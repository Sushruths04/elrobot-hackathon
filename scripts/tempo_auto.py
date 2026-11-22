"""Whole tissue task, autonomous, WITH attempts and camera verification at each stage (no questions to the human):
 1. cap: sweep it open sideways from the top (z 0.081 first), retreat the arm out of view, ask the camera/model twice if the cap is open; retry at other heights;
 2. tissue: pinch the tip near the top, lift 2 cm, camera check that a tissue is held, retry up to 3 times; then pull straight up;
 3. beans: detect the beans (dark brown blobs), straight lanes at table level toward image-right, lifting between lanes, shifted by 2 cm.
 Filmed + recorded via film_task.py tempo_auto."""
import json, sys, time, urllib.request, concurrent.futures as cf
from pathlib import Path
import numpy as np, cv2
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from normakit import er2
T0 = time.time(); log = lambda *m: print(f"[{time.time()-T0:5.1f}s]", *m, flush=True)
A = lambda path, body=None: json.loads(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8891" + path, data=None if body is None else json.dumps(body).encode(), method="GET" if body is None else "POST"), timeout=60).read())
jpg = lambda: urllib.request.urlopen("http://127.0.0.1:8890/frame.jpg", timeout=10).read(); img = lambda: cv2.imdecode(np.frombuffer(jpg(), np.uint8), cv2.IMREAD_COLOR)
cal = json.load(open(ROOT / "calibration/table_tip_front.json"))["planes"]; Hm = {p: np.array(cal[p]["H_pixel_to_robot_xy_m"], np.float64) for p in "BC"}
def xy_at(u, v, z): c, b = [cv2.perspectiveTransform(np.float32([[[u, v]]]), Hm[p])[0, 0] for p in "CB"]; return c + (b - c) * (z - 0.04) / 0.02
def mv(x, y, z, sp, acc=40):
    r = A("/move_tip", {"x": float(x), "y": float(y), "z": float(z), "roll": 0.0, "speed": sp, "timeout": 10, "low_ok": True, "acc": acc}); e = str(r.get("error") or "")
    if e and "timeout" not in e and "cannot reach" not in e: log(f"   STOP: {e}"); A("/stop", {}); sys.exit(2)
    return r
def find(q, n=2):
    jp = jpg()
    with cf.ThreadPoolExecutor(n) as ex: pts = [p for f in [ex.submit(er2.point, jp, q, 640, 480) for _ in range(n)] for p in f.result()["points"]]
    return (float(np.median([p[1] for p in pts])), float(np.median([p[2] for p in pts]))) if pts else None
def yes(question, key, n=2):
    jp = jpg()
    with cf.ThreadPoolExecutor(n) as ex: rs = [f.result() for f in [ex.submit(er2.ask, jp, question) for _ in range(n)]]
    return all(bool(r.get(key)) for r in rs)
log("WAITING: press 'ARM motion' at http://127.0.0.1:8890")
while A("/state")["armed_s"] < 60: time.sleep(0.5)
# 1. cap  (verified by what CHANGED in the picture around the cap: the model said "open" for a closed cap twice, so it is not trusted here)
t = A("/state")["tip"]; mv(t[0], t[1], max(t[2] + 0.1, 0.15), 300, 40); log("lifted clear first")
gray = lambda: cv2.cvtColor(img(), cv2.COLOR_BGR2GRAY).astype(np.float32); cap_ok = False
for k, z0 in enumerate((0.081, 0.078, 0.084, 0.075), 1):
    cap = find("the white flip-top cap on the tissue pack")
    if not cap: log("   no cap found"); continue
    u, v = cap; A("/gripper", {"to": 1321}); x0, y0 = [float(t_) for t_ in xy_at(u - 30, v - 2, z0)]
    mv(x0 + 0.17, y0 - 0.04, 0.17, 400, 40); time.sleep(0.6); before = gray()[int(v) - 45:int(v) + 45, int(u) - 60:int(u) + 60]; log(f"cap attempt {k}: sweep at z {z0} from the cap's left edge")
    mv(x0, y0, z0 + 0.008, 300, 40)
    for j in (1, 2, 3): mv(x0 - 0.025 * j, y0, z0 + 0.008, 140, 30)
    mv(x0 + 0.17, y0 - 0.04, 0.17, 400, 40); time.sleep(0.6); after = gray()[int(v) - 45:int(v) + 45, int(u) - 60:int(u) + 60]; d = float(np.abs(after - before).mean()); (ROOT / f"run/auto_cap_{k}.jpg").write_bytes(jpg())
    log(f"   picture change around the cap: {d:.1f} (open needs more than 9)")
    if d > 9: cap_ok = True; log("   the cap CHANGED a lot - open"); break
    log("   the cap did not change - still closed - retrying")
log(f"cap stage {'done' if cap_ok else 'NOT confirmed'}")
# 2. tissue
held = False
for k, z in enumerate((0.092, 0.086, 0.080), 1):
    tis = find("the white tissue stack sticking out of the opened tissue pack")
    if not tis: log("   no tissue found"); continue
    tx, ty = [float(t) for t in xy_at(tis[0], tis[1], 0.06)]; A("/gripper", {"to": 2650}); log(f"tissue attempt {k}: pinch the tip at z {z}")
    mv(tx, ty + 0.01, 0.13, 350, 40); mv(tx, ty, z, 120, 40); c = A("/gripper", {"to": "close"}); mv(tx, ty, z + 0.02, 60, 30); time.sleep(0.5)
    if yes('Is a white tissue held between the jaws of the robot gripper and lifted up from the pack? Answer JSON {"held": true/false}', "held"): held = True; log("   the camera says a tissue is HELD"); break
    log("   no tissue in the jaws - opening and retrying"); A("/gripper", {"to": 2650}); mv(tx, ty + 0.02, 0.13, 300, 40)
if held: mv(tx, ty, 0.17, 300, 40); log("pulled the tissue straight up")
log(f"tissue stage {'done' if held else 'NOT confirmed'}")
# 3. beans
im = img(); g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY); b, gr, r = [im[..., i].astype(int) for i in range(3)]
m = ((g < 115) & (r > b + 12)).astype(np.uint8); m[:215] = 0; m[330:] = 0; m[:, :60] = 0; m[:, 600:] = 0
n, lab, st, cen = cv2.connectedComponentsWithStats(m); pts = [(cen[i][0], cen[i][1]) for i in range(1, n) if 5 < st[i][4] < 160]
if not pts: log("no beans found"); log("DONE - cap/tissue done, no beans detected"); sys.exit(0)
us = [p[0] for p in pts]; vs = [p[1] for p in pts]; log(f"beans: {len(pts)} blobs, pixel box x {min(us):.0f}-{max(us):.0f}, y {min(vs):.0f}-{max(vs):.0f}")
xs = [float(xy_at(u, v, 0.04)[0]) for u, v in pts]; ys = [float(xy_at(u, v, 0.04)[1]) for u, v in pts]
xa = min(0.05, max(xs) + 0.05); xb = max(-0.33, min(xs) - 0.12); lanes = np.arange(min(ys) - 0.02, max(ys) + 0.021, 0.02)[:8]; ZS = 0.046
for y in lanes:
    mv(xa, y, 0.11, 500, 50); mv(xa, y, ZS, 300, 50); mv((xa + xb) / 2, y, ZS, 400, 40); mv(xb, y, ZS, 400, 40); mv(xb, y, 0.11, 500, 50); log(f"lane y {y:.3f} swept")
mv(xa, float(np.mean(ys)), 0.15, 500, 50); (ROOT / "run/auto_final.jpg").write_bytes(jpg()); log("DONE - cap, tissue and beans attempted: check run/auto_final.jpg")
