"""Final tissue task, one fast autonomous run (about a minute), with the parameters that worked in the guided session:
 1. cap: detect it, start 5.6 cm to its image-left and 1.8 cm back, tip height 10.3 cm commanded (the fingers catch the cap lip there), sweep 14.5 cm to image-right at that SAME height; the arm steps aside before/after and the picture change around the cap is checked; one retry slightly lower;
 2. tissue: detect the stack, jaws open above it, down to tip 9.2 cm, close, pull straight up;
 3. beans: detect them, straight lanes at table level toward image-right, lifting between lanes.   Filmed + recorded via film_task.py tempo_final."""
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
    r = A("/move_tip", {"x": float(x), "y": float(y), "z": float(z), "roll": 0.0, "speed": sp, "timeout": 8, "low_ok": True, "acc": acc}); e = str(r.get("error") or "")
    if e and "timeout" not in e and "cannot reach" not in e: log(f"   STOP: {e}"); A("/stop", {}); sys.exit(2)
def find(q, n=3):
    jp = jpg()
    with cf.ThreadPoolExecutor(n) as ex: pts = [p for f in [ex.submit(er2.point, jp, q, 640, 480) for _ in range(n)] for p in f.result()["points"]]
    return (float(np.median([p[1] for p in pts])), float(np.median([p[2] for p in pts]))) if pts else None
log("WAITING: press 'ARM motion' at http://127.0.0.1:8890")
while A("/state")["armed_s"] < 45: time.sleep(0.5)
t = A("/state")["tip"]; mv(t[0], t[1], max(t[2], 0.15), 500, 50); A("/gripper", {"to": 1321})
# 1. cap
gray = lambda: cv2.cvtColor(img(), cv2.COLOR_BGR2GRAY).astype(np.float32); ok = False
for k, zc in enumerate((0.103, 0.097), 1):
    cap = find("the white flip-top cap on the tissue pack")
    if not cap: log("   no cap found"); continue
    u, v = cap; cx, cy = [float(q) for q in xy_at(u, v, 0.08)]; xs, ys, xe = cx + 0.056, cy - 0.018, cx - 0.089
    mv(xs, ys + 0.02, 0.17, 500, 50); time.sleep(0.4); roi = (slice(int(v) - 45, int(v) + 45), slice(int(u) - 60, int(u) + 60)); before = gray()[roi]
    mv(xs, ys, zc, 400, 50); mv(xe, ys, zc, 500, 70); log(f"cap attempt {k}: swept at tip z {zc} from x {xs:+.3f} to {xe:+.3f}")
    mv(xe, ys, 0.17, 500, 50); mv(xs, ys + 0.02, 0.17, 500, 50); time.sleep(0.4); d = float(np.abs(gray()[roi] - before).mean()); (ROOT / f"run/final_cap_{k}.jpg").write_bytes(jpg()); log(f"   picture change around the cap {d:.1f}")
    if d > 12: ok = True; break
log(f"cap stage: {'changed a lot, open' if ok else 'little change (check run/final_cap_*.jpg)'}")
# 2. tissue: the stack is where the model says UNLESS that disagrees with the cap-based estimate (the stack sits ~15 px right of the cap centre, as in the guided session)
held = False
for k, zt in enumerate((0.087, 0.083, 0.091), 1):
    hsv_ = cv2.cvtColor(img(), cv2.COLOR_BGR2HSV); w_ = ((hsv_[..., 2] > 200) & (hsv_[..., 1] < 35)).astype(np.uint8); box = np.zeros_like(w_); cu, cv_ = (cap if cap else (290, 240)); box[int(cv_) - 15:int(cv_) + 65, int(max(cu - 85, 0)):int(cu + 10)] = 1; w_ *= box; yy_, xx_ = np.nonzero(w_)
    tis = (float(np.median(xx_)), float(np.median(yy_))) if len(xx_) > 60 else find("the white tissue stack sticking out of the opened tissue pack"); log(f"   white tissue patch at {tis} ({len(xx_)} white pixels)")
    if not tis: log("no tissue found"); break
    tx, ty = [float(q) for q in xy_at(tis[0], tis[1], 0.06)]; A("/gripper", {"to": 2650}); mv(tx, ty + 0.01, 0.13, 500, 50); mv(tx, ty, zt, 200, 40)
    c = A("/gripper", {"to": "close"}); log(f"tissue attempt {k}: pinched at ({tx:+.3f},{ty:.3f}) tip z {zt}: {c.get('gripper_ticks')} ticks {c.get('mA')} mA"); mv(tx, ty, 0.14, 250, 40); time.sleep(0.4)
    jp = jpg(); (ROOT / f"run/final_tissue_{k}.jpg").write_bytes(jp)
    with cf.ThreadPoolExecutor(2) as ex: rs = [f.result() for f in [ex.submit(er2.ask, jp, 'Is a white tissue paper hanging from the jaws of the robot gripper, lifted out of the pack? Answer JSON {"held": true/false}') for _ in range(2)]]
    if all(bool(r.get("held")) for r in rs): held = True; log("   the camera says a tissue is HELD"); break
    log("   the camera says nothing is held - putting it back/opening and retrying"); A("/gripper", {"to": 2650}); mv(tx, ty + 0.02, 0.14, 300, 40)
if held: mv(tx, ty, 0.17, 300, 40); log("tissue pulled straight up and out")
log(f"tissue stage {'done' if held else 'NOT confirmed'}")
# 3. beans: red-brown small blobs, kept only if they form a cluster (>=3 neighbours within 40 px), away from the pack
im = img(); hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV); gb = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY); m = (gb < 105).astype(np.uint8); mask = np.zeros_like(m); mask[240:330, 270:600] = 1; m *= mask
n, lab, st, cen = cv2.connectedComponentsWithStats(m); allb = [(cen[i][0], cen[i][1]) for i in range(1, n) if 5 < st[i][4] < 120]
pts = [p for p in allb if sum(np.hypot(p[0] - q[0], p[1] - q[1]) < 40 for q in allb) >= 4 and not (cap and abs(p[0] - cap[0]) < 70 and abs(p[1] - cap[1]) < 55)]
if not pts: log("DONE - no bean cluster detected"); sys.exit(0)
xy = [xy_at(u, v, 0.04) for u, v in pts]; xs_ = [p[0] for p in xy]; ys_ = [p[1] for p in xy]; xa = min(0.05, max(xs_) + 0.06); xb = max(-0.33, min(xs_) - 0.14); lanes = np.arange(min(ys_) - 0.02, max(ys_) + 0.021, 0.02)[:6]
log(f"beans: {len(pts)} blobs in the cluster at pixel x {min(p[0] for p in pts):.0f}-{max(p[0] for p in pts):.0f}, y {min(p[1] for p in pts):.0f}-{max(p[1] for p in pts):.0f}; {len(lanes)} lanes from x {xa:.2f} to {xb:.2f}, y {lanes[0]:.3f}..{lanes[-1]:.3f}"); ZS = 0.046
for y in lanes: mv(xa, y, 0.11, 500, 50); mv(xa, y, ZS, 300, 50); mv((xa + xb) / 2, y, ZS, 400); mv(xb, y, ZS, 400); mv(xb, y, 0.11, 500, 50); log(f"lane y {y:.3f} swept")
mv(xa, float(np.mean(ys_)), 0.15, 500, 50); (ROOT / "run/final_beans.jpg").write_bytes(jpg()); log("DONE - cap, tissue and beans attempted: run/final_*.jpg")
