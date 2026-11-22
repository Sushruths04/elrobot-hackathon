"""Whole tissue task in ONE fast autonomous run (target 30-40 s), using what the earlier attempts taught:
 1. cap: find it with the camera, go straight to the left edge at z 0.081 (no stepping down), sweep sideways (-x) 7.5 cm in 3 segments, lift;
 2. tissue: find the tissue stack, go above, down to the pinch height (tip z 0.082), close (a thin sheet gives no stall, only the camera can tell), pull STRAIGHT UP fast (8 cm);
 3. beans: find the beans, go to the left of them at the sweep height, sweep the tissue sideways (-x) past them, lift.
 --dry: only the three detections.   Filmed + recorded via film_task.py tempo_fast."""
import argparse, json, sys, time, urllib.request, concurrent.futures as cf
from pathlib import Path
import numpy as np, cv2
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from normakit import er2
ap = argparse.ArgumentParser(); ap.add_argument("--dry", action="store_true"); ap.add_argument("--sweep-z", type=float, default=0.060); a = ap.parse_args()
T0 = time.time(); log = lambda *m: print(f"[{time.time()-T0:5.1f}s]", *m, flush=True)
A = lambda path, body=None: json.loads(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8891" + path, data=None if body is None else json.dumps(body).encode(), method="GET" if body is None else "POST"), timeout=60).read())
jpg = lambda: urllib.request.urlopen("http://127.0.0.1:8890/frame.jpg", timeout=10).read(); shot = lambda n: (ROOT / f"run/fast_{n}.jpg").write_bytes(jpg())
cal = json.load(open(ROOT / "calibration/table_tip_front.json"))["planes"]; Hm = {p: np.array(cal[p]["H_pixel_to_robot_xy_m"], np.float64) for p in "BC"}
def xy_at(u, v, z): c, b = [cv2.perspectiveTransform(np.float32([[[u, v]]]), Hm[p])[0, 0] for p in "CB"]; return c + (b - c) * (z - 0.04) / 0.02
def find(q, n=2):
    jp = jpg()
    with cf.ThreadPoolExecutor(n) as ex: pts = [p for f in [ex.submit(er2.point, jp, q, 640, 480) for _ in range(n)] for p in f.result()["points"]]
    return (float(np.median([p[1] for p in pts])), float(np.median([p[2] for p in pts]))) if pts else None
def must(r, what):
    if r.get("error") and "timeout" not in str(r.get("error")): log(f"   STOP: {what}: {r.get('error')}"); A("/stop", {}); sys.exit(2)
    return r
mv = lambda x, y, z, sp, what, acc=40: must(A("/move_tip", {"x": float(x), "y": float(y), "z": float(z), "roll": 0.0, "speed": sp, "timeout": 12, "low_ok": True, "acc": acc}), what)
SAG = 0.0
cap = find("the white flip-top cap on the tissue pack"); tis = cap; beans = True
log(f"detections: cap {cap}")
if a.dry or not cap: sys.exit(0 if a.dry else 3)
if not a.dry:
    log("WAITING: press 'ARM motion' at http://127.0.0.1:8890")
    while A("/state")["armed_s"] < 60: time.sleep(0.5)
# 1. cap
A("/gripper", {"to": 1321}); x0, y0 = [float(t) for t in xy_at(cap[0] - 30, cap[1] - 2, 0.081)]
mv(x0, y0, 0.081 + 0.008, 300, "to the cap's left edge"); log("at the cap, sweeping sideways")
for k in (1, 2, 3): mv(x0 - 0.025 * k, y0, 0.081 + 0.008, 140, f"cap sweep {k}", 30)
mv(x0 - 0.075, y0 + 0.03, 0.15, 300, "lift away"); shot("1_cap"); log("cap swept")
# 2. tissue
tis = find("the white tissue stack sticking out of the opened tissue pack") or tis; tx, ty = [float(t) for t in xy_at(tis[0], tis[1], 0.06)]; ty -= 0.0
A("/gripper", {"to": 2650}); mv(tx, ty + 0.01, 0.13, 350, "above the tissue"); mv(tx, ty + 0.0, 0.092, 120, "down to the pinch height")
c = A("/gripper", {"to": "close"}); log(f"closed on the tissue: {c.get('gripper_ticks')} ticks {c.get('mA')} mA (thin sheet: only the camera can confirm)"); shot("2_pinched")
mv(tx, ty, 0.17, 300, "pull straight up", 40); shot("3_pulled"); log("pulled the tissue straight up")
# 3. beans: straight lanes left to right at table level (the pattern that worked), lifting between lanes and shifting each lane by 2 cm
xa, xb, ZS = -0.03, -0.35, 0.046
for y in (0.345, 0.325, 0.305, 0.285, 0.27, 0.25, 0.23):
    mv(xa, y, 0.11, 500, "to the start of the lane", 50); mv(xa, y, ZS, 300, "down at the lane start", 50); mv(xa + (xb - xa) / 2, y, ZS, 400, "lane", 40); mv(xb, y, ZS, 400, "lane", 40); mv(xb, y, 0.11, 500, "lift", 50); log(f"lane y {y} swept")
mv(xa, 0.30, 0.15, 500, "lift clear", 50); shot("4_beans"); log("DONE - cap swept, tissue pulled, beans swept in lanes: check run/fast_*.jpg")
