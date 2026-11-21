#!/usr/bin/env python
"""Zero-shot calibration through the fast daemon (armd): move the jaw tip over a grid, the vision model points at the tip in the table camera, pair that pixel
with the tip position from forward kinematics, fit pixel -> robot xy per plane height. Saves calibration/table_tip_front.json.  Needs ARM (waits for you to press it)."""
import json, sys, time, urllib.request
from pathlib import Path
import numpy as np, cv2
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from normakit import er2
T0 = time.time(); log = lambda *m: print(f"[{time.time()-T0:5.1f}s]", *m, flush=True)
def A(path, body=None, timeout=60):
    req = urllib.request.Request("http://127.0.0.1:8891" + path, data=None if body is None else json.dumps(body).encode(), method="GET" if body is None else "POST")
    with urllib.request.urlopen(req, timeout=timeout) as r: return json.loads(r.read())
frame = lambda: urllib.request.urlopen("http://127.0.0.1:8890/frame.jpg", timeout=10).read()
t = time.time()
while A("/state")["armed_s"] < 240:
    if time.time() - t > 180: sys.exit("not armed (need >= 240 s left) within 3 min")
    if int(time.time() - t) % 10 == 0: log("WAITING: press 'ARM motion' on http://127.0.0.1:8890"); time.sleep(1)
    time.sleep(0.5)
log("armed - going")
grid = [("A", 0.10, x, y) for y in (0.23, 0.29, 0.34) for x in (-0.06, 0.04, 0.14)] + [("B", 0.06, x, y) for y in (0.24, 0.29, 0.33) for x in (-0.08, -0.02, 0.04)]
r = A("/move_tip", {"x": 0.0, "y": 0.20, "z": 0.25, "speed": 700, "low_ok": True}); log("waypoint up/forward:", {k: r.get(k) for k in ("ok", "arrived_in_ms", "error")})
if "error" in r: sys.exit("could not reach the first waypoint: " + str(r))
data = {"A": [], "B": []}
for pl, z, x, y in grid:
    r = A("/move_tip", {"x": x, "y": y, "z": z, "speed": 700})
    if "error" in r: log(f"plane {pl} ({x:+.2f},{y:.2f},{z:.2f}) skipped: {r['error']}"); continue
    time.sleep(0.5); tip = A("/state")["tip"]; jp = frame(); P = []
    for _ in range(5):                                                               # up to 5 asks until two answers agree within 20 px
        try: P += [er2.point(jp, "the tip of the gripper jaws (the lowest point of the black gripper)", 640, 480)["points"][0][1:]]
        except Exception: pass
        ok_pair = [(a, b) for i, a in enumerate(P) for b in P[i + 1:] if np.hypot(a[0] - b[0], a[1] - b[1]) < 20]
        if ok_pair: break
    if ok_pair:
        a_, b_ = ok_pair[0]; px = (np.array(a_) + np.array(b_)) / 2; data[pl].append({"px": [float(px[0]), float(px[1])], "xy": [tip[0], tip[1]], "z": tip[2]}); log(f"plane {pl}: tip ({tip[0]:+.3f},{tip[1]:.3f},{tip[2]:.3f}) <- pixel ({px[0]:.0f},{px[1]:.0f})  [{r.get('arrived_in_ms')} ms move, {len(P)} asks]")
    else: log(f"plane {pl} ({x:+.2f},{y:.2f}): model answers never agreed {[(round(a), round(b)) for a, b in P]} - skipped")
A("/move_tip", {"x": 0.0, "y": 0.20, "z": 0.25, "speed": 700, "low_ok": True})
out = {"note": "pixel -> robot xy of the JAW TIP at the plane height (vision-model tip pixel + forward kinematics)", "planes": {}}
for pl, z in (("A", 0.10), ("B", 0.06)):
    d = data[pl]
    if len(d) < 5: log(f"plane {pl}: only {len(d)} usable points - not fitted"); continue
    src = np.float32([p["px"] for p in d]); dst = np.float32([p["xy"] for p in d]); Hm, mask = cv2.findHomography(src, dst, cv2.RANSAC, 0.02)
    pr = cv2.perspectiveTransform(src.reshape(-1, 1, 2), Hm).reshape(-1, 2); res = np.linalg.norm(pr - dst, axis=1) * 1000
    log(f"plane {pl}: {int(mask.sum())}/{len(d)} inliers, residual mean {res.mean():.0f} mm, max {res.max():.0f} mm"); out["planes"][pl] = {"z": z, "H_pixel_to_robot_xy_m": Hm.tolist(), "points": d, "residual_mm": [round(float(v), 1) for v in res]}
if out["planes"]: (ROOT / "calibration/table_tip_front.json").write_text(json.dumps(out, indent=1)); log("saved calibration/table_tip_front.json")
else: log("NOT saved: no plane had enough points")
