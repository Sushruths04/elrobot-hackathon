#!/usr/bin/env python
"""Add calibration plane C (jaw tip 4 cm above the table, over the BRICK/PEN area x -0.12..0.08, y 0.20..0.36) to calibration/table_tip_front.json.
The tip stays 1.6 cm above the Duplo bricks, so extrapolating to the brick tops (z 0.02) is a short step instead of the long one from 6 and 10 cm.  Needs ARM (waits)."""
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
while A("/state")["armed_s"] < 200:
    if time.time() - t > 150: sys.exit("not armed (need >= 200 s)")
    if int(time.time() - t) % 10 == 0: log("WAITING: press 'ARM motion' at http://127.0.0.1:8890"); time.sleep(1)
    time.sleep(0.4)
Z = 0.04; grid = [(x, y) for y in (0.21, 0.26, 0.31, 0.355) for x in (0.06, -0.01, -0.07, -0.12)]
det = json.loads((ROOT / "run/detect_pose_follower.json").read_text())
tp = A("/state")["tip"]
if np.hypot(tp[0] - det["tip"][0], tp[1] - det["tip"][1]) > 0.04: A("/move_tip", {"x": 0.0, "y": 0.20, "z": 0.26, "roll": 0.0, "speed": 700, "timeout": 20, "low_ok": True})
pts = []
for x, y in grid:
    r = A("/move_tip", {"x": x, "y": y, "z": Z, "speed": 700, "timeout": 15, "low_ok": True})
    if "error" in r: log(f"({x:+.2f},{y:.2f}) skipped: {r['error']}"); continue
    time.sleep(0.5); tip = A("/state")["tip"]; jp = frame(); P = []
    for _ in range(5):
        try: P += [er2.point(jp, "the tip of the gripper jaws (the lowest point of the black gripper)", 640, 480)["points"][0][1:]]
        except Exception: pass
        ok = [(a, b) for i, a in enumerate(P) for b in P[i + 1:] if np.hypot(a[0] - b[0], a[1] - b[1]) < 20]
        if ok: break
    if ok:
        a_, b_ = ok[0]; px = (np.array(a_) + np.array(b_)) / 2; pts.append({"px": [float(px[0]), float(px[1])], "xy": [tip[0], tip[1]], "z": tip[2]}); log(f"tip ({tip[0]:+.3f},{tip[1]:.3f},{tip[2]:.3f}) <- pixel ({px[0]:.0f},{px[1]:.0f})")
    else: log(f"({x:+.2f},{y:.2f}): model answers never agreed - skipped")
A("/move_tip", {"x": 0.0, "y": 0.20, "z": 0.26, "roll": 0.0, "speed": 700, "timeout": 20, "low_ok": True}); A("/move_joints", {"ticks": det["raw"], "speed": 700, "low_ok": True, "timeout": 25})
if len(pts) < 6: sys.exit(f"only {len(pts)} points - not fitted")
src = np.float32([p["px"] for p in pts]); dst = np.float32([p["xy"] for p in pts]); Hm, mask = cv2.findHomography(src, dst, cv2.RANSAC, 0.02)
pr = cv2.perspectiveTransform(src.reshape(-1, 1, 2), Hm).reshape(-1, 2); res = np.linalg.norm(pr - dst, axis=1) * 1000; log(f"plane C: {int(mask.sum())}/{len(pts)} inliers, residual mean {res.mean():.0f} mm, max {res.max():.0f} mm")
f = ROOT / "calibration/table_tip_front.json"; d = json.loads(f.read_text()); d["planes"]["C"] = {"z": Z, "H_pixel_to_robot_xy_m": Hm.tolist(), "points": pts, "residual_mm": [round(float(v), 1) for v in res]}; f.write_text(json.dumps(d, indent=1)); log("saved plane C into calibration/table_tip_front.json")
