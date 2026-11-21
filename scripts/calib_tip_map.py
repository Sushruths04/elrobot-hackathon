"""Re-calibrate pixel -> robot (x,y) with the real arm: move the jaw tip over a grid, Gemini ER-2 points at the tip in the table camera,
pair that pixel with the tip position from forward kinematics, fit a homography per height. Saves calibration/table_tip_front.json.
 plane A z=0.10 (clears the box walls) is used for the box; plane B z=0.04 (x <= 0.04, away from the box) for the marker.
    scripts/calib_tip_map.py          plan only        scripts/calib_tip_map.py --go     (hand near arm, power plug in reach)"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np, cv2
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from normakit import vision, er2
from normakit.core import norm_to_rad, rad_to_norm
from normakit.kin import Kin
from normakit.station_backend import StationBackend, front_camera, REG_TORQUE, REG_GOAL_POS, REG_GOAL_SPEED, REG_ACC
ap = argparse.ArgumentParser(); ap.add_argument("--go", action="store_true"); ap.add_argument("--speed", type=int, default=250); a = ap.parse_args()
J = list(range(1, 9)); wd = lambda x, y: ((x - y + 2048) % 4096) - 2048; T0 = time.time(); log = lambda *m: print(f"[{time.time()-T0:5.1f}s]", *m, flush=True)
be = StationBackend("localhost:8888"); be.start(); time.sleep(1.5); kin = Kin(); st = be.state(); CAM = front_camera(be)
def to_norm(i, t):
    j = st.joints[i]; span = (j.hi - j.lo) % 4096; rel = (t - j.lo) % 4096
    if rel > span + (4096 - span) / 2: rel -= 4096
    return rel / span
def to_ticks(i, n): j = st.joints[i]; return int(round(j.lo + n * ((j.hi - j.lo) % 4096))) % 4096
q_of = lambda t: {i: norm_to_rad(i, to_norm(i, t[i])) for i in range(1, 8)}
tip_of = lambda t: kin.jaw_tip_pose(q_of(t))[0]
live = lambda: {i: be.state().joints[i].raw for i in J}
grid = [("A", 0.10, x, y) for y in (0.23, 0.29, 0.34) for x in (-0.06, 0.04, 0.14)] + [("B", 0.06, x, y) for y in (0.24, 0.29, 0.33) for x in (-0.08, -0.02, 0.04)]
cur = live(); seed = q_of(cur); plan = []
for pl, z, x, y in grid:
    q, pe, oe, tilt = kin.ik_jaw_tip((x, y, z), roll_rad=0.0, seed_rad=seed); ok = pe < 0.015
    log(f"  plane {pl} tip ({x:+.2f},{y:.2f},{z:.2f}): err {pe*100:.1f} cm, tilt {tilt:.0f} deg -> {'ok' if ok else 'SKIP'}")
    if ok: plan.append((pl, z, x, y, {**cur, **{i: to_ticks(i, rad_to_norm(i, q[i])) for i in range(1, 8)}}, q)); seed = q
log(f"{len(plan)} of {len(grid)} points reachable" + ("" if a.go else " - dry run only (add --go)"))
if len(plan) < 10: sys.exit("ABORT: too few reachable grid points")
if not a.go: import os; os._exit(0)
def put(g): be._sync(REG_GOAL_POS, {i: (int(v) % 4096).to_bytes(2, "little") for i, v in g.items()})
def hold(): put({i: be.state().joints[i].raw for i in J})
def move(target, label, tol=40, timeout=40):
    last = {i: float(be.state().joints[i].raw) for i in target}; step = a.speed / 20; t0 = time.time(); hist = []
    while time.time() - t0 < timeout:
        s = be.state()
        if max(s.joints[i].current_ma for i in range(1, 8)) > 700: hold(); log("HIGH CURRENT - holding"); sys.exit(2)
        d = {i: wd(target[i], last[i]) for i in target}; m = max(abs(v) for v in d.values()); sc = 1.0 if m <= step else step / m
        for i in target: last[i] = (last[i] + d[i] * sc) % 4096
        put({i: last[i] for i in target}); time.sleep(0.05)
        err = max(abs(wd(target[i], s.joints[i].raw)) for i in target); hist.append((time.time(), {i: s.joints[i].raw for i in target})); hist[:] = [x for x in hist if x[0] > time.time() - 0.8]
        moving = max(abs(wd(hist[-1][1][i], hist[0][1][i])) for i in target) if len(hist) > 5 else 99
        if err <= tol or (err <= 2 * tol and moving <= 3 and all(abs(wd(target[i], last[i])) < 1 for i in target)): return
    hold(); log(f"TIMEOUT at {label} - holding and STOPPING"); sys.exit(3)
put(cur); be._sync(REG_TORQUE, {i: b"\x01" for i in J}); be._sync(REG_ACC, {i: bytes([40]) for i in J}); be._sync(REG_GOAL_SPEED, {i: a.speed.to_bytes(2, "little") for i in J}); log("TORQUE ON - moving")
data = {"A": [], "B": []}; first = True
for pl, z, x, y, tgt, q in plan:
    if first:                                            # go up first so the way to the first point is high
        up = dict(cur); first = False
    lowz = min(float(tip_of({i: (live()[i] + s_ * wd(tgt[i], live()[i])) % 4096 for i in range(1, 8)})[2]) for s_ in np.linspace(0, 1, 30))
    if lowz < 0.045: log(f"  path to plane {pl} ({x:+.2f},{y:.2f}) would dip to z={lowz:.3f} - point skipped"); continue
    move(tgt, f"plane {pl} ({x:+.2f},{y:.2f})"); time.sleep(0.5); t_fk = tip_of(live()); jp = be.frame_jpeg(CAM); h, w = vision.decode(jp).shape[:2]
    P = []
    for _ in range(2):
        try: P.append(er2.point(jp, "the tip of the gripper jaws (the lowest point of the black gripper)", w, h)["points"][0][1:])
        except Exception as e: log("   ER-2 failed:", str(e)[:80])
    if len(P) == 2 and np.hypot(P[0][0] - P[1][0], P[0][1] - P[1][1]) < 15:
        px = (np.array(P[0]) + np.array(P[1])) / 2; data[pl].append({"px": [float(px[0]), float(px[1])], "xy": [float(t_fk[0]), float(t_fk[1])], "z": float(t_fk[2])}); log(f"  plane {pl}: FK tip ({t_fk[0]:+.3f},{t_fk[1]:.3f},{t_fk[2]:.3f}) <- pixel ({px[0]:.0f},{px[1]:.0f})")
    else: log(f"  plane {pl}: ER-2 tip points disagree or missing ({P}) - point skipped")
hold(); out = {"note": "pixel -> robot xy of the JAW TIP at the given plane height (ER-2 tip pixel + forward kinematics)", "planes": {}}
for pl, z in (("A", 0.10), ("B", 0.06)):
    d = data[pl]
    if len(d) < 5: log(f"plane {pl}: only {len(d)} usable points - not fitted"); continue
    src = np.float32([p["px"] for p in d]); dst = np.float32([p["xy"] for p in d]); Hm, mask = cv2.findHomography(src, dst, cv2.RANSAC, 0.02)
    pr = cv2.perspectiveTransform(src.reshape(-1, 1, 2), Hm).reshape(-1, 2); res = np.linalg.norm(pr - dst, axis=1) * 1000
    log(f"plane {pl}: {int(mask.sum())}/{len(d)} inliers, residual mean {res.mean():.0f} mm, max {res.max():.0f} mm"); out["planes"][pl] = {"z": z, "H_pixel_to_robot_xy_m": Hm.tolist(), "points": d, "residual_mm": [round(float(r), 1) for r in res]}
(ROOT / "calibration/table_tip_front.json").write_text(json.dumps(out, indent=1)); log("saved calibration/table_tip_front.json"); import os; os._exit(0)
