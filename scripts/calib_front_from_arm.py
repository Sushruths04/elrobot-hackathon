"""Re-calibrate the FRONT camera pixel <-> robot xy mapping with the real arm (user: the old inverse parameters are wrong).
Arm folded out of view -> reference photo. Jaw tip to a 3x3 grid at z=0.10 and z=0.14 (above the bowl rim), each: front photo, jaw tip pixel = lowest point
of what changed vs the reference, paired with the FK jaw tip. Fits H(pixel->xy) per height, reports residual mm and the old calibration's error at the same points.
Saves calibration/table_tip_front_arm_<stamp>.json (old file untouched) + results/photos/calib_arm_<stamp>.jpg."""
import json, sys, time, urllib.request
from pathlib import Path
import numpy as np, cv2
ROOT = Path(__file__).resolve().parents[1]
T0 = time.time(); log = lambda *m: print(f"[{time.time() - T0:5.1f}s]", *m, flush=True)
A = lambda p, b=None, t=60: json.loads(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8891" + p, data=None if b is None else json.dumps(b).encode(), method="GET" if b is None else "POST"), timeout=t).read())
front = lambda: cv2.imdecode(np.frombuffer(urllib.request.urlopen("http://127.0.0.1:8893/frame.jpg?c=table", timeout=5).read(), np.uint8), 1)
FOLD = {"1": 2094, "2": 1055, "3": 1554, "4": 3020, "5": 170, "6": 2836, "7": 3021, "8": 2766}
def fold():
    r = A("/move_joints", {"ticks": FOLD, "speed": 300, "wait": True, "timeout": 25})
    if not r.get("ok"):
        for stage in (("5", "6", "7", "8"), ("3", "4"), ("1", "2")): A("/move_joints", {"ticks": {k: FOLD[k] for k in stage}, "speed": 300, "wait": True, "timeout": 25})
_cp0 = json.loads((ROOT / "calibration/front_camera_pose_arm.json").read_text()); _cam0 = json.loads((ROOT / "calibration/camera_front.json").read_text())
def predict(tp): return cv2.projectPoints(np.array([tp], float), np.array(_cp0["rvec"]), np.array(_cp0["tvec"]), np.array(_cam0["K"]), np.array(_cam0["dist"]).ravel())[0].reshape(2)
def jaw_px(im, ref, guess=None):
    d = cv2.absdiff(cv2.GaussianBlur(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), (5, 5), 0), cv2.GaussianBlur(cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY), (5, 5), 0))
    m = (d > 35).astype(np.uint8); m[:120] = 0; m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)); n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    if n < 2: return None
    win = np.zeros_like(m)
    if guess is not None:                                                  # only changed pixels near where the arm should be (people/objects elsewhere ignored)
        u, v = int(guess[0]), int(guess[1]); win[max(0, v - 90):v + 50, max(0, u - 70):u + 70] = 1
    else: win[:] = 1
    mm = m * win; ys, xs = np.nonzero(mm)
    if len(xs) < 80: return None
    low = ys >= ys.max() - 4; return np.array([xs[low].mean(), ys.max()], float)
while A("/state")["armed_s"] < 90: log("WAITING for ARM"); time.sleep(2)
A("/mirror", {"on": False}); A("/gripper", {"to": "close"}); log("folding out of view for the reference photo"); fold(); time.sleep(1.2); ref = front()
(ROOT / "results/calib_frames").mkdir(parents=True, exist_ok=True); cv2.imwrite(str(ROOT / "results/calib_frames/reference.jpg"), ref)
old = json.loads((ROOT / "calibration/table_tip_front.json").read_text())["planes"]
_hp = lambda pl, u, v: cv2.perspectiveTransform(np.float32([[[u, v]]]), np.array(old[pl]["H_pixel_to_robot_xy_m"], np.float64))[0, 0]
old_xy = lambda z, u, v: _hp("A", u, v) + (_hp("B", u, v) - _hp("A", u, v)) * (z - 0.10) / (0.06 - 0.10)
pairs = {0.10: [], 0.15: []}; vis = ref.copy()
for z in (0.15, 0.10):
    for x in (-0.24, -0.16, -0.08, 0.0, 0.08, 0.15):
        for y in (0.17, 0.25, 0.31):
            r = A("/move_tip", {"x": x, "y": y, "z": z, "roll": 1.5708, "tilt": 18, "speed": 350, "timeout": 15})
            if not r.get("ok"): log(f"  ({x},{y},{z}) refused: {r.get('error')}"); continue
            time.sleep(0.7); tp = np.array(A("/state")["tip"], float); im = front(); gp = predict(tp); p = jaw_px(im, ref, gp)
            (ROOT / "results/calib_frames").mkdir(parents=True, exist_ok=True); cv2.imwrite(str(ROOT / f"results/calib_frames/{x:+.2f}_{y:.2f}_{z:.2f}.jpg"), im)
            if p is None: log(f"  ({x},{y},{z}) jaws not seen"); continue
            pairs[z].append((p.tolist(), tp.tolist())); e_old = np.linalg.norm(old_xy(tp[2], *p) - tp[:2]) * 1000
            cv2.circle(vis, (int(p[0]), int(p[1])), 4, (0, 0, 255) if z == 0.10 else (255, 0, 0), -1)
            log(f"  tip {np.round(tp, 3)} -> jaw pixel {np.round(p, 1)} | OLD calibration error {e_old:.0f} mm")
fold()
out = {"note": "pixel (lowest point of the arm vs an arm-out-of-view reference = jaw tips, gripper tilt 18 deg) -> robot xy of the FK jaw tip, fitted from the real arm", "made": time.strftime("%Y-%m-%d %H:%M"), "planes": {}}
for z, pr in pairs.items():
    if len(pr) < 5: log(f"z={z}: only {len(pr)} points - not fitted"); continue
    P = np.float32([q[0] for q in pr]); X = np.float32([q[1][:2] for q in pr]); H, _ = cv2.findHomography(P, X, 0)
    res = np.linalg.norm(cv2.perspectiveTransform(P[None], H)[0] - X, axis=1) * 1000; zz = float(np.mean([q[1][2] for q in pr]))
    out["planes"]["z%.2f" % z] = {"z": zz, "H_pixel_to_robot_xy_m": H.tolist(), "points": [q for q in pr], "residual_mm": res.round(1).tolist()}
    log(f"NEW z={zz:.3f}: {len(pr)} points, residual mean {res.mean():.1f} mm, max {res.max():.1f} mm")
stamp = time.strftime("%Y%m%d_%H%M%S"); fn = ROOT / f"calibration/table_tip_front_arm_{stamp}.json"; fn.write_text(json.dumps(out, indent=1)); cv2.imwrite(str(ROOT / f"results/photos/calib_arm_{stamp}.jpg"), vis)
log(f"DONE saved {fn.relative_to(ROOT)}")
