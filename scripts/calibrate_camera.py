"""Camera calibration from a printed checkerboard (read-only: reads frames from Station, moves nothing).

  A) Intrinsics (do once per camera and per resolution), hold the board in view at many tilts/distances:
     .venv/bin/python scripts/calibrate_camera.py --cam cam0 --frames 15
  B) Table mapping (pixel -> robot xy metres) for the TABLE camera: lay the board FLAT on the table, then
     .venv/bin/python scripts/calibrate_camera.py --cam cam0 --table --origin-x 0.15 --origin-y 0.10
     origin = robot-base-frame position (m) of the board's first inner corner; board x axis is assumed parallel to the robot x axis
     (use --rot-deg if the board is turned).
  Self-test of detection on the printed file:  --selftest calibration/official/opencv_pattern_9x6_25mm_A4_ORIGINAL.png
Outputs: calibration/camera_<cam>.json and calibration/table_<cam>.json"""
import argparse, json, math, sys, time
from pathlib import Path
import cv2, numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
ap = argparse.ArgumentParser()
ap.add_argument("--cam", default="cam0"); ap.add_argument("--cols", type=int, default=9); ap.add_argument("--rows", type=int, default=6)
ap.add_argument("--square-mm", type=float, default=25.0); ap.add_argument("--frames", type=int, default=15)
ap.add_argument("--table", action="store_true"); ap.add_argument("--origin-x", type=float, default=0.0); ap.add_argument("--origin-y", type=float, default=0.0)
ap.add_argument("--rot-deg", type=float, default=0.0); ap.add_argument("--selftest")
ap.add_argument("--ruler-mm", type=float, default=100.0, help="what the printed 100 mm ruler REALLY measures (corrects printer scaling)")
a = ap.parse_args()
a.square_mm *= a.ruler_mm / 100.0
PAT = (a.cols, a.rows)

def find(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    ok, c = cv2.findChessboardCornersSB(g, PAT, cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY)
    if not ok:
        ok, c = cv2.findChessboardCorners(g, PAT, cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
        if ok: c = cv2.cornerSubPix(g, c, (5, 5), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01))
    return (c.reshape(-1, 2) if ok else None)

if a.selftest:
    im = cv2.imread(a.selftest); im = cv2.resize(im, None, fx=0.3, fy=0.3)
    c = find(im); print("selftest: found" if c is not None else "selftest: NOT found", None if c is None else len(c), "corners"); sys.exit(0 if c is not None else 1)

from normakit.station_backend import StationBackend
b = StationBackend("localhost:8888"); b.start(); time.sleep(1)
def grab():
    return cv2.imdecode(np.frombuffer(b.frame_jpeg(a.cam), np.uint8), cv2.IMREAD_COLOR)
objp = np.zeros((PAT[0] * PAT[1], 3), np.float32); objp[:, :2] = np.mgrid[0:PAT[0], 0:PAT[1]].T.reshape(-1, 2) * (a.square_mm / 1000.0)

def finish(code=0):
    sys.stdout.flush(); import os; os._exit(code)

if a.table:
    cam = ROOT / f"calibration/camera_{a.cam}.json"
    for _ in range(40):
        img = grab(); c = find(img)
        if c is not None: break
        time.sleep(0.25)
    if c is None: print("board not found in view - lay it flat, fully visible, well lit"); finish(1)
    K = dist = None
    if cam.exists():
        d = json.loads(cam.read_text()); K, dist = np.array(d["K"]), np.array(d["dist"])
        c = cv2.undistortPoints(c.reshape(-1, 1, 2).astype(np.float32), K, dist, P=K).reshape(-1, 2)
    r = math.radians(a.rot_deg); R = np.array([[math.cos(r), -math.sin(r)], [math.sin(r), math.cos(r)]])
    world = (objp[:, :2] @ R.T) + np.array([a.origin_x, a.origin_y])
    H, _ = cv2.findHomography(c, world, cv2.RANSAC, 0.003)
    proj = cv2.perspectiveTransform(c.reshape(-1, 1, 2).astype(np.float32), H).reshape(-1, 2)
    err = np.linalg.norm(proj - world, axis=1)
    out = ROOT / f"calibration/table_{a.cam}.json"
    out.write_text(json.dumps({"cam": a.cam, "H_pixel_to_robot_xy_m": H.tolist(), "image_size": [img.shape[1], img.shape[0]], "undistorted": K is not None,
                               "board": {"cols": a.cols, "rows": a.rows, "square_mm": a.square_mm, "origin_xy_m": [a.origin_x, a.origin_y], "rot_deg": a.rot_deg},
                               "mean_err_mm": float(err.mean() * 1000), "max_err_mm": float(err.max() * 1000)}, indent=1))
    print(f"saved {out.relative_to(ROOT)}  mean err {err.mean()*1000:.2f} mm, max {err.max()*1000:.2f} mm" + ("" if K is not None else "  (no intrinsics yet: less accurate)")); finish()

pts, imgs, last, t_last, size = [], [], None, 0, None
print(f"hold the {a.cols+1}x{a.rows+1}-square board in view of {a.cam}; tilt it and change distance between captures. Need {a.frames} frames.", flush=True)
t0 = time.time()
while len(pts) < a.frames and time.time() - t0 < 240:
    img = grab(); size = (img.shape[1], img.shape[0]); c = find(img)
    if c is None or time.time() - t_last < 1.5: time.sleep(0.2); continue
    ctr = c.mean(0); diag = np.linalg.norm(c[0] - c[-1])
    key = np.array([ctr[0], ctr[1], diag])
    if last is not None and np.abs(key - last).max() < 12: time.sleep(0.2); continue   # not different enough from the last capture
    pts.append(c.astype(np.float32)); imgs.append(objp); last = key; t_last = time.time(); print(f"  captured {len(pts)}/{a.frames}", flush=True)
if len(pts) < 8: print("too few views captured (need >= 8): is the board visible and sharp?"); finish(1)
rms, K, dist, rv, tv = cv2.calibrateCamera(imgs, pts, size, None, None)
errs = [float(np.sqrt(np.mean(np.sum((cv2.projectPoints(o, r_, t_, K, dist)[0].reshape(-1, 2) - p_) ** 2, 1)))) for o, p_, r_, t_ in zip(imgs, pts, rv, tv)]
out = ROOT / f"calibration/camera_{a.cam}.json"
out.write_text(json.dumps({"cam": a.cam, "image_size": list(size), "K": K.tolist(), "dist": dist.ravel().tolist(), "rms_px": float(rms), "per_view_px": errs, "views": len(pts),
                           "board": {"cols": a.cols, "rows": a.rows, "square_mm": a.square_mm}}, indent=1))
print(f"saved {out.relative_to(ROOT)}  size {size}  rms {rms:.2f} px  fx={K[0,0]:.0f} fy={K[1,1]:.0f}" + ("  <-- rms > 1 px: redo with sharper, more tilted views" if rms > 1 else "  (good)")); finish()
