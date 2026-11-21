"""Fit the table mapping from the 4 touched corners with a RIGID/similarity model (rotation + translation [+ scale], optional mirror).
Unlike an exact 4-point homography this averages out placement noise and shows an honest residual.   scripts/touch_fit.py [--exact]"""
import json, sys, time
from pathlib import Path
import cv2, numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from normakit import vision
from normakit.station_backend import StationBackend
ORDER = ["top-left", "top-right", "bottom-right", "bottom-left"]; SQ, PAT = 0.025, (9, 6)
pts = json.loads((ROOT / "run/touch_points.json").read_text()); R = np.array([pts[o]["xyz"][:2] for o in ORDER])
def find(rgb): return cv2.findChessboardCornersSB(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), PAT, cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY)
be = StationBackend("localhost:8888", require_motors=False); be.start(); time.sleep(1.5)
ok, c = find(vision.decode(be.frame_jpeg("cam1")))
if not ok: ok, c = find(vision.decode((ROOT / "run/board_frame.jpg").read_bytes())); print("note: using the saved board frame", flush=True)
flat = c.reshape(-1, 2).astype(np.float32); cols, rows = np.meshgrid(np.arange(PAT[0]), np.arange(PAT[1])); bxy = np.stack([cols.ravel() * SQ, rows.ravel() * SQ], 1).astype(np.float32)
u, v = flat[:, 0], flat[:, 1]; idx = {"top-left": np.argmin(u + v), "top-right": np.argmax(u - v), "bottom-right": np.argmax(u + v), "bottom-left": np.argmin(u - v)}
B = np.array([bxy[idx[o]] for o in ORDER], float)
def umeyama(X, Y, scale):                       # Y ~ s R X + t   (R may be a reflection: tried both below)
    mx, my = X.mean(0), Y.mean(0); Xc, Yc = X - mx, Y - my; U, S, Vt = np.linalg.svd(Yc.T @ Xc / len(X)); D = np.eye(2)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0: D[1, 1] = -1
    Rm = U @ D @ Vt; s = (S * np.diag(D)).sum() / (Xc ** 2).sum() * len(X) if scale else 1.0; return s, Rm, my - s * Rm @ mx
best = None
for mirror in (False, True):
    for scale in (False, True):
        Bm = B * np.array([-1 if mirror else 1, 1]); s, Rm, t = umeyama(Bm, R, scale); pred = (s * (Rm @ Bm.T)).T + t; err = np.linalg.norm(pred - R, axis=1) * 1000
        print(f"  model mirror={mirror!s:5} scale={'fit ' + format(s, '.3f') if scale else '1.0':9}: residuals mm {np.round(err, 1).tolist()}  mean {err.mean():.1f}", flush=True)
        if not scale and (best is None or err.mean() < best[0]): best = (err.mean(), mirror, s, Rm, t, err)
mean, mirror, s, Rm, t, err = best
M = np.eye(3); M[:2, :2] = s * Rm @ np.diag([-1 if mirror else 1, 1]); M[:2, 2] = t                    # board metres -> robot xy
Hp2b, _ = cv2.findHomography(flat, bxy, cv2.RANSAC, 0.5); Hp2r = M @ Hp2b
(ROOT / "calibration/table_front.json").write_text(json.dumps({"cam": "cam1", "name": "front", "model": f"rigid (mirror={mirror})", "H_pixel_to_robot_xy_m": Hp2r.tolist(), "undistorted": False,
    "touched": pts, "residual_mm": err.tolist(), "mean_residual_mm": float(mean)}, indent=1))
print(f"SAVED calibration/table_front.json  (rigid model, mirror={mirror}); honest residual at the 4 corners: mean {mean:.1f} mm, max {err.max():.1f} mm", flush=True); import os; os._exit(0)
