"""Manual corner registration, one corner per command (you place the gripper tip, I record on your word).
    scripts/touch_capture.py record top-left|top-right|bottom-right|bottom-left
    scripts/touch_capture.py compute            # after all four: fits the table mapping -> calibration/table_front.json"""
import collections, json, sys, time
from pathlib import Path
import cv2, numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from normakit import vision
from normakit.core import norm_to_rad
from normakit.kin import Kin
from normakit.station_backend import StationBackend
ORDER = ["top-left", "top-right", "bottom-right", "bottom-left"]; PTS = ROOT / "run/touch_points.json"; SQ, PAT = 0.025, (9, 6)
cmd = sys.argv[1] if len(sys.argv) > 1 else ""
pts = json.loads(PTS.read_text()) if PTS.exists() else {}
if cmd == "reset": PTS.write_text("{}"); print("cleared"); sys.exit(0)
if cmd == "record":
    corner = sys.argv[2]; assert corner in ORDER
    be = StationBackend("localhost:8888"); be.start(); time.sleep(1.0); kin = Kin(); hist = collections.deque(); t0 = time.time()
    while time.time() - t0 < 8:
        n = {i: j.norm for i, j in be.state().joints.items()}; hist.append((time.time(), n)); [hist.popleft() for _ in range(len(hist)) if hist[0][0] < time.time() - 1.2]; time.sleep(0.1)
        if time.time() - hist[0][0] >= 1.1 and max(max(h[1][i] for h in hist) - min(h[1][i] for h in hist) for i in n) <= 0.004: break
    else: print(f"NOT RECORDED: the arm was still moving - hold it still and ask again", flush=True); import os; os._exit(1)
    xyz = [float(c) for c in kin.fk({i: norm_to_rad(i, x) for i, x in n.items()})]; pts[corner] = {"xyz": xyz, "norms": n}; PTS.write_text(json.dumps(pts, indent=1))
    print(f"RECORDED {corner}: x={xyz[0]:.3f} y={xyz[1]:.3f} z={xyz[2]:.3f} m | done: {', '.join(c for c in ORDER if c in pts)} | missing: {', '.join(c for c in ORDER if c not in pts) or 'none'}", flush=True); import os; os._exit(0)
if cmd == "compute":
    miss = [c for c in ORDER if c not in pts]
    if miss: sys.exit("missing corners: " + ", ".join(miss))
    def find(rgb): return cv2.findChessboardCornersSB(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), PAT, cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY)
    be = StationBackend("localhost:8888", require_motors=False); be.start(); time.sleep(1.5)
    ok, c = find(vision.decode(be.frame_jpeg("cam1")))
    if not ok and (ROOT / "run/board_frame.jpg").exists(): ok, c = find(vision.decode((ROOT / "run/board_frame.jpg").read_bytes())); print("note: using the saved board frame", flush=True)
    if not ok: sys.exit("checkerboard not found")
    flat = c.reshape(-1, 2).astype(np.float32); cols, rows = np.meshgrid(np.arange(PAT[0]), np.arange(PAT[1])); bxy = np.stack([cols.ravel() * SQ, rows.ravel() * SQ], 1).astype(np.float32)
    u, v = flat[:, 0], flat[:, 1]; idx = {"top-left": np.argmin(u + v), "top-right": np.argmax(u - v), "bottom-right": np.argmax(u + v), "bottom-left": np.argmin(u - v)}
    B = np.array([bxy[idx[o]] for o in ORDER], np.float32); R = np.array([pts[o]["xyz"][:2] for o in ORDER], np.float32)
    Hb2r, _ = cv2.findHomography(B, R, 0); Hp2b, _ = cv2.findHomography(flat, bxy, cv2.RANSAC, 0.5); Hp2r = Hb2r @ Hp2b
    err = np.linalg.norm(cv2.perspectiveTransform(B.reshape(-1, 1, 2), Hb2r).reshape(-1, 2) - R, axis=1) * 1000
    side = [float(np.linalg.norm(R[a] - R[b])) for a, b in ((0, 1), (3, 2), (0, 3), (1, 2))]; want = [8 * SQ, 8 * SQ, 5 * SQ, 5 * SQ]
    (ROOT / "calibration/table_front.json").write_text(json.dumps({"cam": "cam1", "name": "front", "H_pixel_to_robot_xy_m": Hp2r.tolist(), "undistorted": False, "touched": pts, "residual_mm": err.tolist(), "board_side_robot_m": side, "board_side_true_m": want}, indent=1))
    print(f"saved calibration/table_front.json | residual at the 4 corners (mm): {np.round(err, 1).tolist()} | board sides measured by the arm (m): {np.round(side, 3).tolist()} vs true {want}", flush=True); import os; os._exit(0)
print(__doc__)
