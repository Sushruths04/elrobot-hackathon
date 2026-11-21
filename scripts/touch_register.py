"""Register the front camera to the robot by TOUCHING 4 corners of the printed checkerboard with the follower's gripper (hand-guided, torque OFF, nothing moves by itself).
Order (as seen from the FRONT camera): 1 top-left, 2 top-right, 3 bottom-right, 4 bottom-left INNER corner of the pattern. Hold the gripper tip on each corner ~2 s.
Result: calibration/table_front.json  (pixel -> robot x,y metres, same format Arm uses).   .venv/bin/python scripts/touch_register.py"""
import collections, json, sys, time
from pathlib import Path
import cv2, numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from normakit import vision
from normakit.core import norm_to_rad
from normakit.kin import Kin
from normakit.station_backend import StationBackend
FRONT, SQ, PAT = "cam1", 0.025, (9, 6)
be = StationBackend("localhost:8888"); be.start(); time.sleep(1.5); kin = Kin()
def frame(): return vision.decode(be.frame_jpeg(FRONT))
def find_board(rgb):
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY); return cv2.findChessboardCornersSB(g, PAT, cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY)
ok, c = find_board(frame())
if not ok and (ROOT / "run/board_frame.jpg").exists():      # arm or hand hides the sheet right now: use the saved frame (the sheet and the camera must NOT have moved since)
    ok, c = find_board(vision.decode((ROOT / "run/board_frame.jpg").read_bytes())); print("note: sheet hidden in the live picture - using the saved frame run/board_frame.jpg", flush=True)
if not ok: print("ERROR: checkerboard not found by the front camera - lay the printed sheet flat and fully visible", flush=True); sys.exit(1)
c = c.reshape(PAT[1], PAT[0], 2); cols, rows = np.meshgrid(np.arange(PAT[0]), np.arange(PAT[1]))                    # c[row, col] = pixel
flat = c.reshape(-1, 2); bxy = np.stack([cols.ravel() * SQ, rows.ravel() * SQ], 1).astype(np.float32)           # board metres
u, v = flat[:, 0], flat[:, 1]; idx = {"top-left": np.argmin(u + v), "top-right": np.argmax(u - v), "bottom-right": np.argmax(u + v), "bottom-left": np.argmin(u - v)}
order = ["top-left", "top-right", "bottom-right", "bottom-left"]
print("READY: checkerboard found. Touch the gripper tip to the INNER corners, in this order: " + ", ".join(order) + ". Hold each ~2 s.", flush=True)
hist = collections.deque(); touched = []; last_xyz = None; k = 0
_st = be.state(); start_xyz = np.array(kin.fk({i: norm_to_rad(i, j.norm) for i, j in _st.joints.items()}))      # pose at start: ignored
while k < 4:
    st = be.state(); n = {i: j.norm for i, j in st.joints.items()}; now = time.time(); hist.append((now, n)); [hist.popleft() for _ in range(len(hist)) if hist[0][0] < now - 1.6]; time.sleep(0.1)
    if now - hist[0][0] < 1.4: continue
    if max(max(h[1][i] for h in hist) - min(h[1][i] for h in hist) for i in n) > 0.004: continue
    xyz = np.array(kin.fk({i: norm_to_rad(i, x) for i, x in n.items()}))
    if start_xyz is not None and np.linalg.norm(xyz - start_xyz) < 0.05: continue                            # still in the pose it started in (not a touch)
    if last_xyz is not None:
        why = None
        if np.linalg.norm(xyz[:2] - last_xyz[:2]) < 0.09: why = f"only {np.linalg.norm(xyz[:2]-last_xyz[:2])*100:.0f} cm sideways from the last touch (corners are 12-20 cm apart) - move to the next corner"
        elif abs(xyz[2] - touched[0][2]) > 0.05: why = f"height differs from touch 1 by {abs(xyz[2]-touched[0][2])*100:.0f} cm - keep the tip ON the table"
        if why:
            if time.time() - globals().get('_w', 0) > 6: print("ignored: " + why, flush=True); globals()['_w'] = time.time()
            continue                              # same place as the previous touch
    touched.append(xyz); last_xyz = xyz; k += 1; print(f"touch {k}/4 recorded ({order[k-1]}): gripper at x={xyz[0]:.3f} y={xyz[1]:.3f} z={xyz[2]:.3f} m" + (f"   -> next: {order[k]} corner" if k < 4 else ""), flush=True); time.sleep(1.0)
B = np.array([bxy[idx[o]] for o in order], np.float32); R = np.array([t[:2] for t in touched], np.float32)
Hb2r, _ = cv2.findHomography(B, R, 0); Hp2b, _ = cv2.findHomography(flat.astype(np.float32), bxy, cv2.RANSAC, 0.5); Hp2r = Hb2r @ Hp2b
pred = cv2.perspectiveTransform(B.reshape(-1, 1, 2), Hb2r).reshape(-1, 2); err = np.linalg.norm(pred - R, axis=1) * 1000
side = np.linalg.norm(R[0] - R[1]), np.linalg.norm(R[3] - R[2]), np.linalg.norm(R[0] - R[3]), np.linalg.norm(R[1] - R[2]); want = (8 * SQ, 8 * SQ, 5 * SQ, 5 * SQ)
out = ROOT / "calibration/table_front.json"
out.write_text(json.dumps({"cam": FRONT, "name": "front", "H_pixel_to_robot_xy_m": Hp2r.tolist(), "undistorted": False, "touched_xyz": [t.tolist() for t in touched],
                           "residual_mm": err.tolist(), "board_side_robot_m": [float(x) for x in side], "board_side_true_m": list(want)}, indent=1))
print(f"saved {out.relative_to(ROOT)} | residual at the 4 touched corners (mm): {np.round(err, 1).tolist()} | board sides seen by the arm (m) {np.round(side, 3).tolist()} vs true {list(want)}", flush=True)
import os; os._exit(0)
