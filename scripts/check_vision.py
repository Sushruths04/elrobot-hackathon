"""Make two pictures to SEE the state of the vision system (read-only):
  run/check_calibration.jpg - the saved table mapping drawn on the board picture: 5 cm robot-frame grid (cyan), the 4 touched corners where the ARM says they are (red) vs where the CAMERA sees them (green), error in mm
  run/check_detection.jpg   - live front camera with every dark blob the detector sees; the chosen black object is green with its robot x,y
    .venv/bin/python scripts/check_vision.py"""
import json, sys, time
from pathlib import Path
import cv2, numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from normakit import vision
from normakit.station_backend import StationBackend
T = json.loads((ROOT / "calibration/table_front.json").read_text()); H = np.array(T["H_pixel_to_robot_xy_m"]); Hi = np.linalg.inv(H)
def to_px(x, y): p = Hi @ np.array([x, y, 1.0]); return (int(round(p[0] / p[2])), int(round(p[1] / p[2])))
def to_xy(u, v): p = H @ np.array([u, v, 1.0]); return (float(p[0] / p[2]), float(p[1] / p[2]))
# ---------- 1. calibration picture (uses the saved picture of the board: it may be removed from the table now)
img = cv2.cvtColor(vision.decode((ROOT / "calibration/board_frame_front.jpg").read_bytes()), cv2.COLOR_RGB2BGR); out = img.copy()
for x in np.arange(-0.15, 0.301, 0.05):
    cv2.line(out, to_px(x, 0.05), to_px(x, 0.50), (255, 255, 0), 1)
for y in np.arange(0.05, 0.501, 0.05):
    cv2.line(out, to_px(-0.15, y), to_px(0.30, y), (255, 255, 0), 1)
for x in np.arange(-0.15, 0.301, 0.10): cv2.putText(out, f"x{x:+.2f}", to_px(x, 0.50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
for y in np.arange(0.10, 0.501, 0.10): cv2.putText(out, f"y{y:.2f}", to_px(-0.15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
ok, c = cv2.findChessboardCornersSB(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (9, 6), cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY); flat = c.reshape(-1, 2)
u, v = flat[:, 0], flat[:, 1]; sel = {"top-left": np.argmin(u + v), "top-right": np.argmax(u - v), "bottom-right": np.argmax(u + v), "bottom-left": np.argmin(u - v)}
print("corner | pixel where the camera sees it | pixel where the arm says it was | gap in pixels | gap in mm (robot plane)")
for name, k in sel.items():
    seen = tuple(int(round(t)) for t in flat[k]); arm = T["touched"][name]["xyz"]; armpx = to_px(arm[0], arm[1]); sx, sy = to_xy(*seen)
    gap_mm = float(np.hypot(sx - arm[0], sy - arm[1]) * 1000)
    cv2.circle(out, seen, 6, (0, 255, 0), 2); cv2.circle(out, armpx, 6, (0, 0, 255), -1); cv2.line(out, seen, armpx, (0, 0, 255), 2)
    cv2.putText(out, f"{name} {gap_mm:.0f}mm", (seen[0] - 40, seen[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3); cv2.putText(out, f"{name} {gap_mm:.0f}mm", (seen[0] - 40, seen[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
    print(f"  {name:13s} {seen} {armpx} {np.hypot(seen[0]-armpx[0], seen[1]-armpx[1]):5.1f}px {gap_mm:5.1f}mm", flush=True)
cv2.putText(out, "cyan = 5 cm robot grid | green = camera | red = where the arm touched", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3); cv2.putText(out, "cyan = 5 cm robot grid | green = camera | red = where the arm touched", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
cv2.imwrite(str(ROOT / "run/check_calibration.jpg"), out)
# ---------- 2. live detection picture
be = StationBackend("localhost:8888", require_motors=False); be.start(); time.sleep(1.5)
rgb = vision.decode(be.frame_jpeg("cam1")); out2 = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR); hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
mask = (hsv[..., 2] < 70).astype(np.uint8) * 255; roi = np.zeros_like(mask); roi[185:470, 60:620] = 255; mask &= roi
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))); n, lab, st, cen = cv2.connectedComponentsWithStats(mask)
blobs = sorted([(int(st[i, 4]), [int(st[i, 0]), int(st[i, 1]), int(st[i, 0] + st[i, 2]), int(st[i, 1] + st[i, 3])]) for i in range(1, n) if st[i, 4] >= 100], reverse=True)
print(f"\nlive detection: {len(blobs)} dark blob(s) in the table region", flush=True)
for k, (a, b) in enumerate(blobs[:5]):
    bx, by = (b[0] + b[2]) / 2, b[3]; x, y = to_xy(bx, by); col = (0, 255, 0) if k == 0 else (0, 165, 255)
    cv2.rectangle(out2, (b[0], b[1]), (b[2], b[3]), col, 2); cv2.circle(out2, (int(bx), int(by)), 4, (0, 0, 255), -1)
    cv2.putText(out2, f"#{k+1} area {a} -> x={x:.2f} y={y:.2f}m", (b[0], max(14, b[1] - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3); cv2.putText(out2, f"#{k+1} area {a} -> x={x:.2f} y={y:.2f}m", (b[0], max(14, b[1] - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
    print(f"  #{k+1}: area {a}, box {b}, contact pixel ({bx:.0f},{by:.0f}) -> robot x={x:.3f} y={y:.3f} m", flush=True)
cv2.imwrite(str(ROOT / "run/check_detection.jpg"), out2); import os; os._exit(0)
