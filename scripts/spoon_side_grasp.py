"""Sideways spoon grasp, closed loop on the ORANGE handle colour (front camera to find it, wrist camera to align).
  1. front camera: orange handle -> grasp point (60 % up the handle) -> xy (table calibration) ; pre-grasp 6 cm closer to the robot base
  2. gripper at tilt 60 deg, roll 90 deg (jaws close ACROSS the handle - look test 13:11), jaws open
  3. wrist camera: handle line must pass through the jaw-gap centre -> lateral steps (image Jacobian from probe moves); then 1 cm steps forward until
     the handle fills the jaw gap
  4. close, wait for the jaws to stop, require the jaws to stay open > 0.08 (handle in between); lift 4 cm; front camera must see the orange handle rise
Prints DONE / STOP (film_task).   --check: segmentation + plan only."""
import argparse, json, math, sys, time, urllib.request
from pathlib import Path
import numpy as np, cv2
ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser(); ap.add_argument("--check", action="store_true"); ap.add_argument("--attempts", type=int, default=2); a = ap.parse_args()
T0 = time.time(); log = lambda *m: print(f"[{time.time() - T0:5.1f}s]", *m, flush=True)
A = lambda p, b=None, t=60: json.loads(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8891" + p, data=None if b is None else json.dumps(b).encode(), method="GET" if b is None else "POST"), timeout=t).read())
img = lambda c: cv2.imdecode(np.frombuffer(urllib.request.urlopen(f"http://127.0.0.1:8893/frame.jpg?c={c}", timeout=5).read(), np.uint8), 1)
cal = json.loads((ROOT / "calibration/table_tip_front.json").read_text())["planes"]
_hp = lambda pl, u, v: cv2.perspectiveTransform(np.float32([[[u, v]]]), np.array(cal[pl]["H_pixel_to_robot_xy_m"], np.float64))[0, 0]
xy_at = lambda z, u, v: _hp("A", u, v) + (_hp("B", u, v) - _hp("A", u, v)) * (z - 0.10) / (0.06 - 0.10)
TILT, ROLL, GAP = 60, 1.5708, np.array([320.0, 410.0])          # jaw-gap centre in the wrist image (look test)
def orange_mask(im, top_cut=0):
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV); m = cv2.inRange(hsv, np.array((0, 40, 120)), np.array((14, 230, 255))); m[:top_cut] = 0
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)); n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    if n < 2: return None
    i = 1 + int(np.argmax(st[1:, 4])); return (lab == i) if st[i, 4] > 120 else None
def handle_front():
    m = orange_mask(img("table"), top_cut=150)
    if m is None: return None
    ys, xs = np.nonzero(m); top = np.array([xs[np.argmin(ys)], ys.min()], float); bot = np.array([xs[np.argmax(ys)], ys.max()], float); return top, bot
def wrist_handle():
    """-> (x of the handle line at the jaw-gap row, pixels of handle inside the jaw-gap box, area) or None"""
    m = orange_mask(img("wrist"))
    if m is None: return None
    ys, xs = np.nonzero(m); area = len(xs)
    if np.ptp(ys) > 10: k, c = np.polyfit(ys, xs, 1); x_gap = k * GAP[1] + c          # x as a function of y (handle runs top -> bottom)
    else: x_gap = xs.mean()
    inbox = int(((np.abs(xs - GAP[0]) < 60) & (ys > GAP[1] - 50) & (ys < GAP[1] + 40)).sum()); return float(x_gap), inbox, area
def state(): return A("/state")
def tip(): return np.array(state()["tip"], float)
def move(p, roll=ROLL, tilt=TILT, speed=250): return A("/move_tip", {"x": float(p[0]), "y": float(p[1]), "z": float(p[2]), "roll": roll, "tilt": tilt, "speed": speed, "timeout": 15})

h = handle_front()
if h is None: log("STOP - orange handle not visible in the front camera"); sys.exit(1)
top, bot = h; G = bot + 0.6 * (top - bot); g = xy_at(0.10, *G); r = math.hypot(*g); u = g / r; lat = np.array([-u[1], u[0]])
g = g.astype(float); u = u.astype(float); lat = lat.astype(float); pre = np.r_[g - 0.06 * u, 0.13]; goal = np.r_[g, 0.10]
log(f"handle front px top {top} bottom {bot} -> grasp xy {np.round(g, 3)} (radial {r:.3f} m); pre-grasp {np.round(pre, 3)}")
ik = A("/ik", {"x": float(pre[0]), "y": float(pre[1]), "z": float(pre[2]), "tilt": TILT, "rolls": [ROLL]})["ik"]; ik2 = A("/ik", {"x": float(goal[0]), "y": float(goal[1]), "z": float(goal[2]), "tilt": TILT, "rolls": [ROLL]})["ik"]
log(f"IK pre {list(ik.values())[0]} | goal {list(ik2.values())[0]}")
if a.check: sys.exit(0)
log("WAITING: arm must be enabled (ARM motion button)")
while state()["armed_s"] < 60: time.sleep(1)
A("/mirror", {"on": False})
for attempt in range(1, a.attempts + 1):
    log(f"attempt {attempt}: pre-grasp, jaws open"); A("/gripper", {"to": "open"}); r1 = move(pre); log(f"  move: ok={r1.get('ok')} {r1.get('error') or ''}"); time.sleep(0.8)
    w = wrist_handle()
    if w is None: log("  handle not visible in the wrist camera"); continue
    log(f"  wrist: handle at x={w[0]:.0f} (gap {GAP[0]:.0f}), in-gap pixels {w[1]}, area {w[2]}")
    # image Jacobian for lateral motion: probe 1 cm sideways
    p0 = tip(); move(p0 + np.r_[0.01 * lat, 0]); time.sleep(0.6); w2 = wrist_handle(); move(p0); time.sleep(0.6)
    kx = (w2[0] - w[0]) / 0.01 if w2 else None; log(f"  probe: 1 cm sideways moves the handle {None if kx is None else round(kx * 0.01)} px in the wrist image")
    if not kx or abs(kx) < 300: log("  probe unclear - using no lateral correction"); kx = None
    pos = tip(); reached = False
    for step in range(14):
        w = wrist_handle()
        if w is None: log("  lost the handle in the wrist image"); break
        ex = w[0] - GAP[0]; log(f"  step {step}: handle x error {ex:+.0f} px, in-gap pixels {w[1]}, tip {np.round(pos, 3)}")
        if w[1] > 900: reached = True; log("  handle fills the jaw gap"); break
        if kx and abs(ex) > 15: pos = pos + np.r_[np.clip(-ex / kx, -0.01, 0.01) * lat, 0]
        else:
            fwd = goal - pos; d = np.linalg.norm(fwd)
            if d < 0.004 and step > 3: pos = pos + np.r_[0.008 * u, -0.004]      # past the estimate: keep creeping forward/down a little
            else: pos = pos + fwd * min(1.0, 0.01 / max(d, 1e-6))
        if pos[2] < 0.075: log("  too low - stop advancing"); break
        rm = move(pos, speed=150)
        if not rm.get("ok"): log(f"  move refused: {rm.get('error')}"); break
        time.sleep(0.5)
    if not reached: log("  handle not in the gap - closing anyway to test")
    A("/gripper", {"to": "close"}); last = None
    for _ in range(20):                                                       # wait until the jaws stop
        time.sleep(0.15); j8 = state()["joints"]["8"]
        if last is not None and abs(j8["raw"] - last) < 3: break
        last = j8["raw"]
    j8 = state()["joints"]["8"]; log(f"  jaws stopped at {j8['norm']:.2f} ({j8['raw']} ticks), current {j8['mA']} mA (empty closes to ~0.01)")
    fr0 = handle_front(); p = tip(); move(p + np.r_[0, 0, 0.04], speed=150); time.sleep(0.8); fr1 = handle_front()
    rose = None if (fr0 is None or fr1 is None) else float(fr0[0][1] - fr1[0][1]); log(f"  lift 4 cm: orange handle top moved up {rose} px in the front camera")
    fn = ROOT / f"results/photos/spoon_side_grasp_attempt{attempt}_{time.strftime('%H%M%S')}.jpg"; cv2.imwrite(str(fn), img("table")); log(f"  photo {fn.relative_to(ROOT)}")
    if j8["norm"] > 0.08 and rose is not None and rose > 8: log("DONE - SPOON GRASPED AND LIFTED (jaws held open by the handle, handle rose in the front camera)"); sys.exit(0)
    log("  not grasped - releasing, back to pre-grasp"); A("/gripper", {"to": "open"}); time.sleep(0.6); move(np.r_[tip()[:2], 0.14])
log("STOP - no grasp")
