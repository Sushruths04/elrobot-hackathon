"""Spoon grasp closed-loop in the FRONT camera only (user: always check from the front view).
  spoon  = orange handle (colour segmentation, measured with the arm out of view)
  gripper = black jaws against the white table/bowl: lowest dark pixel near the predicted spot = jaw tips (observed, not assumed)
  servo  = image Jacobian from two 1 cm probe moves; step until the jaw tips sit on the handle grasp point, first at z=0.13, then at grasp height
  grasp  = jaws open across the handle (tilt 60, roll 90), close, wait until stopped, resistance check, lift 4 cm, handle must rise in the front view
Prints DONE / STOP (film_task).  Saves a front photo with markers at every step in results/photos/front_servo_*."""
import json, math, sys, time, urllib.request
from pathlib import Path
import numpy as np, cv2
ROOT = Path(__file__).resolve().parents[1]
T0 = time.time(); log = lambda *m: print(f"[{time.time() - T0:5.1f}s]", *m, flush=True)
A = lambda p, b=None, t=60: json.loads(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8891" + p, data=None if b is None else json.dumps(b).encode(), method="GET" if b is None else "POST"), timeout=t).read())
front = lambda: cv2.imdecode(np.frombuffer(urllib.request.urlopen("http://127.0.0.1:8893/frame.jpg?c=table", timeout=5).read(), np.uint8), 1)
cal = json.loads((ROOT / "calibration/table_tip_front.json").read_text())["planes"]
Hinv = {k: np.linalg.inv(np.array(v["H_pixel_to_robot_xy_m"], float)) for k, v in cal.items()}
_hp = lambda pl, u, v: cv2.perspectiveTransform(np.float32([[[u, v]]]), np.array(cal[pl]["H_pixel_to_robot_xy_m"], np.float64))[0, 0]
_cp = json.loads((ROOT / "calibration/front_camera_pose_arm.json").read_text()); _cam = json.loads((ROOT / "calibration/camera_front.json").read_text())
_K = np.array(_cam["K"], float); _D = np.array(_cam["dist"], float).ravel(); _R = cv2.Rodrigues(np.array(_cp["rvec"], float))[0]; _C = np.array(_cp["camera_pos_robot_m"], float)
def xy_at(z, u, v):                               # pixel -> ray -> plane z (camera pose solved from the real arm, 13:30)
    d = cv2.undistortPoints(np.array([[[u, v]]], np.float64), _K, _D).reshape(2); ray = _R.T @ np.array([d[0], d[1], 1.0]); return (_C + (z - _C[2]) / ray[2] * ray)[:2].astype(float)
def project(p):                                   # predicted pixel of a 3D tip (only used to centre the search window)
    pa = cv2.perspectiveTransform(np.float32([[p[:2]]]), Hinv["A"])[0, 0]; pb = cv2.perspectiveTransform(np.float32([[p[:2]]]), Hinv["B"])[0, 0]; return pa + (pb - pa) * (p[2] - 0.10) / (0.06 - 0.10)
TILT, ROLL, Z_HI, Z_GRASP = 18, 1.3, 0.14, 0.12          # operator's grip from the demos: ~18 deg from vertical, j7 ~0.76
STEP = [0]
def orange(im):
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV); m = cv2.inRange(hsv, np.array((0, 40, 120)), np.array((14, 230, 255))); m[:150] = 0
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)); n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    if n < 2: return None
    i = 1 + int(np.argmax(st[1:, 4])); return (lab == i) if st[i, 4] > 120 else None
REF = {}
def jaw_tips(im, guess=None):
    """arm = what changed against the arm-out-of-view reference photo (beans/cup/handle are static); jaw tips = lowest point of the changed blob over the table"""
    if "im" not in REF: return None
    d = cv2.absdiff(cv2.GaussianBlur(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), (5, 5), 0), cv2.GaussianBlur(cv2.cvtColor(REF["im"], cv2.COLOR_BGR2GRAY), (5, 5), 0))
    m = (d > 35).astype(np.uint8); m[:150] = 0; m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    if n < 2: return None
    best = None
    for i in range(1, n):
        if st[i, 4] < 150: continue
        ys, xs = np.nonzero(lab == i); low = ys >= ys.max() - 4; p = np.array([xs[low].mean(), ys.max()], float)
        if guess is not None and np.linalg.norm(p - guess) > 60: continue            # tracking gate: the jaws cannot jump across the image
        if best is None or st[i, 4] > best[0]: best = (st[i, 4], p)
    return None if best is None else best[1]
def snap(im, G, J, note):
    STEP[0] += 1; vis = im.copy()
    if G is not None: cv2.circle(vis, (int(G[0]), int(G[1])), 7, (0, 255, 0), 2)
    if J is not None: cv2.circle(vis, (int(J[0]), int(J[1])), 7, (255, 255, 0), 2)
    cv2.putText(vis, note, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1); fn = ROOT / f"results/photos/front_servo_{time.strftime('%H%M%S')}_{STEP[0]:02d}.jpg"; cv2.imwrite(str(fn), vis); return fn
def st(): return A("/state")
def tip(): return np.array(st()["tip"], float)
def move(p, speed=200): return A("/move_tip", {"x": float(p[0]), "y": float(p[1]), "z": float(p[2]), "roll": ROLL, "tilt": TILT, "speed": speed, "timeout": 15})
LAST = {"J": None}
def look(p_guess=None):
    im = front(); J = jaw_tips(im, LAST["J"])
    if J is not None: LAST["J"] = J
    return im, J

log("WAITING: arm must be enabled (ARM motion button)")
while st()["armed_s"] < 60: time.sleep(1)
A("/mirror", {"on": False})
log("arm out of view for the spoon measurement")
FOLD = {"1": 2094, "2": 1055, "3": 1554, "4": 3020, "5": 170, "6": 2836, "7": 3021, "8": 2766}     # folded start pose (out of the camera's way)
r = A("/move_joints", {"ticks": FOLD, "speed": 300, "wait": True, "timeout": 25})
if not r.get("ok"):
    for stage in (("5", "6", "7", "8"), ("3", "4"), ("1", "2")): A("/move_joints", {"ticks": {k: FOLD[k] for k in stage}, "speed": 300, "wait": True, "timeout": 25})
time.sleep(1.0)
im = front(); REF["im"] = im; m = orange(im)
if m is None: log("STOP - orange handle not visible in the front camera"); sys.exit(1)
ys, xs = np.nonzero(m); top = np.array([xs[np.argmin(ys)], ys.min()], float); bot = np.array([xs[np.argmax(ys)], ys.max()], float)
def ray(u, v):
    d = cv2.undistortPoints(np.array([[[u, v]]], np.float64), _K, _D).reshape(2); r = _R.T @ np.array([d[0], d[1], 1.0]); return r / np.linalg.norm(r)
def proj(P): return cv2.projectPoints(np.array([P], np.float64), np.array(_cp["rvec"]), np.array(_cp["tvec"]), _K, _D)[0].reshape(2)
BEANS, HANDLE_L = 0.045, 0.095                     # operator: bean surface 4.5 cm above the table, handle sticks out 9-10 cm
rb = ray(*bot); B3 = _C + (BEANS - _C[2]) / rb[2] * rb
rt = ray(*top); w = _C - B3; bq = 2 * rt @ w; cq = w @ w - HANDLE_L ** 2; disc = bq * bq - 4 * cq
st_ = (-bq - np.sqrt(max(disc, 0))) / 2 if disc > 0 else -(rt @ w); T3 = _C + st_ * rt
if T3[2] < BEANS + 0.03: st_ = (-bq + np.sqrt(max(disc, 0))) / 2; T3 = _C + st_ * rt
P3 = B3 + 0.65 * (T3 - B3); Z_GRASP = float(np.clip(P3[2], 0.075, 0.14)); P3[2] = Z_GRASP; Z_HI = Z_GRASP + 0.03
G = proj(P3); G_HI = proj(P3 + np.array([0, 0, 0.03]))
log(f"handle 3D: base {np.round(B3, 3)} top {np.round(T3, 3)} -> grasp point {np.round(P3, 3)} (pixel {np.round(G, 1)}), approach height {Z_HI:.3f}")
log(f"handle top {top}, bottom {bot}, grasp point {np.round(G, 1)} px"); snap(im, G, None, "handle measured")
for attempt in range(1, 3):
    LAST["J"] = None; A("/gripper", {"to": "open"}); p = np.r_[P3[:2], Z_HI]; rm = move(p); log(f"attempt {attempt}: above the estimate {np.round(p, 3)} ok={rm.get('ok')} {rm.get('error') or ''}"); time.sleep(0.8)
    im, J = look(tip()); log(f"  jaw tips seen at {None if J is None else np.round(J, 1)} px (handle {np.round(G, 1)})"); snap(im, G, J, f"attempt {attempt} start")
    if J is None: log("  jaws not found in the front image"); continue
    # image Jacobian from probes
    p0 = tip(); cols = []
    for d in ((0.01, 0, 0), (0, 0.01, 0)):
        move(p0 + np.array(d)); time.sleep(0.6); _, Jd = look(tip()); move(p0); time.sleep(0.5)
        cols.append((Jd - J) / 0.01 if Jd is not None else np.array([np.nan, np.nan]))
    Jm = np.array(cols).T; log(f"  image Jacobian (px per m): {np.round(Jm).tolist()}")
    if not np.isfinite(Jm).all() or abs(np.linalg.det(Jm)) < 1e4: log("  probe failed"); continue
    z = Z_HI; ok = False; best = None; worse = 0
    for k in range(14):
        for _try in range(3):
            im, J = look()
            if J is not None: break
            time.sleep(0.4)
        if J is None: log("  lost the jaws (gate rejected 3 looks) - NOT moving"); break
        tgt = G_HI if z > Z_GRASP + 0.005 else G; e = tgt - J; snap(im, tgt, J, f"step {k} err {e[0]:+.0f},{e[1]:+.0f}px z={z:.3f}"); log(f"  step {k}: jaws {np.round(J, 1)} handle {np.round(G, 1)} error {np.round(e, 1)} px, z {z:.3f}")
        en = float(np.linalg.norm(e))
        if best is None or en < best[0]: best = (en, tip().copy()); worse = 0
        else: worse += 1
        if worse >= 2:                                   # getting worse: go back to the best pose and descend there
            log(f"  error growing - back to the best pose ({best[0]:.0f} px) and descending"); move(np.r_[best[1][:2], z], 150); time.sleep(0.5)
            move(np.r_[best[1][:2], Z_GRASP], 120); time.sleep(0.6); ok = True; break
        if en < 15:
            if z <= Z_GRASP + 1e-3: ok = True; break
            z = Z_GRASP; move(np.r_[tip()[:2], z], 120); time.sleep(0.6); best = None; worse = 0; continue
        dxy = np.linalg.solve(Jm, e); dxy *= min(1.0, 0.015 / max(np.linalg.norm(dxy), 1e-9)); p = tip(); p[:2] += dxy; p[2] = z
        rm = move(p, 150)
        if not rm.get("ok"): log(f"  move refused: {rm.get('error')}"); break
        time.sleep(0.6)
    if not ok:
        log("  could not align - not closing; lifting and retrying"); move(np.r_[tip()[:2], 0.16]); continue
    A("/gripper", {"to": "close"}); last = None
    for _ in range(25):
        time.sleep(0.15); j8 = st()["joints"]["8"]
        if last is not None and abs(j8["raw"] - last) < 3: break
        last = j8["raw"]
    j8 = st()["joints"]["8"]; log(f"  jaws stopped at {j8['norm']:.2f} ({j8['raw']} ticks), {j8['mA']} mA (empty = 1321-1326 ticks)")
    m0 = orange(front()); p = tip(); move(p + np.r_[0, 0, 0.04], 150); time.sleep(0.8); im = front(); m1 = orange(im)
    rose = None if (m0 is None or m1 is None) else float(np.nonzero(m0)[0].min() - np.nonzero(m1)[0].min()); snap(im, G, None, f"after lift: handle rose {rose}px")
    log(f"  lifted 4 cm: orange handle top rose {rose} px in the front view")
    if j8["raw"] > 1345 and rose is not None and rose > 8: log("DONE - SPOON GRASPED AND LIFTED (checked in the front view)"); sys.exit(0)
    log("  not grasped - opening, lifting"); A("/gripper", {"to": "open"}); time.sleep(0.6); move(np.r_[tip()[:2], 0.15])
log("STOP - no grasp")
