#!/usr/bin/env python
"""Pick the pen and put it in the box - and KEEP TRYING until the gripper really holds it.
The jaw stalls at the pen's width (~1590 ticks) when it holds something and closes fully (~1322) on air, so 'held' is a SENSOR reading (instant, no model).
After each miss the aiming correction along the jaw axis (a, metres) is stepped through -1.6 cm (estimated from the photos), then +-1, +-2, +-3 cm around it.
The offset that worked is saved to run/grasp_offset.json and used first next time.   Needs ARM (waits).   .venv/bin/python scripts/pick_loop.py [--max 7]"""
import argparse, json, math, sys, time, urllib.request
from pathlib import Path
import numpy as np, cv2
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from normakit import er2
ap = argparse.ArgumentParser(); ap.add_argument("--max", type=int, default=7); ap.add_argument("--speed", type=int, default=800); ap.add_argument("--grasp-z", type=float, default=0.015); ap.add_argument("--hover", type=float, default=0.06); a = ap.parse_args()
T0 = time.time(); log = lambda *m: print(f"[{time.time()-T0:5.1f}s]", *m, flush=True)
def A(path, body=None, timeout=60):
    req = urllib.request.Request("http://127.0.0.1:8891" + path, data=None if body is None else json.dumps(body).encode(), method="GET" if body is None else "POST")
    with urllib.request.urlopen(req, timeout=timeout) as r: return json.loads(r.read())
frame = lambda: urllib.request.urlopen("http://127.0.0.1:8890/frame.jpg", timeout=10).read()
cal = json.loads((ROOT / "calibration/table_tip_front.json").read_text())["planes"]
def _hp(pl, u, v): return cv2.perspectiveTransform(np.float32([[[u, v]]]), np.array(cal[pl]["H_pixel_to_robot_xy_m"], np.float64))[0, 0]
def xy_at(z, u, v): a_, b_ = _hp("A", u, v), _hp("B", u, v); return a_ + (b_ - a_) * (z - 0.10) / (0.06 - 0.10)
def pen_geometry(m):
    """centre from the height-extrapolated positions (z=0.01); length and direction from the plane-B positions (the extrapolation amplifies calibration noise 2.3x and inflated a thin pen to 17-22 cm)."""
    Pz = [xy_at(0.01, x, y) for _, x, y in m[:2]]; Pb = [_hp("B", x, y) for _, x, y in m[:2]]
    L = float(np.linalg.norm(Pb[0] - Pb[1])); d = (Pb[1] - Pb[0]) / max(L, 1e-9); return (Pz[0] + Pz[1]) / 2, L, d
def must(r, w):
    if "error" in r: log(f"STOP at {w}: {r['error']}"); A("/stop", {}); sys.exit(2)
    return r
det = json.loads((ROOT / "run/detect_pose_follower.json").read_text())["raw"]; OFF = ROOT / "run/grasp_offset.json"
a0 = json.loads(OFF.read_text())["a"] if OFF.exists() else -0.016; seq = [a0] + [a0 + d for d in (-0.01, 0.01, -0.02, 0.02, -0.03, 0.03)]
t_w = time.time()
while A("/state")["armed_s"] < 120:
    if time.time() - t_w > 180: sys.exit("NOT ARMED (need >= 120 s) within 3 min")
    if int(time.time() - t_w) % 10 == 0: log("WAITING: press 'ARM motion' at http://127.0.0.1:8890"); time.sleep(1)
    time.sleep(0.4)
log("armed - going"); WP = {"x": 0.0, "y": 0.20, "z": 0.26, "roll": 0.0, "speed": a.speed, "timeout": 20}
det_tip = json.loads((ROOT / "run/detect_pose_follower.json").read_text())["tip"]
def to_detection():
    tp = A("/state")["tip"]
    if np.hypot(tp[0] - det_tip[0], tp[1] - det_tip[1]) < 0.04 and abs(tp[2] - det_tip[2]) < 0.04: return        # already there: no detour (saves ~5 s)
    low = A("/state")["tip"][2] < 0.05                                          # starting at the grasp height (e.g. after a stopped run): the path check would refuse, the move is upward
    must(A("/move_tip", {**WP, "low_ok": True} if low else WP), "waypoint"); must(A("/move_joints", {"ticks": det, "speed": a.speed, "low_ok": True, "timeout": 25}), "detection pose"); time.sleep(0.3)
import concurrent.futures as cf
def detect():
    for _ in range(3):
        jp = frame()
        with cf.ThreadPoolExecutor(4) as ex:
            fm = ex.submit(er2.point, jp, "the two ends of the single pen lying loose on the table OUTSIDE the white box (ignore any pen inside the box; one point per end)", 640, 480)
            fb = [ex.submit(er2.point, jp, "the center of the open white box", 640, 480) for _ in range(3)]      # 3 parallel asks: the box point varied by 4 cm between single calls
            m = fm.result()["points"]; bs = [f.result()["points"][0] for f in fb if f.result()["points"]]
        if len(m) >= 2 and bs: return m[:2], (bs[0][0], float(np.median([q[1] for q in bs])), float(np.median([q[2] for q in bs])))
    return None, None
def wait_for_pen(secs=150):
    t0 = time.time(); last = None; stable = 0
    while time.time() - t0 < secs:
        try: m, b = detect()
        except Exception: m = None
        if m is not None:
            c, L, _d = pen_geometry(m); bxy_ = _hp("A", b[1], b[2])
            ok = 0.05 < L < 0.17 and -0.22 < c[0] < 0.28 and 0.10 < c[1] < 0.40 and np.hypot(*(c - bxy_)) > 0.09
            stable = stable + 1 if ok and last is not None and np.hypot(*(c - last)) < 0.02 else 0; last = c if ok else None
            if stable >= 1: return
        time.sleep(1.5)
    log("no stable pen outside the box within 150 s - stopping"); sys.exit(7)
to_detection(); log("waiting for the pen to be placed (still, on the table, outside the box)..."); wait_for_pen(); t_run = time.time(); log("pen is placed - starting the timed run")
got = False
for k, aa in enumerate(seq[:a.max]):
    to_detection(); m, b = detect()
    if m is None: log("could not see the pen and the box - stopping"); sys.exit(3)
    c, L, d = pen_geometry(m); bxy = _hp("A", b[1], b[2])
    if not (0.05 < L < 0.17 and -0.22 < c[0] < 0.28 and 0.10 < c[1] < 0.40): log(f"implausible pen: centre ({c[0]:+.3f},{c[1]:.3f}) length {L*100:.1f} cm - stopping"); sys.exit(4)
    jaw = np.array([-d[1], d[0]]); psi = math.atan2(c[0], c[1]); roll = (math.atan2(jaw[1], jaw[0]) + psi + math.pi / 2) % math.pi - math.pi / 2; phi = roll - psi; uj = np.array([math.cos(phi), math.sin(phi)])
    tgt = c + aa * uj; tilt = None
    for t_ in (0, 5, 10, 15, 20, 25, 30, 40):
        ok = True
        for zz in (a.grasp_z + a.hover, a.grasp_z):
            v = list(A("/ik", {"x": float(tgt[0]), "y": float(tgt[1]), "z": zz, "tilt": t_, "rolls": [roll]})["ik"].values())[0]
            if v["pos_err_mm"] > 4 or v["ori_err_deg"] > 4: ok = False; break
        if ok: tilt = t_; break
    if tilt is None: log("no reachable tool tilt at this spot - move the pen closer/centre"); sys.exit(5)
    log(f"attempt {k+1}/{min(a.max, len(seq))}: pen ({c[0]:+.3f},{c[1]:.3f}) {L*100:.0f} cm, roll {math.degrees(roll):.0f}, aim offset a={aa*100:+.1f} cm along the jaw axis, tilt {tilt}")
    must(A("/gripper", {"to": "open"}), "open"); T = {"roll": roll, "tilt": tilt, "speed": a.speed, "timeout": 15}
    must(A("/move_tip", {"x": float(tgt[0]), "y": float(tgt[1]), "z": a.grasp_z + a.hover, **T}), "hover")
    must(A("/move_tip", {"x": float(tgt[0]), "y": float(tgt[1]), "z": a.grasp_z, **{**T, "speed": 250}, "low_ok": True}), "descent")
    g = must(A("/gripper", {"to": "close"}), "close"); log(f"   gripper stalled at {g['gripper_ticks']} ticks, {g['mA']} mA -> {'HOLDING SOMETHING' if g['closed_on_object'] else 'closed on air (miss)'}")
    if g["closed_on_object"]:
        time.sleep(2.0); must(A("/move_tip", {"x": float(tgt[0]), "y": float(tgt[1]), "z": 0.15, "roll": roll, "speed": a.speed, "timeout": 15, "low_ok": True}), "lift high"); g2 = A("/state")["joints"]["8"]["raw"]
        if g2 > 1360: got = True; OFF.write_text(json.dumps({"a": aa, "saved": time.strftime("%F %T"), "note": "aim correction along the jaw axis that produced a hold"})); log(f"HOLD CONFIRMED after lift (gripper {g2}); saved offset a={aa*100:+.1f} cm"); break
        log("   it slipped out during the lift - counting as a miss")
    must(A("/gripper", {"to": "open"}), "open after miss"); must(A("/move_tip", {"x": float(tgt[0]), "y": float(tgt[1]), "z": a.grasp_z + a.hover, **{**T, "speed": 400}, "low_ok": True}), "lift after miss")
if not got: log(f"gave up after {min(a.max, len(seq))} attempts"); A("/stop", {}); sys.exit(6)
bx, by = float(bxy[0]), float(bxy[1]); psi_b = math.atan2(bx, by); roll_rel = (-math.pi / 4 + psi_b + math.pi / 2) % math.pi - math.pi / 2     # jaw axis = pen axis + 90 deg; pen on the box diagonal (45 deg)
log(f"carrying to the box at ({bx:+.3f},{by:.3f}), release roll {math.degrees(roll_rel):.0f} deg (pen along the diagonal)")
must(A("/move_tip", {"x": bx, "y": by, "z": 0.19, "roll": roll_rel, "speed": a.speed, "timeout": 20}), "over the box (high, clears the walls)"); must(A("/move_tip", {"x": bx, "y": by, "z": 0.145, "roll": roll_rel, "speed": 500, "timeout": 15}), "lower into the box (stays above the rim)")
must(A("/gripper", {"to": "open"}), "release"); time.sleep(0.4); must(A("/move_tip", {"x": bx, "y": by, "z": 0.19, "roll": 0.0, "speed": a.speed, "timeout": 15}), "lift away"); to_detection()
try: v = er2.ask(frame(), 'Is the marker (pen) inside the white box now? Reply {"inside": true or false}'); log("model check inside box:", v.get("inside"))
except Exception as e: log("final check failed:", str(e)[:70])
log(f"DONE - {time.time()-t_run:.1f} s from the moment the pen was seen")
