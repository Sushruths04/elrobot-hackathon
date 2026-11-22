#!/usr/bin/env python
"""Pick the magnetic DUSTER (magnet in its centre), put its centre on a metal bottle cap so the magnet takes the cap, lift, and carry duster+cap into the white bin.
 1. grasp the duster at its centre across its short side (search 4 mm steps; hold = jaws stall 1900..2560 ticks)   2. carry over the chosen cap (nearest to the bin), lower to 3 mm above the cap
 3. spiral search (+-8..12 mm) because the table mapping is ~1.5 cm: after each try go to the detection pose and look: cap still at its spot -> try the next offset, cap gone -> it is on the magnet
 4. carry over the bin, lower, release the duster (with the cap on it) into the bin, retreat, verify with the model.    scripts/duster_caps.py [--dry]   (needs ARM, waits)"""
import argparse, concurrent.futures as cf, json, math, sys, time, urllib.request
from pathlib import Path
import numpy as np, cv2
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from normakit import duster as dd, er2
ap = argparse.ArgumentParser(); ap.add_argument("--dry", action="store_true"); ap.add_argument("--speed", type=int, default=800); ap.add_argument("--duster-height", type=float, default=0.03); ap.add_argument("--caps", type=int, default=2, help="how many caps must end up on the magnet"); a = ap.parse_args()
T0 = time.time(); log = lambda *m: print(f"[{time.time()-T0:5.1f}s]", *m, flush=True)
def A(path, body=None, timeout=60):
    req = urllib.request.Request("http://127.0.0.1:8891" + path, data=None if body is None else json.dumps(body).encode(), method="GET" if body is None else "POST")
    with urllib.request.urlopen(req, timeout=timeout) as r: return json.loads(r.read())
def jpg(): return urllib.request.urlopen("http://127.0.0.1:8890/frame.jpg", timeout=10).read()
def frame_bgr(): return cv2.imdecode(np.frombuffer(jpg(), np.uint8), cv2.IMREAD_COLOR)
cal = json.loads((ROOT / "calibration/table_tip_front.json").read_text())["planes"]
def _hp(pl, u, v): return cv2.perspectiveTransform(np.float32([[[u, v]]]), np.array(cal[pl]["H_pixel_to_robot_xy_m"], np.float64))[0, 0]
def xy_low(u, v): return _hp("C", u, v)                                                      # plane measured 4 cm above the table: duster/caps are at 0.5-3 cm
def xy_box(u, v): return _hp("A", u, v)                                                   # the duster is released at tip height ~0.115: use the plane measured at 0.10 directly (extrapolating to the rim put the bin out of reach)
GENTLE = {"acc": 12}                                                                           # soft acceleration while carrying the duster (it slipped out at speed 800 / acc 60)
def must(r, w):
    if "error" in r: log(f"STOP at {w}: {r['error']}"); A("/stop", {}); sys.exit(2)
    return r
def roll_across(c, d):
    jaw = np.array([-d[1], d[0]]); psi = math.atan2(c[0], c[1]); roll = (math.atan2(jaw[1], jaw[0]) + psi + math.pi / 2) % math.pi - math.pi / 2; return roll, np.array([math.cos(roll - psi), math.sin(roll - psi)])
def choose_tilt(xy, roll, zs):
    for t in (0, 5, 10, 15, 20, 25, 30, 40):
        if all((lambda v: v["pos_err_mm"] <= 4 and v["ori_err_deg"] <= 4)(list(A("/ik", {"x": float(xy[0]), "y": float(xy[1]), "z": z, "tilt": t, "rolls": [roll]})["ik"].values())[0]) for z in zs): return t
    return None
carrying = [False]; det_pose = json.loads((ROOT / "run/detect_pose_follower.json").read_text()); WP = {"x": 0.0, "y": 0.20, "z": 0.26, "roll": 0.0, "speed": a.speed, "timeout": 20}
def to_detection():
    tp = A("/state")["tip"]
    if np.hypot(tp[0] - det_pose["tip"][0], tp[1] - det_pose["tip"][1]) < 0.04 and abs(tp[2] - det_pose["tip"][2]) < 0.04: return
    must(A("/move_tip", {**WP, "low_ok": True} if tp[2] < 0.06 else WP), "waypoint"); must(A("/move_joints", {"ticks": det_pose["raw"], "speed": 450 if carrying[0] else a.speed, "low_ok": True, "timeout": 45, "acc": 12 if carrying[0] else 60}), "detection pose"); time.sleep(0.3)
def scene():
    img = frame_bgr(); d = dd.detect(img)
    for sat in (50, 42, 35):                                                                       # a cap next to the duster or in shadow can be missed: relax until the expected number is found
        if len(d["caps"]) >= a.caps: break
        d2 = dd.detect(img, sat=sat)
        if len(d2["caps"]) > len(d["caps"]): d = d2
    du = d["duster"]
    if du is None or len(d["caps"]) == 0: return None
    u, v = du["center"]; th = math.radians(du["angle"]); h = du["long"] / 2; c = xy_low(u, v); p1 = xy_low(u + h * math.cos(th), v + h * math.sin(th)); p2 = xy_low(u - h * math.cos(th), v - h * math.sin(th)); dv = (p1 - p2) / max(np.linalg.norm(p1 - p2), 1e-9)
    return {"duster_c": c, "duster_d": dv, "duster_px": du["center"], "caps": [{"px": cp["center"], "xy": xy_low(*cp["center"])} for cp in d["caps"]]}
def bin_xy():
    jp = jpg()
    with cf.ThreadPoolExecutor(3) as ex: pts = [f.result()["points"][0] for f in [ex.submit(er2.point, jp, "the center of the open white box", 640, 480) for _ in range(3)]]
    return xy_box(float(np.median([p[1] for p in pts])), float(np.median([p[2] for p in pts])))
if not a.dry:
    t_w = time.time()
    while A("/state")["armed_s"] < 240:
        if time.time() - t_w > 150: sys.exit("NOT ARMED (need >= 240 s)")
        if int(time.time() - t_w) % 10 == 0: log("WAITING: press 'ARM motion' at http://127.0.0.1:8890"); time.sleep(1)
        time.sleep(0.4)
to_detection() if not a.dry else None
S = scene()
if S is None: sys.exit("ABORT: need the duster and at least one cap in view")
bxy = bin_xy(); caps = sorted(S["caps"], key=lambda c: np.hypot(*(c["xy"] - bxy))); cap = caps[0]
log(f"duster ({S['duster_c'][0]:+.3f},{S['duster_c'][1]:.3f}) axis {math.degrees(math.atan2(S['duster_d'][1], S['duster_d'][0])):.0f} deg | {len(caps)} caps, using the one at ({cap['xy'][0]:+.3f},{cap['xy'][1]:.3f}) nearest the bin | bin ({bxy[0]:+.3f},{bxy[1]:.3f})")
roll_g, uj = roll_across(S["duster_c"], S["duster_d"]); GZ = 0.016; HOV = GZ + 0.06
if a.dry: log(f"dry run: grasp roll {math.degrees(roll_g):.0f} deg, tilt {choose_tilt(S['duster_c'], roll_g, (HOV, GZ))}"); sys.exit(0)
held = False
DO = ROOT / "run/duster_offset.json"; b_saved = json.loads(DO.read_text())["b"] if DO.exists() else 0.0
for k, (da, db) in enumerate([(0, b_saved), (0, b_saved + 0.02), (0, b_saved - 0.02), (0, b_saved + 0.035), (0, b_saved - 0.035), (-0.004, b_saved), (0.004, b_saved), (-0.004, b_saved + 0.02), (0.004, b_saved - 0.02), (-0.008, b_saved), (0.008, b_saved), (0, b_saved + 0.05), (0, b_saved - 0.05)]):    # along-the-duster error is the big one near x>=0.12
    tgt = S["duster_c"] + da * uj + db * S["duster_d"]; tilt = choose_tilt(tgt, roll_g, (HOV, GZ))
    if tilt is None: log("no reachable tilt for the duster"); sys.exit(4)
    T = {"roll": roll_g, "tilt": tilt, "timeout": 15}; must(A("/gripper", {"to": 2840}), "open fully (5 cm)")
    must(A("/move_tip", {"x": float(tgt[0]), "y": float(tgt[1]), "z": HOV, "speed": a.speed, **T}), "hover"); must(A("/move_tip", {"x": float(tgt[0]), "y": float(tgt[1]), "z": GZ, "speed": 250, "low_ok": True, **T}), "descend")
    g = must(A("/gripper", {"to": "close"}), "close"); good = 1600 < g["gripper_ticks"] < 2560; log(f"   duster try {k+1}: across {da*1000:+.0f} mm, along {db*1000:+.0f} mm, jaws {g['gripper_ticks']} ticks {g['mA']} mA -> {'HOLD' if good else 'no good hold'}")
    if good:
        time.sleep(0.8); must(A("/move_tip", {"x": float(tgt[0]), "y": float(tgt[1]), "z": 0.14, "roll": roll_g, "speed": 350, "timeout": 20, "low_ok": True, **GENTLE}), "lift high")
        if 1600 < A("/state")["joints"]["8"]["raw"] < 2560: held = True; DO.write_text(json.dumps({"b": db, "note": "along-the-duster aim offset (m) that gave a hold"})); break
    must(A("/gripper", {"to": 2840}), "open fully"); must(A("/move_tip", {"x": float(tgt[0]), "y": float(tgt[1]), "z": HOV, "roll": roll_g, "tilt": tilt, "speed": 500, "timeout": 15, "low_ok": True}), "lift after miss")
    if k % 2 == 1:
        to_detection(); S = scene()
        if S is None: log("lost sight of the duster"); sys.exit(3)
        roll_g, uj = roll_across(S["duster_c"], S["duster_d"])
if not held: log("could not hold the duster"); A("/stop", {}); sys.exit(5)
log(f"duster held - holding it over each of the {len(caps)} caps (no sliding, no pressing), then a camera check, then the bin"); carrying[0] = True
holding = lambda: 1600 < A("/state")["joints"]["8"]["raw"] < 2560
VIEW = (-0.15, 0.30)                                                                                       # parking spot to the right of the table centre: the camera sees the caps while the duster waits there
def over_caps(targets, passno):
    global attached
    for ci, cap in enumerate(targets):
        Zp = a.duster_height / 2 + 0.006 + 0.012 + 0.006 * attached                                         # ~1.2 cm above the cap: the strong magnet takes it from this distance
        tl = choose_tilt(cap["xy"], roll_g, (Zp + 0.04, Zp)); T = {"roll": roll_g, "tilt": tl, "timeout": 25, **GENTLE}
        must(A("/move_tip", {"x": float(cap["xy"][0]), "y": float(cap["xy"][1]), "z": 0.12, "roll": roll_g, "speed": 450, "timeout": 45, "low_ok": True, **GENTLE}), f"pass {passno}: over cap {ci+1} (high)")
        if not holding(): log("   the duster slipped out of the jaws - stopping"); A("/stop", {}); sys.exit(7)
        must(A("/move_tip", {"x": float(cap["xy"][0]), "y": float(cap["xy"][1]), "z": Zp + 0.03, "speed": 300, **T}), "lower close above")
        must(A("/move_tip", {"x": float(cap["xy"][0]), "y": float(cap["xy"][1]), "z": Zp, "speed": 100, "low_ok": True, **T}), f"hold the duster over cap {ci+1}"); time.sleep(1.0)
        if not holding(): log("   the duster slipped out of the jaws - stopping"); A("/stop", {}); sys.exit(7)
        must(A("/move_tip", {"x": float(cap["xy"][0]), "y": float(cap["xy"][1]), "z": 0.12, "roll": roll_g, "speed": 300, "timeout": 25, "low_ok": True, **GENTLE}), "lift a little"); attached += 1
        log(f"   pass {passno}: held over cap {ci+1}/{len(targets)} (duster still held)")
attached = 0; remaining = sorted(caps, key=lambda c: np.hypot(*(c["xy"] - S["duster_c"])))[:a.caps]; left = remaining
for passno in (1, 2, 3):
    over_caps(remaining, passno)
    must(A("/move_tip", {"x": VIEW[0], "y": VIEW[1], "z": 0.16, "roll": roll_g, "speed": 450, "timeout": 45, "low_ok": True, **GENTLE}), "park to the side for the camera check"); time.sleep(1.2)
    if not holding(): log("   the duster slipped out of the jaws - stopping"); A("/stop", {}); sys.exit(7)
    left = dd.detect(frame_bgr())["caps"]                                                          # EVERY cap still on the table counts, not only the ones I aimed at
    log(f"   camera check after pass {passno}: {len(left)} cap(s) still on the table")
    if not left: break
    remaining = [{"px": c["center"], "xy": xy_low(*c["center"])} for c in left]
log(f"caps still on the table before the bin: {len(left)}")
if left: log("   WARNING: going to the bin although a cap is still on the table")
tb = bxy
must(A("/move_tip", {"x": float(tb[0]), "y": float(tb[1]), "z": 0.17, "roll": roll_g, "speed": 450, "timeout": 45, **GENTLE}), "over the bin (high)")
must(A("/move_tip", {"x": float(tb[0]), "y": float(tb[1]), "z": 0.115, "roll": roll_g, "speed": 250, "timeout": 20, **GENTLE}), "lower into the bin (above the rim)"); must(A("/gripper", {"to": "open"}), "release the duster with its caps into the bin"); time.sleep(0.5)
must(A("/move_tip", {"x": float(tb[0]), "y": float(tb[1]), "z": 0.17, "speed": 500, "timeout": 15, "low_ok": True}), "lift away"); to_detection(); time.sleep(0.5); cv2.imwrite(str(ROOT / "run/duster_result.jpg"), frame_bgr())
try: v = er2.ask(jpg(), 'Is the dark duster inside the white box, and how many metal bottle caps are still on the table? Reply {"duster_in_box": true or false, "caps_on_table": number}'); log("model check:", v)
except Exception as e: log("check failed:", str(e)[:60])
log("DONE - picture run/duster_result.jpg")
