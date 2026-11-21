#!/usr/bin/env python
"""Pick the RED brick and press it onto the YELLOW/ORANGE brick (Duplo-size). Colour detection (src/normakit/bricks.py) + the jaw-tip calibration + fast daemon.
  detect -> grasp across the short side (retry aim offsets, hold = gripper stall) -> lift high -> carry over the orange brick, long axes parallel
  -> refine over the orange (camera: held red vs orange) -> lower to touch -> press ~4.5 mm (studs) -> release -> press again from above with the closed jaws -> verify.
    scripts/stack_bricks.py --dry     detection + plan only (no motion)        scripts/stack_bricks.py     run (needs ARM)"""
import argparse, json, math, sys, time, urllib.request
from pathlib import Path
import numpy as np, cv2
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from normakit import bricks, er2
ap = argparse.ArgumentParser(); ap.add_argument("--dry", action="store_true"); ap.add_argument("--speed", type=int, default=800); ap.add_argument("--max", type=int, default=5); a = ap.parse_args()
T0 = time.time(); log = lambda *m: print(f"[{time.time()-T0:5.1f}s]", *m, flush=True)
def A(path, body=None, timeout=60):
    req = urllib.request.Request("http://127.0.0.1:8891" + path, data=None if body is None else json.dumps(body).encode(), method="GET" if body is None else "POST")
    with urllib.request.urlopen(req, timeout=timeout) as r: return json.loads(r.read())
def frame_bgr(): return cv2.imdecode(np.frombuffer(urllib.request.urlopen("http://127.0.0.1:8890/frame.jpg", timeout=10).read(), np.uint8), cv2.IMREAD_COLOR)
cal = json.loads((ROOT / "calibration/table_tip_front.json").read_text())["planes"]
def _hp(pl, u, v): return cv2.perspectiveTransform(np.float32([[[u, v]]]), np.array(cal[pl]["H_pixel_to_robot_xy_m"], np.float64))[0, 0]
def xy_at(z, u, v): a_, b_ = _hp("A", u, v), _hp("B", u, v); return a_ + (b_ - a_) * (z - 0.10) / (0.06 - 0.10)
def must(r, w):
    if "error" in r: log(f"STOP at {w}: {r['error']}"); A("/stop", {}); sys.exit(2)
    return r
# geometry of the bricks (Duplo 2x4: 6.4 x 3.2 cm, body 1.9 cm + studs 0.45 cm)
H_BODY, H_STUD, TOP_Z = 0.0192, 0.0045, 0.02
def brick_geom(b, z=TOP_Z):
    u, v = b["center"]; th = math.radians(b["angle"]); h = b["long"] / 2; c = xy_at(z, u, v)
    p1 = xy_at(z, u + h * math.cos(th), v + h * math.sin(th)); p2 = xy_at(z, u - h * math.cos(th), v - h * math.sin(th)); d = (p1 - p2) / max(np.linalg.norm(p1 - p2), 1e-9)
    return c, d, float(np.linalg.norm(p1 - p2)), b["short"] / b["long"] * float(np.linalg.norm(p1 - p2))
def roll_across(c, d):
    jaw = np.array([-d[1], d[0]]); psi = math.atan2(c[0], c[1]); roll = (math.atan2(jaw[1], jaw[0]) + psi + math.pi / 2) % math.pi - math.pi / 2; phi = roll - psi
    return roll, np.array([math.cos(phi), math.sin(phi)])
def choose_tilt(xy, roll, zs):
    for t in (0, 5, 10, 15, 20, 25, 30, 40):
        ok = True
        for z in zs:
            v = list(A("/ik", {"x": float(xy[0]), "y": float(xy[1]), "z": z, "tilt": t, "rolls": [roll]})["ik"].values())[0]
            if v["pos_err_mm"] > 4 or v["ori_err_deg"] > 4: ok = False; break
        if ok: return t
    return None
def pick_blobs():
    det = bricks.detect(frame_bgr())
    if not det["orange"] or not det["red"]: return None, None, det
    o = det["orange"][0]; ra = o["angle"]; red = sorted(det["red"], key=lambda b: (abs(((b["angle"] - ra + 90) % 180) - 90) > 25, math.hypot(b["center"][0] - o["center"][0], b["center"][1] - o["center"][1])))[0]
    return red, o, det
red, org, det = pick_blobs()
if red is None: sys.exit(f"ABORT: need one red and one yellow/orange brick in view, saw {len(det['red'])} red, {len(det['orange'])} orange")
cr, dr, Lr, Wr = brick_geom(red); co, do_, Lo, Wo = brick_geom(org)
log(f"red brick centre ({cr[0]:+.3f},{cr[1]:.3f}) {Lr*100:.1f} x {Wr*100:.1f} cm, long axis {math.degrees(math.atan2(dr[1], dr[0])):.0f} deg | orange brick centre ({co[0]:+.3f},{co[1]:.3f}) {Lo*100:.1f} x {Wo*100:.1f} cm, long axis {math.degrees(math.atan2(do_[1], do_[0])):.0f} deg")
if not (0.04 < Lr < 0.10 and 0.02 < Wr < 0.05): sys.exit("ABORT: red brick size is implausible for a Duplo 2x4")
if Wr > 0.042: sys.exit("ABORT: the red brick is wider than the jaw gap")
roll_g, uj_g = roll_across(cr, dr); roll_p, uj_p = roll_across(co, do_)
GZ = H_BODY / 2 + 0.002; HOV = GZ + 0.06; log(f"grasp roll {math.degrees(roll_g):.0f} deg, place roll {math.degrees(roll_p):.0f} deg, grasp tip z {GZ:.3f}")
if a.dry: log("dry run: stopping before any motion"); sys.exit(0)
t_w = time.time()
while A("/state")["armed_s"] < 150:
    if time.time() - t_w > 150: sys.exit("NOT ARMED (need >= 150 s)")
    if int(time.time() - t_w) % 10 == 0: log("WAITING: press 'ARM motion' at http://127.0.0.1:8890"); time.sleep(1)
    time.sleep(0.4)
det_pose = json.loads((ROOT / "run/detect_pose_follower.json").read_text()); WP = {"x": 0.0, "y": 0.20, "z": 0.26, "roll": 0.0, "speed": a.speed, "timeout": 20}
def to_detection():
    tp = A("/state")["tip"]
    if np.hypot(tp[0] - det_pose["tip"][0], tp[1] - det_pose["tip"][1]) < 0.04 and abs(tp[2] - det_pose["tip"][2]) < 0.04: return
    must(A("/move_tip", {**WP, "low_ok": True} if tp[2] < 0.05 else WP), "waypoint"); must(A("/move_joints", {"ticks": det_pose["raw"], "speed": a.speed, "low_ok": True, "timeout": 25}), "detection pose"); time.sleep(0.3)
OFF = ROOT / "run/grasp_offset.json"; a0 = json.loads(OFF.read_text())["a"] if OFF.exists() else -0.006
held = False
for k, da in enumerate((0.0, -0.008, 0.008, -0.016, 0.016)[:a.max]):
    to_detection(); red, org, det = pick_blobs()
    if red is None: log("lost sight of the bricks - stopping"); sys.exit(3)
    cr, dr, Lr, Wr = brick_geom(red); co, do_, Lo, Wo = brick_geom(org); roll_p, uj_p = roll_across(co, do_); roll_g, uj_g = roll_across(cr, dr); aa = a0 + da; tgt = cr + aa * uj_g; tilt = choose_tilt(tgt, roll_g, (HOV, GZ))
    if tilt is None: log("no reachable tilt for the grasp"); sys.exit(4)
    log(f"grasp attempt {k+1}: red at ({cr[0]:+.3f},{cr[1]:.3f}), aim offset {aa*100:+.1f} cm, tilt {tilt}")
    must(A("/gripper", {"to": "open"}), "open"); T = {"roll": roll_g, "tilt": tilt, "timeout": 15}
    must(A("/move_tip", {"x": float(tgt[0]), "y": float(tgt[1]), "z": HOV, "speed": a.speed, **T}), "hover"); must(A("/move_tip", {"x": float(tgt[0]), "y": float(tgt[1]), "z": GZ, "speed": 250, "low_ok": True, **T}), "descend")
    g = must(A("/gripper", {"to": "close"}), "close"); log(f"   gripper stalled at {g['gripper_ticks']} ticks, {g['mA']} mA -> {'HOLD' if g['closed_on_object'] else 'miss'}")
    if g["closed_on_object"]:
        time.sleep(1.0); must(A("/move_tip", {"x": float(tgt[0]), "y": float(tgt[1]), "z": 0.14, "roll": roll_g, "speed": a.speed, "timeout": 15, "low_ok": True}), "lift high")
        if A("/state")["joints"]["8"]["raw"] > 1360: held = True; log("   hold confirmed after the lift"); break
        log("   slipped during the lift")
    must(A("/gripper", {"to": "open"}), "open after miss"); must(A("/move_tip", {"x": float(tgt[0]), "y": float(tgt[1]), "z": HOV, "roll": roll_g, "tilt": tilt, "speed": 400, "timeout": 15, "low_ok": True}), "lift after miss")
if not held: log("could not hold the red brick"); A("/stop", {}); sys.exit(5)
grip = A("/state")["joints"]["8"]["raw"]; tgt_p = co + (a0) * uj_p; zt = GZ + (H_BODY + H_STUD) - H_BODY / 2 + 0.0   # tip z when the red bottom just touches the orange stud tops
z_touch = (H_BODY + H_STUD) + (H_BODY / 2 - 0.002) + 0.001; z_press = z_touch - 0.0045; z_over = z_touch + 0.035
log(f"carry over the orange brick ({tgt_p[0]:+.3f},{tgt_p[1]:.3f}); tip z touch {z_touch:.3f}, press {z_press:.3f}")
tiltp = choose_tilt(tgt_p, roll_p, (z_over, z_press))
if tiltp is None: log("no reachable tilt over the orange brick"); sys.exit(6)
must(A("/move_tip", {"x": float(tgt_p[0]), "y": float(tgt_p[1]), "z": 0.14, "roll": roll_p, "speed": a.speed, "timeout": 20}), "over the orange brick (high)")
must(A("/move_tip", {"x": float(tgt_p[0]), "y": float(tgt_p[1]), "z": z_over, "roll": roll_p, "tilt": tiltp, "speed": 500, "timeout": 15}), "lower close above")
must(A("/move_tip", {"x": float(tgt_p[0]), "y": float(tgt_p[1]), "z": z_touch + 0.004, "roll": roll_p, "tilt": tiltp, "speed": 150, "timeout": 15}), "lower to just above the studs")
must(A("/move_tip", {"x": float(tgt_p[0]), "y": float(tgt_p[1]), "z": z_touch, "roll": roll_p, "tilt": tiltp, "speed": 60, "timeout": 15, "low_ok": True}), "lower until the red brick just rests on the studs (NOT pressing yet)"); log("resting on the studs, releasing without pressing")
must(A("/gripper", {"to": "open"}), "release"); time.sleep(0.4)
must(A("/move_tip", {"x": float(tgt_p[0]), "y": float(tgt_p[1]), "z": 0.14, "speed": 500, "timeout": 15, "low_ok": True}), "lift away (normal tilt: a constant tilt cannot reach this height)")
time.sleep(0.5); must(A("/move_tip", {"x": float(tgt_p[0]), "y": float(tgt_p[1]), "z": 0.14, "speed": 500, "timeout": 15, "low_ok": True}), "lift clear (normal tilt)"); to_detection(); time.sleep(0.5)
d3 = bricks.detect(frame_bgr()); stacked_px = None
if d3["orange"] and d3["red"]:
    o3 = d3["orange"][0]; r3 = min(d3["red"], key=lambda b: math.hypot(b["center"][0] - o3["center"][0], b["center"][1] - o3["center"][1])); stacked_px = math.hypot(r3["center"][0] - o3["center"][0], r3["center"][1] - o3["center"][1])
log(f"after release: red centroid {stacked_px if stacked_px is None else round(stacked_px, 1)} px from the orange centroid (on top = under ~12 px)")
if stacked_px is not None and stacked_px < 14:
    z_top = 2 * H_BODY + H_STUD; z_push = z_top + 0.004; c3 = xy_at(z_top, *r3["center"])
    must(A("/gripper", {"to": "close"}), "close the jaws as a pusher"); must(A("/move_tip", {"x": float(c3[0]), "y": float(c3[1]), "z": z_top + 0.03, "roll": roll_p, "speed": 400, "timeout": 15}), "above the stack")
    must(A("/move_tip", {"x": float(c3[0]), "y": float(c3[1]), "z": z_push, "roll": roll_p, "speed": 80, "timeout": 15, "low_ok": True}), "press from above with the closed jaws"); log("pressed from above")
    must(A("/move_tip", {"x": float(c3[0]), "y": float(c3[1]), "z": 0.14, "speed": 500, "timeout": 15, "low_ok": True}), "lift"); must(A("/gripper", {"to": "open"}), "open"); to_detection(); time.sleep(0.4)
else: log("the red brick is not on top of the orange one: no second press")
cv2.imwrite(str(ROOT / "run/stack_result.jpg"), frame_bgr()); jp = urllib.request.urlopen("http://127.0.0.1:8890/frame.jpg").read()
try: v = er2.ask(jp, 'Is the red brick sitting on top of the yellow-orange brick (stacked, not beside it)? Reply {"stacked": true or false, "reason": short text}'); log("model check:", v.get("stacked"), "-", str(v.get("reason", ""))[:80])
except Exception as e: log("check failed:", str(e)[:70])
log("DONE - picture in run/stack_result.jpg")
