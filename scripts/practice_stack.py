#!/usr/bin/env python
"""PRACTICE stacking the red Duplo brick on the yellow/orange one, and LEARN from every round.
Round: grasp the red brick -> carry -> lower until it rests on the studs -> release (no early press) -> retreat -> measure with the camera how far the red brick is from the
orange one (both heights respected) -> update the persistent place correction run/place_corr.json -> take the red brick off again and put it back on the table spot
(automatic reset) -> next round. The LAST round also presses from above with the closed jaws if the brick sits on top.
    scripts/practice_stack.py --rounds 3        (needs ARM, >= 400 s recommended)"""
import argparse, json, math, sys, time, urllib.request
from pathlib import Path
import numpy as np, cv2
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from normakit import bricks
ap = argparse.ArgumentParser(); ap.add_argument("--rounds", type=int, default=3); ap.add_argument("--speed", type=int, default=800); a = ap.parse_args()
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
H_BODY, H_STUD, TOP_Z = 0.0192, 0.0045, 0.02
def geom(b, z=TOP_Z):
    u, v = b["center"]; th = math.radians(b["angle"]); h = b["long"] / 2; c = xy_at(z, u, v)
    p1 = xy_at(z, u + h * math.cos(th), v + h * math.sin(th)); p2 = xy_at(z, u - h * math.cos(th), v - h * math.sin(th)); return c, (p1 - p2) / max(np.linalg.norm(p1 - p2), 1e-9)
def roll_across(c, d):
    jaw = np.array([-d[1], d[0]]); psi = math.atan2(c[0], c[1]); roll = (math.atan2(jaw[1], jaw[0]) + psi + math.pi / 2) % math.pi - math.pi / 2; phi = roll - psi; return roll, np.array([math.cos(phi), math.sin(phi)])
def choose_tilt(xy, roll, zs):
    for t in (0, 5, 10, 15, 20, 25, 30, 40):
        ok = True
        for z in zs:
            v = list(A("/ik", {"x": float(xy[0]), "y": float(xy[1]), "z": z, "tilt": t, "rolls": [roll]})["ik"].values())[0]
            if v["pos_err_mm"] > 4 or v["ori_err_deg"] > 4: ok = False; break
        if ok: return t
    return None
def blobs():
    det = bricks.detect(frame_bgr())
    if not det["orange"] or not det["red"]: return None, None
    o = det["orange"][0]; ra = o["angle"]
    red = sorted(det["red"], key=lambda b: (abs(((b["angle"] - ra + 90) % 180) - 90) > 25, math.hypot(b["center"][0] - o["center"][0], b["center"][1] - o["center"][1])))[0]; return red, o
det_pose = json.loads((ROOT / "run/detect_pose_follower.json").read_text()); WP = {"x": 0.0, "y": 0.20, "z": 0.26, "roll": 0.0, "speed": a.speed, "timeout": 20}
def to_detection():
    tp = A("/state")["tip"]
    if np.hypot(tp[0] - det_pose["tip"][0], tp[1] - det_pose["tip"][1]) < 0.04 and abs(tp[2] - det_pose["tip"][2]) < 0.04: return
    must(A("/move_tip", {**WP, "low_ok": True} if tp[2] < 0.06 else WP), "waypoint"); must(A("/move_joints", {"ticks": det_pose["raw"], "speed": a.speed, "low_ok": True, "timeout": 25}), "detection pose"); time.sleep(0.3)
OFF = ROOT / "run/grasp_offset.json"; a0 = json.loads(OFF.read_text())["a"] if OFF.exists() else -0.006; CF = ROOT / "run/place_corr.json"
corr = np.array(json.loads(CF.read_text())["xy"]) if CF.exists() else np.array([0.016, -0.012])        # first guess from the photo of the last attempt
def grasp(red, z_g, ticks_min=1700):
    """-> (held, tip_xy_used, roll). Brick short side 3.2 cm: a real hold stalls the jaws at ~1980 ticks; ~1390 means only an edge was caught."""
    c, d = geom(red); roll, uj = roll_across(c, d); hov = z_g + 0.06
    for k, da in enumerate((0.0, -0.008, 0.008, -0.016)):
        tgt = c + (a0 + da) * uj; tilt = choose_tilt(tgt, roll, (hov, z_g))
        if tilt is None: log("no reachable tilt"); return False, tgt, roll
        T = {"roll": roll, "tilt": tilt, "timeout": 15}; must(A("/gripper", {"to": "open"}), "open")
        must(A("/move_tip", {"x": float(tgt[0]), "y": float(tgt[1]), "z": hov, "speed": a.speed, **T}), "hover"); must(A("/move_tip", {"x": float(tgt[0]), "y": float(tgt[1]), "z": z_g, "speed": 250, "low_ok": True, **T}), "descend")
        g = must(A("/gripper", {"to": "close"}), "close"); good = g["gripper_ticks"] > ticks_min; log(f"   grasp try {k+1}: offset {(a0+da)*100:+.1f} cm, jaws {g['gripper_ticks']} ticks {g['mA']} mA -> {'HOLD' if good else 'no good hold'}")
        if good:
            time.sleep(0.8); must(A("/move_tip", {"x": float(tgt[0]), "y": float(tgt[1]), "z": 0.14, "roll": roll, "speed": a.speed, "timeout": 15, "low_ok": True}), "lift high")
            if A("/state")["joints"]["8"]["raw"] > ticks_min: return True, tgt, roll
        must(A("/gripper", {"to": "open"}), "open"); must(A("/move_tip", {"x": float(tgt[0]), "y": float(tgt[1]), "z": hov, "roll": roll, "tilt": tilt, "speed": 400, "timeout": 15, "low_ok": True}), "lift after miss")
        to_detection(); red, _o = blobs(); c, d = geom(red); roll, uj = roll_across(c, d)
    return False, tgt, roll
def put_down(xy, roll, z_rest, hover_z=0.14):
    tilt = choose_tilt(xy, roll, (z_rest + 0.03, z_rest))
    must(A("/move_tip", {"x": float(xy[0]), "y": float(xy[1]), "z": hover_z, "roll": roll, "speed": a.speed, "timeout": 20}), "over the spot (high)")
    must(A("/move_tip", {"x": float(xy[0]), "y": float(xy[1]), "z": z_rest + 0.035, "roll": roll, "tilt": tilt, "speed": 500, "timeout": 15}), "lower close above")
    must(A("/move_tip", {"x": float(xy[0]), "y": float(xy[1]), "z": z_rest, "roll": roll, "tilt": tilt, "speed": 60, "timeout": 15, "low_ok": True}), "lower until resting"); must(A("/gripper", {"to": "open"}), "release"); time.sleep(0.4)
    must(A("/move_tip", {"x": float(xy[0]), "y": float(xy[1]), "z": 0.14, "speed": 500, "timeout": 15, "low_ok": True}), "lift away (normal tilt)")
t_w = time.time()
while A("/state")["armed_s"] < 200:
    if time.time() - t_w > 150: sys.exit("NOT ARMED (need >= 200 s)")
    if int(time.time() - t_w) % 10 == 0: log("WAITING: press 'ARM motion' at http://127.0.0.1:8890"); time.sleep(1)
    time.sleep(0.4)
to_detection(); red0, org0 = blobs()
if red0 is None: sys.exit("ABORT: need a red and an orange brick in view")
c_r0, d_r0 = geom(red0); roll0, uj0 = roll_across(c_r0, d_r0); co, do_ = geom(org0); roll_p, uj_p = roll_across(co, do_); org_px = org0["center"]
z_touch = (H_BODY + H_STUD) + (H_BODY / 2 - 0.002) + 0.001; z_table = H_BODY / 2 + 0.002; z_stack_grasp = (H_BODY + H_STUD) + (H_BODY / 2 - 0.002) + 0.002
log(f"home spot of the red brick ({c_r0[0]:+.3f},{c_r0[1]:.3f}); orange at ({co[0]:+.3f},{co[1]:.3f}); place correction {corr*100} cm"); errs = []
for rd in range(1, a.rounds + 1):
    log(f"=== round {rd}/{a.rounds}: place correction {corr[0]*100:+.1f},{corr[1]*100:+.1f} cm ===")
    to_detection(); red, org = blobs()
    if red is None: log("lost the bricks"); sys.exit(3)
    held, tgt, roll_g = grasp(red, z_table)
    if not held: log("could not hold the red brick"); A("/stop", {}); sys.exit(4)
    tgt_p = co + a0 * uj_p + corr; must(A("/move_tip", {"x": float(tgt_p[0]), "y": float(tgt_p[1]), "z": 0.14, "roll": roll_p, "speed": a.speed, "timeout": 20}), "over the orange brick (high)")
    tiltp = choose_tilt(tgt_p, roll_p, (z_touch + 0.035, z_touch)); must(A("/move_tip", {"x": float(tgt_p[0]), "y": float(tgt_p[1]), "z": z_touch + 0.035, "roll": roll_p, "tilt": tiltp, "speed": 500, "timeout": 15}), "lower close above")
    must(A("/move_tip", {"x": float(tgt_p[0]), "y": float(tgt_p[1]), "z": z_touch, "roll": roll_p, "tilt": tiltp, "speed": 60, "timeout": 15, "low_ok": True}), "lower until it rests on the studs"); must(A("/gripper", {"to": "open"}), "release"); time.sleep(0.4)
    must(A("/move_tip", {"x": float(tgt_p[0]), "y": float(tgt_p[1]), "z": 0.14, "speed": 500, "timeout": 15, "low_ok": True}), "lift away"); to_detection(); time.sleep(0.5)
    rr, _o = (lambda d: (min(d["red"], key=lambda b: math.hypot(b["center"][0] - org_px[0], b["center"][1] - org_px[1])) if d["red"] else None, None))(bricks.detect(frame_bgr()))
    if rr is None: log("cannot see the red brick after the release"); sys.exit(5)
    c_red = xy_at(0.04, *rr["center"]); c_org = xy_at(TOP_Z, *org_px); e = c_red - c_org; errs.append(float(np.hypot(*e)))
    if np.hypot(*e) > 0.06: log(f"   measurement not believable ({np.hypot(*e)*100:.1f} cm: another red brick in view, or the brick is not near the orange one) - NOT learning from it, stopping"); A("/stop", {}); sys.exit(7)
    log(f"   RESULT round {rd}: red centroid is {e[0]*100:+.1f} cm (x) {e[1]*100:+.1f} cm (y) from the orange centroid = {np.hypot(*e)*100:.1f} cm off")
    cv2.imwrite(str(ROOT / f"run/practice_round{rd}.jpg"), frame_bgr())
    corr = corr - 0.8 * e; CF.write_text(json.dumps({"xy": [float(corr[0]), float(corr[1])], "updated": time.strftime("%F %T"), "errors_cm": [round(x * 100, 1) for x in errs]})); log(f"   learned place correction -> {corr[0]*100:+.1f},{corr[1]*100:+.1f} cm")
    if rd == a.rounds:
        if np.hypot(*e) < 0.02:
            c3 = xy_at(2 * H_BODY + H_STUD, *rr["center"]); z_top = 2 * H_BODY + H_STUD
            must(A("/gripper", {"to": "close"}), "close as a pusher"); must(A("/move_tip", {"x": float(c3[0]), "y": float(c3[1]), "z": z_top + 0.03, "roll": roll_p, "speed": 400, "timeout": 15}), "above the stack")
            must(A("/move_tip", {"x": float(c3[0]), "y": float(c3[1]), "z": z_top + 0.002, "roll": roll_p, "speed": 70, "timeout": 15, "low_ok": True}), "press from above"); log("pressed from above")
            must(A("/move_tip", {"x": float(c3[0]), "y": float(c3[1]), "z": 0.14, "speed": 500, "timeout": 15, "low_ok": True}), "lift"); must(A("/gripper", {"to": "open"}), "open"); to_detection()
        else: log("last round: not pressing, the brick is more than 2 cm off the orange one")
        break
    red, org = blobs(); held, tgt, roll_s = grasp(red, z_stack_grasp)                                   # automatic reset: take it off the stack
    if not held: log("could not take the red brick back off the stack - please reset it by hand"); A("/stop", {}); sys.exit(6)
    put_down(c_r0 + a0 * uj0, roll0, z_table); to_detection()
log("SUMMARY: errors per round (cm):", [round(x * 100, 1) for x in errs]); cv2.imwrite(str(ROOT / "run/practice_final.jpg"), frame_bgr())
