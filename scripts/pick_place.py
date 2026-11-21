#!/usr/bin/env python
"""ZERO-SHOT pick the marker and put it in the white box, through the fast daemon (armd). No taught positions:
  detection pose -> vision model (run/vlm.json) finds the marker's two ends + the box -> calibrated pixel->xy (calibration/table_tip_front.json) ->
  IK with wrist roll across the marker -> hover -> visual check/correction (model sees tip + marker) -> descend -> close (+2 s) -> lift -> check held ->
  over the box -> release -> check inside -> back to the detection pose.   Needs ARM.   .venv/bin/python scripts/pick_place.py [--dry]"""
import argparse, json, math, sys, time, urllib.request
from pathlib import Path
import numpy as np, cv2
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from normakit import er2
ap = argparse.ArgumentParser(); ap.add_argument("--dry", action="store_true"); ap.add_argument("--grasp-z", type=float, default=0.015); ap.add_argument("--hover", type=float, default=0.06)
ap.add_argument("--diag", action="store_true", help="photograph hover and grasp height, then STOP before closing"); ap.add_argument("--align", action="store_true", help="visual tip-vs-marker correction (the tip pointing is unreliable: off by default, the calibration is accurate to <1 cm)"); ap.add_argument("--speed", type=int, default=700); ap.add_argument("--object", default="the marker (pen)"); a = ap.parse_args()
T0 = time.time(); log = lambda *m: print(f"[{time.time()-T0:5.1f}s]", *m, flush=True)
def A(path, body=None, timeout=60):
    req = urllib.request.Request("http://127.0.0.1:8891" + path, data=None if body is None else json.dumps(body).encode(), method="GET" if body is None else "POST")
    with urllib.request.urlopen(req, timeout=timeout) as r: return json.loads(r.read())
frame = lambda: urllib.request.urlopen("http://127.0.0.1:8890/frame.jpg", timeout=10).read()
cal = json.loads((ROOT / "calibration/table_tip_front.json").read_text())["planes"]
def _hp(plane, u, v): return cv2.perspectiveTransform(np.float32([[[u, v]]]), np.array(cal[plane]["H_pixel_to_robot_xy_m"], np.float64))[0, 0]
def px2xy(plane, u, v): return _hp(plane, u, v).tolist()
def px2xy_at(z, u, v):
    """Table position of what the camera sees at pixel (u,v) if that point is at height z. The two calibrated planes (z=0.10 'A', z=0.06 'B') give two points on the SAME viewing ray;
    xy is linear in z along a ray, so extrapolate to the real height (pen centre ~0.01 m, box rim ~0.05 m). Using plane B for the pen was ~5 cm off in y (perspective)."""
    a_, b_ = _hp("A", u, v), _hp("B", u, v); return (a_ + (b_ - a_) * (z - 0.10) / (0.06 - 0.10)).tolist()
def must(r, what):
    if "error" in r: log(f"STOP at {what}: {r['error']}"); A("/stop", {}); sys.exit(2)
    return r
if not a.dry:
    t_w = time.time()
    while A("/state")["armed_s"] < 90:
        if time.time() - t_w > 150: sys.exit("NOT ARMED within 150 s")
        if int(time.time() - t_w) % 10 == 0: log("WAITING: press 'ARM motion' at http://127.0.0.1:8890"); time.sleep(1)
        time.sleep(0.4)
det = json.loads((ROOT / "run/detect_pose_follower.json").read_text())["raw"]
if not a.dry: must(A("/move_joints", {"ticks": {k: v for k, v in det.items()}, "speed": a.speed, "low_ok": True, "timeout": 25}), "detection pose"); log("at the detection pose")
jp = frame(); m = er2.point(jp, f"the two ends of {a.object} lying on the table (one point per end)", 640, 480)["points"]; b = er2.point(jp, "the center of the white box", 640, 480)["points"]
log(f"detected marker ends {[(round(x), round(y)) for _, x, y in m]}, box {[(round(x), round(y)) for _, x, y in b]}")
if len(m) < 2 or not b: sys.exit("ABORT: need both marker ends and the box")
P = [np.array(px2xy_at(0.01, x, y)) for _, x, y in m[:2]]; mc = (P[0] + P[1]) / 2; L = float(np.linalg.norm(P[0] - P[1])); d = (P[1] - P[0]) / max(L, 1e-9); bx, by = px2xy_at(0.05, b[0][1], b[0][2])
log(f"marker centre ({mc[0]:+.3f},{mc[1]:.3f}) m, length {L*100:.1f} cm, direction {math.degrees(math.atan2(d[1], d[0])):.0f} deg | box ({bx:+.3f},{by:.3f})")
if not (0.04 < L < 0.22): sys.exit(f"ABORT: marker length {L*100:.1f} cm implausible")
for nm, (x, y) in (("marker", mc), ("box", (bx, by))):
    if not (-0.22 < x < 0.28 and 0.10 < y < 0.42): sys.exit(f"ABORT: {nm} at ({x:+.3f},{y:.3f}) outside the reachable area")
jaw = np.array([-d[1], d[0]]); psi = math.atan2(mc[0], mc[1]); roll = (math.atan2(jaw[1], jaw[0]) + psi + math.pi / 2) % math.pi - math.pi / 2
log(f"wrist roll {math.degrees(roll):.0f} deg (jaws across the marker)")
tilt = None                                                                       # TOP-TO-BOTTOM: one constant, as-vertical-as-reachable tool tilt for hover, descent and lift (the adaptive tilt changed by ~17 deg during the descent and swung the jaws into the pen)
for t_ in (0, 5, 10, 15, 20, 25, 30, 40):
    ok_t = True
    for zz in (a.grasp_z + a.hover, a.grasp_z):
        k = A("/ik", {"x": float(mc[0]), "y": float(mc[1]), "z": zz, "tilt": t_, "rolls": [roll]})["ik"]; v = list(k.values())[0]
        if v["pos_err_mm"] > 4 or v["ori_err_deg"] > 4: ok_t = False; break
    if ok_t: tilt = t_; break
if tilt is None: sys.exit("ABORT: no tool tilt (0-40 deg) reaches both the hover and the grasp height at this position")
log(f"constant tool tilt {tilt} deg for the whole hover -> descent -> lift (0 = straight down)")
if a.dry: log("dry run: stopping before any motion"); sys.exit(0)
must(A("/gripper", {"to": "open"}), "open gripper")
must(A("/move_tip", {"x": 0.0, "y": 0.20, "z": 0.26, "roll": 0.0, "speed": a.speed, "timeout": 20}), "transit waypoint (high, in front of the base)")
r = must(A("/move_tip", {"x": float(mc[0]), "y": float(mc[1]), "z": a.grasp_z + a.hover, "roll": roll, "tilt": tilt, "speed": a.speed, "timeout": 20}), "hover"); log(f"hover above marker: {r.get('arrived_in_ms')} ms, IK err {r.get('ik_err_mm')} mm, ori err {r.get('ori_err_deg')} deg, tilt {r.get('tilt')}")
if a.diag: time.sleep(0.6); open(ROOT / "run/diag_hover.jpg", "wb").write(frame()); log("saved run/diag_hover.jpg")
cx, cy = float(mc[0]), float(mc[1])
for it in range(3 if a.align else 0):                                            # visual check + correction with the calibrated map
    time.sleep(0.3); jp = frame(); tp = er2.point(jp, "the tip of the gripper jaws (the lowest point of the black gripper)", 640, 480)["points"]; mp = er2.point(jp, f"the center of {a.object}", 640, 480)["points"]
    if not tp or not mp: log("alignment: no tip/marker point - continuing without correction"); break
    off = np.array([mp[0][1] - tp[0][1], mp[0][2] - tp[0][2]]); log(f"alignment {it+1}: tip px ({tp[0][1]:.0f},{tp[0][2]:.0f}) marker px ({mp[0][1]:.0f},{mp[0][2]:.0f}) -> offset {off[0]:+.0f},{off[1]:+.0f} px")
    if np.hypot(*off) > 160: log("ABORT: tip and marker are far apart in the image - mapping or detection is wrong"); A("/stop", {}); sys.exit(3)
    if np.hypot(*off) < 14: log("aligned"); break
    u, v = mp[0][1], mp[0][2]; X0 = np.array(px2xy("B", u, v)); X1 = np.array(px2xy("B", u - off[0], v - off[1])); dxy = np.clip(X0 - X1, -0.03, 0.03)      # tip should move by (marker - tip) in table terms
    cx, cy = cx + float(dxy[0]), cy + float(dxy[1]); log(f"   correcting by dx={dxy[0]*100:+.1f} dy={dxy[1]*100:+.1f} cm"); must(A("/move_tip", {"x": cx, "y": cy, "z": a.grasp_z + a.hover, "roll": roll, "tilt": tilt, "speed": 400, "timeout": 15}), "correction")
must(A("/move_tip", {"x": cx, "y": cy, "z": a.grasp_z + 0.04, "roll": roll, "tilt": tilt, "speed": 350, "timeout": 15}), "descent 1"); r = must(A("/move_tip", {"x": cx, "y": cy, "z": a.grasp_z, "roll": roll, "tilt": tilt, "speed": 200, "timeout": 15, "low_ok": True}), "descent 2"); log(f"at grasp height ({r.get('arrived_in_ms')} ms), tip now {A('/state')['tip']}")
if a.diag: time.sleep(0.6); open(ROOT / "run/diag_grasp.jpg", "wb").write(frame()); log("saved run/diag_grasp.jpg - stopping before closing (holding here)"); A("/stop", {}); sys.exit(0)
must(A("/gripper", {"to": "close"}), "close"); time.sleep(2.0); log("gripper closed, waited 2 s")
must(A("/move_tip", {"x": cx, "y": cy, "z": a.grasp_z + a.hover, "roll": roll, "tilt": tilt, "speed": 350, "timeout": 15, "low_ok": True}), "lift")
try: h = er2.ask(frame(), f'Is {a.object} held in the robot gripper, lifted off the table? Reply {{"held": true or false, "reason": short text}}'); log(f"model check held: {h.get('held')} - {h.get('reason')}")
except Exception as e: h = {}; log("held-check failed, continuing:", str(e)[:70])
if h.get("held") is False: log("not held - stopping before the drop"); A("/stop", {}); sys.exit(4)
must(A("/move_tip", {"x": bx, "y": by, "z": 0.17, "roll": 0.0, "speed": a.speed, "timeout": 20}), "over the box"); must(A("/move_tip", {"x": bx, "y": by, "z": 0.12, "roll": 0.0, "speed": 300, "timeout": 15}), "descend into box")
must(A("/gripper", {"to": "open"}), "release"); time.sleep(1.0); must(A("/move_tip", {"x": bx, "y": by, "z": 0.17, "roll": 0.0, "speed": 400, "timeout": 15}), "lift away")
must(A("/move_tip", {"x": 0.0, "y": 0.20, "z": 0.26, "roll": 0.0, "speed": a.speed, "timeout": 20}), "transit waypoint back"); must(A("/move_joints", {"ticks": {k: v for k, v in det.items()}, "speed": a.speed, "low_ok": True, "timeout": 25}), "back to detection pose"); time.sleep(0.5)
try: v = er2.ask(frame(), f'Is {a.object} inside the white box now? Reply {{"inside": true or false, "reason": short text}}'); log(f"model check inside box: {v.get('inside')} - {v.get('reason')}")
except Exception as e: log("final check failed:", str(e)[:70])
log("DONE")
