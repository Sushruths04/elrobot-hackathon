"""Pick the MARKER lying on the table and put it into the white BOX. Everything located by Gemini Robotics-ER 2 from the table camera.
 perception: ER-2 points at both ends of the marker (-> position + direction) and at the box (-> drop point), via the saved table mapping (+-3 cm)
 planning:   7-DoF jaw-tip IK (kin.ik_jaw_tip, wrist roll chosen so the jaws close ACROSS the marker); every segment checked with forward kinematics
 execution:  proportional joint-space moves, torque on, timeout/high-current -> hold and STOP the whole script. Logs go to stdout (shown in the web UI).
    scripts/grab_marker.py            plan only (no motion)        scripts/grab_marker.py --go   execute (hand near arm, power plug in reach)"""
import argparse, json, math, sys, time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from normakit import vision, er2
from normakit.core import norm_to_rad, rad_to_norm
from normakit.kin import Kin
from normakit.station_backend import StationBackend, front_camera, REG_TORQUE, REG_GOAL_POS, REG_GOAL_SPEED, REG_ACC
ap = argparse.ArgumentParser(); ap.add_argument("--go", action="store_true"); ap.add_argument("--speed", type=int, default=250); ap.add_argument("--hover", type=float, default=0.10)
ap.add_argument("--grasp-z", type=float, default=0.012, help="jaw-tip height at the grasp (table = 0)"); ap.add_argument("--release-z", type=float, default=0.12)
ap.add_argument("--object", default="the marker"); ap.add_argument("--target", default="the white box"); a = ap.parse_args()
J = list(range(1, 9)); wd = lambda x, y: ((x - y + 2048) % 4096) - 2048; T0 = time.time(); log = lambda *m: print(f"[{time.time()-T0:5.1f}s]", *m, flush=True)
be = StationBackend("localhost:8888"); be.start(); time.sleep(1.5); kin = Kin(); st = be.state(); CAM = front_camera(be)
def to_norm(i, t):
    j = st.joints[i]; span = (j.hi - j.lo) % 4096; rel = (t - j.lo) % 4096
    if rel > span + (4096 - span) / 2: rel -= 4096
    return rel / span
def to_ticks(i, n): j = st.joints[i]; return int(round(j.lo + n * ((j.hi - j.lo) % 4096))) % 4096
q_of = lambda t: {i: norm_to_rad(i, to_norm(i, t[i])) for i in range(1, 8)}
def ticks_of(q, base): return {**{i: base[i] for i in J}, **{i: to_ticks(i, rad_to_norm(i, q[i])) for i in range(1, 8)}}
tip_of = lambda t: kin.jaw_tip_pose(q_of(t))[0]
# ---- perception (ER-2 only)
jpg = be.frame_jpeg(CAM); rgb = vision.decode(jpg); h, w = rgb.shape[:2]; H = vision.Homography.load(str(ROOT / "calibration/table_front.json"))
r1 = er2.point(jpg, f"the two ends of {a.object} lying on the table (one point per end)", w, h); r2 = er2.point(jpg, f"the center of {a.target}", w, h)
log(f"ER-2 marker ends ({r1['seconds']} s): {[(round(x), round(y)) for _, x, y in r1['points']]}   box ({r2['seconds']} s): {[(round(x), round(y)) for _, x, y in r2['points']]}")
import cv2
im = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
for lab, x, y in r1["points"]: cv2.circle(im, (int(x), int(y)), 6, (255, 0, 255), 2)
for lab, x, y in r2["points"]: cv2.circle(im, (int(x), int(y)), 9, (0, 200, 255), 3)
cv2.imwrite(str(ROOT / "run/grab_target.jpg"), im)
if len(r1["points"]) < 2: sys.exit("ABORT: ER-2 did not return both ends of the marker (need the direction to turn the wrist)")
if not r2["points"]: sys.exit("ABORT: ER-2 did not find the box")
P = [np.array(H(x, y), float) for _, x, y in r1["points"][:2]]; mc = (P[0] + P[1]) / 2; L = float(np.linalg.norm(P[0] - P[1])); mdir = (P[1] - P[0]) / max(L, 1e-9)
bx, by = H(r2["points"][0][1], r2["points"][0][2])
log(f"marker centre x={mc[0]:.3f} y={mc[1]:.3f} m, length {L*100:.1f} cm, direction {math.degrees(math.atan2(mdir[1], mdir[0])):.0f} deg | box x={bx:.3f} y={by:.3f} m")
if not (0.03 < L < 0.22): sys.exit(f"ABORT: marker length {L*100:.1f} cm is implausible (ER-2 points wrong?)")
for nm, (px, py) in (("marker", mc), ("box", (bx, by))):
    if not (-0.20 < px < 0.25 and 0.10 < py < 0.40): sys.exit(f"ABORT: {nm} at ({px:.3f},{py:.3f}) is outside the reachable table area")
# ---- planning
cur = {i: st.joints[i].raw for i in J}; qc = q_of(cur); tip0 = tip_of(cur); log(f"jaw tip now (from joints): x={tip0[0]:.3f} y={tip0[1]:.3f} z={tip0[2]:.3f}")
def roll_for(px, py, jaw_dir):                                  # jaws open along gripper-base x; make that axis perpendicular to the marker
    psi = math.atan2(px, py); phi = math.atan2(jaw_dir[1], jaw_dir[0]); r = (phi + psi + math.pi / 2) % math.pi - math.pi / 2; return r
jaw = np.array([-mdir[1], mdir[0]]); roll = roll_for(mc[0], mc[1], jaw)
def plan(x, y, z, roll, seed, name):
    q, pe, oe, tilt = kin.ik_jaw_tip((x, y, z), roll_rad=roll, seed_rad=seed); R = kin.jaw_tip_pose(q)[1]; jx = R[:, 0]; jy = math.degrees(math.atan2(jx[1], jx[0]))
    log(f"  IK {name}: tip z={z:.3f}  pos err {pe*100:.1f} cm, orientation err {oe:.0f} deg, tilt {tilt:.0f} deg, jaw axis yaw {jy:.0f} deg"); return q, pe, oe
seed = qc; steps = {}
for nm, z in (("hover", a.grasp_z + a.hover), ("mid", a.grasp_z + 0.04), ("grasp", a.grasp_z)):
    q, pe, oe = plan(mc[0], mc[1], z, roll, seed, f"marker {nm}"); steps[nm] = (q, pe, oe); seed = q
want = math.degrees(math.atan2(jaw[1], jaw[0])); log(f"  wanted jaw axis yaw (across the marker): {want:.0f} deg (mod 180)")
qb, peb, oeb = plan(bx, by, a.release_z + 0.05, 0.0, steps["hover"][0], "box above"); qd, ped, oed = plan(bx, by, a.release_z, 0.0, qb, "box release")
bad = [n for n, (q, pe, oe) in list(steps.items()) + [("box above", (qb, peb, oeb)), ("box release", (qd, ped, oed))] if pe > 0.025]
if bad: sys.exit(f"ABORT: IK cannot reach: {bad}")
T = lambda q: ticks_of(q, {**cur})
W = {"hover": T(steps["hover"][0]), "mid": T(steps["mid"][0]), "grasp": T(steps["grasp"][0]), "boxhi": T(qb), "boxlo": T(qd)}
OPEN, CLOSED = 2183, 1321
for k in W: W[k][8] = OPEN
big = {i: wd(W["hover"][i], cur[i]) for i in range(1, 8)}; log("move to hover (ticks per joint):", big)
def path_ok(f, t, zmin, name):
    low = 9
    for s in np.linspace(0, 1, 50): tt = {i: (f[i] + s * wd(t[i], f[i])) % 4096 for i in range(1, 8)}; low = min(low, float(tip_of(tt)[2]))
    ok = low >= zmin; log(f"  path {name}: lowest tip z {low:.3f} (needs >= {zmin:.3f}) -> {'ok' if ok else 'NOT OK'}"); return ok
ok = path_ok(cur, W["hover"], 0.06, "now -> marker hover") & path_ok(W["hover"], W["mid"], 0.02, "descent") & path_ok(W["grasp"], W["hover"], a.grasp_z - 0.005, "lift") & path_ok(W["hover"], W["boxhi"], 0.10, "marker hover -> box (carrying)") & path_ok(W["boxhi"], W["boxlo"], 0.05, "box descent")
if not ok: sys.exit("ABORT: a planned path is not safe")
log(f"PLAN OK" + ("" if a.go else " - dry run only (add --go)"))
if not a.go: import os; os._exit(0)
# ---- execution
def put(g): be._sync(REG_GOAL_POS, {i: (int(v) % 4096).to_bytes(2, "little") for i, v in g.items()})
def hold(): put({i: be.state().joints[i].raw for i in J})
def move(target, label, speed=None, tol=40, timeout=45):
    last = {i: float(be.state().joints[i].raw) for i in target}; step = (speed or a.speed) / 20; t0 = time.time(); hist = []
    while time.time() - t0 < timeout:
        s = be.state()
        if max(s.joints[i].current_ma for i in range(1, 8)) > 700: hold(); log("HIGH CURRENT - holding"); sys.exit(2)
        d = {i: wd(target[i], last[i]) for i in target}; m = max(abs(v) for v in d.values()); sc = 1.0 if m <= step else step / m
        for i in target: last[i] = (last[i] + d[i] * sc) % 4096
        put({i: last[i] for i in target}); time.sleep(0.05)
        err = max(abs(wd(target[i], s.joints[i].raw)) for i in target); hist.append((time.time(), {i: s.joints[i].raw for i in target})); hist[:] = [x for x in hist if x[0] > time.time() - 0.8]
        moving = max(abs(wd(hist[-1][1][i], hist[0][1][i])) for i in target) if len(hist) > 5 else 99
        if err <= tol or (err <= 2 * tol and moving <= 3 and all(abs(wd(target[i], last[i])) < 1 for i in target)): log(f"reached {label} (worst joint {err} ticks off)"); return
    sN = be.state(); log(f"TIMEOUT at {label} (worst {err} ticks off) per joint [target present error mA]:")
    for i in sorted(target): log(f"   j{i}: {int(target[i])%4096} {sN.joints[i].raw} {wd(target[i], sN.joints[i].raw):+d} {sN.joints[i].current_ma}")
    hold(); log("holding and STOPPING"); sys.exit(3)
def gripper(target, label):
    be._sync(REG_GOAL_SPEED, {8: (300).to_bytes(2, "little")}); put({8: target}); last = None; t0 = time.time()
    while time.time() - t0 < 6:
        v = be.state().joints[8].raw
        if last is not None and abs(v - last) <= 2: break
        last = v; time.sleep(0.15)
    time.sleep(2.0 if label == "closing" else 1.0); g = be.state().joints[8]; log(f"gripper {label}: steady at {g.raw}, {g.current_ma} mA"); return g.raw
put(cur); be._sync(REG_TORQUE, {i: b"\x01" for i in J}); be._sync(REG_ACC, {i: bytes([40]) for i in J}); be._sync(REG_GOAL_SPEED, {i: a.speed.to_bytes(2, "little") for i in J}); log("TORQUE ON - moving")
PROCEED = ROOT / "run/proceed"
def wait_proceed(msg, secs=90):
    PROCEED.unlink(missing_ok=True); log(f"WAITING for you: {msg}  (press 'Proceed' in the web page; {secs} s, else I hold and stop)"); t0 = time.time()
    while time.time() - t0 < secs:
        if PROCEED.exists(): PROCEED.unlink(missing_ok=True); log("proceed received"); return
        time.sleep(0.3)
    hold(); log("no answer - holding and STOPPING"); sys.exit(4)
def annotated(name, pts):
    imx = cv2.cvtColor(vision.decode(be.frame_jpeg(CAM)), cv2.COLOR_RGB2BGR)
    for (u, v), col, txt in pts: cv2.circle(imx, (int(u), int(v)), 8, col, 2); cv2.putText(imx, txt, (int(u) + 10, int(v)), 0, 0.5, col, 2)
    cv2.imwrite(str(ROOT / "run/grab_target.jpg"), imx)
def look(what):
    jp = be.frame_jpeg(CAM); r = er2.point(jp, what, w, h); return jp, r["points"]
def local_A(u, v):
    X0, Y0 = H(u, v); X1, Y1 = H(u + 10, v); X2, Y2 = H(u, v + 10); return np.array([[X1 - X0, X2 - X0], [Y1 - Y0, Y2 - Y0]]) / 10
def replan(x, y, roll, seed):
    out = {}
    for nm, z in (("hover", a.grasp_z + a.hover), ("mid", a.grasp_z + 0.04), ("grasp", a.grasp_z)):
        q, pe, oe, _ = kin.ik_jaw_tip((x, y, z), roll_rad=roll, seed_rad=seed); out[nm] = T(q); out[nm][8] = OPEN; seed = q
        if pe > 0.025: sys.exit(f"ABORT: IK failed at {nm} after alignment (err {pe*100:.1f} cm)")
    return out
gripper(OPEN, "opening"); move(W["hover"], "hover above the marker (estimate from the table mapping)"); move(W["mid"], "down to alignment height", speed=200)
off = np.zeros(2); cx_, cy_ = float(mc[0]), float(mc[1])
for it in range(1, 5):                                                              # visual alignment: jaw tip vs marker centre in the camera image
    jp, tp = look("the tip of the gripper jaws (the lowest point of the black gripper)"); _, mp = look(f"the center of {a.object}")
    if not tp or not mp: log("alignment: ER-2 gave no tip/marker point - stopping"); hold(); sys.exit(5)
    tu, tv = tp[0][1], tp[0][2]; mu_, mv_ = mp[0][1], mp[0][2]; d = np.array([mu_ - tu, mv_ - tv]); annotated("al", [((tu, tv), (0, 255, 255), "tip"), ((mu_, mv_), (255, 0, 255), "marker")])
    log(f"alignment {it}: tip px ({tu:.0f},{tv:.0f}) marker px ({mu_:.0f},{mv_:.0f}) -> offset {d[0]:+.0f},{d[1]:+.0f} px")
    if np.hypot(*d) < 12: log("aligned"); break
    A = local_A(mu_, mv_); dm = np.clip(A @ d, -0.08, 0.08); tipxy = tip_of({i: be.state().joints[i].raw for i in J})[:2]
    nx, ny = float(tipxy[0] + dm[0]), float(tipxy[1] + dm[1]); log(f"   correcting by dx={dm[0]*100:+.1f} cm dy={dm[1]*100:+.1f} cm -> tip target ({nx:.3f},{ny:.3f})")
    qn, pe, oe, _ = kin.ik_jaw_tip((nx, ny, a.grasp_z + 0.04), roll_rad=roll, seed_rad=q_of({i: be.state().joints[i].raw for i in J}))
    if pe > 0.025: sys.exit(f"ABORT: IK for the correction failed ({pe*100:.1f} cm)")
    cx_, cy_ = nx, ny; move({**T(qn), 8: OPEN}, f"alignment move {it}", speed=150)
off = np.array([cx_ - mc[0], cy_ - mc[1]]); log(f"learned table offset vs the saved mapping: dx={off[0]*100:+.1f} cm dy={off[1]*100:+.1f} cm")
Wn = replan(cx_, cy_, roll, q_of({i: be.state().joints[i].raw for i in J})); W.update(Wn)
wait_proceed("jaws above the marker? check the picture, then Proceed to grasp")
move(W["grasp"], "descent to grasp height", speed=150)
gv = gripper(CLOSED, "closing"); move({**W["hover"], 8: gv}, "lift with the marker")
nbx, nby = bx + off[0], by + off[1]; qb2, pb2, _, _ = kin.ik_jaw_tip((nbx, nby, a.release_z + 0.05), roll_rad=0.0, seed_rad=q_of({i: be.state().joints[i].raw for i in J})); qd2, pd2, _, _ = kin.ik_jaw_tip((nbx, nby, a.release_z), roll_rad=0.0, seed_rad=qb2)
if max(pb2, pd2) > 0.025: sys.exit("ABORT: IK cannot reach the box after offset correction")
BH, BL = T(qb2), T(qd2); log(f"box target with the learned offset: ({nbx:.3f},{nby:.3f})")
move({**BH, 8: gv}, "travel to the box")
try: v = er2.ask(be.frame_jpeg(CAM), f'Is {a.object} held in the robot gripper now? Reply {{"held": true or false, "reason": short text}}'); log(f"ER-2 check (holding?): held={v.get('held')} - {v.get('reason')}")
except Exception as e: log("ER-2 check failed:", str(e)[:100])
wait_proceed("gripper above the box? check the picture, then Proceed to release")
move({**BL, 8: gv}, "descent into the box", speed=200); gripper(OPEN, "opening (release)"); move({**BH, 8: OPEN}, "lift away")
try: v = er2.ask(be.frame_jpeg(CAM), f'Is {a.object} now inside {a.target}? Reply {{"inside": true or false, "reason": short text}}'); log(f"ER-2 check (in box?): inside={v.get('inside')} - {v.get('reason')}")
except Exception as e: log("ER-2 check failed:", str(e)[:100])
hold(); log("DONE: holding."); import os; os._exit(0)
