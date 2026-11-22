"""Spoon GRASP stage v3, after the Codex review (2026-11-22). Grasp + proof lift only; scoop/place come after the grasp is proven.
Fixes vs v1/v2:
  * orientation: the operator's FULL gripper orientation from demo ep14 (gripper-base rotation, ~87 deg from vertical), not a tilt number
  * approach along the demo's tool axis (gripper-base +Y), pre-grasp 6 cm back along it
  * one motion owner: only armd (/move_joints, path-checked for every move); no direct register writes
  * every failed prerequisite STOPS (hold); no fold fallback that splits joints
  * close gate: VLM must say the open jaws are on opposite sides of the handle; grip = settled + sustained load + 2 s dwell;
    proof lift 2 cm + VLM on a before/after image pair: did the spoon move up with the jaws?
  --check: IK + geometry only, no motion.   Target: --grasp-px u,v,z (read from the live front image) """
import argparse, glob, io, json, sys, time, urllib.request
from pathlib import Path
import numpy as np, cv2, pandas as pd
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from normakit import llm
from normakit.core import norm_to_rad, rad_to_norm
from normakit.kin import Kin
from normakit.station_backend import StationBackend
ap = argparse.ArgumentParser(); ap.add_argument("--check", action="store_true"); ap.add_argument("--grasp-px", required=True); ap.add_argument("--demo", type=int, default=14); a = ap.parse_args()
T0 = time.time(); log = lambda *m: print(f"[{time.time() - T0:5.1f}s]", *m, flush=True)
A = lambda p, b=None, t=60: json.loads(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8891" + p, data=None if b is None else json.dumps(b).encode(), method="GET" if b is None else "POST"), timeout=t).read())
def front_jpg():
    for _ in range(5):
        try: return urllib.request.urlopen("http://127.0.0.1:8893/frame.jpg?c=table", timeout=5).read()
        except Exception: time.sleep(0.5)
    raise RuntimeError("no fresh front frame")
def stop(msg): A("/stop", {}); log("STOP -", msg); sys.exit(1)
cp = json.loads((ROOT / "calibration/front_camera_pose_arm_wide.json").read_text()); cam = json.loads((ROOT / "calibration/camera_front.json").read_text())
K = np.array(cam["K"]); DD = np.array(cam["dist"]).ravel(); Rc = cv2.Rodrigues(np.array(cp["rvec"]))[0]; Cc = np.array(cp["camera_pos_robot_m"])
def at_z(u, v, z):
    d = cv2.undistortPoints(np.array([[[u, v]]], np.float64), K, DD).reshape(2); r = Rc.T @ np.array([d[0], d[1], 1.0]); return Cc + (z - Cc[2]) / r[2] * r
kin = Kin()
def fk(q): p, R = kin.jaw_tip_pose(q); return np.array(p), np.array(R)
def rotvec(Rm): return cv2.Rodrigues(Rm)[0].ravel()
def ik_full(tp, tR, seed, iters=150):
    """7-joint damped least squares on the MuJoCo jaw_tip_site: position + full gripper-base orientation"""
    q = dict(seed); keys = list(range(1, 8)); h = 1e-4
    for _ in range(iters):
        p, R = fk(q); e = np.r_[tp - p, 0.05 * rotvec(tR @ R.T)]
        if np.linalg.norm(e[:3]) < 5e-4 and np.linalg.norm(e[3:]) < 0.05 * np.radians(1): break
        Jm = np.zeros((6, 7))
        for c, k in enumerate(keys):
            q2 = dict(q); q2[k] += h; p2, R2 = fk(q2); Jm[:, c] = (np.r_[p2 - p, 0.05 * rotvec(R2 @ R.T)]) / h
        dq = Jm.T @ np.linalg.solve(Jm @ Jm.T + 1e-5 * np.eye(6), e); dq = np.clip(dq, -0.05, 0.05)
        for c, k in enumerate(keys): q[k] += float(dq[c])
    p, R = fk(q); return q, float(np.linalg.norm(tp - p)), float(np.degrees(np.linalg.norm(rotvec(tR @ R.T))))
# demo grasp configuration
D = ROOT / "datasets/spoon_beans_v1"; df = pd.concat(pd.read_parquet(f) for f in sorted(glob.glob(str(D / "data/*/*.parquet")))); g = df[df.episode_index == a.demo]
S = np.stack(g["observation.state"].values); AC = np.stack(g["action"].values); op = np.flatnonzero(AC[:, 7] > 0.5); ig = int(np.flatnonzero((AC[:, 7] < 0.5) & (np.arange(len(AC)) > op[0]))[0])
qd = {i: norm_to_rad(i, float(S[ig, i - 1])) for i in range(1, 8)}; tip_d, R_d = fk(qd); axis = R_d[:, 1]
log(f"demo ep{a.demo} grasp at {ig / 30:.1f}s: tip {np.round(tip_d, 3)}, tool axis {np.round(axis, 2)} ({np.degrees(np.arccos(np.clip(-axis[2], -1, 1))):.0f} deg from straight down)")
u, v, z = map(float, a.grasp_px.split(",")); T = at_z(u, v, z)
# keep the operator's grip RELATIVE to the arm's direction: rotate the demo orientation about vertical by the azimuth difference (fix 2026-11-22: j1 out of range)
dpsi = np.arctan2(T[1], T[0]) - np.arctan2(tip_d[1], tip_d[0]); Rz = np.array([[np.cos(dpsi), -np.sin(dpsi), 0], [np.sin(dpsi), np.cos(dpsi), 0], [0, 0, 1]])
R_d = Rz @ R_d; axis = R_d[:, 1]; qd0 = dict(qd)
log(f"rotated the demo grip by {np.degrees(dpsi):+.0f} deg about vertical -> tool axis {np.round(axis, 2)}")
pre = T - 0.06 * axis
inr = lambda qq: all(0.01 <= rad_to_norm(i, qq[i]) <= 0.99 for i in range(1, 8))
best = None
for sgn in (+1, -1, 0):                                  # seed the base rotation both ways; keep a solution with every joint in range
    seed = dict(qd0); seed[1] = qd0[1] + sgn * float(dpsi)
    q_g, eg, og = ik_full(T, R_d, seed); q_p, ep_, op_ = ik_full(pre, R_d, q_g); q_l, el, ol = ik_full(T + [0, 0, 0.02], R_d, q_g)
    ok_ = max(eg, ep_, el) < 0.006 and max(og, op_, ol) < 6 and inr(q_g) and inr(q_p) and inr(q_l)
    log(f"  seed j1 {'+' if sgn > 0 else '-' if sgn < 0 else '0'}: errs {eg * 1000:.1f}/{ep_ * 1000:.1f}/{el * 1000:.1f} mm, in range {inr(q_g)}/{inr(q_p)}/{inr(q_l)}")
    if ok_: best = (q_g, eg, og, q_p, ep_, op_, q_l, el, ol); break
if best is None: stop("no IK solution with the operator's grip and all joints in range")
q_g, eg, og, q_p, ep_, op_, q_l, el, ol = best
log(f"target {np.round(T, 3)}: grasp IK err {eg * 1000:.1f} mm / {og:.1f} deg | pre-grasp {np.round(pre, 3)} err {ep_ * 1000:.1f} mm / {op_:.1f} deg | proof-lift err {el * 1000:.1f} mm")
if max(eg, ep_, el) > 0.006 or max(og, op_, ol) > 6: stop("demo orientation not reachable accurately at this target")
_be0 = StationBackend("localhost:8888", require_motors=False); _be0.start(); time.sleep(1.5); _st = _be0.state()
for nm, qq in (("pre-grasp", q_p), ("grasp", q_g), ("proof lift", q_l)):
    bad = [f"j{i} {rad_to_norm(i, qq[i]):.2f}" for i in range(1, 8) if not (0.01 <= rad_to_norm(i, qq[i]) <= 0.99)]
    if bad: stop(f"{nm} needs joints outside their calibrated range: {', '.join(bad)}")
log("joint ranges OK for pre-grasp, grasp and proof lift")
if a.check: log("checks passed (--check: no motion)"); sys.exit(0)
be = StationBackend("localhost:8888", require_motors=False); be.start(); time.sleep(1.5); st = be.state()          # read-only: calibration ranges for ticks
lo = {i: st.joints[i].lo for i in range(1, 9)}; span = {i: (st.joints[i].hi - st.joints[i].lo) % 4096 for i in range(1, 9)}
tk = lambda q: {str(i): int(round((lo[i] + rad_to_norm(i, q[i]) * span[i]) % 4096)) for i in range(1, 8)}
def move(q, what, speed=250):
    r = A("/move_joints", {"ticks": tk(q), "speed": speed, "wait": True, "timeout": 20})
    if not r.get("ok"): stop(f"{what} refused/failed: {r.get('error')}")
    log(f"  {what}: ok")
def ask_yes(jp, q):
    c = llm.config(); r = llm.ask(c["provider"], c["model"], jp, q + ' Reply as JSON: {"answer": "yes" | "no" | "uncertain", "reason": "short"}', max_tokens=3000, timeout=120)
    if r.get("incomplete"): return "uncertain", "model answer incomplete"
    return str(r.get("answer", "uncertain")).lower(), str(r.get("reason", ""))[:120]
log("WAITING: arm must be enabled (ARM motion button)")
while A("/state")["armed_s"] < 120: time.sleep(1)
A("/mirror", {"on": False})
r = A("/gripper", {"to": "open"}); time.sleep(0.6)
move(q_p, "pre-grasp (demo orientation, 6 cm back along the tool axis)"); move(q_g, "slide in to the handle", speed=150); time.sleep(0.5)
jp = front_jpg(); (ROOT / f"results/photos/v3_before_close_{time.strftime('%H%M%S')}.jpg").write_bytes(jp)
ans, why = ask_yes(jp, "Look at the black robot gripper and the wooden spoon. Are the two open gripper jaws on opposite sides of the spoon's handle, so that closing them would grab the handle?")
log(f"  VLM close gate: {ans} ({why})")
if not ans.startswith("yes"): move(q_p, "back off"); stop(f"close gate not passed ({ans})")
r = A("/gripper", {"to": "close"}); samples = []
for _ in range(14): time.sleep(0.15); j8 = A("/state")["joints"]["8"]; samples.append((j8["raw"], j8["mA"]))
raws = [s_[0] for s_ in samples[-6:]]; mas = [s_[1] for s_ in samples[-6:]]; log(f"  jaws after close: raw {raws}, mA {mas}")
settled = max(raws) - min(raws) <= 4; loaded = min(mas) >= 40
if not (settled and loaded): A("/gripper", {"to": "open"}); move(q_p, "back off"); stop(f"no grip (settled={settled}, sustained load={loaded})")
time.sleep(2.0); before = cv2.imdecode(np.frombuffer(front_jpg(), np.uint8), 1); move(q_l, "proof lift 2 cm", speed=100); time.sleep(0.6); after = cv2.imdecode(np.frombuffer(front_jpg(), np.uint8), 1)
pair = np.hstack([before, after]); cv2.putText(pair, "BEFORE", (10, 30), 0, 1, (0, 0, 255), 2); cv2.putText(pair, "AFTER 2 cm lift", (650, 30), 0, 1, (0, 0, 255), 2)
ok_, buf = cv2.imencode(".jpg", pair); (ROOT / f"results/photos/v3_proof_lift_{time.strftime('%H%M%S')}.jpg").write_bytes(buf.tobytes())
ans, why = ask_yes(buf.tobytes(), "Left image BEFORE, right image AFTER the robot gripper lifted by 2 cm. Did the wooden spoon move up together with the gripper (held between the jaws), relative to the bowl?")
log(f"  VLM proof-lift: {ans} ({why})")
if ans.startswith("yes"): log("DONE - GRASP PROVEN (settled load + spoon moved with the jaws)"); sys.exit(0)
A("/gripper", {"to": "open"}); stop(f"proof lift not confirmed ({ans})")
