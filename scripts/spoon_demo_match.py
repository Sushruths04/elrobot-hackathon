"""Spoon task from the operator's own grasps (user: reverse-engineer the teleop data).
  front camera: orange handle -> 3D handle (base on the beans at 4.5 cm, length 9.5 cm, camera pose solved from the real arm)
  pick the demo whose handle was closest; shift THAT demo (approach + grasp, operator's angle) so its grasp lands at the same place on today's handle
  firm squeeze from the grasp until the operator's final release; check: jaws stopped on the handle + handle rises in the front view; then scoop + pour
  empty -> next-closest demo.   Prints DONE / STOP (film_task).  --check: plan only."""
import argparse, glob, json, sys, time, urllib.request
from pathlib import Path
import numpy as np, cv2, pandas as pd
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from normakit.core import norm_to_rad
from normakit.kin import Kin
from normakit.station_backend import StationBackend, REG_TORQUE, REG_GOAL_POS, REG_GOAL_SPEED, REG_ACC
ap = argparse.ArgumentParser(); ap.add_argument("--check", action="store_true"); ap.add_argument("--speed", type=float, default=0.8); ap.add_argument("--tries", type=int, default=3); a = ap.parse_args()
T0 = time.time(); log = lambda *m: print(f"[{time.time() - T0:5.1f}s]", *m, flush=True)
A = lambda p, b=None, t=60: json.loads(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8891" + p, data=None if b is None else json.dumps(b).encode(), method="GET" if b is None else "POST"), timeout=t).read())
front = lambda: cv2.imdecode(np.frombuffer(urllib.request.urlopen("http://127.0.0.1:8893/frame.jpg?c=table", timeout=5).read(), np.uint8), 1)
cp = json.loads((ROOT / "calibration/front_camera_pose_arm.json").read_text()); cam = json.loads((ROOT / "calibration/camera_front.json").read_text())
K = np.array(cam["K"]); DD = np.array(cam["dist"]).ravel(); R = cv2.Rodrigues(np.array(cp["rvec"]))[0]; C = np.array(cp["camera_pos_robot_m"])
def ray(u, v):
    d = cv2.undistortPoints(np.array([[[u, v]]], np.float64), K, DD).reshape(2); r = R.T @ np.array([d[0], d[1], 1.0]); return r / np.linalg.norm(r)
def handle3d(top, bot, BE=0.045, L=0.095):
    rb = ray(*bot); B = C + (BE - C[2]) / rb[2] * rb; rt = ray(*top); w = C - B; bq = 2 * rt @ w; cq = w @ w - L * L; disc = bq * bq - 4 * cq
    s = (-bq - np.sqrt(max(disc, 0))) / 2 if disc > 0 else -(rt @ w); T = C + s * rt
    if T[2] < BE + 0.03: s = (-bq + np.sqrt(max(disc, 0))) / 2; T = C + s * rt
    return B, T
def orange(im):
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV); m = cv2.inRange(hsv, np.array((0, 40, 120)), np.array((14, 230, 255))); m[:150] = 0
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)); n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    if n < 2: return None
    i = 1 + int(np.argmax(st[1:, 4])); return (lab == i) if st[i, 4] > 120 else None
kin = Kin(); wd = lambda x, y: ((x - y + 2048) % 4096) - 2048
def tipf(n): return np.array(kin.jaw_tip_pose({i: norm_to_rad(i, n[i - 1]) for i in range(1, 8)})[0], float)
def shift_tip(n, d):
    n = np.array(n, float); base = tipf(n); goal = base + np.asarray(d, float); out = n.copy(); J3 = np.zeros((3, 4)); h = 0.004
    for _ in range(12):
        cur = tipf(out); err = goal - cur
        if np.linalg.norm(err) < 0.001: break
        for c in range(4):
            m = out.copy(); m[c] += h; J3[:, c] = (tipf(m) - cur) / h
        out[:4] += J3.T @ np.linalg.solve(J3 @ J3.T + 1e-4 * np.eye(3), err * min(1.0, 0.01 / np.linalg.norm(err)))
    return out
# ---- demos
D = ROOT / "datasets/spoon_beans_v1"; df = pd.concat(pd.read_parquet(f) for f in sorted(glob.glob(str(D / "data/*/*.parquet"))))
GR = [g for g in json.loads((ROOT / "results/spoon_demo_grasps.json").read_text()) if g["grasp_t"] > 1.0 and np.linalg.norm(g["tip_minus_handle_mm"]) < 60]
def demo(e):
    g = df[df.episode_index == e]; S = np.stack(g["observation.state"].values).astype(float); AC = np.stack(g["action"].values).astype(float); T = g["timestamp"].values - g["timestamp"].values[0]
    opened = np.flatnonzero(AC[:, 7] > 0.5); ig = int(np.flatnonzero((AC[:, 7] < 0.5) & (np.arange(len(AC)) > opened[0]))[0]); rel = int(np.flatnonzero(AC[:, 7] < 0.5).max()) + 1
    return S, AC, T, ig, rel
# ---- today's handle (arm folded out of view)
FOLD = {"1": 2094, "2": 1055, "3": 1554, "4": 3020, "5": 170, "6": 2836, "7": 3021, "8": 2766}
def fold():
    r = A("/move_joints", {"ticks": FOLD, "speed": 300, "wait": True, "timeout": 25})
    if not r.get("ok"):
        for stg in (("5", "6", "7", "8"), ("3", "4"), ("1", "2")): A("/move_joints", {"ticks": {k: FOLD[k] for k in stg}, "speed": 300, "wait": True, "timeout": 25})
if not a.check:
    log("WAITING: arm must be enabled (ARM motion button)")
    while A("/state")["armed_s"] < 90: time.sleep(1)
    A("/mirror", {"on": False}); fold(); time.sleep(1.0)
F_TARGET = 0.50                                    # middle of the handle part sticking out of the bowl (user: not the tip)
def measure():
    m = orange(front())
    if m is None: return None
    ys, xs = np.nonzero(m); top = (float(xs[np.argmin(ys)]), float(ys.min())); bot = (float(xs[np.argmax(ys)]), float(ys.max())); B, T_ = handle3d(top, bot); return B, T_, top, bot
mm = measure()
if mm is None: log("STOP - orange handle not visible"); sys.exit(1)
Bn, Tn, top, bot = mm
log(f"today's handle: base {np.round(Bn, 3)} top {np.round(Tn, 3)} (pixels top {top} bottom {bot})")
cands = sorted(GR, key=lambda g: np.linalg.norm(np.array(g["base"]) - Bn) + np.linalg.norm(np.array(g["top"]) - Tn))
for g in cands[:5]: log(f"  demo ep{g['ep']} ({g['src']}): handle base {g['base']} top {g['top']} | distance {np.linalg.norm(np.array(g['base']) - Bn) * 100 + np.linalg.norm(np.array(g['top']) - Tn) * 100:.1f} cm | grasped {g['frac']:.0%} up")
def make_plan(g, Bn, Tn):
    S, AC, T, ig, rel = demo(g["ep"]); f = g["frac"]; Pd = np.array(g["base"]) + f * (np.array(g["top"]) - np.array(g["base"])); Pn = Bn + F_TARGET * (Tn - Bn); d = Pn - Pd
    t1, t2 = T[ig] - 6.0, T[ig] - 2.0; P = S.copy(); idx = list(range(0, len(S), 5)) + [len(S) - 1]; key = {}
    for k in idx:
        w = float(np.clip((T[k] - t1) / (t2 - t1), 0, 1)); key[k] = shift_tip(S[k], w * d) if w > 0 else S[k].copy()
    ks = sorted(key)
    for j in range(7): P[:, j] = np.interp(np.arange(len(S)), ks, [key[k][j] for k in ks])
    P[:, 7] = AC[:, 7]; P[ig:rel, 7] = 0.0
    zs = [tipf(P[k])[2] for k in range(0, len(P), 5)]
    log(f"plan ep{g['ep']}: shift {np.round(d * 100, 1)} cm, grasp tip {np.round(tipf(P[ig]), 3)}, path min {min(zs):.3f} m"); return (g, P, T, ig, rel, d), min(zs)
plans = [make_plan(g, Bn, Tn)[0] for g in cands[:a.tries]]
if a.check: sys.exit(0)
be = StationBackend("localhost:8888"); be.start(); time.sleep(1.5); s0 = be.state(); J = list(range(1, 9))
lo = {i: s0.joints[i].lo for i in J}; span = {i: (s0.joints[i].hi - s0.joints[i].lo) % 4096 for i in J}; ticks = lambda i, n: (lo[i] + n * span[i]) % 4096
put = lambda gg: be._sync(REG_GOAL_POS, {i: (int(round(v)) % 4096).to_bytes(2, "little") for i, v in gg.items()})
hold = lambda: put({i: be.state().joints[i].raw for i in J})
def follow(P, T, t_from, t_to):
    be._sync(REG_TORQUE, {i: b"\x01" for i in J}); be._sync(0x30, {i: (800 if i in (2, 3, 4) else 1000).to_bytes(2, "little") for i in J[:7]})
    be._sync(REG_ACC, {i: bytes([20 if i in (2, 3, 4) else 40]) for i in J}); be._sync(REG_GOAL_SPEED, {i: (500).to_bytes(2, "little") for i in J})
    t_start = time.time(); bad = 0
    while True:
        tp = t_from + (time.time() - t_start) * a.speed
        if tp >= t_to: break
        n = [np.interp(tp, T, P[:, j]) for j in range(8)]; gg = {i: ticks(i, n[i - 1]) for i in J}; put(gg); s = be.state()
        if max(s.joints[i].current_ma for i in range(1, 8)) > 700: hold(); log("HIGH CURRENT - held"); return False
        lag = max(abs(wd(gg[i], s.joints[i].raw)) for i in (2, 3, 4)); bad = bad + 1 if lag > 150 else 0
        if bad > 40: hold(); log(f"arm not following (lag {lag:.0f}) - held"); return False
        time.sleep(0.04)
    return True
def goto(n):
    tk = {str(i): int(round(ticks(i, n[i - 1]))) for i in J}; r = A("/move_joints", {"ticks": tk, "speed": 300, "wait": True, "timeout": 25})
    if not r.get("ok"):
        for stg in ((5, 6, 7, 8), (3, 4), (1, 2)): A("/move_joints", {"ticks": {str(i): tk[str(i)] for i in stg}, "speed": 300, "wait": True, "timeout": 25})
for n_try, (g, P, T, ig, rel, d) in enumerate(plans, 1):
    if n_try > 1:
        mm = measure()
        if mm is None: log("  handle not visible any more"); break
        Bn, Tn = mm[0], mm[1]; log(f"  re-measured handle: base {np.round(Bn, 3)} top {np.round(Tn, 3)}"); (g, P, T, ig, rel, d), zmin = make_plan(g, Bn, Tn)
        if zmin < 0.04: log("  path too low - skipping this demo"); continue
    log(f"try {n_try}: replaying demo ep{g['ep']} shifted {np.round(d * 100, 1)} cm"); goto(P[0]); first = max(0, int(np.argmax(np.abs(P[:, :7] - P[0, :7]).max(1) > 0.004)) - 10)
    if not follow(P, T, T[first], T[ig] - 0.4): log("STOP - approach interrupted"); sys.exit(2)
    fn = ROOT / f"results/photos/spoon_match_before_close_try{n_try}_{time.strftime('%H%M%S')}.jpg"; cv2.imwrite(str(fn), front()); log(f"  photo before closing: {fn.relative_to(ROOT)}")
    if not follow(P, T, T[ig] - 0.4, T[ig] + 0.6): log("STOP - interrupted at the grasp"); sys.exit(2)
    last = None
    for _ in range(25):
        time.sleep(0.15); raw = be.state().joints[8].raw
        if last is not None and abs(raw - last) < 3: break
        last = raw
    s = be.state(); log(f"  jaws stopped at {s.joints[8].raw} ticks, {s.joints[8].current_ma} mA (empty = 1321-1326)")
    k_lift = int(np.searchsorted(T, T[ig] + 0.6))
    if s.joints[8].raw > 1300 and s.joints[8].current_ma > 40:
        log("  SPOON IN THE JAWS (jaws stopped on the handle under load) - scooping and pouring like the demo")
        ok = follow(P, T, T[k_lift], T[-1]); fn = ROOT / f"results/photos/spoon_match_end_{time.strftime('%H%M%S')}.jpg"; cv2.imwrite(str(fn), front()); log(f"  end photo {fn.relative_to(ROOT)}")
        log("DONE - full task run finished" if ok else "STOP - interrupted while scooping"); sys.exit(0 if ok else 2)
    log("  not grasped - opening, next demo"); put({8: ticks(8, 0.9)}); time.sleep(0.6); fold(); time.sleep(0.5)
hold(); log(f"STOP - no grasp after {len(plans)} demos")
