"""Lift the follower straight up, then go to the HOME pose (default: the first pose of the recorded demo run/demo_traj_pick1.json, saved to run/home_follower.json).
    scripts/go_home.py            plan + execute (slow, FK-checked)        scripts/go_home.py --plan   plan only"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from normakit.core import norm_to_rad, rad_to_norm
from normakit.kin import Kin
from normakit.station_backend import StationBackend, REG_TORQUE, REG_GOAL_POS, REG_GOAL_SPEED, REG_ACC
ap = argparse.ArgumentParser(); ap.add_argument("--plan", action="store_true"); ap.add_argument("--speed", type=int, default=150); a = ap.parse_args()
J = list(range(1, 9)); wd = lambda x, y: ((x - y + 2048) % 4096) - 2048; T0 = time.time(); log = lambda *m: print(f"[{time.time()-T0:4.1f}s]", *m, flush=True)
be = StationBackend("localhost:8888"); be.start(); time.sleep(1.5); kin = Kin(); st = be.state()
def to_norm(i, t):
    j = st.joints[i]; span = (j.hi - j.lo) % 4096; rel = (t - j.lo) % 4096
    if rel > span + (4096 - span) / 2: rel -= 4096
    return rel / span
def to_ticks(i, n): j = st.joints[i]; return int(round(j.lo + n * ((j.hi - j.lo) % 4096))) % 4096
q_of = lambda t: {i: norm_to_rad(i, to_norm(i, t[i])) for i in range(1, 8)}; tip_of = lambda t: kin.jaw_tip_pose(q_of(t))[0]
cur = {i: st.joints[i].raw for i in J}; hp = ROOT / "run/home_follower.json"
if hp.exists(): home = {int(k): v for k, v in json.loads(hp.read_text())["raw"].items()}
else:
    D = json.loads((ROOT / "run/demo_traj_pick1.json").read_text()); home = {i: D[0][1][str(i)] for i in range(1, 8)}; hp.write_text(json.dumps({"raw": home, "source": "first pose of run/demo_traj_pick1.json"}, indent=1))
home[8] = cur[8]; t0 = tip_of(cur); th = tip_of(home); log(f"tip now ({t0[0]:+.3f},{t0[1]:.3f},{t0[2]:.3f}) -> home tip ({th[0]:+.3f},{th[1]:.3f},{th[2]:.3f}); joint ticks to travel: { {i: wd(home[i], cur[i]) for i in range(1, 8)} }")
STAGES = [("base + wrist first (j1,j5,j6,j7: unloaded joints, shortens the lever)", (1, 5, 6, 7)), ("shoulder + elbow together (j2,j3,j4, throttled)", (2, 3, 4))]
pose = dict(cur); seq = []
for name, js in STAGES:
    nxt = {**pose, **{i: home[i] for i in js}}; lo = min(float(tip_of({i: (pose[i] + s_ * wd(nxt[i], pose[i])) % 4096 for i in range(1, 8)})[2]) for s_ in np.linspace(0, 1, 40)); seq.append((name, nxt, lo)); log(f"stage {name}: lowest tip z {lo:.3f}"); pose = nxt
if any(lo < 0.03 for _, _, lo in seq): sys.exit("ABORT: a stage would take the tip below 3 cm")
if a.plan: import os; os._exit(0)
def put(g): be._sync(REG_GOAL_POS, {i: (int(v) % 4096).to_bytes(2, "little") for i, v in g.items()})
def hold(): put({i: be.state().joints[i].raw for i in J})
def move(target, label, tol=45, timeout=240):
    """Throttled ramp: the goal only advances while the loaded joints (2,3,4) keep up (lag < 80 ticks); progress watchdog; never lets the goal run away from a stalled joint."""
    last = {i: float(be.state().joints[i].raw) for i in target}; step = a.speed / 20; t0 = time.time(); prog = (time.time(), 1e9); lagmax = 0
    while time.time() - t0 < timeout:
        s = be.state()
        if max(s.joints[i].current_ma for i in range(1, 8)) > 700: hold(); log("HIGH CURRENT - holding"); sys.exit(2)
        lag = max(abs(wd(last[i], s.joints[i].raw)) for i in (2, 3, 4) if i in target) if any(i in target for i in (2, 3, 4)) else 0
        if lag < 80:
            d = {i: wd(target[i], last[i]) for i in target}; m = max(abs(v) for v in d.values()); sc = 1.0 if m <= step else step / m
            for i in target: last[i] = (last[i] + d[i] * sc) % 4096
        put({i: last[i] for i in target}); time.sleep(0.05)
        err = max(abs(wd(target[i], s.joints[i].raw)) for i in target)
        if err < prog[1] - 12: prog = (time.time(), err)
        elif time.time() - prog[0] > 25: hold(); log(f"NO PROGRESS for 25 s at {label} (worst {err}) per joint [target present]: " + str({i: (int(target[i]) % 4096, s.joints[i].raw) for i in sorted(target)})); sys.exit(3)
        if err <= tol: log(f"reached {label} (worst {err} ticks off, t={time.time()-T0:.0f}s)"); return
    hold(); log(f"TIMEOUT at {label}"); sys.exit(3)
put(cur); be._sync(REG_TORQUE, {i: b"\x01" for i in J}); be._sync(0x30, {i: (800).to_bytes(2, "little") for i in (2, 3, 4)}); be._sync(REG_ACC, {i: bytes([10 if i in (2, 3, 4) else 40]) for i in J}); be._sync(REG_GOAL_SPEED, {i: (100 if i in (2, 3, 4) else a.speed).to_bytes(2, "little") for i in J}); log("TORQUE ON - folding towards home")
for name, nxt, lo in seq: move(nxt, name)
hold(); th2 = tip_of({i: be.state().joints[i].raw for i in J}); log(f"HOME reached, tip ({th2[0]:+.3f},{th2[1]:.3f},{th2[2]:.3f}). Holding."); import os; os._exit(0)
