"""Replay a recorded hand-guided demo (follower joint trajectory) on the follower arm. Filmed + LeRobot-recorded via:  scripts/film_task.py tempo_replay
   scripts/tempo_replay.py [--samples results/films/<demo>_samples.json] [--speed 0.8]
   Safety: starts only if the arm is already at the demo's first pose (within 80 ticks), checks the FK tip height of the whole path, follows with a lag watchdog + current limit."""
import argparse, json, sys, time, urllib.request
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from normakit.core import norm_to_rad
from normakit.kin import Kin
from normakit.station_backend import StationBackend, REG_TORQUE, REG_GOAL_POS, REG_GOAL_SPEED, REG_ACC
ap = argparse.ArgumentParser(); ap.add_argument("--samples", default="results/films/tempo_open_lid_hand_demo_20261122_013433_samples.json"); ap.add_argument("--speed", type=float, default=0.8); ap.add_argument("--check", action="store_true", help="only run the safety checks, never move"); a = ap.parse_args()
T0 = time.time(); log = lambda *m: print(f"[{time.time()-T0:5.1f}s]", *m, flush=True)
A = lambda path, body=None: json.loads(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8891" + path, data=None if body is None else json.dumps(body).encode(), method="GET" if body is None else "POST"), timeout=60).read())
D = json.loads((ROOT / a.samples).read_text()); T = np.array([x[0] for x in D]); S = np.array([x[1] for x in D]); J = list(range(1, 9)); wd = lambda x, y: ((x - y + 2048) % 4096) - 2048
mv = np.abs(S[:, :7] - S[0, :7]).max(1) > 0.004; first = max(0, int(np.argmax(mv)) - 10); last = len(S) - 1 - int(np.argmax(mv[::-1])); last = min(len(S) - 1, last + 10)
if last <= first: sys.exit("the demo has no movement")
t_a, t_b = T[first], T[last]; log(f"demo window {t_a:.1f}..{t_b:.1f} s ({t_b - t_a:.0f} s of movement), replay at {a.speed}x")
if not a.check: log("WAITING: arm must be enabled (ARM motion button)")
while not a.check and A("/state")["armed_s"] < 30: time.sleep(1)
be = StationBackend("localhost:8888"); be.start(); time.sleep(1.5); st = be.state(); kin = Kin()
def ticks(i, n): j = st.joints[i]; return (j.lo + n * ((j.hi - j.lo) % 4096)) % 4096
start = {i: ticks(i, S[first, i - 1]) for i in J[:7]}; err = max(abs(wd(start[i], st.joints[i].raw)) for i in start); log(f"current pose vs demo start: worst {err:.0f} ticks off")
if err > 80: sys.exit("ABORT: the arm is not at the demo's first pose - move it to the initial pose first (scripts/go_home or the detection pose)")
zs = [float(kin.jaw_tip_pose({i: norm_to_rad(i, S[k, i - 1]) for i in range(1, 8)})[0][2]) for k in range(first, last + 1, 5)]; log(f"path tip height: min {min(zs):.3f} m, max {max(zs):.3f} m")
if min(zs) < 0.02: sys.exit("ABORT: the path goes below 2 cm")
if a.check: log("checks passed (--check: not moving)"); sys.exit(0)
put = lambda g: be._sync(REG_GOAL_POS, {i: (int(round(v)) % 4096).to_bytes(2, "little") for i, v in g.items()})
hold = lambda: put({i: be.state().joints[i].raw for i in J})
put({i: st.joints[i].raw for i in J}); be._sync(REG_TORQUE, {i: b"\x01" for i in J}); be._sync(0x30, {i: (800 if i in (2, 3, 4) else 1000).to_bytes(2, "little") for i in J[:7]})
be._sync(REG_ACC, {i: bytes([20 if i in (2, 3, 4) else 40]) for i in J}); be._sync(REG_GOAL_SPEED, {i: (500).to_bytes(2, "little") for i in J})
log("REPLAYING"); t_start = time.time(); bad = 0
while True:
    tp = t_a + (time.time() - t_start) * a.speed
    if tp >= t_b: break
    n = [np.interp(tp, T, S[:, j]) for j in range(8)]; g = {i: ticks(i, n[i - 1]) for i in J}; put(g); s = be.state()
    if max(s.joints[i].current_ma for i in range(1, 8)) > 700: hold(); log("HIGH CURRENT - held, stopping"); sys.exit(2)
    lag = max(abs(wd(g[i], s.joints[i].raw)) for i in (2, 3, 4)); bad = bad + 1 if lag > 150 else 0
    if bad > 40: hold(); log(f"the arm is not following (lag {lag:.0f} ticks) - held, stopping"); sys.exit(3)
    time.sleep(0.04)
hold(); log("DONE - replay finished, holding the final pose")
