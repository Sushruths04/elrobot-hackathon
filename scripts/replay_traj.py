"""Replay a recorded hand-guided/teleoperated demonstration on the FOLLOWER, slowly and safely.
    scripts/replay_traj.py [--file run/demo_traj_pick1.json] [--slow 2.0]            # dry run: summary + how far the arm is from the start pose
    scripts/replay_traj.py --yes [--slow 2.0]                                         # torque ON, replay, hold
Safety: refuses unless the arm is within 200 ticks of the demo's start pose on every joint; goal=present before torque on; per-cycle step limit;
stops and holds on high current; trims the idle head/tail; one writer on the bus (Station mirroring must be OFF: both cards Self-controlled)."""
import argparse, json, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from normakit.station_backend import StationBackend, REG_TORQUE, REG_GOAL_POS, REG_GOAL_SPEED, REG_ACC
ap = argparse.ArgumentParser(); ap.add_argument("--file", default="run/demo_traj_pick1.json"); ap.add_argument("--slow", type=float, default=2.0); ap.add_argument("--yes", action="store_true"); a = ap.parse_args()
J = list(range(1, 9)); T = json.loads((ROOT / a.file).read_text())
mv = [k for k in range(1, len(T)) if max(abs(T[k][1][str(i)] - T[k-1][1][str(i)]) for i in J) > 3]; k0, k1 = max(0, mv[0] - 5), min(len(T) - 1, mv[-1] + 5); T = T[k0:k1 + 1]
t0 = T[0][0]; traj = [(s[0] - t0, {i: s[1][str(i)] for i in J}) for s in T]; dur = traj[-1][0]
be = StationBackend("localhost:8888"); be.start(); time.sleep(1.0); cur = {i: be.state().joints[i].raw for i in J}; start = traj[0][1]
wd = lambda a, b: ((a - b + 2048) % 4096) - 2048                  # shortest signed distance on the 0..4095 encoder circle
far = {i: wd(cur[i], start[i]) for i in J}
print(f"demo: {len(traj)} samples, {dur:.1f} s recorded -> {dur * a.slow:.1f} s at {1/a.slow:.1f}x speed")
print("start pose :", start); print("present    :", cur); print("difference :", far, flush=True)
if not a.yes: print("dry run only. To replay: put the arm within 200 ticks of the start pose on every joint, cup back at its original spot, then add --yes"); import os; os._exit(0)
if max(abs(v) for v in far.values()) > 200: sys.exit(f"refusing: the arm is {max(abs(v) for v in far.values())} ticks from the start pose (limit 200). Hand-place it closer first.")
def put(g): be._sync(REG_GOAL_POS, {i: (int(v) % 4096).to_bytes(2, "little") for i, v in g.items()})
put(cur); be._sync(REG_TORQUE, {i: b"\x01" for i in J}); be._sync(REG_ACC, {i: bytes([30]) for i in J}); be._sync(REG_GOAL_SPEED, {i: (900).to_bytes(2, "little") for i in J})
print("TORQUE ON - replay starting", flush=True); last = dict(cur); t_start = time.time(); k = 0
while True:
    el = (time.time() - t_start) / a.slow
    if el > dur + 0.5: break
    while k + 1 < len(traj) and traj[k + 1][0] <= el: k += 1
    tgt = traj[min(k + 1, len(traj) - 1)][1] if el > traj[k][0] else traj[k][1]
    st = be.state()
    if max(st.joints[i].current_ma for i in J) > 700: print("HIGH CURRENT - stopping and holding", flush=True); put({i: st.joints[i].raw for i in J}); sys.exit(2)
    g = {i: last[i] + max(-40, min(40, wd(tgt[i], last[i]))) for i in J}; last = g; put(g); time.sleep(0.05)
put({i: be.state().joints[i].raw for i in J}); print("DONE - holding the final pose (torque still ON). Use scripts/teach_pick.py limp --yes to free the arm.", flush=True); import os; os._exit(0)
