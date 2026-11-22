"""Kinesthetic demo: the HUMAN moves the FOLLOWER by hand (soft hold on joints 1-7), film_task records video + joints as a LeRobot episode
   (action = the next joint state, since there is no leader arm). Gripper torque stays ON (closed, used as the lifting finger).
   Ends when run/STOP_HAND_DEMO exists (or after --max s); then it holds the pose it was left in and turns torque back on.
   Run via:  scripts/film_task.py tempo_hand"""
import argparse, sys, time, urllib.request, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from normakit.station_backend import StationBackend, REG_TORQUE, REG_GOAL_POS
REG_TLIM = 0x30                                                                                    # torque limit RAM register (LOWER is safe; never above the approved 800)
ap = argparse.ArgumentParser(); ap.add_argument("--max", type=int, default=150); ap.add_argument("--countdown", type=int, default=10); ap.add_argument("--open-gripper", action="store_true", help="start with the gripper OPEN (the human says when to close it)"); a = ap.parse_args()
T0 = time.time(); log = lambda *m: print(f"[{time.time()-T0:5.1f}s]", *m, flush=True)
A = lambda path, body=None: json.loads(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8891" + path, data=None if body is None else json.dumps(body).encode(), method="GET" if body is None else "POST"), timeout=60).read())
stop = ROOT / "run/STOP_HAND_DEMO"; stop.unlink(missing_ok=True)
start = ROOT / "run/START_HAND_DEMO"; start.unlink(missing_ok=True)
log("WAITING: arm must be enabled (ARM motion button) and the human must say they are holding the arm")
while A("/state")["armed_s"] < 60 or not start.exists(): time.sleep(0.5)
A("/gripper", {"to": 2650 if a.open_gripper else 1321}); be = StationBackend("localhost:8888"); be.start(); time.sleep(1.5); J = list(range(1, 8))
for k in range(a.countdown, 0, -1): log(f"HOLD THE ARM NOW - torque goes OFF in {k} s"); time.sleep(1)
LIM = {1: 120, 2: 260, 3: 220, 4: 160, 5: 100, 6: 100, 7: 100}                                      # soft hold: enough to carry the arm's weight, weak enough to push by hand
be._sync(REG_TLIM, {i: LIM[i].to_bytes(2, "little") for i in J}); log("SOFT HOLD: low torque limit, goal follows your hand - the arm stays where you leave it - RECORDING")
try:
    while time.time() - T0 < a.max and not stop.exists():
        st = be.state(); be._sync(REG_GOAL_POS, {i: (st.joints[i].raw % 4096).to_bytes(2, "little") for i in J}); time.sleep(0.04)
finally:
    st = be.state(); be._sync(REG_GOAL_POS, {i: (st.joints[i].raw % 4096).to_bytes(2, "little") for i in J}); be._sync(REG_TLIM, {i: (800 if i in (2, 3, 4) else 1000).to_bytes(2, "little") for i in J})
    log("normal torque restored, holding the pose it was left in")
log("DONE - hand demo recorded")
