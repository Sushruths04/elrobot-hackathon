"""TEACH-AND-REPLAY pick (EBIM lesson: ask for a demonstration instead of trusting an off calibration).
You hand-guide the FOLLOWER (torque off) through three poses; I record the joint ticks. Then 'go' replays them slowly by itself.
    scripts/teach_pick.py record hover      # gripper OPEN, ~8-10 cm above the cup
    scripts/teach_pick.py record grasp      # gripper OPEN, jaws around the cup, ready to close
    scripts/teach_pick.py record closed     # gripper CLOSED on the cup (only the gripper joint is used)
    scripts/teach_pick.py go --yes          # torque on: start must be near 'hover'; hover -> grasp -> close -> wait -> lift to hover; holds
    scripts/teach_pick.py open --yes        # open the gripper (release the cup), hold the arm
    scripts/teach_pick.py limp --yes        # torque off (arm drops - support it)
Safety: goal=present before torque on, slow speed cap, step limit per cycle, stop and hold on high current, one writer on the bus."""
import json, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from normakit.station_backend import StationBackend, REG_TORQUE, REG_GOAL_POS, REG_GOAL_SPEED, REG_ACC
F = ROOT / "run/pick_poses.json"; poses = json.loads(F.read_text()) if F.exists() else {}
cmd = sys.argv[1] if len(sys.argv) > 1 else ""; yes = "--yes" in sys.argv; J = list(range(1, 9)); SPEED = 250; HZ = 20
be = StationBackend("localhost:8888"); be.start(); time.sleep(1.0)
def ticks(): return {i: be.state().joints[i].raw for i in J}
def put(goals): be._sync(REG_GOAL_POS, {i: (int(g) % 4096).to_bytes(2, "little") for i, g in goals.items()})
def stop_hold():
    put(ticks()); print("holding the pose (torque stays ON).", flush=True)
def enable():
    cur = ticks(); put(cur); be._sync(REG_TORQUE, {i: b"\x01" for i in J}); be._sync(REG_ACC, {i: bytes([30]) for i in J}); be._sync(REG_GOAL_SPEED, {i: SPEED.to_bytes(2, "little") for i in J}); return cur
def move(target, joints, label, tol=12, timeout=40):
    cur = {i: float(be.state().joints[i].raw) for i in joints}; t0 = time.time(); step = SPEED / HZ
    while time.time() - t0 < timeout:
        st = be.state()
        if max(st.joints[i].current_ma for i in J) > 700: print("HIGH CURRENT - stopping and holding", flush=True); stop_hold(); sys.exit(2)
        for i in joints: cur[i] += max(-step, min(step, target[i] - cur[i]))
        put({i: cur[i] for i in joints}); time.sleep(1 / HZ)
        if all(abs(target[i] - st.joints[i].raw) <= tol for i in joints) and all(abs(target[i] - cur[i]) < 1 for i in joints): print(f"  reached {label}", flush=True); return
    print(f"  timeout moving to {label} (continuing to hold)", flush=True)
if cmd == "record":
    name = sys.argv[2]; assert name in ("hover", "grasp", "closed"); hist = []
    for _ in range(80):
        t = ticks(); hist.append(t); hist[:] = hist[-12:]; time.sleep(0.1)
        if len(hist) == 12 and max(max(h[i] for h in hist) - min(h[i] for h in hist) for i in J) <= 8: break
    else: print("NOT RECORDED: the arm was still moving - hold it still and ask again", flush=True); import os; os._exit(1)
    poses[name] = {str(i): v for i, v in hist[-1].items()}; F.write_text(json.dumps(poses, indent=1))
    print(f"RECORDED {name}: " + " ".join(f"j{i}={v}" for i, v in hist[-1].items()) + f" | have: {', '.join(poses)}", flush=True); import os; os._exit(0)
if cmd in ("go", "open", "limp") and not yes: sys.exit("add --yes once a person is at the arm with the power plug in reach")
if cmd == "limp": be._sync(REG_TORQUE, {i: b"\x00" for i in J}); print("torque OFF", flush=True); import os; os._exit(0)
if cmd == "open":
    enable(); tgt = {8: poses["hover"]["8"]}; move(tgt, [8], "gripper open"); stop_hold(); import os; os._exit(0)
if cmd == "go":
    miss = [n for n in ("hover", "grasp", "closed") if n not in poses]
    if miss: sys.exit("missing recordings: " + ", ".join(miss))
    P = {n: {int(k): v for k, v in poses[n].items()} for n in poses}; cur = ticks(); far = {i: abs(cur[i] - P["hover"][i]) for i in J if i != 8}
    print("distance of the arm from the recorded HOVER pose (ticks):", far, flush=True)
    if max(far.values()) > 500: sys.exit("refusing to start: put the follower roughly in the hover pose first (every joint within ~500 ticks), then run again")
    enable(); print("TORQUE ON - slow replay starting", flush=True)
    move(P["hover"], J, "hover"); time.sleep(1.0)
    move(P["grasp"], J, "grasp"); time.sleep(1.0)
    move({8: P["closed"][8]}, [8], "gripper closed"); print("  waiting for the jaw to stop + 2 s (close fully, wait, then lift)", flush=True)
    last = None
    for _ in range(80):
        v = be.state().joints[8].raw
        if last is not None and abs(v - last) <= 2: break
        last = v; time.sleep(0.15)
    time.sleep(2.0); g = be.state().joints[8]; print(f"  gripper at {g.raw} ticks, current {g.current_ma} mA", flush=True)
    lift = dict(P["hover"]); lift[8] = P["closed"][8]; move(lift, J, "lift to hover (cup held)"); stop_hold()
    print("DONE: the cup should be held above the table. Check it, then 'open --yes' to release or 'limp --yes'.", flush=True); import os; os._exit(0)
print(__doc__)
