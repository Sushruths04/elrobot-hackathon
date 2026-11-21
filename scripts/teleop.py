"""Our own leader->follower teleop (replaces Station's percent-mirroring, which went out of sync).
Mapping per joint:  follower_goal = rest_f + sign * (leader - rest_l)   (ticks; both arms use the same 4096 ticks/rev encoders, scale 1)
Gripper (joint 8) is mapped by percentage of range (trigger -> jaws), with its own sign.
Safety: refuses to start unless the arms are already near-matching; goal=present before torque on; speed cap; step limit per cycle;
        clamped inside the follower's saved calibration; holds position (goal=present) on exit or on high current.
Setup (both cards must be (Self-controlled): ONE writer on the follower bus):
  1. put BOTH arms in the same rest pose (folded, resting on their stops), then:   scripts/teleop.py --set-rest
  2. harmless check (no motion, torque stays off):                                  scripts/teleop.py --dry-run
  3. first supervised test, ONE joint, person at the arm:                           scripts/teleop.py --joints 6 --go
  4. wrong direction on a joint? flip it:                                           scripts/teleop.py --flip 2,7     (toggles sign in the map)"""
import argparse, asyncio, json, os, signal, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from normakit.busread import snapshot
from normakit.station_backend import StationBackend, REG_TORQUE, REG_GOAL_POS, REG_GOAL_SPEED, REG_ACC
MAP = ROOT / "calibration/teleop_map.json"; B = json.loads((ROOT / "run/buses.json").read_text()); L, F = B["leader"], B["follower"]
ap = argparse.ArgumentParser(); ap.add_argument("--set-rest", action="store_true"); ap.add_argument("--dry-run", action="store_true"); ap.add_argument("--go", action="store_true")
ap.add_argument("--flip", default=""); ap.add_argument("--joints", default="1,2,3,4,5,6,7,8"); ap.add_argument("--speed", type=int, default=700, help="follower speed cap, ticks/s")
ap.add_argument("--hz", type=float, default=30); ap.add_argument("--start-tol", type=int, default=150, help="max |target-present| ticks to allow starting"); ap.add_argument("--release", action="store_true", help="torque off at exit (arm drops!)")
ap.add_argument("--seconds", type=float, default=600); a = ap.parse_args()
async def read(c, q):
    s, _, _ = await snapshot(client=c, queue=q); return s[L], s[F]
async def main():
    s, c, q = await snapshot(); ls, fs = s[L], s[F]
    if len(ls) < 8 or len(fs) < 8: sys.exit("need all 8 motors on both arms")
    m = json.loads(MAP.read_text()) if MAP.exists() else {"sign": {str(i): 1 for i in range(1, 9)}, "rest_l": None, "rest_f": None}
    if a.flip:
        for j in a.flip.split(","): m["sign"][j] = -m["sign"][j]
        MAP.write_text(json.dumps(m, indent=1)); print("signs now:", m["sign"]); (sys.stdout.flush(), os._exit(0))
    if a.set_rest:
        m["rest_l"] = {str(i): ls[i]["pos"] for i in ls}; m["rest_f"] = {str(i): fs[i]["pos"] for i in fs}; MAP.write_text(json.dumps(m, indent=1))
        print("saved rest pose (both arms must have been in the SAME physical pose):"); [print(f"  j{i}: leader {ls[i]['pos']}  follower {fs[i]['pos']}") for i in sorted(ls)]; (sys.stdout.flush(), os._exit(0))
    if not m["rest_l"]: sys.exit("no rest pose yet: put both arms in the same rest pose and run --set-rest")
    joints = [int(j) for j in a.joints.split(",")]
    def target(i, ls, fs):
        if i == 8:                                                          # trigger -> gripper by percentage of range
            l = (ls[i]["pos"] - ls[i]["lo"]) / max(1, (ls[i]["hi"] - ls[i]["lo"])); l = min(1, max(0, l)); l = l if m["sign"]["8"] > 0 else 1 - l
            return fs[i]["lo"] + l * (fs[i]["hi"] - fs[i]["lo"])
        t = m["rest_f"][str(i)] + m["sign"][str(i)] * (ls[i]["pos"] - m["rest_l"][str(i)])
        margin = 25; return min(fs[i]["hi"] - margin, max(fs[i]["lo"] + margin, t))
    print("joint  leader  follower  target  diff   sign")
    worst = 0
    for i in joints:
        t = target(i, ls, fs); d = t - fs[i]["pos"]; worst = max(worst, abs(d)); print(f"  j{i}   {ls[i]['pos']:5d}   {fs[i]['pos']:5d}    {t:6.0f}  {d:+6.0f}   {m['sign'].get(str(i))}")
    if a.dry_run or not a.go: print("dry run only - nothing moved." + ("" if worst <= a.start_tol else f" (largest gap {worst:.0f} ticks > {a.start_tol}: pose the arms alike before --go)")); (sys.stdout.flush(), os._exit(0))
    if worst > a.start_tol: sys.exit(f"refusing to start: follower is {worst:.0f} ticks from the leader-mapped target (> {a.start_tol}). Pose the follower like the leader first.")
    be = StationBackend("localhost:8888"); be.start(); time.sleep(1.0)
    for i in joints: be._sync(REG_GOAL_POS, {i: (int(fs[i]["pos"]) % 4096).to_bytes(2, "little")})      # goal = present BEFORE torque on
    be._sync(REG_TORQUE, {i: b"\x01" for i in joints}); be._sync(REG_ACC, {i: bytes([40]) for i in joints}); be._sync(REG_GOAL_SPEED, {i: a.speed.to_bytes(2, "little") for i in joints})
    print(f"TELEOP ON for joints {joints}: speed cap {a.speed} ticks/s. Move the leader slowly. Ctrl-C holds the pose.", flush=True)
    stop = {"v": False}; signal.signal(signal.SIGINT, lambda *x: stop.update(v=True)); t_end = time.time() + a.seconds; last = {i: fs[i]["pos"] for i in joints}; nlog = 0
    while not stop["v"] and time.time() < t_end:
        ls, fs = await read(c, q); hold = [i for i in joints if fs[i]["cur"] > 700]
        if hold: print("HIGH CURRENT on joints", hold, "- holding pose and stopping"); break
        goals = {}
        for i in joints:
            t = target(i, ls, fs); step = max(-60, min(60, t - last[i]))                      # at most 60 ticks per cycle toward the target
            g = int(round(last[i] + step)); last[i] = g; goals[i] = g.to_bytes(2, "little")
        be._sync(REG_GOAL_POS, goals); nlog += 1
        if nlog % int(a.hz * 5) == 0: print("  gaps (target-present):", {i: int(last[i] - fs[i]["pos"]) for i in joints}, flush=True)
        await asyncio.sleep(1 / a.hz)
    for i in joints: be._sync(REG_GOAL_POS, {i: (int(fs[i]["pos"]) % 4096).to_bytes(2, "little")})
    if a.release: be._sync(REG_TORQUE, {i: b"\x00" for i in joints}); print("torque OFF (arm may drop)")
    else: print("holding the follower at its last position (torque still ON). Use --release or the UI to free it.")
    (sys.stdout.flush(), os._exit(0))
asyncio.run(main())
