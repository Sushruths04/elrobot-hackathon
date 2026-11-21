"""FIRST MOTION TEST (moves hardware!). One follower joint, tiny delta, slow, then back.
Preconditions: mirroring stopped (both cards (Self-controlled)), a human at the table, arm supported if the joint carries load.
    .venv/bin/python scripts/first_motion.py --joint 8 --delta 0.03 --yes
Steps: read present -> write goal=present -> torque ON for that joint only -> move +delta (slow) -> back -> leave torque ON.
Aborts if the joint is uncalibrated or the target would leave 2%..98% of the range."""
import argparse, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from normakit.station_backend import StationBackend, REG_TORQUE, REG_GOAL_POS
ap = argparse.ArgumentParser(); ap.add_argument("--joint", type=int, default=8); ap.add_argument("--delta", type=float, default=0.03)
ap.add_argument("--speed", type=float, default=0.05); ap.add_argument("--yes", action="store_true"); a = ap.parse_args()
if not a.yes: sys.exit("refusing: add --yes once a human is at the table and mirroring is stopped")
b = StationBackend("localhost:8888"); b.start(); time.sleep(1)
j = b.state().joints[a.joint]; span = (j.hi - j.lo) % 4096
assert j.norm_ok if hasattr(j, "norm_ok") else True
print(f"joint {a.joint}: norm={j.norm:.3f} raw={j.raw} span={span} ticks, delta={a.delta*span:.0f} ticks")
if abs(a.delta) * span > 60: sys.exit("delta > 60 ticks: refusing (AGENTS.md rule 4)")
tgt = j.norm + a.delta
if not (0.02 <= tgt <= 0.98 and 0.02 <= j.norm <= 0.98): sys.exit(f"target {tgt:.3f} or start {j.norm:.3f} outside 2-98% of range: refusing")
b._sync(REG_GOAL_POS, {a.joint: (j.raw % 4096).to_bytes(2, "little")})   # goal = present BEFORE torque on (no jump)
b._sync(REG_TORQUE, {a.joint: b"\x01"}); time.sleep(0.5)
b.send_goals({a.joint: tgt}, a.speed); time.sleep(2.0); print("moved to", round(b.state().joints[a.joint].norm, 3))
b.send_goals({a.joint: j.norm}, a.speed); time.sleep(2.0); print("back at", round(b.state().joints[a.joint].norm, 3), "(torque left ON)")
import os; os._exit(0)
