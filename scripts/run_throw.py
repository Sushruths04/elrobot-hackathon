#!/usr/bin/env python3
"""Deploy the throw policy. Default: dry run in the MuJoCo twin, never touches hardware.

  .venv/bin/python scripts/run_throw.py                       # sim dry run, box at (0, 0.32)
  .venv/bin/python scripts/run_throw.py --box 0.05 0.30
  .venv/bin/python scripts/run_throw.py --real --i-have-a-human --host localhost:8888   # real arm, see docs/11

Real arm: needs --real AND --i-have-a-human. Every command goes through Guard.move (clamp, 0.35 step limit,
10 Hz, overcurrent e-stop), so the toss runs at Guard speed. Step 1 is a
slow one-motor test (joint 1, +0.05 norm and back). Faster execution is refused: --unsafe-fast only prints a
refusal; core.py limits are never edited. A faster throw mode needs an owner-approved Guard change (ISSUE-016).
Success of a real toss is NOT validated: the sim policy was trained in a faster servo regime.

Serving through server/policy_server.py (optional): export the open-loop plan with --export-plan out.json and load
it in a custom adapter; the standard LeRobot server expects an ACT checkpoint (see docs/11).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from normakit.core import JOINT_IDS, Guard  # noqa: E402
from normakit.throw_env import START_NORMS  # noqa: E402
from normakit.throw_policy import BOX_X_SCALE, BOX_Y_MID, BOX_Y_SCALE, plan  # noqa: E402

START_TOL = 0.10          # max |joint - start| (norm) before a real run is refused
BOX_X_RANGE = (-BOX_X_SCALE, BOX_X_SCALE)          # trained range, see throw_env.py
BOX_Y_RANGE = (0.25, 0.38)
# Keep-out check on the simulated jaw-tip path of the plan (Guard.cfg.forbidden_boxes exists but Guard.move does not
# enforce it, so the check is done here before any motion). ASSUMED geometry, owner must confirm.
MIN_TIP_Z = 0.03         # m above the table
MIN_TIP_Y = 0.05         # m in front of the base (person side is behind the base)
N_STEPS = 30
PLAN_DT_SIM, PLAN_DT_REAL = 0.10, 0.10   # trained at the Guard rate (10 Hz), so replay timing matches

SAFETY = """SAFETY (AGENTS.md): the arm will move on its own. A human must stand at the e-stop (tools/estop.py),
the workspace must be clear (box in front, nobody within reach), leader mirroring OFF, no ball on the first run."""


def sim_dry_run(theta, box):
    from normakit.throw_env import ThrowIntoBoxEnv
    env = ThrowIntoBoxEnv(seed=0)
    env.reset(seed=1)
    env.set_box(*box)
    acts = plan(theta, box, env.max_steps)
    info = {}
    for t in range(env.max_steps):
        _, r, term, trunc, info = env.step(acts[t])
        if term or trunc:
            break
    print(f"sim dry run: success={info['success']} miss_distance={info['ball_xy_err']:.3f} m")
    return 0


def box_in_range(box):
    return BOX_X_RANGE[0] <= box[0] <= BOX_X_RANGE[1] and BOX_Y_RANGE[0] <= box[1] <= BOX_Y_RANGE[1]


def validate_plan(theta, box):
    """Run the plan in the sim twin and check the jaw-tip path against the keep-out limits. Returns problems."""
    from normakit.throw_env import ThrowIntoBoxEnv
    env = ThrowIntoBoxEnv(seed=0)
    env.reset(seed=1)
    env.set_box(*box)
    acts = plan(theta, box, env.max_steps)
    zs, ys = [], []
    for t in range(env.max_steps):
        env.step(acts[t])
        tip = env.b.jaw_tip_pos()
        zs.append(tip[2]); ys.append(tip[1])
    print(f"  sim jaw-tip path: z min {min(zs):.3f} m, y min {min(ys):.3f} m")
    out = []
    if min(zs) < MIN_TIP_Z:
        out.append(f"jaw tip goes down to z={min(zs):.3f} m (< {MIN_TIP_Z})")
    if min(ys) < MIN_TIP_Y:
        out.append(f"jaw tip comes back to y={min(ys):.3f} m (< {MIN_TIP_Y})")
    return out


def list_plan(acts, ball):
    """Dry-run listing of the plan that would be sent; returns the max joint target per joint."""
    print("plan (every 6th frame, norms of joints 1..8):")
    for t in range(0, len(acts), 6):
        print(f"  t={t:2d}  " + " ".join(f"{v:.2f}" for v in acts[t]))
    print(f"  playback: {len(acts)} frames at {1/PLAN_DT_REAL:.0f} Hz = {len(acts)*PLAN_DT_REAL:.1f} s "
          f"(trained at {1/PLAN_DT_SIM:.0f} Hz, same timing as the sim; "
          "servo response on the real arm still differs, NOT validated)")
    print("  gripper:", "opens during the plan" if ball else "stays put")
    print("  max target per joint:", " ".join(f"{v:.2f}" for v in acts.max(axis=0)))


def real_run(theta, box, host, hz, ball):
    from normakit.station_backend import StationBackend
    be = StationBackend(host)
    be.start()
    time.sleep(1.0)
    guard = Guard(be)
    guard.cfg.max_cmd_hz = min(guard.cfg.max_cmd_hz, hz)
    try:
        # 0. Start-pose check: frame 0 of the plan assumes the arm is at START_NORMS.
        cur0 = be.state().norms()
        off = {j: abs(cur0[j] - START_NORMS[j]) for j in JOINT_IDS if j != 8}
        bad = {j: round(d, 2) for j, d in off.items() if d > START_TOL}
        if bad:
            print(f"refusing: arm is not at the start pose (joint: |error| > {START_TOL}): {bad}")
            return 6
        # 1. Slow one-motor test: joint 1 only, small delta and back.
        cur = be.state().norms()
        print("one-motor test: joint 1 +0.05 and back, slow")
        for g in (cur[1] + 0.05, cur[1]):
            r = guard.move({1: g}, speed=0.2)
            if not r["ok"]:
                print("guard refused:", r["error"])
                return 3
            guard.settle(timeout=4.0)
        if input("Joint 1 moved correctly and safely? type 'yes' to continue: ").strip() != "yes":
            return 4
        # 2. Gentle toss at Guard speed: every command goes through Guard.move (10 Hz, 0.35 step cap).
        acts = plan(theta, box, N_STEPS)
        if not ball:
            acts[:, 7] = acts[0, 7]  # no ball: keep the gripper where it is
        for t in range(len(acts)):
            if guard.estopped:
                print("e-stop active, stopping")
                return 5
            r = guard.move({jid: float(acts[t][jid - 1]) for jid in JOINT_IDS}, speed=0.5)
            if not r["ok"]:
                print("guard refused:", r["error"])
                return 3
            for n in r["notes"]:
                print("  note:", n)
        print("done (not validated against sim success; see docs/11)")
        return 0
    except KeyboardInterrupt:
        guard.estop()
        return 130
    except BaseException as e:   # incl. EOFError from input(): never leave the last goal active
        print(f"error: {e!r}, e-stopping", file=sys.stderr)
        guard.estop()
        return 1
    finally:
        be.stop_backend()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--policy", default="models/throw/policy.npz")
    ap.add_argument("--box", type=float, nargs=2, default=[0.0, 0.32], metavar=("X", "Y"))
    ap.add_argument("--real", action="store_true", help="drive the real arm through Guard (needs --i-have-a-human)")
    ap.add_argument("--i-have-a-human", action="store_true")
    ap.add_argument("--host", default="localhost:8888")
    ap.add_argument("--hz", type=float, default=10.0, help="Guard command rate, never above 10")
    ap.add_argument("--with-ball", action="store_true", help="real run: allow the gripper to open (default keeps it)")
    ap.add_argument("--unsafe-fast", action="store_true", help="refused, see docstring")
    ap.add_argument("--export-plan", help="write the open-loop plan for this box to JSON and exit")
    a = ap.parse_args()

    if a.unsafe_fast:
        print("refusing: no fast throw mode exists. It needs an owner-approved Guard change (ISSUE-016).", file=sys.stderr)
        return 2
    if a.real:
        print(SAFETY)
        if not a.i_have_a_human:
            print("refusing: --real needs --i-have-a-human", file=sys.stderr)
            return 2
    p = Path(a.policy)
    theta = np.load(p if p.is_absolute() else ROOT / p)["theta"]
    if a.export_plan:
        Path(a.export_plan).write_text(json.dumps(plan(theta, a.box).tolist()))
        return 0
    if not box_in_range(a.box):
        print(f"box {a.box} is outside the trained range x {BOX_X_RANGE}, y {BOX_Y_RANGE}", file=sys.stderr)
        if a.real:
            print("refusing on the real arm", file=sys.stderr)
            return 2
        print("warning: sim result is extrapolation", file=sys.stderr)
    if not a.real:
        return sim_dry_run(theta, a.box)
    acts = plan(theta, a.box, N_STEPS)
    list_plan(acts, a.with_ball)
    problems = validate_plan(theta, a.box)
    if problems:
        print("refusing: plan violates keep-out limits: " + "; ".join(problems), file=sys.stderr)
        return 7
    if acts[:, :6].max() > 0.98 or acts[:, :6].min() < 0.02:
        print("note: plan targets are clamped by Guard to 0.02..0.98")
    return real_run(theta, a.box, a.host, min(a.hz, 10.0), a.with_ball)


if __name__ == "__main__":
    sys.exit(main())
