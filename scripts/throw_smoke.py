#!/usr/bin/env python
"""Headless smoke test of the throw-into-box sim env. Sim only, never touches hardware."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from normakit.throw_env import START_NORMS, ThrowIntoBoxEnv  # noqa: E402

fails = 0


def check(name, ok, extra=""):
    global fails
    print(("PASS " if ok else "FAIL ") + name, extra)
    fails += 0 if ok else 1


env = ThrowIntoBoxEnv(seed=1)
obs, info = env.reset(seed=1)
check("obs shape", obs.shape == (16,) and np.isfinite(obs).all())
hold = np.array([START_NORMS[i] for i in range(1, 9)], dtype=np.float32)

# 1. Ball is held by the gripper at reset (stays near the jaws while the arm holds still).
p0 = env.ball_pos()
for _ in range(5):
    obs, r, term, trunc, info = env.step(hold)
check("ball held at start", env.ball_pos()[2] > 0.1 and not term, f"z={env.ball_pos()[2]:.3f}")

# 2. Scripted drop above the box registers success.
env.reset(seed=2)
env.set_ball([*env.box_xy, 0.20])
total, ok = 0.0, False
for _ in range(env.max_steps):
    obs, r, term, trunc, info = env.step(hold)
    total += r
    if term or trunc:
        ok = info["success"]
        break
check("drop above box -> success", ok and total > 5, f"reward={total:.2f}")

# 3. Drop away from the box is a miss with negative reward.
env.reset(seed=3)
env.set_ball([env.box_xy[0] + 0.2, env.box_xy[1], 0.20])
total, ok = 0.0, None
for _ in range(env.max_steps):
    obs, r, term, trunc, info = env.step(hold)
    total += r
    if term or trunc:
        ok = info["success"]
        break
check("drop beside box -> miss", ok is False and total < 0, f"reward={total:.3f}")

# 4. Box position is randomised and stays in range.
xs = [tuple(np.round(env.reset(seed=s)[1]["box_xy"], 3)) for s in range(5)]
check("box randomised", len(set(xs)) == 5, str(xs[:2]))

print("ALL OK" if not fails else f"{fails} FAILED")
sys.exit(1 if fails else 0)
