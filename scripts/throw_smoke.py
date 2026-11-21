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

def run(seed, ball, vel=(0.0, 0.0, 0.0)):
    env.reset(seed=seed)
    env.set_ball(ball, vel)
    total, info = 0.0, {}
    for _ in range(env.max_steps):
        obs, r, term, trunc, info = env.step(hold)
        total += r
        if term or trunc:
            break
    return info, total


# 2. A toss from 30 cm away that lands in the box is a success.
env.reset(seed=2)
bx, by = env.box_xy
info, total = run(2, [bx, by - 0.30, 0.15], (0.0, 1.92, 0.0))
check("toss into box -> success", info["success"] and info["release_dist"] >= 0.08 and total > 5,
      f"reward={total:.2f} release_dist={info['release_dist']:.3f}")

# 2b. Dropping the ball straight into the box is a placement, not a throw: no success.
env.reset(seed=2)
info, total = run(2, [*env.box_xy, 0.20])
check("drop above box (placement) -> no success", info["in_box"] and not info["success"], f"reward={total:.2f}")

# 2c. A ball still held in the gripper never counts, even if the box were under it.
info, total = run(2, env.b.jaw_tip_pos() + np.array([0.0, -0.012, 0.0]))
check("held ball -> no success", not info["success"] and not info["released"])

# 3. Drop away from the box is a miss with negative reward.
env.reset(seed=3)
info, total = run(3, [env.box_xy[0] + 0.2, env.box_xy[1], 0.20])
check("drop beside box -> miss", info["success"] is False and total < 0, f"reward={total:.3f}")

# 4. Box position is randomised and stays in range.
xs = [tuple(np.round(env.reset(seed=s)[1]["box_xy"], 3)) for s in range(5)]
check("box randomised", len(set(xs)) == 5, str(xs[:2]))

# 5. run_throw.py gating, all checks stop before any hardware connection.
import subprocess  # noqa: E402

RT = [sys.executable, str(Path(__file__).resolve().parent / "run_throw.py")]
r = subprocess.run(RT + ["--real"], capture_output=True, text=True, timeout=60)
check("run_throw --real without --i-have-a-human refuses", r.returncode == 2 and "refusing" in r.stderr)
r = subprocess.run(RT + ["--real", "--i-have-a-human", "--box", "0.30", "0.50"], capture_output=True, text=True, timeout=60)
check("run_throw --real with out-of-range box refuses", r.returncode == 2 and "outside the trained range" in r.stderr)
r = subprocess.run(RT + ["--unsafe-fast"], capture_output=True, text=True, timeout=60)
check("run_throw --unsafe-fast refuses", r.returncode == 2)

# 6. Plans stay inside the Guard clamp.
from normakit.throw_policy import THETA_DIM, plan  # noqa: E402

pl = plan(np.random.default_rng(0).normal(0, 5, THETA_DIM), (0.0, 0.32))
check("plan inside Guard clamp 0.02..0.98", pl.min() >= 0.02 and pl.max() <= 0.98)

print("ALL OK" if not fails else f"{fails} FAILED")
sys.exit(1 if fails else 0)
