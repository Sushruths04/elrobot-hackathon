"""Parametric throw policy conditioned on the box position (SIM tooling, no hardware access).

theta has 3 * 9 = 27 numbers: a base vector plus two slopes for (box_x, box_y), each of length 9:
  0..5  target norms of joints 1..6 (sigmoid of the raw value)
  6     gripper norm after release (sigmoid)
  7     move duration in control steps (1..14)
  8     release step (2..30)
The action at step t is the start pose moved linearly towards the target over `duration` steps,
gripper held at START until the release step, then set to the release value.
"""
from __future__ import annotations

import numpy as np

from .throw_env import START_NORMS

N_P = 9
THETA_DIM = 3 * N_P
BOX_X_SCALE, BOX_Y_MID, BOX_Y_SCALE = 0.12, 0.315, 0.065
START = np.array([START_NORMS[i] for i in range(1, 9)], dtype=np.float64)


def _sig(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def decode(theta, box_xy):
    th = np.asarray(theta, dtype=np.float64).reshape(3, N_P)
    f = np.array([1.0, box_xy[0] / BOX_X_SCALE, (box_xy[1] - BOX_Y_MID) / BOX_Y_SCALE])
    raw = f @ th
    target = _sig(raw[:6])
    grip = float(_sig(raw[6]))
    dur = 1.0 + 13.0 * float(_sig(raw[7]))
    rel = int(round(2 + 28 * float(_sig(raw[8]))))
    return target, grip, dur, rel


def plan(theta, box_xy, steps: int = 60) -> np.ndarray:
    """Open-loop (steps, 8) action sequence for this box position."""
    target, grip, dur, rel = decode(theta, box_xy)
    out = np.tile(START, (steps, 1))
    for t in range(steps):
        a = min(1.0, (t + 1) / dur)
        out[t, :6] = START[:6] + (target - START[:6]) * a
        out[t, 6] = START[6]
        out[t, 7] = grip if t >= rel else START[7]
    return out


def rollout(env, theta, seed=None, record=False):
    """Run one episode. Returns (success, reward, ball_xy_err, records or None)."""
    obs, info = env.reset(seed=seed)
    acts = plan(theta, info["box_xy"], env.max_steps)
    total, rec = 0.0, []
    for t in range(env.max_steps):
        if record:
            rec.append((obs.copy(), acts[t].astype(np.float32)))
        obs, r, term, trunc, inf = env.step(acts[t])
        total += r
        if term or trunc:
            break
    return bool(inf["success"]), total, inf["ball_xy_err"], rec if record else None
