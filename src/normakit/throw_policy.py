"""Parametric throw policy conditioned on the box position (SIM tooling, no hardware access).

Two phases: a windup to pose A, then a swing to pose B with the gripper opening at a chosen step of the swing.
theta has 3 * N_P numbers: a base vector plus two slopes for (box_x, box_y), each of length N_P = 16:
  0..5   windup pose A, joints 1..6 (sigmoid)
  6..11  swing end pose B, joints 1..6 (sigmoid)
  12     gripper norm after release (sigmoid)
  13     windup duration in control steps (1..10)
  14     swing duration in control steps (1..10)
  15     release step counted from the start of the swing (0..swing duration + 2)
The action at step t moves linearly START -> A, then A -> B; the gripper stays at START until the release step.
All targets are clipped to Guard's 0.02..0.98 clamp.
"""
from __future__ import annotations

import numpy as np

from .throw_env import START_NORMS

N_P = 16
THETA_DIM = 3 * N_P
BOX_X_SCALE, BOX_Y_MID, BOX_Y_SCALE = 0.12, 0.315, 0.065
GUARD_LO, GUARD_HI = 0.02, 0.98   # same clamp Guard.move applies on the real arm, so sim sees what the arm would get
START = np.array([START_NORMS[i] for i in range(1, 9)], dtype=np.float64)


def _sig(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def decode(theta, box_xy):
    """Returns (pose_a, pose_b, grip, windup_steps, swing_steps, release_step_absolute)."""
    th = np.asarray(theta, dtype=np.float64).reshape(3, N_P)
    f = np.array([1.0, box_xy[0] / BOX_X_SCALE, (box_xy[1] - BOX_Y_MID) / BOX_Y_SCALE])
    raw = f @ th
    pa, pb = _sig(raw[:6]), _sig(raw[6:12])
    grip = float(_sig(raw[12]))
    d1 = int(round(1 + 9 * float(_sig(raw[13]))))
    d2 = int(round(1 + 9 * float(_sig(raw[14]))))
    rel = d1 + int(round((d2 + 2) * float(_sig(raw[15]))))
    return pa, pb, grip, d1, d2, rel


def plan(theta, box_xy, steps: int = 30) -> np.ndarray:
    """Open-loop (steps, 8) action sequence for this box position."""
    pa, pb, grip, d1, d2, rel = decode(theta, box_xy)
    out = np.tile(START, (steps, 1))
    for t in range(steps):
        if t < d1:
            out[t, :6] = START[:6] + (pa - START[:6]) * (t + 1) / d1
        else:
            out[t, :6] = pa + (pb - pa) * min(1.0, (t - d1 + 1) / d2)
        out[t, 7] = grip if t >= rel else START[7]
    return np.clip(out, GUARD_LO, GUARD_HI)


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
