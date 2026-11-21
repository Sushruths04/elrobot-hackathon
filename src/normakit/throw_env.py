"""Gymnasium env: throw a small ball into an open box with the ElRobot twin (SIM ONLY).

Uses assets/mujoco/throw_scene.xml (includes elrobot_scene.xml unchanged, adds a box and a ball) and the
normal MujocoBackend, so the ST3215 servo model (speed/accel profile, torque cap, bus lag) limits the motion.

Action: 8 normalised joint targets in 0..1 (joints 1..8, 8 = gripper), absolute, one set per control step.
Observation (16): 8 joint norms, ball xyz, ball linear velocity xyz, box xy.
Reward: 0 while in flight; at the end of the throw -distance(ball, box centre) in metres, +10 if the ball is in the box.
No Guard here: this is a simulator. A real-arm throw needs its own approved safety path (docs/11_THROW_INTO_BOX.md).
"""
from __future__ import annotations

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from . import mujoco_backend as mb
from .core import JOINT_IDS

THROW_SCENE = mb.ROOT / "assets/mujoco/throw_scene.xml"
BALL_R = 0.0125
BOX_INNER_HALF = 0.06        # inner half width (m)
BOX_WALL_H = 0.06            # rim height above table (m)
# Park the cup and bottle of the base scene out of the way.
_CUP_PARK = (0.40, 0.75)
_BOTTLE_PARK = (-0.40, 0.75)
# Start pose: arm raised, gripper holds the ball.
START_NORMS = {1: 0.50, 2: 0.25, 3: 0.28, 4: 0.72, 5: 0.50, 6: 0.65, 7: 0.50, 8: 0.45}


class ThrowIntoBoxEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, control_dt: float = 0.05, max_steps: int = 60, seed: int = 0,
                 box_x: tuple[float, float] = (-0.12, 0.12), box_y: tuple[float, float] = (0.25, 0.38),
                 servo_preset: str = "calibrated_fix"):
        super().__init__()
        # MujocoBackend reads the module-level SCENE_XML at construction time.
        old = mb.SCENE_XML
        mb.SCENE_XML = THROW_SCENE
        try:
            self.b = mb.MujocoBackend(realtime=False, servo_preset=servo_preset, seed=seed)
        finally:
            mb.SCENE_XML = old
        m = self.b.model
        self.ball_jnt = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "ball_freejoint")
        self.ball_qpos = int(m.jnt_qposadr[self.ball_jnt])
        self.ball_dof = int(m.jnt_dofadr[self.ball_jnt])
        self.box_mocap = int(m.body_mocapid[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "throw_box")])
        self.control_dt = float(control_dt)
        self.max_steps = int(max_steps)
        self.box_x, self.box_y = box_x, box_y
        self.action_space = spaces.Box(0.0, 1.0, shape=(8,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(16,), dtype=np.float32)
        self.box_xy = np.array([0.0, 0.32])
        self._t = 0
        self.reset(seed=seed)

    # ---------------------------------------------------------------- helpers
    def ball_pos(self) -> np.ndarray:
        return self.b.data.qpos[self.ball_qpos:self.ball_qpos + 3].copy()

    def ball_vel(self) -> np.ndarray:
        return self.b.data.qvel[self.ball_dof:self.ball_dof + 3].copy()

    def set_box(self, x: float, y: float) -> None:
        self.box_xy = np.array([x, y], dtype=float)
        self.b.data.mocap_pos[self.box_mocap] = [x, y, 0.0]
        mujoco.mj_forward(self.b.model, self.b.data)

    def set_ball(self, pos, vel=(0.0, 0.0, 0.0)) -> None:
        self.b.data.qpos[self.ball_qpos:self.ball_qpos + 7] = [*pos, 1.0, 0.0, 0.0, 0.0]
        self.b.data.qvel[self.ball_dof:self.ball_dof + 6] = [*vel, 0.0, 0.0, 0.0]
        mujoco.mj_forward(self.b.model, self.b.data)

    def in_box(self) -> bool:
        p = self.ball_pos()
        d = np.abs(p[:2] - self.box_xy)
        return bool(d[0] < BOX_INNER_HALF - 0.005 and d[1] < BOX_INNER_HALF - 0.005 and p[2] < BOX_WALL_H - 0.005)

    def _obs(self) -> np.ndarray:
        st = self.b.state()
        j = [st.joints[i].norm for i in JOINT_IDS]
        return np.array([*j, *self.ball_pos(), *self.ball_vel(), *self.box_xy], dtype=np.float32)

    def _landed(self) -> bool:
        p, v = self.ball_pos(), self.ball_vel()
        return bool(p[2] < BALL_R + 0.004 and np.linalg.norm(v) < 0.05)

    # ---------------------------------------------------------------- gym API
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        b = self.b
        b.reset_arm_norms(START_NORMS)
        b.set_cup_pose(*_CUP_PARK)
        b.set_bottle_pose(*_BOTTLE_PARK)
        x = float(self.np_random.uniform(*self.box_x))
        y = float(self.np_random.uniform(*self.box_y))
        self.set_box(x, y)
        # Ball starts between the jaw tips.
        self.set_ball(b.jaw_tip_pos() + np.array([0.0, -0.012, 0.0]))
        b.step_seconds(0.3)  # let the grip settle
        self._t = 0
        self._released = False
        return self._obs(), {"box_xy": self.box_xy.copy()}

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=np.float64), 0.0, 1.0)
        self.b.send_goals({i: float(a[i - 1]) for i in JOINT_IDS}, speed=1.0)
        self.b.step_seconds(self.control_dt)
        self._t += 1
        success = self.in_box()
        landed = self._landed()
        terminated = success or landed
        truncated = (not terminated) and self._t >= self.max_steps
        reward = 0.0
        if terminated or truncated:
            dist = float(np.linalg.norm(self.ball_pos()[:2] - self.box_xy))
            reward = -dist + (10.0 if success else 0.0)
        info = {"success": success, "ball_xy_err": float(np.linalg.norm(self.ball_pos()[:2] - self.box_xy))}
        return self._obs(), reward, terminated, truncated, info
