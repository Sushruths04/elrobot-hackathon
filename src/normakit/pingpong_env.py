"""Gymnasium MuJoCo 3.x environment matched 1:1 to the real sideways JOOLA bat grip on ElRobot
(`results/photos/bat_grip_front_20261122_105923_2.jpg`, `{1:0.505, 2:0.019, 3:0.974, 4:0.464, 5:0.464, 6:0.35, 7:0.80, 8:0.341}`).

Key match to real hardware:
  - Jaws clamp the wooden handle neck at gripper norm = 0.341 (`q8 = 0.751 rad`), red JOOLA blade extends 11.8 cm to -X (image-right).
  - Shoulder/elbow joints 2 & 3 stay parked steady (`norm = 0.030, 0.960`); active balancing uses the 5 fast mid-range joints `[1, 4, 5, 6, 7]`
    where `(j4, j6)` give pure fore/aft pitch and `(j5, j7)` give pure left/right roll!
  - Camera mode uses calibration-free `detect_bat_and_ball` (ball offset relative to live red JOOLA ellipse in the front camera).
"""
from __future__ import annotations

import math
import os
import xml.etree.ElementTree as ET
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from .core import URDF_LIMITS, norm_to_rad
from .pingpong_vision import (
    BALL_RADIUS_M,
    BLADE_HALF_THICK_M,
    BLADE_RADIUS_M,
    FdVel3D,
    detect_bat_and_ball,
)

ROOT = Path(__file__).resolve().parents[2]
PINGPONG_XML = ROOT / "assets/mujoco/pingpong_scene.xml"

# Exact real grip norms from results/photos/bat_grip_joints_20261122_105923.txt (with j2=0.030, j3=0.960 safely inside 0.02..0.98)
REAL_GRIP_NORMS = {1: 0.505, 2: 0.030, 3: 0.960, 4: 0.464, 5: 0.464, 6: 0.350, 7: 0.800, 8: 0.341}
HOME_Q = {i: float(norm_to_rad(i, REAL_GRIP_NORMS[i])) for i in range(1, 9)}

# 5 active balancing joints (all near the middle of their travel!): j1 (yaw), j4 & j6 (fore/aft pitch), j5 & j7 (left/right roll)
ACTIVE_JIDS = [1, 4, 5, 6, 7]
LOCKED_JIDS = [2, 3, 8]
MAX_Q_DEV = np.array([0.35, 0.40, 0.45, 0.45, 0.45], dtype=np.float64)
MAX_JVEL_RAD_S = np.array([1.84, 1.23, 2.30, 2.30, 2.30], dtype=np.float64)

AIR_DRAG_COEF = 3.39e-4


def _load_pingpong_model(light: bool = True) -> mujoco.MjModel:
    if not light:
        return mujoco.MjModel.from_xml_path(str(PINGPONG_XML))
    tree = ET.parse(PINGPONG_XML)
    root = tree.getroot()
    comp = root.find("compiler")
    comp.set("meshdir", str((ROOT / "assets/mujoco").resolve()))
    wb = root.find("worldbody")
    for b in list(wb.findall("body")):
        if b.attrib.get("name") in ("black_cup", "distractor_bottle", "place_target"):
            wb.remove(b)
    for g in list(wb.findall("geom")):
        if g.attrib.get("name") == "back_wall":
            wb.remove(g)
        elif g.attrib.get("name") == "table_top":
            g.attrib.pop("material", None)
    for b in wb.iter("body"):
        for g in list(b.findall("geom")):
            gname = g.attrib.get("name", "")
            if g.attrib.get("contype") == "0" and g.attrib.get("conaffinity") == "0":
                b.remove(g)
            elif gname.startswith("col_jaw") or gname.startswith("col_Gripper_Base"):
                b.remove(g)
            else:
                g.attrib.pop("material", None)
    asset = root.find("asset")
    used_meshes = {g.attrib["mesh"] for g in wb.iter("geom") if "mesh" in g.attrib}
    for m_el in list(asset.findall("mesh")):
        if m_el.attrib["name"] not in used_meshes:
            asset.remove(m_el)
    for tag in ("texture", "material"):
        for el in list(asset.findall(tag)):
            asset.remove(el)
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))


class ElRobotPingPongEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(
        self,
        task: str = "balance",
        obs_mode: str = "controller",
        max_steps: int = 300,
        act_hist: int = 3,
        max_latency: int = 3,
        domain_rand: bool = True,
        ball_noise: float = 1.0,
        spawn_scale: float = 1.0,
        light_model: bool | None = None,
        seed: int = 0,
    ):
        super().__init__()
        assert task in ("balance", "dribble")
        assert obs_mode in ("controller", "camera")
        self.task = task
        self.obs_mode = obs_mode
        self.max_steps = int(max_steps)
        self.act_hist = int(act_hist)
        self.max_latency = int(max_latency)
        self.domain_rand = bool(domain_rand)
        self.ball_noise = float(ball_noise)
        self.spawn_scale = float(spawn_scale)
        self.dt = 1.0 / 30.0
        self.n_substeps = 16

        use_light = (obs_mode == "controller") if light_model is None else bool(light_model)
        self.model = _load_pingpong_model(light=use_light)
        self.data = mujoco.MjData(self.model)
        self._renderer: mujoco.Renderer | None = None

        m = self.model
        self.all_qadr = {i: int(m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"rev_motor_{i:02d}")]) for i in range(1, 9)}
        self.all_dadr = {i: int(m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"rev_motor_{i:02d}")]) for i in range(1, 9)}
        self.all_aid = {i: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, f"act_motor_{i:02d}") for i in range(1, 9)}

        self.act_qadr = np.array([self.all_qadr[i] for i in ACTIVE_JIDS], dtype=int)
        self.act_dadr = np.array([self.all_dadr[i] for i in ACTIVE_JIDS], dtype=int)
        self.action_scale = 0.028 if task == "balance" else 0.075

        self.blade_sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "blade_site")
        self.blade_gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "paddle_blade")
        self.ball_jnt = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "pp_ball_freejoint")
        self.ball_qadr = int(m.jnt_qposadr[self.ball_jnt])
        self.ball_dadr = int(m.jnt_dofadr[self.ball_jnt])
        self.ball_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pingpong_ball")
        self.ball_gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "pp_ball_geom")
        self.cup_jnt = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "cup_freejoint")
        self.bot_jnt = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "bottle_freejoint")

        self._kp_nom = np.array([140.0, 150.0, 125.0, 120.0, 110.0], dtype=np.float64)
        self._kd_nom = np.array([2.8, 3.0, 2.2, 2.0, 1.8], dtype=np.float64)
        self._kp = self._kp_nom.copy()
        self._kd = self._kd_nom.copy()
        self._tau_cap = 2.35

        self.rng = np.random.default_rng(seed)
        self._fd = FdVel3D(alpha=0.50, dt=self.dt)
        self._hist = np.zeros((self.act_hist, len(ACTIVE_JIDS)), dtype=np.float32)
        self._delay_buf: list[np.ndarray] = []
        self._latency_steps = 1
        self._bias_err = np.zeros(3)
        self._ou_err = np.zeros(3)
        self._last_vis_uwh = np.zeros(3)

        # Auto-trim wrist pitch (j6) & wrist roll (j7) around your real grip so settled blade normal == [0, 0, +1]!
        self.settled_home_target = dict(HOME_Q)
        for _trim_it in range(6):
            self.data.qpos[:] = 0.0
            self.data.qvel[:] = 0.0
            for i, qv in self.settled_home_target.items():
                self.data.qpos[self.all_qadr[i]] = qv
            for _ in range(500):
                for idx_a, jid in enumerate(ACTIVE_JIDS):
                    err = self.settled_home_target[jid] - float(self.data.qpos[self.all_qadr[jid]])
                    vel = float(self.data.qvel[self.all_dadr[jid]])
                    self.data.ctrl[self.all_aid[jid]] = float(np.clip(self._kp_nom[idx_a] * err - self._kd_nom[idx_a] * vel, -2.35, 2.35))
                for fj in LOCKED_JIDS:
                    ef = self.settled_home_target[fj] - float(self.data.qpos[self.all_qadr[fj]])
                    vf = float(self.data.qvel[self.all_dadr[fj]])
                    kp_f = 160.0 if fj in (2, 3) else 18.0
                    kd_f = 3.4 if fj in (2, 3) else 0.6
                    cap_f = 2.35 if fj in (2, 3) else 1.4
                    self.data.ctrl[self.all_aid[fj]] = float(np.clip(kp_f * ef - kd_f * vf, -cap_f, cap_f))
                mujoco.mj_step(self.model, self.data)
            _, r_eq = self.blade_pose()
            n_eq = r_eq[:, 2]
            # At real grip: +dq6 gives wx = -1.0 (tilts normal toward +Y -> subtract ny!), +dq7 gives wy = -0.96 (tilts normal toward -X -> add nx!)
            self.settled_home_target[6] = float(np.clip(self.settled_home_target[6] - math.asin(float(np.clip(n_eq[1], -0.5, 0.5))), URDF_LIMITS[6][0] + 0.05, URDF_LIMITS[6][1] - 0.05))
            self.settled_home_target[7] = float(np.clip(self.settled_home_target[7] + math.asin(float(np.clip(n_eq[0], -0.5, 0.5))), URDF_LIMITS[7][0] + 0.05, URDF_LIMITS[7][1] - 0.05))

        self.settled_qpos_arm = {i: float(self.data.qpos[self.all_qadr[i]]) for i in range(1, 9)}
        self.settled_ctrl_arm = {i: float(self.data.ctrl[self.all_aid[i]]) for i in range(1, 9)}
        self.home_vec = np.array([self.settled_home_target[i] for i in ACTIVE_JIDS], dtype=np.float64)
        self.q_lo = np.maximum(
            np.array([URDF_LIMITS[i][0] + 0.04 for i in ACTIVE_JIDS]),
            self.home_vec - MAX_Q_DEV,
        )
        self.q_hi = np.minimum(
            np.array([URDF_LIMITS[i][1] - 0.04 for i in ACTIVE_JIDS]),
            self.home_vec + MAX_Q_DEV,
        )
        self._q_target = self.home_vec.copy()
        self.home_blade_pos = self.data.site_xpos[self.blade_sid].copy()

        # Obs dim = 3 (ball u,w,h) + 3 (FdVel vu,vw,vh) + 5 (dq) + 5 (qvel) + 5*act_hist = 31 (pure camera + joint encoders!)
        n_act = len(ACTIVE_JIDS)
        obs_dim = 6 + 2 * n_act + n_act * self.act_hist
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(n_act,), dtype=np.float32)

        self.t_step = 0
        self.n_bounces = 0
        self._peak_h_since_hit = 0.0
        self._was_above = False
        self._steps_since_bounce = 0

    def blade_pose(self) -> tuple[np.ndarray, np.ndarray]:
        return self.data.site_xpos[self.blade_sid].copy(), self.data.site_xmat[self.blade_sid].reshape(3, 3).copy()

    def true_ball_in_blade(self) -> tuple[np.ndarray, np.ndarray]:
        c, R_b = self.blade_pose()
        u_ax, w_ax, n_ax = R_b[:, 0], R_b[:, 1], R_b[:, 2]
        surf_c = c + (BLADE_HALF_THICK_M + BALL_RADIUS_M) * n_ax
        rel = self.data.qpos[self.ball_qadr : self.ball_qadr + 3] - surf_c
        v_w = self.data.qvel[self.ball_dadr : self.ball_dadr + 3]
        uwh = np.array([np.dot(rel, u_ax), np.dot(rel, w_ax), np.dot(rel, n_ax)], dtype=np.float64)
        vuwh = np.array([np.dot(v_w, u_ax), np.dot(v_w, w_ax), np.dot(v_w, n_ax)], dtype=np.float64)
        return uwh, vuwh

    def _observed_ball_uwh(self) -> np.ndarray:
        if self.obs_mode == "camera":
            rgb = self.render()
            det = detect_bat_and_ball(rgb, ball_color="auto")
            if det.get("visible") and det.get("uwh_m") is not None:
                self._last_vis_uwh = det["uwh_m"].copy()
                if self.task == "balance":
                    self._last_vis_uwh[2] = 0.0
            return self._last_vis_uwh.copy()

        true_uwh, _ = self.true_ball_in_blade()
        if self.ball_noise <= 0.0:
            return true_uwh
        self._ou_err = 0.96 * self._ou_err + self.rng.normal(0.0, [0.0012, 0.0015, 0.0020]) * self.ball_noise
        white = self.rng.normal(0.0, [0.0018, 0.0022, 0.0035]) * self.ball_noise
        noisy = true_uwh + self._bias_err + self._ou_err + white
        if self.task == "balance":
            noisy[2] = 0.0
        else:
            noisy[2] = max(0.0, noisy[2])
        return noisy

    def _build_obs(self, uwh_obs: np.ndarray, vel_fd: np.ndarray) -> np.ndarray:
        q_cur = self.data.qpos[self.act_qadr]
        qv_cur = self.data.qvel[self.act_dadr]
        return np.concatenate([
            [uwh_obs[0] / BLADE_RADIUS_M, uwh_obs[1] / BLADE_RADIUS_M, uwh_obs[2] / 0.20],
            vel_fd / 0.8,
            (q_cur - self.home_vec) / 0.25,
            qv_cur * 0.15,
            self._hist.ravel(),
        ]).astype(np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        r = self.rng
        n_act = len(ACTIVE_JIDS)

        if self.domain_rand:
            self._kp = self._kp_nom * r.uniform(0.80, 1.20, size=n_act)
            self._kd = self._kd_nom * r.uniform(0.75, 1.35, size=n_act)
            self._tau_cap = float(r.uniform(2.05, 2.45))
            self._latency_steps = int(r.integers(0, self.max_latency + 1))
            s_mass = float(r.uniform(0.92, 1.08))
            self.model.body_mass[self.ball_bid] = 0.0027 * s_mass
            self.model.body_inertia[self.ball_bid] = 7.2e-7 * s_mass
            damp_jit = float(r.uniform(-23.0, -14.0))
            self.model.geom_solref[self.ball_gid, 1] = damp_jit
            self.model.geom_solref[self.blade_gid, 1] = damp_jit
            fric = float(r.uniform(0.35, 0.65))
            self.model.geom_friction[self.ball_gid, 0] = fric
            self.model.geom_friction[self.blade_gid, 0] = fric
            self._bias_err = r.normal(0.0, [0.0025, 0.0030, 0.0040]) * self.ball_noise
        else:
            self._kp = self._kp_nom.copy()
            self._kd = self._kd_nom.copy()
            self._tau_cap = 2.35
            self._latency_steps = 1
            self._bias_err[:] = 0.0

        self._ou_err[:] = 0.0
        self.data.qpos[:] = 0.0
        self.data.qvel[:] = 0.0
        for i, qv in self.settled_qpos_arm.items():
            noise_q = float(r.normal(0.0, 0.006)) if (self.domain_rand and i in ACTIVE_JIDS) else 0.0
            self.data.qpos[self.all_qadr[i]] = qv + noise_q
        if self.cup_jnt >= 0:
            cq = int(self.model.jnt_qposadr[self.cup_jnt])
            self.data.qpos[cq : cq + 3] = [0.45, 0.78, 0.01]
        if self.bot_jnt >= 0:
            bq = int(self.model.jnt_qposadr[self.bot_jnt])
            self.data.qpos[bq : bq + 3] = [-0.45, 0.78, 0.01]
        if self.model.nmocap > 0:
            self.data.mocap_pos[0] = [0.45, 0.78, -0.05]

        for i, cv in self.settled_ctrl_arm.items():
            self.data.ctrl[self.all_aid[i]] = cv
        self._q_target = self.home_vec.copy() + (r.normal(0.0, 0.006, size=n_act) if self.domain_rand else 0.0)
        for _ in range(16):
            for idx_a, jid in enumerate(ACTIVE_JIDS):
                err = self._q_target[idx_a] - float(self.data.qpos[self.all_qadr[jid]])
                vel = float(self.data.qvel[self.all_dadr[jid]])
                self.data.ctrl[self.all_aid[jid]] = float(np.clip(self._kp[idx_a] * err - self._kd[idx_a] * vel, -self._tau_cap, self._tau_cap))
            for fj in LOCKED_JIDS:
                ef = self.settled_home_target[fj] - float(self.data.qpos[self.all_qadr[fj]])
                vf = float(self.data.qvel[self.all_dadr[fj]])
                kp_f = 160.0 if fj in (2, 3) else 18.0
                kd_f = 3.4 if fj in (2, 3) else 0.6
                cap_f = 2.35 if fj in (2, 3) else 1.4
                self.data.ctrl[self.all_aid[fj]] = float(np.clip(kp_f * ef - kd_f * vf, -cap_f, cap_f))
            mujoco.mj_step(self.model, self.data)

        c, R_b = self.blade_pose()
        u_ax, w_ax, n_ax = R_b[:, 0], R_b[:, 1], R_b[:, 2]
        ang = float(r.uniform(0.0, 2.0 * math.pi))
        if self.task == "balance":
            rad = float(np.sqrt(r.uniform(0.0, 1.0)) * 0.042 * self.spawn_scale)
            du, dw = rad * math.cos(ang), rad * math.sin(ang)
            dh = 0.001
            v_init = r.uniform(-0.12, 0.12, size=2) * self.spawn_scale
            vz_init = 0.0
        else:
            rad = float(np.sqrt(r.uniform(0.0, 1.0)) * 0.026 * self.spawn_scale)
            du, dw = rad * math.cos(ang), rad * math.sin(ang)
            dh = float(r.uniform(0.065, 0.115))
            v_init = r.uniform(-0.05, 0.05, size=2) * self.spawn_scale
            vz_init = 0.0

        surf_c = c + (BLADE_HALF_THICK_M + BALL_RADIUS_M) * n_ax
        b_pos = surf_c + du * u_ax + dw * w_ax + dh * n_ax
        b_vel = v_init[0] * u_ax + v_init[1] * w_ax + vz_init * n_ax
        self.data.qpos[self.ball_qadr : self.ball_qadr + 7] = [*b_pos, 1.0, 0.0, 0.0, 0.0]
        self.data.qvel[self.ball_dadr : self.ball_dadr + 6] = [*b_vel, 0.0, 0.0, 0.0]
        mujoco.mj_forward(self.model, self.data)

        self._fd.reset()
        self._hist[:] = 0.0
        self._delay_buf = [self._q_target.copy() for _ in range(self._latency_steps + 1)]
        self.t_step = 0
        self.n_bounces = 0
        self._peak_h_since_hit = float(dh)
        self._was_above = dh >= 0.035
        self._steps_since_bounce = 0

        uwh0 = self._observed_ball_uwh()
        self._last_vis_uwh = uwh0.copy()
        v0 = self._fd.update(uwh0, self.dt)
        return self._build_obs(uwh0, v0), {}

    def step(self, action: np.ndarray):
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        prev_a = self._hist[0].copy()
        self._hist = np.roll(self._hist, shift=1, axis=0)
        self._hist[0] = a.astype(np.float32)

        max_dq = np.minimum(self.action_scale, MAX_JVEL_RAD_S * self.dt)
        self._q_target = np.clip(self._q_target + max_dq * a, self.q_lo, self.q_hi)
        self._delay_buf.append(self._q_target.copy())
        cmd_q = self._delay_buf.pop(0)

        if self.task == "balance" and self.domain_rand and self.t_step in (90, 190):
            kick = self.rng.normal(0.0, 0.12, size=2)
            self.data.qvel[self.ball_dadr : self.ball_dadr + 2] += kick

        dt_sub = float(self.model.opt.timestep)
        q_prof = self.data.qpos[self.act_qadr].copy()
        bounce_scored = False
        bounce_bonus = 0.0

        for _ in range(self.n_substeps):
            dq_lim = MAX_JVEL_RAD_S * dt_sub
            q_prof = q_prof + np.clip(cmd_q - q_prof, -dq_lim, dq_lim)
            for idx_a, jid in enumerate(ACTIVE_JIDS):
                err = q_prof[idx_a] - float(self.data.qpos[self.all_qadr[jid]])
                vel = float(self.data.qvel[self.all_dadr[jid]])
                tau = self._kp[idx_a] * err - self._kd[idx_a] * vel
                self.data.ctrl[self.all_aid[jid]] = float(np.clip(tau, -self._tau_cap, self._tau_cap))
            for fj in LOCKED_JIDS:
                ef = self.settled_home_target[fj] - float(self.data.qpos[self.all_qadr[fj]])
                vf = float(self.data.qvel[self.all_dadr[fj]])
                kp_f = 160.0 if fj in (2, 3) else 18.0
                kd_f = 3.4 if fj in (2, 3) else 0.6
                cap_f = 2.35 if fj in (2, 3) else 1.4
                self.data.ctrl[self.all_aid[fj]] = float(np.clip(kp_f * ef - kd_f * vf, -cap_f, cap_f))

            bv = self.data.qvel[self.ball_dadr : self.ball_dadr + 3]
            bspd = float(np.linalg.norm(bv))
            if bspd > 1e-4:
                self.data.qfrc_applied[self.ball_dadr : self.ball_dadr + 3] = -AIR_DRAG_COEF * bspd * bv
            else:
                self.data.qfrc_applied[self.ball_dadr : self.ball_dadr + 3] = 0.0

            vh_before = float(self.data.qvel[self.ball_dadr + 2])
            mujoco.mj_step(self.model, self.data)
            uwh_sub, vuwh_sub = self.true_ball_in_blade()
            if uwh_sub[2] > self._peak_h_since_hit:
                self._peak_h_since_hit = float(uwh_sub[2])
            if self._peak_h_since_hit >= 0.035:
                self._was_above = True

            if self.task == "dribble" and uwh_sub[2] < 0.010 and vh_before < -0.15 and vuwh_sub[2] > 0.30:
                if self.domain_rand:
                    scatter = self.rng.normal(0.0, 0.030, size=2)
                    self.data.qvel[self.ball_dadr : self.ball_dadr + 2] += scatter * abs(vuwh_sub[2])
                if self._was_above and math.hypot(uwh_sub[0], uwh_sub[1]) <= BLADE_RADIUS_M + 0.015:
                    self.n_bounces += 1
                    bounce_scored = True
                    r_hit = math.hypot(uwh_sub[0], uwh_sub[1])
                    pred_apex = (vuwh_sub[2] ** 2) / (2.0 * 9.81)
                    q_apex = math.exp(-(((pred_apex - 0.085) / 0.055) ** 2))
                    q_cent = math.exp(-((r_hit / 0.050) ** 2))
                    q_vert = math.exp(-((math.hypot(vuwh_sub[0], vuwh_sub[1]) / 0.55) ** 2))
                    bounce_bonus += 3.5 * (0.25 + 0.75 * q_apex) * (0.35 + 0.65 * q_cent) * q_vert
                    self._steps_since_bounce = 0
                self._peak_h_since_hit = 0.0
                self._was_above = False

        self.t_step += 1
        self._steps_since_bounce += 1

        true_uwh, true_vuwh = self.true_ball_in_blade()
        obs_uwh = self._observed_ball_uwh()
        vel_fd = self._fd.update(obs_uwh, self.dt)
        obs = self._build_obs(obs_uwh, vel_fd)

        r_uv = math.hypot(true_uwh[0], true_uwh[1])
        q_cur = self.data.qpos[self.act_qadr]
        home_cost = float(np.sum(((q_cur - self.home_vec) / 0.35) ** 2))
        rate_cost = float(np.sum((a - prev_a) ** 2))

        if self.task == "balance":
            terminated = bool(r_uv > 0.105 or true_uwh[2] < -0.025 or true_uwh[2] > 0.090)
            truncated = (not terminated) and (self.t_step >= self.max_steps)
            if terminated:
                reward = -2.0
            else:
                v_plane_sq = true_vuwh[0] ** 2 + true_vuwh[1] ** 2
                reward = (
                    1.0
                    - 0.85 * ((r_uv / BLADE_RADIUS_M) ** 2)
                    - 0.20 * min(4.0, v_plane_sq)
                    - 0.05 * home_cost
                    - 0.02 * rate_cost
                )
            success = bool(truncated and r_uv <= 0.060)
        else:
            stalled = self._steps_since_bounce > 45
            terminated = bool(true_uwh[2] < -0.045 or r_uv > 0.165 or true_uwh[2] > 0.55 or stalled)
            truncated = (not terminated) and (self.t_step >= self.max_steps)
            if terminated:
                reward = -1.5
            else:
                track_shaping = 0.35 * math.exp(-((r_uv / 0.055) ** 2))
                reward = 0.15 + track_shaping + bounce_bonus - 0.04 * home_cost - 0.015 * rate_cost
            success = bool((truncated or self.n_bounces >= 12) and not terminated and self.n_bounces >= 10)

        info = {
            "success": success,
            "r_uv_cm": round(r_uv * 100.0, 2),
            "h_cm": round(float(true_uwh[2]) * 100.0, 2),
            "bounces": int(self.n_bounces),
            "steps": int(self.t_step),
            "bounce_scored": bounce_scored,
        }
        return obs, float(reward), terminated, truncated, info

    def render(self, camera: str = "front") -> np.ndarray:
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=480, width=640)
        self._renderer.update_scene(self.data, camera=camera)
        return self._renderer.render().copy()

    def close(self) -> None:
        if self._renderer is not None:
            try:
                self._renderer.close()
            except Exception:
                pass
            self._renderer = None
