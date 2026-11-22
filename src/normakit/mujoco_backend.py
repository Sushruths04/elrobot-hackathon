"""Physics-faithful MuJoCo 3.x digital twin of the NormaCore ElRobot follower arm + table scene.

Exposes the exact same Backend interface (plus StationBackend._sync register writes) so
scripts/grab_cup.py, scripts/autorun.py, scripts/replay_traj.py, and scripts/sim_benchmark.py
run unchanged with `--sim mujoco` / `--backend mujoco`.
"""
from __future__ import annotations

import collections
import io
import json
import math
import os
import struct
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

# Import torch (if installed) on CPU before mujoco EGL to avoid libGL/NVML load-order segfaults when YOLO is used later.
os.environ.setdefault("MUJOCO_GL", "egl")
try:
    import torch as _torch  # noqa: F401
except Exception:
    pass

import cv2
import mujoco
import numpy as np
from PIL import Image

from .core import ArmState, Backend, JOINT_IDS, JointState, URDF_LIMITS, norm_to_rad, rad_to_norm

ROOT = Path(__file__).resolve().parents[2]
SCENE_XML = ROOT / "assets/mujoco/elrobot_scene.xml"
FOLLOWER_CAL = ROOT / "calibration/5B8E114691_follower_20261121_172322.json"

# ST3215 EEPROM & RAM register addresses matching station_backend.py / protocol/memory.rs
REG_MAX_TORQUE = 0x10
REG_P_COEF = 0x15
REG_D_COEF = 0x16
REG_I_COEF = 0x17
REG_PROT_CURRENT = 0x1C
REG_PROT_TORQUE = 0x22
REG_PROT_TIME = 0x23
REG_OVERLOAD_TORQUE = 0x24
REG_TORQUE = 0x28
REG_ACC = 0x29
REG_GOAL_POS = 0x2A
REG_GOAL_SPEED = 0x2E
REG_TORQUE_LIMIT = 0x30

W, H = 640, 480


def _shortest_tick_diff(target: float, current: float) -> float:
    return ((target - current + 2048.0) % 4096.0) - 2048.0


@dataclass
class ST3215ServoParams:
    """Physical + firmware parameters of the 12V Feetech ST3215 servos."""
    stall_torque_nm: float = 2.94        # 30 kg*cm @ 12V
    supply_voltage: float = 12.4
    rated_voltage: float = 12.0
    p_coef: int = 32                     # Station presets.rs uses 16; calibrated fix uses 32
    i_coef: int = 12                     # Station presets.rs uses 0; calibrated fix uses 12 (like SO101)
    d_coef: int = 20                     # Station presets.rs uses 0; calibrated fix uses 20
    max_torque_reg: int = 850            # 0..1000 (Station presets.rs uses 500)
    torque_limit_reg: int = 850          # 0..1000 (Station presets.rs uses 500)
    overload_torque_pct: int = 85        # 0..100 (Station presets.rs uses 25!)
    protection_torque_pct: int = 55      # 0..100 (Station presets.rs uses 20!)
    protection_time_s: float = 1.2       # overload integration window
    deadband_ticks: float = 1.5          # CwDeadZone / CcwDeadZone (~1-2 ticks)
    backlash_ticks: float = 2.5          # 1:345 spur gear play (~0.22 deg)
    cmd_lag_s: float = 0.025             # NormFS + USB-serial bus delay
    pwm_per_tick_scale: float = 0.30     # ST3215 internal duty slope per (P * err_ticks)
    i_pwm_per_tick_s: float = 1.80       # ST3215 internal integral slope per (I * int_err_ticks_s)
    vel_damp_pwm_s: float = 0.022        # internal speed loop damping per D_coef


SERVO_PRESETS: dict[str, ST3215ServoParams] = {
    "station_default": ST3215ServoParams(
        p_coef=16,
        i_coef=0,
        d_coef=0,
        max_torque_reg=500,
        torque_limit_reg=500,
        overload_torque_pct=25,
        protection_torque_pct=20,
        protection_time_s=1.1,
    ),
    "calibrated_fix": ST3215ServoParams(
        p_coef=36,
        i_coef=12,
        d_coef=20,
        max_torque_reg=850,
        torque_limit_reg=850,
        overload_torque_pct=85,
        protection_torque_pct=55,
        protection_time_s=2.0,
    ),
}


@dataclass
class PhotometricConfig:
    exposure_gain: float = 1.0
    brightness_bias: float = 0.0
    noise_std: float = 2.0


class MujocoBackend(Backend):
    name = "mujoco"

    def __init__(
        self,
        realtime: bool = True,
        servo_preset: str = "calibrated_fix",
        servo_params: ST3215ServoParams | None = None,
        enable_torque: bool = True,
        viewer: bool = False,
        photometric: PhotometricConfig | None = None,
        seed: int = 0,
        scene_xml: str | None = None,
    ):
        scene_path = Path(scene_xml) if scene_xml is not None else SCENE_XML
        if not scene_path.exists():
            raise FileNotFoundError(f"Missing {scene_path}; run scripts/build_mujoco_assets.py first.")
        self.model = mujoco.MjModel.from_xml_path(str(scene_path))
        self.data = mujoco.MjData(self.model)
        self.realtime = realtime
        self.params = servo_params or ST3215ServoParams(**SERVO_PRESETS[servo_preset].__dict__)
        self.photo = photometric or PhotometricConfig()
        self._rng = np.random.default_rng(seed)
        self._lock = threading.RLock()
        self._renderer: mujoco.Renderer | None = None
        self._render_thread_id: int | None = None
        self._viewer = None
        self._want_viewer = viewer

        # Load real follower tick calibration so raw tick commands (e.g. grab_cup.py, demo_traj_pick1.json) match 1:1
        cal_data = json.loads(FOLLOWER_CAL.read_text())["motors"]
        self.cal_lo = {int(k): int(v["min"]) for k, v in cal_data.items()}
        self.cal_hi = {int(k): int(v["max"]) for k, v in cal_data.items()}
        self.cal_span = {i: (self.cal_hi[i] - self.cal_lo[i]) % 4096 for i in JOINT_IDS}

        # Cache MuJoCo IDs
        self.jnt_ids = {i: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"rev_motor_{i:02d}") for i in JOINT_IDS}
        self.qpos_adr = {i: int(self.model.jnt_qposadr[self.jnt_ids[i]]) for i in JOINT_IDS}
        self.dof_adr = {i: int(self.model.jnt_dofadr[self.jnt_ids[i]]) for i in JOINT_IDS}
        self.act_ids = {i: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"act_motor_{i:02d}") for i in JOINT_IDS}
        self.jaw1_qpos = int(self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "rev_motor_08_2")])
        self.jaw2_qpos = int(self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "rev_motor_08_1")])
        # Optional scene objects: scenes without them load fine, but the matching
        # setters/getters below raise a clear RuntimeError when the body is absent.
        self.cup_jnt = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "cup_freejoint")
        self.cup_qpos = int(self.model.jnt_qposadr[self.cup_jnt]) if self.cup_jnt >= 0 else None
        self.cup_dof = int(self.model.jnt_dofadr[self.cup_jnt]) if self.cup_jnt >= 0 else None
        self.cup_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "black_cup")
        self.bot_jnt = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "distractor_bottle")
        if self.bot_jnt < 0:
            self.bot_jnt = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "bottle_freejoint")
        self.bot_qpos = int(self.model.jnt_qposadr[self.bot_jnt]) if self.bot_jnt >= 0 else None
        self.bot_dof = int(self.model.jnt_dofadr[self.bot_jnt]) if self.bot_jnt >= 0 else None
        self.ee_sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
        self.tip_sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "jaw_tip_site")
        if self.ee_sid < 0 or self.tip_sid < 0:
            raise ValueError(f"Scene {scene_path} is missing the required ee_site / jaw_tip_site sites.")
        place_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "place_target")
        self.place_mocap_id = int(self.model.body_mocapid[place_bid]) if place_bid >= 0 else None

        # Per-servo register & profile state
        self.torque_on = {i: bool(enable_torque) for i in JOINT_IDS}
        self.reg_acc = {i: 40 for i in JOINT_IDS}
        self.reg_speed = {i: 450 for i in JOINT_IDS}
        self.reg_p = {i: self.params.p_coef for i in JOINT_IDS}
        self.reg_i = {i: self.params.i_coef for i in JOINT_IDS}
        self.reg_d = {i: self.params.d_coef for i in JOINT_IDS}
        self.reg_max_torque = {i: self.params.max_torque_reg for i in JOINT_IDS}
        self.reg_torque_limit = {i: self.params.torque_limit_reg for i in JOINT_IDS}
        self.reg_overload = {i: (100 if i == 8 else self.params.overload_torque_pct) for i in JOINT_IDS}
        self.goal_ticks = {i: 2048 for i in JOINT_IDS}
        self.prof_ticks = {i: 2048.0 for i in JOINT_IDS}
        self.prof_vel_tps = {i: 0.0 for i in JOINT_IDS}
        self.int_err = {i: 0.0 for i in JOINT_IDS}
        self.rotor_play = {i: 0.0 for i in JOINT_IDS}
        self.overload_timer = {i: 0.0 for i in JOINT_IDS}
        self.overloaded = {i: False for i in JOINT_IDS}
        self.current_ma = {i: 0 for i in JOINT_IDS}
        self._cur_filt = {i: 0.0 for i in JOINT_IDS}
        self._cmd_queue: collections.deque[tuple[float, dict[int, int]]] = collections.deque()

        # Video recording state (supports both immediate RGB capture and lazy qpos keyframe buffer for fast 100-trial benchmarks)
        self._video_frames: list[np.ndarray] = []
        self._video_qpos_trace: list[tuple[np.ndarray, np.ndarray]] = []
        self._video_lazy: bool = False
        self._video_path: Path | None = None
        self._video_fps: int = 20
        self._video_cams: tuple[str, ...] = ("front", "wrist")
        self._last_vid_sim_t: float = -1.0

        # Initialize robot in raised non-occluding rest pose (EE high near base, gripper open)
        self._run = False
        self._thr: threading.Thread | None = None
        self.reset_arm_norms({1: 0.50, 2: 0.25, 3: 0.28, 4: 0.72, 5: 0.50, 6: 0.65, 7: 0.50, 8: 0.90})
        if self.cup_jnt >= 0 and self.cup_bid >= 0:
            self.set_cup_pose(0.065, 0.255)
        if self.bot_jnt >= 0:
            self.set_bottle_pose(-0.12, 0.36)
        if self.place_mocap_id is not None:
            self.set_place_target(-0.085, 0.250)

    # ------------------------------------------------------------------ conversions
    def norm_to_qpos(self, jid: int, norm: float) -> float:
        if jid == 8:
            return URDF_LIMITS[8][1] * (1.0 - norm)
        return norm_to_rad(jid, norm)

    def qpos_to_norm(self, jid: int, q: float) -> float:
        if jid == 8:
            return 1.0 - (q / URDF_LIMITS[8][1])
        return rad_to_norm(jid, q)

    def norm_to_ticks(self, jid: int, norm: float) -> int:
        return int(round(self.cal_lo[jid] + norm * self.cal_span[jid])) % 4096

    def ticks_to_norm(self, jid: int, ticks: float) -> float:
        rel = (ticks - self.cal_lo[jid]) % 4096.0
        span = float(self.cal_span[jid])
        if rel > span + (4096.0 - span) / 2.0:
            rel -= 4096.0
        return rel / span

    def _present_ticks_float(self, jid: int) -> float:
        n = self.qpos_to_norm(jid, float(self.data.qpos[self.qpos_adr[jid]]))
        return (self.cal_lo[jid] + n * self.cal_span[jid]) % 4096.0

    # ------------------------------------------------------------------ scene setters
    def reset_arm_norms(self, norms: dict[int, float]) -> None:
        with self._lock:
            for i in JOINT_IDS:
                n = float(norms.get(i, 0.5))
                q = self.norm_to_qpos(i, n)
                self.data.qpos[self.qpos_adr[i]] = q
                self.data.qvel[self.dof_adr[i]] = 0.0
                tk = self.norm_to_ticks(i, n)
                self.goal_ticks[i] = tk
                self.prof_ticks[i] = float(tk)
                self.prof_vel_tps[i] = 0.0
                self.int_err[i] = 0.0
                self.overload_timer[i] = 0.0
                self.overloaded[i] = False
                self.current_ma[i] = 0
                self._cur_filt[i] = 0.0
            n8 = float(norms.get(8, 0.584))
            self.data.qpos[self.jaw2_qpos] = -0.0255 * (1.0 - n8)
            self.data.qpos[self.jaw1_qpos] = +0.0255 * (1.0 - n8)
            self._cmd_queue.clear()
            mujoco.mj_forward(self.model, self.data)

    def reset_arm_ticks(self, ticks: dict[int, int]) -> None:
        self.reset_arm_norms({int(i): self.ticks_to_norm(int(i), int(v)) for i, v in ticks.items()})

    def set_cup_pose(self, x: float, y: float, z: float = 0.001, mass_kg: float | None = None, friction: float | None = None) -> None:
        if self.cup_jnt < 0 or self.cup_bid < 0:
            raise RuntimeError("set_cup_pose(): scene has no 'black_cup' body with 'cup_freejoint' (scene_xml without a cup).")
        with self._lock:
            self.data.qpos[self.cup_qpos : self.cup_qpos + 7] = [x, y, z, 1.0, 0.0, 0.0, 0.0]
            self.data.qvel[self.cup_dof : self.cup_dof + 6] = 0.0
            if mass_kg is not None:
                scale = float(mass_kg) / float(self.model.body_mass[self.cup_bid])
                self.model.body_mass[self.cup_bid] = float(mass_kg)
                self.model.body_inertia[self.cup_bid] *= scale
            if friction is not None:
                for gid in range(self.model.ngeom):
                    if self.model.geom_bodyid[gid] == self.cup_bid and self.model.geom_contype[gid] != 0:
                        self.model.geom_friction[gid, 0] = float(friction)
            mujoco.mj_forward(self.model, self.data)

    def set_bottle_pose(self, x: float, y: float, z: float = 0.001) -> None:
        if self.bot_jnt < 0:
            raise RuntimeError("set_bottle_pose(): scene has no 'distractor_bottle'/'bottle_freejoint' (scene_xml without a bottle).")
        with self._lock:
            self.data.qpos[self.bot_qpos : self.bot_qpos + 7] = [x, y, z, 1.0, 0.0, 0.0, 0.0]
            self.data.qvel[self.bot_dof : self.bot_dof + 6] = 0.0
            mujoco.mj_forward(self.model, self.data)

    def set_place_target(self, x: float, y: float) -> None:
        if self.place_mocap_id is None or self.place_mocap_id < 0:
            raise RuntimeError("set_place_target(): scene has no mocap 'place_target' body (scene_xml without a place target).")
        with self._lock:
            self.data.mocap_pos[self.place_mocap_id] = [x, y, 0.0008]
            mujoco.mj_forward(self.model, self.data)

    def cup_pos(self) -> np.ndarray:
        if self.cup_qpos is None:
            raise RuntimeError("cup_pos(): scene has no 'black_cup' body with 'cup_freejoint' (scene_xml without a cup).")
        with self._lock:
            return self.data.qpos[self.cup_qpos : self.cup_qpos + 3].copy()

    def cup_upright_cos(self) -> float:
        if self.cup_bid < 0:
            raise RuntimeError("cup_upright_cos(): scene has no 'black_cup' body (scene_xml without a cup).")
        with self._lock:
            R = self.data.xmat[self.cup_bid].reshape(3, 3)
            return float(R[2, 2])

    def ee_pos(self) -> np.ndarray:
        with self._lock:
            return self.data.site_xpos[self.ee_sid].copy()

    def jaw_tip_pos(self) -> np.ndarray:
        with self._lock:
            return self.data.site_xpos[self.tip_sid].copy()

    # ------------------------------------------------------------------ physics & ST3215 stepping
    def _step_once_locked(self) -> None:
        dt = float(self.model.opt.timestep)
        sim_t = float(self.data.time)

        # Pop delayed commands whose bus lag has elapsed
        while self._cmd_queue and self._cmd_queue[0][0] <= sim_t:
            _, gmap = self._cmd_queue.popleft()
            for i, tk in gmap.items():
                self.goal_ticks[i] = int(tk) % 4096
                # New goal clears overload latch if direction changes
                if abs(_shortest_tick_diff(self.goal_ticks[i], self._present_ticks_float(i))) < 15:
                    self.overloaded[i] = False
                    self.overload_timer[i] = 0.0

        v_scale = self.params.supply_voltage / self.params.rated_voltage
        qpos = self.data.qpos
        qvel = self.data.qvel
        ctrl = self.data.ctrl
        db = self.params.deadband_ticks
        hb = self.params.backlash_ticks * 0.5
        p_scale = self.params.pwm_per_tick_scale
        i_scale = self.params.i_pwm_per_tick_s
        d_scale = self.params.vel_damp_pwm_s
        prot_t = self.params.protection_time_s
        prot_cap = float(self.params.protection_torque_pct) * 10.0
        stall_tau = self.params.stall_torque_nm * v_scale

        for i in JOINT_IDS:
            aid = self.act_ids[i]
            q_i = float(qpos[self.qpos_adr[i]])
            u_lo, u_hi = URDF_LIMITS[i]
            span_rad = u_hi - u_lo
            norm_i = 1.0 - (q_i / u_hi) if i == 8 else (q_i - u_lo) / span_rad
            pres_float = (self.cal_lo[i] + norm_i * self.cal_span[i]) % 4096.0

            if not self.torque_on[i]:
                ctrl[aid] = 0.0
                self.current_ma[i] = 0
                self.prof_ticks[i] = pres_float
                self.prof_vel_tps[i] = 0.0
                self.int_err[i] = 0.0
                continue

            # 1) ST3215 trapezoidal velocity/acceleration profile generator (pure float math for speed)
            vmax = 20.0 if self.reg_speed[i] < 20 else float(self.reg_speed[i])
            amax = 200.0 if self.reg_acc[i] < 2 else float(self.reg_acc[i]) * 100.0
            dist = ((float(self.goal_ticks[i]) - self.prof_ticks[i] + 2048.0) % 4096.0) - 2048.0
            pv = self.prof_vel_tps[i]
            if -0.5 <= dist <= 0.5 and -amax * dt <= pv <= amax * dt:
                self.prof_ticks[i] = float(self.goal_ticks[i])
                self.prof_vel_tps[i] = 0.0
            else:
                adist = -dist if dist < 0.0 else dist
                v_stop = math.sqrt(2.0 * amax * adist)
                v_mag = vmax if vmax < v_stop else v_stop
                v_des = -v_mag if dist < 0.0 else v_mag
                dv = v_des - pv
                max_dv = amax * dt
                if dv > max_dv:
                    dv = max_dv
                elif dv < -max_dv:
                    dv = -max_dv
                pv += dv
                self.prof_vel_tps[i] = pv
                step_tk = pv * dt
                if (step_tk < 0.0 and -step_tk > adist) or (step_tk >= 0.0 and step_tk > adist):
                    self.prof_ticks[i] = float(self.goal_ticks[i])
                    self.prof_vel_tps[i] = 0.0
                else:
                    self.prof_ticks[i] = (self.prof_ticks[i] + step_tk) % 4096.0

            # 2) Quantised 12-bit encoder + deadband + gearbox backlash
            pres_quant = float(int(pres_float + 0.5) & 4095)
            raw_err = ((self.prof_ticks[i] - pres_quant + 2048.0) % 4096.0) - 2048.0
            sign_q = -1.0 if i == 8 else 1.0
            qvel_rad = float(qvel[self.dof_adr[i]])
            qvel_tps = sign_q * qvel_rad * (self.cal_span[i] / span_rad)

            if -db <= raw_err <= db:
                eff_err = 0.0
            else:
                pt = hb if raw_err > hb else (-hb if raw_err < -hb else raw_err)
                rp = 0.7 * self.rotor_play[i] + 0.3 * pt
                self.rotor_play[i] = rp
                eff_err = raw_err - rp
                if -0.5 * db <= eff_err <= 0.5 * db:
                    eff_err = 0.0

            # 3) ST3215 PID duty calculation + anti-windup + TorqueLimit + OverloadTorque thermal/time derating
            ri = self.reg_i[i]
            if ri > 0:
                ie = self.int_err[i] + eff_err * dt
                self.int_err[i] = 120.0 if ie > 120.0 else (-120.0 if ie < -120.0 else ie)
            else:
                self.int_err[i] = 0.0
            pwm = self.reg_p[i] * p_scale * eff_err + ri * i_scale * self.int_err[i] - self.reg_d[i] * d_scale * qvel_tps
            cap = float(self.reg_max_torque[i] if self.reg_max_torque[i] < self.reg_torque_limit[i] else self.reg_torque_limit[i])
            ov_thresh = float(self.reg_overload[i]) * 10.0
            apwm = -pwm if pwm < 0.0 else pwm
            if apwm > ov_thresh:
                ot = self.overload_timer[i] + dt
                self.overload_timer[i] = ot
                if ot >= prot_t:
                    self.overloaded[i] = True
            elif self.overload_timer[i] > 0.0:
                ot = self.overload_timer[i] - 2.0 * dt
                if ot <= 0.0:
                    self.overload_timer[i] = 0.0
                    self.overloaded[i] = False
                else:
                    self.overload_timer[i] = ot

            if self.overloaded[i] and prot_cap < cap:
                cap = prot_cap
            if pwm > cap:
                pwm = cap
            elif pwm < -cap:
                pwm = -cap

            tau_nm = sign_q * (pwm * 0.001) * stall_tau
            ctrl[aid] = 2.94 if tau_nm > 2.94 else (-2.94 if tau_nm < -2.94 else tau_nm)
            aqv = -qvel_rad if qvel_rad < 0.0 else qvel_rad
            moving_relief = 0.58 if aqv > 0.06 else (aqv / 0.06) * 0.58
            inst_ma = (pwm if pwm >= 0.0 else -pwm) * 1.05 * (1.0 - moving_relief)
            cf = self._cur_filt[i] + (dt / 0.18) * (inst_ma - self._cur_filt[i])
            self._cur_filt[i] = cf
            self.current_ma[i] = int(cf) if cf > 10.0 else 0

        mujoco.mj_step(self.model, self.data)

        # Capture video frame (or lightweight qpos+mocap keyframe in lazy mode) if recording
        if self._video_path is not None and (self.data.time - self._last_vid_sim_t) >= (1.0 / self._video_fps - 1e-6):
            self._last_vid_sim_t = float(self.data.time)
            if self._video_lazy:
                self._video_qpos_trace.append((self.data.qpos.copy(), self.data.mocap_pos.copy()))
            else:
                panels = [self._render_rgb_locked(c, noisy=False) for c in self._video_cams]
                self._video_frames.append(np.hstack(panels) if len(panels) > 1 else panels[0])

    def step_seconds(self, duration_s: float) -> None:
        """Advance physics by `duration_s` seconds (sleeps wall-clock if realtime=True, instant if realtime=False)."""
        if self.realtime and self._run:
            time.sleep(max(0.0, duration_s))
            return
        n_steps = max(1, int(round(duration_s / float(self.model.opt.timestep))))
        with self._lock:
            for _ in range(n_steps):
                self._step_once_locked()
        if self._viewer is not None:
            self._viewer.sync()

    def sleep(self, duration_s: float) -> None:
        """Drop-in replacement for `time.sleep` in scripts so fast-sim (`realtime=False`) advances exact physics time."""
        self.step_seconds(duration_s)

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> None:
        if self._want_viewer and self._viewer is None:
            import mujoco.viewer
            self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
        if self.realtime and not self._run:
            self._run = True
            self._thr = threading.Thread(target=self._realtime_loop, daemon=True)
            self._thr.start()

    def stop_backend(self) -> None:
        self._run = False
        if self._thr is not None:
            self._thr.join(timeout=1.5)
            self._thr = None
        if self._video_path is not None:
            self.stop_video()
        with self._lock:
            if self._renderer is not None:
                try:
                    self._renderer.close()
                except Exception:
                    pass
                self._renderer = None
        if self._viewer is not None:
            try:
                self._viewer.close()
            except Exception:
                pass
            self._viewer = None

    def _realtime_loop(self) -> None:
        dt = float(self.model.opt.timestep)
        batch = 5  # 10 ms batches
        while self._run:
            t0 = time.perf_counter()
            with self._lock:
                for _ in range(batch):
                    self._step_once_locked()
            if self._viewer is not None:
                self._viewer.sync()
            rem = batch * dt - (time.perf_counter() - t0)
            if rem > 0:
                time.sleep(rem)

    # ------------------------------------------------------------------ Backend API + Station _sync compatibility
    def state(self) -> ArmState:
        with self._lock:
            js = {}
            for i in JOINT_IDS:
                pres_tk = int(round(self._present_ticks_float(i))) % 4096
                goal_tk = int(self.goal_ticks[i]) % 4096
                js[i] = JointState(
                    id=i,
                    norm=self.ticks_to_norm(i, pres_tk),
                    goal_norm=self.ticks_to_norm(i, goal_tk),
                    current_ma=self.current_ma[i],
                    temp_c=35,
                    raw=pres_tk,
                    lo=self.cal_lo[i],
                    hi=self.cal_hi[i],
                )
            return ArmState(js, t=float(self.data.time) if not self.realtime else time.time())

    def cameras(self) -> list[str]:
        return ["cam0", "cam1"]

    def frame_age(self, camera: str | None = None) -> float:
        return 0.0

    def _resolve_cam(self, camera: str | None) -> str:
        if camera in (None, "cam0", "front", "table"):
            return "front"
        if camera in ("cam1", "wrist"):
            return "wrist"
        raise ValueError(f"Unknown camera {camera!r}")

    def _ensure_renderer_locked(self) -> mujoco.Renderer:
        tid = threading.get_ident()
        if self._renderer is None or self._render_thread_id != tid:
            if self._renderer is not None:
                try:
                    self._renderer.close()
                except Exception:
                    pass
            self._renderer = mujoco.Renderer(self.model, height=H, width=W)
            self._render_thread_id = tid
        return self._renderer

    def _render_rgb_locked(self, cam_name: str, noisy: bool = True) -> np.ndarray:
        rend = self._ensure_renderer_locked()
        rend.update_scene(self.data, camera=cam_name)
        rgb = rend.render().copy()
        if noisy and (self.photo.exposure_gain != 1.0 or self.photo.brightness_bias != 0.0 or self.photo.noise_std > 0.0):
            f = rgb.astype(np.float32) * self.photo.exposure_gain + self.photo.brightness_bias
            if self.photo.noise_std > 0.0:
                f += self._rng.normal(0.0, self.photo.noise_std, size=f.shape).astype(np.float32)
            rgb = np.clip(f, 0, 255).astype(np.uint8)
        return rgb

    def frame_rgb(self, camera: str | None = None, noisy: bool = True) -> np.ndarray:
        cam_name = self._resolve_cam(camera)
        with self._lock:
            return self._render_rgb_locked(cam_name, noisy=noisy)

    def frame_jpeg(self, camera: str | None = None) -> bytes:
        rgb = self.frame_rgb(camera, noisy=True)
        buf = io.BytesIO()
        Image.fromarray(rgb).save(buf, "JPEG", quality=88)
        return buf.getvalue()

    def _sync(self, address: int, values: dict[int, bytes]) -> None:
        """Exact StationBackend._sync register write compatibility."""
        with self._lock:
            goal_updates: dict[int, int] = {}
            for raw_id, b in values.items():
                i = int(raw_id)
                if address == REG_TORQUE:
                    on = bool(b[0])
                    if on and not self.torque_on[i]:
                        # Match real servo: latch present position if torque was off
                        self.prof_ticks[i] = self._present_ticks_float(i)
                        self.prof_vel_tps[i] = 0.0
                    self.torque_on[i] = on
                elif address == REG_ACC:
                    self.reg_acc[i] = int(b[0])
                elif address == REG_GOAL_SPEED:
                    self.reg_speed[i] = struct.unpack("<H", b[:2])[0]
                elif address == REG_GOAL_POS:
                    goal_updates[i] = struct.unpack("<H", b[:2])[0] & 4095
                elif address == REG_TORQUE_LIMIT:
                    self.reg_torque_limit[i] = struct.unpack("<H", b[:2])[0]
                elif address == REG_MAX_TORQUE:
                    self.reg_max_torque[i] = struct.unpack("<H", b[:2])[0]
                elif address == REG_P_COEF:
                    self.reg_p[i] = int(b[0])
                elif address == REG_I_COEF:
                    self.reg_i[i] = int(b[0])
                elif address == REG_D_COEF:
                    self.reg_d[i] = int(b[0])
                elif address == REG_OVERLOAD_TORQUE:
                    self.reg_overload[i] = int(b[0])
            if goal_updates:
                due = float(self.data.time) + self.params.cmd_lag_s
                self._cmd_queue.append((due, goal_updates))

    def send_goals(self, goals_norm: dict[int, float], speed: float = 0.5) -> None:
        spd = int(150 + 1500 * float(speed))
        self._sync(REG_ACC, {int(i): bytes([30]) for i in goals_norm})
        self._sync(REG_GOAL_SPEED, {int(i): spd.to_bytes(2, "little") for i in goals_norm})
        vals = {int(i): self.norm_to_ticks(int(i), float(g)).to_bytes(2, "little") for i, g in goals_norm.items()}
        self._sync(REG_GOAL_POS, vals)

    def halt(self) -> None:
        st = self.state()
        with self._lock:
            self._cmd_queue.clear()
            for i, j in st.joints.items():
                self.goal_ticks[i] = j.raw
                self.prof_ticks[i] = float(j.raw)
                self.prof_vel_tps[i] = 0.0

    # ------------------------------------------------------------------ video recording
    def start_video(
        self,
        out_path: str | Path,
        fps: int = 20,
        cameras: tuple[str, ...] = ("front", "wrist"),
        lazy: bool = False,
    ) -> None:
        with self._lock:
            self._video_frames = []
            self._video_qpos_trace = []
            self._video_lazy = bool(lazy)
            self._video_path = Path(out_path)
            self._video_fps = fps
            self._video_cams = tuple(self._resolve_cam(c) for c in cameras)
            self._last_vid_sim_t = -1.0

    def stop_video(self, save: bool = True) -> Path | None:
        with self._lock:
            path = self._video_path
            frames = list(self._video_frames)
            qtrace = list(self._video_qpos_trace)
            was_lazy = self._video_lazy
            cams = self._video_cams
            fps = self._video_fps
            self._video_path = None
            self._video_frames = []
            self._video_qpos_trace = []
            if save and was_lazy and qtrace and path is not None:
                saved_qpos = self.data.qpos.copy()
                saved_mocap = self.data.mocap_pos.copy()
                for qp, mp in qtrace:
                    self.data.qpos[:] = qp
                    self.data.mocap_pos[:] = mp
                    mujoco.mj_forward(self.model, self.data)
                    panels = [self._render_rgb_locked(c, noisy=False) for c in cams]
                    frames.append(np.hstack(panels) if len(panels) > 1 else panels[0])
                self.data.qpos[:] = saved_qpos
                self.data.mocap_pos[:] = saved_mocap
                mujoco.mj_forward(self.model, self.data)
        if not save or path is None or not frames:
            return None
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            import av
            container = av.open(str(path), mode="w")
            stream = container.add_stream("libx264", rate=fps)
            stream.height, stream.width = frames[0].shape[:2]
            stream.pix_fmt = "yuv420p"
            stream.options = {"crf": "23", "preset": "veryfast"}
            for rgb in frames:
                vf = av.VideoFrame.from_ndarray(rgb, format="rgb24")
                for pkt in stream.encode(vf):
                    container.mux(pkt)
            for pkt in stream.encode():
                container.mux(pkt)
            container.close()
        except Exception:
            h, w = frames[0].shape[:2]
            vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), self._video_fps, (w, h))
            for rgb in frames:
                vw.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            vw.release()
        return path
