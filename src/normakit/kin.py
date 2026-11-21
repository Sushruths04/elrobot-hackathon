"""Kinematics for ElRobot from its official URDF (pybullet, DIRECT mode, no GUI).

fk(q_rad)->xyz and ik(xyz)->q. This is the 'action decoder': the LLM never emits joint
angles for reaching - it says *where* (a pixel or a point) and IK does the rest.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pybullet as p

from .core import URDF_LIMITS, norm_to_rad, rad_to_norm

URDF = Path(__file__).resolve().parents[2] / "vendor/norma-core-src/hardware/elrobot/simulation/elrobot_follower.urdf"
ACT = {1: 1, 2: 3, 3: 5, 4: 7, 5: 9, 6: 11, 7: 13, 8: 15}   # motor id -> pybullet joint index
EE_LINK = 13


class Kin:
    def __init__(self):
        self.cid = p.connect(p.DIRECT)
        self.rid = p.loadURDF(str(URDF), useFixedBase=True, physicsClientId=self.cid)
        self.lo = [URDF_LIMITS[i][0] for i in range(1, 7)]
        self.hi = [URDF_LIMITS[i][1] for i in range(1, 7)]

    def _set(self, q: dict[int, float]):
        for jid, r in q.items():
            p.resetJointState(self.rid, ACT[jid], r, physicsClientId=self.cid)

    def fk(self, q_rad: dict[int, float]) -> np.ndarray:
        self._set(q_rad)
        return np.array(p.getLinkState(self.rid, EE_LINK, computeForwardKinematics=True,
                                       physicsClientId=self.cid)[4])

    def ik(self, xyz, seed_rad: dict[int, float] | None = None, iters: int = 200):
        """Position IK on joints 1-6. Returns (q_rad dict, error_m)."""
        if seed_rad:
            self._set(seed_rad)
        rng = np.arange(1, 7)
        sol = p.calculateInverseKinematics(
            self.rid, EE_LINK, list(xyz),
            lowerLimits=self.lo, upperLimits=self.hi,
            jointRanges=[h - l for l, h in zip(self.lo, self.hi)],
            restPoses=[(seed_rad or {}).get(int(i), 0.0) for i in rng],
            maxNumIterations=iters, residualThreshold=1e-4, physicsClientId=self.cid)
        # sol is ordered by non-fixed joints: motors 1..6 first
        q = {int(i): float(np.clip(sol[k], self.lo[k], self.hi[k])) for k, i in enumerate(rng)}
        err = float(np.linalg.norm(self.fk(q) - np.array(xyz)))
        return q, err

    def pose(self, q_rad):
        self._set(q_rad); st = p.getLinkState(self.rid, EE_LINK, computeForwardKinematics=True, physicsClientId=self.cid)
        return np.array(st[4]), np.array(st[5])           # position, orientation quaternion (x,y,z,w)

    def ik_pose(self, xyz, quat, seed_rad: dict[int, float], iters: int = 300):
        """Position + orientation IK on joints 1-6, seeded (stays near the seed configuration). Returns (q_rad, pos_err_m, ori_err_deg)."""
        self._set(seed_rad); rng = np.arange(1, 7)
        sol = p.calculateInverseKinematics(self.rid, EE_LINK, list(xyz), targetOrientation=list(quat), lowerLimits=self.lo, upperLimits=self.hi,
                                           jointRanges=[h - l for l, h in zip(self.lo, self.hi)], restPoses=[seed_rad.get(int(i), 0.0) for i in rng],
                                           maxNumIterations=iters, residualThreshold=1e-5, physicsClientId=self.cid)
        q = {int(i): float(np.clip(sol[k], self.lo[k], self.hi[k])) for k, i in enumerate(rng)}
        pos, ori = self.pose({**seed_rad, **q}); dot = abs(float(np.dot(ori, np.array(quat)))); ang = 2 * np.degrees(np.arccos(min(1.0, dot)))
        return q, float(np.linalg.norm(pos - np.array(xyz))), float(ang)

    def ik_norm(self, xyz, seed_norm: dict[int, float] | None = None):
        seed = {i: norm_to_rad(i, v) for i, v in (seed_norm or {}).items() if i <= 6}
        q, err = self.ik(xyz, seed)
        return {i: rad_to_norm(i, r) for i, r in q.items()}, err

    # --- 7-DoF jaw-tip FK/IK (uses exact URDF/MJCF chain including wrist roll joint 7) ---
    def _ensure_mj(self):
        if getattr(self, "_mj_model", None) is None:
            import mujoco
            scene_xml = Path(__file__).resolve().parents[2] / "assets/mujoco/elrobot_scene.xml"
            self._mj_model = mujoco.MjModel.from_xml_path(str(scene_xml))
            self._mj_data = mujoco.MjData(self._mj_model)
            self._mj_tip_sid = mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_SITE, "jaw_tip_site")
            self._mj_ee_sid = mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
            self._mj_gb_bid = mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_BODY, "Gripper_Base_v1_1")
            self._mj_wcam_id = mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist")
            self._mj_jids = [mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_JOINT, f"rev_motor_{i:02d}") for i in range(1, 8)]
            self._mj_qadr = [int(self._mj_model.jnt_qposadr[j]) for j in self._mj_jids]
            self._mj_dadr = [int(self._mj_model.jnt_dofadr[j]) for j in self._mj_jids]
            self._mj_lo = np.array([URDF_LIMITS[i][0] + 0.02 for i in range(1, 8)], dtype=float)
            self._mj_hi = np.array([URDF_LIMITS[i][1] - 0.02 for i in range(1, 8)], dtype=float)

    def jaw_tip_pose(self, q_rad: dict[int, float]) -> tuple[np.ndarray, np.ndarray]:
        """Returns (jaw_tip_xyz, Gripper_Base_3x3_rot) for joints 1..7."""
        import mujoco
        self._ensure_mj()
        for k in range(7):
            self._mj_data.qpos[self._mj_qadr[k]] = float(q_rad.get(k + 1, 0.0))
        mujoco.mj_kinematics(self._mj_model, self._mj_data)
        return self._mj_data.site_xpos[self._mj_tip_sid].copy(), self._mj_data.xmat[self._mj_gb_bid].reshape(3, 3).copy()

    def wrist_cam_extrinsics(self, q_rad: dict[int, float]) -> tuple[np.ndarray, np.ndarray]:
        """Returns (cam_pos_world, R_wc_opencv) of the wrist camera from live joint angles 1..7."""
        import mujoco
        self._ensure_mj()
        for k in range(7):
            self._mj_data.qpos[self._mj_qadr[k]] = float(q_rad.get(k + 1, 0.0))
        mujoco.mj_kinematics(self._mj_model, self._mj_data)
        mujoco.mj_camlight(self._mj_model, self._mj_data)
        c_pos = self._mj_data.cam_xpos[self._mj_wcam_id].copy()
        r_mj = self._mj_data.cam_xmat[self._mj_wcam_id].reshape(3, 3)
        r_cv = np.column_stack([r_mj[:, 0], -r_mj[:, 1], -r_mj[:, 2]])
        return c_pos, r_cv

    def ik_jaw_tip(self, target_tip_xyz, tilt_deg: float | None = None, roll_rad: float = 0.0, seed_rad: dict[int, float] | None = None, lock_planar_roll: bool = False):
        """7-DoF damped least-squares IK on jaw_tip_site with overhead/adaptive pitch.
        When lock_planar_roll=True, locks q5=0 and q7=roll_rad so the gripper moves purely in the vertical radial plane (zero sideways cant!).
        Returns (q_rad dict 1..7, pos_err_m, ori_err_deg, used_tilt_deg)."""
        import math
        import mujoco
        from scipy.spatial.transform import Rotation as R
        self._ensure_mj()
        tgt = np.asarray(target_tip_xyz, dtype=float)
        x, y, z = float(tgt[0]), float(tgt[1]), float(tgt[2])
        yaw = math.atan2(x, y)
        r_h = math.hypot(x, y)
        if tilt_deg is None:
            if r_h < 0.24:
                t_d = 28.0 + max(0.0, z - 0.11) * 180.0
            else:
                t_d = 18.0 + (r_h - 0.24) * 310.0 + max(0.0, z - 0.11) * 140.0
            t_d = float(np.clip(t_d, 15.0, 76.0))
        else:
            t_d = float(tilt_deg)
        tilt = math.radians(t_d)
        rad_xy = np.array([math.sin(yaw), math.cos(yaw), 0.0])
        y_ax = math.sin(tilt) * rad_xy + math.cos(tilt) * np.array([0.0, 0.0, -1.0])
        x_base = np.array([math.cos(yaw), -math.sin(yaw), 0.0])
        z_base = np.cross(x_base, y_ax)
        x_ax = math.cos(roll_rad) * x_base + math.sin(roll_rad) * z_base
        z_ax = np.cross(x_ax, y_ax)
        target_rot = np.column_stack([x_ax, y_ax, z_ax])

        seeds = []
        if seed_rad:
            seeds.append([float(seed_rad.get(i, 0.0)) for i in range(1, 8)])
        seeds.extend([
            [-yaw, -0.85, 0.95, 1.05, 0.0, 0.85, roll_rad],
            [-yaw, -0.35, 0.65, 0.80, 0.0, 0.55, roll_rad],
            [-yaw, 0.25, 0.35, 0.45, 0.0, 0.30, roll_rad],
            [-yaw, 0.65, 0.10, 0.15, 0.0, 0.10, roll_rad],
        ])
        m, d = self._mj_model, self._mj_data
        jacp = np.zeros((3, m.nv))
        jacr = np.zeros((3, m.nv))
        best = None
        seed_vec = np.array([float(seed_rad.get(i, 0.0)) for i in range(1, 8)]) if seed_rad else None
        active_cols = [0, 1, 2, 3, 5] if lock_planar_roll else [0, 1, 2, 3, 4, 5, 6]
        active_dadr = [self._mj_dadr[c] for c in active_cols]
        for s_idx, s in enumerate(seeds):
            q = np.clip(np.array(s, dtype=float), self._mj_lo, self._mj_hi)
            if lock_planar_roll:
                q[4] = 0.0
                q[6] = float(np.clip(roll_rad, self._mj_lo[6], self._mj_hi[6]))
            max_it = 90 if s_idx == 0 and seed_rad else 80
            for _ in range(max_it):
                for k in range(7):
                    d.qpos[self._mj_qadr[k]] = q[k]
                mujoco.mj_kinematics(m, d)
                mujoco.mj_comPos(m, d)
                ep = tgt - d.site_xpos[self._mj_tip_sid]
                er = R.from_matrix(target_rot @ d.xmat[self._mj_gb_bid].reshape(3, 3).T).as_rotvec()
                if np.linalg.norm(ep) < 5e-4 and np.linalg.norm(er) < 1e-2:
                    break
                mujoco.mj_jacSite(m, d, jacp, jacr, self._mj_tip_sid)
                w_ori = 0.25 if r_h < 0.33 else 0.08
                J = np.vstack([jacp[:, active_dadr], w_ori * jacr[:, active_dadr]])
                e = np.concatenate([ep, w_ori * er])
                dq_sub = J.T @ np.linalg.solve(J @ J.T + 6e-4 * np.eye(6), e)
                for ci, col_k in enumerate(active_cols):
                    q[col_k] = float(np.clip(q[col_k] + 0.85 * dq_sub[ci], self._mj_lo[col_k], self._mj_hi[col_k]))
            for k in range(7):
                d.qpos[self._mj_qadr[k]] = q[k]
            mujoco.mj_kinematics(m, d)
            pe = float(np.linalg.norm(tgt - d.site_xpos[self._mj_tip_sid]))
            oe = float(np.degrees(np.linalg.norm(R.from_matrix(target_rot @ d.xmat[self._mj_gb_bid].reshape(3, 3).T).as_rotvec())))
            reg = 0.0015 * float(np.linalg.norm(q - seed_vec)) if seed_vec is not None else 0.0
            score = pe + 0.00015 * oe + reg
            if best is None or score < best[0]:
                best = (score, q.copy(), pe, oe)
            if s_idx == 0 and seed_rad and pe < 0.0025 and oe < 4.0:
                break
            if pe < 0.0015 and oe < 3.5:
                break
        _, q_best, pe_best, oe_best = best
        return {i + 1: float(q_best[i]) for i in range(7)}, pe_best, oe_best, t_d

