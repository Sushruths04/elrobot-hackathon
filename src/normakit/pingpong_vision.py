"""External front-camera orange ping-pong ball detector, 3D blade-frame localizer, and FdVel velocity filter.

Runs identically in MuJoCo camera-in-the-loop evaluation and on the real ElRobot arm (`StationBackend` front camera).
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BALL_RADIUS_M = 0.020
BLADE_HALF_THICK_M = 0.0035
BLADE_RADIUS_M = 0.078


class FdVel3D:
    """Smoothed finite-difference ball velocity in the blade frame (u, w, h).
    Exactly the same filter runs in PPO training and on real hardware (zecice/so101-pingpong-balance)."""

    def __init__(self, alpha: float = 0.50, dt: float = 1.0 / 30.0):
        self.alpha = float(alpha)
        self.dt = float(dt)
        self.pos: np.ndarray | None = None
        self.vel = np.zeros(3, dtype=np.float64)

    def reset(self) -> None:
        self.pos = None
        self.vel[:] = 0.0

    def update(self, pos_uwh: np.ndarray, dt: float | None = None) -> np.ndarray:
        p = np.asarray(pos_uwh, dtype=np.float64)
        step_dt = max(1e-3, float(dt or self.dt))
        if self.pos is not None:
            raw_v = (p - self.pos) / step_dt
            self.vel = (1.0 - self.alpha) * self.vel + self.alpha * raw_v
        self.pos = p.copy()
        return self.vel.copy()


def detect_orange_ball(rgb: np.ndarray, min_area: int = 45) -> dict:
    """Detect an orange 40 mm ping-pong ball in an RGB 640x480 frame.
    Returns {"visible": bool, "u": float, "v": float, "r_px": float}."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, np.array([8, 145, 110], np.uint8), np.array([25, 255, 255], np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return {"visible": False, "u": 0.0, "v": 0.0, "r_px": 0.0}
    c = max(cnts, key=cv2.contourArea)
    area = float(cv2.contourArea(c))
    if area < min_area:
        return {"visible": False, "u": 0.0, "v": 0.0, "r_px": 0.0}
    (cx, cy), r_encl = cv2.minEnclosingCircle(c)
    r_area = float(np.sqrt(area / np.pi))
    r_px = 0.5 * (float(r_encl) + r_area)
    return {"visible": True, "u": float(cx), "v": float(cy), "r_px": r_px}


class FrontCameraBallTracker:
    """Maps 2D orange-ball detections from the external front camera into 3D blade-frame (u, w, h)."""

    def __init__(
        self,
        cam_calib_path: str | Path = ROOT / "calibration/camera_front.json",
        cam_pos_world: np.ndarray | None = None,
        R_wc_opencv: np.ndarray | None = None,
        use_mujoco_intrinsics: bool = False,
    ):
        if use_mujoco_intrinsics:
            import math
            import mujoco
            from .mujoco_backend import SCENE_XML
            m = mujoco.MjModel.from_xml_path(str(SCENE_XML))
            d = mujoco.MjData(m)
            mujoco.mj_forward(m, d)
            cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "front")
            fovy_rad = math.radians(float(m.cam_fovy[cid]))
            f_mj = 240.0 / math.tan(0.5 * fovy_rad)
            self.fx = self.fy = self.f_mean = f_mj
            self.cx, self.cy = 320.0, 240.0
            self.K = np.array([[f_mj, 0.0, 320.0], [0.0, f_mj, 240.0], [0.0, 0.0, 1.0]])
            if cam_pos_world is None or R_wc_opencv is None:
                cam_pos_world = d.cam_xpos[cid].copy()
                r_mj = d.cam_xmat[cid].reshape(3, 3)
                R_wc_opencv = np.column_stack([r_mj[:, 0], -r_mj[:, 1], -r_mj[:, 2]])
        else:
            cal = json.loads(Path(cam_calib_path).read_text())
            self.K = np.array(cal["K"], dtype=np.float64)
            self.fx = float(self.K[0, 0])
            self.fy = float(self.K[1, 1])
            self.cx = float(self.K[0, 2])
            self.cy = float(self.K[1, 2])
            self.f_mean = 0.5 * (self.fx + self.fy)
            if cam_pos_world is None or R_wc_opencv is None:
                from .mujoco_backend import SCENE_XML
                import mujoco
                m = mujoco.MjModel.from_xml_path(str(SCENE_XML))
                d = mujoco.MjData(m)
                mujoco.mj_forward(m, d)
                cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "front")
                cam_pos_world = d.cam_xpos[cid].copy()
                r_mj = d.cam_xmat[cid].reshape(3, 3)
                R_wc_opencv = np.column_stack([r_mj[:, 0], -r_mj[:, 1], -r_mj[:, 2]])
        self.cam_pos = np.asarray(cam_pos_world, dtype=np.float64)
        self.R_wc = np.asarray(R_wc_opencv, dtype=np.float64)
        self._r_px_filt: float | None = None

    def localize(
        self,
        det: dict,
        blade_pos: np.ndarray,
        blade_mat: np.ndarray,
        mode: str = "balance",
    ) -> np.ndarray | None:
        """Returns ball offset (u, w, h) in metres in the paddle blade frame:
          u = right (+X_blade), w = forward (+Y_blade), h = height above top rubber surface (+Z_blade)."""
        if not det.get("visible"):
            return None
        u_px, v_px, r_px = float(det["u"]), float(det["v"]), float(det["r_px"])
        ray_cv = np.array([(u_px - self.cx) / self.fx, (v_px - self.cy) / self.fy, 1.0], dtype=np.float64)
        d_world = self.R_wc @ ray_cv

        u_ax = blade_mat[:, 0]
        w_ax = blade_mat[:, 1]
        n_ax = blade_mat[:, 2]
        surf_origin = blade_pos + (BLADE_HALF_THICK_M + BALL_RADIUS_M) * n_ax

        if mode == "balance":
            # Exact ray-plane intersection onto the live FK paddle contact plane!
            denom = float(np.dot(d_world, n_ax))
            if abs(denom) < 1e-4:
                return None
            lam = float(np.dot(surf_origin - self.cam_pos, n_ax)) / denom
            p_world = self.cam_pos + lam * d_world
            rel = p_world - surf_origin
            return np.array([float(np.dot(rel, u_ax)), float(np.dot(rel, w_ax)), 0.0], dtype=np.float64)

        # 3D flight mode (dribble / keep-up): apparent diameter depth Z_cam = f * R_ball / r_px along ray
        if r_px < 4.0:
            return None
        self._r_px_filt = r_px if self._r_px_filt is None else (0.30 * self._r_px_filt + 0.70 * r_px)
        z_cam = self.f_mean * BALL_RADIUS_M / self._r_px_filt
        p_world = self.cam_pos + z_cam * d_world
        rel = p_world - surf_origin
        h_est = max(0.0, float(np.dot(rel, n_ax)))
        return np.array([float(np.dot(rel, u_ax)), float(np.dot(rel, w_ax)), h_est], dtype=np.float64)
