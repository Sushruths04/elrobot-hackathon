"""Calibration-free front-camera red-JOOLA-bat + white/orange ping-pong ball tracker & FdVel3D filter.

Verified directly against `results/photos/bat_grip_front_20261122_105923_{1,2,3}.jpg`:
  1. Finds the crimson/red JOOLA paddle blade convex hull inside central workspace ROI (ignoring the red box at u < 190).
  2. Finds the 40 mm ping-pong ball (WHITE or ORANGE automatically) inside/above the detected blade hull.
  3. Computes ball offset directly relative to the live detected blade ellipse center & radii:
       u_blade (-X_world, along handle->blade) = +(u_ball - u_pad) / rx_px * BLADE_RADIUS_M
       w_blade (+Y_world, toward front camera) = +(v_ball - v_pad) / ry_px * BLADE_RADIUS_M
     Because both bat and ball are measured in the same camera image, centering is immune to camera extrinsics error
     or small shifts of the wooden handle in the jaws!
"""
from __future__ import annotations

import cv2
import numpy as np

BALL_RADIUS_M = 0.020
BLADE_HALF_THICK_M = 0.0035
BLADE_RADIUS_M = 0.077


class FdVel3D:
    """Smoothed finite-difference ball velocity in the blade frame (u, w, h)."""

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


def detect_bat_and_ball(
    rgb: np.ndarray,
    roi: tuple[int, int, int, int] = (195, 100, 565, 380),
    ball_color: str = "auto",
) -> dict:
    """Detect both the red JOOLA bat ellipse and the ball (white or orange) on/above it.
    Returns:
      {
        "bat_visible": bool, "bat_u": float, "bat_v": float, "bat_rx": float, "bat_ry": float,
        "visible": bool, "u": float, "v": float, "r_px": float,
        "uwh_m": np.ndarray([u_m, w_m, h_m]) or None,
      }
    """
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    u0, v0, u1, v1 = roi

    # 1) Red/crimson JOOLA rubber mask inside workspace ROI (excludes left orange-red box at u < 195!)
    m_red1 = cv2.inRange(hsv, np.array([158, 28, 95], np.uint8), np.array([180, 245, 255], np.uint8))
    m_red2 = cv2.inRange(hsv, np.array([0, 30, 95], np.uint8), np.array([7, 220, 255], np.uint8))
    bat_m = m_red1 | m_red2
    roi_m = np.zeros_like(bat_m)
    roi_m[v0:v1, u0:u1] = 255
    bat_m &= roi_m
    bat_m = cv2.morphologyEx(bat_m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17)))
    bat_m = cv2.morphologyEx(bat_m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))

    cnts, _ = cv2.findContours(bat_m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return {"bat_visible": False, "visible": False, "uwh_m": None}
    c_bat = max(cnts, key=cv2.contourArea)
    hull = cv2.convexHull(c_bat)
    if cv2.contourArea(hull) < 1800:
        return {"bat_visible": False, "visible": False, "uwh_m": None}

    bx, by, bw, bh = cv2.boundingRect(hull)
    M = cv2.moments(hull)
    bat_u = float(M["m10"] / max(1e-5, M["m00"]))
    bat_v = float(M["m01"] / max(1e-5, M["m00"]))
    bat_rx = max(35.0, 0.5 * float(bw))
    bat_ry = max(22.0, 0.5 * float(bh))

    # Build search mask inside the convex hull of the red bat (dilated slightly for balls near the rim)
    hull_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    cv2.drawContours(hull_mask, [hull], -1, 255, thickness=-1)
    search_mask = cv2.dilate(hull_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19)))

    ball_det = None
    # 2a) Check bright orange ping-pong ball first (H in 9..26, high saturation)
    if ball_color in ("auto", "orange"):
        m_org = cv2.inRange(hsv, np.array([9, 130, 115], np.uint8), np.array([26, 255, 255], np.uint8)) & search_mask
        m_org = cv2.morphologyEx(m_org, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        oc, _ = cv2.findContours(m_org, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if oc:
            co = max(oc, key=cv2.contourArea)
            ao = float(cv2.contourArea(co))
            if 35 <= ao <= 3500:
                (cu, cv_p), r_e = cv2.minEnclosingCircle(co)
                ball_det = (float(cu), float(cv_p), 0.5 * (float(r_e) + float(np.sqrt(ao / np.pi))))

    # 2b) Check white ping-pong ball on the red JOOLA rubber (low saturation S <= 38, bright V >= 150)
    if ball_det is None and ball_color in ("auto", "white"):
        m_wht = cv2.inRange(hsv, np.array([0, 0, 150], np.uint8), np.array([180, 38, 255], np.uint8)) & search_mask
        m_wht = cv2.morphologyEx(m_wht, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        wc, _ = cv2.findContours(m_wht, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best_w = None
        for cw in wc:
            aw = float(cv2.contourArea(cw))
            if 35 <= aw <= 3500:
                (cu, cv_p), r_e = cv2.minEnclosingCircle(cw)
                circ = aw / max(1.0, np.pi * (r_e ** 2))
                if circ >= 0.50 and (best_w is None or aw > best_w[0]):
                    best_w = (aw, (float(cu), float(cv_p), 0.5 * (float(r_e) + float(np.sqrt(aw / np.pi)))))
        if best_w is not None:
            ball_det = best_w[1]

    if ball_det is None:
        return {
            "bat_visible": True, "bat_u": bat_u, "bat_v": bat_v, "bat_rx": bat_rx, "bat_ry": bat_ry,
            "visible": False, "u": 0.0, "v": 0.0, "r_px": 0.0, "uwh_m": None,
        }

    bu, bv, br = ball_det
    # In front camera:
    #   +u_blade (+X_paddle = -Z_gb = +X_world) points IMAGE-LEFT (-du_px)!
    #   +w_blade (+Y_paddle = +Y_gb = +Y_world, toward camera) points IMAGE-DOWN (+dv_px)!
    # At ~30 deg camera elevation, a 20mm ball's center projects ~1.48 * r_px above its blade contact point!
    contact_v = bv + 1.48 * br
    u_m = -((bu - bat_u) / bat_rx) * BLADE_RADIUS_M
    w_m = ((contact_v - bat_v) / bat_ry) * BLADE_RADIUS_M
    # Apparent ball radius on blade is ~0.26 * bat_rx (~17 px); larger br means ball is higher above the blade!
    nom_r = 0.26 * bat_rx
    h_m = max(0.0, (br / max(6.0, nom_r) - 1.0) * 0.38)
    return {
        "bat_visible": True, "bat_u": bat_u, "bat_v": bat_v, "bat_rx": bat_rx, "bat_ry": bat_ry,
        "visible": True, "u": bu, "v": bv, "r_px": br,
        "uwh_m": np.array([float(u_m), float(w_m), float(h_m)], dtype=np.float64),
    }
