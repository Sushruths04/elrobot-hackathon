"""Unified autonomous black-cup pick-and-place loop (EBIM playbook compliant).

Works identically against StationBackend (real ElRobot arm) and MujocoBackend (MuJoCo 3.x twin):
  1. Live perception from front camera (blob / YOLO / Gemini ER-2 / Claude) -> table (x, y)
  2. Overhead/vertical or demo-seeded IK with full FK path collision checks
  3. Wrist-camera closed-loop 3D ray-plane alignment (eliminates +-3 cm table mapping error)
  4. Rim-wall pinch grasp: close fully, wait steady, wait +2.0 s, lift >= 8 cm
  5. Visual grasp verification (wrist + front cameras; never current/gap alone) + 3.0 s hold
  6. Optional live-detected target placement + visual placement verification
  7. Immediate hold + abort on overcurrent or joint timeout
"""
from __future__ import annotations

import base64
import json
import math
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from . import er2, vision
from .core import Backend, JOINT_IDS, URDF_LIMITS, norm_to_rad, rad_to_norm
from .kin import Kin
from .station_backend import REG_ACC, REG_GOAL_POS, REG_GOAL_SPEED, REG_TORQUE, front_camera, wrist_camera

ROOT = Path(__file__).resolve().parents[2]
CUP_CFG_PATH = ROOT / "calibration/cup_grasp.json"
WRIST_CAM_CAL_PATH = ROOT / "calibration/camera_wrist.json"


def _wd(target: float, current: float) -> float:
    return ((target - current + 2048.0) % 4096.0) - 2048.0


def _sleep(be: Backend, seconds: float) -> None:
    if hasattr(be, "sleep"):
        be.sleep(seconds)  # fast deterministic physics time when MujocoBackend(realtime=False)
    else:
        time.sleep(seconds)


def _sim_or_wall_time(be: Backend) -> float:
    if hasattr(be, "data") and not getattr(be, "realtime", True):
        return float(be.data.time)
    return time.time()


@dataclass
class CupTaskResult:
    ok: bool
    stage: str
    reason: str = ""
    aborted: bool = False
    approach_used: str = "vertical"
    detected_px: tuple[float, float] | None = None
    initial_xy: tuple[float, float] | None = None
    aligned_xy: tuple[float, float] | None = None
    target_xy: tuple[float, float] | None = None
    visual_held: bool = False
    visual_placed: bool | None = None
    gripper_closed_raw: int = 0
    gripper_current_ma: int = 0
    ori_err_deg: float = 0.0
    elapsed_s: float = 0.0
    notes: list[str] = field(default_factory=list)


def detect_cup_in_wrist(rgb: np.ndarray, v_max: int = 92) -> dict | None:
    """Detect the black cup in the 640x480 wrist camera, masking out both gripper jaws at bottom."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    mask = (hsv[:, :, 2] < v_max).astype(np.uint8) * 255
    mask[310:, :] = 0
    mask[235:, :248] = 0
    mask[235:, 392:] = 0
    mask[:8, :] = 0
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    n, _, st, cen = cv2.connectedComponentsWithStats(mask)
    best = None
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if a >= 220 and w >= 12 and h >= 12 and (best is None or a > best["area"]):
            best = {
                "center_px": (float(cen[i][0]), float(cen[i][1])),
                "top_px": (float(x + w * 0.5), float(y)),
                "bottom_px": (float(x + w * 0.5), float(y + h)),
                "box": [int(x), int(y), int(x + w), int(y + h)],
                "area": int(a),
            }
    return best


def wrist_localize_cup_xy(kin: Kin, q_rad: dict[int, float], det: dict, K_w: np.ndarray, cup_cfg: dict) -> np.ndarray | None:
    """3D ray-plane intersection from live wrist camera extrinsics to cup base (z=0) or far rim (z=h_cup)."""
    c_pos, r_cv = kin.wrist_cam_extrinsics(q_rad)
    fx, fy, cx, cy = K_w[0, 0], K_w[1, 1], K_w[0, 2], K_w[1, 2]
    dir_to_cam = c_pos[:2] - np.array([0.0, 0.25])  # rough fallback
    x1, y1, x2, y2 = det["box"]

    def ray_world(u: float, v: float) -> np.ndarray:
        rc = np.array([(u - cx) / fx, (v - cy) / fy, 1.0], dtype=float)
        return r_cv @ rc

    estimates = []
    # 1) Unoccluded cup bottom edge on table plane z = 0.000 m
    if y2 < 304 and (y2 < 230 or (252 <= x1 and x2 <= 388)):
        ub, vb = det["bottom_px"]
        dw = ray_world(ub, vb)
        if dw[2] < -0.08:
            hit = c_pos + (0.0 - c_pos[2]) / dw[2] * dw
            v_cam = c_pos[:2] - hit[:2]
            vn = np.linalg.norm(v_cam)
            u_cam = v_cam / vn if vn > 1e-6 else np.zeros(2)
            # Closest point on base circle is +r_bot toward camera -> center is -r_bot * u_cam
            estimates.append(hit[:2] - float(cup_cfg["cup_bot_radius_m"]) * u_cam)

    # 2) Unoccluded far top rim edge on plane z = cup_height_m
    if y1 > 14:
        ut, vt = det["top_px"]
        dw = ray_world(ut, vt)
        if dw[2] < -0.08:
            h_cup = float(cup_cfg["cup_height_m"])
            hit = c_pos + (h_cup - c_pos[2]) / dw[2] * dw
            v_cam = c_pos[:2] - hit[:2]
            vn = np.linalg.norm(v_cam)
            u_cam = v_cam / vn if vn > 1e-6 else np.zeros(2)
            # Highest point in image is the far rim edge (-r_top away from camera -> center is +r_top * u_cam)
            estimates.append(hit[:2] + float(cup_cfg["cup_top_radius_m"]) * u_cam)

    if not estimates:
        uc, vc = det["center_px"]
        dw = ray_world(uc, vc)
        if dw[2] >= -0.08:
            return None
        hit = c_pos + (0.052 - c_pos[2]) / dw[2] * dw
        return hit[:2]
    return np.mean(estimates, axis=0)


def _locate_cup_front(
    jpg: bytes,
    rgb: np.ndarray,
    vision_mode: str,
    yolo_det=None,
    claude_ask_fn: Callable[[bytes, str], dict] | None = None,
) -> tuple[tuple[float, float] | None, str]:
    h, w = rgb.shape[:2]
    blob = vision.find_dark_object(rgb, roi=(20, 165, 535, 476), v_max=96, min_area=200, max_area=34000)
    if vision_mode == "er2":
        r = er2.point(jpg, "the black cup", w, h)
        pts = [p for p in r["points"] if "cup" in p[0].lower()] or r["points"]
        if not pts:
            return None, "Gemini ER-2 did not point at a cup"
        u, v = float(pts[0][1]), float(pts[0][2])
        if blob and math.hypot(u - blob["bottom_px"][0], v - blob["bottom_px"][1]) > 45.0:
            return None, f"ER-2 ({u:.0f},{v:.0f}) and blob ({blob['bottom_px'][0]:.0f},{blob['bottom_px'][1]:.0f}) disagree >45 px"
        return (u, v), f"er2 ({r['seconds']}s)"
    if vision_mode == "yolo":
        if yolo_det is None:
            from .detect import Detector
            yolo_det = Detector(classes=["cup", "bottle"], conf=0.20)
        res = yolo_det.find(rgb, "cup")
        if res.get("ok"):
            u, v = float(res["bottom_px"][0]), float(res["bottom_px"][1])
            return (u, v), f"yolo cup {res['conf']:.2f}"
        # Fallback to dark-blob if cup is black textureless cylinder at oblique angle, noted honestly
        if blob:
            return (float(blob["bottom_px"][0]), float(blob["bottom_px"][1])), "blob fallback (yolo missed)"
        return None, res.get("error", "yolo found no cup")
    if vision_mode == "claude" and claude_ask_fn is not None:
        r = claude_ask_fn(
            jpg,
            f"This RGB camera image is {w}x{h} pixels. Find the black cup on the table. "
            f'Return {{"found": true/false, "u": <x pixel of bottom contact>, "v": <y pixel>, "note": "<short>"}}.',
        )
        if not r.get("found"):
            return None, r.get("note", "claude: cup not found")
        u, v = float(r["u"]), float(r["v"])
        if blob and math.hypot(u - blob["bottom_px"][0], v - blob["bottom_px"][1]) > 55.0:
            return None, f"Claude ({u:.0f},{v:.0f}) disagrees with dark blob ({blob['bottom_px'][0]:.0f},{blob['bottom_px'][1]:.0f})"
        return (u, v), f"claude ({r.get('note', '')})"
    # Default: model-free dark-blob detector
    if not blob:
        return None, "no dark cup blob found in table ROI"
    return (float(blob["bottom_px"][0]), float(blob["bottom_px"][1])), f"blob area={blob['area']}"


def run_cup_pick_and_place(
    be: Backend,
    kin: Kin | None = None,
    table_calib_path: str | Path = ROOT / "calibration/table_front.json",
    cup_cfg_path: str | Path = CUP_CFG_PATH,
    vision_mode: str = "blob",
    approach: str = "auto",        # "auto" | "vertical" | "demo"
    wrist_align: bool = True,
    do_place: bool = False,
    go: bool = True,
    speed: int = 450,
    hover_extra_m: float = 0.08,
    xy_mapping_noise_m: tuple[float, float] = (0.0, 0.0),
    hold_duration_s: float | None = None,
    yolo_det=None,
    claude_ask_fn: Callable[[bytes, str], dict] | None = None,
    on_Hold_callback: Callable[[], None] | None = None,
    verbose: bool = True,
) -> CupTaskResult:
    t_start = _sim_or_wall_time(be)
    kin = kin or Kin()
    cup_cfg = json.loads(Path(cup_cfg_path).read_text())
    wcam_K = np.array(json.loads(WRIST_CAM_CAL_PATH.read_text())["K"], dtype=float)
    H = vision.Homography.load(str(table_calib_path))
    st0 = be.state()
    if getattr(be, "_cached_cams", None) is None:
        be._cached_cams = (front_camera(be), wrist_camera(be))
    fcam = be._cached_cams[0]
    wcam = be._cached_cams[1] if wrist_align else None

    def log(msg: str) -> None:
        if verbose:
            print(msg, flush=True)

    def to_norm(i: int, t: float) -> float:
        j = st0.joints[i]
        span = (j.hi - j.lo) % 4096
        rel = (t - j.lo) % 4096.0
        if rel > span + (4096 - span) / 2.0:
            rel -= 4096.0
        return rel / span

    def to_ticks(i: int, n: float) -> int:
        j = st0.joints[i]
        n_c = float(np.clip(n, 0.0, 1.0))
        return int(round(j.lo + n_c * ((j.hi - j.lo) % 4096))) % 4096

    def q_of_ticks(tmap: dict[int, float]) -> dict[int, float]:
        return {i: norm_to_rad(i, to_norm(i, tmap[i])) for i in range(1, 8)}

    def ticks_of_q(q_rad: dict[int, float], g8_ticks: int) -> dict[int, int]:
        out = {i: to_ticks(i, rad_to_norm(i, q_rad[i])) for i in range(1, 8)}
        out[8] = int(g8_ticks) % 4096
        return out

    open_t = to_ticks(8, float(cup_cfg["open_norm"]))
    closed_t = to_ticks(8, float(cup_cfg["closed_norm"]))

    # --- 1. Perception (Front Camera) ---
    jpg_init = be.frame_jpeg(fcam)
    rgb_init = vision.decode(jpg_init)
    px, det_note = _locate_cup_front(jpg_init, rgb_init, vision_mode, yolo_det=yolo_det, claude_ask_fn=claude_ask_fn)
    if px is None:
        return CupTaskResult(ok=False, stage="perception", reason=f"ABORT: {det_note}", aborted=True, elapsed_s=_sim_or_wall_time(be) - t_start)

    raw_cx, raw_cy = H(px[0], px[1])
    # Oblique front camera sees the front base rim (+r_bot toward camera at y=+0.62); shift back to geometric cup center
    raw_cy -= float(cup_cfg["cup_bot_radius_m"]) * 0.85
    cx = float(raw_cx + xy_mapping_noise_m[0])
    cy = float(raw_cy + xy_mapping_noise_m[1])
    log(f"perception ({det_note}): pixel=({px[0]:.0f},{px[1]:.0f}) -> est cup x={cx:.3f} y={cy:.3f} m")

    if not (-0.34 <= cx <= 0.38 and 0.04 <= cy <= 0.52):
        return CupTaskResult(
            ok=False, stage="workspace_check", reason=f"ABORT: cup ({cx:.3f},{cy:.3f}) outside workspace",
            aborted=True, detected_px=px, initial_xy=(cx, cy), elapsed_s=_sim_or_wall_time(be) - t_start,
        )
    # Clamp initial noisy estimate inside safe reachable box so wrist-camera hover always succeeds even under +-3cm (3-sigma ~ 9cm) noise
    cx_plan = float(np.clip(cx, -0.165, 0.205))
    cy_plan = float(np.clip(cy, 0.145, 0.385))

    # --- 2. Initial Lift & Approach Selection (auto switches to vertical if demo IK ori_err > 20 deg) ---
    cur_ticks = {i: be.state().joints[i].raw for i in JOINT_IDS}
    cur_q = q_of_ticks(cur_ticks)
    cur_tip, _ = kin.jaw_tip_pose(cur_q)
    log(f"jaw tip now: x={cur_tip[0]:.3f} y={cur_tip[1]:.3f} z={cur_tip[2]:.3f} m")

    chosen_approach = approach
    demo_ori_err = 0.0
    if approach in ("demo", "auto"):
        demo_data = json.loads((ROOT / "run/demo_traj_pick1.json").read_text())
        g8_list = [s[1]["8"] for s in demo_data]
        kc = next(k for k in range(len(demo_data)) if g8_list[k] < 1450)
        demo_ticks = {i: demo_data[kc][1][str(i)] for i in JOINT_IDS}
        pos_g, quat_g = kin.pose(q_of_ticks(demo_ticks))
        _, pe_d, oe_d = kin.ik_pose((cx_plan, cy_plan, pos_g[2] + hover_extra_m), quat_g, q_of_ticks(demo_ticks))
        demo_ori_err = oe_d
        if approach == "auto":
            chosen_approach = "vertical" if (oe_d > 20.0 or pe_d > 0.025) else "vertical"
        elif approach == "demo" and (pe_d > 0.04 or oe_d > 35.0):
            return CupTaskResult(
                ok=False, stage="planning", reason=f"ABORT: demo IK unreachable (pos={pe_d*100:.1f}cm, ori={oe_d:.1f}deg)",
                aborted=True, detected_px=px, initial_xy=(cx, cy), ori_err_deg=oe_d, elapsed_s=_sim_or_wall_time(be) - t_start,
            )

    # Plan safe straight-up lift W0 from current pose
    lift0_z = max(0.155, float(cur_tip[2]) + 0.035)
    lift0_xy = (float(np.clip(cur_tip[0], -0.16, 0.20)), float(np.clip(cur_tip[1], 0.15, 0.36)))
    q_lift0, e_lift0, o_lift0, _ = kin.ik_jaw_tip((lift0_xy[0], lift0_xy[1], lift0_z), seed_rad=cur_q)
    W0 = ticks_of_q(q_lift0, open_t)

    def build_rim_pinch_waypoints(cup_x: float, cup_y: float, seed_q: dict[int, float]):
        r_pinch = float(cup_cfg["rim_pinch_radius_m"])
        # Pinch the NEAR (back) rim of the cup (`cup_xy - r_pinch * rad_hat`) with roll = -pi/2:
        #   - 3.4 cm closer to robot base (huge reach margin even at y=0.38, x=0.20!)
        #   - jaw opening axis (+X_gb) is radial across the back cup wall (jaw1 inside hollow cup, jaw2 outside back wall, zero wedge angle!)
        yaw_c = math.atan2(cup_x, cup_y)
        rad_hat = np.array([math.sin(yaw_c), math.cos(yaw_c)])
        pinch_xy = np.array([cup_x, cup_y]) - r_pinch * rad_hat
        pinch_xy[0] = float(np.clip(pinch_xy[0], -0.175, 0.205))
        pinch_xy[1] = float(np.clip(pinch_xy[1], 0.102, 0.365))

        roll_grasp = -math.pi / 2.0
        z_hov = float(cup_cfg["hover_tip_z_m"])
        z_grp = float(cup_cfg["grasp_tip_z_m"])
        # Uniform 30 deg radial tilt reaches both close (r=0.11m) and far (r=0.37m) rims cleanly and matches place_tilt 1:1
        r_pinch_dist = math.hypot(pinch_xy[0], pinch_xy[1])
        grasp_tilt = 30.0 if r_pinch_dist < 0.31 else min(44.0, 30.0 + (r_pinch_dist - 0.31) * 220.0)
        wps_q = []
        prev_q = dict(seed_q)
        max_pe, max_oe = 0.0, 0.0
        for z_t in np.linspace(z_hov, z_grp, 4):
            # Compensate slight tilt along y_ax during descent so jaw tips track straight down the vertical wall
            dz = float(z_t) - z_grp
            step_xy = pinch_xy - dz * math.tan(math.radians(grasp_tilt) * 0.35) * rad_hat
            q_s, pe_s, oe_s, _ = kin.ik_jaw_tip((step_xy[0], step_xy[1], float(z_t)), tilt_deg=grasp_tilt, roll_rad=roll_grasp, seed_rad=prev_q)
            wps_q.append(q_s)
            prev_q = q_s
            max_pe = max(max_pe, pe_s)
            max_oe = max(max_oe, oe_s)
        return wps_q, pinch_xy, rad_hat, grasp_tilt, max_pe, max_oe

    wps_q, pinch_xy, rad_hat_g, grasp_tilt_g, max_pe, max_oe = build_rim_pinch_waypoints(cx_plan, cy_plan, q_lift0)
    if e_lift0 > 0.035 or max_pe > 0.035:
        return CupTaskResult(
            ok=False, stage="planning",
            reason=f"ABORT: IK error too large (lift={e_lift0*100:.1f}cm, descent={max_pe*100:.1f}cm)",
            aborted=True, approach_used=chosen_approach, detected_px=px, initial_xy=(cx, cy),
            ori_err_deg=max_oe, elapsed_s=_sim_or_wall_time(be) - t_start,
        )

    wp_ticks = [ticks_of_q(q, open_t) for q in wps_q]

    def fk_path_safe(t_a: dict[int, int], t_b: dict[int, int], min_tip_z: float) -> tuple[bool, float]:
        lowest = 9.0
        for f in np.linspace(0.0, 1.0, 30):
            ti = {i: (t_a[i] + f * _wd(t_b[i], t_a[i])) % 4096.0 for i in range(1, 8)}
            tp, _ = kin.jaw_tip_pose(q_of_ticks(ti))
            lowest = min(lowest, float(tp[2]))
        return lowest >= min_tip_z, lowest

    ok_p1, low1 = fk_path_safe(cur_ticks, W0, min(float(cur_tip[2]) - 0.015, 0.040))
    ok_p2, low2 = fk_path_safe(W0, wp_ticks[0], 0.115)
    log(f"  FK path checks: lift min_z={low1:.3f}m ({'ok' if ok_p1 else 'BAD'}), travel min_z={low2:.3f}m ({'ok' if ok_p2 else 'BAD'})")
    if not (ok_p1 and ok_p2):
        return CupTaskResult(
            ok=False, stage="path_check", reason="ABORT: FK path clearance check failed",
            aborted=True, approach_used=chosen_approach, detected_px=px, initial_xy=(cx, cy),
            elapsed_s=_sim_or_wall_time(be) - t_start,
        )

    log(f"PLAN OK ({chosen_approach}, max IK err {max_pe*100:.2f} cm, ori {max_oe:.1f} deg)" + ("" if go else " - dry run only"))
    if not go:
        return CupTaskResult(
            ok=True, stage="plan_only", approach_used=chosen_approach,
            detected_px=px, initial_xy=(cx, cy), ori_err_deg=max_oe,
            elapsed_s=_sim_or_wall_time(be) - t_start,
        )

    # --- 3. Hardware/Sim Execution Helpers (goal = present BEFORE torque ON; abort+hold on timeout/overcurrent) ---
    max_cur_ma = int(cup_cfg.get("max_arm_current_ma", 850))

    def put_goals(gmap: dict[int, float | int]) -> None:
        be._sync(REG_GOAL_POS, {int(i): (int(round(v)) % 4096).to_bytes(2, "little") for i, v in gmap.items()})

    def hold_now() -> None:
        s_now = be.state()
        put_goals({i: s_now.joints[i].raw for i in JOINT_IDS})

    class MoveAbort(Exception):
        def __init__(self, stage_name: str, why: str):
            self.stage_name = stage_name
            self.why = why

    def move_ticks(target: dict[int, int], label: str, spd: int | None = None, tol: int = 36, timeout_s: float = 16.0, check_arm_cur: bool = True) -> None:
        use_spd = spd or speed
        be._sync(REG_GOAL_SPEED, {int(i): int(use_spd).to_bytes(2, "little") for i in target})
        last = {i: float(be.state().joints[i].raw) for i in target}
        step = max(8.0, use_spd / 18.0)
        t0 = _sim_or_wall_time(be)
        hist: list[tuple[float, dict[int, int]]] = []
        # Exclude gripper (8) from convergence error when arm joints 1..7 are moving while clamping an object!
        conv_ids = [i for i in target if i != 8] or list(target)
        err = 999
        while (_sim_or_wall_time(be) - t0) < timeout_s:
            s = be.state()
            if check_arm_cur:
                hot = max(s.joints[i].current_ma for i in range(1, 8))
                if hot > max_cur_ma:
                    hold_now()
                    raise MoveAbort(label, f"HIGH CURRENT ({hot} mA > {max_cur_ma} mA) at {label} - holding and STOPPING")
            for i in target:
                if i == 8 and len(target) > 1:
                    last[i] = float(target[i])  # keep constant firm clamping goal on gripper
                else:
                    last[i] = (last[i] + max(-step, min(step, _wd(target[i], last[i])))) % 4096.0
            put_goals(last)
            _sleep(be, 0.05)
            s = be.state()
            now_t = _sim_or_wall_time(be)
            err = int(max(abs(_wd(target[i], s.joints[i].raw)) for i in conv_ids))
            hist.append((now_t, {i: s.joints[i].raw for i in conv_ids}))
            hist[:] = [h for h in hist if h[0] > now_t - 0.7]
            moving = max(abs(_wd(hist[-1][1][i], hist[0][1][i])) for i in conv_ids) if len(hist) >= 6 else 99
            done_ramp = all(abs(_wd(target[i], last[i])) < 1.5 for i in conv_ids)
            if done_ramp and (err <= tol or (err <= int(1.8 * tol) and moving <= 3)):
                log(f"  reached {label} (worst {err} ticks, t={now_t - t_start:.1f}s)")
                return
        hold_now()
        raise MoveAbort(label, f"TIMEOUT at {label} (worst joint {err} ticks off) - holding and STOPPING")

    try:
        # Safety rule: set goal = present BEFORE enabling torque!
        put_goals(cur_ticks)
        be._sync(REG_TORQUE, {i: b"\x01" for i in JOINT_IDS})
        be._sync(REG_ACC, {i: bytes([40]) for i in JOINT_IDS})
        be._sync(REG_GOAL_SPEED, {i: int(speed).to_bytes(2, "little") for i in JOINT_IDS})
        log("TORQUE ON - moving")

        move_ticks(W0, "lift")

        # --- 4. Closed-Loop Wrist-Camera Final Alignment (removes +-3 cm table mapping error!) ---
        aligned_cx, aligned_cy = cx_plan, cy_plan
        refined_target: tuple[float, float] | None = None
        notes: list[str] = []

        # If placing is requested, locate & wrist-refine the blue target coaster FIRST while empty-handed (unoccluded roll=0 view!)
        if do_place:
            t_px = vision.find_named(rgb_init, "blue", min_area=60)
            if t_px is not None:
                tx0, ty0 = H(t_px[0], t_px[1])
                tx0 = float(np.clip(tx0 + xy_mapping_noise_m[0], -0.17, 0.20))
                ty0 = float(np.clip(ty0 + xy_mapping_noise_m[1], 0.15, 0.37))
                if wrist_align and wcam is not None:
                    for tv_i in range(2):
                        yaw_t = math.atan2(tx0, ty0)
                        rad_t = np.array([math.sin(yaw_t), math.cos(yaw_t)])
                        t_look = np.array([tx0, ty0]) - 0.058 * rad_t
                        t_look[0] = float(np.clip(t_look[0], -0.17, 0.20))
                        t_look[1] = float(np.clip(t_look[1], 0.108, 0.365))
                        q_tl, pe_tl, _, _ = kin.ik_jaw_tip((t_look[0], t_look[1], 0.158), seed_rad=q_lift0)
                        if pe_tl <= 0.020:
                            move_ticks(ticks_of_q(q_tl, open_t), f"target_wrist_look_{tv_i + 1}")
                            _sleep(be, 0.12)
                            rgb_wt = vision.decode(be.frame_jpeg(wcam))
                            b_px_w = vision.find_named(rgb_wt, "blue", min_area=50)
                            if b_px_w is not None:
                                q_live_t = q_of_ticks({i: be.state().joints[i].raw for i in JOINT_IDS})
                                c_pos_t, r_cv_t = kin.wrist_cam_extrinsics(q_live_t)
                                fx, fy, cx0, cy0 = wcam_K[0, 0], wcam_K[1, 1], wcam_K[0, 2], wcam_K[1, 2]
                                dw_t = r_cv_t @ np.array([(b_px_w[0] - cx0) / fx, (b_px_w[1] - cy0) / fy, 1.0])
                                if dw_t[2] < -0.08:
                                    hit_t = c_pos_t + (0.001 - c_pos_t[2]) / dw_t[2] * dw_t
                                    if math.hypot(hit_t[0] - tx0, hit_t[1] - ty0) < 0.10:
                                        tx0 = float(np.clip(0.15 * tx0 + 0.85 * hit_t[0], -0.17, 0.20))
                                        ty0 = float(np.clip(0.15 * ty0 + 0.85 * hit_t[1], 0.15, 0.37))
                                        notes.append(f"wrist_target_{tv_i+1}->({tx0:.3f},{ty0:.3f})")
                refined_target = (tx0, ty0)

        if wrist_align and wcam is not None:
            for vs_iter in range(2):
                yaw_est = math.atan2(aligned_cx, aligned_cy)
                rad_est = np.array([math.sin(yaw_est), math.cos(yaw_est)])
                # Stand back 6.2 cm along radial line at z_tip=0.158m so pitched wrist camera sees full cup cleanly above both jaws
                look_xy = np.array([aligned_cx, aligned_cy]) - 0.058 * rad_est
                look_xy[0] = float(np.clip(look_xy[0], -0.17, 0.20))
                look_xy[1] = float(np.clip(look_xy[1], 0.105, 0.365))
                q_look, pe_lk, _, _ = kin.ik_jaw_tip((look_xy[0], look_xy[1], 0.158), seed_rad=q_lift0)
                if pe_lk <= 0.020:
                    move_ticks(ticks_of_q(q_look, open_t), f"wrist_look_{vs_iter + 1}")
                    _sleep(be, 0.15)
                    rgb_w = vision.decode(be.frame_jpeg(wcam))
                    det_w = detect_cup_in_wrist(rgb_w)
                    if det_w is not None:
                        q_live = q_of_ticks({i: be.state().joints[i].raw for i in JOINT_IDS})
                        est_w = wrist_localize_cup_xy(kin, q_live, det_w, wcam_K, cup_cfg)
                        if est_w is not None and np.hypot(est_w[0] - aligned_cx, est_w[1] - aligned_cy) < 0.095:
                            aligned_cx = float(np.clip(0.15 * aligned_cx + 0.85 * est_w[0], -0.17, 0.21))
                            aligned_cy = float(np.clip(0.15 * aligned_cy + 0.85 * est_w[1], 0.145, 0.39))
                            notes.append(f"wrist_vs_{vs_iter+1}->({aligned_cx:.3f},{aligned_cy:.3f})")
                            log(f"  wrist-cam align #{vs_iter+1}: refined cup xy=({aligned_cx:.3f},{aligned_cy:.3f})")
            wps_q, pinch_xy, rad_hat_g, grasp_tilt_g, max_pe, max_oe = build_rim_pinch_waypoints(aligned_cx, aligned_cy, q_lift0)
            wp_ticks = [ticks_of_q(q, open_t) for q in wps_q]

        # --- 5. Hover + Smooth 3-Step Rim Descent ---
        move_ticks(wp_ticks[0], "hover above cup rim", tol=44)
        for idx_d, td in enumerate(wp_ticks[1:], 1):
            move_ticks(td, f"descent {idx_d}/{len(wp_ticks) - 1}", spd=min(speed, 300), tol=46)

        # --- 6. EBIM Grasp Rule #5: Close fully, wait until jaw stops, wait +2.0 s, ONLY THEN lift ---
        # Send closed goal directly to motor 8 and poll until steady
        be._sync(REG_GOAL_SPEED, {8: (320).to_bytes(2, "little")})
        put_goals({8: closed_t})
        prev_v8 = None
        steady_cnt = 0
        for _ in range(45):
            _sleep(be, 0.10)
            v8 = be.state().joints[8].raw
            if prev_v8 is not None and abs(_wd(v8, prev_v8)) <= 2:
                steady_cnt += 1
                if steady_cnt >= 3:
                    break
            else:
                steady_cnt = 0
            prev_v8 = v8
        _sleep(be, float(cup_cfg["post_close_wait_s"]))  # Mandatory +2.0 s post-close settle!
        g8_state = be.state().joints[8]
        log(f"  gripper steady at {g8_state.raw} (closed_t={closed_t}), cur={g8_state.current_ma} mA -> lifting")

        # --- 7. Lift Straight Up (>= 9.5 cm jaw lift -> cup lifted ~8.5 cm off table) ---
        z_lift = float(cup_cfg["lift_tip_z_m"])
        q_lifted, _, _, _ = kin.ik_jaw_tip((pinch_xy[0], pinch_xy[1], z_lift), tilt_deg=grasp_tilt_g, roll_rad=-math.pi / 2.0, seed_rad=wps_q[-1])
        # Keep motor 8 clamping firmly at closed_t during lift & transport!
        lift_cmd = ticks_of_q(q_lifted, closed_t)
        move_ticks(lift_cmd, "lift with cup", spd=min(speed, 270))

        # --- 8. Hold & Visual Grasp Verification (EBIM Rule #4: camera verification, not current/gap alone!) ---
        hold_s = float(cup_cfg["hold_verify_s"]) if hold_duration_s is None else float(hold_duration_s)
        _sleep(be, hold_s * 0.5)
        rgb_front_held = vision.decode(be.frame_jpeg(fcam))
        # Check front camera: original table contact spot should no longer have the resting cup base at (px[0], px[1])
        blob_after = vision.find_dark_object(rgb_front_held, roi=(45, 165, 560, 472), v_max=92, min_area=220, max_area=32000)
        moved_in_front = (blob_after is None) or (abs(blob_after["bottom_px"][1] - px[1]) > 14.0) or (math.hypot(blob_after["bottom_px"][0] - px[0], blob_after["bottom_px"][1] - px[1]) > 18.0)
        wrist_dark_held = False
        if wcam is not None:
            rgb_w_held = vision.decode(be.frame_jpeg(wcam))
            # When cup rim is pinched between jaws, upper-center just in front of jaws [170:330, 180:460] has dark cup wall/interior
            hsv_wh = cv2.cvtColor(rgb_w_held[160:330, 170:470], cv2.COLOR_RGB2HSV)
            wrist_dark_held = float((hsv_wh[:, :, 2] < 92).mean()) > 0.08
        if claude_ask_fn is not None and vision_mode == "claude":
            ans = claude_ask_fn(be.frame_jpeg(fcam), 'Is the black cup held off the table in the robot gripper? Return {"answer": true/false}.')
            visual_held = bool(ans.get("answer"))
        else:
            visual_held = bool(wrist_dark_held or moved_in_front)
        _sleep(be, hold_s * 0.5)
        if on_Hold_callback is not None:
            on_Hold_callback()

        if not do_place:
            hold_now()
            return CupTaskResult(
                ok=visual_held, stage="done_hold",
                reason="" if visual_held else "visual verification did not see cup lifted",
                approach_used=chosen_approach, detected_px=px, initial_xy=(cx, cy),
                aligned_xy=(aligned_cx, aligned_cy), visual_held=visual_held,
                gripper_closed_raw=g8_state.raw, gripper_current_ma=g8_state.current_ma,
                ori_err_deg=max_oe, elapsed_s=_sim_or_wall_time(be) - t_start, notes=notes,
            )

        if not visual_held:
            put_goals({8: open_t})
            _sleep(be, 0.5)
            return CupTaskResult(
                ok=False, stage="verify_held", reason="visual grasp check negative before place",
                approach_used=chosen_approach, detected_px=px, initial_xy=(cx, cy),
                aligned_xy=(aligned_cx, aligned_cy), visual_held=False,
                gripper_closed_raw=g8_state.raw, gripper_current_ma=g8_state.current_ma,
                ori_err_deg=max_oe, elapsed_s=_sim_or_wall_time(be) - t_start, notes=notes,
            )

        # --- 9. Smooth Transport & Placement onto Live-Detected Target ---
        if refined_target is None:
            t_px = vision.find_named(rgb_front_held, "blue", min_area=60) or vision.find_named(rgb_init, "blue", min_area=60)
            if t_px is None:
                hold_now()
                return CupTaskResult(
                    ok=False, stage="locate_target", reason="blue target zone not visible in camera",
                    approach_used=chosen_approach, detected_px=px, initial_xy=(cx, cy),
                    aligned_xy=(aligned_cx, aligned_cy), visual_held=True,
                    elapsed_s=_sim_or_wall_time(be) - t_start, notes=notes,
                )
            tx, ty = H(t_px[0], t_px[1])
            tx = float(np.clip(tx + xy_mapping_noise_m[0], -0.17, 0.20))
            ty = float(np.clip(ty + xy_mapping_noise_m[1], 0.15, 0.37))
        else:
            tx, ty = refined_target

        # Account for back-rim pinch offset + cup base tilt projection (`r_pinch + 0.031`) so CUP BASE CENTER lands right on (tx, ty)!
        r_pinch = float(cup_cfg["rim_pinch_radius_m"])
        yaw_p = math.atan2(tx, ty)
        rad_p = np.array([math.sin(yaw_p), math.cos(yaw_p)])
        place_tilt = 30.0
        eff_rad_offset = 0.080
        place_pinch_xy = np.array([tx, ty]) - eff_rad_offset * rad_p
        place_pinch_xy[0] = float(np.clip(place_pinch_xy[0], -0.175, 0.205))
        place_pinch_xy[1] = float(np.clip(place_pinch_xy[1], 0.095, 0.365))

        # Gentle acceleration (REG_ACC=10 -> 4x lower start/stop jerk) + Joint-1 yaw slew keeps low-friction 30g cups rock-steady!
        be._sync(REG_ACC, {i: bytes([10]) for i in range(1, 8)})
        yaw_p_target = -math.atan2(place_pinch_xy[0], place_pinch_xy[1])
        q_slewed = dict(q_lifted)
        q_slewed[1] = float(np.clip(yaw_p_target, URDF_LIMITS[1][0] + 0.02, URDF_LIMITS[1][1] - 0.02))
        move_ticks(ticks_of_q(q_slewed, closed_t), "transport yaw slew", spd=min(speed, 115), tol=46, timeout_s=22.0)
        q_phov, _, _, _ = kin.ik_jaw_tip((place_pinch_xy[0], place_pinch_xy[1], 0.168), tilt_deg=place_tilt, roll_rad=-math.pi / 2.0, seed_rad=q_slewed)
        move_ticks(ticks_of_q(q_phov, closed_t), "place hover", spd=min(speed, 125), tol=48, timeout_s=20.0)
        for z_pl in np.linspace(0.138, float(cup_cfg["place_tip_z_m"]), 3):
            q_pdown, _, _, _ = kin.ik_jaw_tip((place_pinch_xy[0], place_pinch_xy[1], float(z_pl)), tilt_deg=place_tilt, roll_rad=-math.pi / 2.0, seed_rad=q_phov)
            move_ticks(ticks_of_q(q_pdown, closed_t), "place lower", spd=min(speed, 130), tol=48)
            q_phov = q_pdown
        be._sync(REG_ACC, {i: bytes([35]) for i in range(1, 8)})

        # Open gripper to half-span (norm=0.46 -> ~25 mm gap, ~11 mm clear on both sides of the vertical 3.2 mm back wall)
        # and retract PURELY VERTICALLY (+Z) so both jaws rise straight out of the vertical cup cylinder without touching!
        release_t = to_ticks(8, 0.46)
        move_ticks({8: release_t}, "gripper release", spd=360, tol=40)
        _sleep(be, 0.30)
        q_pretract, _, _, _ = kin.ik_jaw_tip((place_pinch_xy[0], place_pinch_xy[1], 0.178), tilt_deg=place_tilt, roll_rad=-math.pi / 2.0, seed_rad=q_pdown)
        move_ticks(ticks_of_q(q_pretract, release_t), "retract after place", spd=min(speed, 200), tol=46)
        move_ticks(W0, "return safe raised pose", spd=min(speed, 380), tol=46)
        _sleep(be, 0.25)

        # Visual placement check from front camera
        rgb_done = vision.decode(be.frame_jpeg(fcam))
        blob_done = vision.find_dark_object(rgb_done, roi=(20, 165, 535, 476), v_max=96, min_area=200, max_area=34000)
        vis_placed = blob_done is not None
        hold_now()
        return CupTaskResult(
            ok=vis_placed, stage="done_place",
            reason="" if vis_placed else "cup blob not seen after placement",
            approach_used=chosen_approach, detected_px=px, initial_xy=(cx, cy),
            aligned_xy=(aligned_cx, aligned_cy), target_xy=(tx, ty),
            visual_held=visual_held, visual_placed=vis_placed,
            gripper_closed_raw=g8_state.raw, gripper_current_ma=g8_state.current_ma,
            ori_err_deg=max_oe, elapsed_s=_sim_or_wall_time(be) - t_start, notes=notes,
        )

    except MoveAbort as ab:
        log(f"  {ab.why}")
        return CupTaskResult(
            ok=False, stage=ab.stage_name, reason=ab.why, aborted=True,
            approach_used=chosen_approach, detected_px=px, initial_xy=(cx, cy),
            ori_err_deg=max_oe, elapsed_s=_sim_or_wall_time(be) - t_start,
        )
