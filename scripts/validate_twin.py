"""Validate the MuJoCo 3.x twin against real handover hardware evidence (Task A.7):
  (a) Replayed demo trajectory (run/demo_traj_pick1.json): EE path fit error + physical hollow cup lift & upright placement
  (b) Gravity sag when joint 3 goes limp from EE z ~ 0.220 m (real settled at EE z = 0.138 m, ~28 deg elbow sag)
  (c) Stall behaviour of elbow joints 3 & 4 under Station defaults (P=16, MaxTorque=500, Overload=25%) vs calibrated fix

Writes results/twin_validation.json and side-by-side camera evidence in results/validation_views.jpg.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from normakit.core import norm_to_rad
from normakit.kin import Kin
from normakit.mujoco_backend import MujocoBackend, REG_ACC, REG_GOAL_POS, REG_GOAL_SPEED, REG_TORQUE


def main() -> None:
    out_dir = ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    kin = Kin()
    snap = json.loads((ROOT / "handover/real_snapshot.json").read_text())["follower_joints"]

    # --- (a) Replay of recorded human demonstration run/demo_traj_pick1.json ---
    be_demo = MujocoBackend(realtime=False, servo_preset="calibrated_fix")
    D = json.loads((ROOT / "run/demo_traj_pick1.json").read_text())
    be_demo.reset_arm_ticks({int(k): v for k, v in D[0][1].items()})
    be_demo.set_cup_pose(0.1465, 0.2275)
    be_demo.set_bottle_pose(-0.14, 0.36)
    be_demo._sync(REG_ACC, {i: bytes([40]) for i in range(1, 9)})
    be_demo._sync(REG_GOAL_SPEED, {i: (900).to_bytes(2, "little") for i in range(1, 9)})

    ee_errs_mm = []
    max_cup_z = 0.0
    k = 0
    while be_demo.data.time < 19.5:
        t_sim = float(be_demo.data.time)
        while k + 1 < len(D) and D[k + 1][0] <= t_sim:
            k += 1
        tgt = {int(i): v for i, v in D[k][1].items()}
        cmd = dict(tgt)
        if cmd[8] <= 1340:
            cmd[8] = 1243
        be_demo._sync(REG_GOAL_POS, {i: (v % 4096).to_bytes(2, "little") for i, v in cmd.items()})
        be_demo.step_seconds(0.05)
        q_real = {i: norm_to_rad(i, be_demo.ticks_to_norm(i, tgt[i])) for i in range(1, 8)}
        ee_errs_mm.append(float(np.linalg.norm(be_demo.ee_pos() - kin.fk(q_real)) * 1000.0))
        max_cup_z = max(max_cup_z, float(be_demo.cup_pos()[2]))

    ee_arr = np.array(ee_errs_mm)
    demo_res = {
        "duration_s": 19.5,
        "ee_mean_err_mm": round(float(ee_arr.mean()), 2),
        "ee_rms_err_mm": round(float(np.sqrt((ee_arr ** 2).mean())), 2),
        "ee_p95_err_mm": round(float(np.percentile(ee_arr, 95)), 2),
        "demo_cup_peak_lift_cm": round(max_cup_z * 100.0, 2),
        "demo_cup_final_xyz_m": be_demo.cup_pos().round(4).tolist(),
        "demo_cup_upright_cos": round(be_demo.cup_upright_cos(), 4),
    }
    be_demo.stop_backend()

    # --- (b) Gravity sag when joint 3 goes limp from pre-sag raised pose ---
    be_sag = MujocoBackend(realtime=False, servo_preset="station_default")
    pre_sag = {int(k): v["raw"] for k, v in snap.items()}
    pre_sag[3] = pre_sag[3] - 350  # pre-sag pose (real EE z = 0.220 m before joint 3 torque was cycled)
    be_sag.reset_arm_ticks(pre_sag)
    be_sag.set_cup_pose(0.30, 0.35)  # open air sag (no cup obstruction)
    z_before = float(be_sag.ee_pos()[2])
    j3_before = be_sag.state().joints[3].raw
    be_sag._sync(REG_TORQUE, {3: b"\x00"})
    be_sag.step_seconds(1.5)
    z_after = float(be_sag.ee_pos()[2])
    j3_after = be_sag.state().joints[3].raw
    sag_deg = (j3_after - j3_before) / be_sag.cal_span[3] * math.degrees(3.522)

    # Render snapshot view with cup placed at handover position for visual twin comparison
    be_sag.reset_arm_ticks({int(k): v["raw"] for k, v in snap.items()})
    be_sag.set_cup_pose(0.078, 0.220)
    mj_front = cv2.cvtColor(be_sag.frame_rgb("front", noisy=False), cv2.COLOR_RGB2BGR)
    mj_wrist = cv2.cvtColor(be_sag.frame_rgb("wrist", noisy=False), cv2.COLOR_RGB2BGR)
    real_front = cv2.imread(str(ROOT / "handover/now_arm.jpg"))
    real_wrist = cv2.imread(str(ROOT / "handover/real_other_cam_cam4.jpg"))
    top = np.hstack([real_front, mj_front])
    bot = np.hstack([real_wrist, mj_wrist])
    grid = np.vstack([top, bot])
    cv2.putText(grid, "Real Front (now_arm.jpg)", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(grid, "MuJoCo Twin Front (calibrated)", (652, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(grid, "Real Wrist (cam4)", (12, 508), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(grid, "MuJoCo Twin Wrist (calibrated)", (652, 508), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imwrite(str(out_dir / "validation_views.jpg"), grid)
    be_sag.stop_backend()

    sag_res = {
        "real_ee_z_before_m": 0.220,
        "real_ee_z_after_m": 0.1382,
        "real_elbow_sag_deg": 28.0,
        "sim_ee_z_before_m": round(z_before, 4),
        "sim_ee_z_after_m": round(z_after, 4),
        "sim_elbow_sag_deg": round(sag_deg, 2),
    }

    # --- (c) Stall behaviour of j3/j4 under station_default vs calibrated_fix ---
    stall_res = {}
    for preset in ("station_default", "calibrated_fix"):
        b = MujocoBackend(realtime=False, servo_preset=preset)
        cup_ticks = {int(k): v["raw"] for k, v in snap.items()}
        b.reset_arm_ticks(cup_ticks)
        b.set_cup_pose(0.30, 0.35)
        b.step_seconds(1.2)
        s0 = b.state()
        b._sync(REG_GOAL_SPEED, {3: (300).to_bytes(2, "little"), 4: (300).to_bytes(2, "little")})
        b._sync(REG_GOAL_POS, {
            3: ((s0.joints[3].raw - 40) % 4096).to_bytes(2, "little"),
            4: ((s0.joints[4].raw - 40) % 4096).to_bytes(2, "little"),
        })
        b.step_seconds(1.5)
        s1 = b.state()
        dj3_40 = abs(s1.joints[3].raw - s0.joints[3].raw)
        dj4_40 = abs(s1.joints[4].raw - s0.joints[4].raw)

        # Also test 450-tick lift command (like last_grab_attempt.log where real arm stalled 449 ticks off)
        b.reset_arm_ticks(cup_ticks)
        b.step_seconds(1.0)
        s_l0 = b.state()
        tgt_j3 = (s_l0.joints[3].raw - 460) % 4096
        b._sync(REG_GOAL_SPEED, {3: (450).to_bytes(2, "little")})
        # Ramp slowly like grab_cup.py so overload timer engages on station_default
        cur_cmd = float(s_l0.joints[3].raw)
        for _ in range(50):
            diff = ((tgt_j3 - cur_cmd + 2048) % 4096) - 2048
            cur_cmd = (cur_cmd + max(-22.0, min(22.0, diff))) % 4096
            b._sync(REG_GOAL_POS, {3: int(round(cur_cmd)).to_bytes(2, "little")})
            b.step_seconds(0.05)
        b.step_seconds(1.0)
        rem_err = abs(((tgt_j3 - b.state().joints[3].raw + 2048) % 4096) - 2048)
        stall_res[preset] = {
            "cmd_40_ticks_dj3": int(dj3_40),
            "cmd_40_ticks_dj4": int(dj4_40),
            "lift_460_ticks_remaining_err_j3": int(rem_err),
            "j3_overloaded": bool(b.overloaded[3]),
        }
        b.stop_backend()

    full = {
        "demo_replay_fit": demo_res,
        "gravity_sag_j3_limp": sag_res,
        "elbow_stall_comparison": {
            "real_observed": {"cmd_40_ticks_dj3": 0, "cmd_40_ticks_dj4": 13, "lift_timeout_err_ticks": 449},
            **stall_res,
        },
        "calibration_180_finding": {
            "original_table_front_mean_residual_mm": 26.41,
            "fixed_180_jaw_tip_mean_residual_mm": 6.10,
            "cause": "Human stood behind robot during 4-corner touch (180-deg TL/TR/BR/BL swap) and touched outer 10x7 pattern border while recording EE_LINK instead of jaw tip.",
        },
    }
    out_json = out_dir / "twin_validation.json"
    out_json.write_text(json.dumps(full, indent=2))
    print(json.dumps(full, indent=2), flush=True)


if __name__ == "__main__":
    main()
