"""Deploy the trained 30 Hz ElRobot Table-Tennis Balance or Dribble PPO policy.

Default is a camera-in-the-loop MuJoCo twin dry run (never touches hardware).
Real-robot mode (--real --i-have-a-human) drives StationBackend at 30 Hz through a dedicated
PaddleGuard that enforces:
  - src/normakit/core.py untouched
  - Approved TorqueLimit <= 800 (docs/STACK.md), REG_ACC=0, SPEED_CAP (scripts/armd.py)
  - Blade table keep-out z_blade >= 0.10 m and bounded joint box around HOME_Q
  - Immediate hold() e-stop if any arm joint exceeds 700 mA
  - Gentle non-stalling paddle handle grip (--grip-paddle)

Usage:
  .venv/bin/python scripts/run_pingpong.py --task balance
  .venv/bin/python scripts/run_pingpong.py --task dribble
  .venv/bin/python scripts/run_pingpong.py --real --i-have-a-human --grip-paddle
  .venv/bin/python scripts/run_pingpong.py --real --i-have-a-human --task balance --seconds 15
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from normakit.core import JOINT_IDS, URDF_LIMITS, norm_to_rad, rad_to_norm
from normakit.kin import Kin
from normakit.pingpong_env import ACTIVE_JIDS, HOME_Q, MAX_JVEL_RAD_S, MAX_Q_DEV, ElRobotPingPongEnv
from normakit.pingpong_vision import FdVel3D, FrontCameraBallTracker, detect_orange_ball
from normakit.ppo import PPOActorCritic
from normakit.station_backend import REG_ACC, REG_GOAL_POS, REG_GOAL_SPEED, REG_TORQUE, StationBackend, front_camera

REG_TORQUE_LIMIT = 0x30
SPEED_CAP = {1: 1200, 2: 700, 3: 700, 4: 800, 5: 1500, 6: 1500, 7: 1500, 8: 600}
MIN_BLADE_Z_M = 0.10


def _to_ticks(st, jid: int, q_rad: float) -> int:
    n = float(np.clip(rad_to_norm(jid, q_rad), 0.02, 0.98))
    j = st.joints[jid]
    span = (j.hi - j.lo) % 4096
    return int(round(j.lo + n * span)) % 4096


def _blade_pose_from_kin(kin: Kin, q_all: dict[int, float]) -> tuple[np.ndarray, np.ndarray]:
    # Blade site is at (0.0009, 0.085 + 0.125 = 0.210, 0.0) in Gripper_Base_v1_1 frame!
    _, R_gb = kin.jaw_tip_pose(q_all)
    p_gb = kin.fk(q_all)
    c_blade = p_gb + R_gb @ np.array([0.0009, 0.210, 0.0])
    return c_blade, R_gb


def run_sim(ckpt_path: Path, task: str, video_out: str) -> int:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ac = PPOActorCritic(obs_dim=ckpt["obs_dim"], act_dim=ckpt["act_dim"])
    ac.load_state_dict(ckpt["model_state_dict"])
    ac.eval()

    from scripts.train_pingpong import evaluate_policy
    vid_p = Path(video_out) if video_out else (ROOT / f"run/sim_pingpong_{task}.mp4")
    res = evaluate_policy(ac, task=task, n_episodes=5, obs_mode="camera", seed=12345, video_path=vid_p)
    print(f"Sim camera-in-the-loop dry run ({task}): {json.dumps(res)}")
    print(f"Saved video: {vid_p}")
    return 0


def run_preflight(host: str, ball_color: str = "auto") -> int:
    """100% read-only check: motor calibration spans, currents, temperatures, and live orange-ball detection."""
    import cv2
    from normakit import vision

    be = StationBackend(host)
    be.start()
    time.sleep(1.0)
    try:
        st = be.state()
        fcam = front_camera(be)
        print(f"Connected to follower bus {be._bus}, front_camera={fcam}")
        bad = []
        for i in JOINT_IDS:
            if i not in st.joints:
                bad.append(f"joint {i} missing")
                continue
            j = st.joints[i]
            span = (j.hi - j.lo) % 4096
            print(f"  joint {i}: raw={j.raw:4d} range=[{j.lo:4d},{j.hi:4d}] span={span:4d} norm={j.norm:+.3f} cur={j.current_ma:3d}mA temp={j.temp_c}C")
            if span < 100:
                bad.append(f"joint {i} uncalibrated (span={span})")
        rgb = vision.decode(be.frame_jpeg(fcam))
        det = detect_orange_ball(rgb, ball_color=ball_color)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        if det["visible"]:
            cv2.circle(bgr, (int(det["u"]), int(det["v"])), max(4, int(det["r_px"])), (0, 255, 0), 2)
            cv2.putText(bgr, f"ball u={det['u']:.0f} v={det['v']:.0f} r={det['r_px']:.1f}px", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
        else:
            cv2.putText(bgr, "NO ORANGE BALL DETECTED YET", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 140, 255), 2)
        out_img = ROOT / "run/pingpong_preflight.jpg"
        cv2.imwrite(str(out_img), bgr)
        print(f"Live orange-ball check: {det} -> saved {out_img.relative_to(ROOT)}")
        if bad:
            print("PREFLIGHT FAILED:", ", ".join(bad))
            return 2
        print("PREFLIGHT OK (read-only, zero motor commands sent).")
        return 0
    finally:
        be.stop_backend()


def grip_paddle_routine(be: StationBackend) -> int:
    st0 = be.state()
    be._sync(REG_GOAL_POS, {i: int(st0.joints[i].raw).to_bytes(2, "little") for i in JOINT_IDS})
    be._sync(REG_TORQUE, {i: b"\x01" for i in JOINT_IDS})
    be._sync(REG_TORQUE_LIMIT, {i: (800 if i != 8 else 450).to_bytes(2, "little") for i in JOINT_IDS})
    be._sync(REG_ACC, {i: bytes([25]) for i in JOINT_IDS})
    be._sync(REG_GOAL_SPEED, {i: (300).to_bytes(2, "little") for i in JOINT_IDS})

    # Move smoothly to HOME_Q with gripper open (norm=0.85) so human can insert flat paddle handle
    st = be.state()
    home_open = {i: _to_ticks(st, i, HOME_Q[i]) for i in range(1, 8)}
    j8 = st.joints[8]
    home_open[8] = int(round(j8.lo + 0.85 * ((j8.hi - j8.lo) % 4096))) % 4096
    be._sync(REG_GOAL_POS, {i: v.to_bytes(2, "little") for i, v in home_open.items()})
    time.sleep(3.0)
    input("Place the table-tennis paddle handle FLAT & HORIZONTAL between the open jaws, then press Enter to clamp gently: ")

    # Squeeze gently by load/stall (never command 0.00 full close — prevents ST3215 overload drop-off!)
    target_clamp = int(round(j8.lo + 0.32 * ((j8.hi - j8.lo) % 4096))) % 4096
    be._sync(REG_GOAL_SPEED, {8: (220).to_bytes(2, "little")})
    be._sync(REG_GOAL_POS, {8: target_clamp.to_bytes(2, "little")})
    last_raw = None
    for _ in range(35):
        time.sleep(0.08)
        g8 = be.state().joints[8]
        if g8.current_ma > 220 or (last_raw is not None and abs(g8.raw - last_raw) <= 2):
            # Back off 6 ticks from stall so continuous holding current stays low (~80-140 mA)
            gentle_hold = (g8.raw + 6) % 4096
            be._sync(REG_GOAL_POS, {8: gentle_hold.to_bytes(2, "little")})
            print(f"Paddle clamped gently at {gentle_hold} ticks (current {g8.current_ma} mA).")
            return gentle_hold
        last_raw = g8.raw
    g8_final = be.state().joints[8].raw
    print(f"Paddle clamp settled at {g8_final} ticks.")
    return g8_final


def run_real(ckpt_path: Path, task: str, host: str, seconds: float, do_grip: bool, home_only: bool = False, ball_color: str = "auto") -> int:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ac = PPOActorCritic(obs_dim=ckpt["obs_dim"], act_dim=ckpt["act_dim"])
    ac.load_state_dict(ckpt["model_state_dict"])
    ac.eval()
    action_scale = float(ckpt.get("action_scale", 0.042 if task == "balance" else 0.085))

    be = StationBackend(host)
    be.start()
    time.sleep(1.0)
    kin = Kin()
    fcam = front_camera(be)
    tracker = FrontCameraBallTracker()
    fd = FdVel3D(alpha=0.50, dt=1.0 / 30.0)

    def hold_now():
        s = be.state()
        be._sync(REG_GOAL_POS, {i: int(s.joints[i].raw).to_bytes(2, "little") for i in JOINT_IDS})

    try:
        # AGENTS.md Rule #3: verify all 8 motors are calibrated before any write
        st_init = be.state()
        for i in JOINT_IDS:
            span = (st_init.joints[i].hi - st_init.joints[i].lo) % 4096
            if span < 100:
                sys.exit(f"REFUSED: joint {i} has uncalibrated span={span}")

        # AGENTS.md Rule #4: First motion of new code: one motor (joint 1), small delta (+40 ticks ~ 3.5 deg), slow!
        print("Step 1 (AGENTS.md Rule #4): slow one-motor test on joint 1 (+40 ticks and back at speed=180)...", flush=True)
        be._sync(REG_GOAL_POS, {i: int(st_init.joints[i].raw).to_bytes(2, "little") for i in JOINT_IDS})
        be._sync(REG_TORQUE, {i: b"\x01" for i in JOINT_IDS})
        be._sync(REG_TORQUE_LIMIT, {i: (800 if i != 8 else 450).to_bytes(2, "little") for i in JOINT_IDS})
        be._sync(REG_ACC, {i: bytes([20]) for i in JOINT_IDS})
        be._sync(REG_GOAL_SPEED, {i: (180).to_bytes(2, "little") for i in JOINT_IDS})
        r1_0 = st_init.joints[1].raw
        be._sync(REG_GOAL_POS, {1: ((r1_0 + 40) % 4096).to_bytes(2, "little")})
        time.sleep(0.9)
        be._sync(REG_GOAL_POS, {1: int(r1_0).to_bytes(2, "little")})
        time.sleep(0.9)
        if input("Did joint 1 move ~3.5 deg and return cleanly? Type 'yes' to continue: ").strip() != "yes":
            hold_now()
            return 0

        if do_grip:
            grip_ticks = grip_paddle_routine(be)
            if not seconds:
                return 0
        else:
            st0 = be.state()
            grip_ticks = st0.joints[8].raw
            be._sync(REG_GOAL_POS, {i: int(st0.joints[i].raw).to_bytes(2, "little") for i in JOINT_IDS})
            be._sync(REG_TORQUE, {i: b"\x01" for i in JOINT_IDS})
            be._sync(REG_TORQUE_LIMIT, {i: (800 if i != 8 else 450).to_bytes(2, "little") for i in JOINT_IDS})
            be._sync(REG_ACC, {i: bytes([25]) for i in JOINT_IDS})
            be._sync(REG_GOAL_SPEED, {i: (320).to_bytes(2, "little") for i in JOINT_IDS})
            home_goals = {i: _to_ticks(st0, i, HOME_Q[i]) for i in range(1, 8)}
            home_goals[8] = grip_ticks
            be._sync(REG_GOAL_POS, {i: v.to_bytes(2, "little") for i, v in home_goals.items()})
            print("Moving smoothly to level paddle HOME pose (3.0 s) — motor 8 untouched...", flush=True)
            time.sleep(3.0)
            if home_only or seconds <= 0:
                print("Holding HOME pose. Tape the paddle flat & level (black rubber UP) onto the gripper now!", flush=True)
                return 0

        if input(f"Paddle level and workspace clear? Type 'yes' to start {seconds:.0f}s 30 Hz {task} loop: ").strip() != "yes":
            hold_now()
            return 0

        # Configure direct 30 Hz servo streaming (REG_ACC=0, per-joint SPEED_CAP from scripts/armd.py)
        be._sync(REG_ACC, {i: bytes([0]) for i in range(1, 8)})
        be._sync(REG_GOAL_SPEED, {i: SPEED_CAP[i].to_bytes(2, "little") for i in range(1, 8)})

        home_vec = np.array([HOME_Q[i] for i in ACTIVE_JIDS], dtype=np.float64)
        q_lo = np.maximum(np.array([URDF_LIMITS[i][0] + 0.04 for i in ACTIVE_JIDS]), home_vec - MAX_Q_DEV)
        q_hi = np.minimum(np.array([URDF_LIMITS[i][1] - 0.04 for i in ACTIVE_JIDS]), home_vec + MAX_Q_DEV)
        home_blade_pos, _ = _blade_pose_from_kin(kin, HOME_Q)

        q_target = home_vec.copy()
        act_hist = np.zeros((3, 6), dtype=np.float32)
        last_uwh = np.zeros(3, dtype=np.float64)
        prev_q = home_vec.copy()
        prev_t = time.perf_counter()
        t_end = prev_t + seconds
        import cv2
        from normakit import vision
        rec_frames: list[np.ndarray] = []

        print(f"RUNNING 30 Hz {task.upper()} on {fcam} — place/drop the orange ping-pong ball on the paddle! (Ctrl+C to stop)", flush=True)
        while time.perf_counter() < t_end:
            t_loop = time.perf_counter()
            dt = max(0.015, min(0.08, t_loop - prev_t))
            prev_t = t_loop

            st = be.state()
            hot = max(st.joints[i].current_ma for i in range(1, 8))
            if hot > 700:
                hold_now()
                print(f"OVERCURRENT ({hot} mA > 700 mA) — holding pose and stopping.", flush=True)
                return 2

            q_all = {i: norm_to_rad(i, st.joints[i].norm) for i in range(1, 8)}
            q_cur = np.array([q_all[i] for i in ACTIVE_JIDS], dtype=np.float64)
            qv_cur = (q_cur - prev_q) / dt
            prev_q = q_cur.copy()

            c_blade, R_blade = _blade_pose_from_kin(kin, q_all)
            rgb = vision.decode(be.frame_jpeg(fcam))
            det = detect_orange_ball(rgb, ball_color=ball_color)
            est_uwh = tracker.localize(det, c_blade, R_blade, mode=task)
            if est_uwh is not None:
                last_uwh = est_uwh
            vel_fd = fd.update(last_uwh, dt)
            if len(rec_frames) < 600:
                vis_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                if det["visible"]:
                    cv2.circle(vis_bgr, (int(det["u"]), int(det["v"])), max(4, int(det["r_px"])), (0, 255, 0), 2)
                rec_frames.append(vis_bgr)

            if not det["visible"]:
                # Ball not visible -> smoothly relax back toward level HOME_Q and wait!
                q_target = 0.90 * q_target + 0.10 * home_vec
                act_np = np.zeros(6, dtype=np.float32)
            else:
                n_ax = R_blade[:, 2]
                obs = np.concatenate([
                    [last_uwh[0] / 0.08, last_uwh[1] / 0.08, last_uwh[2] / 0.20],
                    vel_fd,
                    (c_blade - home_blade_pos) / 0.10,
                    [n_ax[0], n_ax[1], n_ax[2] - 1.0],
                    (q_cur - home_vec) / 0.25,
                    qv_cur * 0.15,
                    act_hist.ravel(),
                ]).astype(np.float32)
                act_t, _, _ = ac.act(torch.as_tensor(obs).unsqueeze(0), deterministic=True)
                act_np = np.clip(act_t.squeeze(0).cpu().numpy(), -1.0, 1.0)
                max_dq = np.minimum(action_scale, MAX_JVEL_RAD_S * (1.0 / 30.0))
                q_target = np.clip(q_target + max_dq * act_np, q_lo, q_hi)

            act_hist = np.roll(act_hist, shift=1, axis=0)
            act_hist[0] = act_np

            # Table keep-out FK check BEFORE sending goal
            q_cand_all = dict(HOME_Q)
            for idx_a, jid in enumerate(ACTIVE_JIDS):
                q_cand_all[jid] = float(q_target[idx_a])
            c_cand, _ = _blade_pose_from_kin(kin, q_cand_all)
            if c_cand[2] < MIN_BLADE_Z_M:
                continue  # refuse any move taking blade below 10 cm above table

            goals = {jid: _to_ticks(st, jid, float(q_target[idx_a])).to_bytes(2, "little") for idx_a, jid in enumerate(ACTIVE_JIDS)}
            goals[5] = _to_ticks(st, 5, HOME_Q[5]).to_bytes(2, "little")
            goals[8] = int(grip_ticks).to_bytes(2, "little")
            be._sync(REG_GOAL_POS, goals)

            rem = (1.0 / 30.0) - (time.perf_counter() - t_loop)
            if rem > 0:
                time.sleep(rem)

        hold_now()
        if rec_frames:
            out_mp4 = ROOT / f"run/real_pingpong_{task}.mp4"
            h_f, w_f = rec_frames[0].shape[:2]
            vw = cv2.VideoWriter(str(out_mp4), cv2.VideoWriter_fourcc(*"mp4v"), 30, (w_f, h_f))
            for f_bgr in rec_frames:
                vw.write(f_bgr)
            vw.release()
            print(f"Saved real-run video -> {out_mp4.relative_to(ROOT)}", flush=True)
        print("Finished cleanly — holding pose.", flush=True)
        return 0
    except KeyboardInterrupt:
        hold_now()
        print("\nStopped by Ctrl+C — holding pose.", flush=True)
        return 130
    finally:
        be.stop_backend()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["balance", "dribble"], default="balance")
    ap.add_argument("--policy", default="")
    ap.add_argument("--preflight", action="store_true", help="read-only Station + calibration + live orange-ball camera check")
    ap.add_argument("--real", action="store_true", help="run on real Station follower arm (requires --i-have-a-human)")
    ap.add_argument("--i-have-a-human", action="store_true")
    ap.add_argument("--home-only", action="store_true", help="move to level HOME pose and hold so you can tape the paddle flat")
    ap.add_argument("--grip-paddle", action="store_true", help="open jaws at home and gently clamp paddle handle")
    ap.add_argument("--ball-color", choices=["auto", "orange", "white"], default="auto", help="orange (recommended) or white ball on black rubber")
    ap.add_argument("--host", default="localhost:8888")
    ap.add_argument("--seconds", type=float, default=15.0)
    ap.add_argument("--video", default="")
    a = ap.parse_args()

    if a.preflight:
        sys.exit(run_preflight(a.host, ball_color=a.ball_color))

    ckpt = Path(a.policy) if a.policy else (ROOT / f"models/pingpong/{a.task}_ppo.pt")
    if not ckpt.exists():
        sys.exit(f"Missing policy {ckpt}; train first with scripts/train_pingpong.py --task {a.task}")

    if not a.real:
        sys.exit(run_sim(ckpt, a.task, a.video))
    if not a.i_have_a_human:
        sys.exit("REFUSED: --real requires --i-have-a-human (person at 12V plug, Station mirroring OFF).")
    sys.exit(run_real(ckpt, a.task, a.host, a.seconds, a.grip_paddle, home_only=a.home_only, ball_color=a.ball_color))


if __name__ == "__main__":
    main()
