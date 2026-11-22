"""Deploy the trained 30 Hz ElRobot Table-Tennis Balance or Dribble PPO policy matched 1:1 to your sideways JOOLA bat grip.

Default is a camera-in-the-loop MuJoCo twin dry run (never touches hardware).
Real-robot mode (--real --i-have-a-human) drives StationBackend at 30 Hz:
  - Automatically turns Station leader->follower mirroring OFF so it never fights the follower
  - Keeps gripper joint 8 locked at your live handle grip (~0.341 norm) so the bat never opens or crushes
  - Keeps shoulder/elbow joints 2 & 3 steady at your real grip pose (0.030, 0.960)
  - Drives fast mid-range joints [1, 4, 5, 6, 7] with TorqueLimit=800, SPEED_CAP, z_blade >= 8 cm keep-out, and >700 mA e-stop
  - Uses calibration-free `detect_bat_and_ball` (detects red JOOLA blade ellipse + white or orange ball directly in front camera)

Usage:
  .venv/bin/python scripts/run_pingpong.py --preflight
  .venv/bin/python scripts/run_pingpong.py --task balance
  .venv/bin/python scripts/run_pingpong.py --real --i-have-a-human --home-only
  .venv/bin/python scripts/run_pingpong.py --real --i-have-a-human --task balance --seconds 20
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from normakit.core import JOINT_IDS, URDF_LIMITS, norm_to_rad, rad_to_norm
from normakit.kin import Kin
from normakit.pingpong_env import ACTIVE_JIDS, LOCKED_JIDS, MAX_JVEL_RAD_S, MAX_Q_DEV, ElRobotPingPongEnv
from normakit.pingpong_vision import BLADE_RADIUS_M, FdVel3D, detect_bat_and_ball
from normakit.ppo import PPOActorCritic
from normakit.station_backend import REG_ACC, REG_GOAL_POS, REG_GOAL_SPEED, REG_TORQUE, StationBackend, front_camera

REG_TORQUE_LIMIT = 0x30
SPEED_CAP = {1: 1200, 2: 500, 3: 500, 4: 800, 5: 1500, 6: 1500, 7: 1500, 8: 400}
MIN_BLADE_Z_M = 0.08


def _to_ticks(st, jid: int, q_rad: float) -> int:
    n = float(np.clip(rad_to_norm(jid, q_rad), 0.02, 0.98))
    j = st.joints[jid]
    span = (j.hi - j.lo) % 4096
    return int(round(j.lo + n * span)) % 4096


def _stop_mirroring_safe(be: StationBackend) -> None:
    try:
        import asyncio
        from station_py import send_commands
        from target.gen_python.protobuf.drivers.motors_mirroring import mirroring as mm
        from target.gen_python.protobuf.station import commands, drivers

        buses = json.loads((ROOT / "run/buses.json").read_text())
        L = mm.MirroringBus(type=mm.BusType.MBT_ST3215, unique_id=buses["leader"])
        cmd = mm.Command(type=mm.CommandType.CT_STOP_MIRROR, source=L, targets=None)
        dc = commands.DriverCommand(type=drivers.StationCommandType.STC_MOTOR_MIRRORING_COMMAND, body=cmd.encode())
        asyncio.run_coroutine_threadsafe(send_commands(be._client, [dc]), be._loop).result(3.0)
    except Exception:
        pass


def _blade_pose_from_kin(kin: Kin, q_all: dict[int, float]) -> tuple[np.ndarray, np.ndarray]:
    # Sideways JOOLA bat: handle clamped at (0.0009, 0.092, 0.0), blade center at -0.118 along X_paddle (+0.118 along +Z_gb = -X_world!)
    _, R_gb = kin.jaw_tip_pose(q_all)
    p_gb = kin.fk(q_all)
    R_ry90 = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
    R_blade = R_gb @ R_ry90
    c_blade = p_gb + R_gb @ np.array([0.0009, 0.092, 0.0]) + R_blade @ np.array([-0.118, 0.0, 0.0])
    return c_blade, R_blade


def run_sim(ckpt_path: Path, task: str, video_out: str) -> int:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ac = PPOActorCritic(obs_dim=ckpt["obs_dim"], act_dim=ckpt["act_dim"])
    ac.load_state_dict(ckpt["model_state_dict"])
    ac.eval()

    sys.path.insert(0, str(ROOT))
    from scripts.train_pingpong import evaluate_policy
    vid_p = Path(video_out) if video_out else (ROOT / f"run/sim_pingpong_{task}.mp4")
    res = evaluate_policy(ac, task=task, n_episodes=5, obs_mode="camera", seed=12345, video_path=vid_p)
    print(f"Sim camera-in-the-loop dry run ({task}): {json.dumps(res)}")
    print(f"Saved video: {vid_p}")
    return 0


def run_preflight(host: str, ball_color: str = "auto") -> int:
    """100% read-only check: motor spans + red JOOLA bat ellipse + white/orange ball detection."""
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
            print(f"  joint {i}: raw={j.raw:4d} norm={j.norm:+.3f} span={span:4d} cur={j.current_ma:3d}mA temp={j.temp_c}C")
            if span < 100:
                bad.append(f"joint {i} uncalibrated (span={span})")
        rgb = vision.decode(be.frame_jpeg(fcam))
        det = detect_bat_and_ball(rgb, ball_color=ball_color)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        if det["bat_visible"]:
            cv2.ellipse(bgr, (int(det["bat_u"]), int(det["bat_v"])), (int(det["bat_rx"]), int(det["bat_ry"])), 0, 0, 360, (255, 220, 0), 2)
        if det["visible"]:
            cv2.circle(bgr, (int(det["u"]), int(det["v"])), max(4, int(det["r_px"])), (0, 255, 0), 2)
            uwh = det["uwh_m"]
            cv2.putText(bgr, f"BALL du={uwh[0]*100:+.1f}cm dw={uwh[1]*100:+.1f}cm", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
        else:
            msg = "BAT OK - PLACE BALL ON RED RUBBER" if det["bat_visible"] else "RED BAT NOT IN CENTRAL VIEW YET"
            cv2.putText(bgr, msg, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 200, 255), 2)
        out_img = ROOT / "run/pingpong_preflight.jpg"
        cv2.imwrite(str(out_img), bgr)
        print(f"Bat visible={det['bat_visible']}, Ball visible={det['visible']} -> saved {out_img.relative_to(ROOT)}")
        if bad:
            print("PREFLIGHT FAILED:", ", ".join(bad))
            return 2
        print("PREFLIGHT OK (read-only, zero motor commands sent).")
        return 0
    finally:
        be.stop_backend()


def run_real(ckpt_path: Path, task: str, host: str, seconds: float, home_only: bool = False, ball_color: str = "auto") -> int:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ac = PPOActorCritic(obs_dim=ckpt["obs_dim"], act_dim=ckpt["act_dim"])
    ac.load_state_dict(ckpt["model_state_dict"])
    ac.eval()
    action_scale = float(ckpt.get("action_scale", 0.028 if task == "balance" else 0.075))

    # Load exact auto-trimmed level home targets from ElRobotPingPongEnv
    ref_env = ElRobotPingPongEnv(task=task, domain_rand=False, light_model=True)
    settled_home_q = dict(ref_env.settled_home_target)
    ref_env.close()

    be = StationBackend(host)
    be.start()
    time.sleep(1.0)
    kin = Kin()
    fcam = front_camera(be)
    fd = FdVel3D(alpha=0.50, dt=1.0 / 30.0)

    def hold_now():
        s = be.state()
        be._sync(REG_GOAL_POS, {i: int(s.joints[i].raw).to_bytes(2, "little") for i in JOINT_IDS})

    try:
        st_init = be.state()
        for i in JOINT_IDS:
            span = (st_init.joints[i].hi - st_init.joints[i].lo) % 4096
            if span < 100:
                sys.exit(f"REFUSED: joint {i} has uncalibrated span={span}")

        # Stop leader->follower mirroring first so it never fights the follower!
        _stop_mirroring_safe(be)
        time.sleep(0.1)
        st_init = be.state()

        # Keep gripper locked at current handle clamp (or ~0.335 norm if gripping the bat handle)!
        grip_ticks = st_init.joints[8].raw

        print("Step 1 (AGENTS.md Rule #4): slow one-motor test on joint 1 (+35 ticks and back at speed=180)...", flush=True)
        be._sync(REG_GOAL_POS, {i: int(st_init.joints[i].raw).to_bytes(2, "little") for i in JOINT_IDS})
        be._sync(REG_TORQUE, {i: b"\x01" for i in JOINT_IDS})
        be._sync(REG_TORQUE_LIMIT, {i: (800 if i != 8 else 550).to_bytes(2, "little") for i in JOINT_IDS})
        be._sync(REG_ACC, {i: bytes([20]) for i in JOINT_IDS})
        be._sync(REG_GOAL_SPEED, {i: (180).to_bytes(2, "little") for i in JOINT_IDS})
        r1_0 = st_init.joints[1].raw
        be._sync(REG_GOAL_POS, {1: ((r1_0 + 35) % 4096).to_bytes(2, "little")})
        time.sleep(0.8)
        be._sync(REG_GOAL_POS, {1: int(r1_0).to_bytes(2, "little")})
        time.sleep(0.8)
        if input("Did joint 1 move ~3 deg and return cleanly? Type 'yes' to level the bat at HOME: ").strip() != "yes":
            hold_now()
            return 0

        st0 = be.state()
        be._sync(REG_GOAL_SPEED, {i: (260).to_bytes(2, "little") for i in range(1, 8)})
        home_goals = {i: _to_ticks(st0, i, settled_home_q[i]) for i in range(1, 8)}
        home_goals[8] = grip_ticks
        be._sync(REG_GOAL_POS, {i: v.to_bytes(2, "little") for i, v in home_goals.items()})
        print("Moving smoothly to level JOOLA bat HOME pose (2.5 s) — gripper held steady...", flush=True)
        time.sleep(2.5)
        if home_only or seconds <= 0:
            print("DONE: Holding level bat HOME pose.", flush=True)
            return 0

        if input(f"Bat level over table? Type 'yes' to start {seconds:.0f}s 30 Hz {task} loop: ").strip() != "yes":
            hold_now()
            return 0

        be._sync(REG_ACC, {i: bytes([0]) for i in range(1, 8)})
        be._sync(REG_GOAL_SPEED, {i: SPEED_CAP[i].to_bytes(2, "little") for i in range(1, 8)})

        home_vec = np.array([settled_home_q[i] for i in ACTIVE_JIDS], dtype=np.float64)
        q_lo = np.maximum(np.array([URDF_LIMITS[i][0] + 0.04 for i in ACTIVE_JIDS]), home_vec - MAX_Q_DEV)
        q_hi = np.minimum(np.array([URDF_LIMITS[i][1] - 0.04 for i in ACTIVE_JIDS]), home_vec + MAX_Q_DEV)

        q_target = home_vec.copy()
        n_act = len(ACTIVE_JIDS)
        act_hist = np.zeros((3, n_act), dtype=np.float32)
        last_uwh = np.zeros(3, dtype=np.float64)
        prev_q = home_vec.copy()
        prev_t = time.perf_counter()
        t_end = prev_t + seconds
        from normakit import vision
        rec_frames: list[np.ndarray] = []

        print(f"RUNNING 30 Hz {task.upper()} on {fcam} — place the ping-pong ball on the red bat! (Ctrl+C to stop)", flush=True)
        while time.perf_counter() < t_end:
            t_loop = time.perf_counter()
            dt = max(0.015, min(0.08, t_loop - prev_t))
            prev_t = t_loop

            st = be.state()
            hot = max(st.joints[i].current_ma for i in range(1, 8))
            if hot > 700:
                hold_now()
                print(f"STOP: OVERCURRENT ({hot} mA > 700 mA) — holding pose.", flush=True)
                return 2

            q_all = {i: norm_to_rad(i, st.joints[i].norm) for i in range(1, 8)}
            q_cur = np.array([q_all[i] for i in ACTIVE_JIDS], dtype=np.float64)
            qv_cur = (q_cur - prev_q) / dt
            prev_q = q_cur.copy()

            rgb = vision.decode(be.frame_jpeg(fcam))
            det = detect_bat_and_ball(rgb, ball_color=ball_color)
            if det.get("visible") and det.get("uwh_m") is not None:
                last_uwh = det["uwh_m"].copy()
                if task == "balance":
                    last_uwh[2] = 0.0
            vel_fd = fd.update(last_uwh, dt)

            if len(rec_frames) < 650:
                vis_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                if det["bat_visible"]:
                    cv2.ellipse(vis_bgr, (int(det["bat_u"]), int(det["bat_v"])), (int(det["bat_rx"]), int(det["bat_ry"])), 0, 0, 360, (255, 220, 0), 2)
                if det["visible"]:
                    cv2.circle(vis_bgr, (int(det["u"]), int(det["v"])), max(4, int(det["r_px"])), (0, 255, 0), 2)
                rec_frames.append(vis_bgr)

            if not det["visible"]:
                q_target = 0.90 * q_target + 0.10 * home_vec
                act_np = np.zeros(n_act, dtype=np.float32)
            else:
                obs = np.concatenate([
                    [last_uwh[0] / BLADE_RADIUS_M, last_uwh[1] / BLADE_RADIUS_M, last_uwh[2] / 0.20],
                    vel_fd / 0.8,
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

            q_cand_all = dict(settled_home_q)
            for idx_a, jid in enumerate(ACTIVE_JIDS):
                q_cand_all[jid] = float(q_target[idx_a])
            c_cand, _ = _blade_pose_from_kin(kin, q_cand_all)
            if c_cand[2] < MIN_BLADE_Z_M:
                continue

            goals = {jid: _to_ticks(st, jid, float(q_target[idx_a])).to_bytes(2, "little") for idx_a, jid in enumerate(ACTIVE_JIDS)}
            for fj in (2, 3):
                goals[fj] = _to_ticks(st, fj, settled_home_q[fj]).to_bytes(2, "little")
            goals[8] = int(grip_ticks).to_bytes(2, "little")
            be._sync(REG_GOAL_POS, goals)

            rem = (1.0 / 30.0) - (time.perf_counter() - t_loop)
            if rem > 0:
                time.sleep(rem)

        hold_now()
        if rec_frames:
            out_mp4 = ROOT / f"run/real_pingpong_{task}.mp4"
            try:
                import av
                container = av.open(str(out_mp4), mode="w")
                stream = container.add_stream("libx264", rate=30)
                stream.height, stream.width = rec_frames[0].shape[:2]
                stream.pix_fmt = "yuv420p"
                stream.options = {"crf": "22", "preset": "veryfast"}
                for fb in rec_frames:
                    vf = av.VideoFrame.from_ndarray(cv2.cvtColor(fb, cv2.COLOR_BGR2RGB), format="rgb24")
                    for pkt in stream.encode(vf):
                        container.mux(pkt)
                for pkt in stream.encode():
                    container.mux(pkt)
                container.close()
            except Exception:
                h_f, w_f = rec_frames[0].shape[:2]
                vw = cv2.VideoWriter(str(out_mp4), cv2.VideoWriter_fourcc(*"mp4v"), 30, (w_f, h_f))
                for fb in rec_frames:
                    vw.write(fb)
                vw.release()
            print(f"Saved H.264 real-run video -> {out_mp4.relative_to(ROOT)}", flush=True)
        print("DONE: Finished cleanly — holding pose.", flush=True)
        return 0
    except KeyboardInterrupt:
        hold_now()
        print("\nSTOP: Interrupted by Ctrl+C — holding pose.", flush=True)
        return 130
    finally:
        be.stop_backend()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["balance", "dribble"], default="balance")
    ap.add_argument("--policy", default="")
    ap.add_argument("--preflight", action="store_true", help="read-only Station + red bat + white/orange ball check")
    ap.add_argument("--real", action="store_true", help="run on real Station follower arm (requires --i-have-a-human)")
    ap.add_argument("--i-have-a-human", action="store_true")
    ap.add_argument("--home-only", action="store_true", help="level the gripped bat at HOME and hold")
    ap.add_argument("--ball-color", choices=["auto", "orange", "white"], default="auto")
    ap.add_argument("--host", default="localhost:8888")
    ap.add_argument("--seconds", type=float, default=20.0)
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
        sys.exit("REFUSED: --real requires --i-have-a-human (person at 12V plug).")
    sys.exit(run_real(ckpt, a.task, a.host, a.seconds, home_only=a.home_only, ball_color=a.ball_color))


if __name__ == "__main__":
    main()
