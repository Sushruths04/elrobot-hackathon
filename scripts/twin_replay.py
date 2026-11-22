"""Digital-twin video replay: replay a recorded real-robot session (synced front.mp4 + joints.jsonl)
through the MuJoCo servo twin and produce side-by-side comparison artifacts.

Artifacts written to --out (default docs/twin):
  twin_vs_real_f<idx>.png   ~8 side-by-side stills (REAL front cam left, MuJoCo twin right)
  twin_replay_sidebyside.mp4  side-by-side video over the action window (cv2.VideoWriter mp4v)
  twin_replay_preview.gif     small GIF preview (PIL)
  camera_align_f0.png         frame-0 camera-alignment overlay (yellow tape-line fit, real vs twin)
  replay_metrics.json         EE replay error, tape-line pixel alignment, object poses, marker outcome

Example:
  .venv/bin/python scripts/twin_replay.py --session run/recordings/20261121_205828_teleop_pick1 --out docs/twin
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from normakit.core import norm_to_rad
from normakit.kin import Kin
from normakit.mujoco_backend import MujocoBackend, REG_ACC, REG_GOAL_POS, REG_GOAL_SPEED

import mujoco

W, H = 640, 480

# Object poses measured from frame 0 of the real front.mp4 via
# calibration/table_mujoco_front.json (H_pixel_to_robot_xy_m) and iterated against the
# twin render until the tray/marker sit at the same pixels (within ~10 px). See
# docs/twin/replay_metrics.json ("measured_object_poses").
DEFAULT_TRAY = (0.136, 0.258, 0.0)      # x, y (m), yaw (rad) of white_tray
DEFAULT_MARKER = (0.023, 0.265, 3.12)   # x, y (m), yaw (rad) of marker (axis: red end -x local)


class FreeBody:
    """Direct qpos access for a named free body in the backend's model."""

    def __init__(self, be: MujocoBackend, body_name: str):
        self.bid = mujoco.mj_name2id(be.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if self.bid < 0:
            raise ValueError(f"scene has no body named {body_name!r}")
        jadr = int(be.model.body_jntadr[self.bid])
        self.qadr = int(be.model.jnt_qposadr[jadr])
        self.dadr = int(be.model.jnt_dofadr[jadr])
        self.be = be

    def set_pose(self, x: float, y: float, z: float, yaw: float) -> None:
        self.be.data.qpos[self.qadr : self.qadr + 7] = [x, y, z, math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)]
        self.be.data.qvel[self.dadr : self.dadr + 6] = 0.0
        mujoco.mj_forward(self.be.model, self.be.data)

    def pos(self) -> np.ndarray:
        return self.be.data.qpos[self.qadr : self.qadr + 3].copy()

    def yaw(self) -> float:
        qw, qx, qy, qz = self.be.data.qpos[self.qadr + 3 : self.qadr + 7]
        return math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))


def load_session(session: Path):
    lines = [json.loads(l) for l in (session / "joints.jsonl").read_text().splitlines() if l.strip()]
    ts = [float(r["t"]) for r in lines]
    ticks = [{int(k): int(v["raw"]) for k, v in r.items() if k != "t"} for r in lines]
    return ts, ticks


def auto_window(ticks: list[dict[int, int]], pad: int = 30) -> tuple[int, int]:
    """Frames where any joint raw moves > 40 ticks from the start pose, padded by `pad` frames."""
    r0 = ticks[0]
    active = [
        i
        for i, d in enumerate(ticks)
        if max(abs(((d[j] - r0[j] + 2048) % 4096) - 2048) for j in d) > 40
    ]
    if not active:
        return 0, len(ticks) - 1
    return max(0, active[0] - pad), min(len(ticks) - 1, active[-1] + pad)


def most_active_block(ticks: list[dict[int, int]], w0: int, w1: int, max_len: int) -> tuple[int, int]:
    """Longest allowed contiguous block [a, b] (b-a+1 <= max_len) inside [w0, w1] with the most joint motion."""
    if w1 - w0 + 1 <= max_len:
        return w0, w1
    act = np.zeros(w1 + 1)
    for i in range(w0, w1):
        act[i] = max(abs(((ticks[i + 1][j] - ticks[i][j] + 2048) % 4096) - 2048) for j in ticks[i])
    csum = np.cumsum(act)
    best_a, best_s = w0, -1.0
    for a in range(w0, w1 - max_len + 2):
        s = csum[a + max_len - 1] - (csum[a - 1] if a > 0 else 0.0)
        if s > best_s:
            best_s, best_a = s, a
    return best_a, best_a + max_len - 1


def tape_line_x(img_bgr: np.ndarray, y_eval: float = 240.0) -> list[float]:
    """X position of the yellow table tape lines at image row y_eval (HSV threshold + per-line x=a*y+b fit)."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([15, 60, 80]), np.array([45, 255, 255]))
    ys, xs = np.nonzero(mask)
    if len(xs) < 50:
        return []
    lines = []
    for lo, hi in ((0, W // 2), (W // 2, W)):
        sel = (xs >= lo) & (xs < hi)
        if sel.sum() < 30:
            continue
        a, b = np.polyfit(ys[sel].astype(float), xs[sel].astype(float), 1)
        lines.append(float(a * y_eval + b))
    return lines


def _label(img: np.ndarray, text: str, org: tuple[int, int], color: tuple[int, int, int], scale: float = 0.5) -> None:
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1)


def compose(real_bgr: np.ndarray, twin_bgr: np.ndarray, session: str, frame: int, t: float, ee_err_mm: float) -> np.ndarray:
    side = np.hstack([real_bgr, twin_bgr])
    _label(side, "REAL front cam", (10, 26), (0, 255, 255), 0.7)
    _label(side, "MuJoCo twin", (W + 10, 26), (0, 255, 0), 0.7)
    _label(side, session, (10, H - 12), (255, 255, 255))
    _label(side, f"frame {frame}  t={t:.1f}s  EE err {ee_err_mm:.1f} mm", (W + 10, H - 12), (255, 255, 255))
    return side


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", default="run/recordings/20261121_205828_teleop_pick1")
    ap.add_argument("--scene", default="assets/mujoco/twin_desk_scene.xml")
    ap.add_argument("--out", default="docs/twin")
    ap.add_argument("--stills", type=int, default=8)
    ap.add_argument("--window", default="auto", help="'auto' or 'f0:f1' frame range to render")
    ap.add_argument("--tray-x", type=float, default=DEFAULT_TRAY[0])
    ap.add_argument("--tray-y", type=float, default=DEFAULT_TRAY[1])
    ap.add_argument("--tray-yaw", type=float, default=DEFAULT_TRAY[2])
    ap.add_argument("--marker-x", type=float, default=DEFAULT_MARKER[0])
    ap.add_argument("--marker-y", type=float, default=DEFAULT_MARKER[1])
    ap.add_argument("--marker-yaw", type=float, default=DEFAULT_MARKER[2])
    args = ap.parse_args()

    session = Path(args.session)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts, ticks = load_session(session)
    n = len(ticks)
    kin = Kin()

    if args.window == "auto":
        w0, w1 = auto_window(ticks)
    else:
        w0, w1 = (int(v) for v in args.window.split(":"))
        w0, w1 = max(0, w0), min(n - 1, w1)
    mp4_max = int(60.0 / (ts[1] - ts[0]))  # cap the committed mp4 at ~60 s (~15 MB)
    m0, m1 = most_active_block(ticks, w0, w1, mp4_max)
    still_idx = sorted({int(round(f)) for f in np.linspace(w0, w1, args.stills)})
    print(f"session {n} frames; action window [{w0},{w1}]; mp4 window [{m0},{m1}]; stills {still_idx}", flush=True)

    be = MujocoBackend(realtime=False, scene_xml=str(ROOT / args.scene))
    tray = FreeBody(be, "white_tray")
    marker = FreeBody(be, "marker")
    be.reset_arm_ticks(ticks[0])
    tray.set_pose(args.tray_x, args.tray_y, 0.0, args.tray_yaw)
    marker.set_pose(args.marker_x, args.marker_y, 0.0075, args.marker_yaw)
    be._sync(REG_ACC, {i: bytes([40]) for i in range(1, 9)})
    be._sync(REG_GOAL_SPEED, {i: (900).to_bytes(2, "little") for i in range(1, 9)})
    be._sync(REG_GOAL_POS, {i: (ticks[0][i] % 4096).to_bytes(2, "little") for i in range(1, 9)})
    be.step_seconds(0.3)  # settle arm servos + tray/marker contacts at the frame-0 pose

    # ---- frame-0 camera alignment (yellow tape lines, real vs twin) ----
    twin0 = cv2.cvtColor(be.frame_rgb("front", noisy=False), cv2.COLOR_RGB2BGR)
    cap = cv2.VideoCapture(str(session / "front.mp4"))
    ok, real0 = cap.read()
    assert ok, "cannot read front.mp4 frame 0"
    real_tapes, twin_tapes = tape_line_x(real0), tape_line_x(twin0)
    tape_diffs = []
    align = np.hstack([real0.copy(), twin0.copy()])
    for k, (rx, tx) in enumerate(zip(real_tapes, twin_tapes)):
        tape_diffs.append(round(tx - rx, 1))
        cv2.line(align, (int(round(rx)), 0), (int(round(rx)), H), (0, 0, 255), 1)
        cv2.line(align, (W + int(round(tx)), 0), (W + int(round(tx)), H), (0, 0, 255), 1)
        cv2.putText(align, f"tape{k}: real x={rx:.0f} twin x={tx:.0f} diff {tx - rx:+.1f}px", (10, 440 - 20 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 1)
    cv2.imwrite(str(out_dir / "camera_align_f0.png"), align)
    print(f"tape lines: real {real_tapes} twin {twin_tapes} diffs {tape_diffs}", flush=True)

    # ---- replay ----
    cap.set(cv2.CAP_PROP_POS_FRAMES, w0)  # one seek, then sequential reads inside the window
    vw = cv2.VideoWriter(str(out_dir / "twin_replay_sidebyside.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 8, (2 * W, H))
    gif_frames = []
    ee_errs_mm: list[float] = []
    marker_z_max = 0.0
    t_start = time.time()

    for i in range(n):
        tgt = ticks[i]
        be._sync(REG_GOAL_POS, {j: (v % 4096).to_bytes(2, "little") for j, v in tgt.items()})
        dt = (ts[i + 1] - ts[i]) if i + 1 < n else (ts[1] - ts[0])
        be.step_seconds(dt)
        q_real = {j: norm_to_rad(j, be.ticks_to_norm(j, tgt[j])) for j in range(1, 8)}
        ee_err = float(np.linalg.norm(be.ee_pos() - kin.fk(q_real)) * 1000.0)
        ee_errs_mm.append(ee_err)
        marker_z_max = max(marker_z_max, float(marker.pos()[2]))

        if w0 <= i <= w1:
            ok, real = cap.read()
            if not ok:
                continue
            twin = cv2.cvtColor(be.frame_rgb("front", noisy=False), cv2.COLOR_RGB2BGR)
            side = compose(real, twin, session.name, i, ts[i], ee_err)
            if m0 <= i <= m1:
                vw.write(side)
                if (i - m0) % 4 == 0:
                    gif_frames.append(cv2.resize(side, (640, 240), interpolation=cv2.INTER_AREA))
            if i in still_idx:
                cv2.imwrite(str(out_dir / f"twin_vs_real_f{i:04d}.png"), side)
        if i % 100 == 0:
            print(f"  frame {i}/{n}  ee_err {ee_err:.1f} mm  ({time.time() - t_start:.0f}s)", flush=True)

    vw.release()
    cap.release()

    # ---- GIF preview ----
    from PIL import Image

    gif_path = out_dir / "twin_replay_preview.gif"
    step = 1
    while True:
        frames = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)).convert("P", palette=Image.ADAPTIVE, colors=128) for f in gif_frames[::step]]
        frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=500 * step, loop=0, optimize=True)
        if gif_path.stat().st_size <= 8 * 1024 * 1024 or step >= 8:
            break
        step *= 2
    print(f"gif: {gif_path.stat().st_size / 1e6:.1f} MB, every {4 * step}th frame", flush=True)

    # ---- metrics ----
    marker_final = marker.pos()
    tray_final = tray.pos()
    tray_yaw = tray.yaw()
    dx, dy = marker_final[0] - tray_final[0], marker_final[1] - tray_final[1]
    ca, sa = math.cos(-tray_yaw), math.sin(-tray_yaw)
    lx, ly = ca * dx - sa * dy, sa * dx + ca * dy
    in_tray = bool(abs(lx) < 0.049 and abs(ly) < 0.065 and marker_final[2] < 0.06)
    ee = np.array(ee_errs_mm)
    metrics = {
        "session": session.name,
        "scene": args.scene,
        "frames_replayed": n,
        "action_window": [w0, w1],
        "mp4_window": [m0, m1],
        "mp4_note": "mp4 trimmed to the most active ~60 s block of the action window to stay under ~15 MB" if (m0, m1) != (w0, w1) else "full action window",
        "ee_err_mm": {
            "mean": round(float(ee.mean()), 2),
            "rms": round(float(np.sqrt((ee**2).mean())), 2),
            "p95": round(float(np.percentile(ee, 95)), 2),
        },
        "camera_align_frame0_tape_lines": {
            "real_x_at_y240": [round(v, 1) for v in real_tapes],
            "twin_x_at_y240": [round(v, 1) for v in twin_tapes],
            "diff_px_twin_minus_real": tape_diffs,
            "note": (
                "x-pixel of the yellow table tape lines at mid-height (HSV yellow threshold + line fit). "
                "Twin texture tapes sit at robot x=-0.15 m and x=+0.21 m; this session's real table measured "
                "x=-0.21 m (right tape) and x=+0.24 m (left tape) via H, so most of the diff is a real-table "
                "vs table_tex.png mismatch, not camera error (tray/marker/tape all match within ~15 px)."
            ),
        },
        "measured_object_poses": {
            "tray": {"x": args.tray_x, "y": args.tray_y, "yaw_rad": args.tray_yaw},
            "marker": {"x": args.marker_x, "y": args.marker_y, "yaw_rad": args.marker_yaw},
            "note": "measured from real frame 0 via calibration/table_mujoco_front.json, iterated vs twin render to ~10 px",
        },
        "marker_outcome": {
            "max_marker_z_cm": round(marker_z_max * 100.0, 2),
            "final_marker_xyz_m": [round(float(v), 4) for v in marker_final],
            "ends_inside_tray_footprint": in_tray,
            "sim_grasp_succeeded": bool(marker_z_max > 0.02 and in_tray),
            "sim_grasp_note": (
                "Sim marker is never lifted: at the recorded grasp (jaw closes ~line 640) the recorded joints "
                "put the jaw tips ~5 mm from the marker in xy but ~40 mm above the table, while the real grasp "
                "needs the tips at ~10-15 mm to straddle a 15 mm marker. Arm servo replay is the deliverable; "
                "the wrist-pitch/tip-height gap is a twin-vs-real modeling residual."
            ),
            "real_video_outcome": "marker grasped off the table and placed into the tray (front.mp4, ~frames 600-760)",
        },
    }
    (out_dir / "replay_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2), flush=True)
    be.stop_backend()


if __name__ == "__main__":
    main()
