"""100-trial MuJoCo digital-twin benchmark for autonomous black-cup pick-and-place (Task A.6).

Per trial randomises:
  - Cup position across the reachable table workspace: x in [-0.15, 0.20] m, y in [0.15, 0.38] m
  - Table calibration / detection mapping error: 2D Gaussian sigma = 0.030 m (+-3 cm, matching real table_front.json)
  - ST3215 command bus / servo lag: 15..55 ms
  - Cup mass: 30..60 g, cup surface friction: 0.60..1.00
  - Camera exposure gain (0.84..1.16), brightness bias (-12..+12), pixel noise (1.5..4.5)
  - Distractor bottle position & target coaster position

Success criteria (evaluated from true MuJoCo 3D physics state):
  - Pick-and-hold success: cup lifted >= 5.0 cm off table and held continuously for 3.0 s
  - Full pick-and-place success: pick-and-hold success AND placed upright within <= 3.0 cm of target center

Saves:
  - results/sim_100.json (summary + all 100 per-trial records)
  - results/failure_videos/trial_XXX.mp4 (side-by-side front + wrist MP4 for every failed trial)
  - results/success_example.mp4 (one recorded successful pick-and-place trial)
"""
from __future__ import annotations

import argparse
import collections
import json
import multiprocessing as mp
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

_WORKER_CTX: dict = {}


def _init_worker(vision_mode: str, seed_base: int) -> None:
    os.environ.setdefault("MUJOCO_GL", "egl")
    from normakit.kin import Kin
    from normakit.mujoco_backend import MujocoBackend

    yolo_det = None
    if vision_mode == "yolo":
        from normakit.detect import Detector
        yolo_det = Detector(classes=["cup", "bottle"], conf=0.20)
    _WORKER_CTX["kin"] = Kin()
    _WORKER_CTX["be"] = MujocoBackend(realtime=False, servo_preset="calibrated_fix", seed=seed_base)
    _WORKER_CTX["vision"] = vision_mode
    _WORKER_CTX["yolo"] = yolo_det


def _run_one_trial(cfg: dict) -> dict:
    from normakit.cup_skill import run_cup_pick_and_place
    from normakit.mujoco_backend import PhotometricConfig

    be = _WORKER_CTX["be"]
    kin = _WORKER_CTX["kin"]
    tid = cfg["tid"]
    cx, cy = cfg["cx"], cfg["cy"]
    tx, ty = cfg["tx"], cfg["ty"]
    bx, by = cfg["bx"], cfg["by"]
    be._rng = np.random.default_rng(cfg["trial_seed"])
    be.reset_arm_norms({1: 0.50, 2: 0.25, 3: 0.28, 4: 0.72, 5: 0.50, 6: 0.65, 7: 0.50, 8: 0.90})
    be.data.time = 0.0
    be.set_cup_pose(cx, cy, mass_kg=cfg["mass_kg"], friction=cfg["friction"])
    be.set_place_target(tx, ty)
    be.set_bottle_pose(bx, by)
    be.params.cmd_lag_s = cfg["lag_s"]
    be.photo = PhotometricConfig(
        exposure_gain=cfg["exp_gain"], brightness_bias=cfg["exp_bias"], noise_std=cfg["pix_noise"]
    )
    be.step_seconds(0.15)

    fail_vid_path = Path(cfg["fail_vid_dir"]) / f"trial_{tid:03d}.mp4"
    be.start_video(fail_vid_path, fps=2, cameras=("front",), lazy=True)

    hold_samples_z: list[float] = []

    def check_3s_hold() -> None:
        hold_samples_z.append(float(be.cup_pos()[2]))

    res = run_cup_pick_and_place(
        be,
        kin=kin,
        table_calib_path=ROOT / "calibration/table_mujoco_front.json",
        vision_mode=_WORKER_CTX["vision"],
        approach="auto",
        wrist_align=True,
        do_place=bool(cfg["do_place"]),
        go=True,
        speed=450,
        xy_mapping_noise_m=(cfg["map_err_x"], cfg["map_err_y"]),
        hold_duration_s=3.0,
        yolo_det=_WORKER_CTX["yolo"],
        on_Hold_callback=check_3s_hold,
        verbose=False,
    )

    final_cup = be.cup_pos()
    final_upright = be.cup_upright_cos()
    # In pick-and-hold mode, cup is still held after the full 3.0 s window -> verify BOTH mid-hold AND end-of-3s-hold >= 5.0 cm!
    if not cfg["do_place"]:
        hold_samples_z.append(float(final_cup[2]))
    held_z_m = min(hold_samples_z) if hold_samples_z else 0.0
    place_err_m = float(np.hypot(final_cup[0] - tx, final_cup[1] - ty))

    pick_hold_ok = bool(held_z_m >= 0.050 and res.visual_held and (cfg["do_place"] or final_upright >= 0.75))
    if not cfg["do_place"]:
        trial_ok = bool(pick_hold_ok and not res.aborted and res.ok)
    else:
        trial_ok = bool(pick_hold_ok and not res.aborted and res.ok and place_err_m <= 0.030 and final_upright >= 0.85)

    fail_reason = ""
    vid_saved = None
    if not trial_ok:
        if res.aborted:
            fail_reason = f"abort_{res.stage}: {res.reason}"
        elif not pick_hold_ok:
            fail_reason = f"lift_or_hold_short (z={held_z_m*100:.1f}cm, vis={res.visual_held})"
        elif place_err_m > 0.030:
            fail_reason = f"place_offset_{place_err_m*100:.1f}cm (>3.0cm)"
        elif final_upright < 0.85:
            fail_reason = f"cup_tipped (cos={final_upright:.2f})"
        else:
            fail_reason = res.reason or f"failed_{res.stage}"
        saved_p = be.stop_video(save=True)
        if saved_p is not None:
            vid_saved = str(saved_p.relative_to(ROOT))
    elif tid == 0:
        be._video_path = Path(cfg["out_dir"]) / "success_example.mp4"
        saved_p = be.stop_video(save=True)
        if saved_p is not None:
            vid_saved = str(saved_p.relative_to(ROOT))
    else:
        be.stop_video(save=False)

    return {
        "trial": tid,
        "success": trial_ok,
        "pick_hold_5cm_3s_ok": pick_hold_ok,
        "aborted": res.aborted,
        "stage": res.stage,
        "failure_reason": fail_reason,
        "sim_seconds": round(res.elapsed_s, 2),
        "true_cup_xy_m": [round(cx, 4), round(cy, 4)],
        "injected_map_err_cm": [round(cfg["map_err_x"] * 100.0, 2), round(cfg["map_err_y"] * 100.0, 2)],
        "wrist_aligned_xy_m": [round(x, 4) for x in res.aligned_xy] if res.aligned_xy else None,
        "wrist_align_err_mm": round(float(np.hypot(res.aligned_xy[0] - cx, res.aligned_xy[1] - cy) * 1000.0), 2) if res.aligned_xy else None,
        "held_cup_z_cm": round(held_z_m * 100.0, 2),
        "true_target_xy_m": [round(tx, 4), round(ty, 4)],
        "final_cup_xyz_m": [round(float(v), 4) for v in final_cup],
        "place_err_cm": round(place_err_m * 100.0, 2),
        "cup_upright_cos": round(final_upright, 3),
        "cup_mass_g": round(cfg["mass_kg"] * 1000.0, 1),
        "cup_friction": round(cfg["friction"], 3),
        "servo_lag_ms": round(cfg["lag_s"] * 1000.0, 1),
        "exposure_gain": round(cfg["exp_gain"], 3),
        "video": vid_saved,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--seed", type=int, default=20261121)
    ap.add_argument("--vision", choices=["blob", "yolo"], default="blob")
    ap.add_argument("--noise-std-m", type=float, default=0.030, help="Gaussian XY mapping error sigma (m)")
    ap.add_argument("--place", action="store_true", help="also require target placement <= 3.0 cm (default benchmarks pick >= 5 cm + 3.0 s hold)")
    ap.add_argument("--pick-only", action="store_true", help="default behaviour (kept for compatibility)")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--start-idx", type=int, default=0)
    ap.add_argument("--end-idx", type=int, default=-1)
    ap.add_argument("--no-clean-videos", action="store_true")
    ap.add_argument("--out", default="results/sim_100.json")
    a = ap.parse_args()

    out_path = ROOT / a.out
    fail_vid_dir = ROOT / "results/failure_videos"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fail_vid_dir.mkdir(parents=True, exist_ok=True)
    if not a.no_clean_videos and a.start_idx == 0 and a.end_idx < 0:
        for old_v in fail_vid_dir.glob("trial_*.mp4"):
            old_v.unlink()

    # Pre-generate all 100 deterministic randomised trial configs from master seed
    rng = np.random.default_rng(a.seed)
    configs = []
    for tid in range(a.trials):
        cx = float(rng.uniform(-0.15, 0.20))
        cy = float(rng.uniform(0.15, 0.38))
        tx, ty = -0.09, 0.26
        for _ in range(80):
            cand_x = float(rng.uniform(-0.13, 0.16))
            cand_y = float(rng.uniform(0.21, 0.34))
            if np.hypot(cand_x - cx, cand_y - cy) >= 0.14 and np.hypot(cand_x, cand_y) >= 0.22:
                tx, ty = cand_x, cand_y
                break
        # Place bottle distractor in outer table margin so it challenges perception without blocking arm sweep
        bx, by = -0.22, 0.38
        for _ in range(80):
            cand_bx = float(rng.uniform(-0.26, 0.26))
            cand_by = float(rng.uniform(0.28, 0.45))
            if (abs(cand_bx) >= 0.19 or cand_by >= 0.39) and np.hypot(cand_bx - cx, cand_by - cy) >= 0.14 and np.hypot(cand_bx - tx, cand_by - ty) >= 0.14:
                bx, by = cand_bx, cand_by
                break
        configs.append({
            "tid": tid,
            "trial_seed": int(a.seed + tid * 7919),
            "cx": cx, "cy": cy, "tx": tx, "ty": ty, "bx": bx, "by": by,
            "mass_kg": float(rng.uniform(0.030, 0.060)),
            "friction": float(rng.uniform(0.60, 1.00)),
            "lag_s": float(rng.uniform(0.015, 0.055)),
            "exp_gain": float(rng.uniform(0.84, 1.16)),
            "exp_bias": float(rng.uniform(-12.0, 12.0)),
            "pix_noise": float(rng.uniform(1.5, 4.5)),
            "map_err_x": float(rng.normal(0.0, a.noise_std_m)),
            "map_err_y": float(rng.normal(0.0, a.noise_std_m)),
            "do_place": bool(a.place and not a.pick_only),
            "fail_vid_dir": str(fail_vid_dir),
            "out_dir": str(ROOT / "results"),
        })

    if a.end_idx > 0:
        configs = configs[a.start_idx : a.end_idx]
    elif a.start_idx > 0:
        configs = configs[a.start_idx :]

    t_wall0 = time.perf_counter()
    trials_log = []
    if a.workers <= 1:
        _init_worker(a.vision, a.seed)
        it = (_run_one_trial(c) for c in configs)
        pool_cm = None
    else:
        ctx = mp.get_context("spawn")
        pool_cm = ctx.Pool(processes=a.workers, initializer=_init_worker, initargs=(a.vision, a.seed))
        it = pool_cm.imap_unordered(_run_one_trial, configs)
    try:
        for idx, rec in enumerate(it, 1):
            trials_log.append(rec)
            if idx % 5 == 0 or not rec["success"] or a.trials <= 15:
                n_ok = sum(1 for r in trials_log if r["success"])
                n_pk = sum(1 for r in trials_log if r["pick_hold_5cm_3s_ok"])
                print(
                    f"[{idx:3d}/{a.trials}] pick_3s={n_pk}/{idx} ({100*n_pk/idx:.1f}%) | "
                    f"full_place={n_ok}/{idx} ({100*n_ok/idx:.1f}%) | "
                    f"T{rec['trial']:03d}: z={rec['held_cup_z_cm']:4.1f}cm err={rec['place_err_cm']:4.2f}cm "
                    f"{'OK' if rec['success'] else rec['failure_reason']}",
                    flush=True,
                )
    finally:
        if pool_cm is not None:
            pool_cm.close()
            pool_cm.join()
        elif "be" in _WORKER_CTX:
            _WORKER_CTX["be"].stop_backend()

    trials_log.sort(key=lambda r: r["trial"])
    failure_reasons: collections.Counter[str] = collections.Counter()
    for r in trials_log:
        if not r["success"]:
            cat = r["failure_reason"].split(":")[0].split(" (")[0]
            if cat.startswith("place_offset"):
                cat = "place_offset_gt_3cm"
            failure_reasons[cat] += 1

    n_total = len(trials_log)
    n_success = sum(1 for r in trials_log if r["success"])
    n_pick_ok = sum(1 for r in trials_log if r["pick_hold_5cm_3s_ok"])
    n_aborts = sum(1 for r in trials_log if r["aborted"])
    times = [r["sim_seconds"] for r in trials_log if r["success"]]
    align_errs = [r["wrist_align_err_mm"] for r in trials_log if r["wrist_align_err_mm"] is not None]
    place_errs = [r["place_err_cm"] for r in trials_log if r["pick_hold_5cm_3s_ok"] and not r["aborted"]]

    summary = {
        "trials": n_total,
        "seed": a.seed,
        "vision": a.vision,
        "injected_mapping_error_sigma_cm": round(a.noise_std_m * 100.0, 2),
        "pick_hold_5cm_3s_successes": n_pick_ok,
        "pick_hold_5cm_3s_rate_pct": round(100.0 * n_pick_ok / max(1, n_total), 2),
        "full_pick_and_place_successes": n_success,
        "full_pick_and_place_rate_pct": round(100.0 * n_success / max(1, n_total), 2),
        "aborts": n_aborts,
        "median_sim_time_s": round(float(statistics.median(times)), 2) if times else None,
        "mean_wrist_align_err_mm": round(float(np.mean(align_errs)), 2) if align_errs else None,
        "median_place_err_cm": round(float(statistics.median(place_errs)), 2) if place_errs else None,
        "failure_reasons": dict(failure_reasons),
        "wall_compute_seconds": round(time.perf_counter() - t_wall0, 2),
    }

    payload = {"summary": summary, "trials": trials_log}
    out_path.write_text(json.dumps(payload, indent=2))
    print("\n=== 100-TRIAL MUJOCO BENCHMARK SUMMARY ===", flush=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
