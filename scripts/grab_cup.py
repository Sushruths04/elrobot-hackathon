"""Autonomous GRAB (and optional PLACE) of the black cup. Plan first (no motion), then execute with --go.
Works identically on the real Station arm and the MuJoCo 3.x digital twin (--sim mujoco).

    .venv/bin/python scripts/grab_cup.py                                # real arm: plan only
    .venv/bin/python scripts/grab_cup.py --go --speed 250               # real arm: execute (person at plug)
    .venv/bin/python scripts/grab_cup.py --sim mujoco --vision blob --go --place --video run/demo_mujoco.mp4
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from normakit.cup_skill import run_cup_pick_and_place
from normakit.kin import Kin

ap = argparse.ArgumentParser()
ap.add_argument("--go", action="store_true", help="execute motion (without --go: plan-only dry run)")
ap.add_argument("--sim", choices=["none", "mujoco"], default="none", help="use MuJoCo digital twin instead of real Station")
ap.add_argument("--servo-preset", choices=["calibrated_fix", "station_default"], default="calibrated_fix")
ap.add_argument("--realtime", action="store_true", help="run MuJoCo in 1x wall-clock realtime")
ap.add_argument("--viewer", action="store_true", help="open interactive MuJoCo 3D viewer")
ap.add_argument("--video", default="", help="optional MP4 path to record front+wrist cameras")
ap.add_argument("--hover", type=float, default=0.08)
ap.add_argument("--speed", type=int, default=450)
ap.add_argument("--vision", choices=["er2", "blob", "yolo"], default="er2")
ap.add_argument("--approach", choices=["auto", "vertical", "demo"], default="auto")
ap.add_argument("--no-wrist-align", action="store_true", help="disable wrist-camera closed-loop final alignment")
ap.add_argument("--place", action="store_true", help="also place the cup on the live-detected blue target zone")
ap.add_argument("--table-calib", default="", help="override pixel->robot table calibration JSON")
a = ap.parse_args()

if a.sim == "mujoco":
    from normakit.mujoco_backend import MujocoBackend
    vision_mode = "blob" if a.vision == "er2" and not os.environ.get("GEMINI_API_KEY") and not (Path.home() / ".gemini_er_key").exists() else a.vision
    be = MujocoBackend(realtime=a.realtime or a.viewer, servo_preset=a.servo_preset, viewer=a.viewer)
    be.start()
    calib = Path(a.table_calib) if a.table_calib else (ROOT / "calibration/table_mujoco_front.json")
    if a.video:
        be.start_video(a.video)
else:
    from normakit.station_backend import StationBackend
    vision_mode = a.vision
    be = StationBackend("localhost:8888")
    be.start()
    time.sleep(1.5)
    default_cal = ROOT / "calibration/table_front_180fix.json"
    calib = Path(a.table_calib) if a.table_calib else (default_cal if default_cal.exists() else ROOT / "calibration/table_front.json")

try:
    res = run_cup_pick_and_place(
        be,
        kin=Kin(),
        table_calib_path=calib,
        vision_mode=vision_mode,
        approach=a.approach,
        wrist_align=not a.no_wrist_align,
        do_place=a.place,
        go=a.go,
        speed=a.speed,
        hover_extra_m=a.hover,
        verbose=True,
    )
    if a.sim == "mujoco" and a.video:
        saved = be.stop_video(save=True)
        if saved:
            print(f"Saved video: {saved}", flush=True)
    if not res.ok:
        print(f"FAILED at {res.stage}: {res.reason}", flush=True)
        code = 2 if "HIGH CURRENT" in res.reason else (3 if "TIMEOUT" in res.reason else 1)
        be.stop_backend()
        os._exit(code)
    print(f"DONE ({res.stage}) in {res.elapsed_s:.1f} s | held={res.visual_held} placed={res.visual_placed}", flush=True)
finally:
    be.stop_backend()
os._exit(0)
