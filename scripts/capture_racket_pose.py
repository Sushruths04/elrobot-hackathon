"""Read-only capture of your real racket grip, level home pose, and live front + wrist camera views.

Zero motor commands are sent. Pose the follower arm holding the racket level (ideally with the ping-pong ball
resting in the center of the blade), then run:
  .venv/bin/python scripts/capture_racket_pose.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from normakit import vision
from normakit.core import JOINT_IDS, norm_to_rad
from normakit.kin import Kin
from normakit.pingpong_vision import detect_orange_ball
from normakit.station_backend import StationBackend, front_camera, wrist_camera


def main() -> None:
    be = StationBackend("localhost:8888")
    be.start()
    time.sleep(1.0)
    kin = Kin()
    try:
        st = be.state()
        fcam = front_camera(be)
        wcam = wrist_camera(be)

        q_rad = {i: round(float(norm_to_rad(i, st.joints[i].norm)), 5) for i in range(1, 9)}
        ee_xyz = [round(float(x), 5) for x in kin.fk({i: q_rad[i] for i in range(1, 8)})]
        tip_xyz, R_gb = kin.jaw_tip_pose({i: q_rad[i] for i in range(1, 8)})

        f_jpg = be.frame_jpeg(fcam)
        f_rgb = vision.decode(f_jpg)
        f_det = detect_orange_ball(f_rgb)
        (ROOT / "run/racket_grip_front.jpg").write_bytes(f_jpg)

        w_det = None
        if wcam:
            w_jpg = be.frame_jpeg(wcam)
            w_rgb = vision.decode(w_jpg)
            w_det = detect_orange_ball(w_rgb)
            (ROOT / "run/racket_grip_wrist.jpg").write_bytes(w_jpg)

        payload = {
            "bus": be._bus,
            "front_camera": fcam,
            "wrist_camera": wcam,
            "joints": {
                i: {
                    "raw": st.joints[i].raw,
                    "lo": st.joints[i].lo,
                    "hi": st.joints[i].hi,
                    "norm": round(float(st.joints[i].norm), 5),
                    "rad": q_rad[i],
                    "current_ma": st.joints[i].current_ma,
                }
                for i in JOINT_IDS
            },
            "gripper_base_xyz_m": ee_xyz,
            "jaw_tip_xyz_m": [round(float(x), 5) for x in tip_xyz],
            "gripper_base_R": [[round(float(v), 5) for v in row] for row in R_gb],
            "front_orange_ball": f_det,
            "wrist_orange_ball": w_det,
        }
        out_json = ROOT / "calibration/racket_grip.json"
        out_json.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"Captured follower pose -> {out_json.relative_to(ROOT)}")
        print(f"Saved frames -> run/racket_grip_front.jpg, run/racket_grip_wrist.jpg")
        print(f"jaw_tip_xyz_m = {payload['jaw_tip_xyz_m']}")
        print(f"front ball = {f_det} | wrist ball = {w_det}")
    finally:
        be.stop_backend()


if __name__ == "__main__":
    main()
