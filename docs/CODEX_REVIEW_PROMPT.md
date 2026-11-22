# Review request: autonomous spoon pick-and-place on the ElRobot arm (2026-11-22)

You are reviewing Claude Code's work in this repo (`normacore-hackathon`). Be critical. Find what is wrong, then propose ONE simple, robust approach. **Do not move the arm. Read code, logs, photos and data only.**

## Goal (operator's words)
The arm must **autonomously**:
1. pick up a wooden spoon lying in a bowl of coffee beans,
2. scoop beans,
3. lift high, move over a black cup and release the spoon so it drops into the cup (spoon ends up inside the cup).

Rules:
- No hard-coded positions.
- Use the cameras, a good VLM (Nebius) as the "brain" in a look-act-verify loop, and the operator's teleop demos to learn the how (grip angle, heights, release offset).
- The operator is also training SmolVLA online separately on the teleop data; that's not part of this review.

## Hardware and software
- **Arm:** 8 ST3215 servos (j1-j7 arm, j8 gripper). Leader arm for teleop.
- **Station** (Rust, `../bin/station`, :8888) owns the bus.
- **`scripts/armd.py`** (:8891) is the arm daemon:
  - `/state` returns joints, FK jaw tip and armed_s.
  - `/move_tip {x,y,z,roll,tilt}` runs IK on the MuJoCo model with a path check.
  - `/move_joints`, `/gripper`, `/ik` (dry IK).
  - Moves need the human's ARM button (`scripts/ui.py` :8890).
- **Cameras:** front/table camera (640x480) and wrist camera. Frames come from `scripts/teleop_panel.py` (:8893) at `/frame.jpg?c=table|wrist`.
- **Vision:** `src/normakit/er2.py` `point()` / `src/normakit/llm.py` `ask()`. The provider is set in `run/vlm.json` (Nebius `nvidia/Nemotron-3.5-Super-VL`, a reasoning model, needs `max_tokens` ≥ 3000).
- **Kinematics:** `src/normakit/kin.py` (`jaw_tip_pose`, `ik_jaw_tip` with tilt/roll).

## Data
- **Teleop demos:** `datasets/spoon_beans_v1` (24 episodes), with the 19 good ones in `TRAIN_MANIFEST.json`. A filtered copy is at `datasets/spoon_beans_v1_train` (19 episodes, 30 fps). Format:
  - state = follower joints 0..1
  - action = leader joints 0..1
  - images.table and images.wrist
- **Demo analysis** (by Claude):
  - `results/spoon_demo_phases.json` (grasp / scoop / lift / release tips per demo)
  - `results/spoon_demo_grasps.json` (where on the handle the operator grasped, about 85% up)
  - `results/spoon_demo_release_vs_cup.json` (release tip ≈ cup + (+7 cm x, -2 cm y), z ≈ 0.165)
  - `results/spoon_demo_handle_pixels.json`
- **Operator grip from demos:** tilt about 18° from vertical, wrist roll j7 ≈ 0.76 (IK: tilt 18, roll 1.3 rad gives j7 0.77). Grasp tip z about 0.12-0.13. Scoop dip about 3 cm. Lift to 0.20-0.23. Release at 0.15-0.17, then the gripper opens and the spoon drops into the cup.

## Calibration (suspect area)
- The old `calibration/table_tip_front.json` (pixel to xy homographies per height) was **4-9 cm off today**. The table camera was unplugged and replugged and probably moved.
- **New calibration, made by Claude with the real arm:**
  - `scripts/calib_front_from_arm.py`: arm at a grid at tilt 18; jaw pixel = lowest changed pixel against an arm-out-of-view reference, gated near a predicted pixel; paired with the FK jaw tip.
  - Camera pose: `calibration/front_camera_pose_arm_wide.json` (solvePnP RANSAC, 25/53 inliers, about 6 px reprojection, about 1.1 cm mean / 2 cm max pixel-to-xy at the true z; covers x -0.24..0.15, y 0.17..0.33).
  - Intrinsics: `calibration/camera_front.json`.
  - Raw frames: `results/calib_frames/`.
  - **Please check:** are the inlier count, the jaw-pixel definition (the lowest point of the arm ≠ the FK jaw_tip_site) and the intrinsics valid? Is pixel → ray ∩ plane z correct in `spoon_vlm_pick.py` `at_z()`?

## What was tried (all filmed: `results/films/spoon_*`, photos: `results/photos/`)
1. **Blind replay of a demo** (`scripts/tempo_replay.py` via `spoon_replay.py`): reproduced the demo path within 3 mm, but the spoon lay 1-2 cm differently, so it missed.
2. **Wrist-camera VLM servo** (`spoon_auto.py`): VLM points in the wrist image jumped around, and the arm was pushed 12 cm off.
3. **Demo + front-camera shift** (`spoon_auto2.py`), plus a VLM direction loop ("move left/right/closer/down"): the gripper only pinched to 0.21 because the replay used the follower state, not a firm close. A false grip-success check (it read the jaws while they were still closing).
4. **Sideways grasp with orange-colour segmentation** (`spoon_side_grasp.py`): the wrist-image "handle in gap" test fired early, and the arm never advanced.
5. **Front-camera image servo** (`spoon_front_servo.py`): the jaw detector found beans, then the arm's shadow. A pixel match ≠ 3D match (no depth).
6. **Closest-demo replay shifted to the measured handle** (`spoon_demo_match.py`): **one real grasp (97 mA load, spoon lifted out)**, but the carry was shifted by the grasp offset, so the spoon was dropped on the table, not in the cup.
7. **VLM pick-and-place** (`spoon_vlm_pick.py`; Nemotron points for handle top/base and cup, plus a 3D handle model with base on beans at 4.5 cm and length 9.5 cm): the jaws closed behind and above the handle. The VLM "handle top" was about 30 px off, and the fixed-length model is wrong when the spoon rests on the rim.
8. **Same with grasp points read by eye from the grid image** (`--grasp-px 157,223,z`): at z 0.078 the jaws were about 1.5-2 cm above the handle end; at z 0.062 still empty (photo `results/photos/spoon_vlm_before_close_t1_143856.jpg`).

## Questions for you
1. **Root cause of the persistent 1-3 cm grasp misses:** calibration (camera pose / jaw-pixel definition / FK jaw_tip_site vs the real jaw contact point), IK tilt/roll making the jaws close along the handle, or perception?
2. **Is the grip check right?** Empty = 1263-1326 ticks; holding the handle ≈ 1352-1355 ticks at ~100 mA.
3. **Propose the simplest robust closed loop:** VLM look → act → verify each step (grasp verified by the camera, not just ticks), using the demos for orientation and heights. Specify exactly which VLM questions and which geometry to use.
4. Anything unsafe or wrong in the scripts?

Do not push to GitHub; work stays local.
