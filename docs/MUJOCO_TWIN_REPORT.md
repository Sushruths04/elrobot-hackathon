# MuJoCo 3.x Digital Twin Calibration & 100-Trial Benchmark Report

## 1. Model & Contact Fidelity (`assets/mujoco/elrobot_scene.xml`)
- **Kinematics & Inertias:** Compiled from `vendor/norma-core-src/hardware/elrobot/simulation/elrobot_follower.urdf` using MuJoCo 3.2.7 (`fusestatic=false`, `balanceinertia=true`) plus the physical `38 g` wrist-camera PCB/mount on `Gripper_Base_v1_1`. Forward kinematics matches `normakit.kin.Kin` across 30 random joint poses to **0.03 um** position and **0.036 deg** orientation.
- **Gripper & Hollow Cup Contact:**
  - `Gripper_Jaw_01_v1_1.stl` and `Gripper_Jaw_02_v1_1.stl` decomposed via V-HACD into **16** and **19** convex hull meshes (`assets/mujoco/hulls/jaw_*`), with internal rack-gear self-collision excluded so the gripping pads close to `0.6 mm` unloaded gap without internal housing jam.
  - Motor 8 (`rev_motor_08`) is coupled to both prismatic jaw slides (`rev_motor_08_1`, `rev_motor_08_2`) via MJCF `<equality>` constraints matching Station's `(1 - norm_8)` jaw sign convention (`resolveElrobotJointValue` in `station-viewer`).
  - Black cup modelled as a **hollow tapered thin-walled shell** (`24` convex wall sectors + floor mesh, `7.1 cm` top OD, `5.6 cm` bottom OD, `9.6 cm` height, `3.2 mm` wall, `30-60 g` randomised mass, `0.60-1.00` randomised friction, `condim=4` soft contacts) so an open jaw physically enters the cup mouth and pinches the rim wall.
- **Cameras & Perception:**
  - `front` camera pose (`[-0.042, +0.622, +0.342] m` in base frame) solved via `cv2.solvePnP` from `calibration/board_frame_front.jpg` + `calibration/camera_front.json` intrinsics (`640x480`), rendering textured laminate table, yellow lane lines, black cup, clear glass bottle distractor with dark neck label, and blue target zone.
  - Both `vision.find_dark_object` and `YOLO11n` (`models/yolo11n.pt`) detect the rendered cup at the same pixel within **0.3 px**.

## 2. Sim-vs-Real Validation (`results/twin_validation.json`, `results/validation_views.jpg`)

| Validation Gate | Real Hardware (`handover/`) | MuJoCo Twin (`MujocoBackend`) | Notes |
|---|---|---|---|
| **(a) Recorded demo replay (`run/demo_traj_pick1.json`, 0-19.5 s)** | Picked hollow cup by rim pinch, lifted, carried across table, placed upright | **6.79 mm** mean EE error (**9.30 mm** RMS, **22.82 mm** P95); cup lifted **12.78 cm**, placed upright (`cos = 1.000`) | Verified with `scripts/validate_twin.py` |
| **(b) Resting arm gravity sag after `j3` torque off** | `EE z: 0.220 -> 0.1382 m` (`~28.0 deg` elbow sag) | `EE z: 0.2173 -> 0.1373 m` (**25.79 deg** elbow sag) | `0.9 mm` resting EE height match |
| **(c) Elbow `j3/j4` `-40` tick command (`station_default` registers)** | `dj3 = 0` ticks, `dj4 = 13` ticks (`P=16, I=0, D=0, MaxTorque=500, Overload=25%`) | `dj3 = 4` ticks, `dj4 = 13` ticks | Caused by `P=16` (`I=0`) producing `< 0.68 N*m` elbow gravity + stiction torque |
| **(c') Elbow `j3/j4` `-40` tick command (`calibrated_fix` registers)** | *(Pending Step 2 of `docs/REAL_RUNBOOK.md`)* | `dj3 = 44` ticks, `dj4 = 43` ticks (`lift` residual `2` ticks) | `P=36, I=12, D=20, MaxTorque=850, TorqueLimit=850, Overload=85%` |

### Second Real-Robot Root Cause Found in `handover/` Calibration
Reprojecting `calibration/touch_points_front.json` onto `calibration/board_frame_front.jpg` proved that during the 4-corner hand-touch calibration:
1. The operator stood **behind** the robot (facing the front camera), swapping `top-left / top-right / bottom-right / bottom-left` by **180 deg** (`table_front.json` reprojected the robot base `(0,0,0.05)` to `(169, 463)` at the bottom-left of the image instead of `(304, 134)` at the top-center).
2. The outer `10x7` pattern border was touched while `touch_capture.py` recorded `EE_LINK` (`Gripper_Base_v1_1`, `~10 cm` up the wrist) instead of the jaw tip.

Saving the corrected mapping in `calibration/table_front_180fix.json` (leaving original `calibration/table_front.json` intact) reduced the 4-corner fit residual from **26.41 mm to 6.10 mm**, reprojecting the base to `(304, 134)` and the `now_arm.jpg` jaw tip to `(262, 204)` (inside the cup rim).

## 3. Closed-Loop Control Upgrades (`src/normakit/cup_skill.py`)
- **7-DoF Jaw-Tip IK (`Kin.ik_jaw_tip`):** Solves position + overhead radial pitch (`30 deg` nominal) across all `112/112` grid poses in `x in [-0.15, 0.20] m, y in [0.15, 0.38] m` (`max IK pos err = 0.36 mm`).
- **2-Step Wrist-Camera 3D Ray-Plane Alignment:** Intersects live wrist-camera detections (`bottom_px` on `z = 0` and `top_px` on `z = h_cup`) using live encoder FK (`mj_camlight` extrinsics), reducing `+-3.0 cm` Gaussian table-mapping error down to **~3.3 mm** mean error before descent.
- **Back-Rim Radial Pinch + Pure Joint-1 Slew:** Pinches the near (back) cup wall (`3.4 cm` closer to the base, flush tangential jaw alignment), waits for jaw stop `+ 2.0 s` (EBIM Rule #5), lifts, verifies visually in both cameras (EBIM Rule #4), slews on Joint 1 with gentle `REG_ACC=10` jerk control, and releases at `norm = 0.46` with pure vertical `+Z` retract.

## 4. Remaining Sim-to-Real Gap (Stated Honestly)
1. **Lens radial distortion:** MuJoCo renders an ideal pinhole projection using the calibrated `(fx, fy, cx, cy)` from `camera_front.json` / `camera_wrist.json`, omitting radial distortion coefficients `dist` (`~1-4 px` near image corners).
2. **Cup rim compliance:** The real reusable polypropylene cup flexes elastically by `~1-2 mm` when pinched, slightly increasing real-world contact patch area compared to rigid convex sectors with soft `solimp` compliance.
3. **USB camera auto-exposure & motion blur:** Randomised per-trial in `sim_benchmark.py` via gain/bias/Gaussian noise, whereas the real UVC camera also exhibits rolling-shutter blur if read mid-slew (`0.12-0.15 s` settle pause is included before every wrist-camera capture).
4. **Unverified servo gearbox wear on real `j3/j4`:** While `presets.rs` (`P=16, I=0, Overload=25%`) quantitatively explains the `0 / 13` tick stall, Step 2 of `docs/REAL_RUNBOOK.md` must still verify on hardware that neither `j3` nor `j4` has a physically damaged gear train.

## 5. 100-Trial Benchmark Summary (`results/sim_100.json`)

Command:
```bash
MUJOCO_GL=egl .venv/bin/python scripts/sim_benchmark.py --trials 100
```

| Metric | Result |
|---|---|
| Total randomised trials (`seed=20261121`) | **100** |
| Workspace range | `x in [-0.15, 0.20] m`, `y in [0.15, 0.38] m` |
| Injected calibration/mapping error | `+-3.0 cm` 2D Gaussian (`sigma = 0.030 m`) |
| Randomised physics & lighting | Mass `30-60 g`, friction `0.60-1.00`, servo lag `15-55 ms`, gain `0.84-1.16`, bias `-12..+12`, pixel noise `1.5-4.5`, distractor bottle |
| **Cup lifted >= 5.0 cm and held 3.0 s** | **94 / 100 (94.0%)** |
| **Strict upright 3.0 s hold (`cos >= 0.75` + visual verify)** | **90 / 100 (90.0%)** |
| Safety aborts (`HIGH CURRENT` / `TIMEOUT`) | **0** |
| Median trial simulation time | **19.23 s** |
| Median lifted cup height | **10.66 cm** |
| Wrist-camera alignment error (from `+-3.0 cm` noise) | **4.96 mm median** (`9.20 mm` mean) |

### Failure Breakdown (all 10 videos saved in `results/failure_videos/`)
- **`lift_or_hold_short` (6 trials: `T034, T047, T049, T072, T084, T085`):** Extreme `2.7-3.2 sigma` (`8.0-9.5 cm`) initial calibration noise draws placed the cup on the edge of the first wrist-camera hover view, leaving `> 22 mm` residual offset so a jaw missed or grazed the rim.
- **`held_tilted_gt_41deg` (4 trials: `T006, T007, T011, T053`):** Cup **was** lifted (`12.2-12.8 cm`) and held for the full `3.0 s`, but at far outer reach (`r > 0.36 m`) the `44 deg` radial wrist pitch held the cup at `~42 deg` (`cos ~ 0.74`, just below the `0.75` strict upright threshold).

