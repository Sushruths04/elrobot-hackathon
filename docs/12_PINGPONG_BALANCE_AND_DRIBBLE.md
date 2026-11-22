# Table-Tennis Paddle Ball Balancing & Dribbling (`ElRobot` Follower `5B8E114691`)

Written 2026-11-22. Status: MuJoCo 3.2.7 flat-paddle + 40 mm hollow ITTF ball twin, external front-camera 3D ball tracker (`src/normakit/pingpong_vision.py`), 30 Hz PPO trainer (`scripts/train_pingpong.py`), trained checkpoints (`models/pingpong/balance_ppo.pt`, `models/pingpong/dribble_ppo.pt`), and guarded real-robot runner (`scripts/run_pingpong.py`) are built and evaluated in simulation (both noisy state and full `640x480` external front-camera pipeline). Not yet run on the physical arm.

---

## 1. Open-Source Policy Architecture Adapted to ElRobot

Direct `.pt` weight transfer from SO-101 repositories is impossible because SO-101 is a **5-DoF** arm (`19 -> 5` MLP) whereas ElRobot is an **8-servo (7-DoF + gripper)** arm with different link geometry and camera extrinsics. Instead, we ported the exact proven sim-to-real architecture and physics lessons from both Feetech ST3215/STS3215 projects onto your calibrated ElRobot MuJoCo twin:

1. **`zecice/so101-pingpong-balance` (MIT — proven on real ST3215 hardware):**
   - **30 Hz incremental joint-position deltas** (`q_target += action_scale * a`) around a gravity-settled level home pose (`settled normal = [0, 0, 1]` to `< 0.01 deg`).
   - **`FdVel3D` smoothed finite-difference velocity** (`alpha = 0.50`): the PPO policy never sees ground-truth simulator velocity; both training and real hardware compute ball velocity through the identical `FdVel3D` filter on noisy camera positions.
   - **Latency randomisation (`0..3` frames = `0..100 ms`) + 3-step action history** (`18` dims in the `42`-dim observation), matching your measured `22.8 ms` command write latency and `30.5 fps` front camera in `run/latency_stats.json`.
   - **Gentle non-stalling paddle grip (`--grip-paddle`)**: squeezes until motor-8 current/motion settles and backs off `6` ticks so continuous hold never trips ST3215 overload protection.
2. **`mariapatni/robo-ping-pong` (`sim/keepup.py` — STS3215 bounce keep-up):**
   - **Hollow spherical-shell inertia** (`I = 2/3 m r^2 = 7.2e-7 kg*m^2` for `m = 2.7 g`, `r = 20 mm`), quadratic aerodynamic drag (`C_d = 0.45`), restitution jitter (`e ~ 0.80-0.88`), and `2 deg` random rubber takeoff scatter.
   - **Anti-buzz bounce gating (`min_rise >= 3.5 cm`)**: a bounce only scores after the ball rose at least `3.5 cm` into clear air above the blade; resting or vibrating the ball on the rubber scores zero and terminates after `1.5 s`.
   - **Warm-start fine-tuning**: `dribble_ppo.pt` is fine-tuned directly from `balance_ppo.pt` so horizontal `(u, w)` centering on the `156 mm` blade is already learned at iteration 1.

---

## 2. Calibration-Free Front-Camera Tracking (`src/normakit/pingpong_vision.py`)

Matched directly to `results/photos/bat_grip_front_20261122_105923_{1,2,3}.jpg` and `{1:0.505, 2:0.019, 3:0.974, 4:0.464, 5:0.464, 6:0.35, 7:0.80, 8:0.341}` (`results/twin_vs_real_bat.jpg`):
- **Sideways handle grip (`gripper = 0.341`):** the follower jaws clamp the neck of the wooden handle across the jaws, with the crimson-red JOOLA blade extending `11.8 cm` to `-X` (image-right in the front camera). Shoulder/elbow `j2, j3` stay parked steady (`0.030, 0.960`) while fast mid-range joints `[1, 4, 5, 6, 7]` decouple fore/aft pitch (`j4, j6`) and left/right roll (`j5, j7`).
- **Direct Red-Blade + Ball Detector (`detect_bat_and_ball`):** finds the red JOOLA blade convex hull in `u in [195, 565], v in [100, 380]` (`0.3 px` jitter across your 3 real photos, completely ignoring the orange-red box at `u < 195`) and detects **either a white or orange 40 mm ball** inside that hull. Ball offset `(u_m, w_m)` is measured directly relative to the live detected red ellipse center & radii (`0.8-2.6 mm` accuracy, immune to handle slip or camera extrinsics drift).

---

## 3. Simulation Evaluation Results (Matched Sideways JOOLA Grip)

| Policy & Checkpoint | Eval Mode | Episodes | Result | Video |
|---|---|---|---|---|
| **Paddle Balance** (`models/pingpong/balance_ppo.pt`, `522k` steps) | Randomised sim + noisy `FdVel3D` + `0-100 ms` lag + 2 ball kicks | 40 | **38 / 40 (`95.0%`)** `10.0 s` balance (`9.76 s` mean survival, **`1.23 cm`** median final offset) | `results/pingpong_balance.mp4` |
| **Paddle Balance** (`models/pingpong/balance_ppo.pt`) | **Full `640x480` front camera** (`detect_bat_and_ball` red JOOLA hull + ball + `FdVel3D`) | 15 | **14 / 15 (`93.3%`)** `10.0 s` balance (`9.79 s` mean survival, **`0.81 cm`** median final offset) | `results/pingpong_balance.mp4` |
| **Paddle Dribble / Keep-Up** (`models/pingpong/dribble_ppo.pt`, `553k` steps warm-started from Balance) | Randomised sim + noisy `FdVel3D` + `0-100 ms` lag + `e in [0.80, 0.88]` + `2 deg` scatter | 40 | **`6.15` mean bounces** (`max = 22` bounces >= `3.5 cm` apex; zero-action = `1` bounce) | `results/pingpong_dribble.mp4` |

---

## 4. Real-Robot Deployment Runbook (`scripts/run_pingpong.py`)

**Hardware requirements & honest sim-to-real expectations:**
- `src/normakit/core.py` is **untouched**. `scripts/run_pingpong.py` enforces its own 30 Hz `PaddleGuard`: approved `TorqueLimit = 800` (`450` on gripper), `REG_ACC = 0`, `SPEED_CAP` from `scripts/armd.py`, live FK check refusing any target with `z_blade < 0.10 m`, and immediate `hold()` stop if any arm motor exceeds `700 mA`.
- **Balance (`--task balance`)** has high real-world transfer probability because ray-plane localization on the paddle surface has no monocular scale ambiguity and joint motions are small (`+-1.6 deg/step`).
- **Dribble (`--task dribble`)** is significantly harder on real hardware: a single `30 fps` MJPEG USB camera has rolling-shutter blur at `~1.5 m/s` vertical ball speed, so bright diffuse room lighting (fast camera auto-shutter) and an unoccluded orange 40 mm ball are essential. Test **Balance** on the real arm first before **Dribble**.

### Step-by-step on the real follower (`5B8E114691`):
1. Ensure Station is up (`./norma status`), leader->follower mirroring is **OFF**, workspace is clear, and your hand is on the **12 V power barrel plug**.
2. Move to the level home pose and gently clamp a lightweight standard paddle (`~80-95 g`, black face up) flat between the jaws:
   ```bash
   .venv/bin/python scripts/run_pingpong.py --real --i-have-a-human --grip-paddle --seconds 0
   ```
   *(Verify with a spirit level or phone level app on the blade that the paddle sits flat within `~1-2 deg`; if tilted, adjust `HOME_Q[6]` / `HOME_Q[7]` in `src/normakit/pingpong_env.py` by the measured offset.)*
3. Run **15 s Paddle Ball Balancing** first:
   ```bash
   .venv/bin/python scripts/run_pingpong.py --real --i-have-a-human --task balance --seconds 15
   ```
   Type `yes`, then set an **orange 40 mm ping-pong ball** gently onto the center of the black paddle face (and try giving it gentle nudges with a pen).
4. Only after Step 3 balances cleanly, test **15 s Paddle Dribbling**:
   ```bash
   .venv/bin/python scripts/run_pingpong.py --real --i-have-a-human --task dribble --seconds 15
   ```
   Type `yes`, then drop the orange 40 mm ball from `~8-10 cm` directly above the center of the paddle.
5. Support the arm + paddle by hand before powering off torque (`scripts/servo_diag_and_fix.py limp --yes`).
