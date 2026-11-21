# PROMPT FOR DEVIN (paste everything below the line)

---

You are taking over a robotics hackathon project (HackLab Robotics x BRISA, Berlin 21-22 Nov 2026, NormaCore track). Goal: **a robot arm autonomously picks a black cup from a table using a camera and puts it down**, driven by Claude-class AI for perception/decisions, tested first in simulation (100 trials), then on the real robot. Repo: https://github.com/Sushruths04/normacore-hackathon (read `AGENTS.md`, `docs/HANDOVER.md`, `docs/09_EBIM_PLAYBOOK.md`, `docs/ISSUES.md`, `docs/NEXT_STEPS.md` first). Latest state is in `handover/` (real + sim snapshots, calibration, logs, pictures).

## Hard constraints
- You run in the cloud: you CANNOT touch the real robot (USB, Station, cameras). Do everything that does not need hardware (simulation, planning, tests, code, docs). For anything on the real robot, write a short numbered runbook that the human at the laptop executes and pastes back the output. Never claim a real-robot result you did not see.
- No secrets in the repo or in logs. The Gemini key and any Anthropic key come from environment variables (`GEMINI_API_KEY`, `ANTHROPIC_API_KEY`, or `~/.gemini_er_key`); ask for them via Devin secrets. Never print them.
- Git: small commits, no co-author/attribution trailers or mentions of an AI in commit messages or PR text (the owner's rule for this repo). Do not rewrite history.
- Be terse. Evidence before theory. Report before editing working code; never delete what cannot be regenerated.

## Hardware and software (what exists)
- NormaCore **ElRobot**: two 8-servo arms (ST3215, joints 1-7 + gripper 8). Follower = bus serial `5B8E114691` (does the work), leader = `5B79016336` (hand-operated, for teleop/demos). Config in `run/buses.json`, `run/station.yaml`.
- **Station** (Rust, `vendor/`, started with `./norma start`): NormFS queues on TCP :8888, web UI :8889 (use Firefox). Calibration (per-motor min/max) is stored by Station and survives restarts; **never open a bus's calibration page casually: it resets that calibration.** Calibration backups: `calibration/*.json`, `./norma backup`.
- Cameras (USB, 640x480 MJPEG): table/front camera + wrist camera. Camera names (cam0..cam6) **renumber at every Station restart**; always use `front_camera(be)` in `src/normakit/station_backend.py`.
- `src/normakit/`: `station_backend.py` (real arm over Station), `sim_backend.py` (pybullet twin, same interface: `state()`, `send_goals()`, `frame_jpeg()`), `kin.py` (URDF IK/FK incl. orientation `ik_pose`), `arm.py` (grasp helpers: close gripper, wait steady, +2 s), `vision.py` (table homography, `find_dark_object`), `detect.py` (YOLO11n), `er2.py` (Gemini Robotics-ER 2 pointing, key via header only), `mcp_server.py` (MCP tools for Claude), `agent.py`.
- Perception options for the cup: dark-blob detector (reliable, model-free), YOLO11n ("cup" works, conf ~0.6), Gemini Robotics-ER 2 (`er2.point`, 1.5-3 s; once pointed at a wrong spot, so always cross-check against the blob; the script aborts if >45 px apart).
- Pixel -> table x,y: `calibration/table_front.json` (hand-touched 4 board corners), **accuracy only about +-3 cm**; lens calibrations `calibration/camera_front.json` (0.39 px), `camera_wrist.json`.
- Main script: `scripts/grab_cup.py` (plan-only without `--go`). Plan = detect cup -> IK seeded from the owner's demonstrated grasp (`run/demo_traj_pick1.json`: same gripper orientation/height) -> straight-up lift -> hover -> 4 descents, every path FK-checked (stay high, do not sweep through the cup) -> close, wait jaw stop + 2 s -> lift. Safety: high current (>700 mA) or timeout holds pose and aborts the whole run.
- Other scripts: `check_vision.py` (calibration+detection pictures), `teach_pick.py` (record/replay poses), `demo_record.py`/`replay_traj.py`, `teleop.py`, `autorun.py` (one-process pick-place loop, `--vision color|dark|yolo|claude`), `record_demo.py` (LeRobot dataset). Run with `.venv/bin/python`.

## Current status (honest)
- Works: Station, calibration persistence, camera streams + lens calibration, table mapping (+-3 cm), cup detection (blob/YOLO/ER-2), recorded demo of a full pick+place, planning in plan-only mode.
- Fails: **execution on the real arm.** The lift (first move) times out: joints 3 and 4 (elbow) barely move (j3 0 ticks for a +-40 command, j4 13). Registers look normal (torque limit 500, protection current 500, status 0x00, currents <10 mA). The camera showed the follower's jaw inside the cup opening; unknown whether it was caught or lacks torque. Joint direction/URDF sign check on the follower never done (ISSUE-107). Leader joint 4 range odd (ISSUE-114).
- No grasp has ever been attempted/verified. Simulation: `autorun.py` loop runs end-to-end but the sim grasp itself fails (`timeout reaching goal`, contact at table height) - fix this.

## Lessons from the owner's earlier EBIM robot project (apply unchanged)
1. Build the loop, don't be the loop: ONE process holds the decision loop; AI is used for perception only; decisions and retries are code. Do not chain scripts by hand.
2. Look at the camera image first, before theorising about any failure.
3. Calibration is not hardcoding: fixed object properties (grip depth, jaw opening) live in JSON taught from a demo; **where** things are is measured live every run, never a constant.
4. Never judge a grasp by jaw gap or current alone: verify visually (camera + model). Current is telemetry.
5. Close fully, wait until the jaw stops, wait 2 s more, then lift.
6. When geometry is unclear, ask the owner for a leader-arm demonstration (5 s beats hours of measuring).
7. Evidence first (what changed since the last good run), then ask before editing working code.
8. The organisers' brief is the source of truth for scoring.
Also learned this session: set goal = present position before torque on; never cycle torque on a single joint while the arm is raised (joint 3 went limp and the elbow sagged 28 deg); never leave background monitors spamming; do not call something wrong without evidence (the simulation was right, the earlier claim was wrong); the owner wants short, one-step-at-a-time instructions and no long delays.

## Your tasks, in order
**A. Realistic MuJoCo simulation (you do this fully, no hardware). Not the pybullet toy in `sim_backend.py`: build a physics-faithful twin in MuJoCo (`pip install mujoco`, 2.3.3 is installed locally; use a current 3.x).**
1. Robot model: convert the official URDF `vendor/norma-core-src/hardware/elrobot/simulation/elrobot_follower.urdf` (+ STL meshes in `assets/`) to MJCF (`mujoco` URDF compiler, then hand-clean). Keep joint limits, inertias and mesh collision for the gripper jaws (use convex decomposition for the jaws and cup contact; not boxes).
2. Actuators = the real servos (ST3215, 12 V): position actuators with kp/kv and force limit matching a measured torque limit (Station register limit 500/1000), goal speed + acceleration ramps like the real `REG_GOAL_SPEED/ACC` (speed 300-450 ticks/s class, acc 40), 4096 ticks/rev quantisation, deadband ~1 tick, backlash, joint friction/damping, gravity ON. The elbow joints (3,4) must sag and stall like the real arm: tune stiffness/torque limit so that the sim reproduces the real symptom (j3/j4 barely move under load) and then find the torque setting that fixes it. Calibrate these parameters from the real data in `handover/` (`real_snapshot.json`, `real_buses_dump.txt`, `last_grab_attempt.log`, `run/demo_traj_pick1.json` = a real recorded trajectory: replay it in the sim and fit parameters until the sim end-effector path matches the real one; report the fit error).
3. World: table plane with realistic friction, black cup (cylinder shell, ~7 cm top width, ~9 cm high, mass ~30-60 g, friction 0.6-1.0, hollow so a jaw can enter), a bottle distractor, bottle and cup randomised. Jaw contact: soft contacts (`solref/solimp`), `condim 4`+ for grasp friction, tune until a human-demo grasp (replay of `demo_traj_pick1.json` at the demo cup position) lifts the cup in sim as it did in reality.
4. Cameras: MuJoCo cameras placed exactly like the real table camera (fit pose from `calibration/camera_front.json` intrinsics + `table_front.json` mapping; 640x480), textured table, lighting; render with the SAME perception code (blob / YOLO / ER-2) as the real pipeline. Wrist camera likewise. Add image noise/exposure randomisation.
5. `MujocoBackend(Backend)` with the same interface as `StationBackend` (`state()`, `send_goals()`, `frame_jpeg()`, camera names, tick units) so `scripts/grab_cup.py` and `autorun.py` run unchanged with `--sim mujoco`. Add a live viewer option and video recording of each trial.
6. **100-trial benchmark** `scripts/sim_benchmark.py`: randomise cup position over the reachable area (x -0.15..0.20, y 0.15..0.38 m), inject +-3 cm Gaussian detection/mapping error (like the real calibration), servo lag, mass/friction, lighting, distractor position. Success = cup lifted >= 5 cm and held 3 s (pick-and-place: placed within 3 cm of target). Save per-trial JSON + videos of failures, summary (rate, failure reasons, median time, how many aborts). Target >= 90 %; if lower, add closed-loop correction (wrist-camera visual servo / vertical approach), not tighter calibration. Timeouts must stop the sequence.
7. Validation of the twin (report honestly): sim-vs-real for (a) the replayed demo path, (b) gravity sag of the resting arm (real: EE z 0.220 -> 0.138 m after joint 3 went limp), (c) stall behaviour of j3/j4. State the remaining sim-to-real gap.
8. Commit results (`results/sim_100.json`, summary in `docs/`).

**B. Real robot (human executes, you write and review)**
1. Write `docs/REAL_RUNBOOK.md`: numbered steps, each with expected output and a stop condition. Order: (a) `./norma status`, free the arm from the cup, support the arm; (b) open-air torque test of j3/j4 (+-60 ticks, expect movement); if they do not move, raise their torque/protection limits in `run/station.yaml` (`motor-current-thresholds`) and test again, or diagnose the servos; (c) joint direction check vs URDF (small moves, compare FK with camera); (d) plan-only run; (e) first `--go` at low speed, cup moved aside from the jaws, human hand on the power plug; (f) then speed up; (g) visual grasp verification; (h) place.
2. Add a vertical (top-down) approach option if orientation error stays > 20 deg, and a wrist-camera final alignment to remove the +-3 cm mapping error.
3. The sim harness must be re-run with any parameter you change, before the human tries it on the real arm.

## Definition of done
Sim (MuJoCo, calibrated to real data): 100-trial report with success rate and failure analysis committed. Real: the human runs the runbook; the cup is picked autonomously (camera-located, no hardcoded positions), verified visually, and placed; log + video saved. Report at each gate with evidence (numbers, images), not claims.
