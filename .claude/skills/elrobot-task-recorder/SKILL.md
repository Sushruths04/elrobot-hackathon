---
name: elrobot-task-recorder
description: Run, film and LeRobot-record any task on the NormaCore ElRobot follower arm (camera + vision model + kinematics), teach tasks by hand-guided demo, replay them, and verify results honestly. Use when the user wants the arm to do a new desk task, record a demo, replay a demo, or save runs for training.
---

# ElRobot task recorder (transferable workflow)

Everything lives in the repo `normacore-hackathon`. Station (Rust) owns the motor bus; `scripts/armd.py` (:8891) is the fast arm daemon; `scripts/ui.py` (:8890) is the web UI with the **ARM motion** button; the front camera is picked automatically (`normakit.station_backend.front_camera`, names renumber on every restart).

## Standing rules (from the user)
1. **Film every task AND save it as a LeRobot episode**, trimmed to the task. No unfilmed runs, not even "just a check": give every script a `--check`/`--dry` flag that never moves the arm.
2. **The human presses ARM** (10 min window, `armed_s` in `/state`). Never work around it. Re-check `armed_s` before each batch of moves.
3. Success is decided by the human or by a clean camera photo with the arm out of the way. Never by the vision model alone (it said "lid open" while the arm hid a closed lid) and never by the script's own "reached" message.
4. Never press, sweep or push down unless the task says so. Hold/lift gently (`acc` 12-20), slow when close to objects.
5. Push only to the private repo, only when asked; no Claude attribution in commits.

## How to run a task (filmed + recorded)
`.venv/bin/python scripts/film_task.py <task>` where `<task>` is registered in `TASKS` in `scripts/film_task.py` (script, log file, dataset name). It starts the task script, records captioned and caption-free video (H.264, real fps), joint samples, builds `datasets/autonomous/<name>_<stamp>/` (LeRobot v3, observation.state(8), action(8), observation.images.table, 30 fps) plus `OUTCOME.json`, and rebuilds `showcase/index.html`.
Registered: `pen`, `duster`, `tempo` (autonomous lid push), `tempo_hand`, `tempo_pinch` (hand-guided demos), `tempo_replay`. A task script must print a line containing `DONE` (or `STOP`/`ABORT`) at the end and wait for ARM itself.
Always set `OUTCOME.json` `success` only after the human's verdict; add a `note`.

## Teaching by hand (kinesthetic demo, no leader arm needed)
`film_task.py tempo_hand` / `tempo_pinch`: after the human says "holding" a 10 s countdown starts, then **soft hold**: torque limit lowered (j1 120, j2 260, j3 220, j4 160, j5-7 100) and the goal position follows the arm at 25 Hz, so the arm carries its weight, stays where it is left and can be pushed by hand. Fully torque-off makes the arm collapse. Ends on `run/STOP_HAND_DEMO`; then normal limits (800 on j2-4, 1000 on the rest) are restored. The gripper is commanded by chat (`POST :8891/gripper {"to":"close"}`), action = the follower's own next state.

## Replaying a demo
`scripts/tempo_replay.py [--check] [--samples results/films/<demo>_samples.json] [--speed 0.8]`: trims the idle part, refuses to start unless the arm is within 80 ticks of the demo's first pose, checks the FK tip height of the whole path (>2 cm), then follows with lag and current watchdogs. Works only if the objects are within ~1-2 cm of where they were in the demo. Start poses: `run/detect_pose_follower.json` (folded upright at the back = "initial position"). Verified: the lid demo replay left the lid semi-open with the pack 2 cm off.

## Autonomous steps (camera + model + IK)
- Pixel to table xy uses planes in `calibration/table_tip_front.json` (A z=0.10, B 0.06, C 0.04): xy is linear in z along the viewing ray, interpolate between planes for the object's height. Detect with the vision model (`normakit.er2.point`, median of 3 calls) or colour detectors (`bricks.py`, `duster.py`).
- Move with `POST :8891/move_tip {x,y,z,roll,speed,acc,low_ok,timeout}`; the tip sags ~8 mm at low stretched poses, so compensate or verify by hovering and photographing before touching.
- `/home` on armd replies ok in 0.4 s even when nothing moves, use `scripts/go_home.py` or move to the detection pose with `/move_joints` and check the photo.
- Gripper: open 2650, closed 1321; hold detection by stall ticks (thin pen ~1397, marker ~1550-1590, duplo ~1890-1980, cut duster ~1888 with ~310 mA). A close that ends at ~1322-1326 with 0 mA grasped nothing.
- Mirroring (leader to follower) must be OFF for autonomous control: `POST /mirror {"on":false}`.

## Lessons learned
- Duster (magnet over caps): hold the duster over each cap with no pressing; the camera check counting caps in a fixed ROI can miss a cap, always look at the photo; the duster dropped at fast waypoint moves, carry gently.
- Tissue pack: a closed gripper pressing down on the lid fails; the lid opens by a sideways lift from the front lip. A hand-guided demo opened it, its replay semi-opened it. Pinching a single tissue is not solved (no clean pinch recorded yet).
- Check the physical scene after every run (pack nudged, cap moved) before trusting the next run.

## Lessons from the tissue-pack task (2026-11-22), what actually worked and what did not
- **Always measure the object position from a fresh camera picture right before moving.** The pack was moved (by the user and by my own sweeps) between looks; ten pinch attempts missed because I aimed at old pixels about 10 cm off. Make a zoomed picture with a labelled pixel grid, read the position, convert with `xy_at(u, v, z)`, move above it, look again, then go down.
- **Never trust a model yes/no for physical success** ("cap open", "tissue held" were false positives many times). Trust: a before/after picture difference with the arm out of view for the cap, and your own look at a clean photo for the tissue. A thin tissue does not stall the gripper: closed-empty reads 1324-1334 ticks / 0-2 mA and so does a held sheet.
- **Cap (flip-top, hinged on one side):** the fingers must catch the cap lip. Working recipe: tip height 9.7-10.3 cm commanded, start 5.6 cm image-left of the cap and 1.8 cm back, one 14.5 cm sideways stroke toward image-right at the SAME height (no stepping down), arm aside before/after, picture change > 12 means opened. Lower than 7 cm pushes the whole pack; 8-9 cm skims the cap.
- **Tissue:** pinch only works at the tissue's standing edge or fold, with the jaws about 8-9 cm tip height, straight down, close fully, then pull STRAIGHT UP (not forward), then carry. A flush flat tissue cannot be pinched with smooth plastic fingertips: drag the closed jaws through it to raise a fold, or put a grippy pad (rubber/tape) on the fingertips.
- **Beans:** straight lanes at table level (tip 4.4-4.8 cm), each lane one length toward image-right, lift at the end, shift the next lane by 2 cm, 3-8 lanes covering the pile; the pile position must come from the picture (my colour/dark-blob bean finder failed three times; read the pixel by eye or use the user's pointer).
- **Moves:** one move to the start position at the working height beats stepping down in small steps (the user hated stepping). Speed 400-500, acc 40-70 for travel; 40-120 near the object. IK errors "cannot reach" appear at x > 0.2 or beyond about -0.35.
- **The user guides in short chat commands ("left 2", "down 1", "sweep"):** execute them exactly, keep recording, no questions in between; when told "do it autonomously", loop with attempts and report each stage in one line.
- **Recording:** `film_task.py tempo_guided` keeps a session open; stop it with `touch run/STOP_GUIDED`; killing film_task loses the video (moov atom missing).

## Teleop data collection for training (leader arm -> LeRobot), lessons from 2026-11-22
Tools: `scripts/teleop_panel.py` (http://127.0.0.1:8893: both cameras, leader-vs-follower table, MIRRORING ON/OFF, START/END SESSION, GO / SAVE OK / SAVE FAIL / DISCARD, keys G/S/F/D) runs `scripts/record_teleop.py` (LeRobot v3, state = follower 8, action = leader 8, images.table + images.wrist, 30 fps, labels in `<root>/EPISODE_LABELS.json`). Review doc: `docs/TELEOP_SMOLVLA_PLAN.md`. Target: 50+ successful episodes per task for SmolVLA; training runs on the cluster.
The first session (`datasets/teleop_spoon_beans_20261122_093829`, 5 episodes) was unusable. **Run this checklist before the operator records anything:**
1. **USB cameras stable.** `journalctl -k --since "-10 min" | grep -E "USB disconnect"` must be empty. A camera on hub port 1.1.2 dropped ~25 times; wrist cam froze for 935 frames. Plug cameras directly into the laptop. After a replug, Station often does NOT re-attach: backup calibration (`./norma backup`), restart Station, check frames are live (two fetches 1 s apart must differ; `be.frame_age(cam)` < 0.5 s).
2. **Leader calibration covers the leader's real motion.** Move the leader through its full range and check every joint's norm stays within 0..1. The leader's j4 range (1404-2692) was narrower than its real motion (3316) -> action up to 1.49 while the follower was clamped at 0.99 (29-58% of frames in some episodes). Fix the range before recording.
3. **Match before mirroring.** Every arm joint |leader_norm - follower_norm| < 0.08, otherwise the follower jumps (the panel refuses).
4. **Real 30 fps.** `LeRobotDataset.create(..., image_writer_threads=8)` is required. Without it add_frame takes ~96 ms with 2 cameras -> ~10 fps real, but frames are stamped 30 fps -> motion 3x too fast, and it can't be repaired afterwards. Check `real_fps` (>= 28.5) in EPISODE_LABELS for every episode.
5. **Fresh readings.** `normakit.busread.snapshot` with a reused queue must drain the backlog (fixed), otherwise the readings lag further and further (the panel showed old joints, which blocked mirroring).
6. **One recorder only.** Restarting the panel orphaned a record_teleop process; then two recorders raced on the same `run/TELEOP_*` files. Close a recorder with `touch run/TELEOP_QUIT` (killing it skips `finalize()` and corrupts the v3 dataset). Check with `ps -eo args | grep [r]ecord_teleop`.
7. **Record 1 test episode, then verify** before mass collection: real_fps, stale_camera_frames = 0, the longest run of identical video frames is small, j4 action stays <= 1.0, then play the video and look at it.
8. Never `pkill -f <pattern>` from a Bash command that contains the same pattern (it kills the shell itself). Use `ps -eo pid,args | awk '$2 ~ /python/ && /name/ {print $1}' | xargs -r kill`.
9. Local video decoding (torchcodec) is broken on this laptop (CUDA 13 build, libavdevice58 missing). Check videos with PyAV. Training uses the cluster env.

## Lessons from the spoon pick-and-place (2026-11-22), read before any camera-guided grasp
- **The front camera calibration goes stale when the camera is replugged or moved.** `table_tip_front.json` was 4-9 cm off. Re-measure with the arm: `scripts/calib_front_from_arm.py` (clear the table, keep people out of view; it saves frames to `results/calib_frames/`). The camera pose goes to `calibration/front_camera_pose_arm_wide.json` (pixel → ray ∩ plane z, about 1 cm). Check the coverage includes the target area (the cup was outside the first grid).
- **The front view has no depth.** Matching the jaw pixel to the object pixel is only a 3D match if both points are at the same height. Compute the object in 3D and command that z.
- **Don't detect the gripper with "dark pixels"** (beans, cup and shadows are dark too). Background difference against an arm-out-of-view photo, gated near the predicted pixel, is OK.
- **Replaying teleop state is not enough for the gripper:** command a firm close (norm 0), not the position the jaws reached on the object. Grip check: wait until the jaws stop. Empty ≈ 1263-1326 ticks; spoon handle ≈ 1352-1355 at ~100 mA.
- **When shifting a demo to a new object position, shift only the approach and grasp.** The carry and release must be aligned to the TARGET (cup), not the grasp offset.
- **Nemotron-VL (Nebius) is a reasoning model:** `llm.ask(..., max_tokens=3000)`, about 20 s per answer. Its points can be 30 px off on thin objects (spoon handle top). Look at a zoomed grid image yourself to confirm.
- **Operator spoon demos:** grasp about 85% up the handle at z 0.12-0.13, tilt about 18°, j7 0.76; scoop dip 3 cm; lift to 0.20-0.23; release with the gripper about 7 cm from the cup centre toward the bowl, z 0.165; the spoon drops into the cup. Analysis files: `results/spoon_demo_*.json`.
- Review prompt with the full history: `docs/CODEX_REVIEW_PROMPT.md`.
- **Codex review (2026-11-22):** the operator's spoon grasp is near-horizontal (~87° from vertical, tool axis = gripper-base +Y), so reuse the demo's full orientation (`scripts/spoon_grasp_v3.py`). armd now path-checks partial moves and rejects IK error > 12 mm / 20°. llm.point uses 3000 tokens and returns nothing on incomplete reasoning. The panel refuses frames older than 0.5 s. Pickers must STOP on any failed step or non-"yes" verdict.
