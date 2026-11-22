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
