# State at the end of the 2026-11-21 session (stopped by the user; resume tomorrow)

Everything is stopped: no motion script or recorder is running, **ARM is off** (moving is refused), mirroring is OFF, the arm holds the detection pose (folded back, out of the camera view) with torque on.
Station, `scripts/armd.py` (:8891) and `scripts/ui.py` (:8890) are still running; `./norma check` shows the stack state, `./norma down` stops the UI and armd.

## What works (real arm)
- Fast stack: command latency ~25 ms (the backend now keeps only the newest state message), `armd` daemon with servo-side trajectories, ARM gate, MCP `elrobot-ui` (20 tools), web UI with jog/demos/record/ARM/STOP/keys.
- Vision: Nemotron-3.5-Super-VL through Nebius (`run/vlm.json`, benchmark in `run/vlm_bench.json`); Gemini ER-2 key works but has a small quota.
- Calibration `calibration/table_tip_front.json`: planes A (jaw tip 10 cm, mean 10 mm), B (6 cm, 6 mm), C (4 cm over the brick area, 14 mm, 10 of 14 inliers).
- **Pen/marker into the box** (`scripts/pick_loop.py`): ~23 s from seeing the pen, first-try holds on several pens; hold = gripper stall reading (thick marker ~1550-1590 ticks, thin pen ~1397, nothing ~1325).
- Teleop demos with follower+leader recorded (`run/demos/teleop_pick1/2`, labels in `episodes.json`).

## What did not work today (be honest tomorrow)
- **Stacking the red brick on the yellow one** (`stack_bricks.py`, `practice_stack.py`, `stack_loop.py`, `bricks.py`): one run placed it well (the user saw it land perfectly, then my early press knocked it off); later runs missed or placed 2-4 cm off. Cause found at the very end: the table-level pixel->xy mapping is only good to ~1.5-4 cm (extrapolation from 6/10 cm planes; plane C helped: the photo test put the jaws around the brick). The last run held the brick on try 1 (1933 ticks) but then the objects vanished from the picture (the user may have moved them) and the loop stopped on "lost sight". Treat these scripts as untested prototypes. `practice_stack.py` has a known weakness (two red bricks confused its measurements); `stack_loop.py` is the better one.
- The brick colour detector (`bricks.py`) is fine; the red-vs-orange identification with two red bricks needs care.

## Settings I changed that are still in effect (check before use)
- Servo torque limit (RAM): j2-j4 = 800 (was 500; raised on a wrong diagnosis - the "weak elbow" was command lag/mirroring), gripper = 850 (firmer grip). These reset on servo power-cycle. Station restart not needed.
- `armd.py`: gripper OPEN = 2650 ticks (was 2183, wide gap), close = stall detection, SPEED_CAP per joint, ZMIN 0.03 path check.
- `run/grasp_offset.json` a = -0.006 m (aim along the jaw axis, last value that held); `run/place_corr.json` = 0; `run/grasp_offset_b.json` = 0.
- Gemini/Nebius keys are in `~/.gemini_er_key`, `~/.nebius_key` (chmod 600), never in the repo.

## Root causes found today (do not repeat)
1. Backend parsed ~800 state msgs/s -> command latency 0.15-2.9 s (fixed in `station_backend.py`).
2. **Station mirroring was left ON after teleop** and overrode every command (`./armd /mirror {"on":false}` turns it off).
3. Old table mapping had x mirrored vs the kinematics (pushed the box once); recalibrated with the jaw tip.
4. Objects at z != plane height are mis-mapped by 1-2 cm (perspective): use the right height / plane C.
5. The model's "tip of the gripper" answer is biased/noisy: averaged over a grid it is fine, per-run alignment with it is not.

## Ideas for tomorrow (the user will bring new ones)
- Wrist-camera visual servo for the last 5 cm (removes the mapping error), or teach-by-demonstration grasps fitted in pixel space.
- Record proper 25-30 fps video + joints for every run (`scripts/record_run.py`) and convert the demos to a LeRobot dataset.
- Commit/push when the user says so (everything is committed locally only).
