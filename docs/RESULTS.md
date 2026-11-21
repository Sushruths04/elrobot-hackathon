# Real-arm results, 2026-11-21 (zero-shot pen/marker pick and place into the white box)

Setup: ElRobot follower, table camera, calibration `calibration/table_tip_front.json` (error < 1 cm), vision model Nemotron-3.5-Super-VL (run/vlm.json),
fast daemon `scripts/armd.py`, loop `scripts/pick_loop.py` (hold = gripper stall reading). "Seconds" = from the moment the pen was seen to DONE.
Honest notes: success is judged by the gripper reading (hold) and by the model + my own look at a camera picture afterwards. Only some runs have a video (see the end).

| # | Object (position m) | Result | Time | Notes |
|---|---|---|---|---|
| 1 | marker (0.016, 0.325), front-back | held (1589 ticks), placed, model "inside", picture: in box | ~33 s motion | first full run; last steps finished by hand because of two script bugs (gripper-close timeout, lift check) - since fixed |
| 2 | marker, side-to-side (-0.011, 0.299) | missed (jaws closed on air) | 25 s | jaw gap too narrow for a front-back error; later also a mapping height error |
| 3 | marker, constant tilt 15 deg (-0.022,0.326) | missed | 24 s | |
| 4 | marker (-0.081,0.264), tilt 0 | missed | 20 s | photographed at the bottom: jaws ~3 cm off the pen (calibration aiming bias) |
| 5 | marker (-0.053, 0.242) | held at attempt 1 (1589 ticks, 117 mA), model "inside" | 35.6 s | retry loop with aim offset -1.6 cm along the jaw axis |
| 6 | marker (-0.011, 0.255), roll -52 | held at attempt 1 (121 mA), model "inside", picture: in box on the rim | 22.8 s | higher carry/release |
| 7 | two pens on the table | 2 "misses" | - | model took the ends of two different pens (18-22 cm); fixed by asking for the loose pen outside the box |
| 8 | thin blue pen (-0.041, 0.239) | held at attempt 1 (1397 ticks, 43 mA), model "inside", picture: across the box, one end on the rim | 23.1 s | a thin pen stalls the jaws at ~1397, I had counted that as a miss before |
| 9 | marker, far spot (0.001..0.046, 0.285), rolled ~45 deg | 2 real misses, held at attempt 3 (1549 ticks, 320 mA, offset -0.6 cm), model "inside: False", picture: lying across the front rim | 67.5 s | box centre from one model call varied by 4 cm; marker is as long as the box is wide -> now median of 3 calls + release along the box diagonal |

Hold readings: thick marker ~1550-1590 ticks (120-320 mA), thin pen ~1397 ticks (45 mA), nothing 1322-1325 ticks (0 mA).
Not yet shown: reliability over many positions (a handful of runs only), the placement fix (box median + diagonal release) is untested at the time of writing.

## Data
- teleop demos (follower 40 Hz + leader 50 Hz + 8 fps camera): `run/demos/teleop_pick1*` (failure, placed wrongly), `teleop_pick2*` (success); labels `run/demos/episodes.json`; videos in `run/recordings/` (not committed, 50 MB).
- run videos with joints and logs: `results/runs/<time>/` (25 fps overlay video, joints.jsonl, log) - only runs after 23:00.
- Earlier run logs were overwritten (logs/pick_loop.log is truncated per run); this table is the record.

## Later runs (2026-11-22 00:15) - duster with a magnet in its centre collects two metal bottle caps and goes into the white box
Script `scripts/duster_caps.py` (detector `src/normakit/duster.py`). Last run: duster grasped at its centre on the FIRST try (jaws 1889 ticks, ~310 mA, the cut duster is ~3 cm wide), held ~1.2 cm above each of the 2 caps for 1 s (no sliding, no pressing - the magnet is strong and takes the cap from a distance),
parked to the side, camera check: 0 of the 2 targeted caps left on the table, then straight over the white box (no waypoint, slow carry moves, acc 12) and released. Final picture `results/images/duster_result.jpg`: duster lies in the box, both caps gone from the table (they are on the magnet underneath, not visible).
Earlier attempts the same evening failed for known reasons: jaws closed on a thin edge because the old, thicker duster was taller than the clearance under the wrist-camera board; duster mapped 4-5 cm off at x >= 0.12 (calibration plane C did not reach there; extended to x = 0.17); verification that went to the detection pose and a waypoint move that dropped the duster twice (removed); the strong magnet also dragged caps around the table.

## Brick stacking (red brick on the yellow one) - not finished
Scripts `stack_bricks.py`, `stack_loop.py`, `practice_stack.py`, colour detector `src/normakit/bricks.py`. One run placed the red brick perfectly on the yellow one (user's words), then pressing knocked it off; later runs held the brick (1933 ticks) but the table-level position error (1.5-4 cm) made placement unreliable. Pen/marker into the box (`pick_loop.py`) is the reliable task (~23 s, first-try holds).

## What is saved where
- Images of the results: `results/images/`. Teleop demos with video: `demo_backup/*.mp4` (H.264) and the LeRobot v3.0 dataset `datasets/elrobot_teleop/` (113 MB, left out of git because its video file is 117 MB; on disk only).
- NOT filmed: the pen runs, the brick runs and the duster runs - only photos and logs exist for them (`logs/` is not in git; the log lines are summarised above).
