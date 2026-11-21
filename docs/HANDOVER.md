# Handover (2026-11-21) - autonomous grab of the black cup

Everything captured at handover time is in `handover/` (real robot + simulation snapshots, configs, calibration, last log, pictures).
No API key is stored in the repo: the Gemini key lives in `~/.gemini_er_key` (chmod 600) or `$GEMINI_API_KEY`. Nothing here is committed yet.

## State: what works / what does not
| Part | State |
|---|---|
| Station, 2 buses (follower 5B8E114691, leader 5B79016336), calibration persisted | works |
| Table camera 640x480, lens calibration 0.39 px, table mapping (pixel -> robot x,y) | works, accuracy only about +-3 cm |
| Cup detection: dark-blob (`vision.find_dark_object`), YOLO11n, Gemini Robotics-ER 2 (`normakit/er2.py`, ~1.5-3 s) | works; ER-2 once pointed at a wrong spot (pixel 74,202 vs cup 253,243) -> `grab_cup.py` cross-checks it against the blob and aborts if > 45 px apart |
| Planning in `scripts/grab_cup.py` (IK seeded from demo `run/demo_traj_pick1.json`, straight-up lift, hover, descents, FK path checks) | works; plan-only mode passes |
| **Execution on the real arm** | **fails**: first step (lift) times out, joints 3 and 4 hardly move (j3: 0 ticks for +-40 command, j4: 13 ticks) |
| An actual grasp | never reached or verified |

## Open problems (in the order to attack them)
1. **Elbow joints 3/4 do not follow commands** in the cup pose. Registers looked normal (torque limit 500, protection current 500, status 0x00, lock 1, currents < 10 mA). Unknown whether the jaw was caught in the cup (the camera showed a jaw inside the cup opening) or the servos lack torque. Test: free the arm from the cup first, then command +-60 ticks on j3/j4 in open air.
2. Joint direction / URDF sign check on the follower not done (ISSUE-107). Leader j4 range odd (ISSUE-114).
3. Mapping accuracy +-3 cm: use a visual check (wrist camera) before the final descent, or ArUco on the cup/table.
4. Descent uses a demo orientation; IK orientation error 13-33 deg at hover - consider a vertical (top-down) approach.
5. The cup was moved/out of its place during tests: re-detect before every run (the script does).

## How to run
```
./norma status                     # Station up? (./norma start / restart)
.venv/bin/python scripts/grab_cup.py            # plan only, no motion
.venv/bin/python scripts/grab_cup.py --go       # execute (person at arm, power plug in reach, bottle clear)
.venv/bin/python scripts/grab_cup.py --vision blob   # without Gemini
.venv/bin/python scripts/check_vision.py        # calibration + detection pictures
.venv/bin/python scripts/teach_pick.py limp --yes    # release the arm
```
Stage logic: a timeout or high current (> 700 mA) holds the pose and **stops the whole script** (exit 2/3).
Offline: `normakit.sim_backend.SimBackend(realtime=False)` has the same interface (state/send_goals/frame_jpeg) as `StationBackend`; the sim cannot show servo torque, gravity sag, jaw contact or mapping error.

## Safety lessons from this session
- Never cycle torque on one joint while the arm is up: an interrupted test left joint 3 limp and the elbow sagged 28 deg. Always set goal = present before torque on.
- Camera names (cam0..cam6) renumber at every restart; use `front_camera(be)`.
- Do not paste API keys into chat; keep them in `~/.gemini_er_key`.

## Files in `handover/`
real_snapshot.json (follower joints, ER-2/blob/YOLO detections, EE position), real_buses_dump.txt (both buses), real_front_cam.jpg, after_grab_attempt.jpg, now_arm.jpg, er2_out.jpg,
sim_snapshot.json, sim_front_cam.jpg, station.yaml, buses.json, camera_front/wrist.json, table_front.json, teleop_map.json, demo_traj_pick1.json, last_grab_attempt.log.

## Incident 2026-11-21 ~20:15: marker run pushed the white box
`grab_marker.py --go` (first real run): the table mapping `calibration/table_front.json` has **x mirrored relative to the forward kinematics / IK frame** (pixel 300 -> x=+0.105 by the mapping, but moving the jaw tip to x=+0.105 put it at pixel 227; +x goes LEFT in the image). The visual-alignment loop used that same mapping's local Jacobian, so each correction (clipped at 8 cm) moved the arm further the wrong way (tip px 227 -> 186 -> 140) at tip height 5 cm = box-wall height, and the jaw pushed the box about 12 cm to the left (box px 205 -> 75). Marker untouched, no fault/overcurrent. The script aborted itself only because IK failed at the next step.
Fixes needed before the next real run: (1) re-fit pixel->xy from the jaw tip itself (`scripts/calib_tip_map.py`, plan checked, not yet run); (2) alignment loop must stop if the pixel offset grows after a correction and limit steps to 3 cm; (3) path checks must treat the box as an obstacle (keep tip z >= 0.10 within 8 cm of the box); (4) never start a run when the mapping disagrees with the arm's own pose.
