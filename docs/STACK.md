# Software stack (what runs, how it fits, how to bring it up) - 2026-11-21

## One-command bring-up / check
```
./norma up        # starts Station if needed, the web UI (http://127.0.0.1:8890), then the health check
./norma check     # read-only health check (add --er2 for a live Gemini call) -> run/stack_check.json
./norma down      # stops the UI only (Station keeps running: restarting it may lose calibration, ./norma backup first)
```
Health check on 2026-11-21 20:5x: PASS 11, WARN 1 (table mapping mirrored, see below), FAIL 1 (Gemini HTTP 429 = quota/rate limit of the key).

## Layers
| Layer | Where | Notes |
|---|---|---|
| Hardware | 2x ElRobot (8 ST3215 servos each): follower `5B8E114691`, leader `5B79016336`; table camera + wrist camera | `run/buses.json` |
| Station (Rust) | `bin/station`, `run/station.yaml`, logs `logs/station.log` | NormFS TCP :8888, web :8889 (Firefox). Calibration per motor (range lo/hi, frozen) is stored in Station; **never open a bus's calibration page casually (it resets)** |
| Python backend | `src/normakit/station_backend.py` (real), `sim_backend.py` (pybullet), `mujoco_backend.py` (Devin's twin) | same interface: `state()`, `send_goals()`, `frame_jpeg()`; `front_camera(be)` because camera names renumber at each restart; `be._sync(REG, {motor: bytes})` writes servo registers |
| Kinematics | `src/normakit/kin.py` | see below |
| Perception | `er2.py` (Gemini Robotics-ER 2: `point`, `ask`), `vision.py` (homography, dark-blob), `detect.py` (YOLO, now unused) | key from `~/.gemini_er_key` or `$GEMINI_API_KEY`, sent in a header only |
| Tasks | `scripts/grab_marker.py` (pen -> white box), `grab_cup.py`, `go_home.py`, `calib_tip_map.py`, `teach_pick.py`, `demo_record.py`, `replay_traj.py`, `teleop.py` | every moving script: plan-only without `--go`; timeout / high current holds the pose and stops |
| Interfaces | `scripts/ui.py` (web UI: cameras, joints, jog, detect, run, PROCEED/STOP, recording, live log) ; `normakit/mcp_ui.py` (MCP `elrobot-ui`, 9 tools, drives the UI) ; `normakit/mcp_server.py` (`elrobot` sim / `elrobot-real`) | `.mcp.json` |

## Kinematics in detail
- **Frames:** robot base at origin, **y forward** (towards the table/camera), x lateral, z up, table plane z = 0 for the jaw tip. Base yaw of a point = `atan2(x, y)`.
- **Tick <-> radian:** encoder 4096 ticks/rev, wrap-aware difference `wd(a,b) = ((a-b+2048) % 4096) - 2048`. `norm = (raw - lo) / ((hi - lo) % 4096)` over the calibrated range; `rad = URDF_LIMITS lo + norm * (hi - lo)` (`core.py`). **Linear map assumption, joint signs vs the URDF are NOT verified on the follower (ISSUE-107).**
- **Two kinematic models, agreeing to 0.0 mm at the EE link:** (1) pybullet on the official URDF (`Kin.fk/ik/pose/ik_pose`, EE link 13 = `Gripper_Base`, IK on joints 1-6); (2) MuJoCo on `assets/mujoco/elrobot_scene.xml` (`Kin.jaw_tip_pose`, `ik_jaw_tip`: 7-DoF damped least squares incl. wrist roll joint 7, tilt policy chosen from reach, several seeds, regularised towards the seed). `jaw_tip_site` = (0.0009, 0.092, 0) in the Gripper_Base frame: **approach axis = gripper-base +y, jaw opening axis = gripper-base x** (jaw slide axis -x). Roll for a thin object: choose `roll_rad` so the jaw axis is perpendicular to it (`grab_marker.roll_for`).
- **Gripper:** joint 8, follower range 1243..2853; open ~2183, closed ~1321 (the demo cup grasp stalled there). After closing: wait until the jaw stops, +2 s, then lift.
- **Servo registers used** (`station_backend.py`): torque 0x28, acc 0x29, goal 0x2A, speed 0x2E, **torque limit 0x30 (RAM)**, lock 0x37, P/D/I 0x15/0x16/0x17 (EEPROM: a write via lock-off did not read back as changed), present pos 0x38, current 0x45.
- **Motion:** proportional joint-space ramp, `speed/20` ticks per 50 ms cycle for the largest joint, others scaled; success = all joints within 40 ticks (or steady within 2x); FK path checks sample 40-60 points along each segment.

## Known problems (read before running anything that moves)
1. **Table mapping mirrored:** `calibration/table_front.json` (pixel -> robot xy) has **x mirrored** relative to the kinematics: moving the tip to x=+0.105 put it at pixel 227, the mapping said pixel 300. A run used it and the jaw pushed the white box. Fix: `scripts/calib_tip_map.py --go` (ER-2 sees the jaw tip, FK knows it; 18 grid points planned and reachable, never run) **or** avoid the mapping with taught poses (below).
2. **Shoulder/elbow (j2, j3, j4) weak/sagging when the arm is extended:** with torque limit 500 j3 stalled 285 ticks short of a lift; with **800 (user-approved, 1000 NOT approved)** and speed 100 / acc 10 it reached the target, but a straight combined lift still lags. Each `hold()` (goal = present) locks in the droop: do not use it as a safe stop at a sagged pose. Lift order that worked: elbow j3 alone, slowly; then j2; then the rest. The June hackathon repo (tejavadlamudi03/norma-robot) hit the same overload and records: retract elbow first, keep shoulder moderate.
3. **Gemini ER-2:** HTTP 429 (quota/rate limit) at 20:5x - a free-tier key allows very few calls per day for this model; occasionally returns malformed JSON (parser has a regex fallback) or a wrong point (cross-check with a second call or the blob).
4. Follower j1-j7 positions drift when the human moves the arm; always re-read state, never reuse old poses.
5. Leader joint 4 range odd (ISSUE-114). Station `current-threshold` 100 (j4: 250) - the June repo used 200.

## Recommended path to a working pick-and-place (fast)
**Teach-and-replay, not IK:** the operator moves the leader arm through the pick and the drop once (`demo_record.py`), `replay_traj.py` replays it on the follower (wrap-aware), ER-2 only verifies the result. For several object positions: record several grasps with the ER-2 pixel of the object each time and fit pixel -> joint pose (bilinear/affine) = the June repo's `grasp_model.py` idea, but in pixel space, so no robot-frame calibration is needed.

## UPDATE 2026-11-21 20:5x - root cause of the "everything is slow / joints stall" problem (FIXED)
Symptom: command writes returned after 0.15-2.9 s, ramps stalled, "no progress", joints seemed weak, state looked stale. Cause: `StationBackend` parsed **every** message of `st3215/inference` (~800 msg/s, 3 kB each, both buses) in the same asyncio loop that sends the commands -> the loop was saturated and every write queued behind the parsing backlog. Raw Station is fast (a write takes 0.04 s). Fix in `src/normakit/station_backend.py`: each follow loop drains its queue and parses only the newest message. After the fix: write latency 13-44 ms, a joint command is visible in the state after 0.4 s, `start()` 0.3 s, and `go_home.py` reached home in 26 s.
Consequences: (1) problem 2 above ("shoulder/elbow weak") was at least partly this lag - the torque limit of j2-4 was raised to 800 and a P-gain write was attempted on that wrong evidence; re-test with the original limit (500) before keeping 800; the P write did not read back as changed. (2) Any older "timeout"/"no progress" result from before this fix is suspect. (3) Restarting Station was NOT needed (it is not the bottleneck; its `run/` data dir is 21 GB, harmless so far).
