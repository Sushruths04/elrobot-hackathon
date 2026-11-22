# Prompt for the next agent (paste this whole file)

You are taking over the NormaCore ElRobot hackathon repo (`normacore-hackathon`, private repo `Sushruths04/elrobot-hackathon-private`). Read `.claude/skills/elrobot-task-recorder/SKILL.md` first, then `docs/RESULTS.md`, `docs/NEXT_SESSION.md`.

## Rules from the user (non-negotiable)
- The human presses **ARM motion** (http://127.0.0.1:8890, 10-minute window). Never work around it. Check `armed_s` in `GET :8891/state` before moving.
- **Film every task and save a LeRobot episode**: `.venv/bin/python scripts/film_task.py <task>` (tasks registered in `TASKS` in that file). Never kill `film_task.py` mid-run (the video is lost). Stop guided sessions with `touch run/STOP_GUIDED`.
- Success is the human's verdict or a clean camera photo with the arm out of view. Never trust a vision-model yes/no (it was wrong about "cap open" and "tissue held" many times). A thin sheet does not stall the gripper (empty 1324-1334 ticks / 0-2 mA, held the same).
- Be fast: the user wants a task in 30-60 s, no long analysis, no repeated questions. They guide in short commands ("left 2", "down 1", "sweep"): execute exactly.
- No Claude attribution in commits. Push only to the private repo `origin`; the `public` remote push is disabled on purpose.
- Measure object positions from a FRESH camera picture right before moving (zoom + labelled pixel grid); ten pinches missed because the pack had moved.

## State of the work (2026-11-22)
- Done and saved (video + LeRobot episode, see `results/films/`, `datasets/autonomous/`, `showcase/index.html`): duster+magnet task, pen into box, tissue task (cap opened by a sideways sweep at tip height ~10 cm, one tissue pinched and carried home), pack-push through coffee beans (needs a LONGER push to clear them), guided hand-taught sessions.
- Tissue hand-over for the presentation: carry the tissue to the front at ~12 cm and release it on the user's word (not throw).
- Pinching a flush, flat tissue with smooth plastic fingertips fails; the user will add grip pads. Dragging the closed jaws through the tissue raises a fold.
- **Next task: cut sellotape with scissors.** The user hands the scissors over; each jaw must sit inside one finger loop (open = blades open, close = cut). The user sets the location (two boxes with the tape between them, right side of the table near the yellow line) and teaches by hand with the soft hold (`film_task.py scissors`, 15 min, 5 s countdown, jaws open). Reach limit: the spot between the boxes is about (-0.20, 0.43) m, **out of reach at 13 cm in every orientation** (IK error 20-26 mm); the nearer box (-0.17, 0.39) is reachable (tilt 90, roll +-90 deg, 6-12 cm). A perfectly vertical tool (tilt 0) is only reachable at (-0.14, 0.20, 0.08). Ask the user to move the boxes closer (y < 0.40) or accept a horizontal tool.
- **Table-balancing / ping-pong idea from Devin:** see `docs/12_PINGPONG_BALANCE_AND_DRIBBLE.md` (balancing a ball on a plate with the arm). Not started on the real arm; start in the MuJoCo twin (`assets/mujoco/elrobot_scene.xml`, `src/normakit/kin.py`) before touching hardware, use low torque limits, and film it.

## Tools you have
`scripts/armd.py` fast daemon (:8891: /state /move_tip /move_joints /gripper /ik /stop /mirror), `scripts/ui.py` UI (:8890, /frame.jpg), `scripts/film_task.py` recorder, `scripts/tempo_*.py` task scripts (cap sweep, pinch, lanes, replay with safety checks), `scripts/go_home.py`, soft-hold hand teaching in `scripts/tempo_hand_demo.py`. Mirroring (leader to follower) must be OFF for autonomous work: `POST :8891/mirror {"on":false}`. Front camera = the live camera whose bottom-centre is bright (`front_camera(be)`).

## Known numbers
Gripper open 2650 (2840 allowed), closed 1321; joint torque limits 800 on j2-4, 1000 others (lower is safe, never higher). Pixel to table xy: `calibration/table_tip_front.json`, planes z=0.10/0.06/0.04, interpolate in z. Initial pose: `run/detect_pose_follower.json`.
