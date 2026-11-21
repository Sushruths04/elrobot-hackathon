# Next steps (updated 2026-11-21 ~14:15)

**Hardware owner:** Sushruth's laptop (both arms + both cameras plugged in there). Station runs from `run/` (`./norma start`).
Claim a task by writing your name next to it. Ask a human before any arm motion (AGENTS.md safety rules).

## State right now
- Leader `5B79016336` (/dev/ttyACM0, 7.5 V, handle + trigger) and follower `5B8E114691` (/dev/ttyACM1, 12 V). Roles confirmed by a movement test.
- Follower calibrated (auto, frozen, spans match the URDF within ~1% except joint 1). Leader calibrated manually, frozen, but **joints 5 and 7 are short** (46% / 57% of expected) - ISSUE-103.
- Station web-UI mirroring leader -> follower was ON at last check. Station-restart behaviour of the calibration is untested (it is stored in `st3215/meta`, see docs/SETUP.md).
- Cameras are the main blocker: see ISSUE-106.

## Tasks
| # | Task | Exit check | Owner |
|---|---|---|---|
| 1 | Stabilise both cameras (cables / direct laptop ports; udev rule for 2c7f:4a60) | `snap_cams.py` returns a clear frame from both | open |
| 2 | Restart Station with `resize_target: 0` (already in `run/station.yaml`), `./norma backup` first | frames larger than 299x224 | open |
| 3 | Camera calibration with the official OpenCV board (`calibration/official/`) - `scripts/calibrate_camera.py` | rms < 1 px, `calibration/camera_<cam>.json` | open |
| 4 | Find which joints move opposite between leader and follower (incl. trigger vs gripper) - ISSUE-107 | table joint -> same/opposite | open |
| 5 | Custom teleop loop with per-joint invert, speed cap, home pose (replaces Station mirroring) | follower tracks leader, one writer on the bus | open |
| 6 | Redo leader joints 5 and 7 (manual sweep) - ISSUE-103 | spans >= ~3500 ticks | open |
| 7 | First motion test `scripts/first_motion.py --joint 8 --delta 0.03 --yes` (mirroring OFF, human present) | joint moves ~3% and returns | open |
| 8 | Table camera -> robot mapping (`calibrate_camera.py --table`), then Claude tool loop (`src/normakit/mcp_server.py`, `elrobot-real`) | pick a block by pixel | open |

## Notes
- 2026-11-21 15:40 — Bart's macOS repo merged in (tools/, start/restart scripts, ISSUE-001..015). Bart's repo `bpachniewsk1/normacore-hackathon` is superseded by this one.
- 2026-11-21 15:58 — **Bart's Mac has the arms** (both buses + 1 camera b309a703). Both arms NOT SAVED here (ISSUE-109) → recalibrating: follower auto, leader manual (`tools/calibrate.py`) — in progress, Bart + Claude.
- 2026-11-21 15:45 — Follower calibrated on Bart's Mac (all 8 saved, spans match reference; m1 now 1972 ≈ URDF 2021). **Leader manual calibration next** (needs a human sweeping every joint). Local link `vendor/norma-core-src -> ../repo/norma-core` makes Sushruth's scripts work on the Mac.
- 2026-11-21 15:52 — Leader manual sweep abandoned (hard to tell joints apart); team switching to **web-UI Auto Calibrate for the leader**. Watch for ISSUE-003 (erratic leader auto-cal earlier today).
- 2026-11-21 15:55 — Leader auto-cal done but joints 3/4/5 short (ISSUE-111). **Do not start mirroring.** Next: manual leader calibration.
