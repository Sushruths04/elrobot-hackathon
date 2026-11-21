# Changelog (newest first)

## [Unreleased]
### Added
- 2026-11-21 16:35 - `docs/NEXT_STEPS.md`: handoff header (state, follower-only project, joint-check + camera steps with exact commands) - Bart + Claude
- 2026-11-21 15:58 - `tools/calibrate.py`: watch (read-only live spans vs reference) / reset / save (freeze) / auto / stop over TCP, no web UI needed - Claude [ISSUE-109]
- 2026-11-21 15:40 - Merged Bart's macOS work onto this repo: `tools/` (verify_station, estop, reset_calibration, speed_limit), `start_station.sh`, `restart_station.sh`, `patches/teleop-max-speed.patch`, `docs/CONCEPTS.md`; ISSUE-001..015 appended to `docs/ISSUES.md`; macOS section in `docs/SETUP.md`; tools now find upstream at `vendor/norma-core-src`; `.gitignore` += `repo/`, `snapshots/` - Bart + Claude
- 2026-11-21 - `scripts/twin_view.py`: renders both real arms from live joint data (`run/twin.png`), using the Station viewer's mapping
- 2026-11-21 - `scripts/identify_arm.py`, `set_home.py`, `dump_buses.py`, `monitor_calib.py`, `backup_calibration.py`, `mirror_state.py`; `./norma` shortcuts
- 2026-11-21 - `src/normakit/busread.py` read-only bus snapshot; `StationBackend` reads follower bus from `run/buses.json`
- 2026-11-21 - MCP server `elrobot-real` entry in `.mcp.json` (station backend, untested for motion)
- 2026-11-21 - `calibration/`: per-bus calibration files (min, max, midpoint, span, frozen)
### Fixed
- 2026-11-21 15:50 - `tools/calibrate.py watch`: subscribe once (it re-subscribed every refresh and crashed with TimeoutError after a few minutes); shows `pos` and `<<< MOVING` on the joint being moved - Claude
- 2026-11-21 15:45 - `tools/verify_station.py`: saved motors use raw Present Position (offset was counted twice -> false `OUTSIDE`, wrong `--move-test` start) - Claude [ISSUE-110]
- 2026-11-21 - `StationBackend`: wrapped ranges (max < min) and positions just outside the range no longer break normalisation
- 2026-11-21 - `monitor_calib.py`: no longer crashes on short motor reads
### Hardware
- 2026-11-21 16:25 - Leader manual calibration abandoned (accidental web-UI reset 16:10:56, m4 not re-swept); leader left NOT SAVED. Team decision: follower only - Bart [ISSUE-112]
- 2026-11-21 15:52 - Leader `5B79016336` web-UI auto-calibrated on Bart's Mac: all 8 saved, but m3/m4/m5 spans short (1526/996/436) - backup `calibration/5B79016336_leader_*` is NOT usable for mirroring - Bart + Claude [ISSUE-111]
- 2026-11-21 15:40 - Follower `5B8E114691` reset + auto-calibrated on Bart's Mac (`tools/calibrate.py`, 32 steps, 156 s, all 8 saved; spans m1 1972, m2 2012, m3 2322, m4 2326, m5 3836, m6 2050, m7 3874, m8 1610; torque off afterwards). Backup `calibration/5B8E114691_follower_20261121_154324.json` - Bart + Claude [ISSUE-109]
- 2026-11-21 - Follower `5B8E114691` auto-calibrated and frozen. Leader `5B79016336` calibrated manually; joints 5 and 7 short (see ISSUE-103)
- 2026-11-21 - Roles confirmed by movement test: leader `5B79016336` (/dev/ttyACM0), follower `5B8E114691` (/dev/ttyACM1)

## [Unreleased] - 2026-11-21 afternoon
### Added
- `calibration/official/`: OpenCV's official 9x6 chessboard (`pattern.png`, unmodified, 25 mm squares) + A4 print PDF
- `scripts/calibrate_camera.py` (lens + table homography, `--ruler-mm` print-scale correction), `snap_cams.py`, `first_motion.py` (guarded, needs `--yes`), `make_checkerboard.py`, `make_aruco.py`
- `docs/NEXT_STEPS.md`, `docs/SETUP.md`; strategy docs 02, 03, 06, 07 added to the repo at the owner's request
### Fixed
- `StationBackend.send_goals`: goal wraps modulo 4096
### Changed
- `run/station.yaml`: `resolution: auto`, `resize_target: 0` (not applied until Station restart)
