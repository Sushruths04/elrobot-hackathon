# Changelog (newest first)

## [Unreleased]
### Added
- 2026-11-21 - `scripts/twin_view.py`: renders both real arms from live joint data (`run/twin.png`), using the Station viewer's mapping
- 2026-11-21 - `scripts/identify_arm.py`, `set_home.py`, `dump_buses.py`, `monitor_calib.py`, `backup_calibration.py`, `mirror_state.py`; `./norma` shortcuts
- 2026-11-21 - `src/normakit/busread.py` read-only bus snapshot; `StationBackend` reads follower bus from `run/buses.json`
- 2026-11-21 - MCP server `elrobot-real` entry in `.mcp.json` (station backend, untested for motion)
- 2026-11-21 - `calibration/`: per-bus calibration files (min, max, midpoint, span, frozen)
### Fixed
- 2026-11-21 - `StationBackend`: wrapped ranges (max < min) and positions just outside the range no longer break normalisation
- 2026-11-21 - `monitor_calib.py`: no longer crashes on short motor reads
### Hardware
- 2026-11-21 - Follower `5B8E114691` auto-calibrated and frozen. Leader `5B79016336` calibrated manually; joints 5 and 7 short (see ISSUE-103)
- 2026-11-21 - Roles confirmed by movement test: leader `5B79016336` (/dev/ttyACM0), follower `5B8E114691` (/dev/ttyACM1)
