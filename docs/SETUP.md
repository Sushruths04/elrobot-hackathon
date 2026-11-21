# Setup facts (verified on this rig unless marked)

- **Buses:** leader `5B79016336` = /dev/ttyACM0 (USB `1a86:55d3`), 7.5 V; follower `5B8E114691` = /dev/ttyACM1, 12 V. Both 8 ST3215 motors, ids 1-8. `run/buses.json` is the source of truth for our scripts.
- **Linux perms:** serial ports need `dialout`; Station opens cameras with libusb, so the USB device node must be writable (`sudo chmod 666 /dev/bus/usb/001/NNN`, or a udev rule for `2c7f:4a60`). Error in the log when not: `Failed to open device for camera ...: -3`.
- **Start Station:** `./norma start` (runs `bin/station --web --tcp --config run/station.yaml` in `run/`). Web UI http://localhost:8889 - use Firefox (WebGL is off in the Chrome here).
- **Calibration storage:** motors keep only the centring offset (EEPROM). Min/max ranges + frozen flag live in Station's `st3215/meta` queue under `run/station_data/`, and are restored from it at startup unless the last entry is a reset (read in `drivers/st3215/src/port_meta.rs`; restart NOT yet tested). Backups: `calibration/*.json` via `./norma backup`.
- **Opening a bus's calibration page resets that bus** and stops mirroring (torque off).
- **Mirroring** (`motors-mirroring`) copies leader % of range to follower %; it cannot invert a joint. A second start does not cancel an opposite one (two mirrors can fight).
- **Station torque defaults:** all motors max torque 500/1000, torque limit 1000, P16 I0 D0, protection current 500, overload torque 25 (ISSUE-101).
- **3D viewer mapping** (`station-viewer .../elrobot/config.ts`): joints 1-7 linear over URDF limits; gripper (motor 8) flipped.
- **Official board:** `calibration/official/opencv_pattern_9x6_25mm_A4_*` = OpenCV's `doc/pattern.png`, 9x6 inner corners, 25.00 mm squares (measured), A4 landscape.
- **Read-only tools:** `./norma status|identify|backup|smoke`, `scripts/dump_buses.py`, `monitor_calib.py`, `twin_view.py`, `snap_cams.py`. Moving tools: `first_motion.py` (needs `--yes`).
- **Camera format (Station beta.10):** set in `run/station.yaml` as `usb-video: formats: [ {format: "640x480@30fps,mjpeg"} ]` and `resize_target: 0`; restart Station. The `resolution:` key is ignored in this build. Default (empty list) opens MJPG 320x240 which gave washed-out pictures. Camera controls (exposure etc.) can be read/set with `scripts/uvc_ctl.py` only while Station is NOT holding the camera.
- **USB hub:** the shared hub dropped repeatedly (kernel `error -71`, 15 disconnects on one port in 3 h). Cameras plugged directly into laptop ports are stable so far. udev rule `/etc/udev/rules.d/99-norma.rules` makes the serial boards and cameras world-writable (applies on re-plug).
- **Camera calibration needs no arms and no hub:** `scripts/calibrate_camera.py` only reads frames from Station (`require_motors=False`).

## macOS laptop (Bart) — merged from bpachniewsk1/normacore-hackathon
- Serial ports appear as `/dev/cu.usbmodem<bus-serial>1` (leader `/dev/cu.usbmodem5B790163361`, follower `/dev/cu.usbmodem5B8E1146911`), not `/dev/ttyACM*`. `./norma` and `run/buses.json` assume Linux.
- Station binary (macOS arm64 zip) lives in `~/Downloads` by default: `./restart_station.sh` (background, log `logs/station-latest.log`), `./restart_station.sh stop`, `./start_station.sh` (foreground). Override with `STATION_DIR=/path`.
- `tools/` finds the upstream Python client at `vendor/norma-core-src`, else `repo/norma-core`, else `$NORMA_CORE`.
- Read-only: `python3 tools/verify_station.py [--watch]` (USB, Station, per-motor pos=raw+offset / torque / current / temp / voltage / range / saved flag / error status, cameras + black-frame check).
- Moving / torque: `tools/estop.py` (HOLD = stop mirroring + goal = current; `--limp` = torque off), `tools/reset_calibration.py --bus SN` (= web-UI Reset, torque off), `tools/speed_limit.py` (Goal Speed 0x2E + Acc 0x29; never applied on hardware).
- `patches/teleop-max-speed.patch`: adds `--max-speed`/`--max-accel` to upstream `st3215-remote-teleop-py` (apply with `git -C vendor/norma-core-src apply ../../patches/teleop-max-speed.patch`; untested on hardware).
- Auto-calibration (`elrobot.rs`): only for exactly 8 motors IDs 1–8; stall-sweeps each joint up to ~360° from the current pose at fixed speed 365 / accel 50 / torque limit 300 — clear the workspace. Mirroring max speed is hard-coded (3300) in `motors-mirroring/src/config.rs`.
- Calibration over TCP (`tools/calibrate.py`): `st3215.Command` has `reset_calibration` (14), `freeze_calibration` (15, freeze=True without arcs = web-UI Save), `auto_calibrate` (17), `stop_auto_calibrate` (18); progress in `BusState.auto_calibration` (`protobufs/drivers/st3215/st3215.proto`). Moving the arms to another laptop loses the saved ranges (ISSUE-109).
- **Table markers** (`calibration/markers.json`, sheet `aruco_markers_A4.pdf`): DICT_4X4_50, 40 mm. ID 0 corner front-left, 1 front-right, 2 back-right, 3 back-left, 4 zone A, 5 zone B, **6 zone C = DROP**, 7 bowl. Live check: `.venv/bin/python tools/detect_live.py --classes cup,bottle --markers`.
