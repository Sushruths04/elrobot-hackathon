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
