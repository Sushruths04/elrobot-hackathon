# NormaCore hackathon - live session notes (HackLab Berlin 21-22 Nov 2026)
Work from this folder. bin/, vendor/, .venv are symlinks to ../hacklab-normacore-kit.

## Bring-up checklist
1. Plug in arm USB(s) + cameras. Check: ls /dev/ttyACM* /dev/ttyUSB* /dev/video*
2. Linux serial perms: sudo usermod -a -G dialout $USER  (re-login if newly added)
3. cd run && ../bin/station --web --tcp --config station.yaml   (UI: http://localhost:8889)
4. .venv/bin/python scripts/smoke_station.py
5. Calibrate arm in web UI, then close the control panel (single writer on the bus).

## Log (append below)
- 12:50 Station up. ACM1 = bus 5B8E114691 = FOLLOWER (12.4 V). ACM0 = bus 5B79016336 = LEADER (7.5 V). Both 8 motors found.
  Neither calibrated yet (ranges ~0 wide) -> backend skipped motor 8 -> "only saw 1..7". Not a bug: calibrate first.
  KD cameras stream at 320x240 MJPG (Station picks first format) - need higher res later. Laptop webcam errors in log: ignore.
- 13:0x User confirmed: LEADER = 5B8E114691 (12.4 V, ttyACM1), FOLLOWER = 5B79016336 (7.5 V, ttyACM0). run/buses.json updated. Follower runs at 7.5 V: check torque; BOM expects 12 V on the follower.
