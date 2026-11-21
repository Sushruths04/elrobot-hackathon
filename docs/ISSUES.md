# Issues (IDs 101+ = Sushruth's rig; 001..015 = Bart's macOS session, listed at the bottom). Next ID: ISSUE-113

### ISSUE-101 - Station default torque caps may make arms weak / trip overload
- **Status:** open (verified registers, effect suspected)
- **Evidence:** all 16 motors: max torque 500/1000, torque limit 1000, P16 I0 D0, protection current 500, overload torque 25, protective torque 20. Leader motor 4 showed status `0x20` (Overload) once.
- **Cause:** Station writes these defaults at freeze (`drivers/st3215/src/presets.rs`). Not changed by us.
- **Next:** decide with a human before writing EEPROM.

### ISSUE-102 - Two opposite mirrors can run simultaneously
- **Status:** known gotcha (verified in log and `motors-mirroring/src/inference/mod.rs`)
- **Symptom:** starting leader->follower then follower->leader leaves both active; all 16 motors torque on and arms fight.
- **Fix:** set both cards to (Self-controlled), then start one direction only.

### ISSUE-103 - Leader calibration short on joints 5 and 7
- **Status:** open
- **Evidence:** saved spans vs URDF-expected: j5 1758/3826 (46%), j7 2222/3866 (57%), j4 84%, j8 81%. Follower matches URDF within ~1% on joints 2-8 (joint 1 1218/2021).
- **Cause:** incomplete manual sweep (suspected). Redo those joints; opening the calibration page resets the whole bus.

### ISSUE-104 - Follower motor 6 temperature spikes
- **Status:** investigating
- **Evidence:** single-sample reads of 66 C and 86 C, back to 33 C three seconds later, three times in ~5 min. Not physically plausible.
- **Cause:** suspected bad bus read or loose cable on motor 6. Unverified; check by touch.

### ISSUE-105 - Station camera frames 299x224
- **Status:** open. `usb-video.resize_target: 224` in the original config; `run/station.yaml` now sets `resolution: auto`, `resize_target: 0` (needs Station restart, not yet applied).

### ISSUE-106 - Cameras drop off USB / Station cannot open them (-3)
- **Status:** open (blocker)
- **Evidence:** one KD camera (`2c7f:4a60`) of two is missing; the other re-enumerates with new device numbers (11, 13, 26, 27, 30) and is then not writable (`Failed to open device ... -3`). At 14:10 `lsusb` showed 0 KD cameras. Camera 0 when it did stream: brightness 13-16/255 (almost black); its queue grew, the other queue stayed flat.
- **Cause:** suspected loose/damaged cable or overloaded shared hub (unverified). Permission part is certain (device node root-owned).
- **Next:** direct laptop ports, one camera at a time; udev rule `SUBSYSTEM=="usb", ATTR{idVendor}=="2c7f", ATTR{idProduct}=="4a60", MODE="0666"`.

### ISSUE-107 - Leader/follower motion may be opposite on some joints (unverified)
- **Status:** open, needs a human to look
- **Detail:** Station mirroring maps percent -> percent, so a servo mounted reversed gives opposite physical motion. Calibration records only min/max ticks, not direction. URDF prediction when ticks increase: tip goes down for joints 2, 3, 4 and 6; joint 1 yaws +. Gripper direction not derivable.
- **Next:** move each joint slowly and record same/opposite (incl. trigger vs gripper); then custom teleop with invert flags.

### ISSUE-108 - Follower joint 2 lags the leader while moving
- **Status:** investigating
- **Evidence:** 40 s sample: leader j2 54-62% (14 deg), follower 65-82% (31 deg), mean lag 18 points; no faults, current 4 mA, no protection-block lines in that window. Gap closed to ~1 point at rest.
- **Cause:** unknown (delay/speed vs start offset). Re-test with slow single-joint moves.

### ISSUE-113 - Camera pictures washed out / black-and-white (RESOLVED 2026-11-21 15:35)
- **Status:** resolved
- **Symptom:** both KD cameras gave ~242/255 mean brightness, 86% pure-white pixels, one nearly grayscale, not changing over 24 s.
- **Evidence:** camera controls read over USB were normal (auto exposure mode 8, brightness 0, gain 0, contrast 40, saturation 64). Station was opening MJPG **320x240** (its smallest mode). The `resolution:` key I had put in `run/station.yaml` is NOT understood by this Station build (beta.10) and was silently ignored.
- **Fix:** `usb-video: formats: [ {format: "640x480@30fps,mjpeg"} ]` plus `resize_target: 0`, then restart Station. Result: 640x480, mean brightness 124/255, correct colours.
- **Also seen:** one `Failed to get stream handle: -51` on the first open attempt, then fine.

### ISSUE-114 - Leader joint 4 range narrow / odd at one location (parked 2026-11-21 17:25)
- **Status:** open, parked by the owner to move on to camera calibration
- **Evidence:** leader j4 saved range [1404,2692] = 1288 ticks vs follower [884,3212] = 2328 ticks, so Station's percent mirroring moves the follower j4 ~1.8x the leader. Automatic calibration of the leader gives j4 ~1000-1290 (994, 1000, 1002, 1023, 1064, 1288 over several runs); a manual sweep reached 1918. The owner reports the leader behaving oddly at one position on that joint.
- **Cause:** suspected weak torque in the automatic sweep (7.5 V, current limit 50) and/or a mechanical issue on the leader's elbow. Unverified.
- **Also fixed:** follower j4 tripped Station's current protection (108/122 vs limit 100) and stuck; `motor-current-thresholds: {4: 250}` added to `run/station.yaml`.
- **Workarounds if needed:** manual leader sweep including j4 to both real stops; or `scripts/teleop.py` (1:1 tick mapping, needs a fresh `--set-rest` after any leader recalibration).
- **Note:** the leader's saved ranges changed between 17:06:58 and 17:15:04 (another calibration run, not by the agent).

### ISSUE-109 - Calibration does not follow the arms to another laptop
- **Status:** workaround (follower recalibrated on Bart's Mac 15:40; leader pending)
- **Found:** 2026-11-21 15:55 by Claude
- **Where:** station (both buses)
- **Symptom:** arms moved from Sushruth's laptop (both frozen) to Bart's Mac: `verify_station.py` shows all 16 motors `NOT SAVED`, ranges are partial leftovers (e.g. follower m2 [2142,2142], m4 [497,3284] = 120% of reference).
- **Cause (known):** min/max/frozen live in Station's `st3215/meta` under the local `station_data/`, not in the servos (docs/SETUP.md). EEPROM offsets from the old calibration are still in the motors, so seen ranges mix old offsets with new tracking.
- **Fix / workaround:** recalibrate on whichever laptop runs Station (reset -> sweep -> save, `tools/calibrate.py`). Keep one Station laptop for the rest of the hackathon, or copy its `station_data/` along with the arms (untested).


### ISSUE-110 - `verify_station.py` added the servo offset twice on saved motors
- **Status:** resolved (2026-11-21 15:45, Claude)
- **Found:** 2026-11-21 15:43 by Claude
- **Where:** `tools/verify_station.py` (also feeds `--move-test`)
- **Symptom:** right after the follower auto-cal, 5 of 8 motors showed `! pos OUTSIDE range` (e.g. m8 pos=4021, range [1243,2853]).
- **Evidence:** raw Present Position m2=1066 (= 1% of [1042,3054]) and m4=3096 (= 95% of [885,3211]) match the last auto-cal steps "Move to 1%" / "Move to 95%". raw + offset did not.
- **Cause (known):** Station adds the EEPROM offset only while collecting an UNSAVED range (`port_meta.rs` ~l.410); once saved, ranges are in Present-Position space (the servo applies the offset itself). The 12:25 "fix" generalised that wrongly.
- **Impact:** `--move-test` would have computed its goal from the wrong position (up to ~1900 steps off). It was never run.
- **Fix:** frozen motors use raw Present Position; unsaved motors keep raw + offset.


### ISSUE-111 - Leader auto-calibration stops early on joints 3, 4, 5 (false stalls)
- **Status:** open
- **Found:** 2026-11-21 15:53 by Claude (web-UI Auto Calibrate on Bart's Mac, completed "successfully" 14:52:00Z)
- **Where:** leader `5B79016336`
- **Evidence:** saved spans m3 1526, m4 996, m5 **436** (vs follower 2322 / 2326 / 3836; Sushruth's manual leader cal 2274 / 1918 / 1758). Log: `Motor 3 - Stalled at position 4002` in the same second it started; `Motor 5 - Stalled at position 3750` after ~1 s, then `Motor 5 in error state, recovering`. Auto-cal sets current limits m5 = 15, m7 = 15, m6 = 20, m8 = 10.
- **Cause (suspected):** the 7.4 V low-gear leader servos with the handle hit the low stall-current threshold from load/gravity before the real hard stop. Unverified. Related: ISSUE-003, ISSUE-103.
- **Impact:** with mirroring, leader m5's 38 deg range maps onto the follower's full m5 range (~9x gain) -> jerky / unexpected follower wrist motion. Don't mirror until fixed.
- **Fix / workaround:** calibrate the leader by hand (`tools/calibrate.py reset` + `watch` with the `<<< MOVING` marker, then `save`).


### ISSUE-112 - Leader bus drops / shifts replies (timeouts, "Motor ID mismatch")
- **Status:** open
- **Found:** 2026-11-21 16:05 by Claude
- **Where:** leader bus `5B79016336` (CH343 board -> USB hub -> WCH adapter on Bart's Mac)
- **Symptom:** a ResetCalibration at 15:05:13Z finished on only 3 of 8 motors: `Failed to reset calibration for motor 1: Timeout`, `motor 3: Motor ID mismatch: expected 3, got 1`, `motor 5 ... expected 5, got 3`, `motor 6 ... expected 6, got 5`, `motor 8: Timeout`. Station then keeps the bus frozen (meta reset needs a fully successful command), leaving a half-reset calibration.
- **Evidence:** 20 such failures, all in that one command; the next 3 resets were clean (all offsets 0, NOT SAVED). Leader latency in `verify_station.py` ~14 ms vs follower ~1 ms.
- **Cause (suspected):** after one timeout, replies arrive one motor late (each mismatch reports the previous ID). Possibly a marginal cable / hub / 7.4 V supply. Unverified. **Suspected to also cause the leader's false auto-cal stalls (ISSUE-111, ISSUE-003).**
- **Fix / workaround:** after any reset, check every offset is 0 and the bus shows NOT SAVED; resend the reset if not (it is idempotent). Try the leader board directly on a laptop port.

---

## ISSUE-001..015 — from Bart's macOS repo (bpachniewsk1/normacore-hackathon), merged 2026-11-21
Original IDs kept (no clash with 101+). Some overlap with 101+: 002 ~ 106 (cameras), 008 is superseded by the follower auto-calibration in this repo.

| ID | Title | Status |
|---|---|---|
| 001 | Servo boards didn't enumerate on USB | resolved |
| 002 | Camera `386869c0` sends black frames | open |
| 003 | Leader auto-calibration moved erratically | open |
| 004 | Leader saved calibration inconsistent with servo offsets | resolved |
| 005 | Leader motor 6 error status `0x08` (latched) | resolved |
| 006 | Leader servos stopped responding after power cycle | resolved |
| 007 | Torque went off after `estop.py` HOLD on leader | investigating |
| 008 | Follower never calibrated | open |
| 009 | Station has no dedicated e-stop | won't fix (workaround) |
| 010 | Opening web-UI calibration page silently resets the bus | known gotcha |
| 011 | Speed caps are hard-coded in Station (mirroring 3300, auto-cal 365) | workaround |
| 012 | All USB devices share one bus-powered hub (~1.3 A) | risk |
| 013 | Stored camera frames only 299×224 | open |
| 014 | No documented start pose for ElRobot auto-calibration | open (ask staff) |
| 015 | Our `repo/norma-core` copy has no `.git` | known |

---

### ISSUE-001 — Servo boards didn't enumerate on USB
- **Status:** resolved
- **Found:** 2026-11-21 ~11:50 by Bart
- **Where:** both servo boards
- **Symptom:** no `/dev/cu.usbmodem*`, Station log `joints=0`.
- **Cause:** suspected cable/port/power (not confirmed which).
- **Fix:** after re-cabling, both boards show as `1a86:55d3` (CH343) → `/dev/cu.usbmodem5B790163361` (leader), `/dev/cu.usbmodem5B8E1146911` (follower).

### ISSUE-002 — Camera `386869c0` sends black frames
- **Status:** open
- **Found:** 2026-11-21 12:10 by Claude
- **Where:** USB camera, Station queue `usbvideo/386869c07ab99f1d5b103f3b6a381f2f`
- **Symptom:** streams 30 fps but every JPEG is a constant 2489 B, solid black (`snapshots/cam_386869c0.jpg`). Camera `191fd589` works.
- **Cause:** unknown — lens cap / covered / faulty sensor suspected.
- **Fix / workaround:** check physically; `verify_station.py` warns on black frames.

### ISSUE-003 — Leader auto-calibration moved erratically
- **Status:** open
- **Found:** 2026-11-21 ~12:05 by team
- **Where:** leader `5B79016336`, web-UI Auto Calibrate
- **Symptom:** "moving weird and uncontrolled". Station restarted afterwards, so the progress state was lost.
- **Cause (suspected, from code reading):** (a) the single ElRobot routine (`drivers/st3215/src/auto_calibrate/elrobot.rs`) drives into hard stops and releases torque on some joints between steps by design; (b) the leader is a different build (handle + trigger, no gripper) but gets the same fixed shift offsets; (c) the leader's low-ratio 7.4 V servos are back-drivable, so gravity confuses stall detection (velocity < 10) at the reduced torque limit 300.
- **Fix / workaround:** manual calibration (Reset → move each joint to both stops → Save). See SETUP.md. Blocked by ISSUE-005/006.

### ISSUE-004 — Leader saved calibration inconsistent with servo offsets
- **Status:** resolved
- **Found:** 2026-11-21 12:15 by Claude
- **Symptom:** ranges saved (centred on 2048) but servo offsets = 0 → 6/8 motors outside their range. Motors 3 and 5 had torque on with goals far from position (3398→244, 2366→157), straining.
- **Fix:** reset via `tools/reset_calibration.py` at 12:26 → NOT SAVED.

### ISSUE-005 — Leader motor 6 error status `0x08` (latched)
- **Status:** resolved (2026-11-21 15:55, Claude: `verify_station.py` shows no error bits on leader m6 and its range is tracking [1636,3813]; cleared by the later power cycles, exact moment unknown)
- **Found:** 2026-11-21 12:35 by Claude (log: `Motor 6 has errors: [Range], skip calibration data`, ~47×/s)
- **Where:** leader `5B79016336`, motor 6
- **Evidence:** status byte (0x41) = `0x08` while torque = 0, current = 0, load = 0, temp 30 °C. Its goal was 2905 while other leader motors showed 0 after reset. Follower motor 6 is clean.
- **Cause (suspected):** Station labels bit 3 "Range"; on Feetech STS servos bit 3 is usually over-current → probably latched from earlier strain. Unverified.
- **Impact:** Station skips motor 6 data → range stuck `[0,0]` → **leader can't be calibrated** (auto or manual).
- **Fix / workaround:** power-cycle the servo supply; if it persists at rest, check the joint mechanically / swap servo (ask staff).

### ISSUE-006 — Leader servos stopped responding after power cycle
- **Status:** resolved (2026-11-21 15:55, Claude: all 8 leader motors answer at 7.5–7.7 V on Bart's Mac)
- **Found:** 2026-11-21 12:40 by Claude
- **Symptom:** leader board opens (`Successfully opened ST3215 port`) but 0 motors answer, even after 2 Station restarts. No error lines. Follower re-detected fine after its power cycle.
- **Cause (suspected):** leader servo power not reconnected (PSU, barrel jack, board→servo cable). Unverified — leader then disconnected.
- **Fix / workaround:** check 7.4 V PSU LED, barrel jack, servo LED blink on power-up, 3-pin cable. Station re-detects motors automatically, no restart needed.

### ISSUE-007 — Torque went off after `estop.py` HOLD on leader
- **Status:** investigating
- **Found:** 2026-11-21 12:22 by Claude
- **Symptom:** HOLD wrote goal = current position correctly, but ~1 s later all leader motors had torque 0 and the arm sagged (shoulder ~83°, wrist ~90°).
- **Evidence:** HOLD sends no torque write. `current-threshold` only applies inside mirroring (`motors-mirroring/src/inference/mirror.rs`).
- **Cause:** unknown — suspects: Station's stop-mirror handling, or servo overload protection after the earlier strain.
- **Fix / workaround:** treat HOLD as possibly dropping the arm until understood; power plug is the real e-stop. Re-test on a healthy, calibrated arm.

### ISSUE-008 — Follower never calibrated
- **Status:** open
- **Where:** follower `5B8E114691` (12 V)
- **Symptom:** ranges are only "seen so far" (e.g. `[114,114]`), `NOT SAVED`; Station inference skips every frame (`invalid_range`).
- **Fix:** calibrate (auto or manual). Support the arm — reset turns torque off.

### ISSUE-009 — Station has no dedicated e-stop
- **Status:** won't fix (workaround)
- **Detail:** only "Stop Calibration" (torque off during auto-cal) and mirroring current threshold. Workaround: `tools/estop.py` + physical power plug within reach.

### ISSUE-010 — Opening web-UI calibration page silently resets the bus
- **Status:** known gotcha
- **Evidence:** `station-viewer/src/pages/St3215BusCalibrationPage.tsx` sends stop-mirror + `resetCalibration` in a `useEffect` on page open → torque off, calibration cleared.
- **Workaround:** only open it when you intend to (re)calibrate; support the arm first.

### ISSUE-011 — Speed caps hard-coded in Station
- **Status:** workaround
- **Detail:** web-UI mirroring max_speed 3300 / accel 100 (`motors-mirroring/src/config.rs`, not in station.yaml); auto-cal speed 365 / accel 50 (`st3215/src/presets.rs`). Writing Goal Speed ourselves is overwritten by these.
- **Workaround:** use Python teleop example with `--max-speed` (our patch); own tools write speed before moves. Changing Station requires a Rust rebuild.

### ISSUE-012 — All USB devices on one bus-powered hub
- **Status:** risk
- **Detail:** Mac USB-C → WCH adapter (1a86:8095) → Genesys USB 2.0 hub → 2 servo boards (138 mA each) + 2 cameras (500 mA each) ≈ 1.3 A, 480 Mb/s shared. Works now; if devices drop out under load, power the hub or split cameras to another port.

### ISSUE-013 — Stored camera frames only 299×224
- **Status:** open
- **Detail:** `usb-video.resize_target: 224` in station.yaml → ~2.7 mm/px over 60 cm. Raise for perception if needed (more disk + CPU).

### ISSUE-014 — No documented start pose for ElRobot auto-calibration
- **Status:** open (ask NormaCore staff)
- **Detail:** code sweeps each joint ≤ ~360° from wherever it is until stall, so pose only matters for not hitting table/frame first. Our recommendation (unverified): folded compact rest pose, base centred, wrist straight, gripper half open.

### ISSUE-015 — Our `repo/norma-core` copy has no `.git`
- **Status:** known
- **Detail:** session move didn't copy `.git`. Fine for running examples; to update, re-clone (see SETUP.md). Our only upstream edit is the teleop `--max-speed` patch (CHANGELOG).
