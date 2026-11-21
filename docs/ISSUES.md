# Issues (IDs start at 101 to avoid clashing with the teammate repo's ISSUE-001..015)

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
