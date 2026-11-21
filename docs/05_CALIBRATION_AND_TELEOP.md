# Calibration + teleoperation (from the Station source)

Sources: `software/drivers/st3215/src/auto_calibrate/{mod,elrobot,calibrator}.rs`,
`software/station/clients/station-viewer/src/pages/St3215BusCalibrationPage.tsx`,
`software/drivers/motors-mirroring/`, `software/station/examples/st3215-remote-teleop-py/`.
Not yet run against hardware: everything below is from reading the code.

## How NormaCore fits together
- `station` (Rust) opens each serial bus, pings motor IDs 1..N, and publishes state to NormFS queues
  (`st3215/inference`, `commands`, `usbvideo/*`, ...). Web UI on :8889, TCP/NormFS on :8888.
- Each bus is identified by the adapter's serial number. A follower and a leader are two separate buses.
- Station detects the robot type by motor IDs: 6 motors (1-6) = SO-101, 8 motors (1-8) = ElRobot.
  Motor IDs must already be unique and correct (1..8) or auto-calibration will not start.

## Calibration (what it is)
- It finds each joint's physical min/max by driving it into its end stops and detecting a stall.
  The result is saved as `range_min`/`range_max` per motor, and the offset (midpoint) is adjusted so the
  range does not wrap at the 0/4095 register boundary. All control in Station is in this calibrated range.
- ElRobot sequence (`elrobot.rs`): 32 steps. Prepare all 8 motors, set low protection-current limits
  (motors 1-4: 50, 5: 15, 6: 20, 7: 15, 8 gripper: 10), then sweep motor 8 (gripper), 1 (base), 6, then
  2/3/4 together (torque held on motor 4 so the arm does not fall), then 7, 5, and finish at a rest pose
  (3 at 50%, 4 at 95%, 2 at 1%). Takes a few minutes.
- The web page `/st3215-bus-calibration` has: Auto Calibrate button, a reset/"freeze" control
  (frozen = range locked), and warnings for supply voltage below 7.0 V and for motors whose range is
  under 100 ticks (not calibrated yet).

### Before you start it
1. 12 V supply on, adapter powered, all 8 motors visible (the page shows each motor's voltage).
2. Clear the workspace: the arm sweeps into every limit. Hold or support the arm when it starts.
3. Close any other client writing to the bus (one writer at a time).
4. Open `http://localhost:8889`, select the follower bus, run Auto Calibrate, wait for "done".
5. Do the same for the leader bus if it is also 8 motors (SO-101 style 6-motor leaders use the so101 routine).
   If the auto routine does not apply to your leader, calibrate the range manually from the same page.

### After it: check
`scripts/smoke_station.py` must show `range_min < range_max` for all 8 motors. Our code maps the
calibrated range linearly onto the URDF limits (assumption, verify with a few poses).

## Teleoperation (leader -> follower)
Two ways, both already in the repo:

**A. Station mirroring driver (`motors-mirroring`)**
- Runs inside Station (20 ms loop). Modes are set per bus through the `motors_mirroring/modes` queue
  and state shows up on `inference/mirroring`. The web UI exposes it on the bus page.
- Safety built in: per-motor current threshold (default 100 raw): if the follower's current exceeds it,
  the goal is set to the current position, so it will not stall-burn on contact. Also deadband 20,
  speed 300..3300, accel 5..100.

**B. Python example `software/station/examples/st3215-remote-teleop-py`**
```bash
cd vendor/norma-core-src/software/station/examples/st3215-remote-teleop-py
uv run python main.py --leader-server localhost --leader-bus <LEADER_SERIAL> \
                      --follower-server localhost --follower-bus <FOLLOWER_SERIAL>
```
Bus serials come from the Station UI or `smoke_station.py` (`bus:` line). Logs freq/age per bus every second.
On Ctrl+C it disables follower torque (arm goes limp): support it first.

## Recording demos (for the skill library / SmolVLA fallback)
Station keeps all queues, including cameras and `st3215/inference`, so recording a demo means
teleoperating while logging those queues. Next step once the arm is live: add `scripts/record_demo.py`
that tags each episode with a text label (our skill library idea).

## Order of work once the boards show up
1. `lsusb` shows two adapters; `dialout` group set.
2. Station up, both buses visible, all motor IDs found.
3. Auto-calibrate follower, then leader.
4. Teleop test (B above, small movements first), then verify camera frames at full resolution.
5. `smoke_station.py`, then `calibrate_table`, then the Claude loop.

## Home position (checked in the source, 2026-11-21)
- Station has NO home/rest-pose feature (no hits for home/rest/park in software/). The only "rest pose" is the
  last calibration step for ElRobot: motor 3 at 50%, motor 4 at 95%, motor 2 at 1%.
- Teleop maps by percentage: leader joint % of its calibrated range -> same % of the follower's range
  (`mirror.py::normalize_position`), speed proportional to remaining distance. So a home pose is just a list of
  percentages and ONE list serves both arms.
- DANGER at start: `main.py` enables follower torque first, so the follower snaps toward the leader's pose at once.
  Start with both arms in the same pose (home). On Ctrl+C torque is disabled and the follower goes limp: park it first.
- Our tools: `scripts/set_home.py <leader|follower>` records the current hand-posed pose to `run/home.json`;
  `scripts/identify_arm.py` finds which bus is the one you hold.
