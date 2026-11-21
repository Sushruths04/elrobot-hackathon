# Real-Robot Execution Runbook (`ElRobot` Follower `5B8E114691`)

Execute **one numbered step at a time** at the robot table. Paste back the terminal output (and saved image path) after each gate before proceeding to the next.

**Pre-flight safety invariants (apply to every step):**
- Physical 12 V servo power barrel plug is in your hand (the real e-stop).
- Station Web UI control/mirroring panel is **closed** or both buses set to `(Self-controlled)` (one writer on the bus).
- **Never** open a bus's Calibration page in the Web UI (opening it silently resets calibration).

---

## Step 1 — Check Station, support the arm, and free the jaw from the cup
*(At handover, the follower's jaw was resting inside the black cup opening with torque ON.)*

1. Check Station & calibration backup:
   ```bash
   ./norma status
   .venv/bin/python scripts/servo_diag_and_fix.py status
   ```
   - **Expected output:** Both `cam` streams (`front`, `wrist`) discovered; all 8 follower motors show `span >= 1600` (`m1~1978, m2~2034, m3~2324, m4~2328, m5~3828, m6~2052, m7~3866, m8~1610`).
   - **STOP condition:** Any motor missing, `span < 100`, or `Station not reachable`. Run `./norma start` (and never open Web UI calibration).

2. **Hold the follower forearm/wrist firmly with one hand** (torque-off will let the arm drop under gravity), then run:
   ```bash
   .venv/bin/python scripts/servo_diag_and_fix.py limp --yes
   ```
   Gently lift the follower jaw clear of the black cup into open air (`~18-22 cm` above the table, nothing touching the jaws) and move the black cup `~15 cm` aside for the open-air tests.
   - **Expected output:** `TORQUE OFF on all 8 follower motors.`
   - **STOP condition:** Jaws snagged on cup or cable strained—unhook gently by hand while torque is off.

---

## Step 2 — Open-air torque test of elbow joints 3 and 4 (`+-60` ticks)

With the follower held/resting safely in open air (away from table, cup, and bottle) and your hand on the 12 V plug:

```bash
.venv/bin/python scripts/servo_diag_and_fix.py test-elbow --delta 60 --yes
```

- **Expected output:** Both `joint 3` and `joint 4` move `~ -50..-60` ticks on `up`, `+50..+60` ticks on `down`, and return within `<= 8` ticks of `start`, with `cur < 500 mA`.
- **If `joint 3` or `joint 4` moves `< 35` ticks in open air (Root cause verified in MuJoCo twin: Station's `presets.rs` writes `P=16, I=0, D=0, MaxTorque=500, OverloadTorque=25%`, which caps sustained torque to `20% = 0.59 N*m < 0.68 N*m` elbow gravity+stiction):**
  1. Support the arm by hand (unlocking EEPROM briefly turns torque off) and apply the simulation-validated ST3215 preset (`P=36, I=12, D=20, MaxTorque=850, TorqueLimit=850, OverloadTorque=85%`):
     ```bash
     .venv/bin/python scripts/servo_diag_and_fix.py apply-torque-fix --yes
     ```
  2. Also ensure `run/station.yaml` has raised mirroring current thresholds if mirroring is ever used (`motor-current-thresholds: {2: 350, 3: 350, 4: 350}`).
  3. Re-run the open-air test:
     ```bash
     .venv/bin/python scripts/servo_diag_and_fix.py test-elbow --delta 60 --yes
     ```
- **STOP condition:** If `j3` or `j4` still moves `< 35` ticks in open air after `apply-torque-fix`, or `status != 0x00`, or `current > 700 mA`, **STOP**—suspect stripped spur gear or loose 3-pin cable on motor 3/4; inspect physically before proceeding.

---

## Step 3 — Follower joint-direction vs URDF check (ISSUE-107)

With the arm in open air (`~20 cm` high, every joint away from `2%..98%` stops), test small `+50`-tick moves and compare physical motion against the printed URDF delta and saved `run/dir_j*_before.jpg` / `run/dir_j*_plus.jpg`:

```bash
.venv/bin/python scripts/servo_diag_and_fix.py check-dir --joint 1 --delta 50 --yes
.venv/bin/python scripts/servo_diag_and_fix.py check-dir --joint 2 --delta 50 --yes
.venv/bin/python scripts/servo_diag_and_fix.py check-dir --joint 3 --delta 50 --yes
.venv/bin/python scripts/servo_diag_and_fix.py check-dir --joint 4 --delta 50 --yes
.venv/bin/python scripts/servo_diag_and_fix.py check-dir --joint 6 --delta 50 --yes
.venv/bin/python scripts/servo_diag_and_fix.py check-dir --joint 8 --delta 80 --yes
```

- **Expected output (URDF convention):**
  - `+delta` on `j1`: base yaws toward `-X` (arm moves **right** in the front camera image).
  - `+delta` on `j2, j3, j4, j6`: `dz < 0` (jaw tip moves **downward** toward the table).
  - `+delta` on `j8`: gripper jaws **open wider** (`norm` increases = open; `norm -> 0` = close).
- **STOP condition:** Any joint moves **opposite** to the URDF sign printed by `check-dir`. Paste the output immediately so we invert that joint in config + re-run the 100-trial sim benchmark before any pick attempt.

---

## Step 4 — Vision & calibration check + Plan-only dry run (no motion)

*(Note: Handover analysis proved `calibration/table_front.json` had a 180-deg corner swap because the 4 corners were touched while standing behind the robot. `calibration/table_front_180fix.json` fixes that with a `6.1 mm` corner residual and is used automatically by `grab_cup.py`.)*

Place the black cup near the middle of the workspace (`~22-28 cm` in front of the follower base) and keep the distractor bottle `>= 14 cm` clear. Run plan-only (zero motor commands):

```bash
.venv/bin/python scripts/grab_cup.py --vision blob
```
*(Or `--vision er2` once `GEMINI_API_KEY` or `~/.gemini_er_key` is set.)*

- **Expected output:**
  - `perception (...): pixel=(u,v) -> est cup x=... y=... m` with `x in [-0.15, 0.20]`, `y in [0.15, 0.38]`, matching the physical side of the robot (`+x` = robot's right = **left** in front camera image).
  - `FK path checks: lift min_z=... (ok), travel min_z=... (ok)`
  - `PLAN OK (vertical, max IK err < 0.5 cm, ori < 5 deg) - dry run only`
- **STOP condition:** `ABORT` on perception, workspace bounds, IK error, or FK path check, or `x` sign opposite to where the cup sits on the table.

---

## Step 5 — First `--go` at low speed (`--speed 200`), cup moved aside

Move the black cup **out of the jaws' path** right after perception? **No**—better: place a soft paper/foam target or place the cup, run `--speed 180` ONLY after moving the cup `10 cm` away as soon as `TORQUE ON - moving` prints, **OR** run with `--no-wrist-align --speed 180` over an empty spot marked by a small black paper square (`4x4 cm`) on the table so the first autonomous descent executes in open air with zero collision risk:

```bash
.venv/bin/python scripts/grab_cup.py --vision blob --no-wrist-align --speed 180 --go
```

- **Expected output:** Smooth slow lift -> hover -> 3-step descent to `z_tip ~ 6.4 cm` above the table -> gripper closes -> waits `2.0 s` -> lifts to `z_tip ~ 16.8 cm` and holds. All currents `< 700 mA`, no `TIMEOUT`.
- **STOP condition:** Pull 12 V plug immediately if arm moves toward table surface (`< 3 cm`), toward a person, or `TIMEOUT` / `HIGH CURRENT` triggers.

---

## Step 6 — Real cup grasp with wrist-camera alignment (`--speed 240`, then `--speed 400`)

Remove the paper marker, place the real hollow black cup upright on the table (`x in [-0.12, 0.16]`, `y in [0.18, 0.34]`), hand on 12 V plug:

```bash
.venv/bin/python scripts/grab_cup.py --vision blob --speed 240 --go
```
Once verified clean at `--speed 240`, release the cup safely (`teach_pick.py open --yes`), re-place the cup at a new unmeasured spot on the table, and run at normal speed (`--speed 400`, with `--vision er2` if Gemini key is loaded):

```bash
.venv/bin/python scripts/grab_cup.py --vision er2 --speed 400 --go
```

- **Expected output:**
  - `wrist-cam align #1` and `#2` refine `cup xy` from live wrist-camera view.
  - Back rim of the hollow cup is straddled cleanly (inner jaw inside cup mouth, outer jaw behind back rim), gripper closes and settles (`raw ~ 1300-1520`), waits `+2.0 s`, lifts `~9-10 cm` straight up, and holds for `3.0 s`.
- **STOP condition:** Jaws hit top of cup rim instead of straddling it, `HIGH CURRENT`, or `TIMEOUT`.

---

## Step 7 — Visual grasp verification (EBIM Rule #4)

While holding after Step 6, capture live frames from both cameras (never trust jaw gap or motor current alone):

```bash
.venv/bin/python scripts/snap_cams.py
```

- **Expected output:** Inspect the saved front and wrist JPEGs visually—confirm the black cup is off the table (`>= 5 cm` air gap under the base) and held upright in the jaws.
- **STOP condition:** Cup still on table or tilted `> 25 deg` in gripper.

---

## Step 8 — Full autonomous pick-and-place + log capture

Place a blue target mat/coaster (or zone marker) on the table `>= 15 cm` from the black cup, place the black cup at a fresh random spot on the table, and run the single-process autonomous pick-and-place loop:

```bash
.venv/bin/python scripts/grab_cup.py --vision er2 --speed 380 --place --go | tee run/real_pick_place.log
```
*(Or with Claude perception + verification: `ANTHROPIC_API_KEY=... .venv/bin/python scripts/autorun.py --backend station --table-calib calibration/table_front_180fix.json --vision claude --object "black cup" --target "blue tray" --i-am-at-the-estop`.)*

When finished and the arm has returned to its safe raised pose, support the arm by hand and power down torque cleanly:
```bash
.venv/bin/python scripts/servo_diag_and_fix.py limp --yes
```

- **Expected output:** `DONE (done_place) ... held=True placed=True`; cup sits upright on the target within `3 cm`; `run/real_pick_place.log` saved.
- **STOP condition:** Any `FAILED`, `TIMEOUT`, or `HIGH CURRENT` exit (`exit code != 0`)—paste `run/real_pick_place.log` and the latest camera frame before retrying.
