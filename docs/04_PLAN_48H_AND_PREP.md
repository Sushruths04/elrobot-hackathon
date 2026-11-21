# Plan

## A. Prep before Saturday (do these now, in order)
1. `scripts/selftest_sim.py` green on your laptop (digital twin) - proves the stack, installs nothing on site.
2. **Rehearse on the sim with Claude**: `ANTHROPIC_API_KEY=... python -m normakit.agent "put the red block in the blue tray"`; also register `.mcp.json` in Claude Code and drive it conversationally. Fix prompt/tool wording until it succeeds 5/5.
3. Get a **second pair of eyes on the Station protocol**: read `docs/01_STATION_DEEP_DIVE.md`; `station_backend.py` is the only untested piece (hardware). Know where each field comes from.
4. Print: magenta sticker (~12 mm), ArUco markers (DICT_4X4) for later markerless calibration, a cheat-sheet of ST3215 registers.
5. Buy/bring: USB-C data cables, small powered hub, 2 UVC webcams if allowed, clamp, coloured blocks, a cup, a foam/felt object, blue-tac. Laptop with Linux/mac; `dialout` group set; Station binary pre-downloaded (`bin/`).
6. Pre-generate a **demo script** (5 tasks of rising difficulty) and **pitch outline** (below).
7. Test with a real arm if you can borrow any SO-101/ElRobot before Saturday (ask HackLab organisers / NormaCore on Discord; the repo is public and they answer).

## B. Hour-by-hour
| When | Goal | Done when |
|---|---|---|
| Sat 11:00-12:00 | Kits, power, `station --web --tcp`, `smoke_station.py` | motors + 2 camera frames printed |
| 12:00-13:00 | Station config: `resize_target: 0`; calibrate arm (web UI); fix bus serial/ids | all 8 `range_min<range_max` |
| 13:00-14:30 | Verify URDF map (norm->rad), table z, workspace; `calibrate_table` with sticker | mean err < 1.5 cm |
| 14:30-16:00 | `reach_cell/pick_cell/place_cell` on real objects, tune grasp current thresholds, gripper opening | 8/10 pick+place of a block |
| 16:00-18:00 | MCP in Claude Code **and** Codex; agent loop; first natural-language run | judge-style sentence works |
| 18:00-22:00 | Generalisation: unknown objects, 2 cameras, recovery, speed (plan-once) | 10-trial benchmark logged |
| 22:00-02:00 | UI "thought bubble" + voice + e-stop theatre; record backup video | video on disk |
| Sun 08:00-12:00 | Fix top-3 failures from the benchmark only. Freeze features at 12:00 | |
| 12:00-16:00 | Rehearse pitch x3 with live robot, hand over unseen objects | |
| 16:00-17:45 | Submit, final video, README, repo public if allowed | submitted before 17:45 |
| 18:00 | Judging; 19:00 pitch | |

**Fallback ladder** (decide at 16:00 Sat): (1) calibrated pixel reach works -> main; (2) grid-cell + manual offsets table; (3) LLM-driven `move_joints` with look-verify (slow but works); (4) teleop-recorded skills replay.

## C. Pitch (3 minutes)
1. *Problem (20 s)*: "LLMs can see but are terrible at precise motor control; giving them joint tools is slow and breaks."
2. *Insight (30 s)*: "Don't micromanage. LLM decides what/where; a calibrated decoder (URDF IK + self-calibrating camera) does how; motor current verifies."
3. *Live demo (90 s)*: judge speaks a task; robot does it; show thought bubble; bump the camera -> press "recalibrate" (60 s montage / pre-calibrated).
4. *Numbers (20 s)*: "N trials, X% success, median Y s, Z model calls per task."
5. *Why NormaCore (20 s)*: works with Claude Code *and* Codex over MCP/CLI; open-source; safety guard; next: fleet via NormFS.

## D. Risks and mitigations
| Risk | Mitigation |
|---|---|
| Station field names drift from my reading | `smoke_station.py` first; fallback = own 30-line client per protocol section |
| norm->URDF map off | measure 3 poses; fit per-joint offset/scale (add to `core.py`) |
| IK solution through table / self-collision | add forbidden boxes in `SafetyConfig`, test in twin before real |
| Servo overheating (30 kg servos, long idle torque) | back off, `temp_c` shown in state; cooldown pauses |
| Claude API latency/limits at a crowded venue | plan-once; cache; Haiku verifier; offline CV colour fallback |
| WiFi dies | tether phone; keep agent local; record backup video |
| Bus contention with web UI | close control panel; single writer |
