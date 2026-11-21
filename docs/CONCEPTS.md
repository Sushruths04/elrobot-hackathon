# Concepts, ideas & decisions

**Project idea: TBA — to be agreed by the team.** Add ideas below; don't delete others' ideas,
add pros/cons instead. When we decide, log it in **Decisions** at the bottom.

## Track brief
NormaCore track — *"AI-Powered Robot Control"*: integrate the NormaCore **Station API** into
Claude / Codex so the LLM can control the robot by looking through the cameras.
NormaCore = "the data layer for physical AI" (data collection, VLA models, fleet management) →
pitches that touch **data** are a strong fit.

## How we'll judge ideas
| Criterion | Question |
|---|---|
| Track fit | Does an LLM control the robot *through cameras* via Station? |
| NormaCore fit | Does it use / showcase their data layer (recording, datasets, SmolVLA)? |
| Feasible | Works with: 2 RGB cameras (no depth), 7+1 DoF arm, no IK API, raw servo writes, our time left? |
| Demo-able | A reliable 3-minute live demo + backup video? |
| Wow | Is there one memorable moment? |

## Candidate ideas

### A. Self-data-collecting robot (from pre-event planning)
Claude runs the follower autonomously: invents a task → executes it by looking through the
camera → judges success → resets the scene → logs the episode as labelled training data.
A MAPE-K supervisor keeps it healthy (detects camera drift, missing objects, low success rate →
pauses + notifies a human). Stretch: fine-tune SmolVLA on the collected data.
- **Pro:** perfect NormaCore data-layer fit; numbers-driven demo; runs overnight.
- **Con:** needs reliable pick-and-place first (IK + calibration + perception) — a lot for the time left.
- **Fallback tiers:** (1) language command → pick & place, (2) autonomous loop, (3) MAPE-K, (4) SmolVLA.

### B. (add yours)
- Pitch:
- Pro:
- Con:

### C. (add yours)

## Architecture concepts (apply to any idea)
- **LLM decides *what*, code decides *how*.** Claude picks targets / tasks / judges results;
  deterministic code does geometry, IK, limits, safety. The model never sees transform matrices.
- **Few coarse tools** (3–7) with readable errors; compact JSON in robot base frame, metres.
- **Agent loop:** while + tool calls + step limit (~15) + per-tool timeout (~30 s) + API budget cap;
  `call_model(messages, tools)` wrapper so Claude/Codex are swappable; JSONL trajectory log.
- **Perception without depth:** objects on a flat table + fixed camera → homography pixel → table (x, y).
- **Station is the recorder:** every sensor reading and command is stored; we only add episode
  boundaries + labels, then export Parquet (`repo/norma-core/docs/datasets/export-parquet/`).
- **Safety in code:** workspace box, min z, speed cap, pre-grasp approach, e-stop path.

## Open questions (for team / NormaCore staff)
- Which idea? How much time is left for build vs demo prep?
- Is unattended running allowed? Judging criteria + demo format?
- Can we write custom queues/annotations (episode labels) into Station?
- Can the Python client get full-resolution frames?

## Decisions
| Date | Decision | Why | Who |
|---|---|---|---|
| 2026-11-21 | No ROS 2 — Station API only | ROS is "planned" in Station, not supported | pre-event |
| 2026-11-21 | Camera self-calibration out of scope; calibrate once manually | too hard for the time | pre-event |
| 2026-11-21 | Agentic workflow: AGENTS.md protocol, every change in CHANGELOG | many agents/people in parallel | Bart |
| 2026-11-21 | Project idea: **TBA** | — | team |
| 2026-11-21 16:25 | Use **only the follower** arm; project = **YOLO-based pick and place (e.g. a cup)**. Leader left uncalibrated. | Leader calibration kept failing (flaky bus ISSUE-112, false stalls ISSUE-111, accidental reset). Follower is calibrated and matches the reference. |
