# Winning angles - the "custom encoder/decoder" lesson

Last time the winner did not follow the normal route: they replaced the generic pipeline with a purpose-built encoder/decoder and beat a number everyone assumed was a ceiling.
The normal route here = "give Claude a `move_joint` tool and hope". Everyone will do that and it is slow, flaky and ugly. Our custom parts:

## 1. A learned/calibrated **action decoder** instead of LLM-emitted joint angles  (BUILT: `kin.py`, `arm.py`)
LLM says *where* (grid cell / pixel / "above the red block"), code turns that into joints:
URDF IK + pixel->table homography. The LLM never outputs a number it is bad at.

## 2. **Self-calibrating hand-eye** (BUILT in sim: `Arm.calibrate_table`)
Arm moves to known XYZ (FK from URDF), finds a coloured sticker in the camera, fits pixel->table. 1 minute, no checkerboard, survives someone bumping the camera. Re-run it live = great demo moment ("it calibrates itself").
Upgrade path: markerless (detect gripper by background subtraction / difference between frames while moving).

## 3. **Sensor-verified skills** (BUILT: `close_gripper` grasp check)
Grasp success from servo current + jaw stall (Station gives `current_ma`), not from asking the LLM "did it work?". Pushing-on-table detection the same way. Cheap, instantaneous, objective.

## 4. **Resolution & perception pipeline** (config trick)
Station downsamples frames to 224 px by default. Set `resize_target: 0`. Add: set-of-marks grid overlay (BUILT `vision.grid_overlay`), crop-and-zoom around candidate (TODO), colour-blob prior for speed (BUILT `find_named`) with LLM fallback for unknown objects.

## 5. **Plan once, verify cheaply**  (TODO in `agent.py`)
One LLM call: image -> JSON plan [{skill, target_cell}, ...]. Execute steps locally. After each step a *verifier* (before/after frames + sensor flags) - small/fast model (Haiku) or pure CV diff. Recover by re-planning only on failure. Target: whole pick-and-place = 2-3 model calls (~10 s) instead of 15.

## 6. **Skill library from teleop + language tags** (use Station's own features)
Record leader-arm demos for tricky primitives (open drawer, pour, hand-over), tag with `inference-tags` ("skill=handover"), replay parametrised. Claude calls `run_skill("handover")`. Gives dexterity without training a policy; also feeds SmolVLA as fallback data.

## 7. **Agent-native packaging** (BUILT: MCP + `CLAUDE.md`/`AGENTS.md`)
Track literally says *Codex or Claude*. Ship (a) an MCP server, (b) a `robotctl` CLI with the same verbs, (c) an `AGENTS.md`/`CLAUDE.md` that tells the agent how to behave. Then **Claude Code and Codex CLI both work with zero glue** and you can demo both, even switch live. Bonus: let the agent *write short Python against the safe `Arm` API* (code-as-policy) for novel tasks.

## 8. **Safety as a feature** (BUILT: `Guard`)
Clamps, per-call step limit, rate limit, overcurrent e-stop, latching `stop`. Judges (and NormaCore) care that an LLM driving hardware does not break it. Show the e-stop live.

## 9. Demo craft
- Live "thought bubble" web view: camera + grid + current plan + sensor verdicts (judges understand it in 5 s).
- Natural-language input by voice (Whisper/browser SpeechRecognition) -> the judge can talk to the robot.
- Pre-record a backup video of the best run. Pitch in 3 minutes: problem -> insight (decode, don't micromanage) -> live run -> numbers (success %, seconds per task over N trials).
- **Measure**: keep a log of N trials with success/time. A number on a slide beats an adjective.

## 10. Crazy-but-plausible extras (only after the core is bulletproof)
- **Dual-camera triangulation**: table cam gives XY, wrist cam gives final alignment (visual servoing on the last 5 cm).
- **World-model preview**: render the planned motion in the URDF twin and let a critic LLM veto collisions before executing.
- **Dogzilla bridge**: the repo already has `arm2dog`; a "robot dog fetches / arm hands over" story if a dog is present.
- **Self-improving skill memory**: store successful (pixel, joints, outcome) tuples; re-use for similar objects -> arm gets faster during the demo.
- **Failure-driven teleop**: when the agent fails twice it asks the human to teleop once, records, and adds a skill.
