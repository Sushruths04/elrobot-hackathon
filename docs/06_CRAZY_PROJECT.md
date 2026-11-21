# The crazy project: "Skill Foundry" (teach by showing, run by saying, improve by itself)

## Evidence this is the right bet
- A participant from the earlier HackLab NormaCore track wrote that their robot still could not pick a single cube
  at 2 AM, and what saved them was a **modular skill library with Claude choosing the skill**
  (linkedin.com/in/rajveer-shaw, self-reported). Raw "Claude drives joints" is what most teams will try and fail at.
- NormaCore positions ElRobot as "optimized for teleoperation and demonstration recording" and Station as the
  record/replay layer (Nebius customer story). A teleop-recorded skill library is *their own story*, so judges get it instantly.
- The repo has no LLM/MCP code, so any working Claude link already stands out; the skill layer is what makes it reliable.

## The product (one sentence for judges)
"You show the robot a move once with the leader arm. Claude names it, understands when to use it, chains skills
from a spoken sentence, watches the result through the cameras, and when something fails it fixes the skill and
saves the better version - live."

## Architecture
```
leader arm --teleop--> record_skill("pick_cube", tags, camera keyframes, joint trajectory, grip current profile)
                                   |
                         skills/  (JSON: trajectory + preconditions + success check + examples)
                                   |
voice/text -> Claude (MCP) -> plan = [skill, params, skill...] -> Guard -> follower arm
                 ^                                              |
                 +---- look (2 cameras) + motor current <-------+  (verify, retry, repair)
```

## Ideas, ranked by (wow x chance it works in 48 h)
1. **Skill library from teleop (core, must ship).** `record_skill`, `play_skill`, `list_skills` as MCP tools.
   Parametrise by replaying the trajectory relative to a detected object pixel (our homography + IK from `arm.py`),
   so a skill generalises beyond the exact recorded position.
2. **Sensor-verified success.** Gripper current/position decides "grasped?" with no vision call. Failure -> Claude
   retries with an offset. Show the live current graph on screen.
3. **Self-repairing skills (the crazy part).** When a skill fails twice, Claude gets the failure frames + trajectory,
   proposes a modified parameter set (offset, speed, grip width), tries it, and on success saves it as `v2`.
   Demo line: "the robot got better during this pitch." Log before/after success rate.
4. **Kinesthetic correction.** Judge grabs the leader arm mid-task to correct the robot; the correction is
   stored as a new demo for that skill (DAgger-lite). Uses hardware only this kit has.
5. **Dream before act.** Run the plan in our pybullet twin first (URDF is official), show it on screen, abort on
   predicted collision, then execute. Judges see the robot "think".
6. **Show-and-tell objects.** A judge hands over an unseen object; Claude names it, picks a grasp strategy
   (top/side, force from current), asks a clarifying question if ambiguous, and does it.
7. **Live thought bubble UI + voice.** Camera feed with Claude's plan overlaid (grid cells, target, current).
   Voice in via browser speech API. Cheap, high stage impact.
8. **Crowd-pleaser add-on only if 1-3 are solid:** the arm plays a game (tic-tac-toe on a printed grid) -
   Claude picks the move, skill library places the piece. High wow, high precision risk: do last.
9. **Fleet angle for the pitch (no build):** skills are files; the same library runs on any Station arm
   (SO-101 too) via NormFS. One slide, ties to NormaCore's business story.

## Best use case to pitch: "teachable desk assistant"
Desk tasks (sort, hand over, tidy, stack, put-in-box) match the hackathon's own "desk automation" framing and the
June track's "put the waffle in the black box". Benchmark it: 10 judge-style commands, log success rate, median
time, model calls per task, and number of self-repairs.

## Build order (timeboxed, today)
| Block | Deliverable | Gate |
|---|---|---|
| 0 | Boards visible, Station up, calibrated, teleop works | follower mirrors leader |
| 1 | `record_skill`/`play_skill` on real arm, 3 skills (pick, place, push/wave) | each replays 8/10 |
| 2 | MCP tools + Claude composes a 2-skill plan from one sentence | works from Claude Code |
| 3 | Object-relative parameters via camera calibration | skill hits an object moved by 5 cm |
| 4 | Verification + retry + self-repair + benchmark log | before/after numbers |
| 5 | Thought-bubble UI, voice, backup video | recorded |
Do not start a block until the previous gate passes. Cut from the bottom.

## Risks
- Hardware not yet detected (motor boards missing on USB as of last check): everything above waits on it.
- Event dates conflict between listings (14-15 vs 21-22 Nov); confirm with organisers. If it is today, skip blocks 5+ and
  go straight 0 -> 1 -> 2.
- Replay of raw trajectories drifts if objects move: that is why block 3 exists.
