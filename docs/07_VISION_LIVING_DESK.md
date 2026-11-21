# Vision: "Norma" - a desk creature with a mind (skill library is only its motor cortex)

Idea: not a robot that executes commands, but an embodied agent that perceives, remembers, feels, talks and learns.
Everything is built from signals this kit already exposes (2 cameras, per-motor current/temp/position, leader arm,
URDF twin, Claude via MCP).

## Brain layers
| Layer | What it is | Built from |
|---|---|---|
| Senses | 2 cameras + proprioception + **touch = motor current** | Station queues |
| Touch memory | per-object card: mass (from current at lift), width (gripper stall), softness (current ramp), name | grip probe routine |
| Episodic memory | timestamped snapshots + object locations ("where is the pen?") | camera + vision.py + JSON/SQLite |
| Cortex | Claude: plans, chats, decides when to explore, asks questions | MCP tools |
| Cerebellum | skills (teleop-recorded), IK, calibration, Guard | normakit |
| Reflexes | overcurrent / limit / e-stop, always on, no LLM | core.Guard |
| Dreams | rehearse plan in pybullet twin before acting | sim_backend |
| Telepathy | leader arm as a two-way link (below) | 2 buses |

## The five demo moments (3-minute pitch)
1. **"Touch to know."** Judge drops an unseen object. Norma grips it gently, says "~40 g, soft, 3 cm wide - a sponge?"
   Mass/softness come from servo current, not vision. Nobody else will have a haptic sense.
2. **"Remember."** Judge hides it under a cup. Later: "where is the sponge?" Norma answers from memory and fetches it.
3. **"Feel it."** *Bilateral teleop* (stretch): follower grip current is fed back to the leader arm's torque limit, so the
   operator feels resistance when the follower squeezes something. Needs our own 2-bus loop and strict safety clamps;
   untested, do only after plain teleop works.
4. **"Learn from a nudge."** Norma fails a grasp, Claude proposes a fix, retries, and saves skill v2. Judge corrects by hand
   on the leader arm; correction stored as a demo.
5. **"Think, then act."** Plan shown as a ghost arm in the twin on screen; abort on predicted collision; then it moves.

## Personality / stage
Voice in (browser speech), voice out (TTS), live thought-bubble overlay, narrated reasoning. A name and character make judges
remember it. Keep replies short; never let it move without Guard.

## Build order (cut from the bottom)
1. Teleop + calibration working (gate for everything).
2. Skill record/play + MCP (docs/06).
3. Touch probe -> object cards (1 hour: lift, read current, write JSON).
4. Memory: log detections each `look`, answer "where is X" tool.
5. Failure -> Claude repair -> v2.
6. Thought-bubble UI + voice.
7. Twin preview.
8. Bilateral haptics (only if everything else is solid).

## Honesty
None of this is built yet beyond the base kit (IK, guard, calibration, MCP in sim). Touch-based mass/softness is a
hypothesis: ST3215 current is noisy and needs per-motor baselines; test it before promising numbers.
