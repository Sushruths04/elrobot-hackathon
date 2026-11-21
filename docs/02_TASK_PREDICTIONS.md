# Predicting the NormaCore challenge (HackLab Berlin, 21-22 Nov 2026)

**Hard evidence**
- Track text (identical in June and Nov): "AI-Powered Robot Control - Integrate the Station API into Codex or Claude so that it can control the robot by looking through the cameras."
- June: NormaCore supplied ElRobot kits + Station + judges; NormaCore's own site lists "ElRobot kits, Station software, judging and support".
- June team "Waffle" (Team 4): ~60 teleop demos, SmolVLA fine-tune on a Nebius GPU, patched example for 8 joints, instruction "put the waffle in the black box". NormaCore said the Claude/ChatGPT *supervisor was planned but did not finish*. => the LLM-in-the-loop part is the unsolved, differentiating part.
- Repo head 2026-11-10: new **28 mm gripper camera mount** -> wrist camera is part of the 2026 rig. Station has no MCP/LLM code -> they want to *see* what hackers build.
- Same org supplied kits to a separate "Robotics Pizza Hackathon" (SO-101 + ElRobot, scored Precision Topping 30 / Recipe Assembly 40 / Mystery 30, "mystery" = generalisation). Shows their taste: **timed, scored, includes a hidden task**.
- HackLab judging: Sunday 18:00 judging, 19:00 live pitches. Pitch + live demo decide.

**Not known**: scoring rubric, object set, whether SO-101 is also provided, whether cameras are pre-mounted, whether Claude API credits are supplied (sponsors give credits: Anthropic, Nebius).

## Predicted task formats (my probabilities)
| # | Format | P | What wins |
|---|---|---|---|
| A | **Open-ended**: "make Claude/Codex do something useful with the arm by looking" (judged on demo + pitch) | 45% | reliability, wow, story, working live |
| B | **Fixed task list scored in a time box** (pick X into Y; sort by colour; stack; hand over) incl. a **surprise/mystery object** | 35% | generalisation + speed + zero crashes |
| C | **Natural-language command from judge at demo time** ("put the red thing in the cup", "tidy the desk") | 40% (overlaps A/B) | grounding any phrase, recovery |
| D | Needs a trained policy (SmolVLA) | 10% | irrelevant if you keep LLM-skills primary; keep as fallback |
| E | Dog (Dogzilla) involvement / arm2dog mirroring | 15% | add a bonus integration only if time |

Likely objects: small light items (blocks, cap, waffle/foam, marker) - payload of 30 kg-cm ST3215 on a 430 mm arm is ~100-200 g.
Likely scene: arm clamped to a table, table camera overhead/front + wrist camera, 2-4 objects + a box/tray.

## What this means for design
1. Do not depend on a trained policy for the main demo (June teams spent the night collecting + training).
2. Make *one deterministic, calibrated reach/pick/place skill* and let the LLM only decide "what and where".
3. The judge will say a sentence you have never heard => LLM must ground arbitrary language to the image => keep the model in the loop for perception/planning; keep the motor loop out of the LLM.
4. Speed is scored implicitly: minimise model round-trips (one look -> plan multiple steps -> verify at the end of each step with a cheap check).
5. Prepare for failure modes live judges love to create: moved object, occlusion by hand, new object, lighting change.
