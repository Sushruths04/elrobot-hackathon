# Throw a small object into a box (sim first)

Written 2026-11-22. Status: sim environment exists (`src/normakit/throw_env.py`, `assets/mujoco/throw_scene.xml`, `scripts/throw_smoke.py`). No policy is trained yet. Nothing here has touched the real arm.

## Assumptions (the request was "throw a small object inside the box, something like that")
No reference clip, photo or description of "that" exists in the repo. These are guesses, change them if wrong:
- Object: a ball, 25 mm diameter, 8 g (foam or rubber class). Not a cup, not a block.
- Box: open top, 12 x 12 cm inside, 6 cm walls, standing on the table.
- Distance: box centre 25 to 38 cm in front of the base, x within +-12 cm. This is a short toss/lob, not an overarm throw.
- Start: the gripper holds the ball with the arm in the raised rest pose. Release happens by opening joint 8 during the motion.
- Success: the ball ends inside the box (inside the walls, below the rim).
- Sim only for now. A real throw needs the safety decision below.

## Feasibility (measured in the twin, ST3215 model with the `calibrated_fix` preset)
- Joint speed with servo speed register at max (`send_goals(speed=1.0)`): joints 1..6 reach 0.5 to 0.86 normalised units per second, which is about 1.5 to 2.7 rad/s (joint 2: 0.69 norm/s, about 2.2 rad/s). The firmware profile caps at 1650 ticks/s (about 2.5 rad/s) and the acceleration register is set to 30.
- Random search (150 episodes, box fixed at x=0, y=0.32, random target pose for joints 2..5 plus a random release step): 14 of 150 landed in the box. So the arm can toss that far in the twin without any learning, and a trained policy has room to improve on 9%.
- Caveat: peak ball speeds in that search reached several m/s (p90 5.9 m/s, max 13.6). Those come from jaw/contact impulses in the simulator when the jaw snaps open, and are not trustworthy as real release speeds. Treat range numbers as sim-only until compared with a real toss. Contact parameters of the ball are a guess.
- Control step is 50 ms (20 Hz) in the env, episode 60 steps (3 s). One env step costs about 20 ms of CPU, so about 50 steps/s per process. 1M PPO steps is about 6 h on one core; run many envs in parallel on the A100 host CPUs.

## Guard blocks a real throw
`Guard.move()` limits calls to 10 Hz and non-gripper steps to 0.35 normalised, and clamps to 0.02..0.98. A throw needs a fast, large joint move and a gripper release timed within about 50 ms. At 10 Hz with 100 ms granularity the release timing is too coarse, and 0.35 step caps cut the swing. A throw also leaves no way to stop mid-flight.
Proposal (not implemented, `core.py` limits are untouched): a separate opt-in "throw mode" that
- is off by default and needs a human at the table to enable it (AGENTS.md hardware rules apply),
- uses a short, fixed, pre-validated joint trajectory family (small parameter set, bounded joint ranges inside the existing min/max norms),
- keeps the overcurrent estop and a keep-out zone (nothing toward the person side, box in front),
- first runs with a tiny amplitude and no ball, then increases in steps.
The owner must approve this before any real-arm test. Until then the policy trains and evaluates in sim only.

## Model choice
1. Sim RL or trajectory search. The task is a short ballistic motion, so a low-dimensional release trajectory works: target pose for joints 1..5, a movement duration and a release time (about 8 parameters), conditioned on box x,y. Use CEM or SAC/PPO (stable-baselines3 or plain PyTorch) on `ThrowIntoBoxEnv`. Reward: -distance at landing, +10 if in box.
2. Distil the best behaviour into a conditioned policy: generate successful episodes across random box positions and train ACT (or an MLP) on (state, box position) to action chunks. ACT is already the chosen model in `docs/10_MODEL_AND_A100_PLAN.md`; serve it with `server/policy_server.py`. The throw needs no camera if the box position comes from the table calibration. A camera input can be added later.
3. Train on the A100 (the laptop GPU is broken). The sim runs on CPU, so run rollouts on the A100 host cores and the network update on its GPU.

## Next steps
- Add a CEM/PPO training script and a held-out evaluation over random box positions (success rate with 100 episodes).
- Domain randomisation: ball mass/friction, box position, servo lag.
- Owner decision on the real-arm throw mode and a reference of the intended throw ("something like that").
