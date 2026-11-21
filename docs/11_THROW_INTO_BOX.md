# Throw a small object into a box (sim first)

Written 2026-11-22. Status: sim environment exists (`src/normakit/throw_env.py`, `assets/mujoco/throw_scene.xml`, `scripts/throw_smoke.py`). A CEM-trained policy exists (sim only, see Results). Nothing here has touched the real arm.

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

## Results (measured, sim only)
Method: CEM over a 27-number box-conditioned release trajectory (`src/normakit/throw_policy.py`), 40 iterations x 64 candidates x 4 episodes (10,240 sim episodes, 340 s on 16 CPU cores, no GPU). Command: `.venv/bin/python scripts/train_throw.py --iters 40 --pop 64 --eps-per-cand 4 --workers 16 --out models/throw/`.
- Held-out eval (`scripts/eval_throw.py --episodes 100 --baseline`, random box positions, seeds not used in training): **85 of 100 in the box (85%)**, mean miss distance 3.0 cm. An untrained zero policy gets 0%.
- Training-time success averaged only about 40% because CEM samples noisy candidates; 85% is the final selected mean policy.
- Dataset: 176 of 200 rollouts succeeded, 3,971 (obs, action) frames in `models/throw/throw_dataset.npz` (keys obs, action, episode; obs is the 16-dim env observation, action the 8 normalised joint targets). Converting to LeRobot format for ACT is not done: write a small script that builds a LeRobotDataset with `observation.state` = first 8 obs values and `action`. Without a camera image, ACT needs a dummy or rendered frame, so an MLP on (state, box xy) may be the simpler distillation.
- Limits, honestly: this is the sim number. The ball release speed comes from jaw contact impulses that are probably inflated (see Feasibility), the policy is open loop, and nothing was tried on the real arm. Expect the real success rate to be far lower, possibly zero, until the sim ball/jaw contact is compared with a real toss.

## Scripts
- `scripts/train_throw.py [--smoke]`: CEM training, writes `models/throw/policy.npz`, `train_log.json`, `throw_dataset.npz`. `--smoke` runs in about 15 s on CPU.
- `scripts/eval_throw.py --episodes N`: success rate, writes `results/throw_eval.json`.
- `scripts/run_throw.py`: default is a sim dry run and never touches hardware. `--real` requires `--i-have-a-human`, then does a one-motor test (joint 1, +0.05 and back, speed 0.2) and asks for a typed `yes`. The toss then goes through `Guard.move` at 10 Hz and 0.35 step cap, so it is a slow, gentle version of the sim motion and will not throw the ball as in sim. The gripper stays put unless `--with-ball`. `--unsafe-fast` is refused; a faster mode needs an owner-approved Guard change (ISSUE-016). `--export-plan f.json` writes the open-loop (60, 8) plan for one box position; serving it through `server/policy_server.py` needs a small adapter (the server loads LeRobot ACT checkpoints), not written.

## A100 instructions
CEM needs CPU cores, not a GPU, so use the host's cores; the GPU matters only for ACT distillation.
```
export A100_HOST=<host>            # no secrets in the repo
export POLICY_API_TOKEN=<token>    # only for serving, set in your shell
scp -r src scripts assets/mujoco <user>@$A100_HOST:~/normacore/
ssh <user>@$A100_HOST 'cd ~/normacore && python scripts/train_throw.py --device cuda --iters 100 --pop 128 --workers 32 --out models/throw/'
scp <user>@$A100_HOST:~/normacore/models/throw/* models/throw/
```
`assets/mujoco/elrobot_scene.xml` and `hulls/` are needed on the host (untracked in git here). Then run `eval_throw.py` locally.

## Next steps
- Compare sim ball release speed with a real toss, then fix contact parameters before trusting 85%.
- Domain randomisation (ball mass/friction, servo lag), closed-loop policy, ACT distillation.
- Owner decision on the real-arm throw mode and a reference of the intended throw ("something like that").
