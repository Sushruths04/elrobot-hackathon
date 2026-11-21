# Throw a small object into a box (sim first)

Written 2026-11-22. Status: sim environment exists (`src/normakit/throw_env.py`, `assets/mujoco/throw_scene.xml`, `scripts/throw_smoke.py`). A CEM-trained policy exists (sim only, 72/100 strict throws, see Results). Nothing here has touched the real arm.

## Assumptions (the request was "throw a small object inside the box, something like that")
No reference clip, photo or description of "that" exists in the repo. These are guesses, change them if wrong:
- Object: a ball, 25 mm diameter, 8 g (foam or rubber class). Not a cup, not a block.
- Box: open top, 12 x 12 cm inside, 6 cm walls, standing on the table.
- Distance: box centre 25 to 38 cm in front of the base, x within +-12 cm. This is a short toss/lob, not an overarm throw.
- Start: the gripper holds the ball with the arm in the raised rest pose. Release happens by opening joint 8 during the motion.
- Success (strict, since the 2026-11-22 review): the ball has left the gripper (more than 3 cm from the jaw tip), it left at least 8 cm (horizontal) from the box centre, so outside the box footprint, and it ends at rest inside the box (inside the walls, below the rim). A ball carried over the box and dropped is a placement and does not count. The episode runs until the released ball is at rest or the 3 s horizon ends, so a bounce-out is a miss.
- Sim only for now. A real throw needs the safety decision below.

## Feasibility (measured in the twin, ST3215 model with the `calibrated_fix` preset)
- Joint speed with servo speed register at max (`send_goals(speed=1.0)`): joints 1..6 reach 0.5 to 0.86 normalised units per second, which is about 1.5 to 2.7 rad/s (joint 2: 0.69 norm/s, about 2.2 rad/s). The firmware profile caps at 1650 ticks/s (about 2.5 rad/s) and the acceleration register is set to 30.
- Random search (150 episodes, box fixed at x=0, y=0.32, random target pose for joints 2..5 plus a random release step): 14 of 150 landed in the box. So the arm can toss that far in the twin without any learning, and a trained policy has room to improve on 9%.
- Caveat: peak ball speeds in that search reached several m/s (p90 5.9 m/s, max 13.6). Those come from jaw/contact impulses in the simulator when the jaw snaps open, and are not trustworthy as real release speeds. Treat range numbers as sim-only until compared with a real toss. Contact parameters of the ball are a guess.
- Control step is 100 ms (10 Hz, the Guard rate) in the env, episode 30 steps (3 s). It was 50 ms before the review. One env step costs about 20 ms of CPU, so about 50 steps/s per process. 1M PPO steps is about 6 h on one core; run many envs in parallel on the A100 host CPUs.

## Guard blocks a real throw
`Guard.move()` limits calls to 10 Hz and non-gripper steps to 0.35 normalised, and clamps to 0.02..0.98. A throw needs a fast, large joint move and a gripper release timed within about 50 ms. At 10 Hz with 100 ms granularity the release timing is too coarse, and 0.35 step caps cut the swing. A throw also leaves no way to stop mid-flight.
Proposal (not implemented, `core.py` limits are untouched): a separate opt-in "throw mode" that
- is off by default and needs a human at the table to enable it (AGENTS.md hardware rules apply),
- uses a short, fixed, pre-validated joint trajectory family (small parameter set, bounded joint ranges inside the existing min/max norms),
- keeps the overcurrent estop and a keep-out zone (nothing toward the person side, box in front),
- first runs with a tiny amplitude and no ball, then increases in steps.
The owner must approve this before any real-arm test. Until then the policy trains and evaluates in sim only.

## Model choice
1. Sim RL or trajectory search. The task is a short ballistic motion, so a low-dimensional release trajectory works: target pose for joints 1..5, a movement duration and a release time (about 8 parameters), conditioned on box x,y. Use CEM or SAC/PPO (stable-baselines3 or plain PyTorch) on `ThrowIntoBoxEnv`. Reward: -distance at the end, +10 on a strict success, +2 to +8 if the ball is in the box but was placed (shaped by release distance, so CEM is pushed from placement towards a toss).
2. Distil the best behaviour into a conditioned policy: generate successful episodes across random box positions and train ACT (or an MLP) on (state, box position) to action chunks. ACT is already the chosen model in `docs/10_MODEL_AND_A100_PLAN.md`; serve it with `server/policy_server.py`. The throw needs no camera if the box position comes from the table calibration. A camera input can be added later.
3. Train on the A100 (the laptop GPU is broken). The sim runs on CPU, so run rollouts on the A100 host cores and the network update on its GPU.

## Results (measured, sim only)
History: the first version reported 85/100. The review on 2026-11-21 showed that this was a placement, not a throw: the arm carried the ball over the box and opened the jaw 0 to 3 cm from the centre, and `in_box` also counted a ball still in the gripper. That number is withdrawn.

Current method: CEM over a 48-number box-conditioned two-phase trajectory (`src/normakit/throw_policy.py`): windup pose A, swing to pose B, gripper opens at a chosen step of the swing, all targets clipped to Guard's 0.02..0.98. Trained at 10 Hz with the strict success above. Training (about 40 min on 18 CPU cores, no GPU), each run warm-started from the previous one:
```
.venv/bin/python scripts/train_throw.py --iters 50 --pop 96 --eps-per-cand 4 --workers 18 --out models/throw/
.venv/bin/python scripts/train_throw.py --init r1.npz --sd0 0.5 --iters 40 --pop 64 --eps-per-cand 8 --workers 18 --out models/throw/
.venv/bin/python scripts/train_throw.py --init r2.npz --sd0 0.3 --iters 40 --pop 64 --eps-per-cand 8 --workers 18 --seed 1 --out models/throw/
.venv/bin/python scripts/train_throw.py --init r3.npz --sd0 0.3 --iters 40 --pop 96 --eps-per-cand 8 --workers 18 --seed 2 --out models/throw/
```
- Held-out eval (`scripts/eval_throw.py --episodes 100 --baseline`, seeds 900000+): **72 of 100 strict throws into the box (72%)**, 95% interval about 63 to 80%. 92 of 100 balls end in the box; 20 of those left the gripper less than 8 cm from the centre and are not counted. Median miss distance 5.2 cm, mean 13 cm (a few balls roll far away). Mean release distance 9.8 cm, mean release speed 0.84 m/s. Zero policy: 0%. `--episodes 50`: 35/50 (70%).
- The learned motion: the jaw opens at the start of the swing and the swing flings the ball about 9 to 13 cm into the box. This is a short lob just past the 8 cm threshold, not a long throw. Failures cluster at the corners of the box range (far corners and the near edge).
- Rounds 3 and 4 both scored 72/100 on eval; more CEM did not help, so the linear box conditioning or the 10 Hz release timing is probably the limit (guess).
- "Held-out" means held-out box positions only: ball mass, friction and servo parameters are fixed, the simulator is deterministic.
- Dataset: 82 of 100 rollouts succeeded, 1,293 (obs, action) frames in `models/throw/throw_dataset.npz` (gitignored; keys obs, action, episode; obs is the 16-dim env observation, action the 8 normalised joint targets). Converting to LeRobot format for ACT is not done: build a LeRobotDataset with `observation.state` = first 8 obs values and `action`. Without a camera image, ACT needs a dummy or rendered frame, so an MLP on (state, box xy) may be the simpler distillation.
- Limits: sim number only. The ball leaves the jaw through contact impulses that may be inflated (see Feasibility), the policy is open loop, and nothing was tried on the real arm.

## Real-arm gap (why `run_throw.py --real` will usually refuse)
- Guard caps each call at 0.35 normalised per joint; the sim plan has no such cap and jumps up to 0.95 in one step (frame 0 goes START to pose A). On the real arm the same plan would lag and the swing would be slower, so the sim result does not transfer. Making the env apply the same per-call cap and retraining is the next step.
- The plan drives joints to the Guard clamp (0.02 and 0.98) and sweeps the jaw tip low (1.5 cm above the table) or back towards the base. The keep-out check below rejects 15 of 20 random box positions, including the default (0, 0.32). This is intended: the plan is not safe to replay as is.
- `Guard.cfg.forbidden_boxes` exists but `Guard.move` does not enforce it, so `run_throw.py` checks the plan itself.

## Scripts
- `scripts/train_throw.py [--smoke]`: CEM training, writes `models/throw/policy.npz`, `train_log.json`, `throw_dataset.npz`. `--smoke` runs in about 15 s on CPU. `--init policy.npz --sd0 0.3` warm-starts.
- `scripts/eval_throw.py --episodes N`: success rate, writes `results/throw_eval.json`.
- `scripts/run_throw.py`: default is a sim dry run and never touches hardware. `--real` requires `--i-have-a-human` and refuses a box outside the trained range (x within 0.12, y 0.25 to 0.38). Before connecting it prints the plan (frames, max target per joint, 30 frames at 10 Hz = 3 s) and replays it in the twin to check the jaw-tip path: it refuses if the tip goes below 3 cm above the table or closer than 5 cm in front of the base (ASSUMED keep-out, owner must confirm). After connecting it refuses unless every joint 1..7 is within 0.1 of `START_NORMS`, then does a one-motor test (joint 1, +0.05 and back, speed 0.2) and asks for a typed `yes`. The toss then goes through `Guard.move` (10 Hz, 0.35 step cap). Any exception, including `EOFError` from the prompt, calls `guard.estop()`. The gripper stays put unless `--with-ball`. `--unsafe-fast` is refused; a faster mode needs an owner-approved Guard change (ISSUE-016). `--export-plan f.json` writes the open-loop (30, 8) plan for one box position; serving it through `server/policy_server.py` needs a small adapter (the server loads LeRobot ACT checkpoints), not written.

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
- Apply Guard's 0.35 per-call step cap and keep-out limits inside the env, retrain, so the plan can pass the real-arm checks.
- Compare sim ball release speed with a real toss, then fix contact parameters before trusting 72%.
- Domain randomisation (ball mass/friction, servo lag), closed-loop policy, ACT distillation.
- Owner decision on the real-arm throw mode and a reference of the intended throw ("something like that").

## Review 2026-11-21: what was changed and what was not
- Fixed: placement counted as a throw, held ball counted, first-contact termination (strict success, run until rest, retrained, number now 72%).
- Fixed: start-pose check, exception e-stop, box range check, plan listing, keep-out check of the simulated plan, 10 Hz training so replay timing matches.
- Not fixed, by design: the real plan still lacks the Guard step cap in training (see Real-arm gap); the keep-out check therefore refuses most plans.
- Not fixed, not ours: `.mcp.json`, `norma`, `scripts/autorun.py`, `src/normakit/kin.py`, `station_backend.py`, `vision.py` are the team's uncommitted working-tree changes. They are not in any throw commit and were left untouched.
- Not fixed: `elrobot_scene.xml` and `hulls/` stay untracked (team assets, large meshes); a clean checkout cannot run the env without them.
