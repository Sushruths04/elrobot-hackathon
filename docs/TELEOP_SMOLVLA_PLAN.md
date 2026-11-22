# Teleop data collection for SmolVLA: spoon, scoop beans, pour into cup (review document)

Written 2026-11-22 for reviewers. Status: the first real session (5 episodes, `datasets/teleop_spoon_beans_20261122_093829`) was **unusable** (recorded at ~10 fps but stamped 30, leader j4 out of range, wrist cam frozen); see the DO_NOT_TRAIN.md in that folder. The recorder is fixed and tested at a real 30.0 fps with the cameras. **Before recording, run the preflight checklist in `.claude/skills/elrobot-task-recorder/SKILL.md` (Teleop section).** Control panel: `scripts/teleop_panel.py` -> http://127.0.0.1:8893.
Arms and external cameras were NOT plugged in when this was written (lsusb showed only mouse + laptop webcam).

## Task
Language instruction (identical in every episode, this exact string):
`pick up the spoon, scoop beans and pour them into the cup`
One episode = start from the folded rest pose, grasp spoon, scoop from the bean pile, carry, pour into cup, put the spoon down (or hold it), return near rest.
Decide before recording whether the spoon is put back down at the end; keep it the same in every episode.

## How it is recorded
Script: `scripts/record_teleop.py` (new; the older `scripts/record_demo.py` is untouched).
- Operator moves the **leader arm**; Station mirroring (`POST :8891/mirror {"on":true}`) makes the follower copy it.
- `observation.state` = follower joints 1-8, normalised 0..1 of each joint's calibrated range (`normakit.busread.snapshot`, `norm`).
- `action` = leader joints 1-8, same normalisation. Joint 8 = gripper.
- `observation.images.table` = front camera (`front_camera()` auto-detect), `observation.images.wrist` = wrist camera (`wrist_camera()`), native resolution (640x480 expected).
- 30 fps fixed clock. Arm and cameras are read by background threads; each frame uses the newest sample, so timestamps are real time even if a camera is slower than 30 fps (frames then repeat). Episodes with arm data older than 0.5 s get a `stale_arm_frames` count in the labels.
- Format: LeRobot v3.0 via lerobot 0.4.4 `LeRobotDataset.create`, videos AV1 (lerobot default), `robot_type=elrobot`.
- Control by files in `run/` (the agent touches them when the operator says so in chat): `TELEOP_GO`, `TELEOP_DONE_OK`, `TELEOP_DONE_FAIL`, `TELEOP_DISCARD`, `TELEOP_QUIT`. Max 120 s per episode (then saved as `timeout`, success=false).
- Labels: `<root>/EPISODE_LABELS.json` (`success`, `end`, `frames`, `seconds`, `stale_arm_frames`). **Failures are kept in the dataset, so training must filter to success=true episodes** (`--dataset.episodes=[...]`) or failed demos must be discarded instead of kept.
- Command:
  `.venv/bin/python scripts/record_teleop.py --task "pick up the spoon, scoop beans and pour them into the cup" --root datasets/teleop_spoon_beans_$(date +%Y%m%d_%H%M%S)`

## What was verified (dry run, synthetic data)
- go / ok / discard / fail / quit cycle: 2 episodes saved, 1 discarded, labels correct.
- `meta/info.json`: v3.0, 2 episodes, 180 frames, features state(8), action(8), images.table, images.wrist.
- Both mp4 files decode with PyAV (180 frames each, 30 fps); parquet timestamps 0..2.967 s per episode.

## Known problems / open questions for reviewers
1. **Local video decoding is broken on this laptop.** `torchcodec` 0.14 in `~/.local` is built for CUDA 13 and needs `libavdevice.so.58` (missing); torchvision 0.27 has no `VideoReader`. So `LeRobotDataset[i]` fails here. The files themselves are fine. Training must run in an env where torchcodec loads (cluster L40S env, or fix locally: `apt install libavdevice58` + torchcodec build matching torch 2.12/cu126). Not changed yet.
2. **Leader to follower mapping must be validated before recording** (`action` is the leader's normalised position; at deployment the policy's action is sent as the follower's goal in follower-normalised units). `scripts/teleop.py` says Station percent-mirroring once "went out of sync". Check: mirroring ON, hold 3 different poses, per-joint |follower_norm - leader_norm| should be small (< 0.03) and stable. If not, either fix calibration or record follower goal instead of leader position as the action.
3. **Gripper with a spoon:** confirm the follower jaws actually hold the spoon firmly when the leader trigger is closed (stall ticks / current), and that the spoon doesn't rotate in the jaws while scooping.
4. **SmolVLA camera keys:** `lerobot/smolvla_base` was pretrained with its own camera names. In lerobot 0.4.4 `TrainPipelineConfig` has `rename_map`; reviewer: confirm whether our keys `observation.images.table` / `observation.images.wrist` need `--rename_map` to the base model's keys, or whether fine-tuning with new keys is fine.
5. State/action are 0..1 normalised joint positions (not radians). SmolVLA pads to 32 dims and applies its own mean/std normalisation from dataset stats, so this should be fine. Reviewer confirm.

## SmolVLA defaults (read from installed lerobot 0.4.4 `SmolVLAConfig`)
chunk_size 50, n_action_steps 50, max_state_dim 32, max_action_dim 32, images resized+padded to 512x512, freeze_vision_encoder True, train_expert_only True, lr 1e-4.

## How much data
- SmolVLA authors / LeRobot docs fine-tune on about **50 episodes per task** (SO-100 tasks). Fewer than ~25 usually fails.
- Plan: **50 successful episodes minimum**, aim for 60-80 to allow discards.
  - Vary the scene on purpose: about 5 spoon positions x 2-3 cup positions x pile shape; ~10 episodes per spoon position.
  - Keep lighting, camera mounts and the instruction string fixed.
  - Each episode ~20-40 s, so 50 episodes is roughly 30 min of recording + resets, 1-2 hours total in practice.
- Keep motions smooth and consistent (same grasp on the spoon, same scoop direction). Pauses and hesitations get learned too.

## Training (to be confirmed by reviewer, run on the cluster GPU, not the 8 GB laptop A2000)
```
lerobot-train --policy.path=lerobot/smolvla_base \
  --dataset.repo_id=local/elrobot_teleop --dataset.root=datasets/teleop_spoon_beans_<stamp> \
  --dataset.episodes='[<success episode indices>]' \
  --batch_size=64 --steps=20000 --output_dir=outputs/smolvla_spoon --policy.device=cuda
```
(+ `--rename_map` if item 4 requires it.)

## Rules that still apply
Human presses ARM; mirroring OFF for autonomous control afterwards; film every run; success decided by the human; push only to the private repo, only when asked.
