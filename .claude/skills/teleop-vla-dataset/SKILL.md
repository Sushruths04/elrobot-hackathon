---
name: teleop-vla-dataset
description: Collect leader-arm teleop demonstrations on the NormaCore ElRobot as LeRobot v3 datasets for fine-tuning SmolVLA (or ACT/pi0) for ANY new task - preflight checks, the teleop panel, per-episode quality check, how many demos, merging sessions, training on the cluster, deployment. Use when the user wants to record teleop data, check recorded episodes, plan a new task's dataset, or train/deploy a VLA on the arm.
---

# Teleop -> LeRobot -> SmolVLA (any task)
Repo `normacore-hackathon`. Training runs on the cluster GPU (see the `wzl-cluster-gpu` skill), never on the laptop (8 GB GPU, broken torchcodec).
First working session: `datasets/teleop_spoon_beans_20261122_103601` (6 episodes, 5 TRAIN; spoon -> scoop beans -> pour into cup).

## 0. Preflight (do it yourself before saying "ready to record")
1. **Laptop on the charger** (a low battery shut it down mid-calibration; unsaved calibration is lost).
2. Station up (`./norma status`): both buses, 16 motors `frozen=True`. After a reboot start it: `(cd run && nohup ../bin/station --web --tcp --config station.yaml > ../logs/station.log 2>&1 &)`.
3. **Leader calibration covers its real motion**: move the leader around, and every joint norm stays within 0..1 (panel table). If not: `./norma backup`, then `python3 tools/calibrate.py reset --bus leader`, and check `logs/station.log` for "Failed to reset" (leader bus glitch ISSUE-112: resend the reset until it's clean). Then run `python3 -u tools/calibrate.py watch --bus leader > run/calib_watch.log &`. The human sweeps every joint to BOTH hard stops, incl. the gripper trigger. Compare spans with the good leader cal (j1 ~1800, j2 2100, j3 2290, j4 >= 2000, j5 >= 1750, j6 1990, j7 >= 2200, j8 >= 1200; the watcher's "ok" uses follower spans, so ignore it for j5/j7). Then `python3 tools/calibrate.py save --bus leader` and `./norma backup`.
4. **Cameras stable**: `journalctl -k -b | grep -c "USB disconnect"` = 0. Cameras directly on the laptop, not via the hub (hub port 1.1.2 dropped ~25x). A replug renumbers camN; Station sometimes doesn't re-attach -> backup + restart Station.
5. Only one recorder: `ps -eo args | grep [r]ecord_teleop` is empty before a session starts.

## 1. Recording (the human drives, from the panel)
`nohup .venv/bin/python -u scripts/teleop_panel.py > run/teleop_panel.log 2>&1 &` -> **http://127.0.0.1:8893** (local only)
- Match the arms (all table rows green, |diff| < 0.08) -> **MIRRORING ON** (it refuses while red).
- **START SESSION** (set the task text first; the SAME exact string for every episode of a task) -> new `datasets/teleop_<task>_<stamp>/`.
- Per episode: **GO** (G), do the task, then **SAVE success** (S) / **SAVE failed** (F) / **DISCARD** (D). It refuses GO if a camera is dead. Max 120 s.
- **END SESSION** before checking (the parquet is only readable once it's finalized). Close a recorder only with `touch run/TELEOP_QUIT` (killing it corrupts the v3 dataset). Then MIRRORING OFF.
- Recorder: `scripts/record_teleop.py`. state = follower joints 1-8, action = leader joints 1-8 (both normalised 0..1 of the calibrated range), `observation.images.table` + `observation.images.wrist` 640x480, 30 fps, `image_writer_threads=8` (without it only ~10 fps real). Labels: `EPISODE_LABELS.json` (success, real_fps, stale_camera_frames ...). To change the task text, edit `DEFAULT_TASK` in the panel or type it in the box.

## 2. Check every session (after the FIRST episode of a new task, then every session)
`.venv/bin/python scripts/check_teleop_dataset.py datasets/<session> --sheet` -> per episode TRAIN / FAIL-LABEL / REJECT plus reasons, and the TRAIN episode list. `--sheet` writes `check/epN.jpg` (6 frames). **Look at the sheets** (the task really done? spoon in the cup?). The human's label decides success; the checker only rejects data problems (fps, out-of-range actions, frozen camera, tracking error, unused gripper).
Healthy values seen: real fps 30.0, follower lag 10-13 frames, track-err <= 0.015, frozen <= 3 frames.

## 3. How much data (SmolVLA official docs)
- **~50 successful episodes per task** as the starting point; 25 was "not enough, bad performance". The reference dataset: 50 = 5 object positions x 10 demos each. **Repeat each variation several times** rather than 50 random scenes.
- Plan per task: decide the variations (e.g. spoon 5 spots x cup 2 spots), ~5-10 demos per combination, 60-80 attempts -> 50 good ones.
- Keep fixed: camera mounts, lighting, the task string, the start pose (folded rest), the end state (decide: spoon put down or held). Keep the operator's body out of the table camera if possible (at deployment nobody is there).
- Smooth and consistent moves; short pauses at the start/end are fine (trim not required); hesitations get learned.
- Many tasks: one dataset per task, each with its own task string; the same robot/cameras let you merge tasks for multi-task training later.

## 4. Merge sessions + train (cluster)
- Merge sessions of one task: `lerobot.datasets.aggregate.aggregate_datasets` (lerobot 0.4.4 has it) into one root; carry the TRAIN episode indices over.
- SmolVLA expects cameras `observation.images.camera1/2/3`. Ours: `--rename_map='{"observation.images.table": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}'`. With fewer than 3 cams: `--policy.empty_cameras=1` (verify on the cluster's lerobot version).
- ```
  lerobot-train --policy.path=lerobot/smolvla_base --dataset.repo_id=local/<task> --dataset.root=<root> \
    --dataset.episodes='[<TRAIN list>]' --rename_map='{...}' --batch_size=64 --steps=20000 \
    --output_dir=outputs/train/smolvla_<task> --job_name=smolvla_<task> --policy.device=cuda
  ```
  Docs: 20k steps ~4 h on one A100 (L40S similar order). Tune the steps on results. State/action are 8-dim (SmolVLA pads to 32). The fps is 30 (chunk 50 = 1.7 s).
- Copy the dataset to the cluster (`wzl-cluster-gpu` skill). Datasets are gitignored; never push them to GitHub.

## 5. Deploy on the arm
The lerobot `lerobot-rollout` command needs a lerobot robot type; ElRobot is driven through Station, so use our runner: `scripts/run_policy.py` (remote policy server, every command through Guard: clamp, step limit, overcurrent stop; `--backend sim` first, then `--backend station --i-have-a-human`) with `server/policy_server.py` on the GPU. The policy outputs the leader-normalised action -> send it as the follower goal in follower-normalised units (the same mapping as Station mirroring). Use the same task string as in training. Mirroring OFF during the rollout. Film it, human verdict.

## Mistakes already made (don't repeat)
Recorded at 10 fps stamped 30 (no image writer threads); leader j4 range too narrow (action 1.49 vs follower clamped 0.99); stale readings from a reused snapshot queue (fixed: drain); orphaned recorder after a panel restart (two recorders raced); camN renumbering after a replug (fixed: stable queue ids + live check before GO); `pkill -f` killing the agent's own shell; pushing unproven work to GitHub. Unusable sessions are kept with `DO_NOT_TRAIN.md`.

Sources: https://huggingface.co/docs/lerobot/smolvla , https://huggingface.co/docs/lerobot/rename_map

## 2026-11-22 status
- Training set for online training: `datasets/spoon_beans_v1_train` (19 good episodes, 30351 frames, made with `dataset_tools.delete_episodes`). The operator trains online, then gives us the checkpoint.
- Remove marker colours from objects before recording or deploying (the orange handle was only for segmentation; the demos have a plain wooden spoon).
- Deployment runner for a trained SmolVLA checkpoint is still TODO: load the policy plus pre/post processors (`lerobot.policies.factory.make_pre_post_processors`), cameras table→camera1, wrist→camera2, send actions as follower goals through Guard/armd watchdogs, film it.
