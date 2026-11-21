# Plan: fast model on a remote A100 for the ElRobot arm

Goal: a low-latency policy for the 8-DoF ST3215 follower (joints 1..8, 8 = gripper, normalised 0..1) with one 640x480 camera, served from one A100 over an API. This laptop has no working NVIDIA GPU, so training and inference both run on the A100. Written 2026-11-22. Latency numbers are **estimates** unless marked measured. Nothing here has been run on the A100 yet.

## What the repo already gives us
- `scripts/record_demo.py` writes a LeRobot v3 dataset: `observation.state` (8), `action` (8), `observation.images.table` 640x480, 30 fps, `task` text. Matches what ACT, SmolVLA, Pi0 and GR00T training expect (camera name needs a rename map for SmolVLA, see below).
- `Guard.move()` in `src/normakit/core.py` clamps goals to 0.02..0.98, limits steps to 0.35 for non-gripper joints, rate limits to 10 Hz, estops on overcurrent. Every policy action must go through it.
- `docs/08_POLICY_AND_CLAUDE_PLAN.md`: Claude/Gemini decide what and where (seconds), a fast policy does the how (~30 Hz).

## Candidate comparison

| Model | Params | Latency per action chunk on 1 A100 | Demos needed | Fit to 8-DoF + 1 camera | LeRobot | License | Source |
|---|---|---|---|---|---|---|---|
| **ACT** | ~80M (ResNet-18 + transformer) | ~5-15 ms (estimate) | ~50 | Direct: any state/action dim, any camera count. No language. | Native, `--policy.type=act` | Apache-2.0 | [LeRobot ACT](https://huggingface.co/docs/lerobot/act) |
| **Diffusion Policy** | ~80-250M depending on config (estimate) | ~50-150 ms with default 100 denoise steps, less with DDIM (estimate) | ~50-200 | Direct, no language. Slower than ACT. | Native, `--policy.type=diffusion` | Apache-2.0 | [diffusion policy](https://diffusion-policy.cs.columbia.edu/) |
| **SmolVLA** | 450M | ~50-100 ms (estimate) | ~50, 5 object positions x 10 (HF reports 25 was not enough) | Good. Language conditioned. Expects camera names camera1..3, use `--rename_map`. State/action padded internally. | Native, `--policy.path=lerobot/smolvla_base` | Apache-2.0 | [LeRobot SmolVLA](https://huggingface.co/docs/lerobot/smolvla) (20k steps ~4 h on one A100) |
| **Pi0 / Pi0.5** | ~3B | ~100-200 ms (estimate, JAX or torch.compile) | 50+ with LoRA; HF/openpi examples use tens to hundreds | Works with custom robots but "may or may not work" per authors. Heavy for 1 camera, 8 DoF. | Supported in LeRobot, and openpi has its own websocket server | Apache-2.0 code (check weights terms) | [openpi](https://github.com/Physical-Intelligence/openpi): inference >8 GB, LoRA fine-tune >22.5 GB, full fine-tune >70 GB (A100 80GB) |
| **GR00T N1.5 / N1.6 / N1.7** | ~3B class (N1.7: Cosmos-Reason2-2B backbone + flow-matching DiT head) | ~50-150 ms (estimate; NVIDIA lists TensorRT export in N1.7) | Embodiment-specific fine-tune, tens of demos (estimate) | Needs a new embodiment tag and modality config for 8 DoF. Most setup work of all candidates. | Partial (policy in LeRobot is N1.5 era, verify) | NVIDIA license, check terms | [Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T): inference 1 GPU 16 GB+, fine-tune 40 GB+ recommended; ships a policy server. N1.7 action horizon 40 |
| **OpenVLA / OpenVLA-OFT** | 7B | OpenVLA ~150-250 ms on A100 (estimate); OFT faster with parallel decoding (estimate) | 50-150 | Single-arm 7-DoF EEF delta style. Awkward for joint-space 8 DoF. | Not native | MIT (code), Llama-2 terms for weights | [openvla-oft](https://github.com/moojink/openvla-oft) (not re-verified this session) |
| **Octo** | 27M / 93M | ~20-40 ms (estimate) | ~100 | Works, JAX only, aging. Weak in the V-JEPA 2 paper's pick-and-place table (Octo 10% on cup/box). | No | MIT | [octo](https://github.com/octo-models/octo), [V-JEPA 2 table](https://github.com/facebookresearch/vjepa2) |
| **V-JEPA 2 / 2-AC (world model)** | 300M to 1B encoder (ViT-L/H/g) | Planning by sampling action sequences, hundreds of ms to seconds (estimate) | Needs ~62 h robot data for AC post-training per paper (not re-verified) | Research use. Not a drop-in controller for our 8-DoF arm. Reported 100% reach, 60%/20% cup/box pick on Franka. | No | MIT code | [vjepa2](https://github.com/facebookresearch/vjepa2) |
| **NVIDIA Cosmos / Cosmos-Reason** | 2B to 7B+ | 100s of ms to seconds | n/a | World model / reasoning VLM, data generation and high-level reasoning. Cosmos-Reason2-2B is the GR00T N1.7 backbone. Not a low-level controller. | No | NVIDIA Open Model License | [Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T) |
| **Gemini Robotics-ER** (in use) | API | 1-5 s (measured in team notes, see `src/normakit/er2.py`) | none | Pointing and planning from images. Keep as high-level. | n/a | API terms | `src/normakit/er2.py` |
| **YOLO / Grounding DINO / OWL-ViT** | 3M to ~200M | YOLO <10 ms, Grounding DINO ~50-100 ms (estimates) | none (open vocab) | Perception only: object boxes into table coordinates via the existing calibration. | n/a | AGPL (Ultralytics YOLO, check), Apache-2.0 (Grounding DINO, OWL-ViT) | [Ultralytics](https://docs.ultralytics.com/), [OWL-ViT](https://huggingface.co/docs/transformers/model_doc/owlvit) |

`/home/laptop/src/GR00T-WholeBodyControl` only contains `docs` and `gear_sonic`. It is a humanoid whole-body controller, not relevant to a tabletop arm.

## Recommendation

**PICK: ACT.** Reasons: fastest inference of any trainable option, about 50 demos, trains natively on our dataset format with no renames, 8 DoF and one camera need no special configuration, and action chunks hide network latency. Fits the 2-day hackathon budget.

**FALLBACK / UPGRADE:**
1. **SmolVLA** if we need a text instruction ("pick the cup") from one model. Same `lerobot-train`, same dataset, roughly 4 h of A100 for 20k steps per HF. Use `--rename_map` to map `observation.images.table` to `observation.images.camera1`.
2. **GR00T N1.5/N1.6/N1.7** only if there is time for the embodiment-tag setup. It has an official policy server and TensorRT path, but the most integration work.
3. **Diffusion Policy** if ACT jitters on multimodal demos.

**Stay as-is:** Gemini ER-2 (and YOLO) for high-level perception and verification. They choose the target and the task string, the policy executes.

Skip for this hackathon: OpenVLA(-OFT) (7B, EEF-delta oriented), Octo (JAX, weak results), V-JEPA 2 and Cosmos (research world models, not controllers).

## Latency budget (Mac/laptop to A100)

All numbers are estimates. Measure with the client in FEAT-003.

| Stage | Estimate |
|---|---|
| Camera capture + JPEG encode 640x480 (quality ~80, ~40-60 KB) | 10-20 ms |
| Uplink laptop to A100 (WAN RTT, upload at 10+ Mbit/s) | 30-120 ms RTT |
| Server decode + preprocess | 5-10 ms |
| ACT forward (one chunk, e.g. 50 steps) | 5-15 ms |
| Downlink (chunk of 50x8 floats, ~2 KB) | included in RTT |
| **Total per chunk** | **~60-170 ms** |

Action chunking is what makes this work. ACT predicts k future actions per call. At 30 fps, a chunk of 50 covers ~1.6 s of motion, far longer than the 60-170 ms round trip. The real limit is the **Guard 10 Hz rate limit**: the arm receives goals at 10 Hz, so `scripts/run_policy.py` interpolates the chunk in time. Each 0.1 s tick advances `0.1 / chunk_dt` rows (3 rows at 30 fps), so playback runs at the demonstrated speed. v1 fetches synchronously: once half the chunk has been played it blocks on the next `/act` while the arm holds its last goal (one RTT, 60-170 ms estimated). The second half of each chunk is dropped because it was predicted from an older observation. Asynchronous prefetch and ACT temporal ensembling are possible upgrades once the RTT is measured. `--max-delta` (default 0.05 norm per tick, 0.5 norm/s at 10 Hz) caps speed below the Guard's 0.35 step limit; raise it if fast demonstrated motions lag.

Also keep server and robot clocks out of the control loop. The client owns the clock and the Guard.

## Risks (honest)
- **Elbow joints 3/4 did not follow commands** in the last real-arm test (`docs/HANDOVER.md`: j3 0 ticks for +-40 command, j4 13 ticks). A policy cannot work around a joint that does not move. Demos recorded with this fault would teach bad actions. Fix or rule out the hardware first.
- **Leader j4 range odd** (ISSUE-114) and follower j4 moves ~1.8x the leader under Station percent mirroring. That corrupts the action labels. Check before recording.
- **Video decode problem** (`docs/08`): TorchCodec fails locally (`libnvrtc.so.13` missing), PyAV fallback lacks `VideoReader`. Training happens on the A100, where a matching TorchCodec install should work, but this is untested. Fallback is recording with `use_videos=False` (PNG, large).
- **Real recording is untested**; only synthetic data was written so far.
- **Calibration must stay frozen** between recording and deployment, otherwise normalised values shift.
- **Single camera, fixed table**: ACT generalises poorly to new positions. Record each object position about 10 times.
- **Unknown A100 host**: network path, ports and firewall are placeholders (`A100_HOST`). Use an SSH tunnel rather than exposing the port.
- **No real motion without a human at the E-stop.** Test the client against the sim backend and the server in `--dummy` mode first.

## User steps
Files marked FEAT-002/003 are written in the next features of this task and may not exist yet.

1. **A100 setup.** On the A100 host (`A100_HOST`, user and key are yours, nothing is stored in the repo): create a venv, `pip install "lerobot==0.4.4"` (same as the laptop) plus `fastapi uvicorn`. For SmolVLA add `pip install -e ".[smolvla]"` from a LeRobot checkout. Check `nvidia-smi`.
2. **Fix hardware first.** Resolve elbow j3/j4 and leader j4 (see Risks).
3. **Record demos** on the laptop with leader to follower mirroring: `.venv/bin/python scripts/record_demo.py` (about 50 episodes, 5 object positions x 10). Do not recalibrate afterwards.
4. **Upload the dataset:** `rsync -avz <dataset_dir> $A100_USER@$A100_HOST:~/datasets/elrobot_demo/`
5. **Train on the A100:**
   ```
   lerobot-train --dataset.repo_id=local/elrobot_demo --dataset.root=~/datasets/elrobot_demo \
     --policy.type=act --policy.device=cuda --policy.push_to_hub=false \
     --output_dir=outputs/train/act_elrobot --job_name=act_elrobot --batch_size=16 --steps=100000
   ```
   SmolVLA variant: `--policy.path=lerobot/smolvla_base --rename_map='{"observation.images.table": "observation.images.camera1"}' --steps=20000`.
6. **Start the server** on the A100 (FEAT-002, `server/`): first `--dummy` to test the API, then with `--checkpoint outputs/train/act_elrobot/checkpoints/last/pretrained_model`. Open it via tunnel: `ssh -L 8000:localhost:8000 $A100_USER@$A100_HOST`.
7. **Run the client** on the laptop (FEAT-003, `scripts/run_policy.py`): first against `SimBackend`, then on the real arm only with a person at the E-stop. Every action goes through `Guard.move()`.

## Sources checked in this session
[LeRobot ACT](https://huggingface.co/docs/lerobot/act), [LeRobot SmolVLA](https://huggingface.co/docs/lerobot/smolvla), [openpi](https://github.com/Physical-Intelligence/openpi), [Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T), [V-JEPA 2](https://github.com/facebookresearch/vjepa2). OpenVLA-OFT, Octo, Diffusion Policy, Cosmos and detector rows rely on prior knowledge and are labelled estimates. Re-verify before relying on them.
