# Plan: learned policy + Claude API on the real arms (written 2026-11-21 ~15:50)

**Split of work:** Claude (API, 2-6 s per call) decides *what/where* and checks the result. A fast low-level policy (learned from teleop demos, or scripted IK) does the *how* at ~30 Hz. Claude cannot do 30 Hz control.

## Pipeline and status
| Step | Tool | Status |
|---|---|---|
| 0. Stable hardware (hub, cameras) | - | BLOCKER: hub dropped, camera also dropped on a direct port |
| 1. Camera calibration (official OpenCV board) | `scripts/calibrate_camera.py` | script ready; needs camera + printed board |
| 2. Record demos (leader->follower mirroring ON) | `scripts/record_demo.py` | dataset writing TESTED with synthetic data; real recording untested |
| 3. Train a policy | `lerobot-train` (LeRobot 0.4.4 installed) | NOT run; see decode problem below |
| 4. Run policy on the follower + Guard | to write: `scripts/run_policy.py` | not written |
| 5. Claude API agent (tool use, looks at camera, picks task, verifies) | `src/normakit/agent.py` | written, never run (needs `ANTHROPIC_API_KEY` in the shell env, never in files) |
| 6. MCP for Claude Code / Agent SDK | `src/normakit/mcp_server.py`, `.mcp.json` `elrobot-real` | registered, motion untested; `claude_agent_sdk` not installed |

## Recording format
LeRobot dataset v3: `observation.state` = follower joints 0..1 (8), `action` = leader joints 0..1 (8), `observation.images.table` 640x480, `task` text per frame. Normalised to each joint's calibrated range, so the follower can be commanded with `send_goals` directly. Calibration must stay frozen between recording and deployment.

## Known problems found while testing the pipeline (no hardware)
- **Reading videos back fails locally:** TorchCodec cannot load (`libnvrtc.so.13` missing; torch is `2.12+cu126`), and the PyAV fallback fails because this torchvision has no `VideoReader`. Dataset *writing* works (AV1 mp4). Fix options: install a matching TorchCodec / CUDA-13 nvrtc, train on a machine where `lerobot-train` already works, or record with `use_videos=False` (PNG images, large on disk).
- **GPU here:** RTX A2000 8 GB. ACT (~50 M params) should fit; SmolVLA (450 M) is tight. Not tried.
- **Claude API:** `api.anthropic.com` resolves from this machine. Key presence not checked (kept out of scope on purpose).

## Time budget (estimates, not measured)
Camera calibration 15 min -> 30-50 demos x ~28 s incl. reset = 15-25 min -> training 1-3 h (GPU dependent) -> deploy/debug 1 h. Freeze features Sunday 12:00.
