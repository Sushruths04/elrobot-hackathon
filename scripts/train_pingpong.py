"""Train & evaluate 30 Hz PPO policies for ElRobot flat-paddle ball balancing and dribbling.

Examples:
  # 1) Train Paddle-Balance policy (~2-3 min on 12 CPU cores), evaluate on camera pipeline, and render video:
  .venv/bin/python scripts/train_pingpong.py --task balance --iters 180 --workers 12 \
      --out models/pingpong/balance_ppo.pt --eval-camera 20 --video results/pingpong_balance.mp4

  # 2) Train / fine-tune Paddle-Dribble (Keep-Up) policy from scratch or warm-started from balance_ppo.pt:
  .venv/bin/python scripts/train_pingpong.py --task dribble --init models/pingpong/balance_ppo.pt \
      --iters 260 --workers 12 --out models/pingpong/dribble_ppo.pt --eval-camera 20 --video results/pingpong_dribble.mp4
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("LP_NUM_THREADS", "2")
os.environ.setdefault("MUJOCO_GL", "egl")

import cv2
import gymnasium as gym
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from normakit.pingpong_env import ACTIVE_JIDS, HOME_Q, ElRobotPingPongEnv
from normakit.ppo import PPOActorCritic, compute_gae, ppo_update


def make_env_fn(task: str, seed: int, spawn_scale: float = 1.0):
    def _thunk():
        return ElRobotPingPongEnv(
            task=task,
            obs_mode="controller",
            domain_rand=True,
            ball_noise=1.0,
            spawn_scale=spawn_scale,
            light_model=True,
            seed=seed,
        )
    return _thunk


def evaluate_policy(
    ac: PPOActorCritic,
    task: str,
    n_episodes: int = 30,
    obs_mode: str = "controller",
    seed: int = 9000,
    video_path: Path | None = None,
) -> dict:
    env = ElRobotPingPongEnv(
        task=task,
        obs_mode=obs_mode,
        domain_rand=True,
        ball_noise=1.0,
        spawn_scale=1.0,
        light_model=(obs_mode == "controller"),
        seed=seed,
    )
    ac.eval()
    successes = 0
    ep_steps = []
    ep_bounces = []
    final_r_cm = []
    vid_frames: list[np.ndarray] = []

    # If rendering a video in controller mode, create a full visual env for episode 0
    vis_env = None
    if video_path is not None and obs_mode == "controller":
        vis_env = ElRobotPingPongEnv(task=task, obs_mode="controller", domain_rand=True, ball_noise=1.0, light_model=False, seed=seed)

    for ep in range(n_episodes):
        use_env = vis_env if (ep == 0 and vis_env is not None) else env
        obs, _ = use_env.reset(seed=seed + ep * 97)
        done = False
        info = {}
        while not done:
            if ep == 0 and video_path is not None:
                rgb = use_env.render()
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                cv2.putText(
                    bgr,
                    f"ElRobot {task.upper()} ({obs_mode}) t={use_env.t_step/30:.1f}s bounces={use_env.n_bounces}",
                    (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 180), 2,
                )
                vid_frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            act_t, _, _ = ac.act(obs_t, deterministic=True)
            obs, _, term, trunc, info = use_env.step(act_t.squeeze(0).cpu().numpy())
            done = bool(term or trunc)
        successes += int(bool(info.get("success", False)))
        ep_steps.append(int(info.get("steps", 0)))
        ep_bounces.append(int(info.get("bounces", 0)))
        final_r_cm.append(float(info.get("r_uv_cm", 0.0)))

    if video_path is not None and vid_frames:
        video_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            import av
            container = av.open(str(video_path), mode="w")
            stream = container.add_stream("libx264", rate=30)
            stream.height, stream.width = vid_frames[0].shape[:2]
            stream.pix_fmt = "yuv420p"
            stream.options = {"crf": "22", "preset": "veryfast"}
            for f in vid_frames:
                vf = av.VideoFrame.from_ndarray(f, format="rgb24")
                for pkt in stream.encode(vf):
                    container.mux(pkt)
            for pkt in stream.encode():
                container.mux(pkt)
            container.close()
        except Exception:
            h, w = vid_frames[0].shape[:2]
            vw = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h))
            for f in vid_frames:
                vw.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
            vw.release()

    if vis_env is not None:
        vis_env.close()
    env.close()
    return {
        "episodes": n_episodes,
        "obs_mode": obs_mode,
        "successes": successes,
        "success_rate_pct": round(100.0 * successes / max(1, n_episodes), 2),
        "mean_steps": round(float(np.mean(ep_steps)), 1),
        "mean_survival_s": round(float(np.mean(ep_steps)) / 30.0, 2),
        "mean_bounces": round(float(np.mean(ep_bounces)), 2),
        "max_bounces": int(max(ep_bounces)) if ep_bounces else 0,
        "median_final_r_cm": round(float(np.median(final_r_cm)), 2),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["balance", "dribble"], default="balance")
    ap.add_argument("--iters", type=int, default=180)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--envs-per-worker", type=int, default=2)
    ap.add_argument("--steps", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--init", default="", help="optional checkpoint to warm-start / fine-tune from")
    ap.add_argument("--out", default="")
    ap.add_argument("--eval-episodes", type=int, default=40)
    ap.add_argument("--eval-camera", type=int, default=20, help="episodes to evaluate with full 640x480 front-camera HSV+3D tracker")
    ap.add_argument("--video", default="")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    torch.set_num_threads(4)

    out_pt = (Path(a.out) if a.out else (ROOT / f"models/pingpong/{a.task}_ppo.pt")).resolve()
    out_pt.parent.mkdir(parents=True, exist_ok=True)

    n_envs = a.workers * a.envs_per_worker
    env_fns = [make_env_fn(a.task, a.seed + 1000 * i) for i in range(n_envs)]
    vec_env = gym.vector.AsyncVectorEnv(env_fns, context="spawn")

    obs_dim = int(vec_env.single_observation_space.shape[0])
    act_dim = int(vec_env.single_action_space.shape[0])
    init_std = 0.24 if a.task == "balance" else 0.42
    ac = PPOActorCritic(obs_dim=obs_dim, act_dim=act_dim, hidden=256, init_std=init_std)

    if a.init and Path(a.init).exists():
        ckpt = torch.load(a.init, map_location="cpu", weights_only=False)
        ac.load_state_dict(ckpt["model_state_dict"])
        # Re-open exploration std slightly only when cross-task fine-tuning (balance -> dribble)
        if ckpt.get("task") != a.task:
            with torch.no_grad():
                ac.log_std.fill_(float(np.log(0.42)))
        print(f"Loaded warm-start weights from {a.init} (source task={ckpt.get('task')})", flush=True)

    opt = torch.optim.Adam(ac.parameters(), lr=a.lr, eps=1e-5)

    obs, _ = vec_env.reset(seed=a.seed)
    T = a.steps
    obs_buf = np.zeros((T, n_envs, obs_dim), dtype=np.float32)
    act_buf = np.zeros((T, n_envs, act_dim), dtype=np.float32)
    logp_buf = np.zeros((T, n_envs), dtype=np.float32)
    rew_buf = np.zeros((T, n_envs), dtype=np.float32)
    val_buf = np.zeros((T, n_envs), dtype=np.float32)
    val_next_buf = np.zeros((T, n_envs), dtype=np.float32)
    done_buf = np.zeros((T, n_envs), dtype=np.float32)

    best_score = -1e9
    best_state = {k: v.cpu().clone() for k, v in ac.state_dict().items()}
    ep_returns: list[float] = []
    ep_succs: list[int] = []
    ep_bounces: list[int] = []
    running_ret = np.zeros(n_envs, dtype=np.float64)
    t0 = time.perf_counter()

    for it in range(1, a.iters + 1):
        ac.eval()
        # Cosine-anneal lr gently
        frac = 1.0 - (it - 1.0) / max(1, a.iters)
        for pg in opt.param_groups:
            pg["lr"] = a.lr * max(0.15, frac)

        for t in range(T):
            obs_buf[t] = obs
            obs_t = torch.as_tensor(obs, dtype=torch.float32)
            act_t, logp_t, val_t = ac.act(obs_t, deterministic=False)
            act_np = act_t.cpu().numpy()
            act_buf[t] = act_np
            logp_buf[t] = logp_t.cpu().numpy()
            val_buf[t] = val_t.cpu().numpy()

            next_obs, rew, term, trunc, infos = vec_env.step(act_np)
            rew_buf[t] = rew
            done = np.logical_or(term, trunc)
            done_buf[t] = done.astype(np.float32)
            running_ret += rew

            with torch.no_grad():
                v_next = ac.value(torch.as_tensor(next_obs, dtype=torch.float32)).cpu().numpy()
            # On true failure (term=True), bootstrap value is 0; on time-limit truncation, bootstrap from final_observation!
            if "final_observation" in infos:
                for ei in range(n_envs):
                    if trunc[ei] and not term[ei] and infos["final_observation"][ei] is not None:
                        fo = torch.as_tensor(infos["final_observation"][ei], dtype=torch.float32).unsqueeze(0)
                        with torch.no_grad():
                            v_next[ei] = float(ac.value(fo).item())
                    elif term[ei]:
                        v_next[ei] = 0.0
            else:
                v_next = np.where(term, 0.0, v_next)
            val_next_buf[t] = v_next

            if "final_info" in infos:
                for ei in range(n_envs):
                    if done[ei] and infos["final_info"][ei] is not None:
                        fi = infos["final_info"][ei]
                        ep_returns.append(float(running_ret[ei]))
                        ep_succs.append(int(bool(fi.get("success", False))))
                        ep_bounces.append(int(fi.get("bounces", 0)))
                        running_ret[ei] = 0.0
            else:
                for ei in range(n_envs):
                    if done[ei]:
                        ep_returns.append(float(running_ret[ei]))
                        if "success" in infos:
                            ep_succs.append(int(bool(infos["success"][ei])))
                        if "bounces" in infos:
                            ep_bounces.append(int(infos["bounces"][ei]))
                        running_ret[ei] = 0.0

            obs = next_obs

        adv_buf, ret_buf = compute_gae(rew_buf, val_buf, val_next_buf, done_buf, gamma=0.99, lam=0.95)
        ac.train()
        batch = {
            "obs": torch.as_tensor(obs_buf.reshape(-1, obs_dim), dtype=torch.float32),
            "act": torch.as_tensor(act_buf.reshape(-1, act_dim), dtype=torch.float32),
            "logp": torch.as_tensor(logp_buf.reshape(-1), dtype=torch.float32),
            "adv": torch.as_tensor(adv_buf.reshape(-1), dtype=torch.float32),
            "ret": torch.as_tensor(ret_buf.reshape(-1), dtype=torch.float32),
        }
        upd = ppo_update(ac, opt, batch, epochs=8, minibatches=8, ent_coef=0.002 if a.task == "dribble" else 0.0005)

        if len(ep_returns) >= 20:
            mean_ret = float(np.mean(ep_returns[-60:]))
            succ_rate = 100.0 * float(np.mean(ep_succs[-60:])) if ep_succs else 0.0
            mean_bnc = float(np.mean(ep_bounces[-60:])) if ep_bounces else 0.0
            score = succ_rate + 0.05 * mean_ret + (2.0 * mean_bnc if a.task == "dribble" else 0.0)
            if score > best_score:
                best_score = score
                best_state = {k: v.cpu().clone() for k, v in ac.state_dict().items()}
                torch.save(
                    {
                        "task": a.task,
                        "obs_dim": obs_dim,
                        "act_dim": act_dim,
                        "active_jids": ACTIVE_JIDS,
                        "home_q": HOME_Q,
                        "action_scale": 0.028 if a.task == "balance" else 0.080,
                        "model_state_dict": best_state,
                        "iter": it,
                        "train_succ_pct": round(succ_rate, 2),
                        "train_mean_bounces": round(mean_bnc, 2),
                    },
                    out_pt,
                )

        if it % 15 == 0 or it == 1 or it == a.iters:
            steps_done = it * T * n_envs
            fps = steps_done / max(1e-3, time.perf_counter() - t0)
            mr = float(np.mean(ep_returns[-60:])) if ep_returns else 0.0
            sr = 100.0 * float(np.mean(ep_succs[-60:])) if ep_succs else 0.0
            mb = float(np.mean(ep_bounces[-60:])) if ep_bounces else 0.0
            print(
                f"[{a.task} iter {it:3d}/{a.iters}] steps={steps_done:7d} ({fps:5.0f} sps) | "
                f"ret={mr:6.1f} succ={sr:5.1f}% bounces={mb:4.1f} | kl={upd['kl']:.4f}",
                flush=True,
            )

    vec_env.close()

    # Load best checkpoint and evaluate on BOTH controller state AND full 640x480 external front-camera pipeline!
    ac.load_state_dict(best_state)
    torch.save(
        {
            "task": a.task,
            "obs_dim": obs_dim,
            "act_dim": act_dim,
            "active_jids": ACTIVE_JIDS,
            "home_q": HOME_Q,
            "action_scale": 0.042 if a.task == "balance" else 0.085,
            "model_state_dict": best_state,
        },
        out_pt,
    )
    print(f"\nSaved best {a.task} checkpoint -> {out_pt.relative_to(ROOT)}", flush=True)

    eval_state = evaluate_policy(ac, a.task, n_episodes=a.eval_episodes, obs_mode="controller", seed=80000)
    print("Eval (randomised sim + noisy FdVel):", json.dumps(eval_state), flush=True)

    vid_p = (Path(a.video) if a.video else (ROOT / f"results/pingpong_{a.task}.mp4")).resolve()
    eval_cam = evaluate_policy(
        ac, a.task, n_episodes=a.eval_camera, obs_mode="camera", seed=90000, video_path=vid_p
    )
    print("Eval (640x480 front-camera HSV + 3D tracker):", json.dumps(eval_cam), flush=True)

    summary_path = ROOT / f"results/pingpong_{a.task}_eval.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps({"task": a.task, "checkpoint": str(out_pt.relative_to(ROOT)), "state_eval": eval_state, "camera_eval": eval_cam, "video": str(vid_p.relative_to(ROOT))}, indent=2) + "\n")


if __name__ == "__main__":
    main()
