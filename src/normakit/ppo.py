"""Clean PyTorch PPO (orthogonal-init Actor-Critic MLP, GAE, target-KL clipped updates)
adapted from zecice/so101-pingpong-balance (MIT) for 6-DoF ElRobot paddle balance & dribbling."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def build_mlp(in_dim: int, out_dim: int, hidden: int = 256, layers: int = 2, out_gain: float = 1.0) -> nn.Sequential:
    mods: list[nn.Module] = []
    d = in_dim
    for _ in range(layers):
        lin = nn.Linear(d, hidden)
        nn.init.orthogonal_(lin.weight, float(np.sqrt(2.0)))
        nn.init.zeros_(lin.bias)
        mods.extend([lin, nn.Tanh()])
        d = hidden
    last = nn.Linear(d, out_dim)
    nn.init.orthogonal_(last.weight, out_gain)
    nn.init.zeros_(last.bias)
    mods.append(last)
    return nn.Sequential(*mods)


class PPOActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int = 6, hidden: int = 256, init_std: float = 0.45):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.actor = build_mlp(obs_dim, act_dim, hidden=hidden, out_gain=0.01)
        self.critic = build_mlp(obs_dim, 1, hidden=hidden, out_gain=1.0)
        self.log_std = nn.Parameter(torch.full((act_dim,), float(np.log(init_std))))

    def dist(self, obs: torch.Tensor) -> torch.distributions.Normal:
        std = self.log_std.clamp(-2.8, 0.5).exp()
        return torch.distributions.Normal(self.actor(obs), std)

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs).squeeze(-1)

    @torch.no_grad()
    def act(self, obs: torch.Tensor, deterministic: bool = False):
        if deterministic:
            return self.actor(obs), None, self.value(obs)
        d = self.dist(obs)
        a = d.sample()
        return a, d.log_prob(a).sum(-1), self.value(obs)


def compute_gae(
    rew: np.ndarray,
    val: np.ndarray,
    val_next: np.ndarray,
    done: np.ndarray,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> tuple[np.ndarray, np.ndarray]:
    T = rew.shape[0]
    delta = rew + gamma * val_next - val
    adv = np.zeros_like(rew)
    last = np.zeros_like(rew[0])
    for t in reversed(range(T)):
        last = delta[t] + gamma * lam * (1.0 - done[t]) * last
        adv[t] = last
    return adv, adv + val


def ppo_update(
    ac: PPOActorCritic,
    opt: torch.optim.Optimizer,
    batch: dict[str, torch.Tensor],
    epochs: int = 8,
    minibatches: int = 8,
    clip: float = 0.2,
    vf_coef: float = 0.5,
    ent_coef: float = 0.002,
    max_grad: float = 0.5,
    target_kl: float = 0.03,
) -> dict[str, float]:
    n = batch["obs"].shape[0]
    mb = max(32, n // minibatches)
    stats = {"pg": 0.0, "vf": 0.0, "ent": 0.0, "kl": 0.0, "clipfrac": 0.0, "steps": 0}
    for _ in range(epochs):
        perm = torch.randperm(n, device=batch["obs"].device)
        for start in range(0, n - mb + 1, mb):
            idx = perm[start : start + mb]
            d = ac.dist(batch["obs"][idx])
            logp = d.log_prob(batch["act"][idx]).sum(-1)
            log_ratio = logp - batch["logp"][idx]
            ratio = log_ratio.exp()
            adv = batch["adv"][idx]
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            pg = -torch.min(ratio * adv, ratio.clamp(1.0 - clip, 1.0 + clip) * adv).mean()
            v = ac.value(batch["obs"][idx])
            vf = 0.5 * (v - batch["ret"][idx]).pow(2).mean()
            ent = d.entropy().sum(-1).mean()
            loss = pg + vf_coef * vf - ent_coef * ent
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(ac.parameters(), max_grad)
            opt.step()
            with torch.no_grad():
                kl = ((ratio - 1.0) - log_ratio).mean()
                stats["pg"] += float(pg.item())
                stats["vf"] += float(vf.item())
                stats["ent"] += float(ent.item())
                stats["kl"] += float(kl.item())
                stats["clipfrac"] += float(((ratio - 1.0).abs() > clip).float().mean().item())
                stats["steps"] += 1
        if stats["steps"] > 0 and (stats["kl"] / stats["steps"]) > target_kl:
            break
    k = max(1, int(stats.pop("steps")))
    return {key: val / k for key, val in stats.items()}
