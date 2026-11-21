#!/usr/bin/env python
"""Evaluate a trained throw policy in sim over random box positions (SIM ONLY).

  .venv/bin/python scripts/eval_throw.py --episodes 100 [--policy models/throw/policy.npz]
Writes results/throw_eval.json. Eval seeds (900000+) differ from training seeds.
"""
import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from normakit.throw_env import ThrowIntoBoxEnv  # noqa: E402
from normakit.throw_policy import THETA_DIM, rollout  # noqa: E402

_env = None


def _work(args):
    global _env
    theta, seeds = args
    if _env is None:
        _env = ThrowIntoBoxEnv(seed=0)
    out = []
    for s in seeds:
        ok, r, err, _ = rollout(_env, theta, seed=s)
        out.append((ok, r, err, [float(v) for v in _env.box_xy]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--policy", default="models/throw/policy.npz")
    ap.add_argument("--workers", type=int, default=max(1, min(16, (mp.cpu_count() or 2) - 2)))
    ap.add_argument("--baseline", action="store_true", help="also evaluate a zero (untrained) policy")
    a = ap.parse_args()
    theta = np.load(ROOT / a.policy if not Path(a.policy).is_absolute() else a.policy)["theta"]
    seeds = [900_000 + i for i in range(a.episodes)]
    chunks = [seeds[i::a.workers] for i in range(a.workers)]
    t0 = time.time()

    def run(th):
        with mp.get_context("spawn").Pool(a.workers) as p:
            return [r for part in p.map(_work, [(th, c) for c in chunks if c]) for r in part]

    res = run(theta)
    succ = np.array([r[0] for r in res])
    summary = {
        "policy": a.policy, "episodes": len(res), "successes": int(succ.sum()),
        "success_rate": float(succ.mean()), "mean_reward": float(np.mean([r[1] for r in res])),
        "mean_miss_distance_m": float(np.mean([r[2] for r in res])),
        "seconds": round(time.time() - t0, 1), "sim_only": True,
    }
    if a.baseline:
        b = run(np.zeros(THETA_DIM))
        summary["baseline_zero_policy_success_rate"] = float(np.mean([r[0] for r in b]))
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results/throw_eval.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"success rate {summary['success_rate']*100:.1f}% ({summary['successes']}/{summary['episodes']})")


if __name__ == "__main__":
    main()
