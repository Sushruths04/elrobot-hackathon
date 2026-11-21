#!/usr/bin/env python
"""Train a box-conditioned throw policy in the MuJoCo twin with CEM (SIM ONLY, never touches hardware).

  .venv/bin/python scripts/train_throw.py --smoke
  .venv/bin/python scripts/train_throw.py --iters 25 --pop 48 --eps-per-cand 4 --workers 16 --out models/throw/

The policy is a 27-number parametric release trajectory linear in the box position (src/normakit/throw_policy.py).
The simulator runs on CPU; --device is accepted for A100 hosts but CEM needs no GPU (see docs/11_THROW_INTO_BOX.md).
Outputs in --out: policy.npz (best theta), train_log.json, throw_dataset.npz (successful rollouts: obs, action,
episode id; convert to LeRobot format later, see docs/11).
"""
import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from normakit.throw_env import ThrowIntoBoxEnv  # noqa: E402
from normakit.throw_policy import THETA_DIM, rollout  # noqa: E402

_env = None


def _work(args):
    global _env
    theta, seeds = args
    if _env is None:
        _env = ThrowIntoBoxEnv(seed=0)
    res = [rollout(_env, theta, seed=s)[:2] for s in seeds]
    return float(np.mean([r[1] for r in res])), float(np.mean([r[0] for r in res]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--smoke", action="store_true", help="tiny CPU run (2 iters)")
    ap.add_argument("--device", default="cpu", help="accepted for A100 hosts; CEM runs on CPU cores")
    ap.add_argument("--episodes", type=int, default=0, help="stop once this many sim episodes are used (0 = use --iters)")
    ap.add_argument("--iters", type=int, default=25)
    ap.add_argument("--pop", type=int, default=48)
    ap.add_argument("--eps-per-cand", type=int, default=4)
    ap.add_argument("--elite-frac", type=float, default=0.15)
    ap.add_argument("--workers", type=int, default=max(1, min(16, (mp.cpu_count() or 2) - 2)))
    ap.add_argument("--dataset-episodes", type=int, default=100, help="rollouts tried for the successful-rollout dataset")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="models/throw/")
    a = ap.parse_args()
    if a.smoke:
        a.iters, a.pop, a.eps_per_cand, a.workers, a.dataset_episodes = 2, 6, 2, 2, 6
    if a.episodes:
        a.iters = max(1, a.episodes // (a.pop * a.eps_per_cand))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(a.seed)
    mu, sd = np.zeros(THETA_DIM), np.full(THETA_DIM, 1.5)
    n_el = max(2, int(a.pop * a.elite_frac))
    best_theta, best_score, log = mu.copy(), -1e9, []
    t0 = time.time()
    pool = mp.get_context("spawn").Pool(a.workers)
    try:
        for it in range(a.iters):
            cands = [mu + sd * rng.standard_normal(THETA_DIM) for _ in range(a.pop)]
            jobs = [(c, [int(rng.integers(1 << 30)) for _ in range(a.eps_per_cand)]) for c in cands]
            res = pool.map(_work, jobs)
            sc = np.array([r[0] for r in res])
            sr = np.array([r[1] for r in res])
            idx = np.argsort(-sc)[:n_el]
            mu = 0.5 * mu + 0.5 * np.mean([cands[i] for i in idx], axis=0)
            sd = 0.5 * sd + 0.5 * np.std([cands[i] for i in idx], axis=0) + 0.05
            if sc[idx[0]] > best_score:
                best_score, best_theta = float(sc[idx[0]]), cands[idx[0]].copy()
            row = dict(iter=it, mean_reward=float(sc.mean()), best_reward=float(sc.max()),
                       mean_success=float(sr.mean()), best_success=float(sr[idx[0]]), t=round(time.time() - t0, 1))
            log.append(row)
            print(row, flush=True)
    finally:
        pool.terminate()

    # CEM scores are noisy (few episodes per candidate): pick the final policy by a re-evaluation of the
    # best sampled candidate and the final mean on fresh seeds.
    pool = mp.get_context("spawn").Pool(a.workers)
    try:
        sel_seeds = [10_000 + i for i in range(max(6, a.eps_per_cand * 3))]
        cand = [best_theta, mu]
        res = pool.map(_work, [(c, sel_seeds) for c in cand])
    finally:
        pool.terminate()
    theta = cand[int(np.argmax([r[0] for r in res]))]
    print("selection (best-sample, final-mean) reward/success:", res)
    np.savez(out / "policy.npz", theta=theta)
    (out / "train_log.json").write_text(json.dumps(
        {"args": vars(a), "log": log, "selection": res, "seconds": round(time.time() - t0, 1)}, indent=2))

    # Dataset of successful rollouts for later ACT distillation.
    env = ThrowIntoBoxEnv(seed=0)
    from normakit.throw_policy import rollout as ro
    S, A, E = [], [], []
    ok = 0
    for k in range(a.dataset_episodes):
        s, _, _, rec = ro(env, theta, seed=50_000 + k, record=True)
        if s:
            ok += 1
            S += [r[0] for r in rec]
            A += [r[1] for r in rec]
            E += [ok - 1] * len(rec)
    if S:
        np.savez_compressed(out / "throw_dataset.npz", obs=np.array(S), action=np.array(A), episode=np.array(E))
    print(f"dataset: {ok}/{a.dataset_episodes} successful rollouts, {len(S)} frames -> {out}")
    print("saved", out / "policy.npz")


if __name__ == "__main__":
    main()
