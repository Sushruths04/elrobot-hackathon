"""Train ACT or SmolVLA on the merged spoon demonstrations inside a cluster GPU allocation."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pyarrow.parquet as pq
from lerobot.datasets.compute_stats import aggregate_stats
from lerobot.datasets.utils import unflatten_dict, write_stats


def prepare_run(dataset, run):
    manifest = json.loads((dataset / "TRAIN_MANIFEST.json").read_text())
    usable = manifest["train_episodes"]
    validation = [ep for ep in usable if manifest["sources"][ep].split()[0] == "110808"]
    training = [ep for ep in usable if ep not in validation]
    if not training or not validation:
        raise ValueError("Expected training demonstrations and held-out session 110808")
    stats = {}
    for path in sorted((dataset / "meta/episodes").rglob("*.parquet")):
        for row in pq.read_table(path).to_pylist():
            ep = row["episode_index"]
            if ep in training:
                stats[ep] = unflatten_dict({k.removeprefix("stats/"): np.asarray(v)
                                            for k, v in row.items() if k.startswith("stats/")})
    combined = aggregate_stats([stats[ep] for ep in training])
    # LeRobot 0.4.4 selects samples by episode but reads normalization from meta/stats.json.
    # Keep a run-specific metadata view so the validation session cannot affect these statistics.
    view = run / "dataset"
    run.mkdir(parents=True, exist_ok=False)
    (view / "meta").mkdir(parents=True)
    for name in ("data", "videos"):
        (view / name).symlink_to(dataset / name)
    for path in (dataset / "meta").iterdir():
        if path.name != "stats.json":
            (view / "meta" / path.name).symlink_to(path)
    write_stats(combined, view)
    split = {"train": training, "validation": validation}
    (run / "episodes.json").write_text(json.dumps(split, indent=2) + "\n")
    return view, split, manifest["rename_map"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("run", type=Path, help="New run directory on persistent storage")
    parser.add_argument("--policy", choices=("act", "smolvla"), default="act")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    if min(args.steps, args.batch_size) < 1:
        parser.error("steps and batch size must both be positive")
    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    if not torch.cuda.is_available():
        sys.exit("No CUDA device visible. Run this inside the cluster GPU allocation.")
    run = args.run.resolve()
    view, split, rename = prepare_run(args.dataset.resolve(), run)
    ds = LeRobotDataset("local/spoon_beans_v1", root=view, episodes=split["train"], video_backend="pyav")
    sample = ds[0]
    for key in ("observation.images.table", "observation.images.wrist"):
        if tuple(sample[key].shape) != (3, 480, 640):
            raise ValueError(f"Unexpected camera input for {key}: {sample[key].shape}")
    print(torch.cuda.get_device_name(), split, flush=True)
    if args.policy == "act":
        policy_args = ["--policy.type=act", "--policy.chunk_size=30", "--policy.n_action_steps=30"]
    else:
        policy_args = ["--policy.path=lerobot/smolvla_base", f"--rename_map={json.dumps(rename)}"]
    subprocess.run([sys.executable, "-m", "lerobot.scripts.lerobot_train", *policy_args,
                    "--dataset.repo_id=local/spoon_beans_v1", f"--dataset.root={view}",
                    f"--dataset.episodes={json.dumps(split['train'])}", "--dataset.video_backend=pyav",
                    "--policy.device=cuda", "--policy.push_to_hub=false", "--wandb.enable=false",
                    f"--steps={args.steps}", f"--batch_size={args.batch_size}", "--num_workers=4",
                    f"--save_freq={min(1000, args.steps)}", f"--log_freq={min(50, args.steps)}",
                    f"--output_dir={run / 'training'}"], check=True)


if __name__ == "__main__":
    main()
