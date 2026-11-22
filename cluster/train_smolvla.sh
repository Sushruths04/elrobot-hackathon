#!/usr/bin/env bash
# Runs INSIDE the smolvla-train pod. Installs lerobot 0.4.4 (same version that wrote the dataset), then fine-tunes SmolVLA.
#   bash /data/smolvla/train_smolvla.sh spoon_beans_v1 [steps]
set -euo pipefail
NAME=${1:?dataset folder name under /data/smolvla/datasets}; STEPS=${2:-20000}
DS=/data/smolvla/datasets/$NAME; OUT=/data/smolvla/outputs/smolvla_${NAME}_$(date +%Y%m%d_%H%M)
test -f $DS/TRAIN_MANIFEST.json || { echo "missing $DS/TRAIN_MANIFEST.json"; exit 1; }
EPISODES=$(python -c "import json;print(json.dumps(json.load(open('$DS/TRAIN_MANIFEST.json'))['train_episodes']))")
RENAME=$(python -c "import json;print(json.dumps(json.load(open('$DS/TRAIN_MANIFEST.json'))['rename_map']))")
if ! python -c "import lerobot" 2>/dev/null; then
  apt-get update -qq && apt-get install -y -qq ffmpeg git >/dev/null
  pip install -q "lerobot[smolvla]==0.4.4"
fi
python -c "import torch,lerobot;print('torch',torch.__version__,'cuda',torch.cuda.is_available(),torch.cuda.get_device_name(0),'lerobot',lerobot.__version__)"
python -c "from lerobot.datasets.lerobot_dataset import LeRobotDataset as L; d=L('local/$NAME',root='$DS'); x=d[0]; print('dataset ok', d.num_episodes, 'episodes', len(d), 'frames', sorted(k for k in x if k.startswith('observation')))"
echo "episodes: $EPISODES"; echo "rename: $RENAME"; echo "output: $OUT"
lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --dataset.repo_id=local/$NAME --dataset.root=$DS \
  --dataset.episodes="$EPISODES" \
  --rename_map="$RENAME" \
  --policy.push_to_hub=false \
  --batch_size=64 --steps=$STEPS --save_freq=5000 --log_freq=100 --num_workers=8 \
  --output_dir=$OUT --job_name=smolvla_$NAME --policy.device=cuda 2>&1 | tee $OUT.log
echo "DONE -> $OUT"
