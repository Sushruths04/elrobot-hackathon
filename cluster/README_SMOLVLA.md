# Spoon and beans on a cluster GPU

For a first comparison use `train_spoon.py` with ACT and both cameras at the recorded 30 Hz. It uses 16 approved demonstrations (24,231 frames) and reserves the three approved demonstrations from session 110808 for evaluation. Both policies use the same split. Each run saves its source episode IDs in `episodes.json`.

### Stage the data

For an SSH cluster, copy the following from the repository root, replacing `cluster-login` with your SSH alias. Use a directory shared by the login and compute nodes.

```bash
ssh cluster-login 'mkdir -p ~/spoon/datasets'
rsync -aP datasets/spoon_beans_v1 cluster-login:~/spoon/datasets/
rsync -aP cluster/train_spoon.py cluster-login:~/spoon/
```

For the WZL Kubernetes cluster, connect the RWTH VPN and run the following from the repository root. The supplied pod requests an L40S. An existing A100 allocation can use the same Python entry point.

```bash
kubectl -n isaac-sim apply -f cluster/smolvla_train_pod.yaml
kubectl -n isaac-sim wait --for=condition=Ready pod/smolvla-train --timeout=15m
kubectl -n isaac-sim exec smolvla-train -- mkdir -p /data/spoon/datasets
kubectl -n isaac-sim cp datasets/spoon_beans_v1 smolvla-train:/data/spoon/datasets/
kubectl -n isaac-sim cp cluster/train_spoon.py smolvla-train:/data/spoon/train_spoon.py
kubectl -n isaac-sim exec -it smolvla-train -- bash
cd /data/spoon
```

### Train inside the GPU allocation

On an SSH cluster enter your scheduled GPU shell and `cd ~/spoon`. On Kubernetes use the pod shell above. These example PyTorch CUDA 12.6 wheels need a compatible cluster driver. Torchvision 0.22.1 supplies the PyAV video reader used by this launcher. The first run downloads pretrained weights.

```bash
nvidia-smi
python -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126
python -m pip install 'lerobot[smolvla]==0.4.4' torchcodec==0.5
python train_spoon.py datasets/spoon_beans_v1 runs/act_check --steps 10
python train_spoon.py datasets/spoon_beans_v1 runs/act_5k --steps 5000
```

The 10-step run verifies the worker can decode both cameras, update the policy, and save a checkpoint. The 5,000-step run saves every 1,000 steps. Use a fresh run directory for each job. For a pretrained SmolVLA comparison run `python train_spoon.py datasets/spoon_beans_v1 runs/smolvla_5k --policy smolvla --steps 5000 --batch-size 4`. Training writes to the cluster filesystem and needs no Nebius inference key.

Copy the entire `runs/act_5k/training/checkpoints/last/pretrained_model/` directory, including the processors. On WZL, run `kubectl -n isaac-sim cp smolvla-train:/data/spoon/runs/act_5k/training/checkpoints/last/pretrained_model ./act_spoon_checkpoint` from your laptop. Release the pod after copying with `kubectl -n isaac-sim delete pod smolvla-train`.

For physical evaluation, score spoon grasp, successful scoop, and beans landing in the cup separately on ten fresh starts. The current robot-side policy server accepts one image. Serving this checkpoint requires passing both named camera views to the policy.

### Earlier all-eligible SmolVLA run on WZL
Dataset: `datasets/spoon_beans_v1` (24 episodes merged from 5 sessions, 19 TRAIN listed in TRAIN_MANIFEST.json, 2.0 GB).
```
export PATH=$HOME/.local/bin:$PATH
kubectl auth whoami && kubectl get pods -n isaac-sim -o wide          # VPN ok? GPU free?
kubectl apply -f cluster/smolvla_train_pod.yaml -n isaac-sim --dry-run=server
kubectl apply -f cluster/smolvla_train_pod.yaml -n isaac-sim
kubectl get pod smolvla-train -n isaac-sim -w                          # wait for Running (Pending = GPU busy)
kubectl exec smolvla-train -n isaac-sim -- mkdir -p /data/smolvla/datasets
kubectl cp datasets/spoon_beans_v1 isaac-sim/smolvla-train:/data/smolvla/datasets/spoon_beans_v1
kubectl cp cluster/train_smolvla.sh isaac-sim/smolvla-train:/data/smolvla/train_smolvla.sh
kubectl exec smolvla-train -n isaac-sim -- bash -c "nohup bash /data/smolvla/train_smolvla.sh spoon_beans_v1 20000 > /data/smolvla/train.out 2>&1 &"
kubectl exec smolvla-train -n isaac-sim -- tail -f /data/smolvla/train.out   # watch (loss every 100 steps)
```
~4 h for 20k steps (docs: A100; L40S similar). Checkpoints every 5k steps in /data/smolvla/outputs/<run>/checkpoints/.
Copy the result back:  `kubectl cp isaac-sim/smolvla-train:/data/smolvla/outputs/<run>/checkpoints/last/pretrained_model ./models/smolvla_spoon_beans`
Free the GPU when done:  `kubectl delete pod smolvla-train -n isaac-sim`  (the data stays on the PVC). The pod also stops itself after 12 h.
