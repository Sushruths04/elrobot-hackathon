"""Record teleop demonstrations into a LeRobot dataset (v3) from Station: this is the training data for a policy.
  state  = follower joints (0..1 of each calibrated range), 8 values
  action = leader joints   (0..1 of each calibrated range), 8 values  (what the operator commands)
  image  = table camera, 640x480 RGB
Needs: Station running, leader->follower mirroring ON (so the follower follows while you teleoperate), a working camera.
    .venv/bin/python scripts/record_demo.py --task "put the red block in the blue tray" --episodes 10 --seconds 20
    .venv/bin/python scripts/record_demo.py --dry-run          # synthetic data, no hardware: tests the whole dataset pipeline
Then train (local 8 GB GPU: ACT) e.g.  lerobot-train --dataset.repo_id=local/elrobot --dataset.root=datasets/elrobot --policy.type=act ..."""
import argparse, asyncio, io, json, shutil, sys, time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
ap = argparse.ArgumentParser()
ap.add_argument("--task", default="pick and place"); ap.add_argument("--episodes", type=int, default=5); ap.add_argument("--seconds", type=float, default=20)
ap.add_argument("--reset-seconds", type=float, default=8); ap.add_argument("--fps", type=int, default=30); ap.add_argument("--cam")
ap.add_argument("--repo-id", default="local/elrobot"); ap.add_argument("--root", default=str(ROOT / "datasets/elrobot")); ap.add_argument("--dry-run", action="store_true")
ap.add_argument("--overwrite", action="store_true"); a = ap.parse_args()
from lerobot.datasets.lerobot_dataset import LeRobotDataset
H, W = (96, 128) if a.dry_run else (480, 640)
names = [f"joint_{i}" for i in range(1, 9)]
features = {"observation.state": {"dtype": "float32", "shape": (8,), "names": names},
            "action": {"dtype": "float32", "shape": (8,), "names": names},
            "observation.images.table": {"dtype": "video", "shape": (H, W, 3), "names": ["height", "width", "channels"]}}
root = Path(a.root)
if root.exists():
    if a.overwrite: shutil.rmtree(root)
    else: sys.exit(f"{root} exists - pass --overwrite or use another --root (the existing dataset is kept)")
ds = LeRobotDataset.create(a.repo_id, a.fps, features, root=root, robot_type="elrobot", use_videos=True)

if a.dry_run:
    def source():
        t = 0
        while True:
            t += 1; x = (np.sin(t / 10 + np.arange(8)) * 0.5 + 0.5).astype(np.float32)
            yield x, x, np.full((H, W, 3), (t * 5) % 255, np.uint8)
    nxt = source(); get = lambda: next(nxt); close = lambda: None
else:
    from PIL import Image
    from normakit.busread import snapshot
    from normakit.station_backend import StationBackend
    buses = json.loads((ROOT / "run/buses.json").read_text()); L, F = buses["leader"], buses["follower"]
    be = StationBackend("localhost:8888", require_motors=False); be.start(); time.sleep(1.5)
    cam = a.cam
    if not cam:
        for c in be.cameras():
            if c.startswith("cam"):
                try: be.frame_jpeg(c); cam = c; break
                except Exception: pass
    if not cam: sys.exit("no camera with frames")
    loop = asyncio.new_event_loop(); box = {}
    async def init():
        s, box["c"], box["q"] = await snapshot(); return s
    loop.run_until_complete(init())
    def get():
        snap, _, _ = loop.run_until_complete(snapshot(client=box["c"], queue=box["q"]))
        st = np.array([snap[F][i]["norm"] for i in range(1, 9)], np.float32); ac = np.array([snap[L][i]["norm"] for i in range(1, 9)], np.float32)
        img = np.asarray(Image.open(io.BytesIO(be.frame_jpeg(cam))).convert("RGB"))
        return st, ac, img
    close = lambda: None
    print("camera:", cam)

dt = 1.0 / a.fps
for ep in range(a.episodes):
    if ep: print(f"reset the scene... {a.reset_seconds:.0f} s"); time.sleep(a.reset_seconds if not a.dry_run else 0)
    print(f"episode {ep+1}/{a.episodes}: GO ({a.seconds:.0f} s)", flush=True); n = int(a.seconds * a.fps); t0 = time.time()
    for k in range(n):
        st, ac, img = get()
        ds.add_frame({"observation.state": st, "action": ac, "observation.images.table": img, "task": a.task})
        time.sleep(max(0.0, t0 + (k + 1) * dt - time.time()))
    ds.save_episode(); print(f"  saved episode {ep+1} ({n} frames)", flush=True)
if hasattr(ds, "finalize"): ds.finalize()
print("done ->", root); sys.stdout.flush(); import os; os._exit(0)
