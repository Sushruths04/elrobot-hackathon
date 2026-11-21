#!/usr/bin/env python
"""Convert the recorded teleop demos (follower joints + leader joints + camera video) into a LeRobot dataset (v3 via lerobot 0.4.4) and make playable H.264 backups of the videos.
  observation.state = follower joints (0..1 of the calibrated range, 8), action = leader joints (0..1, 8), observation.images.table = table camera 640x480 (30 fps timeline)
  The leader/follower clocks are aligned by cross-correlation (follower copies the leader with a small lag); the 8 fps video is repeated onto the 30 fps timeline.
    .venv/bin/python scripts/demos_to_lerobot.py [--root datasets/elrobot_teleop]"""
import argparse, glob, json, shutil, sys, time
from pathlib import Path
import cv2, numpy as np
ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser(); ap.add_argument("--root", default=str(ROOT / "datasets/elrobot_teleop")); ap.add_argument("--fps", type=int, default=30); ap.add_argument("--overwrite", action="store_true"); a = ap.parse_args()
TASK = "pick up the pen or marker and place it in the white box"
LABELS = json.loads((ROOT / "run/demos/episodes.json").read_text())
def ranges(role):
    f = sorted(glob.glob(str(ROOT / f"calibration/*_{role}_20*.json")))[-1]; m = json.loads(Path(f).read_text())["motors"]; return {int(k): (v["min"], v["span"]) for k, v in m.items()}
FR, LR = ranges("follower"), ranges("leader")
norm = lambda raw, r, i: ((raw - r[i][0] + 2048) % 4096 - 2048 + 0.0) / r[i][1] if False else np.clip((raw - r[i][0]) / r[i][1], -0.2, 1.2)
def load(name):
    fol = json.loads((ROOT / f"run/demos/{name}.json").read_text()); tf = np.array([x[0] for x in fol]); F = np.array([[norm(x[1][str(i)], FR, i) for i in range(1, 9)] for x in fol], np.float32)
    led = [json.loads(l) for l in open(ROOT / f"run/demos/{name}_leader.jsonl")]; tl = np.array([x["t"] for x in led]); L = np.array([[norm(x["q"][str(i)], LR, i) for i in range(1, 9)] for x in led], np.float32)
    return tf, F, tl - tl[0], L
def align(tf, F, tl, L):
    """offset d (s) so that leader time tl + d matches follower time tf (follower lags the leader a little); search -3..+8 s (the leader recorder started a few seconds later)."""
    best = None
    for d in np.arange(-3, 8, 0.02):
        t = np.arange(max(tf[0], tl[0] + d), min(tf[-1], tl[-1] + d), 0.1)
        if len(t) < 50: continue
        Fi = np.stack([np.interp(t, tf, F[:, j]) for j in range(8)], 1); Li = np.stack([np.interp(t - d, tl, L[:, j]) for j in range(8)], 1); e = float(np.mean((Fi[:, :7] - Li[:, :7]) ** 2))
        if best is None or e < best[0]: best = (e, d)
    return best[1], best[0]
def video(name):
    d = sorted(glob.glob(str(ROOT / f"run/recordings/*_{name}")))[-1]; meta = json.loads(Path(d, "meta.json").read_text()); cap = cv2.VideoCapture(f"{d}/front.mp4"); fr = []
    while True:
        ok, im = cap.read()
        if not ok: break
        fr.append(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    return fr, meta["seconds"], d
def h264(src_dir, dst):
    import av
    cap = cv2.VideoCapture(f"{src_dir}/front.mp4"); fps = cap.get(cv2.CAP_PROP_FPS) or 8; out = av.open(str(dst), "w"); st = out.add_stream("libx264", rate=int(round(fps))); st.width, st.height, st.pix_fmt = 640, 480, "yuv420p"; st.options = {"crf": "20"}
    while True:
        ok, im = cap.read()
        if not ok: break
        for p in st.encode(av.VideoFrame.from_ndarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), format="rgb24")): out.mux(p)
    for p in st.encode(): out.mux(p)
    out.close()
from lerobot.datasets.lerobot_dataset import LeRobotDataset
names = [f"joint_{i}" for i in range(1, 9)]
features = {"observation.state": {"dtype": "float32", "shape": (8,), "names": names}, "action": {"dtype": "float32", "shape": (8,), "names": names},
            "observation.images.table": {"dtype": "video", "shape": (480, 640, 3), "names": ["height", "width", "channels"]}}
root = Path(a.root)
if root.exists():
    if a.overwrite: shutil.rmtree(root)
    else: sys.exit(f"{root} exists - pass --overwrite")
ds = LeRobotDataset.create("local/elrobot_teleop", a.fps, features, root=root, robot_type="elrobot", use_videos=True); info = []
for name in ("teleop_pick2", "teleop_pick1"):                         # success first
    tf, F, tl, L = load(name); d, err = align(tf, F, tl, L); frames, vsec, vdir = video(name); T = min(tf[-1], vsec); n = int(T * a.fps); fps_v = len(frames) / vsec
    print(f"{name}: follower {len(tf)} samples {tf[-1]:.1f}s, leader {len(tl)} samples, video {len(frames)} frames {vsec:.1f}s; leader clock offset {d:+.2f}s (alignment error {err:.5f}); writing {n} frames", flush=True)
    for k in range(n):
        t = k / a.fps; st = np.array([np.interp(t, tf, F[:, j]) for j in range(8)], np.float32); ac = np.array([np.interp(t + 0.0 - d, tl, L[:, j]) for j in range(8)], np.float32)
        img = frames[min(len(frames) - 1, int(t * fps_v))]; ds.add_frame({"observation.state": st, "action": ac, "observation.images.table": img, "task": TASK})
    ds.save_episode(); lab = LABELS.get(name, {}); info.append({"episode": len(info), "source": name, "success": lab.get("success"), "note": lab.get("note", ""), "frames": n, "seconds": round(n / a.fps, 1), "leader_clock_offset_s": round(float(d), 2)})
    Path(ROOT / "demo_backup").mkdir(exist_ok=True); h264(vdir, ROOT / f"demo_backup/{name}_{'success' if lab.get('success') else 'failure'}.mp4")
if hasattr(ds, "finalize"): ds.finalize()
(root / "EPISODE_LABELS.json").write_text(json.dumps(info, indent=1)); (root / "README.md").write_text("# elrobot_teleop\nTwo leader-arm teleoperation episodes of picking a pen/marker and placing it in the white box, recorded on the real ElRobot.\n" + "\n".join(f"- episode {i['episode']} ({i['source']}): success={i['success']} - {i['note']}" for i in info) + "\n\nTrain only on episodes with success=true. state=follower joints, action=leader joints (0..1 of the calibrated range), 30 fps, camera 640x480 (source video 8 fps, frames repeated).\n")
print("DATASET saved:", root, json.dumps(info))
