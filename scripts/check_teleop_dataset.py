#!/usr/bin/env python
"""Quality check of a teleop LeRobot dataset recorded with scripts/record_teleop.py (run after the session is ended / finalized).
Per episode: label, real fps, out-of-range actions, follower lag, follower tracking error, gripper use, idle start/end, frozen video frames.
Verdict per episode: TRAIN (usable success), FAIL-LABEL (operator said failed: keep out of training), REJECT (data problem).
    .venv/bin/python scripts/check_teleop_dataset.py datasets/teleop_<task>_<stamp> [--sheet]    (--sheet: frame contact sheet per episode in <root>/check/)"""
import argparse, glob, hashlib, json, sys
from pathlib import Path
import numpy as np, pandas as pd, av
ap = argparse.ArgumentParser(); ap.add_argument("root"); ap.add_argument("--sheet", action="store_true"); a = ap.parse_args()
D = Path(a.root); info = json.loads((D / "meta/info.json").read_text())
try: df = pd.concat(pd.read_parquet(f) for f in sorted(glob.glob(str(D / "data/*/*.parquet"))))
except Exception as e: sys.exit(f"cannot read parquet ({e}) - the session is probably still open: END SESSION first (touch run/TELEOP_QUIT)")
labels = {x["episode"]: x for x in json.loads((D / "EPISODE_LABELS.json").read_text())} if (D / "EPISODE_LABELS.json").exists() else {}
cams = [k.split(".")[-1] for k in info["features"] if k.startswith("observation.images.")]
frozen = {}
for cam in cams:                                             # longest run of identical frames per episode (camera frozen/unplugged)
    i = 0; prev = None; run = 0; ep_of = df["episode_index"].values; out = {}
    for f in sorted(glob.glob(str(D / f"videos/observation.images.{cam}/*/*.mp4"))):
        for fr in av.open(f).decode(video=0):
            h = hashlib.md5(fr.to_ndarray(format="gray")[::8, ::8].tobytes()).hexdigest(); run = run + 1 if h == prev else 0; prev = h
            if i < len(ep_of): out[ep_of[i]] = max(out.get(ep_of[i], 0), run)
            i += 1
    frozen[cam] = (out, i)
print(f"{D.name}: {info['total_episodes']} episodes, {info['total_frames']} frames, {info['fps']} fps, cameras {cams}, task: {(D / 'TASK.txt').read_text().strip() if (D / 'TASK.txt').exists() else '?'}")
for cam, (_, n) in frozen.items():
    if n != info["total_frames"]: print(f"  WARNING {cam}: {n} video frames vs {info['total_frames']} data frames")
train = []
for e, g in df.groupby("episode_index"):
    A = np.stack(g["action"].values); S = np.stack(g["observation.state"].values); t = g["timestamp"].values; L = labels.get(e, {})
    out = int(((A < -0.03) | (A > 1.03)).any(1).sum()); j = int(np.argmax(A[:, :7].std(0)))
    lag = max(range(0, 30), key=lambda k: np.corrcoef(A[:len(A) - k, j], S[k:, j])[0, 1])
    err = np.abs(A[:len(A) - lag, :7] - S[lag:, :7]).mean(0); v = np.abs(np.diff(A[:, :7], axis=0)).max(1) > 0.002
    first = int(np.argmax(v)); last = len(v) - int(np.argmax(v[::-1])); fz = max(frozen[c][0].get(e, 0) for c in cams)
    problems = []
    if "real_fps" not in L: problems.append("no real_fps in labels (old recorder: timing unknown)")
    elif L["real_fps"] < 0.95 * info["fps"]: problems.append(f"real fps {L['real_fps']}")
    if out > 0.02 * len(A): problems.append(f"{out} frames action out of range (leader calibration?)")
    if fz > 15: problems.append(f"camera frozen {fz} frames")
    if err.max() > 0.05: problems.append(f"follower tracking error {err.max():.3f} on j{int(err.argmax()) + 1} (clamped/stuck?)")
    if A[:, 7].max() - A[:, 7].min() < 0.2: problems.append("gripper barely used")
    verdict = "REJECT" if problems else ("TRAIN" if L.get("success") else "FAIL-LABEL")
    if verdict == "TRAIN": train.append(int(e))
    print(f"  ep{e:<3} {verdict:<10} {L.get('end', '?'):<7} {len(A) / 30:5.1f}s  fps {L.get('real_fps', '?')}  lag {lag}fr  track-err {err.max():.3f}  "
          f"grip {A[:, 7].min():.2f}-{A[:, 7].max():.2f}  idle start {first / 30:.1f}s end {(len(v) - last) / 30:.1f}s  frozen {fz}fr" + (f"  <- {'; '.join(problems)}" if problems else ""))
    if a.sheet:
        from PIL import Image, ImageDraw
        (D / "check").mkdir(exist_ok=True); idx = np.linspace(0, len(A) - 1, 6).astype(int) + int(g["index"].values[0]); want = set(idx); ims = {}
        k = 0
        for f in sorted(glob.glob(str(D / f"videos/observation.images.{cams[0]}/*/*.mp4"))):
            for fr in av.open(f).decode(video=0):
                if k in want: ims[k] = fr.to_image().resize((320, 240))
                k += 1
                if k > idx.max(): break
        sh = Image.new("RGB", (960, 480))
        for n, i in enumerate(idx):
            if i in ims: ImageDraw.Draw(ims[i]).text((5, 5), f"ep{e} t={(i - idx[0]) / 30:.0f}s", fill=(255, 0, 0)); sh.paste(ims[i], ((n % 3) * 320, (n // 3) * 240))
        sh.save(D / f"check/ep{e}.jpg")
print(f"TRAIN episodes ({len(train)}): {train}")
print(f"  lerobot-train ... --dataset.episodes='{json.dumps(train)}'")
