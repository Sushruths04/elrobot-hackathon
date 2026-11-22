"""Leader-arm teleop recording straight into a LeRobot v3 dataset, controlled by files (so the operator can say "go"/"done" in chat).
  observation.state         = follower joints (0..1 of each calibrated range), 8
  action                    = leader joints   (0..1), 8   (what the operator commanded)
  observation.images.table  = front camera,  observation.images.wrist = wrist camera (if found)
Cameras and the arm are sampled by background threads; the main loop writes on a fixed 30 fps clock using the newest sample,
so timestamps are real time even when a camera delivers fewer frames.
Control files in run/ (touch them):
  TELEOP_GO       start the next episode            TELEOP_DONE_OK    end episode, success
  TELEOP_DONE_FAIL end episode, failure (kept, labelled)   TELEOP_DISCARD  throw the current episode away
  TELEOP_QUIT     finish the session (after the current episode is ended)
Labels go to <root>/EPISODE_LABELS.json (train only on success=true). Needs leader->follower mirroring ON (POST :8891/mirror {"on":true}).
    .venv/bin/python scripts/record_teleop.py --task "pick up the spoon, scoop beans and pour them into the cup" --root datasets/teleop_spoon_beans_<stamp>
    .venv/bin/python scripts/record_teleop.py --dry-run --root /tmp/x        # synthetic, no hardware"""
import argparse, asyncio, io, json, sys, threading, time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
ap = argparse.ArgumentParser()
ap.add_argument("--task", required=True); ap.add_argument("--root", required=True); ap.add_argument("--repo-id", default="local/elrobot_teleop")
ap.add_argument("--fps", type=int, default=30); ap.add_argument("--max-seconds", type=float, default=120); ap.add_argument("--no-wrist", action="store_true")
ap.add_argument("--table-cam", help="Station camera name for the table view (default: auto)"); ap.add_argument("--wrist-cam", help="Station camera name for the wrist view (default: auto)")
ap.add_argument("--dry-run", action="store_true"); a = ap.parse_args()
RUN = ROOT / "run"; F_GO, F_OK, F_FAIL, F_DISC, F_QUIT = (RUN / n for n in ("TELEOP_GO", "TELEOP_DONE_OK", "TELEOP_DONE_FAIL", "TELEOP_DISCARD", "TELEOP_QUIT"))
for f in (F_GO, F_OK, F_FAIL, F_DISC, F_QUIT): f.unlink(missing_ok=True)
root = Path(a.root)
if root.exists(): sys.exit(f"{root} exists - use a new --root (existing data is never overwritten)")
from PIL import Image
latest = {}; stop_threads = threading.Event()

if a.dry_run:
    cams = {"table": None, "wrist": None}
    def arm_loop():
        t = 0
        while not stop_threads.is_set():
            t += 1; x = (np.sin(t / 10 + np.arange(8)) * 0.5 + 0.5).astype(np.float32); latest["arm"] = (x, x); time.sleep(0.02)
    def cam_loop(key, _):
        t = 0
        while not stop_threads.is_set():
            t += 1; latest[key] = np.full((96, 128, 3), (t * 7) % 255, np.uint8); time.sleep(0.06)
else:
    from normakit.busread import snapshot
    from normakit.station_backend import StationBackend, front_camera, wrist_camera
    buses = json.loads((RUN / "buses.json").read_text()); L, F = buses["leader"], buses["follower"]
    be = StationBackend("localhost:8888", require_motors=False); be.start(); time.sleep(1.5)
    cams = {"table": a.table_cam or front_camera(be)}
    if not a.no_wrist:
        w = a.wrist_cam or wrist_camera(be)
        if w and w != cams["table"]: cams["wrist"] = w
        else: sys.exit("wrist camera not found (unplugged or frozen?) - fix it, or pass --no-wrist to record the table camera only")
    qid = lambda c: be._cam_order[int(c[3:])] if c and c.startswith("cam") and c[3:].isdigit() else c   # camN renumbers on every replug; the queue id is stable
    cams = {k: qid(v) for k, v in cams.items()}
    def redetect():
        """Re-find cameras after a replug (new queue). Returns list of roles still not live."""
        try:
            t = qid(front_camera(be)); cams["table"] = t
            if "wrist" in cams:
                w = wrist_camera(be)
                if w and qid(w) != t: cams["wrist"] = qid(w)
        except Exception as e: print("camera re-detect failed:", e, flush=True)
        return [k for k, n in cams.items() if be.frame_age(n) > 0.5]
    print("cameras:", {k: v[-8:] for k, v in cams.items()}, flush=True)
    def arm_loop():
        loop = asyncio.new_event_loop(); c = q = None
        while not stop_threads.is_set():
            try:
                s, c, q = loop.run_until_complete(snapshot(client=c, queue=q))
                if len(s.get(L, {})) < 8 or len(s.get(F, {})) < 8: continue
                latest["arm"] = (np.array([s[F][i]["norm"] for i in range(1, 9)], np.float32), np.array([s[L][i]["norm"] for i in range(1, 9)], np.float32))
                latest["arm_t"] = time.time()
            except Exception as e:
                print("arm read error:", e, flush=True); c = q = None; time.sleep(0.5)
    def cam_loop(key, name):
        while not stop_threads.is_set():
            try: latest[key] = np.asarray(Image.open(io.BytesIO(be.frame_jpeg(cams[key]))).convert("RGB")); latest[key + "_t"] = time.time()
            except Exception: time.sleep(0.1)

threading.Thread(target=arm_loop, daemon=True).start()
for k, n in cams.items(): threading.Thread(target=cam_loop, args=(k, n), daemon=True).start()
t0 = time.time()
while not ("arm" in latest and all(k in latest for k in cams)):
    if time.time() - t0 > 15: sys.exit(f"no data after 15 s: have {sorted(latest)} need arm + {list(cams)}")
    time.sleep(0.1)
names = [f"joint_{i}" for i in range(1, 9)]
features = {"observation.state": {"dtype": "float32", "shape": (8,), "names": names}, "action": {"dtype": "float32", "shape": (8,), "names": names}}
for k in cams: features[f"observation.images.{k}"] = {"dtype": "video", "shape": latest[k].shape, "names": ["height", "width", "channels"]}
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset.create(a.repo_id, a.fps, features, root=root, robot_type="elrobot", use_videos=True, image_writer_threads=8)   # async PNG writing: without it add_frame took ~96 ms -> only ~10 fps real
labels_path = root / "EPISODE_LABELS.json"; labels = []
(root / "TASK.txt").write_text(a.task + "\n")
dt = 1.0 / a.fps; saved = 0
print(f"READY dataset {root} | features {list(features)} | waiting for TELEOP_GO", flush=True)
try:
    while True:
        while not (F_GO.exists() or F_QUIT.exists()): time.sleep(0.05)
        if F_QUIT.exists() and not F_GO.exists(): break
        F_GO.unlink(missing_ok=True)
        if not a.dry_run:
            dead = [k for k, n in cams.items() if be.frame_age(n) > 0.5]
            if dead: dead = redetect()
            if dead:
                print(f"ERROR camera not live: {dead} (unplugged/replugged?) - episode NOT started; fix the cable and press GO again", flush=True); continue
        print(f"RECORDING episode {saved} (max {a.max_seconds:.0f} s)", flush=True)
        n = 0; t0 = time.time(); end = None; stale = 0; stale_cam = 0; run_cam = 0
        while True:
            for f, tag in ((F_OK, "ok"), (F_FAIL, "fail"), (F_DISC, "discard")):
                if f.exists(): f.unlink(); end = tag
            if end or time.time() - t0 > a.max_seconds: break
            st, ac = latest["arm"]
            if not a.dry_run and time.time() - latest.get("arm_t", 0) > 0.5: stale += 1
            if not a.dry_run and any(be.frame_age(n) > 0.5 for n in cams.values()):
                stale_cam += 1; run_cam += 1
                if run_cam == a.fps // 2: print(f"  CAMERA FROZE at {n / a.fps:.0f} s", flush=True)   # live warning in the panel log (ends with ' s' so the panel keeps showing RECORDING)
            else: run_cam = 0
            fr = {"observation.state": st, "action": ac, "task": a.task}
            for k in cams: fr[f"observation.images.{k}"] = latest[k]
            ds.add_frame(fr); n += 1
            if n % (a.fps * 10) == 0: print(f"  {n / a.fps:.0f} s", flush=True)
            time.sleep(max(0.0, t0 + n * dt - time.time()))
        print("SAVING episode (encoding video, wait for READY)", flush=True)
        end = end or "timeout"; wall = time.time() - t0; real_fps = n / max(wall, 1e-6)
        if end == "discard" or n < a.fps:
            ds.clear_episode_buffer(); print(f"DISCARDED ({n} frames)", flush=True)
        else:
            ds.save_episode()
            labels.append({"episode": saved, "success": end == "ok", "end": end, "frames": n, "seconds": round(n / a.fps, 1),
                           "real_fps": round(real_fps, 1), "wall_seconds": round(wall, 1), "stale_arm_frames": stale, "stale_camera_frames": stale_cam, "recorded_at": time.strftime("%Y-%m-%d %H:%M:%S")})
            labels_path.write_text(json.dumps(labels, indent=1))
            print(f"SAVED episode {saved}: {end}, {n / a.fps:.1f} s{f', WARNING real rate {real_fps:.1f} fps < {a.fps} - timestamps wrong, discard' if real_fps < 0.95 * a.fps else ''}{f', WARNING {stale} stale arm frames' if stale else ''}{f', WARNING camera froze for {stale_cam} frames (unplugged?) - discard this episode' if stale_cam > a.fps // 2 else ''}", flush=True); saved += 1
        if F_QUIT.exists(): break
        print("waiting for TELEOP_GO (reset the scene)", flush=True)
finally:
    stop_threads.set()
    if hasattr(ds, "finalize"): ds.finalize()
    print(f"DONE {saved} episodes -> {root}", flush=True); sys.stdout.flush()
import os; os._exit(0)
