#!/usr/bin/env python
"""Record a pick-and-place run: table-camera video (25 fps, with a live overlay of the loop's log lines and the elapsed time) + follower joints/tip at ~20 Hz + a copy of the log.
Stops by itself when logs/pick_loop.log shows DONE / STOP / gave up / stopping (+3 s) or after 300 s.   Output: results/runs/<time>/{run.mp4,joints.jsonl,pick_loop.log,meta.json}"""
import json, os, re, sys, time
from datetime import datetime
from pathlib import Path
import cv2, numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src")); os.chdir(ROOT)
from normakit import vision
from normakit.station_backend import StationBackend, front_camera
out = ROOT / "results/runs" / datetime.now().strftime("%Y%m%d_%H%M%S"); out.mkdir(parents=True)
be = StationBackend("localhost:8888", require_motors=False); be.start(); time.sleep(1.5); cam = front_camera(be)
LOG = ROOT / "logs/pick_loop.log"; vw = None; jl = open(out / "joints.jsonl", "w"); t0 = time.time(); n = 0; done_at = None; last_jt = 0
END = re.compile(r"DONE|STOP at|gave up|stopping|NOT ARMED|Traceback")
def log_lines():
    try: return [l for l in LOG.read_text(errors="replace").splitlines() if l.startswith("[") and not any(k in l for k in ("Requests", "warnings"))]
    except Exception: return []
print("waiting for the loop to see the pen ('pen is placed' in the log) before recording...", flush=True)
t_w = time.time()
while time.time() - t_w < 600 and not any("pen is placed" in l for l in log_lines()): time.sleep(0.2)
t0 = time.time(); print("recording to", out, flush=True)
while time.time() - t0 < 300:
    t = time.time()
    try:
        rgb = vision.decode(be.frame_jpeg(cam)); img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR); L = log_lines(); stage = [l for l in L if "WAITING" not in l][-3:]
        cv2.rectangle(img, (0, 0), (640, 16 + 15 * len(stage)), (0, 0, 0), -1)
        for i, l in enumerate(stage): cv2.putText(img, re.sub(r"\s+", " ", l)[:92], (4, 13 + 15 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img, f"t = {t - t0:5.1f} s", (540, 472), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
        if vw is None: vw = cv2.VideoWriter(str(out / "run.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 25, (img.shape[1], img.shape[0]))
        vw.write(img); n += 1
        if t - last_jt > 0.05: st = be.state(); jl.write(json.dumps({"t": round(t - t0, 3), **{str(i): j.raw for i, j in st.joints.items()}}) + "\n"); last_jt = t
        if done_at is None and L and END.search(L[-1]): done_at = time.time()
    except Exception as e: pass
    if done_at and time.time() - done_at > 3: break
    time.sleep(max(0, 0.04 - (time.time() - t)))
vw and vw.release(); jl.close(); (out / "pick_loop.log").write_text("\n".join(log_lines()) + "\n")
(out / "meta.json").write_text(json.dumps({"frames": n, "seconds": round(time.time() - t0, 1), "log_lines": len(log_lines())}, indent=1)); print("saved", out, n, "frames", flush=True); os._exit(0)
