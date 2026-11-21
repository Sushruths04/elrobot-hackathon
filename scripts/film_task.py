#!/usr/bin/env python
"""Film an autonomous task while it runs: table-camera video (~20 fps, live caption of the task's log lines + timer) + follower joints, saved as H.264 in results/films/ and added to the showcase.
    .venv/bin/python scripts/film_task.py pen       # pick_loop.py: pen/marker into the white box
    .venv/bin/python scripts/film_task.py duster    # duster_caps.py: magnetic duster over the caps, then into the box
Reset the scene first, press ARM; recording starts when the task really starts (after ARM) and ends 4 s after it finishes."""
import json, os, re, subprocess, sys, threading, time, urllib.request
from datetime import datetime
from pathlib import Path
import cv2, numpy as np
ROOT = Path(__file__).resolve().parents[1]
TASKS = {"pen": ("scripts/pick_loop.py", "logs/pick_loop.log", "pen_marker_into_box"), "duster": ("scripts/duster_caps.py", "logs/duster.log", "duster_magnet_caps_into_box")}
task = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] in TASKS else sys.exit("usage: film_task.py pen|duster"); script, logf, name = TASKS[task]
LOG = ROOT / logf; LOG.write_text(""); stamp = datetime.now().strftime("%Y%m%d_%H%M%S"); films = ROOT / "results/films"; films.mkdir(parents=True, exist_ok=True); tmp = films / f"{name}_{stamp}_raw.mp4"; final = films / f"{name}_{stamp}.mp4"
proc = subprocess.Popen([str(ROOT / ".venv/bin/python"), str(ROOT / script)], cwd=ROOT, stdout=open(LOG, "w"), stderr=subprocess.STDOUT)
def lines():
    try: return [l for l in LOG.read_text(errors="replace").splitlines() if l.startswith("[") and "WAITING" not in l and "Requests" not in l and "warn" not in l.lower()]
    except Exception: return []
def frame(): return cv2.imdecode(np.frombuffer(urllib.request.urlopen("http://127.0.0.1:8890/frame.jpg", timeout=10).read(), np.uint8), cv2.IMREAD_COLOR)
def state():
    try: return json.loads(urllib.request.urlopen("http://127.0.0.1:8891/state", timeout=5).read())
    except Exception: return None
print("task started, waiting for ARM / the task to begin; recording starts with its first log line", flush=True)
while proc.poll() is None and not lines(): time.sleep(0.2)
t0 = time.time(); vw = None; jl = open(films / f"{name}_{stamp}_joints.jsonl", "w"); n = 0; done_at = None; last_j = 0; END = re.compile(r"DONE|STOP at|gave up|stopping|NOT ARMED|ABORT|could not|slipped|Traceback")
while time.time() - t0 < 400:
    t = time.time()
    try:
        im = frame(); L = lines()[-3:]; cv2.rectangle(im, (0, 0), (640, 16 + 15 * max(1, len(L))), (0, 0, 0), -1)
        for i, l in enumerate(L): cv2.putText(im, re.sub(r"\s+", " ", l)[:92], (4, 13 + 15 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(im, f"t = {t - t0:5.1f} s", (540, 472), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
        if vw is None: vw = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), 20, (im.shape[1], im.shape[0]))
        vw.write(im); n += 1
        if t - last_j > 0.1:
            st = state()
            if st: jl.write(json.dumps({"t": round(t - t0, 2), "tip": st["tip"], **{k: v["raw"] for k, v in st["joints"].items()}}) + "\n")
            last_j = t
    except Exception: pass
    if done_at is None and (proc.poll() is not None or (lines() and END.search(lines()[-1]))): done_at = time.time()
    if done_at and time.time() - done_at > 4: break
    time.sleep(max(0, 0.05 - (time.time() - t)))
vw and vw.release(); jl.close(); proc.poll() is None and proc.terminate()
import av
cap = cv2.VideoCapture(str(tmp)); out = av.open(str(final), "w"); st = out.add_stream("libx264", rate=20); st.width, st.height, st.pix_fmt = 640, 480, "yuv420p"; st.options = {"crf": "22"}; ok, im = cap.read()
while ok:
    for p in st.encode(av.VideoFrame.from_ndarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), format="rgb24")): out.mux(p)
    ok, im = cap.read()
for p in st.encode(): out.mux(p)
out.close(); tmp.unlink(); (films / f"{name}_{stamp}.log").write_text("\n".join(lines()) + "\n")
print(f"saved {final} ({n} frames, {time.time() - t0:.0f} s)  | last log line: {lines()[-1] if lines() else '-'}", flush=True)
subprocess.run([str(ROOT / ".venv/bin/python"), str(ROOT / "scripts/build_showcase.py")], cwd=ROOT, capture_output=True)
