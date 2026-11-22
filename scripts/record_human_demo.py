"""Record the front camera while a HUMAN demonstrates a task (no robot motion). Stops when run/STOP_HUMAN_DEMO exists or after --max seconds.
   scripts/record_human_demo.py tissue_sweep_coffee_beans"""
import argparse, io, sys, time, urllib.request
from pathlib import Path
import av, numpy as np
from PIL import Image
ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser(); ap.add_argument("name"); ap.add_argument("--max", type=int, default=900); ap.add_argument("--cam", default="front"); a = ap.parse_args()
stop = ROOT / "run/STOP_HUMAN_DEMO"; stop.unlink(missing_ok=True)
out = ROOT / "results/human_demos" / f"{a.name}_{time.strftime('%Y%m%d_%H%M%S')}.mp4"; tmp = out.with_suffix(".raw.mp4")
get = lambda: np.array(Image.open(io.BytesIO(urllib.request.urlopen(f"http://127.0.0.1:8890/frame.jpg?cam={a.cam}", timeout=5).read())).convert("RGB"))
c = av.open(str(tmp), "w"); st = c.add_stream("libx264", rate=15); st.pix_fmt = "yuv420p"; st.options = {"crf": "23", "preset": "veryfast"}
t0 = time.time(); n = 0; ts = []
while time.time() - t0 < a.max and not stop.exists():
    t = time.time()
    try: f = get()
    except Exception: time.sleep(0.1); continue
    if n == 0: st.width, st.height = f.shape[1], f.shape[0]
    for p in st.encode(av.VideoFrame.from_ndarray(f, format="rgb24")): c.mux(p)
    n += 1; ts.append(round(t - t0, 3)); time.sleep(max(0, 1 / 15 - (time.time() - t)))
for p in st.encode(): c.mux(p)
c.close(); real = n / max(time.time() - t0, 1)
# re-encode at the real capture rate so the playback speed matches what happened
r = av.open(str(tmp)); o = av.open(str(out), "w"); s2 = o.add_stream("libx264", rate=max(5, int(round(real)))); s2.pix_fmt = "yuv420p"; s2.width, s2.height = st.width, st.height
for fr in r.decode(video=0):
    for p in s2.encode(fr): o.mux(p)
for p in s2.encode(): o.mux(p)
o.close(); r.close(); tmp.unlink(); (out.with_suffix(".json")).write_text(__import__("json").dumps({"frames": n, "seconds": round(time.time() - t0, 1), "capture_fps": round(real, 1), "timestamps": ts}))
print("saved", out, n, "frames", round(time.time() - t0), "s", flush=True)
