"""Hand-guided camera <-> arm calibration capture (read-only: moves NOTHING, torque stays off).
You move the FOLLOWER by hand (marker on the gripper) and hold each pose ~2-3 s. When the arm is still and the marker is visible in the
front camera, a sample is saved: joints (0..1), marker pixel, forward-kinematics xyz, front + wrist JPEGs. Prints which part of the table is still empty.
    .venv/bin/python scripts/handeye_capture.py --front cam1 --wrist cam2
Output: run/handeye/<time>/samples.json + images. Fit afterwards with scripts/handeye_fit.py (next step)."""
import argparse, collections, json, sys, time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from normakit import vision
from normakit.core import norm_to_rad
from normakit.kin import Kin
from normakit.station_backend import StationBackend
ap = argparse.ArgumentParser(); ap.add_argument("--front", default="cam1"); ap.add_argument("--wrist", default="cam2")
ap.add_argument("--target", type=int, default=32); ap.add_argument("--minutes", type=float, default=15); ap.add_argument("--marker", default="magenta"); a = ap.parse_args()
be = StationBackend("localhost:8888"); be.start(); time.sleep(1.5); kin = Kin()
out = ROOT / "run/handeye" / time.strftime("%Y%m%d_%H%M%S"); out.mkdir(parents=True, exist_ok=True)
U0, U1, V0, V1, NC, NR = 40, 620, 170, 470, 5, 4                       # image region where the table is, split into 5x4 cells
def cell(u, v): return (min(NC - 1, max(0, int((u - U0) / (U1 - U0) * NC))), min(NR - 1, max(0, int((v - V0) / (V1 - V0) * NR))))
hist = collections.deque(); samples = []; last_msg = 0.0; t_end = time.time() + a.minutes * 60
def say(msg, every=0.0):
    global last_msg
    if time.time() - last_msg >= every: print(msg, flush=True); last_msg = time.time()
def fk_xyz(norms): return [float(x) for x in kin.fk({i: norm_to_rad(i, v) for i, v in norms.items()})]
print(f"READY: move the follower by hand, hold each pose 2-3 s. Need {a.target} samples (about 2/3 near the table, 1/3 raised ~10 cm).", flush=True)
while time.time() < t_end and len(samples) < a.target + 8:
    st = be.state(); norms = {i: j.norm for i, j in st.joints.items()}; now = time.time()
    hist.append((now, norms)); [hist.popleft() for _ in range(len(hist)) if hist[0][0] < now - 1.2]
    time.sleep(0.1)
    if now - hist[0][0] < 1.0: continue
    spread = max(max(h[1][i] for h in hist) - min(h[1][i] for h in hist) for i in norms)
    if spread > 0.004: say("moving...", 6); continue
    try: front = be.frame_jpeg(a.front)
    except Exception as e: say(f"no frame from {a.front}: {e}", 5); continue
    px = vision.find_named(vision.decode(front), a.marker)
    if px is None: say("STILL, but the magenta marker is NOT visible in the front camera - turn the marker towards the front camera", 5); continue
    xyz = fk_xyz(norms)
    if any(np.hypot(px[0] - s["px"][0], px[1] - s["px"][1]) < 22 and np.linalg.norm(np.array(xyz) - np.array(s["xyz"])) < 0.03 for s in samples):
        say("this pose is already captured - move to a new place", 6); continue
    n = len(samples) + 1; (out / f"{n:02d}_front.jpg").write_bytes(front)
    try: (out / f"{n:02d}_wrist.jpg").write_bytes(be.frame_jpeg(a.wrist))
    except Exception: pass
    samples.append({"n": n, "t": now, "norms": norms, "px": [float(px[0]), float(px[1])], "xyz": xyz}); (out / "samples.json").write_text(json.dumps(samples, indent=1))
    low = sum(s["xyz"][2] < 0.06 for s in samples); cells = {cell(*s["px"]) for s in samples if s["xyz"][2] < 0.06}
    empty = [f"col{c+1}/row{r+1}" for r in range(NR) for c in range(NC) if (c, r) not in cells]
    print(f"captured {n}/{a.target}  (near-table {low}, raised {n-low}; marker pixel {px[0]:.0f},{px[1]:.0f}; height {xyz[2]*100:.0f} cm)  empty table cells: {', '.join(empty[:8]) or 'none'}", flush=True)
    if n >= a.target and low >= 18 and n - low >= 8 and len(empty) <= 4: print("ENOUGH: coverage is good - you can stop", flush=True); break
    time.sleep(0.8)
print(f"done: {len(samples)} samples -> {out}", flush=True); import os; os._exit(0)
