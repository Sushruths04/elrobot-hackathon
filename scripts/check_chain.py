"""Read-only check of the whole chain before any autonomous pick: does the arm model + table mapping agree with where the arm REALLY went in the demo,
and can the Claude model find the cup in the live picture?   Writes run/check_chain.jpg"""
import base64, json, os, re, sys, time
from pathlib import Path
import cv2, numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from normakit import vision
from normakit.core import norm_to_rad
from normakit.kin import Kin
from normakit.station_backend import StationBackend, front_camera
T = json.loads((ROOT / "calibration/table_front.json").read_text()); H = np.array(T["H_pixel_to_robot_xy_m"]); Hi = np.linalg.inv(H)
to_px = lambda x, y: (lambda p: (int(round(p[0] / p[2])), int(round(p[1] / p[2]))))(Hi @ np.array([x, y, 1.0]))
be = StationBackend("localhost:8888"); be.start(); time.sleep(2.0); kin = Kin(); st = be.state(); CAM = sys.argv[1] if len(sys.argv) > 1 else front_camera(be); print("front camera is", CAM, flush=True)
def fk(ticks):
    n = {}
    for i, v in ticks.items():
        j = st.joints[int(i)]; span = (j.hi - j.lo) % 4096; rel = (v - j.lo) % 4096
        if rel > span + (4096 - span) / 2: rel -= 4096
        n[int(i)] = rel / span
    return [float(c) for c in kin.fk({i: norm_to_rad(i, x) for i, x in n.items()})]
D = json.loads((ROOT / "run/demo_traj_pick1.json").read_text()); g = [s[1]["8"] for s in D]
close_k = next(k for k in range(len(D)) if g[k] < 1450); open_k = next(k for k in range(close_k, len(D)) if g[k] > 1800)
poses = {"grasp (gripper closes)": D[close_k], "release (gripper opens)": D[open_k], "start": D[0]}
rgb = vision.decode(be.frame_jpeg(CAM)); out = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR); cols = {"grasp (gripper closes)": (0, 0, 255), "release (gripper opens)": (255, 0, 0), "start": (0, 200, 255)}
print("demo moments through the arm model + table mapping (pixel in the front picture):", flush=True)
for name, s in poses.items():
    xyz = fk({int(i): v for i, v in s[1].items()}); px = to_px(xyz[0], xyz[1]); print(f"  {name:24s} t={s[0]:5.1f}s  EE xyz=({xyz[0]:.3f},{xyz[1]:.3f},{xyz[2]:.3f}) -> pixel {px}", flush=True)
    cv2.circle(out, px, 9, cols[name], 2); cv2.putText(out, name.split(" ")[0], (px[0] + 10, px[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, cols[name], 2)
# Claude model: find the black cup
try:
    import anthropic; client = anthropic.Anthropic(); jpg = be.frame_jpeg(CAM); h, w = rgb.shape[:2]
    r = client.messages.create(model=os.environ.get("NORMA_MODEL", "claude-sonnet-5-5"), max_tokens=250, messages=[{"role": "user", "content": [
        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": base64.b64encode(jpg).decode()}},
        {"type": "text", "text": f"This RGB camera image is {w}x{h} pixels. Find the black cup (a black open-top mug) on the table. Reply with ONLY one JSON object: {{\"found\": true/false, \"u\": <x pixel of where the cup touches the table>, \"v\": <y pixel>, \"note\": \"<short>\"}}"}]}])
    m = re.search(r"\{.*\}", r.content[0].text, re.S); res = json.loads(m.group(0)); print("Claude model says:", res, flush=True)
    if res.get("found"): cx, cy = int(res["u"]), int(res["v"]); cv2.circle(out, (cx, cy), 9, (0, 255, 0), 3); cv2.putText(out, "Claude: cup", (cx + 10, cy + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2); xy = H @ np.array([cx, cy, 1.0]); print(f"  -> table position of that pixel: x={xy[0]/xy[2]:.3f} y={xy[1]/xy[2]:.3f} m", flush=True)
except Exception as e:
    print("Claude model call FAILED:", type(e).__name__, str(e)[:160], flush=True)
d = vision.find_dark_object(rgb, roi=(60, 185, 500, 470), min_area=300, max_area=30000)
cv2.imwrite(str(ROOT / "run/check_chain.jpg"), out); print("overlay saved: run/check_chain.jpg", flush=True); os._exit(0)
