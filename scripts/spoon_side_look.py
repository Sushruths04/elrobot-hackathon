"""Look test for the sideways spoon grasp: pre-grasp pose (tilt 60, two wrist rolls), jaws open, wrist + front photos, orange segmentation in the wrist image. No contact."""
import json, sys, time, urllib.request
from pathlib import Path
import numpy as np, cv2
ROOT = Path(__file__).resolve().parents[1]
A = lambda p, b=None, t=60: json.loads(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8891" + p, data=None if b is None else json.dumps(b).encode(), method="GET" if b is None else "POST"), timeout=t).read())
img = lambda c: cv2.imdecode(np.frombuffer(urllib.request.urlopen(f"http://127.0.0.1:8893/frame.jpg?c={c}", timeout=5).read(), np.uint8), 1)
def orange(im):
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV); m = cv2.inRange(hsv, np.array((0, 40, 120)), np.array((14, 230, 255))); m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, st, cen = cv2.connectedComponentsWithStats(m)
    if n < 2: return None
    i = 1 + int(np.argmax(st[1:, 4])); return {"area": int(st[i, 4]), "centroid": [round(float(x), 1) for x in cen[i]], "bbox": [int(x) for x in st[i, :4]]}
P = json.loads((ROOT / "run/side_grasp_plan.json").read_text()); pre = P["pre_xy"]
while A("/state")["armed_s"] < 30: print("WAITING for ARM", flush=True); time.sleep(2)
A("/mirror", {"on": False}); A("/gripper", {"to": "open"})
out = {}
for roll in (0.0, 1.5708):
    r = A("/move_tip", {"x": pre[0], "y": pre[1], "z": 0.13, "roll": roll, "tilt": 60, "speed": 300, "timeout": 20}); print("roll", roll, "move:", {k: r.get(k) for k in ("ok", "error", "pos_err_mm", "worst_ticks_off")}, flush=True)
    time.sleep(1.0); w = img("wrist"); f = img("table"); ts = time.strftime("%H%M%S")
    cv2.imwrite(str(ROOT / f"results/photos/side_look_wrist_roll{roll:.1f}_{ts}.jpg"), w); cv2.imwrite(str(ROOT / f"results/photos/side_look_front_roll{roll:.1f}_{ts}.jpg"), f)
    out[roll] = orange(w); print("  wrist orange:", out[roll], flush=True)
print("DONE look test", flush=True)
