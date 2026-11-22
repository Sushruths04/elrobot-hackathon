"""Wipe the coffee beans sideways with the tissue held in the gripper: up to 3 passes, lower each time (tip z 0.075 / 0.065 / 0.055), straight to the start then sideways (-x) at constant y,
no stepping down, camera check after each pass (the bean area must change). Filmed via film_task.py tempo_wipe."""
import json, sys, time, urllib.request
from pathlib import Path
import numpy as np, cv2
ROOT = Path(__file__).resolve().parents[1]
T0 = time.time(); log = lambda *m: print(f"[{time.time()-T0:5.1f}s]", *m, flush=True)
A = lambda path, body=None: json.loads(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8891" + path, data=None if body is None else json.dumps(body).encode(), method="GET" if body is None else "POST"), timeout=60).read())
img = lambda: cv2.imdecode(np.frombuffer(urllib.request.urlopen("http://127.0.0.1:8890/frame.jpg", timeout=10).read(), np.uint8), cv2.IMREAD_COLOR)
cal = json.load(open(ROOT / "calibration/table_tip_front.json"))["planes"]; Hm = {p: np.array(cal[p]["H_pixel_to_robot_xy_m"], np.float64) for p in "BC"}
def xy_at(u, v, z): c, b = [cv2.perspectiveTransform(np.float32([[[u, v]]]), Hm[p])[0, 0] for p in "CB"]; return c + (b - c) * (z - 0.04) / 0.02
def must(r, what):
    if r.get("error") and "timeout" not in str(r.get("error")): log(f"   STOP: {what}: {r.get('error')}"); A("/stop", {}); sys.exit(2)
mv = lambda x, y, z, sp, what, acc=30: must(A("/move_tip", {"x": float(x), "y": float(y), "z": float(z), "roll": 0.0, "speed": sp, "timeout": 12, "low_ok": True, "acc": acc}), what)
ROI = (slice(255, 305), slice(285, 360))                                                          # bean box in the picture (beans at about pixel (318,278))
log("WAITING: press 'ARM motion' at http://127.0.0.1:8890")
while A("/state")["armed_s"] < 40: time.sleep(0.5)
(ROOT / "run/wipe_0_before.jpg").write_bytes(cv2.imencode(".jpg", img())[1].tobytes())
dark = lambda: int((cv2.cvtColor(img(), cv2.COLOR_BGR2GRAY)[ROI] < 115).sum()); base = dark(); Z = 0.040 + 0.008
xc, yc = [float(t) for t in xy_at(318, 278, 0.04)]; log(f"beans at xy ({xc:+.3f},{yc:.3f}); dark bean pixels {base}")
# VACUUM-CLEANER (boustrophedon / C-S) pattern, as the user showed: long lanes, one length each, stepping over to the next lane at the end of each length.
# Direction: the user said my earlier sweeps went the opposite way, so the working strokes go toward IMAGE-LEFT (+x). The return lane is lifted 2.5 cm so it does not drag the beans back.
LANES = (0.0,); XL, XR = xc + 0.12, xc - 0.26                         # lane length 24 cm: from image-right (-x) to image-left (+x)
for k, dy in enumerate(LANES, 1):
    y = yc + dy; log(f"lane {k}/{len(LANES)}: y {y:.3f}, working stroke toward image-left")
    mv(XR, y, Z + 0.03, 400, "to the start of the lane", 40); mv(XR, y, Z, 200, "down at the start of the lane (not on the beans)", 40)
    for j in (1, 2, 3): mv(XR + (XL - XR) * j / 3, y, Z, 220, f"lane {k} stroke {j}")
    mv(XL, y, Z + 0.03, 200, "lift at the end of the lane")
    if k < len(LANES): mv(XL, yc + LANES[k], Z + 0.03, 300, "step over to the next lane", 40)
mv(XL, yc, 0.17, 400, "lift clear", 40); time.sleep(0.6); (ROOT / "run/wipe_final.jpg").write_bytes(cv2.imencode(".jpg", img())[1].tobytes()); log(f"dark bean pixels in the first bean box: {base} before -> {dark()} after")
log("DONE - vacuum-pattern sweep finished, check run/wipe_final.jpg")
