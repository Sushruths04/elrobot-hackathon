"""Tempo tissue pack, step 1: open the white flip-top lid by PUSHING it with the CLOSED gripper (no gripping, no pinching), then check with the camera.
   Step 2 (pinch one tissue and pull it out) comes after this works. Gentle, one attempt, photos at every stage in run/tempo_*.jpg.
   Run via film_task.py tempo so it is filmed and saved as a LeRobot episode."""
import argparse, json, sys, time, urllib.request, concurrent.futures as cf
from pathlib import Path
import numpy as np, cv2
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from normakit import er2
ap = argparse.ArgumentParser(); ap.add_argument("--dry", action="store_true"); ap.add_argument("--lip-z", type=float, default=0.045); ap.add_argument("--z0", type=float, default=0.066); a = ap.parse_args()
T0 = time.time(); log = lambda *m: print(f"[{time.time()-T0:5.1f}s]", *m, flush=True)
def A(path, body=None):
    req = urllib.request.Request("http://127.0.0.1:8891" + path, data=None if body is None else json.dumps(body).encode(), method="GET" if body is None else "POST")
    return json.loads(urllib.request.urlopen(req, timeout=90).read())
jpg = lambda: urllib.request.urlopen("http://127.0.0.1:8890/frame.jpg", timeout=10).read()
shot = lambda n: (ROOT / f"run/tempo_{n}.jpg").write_bytes(jpg())
cal = json.load(open(ROOT / "calibration/table_tip_front.json"))["planes"]; Hm = {p: np.array(cal[p]["H_pixel_to_robot_xy_m"], np.float64) for p in "BC"}
def xy_at(u, v, z):                                                                               # xy is linear in z along the viewing ray: interpolate planes C (0.04) and B (0.06)
    c, b = [cv2.perspectiveTransform(np.float32([[[u, v]]]), Hm[p])[0, 0] for p in "CB"]; return c + (b - c) * (z - 0.04) / 0.02
def must(r, what):
    if not r.get("ok"): log(f"   STOP: {what} failed: {r.get('error')}"); A("/stop", {}); sys.exit(2)
G = {"acc": 12}
if not a.dry:
    log("WAITING: press 'ARM motion' at http://127.0.0.1:8890")
    while A("/state")["armed_s"] < 90: time.sleep(1)
shot("0_start"); jp = jpg()
with cf.ThreadPoolExecutor(3) as ex: pts = [p for f in [ex.submit(er2.point, jp, "the white flip-top cap on the tissue pack", 640, 480) for _ in range(3)] for p in f.result()["points"]]
if not pts: log("STOP: could not find the lid"); sys.exit(3)
u, v = float(np.median([p[1] for p in pts])), float(np.median([p[2] for p in pts])); log(f"lid at pixel ({u:.0f},{v:.0f})")
SAG = 0.008                                                                                       # measured: the tip ends ~8 mm below the commanded z at these low, stretched poses
# What the hand-guided demo showed: the white cap is hinged on the IMAGE-RIGHT side. The closed gripper touched the cap's LEFT edge and swept it sideways to the right: the cap flips over, tissue exposed.
Z0 = a.z0                                                                                        # cap-top edge height (NOT deeper: the last sweep went too deep)
x0, y0 = [float(t) for t in xy_at(u - 30, v - 2, Z0)]; x1, y1 = x0 - 0.075, y0                                                                          # SIDEWAYS = pure robot -x at constant y (the old pixel-line sweep drifted 5 cm FORWARD)
def blue():                                                                                       # blue label pixels around the pack: they vanish from view when the cap flips over
    im = cv2.imdecode(np.frombuffer(jpg(), np.uint8), cv2.IMREAD_COLOR); hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    m = (hsv[..., 0] > 100) & (hsv[..., 0] < 130) & (hsv[..., 1] > 110) & (hsv[..., 2] > 50); r = m[int(v) - 50:int(v) + 50, int(u) - 70:int(u) + 70]; return int(r.sum())
log(f"cap edge start ({x0:+.3f},{y0:.3f}) -> end ({x1:+.3f},{y1:.3f}); blue label pixels now {blue()}")
if a.dry: log("dry run: stopping"); sys.exit(0)
must(A("/gripper", {"to": 1321}), "close the gripper to use it as a sweeper")
mv = lambda x, y, z, sp, what: must(A("/move_tip", {"x": x, "y": y, "z": z + SAG, "roll": 0.0, "speed": sp, "timeout": 40, "low_ok": True, **G}), what)
mv(x0, y0, Z0, 150, "move straight to the start of the sweep (no stepping down)"); shot("1_at_start"); b0 = blue(); log(f"at the start position, label pixels {b0}; sweeping sideways from here at the same height")
n_seg = 6; opened = False
for k in range(1, n_seg + 1):
    f = k / n_seg; mv(x0 + (x1 - x0) * f, y0 + (y1 - y0) * f, Z0 + 0.008 * f, 40, f"sweep segment {k}"); time.sleep(0.8); bk = blue(); log(f"   sweep {k}/{n_seg}: label pixels {bk} (start {b0})")
    if b0 and bk < 0.5 * b0: opened = True; log("   the blue label is hidden: the cap flipped open - stopping the sweep"); break
shot("3_swept"); mv(x1 + 0.17, y1 - 0.04, 0.16, 300, "move the arm far to the side so the camera sees the cap"); time.sleep(1.2); shot("4_after"); b2 = blue(); log(f"after: label pixels {b2} (start {b0}) -> {'cap open' if b2 < 0.5 * b0 else 'cap still closed'}"); log('RESULT: ' + ('cap open' if b2 < 0.5 * b0 else 'cap closed'))
with cf.ThreadPoolExecutor(2) as ex: ans = [f.result() for f in [ex.submit(er2.ask, jpg(), "Is the white flip-top lid of the tissue pack open (tissue visible)? Answer JSON {\"open\": true/false}") for _ in range(2)]]
log("model check:", ans); log("DONE - photos run/tempo_*.jpg")
