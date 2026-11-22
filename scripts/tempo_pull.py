"""Tempo pack, step 2: pinch only the TIP of the tissue sticking out of the opened pack (not deep, never the pack) and pull it out. Autonomous, gentle, photos in run/tempo_pull_*.jpg.
   Jaws straddle the tissue tip sideways at the TOP of the bulge. After a small lift the pack position is re-checked: if the whole pack came up (it moved in the image), the arm
   puts it back down where it was, opens, and retries one level higher. If the gripper is already holding the pack at the start, it puts it down first. --dry: locate only."""
import argparse, json, sys, time, urllib.request, concurrent.futures as cf
from pathlib import Path
import numpy as np, cv2
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from normakit import er2
ap = argparse.ArgumentParser(); ap.add_argument("--dry", action="store_true"); ap.add_argument("--zs", default="0.068,0.074,0.080"); a = ap.parse_args()
T0 = time.time(); log = lambda *m: print(f"[{time.time()-T0:5.1f}s]", *m, flush=True)
A = lambda path, body=None: json.loads(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8891" + path, data=None if body is None else json.dumps(body).encode(), method="GET" if body is None else "POST"), timeout=90).read())
jpg = lambda: urllib.request.urlopen("http://127.0.0.1:8890/frame.jpg", timeout=10).read(); shot = lambda n: (ROOT / f"run/tempo_pull_{n}.jpg").write_bytes(jpg())
cal = json.load(open(ROOT / "calibration/table_tip_front.json"))["planes"]; Hm = {p: np.array(cal[p]["H_pixel_to_robot_xy_m"], np.float64) for p in "BC"}
def xy_at(u, v, z): c, b = [cv2.perspectiveTransform(np.float32([[[u, v]]]), Hm[p])[0, 0] for p in "CB"]; return c + (b - c) * (z - 0.04) / 0.02
def must(r, what):
    if not r.get("ok"): log(f"   STOP: {what} failed: {r.get('error')}"); A("/stop", {}); sys.exit(2)
def find(q, n=3):
    jp = jpg()
    with cf.ThreadPoolExecutor(n) as ex: pts = [p for f in [ex.submit(er2.point, jp, q, 640, 480) for _ in range(n)] for p in f.result()["points"]]
    return (float(np.median([p[1] for p in pts])), float(np.median([p[2] for p in pts]))) if pts else None
G = {"acc": 12}; SAG = 0.008; PACK_Q = "the whole tissue pack bag (blue and pink) lying on the table"; TIS_Q = "the tissue paper sticking out of the opening of the tissue pack"
if not a.dry:
    log("WAITING: press 'ARM motion' at http://127.0.0.1:8890")
    while A("/state")["armed_s"] < 90: time.sleep(1)
def mv(x, y, z, sp, what): must(A("/move_tip", {"x": x, "y": y, "z": z + SAG, "roll": 0.0, "speed": sp, "timeout": 40, "low_ok": True, **G}), what)
st = A("/state"); g8 = st["joints"]["8"]["raw"]; pack_xy = (-0.016, 0.230)                       # where the pack sat when the first grasp started (grasp xy of the first run)
if 1345 < g8 < 2300 and not a.dry:                                                              # still holding the pack from the last run: put it down gently where it was
    log(f"the gripper still holds something ({g8} ticks): putting the pack back down"); x0, y0, z0 = st["tip"]
    mv(pack_xy[0], pack_xy[1] + 0.005, max(z0 - SAG, 0.075), 60, "carry the pack back over its place"); mv(pack_xy[0], pack_xy[1], 0.052, 40, "lower the pack onto the table")
    A("/gripper", {"to": 2650}); time.sleep(1); mv(pack_xy[0], pack_xy[1] + 0.04, 0.14, 150, "retreat"); shot("putdown"); log("pack put down, gripper open")
shot("0_start"); tis = find(TIS_Q)
if not tis: log("STOP: could not find the tissue"); sys.exit(3)
log(f"tissue at pixel ({tis[0]:.0f},{tis[1]:.0f})")
if a.dry: log("dry run: stopping"); sys.exit(0)
for k, z in enumerate([float(s) for s in a.zs.split(",")], 1):
    pk0 = find(PACK_Q, 2); tis = find(TIS_Q) or tis; xy = xy_at(tis[0], tis[1], z); x, y = float(xy[0]), float(xy[1]); log(f"try {k}: pinch the TIP at z {z} xy ({x:+.3f},{y:.3f}); pack at pixel {pk0}")
    A("/gripper", {"to": 2650}); mv(x, y + 0.03, 0.12, 400, "move above the tissue tip"); mv(x, y, z + 0.03, 150, "lower above the tip"); mv(x, y, z, 50, "lower the open jaws around the tip"); shot(f"{k}_around")
    r = A("/gripper", {"to": "close"}); tk = r.get("gripper_ticks") or 0; log(f"   close: ticks {tk} mA {r.get('mA')} closed_on_object {r.get('closed_on_object')}")
    if not (tk > 1345 or r.get("closed_on_object")): log("   nothing gripped - opening and retrying higher"); A("/gripper", {"to": 2650}); mv(x, y + 0.03, 0.14, 300, "retreat"); continue
    mv(x, y, z + 0.035, 30, "lift 3.5 cm"); time.sleep(1); pk1 = find(PACK_Q, 2); shot(f"{k}_lifted"); moved = (pk0 and pk1 and np.hypot(pk1[0] - pk0[0], pk1[1] - pk0[1]) > 10)
    log(f"   pack pixel before {pk0} after the lift {pk1}")
    if moved:
        log("   the WHOLE PACK came up - putting it back down and retrying higher"); mv(x, y, z, 30, "lower the pack back"); A("/gripper", {"to": 2650}); time.sleep(1); mv(x, y + 0.03, 0.14, 150, "retreat"); shot(f"{k}_putback"); continue
    mv(x, y + 0.04, z + 0.07, 50, "pull the tissue out"); shot("final_pulled"); log(f"   holding the tissue: gripper {A('/state')['joints']['8']['raw']} ticks"); log("DONE - tissue tip pinched and pulled (check run/tempo_pull_final_pulled.jpg)"); sys.exit(0)
log("DONE - no clean tissue grip after all tries")
