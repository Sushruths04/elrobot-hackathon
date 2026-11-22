"""VLM pick-and-place of the spoon into the cup (like the pen task): camera + vision model find the objects live, the operator's demos give the HOW.
  perception: Nebius VLM (run/vlm.json, Nemotron-VL) points at handle top, handle-in-beans point, cup opening; 3D via the front camera pose solved
              from the real arm (calibration/front_camera_pose_arm_wide.json, ~1 cm); bean surface 4.5 cm, handle 9.5 cm, cup rim 9.6 cm (operator/twin)
  grasp:      middle of the exposed handle, from above, operator's grip (tilt 18 deg, roll 1.3 rad -> j7 0.77 like the demos), firm close, load check, retry
  scoop+lift: 2.5 cm dip like the demos, lift to 0.21 m
  place:      operator's release point relative to the cup (+7.1 cm x, -2.2 cm y, z 0.165, from demos ep14-17), open -> spoon drops into the cup
  verify:     VLM yes/no on a front photo. Prints DONE / STOP (film_task).  --check: perception + reachability only, no motion."""
import argparse, io, json, sys, time, urllib.request
from pathlib import Path
import numpy as np, cv2
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from normakit import er2, llm
ap = argparse.ArgumentParser(); ap.add_argument("--check", action="store_true"); ap.add_argument("--tries", type=int, default=3)
ap.add_argument("--grasp-px", help="u,v,z: grasp pixel read from the front image + its height (overrides the VLM handle points)"); ap.add_argument("--cup-px", help="u,v of the cup opening centre"); a = ap.parse_args()
T0 = time.time(); log = lambda *m: print(f"[{time.time() - T0:5.1f}s]", *m, flush=True)
A = lambda p, b=None, t=60: json.loads(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8891" + p, data=None if b is None else json.dumps(b).encode(), method="GET" if b is None else "POST"), timeout=t).read())
front_jpg = lambda: urllib.request.urlopen("http://127.0.0.1:8893/frame.jpg?c=table", timeout=5).read()
cp = json.loads((ROOT / "calibration/front_camera_pose_arm_wide.json").read_text()); cam = json.loads((ROOT / "calibration/camera_front.json").read_text())
K = np.array(cam["K"]); DD = np.array(cam["dist"]).ravel(); R = cv2.Rodrigues(np.array(cp["rvec"]))[0]; C = np.array(cp["camera_pos_robot_m"])
BEANS, HANDLE_L, CUP_RIM = 0.045, 0.095, 0.096
TILT, ROLL = 18, 1.3
REL_OFF, REL_Z, CARRY_Z = np.array([0.071, -0.022]), 0.165, 0.21
def ray(u, v):
    d = cv2.undistortPoints(np.array([[[u, v]]], np.float64), K, DD).reshape(2); r = R.T @ np.array([d[0], d[1], 1.0]); return r / np.linalg.norm(r)
def at_z(px, z): r = ray(*px); return C + (z - C[2]) / r[2] * r
def point(jp, what, n=3):
    pts = []
    for _ in range(n):
        try:
            p = er2.point(jp, what, 640, 480).get("points") or []
            if p: pts.append(p[0][1:3] if len(p[0]) == 3 else p[0])
        except Exception as e: log("  point error", str(e)[:60])
    if len(pts) < 2: return None
    P = np.array(pts, float); return np.median(P, 0) if P.std(0).max() < 25 else None
def yes(jp, q):
    c = llm.config(); r = llm.ask(c["provider"], c["model"], jp, q + ' Reply as JSON: {"answer": "yes" or "no", "reason": "short"}', max_tokens=3000, timeout=90)
    ans = str(r.get("answer", r.get("raw", ""))).lower(); return ans.startswith("yes"), str(r.get("reason", ""))[:100]
def perceive():
    if a.grasp_px and a.cup_px:
        u, v, z = map(float, a.grasp_px.split(",")); cu, cv_ = map(float, a.cup_px.split(",")); G = at_z((u, v), z); cupc = at_z((cu, cv_), CUP_RIM)
        log(f"  points read from the front image: grasp px ({u:.0f},{v:.0f}) z {z:.3f} -> {np.round(G, 3)} | cup px ({cu:.0f},{cv_:.0f}) -> {np.round(cupc, 3)}"); return G, cupc
    jp = front_jpg(); top = point(jp, "the top end of the wooden spoon handle (the end sticking up out of the bowl)"); base = point(jp, "the point where the wooden spoon handle enters the coffee beans")
    cup = point(jp, "the center of the opening of the black cup")
    if top is None or base is None or cup is None: log(f"  perception failed: top {top} base {base} cup {cup}"); return None
    B = at_z(base, BEANS); rt = ray(*top); w = C - B; bq = 2 * rt @ w; cq = w @ w - HANDLE_L ** 2; disc = bq * bq - 4 * cq
    s = (-bq - np.sqrt(max(disc, 0))) / 2 if disc > 0 else -(rt @ w); T = C + s * rt
    if T[2] < BEANS + 0.03: s = (-bq + np.sqrt(max(disc, 0))) / 2; T = C + s * rt
    G = B + 0.5 * (T - B); G[2] = max(G[2], 0.075); cupc = at_z(cup, CUP_RIM)
    log(f"  handle base {np.round(B, 3)} top {np.round(T, 3)} -> grasp {np.round(G, 3)} | cup {np.round(cupc, 3)}  (px top {np.round(top)} base {np.round(base)} cup {np.round(cup)})")
    return G, cupc
def move(p, tilt=TILT, speed=250, timeout=15):
    b = {"x": float(p[0]), "y": float(p[1]), "z": float(p[2]), "roll": ROLL, "speed": speed, "timeout": timeout}
    if tilt is not None: b["tilt"] = tilt
    return A("/move_tip", b)
FOLD = {"1": 2094, "2": 1055, "3": 1554, "4": 3020, "5": 170, "6": 2836, "7": 3021, "8": 2766}
def fold():
    r = A("/move_joints", {"ticks": FOLD, "speed": 300, "wait": True, "timeout": 25})
    if not r.get("ok"):
        for stg in (("5", "6", "7", "8"), ("3", "4"), ("1", "2")): A("/move_joints", {"ticks": {k: FOLD[k] for k in stg}, "speed": 300, "wait": True, "timeout": 25})
def photo(tag): fn = ROOT / f"results/photos/spoon_vlm_{tag}_{time.strftime('%H%M%S')}.jpg"; jp = front_jpg(); fn.write_bytes(jp); log(f"  photo {fn.relative_to(ROOT)}"); return jp

if a.check:
    pr = perceive()
    if pr:
        G, cupc = pr; rel = np.r_[cupc[:2] + REL_OFF, REL_Z]
        for nm, p, t in (("pre-grasp", G + [0, 0, 0.05], TILT), ("grasp", G, TILT), ("carry above cup", np.r_[rel[:2], CARRY_Z], None), ("release", rel, None)):
            r = A("/ik", {"x": float(p[0]), "y": float(p[1]), "z": float(p[2]), **({"tilt": t} if t else {}), "rolls": [ROLL]})["ik"]; log(f"  IK {nm} {np.round(p, 3)}: {list(r.values())[0]}")
    sys.exit(0)
log("WAITING: arm must be enabled (ARM motion button)")
while A("/state")["armed_s"] < 120: time.sleep(1)
A("/mirror", {"on": False})
held = False
for t in range(1, a.tries + 1):
    log(f"try {t}: arm out of view, looking"); fold(); time.sleep(1.0); pr = perceive()
    if pr is None: continue
    G, cupc = pr; A("/gripper", {"to": "open"})
    r = move(G + [0, 0, 0.05]); log(f"  pre-grasp ok={r.get('ok')} {r.get('error') or ''}")
    if not r.get("ok"): continue
    r = move(G, speed=150); log(f"  down to the handle ok={r.get('ok')} {r.get('error') or ''}"); time.sleep(0.4); photo(f"before_close_t{t}")
    A("/gripper", {"to": "close"}); last = None
    for _ in range(25):
        time.sleep(0.15); raw = A("/state")["joints"]["8"]["raw"]
        if last is not None and abs(raw - last) < 3: break
        last = raw
    j8 = A("/state")["joints"]["8"]; log(f"  jaws stopped at {j8['raw']} ticks, {j8['mA']} mA (empty ~1263-1326)")
    if j8["raw"] > 1300 and j8["mA"] > 40: held = True; log("  SPOON IN THE JAWS"); break
    log("  jaws empty - opening, retry"); A("/gripper", {"to": "open"}); time.sleep(0.5); move(G + [0, 0, 0.06])
if not held: fold(); log(f"STOP - no grasp after {a.tries} tries"); sys.exit(1)
p = np.array(A("/state")["tip"], float); log("scoop dip"); move(p - [0, 0, 0.025], speed=120); time.sleep(0.5); move(p, speed=120)
log("lift"); r = move(np.r_[p[:2], CARRY_Z], tilt=None, speed=200); log(f"  ok={r.get('ok')} {r.get('error') or ''}")
rel = np.r_[cupc[:2] + REL_OFF, REL_Z]; log(f"carry above the cup -> release point {np.round(rel, 3)}")
r = move(np.r_[rel[:2], CARRY_Z], tilt=None, speed=200); log(f"  ok={r.get('ok')} {r.get('error') or ''}")
r = move(rel, tilt=None, speed=120); log(f"  lowered ok={r.get('ok')} {r.get('error') or ''}"); time.sleep(0.5); jp = photo("before_release")
ok, why = yes(jp, "Is the wooden spoon held by the robot gripper with its end above or inside the black cup?"); log(f"  VLM before release: {'yes' if ok else 'no'} ({why})")
A("/gripper", {"to": "open"}); time.sleep(1.2); move(np.r_[rel[:2], CARRY_Z], tilt=None, speed=150); time.sleep(0.6); jp = photo("after_release")
ok, why = yes(jp, "Is the wooden spoon now inside the black cup?"); log(f"  VLM after release: spoon in the cup = {'yes' if ok else 'no'} ({why})")
fold(); log("DONE - spoon placed in the cup (VLM yes)" if ok else "STOP - VLM says the spoon is not in the cup (look at the photo)")
