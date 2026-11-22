"""Autonomous spoon task v2: FRONT-camera spoon detection -> shift the teleop demo's approach+grasp by the measured spoon shift,
gripper replays the operator's real squeeze command (demo ACTION, not the blocked follower state), hover photo before closing, grip check, retries with +-1.5 cm.
   scripts/film_task.py spoon_auto2      (--check: detection + shift + path checks only, no motion)"""
import argparse, glob, io, json, sys, time, urllib.request
from pathlib import Path
import numpy as np, cv2, pandas as pd
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from PIL import Image
from normakit import er2
from normakit.core import norm_to_rad
from normakit.kin import Kin
from normakit.station_backend import StationBackend, REG_TORQUE, REG_GOAL_POS, REG_GOAL_SPEED, REG_ACC
ap = argparse.ArgumentParser(); ap.add_argument("--check", action="store_true"); ap.add_argument("--speed", type=float, default=0.8); a = ap.parse_args()
T0 = time.time(); log = lambda *m: print(f"[{time.time() - T0:5.1f}s]", *m, flush=True)
A = lambda p, b=None, t=60: json.loads(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8891" + p, data=None if b is None else json.dumps(b).encode(), method="GET" if b is None else "POST"), timeout=t).read())
D = ROOT / "datasets/teleop_spoon_beans_20261122_113219"; df = pd.concat(pd.read_parquet(f) for f in sorted(glob.glob(str(D / "data/*/*.parquet")))); g = df[df.episode_index == 3]
S = np.stack(g["observation.state"].values).astype(float); ACT = np.stack(g["action"].values).astype(float); T = g["timestamp"].values - g["timestamp"].values[0]
IG = int(np.argmax(S[:, 7] < 0.5)); first = max(0, int(np.argmax(np.abs(S[:, :7] - S[0, :7]).max(1) > 0.004)) - 10)
cal = json.loads((ROOT / "calibration/table_tip_front.json").read_text())["planes"]
_hp = lambda pl, u, v: cv2.perspectiveTransform(np.float32([[[u, v]]]), np.array(cal[pl]["H_pixel_to_robot_xy_m"], np.float64))[0, 0]
xy_at = lambda z, u, v: _hp("A", u, v) + (_hp("B", u, v) - _hp("A", u, v)) * (z - 0.10) / (0.06 - 0.10)
REFPX = {k: np.array(v) for k, v in json.loads((ROOT / "run/camscan/auto_points.json").read_text()).items()}   # demo_start top/base pixels (same detector)
Q = {"top": "the top end of the wooden spoon handle (the end sticking up out of the bowl)", "base": "the point where the wooden spoon handle enters the coffee beans"}
kin = Kin(); wd = lambda x, y: ((x - y + 2048) % 4096) - 2048
def front(): return urllib.request.urlopen("http://127.0.0.1:8893/frame.jpg?c=table", timeout=5).read()
def detect(q, n=5):
    pts = []
    for _ in range(n):
        jp = front(); im = Image.open(io.BytesIO(jp))
        try:
            p = er2.point(jp, Q[q], im.width, im.height).get("points") or []
            if p: pts.append(p[0][1:3] if len(p[0]) == 3 else p[0])
        except Exception as e: log("detect error", str(e)[:60])
    if len(pts) < 3: return None, None
    P = np.array(pts, float); return np.median(P, 0), P.std(0)
def tip(n): return np.array(kin.jaw_tip_pose({i: norm_to_rad(i, n[i - 1]) for i in range(1, 8)})[0], float)
def shift_tip(n, dxy):
    n = np.array(n, float); J3 = np.zeros((3, 4)); base = tip(n); h = 0.004
    for c in range(4):
        m = n.copy(); m[c] += h; J3[:, c] = (tip(m) - base) / h
    dz = dxy[2] if len(dxy) > 2 else 0.0; goal = base + np.array([dxy[0], dxy[1], dz]); out = n.copy()
    for _ in range(12):                                            # iterate small damped steps so large shifts stay exact (one linear step went wild for 8-10 cm)
        cur = tip(out); err = goal - cur
        if np.linalg.norm(err) < 0.001: break
        for c in range(4):
            m = out.copy(); m[c] += h; J3[:, c] = (tip(m) - cur) / h
        step = err * min(1.0, 0.01 / np.linalg.norm(err)); out[:4] += J3.T @ np.linalg.solve(J3 @ J3.T + 1e-4 * np.eye(3), step)
    return out
def shifted_path(d):
    """approach (from grasp-6 s) blends in the shift, full shift from grasp-2 s to the end; gripper = operator's squeeze command"""
    t1, t2 = T[IG] - 6.0, T[IG] - 2.0; P = S.copy(); idx = list(range(0, len(S), 5)) + [len(S) - 1]; key = {}
    for k in idx:
        w = float(np.clip((T[k] - t1) / (t2 - t1), 0, 1)); key[k] = shift_tip(S[k], w * np.asarray(d)) if w > 0 else S[k].copy()
    ks = sorted(key)
    for j in range(7): P[:, j] = np.interp(np.arange(len(S)), ks, [key[k][j] for k in ks])
    P[:, 7] = ACT[:, 7]; rel = int(np.flatnonzero(ACT[:, 7] < 0.5).max()) + 1    # the operator's final release (spoon dropped into the cup)
    P[IG:rel, 7] = 0.0; return P                                  # squeeze fully from the grasp until the operator released the spoon

log(f"demo: grasp at {T[IG]:.1f}s, operator squeeze command min {ACT[IG:IG + 60, 7].min():.2f} (follower stopped at {S[IG + 45, 7]:.2f} on the handle)")
st = A("/state"); tipz = st["tip"][2]
if not a.check and tipz < 0.2: log(f"arm tip at {tipz:.2f} m may hide the spoon - moving to the start pose first")
pts = {}
def measure():
    for q in ("top", "base"):
        m, s = detect(q); pts[q] = m; log(f"spoon {q}: now {None if m is None else np.round(m, 1)} px (spread {None if s is None else np.round(s, 1)}), demo {np.round(REFPX['demo_start|' + q], 1)} px")
    if pts["top"] is None or pts["base"] is None: return None
    d_top = xy_at(0.10, *pts["top"]) - xy_at(0.10, *REFPX["demo_start|top"]); d_base = xy_at(0.06, *pts["base"]) - xy_at(0.06, *REFPX["demo_start|base"])
    d = d_top; log(f"spoon shift vs demo: top {np.round(d_top * 100, 1)} cm (base {np.round(d_base * 100, 1)} cm, not used: calibration poor at bean height) -> using the top")
    if np.linalg.norm(d) > 0.12: log("shift > 12 cm - detection probably wrong"); return None
    return d
if a.check:
    d = measure()
    if d is not None:
        P = shifted_path(d); zs = [tip(P[k])[2] for k in range(0, len(P), 5)]; log(f"shifted grasp tip {np.round(tip(P[IG]), 3)} (demo {np.round(tip(S[IG]), 3)}), path min height {min(zs):.3f} m")
    sys.exit(0)

log("WAITING: arm must be enabled (ARM motion button)")
while A("/state")["armed_s"] < 90: time.sleep(1)
A("/mirror", {"on": False})
be = StationBackend("localhost:8888"); be.start(); time.sleep(1.5); s0 = be.state(); J = list(range(1, 9))
lo = {i: s0.joints[i].lo for i in J}; span = {i: (s0.joints[i].hi - s0.joints[i].lo) % 4096 for i in J}; ticks = lambda i, n: (lo[i] + n * span[i]) % 4096
def nrm(i, raw): r = (raw - lo[i]) % 4096; return (r - 4096 if r > span[i] + (4096 - span[i]) / 2 else r) / span[i]
put = lambda gg: be._sync(REG_GOAL_POS, {i: (int(round(v)) % 4096).to_bytes(2, "little") for i, v in gg.items()})
hold = lambda: put({i: be.state().joints[i].raw for i in J})
def stiff():
    be._sync(REG_TORQUE, {i: b"\x01" for i in J}); be._sync(0x30, {i: (800 if i in (2, 3, 4) else 1000).to_bytes(2, "little") for i in J[:7]})
    be._sync(REG_ACC, {i: bytes([20 if i in (2, 3, 4) else 40]) for i in J}); be._sync(REG_GOAL_SPEED, {i: (500).to_bytes(2, "little") for i in J})
def goto(n, timeout=25):
    tk = {str(i): int(round(ticks(i, n[i - 1]))) for i in J}; r = A("/move_joints", {"ticks": tk, "speed": 300, "wait": True, "timeout": timeout})
    if not r.get("ok"):
        for stage in ((5, 6, 7, 8), (3, 4), (1, 2)): r = A("/move_joints", {"ticks": {str(i): tk[str(i)] for i in stage}, "speed": 300, "wait": True, "timeout": timeout})
    return r
def follow(P, t_from, t_to):
    stiff(); t_start = time.time(); bad = 0
    while True:
        tp = t_from + (time.time() - t_start) * a.speed
        if tp >= t_to: break
        n = [np.interp(tp, T, P[:, j]) for j in range(8)]; gg = {i: ticks(i, n[i - 1]) for i in J}; put(gg); s = be.state()
        if max(s.joints[i].current_ma for i in range(1, 8)) > 700: hold(); log("HIGH CURRENT - held"); return False
        lag = max(abs(wd(gg[i], s.joints[i].raw)) for i in (2, 3, 4)); bad = bad + 1 if lag > 150 else 0
        if bad > 40: hold(); log(f"arm not following (lag {lag:.0f}) - held"); return False
        time.sleep(0.04)
    return True
log("start pose (arm out of the camera view)"); goto(S[0]); time.sleep(0.8)
d0 = measure()
if d0 is None: hold(); log("STOP - spoon not found reliably in the front camera"); sys.exit(1)
ASK = ("Close-up from a camera in front of a robot arm. A black two-jaw robot gripper hangs above a wooden spoon whose handle sticks up out of a bowl of coffee beans. "
       "Goal: the open jaws must surround the upper part of the wooden handle so that closing them grabs it. How must the gripper move? Use image directions: "
       "horizontal 'left' or 'right' as seen in the image; depth 'closer' (toward the camera, lower in the image) or 'farther'; vertical 'down' or 'up'; 'ok' if already right. "
       'Reply as JSON: {"horizontal": "left|right|ok", "depth": "closer|farther|ok", "vertical": "down|up|ok", "aligned": true or false, "reason": "short"}')
def crop_jpeg(center):
    im = Image.open(io.BytesIO(front())).convert("RGB"); u, v = int(center[0]), int(center[1]); box = (max(0, u - 110), max(0, v - 140), min(im.width, u + 110), min(im.height, v + 80))
    b = io.BytesIO(); im.crop(box).resize(((box[2] - box[0]) * 2, (box[3] - box[1]) * 2)).save(b, "JPEG"); return b.getvalue()
def guide(center, k_hover, P):
    """VLM tells the direction, we step 1 cm, repeat (max 7). Returns the path re-anchored at the aligned pose."""
    u, v = center; z = 0.10; o = xy_at(z, u, v); r = xy_at(z, u + 20, v) - o; c = xy_at(z, u, v + 20) - o; r /= np.linalg.norm(r); c /= np.linalg.norm(c)
    for it in range(7):
        jp = crop_jpeg(center); (ROOT / "results/photos").mkdir(exist_ok=True); (ROOT / f"results/photos/spoon_guide_{time.strftime('%H%M%S')}_{it}.jpg").write_bytes(jp)
        try: from normakit import llm as _llm; _c = _llm.config(); ans = _llm.ask(_c["provider"], _c["model"], jp, ASK, max_tokens=3000, timeout=90)
        except Exception as e: log("  vision ask failed", str(e)[:80]); break
        hz, dp, vt = str(ans.get("horizontal", "ok")).lower(), str(ans.get("depth", "ok")).lower(), str(ans.get("vertical", "ok")).lower()
        log(f"  vision says: horizontal={hz} depth={dp} vertical={vt} aligned={ans.get('aligned')} ({str(ans.get('reason', ''))[:90]})")
        if ans.get("aligned") is True or (hz.startswith("ok") and dp.startswith("ok") and vt.startswith("ok")): break
        step = np.zeros(3); st_ = 0.01
        if hz.startswith("left"): step[:2] -= st_ * r
        elif hz.startswith("right"): step[:2] += st_ * r
        if dp.startswith("closer"): step[:2] += st_ * c
        elif dp.startswith("farther"): step[:2] -= st_ * c
        if vt.startswith("down"): step[2] -= st_
        elif vt.startswith("up"): step[2] += st_
        cur = np.array([nrm(i, be.state().joints[i].raw) for i in J]); nxt = shift_tip(cur, step)
        if tip(nxt)[2] < 0.05: log("  would go below 5 cm - not stepping down"); nxt = shift_tip(cur, [step[0], step[1], 0])
        goto(nxt, 8); time.sleep(0.4)
    cur = np.array([nrm(i, be.state().joints[i].raw) for i in J]); off = np.zeros(8); off[:7] = cur[:7] - P[k_hover, :7]
    P2 = P.copy(); P2[k_hover:, :7] += off[:7]; return P2
for attempt in range(1, 4):
    P = shifted_path(np.r_[d0, 0.0]); k_hover = int(np.searchsorted(T, T[IG] - 0.3))
    zs = [tip(P[k])[2] for k in range(0, len(P), 5)]
    if min(zs) < 0.03: hold(); log(f"STOP - path too low ({min(zs):.3f} m)"); sys.exit(3)
    log(f"attempt {attempt}: coarse shift {np.round(d0 * 100, 1)} cm, hover {np.round(tip(P[k_hover]), 3)}")
    if attempt == 1:
        if not follow(P, T[first], T[k_hover]): log("STOP - approach interrupted"); sys.exit(2)
    else:
        goto(P[k_hover], 10)
    put({8: ticks(8, 0.9)}); time.sleep(0.6)                                     # jaws wide open for the alignment
    P2 = guide(pts["top"], k_hover, P)
    fn = ROOT / f"results/photos/spoon_auto2_aligned_attempt{attempt}_{time.strftime('%H%M%S')}.jpg"; fn.write_bytes(front()); log(f"aligned photo {fn.relative_to(ROOT)}")
    put({8: ticks(8, 0.0)}); time.sleep(1.5)                                     # squeeze where we are
    s = be.state(); gpos = nrm(8, s.joints[8].raw); log(f"grip check: gripper at {gpos:.2f} (empty closes to ~0.01), current {s.joints[8].current_ma} mA")
    if gpos > 0.08:
        log("SPOON IN THE JAWS - scooping and pouring"); P2[k_hover:int(np.flatnonzero(ACT[:, 7] < 0.5).max()) + 1, 7] = 0.0
        ok = follow(P2, T[k_hover], T[-1]); fn = ROOT / f"results/photos/spoon_auto2_end_{time.strftime('%H%M%S')}.jpg"; fn.write_bytes(front()); log(f"end photo {fn.relative_to(ROOT)}")
        log("DONE - task run finished (check the photo: beans in the cup?)" if ok else "STOP - interrupted while scooping"); sys.exit(0 if ok else 2)
    log("jaws closed EMPTY - opening, lifting, looking again"); put({8: ticks(8, 0.9)}); time.sleep(0.6); goto(P[int(np.searchsorted(T, T[IG] - 3.0))], 10)
hold(); log("STOP - no grasp after 3 attempts")
