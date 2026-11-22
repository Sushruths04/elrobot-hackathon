"""Autonomous spoon task: teleop demo (113219 ep3) + closed-loop WRIST-CAMERA correction of the grasp + grip check + retries.
  1. replay the demo approach up to 0.7 s before the gripper closes
  2. wrist camera: find the spoon handle, compare with where the demo saw it at that moment, nudge the tip (image Jacobian from 2 probe moves) until it matches
  3. continue the demo (grasp, scoop, pour) shifted by the correction; after the close check the jaws stopped on the handle (not closed empty)
  4. empty grip -> open, back to the pre-grasp pose, look again, retry (max --attempts)
Prints DONE / STOP for film_task.  --check: detection + checks only, never moves.
   scripts/film_task.py spoon_auto"""
import argparse, io, json, sys, time, urllib.request
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from PIL import Image
from normakit import er2
from normakit.core import norm_to_rad
from normakit.kin import Kin
from normakit.station_backend import StationBackend, REG_TORQUE, REG_GOAL_POS, REG_GOAL_SPEED, REG_ACC
ap = argparse.ArgumentParser(); ap.add_argument("--check", action="store_true"); ap.add_argument("--speed", type=float, default=0.8); ap.add_argument("--attempts", type=int, default=3)
ap.add_argument("--tol", type=float, default=12, help="wrist-image handle error (px) accepted"); a = ap.parse_args()
T0 = time.time(); log = lambda *m: print(f"[{time.time() - T0:5.1f}s]", *m, flush=True)
A = lambda p, b=None, t=60: json.loads(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8891" + p, data=None if b is None else json.dumps(b).encode(), method="GET" if b is None else "POST"), timeout=t).read())
REF = json.loads((ROOT / "results/spoon_auto_reference.json").read_text())   # built by --build-ref below (demo trajectory + wrist reference point)
T = np.array(REF["t"]); S = np.array(REF["state"]); IG = REF["grasp_index"]; TP = REF["pregrasp_index"]; P_DEMO = np.array(REF["wrist_handle_px"])
HANDLE = "the wooden spoon handle (light brown wood) between or just in front of the black gripper jaws"
kin = Kin(); wd = lambda x, y: ((x - y + 2048) % 4096) - 2048

def wrist_jpeg(): return urllib.request.urlopen("http://127.0.0.1:8893/frame.jpg?c=wrist", timeout=5).read()
def detect(n=3):
    pts = []
    for _ in range(n):
        jp = wrist_jpeg(); im = Image.open(io.BytesIO(jp))
        try:
            r = er2.point(jp, HANDLE, im.width, im.height); p = r.get("points") or []
            if p: pts.append(p[0][1:3] if len(p[0]) == 3 else p[0])
        except Exception as e: log("detect error", str(e)[:80])
    return np.median(np.array(pts, float), 0) if len(pts) >= 2 else None

if a.check:
    p = detect(); log(f"wrist handle now {None if p is None else np.round(p, 1)} | demo reference {np.round(P_DEMO, 1)} (only meaningful at the pre-grasp pose)")
    zs = [float(kin.jaw_tip_pose({i: norm_to_rad(i, S[k, i - 1]) for i in range(1, 8)})[0][2]) for k in range(0, len(S), 5)]; log(f"demo path tip height min {min(zs):.3f} m"); sys.exit(0)

log("WAITING: arm must be enabled (ARM motion button)")
while A("/state")["armed_s"] < 60: time.sleep(1)
A("/mirror", {"on": False})
be = StationBackend("localhost:8888"); be.start(); time.sleep(1.5); st0 = be.state(); J = list(range(1, 9))
lo = {i: st0.joints[i].lo for i in J}; span = {i: (st0.joints[i].hi - st0.joints[i].lo) % 4096 for i in J}
ticks = lambda i, n: (lo[i] + n * span[i]) % 4096
def norm_now():
    s = be.state(); return np.array([((s.joints[i].raw - lo[i]) % 4096) / span[i] if ((s.joints[i].raw - lo[i]) % 4096) <= span[i] + (4096 - span[i]) / 2 else (((s.joints[i].raw - lo[i]) % 4096) - 4096) / span[i] for i in J])
put = lambda g: be._sync(REG_GOAL_POS, {i: (int(round(v)) % 4096).to_bytes(2, "little") for i, v in g.items()})
hold = lambda: put({i: be.state().joints[i].raw for i in J})
def stiff():
    be._sync(REG_TORQUE, {i: b"\x01" for i in J}); be._sync(0x30, {i: (800 if i in (2, 3, 4) else 1000).to_bytes(2, "little") for i in J[:7]})
    be._sync(REG_ACC, {i: bytes([20 if i in (2, 3, 4) else 40]) for i in J}); be._sync(REG_GOAL_SPEED, {i: (500).to_bytes(2, "little") for i in J})
def goto(n, timeout=20):
    """move to a normalised pose with armd's path check (staged if refused)"""
    tk = {str(i): int(round(ticks(i, n[i - 1]))) for i in J}; r = A("/move_joints", {"ticks": tk, "speed": 300, "wait": True, "timeout": timeout})
    if not r.get("ok"):
        for stage in ((5, 6, 7, 8), (3, 4), (1, 2)): r = A("/move_joints", {"ticks": {str(i): tk[str(i)] for i in stage}, "speed": 300, "wait": True, "timeout": timeout})
    return r
def follow(path_t, path_s, t_from, t_to):
    """replay path between demo times t_from..t_to at a.speed with lag + current watchdogs; returns False if stopped"""
    stiff(); t_start = time.time(); bad = 0
    while True:
        tp = t_from + (time.time() - t_start) * a.speed
        if tp >= t_to: break
        n = [np.interp(tp, path_t, path_s[:, j]) for j in range(8)]; g = {i: ticks(i, n[i - 1]) for i in J}; put(g); s = be.state()
        if max(s.joints[i].current_ma for i in range(1, 8)) > 700: hold(); log("HIGH CURRENT - held"); return False
        lag = max(abs(wd(g[i], s.joints[i].raw)) for i in (2, 3, 4)); bad = bad + 1 if lag > 150 else 0
        if bad > 40: hold(); log(f"arm not following (lag {lag:.0f}) - held"); return False
        time.sleep(0.04)
    put({i: ticks(i, path_s[np.searchsorted(path_t, t_to) - 1][i - 1]) for i in J}); time.sleep(0.6); return True
def tip(n): return np.array(kin.jaw_tip_pose({i: norm_to_rad(i, n[i - 1]) for i in range(1, 8)})[0], float)
def shift_tip(n, dxy):
    """joint-space step (joints 1-4, damped least squares) that moves the jaw tip by dxy in x,y with z held"""
    n = np.array(n, float); J3 = np.zeros((3, 4)); base = tip(n); h = 0.004
    for c, j in enumerate(range(4)):
        m = n.copy(); m[j] += h; J3[:, c] = (tip(m) - base) / h
    target = np.array([dxy[0], dxy[1], 0.0]); dq = J3.T @ np.linalg.solve(J3 @ J3.T + 1e-4 * np.eye(3), target); out = n.copy(); out[:4] += dq; return out

first = int(np.argmax(np.abs(S[:, :7] - S[0, :7]).max(1) > 0.004)); t_first = T[max(0, first - 10)]
log(f"demo: movement from {t_first:.1f}s, pre-grasp at {T[TP]:.1f}s, gripper closes at {T[IG]:.1f}s, end {T[-1]:.1f}s")
log("moving to the demo start pose"); log(goto(S[0])); err0 = np.abs(norm_now()[:7] - S[0, :7]).max(); log(f"start pose error {err0:.3f} (norm)")
if err0 > 0.03: hold(); log("STOP - could not reach the start pose"); sys.exit(1)
if not follow(T, S, t_first, T[TP]): log("STOP - approach interrupted"); sys.exit(2)
offset = np.zeros(8)
for attempt in range(1, a.attempts + 1):
    log(f"attempt {attempt}: looking with the wrist camera")
    Jimg = None
    for it in range(5):
        p = detect()
        if p is None: log("handle not found in the wrist image"); break
        e = p - P_DEMO; log(f"  handle at {np.round(p, 1)} px, demo {np.round(P_DEMO, 1)}, error {np.round(e, 1)} px")
        if np.linalg.norm(e) < a.tol: break
        cur = norm_now()
        if Jimg is None:      # image Jacobian from two 1 cm probe moves
            cols = []
            for d in ((0.01, 0), (0, 0.01)):
                goto(shift_tip(cur, d), 8); q = detect(); goto(cur, 8)
                cols.append(((q - p) / 0.01) if q is not None else np.array([np.nan, np.nan]))
            Jimg = np.array(cols).T; log(f"  image Jacobian px/m {np.round(Jimg, 0).tolist()}")
            if not np.isfinite(Jimg).all() or abs(np.linalg.det(Jimg)) < 1e3: log("  probe detection failed - no correction"); Jimg = None; break
        dxy = np.linalg.solve(Jimg, -e); dxy = dxy * min(1.0, 0.03 / max(np.linalg.norm(dxy), 1e-9)); log(f"  correcting tip by {np.round(dxy * 100, 1)} cm")
        goto(shift_tip(cur, dxy), 8)
    offset = np.zeros(8); offset[:7] = norm_now()[:7] - S[TP, :7]; log(f"joint offset applied to the rest of the demo: {np.round(offset[:7], 3).tolist()}")
    S2 = S + offset; S2[:, 7] = S[:, 7]
    zs = [float(tip(S2[k])[2]) for k in range(TP, len(S2), 5)]
    if min(zs) < 0.03: hold(); log(f"STOP - corrected path goes to {min(zs):.3f} m"); sys.exit(3)
    t_check = min(T[-1], T[IG] + 1.5)
    if not follow(T, S2, T[TP], t_check): log("STOP - interrupted during the grasp"); sys.exit(2)
    g = norm_now()[7]; log(f"grip check: gripper at {g:.2f} (demo holding the handle: {S[min(len(S) - 1, IG + 45), 7]:.2f}; empty closes to ~0.0-0.1)")
    if g > 0.25:
        log("handle is in the jaws - scooping and pouring")
        ok = follow(T, S2, t_check, T[-1]); log("DONE - task finished" if ok else "STOP - interrupted while scooping"); sys.exit(0 if ok else 2)
    log("grip EMPTY - opening, back to the pre-grasp pose, looking again")
    A("/gripper", {"to": "open"}); time.sleep(0.8); goto(S2[TP], 10); time.sleep(0.5)
hold(); log(f"STOP - no grasp after {a.attempts} attempts")
