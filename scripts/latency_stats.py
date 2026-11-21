#!/usr/bin/env python
"""Where does the time go? Measures every stage of the stack and writes run/latency_stats.json.
Moves only joint 5 by +-60 ticks (wrist, harmless) for the step-response test.   .venv/bin/python scripts/latency_stats.py [--no-move] [--er2]"""
import json, os, statistics as st, sys, time
from pathlib import Path
T_PROC = time.time(); ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src")); os.chdir(ROOT)
R = {}; ms = lambda s: round(s * 1000, 1)
def rec(k, v, note=""): R[k] = v; print(f"{k:42} {v if isinstance(v, str) else (str(v) + ' ms')}   {note}", flush=True)
t = time.time(); import numpy, cv2; from normakit import station_backend as sb; from normakit.station_backend import StationBackend, REG_GOAL_POS, REG_GOAL_SPEED, REG_ACC, front_camera; rec("python import (numpy,cv2,backend)", ms(time.time() - t))
t = time.time(); be = StationBackend("localhost:8888"); be.start(); rec("backend start() to first full state", ms(time.time() - t))
time.sleep(1.0); s = be.state(); g = {i: (s.joints[i].raw % 4096).to_bytes(2, "little") for i in range(1, 9)}
w = []
for _ in range(20): t = time.time(); be._sync(REG_GOAL_POS, g); w.append(time.time() - t)
rec("command write latency median", ms(st.median(w)), f"p95 {ms(sorted(w)[18])} ms, max {ms(max(w))} ms")
rec("state() read (cached)", ms((lambda t: (t, [be.state() for _ in range(1000)], time.time() - t)[2] / 1000)(time.time())))
# state freshness: how long until a changed value shows up after a physical change is unknowable; measure update rate of the cached state
seen = []; last = None; t0 = time.time()
while time.time() - t0 < 2.0:
    v = be.state().joints[1].current_ma + be.state().joints[5].raw * 7
    if v != last: seen.append(time.time()); last = v
    time.sleep(0.002)
rec("state update changes seen in 2 s", str(len(seen)), "(0 = frozen/stale; many = fresh)")
if "--no-move" not in sys.argv:
    j = 5; p0 = be.state().joints[j].raw; be._sync(REG_ACC, {j: bytes([40])}); be._sync(REG_GOAL_SPEED, {j: (800).to_bytes(2, "little")}); time.sleep(0.2)
    for tgt_off in (60, -60):
        t = time.time(); be._sync(REG_GOAL_POS, {j: ((p0 + tgt_off) % 4096).to_bytes(2, "little")}); tw = time.time() - t; first = settle = None
        while time.time() - t < 4:
            d = abs(be.state().joints[j].raw - (p0 + tgt_off))
            if first is None and abs(be.state().joints[j].raw - p0) > 4: first = time.time() - t
            if d <= 6: settle = time.time() - t; break
            time.sleep(0.005)
        rec(f"step {tgt_off:+d} ticks: write+first motion", ms(first) if first else "none", f"write {ms(tw)} ms; settled in {ms(settle) if settle else 'timeout'} ms (speed 800 ticks/s)")
    be._sync(REG_GOAL_POS, {j: (p0 % 4096).to_bytes(2, "little")}); time.sleep(1.0)
cams = []
for c in be.cameras():
    try: be.frame_jpeg(c); cams.append(c)
    except Exception: pass
fc = front_camera(be); t = time.time(); n = 0; lastb = None; ch = 0
while time.time() - t < 2.0:
    b = be.frame_jpeg(fc)
    if b is not lastb: ch += 1; lastb = b
    time.sleep(0.005)
rec("front camera new frames per second", f"{ch / 2:.1f}", f"{len(cams)} cameras streaming")
t = time.time(); import cv2 as _c; from normakit import vision; arr = vision.decode(be.frame_jpeg(fc)); rec("jpeg decode", ms(time.time() - t))
t = time.time(); d = vision.find_dark_object(arr, roi=(60, 185, 500, 470), min_area=300, max_area=30000); rec("blob detector", ms(time.time() - t))
t = time.time(); from normakit.kin import Kin; kin = Kin(); rec("Kin() load pybullet", ms(time.time() - t))
from normakit.core import norm_to_rad
q = {i: norm_to_rad(i, s.joints[i].norm) for i in range(1, 8)}
t = time.time(); [kin.fk(q) for _ in range(100)]; rec("pybullet fk", ms((time.time() - t) / 100))
t = time.time(); kin.jaw_tip_pose(q); rec("MuJoCo model load + first fk", ms(time.time() - t))
t = time.time(); [kin.jaw_tip_pose(q) for _ in range(100)]; rec("jaw_tip_pose fk", ms((time.time() - t) / 100))
t = time.time(); r = kin.ik_jaw_tip((0.0, 0.27, 0.10), roll_rad=0.0, seed_rad=q); rec("ik_jaw_tip (7-DoF, multi-seed)", ms(time.time() - t), f"err {r[1]*1000:.1f} mm")
t = time.time(); r = kin.ik_jaw_tip((0.0, 0.27, 0.10), roll_rad=0.0, seed_rad=r[0]); rec("ik_jaw_tip warm seed (nearby)", ms(time.time() - t), f"err {r[1]*1000:.1f} mm")
if "--er2" in sys.argv:
    from normakit import er2; jp = be.frame_jpeg(fc); h, ww = arr.shape[:2]
    for k in range(2):
        try: t = time.time(); er2.point(jp, "the robot gripper", ww, h); rec(f"Gemini ER-2 point #{k+1}", ms(time.time() - t))
        except Exception as e: rec(f"Gemini ER-2 point #{k+1}", "FAILED", str(e)[:60]); break
import urllib.request
for path in ("/api/state", "/frame.jpg"):
    try: t = time.time(); urllib.request.urlopen("http://127.0.0.1:8890" + path, timeout=5).read(); rec(f"web UI {path}", ms(time.time() - t))
    except Exception as e: rec(f"web UI {path}", "down")
rec("this whole script", ms(time.time() - T_PROC))
(ROOT / "run/latency_stats.json").write_text(json.dumps({"time": time.strftime("%F %T"), "ms": R}, indent=1)); os._exit(0)
