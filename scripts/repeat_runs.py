#!/usr/bin/env python
"""Repeat the zero-shot pick-and-place N times. Between runs YOU move the pen out of the box to a NEW spot in the reach zone; when the pen is stable
on the table (not in the box) for ~6 s, the next run starts by itself. Results (pen position, outcome, model verdict) -> run/runs.jsonl.
    .venv/bin/python scripts/repeat_runs.py 3"""
import json, math, re, subprocess, sys, time, urllib.request
from pathlib import Path
import numpy as np, cv2
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from normakit import er2
N = int(sys.argv[1]) if len(sys.argv) > 1 else 3; T0 = time.time(); log = lambda *m: print(f"[{time.time()-T0:6.1f}s]", *m, flush=True)
cal = json.loads((ROOT / "calibration/table_tip_front.json").read_text())["planes"]
px2xy = lambda pl, u, v: cv2.perspectiveTransform(np.float32([[[u, v]]]), np.array(cal[pl]["H_pixel_to_robot_xy_m"], np.float64))[0, 0]
frame = lambda: urllib.request.urlopen("http://127.0.0.1:8890/frame.jpg", timeout=10).read()
def pen_and_box():
    jp = frame(); m = er2.point(jp, "the two ends of the marker (pen) lying on the table (one point per end)", 640, 480)["points"]; b = er2.point(jp, "the center of the white box", 640, 480)["points"]
    if len(m) < 2 or not b: return None
    P = [px2xy("B", x, y) for _, x, y in m[:2]]; c = (P[0] + P[1]) / 2; L = float(np.linalg.norm(P[0] - P[1])); bb = px2xy("A", b[0][1], b[0][2])
    return c, L, bb
results = []
det_pose = json.loads((ROOT / "run/detect_pose_follower.json").read_text())["raw"]
def A(path, body=None, timeout=60):
    req = urllib.request.Request("http://127.0.0.1:8891" + path, data=None if body is None else json.dumps(body).encode(), method="GET" if body is None else "POST")
    with urllib.request.urlopen(req, timeout=timeout) as r: return json.loads(r.read())
for k in range(1, N + 1):
    try:
        if A("/state")["armed_s"] > 30: log("moving to the detection pose so the arm does not hide the pen"); A("/move_tip", {"x": 0.0, "y": 0.20, "z": 0.26, "roll": 0.0, "speed": 600, "timeout": 20}); A("/move_joints", {"ticks": det_pose, "speed": 600, "low_ok": True, "timeout": 25})
    except Exception as e: log("could not move to the detection pose:", str(e)[:80])
    log(f"=== run {k}/{N}: put the pen OUT of the box at a NEW spot inside the reach zone (hands away, then wait 6 s) ===")
    last = None; stable = 0; t_w = time.time()
    while stable < 2:
        if time.time() - t_w > 600: log("no pen placed within 10 min - stopping"); sys.exit(1)
        try: r = pen_and_box()
        except Exception as e: r = None
        if r is None: stable = 0; time.sleep(2); continue
        c, L, bb = r; out_of_box = np.hypot(*(c - bb)) > 0.09 and 0.07 < L < 0.22 and -0.2 < c[0] < 0.25 and 0.12 < c[1] < 0.40
        if out_of_box and last is not None and np.hypot(*(c - last)) < 0.02: stable += 1
        else: stable = 0
        last = c if out_of_box else None; time.sleep(3)
    log(f"pen at ({last[0]:+.3f},{last[1]:.3f}) stable - starting the run")
    t = time.time(); p = subprocess.run([str(ROOT / ".venv/bin/python"), str(ROOT / "scripts/pick_place.py")], capture_output=True, text=True, cwd=ROOT, timeout=400); out = p.stdout + p.stderr; (ROOT / f"logs/run_{k}_{int(time.time())}.log").write_text(out)
    inside = re.search(r"inside box: (True|False)", out); done = "DONE" in out; stop = re.search(r"(STOP at [^\n]*|ABORT[^\n]*|not held[^\n]*)", out)
    res = {"run": k, "pen_xy": [round(float(last[0]), 3), round(float(last[1]), 3)], "seconds": round(time.time() - t, 1), "completed": done, "model_says_inside": inside.group(1) if inside else None, "stopped": stop.group(1)[:120] if stop else None}
    results.append(res); open(ROOT / "run/runs.jsonl", "a").write(json.dumps(res) + "\n"); log("RESULT", json.dumps(res))
ok = sum(1 for r in results if r["completed"] and r["model_says_inside"] == "True"); log(f"SUMMARY: {ok}/{len(results)} runs completed with the model confirming the pen in the box")
