"""Replay of teleop demo ep14 (spoon -> scoop -> spoon into the cup) with a firm grip, after the objects are aligned to the demo with the live overlay
(scripts/align_view.py, http://127.0.0.1:8894). Moves to the demo start pose first (armd, path-checked), then scripts/tempo_replay.py (lag + current watchdogs)."""
import json, os, sys, time, urllib.request
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from normakit.station_backend import StationBackend
A = lambda p, b=None, t=60: json.loads(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8891" + p, data=None if b is None else json.dumps(b).encode(), method="GET" if b is None else "POST"), timeout=t).read())
print("WAITING: arm must be enabled (ARM motion button)", flush=True)
while A("/state")["armed_s"] < 120: time.sleep(1)
A("/mirror", {"on": False})
S0 = json.loads((ROOT / "results/films/spoon_demo_ep14_firm_samples.json").read_text())[0][1]
be = StationBackend("localhost:8888", require_motors=False); be.start(); time.sleep(1.5); st = be.state()
tk = {str(i): int(round((st.joints[i].lo + S0[i - 1] * ((st.joints[i].hi - st.joints[i].lo) % 4096)) % 4096)) for i in range(1, 9)}
r = A("/move_joints", {"ticks": tk, "speed": 250, "wait": True, "timeout": 30}); print("demo start pose:", {k: r.get(k) for k in ("ok", "error", "worst_ticks_off")}, flush=True)
if not r.get("ok"): A("/stop", {}); print("STOP - could not reach the demo start pose", flush=True); sys.exit(1)
time.sleep(0.8)
os.execv(str(ROOT / ".venv/bin/python"), [str(ROOT / ".venv/bin/python"), "-u", str(ROOT / "scripts/tempo_replay.py"), "--samples", "results/films/spoon_demo_ep14_firm_samples.json", "--speed", "0.8"])
