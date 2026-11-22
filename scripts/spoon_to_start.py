"""Move the follower slowly to the first pose of the spoon demo (results/films/spoon_teleop_demo_113219_ep3_samples.json). Needs ARM. Turns mirroring OFF first."""
import json, sys, time, urllib.request
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from normakit.station_backend import StationBackend
A = lambda p, b=None: json.loads(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8891" + p, data=None if b is None else json.dumps(b).encode(), method="GET" if b is None else "POST"), timeout=60).read())
if A("/state")["armed_s"] < 20: sys.exit("NOT ARMED - press ARM on http://127.0.0.1:8890 first")
print("mirroring:", A("/mirror", {"on": False}), flush=True); time.sleep(0.5)
S0 = json.loads((ROOT / "results/films/spoon_teleop_demo_113219_ep3_samples.json").read_text())[0][1]
be = StationBackend("localhost:8888"); be.start(); time.sleep(1.5); st = be.state()
ticks = {str(i): int(round((st.joints[i].lo + S0[i - 1] * ((st.joints[i].hi - st.joints[i].lo) % 4096)) % 4096)) for i in range(1, 9)}
print("target ticks", ticks, flush=True); print(A("/move_joints", {"ticks": ticks, "speed": 300, "wait": True}), flush=True)
st = be.state(); wd = lambda x, y: ((x - y + 2048) % 4096) - 2048
print("worst error after move:", max(abs(wd(ticks[str(i)], st.joints[i].raw)) for i in range(1, 8)), "ticks", flush=True)
import os; os._exit(0)
