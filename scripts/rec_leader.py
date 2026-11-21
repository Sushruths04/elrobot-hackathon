"""Log the LEADER arm (the action in LeRobot terms) at ~50 Hz with epoch timestamps -> run/demos/<name>_leader.jsonl. Stop: touch run/rec_leader_stop or Ctrl+C."""
import json, os, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src")); os.chdir(ROOT)
from normakit.station_backend import StationBackend
name = sys.argv[1] if len(sys.argv) > 1 else "demo"; leader = json.loads((ROOT / "run/buses.json").read_text())["leader"]; stop = ROOT / "run/rec_leader_stop"; stop.unlink(missing_ok=True)
be = StationBackend("localhost:8888", bus_serial=leader, require_motors=False); be.start(); time.sleep(1.0)
f = open(ROOT / f"run/demos/{name}_leader.jsonl", "w"); n = 0; print("leader recording", name, flush=True)
while not stop.exists():
    t = time.time(); st = be.state(); f.write(json.dumps({"t": t, "q": {str(i): j.raw for i, j in st.joints.items()}}) + "\n"); n += 1
    if n % 50 == 0: f.flush()
    time.sleep(max(0, 0.02 - (time.time() - t)))
f.close(); print("saved", n, "samples", flush=True)
