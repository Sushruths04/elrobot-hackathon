"""Record the follower's joint ticks continuously (20 Hz) while you hand-guide ONE pick (read-only, torque stays off, nothing moves by itself).
Stop: create the file run/demo_stop (or wait max 120 s).  Output: run/demo_traj.json = [[t_seconds, {joint: ticks}], ...]"""
import json, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from normakit.station_backend import StationBackend
stop = ROOT / "run/demo_stop"; stop.unlink(missing_ok=True)
be = StationBackend("localhost:8888"); be.start(); time.sleep(1.0); traj = []; t0 = time.time()
print("RECORDING: do the whole pick by hand now (above the cup, down, close the gripper, lift). Tell me 'done' when finished.", flush=True)
LIMIT = float(sys.argv[1]) if len(sys.argv) > 1 else 120
while time.time() - t0 < LIMIT and not stop.exists():
    st = be.state(); traj.append([round(time.time() - t0, 3), {str(i): j.raw for i, j in st.joints.items()}]); time.sleep(0.05)
(ROOT / "run/demo_traj.json").write_text(json.dumps(traj)); span = {i: max(t[1][i] for t in traj) - min(t[1][i] for t in traj) for i in traj[0][1]}
print(f"SAVED {len(traj)} samples over {traj[-1][0]:.1f} s | joint travel in ticks: {span}", flush=True); import os; os._exit(0)
