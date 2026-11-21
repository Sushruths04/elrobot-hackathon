"""Read-only: how far is the follower from the START pose of a recorded demo? Prints a line when something changes; MATCHED when every joint is within 200 ticks (about 17 degrees)."""
import json, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from normakit.station_backend import StationBackend
T = json.loads((ROOT / "run/demo_traj_pick1.json").read_text())
mv = [k for k in range(1, len(T)) if max(abs(T[k][1][str(i)] - T[k-1][1][str(i)]) for i in range(1, 9)) > 3]; start = {i: T[max(0, mv[0] - 5)][1][str(i)] for i in range(1, 9)}
wd = lambda a, b: ((a - b + 2048) % 4096) - 2048
be = StationBackend("localhost:8888"); be.start(); time.sleep(1.0); last = None; lt = 0
while True:
    cur = {i: be.state().joints[i].raw for i in range(1, 9)}; d = {i: wd(cur[i], start[i]) for i in cur}; ok = all(abs(v) <= 200 for v in d.values())
    key = (ok, tuple(round(v / 60) for v in d.values()))
    if key != last and time.time() - lt > 2:
        last, lt = key, time.time()
        print("MATCHED - arm is at the start pose" if ok else "to go (ticks, degrees): " + ", ".join(f"j{i} {v:+d} ({v*360/4096:+.0f} deg)" for i, v in d.items() if abs(v) > 200), flush=True)
    time.sleep(0.3)
