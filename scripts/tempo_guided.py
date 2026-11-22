"""Guided session: the human gives the moves in chat, the assistant executes them blindly through the arm daemon while film_task records video + joints as a LeRobot episode.
   This script only keeps the recording open: it ends when run/STOP_GUIDED exists (or after --max s).  Run via film_task.py tempo_guided."""
import argparse, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser(); ap.add_argument("--max", type=int, default=900); a = ap.parse_args(); stop = ROOT / "run/STOP_GUIDED"; stop.unlink(missing_ok=True); T0 = time.time()
print(f"[  0.0s] RECORDING - guided by the human; moves are executed from chat (touch run/STOP_GUIDED to finish)", flush=True)
while time.time() - T0 < a.max and not stop.exists(): time.sleep(0.3)
print(f"[{time.time()-T0:5.1f}s] DONE - guided session recorded", flush=True)
