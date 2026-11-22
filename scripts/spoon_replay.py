"""Single-shot test: replay teleop demo 113219 ep3 (spoon -> scoop beans -> pour into cup; follower joint trajectory) on the follower arm.
Filmed + recorded via  scripts/film_task.py spoon_replay .  Needs: mirroring OFF, arm at the demo start pose (folded rest), ARM pressed. Add --check to only run the checks."""
import os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
os.execv(str(ROOT / ".venv/bin/python"), [str(ROOT / ".venv/bin/python"), "-u", str(ROOT / "scripts/tempo_replay.py"), "--samples", "results/films/spoon_teleop_demo_113219_ep3_samples.json", "--speed", os.environ.get("REPLAY_SPEED", "0.8")] + sys.argv[1:])
