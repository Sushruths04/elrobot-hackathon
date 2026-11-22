"""film_task wrapper: spoon_grasp_v3.py with the grasp point read from the live front image (run/spoon_v3_args.txt)."""
import os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
os.execv(str(ROOT / ".venv/bin/python"), [str(ROOT / ".venv/bin/python"), "-u", str(ROOT / "scripts/spoon_grasp_v3.py")] + (ROOT / "run/spoon_v3_args.txt").read_text().split())
