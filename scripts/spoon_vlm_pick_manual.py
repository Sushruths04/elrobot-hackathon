"""Wrapper for film_task: spoon_vlm_pick.py with the grasp/cup pixels read by eye from the live front image (args in run/spoon_manual_points.txt)."""
import os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; args = (ROOT / "run/spoon_manual_points.txt").read_text().split()
os.execv(str(ROOT / ".venv/bin/python"), [str(ROOT / ".venv/bin/python"), "-u", str(ROOT / "scripts/spoon_vlm_pick.py"), "--tries", "1"] + args)
