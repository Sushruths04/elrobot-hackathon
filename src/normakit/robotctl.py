"""CLI with the same verbs as the MCP server, for agents that prefer a shell (Codex CLI, Claude Code Bash).

Each call is a separate process, so a tiny daemon-less design is used: it reconnects to Station every time
(fast: TCP). Images are written to run/last.jpg and the path is printed; open it with your image-capable tool.

    python -m normakit.robotctl --backend station look --grid
    python -m normakit.robotctl pick C4
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .mcp_server import build

GRID_STATE = Path("run/grid_cells.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="station", choices=["sim", "station"])
    ap.add_argument("--host", default="localhost:8888")
    sub = ap.add_subparsers(dest="cmd", required=True)
    l = sub.add_parser("look"); l.add_argument("--camera", default="table"); l.add_argument("--grid", action="store_true")
    for c in ("pick", "place", "reach"):
        s = sub.add_parser(c); s.add_argument("cell")
    g = sub.add_parser("gripper"); g.add_argument("action", choices=["open", "close"])
    for c in ("state", "home", "stop", "release", "calibrate"):
        sub.add_parser(c)
    a = ap.parse_args()

    arm = build(a.backend, a.host)
    time.sleep(0.5)
    if GRID_STATE.exists():
        arm.grid_cells = {k: tuple(v) for k, v in json.loads(GRID_STATE.read_text()).items()}
    out = None
    if a.cmd == "look":
        Path("run").mkdir(exist_ok=True)
        Path("run/last.jpg").write_bytes(arm.look(a.camera, a.grid))
        if a.grid:
            GRID_STATE.write_text(json.dumps(arm.grid_cells))
        out = {"image": "run/last.jpg"}
    elif a.cmd in ("pick", "place", "reach"):
        fn = {"pick": arm.pick, "place": arm.place, "reach": lambda u, v: arm.reach_pixel(u, v)}[a.cmd]
        out = fn(*arm.grid_cells[a.cell]) if a.cell in arm.grid_cells else {"ok": False, "error": "run `look --grid` first"}
    elif a.cmd == "gripper":
        out = arm.open_gripper() if a.action == "open" else arm.close_gripper()
    elif a.cmd == "state":
        out = arm.state().to_json()
    elif a.cmd == "home":
        out = arm.home()
    elif a.cmd == "stop":
        arm.guard.estop(); out = "estopped"
    elif a.cmd == "release":
        arm.guard.release(); out = "released"
    elif a.cmd == "calibrate":
        out = arm.calibrate_table()
    print(json.dumps(out))


if __name__ == "__main__":
    main()
