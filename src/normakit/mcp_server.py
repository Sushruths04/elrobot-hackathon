"""MCP server exposing the arm to Claude Code / Claude Desktop / Codex (any MCP client).

Run (sim):   python -m normakit.mcp_server --backend sim
Run (real):  python -m normakit.mcp_server --backend station --host localhost:8888

Tool design rules (what makes agents succeed):
  * `look` returns an IMAGE (the model literally sees the scene); `grid=true` adds A1..H6 labels.
  * Reaching is by *where* (grid cell / pixel / xyz); IK + safety live in code, not in the model.
  * Every action returns a small JSON verdict from sensors (reached? grasped? overcurrent?).
  * `stop` is always available and latches an e-stop until `release`.
"""
from __future__ import annotations

import argparse
import json
import sys

from mcp.server.fastmcp import FastMCP, Image

from .arm import Arm

mcp = FastMCP("elrobot")
_arm: Arm | None = None


def arm() -> Arm:
    assert _arm is not None
    return _arm


@mcp.tool()
def look(camera: str = "table", grid: bool = True) -> Image:
    """Capture a camera frame. camera: 'table' (overview) or 'wrist' in sim; 'cam0','cam1' on the real rig.
    With grid=true the image is overlaid with labelled cells (A1..H6) you can pass to reach_cell/pick_cell."""
    return Image(data=arm().look(camera, grid), format="jpeg")


@mcp.tool()
def arm_state() -> str:
    """Joint positions (normalised 0..1 over calibrated range), goals, motor current (mA), temperature."""
    return json.dumps(arm().state().to_json())


@mcp.tool()
def move_joints(goals: dict[str, float], speed: float = 0.5) -> str:
    """Low-level: set normalised joint targets, e.g. {"1":0.6,"4":0.4}. Joint 8 = gripper (0 closed..1 open).
    Prefer reach_cell/pick_cell. Large moves are step-limited; check notes in the reply."""
    return json.dumps(arm().move_joints({int(k): v for k, v in goals.items()}, speed))


@mcp.tool()
def reach_cell(cell: str) -> str:
    """Move the gripper above a grid cell from the most recent look(grid=true), lowering to table height."""
    return json.dumps(arm().reach_cell(cell))


@mcp.tool()
def pick_cell(cell: str) -> str:
    """Open gripper, go to the cell, close, lift. Returns grasped=true/false from motor current/jaw stall (not vision)."""
    a = arm()
    if cell not in a.grid_cells:
        return json.dumps({"ok": False, "error": "call look(grid=true) first"})
    return json.dumps(a.pick(*a.grid_cells[cell]))


@mcp.tool()
def place_cell(cell: str) -> str:
    """Move the held object above a cell, lower slightly and release."""
    a = arm()
    if cell not in a.grid_cells:
        return json.dumps({"ok": False, "error": "call look(grid=true) first"})
    return json.dumps(a.place(*a.grid_cells[cell]))


@mcp.tool()
def reach_xyz(x: float, y: float, z: float) -> str:
    """Move the gripper to a point in the robot base frame (metres). Fails with IK error if unreachable."""
    return json.dumps(arm().reach_xyz((x, y, z)))


@mcp.tool()
def gripper(action: str) -> str:
    """'open' or 'close'. close returns grasped=true/false."""
    return json.dumps(arm().open_gripper() if action == "open" else arm().close_gripper())


@mcp.tool()
def home() -> str:
    """Return to the neutral pose (also use before every fresh look so the arm is out of view)."""
    return json.dumps(arm().home())


@mcp.tool()
def calibrate_table() -> str:
    """One-minute self-calibration: the arm touches known points, finds its magenta marker in the table
    camera, fits pixel->table coordinates. Run once after the camera is fixed in place."""
    return json.dumps(arm().calibrate_table())


@mcp.tool()
def stop() -> str:
    """EMERGENCY STOP: hold position and refuse motion until release()."""
    arm().guard.estop()
    return "estopped"


@mcp.tool()
def release() -> str:
    """Clear the e-stop latch."""
    arm().guard.release()
    return "released"


def build(backend: str, host: str) -> Arm:
    if backend == "sim":
        from .sim_backend import SimBackend
        b = SimBackend()
    else:
        from .station_backend import StationBackend
        b = StationBackend(host)
    b.start()
    return Arm(b)


def main():
    global _arm
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["sim", "station"], default="sim")
    ap.add_argument("--host", default="localhost:8888")
    a = ap.parse_args()
    _arm = build(a.backend, a.host)
    print(f"elrobot MCP ready (backend={a.backend})", file=sys.stderr)
    mcp.run()


if __name__ == "__main__":
    main()
