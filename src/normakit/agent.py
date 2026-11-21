"""Standalone Claude agent loop (no MCP client needed) - useful for the live demo UI and for benchmarking.

    ANTHROPIC_API_KEY=... python -m normakit.agent --backend sim "put the red block in the blue tray"

Loop: the model gets tools (look/pick/place/...), sees images as tool results, and must verify with `look`
after acting. Every step is logged to run/trials.jsonl (success/time) so you can quote real numbers in the pitch.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import time
from pathlib import Path

import anthropic

from .arm import Arm
from .mcp_server import build

MODEL = os.environ.get("NORMA_MODEL", "claude-sonnet-5-5")

SYSTEM = """You control a 7+1 DOF desktop robot arm through tools. You can SEE through `look`.
Rules:
1. Always call look(grid=true) first; the image has labelled cells (A1..H6). Name targets by cell.
2. Prefer pick_cell/place_cell/reach_cell. They use calibrated IK + safety. Do NOT guess joint numbers.
3. After pick_cell read `grasped`; if false, look again, adjust the cell, retry (max 2 retries) then report failure.
4. After the final placement call home() then look() and state plainly whether the task succeeded, based on the image.
5. Never touch things that are not part of the task. If anything looks unsafe call stop().
Be brief; act, don't narrate."""

TOOLS = [
    {"name": "look", "description": "Capture camera image (with grid labels if grid=true).",
     "input_schema": {"type": "object", "properties": {"camera": {"type": "string"}, "grid": {"type": "boolean"}}}},
    {"name": "pick_cell", "description": "Pick the object in a grid cell. Returns grasped true/false.",
     "input_schema": {"type": "object", "properties": {"cell": {"type": "string"}}, "required": ["cell"]}},
    {"name": "place_cell", "description": "Release the held object above a grid cell.",
     "input_schema": {"type": "object", "properties": {"cell": {"type": "string"}}, "required": ["cell"]}},
    {"name": "reach_cell", "description": "Move gripper over a grid cell.",
     "input_schema": {"type": "object", "properties": {"cell": {"type": "string"}}, "required": ["cell"]}},
    {"name": "gripper", "description": "open or close the gripper.",
     "input_schema": {"type": "object", "properties": {"action": {"type": "string", "enum": ["open", "close"]}}, "required": ["action"]}},
    {"name": "home", "description": "Neutral pose, arm out of view.", "input_schema": {"type": "object", "properties": {}}},
    {"name": "arm_state", "description": "Joint/current/temperature readout.", "input_schema": {"type": "object", "properties": {}}},
    {"name": "stop", "description": "Emergency stop.", "input_schema": {"type": "object", "properties": {}}},
]


def run_tool(arm: Arm, name: str, a: dict):
    if name == "look":
        jpg = arm.look(a.get("camera") or "table", a.get("grid", True))
        return [{"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                              "data": base64.b64encode(jpg).decode()}}]
    if name == "pick_cell":
        r = arm.pick(*arm.grid_cells[a["cell"]]) if a["cell"] in arm.grid_cells else {"ok": False, "error": "look(grid=true) first"}
    elif name == "place_cell":
        r = arm.place(*arm.grid_cells[a["cell"]]) if a["cell"] in arm.grid_cells else {"ok": False, "error": "look(grid=true) first"}
    elif name == "reach_cell":
        r = arm.reach_cell(a["cell"])
    elif name == "gripper":
        r = arm.open_gripper() if a["action"] == "open" else arm.close_gripper()
    elif name == "home":
        r = arm.home()
    elif name == "arm_state":
        r = arm.state().to_json()
    elif name == "stop":
        arm.guard.estop(); r = "estopped"
    else:
        r = {"ok": False, "error": f"unknown tool {name}"}
    return json.dumps(r)


def run(arm: Arm, task: str, max_turns: int = 25) -> dict:
    client = anthropic.Anthropic()
    msgs = [{"role": "user", "content": task}]
    t0, calls, tools_used = time.time(), 0, 0
    final = ""
    for _ in range(max_turns):
        resp = client.messages.create(model=MODEL, max_tokens=1024, system=SYSTEM, tools=TOOLS, messages=msgs)
        calls += 1
        msgs.append({"role": "assistant", "content": resp.content})
        uses = [b for b in resp.content if b.type == "tool_use"]
        if not uses:
            final = "".join(b.text for b in resp.content if b.type == "text")
            break
        results = []
        for u in uses:
            tools_used += 1
            print(f"  -> {u.name}({json.dumps(u.input)})", flush=True)
            results.append({"type": "tool_result", "tool_use_id": u.id, "content": run_tool(arm, u.name, u.input)})
        msgs.append({"role": "user", "content": results})
    rec = {"task": task, "model": MODEL, "llm_calls": calls, "tool_calls": tools_used,
           "seconds": round(time.time() - t0, 1), "final": final}
    Path("run").mkdir(exist_ok=True)
    with open("run/trials.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task")
    ap.add_argument("--backend", default="sim", choices=["sim", "station"])
    ap.add_argument("--host", default="localhost:8888")
    a = ap.parse_args()
    arm = build(a.backend, a.host)
    print(run(arm, a.task))


if __name__ == "__main__":
    main()
