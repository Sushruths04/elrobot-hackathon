"""Claude tool-use loop for the web console.

Reuses SYSTEM / TOOLS / run_tool / MODEL from src/normakit/agent.py VERBATIM (read-only import -
that file is never edited and never forked). Claude API is the only model used; there is no
fine-tuning path. Emits one JSON-able event per step through `on_event` so the WebSocket layer
can stream the agent's work to the browser. Writes nothing to the repo (no run/trials.jsonl -
that file belongs to the other loop).
"""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from normakit.agent import MODEL, SYSTEM, TOOLS, run_tool  # noqa: E402  (read-only import)

from safety import MOTION_TOOLS  # noqa: E402

NO_KEY_MSG = "ANTHROPIC_API_KEY is not set - export it and retry"
GATE_MSG = "motion blocked: human-presence confirmation required (station mode)"


def run_chat(arm, text: str, on_event, gate, client=None, model: str | None = None,
             max_turns: int = 25, cancel=None) -> dict:
    """Drive the existing Claude tool-use loop for one chat message.

    on_event receives dicts: tool_use / image / tool_result / final / error / cancelled.
    Motion tools go through the HumanGate first and always through normakit.agent.run_tool
    (which uses Arm skills -> Guard) - never through Backend.send_goals directly.
    """
    t0 = time.time()
    if client is None:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            on_event({"type": "error", "text": NO_KEY_MSG})
            return {"llm_calls": 0, "tool_calls": 0,
                    "seconds": round(time.time() - t0, 1), "final": ""}
        import anthropic
        client = anthropic.Anthropic()
    model = model or MODEL
    msgs = [{"role": "user", "content": text}]
    calls, tools_used, final = 0, 0, ""
    for _ in range(max_turns):
        if cancel is not None and cancel.is_set():
            on_event({"type": "cancelled"})
            break
        resp = client.messages.create(model=model, max_tokens=1024,
                                      system=SYSTEM, tools=TOOLS, messages=msgs)
        calls += 1
        msgs.append({"role": "assistant", "content": resp.content})
        uses = [b for b in resp.content if b.type == "tool_use"]
        if not uses:
            final = "".join(b.text for b in resp.content if b.type == "text")
            break
        results = []
        for u in uses:
            tools_used += 1
            on_event({"type": "tool_use", "name": u.name, "input": u.input})
            if u.name in MOTION_TOOLS and not gate.ok():
                blocked = json.dumps({"ok": False, "error": GATE_MSG})
                on_event({"type": "tool_result", "name": u.name, "result": blocked})
                results.append({"type": "tool_result", "tool_use_id": u.id, "content": blocked})
                continue
            with gate.lock:  # one writer on the bus
                out = run_tool(arm, u.name, u.input)
            if isinstance(out, list):  # look -> image content blocks
                for blk in out:
                    if blk.get("type") == "image":
                        on_event({"type": "image", "jpeg_b64": blk["source"]["data"]})
                results.append({"type": "tool_result", "tool_use_id": u.id, "content": out})
            else:
                on_event({"type": "tool_result", "name": u.name, "result": out})
                results.append({"type": "tool_result", "tool_use_id": u.id, "content": out})
        msgs.append({"role": "user", "content": results})
    if final:
        on_event({"type": "final", "text": final})
    return {"llm_calls": calls, "tool_calls": tools_used,
            "seconds": round(time.time() - t0, 1), "final": final}
