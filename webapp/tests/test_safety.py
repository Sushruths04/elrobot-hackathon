import os
import subprocess

import agent_bridge
from safety import HumanGate
from test_agent_bridge import FakeClient, messages_seen_text, text_block, tool_use

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_station_gate_blocks_motion_until_confirmed(sim_arm):
    gate = HumanGate("station")  # unconfirmed: motion tools must not run
    client = FakeClient([
        [tool_use("home", {}, "tu_home")],
        [text_block("blocked")],
    ])
    events = []
    agent_bridge.run_chat(sim_arm, "go home", events.append, gate, client=client)

    assert "human-presence confirmation required" in messages_seen_text(client)
    results = [e for e in events if e["type"] == "tool_result"]
    assert results and "human-presence confirmation required" in results[0]["result"]
    norms = sim_arm.state().norms()
    assert all(abs(v - 0.5) < 0.08 for v in norms.values()), f"arm moved while gated: {norms}"

    gate.confirm()
    client2 = FakeClient([
        [tool_use("home", {}, "tu_home2")],
        [text_block("done")],
    ])
    events2 = []
    agent_bridge.run_chat(sim_arm, "go home", events2.append, gate, client=client2)
    assert any(e["type"] == "tool_result" and e.get("name") == "home" for e in events2)


def test_run_py_refuses_station_without_flag():
    venv_py = os.path.join(REPO, ".venv", "bin", "python")
    r = subprocess.run([venv_py, os.path.join(REPO, "webapp", "run.py"),
                        "--backend", "station", "--port", "8999"],
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 2
    assert "--i-have-a-human" in r.stderr


def test_estop_during_chat_blocks_via_guard(sim_arm):
    # park joint 1 away from home so home() really has to drive through Guard
    r = sim_arm.guard.move({1: 0.8})
    assert r["ok"] is True
    sim_arm.guard.settle(timeout=5.0)
    gate = HumanGate("sim")
    client = FakeClient([
        [tool_use("home", {}, "tu_h1")],
        [tool_use("home", {}, "tu_h2")],
        [text_block("stopped")],
    ])
    events = []

    def on_event(ev):
        events.append(ev)
        if ev["type"] == "tool_use":
            sim_arm.guard.estop()  # e-stop lands before the tool executes

    try:
        agent_bridge.run_chat(sim_arm, "go home twice", on_event, gate, client=client)
    finally:
        sim_arm.guard.release()

    results = [e for e in events if e["type"] == "tool_result"]
    assert any("e-stop active" in e["result"] for e in results), \
        f"Guard's e-stop refusal never surfaced: {results}"
