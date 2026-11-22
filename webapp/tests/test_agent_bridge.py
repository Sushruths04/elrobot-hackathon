import base64
import io
import json
from types import SimpleNamespace

from PIL import Image

import agent_bridge
from safety import HumanGate


class FakeMessages:
    """Stand-in for anthropic.Anthropic().messages: replays a canned script of turns."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = 0
        self.seen = []

    def create(self, model, max_tokens, system, tools, messages):
        self.calls += 1
        self.seen.append(messages)
        return SimpleNamespace(content=self.script.pop(0))


class FakeClient:
    def __init__(self, script):
        self.messages = FakeMessages(script)


def tool_use(name, args, tid):
    return SimpleNamespace(type="tool_use", name=name, input=args, id=tid)


def text_block(s):
    return SimpleNamespace(type="text", text=s)


def test_run_chat_sim_look_home_done(sim_arm):
    script = [
        [tool_use("look", {"grid": True}, "tu_look")],
        [tool_use("home", {}, "tu_home")],
        [text_block("done")],
    ]
    client = FakeClient(script)
    events = []
    rec = agent_bridge.run_chat(sim_arm, "put the red block in the blue tray",
                                events.append, HumanGate("sim"), client=client)

    tool_events = [e for e in events if e["type"] == "tool_use"]
    assert [e["name"] for e in tool_events] == ["look", "home"]

    images = [e for e in events if e["type"] == "image"]
    assert images, "look should produce an image event"
    Image.open(io.BytesIO(base64.b64decode(images[0]["jpeg_b64"]))).verify()  # real jpeg

    finals = [e for e in events if e["type"] == "final"]
    assert finals and finals[-1]["text"] == "done"

    assert rec["tool_calls"] == 2
    assert rec["final"] == "done"
    assert client.messages.calls == 3


def test_missing_api_key_returns_error_event(sim_arm, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    events = []
    rec = agent_bridge.run_chat(sim_arm, "hello", events.append, HumanGate("sim"), client=None)
    errors = [e for e in events if e["type"] == "error"]
    assert errors and "ANTHROPIC_API_KEY" in errors[0]["text"]
    assert rec["llm_calls"] == 0 and rec["tool_calls"] == 0


def messages_seen_text(fake_client):
    return json.dumps(fake_client.messages.seen, default=str)
