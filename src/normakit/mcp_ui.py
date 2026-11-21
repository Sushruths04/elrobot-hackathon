"""MCP server that drives the REAL arm through the local web interface (scripts/ui.py, http://127.0.0.1:8890) - same safe path the human sees:
STOP, safety-gated runs, PROCEED gates, recording and the live log all stay in one place. Start the UI first:  .venv/bin/python scripts/ui.py
Run:  .venv/bin/python -m normakit.mcp_ui"""
import json, urllib.request
from mcp.server.fastmcp import FastMCP, Image
BASE = "http://127.0.0.1:8890"; mcp = FastMCP("elrobot-ui")
def _get(path, raw=False, timeout=30):
    with urllib.request.urlopen(BASE + path, timeout=timeout) as r: d = r.read(); return d if raw else d.decode()
def _post(path, body=None, timeout=90):
    req = urllib.request.Request(BASE + path, data=json.dumps(body or {}).encode(), method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r: return r.read().decode()

@mcp.tool()
def status() -> str:
    """Follower joints (raw ticks, normalised, mA), gripper xyz, whether a run/recording is active, whether the ER-2 key is set."""
    return _get("/api/state")
@mcp.tool()
def look() -> Image:
    """Current picture of the table camera."""
    return Image(data=_get("/frame.jpg", raw=True), format="jpeg")
@mcp.tool()
def detect_cup() -> str:
    """Gemini ER-2 pointing at the black cup; returns pixel and table x,y. Newest annotated picture is at /detect.jpg."""
    return _post("/api/detect")
@mcp.tool()
def jog(joint: int, ticks: int) -> str:
    """Move ONE joint (1-7, 8 = gripper) by up to +-150 ticks (4096 = full turn), slow; reports how far it really moved and the current."""
    return _post("/api/jog", {"joint": joint, "delta": ticks})
@mcp.tool()
def run_task(task: str = "marker", go: bool = False, speed: int = 250, record: bool = True) -> str:
    """Start scripts/grab_marker.py ('marker': pen -> white box) or grab_cup.py ('cup'). go=False is a plan-only dry run. go=True MOVES THE ARM: only after the human said go and confirmed hands clear + power plug in reach."""
    return _post("/api/run", {"task": task, "go": go, "speed": speed, "rec": record})
@mcp.tool()
def proceed() -> str:
    """Release a 'WAITING for you' gate in the running task. Only call after the human has looked at the picture and said proceed."""
    return _post("/api/proceed")
@mcp.tool()
def stop() -> str:
    """Kill the running task and hold the arm at its present pose."""
    return _post("/api/stop")
@mcp.tool()
def log(lines: int = 40) -> str:
    """Tail of the live run log."""
    return "\n".join(_get("/api/log").splitlines()[-lines:])
@mcp.tool()
def record(on: bool, label: str = "scenario") -> str:
    """Start/stop recording the table camera + joint log into run/recordings/."""
    return _post("/api/record", {"on": on, "label": label})


def _armd(path, body=None, timeout=180):
    req = urllib.request.Request("http://127.0.0.1:8891" + path, data=None if body is None else json.dumps(body).encode(), method="GET" if body is None else "POST")
    with urllib.request.urlopen(req, timeout=timeout) as r: return r.read().decode()

@mcp.tool()
def arm_state() -> str:
    """FAST (25 ms): follower joints in ticks, current mA, jaw-tip xyz (m) and how many seconds motion stays armed."""
    return _armd("/state")
@mcp.tool()
def move_tip(x: float, y: float, z: float, roll: float = 0.0, speed: int = 600) -> str:
    """Move the jaw tip to robot xyz (m; y forward, z up, table z=0) with IK, servo-side trajectory, waits until arrived. Needs the human's ARM."""
    return _armd("/move_tip", {"x": x, "y": y, "z": z, "roll": roll, "speed": speed})
@mcp.tool()
def gripper_to(to: str = "open") -> str:
    """'open', 'close' or a tick value. Needs ARM."""
    return _armd("/gripper", {"to": to})
@mcp.tool()
def go_home(speed: int = 500) -> str:
    """Wrist first, then shoulder+elbow, to run/home_follower.json. Needs ARM."""
    return _armd("/home", {"speed": speed})
@mcp.tool()
def hold_now() -> str:
    """Hold the present pose (goal = present). Always allowed."""
    return _armd("/stop", {})
@mcp.tool()
def list_demos() -> str:
    """Recorded teleop demos with success labels (run/demos/, episodes.json)."""
    return _post("/api/demo/list")
@mcp.tool()
def record_demo(on: bool, name: str = "demo1") -> str:
    """Start/stop recording a teleop demonstration: follower 40 Hz + leader 50 Hz + front camera. The human moves the leader."""
    return _post("/api/demo/record", {"on": on, "name": name})
@mcp.tool()
def mark_demo(name: str, success: bool, note: str = "") -> str:
    """Label a demo as success/failure (for training). Only after the human confirmed."""
    return _post("/api/demo/mark", {"name": name, "success": success, "note": note})
@mcp.tool()
def play_demo(name: str, scale: float = 0.5) -> str:
    """Replay a recorded demo on the follower (scale 0.5 = half speed). Needs ARM. Then call verify() to check the result."""
    return _post("/api/demo/play", {"name": name, "scale": scale})
@mcp.tool()
def verify(question: str) -> str:
    """Ask Gemini Robotics-ER 2 a question about the table camera, e.g. 'Is the marker inside the white box? Reply {"inside": true or false}'. 1.7 s."""
    return _post("/api/ask", {"question": question})
@mcp.tool()
def run_skill_and_verify(name: str = "teleop_pick2", scale: float = 0.5) -> str:
    """Replay demo `name`, then ask Gemini whether the marker is in the white box. Returns both results. Needs ARM."""
    r = _post("/api/demo/play", {"name": name, "scale": scale}); v = _post("/api/ask", {"question": 'Is the marker (pen) inside the white box? Reply {"inside": true or false, "reason": short text}'})
    return json.dumps({"play": json.loads(r), "gemini_check": json.loads(v)})

if __name__ == "__main__":
    mcp.run()
