"""FastAPI factory for the normakit web console.

Routes:
  GET  /                  UI (webapp/static/index.html; placeholder until the frontend lands)
  GET  /api/status        backend / model / e-stop / human-gate / key presence / cameras
  POST /api/estop         latch e-stop (arm.guard.estop) and cancel any running chat
  POST /api/release       clear the e-stop latch
  POST /api/confirm-human open the 10-minute human-presence window (station mode)
  WS   /ws                telemetry + camera push; client sends chat / estop / cancel

Everything that touches the arm runs in a thread (Arm calls are blocking) and every motion
goes through Arm/Guard only - never Backend.send_goals directly. The app writes nothing to
the repo at runtime.
"""
from __future__ import annotations

import asyncio
import base64
import os
import sys
import threading

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect  # noqa: E402
from fastapi.responses import FileResponse, HTMLResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

from agent_bridge import run_chat  # noqa: E402
from normakit.agent import MODEL  # noqa: E402  (read-only import)
from safety import HumanGate  # noqa: E402

STATIC_INDEX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", "index.html")
PLACEHOLDER = ("<html><body><h1>normakit web console</h1><p>API up - the frontend lands in "
               "FEAT-003. See GET /api/status and WS /ws.</p></body></html>")


def create_app(arm, backend: str, gate: HumanGate | None = None, model: str | None = None) -> FastAPI:
    gate = gate or HumanGate(backend)
    model = model or MODEL
    app = FastAPI(title="normakit web console")
    static_dir = os.path.dirname(STATIC_INDEX)
    if os.path.isdir(static_dir):
        app.mount("/static", StaticFiles(directory=static_dir), name="static")
    agent_lock = threading.Lock()
    state = {"agent_active": False, "cancel": None}

    def _estop():
        arm.guard.estop()
        c = state.get("cancel")
        if c is not None:
            c.set()  # stop the agent issuing tools at the next turn boundary

    # ---------------- HTTP ----------------
    @app.get("/")
    async def index():
        if os.path.exists(STATIC_INDEX):
            return FileResponse(STATIC_INDEX)
        return HTMLResponse(PLACEHOLDER)

    @app.get("/api/status")
    async def status():
        cameras = await asyncio.to_thread(arm.backend.cameras)
        return {"backend": backend,
                "model": model,
                "estopped": arm.guard.estopped,
                "human_confirmed": gate.ok(),
                "human_window_s": round(gate.remaining_s(), 1),
                "anthropic_key": bool(os.environ.get("ANTHROPIC_API_KEY")),
                "cameras": cameras}

    @app.post("/api/estop")
    async def estop():
        await asyncio.to_thread(_estop)  # estop calls halt - blocking
        return {"ok": True, "estopped": True}

    @app.post("/api/release")
    async def release():
        await asyncio.to_thread(arm.guard.release)
        return {"ok": True, "estopped": False}

    @app.post("/api/confirm-human")
    async def confirm_human():
        window_s = gate.confirm()  # harmless no-op in sim (gate.ok() is always True there)
        return {"ok": True, "window_s": window_s}

    # ---------------- WebSocket ----------------
    @app.websocket("/ws")
    async def ws(websocket: WebSocket):
        await websocket.accept()
        loop = asyncio.get_running_loop()
        stop_push = asyncio.Event()
        send_lock = asyncio.Lock()

        async def send_safe(payload: dict):
            try:
                async with send_lock:
                    await websocket.send_json(payload)
            except Exception:
                stop_push.set()

        async def state_loop():
            while not stop_push.is_set():
                try:
                    st = await asyncio.to_thread(lambda: arm.state().to_json())
                    await send_safe({"type": "state", "joints": st,
                                     "estopped": arm.guard.estopped,
                                     "agent_active": state["agent_active"],
                                     "backend": backend})
                except Exception as e:
                    await send_safe({"type": "telemetry_error", "text": f"state: {e}"})
                await asyncio.sleep(0.5)

        async def frame_loop():
            cam = None
            while not stop_push.is_set():
                try:
                    if cam is None:
                        if backend == "station":
                            from normakit.station_backend import front_camera
                            cam = await asyncio.to_thread(front_camera, arm.backend)
                        else:
                            cam = "table"
                    jpg = await asyncio.to_thread(arm.backend.frame_jpeg, cam)
                    await send_safe({"type": "frame", "camera": cam,
                                     "jpeg_b64": base64.b64encode(jpg).decode()})
                except Exception as e:
                    if backend == "station":
                        cam = None  # re-detect next tick (station renumbers cameras)
                    await send_safe({"type": "telemetry_error", "text": f"camera: {e}"})
                await asyncio.sleep(0.7)

        tasks = [asyncio.create_task(state_loop()), asyncio.create_task(frame_loop())]
        try:
            while True:
                try:
                    msg = await websocket.receive_json()
                except WebSocketDisconnect:
                    raise
                except Exception:
                    await send_safe({"type": "error", "text": "bad message"})
                    continue
                if not isinstance(msg, dict):
                    continue
                mtype = msg.get("type")
                if mtype == "chat":
                    text = str(msg.get("text", "")).strip()
                    if not text:
                        await send_safe({"type": "error", "text": "empty chat text"})
                        continue
                    with agent_lock:
                        busy = state["agent_active"]
                        if not busy:
                            state["agent_active"] = True
                            state["cancel"] = threading.Event()
                    if busy:
                        await send_safe({"type": "error", "text": "agent already running"})
                        continue
                    cancel = state["cancel"]

                    def on_event(ev):
                        try:
                            asyncio.run_coroutine_threadsafe(send_safe(ev), loop)
                        except Exception:
                            pass

                    def work(_text=text, _cancel=cancel, _emit=on_event):
                        try:
                            run_chat(arm, _text, _emit, gate, client=None,
                                     model=model, cancel=_cancel)
                        except Exception as e:
                            _emit({"type": "error", "text": f"agent error: {e}"})
                        finally:
                            _emit({"type": "agent_done"})
                            with agent_lock:
                                state["agent_active"] = False
                                state["cancel"] = None

                    threading.Thread(target=work, daemon=True).start()
                elif mtype == "estop":
                    await asyncio.to_thread(_estop)
                    await send_safe({"type": "estop_ack", "ok": True, "estopped": True})
                elif mtype == "cancel":
                    c = state.get("cancel")
                    if c is not None:
                        c.set()
                    await send_safe({"type": "cancel_ack", "ok": True})
        except WebSocketDisconnect:
            pass
        except Exception:
            pass
        finally:
            stop_push.set()
            for t in tasks:
                t.cancel()

    return app
