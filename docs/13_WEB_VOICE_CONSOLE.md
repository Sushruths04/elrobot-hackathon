# 13 - Web voice console

A web/voice layer over `normakit`: "Claude Code in a web" for the ElRobot follower arm.
Everything new lives in `webapp/` (+ this page). No existing file was changed.

## Architecture

```
browser (vanilla JS, no build/CDN - works offline)
   |  WS /ws  : state 0.5 s, camera frame 0.7 s, agent events (tool_use/tool_result/image/final/error/cancelled)
   |  client -> server: {type:chat,text} / {type:estop} / {type:cancel}
   |  REST    : GET /api/status, POST /api/estop|release|confirm-human
   v
FastAPI (webapp/app.py, :8920, binds 127.0.0.1)
   |  webapp/agent_bridge.py  -> Claude tool-use loop, one event per step
   |  webapp/safety.py        -> HumanGate (10-min window) + single-writer lock
   v
normakit Arm/Guard (src/normakit, read-only import - nothing edited)
   v
sim (pybullet, default)  |  station (real arm via Station :8888)
```

`agent_bridge.run_chat` reuses `SYSTEM`, `TOOLS`, `run_tool`, `MODEL` from
`src/normakit/agent.py` verbatim by import (that file is not edited or forked). The only
difference from the CLI agent loop is that each step is also emitted as a JSON event for the
WebSocket, motion tools pass the `HumanGate` first, and nothing is written to `run/trials.jsonl`.
Claude API is the only model used; the webapp introduces no fine-tuning path.

## Safety chain

- `Guard` (`src/normakit/core.py`) stays the hard layer: clamps, step limit, overcurrent e-stop.
  Untouched.
- `--backend station` refuses to start without `--i-have-a-human` (mirrors `scripts/run_policy.py`).
- Station mode: a HUMAN AT THE TABLE? modal blocks chat until `POST /api/confirm-human`;
  motion tools are forwarded only inside the 10-minute window (mirrors the :8890 protocol).
- E-STOP button sends both WS `estop` and `POST /api/estop`; it latches `Guard` and cancels the
  running agent. The real e-stop is the servo power plug; software stop is a convenience.
- One writer on the bus: the Station web UI control panel (:8889) must be closed while driving;
  a single `threading.Lock` serialises all motion calls in the webapp.

## Voice (v1, zero GPU)

Browser Web Speech API (`SpeechRecognition`) for STT - interim words fill the input, the final
transcript auto-sends - and `speechSynthesis` for TTS (final replies + short tool summaries).
No audio leaves the browser; no speech model is added to the repo. Without the APIs (non
Chrome/Edge) a hint shows and typing works fully. Upgrade path later on the A100: host Whisper
(STT) + Piper/Coqui (TTS) behind env-var URLs (`WEBAPP_STT_URL`/`WEBAPP_TTS_URL`) - documented
placeholders only, **not implemented** in v1.

## What is NOT changed

Every existing folder and file is untouched: `src/`, `scripts/`, `server/`, `run/`, `docs/01..12`,
`AGENTS.md`, `README.md`, `scripts/selftest_sim.py` (baseline, still green). The only edit to an
existing file is one appended bullet in `CHANGELOG.md` (rule 8). The webapp writes nothing to
the repo at runtime.

## Tests

`.venv/bin/python -m pytest webapp/tests -q` - 10 tests, no hardware, no API key (anthropic
client mocked): status/estop/release/confirm endpoints, WS state+frame push, tool-loop events,
station-gate motion blocking, estop-during-chat (Guard refuses), station refusal exit code.

## Honest limits

- Voice = browser APIs (Chrome/Edge); accuracy and latency are whatever the browser gives.
- Sim physics is logic-level (pybullet twin), not a dynamics-exact model.
- Station mode is untested on live hardware by this layer (sim only so far).
- `NORMA_MODEL` defaults to `claude-sonnet-5-5`; tool accuracy depends on the model's reading of
  the grid image - verify grasps visually like any other loop in this repo.
