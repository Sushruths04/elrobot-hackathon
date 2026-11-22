# normakit web console + voice agent

A self-contained web/voice layer over `normakit`: **Claude Code in a web** for the ElRobot
follower arm. One page (`webapp/static/`, vanilla HTML/CSS/JS - no build step, no npm, no
CDN, works offline) talks to a FastAPI backend (`webapp/`) over WebSocket + REST. The Claude
tool-use loop is reused verbatim from `src/normakit/agent.py` (read-only import); every motion
goes through `Arm`/`Guard` exactly like the MCP tools do. Sim backend by default.

## Run

Sim (zero hardware):

```
.venv/bin/python webapp/run.py --backend sim --port 8920
# open http://127.0.0.1:8920
```

Real arm (Station backend):

```
./norma start            # Station up first
.venv/bin/python webapp/run.py --backend station --i-have-a-human
```

Safety rules for station mode (same as the rest of this repo):

- `--i-have-a-human` is mandatory; the server refuses to start without it.
- A person must be **at the table**: the UI shows a HUMAN AT THE TABLE? modal and motion tools
  stay blocked until it is confirmed. The window lasts 10 minutes, then the modal returns.
- **One writer on the bus**: close the Station web UI control panel (:8889) while the webapp drives.
- The real e-stop is the **servo power plug**; the on-screen E-STOP is a software convenience
  (it latches `Guard`, which refuses all motion until RELEASE).
- Never send motion to an uncalibrated arm (`./norma status` -> `frozen=True`, span >= 100).

## Claude API key / model

`ANTHROPIC_API_KEY` comes from the shell environment only - never written to files or logs.
The server boots and serves the UI without it; chat then replies with a clear
"ANTHROPIC_API_KEY is not set" error event. Model override: `NORMA_MODEL`
(default `claude-sonnet-5-5`, same as `src/normakit/agent.py`).

**Claude API is the only model used by this layer. No fine-tuning path is introduced
anywhere in the webapp.**

## Voice (v1, zero GPU)

- Mic toggle: browser **Web Speech API** (`SpeechRecognition`) turns speech into chat text.
  Interim words appear live in the input; the final transcript auto-sends.
- Speaker toggle: `speechSynthesis` reads out the assistant's final replies and short tool
  summaries ("picking cell C4").
- Everything happens **in the browser**: no audio is sent anywhere, no speech model is
  introduced in the repo.
- Needs Chrome/Edge for recognition. Without it a greyed-out hint shows and the app works
  fully by typing.

## Hosting later (A100)

The app speaks plain HTTP/WebSocket and binds `127.0.0.1` by default. To serve it from the
A100 later, run the same command with `--host 0.0.0.0` and reach it through an SSH tunnel
(`ssh -L 8920:localhost:8920 <user>@A100_HOST`), a VPN (WireGuard/Tailscale), or a TLS
reverse proxy (e.g. Caddy) - same guidance as `server/README.md`. Do not expose the port to
the internet directly. No A100 hostnames or credentials exist in this repo yet: use env-var
placeholders, never commit them.

Optional GPU upgrade path for voice (documented, **not implemented in v1**): host Whisper
(STT) and Piper/Coqui (TTS) on the A100 and point the browser at them via env-var-configured
URLs such as `WEBAPP_STT_URL` / `WEBAPP_TTS_URL`. Those variables are placeholders only;
v1 ignores them and stays on the browser APIs.

## Hackathon demo script

1. `export ANTHROPIC_API_KEY=...` (shell env only)
2. `.venv/bin/python webapp/run.py --backend sim --port 8920`
3. Open `http://127.0.0.1:8920` - camera frames and joint telemetry stream in live.
4. Click the chip **"put the red block in the blue tray"** (the sim ships `red_block`,
   `green_block`, `blue_tray`), SEND, and watch the tool calls stream: `look` snapshots appear
   inline, `pick_cell`/`place_cell` drive the arm.
5. Toggle MIC and say the task instead (Chrome/Edge); toggle SPK for spoken replies.
6. Show the E-STOP: press it mid-task - the agent is cancelled and Guard latches; RELEASE clears it.
7. Mention station mode: same UI, plus `--i-have-a-human` and the 10-minute human-presence gate.

## Tests

```
.venv/bin/python -m pytest webapp/tests -q
```

No hardware and no API key needed (the anthropic client is mocked). Baseline
`.venv/bin/python scripts/selftest_sim.py` is untouched and still passes.

## Changelog

Changes to this layer are documented here and as a single entry in the repo `CHANGELOG.md`
under `## [Unreleased] - 2026-11-22` -> `### Added`. See also `docs/13_WEB_VOICE_CONSOLE.md`
for the architecture page.
