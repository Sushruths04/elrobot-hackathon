# NormaCore hackathon - ElRobot + Claude (HackLab Berlin, 21-22 Nov 2026)

Setup notes: `bin/`, `vendor/` and `.venv` are NOT in git.
- Station binary: https://github.com/norma-core/norma-core/releases/tag/v0.1.0-beta.10 -> `bin/station`
- Upstream source (URDF, protobufs): `git clone https://github.com/norma-core/norma-core vendor/norma-core-src`
- Python: `python3 -m venv --system-site-packages .venv && .venv/bin/pip install 'mcp<2' anthropic pybullet pillow numpy`
- Shortcuts for the rig: `./norma help`. Calibration backups: `calibration/`. Findings: `docs/ISSUES.md`.

# HackLab Berlin 21-22 Nov 2026 - NormaCore track starter kit
"AI-Powered Robot Control: integrate the Station API into Codex or Claude so it controls the robot by looking through the cameras."

```
docs/01_STATION_DEEP_DIVE.md   what Station really is (protocol, queues, registers, gotchas)
docs/02_TASK_PREDICTIONS.md    what the challenge probably is, with probabilities
docs/03_WINNING_ANGLES.md      the non-obvious design (action decoder, self-calibration, sensor-verified skills ...)
docs/04_PLAN_48H_AND_PREP.md   prep checklist, hour-by-hour plan, pitch, risks
src/normakit/                  core.py (types+safety) | kin.py (URDF IK) | vision.py | arm.py (skills)
                               sim_backend.py (digital twin) | station_backend.py (real arm)
                               mcp_server.py | agent.py | robotctl.py
scripts/selftest_sim.py        full logic test with zero hardware
scripts/smoke_station.py       first command on site
bin/station                    Station v0.1.0-beta.10 (linux x86_64)
vendor/norma-core-src/         full upstream source + generated Python protobufs
run/station.yaml               Station config (full-res camera frames)
```

## Quick start
```bash
.venv/bin/python scripts/selftest_sim.py                       # no hardware needed
.venv/bin/python -m normakit.mcp_server --backend sim          # MCP over stdio (Claude Code picks up .mcp.json)
ANTHROPIC_API_KEY=... .venv/bin/python -m normakit.agent "put the red block in the blue tray"
# On site:
cd run && ../bin/station --web --tcp --config station.yaml     # http://localhost:8889
.venv/bin/python scripts/smoke_station.py
.venv/bin/python -m normakit.mcp_server --backend station
```

## Status honesty
- Tested (digital twin, `run/selftest.log`, all PASS): IK (0.07 cm), safety guard + e-stop, camera+grid, self-calibration (0.04 cm fit error, 10 points), pixel->table estimate (1.6 cm off on the red block), MCP tool registry (12 tools).
- Known sim issues: the final descent to table height times out in the twin (probably contact with the block/table) - grasping is NOT verified even in sim. The 1.6 cm error is parallax: calibrate with the sticker at the same height as the objects you will pick.
- `agent.py` (Claude loop) and `robotctl.py` were written but not run (no API call made in this session).
- NOT tested: `station_backend.py` against a live Station (running the downloaded binary was not permitted in the dev session). Written from source; expect small fixes.
- Assumptions to verify on the arm: linear map calibrated-range -> URDF limits; `TABLE_Z`; gripper direction (norm 1 = open); sticker colour.
- The task itself is predicted, not known.
