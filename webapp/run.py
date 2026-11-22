#!/usr/bin/env python3
"""normakit web console entry point.

  .venv/bin/python webapp/run.py --backend sim --port 8920
  .venv/bin/python webapp/run.py --backend station --i-have-a-human   # real arm, human at the table

Mirrors scripts/run_policy.py: the station backend is refused without --i-have-a-human.
Binds 127.0.0.1 by default. ANTHROPIC_API_KEY comes from the shell env only; the server boots
fine without it (chat then returns a clear error event).
"""
from __future__ import annotations

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))
sys.path.insert(0, _HERE)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--backend", choices=["sim", "station"], default="sim")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8920")))
    ap.add_argument("--station-host", default="localhost:8888", help="Station host (station backend)")
    ap.add_argument("--i-have-a-human", action="store_true", help="required for --backend station")
    a = ap.parse_args()

    if a.backend == "station" and not a.i_have_a_human:
        print("refusing: station backend needs --i-have-a-human (a person must be at the table)",
              file=sys.stderr)
        return 2

    from normakit.mcp_server import build
    try:
        arm = build(a.backend, a.station_host)
    except Exception as e:
        print(f"error: could not start {a.backend} backend ({a.station_host}): {e}", file=sys.stderr)
        return 3

    from app import create_app
    from safety import HumanGate
    import uvicorn
    uvicorn.run(create_app(arm, a.backend, gate=HumanGate(a.backend)), host=a.host, port=a.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
