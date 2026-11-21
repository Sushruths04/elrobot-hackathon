#!/usr/bin/env python3
"""Run a remote (A100) policy on the follower arm through the Guard.

  .venv/bin/python scripts/run_policy.py --backend sim --url http://$A100_HOST:8000 --task "pick the cup"
  .venv/bin/python scripts/run_policy.py --backend station --i-have-a-human --url ...

Every command goes through Guard.move (clamp, step limit, 10 Hz, overcurrent e-stop).
Station mode moves the real arm and needs --i-have-a-human.
"""
import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from normakit.core import JOINT_IDS, Guard
from normakit.policy_client import PolicyClient, PolicyError

SAFETY = """SAFETY (AGENTS.md): the arm will move on its own. A human must stand at the e-stop
(tools/estop.py), the workspace must be clear, and leader mirroring must be OFF."""


def run_loop(backend, client, task, hz=10.0, steps=100, max_delta=0.05, guard=None, log=print):
    """Returns list of dicts sent to Guard.move results. Stops on e-stop or server error."""
    guard = guard or Guard(backend)
    guard.cfg.max_cmd_hz = min(guard.cfg.max_cmd_hz, hz)
    chunk, idx, sent = None, 0, []
    for step in range(steps):
        if guard.estopped:
            log("e-stop active, stopping")
            break
        if chunk is None or idx >= max(1, len(chunk) // 2):
            norms = backend.state().norms()
            chunk = client.act([norms[i] for i in JOINT_IDS], backend.frame_jpeg(), task)
            idx = 0
        cur = backend.state().norms()
        goals = {}
        for k, jid in enumerate(JOINT_IDS):
            tgt = float(chunk[idx][k])
            goals[jid] = cur[jid] + max(-max_delta, min(max_delta, tgt - cur[jid]))
        idx += 1
        res = guard.move(goals)
        if not res["ok"]:
            log(f"guard refused: {res['error']}")
            break
        sent.append(res)
    return sent


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--backend", choices=["sim", "station"], default="sim")
    ap.add_argument("--host", default="localhost:8888", help="Station host (station backend)")
    ap.add_argument("--url", default=f"http://{os.environ.get('A100_HOST', 'localhost')}:8000")
    ap.add_argument("--task", default="")
    ap.add_argument("--hz", type=float, default=10.0)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--max-delta", type=float, default=0.05, help="max norm change per tick per joint")
    ap.add_argument("--i-have-a-human", action="store_true", help="required for --backend station")
    a = ap.parse_args()

    if a.backend == "station":
        print(SAFETY)
        if not a.i_have_a_human:
            print("refusing: station backend needs --i-have-a-human", file=sys.stderr)
            return 2

    client = PolicyClient(a.url)
    try:
        print("server:", client.health())
    except PolicyError as e:
        print(e, file=sys.stderr)
        return 3

    if a.backend == "sim":
        from normakit.sim_backend import SimBackend
        be = SimBackend(realtime=True)
    else:
        from normakit.station_backend import StationBackend
        be = StationBackend(a.host)
    be.start()
    time.sleep(1.0)
    guard = Guard(be)
    try:
        sent = run_loop(be, client, a.task, a.hz, a.steps, a.max_delta, guard)
        print(f"sent {len(sent)} commands, last server_ms={client.last_server_ms:.1f}")
        return 0
    except PolicyError as e:
        print(e, file=sys.stderr)
        guard.estop()
        return 3
    except KeyboardInterrupt:
        guard.estop()
        return 130
    finally:
        be.stop_backend()


if __name__ == "__main__":
    sys.exit(main())
