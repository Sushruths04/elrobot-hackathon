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


def sample_chunk(chunk, pos):
    """Linear interpolation of chunk rows at fractional row position pos (clamped to the last row)."""
    n = len(chunk)
    i = min(int(pos), n - 1)
    j = min(i + 1, n - 1)
    f = min(1.0, pos - i) if i < n - 1 else 0.0
    return [(1 - f) * chunk[i][k] + f * chunk[j][k] for k in range(len(chunk[i]))]


def resolve_cameras(backend, wanted):
    """Map recorded camera names ('table', 'wrist') to backend camera ids. Station renumbers cameras on
    every replug, so use the same detectors as scripts/record_teleop.py."""
    if not wanted:
        return {}
    if hasattr(backend, "_cam_order"):  # StationBackend
        from normakit.station_backend import front_camera, wrist_camera
        found = {"table": front_camera(backend), "wrist": wrist_camera(backend)}
    else:                               # SimBackend names its cameras 'table' and 'wrist'
        found = {c: c for c in backend.cameras()}
    missing = [c for c in wanted if not found.get(c)]
    if missing:
        raise RuntimeError(f"policy needs cameras {wanted}; not found on the robot: {missing}")
    if "table" in wanted and "wrist" in wanted and found["table"] == found["wrist"]:
        raise RuntimeError("table and wrist resolved to the same camera; check the wrist camera")
    return {c: found[c] for c in wanted}


def grab(backend, cams):
    return {name: backend.frame_jpeg(cid) for name, cid in cams.items()}


def run_loop(backend, client, task, hz=10.0, steps=100, max_delta=0.05, guard=None, log=print, cams=None):
    """Returns list of dicts sent to Guard.move results. Stops on e-stop or server error.

    Chunk rows are chunk_dt seconds apart (30 fps for ACT). Each tick of 1/hz seconds advances the
    chunk position by 1/(hz*chunk_dt) rows with linear interpolation, so playback runs in real time.
    The next chunk is fetched (synchronously) once half the chunk is consumed; the later half is
    dropped because it is predicted from a staler observation.
    """
    guard = guard or Guard(backend)
    guard.cfg.max_cmd_hz = min(guard.cfg.max_cmd_hz, hz)
    chunk, pos, sent = None, 0.0, []
    for step in range(steps):
        if guard.estopped:
            log("e-stop active, stopping")
            break
        if chunk is None or pos >= max(1.0, len(chunk) / 2):
            norms = backend.state().norms()
            if cams:
                chunk = client.act([norms[i] for i in JOINT_IDS], None, task, images=grab(backend, cams))
            else:
                chunk = client.act([norms[i] for i in JOINT_IDS], backend.frame_jpeg(), task)
            pos = 0.0
        cur = backend.state().norms()
        row = sample_chunk(chunk, pos)
        goals = {}
        for k, jid in enumerate(JOINT_IDS):
            goals[jid] = cur[jid] + max(-max_delta, min(max_delta, float(row[k]) - cur[jid]))
        pos += 1.0 / (hz * max(client.chunk_dt, 1e-6))
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
        health = client.health()
        print("server:", health)
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
        cams = resolve_cameras(be, health.get("cameras") or [])
        if len(cams) > 1:
            print("cameras:", cams)
        else:
            cams = None  # single-camera checkpoint: keep the original one-image path
        client.reset()  # clear any policy-side action queue from a previous episode
        sent = run_loop(be, client, a.task, a.hz, a.steps, a.max_delta, guard, cams=cams)
        print(f"sent {len(sent)} commands, last server_ms={client.last_server_ms:.1f}")
        return 0
    except (PolicyError, RuntimeError) as e:
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
