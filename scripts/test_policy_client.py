#!/usr/bin/env python3
"""No-hardware test: dummy policy server + SimBackend(realtime=False) + Guard.
Run: .venv/bin/python scripts/test_policy_client.py
"""
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from normakit.core import Guard
from normakit.policy_client import PolicyClient, PolicyError
from normakit.sim_backend import SimBackend
from run_policy import run_loop

PORT = 8765
ok = True


def check(name, cond, extra=""):
    global ok
    ok &= bool(cond)
    print(("PASS " if cond else "FAIL ") + name, extra)


env = {k: v for k, v in os.environ.items() if k != "POLICY_API_TOKEN"}
srv = subprocess.Popen([sys.executable, str(ROOT / "server" / "policy_server.py"), "--dummy",
                        "--port", str(PORT)], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
try:
    cl = PolicyClient(f"http://127.0.0.1:{PORT}", timeout=5)
    for _ in range(50):
        try:
            cl.health()
            break
        except PolicyError:
            time.sleep(0.2)
    check("health", cl.health().get("status", "ok") is not None, str(cl.health()))

    be = SimBackend(realtime=False)
    be.start()
    g = Guard(be)
    g.cfg.max_cmd_hz = 100.0  # test speed only; limits below are unchanged
    sent = run_loop(be, cl, "test", hz=100.0, steps=30, guard=g, log=print)
    check("30 commands sent", len(sent) == 30, str(len(sent)))
    c = g.cfg
    vals = [v for s in sent for v in s["sent"].values()]
    check("goals within 0.02..0.98", all(c.min_norm - 1e-9 <= v <= c.max_norm + 1e-9 for v in vals))
    be.stop_backend()

    bad = PolicyClient("http://127.0.0.1:1", timeout=1)
    try:
        bad.act([0.5] * 8)
        check("unreachable raises PolicyError", False)
    except PolicyError:
        check("unreachable raises PolicyError", True)

    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "run_policy.py"), "--backend", "station"],
                       capture_output=True, text=True)
    check("station without flag exits nonzero", r.returncode != 0, f"rc={r.returncode}")
finally:
    srv.terminate()
    srv.wait(timeout=5)

print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
