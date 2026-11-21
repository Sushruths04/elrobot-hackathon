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
import numpy as np
from run_policy import run_loop, sample_chunk

PORT = 8765
ok = True


def check(name, cond, extra=""):
    global ok
    ok &= bool(cond)
    print(("PASS " if cond else "FAIL ") + name, extra)


TOKEN = "test-token"
env = {k: v for k, v in os.environ.items() if k != "POLICY_API_TOKEN"}
env["POLICY_API_TOKEN"] = TOKEN
srv = subprocess.Popen([sys.executable, str(ROOT / "server" / "policy_server.py"), "--dummy",
                        "--port", str(PORT)], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
try:
    os.environ["POLICY_API_TOKEN"] = TOKEN
    cl = PolicyClient(f"http://127.0.0.1:{PORT}", timeout=5)
    for _ in range(50):
        try:
            cl.health()
            break
        except PolicyError:
            time.sleep(0.2)
    h = cl.health()
    check("health ready", h.get("ready") is True and h.get("joints") == 8, str(h))

    os.environ["POLICY_API_TOKEN"] = "wrong"
    for name, call in (("health", lambda c: c.health()), ("act", lambda c: c.act([0.5] * 8)),
                       ("reset", lambda c: c.reset())):
        try:
            call(PolicyClient(f"http://127.0.0.1:{PORT}", timeout=5))
            check(f"bad token rejected on {name}", False)
        except PolicyError as e:
            check(f"bad token rejected on {name}", "401" in str(e), str(e)[:60])
    os.environ["POLICY_API_TOKEN"] = TOKEN

    try:
        PolicyClient(f"http://127.0.0.1:{PORT}").act([0.5] * 7)
        check("short state rejected", False)
    except (PolicyError, ValueError):
        check("short state rejected", True)

    # server refuses to start without a token unless --dummy / --no-auth
    env_nt = {k: v for k, v in os.environ.items() if k != "POLICY_API_TOKEN"}
    r = subprocess.run([sys.executable, str(ROOT / "server" / "policy_server.py"), "--checkpoint", "x"],
                       env=env_nt, capture_output=True, text=True, timeout=30)
    check("server refuses to start without token", r.returncode != 0 and "POLICY_API_TOKEN" in r.stderr,
          f"rc={r.returncode}")

    # 422 instead of 500 for a vision policy without an image, and for NaN state (in-process app)
    sys.path.insert(0, str(ROOT / "server"))
    from fastapi.testclient import TestClient
    from policy_server import DummyPolicy, create_app

    class VisionDummy(DummyPolicy):
        needs_image = True

    tc = TestClient(create_app(VisionDummy(), token="t"))
    hdr = {"Authorization": "Bearer t"}
    r = tc.post("/act", json={"state": [0.5] * 8}, headers=hdr)
    check("vision policy without image gives 422", r.status_code == 422, str(r.status_code))
    r = tc.post("/act", content='{"state": [NaN, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], "image_jpeg_b64": "AA=="}',
                headers={**hdr, "Content-Type": "application/json"})
    check("NaN state gives 422", r.status_code == 422, str(r.status_code))
    check("metrics needs token", tc.get("/metrics").status_code == 401)

    # chunk playback is real time: fake 20-row chunk, refetch after 10 rows consumed
    class Fake:
        def __init__(self, dt):
            self.chunk_dt, self.calls = dt, 0
            self.last_server_ms = 0.0

        def act(self, state, jpeg=None, task=""):
            self.calls += 1
            return np.tile(np.linspace(0.4, 0.6, 20)[:, None], (1, 8))

    ramp = [[i / 10.0] * 8 for i in range(11)]
    check("sample_chunk interpolates", abs(sample_chunk(ramp, 2.5)[0] - 0.25) < 1e-9
          and abs(sample_chunk(ramp, 99.0)[0] - 1.0) < 1e-9)
    be = SimBackend(realtime=False)
    be.start()
    g = Guard(be)
    g.cfg.max_cmd_hz = 100.0  # test speed only; limits below are unchanged
    sent = run_loop(be, cl, "test", hz=100.0, steps=30, guard=g, log=print)
    check("30 commands sent", len(sent) == 30, str(len(sent)))
    c = g.cfg
    vals = [v for s in sent for v in s["sent"].values()]
    check("goals within 0.02..0.98", all(c.min_norm - 1e-9 <= v <= c.max_norm + 1e-9 for v in vals))
    raw = cl.act([0.5] * 8)
    check("server chunk shape (N, 8) and in 0..1", raw.ndim == 2 and raw.shape[1] == 8
          and raw.min() >= 0 and raw.max() <= 1, str(raw.shape))
    # 100 Hz ticks: 30 fps rows advance 0.33 row/tick (1 fetch in 30 ticks); 100 fps rows advance
    # 1 row/tick (3 fetches). The old one-row-per-tick loop gave 3 fetches for both.
    for dt, want in ((1 / 30, 1), (1 / 100, 3)):
        g2 = Guard(be)
        g2.cfg.max_cmd_hz = 100.0
        f = Fake(dt)
        run_loop(be, f, "t", hz=100.0, steps=30, guard=g2, log=print)
        check(f"playback rate honours chunk_dt={dt:.3f}", f.calls == want, f"fetches={f.calls} want={want}")
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
