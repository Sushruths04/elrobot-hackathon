#!/usr/bin/env python3
"""Policy inference server for a remote A100.

  python server/policy_server.py --checkpoint <hf repo or dir> --policy-type act --port 8000
  python server/policy_server.py --dummy --port 8765      # no torch/GPU

Endpoints: GET /health, POST /act, POST /reset, GET /metrics.
Optional auth: set env POLICY_API_TOKEN, clients send "Authorization: Bearer <token>".
"""
from __future__ import annotations

import argparse
import hmac
import io
import math
import os
import sys
import threading
import time
from collections import deque

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from fastapi import Depends, FastAPI, Header, HTTPException  # noqa: E402

from normakit.policy_api import (  # noqa: E402
    DEFAULT_CHUNK, DEFAULT_CHUNK_DT, N_JOINTS, ActRequest, ActResponse, decode_jpeg_b64,
)

IMAGE_KEY = "observation.images.table"
STATE_KEY = "observation.state"


class DummyPolicy:
    """Deterministic smooth chunk: eases from the given state toward a fixed pose."""
    name = "dummy"
    device = "cpu"

    def __init__(self, chunk=DEFAULT_CHUNK):
        self.chunk = chunk
        self.target = [0.5] * (N_JOINTS - 1) + [0.3]

    def reset(self):
        pass

    def act(self, state, jpeg, task):
        out = []
        for k in range(1, self.chunk + 1):
            a = 0.5 - 0.5 * math.cos(math.pi * k / (self.chunk * 4))  # max ~0.15 of the way
            out.append([min(1.0, max(0.0, s + (t - s) * a)) for s, t in zip(state, self.target)])
        return out, DEFAULT_CHUNK_DT


class LeRobotPolicy:
    def __init__(self, checkpoint, policy_type, device):
        import torch
        from lerobot.policies.factory import get_policy_class, make_pre_post_processors

        self.torch = torch
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.name = f"{policy_type}:{checkpoint}"
        cls = get_policy_class(policy_type)
        self.policy = cls.from_pretrained(checkpoint).to(device).eval()
        self.pre, self.post = make_pre_post_processors(
            self.policy.config, pretrained_path=checkpoint,
            preprocessor_overrides={"device_processor": {"device": device}},
        )
        self.chunk = getattr(self.policy.config, "chunk_size", DEFAULT_CHUNK)
        self.fps = getattr(self.policy.config, "fps", None) or 30
        self.use_amp = device.startswith("cuda")

    def reset(self):
        self.policy.reset()

    def act(self, state, jpeg, task):
        torch = self.torch
        import numpy as np
        from PIL import Image

        batch = {STATE_KEY: torch.tensor(state, dtype=torch.float32).unsqueeze(0)}
        if jpeg is not None:
            img = np.asarray(Image.open(io.BytesIO(jpeg)).convert("RGB"))
            t = torch.from_numpy(img).permute(2, 0, 1).float().div(255.0).unsqueeze(0)
            batch[IMAGE_KEY] = t
        if task:
            batch["task"] = [task]
        with torch.inference_mode():
            batch = self.pre(batch)
            chunk = self.policy.predict_action_chunk(batch)  # (1, N, D)
            chunk = self.post(chunk)
        rows = chunk.squeeze(0).float().cpu().clamp(0.0, 1.0).tolist()
        return [r[:N_JOINTS] for r in rows], 1.0 / float(self.fps)


def create_app(policy, token=None):
    app = FastAPI(title="normakit policy server")
    lat = deque(maxlen=1000)
    lock = threading.Lock()  # one GPU inference at a time
    tok = token if token is not None else os.environ.get("POLICY_API_TOKEN")

    def auth(authorization: str = Header(default="")):
        if tok and not hmac.compare_digest(authorization, f"Bearer {tok}"):
            raise HTTPException(401, "bad or missing bearer token")

    @app.get("/health")
    def health():
        return {"ready": True, "model": policy.name, "device": policy.device,
                "chunk": policy.chunk, "joints": N_JOINTS}

    @app.post("/act", response_model=ActResponse, dependencies=[Depends(auth)])
    def act(req: ActRequest):
        t0 = time.perf_counter()
        jpeg = decode_jpeg_b64(req.image_jpeg_b64) if req.image_jpeg_b64 else None
        with lock:
            if req.reset:
                policy.reset()
            chunk, dt = policy.act(req.state, jpeg, req.task)
        ms = (time.perf_counter() - t0) * 1000
        lat.append(ms)
        return ActResponse(action_chunk=chunk, chunk_dt=dt, server_ms=round(ms, 3))

    @app.post("/reset", dependencies=[Depends(auth)])
    def reset():
        with lock:
            policy.reset()
        return {"ok": True}

    @app.get("/metrics")
    def metrics():
        v = sorted(lat)
        if not v:
            return {"n": 0}
        q = lambda p: v[min(len(v) - 1, int(p * len(v)))]
        return {"n": len(v), "p50_ms": q(0.5), "p95_ms": q(0.95), "max_ms": v[-1]}

    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dummy", action="store_true", help="no torch/GPU, deterministic chunk")
    ap.add_argument("--checkpoint", default=os.environ.get("CHECKPOINT"))
    ap.add_argument("--policy-type", default=os.environ.get("POLICY_TYPE", "act"))
    ap.add_argument("--device", default=os.environ.get("DEVICE", "auto"))
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8000")))
    a = ap.parse_args()
    if a.dummy:
        policy = DummyPolicy()
    elif a.checkpoint:
        policy = LeRobotPolicy(a.checkpoint, a.policy_type, a.device)
    else:
        ap.error("--checkpoint (or env CHECKPOINT) required unless --dummy")
    import uvicorn
    uvicorn.run(create_app(policy), host=a.host, port=a.port)


if __name__ == "__main__":
    main()
