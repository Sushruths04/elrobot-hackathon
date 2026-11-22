#!/usr/bin/env python3
"""Policy inference server for a remote A100.

  python server/policy_server.py --checkpoint <hf repo or dir> --policy-type act --port 8000
  python server/policy_server.py --dummy --port 8765      # no torch/GPU

Endpoints: GET /health, POST /act, POST /reset, GET /metrics.
Auth: set env POLICY_API_TOKEN, clients send "Authorization: Bearer <token>". The server refuses to
start without a token unless --dummy or --no-auth. Binds 127.0.0.1 by default. Plain HTTP: use an
SSH tunnel, VPN or a TLS proxy across the WAN (see server/README.md).
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

DEFAULT_IMAGE_KEY = "observation.images.table"
STATE_KEY = "observation.state"


class DummyPolicy:
    """Deterministic smooth chunk: eases from the given state toward a fixed pose."""
    name = "dummy"
    device = "cpu"
    needs_image = False

    def __init__(self, chunk=DEFAULT_CHUNK):
        self.chunk = chunk
        self.target = [0.5] * (N_JOINTS - 1) + [0.3]

    def reset(self):
        pass

    def act(self, state, jpeg, task, images=None):
        out = []
        for k in range(1, self.chunk + 1):
            a = 0.5 - 0.5 * math.cos(math.pi * k / (self.chunk * 4))  # max ~0.15 of the way
            out.append([min(1.0, max(0.0, s + (t - s) * a)) for s, t in zip(state, self.target)])
        return out, DEFAULT_CHUNK_DT


class LeRobotPolicy:
    def __init__(self, checkpoint, policy_type, device, image_key=None):
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
        self._validate_config(image_key)

    def _validate_config(self, image_key):
        cfg = self.policy.config
        inputs = getattr(cfg, "input_features", None) or {}
        out = getattr(cfg, "output_features", None) or {}
        self.image_keys = [k for k in inputs if k.startswith("observation.images.")]
        self.needs_image = bool(self.image_keys)
        # SmolVLA checkpoints store camera1/camera2 and rename the recorded keys in the preprocessor;
        # feed the recorded keys (observation.images.table / .wrist) and let the preprocessor rename.
        inv = {}
        for step in getattr(self.pre, "steps", []):
            for src, dst in (getattr(step, "rename_map", None) or {}).items():
                inv[dst] = src
        if inv:  # keep only cameras that were actually recorded; SmolVLA skips missing slots (e.g. camera3)
            self.image_keys = [k for k in self.image_keys if k in inv]
            self.needs_image = bool(self.image_keys)
        self.feed_keys = {k: inv.get(k, k) for k in self.image_keys}       # checkpoint key -> key we feed
        self.cameras = [v.removeprefix("observation.images.") for v in self.feed_keys.values()]
        if self.needs_image:
            if image_key and image_key not in self.feed_keys.values() and image_key not in self.image_keys:
                raise ValueError(f"--image-key {image_key} not in checkpoint inputs {self.image_keys}")
            self.image_key = list(self.feed_keys.values())[0]
        # SmolVLA pads state to max_state_dim, and fine-tuning from smolvla_base keeps the base's declared
        # state shape (6,) even though it trained on our 8 joints, so only check the shape for other policies.
        if STATE_KEY in inputs and not hasattr(cfg, "max_state_dim"):
            sd = tuple(inputs[STATE_KEY].shape)
            if sd != (N_JOINTS,):
                raise ValueError(f"checkpoint state shape {sd}, expected ({N_JOINTS},)")
        act = out.get("action")
        if act is None or tuple(act.shape) != (N_JOINTS,):
            raise ValueError(f"checkpoint action shape {getattr(act, 'shape', None)}, "
                             f"expected ({N_JOINTS},)")

    def reset(self):
        self.policy.reset()

    def act(self, state, jpeg, task, images=None):
        """images: camera name ('table', 'wrist') -> JPEG. A lone jpeg is used for the first camera."""
        torch = self.torch
        import numpy as np
        from PIL import Image

        batch = {STATE_KEY: torch.tensor(state, dtype=torch.float32).unsqueeze(0)}
        images = dict(images or {})
        if jpeg is not None and self.needs_image and self.cameras[0] not in images:
            images[self.cameras[0]] = jpeg
        missing = [c for c in self.cameras if c not in images]
        if self.needs_image and missing:
            raise ValueError(f"this policy needs camera images {self.cameras}; missing {missing}")
        for cam in self.cameras:
            img = np.asarray(Image.open(io.BytesIO(images[cam])).convert("RGB"))
            batch["observation.images." + cam] = torch.from_numpy(img).permute(2, 0, 1).float().div(255.0).unsqueeze(0)
        if task:
            batch["task"] = [task]
        with torch.inference_mode():
            batch = self.pre(batch)
            chunk = self.policy.predict_action_chunk(batch)  # (1, N, D)
            chunk = self.post(chunk)
        arr = chunk.squeeze(0).float().cpu().clone()
        if arr.ndim != 2 or arr.shape[1] != N_JOINTS:
            raise RuntimeError(f"policy returned action shape {tuple(arr.shape)}, expected (N, {N_JOINTS})")
        if bool(torch.isfinite(arr).all()) and arr.shape[1] >= 8:
            arr[:, 7] = arr[:, 7].clamp(0.0, 1.0)                     # gripper only: ACT 5k predicts -0.12 / 1.12 now and then (2/206 held-out frames); arm joints stay strict
        if not bool(torch.isfinite(arr).all()) or float(arr.min()) < -0.1 or float(arr.max()) > 1.1:
            raise RuntimeError("policy actions are outside the 0..1 joint-norm range "
                               "(checkpoint trained on other units?)")
        return arr.clamp(0.0, 1.0).tolist(), 1.0 / float(self.fps)


def create_app(policy, token=None):
    app = FastAPI(title="normakit policy server")
    lat = deque(maxlen=1000)
    lock = threading.Lock()  # one GPU inference at a time
    tok = token if token is not None else os.environ.get("POLICY_API_TOKEN")

    def auth(authorization: str = Header(default="")):
        if tok and not hmac.compare_digest(authorization, f"Bearer {tok}"):
            raise HTTPException(401, "bad or missing bearer token")

    @app.get("/health", dependencies=[Depends(auth)])
    def health():
        return {"ready": True, "model": policy.name, "device": policy.device,
                "chunk": policy.chunk, "joints": N_JOINTS, "cameras": getattr(policy, "cameras", [])}

    @app.post("/act", response_model=ActResponse, dependencies=[Depends(auth)])
    def act(req: ActRequest):
        t0 = time.perf_counter()
        if not all(math.isfinite(x) for x in req.state):
            raise HTTPException(422, "state contains NaN or Inf")
        if getattr(policy, "needs_image", False) and not (req.image_jpeg_b64 or req.images_jpeg_b64):
            raise HTTPException(422, f"this policy requires images {getattr(policy, 'cameras', [])}")
        try:
            jpeg = decode_jpeg_b64(req.image_jpeg_b64) if req.image_jpeg_b64 else None
            images = {k: decode_jpeg_b64(v) for k, v in (req.images_jpeg_b64 or {}).items()}
        except Exception:
            raise HTTPException(422, "image payload is not valid base64")
        with lock:
            if req.reset:
                policy.reset()
            try:
                chunk, dt = policy.act(req.state, jpeg, req.task, images)
            except ValueError as e:
                raise HTTPException(422, str(e))
        ms = (time.perf_counter() - t0) * 1000
        lat.append(ms)
        return ActResponse(action_chunk=chunk, chunk_dt=dt, server_ms=round(ms, 3))

    @app.post("/reset", dependencies=[Depends(auth)])
    def reset():
        with lock:
            policy.reset()
        return {"ok": True}

    @app.get("/metrics", dependencies=[Depends(auth)])
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
    ap.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"),
                    help="bind address (default 127.0.0.1; use 0.0.0.0 only with a token)")
    ap.add_argument("--image-key", default=os.environ.get("IMAGE_KEY"),
                    help="expected camera key; default read from the checkpoint")
    ap.add_argument("--no-auth", action="store_true", help="allow serving without POLICY_API_TOKEN")
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8000")))
    a = ap.parse_args()
    if not (a.dummy or a.no_auth or os.environ.get("POLICY_API_TOKEN")):
        ap.error("POLICY_API_TOKEN is not set; set it, or pass --no-auth (or --dummy)")
    if a.dummy:
        policy = DummyPolicy()
    elif a.checkpoint:
        policy = LeRobotPolicy(a.checkpoint, a.policy_type, a.device, a.image_key)
    else:
        ap.error("--checkpoint (or env CHECKPOINT) required unless --dummy")

    import uvicorn
    uvicorn.run(create_app(policy), host=a.host, port=a.port)


if __name__ == "__main__":
    main()
