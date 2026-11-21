"""Robot-side HTTP client for the A100 policy server (server/policy_server.py)."""
from __future__ import annotations

import os

import numpy as np
import requests

from .policy_api import N_JOINTS, ActRequest, ActResponse, encode_jpeg_b64


class PolicyError(RuntimeError):
    pass


class PolicyClient:
    def __init__(self, url: str, token_env: str = "POLICY_API_TOKEN", timeout: float = 5.0):
        self.url = url.rstrip("/")
        self.timeout = timeout
        tok = os.environ.get(token_env, "")
        self.headers = {"Authorization": f"Bearer {tok}"} if tok else {}
        self.chunk_dt = 1.0 / 30.0
        self.last_server_ms = 0.0

    def _call(self, method: str, path: str, **kw):
        try:
            r = requests.request(method, self.url + path, headers=self.headers,
                                 timeout=self.timeout, **kw)
        except requests.RequestException as e:
            raise PolicyError(f"policy server unreachable at {self.url}: {e.__class__.__name__}") from e
        if r.status_code != 200:
            raise PolicyError(f"policy server {path} returned HTTP {r.status_code}: {r.text[:200]}")
        return r.json()

    def health(self) -> dict:
        return self._call("GET", "/health")

    def reset(self) -> None:
        self._call("POST", "/reset")

    def act(self, state_norms, jpeg: bytes | None = None, task: str = "") -> np.ndarray:
        """state_norms: 8 floats (joint 1..8) or dict jid->norm. Returns (N, 8) array."""
        if isinstance(state_norms, dict):
            state_norms = [state_norms[i] for i in range(1, N_JOINTS + 1)]
        req = ActRequest(state=[float(x) for x in state_norms],
                         image_jpeg_b64=encode_jpeg_b64(jpeg) if jpeg else None, task=task)
        resp = ActResponse(**self._call("POST", "/act", json=req.model_dump()))
        chunk = np.asarray(resp.action_chunk, dtype=float)
        if chunk.ndim != 2 or chunk.shape[1] != N_JOINTS or len(chunk) == 0:
            raise PolicyError(f"bad action chunk shape {chunk.shape}")
        self.chunk_dt = resp.chunk_dt
        self.last_server_ms = resp.server_ms
        return chunk
