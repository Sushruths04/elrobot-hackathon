"""Shared request/response schema for the A100 policy server (server/policy_server.py).

All joint values are normalised 0..1 (JointState.norm), joints 1..8, 8 = gripper.
"""
from __future__ import annotations

import base64
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

N_JOINTS = 8
DEFAULT_CHUNK = 20
DEFAULT_CHUNK_DT = 1.0 / 30.0  # LeRobot dataset fps


class ActRequest(BaseModel):
    state: List[float] = Field(..., description="8 joint positions, 0..1 norm")
    image_jpeg_b64: Optional[str] = Field(None, description="table camera JPEG, base64")
    images_jpeg_b64: Optional[Dict[str, str]] = Field(
        None, description="camera name (as recorded, e.g. 'table', 'wrist') -> JPEG base64; for multi-camera policies")
    task: str = ""
    reset: bool = False

    @field_validator("state")
    @classmethod
    def _state_len(cls, v):
        if len(v) != N_JOINTS:
            raise ValueError(f"state must have {N_JOINTS} floats")
        return v


class ActResponse(BaseModel):
    action_chunk: List[List[float]] = Field(..., description="N x 8, 0..1 norm")
    chunk_dt: float = Field(..., description="seconds between chunk rows")
    server_ms: float


def encode_jpeg_b64(jpeg_bytes: bytes) -> str:
    return base64.b64encode(jpeg_bytes).decode("ascii")


def decode_jpeg_b64(s: str) -> bytes:
    return base64.b64decode(s)
