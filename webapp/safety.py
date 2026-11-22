"""Safety gating for the web console: motion-tool classification + per-session human-presence window.

Sim backend needs no confirmation. Station mode mirrors the team's ARM-motion protocol (:8890):
a person must confirm presence in the UI, and motion tools are only forwarded while that
confirmation is fresh (default 10-minute window). Guard (src/normakit/core.py) stays the hard
layer underneath - this gate is the convenience layer on top of it.
"""
from __future__ import annotations

import threading
import time

MOTION_TOOLS = {"pick_cell", "place_cell", "reach_cell", "reach_xyz",
                "gripper", "home", "move_joints", "calibrate_table"}
NON_MOTION = {"look", "arm_state", "stop", "release"}

CONFIRM_WINDOW_MIN = 10.0


class HumanGate:
    """Human-presence confirmation window + the single-writer lock for bus access."""

    def __init__(self, backend: str):
        self.backend = backend
        self._confirmed_until = 0.0
        # One writer on the bus (AGENTS.md rule 6): the agent loop and any direct
        # motion call must hold this same lock.
        self.lock = threading.Lock()

    def confirm(self, minutes: float = CONFIRM_WINDOW_MIN) -> float:
        """A human is at the table: open the motion window. Returns the window in seconds."""
        self._confirmed_until = time.time() + minutes * 60.0
        return minutes * 60.0

    def ok(self) -> bool:
        if self.backend == "sim":
            return True
        return time.time() < self._confirmed_until

    def remaining_s(self) -> float:
        if self.backend == "sim":
            return 0.0
        return max(0.0, self._confirmed_until - time.time())
