"""Core types, safety guard and the Backend interface.

Every backend (real Station arm, pybullet digital twin) exposes the same tiny API,
so the MCP server / agent / skills never care which one is behind them.

Joint convention: joint i (1..8) <-> URDF `rev_motor_0i`. Positions are exposed as
`norm` in [0,1] over the *calibrated* servo range (Station's range_min..range_max).
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict

JOINT_IDS = [1, 2, 3, 4, 5, 6, 7, 8]  # 8 = gripper
GRIPPER_ID = 8

# URDF limits (rad) from hardware/elrobot/simulation/elrobot_follower.urdf
URDF_LIMITS = {
    1: (-1.5509, 1.5509), 2: (-1.6122, 1.6122), 3: (-1.761, 1.761), 4: (-1.7533, 1.7533),
    5: (-2.62, 3.252), 6: (-1.3775, 1.7641), 7: (-3.2014, 2.7336), 8: (0.0, 2.2028),
}


@dataclass
class JointState:
    id: int
    norm: float            # 0..1 over calibrated range
    goal_norm: float
    current_ma: int = 0
    temp_c: int = 0
    raw: int = 0
    lo: int = 0
    hi: int = 4095

    def rad(self) -> float:
        lo, hi = URDF_LIMITS[self.id]
        return lo + self.norm * (hi - lo)


@dataclass
class ArmState:
    joints: dict[int, JointState]
    t: float = field(default_factory=time.time)

    def norms(self) -> dict[int, float]:
        return {i: j.norm for i, j in self.joints.items()}

    def to_json(self) -> dict:
        return {
            "t": self.t,
            "joints": {
                str(i): {"norm": round(j.norm, 3), "goal": round(j.goal_norm, 3),
                         "current_ma": j.current_ma, "temp_c": j.temp_c}
                for i, j in sorted(self.joints.items())
            },
        }


def norm_to_rad(jid: int, norm: float) -> float:
    lo, hi = URDF_LIMITS[jid]
    return lo + norm * (hi - lo)


def rad_to_norm(jid: int, rad: float) -> float:
    lo, hi = URDF_LIMITS[jid]
    return (rad - lo) / (hi - lo)


class Backend(ABC):
    """Hardware (or sim) abstraction. All methods are synchronous & thread-safe."""

    name = "abstract"

    @abstractmethod
    def start(self) -> None: ...

    @abstractmethod
    def stop_backend(self) -> None: ...

    @abstractmethod
    def state(self) -> ArmState: ...

    @abstractmethod
    def cameras(self) -> list[str]: ...

    @abstractmethod
    def frame_jpeg(self, camera: str | None = None) -> bytes: ...

    @abstractmethod
    def send_goals(self, goals_norm: dict[int, float], speed: float = 0.5) -> None:
        """Command target positions (normalised). speed in (0,1]."""

    @abstractmethod
    def halt(self) -> None:
        """Hold current position (set goal := present)."""


@dataclass
class SafetyConfig:
    max_step_norm: float = 0.35      # largest single-call joint change (normalised)
    current_limit_ma: int = 900      # abort/back-off above this (gripper excluded when grasping)
    min_norm: float = 0.02           # stay off hard stops
    max_norm: float = 0.98
    max_cmd_hz: float = 10.0
    forbidden_boxes: list[tuple] = field(default_factory=list)  # (xmin,ymin,zmin,xmax,ymax,zmax) in base frame


class Guard:
    """Wraps a Backend: clamps, rate-limits and refuses unsafe commands."""

    def __init__(self, backend: Backend, cfg: SafetyConfig | None = None):
        self.b = backend
        self.cfg = cfg or SafetyConfig()
        self._last = 0.0
        self.estopped = False
        self.log: list[str] = []

    def estop(self):
        self.estopped = True
        self.b.halt()
        self.log.append("ESTOP")

    def release(self):
        self.estopped = False

    def move(self, goals: dict[int, float], speed: float = 0.5) -> dict:
        if self.estopped:
            return {"ok": False, "error": "e-stop active; call release first"}
        st = self.b.state()
        out, notes = {}, []
        for jid, g in goals.items():
            jid = int(jid)
            if jid not in JOINT_IDS:
                return {"ok": False, "error": f"unknown joint {jid}"}
            g2 = min(self.cfg.max_norm, max(self.cfg.min_norm, float(g)))
            if g2 != g:
                notes.append(f"joint {jid} clamped {g:.2f}->{g2:.2f}")
            cur = st.joints[jid].norm
            if jid != GRIPPER_ID and abs(g2 - cur) > self.cfg.max_step_norm:
                g3 = cur + self.cfg.max_step_norm * (1 if g2 > cur else -1)
                notes.append(f"joint {jid} step-limited {g2:.2f}->{g3:.2f} (call again to continue)")
                g2 = g3
            out[jid] = g2
        hot = [j.id for j in st.joints.values()
               if j.id != GRIPPER_ID and j.current_ma > self.cfg.current_limit_ma]
        if hot:
            self.estop()
            return {"ok": False, "error": f"overcurrent on joints {hot}; e-stopped"}
        wait = 1.0 / self.cfg.max_cmd_hz - (time.time() - self._last)
        if wait > 0:
            time.sleep(wait)
        self._last = time.time()
        self.b.send_goals(out, speed=max(0.05, min(1.0, speed)))
        return {"ok": True, "sent": {k: round(v, 3) for k, v in out.items()}, "notes": notes}

    def settle(self, timeout: float = 4.0, tol: float = 0.02) -> ArmState:
        """Block until joints reach goals (or timeout)."""
        t0 = time.time()
        while time.time() - t0 < timeout:
            st = self.b.state()
            if all(abs(j.norm - j.goal_norm) < tol for j in st.joints.values()):
                return st
            time.sleep(0.05)
        return self.b.state()
