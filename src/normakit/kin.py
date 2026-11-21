"""Kinematics for ElRobot from its official URDF (pybullet, DIRECT mode, no GUI).

fk(q_rad)->xyz and ik(xyz)->q. This is the 'action decoder': the LLM never emits joint
angles for reaching - it says *where* (a pixel or a point) and IK does the rest.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pybullet as p

from .core import URDF_LIMITS, norm_to_rad, rad_to_norm

URDF = Path(__file__).resolve().parents[2] / "vendor/norma-core-src/hardware/elrobot/simulation/elrobot_follower.urdf"
ACT = {1: 1, 2: 3, 3: 5, 4: 7, 5: 9, 6: 11, 7: 13, 8: 15}   # motor id -> pybullet joint index
EE_LINK = 13


class Kin:
    def __init__(self):
        self.cid = p.connect(p.DIRECT)
        self.rid = p.loadURDF(str(URDF), useFixedBase=True, physicsClientId=self.cid)
        self.lo = [URDF_LIMITS[i][0] for i in range(1, 7)]
        self.hi = [URDF_LIMITS[i][1] for i in range(1, 7)]

    def _set(self, q: dict[int, float]):
        for jid, r in q.items():
            p.resetJointState(self.rid, ACT[jid], r, physicsClientId=self.cid)

    def fk(self, q_rad: dict[int, float]) -> np.ndarray:
        self._set(q_rad)
        return np.array(p.getLinkState(self.rid, EE_LINK, computeForwardKinematics=True,
                                       physicsClientId=self.cid)[4])

    def ik(self, xyz, seed_rad: dict[int, float] | None = None, iters: int = 200):
        """Position IK on joints 1-6. Returns (q_rad dict, error_m)."""
        if seed_rad:
            self._set(seed_rad)
        rng = np.arange(1, 7)
        sol = p.calculateInverseKinematics(
            self.rid, EE_LINK, list(xyz),
            lowerLimits=self.lo, upperLimits=self.hi,
            jointRanges=[h - l for l, h in zip(self.lo, self.hi)],
            restPoses=[(seed_rad or {}).get(int(i), 0.0) for i in rng],
            maxNumIterations=iters, residualThreshold=1e-4, physicsClientId=self.cid)
        # sol is ordered by non-fixed joints: motors 1..6 first
        q = {int(i): float(np.clip(sol[k], self.lo[k], self.hi[k])) for k, i in enumerate(rng)}
        err = float(np.linalg.norm(self.fk(q) - np.array(xyz)))
        return q, err

    def ik_norm(self, xyz, seed_norm: dict[int, float] | None = None):
        seed = {i: norm_to_rad(i, v) for i, v in (seed_norm or {}).items() if i <= 6}
        q, err = self.ik(xyz, seed)
        return {i: rad_to_norm(i, r) for i, r in q.items()}, err
