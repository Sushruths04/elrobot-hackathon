"""Digital-twin backend: the real ElRobot URDF in pybullet with two rendered cameras.

Use it to develop/test the whole stack with ZERO hardware (laptop, on the plane, in the
hotel). Same Backend API as the real arm. Fidelity is for *logic* testing, not for
tuning grasp physics.
"""
from __future__ import annotations

import io
import threading
import time

import numpy as np
import pybullet as p
from PIL import Image

from .core import (ArmState, Backend, JointState, URDF_LIMITS, GRIPPER_ID, norm_to_rad, rad_to_norm)
from .kin import ACT, URDF, EE_LINK

W, H = 640, 480


class SimBackend(Backend):
    name = "sim"

    def __init__(self, objects: bool = True, realtime: bool = True):
        self.cid = p.connect(p.DIRECT)
        p.setGravity(0, 0, -9.81, physicsClientId=self.cid)
        p.setTimeStep(1 / 240, physicsClientId=self.cid)
        self.plane = p.createMultiBody(
            0, p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.6, 0.6, 0.01], physicsClientId=self.cid),
            p.createVisualShape(p.GEOM_BOX, halfExtents=[0.6, 0.6, 0.01], rgbaColor=[.85, .8, .7, 1],
                                physicsClientId=self.cid),
            basePosition=[0, 0.2, -0.011], physicsClientId=self.cid)
        self.rid = p.loadURDF(str(URDF), useFixedBase=True, physicsClientId=self.cid)
        self.goals = {i: 0.5 for i in range(1, 9)}
        self.speed = 0.5
        self.objs: dict[str, int] = {}
        if objects:
            self.add_box("red_block", [0.12, 0.22, 0.02], [0.02, 0.02, 0.02], [0.9, 0.1, 0.1, 1])
            self.add_box("green_block", [-0.10, 0.20, 0.02], [0.02, 0.02, 0.02], [0.1, 0.8, 0.2, 1])
            self.add_box("blue_tray", [0.0, 0.32, 0.005], [0.07, 0.07, 0.005], [0.15, 0.25, 0.9, 1], mass=0)
        # magenta sticker on the gripper (like the one you stick on the real arm for hand-eye calibration)
        self.marker = p.createMultiBody(
            0, -1, p.createVisualShape(p.GEOM_SPHERE, radius=0.012, rgbaColor=[1, 0, 1, 1], physicsClientId=self.cid),
            basePosition=[0, 0, 1], physicsClientId=self.cid)
        self._lock = threading.Lock()
        self._run = False
        self._thr = None
        self.realtime = realtime
        self._set_all_rad({i: norm_to_rad(i, 0.5) for i in range(1, 9)})

    # --- scene helpers (sim only) ------------------------------------------------
    def add_box(self, name, pos, half, rgba, mass=0.05):
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half, physicsClientId=self.cid)
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half, rgbaColor=rgba, physicsClientId=self.cid)
        self.objs[name] = p.createMultiBody(mass, col, vis, basePosition=pos, physicsClientId=self.cid)

    def object_pos(self, name):
        return np.array(p.getBasePositionAndOrientation(self.objs[name], physicsClientId=self.cid)[0])

    def _set_all_rad(self, q):
        for jid, r in q.items():
            p.resetJointState(self.rid, ACT[jid], r, physicsClientId=self.cid)

    # --- Backend API -----------------------------------------------------------
    def start(self):
        self._run = True
        self._thr = threading.Thread(target=self._loop, daemon=True)
        self._thr.start()

    def stop_backend(self):
        self._run = False

    def _loop(self):
        while self._run:
            with self._lock:
                for jid in range(1, 9):
                    p.setJointMotorControl2(
                        self.rid, ACT[jid], p.POSITION_CONTROL,
                        targetPosition=norm_to_rad(jid, self.goals[jid]),
                        maxVelocity=0.5 + 5.0 * self.speed, force=40.0, physicsClientId=self.cid)
                # gripper jaws follow motor 8
                g = norm_to_rad(8, self.goals[8]) / 2.2028 * 0.0255
                p.setJointMotorControl2(self.rid, 16, p.POSITION_CONTROL, targetPosition=-g, force=5, physicsClientId=self.cid)
                p.setJointMotorControl2(self.rid, 17, p.POSITION_CONTROL, targetPosition=g, force=5, physicsClientId=self.cid)
                for _ in range(4):
                    p.stepSimulation(physicsClientId=self.cid)
                ee = p.getLinkState(self.rid, EE_LINK, computeForwardKinematics=True, physicsClientId=self.cid)[4]
                p.resetBasePositionAndOrientation(self.marker, [ee[0], ee[1], ee[2] + 0.05], [0, 0, 0, 1],
                                                  physicsClientId=self.cid)
            if self.realtime:
                time.sleep(4 / 240)

    def state(self) -> ArmState:
        js = {}
        with self._lock:
            for jid in range(1, 9):
                r = p.getJointState(self.rid, ACT[jid], physicsClientId=self.cid)
                js[jid] = JointState(
                    id=jid, norm=rad_to_norm(jid, r[0]), goal_norm=self.goals[jid],
                    current_ma=int(min(1500, abs(self.goals[jid] - rad_to_norm(jid, r[0])) * 600)), temp_c=30, raw=int(rad_to_norm(jid, r[0]) * 4095),
                    lo=0, hi=4095)
        return ArmState(js)

    def cameras(self):
        return ["table", "wrist"]

    def frame_jpeg(self, camera=None) -> bytes:
        camera = camera or "table"
        with self._lock:
            if camera == "wrist":
                ls = p.getLinkState(self.rid, EE_LINK, computeForwardKinematics=True, physicsClientId=self.cid)
                pos, orn = ls[4], ls[5]
                rot = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
                eye = np.array(pos) + rot @ np.array([0, 0, 0.0])
                tgt = eye + rot @ np.array([0, 0, 0.2])
                up = rot @ np.array([0, 1, 0])
                view = p.computeViewMatrix(eye.tolist(), tgt.tolist(), up.tolist())
            else:
                view = p.computeViewMatrix([0.0, 0.22, 0.75], [0.0, 0.22, 0.0], [0, 1, 0])
            proj = p.computeProjectionMatrixFOV(60, W / H, 0.02, 3.0)
            _, _, rgb, _, _ = p.getCameraImage(W, H, view, proj, renderer=p.ER_TINY_RENDERER, physicsClientId=self.cid)
        img = Image.fromarray(np.reshape(rgb, (H, W, 4))[:, :, :3].astype(np.uint8))
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=88)
        return buf.getvalue()

    def send_goals(self, goals_norm, speed=0.5):
        with self._lock:
            self.goals.update({int(k): float(v) for k, v in goals_norm.items()})
            self.speed = speed

    def halt(self):
        s = self.state()
        with self._lock:
            for j in s.joints.values():
                self.goals[j.id] = j.norm
