"""High-level arm controller used by the MCP server, the agent loop and the CLI.

Skills are deterministic and verified from *sensors* (joint error, motor current), so the
LLM only does perception + task planning. That is what makes it fast and reliable.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from .core import ArmState, Backend, Guard, GRIPPER_ID, SafetyConfig, norm_to_rad
from .kin import Kin
from . import vision

HOME = {1: 0.5, 2: 0.5, 3: 0.5, 4: 0.5, 5: 0.5, 6: 0.5, 7: 0.5, 8: 0.5}
TABLE_Z = 0.02          # gripper-centre height when touching the table plane (metres, base frame) - CALIBRATE
POST_CLOSE_WAIT_S = 2.0
HOVER = 0.08            # extra height above target when travelling


class Arm:
    def __init__(self, backend: Backend, cfg: SafetyConfig | None = None, calib_path: str | Path = "run/table_cam_H.json"):
        self.backend = backend
        self.guard = Guard(backend, cfg)
        self.kin = Kin()
        self.calib_path = Path(calib_path)
        self.H = vision.Homography.load(self.calib_path) if self.calib_path.exists() else vision.Homography()
        self.table_cam = "table"
        self.grid_cells: dict = {}
        self.grasped = False

    # ---------------- perception ----------------
    def look(self, camera: str | None = None, grid: bool = False) -> bytes:
        jpg = self.backend.frame_jpeg(camera or self.table_cam)
        if grid:
            jpg, self.grid_cells = vision.grid_overlay(jpg)
        return jpg

    def state(self) -> ArmState:
        return self.backend.state()

    # ---------------- motion ----------------
    def move_joints(self, goals: dict[int, float], speed=0.5, wait=True) -> dict:
        r = self.guard.move(goals, speed)
        if r.get("ok") and wait:
            self.guard.settle()
        return r

    def home(self, speed=0.4):
        return self.goto_norm({i: v for i, v in HOME.items() if i != GRIPPER_ID}, speed)

    def goto_norm(self, goals: dict[int, float], speed=0.5, timeout=8.0) -> dict:
        """Move to goals in guard-sized steps until reached."""
        t0 = time.time()
        while time.time() - t0 < timeout:
            cur = self.backend.state().norms()
            if all(abs(cur[i] - g) < 0.03 for i, g in goals.items()):
                return {"ok": True}
            r = self.guard.move(goals, speed)
            if not r.get("ok"):
                return r
            self.guard.settle(timeout=2.0)
        return {"ok": False, "error": "timeout reaching goal"}

    def reach_xyz(self, xyz, speed=0.5) -> dict:
        cur = self.backend.state().norms()
        q, err = self.kin.ik_norm(xyz, cur)
        if err > 0.02:
            return {"ok": False, "error": f"unreachable (IK error {err * 100:.1f} cm)", "ik_err_m": err}
        r = self.goto_norm(q, speed)
        r["ik_err_m"] = err
        return r

    def reach_pixel(self, u: float, v: float, z: float = TABLE_Z, speed=0.5) -> dict:
        if self.H.H is None:
            return {"ok": False, "error": "table camera not calibrated; run calibrate_table first"}
        x, y = self.H(u, v)
        r = self.reach_xyz((x, y, z + HOVER), speed)
        if not r.get("ok"):
            return r
        return self.reach_xyz((x, y, z), speed * 0.6) | {"xy": (round(x, 3), round(y, 3))}

    def reach_cell(self, cell: str, **kw):
        if cell not in self.grid_cells:
            return {"ok": False, "error": f"unknown cell {cell}; call look(grid=true) first"}
        return self.reach_pixel(*self.grid_cells[cell], **kw)

    # ---------------- gripper & grasp verification ----------------
    def open_gripper(self):
        self.grasped = False
        return self.move_joints({GRIPPER_ID: 0.95}, speed=0.8)

    def _wait_gripper_steady(self, max_s: float = 6.0, tol: float = 0.003):
        hist, t0 = [], time.time()
        while time.time() - t0 < max_s:
            hist.append(self.backend.state().joints[GRIPPER_ID].norm); time.sleep(0.1)
            if len(hist) >= 6 and max(hist[-6:]) - min(hist[-6:]) < tol:
                return True
        return False

    def close_gripper(self) -> dict:
        """Close and verify: if jaws stall before fully closed AND current rises => object held."""
        self.guard.move({GRIPPER_ID: 0.05}, 0.5)
        self._wait_gripper_steady()
        time.sleep(POST_CLOSE_WAIT_S)                # close fully, wait for the jaw to stop, +2 s, only then lift
        s = self.backend.state().joints[GRIPPER_ID]
        stalled = s.norm > 0.12          # jaws could not close fully
        pushing = s.current_ma > 150 or self.backend.name == "sim"
        self.grasped = stalled and pushing
        return {"ok": True, "grasped": self.grasped, "gripper_norm": round(s.norm, 3), "current_ma": s.current_ma}

    def pick(self, u: float, v: float) -> dict:
        self.open_gripper()
        r = self.reach_pixel(u, v)
        if not r.get("ok"):
            return r
        g = self.close_gripper()
        self.reach_xyz(tuple(np.array(self.kin.fk({i: norm_to_rad(i, x) for i, x in self.backend.state().norms().items()})) + [0, 0, HOVER + 0.04]))
        return {"ok": True, **g}

    def place(self, u: float, v: float) -> dict:
        r = self.reach_pixel(u, v, z=TABLE_Z + 0.03)
        if not r.get("ok"):
            return r
        self.open_gripper()
        return {"ok": True}

    # ---------------- self-calibration (hand-eye) ----------------
    def calibrate_table(self, marker_color: str = "magenta", n: int = 12, z: float = TABLE_Z + 0.02) -> dict:
        """Move the gripper to known XY points; find the marker in the table camera; fit pixel->XY.
        Needs a bright sticker (default magenta) on the gripper. Takes ~1 minute."""
        pts_px, pts_xy = [], []
        rng = np.random.default_rng(1)
        tries = 0
        while len(pts_px) < n and tries < n * 3:
            tries += 1
            x, y = rng.uniform(-0.12, 0.12), rng.uniform(0.17, 0.30)
            r = self.reach_xyz((x, y, z), speed=0.6)
            if not r.get("ok"):
                continue
            time.sleep(0.4)
            fk = self.kin.fk({i: norm_to_rad(i, v) for i, v in self.backend.state().norms().items()})
            px = vision.find_named(vision.decode(self.backend.frame_jpeg(self.table_cam)), marker_color)
            if px:
                pts_px.append(px)
                pts_xy.append((float(fk[0]), float(fk[1])))
        if len(pts_px) < 6:
            return {"ok": False, "error": f"marker seen at only {len(pts_px)} points"}
        err = self.H.fit(pts_px, pts_xy)
        self.calib_path.parent.mkdir(parents=True, exist_ok=True)
        self.H.save(self.calib_path)
        self.home()
        return {"ok": True, "points": len(pts_px), "mean_err_cm": round(err * 100, 2)}
