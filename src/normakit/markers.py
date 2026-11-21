"""ArUco markers on the table: drop zone, table corners (pixel -> table homography), labelled zones.

  from normakit.markers import MarkerDetector
  md = MarkerDetector()                 # auto-detects the dictionary on the first frame that has markers
  ms = md.detect(rgb)                   # [{"id": 3, "center_px": [...], "corners_px": [[...]*4], "side_px": 41.2}, ...]
  md.dictionary                         # e.g. "DICT_4X4_50" once locked

Roles (which ID means what) live in calibration/markers.json, e.g.
  {"dictionary": "DICT_4X4_50", "corners": {"0": [0, 0], "1": [0.40, 0], "2": [0.40, 0.30], "3": [0, 0.30]},
   "drop": 10, "zones": {"A": 11, "B": 12}}
corners = marker ID -> its centre in table metres (T_table frame), used for the homography.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

ROLES_FILE = Path(__file__).resolve().parents[2] / "calibration" / "markers.json"
# most common first; 4x4_50 is what scripts/make_aruco.py prints
CANDIDATES = ["DICT_4X4_50", "DICT_4X4_100", "DICT_5X5_100", "DICT_6X6_250", "DICT_7X7_250", "DICT_ARUCO_ORIGINAL",
              "DICT_APRILTAG_36h11", "DICT_APRILTAG_25h9", "DICT_APRILTAG_16h5"]


def _detector(name):
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name))
    return cv2.aruco.ArucoDetector(d, cv2.aruco.DetectorParameters())


class MarkerDetector:
    def __init__(self, dictionary: str | None = None):
        self.roles = json.loads(ROLES_FILE.read_text()) if ROLES_FILE.exists() else {}
        self.dictionary = dictionary or self.roles.get("dictionary")
        self._det = _detector(self.dictionary) if self.dictionary else None

    def _run(self, det, gray):
        corners, ids, _ = det.detectMarkers(gray)
        out = []
        for c, i in zip(corners, ids.flatten() if ids is not None else []):
            c = c.reshape(4, 2)
            side = float(np.mean([np.linalg.norm(c[k] - c[(k + 1) % 4]) for k in range(4)]))
            out.append({"id": int(i), "center_px": c.mean(0).round(1).tolist(), "corners_px": c.round(1).tolist(),
                        "side_px": round(side, 1)})
        return out

    def detect(self, rgb: np.ndarray) -> list[dict]:
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        if self._det:
            return self._run(self._det, gray)
        best = (None, [])  # auto: try each dictionary, keep the one that finds most markers, then lock it
        for name in CANDIDATES:
            found = self._run(_detector(name), gray)
            if len(found) > len(best[1]):
                best = (name, found)
        if best[0]:
            self.dictionary, self._det = best[0], _detector(best[0])
        return best[1]

    # -------- roles
    def role(self, marker_id: int) -> str:
        r = self.roles
        if r.get("drop") == marker_id:
            return "DROP"
        if str(marker_id) in r.get("corners", {}):
            return f"corner{marker_id}"
        for name, mid in r.get("zones", {}).items():
            if mid == marker_id:
                return f"zone {name}"
        return ""

    def homography(self, markers: list[dict]):
        """H (3x3) mapping pixel -> table metres from >= 4 corner markers, or None."""
        table = self.roles.get("corners", {})
        pts = [(m["center_px"], table[str(m["id"])]) for m in markers if str(m["id"]) in table]
        if len(pts) < 4:
            return None
        H, _ = cv2.findHomography(np.float32([p for p, _ in pts]), np.float32([t for _, t in pts]))
        return H


def px_to_table(H, px) -> list[float]:
    v = H @ np.array([px[0], px[1], 1.0])
    return (v[:2] / v[2]).round(4).tolist()


def draw(img_bgr: np.ndarray, markers: list[dict], md: MarkerDetector | None = None) -> np.ndarray:
    for m in markers:
        c = np.int32(m["corners_px"])
        cv2.polylines(img_bgr, [c], True, (255, 0, 255), 3)
        label = f"ID {m['id']}" + (f" {md.role(m['id'])}" if md and md.role(m["id"]) else "")
        x, y = map(int, m["center_px"])
        cv2.putText(img_bgr, label, (x - 30, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 5)
        cv2.putText(img_bgr, label, (x - 30, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 255), 2)
    return img_bgr
