"""Vision helpers: set-of-marks grid overlay, colour-marker tracking, pixel->table homography.

Why: LLMs are good at *deciding* ('the red block is in cell C4') but bad at emitting precise
pixel/joint numbers. We (1) overlay a labelled grid so the model answers with a cell id and
(2) turn pixels into robot-frame XY with a homography calibrated by the arm itself.
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw


def decode(jpeg: bytes) -> np.ndarray:
    return np.array(Image.open(io.BytesIO(jpeg)).convert("RGB"))


def encode(arr: np.ndarray, q=88) -> bytes:
    b = io.BytesIO()
    Image.fromarray(arr).save(b, "JPEG", quality=q)
    return b.getvalue()


def grid_overlay(jpeg: bytes, cols=8, rows=6) -> tuple[bytes, dict]:
    """Draw an A1..H6 grid. Returns (jpeg, {cell: (cx, cy)}). Model says 'C4', we know the pixel."""
    img = Image.open(io.BytesIO(jpeg)).convert("RGB")
    w, h = img.size
    d = ImageDraw.Draw(img, "RGBA")
    cells = {}
    for c in range(cols):
        for r in range(rows):
            x0, y0 = c * w / cols, r * h / rows
            d.rectangle([x0, y0, x0 + w / cols, y0 + h / rows], outline=(255, 255, 0, 120))
            name = f"{chr(65 + c)}{r + 1}"
            d.text((x0 + 3, y0 + 2), name, fill=(255, 255, 0, 230))
            cells[name] = (x0 + w / cols / 2, y0 + h / rows / 2)
    b = io.BytesIO()
    img.save(b, "JPEG", quality=88)
    return b.getvalue(), cells


def find_color(arr: np.ndarray, lower_hsv, upper_hsv, min_area=20):
    """Centroid (x,y) of the largest blob in an HSV range, or None."""
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, np.array(lower_hsv), np.array(upper_hsv))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < min_area:
        return None
    m = cv2.moments(c)
    return (m["m10"] / m["m00"], m["m01"] / m["m00"])


NAMED_COLORS = {  # HSV (OpenCV: H 0-179)
    "red": [((0, 120, 70), (8, 255, 255)), ((172, 120, 70), (179, 255, 255))],
    "green": [((40, 80, 50), (85, 255, 255))],
    "blue": [((95, 100, 50), (130, 255, 255))],
    "yellow": [((20, 100, 100), (35, 255, 255))],
    "magenta": [((140, 100, 100), (170, 255, 255))],
    "orange": [((9, 120, 100), (19, 255, 255))],
}


def find_named(arr: np.ndarray, name: str, min_area=20):
    best = None
    for lo, hi in NAMED_COLORS[name]:
        r = find_color(arr, lo, hi, min_area)
        if r:
            best = r
            break
    return best


class Homography:
    """pixel (u,v) -> robot-frame table (x,y) metres."""

    def __init__(self, H: np.ndarray | None = None):
        self.H = H

    def fit(self, pixels: list[tuple], xys: list[tuple]):
        H, _ = cv2.findHomography(np.array(pixels, np.float32), np.array(xys, np.float32), cv2.RANSAC, 0.01)
        self.H = H
        pred = [self(*p) for p in pixels]
        err = float(np.mean([np.linalg.norm(np.array(a) - np.array(b)) for a, b in zip(pred, xys)]))
        return err

    def __call__(self, u, v):
        p = self.H @ np.array([u, v, 1.0])
        return (float(p[0] / p[2]), float(p[1] / p[2]))

    def save(self, path):
        Path(path).write_text(json.dumps(self.H.tolist()))

    @classmethod
    def load(cls, path):
        d = json.loads(Path(path).read_text())
        if isinstance(d, dict):                       # calibration/table_<cam>.json from scripts/calibrate_camera.py
            d = d["H_pixel_to_robot_xy_m"]
        return cls(np.array(d))
