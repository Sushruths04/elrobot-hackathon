"""Colour detection of toy bricks (Duplo-size) on the table: centre, long-axis angle, size in pixels. Millisecond-fast, no model calls.
red: S>100 and hue near 0/179;  orange/yellow: 7<=hue<=24, S>70.  ROI keeps the left-edge clutter (pens, bottle caps) out."""
from __future__ import annotations
import cv2, numpy as np
ROI = (200, 190, 620, 400)          # x1, y1, x2, y2 in the 640x480 table camera picture
def _mask(hsv, kind):
    if kind == "red": m = (((hsv[..., 0] >= 165) | (hsv[..., 0] <= 5)) & (hsv[..., 1] > 100) & (hsv[..., 2] > 90))
    else: m = ((hsv[..., 0] >= 7) & (hsv[..., 0] <= 24) & (hsv[..., 1] > 70) & (hsv[..., 2] > 120))
    out = np.zeros(hsv.shape[:2], np.uint8); x1, y1, x2, y2 = ROI; out[y1:y2, x1:x2] = m[y1:y2, x1:x2].astype(np.uint8) * 255
    return cv2.morphologyEx(out, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
def detect(bgr, min_area=120):
    """-> {'red': [blob,...], 'orange': [blob,...]}; blob = {'center': (u,v), 'long': px, 'short': px, 'angle': deg of the long axis in the picture (0 = to the right, 90 = down), 'area': px}"""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV); res = {}
    for kind in ("red", "orange"):
        cs, _ = cv2.findContours(_mask(hsv, kind), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE); blobs = []
        for c in cs:
            a = cv2.contourArea(c)
            if a < min_area: continue
            (cx, cy), (w, h), ang = cv2.minAreaRect(c)
            if w < h: w, h, ang = h, w, ang + 90
            blobs.append({"center": (round(cx, 1), round(cy, 1)), "long": round(w, 1), "short": round(h, 1), "angle": round(ang % 180, 1), "area": int(a)})
        res[kind] = sorted(blobs, key=lambda b: -b["area"])
    return res
def draw(bgr, det):
    out = bgr.copy()
    for kind, col in (("red", (0, 0, 255)), ("orange", (0, 165, 255))):
        for b in det[kind]:
            rect = ((b["center"][0], b["center"][1]), (b["long"], b["short"]), b["angle"]); cv2.polylines(out, [cv2.boxPoints(rect).astype(np.int32)], True, (0, 255, 0), 2)
            cv2.putText(out, f"{kind} {b['angle']:.0f}d", (int(b["center"][0]) - 20, int(b["center"][1]) - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 2)
    return out
