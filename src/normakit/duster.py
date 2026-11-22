"""Detect a dark-grey whiteboard duster and small coloured metal bottle caps on the table (colour/shape, millisecond-fast).  ROI keeps the left-edge clutter out."""
from __future__ import annotations
import cv2, numpy as np
ROI = (185, 200, 400, 340)          # x1, y1, x2, y2 in the 640x480 table-camera picture
def detect(bgr, sat=60):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV); x1, y1, x2, y2 = ROI; res = {"duster": None, "caps": []}
    # duster: dark and unsaturated, big
    m = ((hsv[..., 2] > 75) & (hsv[..., 2] < 150) & (hsv[..., 1] < 45)).astype(np.uint8) * 255;       roi = np.zeros_like(m); roi[y1:y2, x1:x2] = m[y1:y2, x1:x2]; roi = cv2.morphologyEx(roi, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    cs, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE); cs = [c for c in cs if cv2.contourArea(c) > 350]
    if cs:
        c = max(cs, key=cv2.contourArea); (cx, cy), (w, h), ang = cv2.minAreaRect(c)
        if w < h: w, h, ang = h, w, ang + 90
        res["duster"] = {"center": (round(cx, 1), round(cy, 1)), "long": round(w, 1), "short": round(h, 1), "angle": round(ang % 180, 1), "area": int(cv2.contourArea(c))}
    # caps: saturated, round, small (pink/orange/blue); skip the red brick (rectangular, hue ~178, big)
    s = ((hsv[..., 1] > sat) & (hsv[..., 2] > 70)).astype(np.uint8) * 255; roi = np.zeros_like(s); roi[y1:y2, x1:x2] = s[y1:y2, x1:x2]; roi = cv2.morphologyEx(roi, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    cs, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cs:
        a = cv2.contourArea(c)
        if not 50 < a < 700: continue                  # metal caps are shiny: only part of the cap passes the colour test, so small blobs count
        per = cv2.arcLength(c, True); circ = 4 * np.pi * a / max(per * per, 1e-6)
        if circ < 0.1: continue
        (cx, cy), r = cv2.minEnclosingCircle(c); col = hsv[int(cy), int(cx)]
        res["caps"].append({"center": (round(cx, 1), round(cy, 1)), "diameter_px": round(2 * r, 1), "hue": int(col[0]), "area": int(a), "circularity": round(circ, 2)})
    return res
def draw(bgr, det):
    out = bgr.copy()
    if det["duster"]:
        d = det["duster"]; cv2.polylines(out, [cv2.boxPoints(((d["center"][0], d["center"][1]), (d["long"], d["short"]), d["angle"])).astype(np.int32)], True, (0, 255, 0), 2)
    for c in det["caps"]: cv2.circle(out, (int(c["center"][0]), int(c["center"][1])), int(c["diameter_px"] / 2) + 3, (255, 0, 255), 2)
    return out
