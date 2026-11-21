"""Printable ArUco markers from OpenCV's OWN generator (cv2.aruco, DICT_4X4_50). Nothing is drawn by hand: the bit pattern of every
marker comes from cv2.aruco.generateImageMarker; this script only places the modules at exact millimetre sizes on A4.
    .venv/bin/python scripts/make_aruco.py   ->  calibration/ARUCO_markers_A4.pdf (+ PNG pages)
Page 1: IDs 0-3, marker side 60 mm (table corners / robot base).  Page 2: IDs 10-15, marker side 36 mm (wrist / objects)."""
from pathlib import Path
import cv2, numpy as np
from PIL import Image, ImageDraw, ImageFont
ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT / "calibration"
DPI = 300; MM = DPI / 25.4; W, H = round(297 * MM), round(210 * MM)
DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
def font(pt):
    for f in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"):
        try: return ImageFont.truetype(f, round(pt * DPI / 72))
        except Exception: pass
    return ImageFont.load_default()
big, mid = font(20), font(14)

def draw_marker(d, mid_id, x_mm, y_mm, module_mm):
    bits = cv2.aruco.generateImageMarker(DICT, mid_id, 6)           # 6x6 modules incl. the 1-module black border (OpenCV's official pattern)
    for r in range(6):
        for c in range(6):
            if bits[r, c] == 0:
                d.rectangle([round((x_mm + c * module_mm) * MM), round((y_mm + r * module_mm) * MM),
                             round((x_mm + (c + 1) * module_mm) * MM) - 1, round((y_mm + (r + 1) * module_mm) * MM) - 1], fill=0)

def ruler(d, x_mm, y_mm):
    rx, ry = round(x_mm * MM), round(y_mm * MM)
    d.text((rx, ry - round(13 * MM)), "100 mm  (= 10 cm)", fill=0, font=big)
    d.line([rx, ry, rx + round(100 * MM), ry], fill=0, width=8)
    for k in range(0, 101, 10):
        h = 22 if k % 50 == 0 else 14
        d.line([rx + round(k * MM), ry - h, rx + round(k * MM), ry + h], fill=0, width=6)
        d.text((rx + round(k * MM) - 14, ry + 28), str(k // 10), fill=0, font=mid)

def page(ids, module_mm, cols, rows, title):
    im = Image.new("L", (W, H), 255); d = ImageDraw.Draw(im); side = 6 * module_mm
    gx, gy = (297 - 2 * 14 - cols * side) / max(cols - 1, 1), 8
    left, top = 14, 10
    if cols == 2: gx = 60
    for k, i in enumerate(ids):
        cx, cy = k % cols, k // cols
        x = left + 40 + cx * (side + gx) if cols == 2 else left + cx * (side + gx)
        y = top + cy * (side + gy + 12)
        draw_marker(d, i, x, y, module_mm)
        d.text((round(x * MM), round((y + side + 2) * MM)), f"ID {i}", fill=0, font=big)
    ty = 165 if cols == 2 else 150
    ruler(d, 14, ty + 24)
    tx = round((14 + 118) * MM)
    d.text((tx, round(ty * MM)), f"MARKER SIDE = {side:g} mm (black square)", fill=0, font=big)
    d.text((tx, round((ty + 10) * MM)), f"ArUco dictionary DICT_4X4_50, {title}", fill=0, font=mid)
    d.text((tx, round((ty + 17) * MM)), "PRINT AT 100% - Actual size. Keep the white border.", fill=0, font=mid)
    d.text((tx, round((ty + 24) * MM)), "Check: ruler on the left = 100 mm", fill=0, font=mid)
    return im

pages = [page([0, 1, 2, 3], 10, 2, 2, "IDs 0-3"), page([10, 11, 12, 13, 14, 15], 6, 3, 2, "IDs 10-15")]
for k, p in enumerate(pages, 1): p.save(OUT / f"aruco_page{k}.png", dpi=(DPI, DPI))
pages[0].convert("RGB").save(OUT / "ARUCO_markers_A4.pdf", save_all=True, append_images=[p.convert("RGB") for p in pages[1:]], resolution=DPI)
print("wrote calibration/ARUCO_markers_A4.pdf + aruco_page1.png, aruco_page2.png")
