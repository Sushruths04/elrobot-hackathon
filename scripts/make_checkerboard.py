"""Make print-ready checkerboards (A4 landscape PDF+PNG, 300 dpi, exact scale). Print at 100% / 'Actual size' (NOT fit to page).
    .venv/bin/python scripts/make_checkerboard.py            -> calibration/checkerboard_*.pdf
Inner corners = (cols, rows). A 10x7-square board has 9x6 inner corners."""
import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT / "calibration"; OUT.mkdir(exist_ok=True)
DPI = 300; MM = DPI / 25.4; W, H = round(297 * MM), round(210 * MM)

def _font(pt):
    for f in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"):
        try: return ImageFont.truetype(f, round(pt * DPI / 72))
        except Exception: pass
    return ImageFont.load_default()

def board(sq_mm, sx=10, sy=7, name=""):
    im = Image.new("L", (W, H), 255); d = ImageDraw.Draw(im)
    bw, bh = sx * sq_mm, sy * sq_mm; x0, y0 = (297 - bw) / 2, 10.0
    for j in range(sy):
        for i in range(sx):
            if (i + j) % 2 == 0:
                d.rectangle([round((x0 + i * sq_mm) * MM), round((y0 + j * sq_mm) * MM),
                             round((x0 + (i + 1) * sq_mm) * MM) - 1, round((y0 + (j + 1) * sq_mm) * MM) - 1], fill=0)
    big, mid = _font(20), _font(14)
    top = y0 + bh + 6                                   # band under the board (mm)
    rx, ry = round(x0 * MM), round((top + 14) * MM)
    d.text((rx, round(top * MM)), "100 mm  (= 10 cm)", fill=0, font=big)
    d.line([rx, ry, rx + round(100 * MM), ry], fill=0, width=8)
    for k in range(0, 101, 10):
        h = 22 if k % 50 == 0 else 14
        d.line([rx + round(k * MM), ry - h, rx + round(k * MM), ry + h], fill=0, width=6)
        d.text((rx + round(k * MM) - 14, ry + 28), str(k // 10), fill=0, font=mid)
    tx = rx + round(118 * MM)
    d.text((tx, round(top * MM)), f"EACH SQUARE = {sq_mm} mm", fill=0, font=big)
    d.text((tx, round((top + 11) * MM)), f"{sx}x{sy} squares ({sx-1}x{sy-1} inner corners)", fill=0, font=mid)
    d.text((tx, round((top + 18) * MM)), "PRINT AT 100% - Actual size", fill=0, font=mid)
    d.text((tx, round((top + 25) * MM)), "Check: ruler on the left = 100 mm", fill=0, font=mid)
    return im

pages = []
# (square mm, squares x, squares y, tag): inner corners must be ODD x EVEN so OpenCV can tell the board's orientation
for sq, sx, sy, tag in ((22, 10, 7, "table_22mm"), (10, 8, 7, "wrist_10mm")):
    im = board(sq, sx, sy, f"CHECKERBOARD {tag}")
    base = OUT / f"checkerboard_{tag}_{sx-1}x{sy-1}"
    im.save(str(base) + ".png", dpi=(DPI, DPI)); pages.append(im.convert("RGB"))
    print("wrote", base.relative_to(ROOT), f".png  inner corners {sx-1}x{sy-1}  square {sq} mm  board {sx*sq}x{sy*sq} mm")
pages[0].save(str(OUT / "PRINT_BOTH_boards_A4.pdf"), save_all=True, append_images=pages[1:], resolution=DPI)
print("wrote calibration/PRINT_BOTH_boards_A4.pdf  (page 1 = table board, page 2 = wrist board)")
