#!/usr/bin/env python
"""High-definition local video editor & highlight-reel builder for presentation/.

Upgrades over the earlier draft:
  - Uses clean caption-free camera feeds (`clean` / `*_nocaption.mp4`) so zero debug log bars cover the robot!
  - Upscales to 960x720 (Lanczos4) + subtle unsharp-mask clarity enhancement + mild contrast lift.
  - Encodes at smooth 30 fps, H.264 yuv420p CRF 17 (+faststart) for crisp, artifact-free local playback.
  - Builds a sleek ~105 s Hackathon Highlight Reel featuring ONLY positive achievements, led by Wet-Wipes Extraction & Table Cleaning.
"""
from __future__ import annotations

import argparse
import json
import math
import textwrap
from pathlib import Path

import av
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "presentation"
CAT = json.loads((P / "catalog.json").read_text())
(P / "videos").mkdir(parents=True, exist_ok=True)
(P / "posters").mkdir(parents=True, exist_ok=True)

W, H, FPS = 960, 720, 30
ACCENT = (56, 189, 248)   # RGB cyan-sky
EMERALD = (34, 197, 94)   # RGB success green

_fonts: dict = {}


def font(sz: int, bold: bool = False):
    k = (sz, bold)
    if k not in _fonts:
        _fonts[k] = ImageFont.truetype(
            f"/usr/share/fonts/truetype/dejavu/DejaVuSans{'-Bold' if bold else ''}.ttf", sz
        )
    return _fonts[k]


def enhance_frame(bgr: np.ndarray) -> np.ndarray:
    """Upscale to 960x720 with Lanczos4 + gentle unsharp mask & micro-contrast polish."""
    if bgr.shape[:2] != (H, W):
        bgr = cv2.resize(bgr, (W, H), interpolation=cv2.INTER_LANCZOS4)
    blur = cv2.GaussianBlur(bgr, (0, 0), 1.4)
    sharp = cv2.addWeighted(bgr, 1.22, blur, -0.22, 2.0)
    return sharp


def probe(path: Path) -> tuple[float, int]:
    c = cv2.VideoCapture(str(path))
    fps = c.get(cv2.CAP_PROP_FPS) or 0.0
    n = int(c.get(cv2.CAP_PROP_FRAME_COUNT))
    c.release()
    return fps, n


def joint_motion(jpath: Path):
    rows = [json.loads(line) for line in open(jpath) if line.strip()]
    T = np.array([r["t"] for r in rows], float)
    Q = np.array([[r[str(i)] for i in range(1, 9)] for r in rows], float)
    j = np.clip(np.searchsorted(T, T - 0.3), 0, len(T) - 1)
    d = np.abs(((Q - Q[j] + 2048) % 4096) - 2048).max(1)
    return T, d > 5


def frame_motion(path: Path, fps: float):
    c = cv2.VideoCapture(str(path))
    prev = None
    T, M = [], []
    k = 0
    while True:
        ok, im = c.read()
        if not ok:
            break
        g = cv2.GaussianBlur(cv2.cvtColor(cv2.resize(im, (160, 120)), cv2.COLOR_BGR2GRAY), (5, 5), 0)
        T.append(k / fps)
        M.append(prev is not None and (cv2.absdiff(g, prev) > 16).mean() > 0.003)
        prev = g
        k += 1
    c.release()
    return np.array(T), np.array(M)


def keep_segments(T, moving, dur: float, pre=0.8, post=2.0, pad=0.5, gap=2.8):
    idx = np.where(moving)[0]
    if len(idx) == 0:
        return [[0.0, dur]]
    iv = []
    s = e = T[idx[0]]
    for k in idx[1:]:
        if T[k] - e > 0.5:
            iv.append([s, e])
            s = T[k]
        e = T[k]
    iv.append([s, e])
    segs = [[s - pad, e + pad] for s, e in iv]
    segs[0][0] = iv[0][0] - pre
    segs[-1][1] = iv[-1][1] + post
    out = []
    for s, e in segs:
        s, e = max(0.0, s), min(dur, e)
        if out and s - out[-1][1] < max(gap - 2 * pad, 2.2):
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out if sum(e - s for s, e in out) > 2.5 else [[0.0, dur]]


def nice_speed(d: float, target: float, cap: float) -> float:
    return 1.0 if d <= target else min(cap, math.ceil(d / target * 2) / 2)


_patch_cache: dict = {}


def label_patch(text: str, size=18, bold=True, bg=(12, 17, 28, 205), fg=(248, 250, 252, 255), padx=12, pady=6):
    k = (text, size, bold, bg, fg, padx, pady)
    if k not in _patch_cache:
        f = font(size, bold)
        w = min(W - 24, int(f.getlength(text)) + 2 * padx)
        h = size + 2 * pady + 6
        im = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        d = ImageDraw.Draw(im)
        d.rounded_rectangle((0, 0, w - 1, h - 1), radius=8, fill=bg)
        d.text((padx, pady), text, font=f, fill=fg)
        a = np.array(im)
        _patch_cache[k] = (
            cv2.cvtColor(a[:, :, :3], cv2.COLOR_RGB2BGR).astype(np.float32),
            a[:, :, 3:4].astype(np.float32) / 255.0,
        )
    return _patch_cache[k]


def blend(img: np.ndarray, patch, x: int, y: int) -> None:
    rgb, al = patch
    h, w = al.shape[:2]
    x = max(0, min(max(0, W - w), x))
    y = max(0, min(max(0, H - h), y))
    w = min(w, W - x)
    h = min(h, H - y)
    rgb = rgb[:h, :w]
    al = al[:h, :w]
    roi = img[y : y + h, x : x + w].astype(np.float32)
    img[y : y + h, x : x + w] = (roi * (1.0 - al) + rgb * al).astype(np.uint8)


class Enc:
    def __init__(self, path: Path):
        self.o = av.open(str(path), "w", options={"movflags": "+faststart"})
        self.st = self.o.add_stream("libx264", rate=FPS)
        self.st.width, self.st.height, self.st.pix_fmt = W, H, "yuv420p"
        self.st.options = {"crf": "17", "preset": "fast"}
        self.n = 0

    def put(self, bgr: np.ndarray) -> None:
        if bgr.shape[:2] != (H, W):
            bgr = enhance_frame(bgr)
        for p in self.st.encode(av.VideoFrame.from_ndarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), format="rgb24")):
            self.o.mux(p)
        self.n += 1

    def close(self) -> None:
        for p in self.st.encode():
            self.o.mux(p)
        self.o.close()


def plan_of(segs, speed: float):
    plan = []
    for si, (a, b) in enumerate(segs):
        n_frames = max(1, int(round((b - a) / speed * FPS)))
        for j in range(n_frames):
            plan.append((a + j * speed / FPS, si))
    return plan


def frames(path: Path, fps: float, plan):
    c = cv2.VideoCapture(str(path))
    cur_k, cur = -1, None
    for s, si in plan:
        k = int(round(s * fps))
        while cur_k < k:
            ok, im = c.read()
            if not ok:
                break
            cur, cur_k = im, cur_k + 1
        if cur is not None:
            yield enhance_frame(cur), si
    c.release()


def resolve_clean_src(it: dict) -> Path:
    if it.get("clean") and (ROOT / it["clean"]).exists():
        return ROOT / it["clean"]
    return ROOT / it["src"]


def analyse(it: dict):
    src = resolve_clean_src(it)
    if not src.exists():
        return None
    fps, n = probe(src)
    if fps <= 0 or n <= 0:
        return None
    dur = n / fps
    if it.get("no_trim"):
        segs = [[0.0, dur]]
    elif it.get("joints") and (ROOT / it["joints"]).exists():
        T, M = joint_motion(ROOT / it["joints"])
        segs = keep_segments(T, M, dur)
    else:
        T, M = frame_motion(src, fps)
        segs = keep_segments(T, M, dur)
    kept = sum(e - s for s, e in segs)
    return {"src": src, "fps": fps, "dur": dur, "segs": segs, "kept": kept}


def edit_clip(it: dict, a: dict, force: bool):
    out = P / "videos" / f"{it['id']}.mp4"
    speed = 1.0 if it.get("no_trim") else nice_speed(a["kept"], 28, 4)
    cuts = [round(a["segs"][i + 1][0] - a["segs"][i][1], 1) for i in range(len(a["segs"]) - 1)]
    lead, tail = round(a["segs"][0][0], 1), round(a["dur"] - a["segs"][-1][1], 1)
    info = {
        "src_dur": round(a["dur"], 1),
        "kept": round(a["kept"], 1),
        "speed": speed,
        "out_dur": round(a["kept"] / speed, 1),
        "cuts": cuts,
        "lead_cut": lead,
        "tail_cut": tail,
    }
    if out.exists() and out.stat().st_size > 5000 and not force:
        return info

    plan = plan_of(a["segs"], speed)
    enc = Enc(out)
    for im, _ in frames(a["src"], a["fps"], plan):
        enc.put(im)
    enc.close()

    t = plan[min(len(plan) - 1, int(len(plan) * 0.72))][0]
    c = cv2.VideoCapture(str(a["src"]))
    c.set(cv2.CAP_PROP_POS_FRAMES, int(t * a["fps"]))
    ok, im = c.read()
    c.release()
    if ok:
        cv2.imwrite(str(P / "posters" / f"{it['id']}.jpg"), enhance_frame(im), [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"HD {it['id']}: {a['dur']:.1f}s -> {a['kept'] / speed:.1f}s ({speed:g}x @ 30fps CRF17)", flush=True)
    return info


def wrap_lines(text: str, f, width: int):
    words, lines, cur = text.split(), [], ""
    for w in words:
        t = (cur + " " + w).strip()
        if f.getlength(t) <= width:
            cur = t
        else:
            lines.append(cur)
            cur = w
    return lines + [cur] if cur else lines


def title_card(title: str, sub: str = "", kicker: str = "", small: str = "") -> np.ndarray:
    im = Image.new("RGB", (W, H), (8, 12, 22))
    d = ImageDraw.Draw(im)
    for y in range(H):
        d.line([(0, y), (W, y)], fill=(8 + y // 38, 13 + y // 30, 24 + y // 18))
    d.rounded_rectangle((64, 195, 72, 475), radius=4, fill=ACCENT)
    y = 195
    if kicker:
        d.text((96, y), kicker, font=font(20, True), fill=ACCENT)
        y += 40
    for line in wrap_lines(title, font(42, True), W - 180):
        d.text((96, y), line, font=font(42, True), fill=(248, 250, 252))
        y += 54
    y += 14
    for line in wrap_lines(sub, font(23, False), W - 180):
        d.text((96, y), line, font=font(23, False), fill=(165, 180, 205))
        y += 34
    if small:
        d.text((96, H - 68), small, font=font(17, False), fill=(110, 128, 158))
    return cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)


def put_card(enc: Enc, img: np.ndarray, secs: float) -> None:
    n = int(secs * FPS)
    fade = int(0.35 * FPS)
    for j in range(n):
        alpha = min(1.0, (j + 1) / fade, (n - j) / fade)
        enc.put((img.astype(np.float32) * alpha).astype(np.uint8))


def build_reel(items: list[dict], analyses: dict) -> dict:
    out = P / "videos" / "highlight_reel.mp4"
    enc = Enc(out)
    secs = {s["id"]: s for s in CAT["sections"]}
    order = [s["id"] for s in CAT["sections"]]
    total = []

    intro = title_card(
        "Autonomous Wet-Wipes & Desk Care Robot",
        "Opening flip-top packs, extracting single wipes, sweeping spills, collecting metal caps & scooping beans on an 8-servo arm.",
        "NORMACORE ROBOTICS SHOWCASE",
        "30 fps HD Highlight Reel  |  Real Hardware + Physics Digital Twin",
    )
    cv2.imwrite(str(P / "posters" / "reel.jpg"), intro, [cv2.IMWRITE_JPEG_QUALITY, 92])
    put_card(enc, intro, 3.8)

    sec_no = 0
    cur_sec = None
    for it in sorted(items, key=lambda x: order.index(x["section"])):
        if it["section"] != cur_sec:
            cur_sec = it["section"]
            sec_no += 1
            s = secs[cur_sec]
            put_card(enc, title_card(s["title"], s["line"], f"CHAPTER 0{sec_no}"), 2.4)

        a = analyses[it["id"]]
        speed = max(nice_speed(a["kept"], 28, 4), nice_speed(a["kept"], it.get("reel_max", 14), 6))
        badge = label_patch(it.get("tag", it["verdict"]), 15, True, bg=EMERALD + (235,))
        title_p = label_patch(it["title"], 20, True, bg=(10, 15, 26, 210))
        sp_p = label_patch(f"{speed:g}x", 15, True, bg=(15, 23, 42, 200)) if speed > 1 else None
        plan = plan_of(a["segs"], speed)
        nfade = int(0.25 * FPS)

        for j, (im, _) in enumerate(frames(a["src"], a["fps"], plan)):
            # Subtle bottom gradient bar for crisp broadcast lower-third readability
            grad = im[H - 104 :].astype(np.float32) * 0.52
            im[H - 104 :] = grad.astype(np.uint8)
            blend(im, badge, 24, H - 92)
            blend(im, title_p, 24, H - 52)
            if sp_p:
                blend(im, sp_p, W - 72, 18)
            if j < nfade:
                im = (im.astype(np.float32) * ((j + 1) / nfade)).astype(np.uint8)
            enc.put(im)
        total.append((it["id"], round(len(plan) / FPS, 1)))

    outro = title_card(
        "From Kinesthetic Teaching to Autonomous Desk Hygiene",
        "Wet-wipe extraction, surface sweeping, magnetic pickup, 30 fps LeRobot datasets & 30 Hz RL control — built 100% locally.",
        "ELROBOT AUTONOMOUS PLATFORM",
        "Explore every clip interactively in index.html & slides.html",
    )
    put_card(enc, outro, 3.8)
    enc.close()
    print(f"Highlight reel complete: {enc.n / FPS:.1f}s @ 960x720 30fps -> {out}", flush=True)
    return {"reel_s": round(enc.n / FPS, 1), "parts": total}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--no-reel", action="store_true")
    ap.add_argument("--reel-only", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    only = set(filter(None, a.only.split(",")))
    ej = P / "edits.json"
    edits = json.loads(ej.read_text()) if ej.exists() else {}
    analyses = {}

    for it in CAT["items"]:
        if only and it["id"] not in only and not (a.reel_only and it.get("reel")):
            continue
        if a.reel_only and not it.get("reel"):
            continue
        an = analyse(it)
        if an is None:
            print("SKIP (missing):", it["src"], flush=True)
            continue
        analyses[it["id"]] = an
        if not a.reel_only:
            edits[it["id"]] = edit_clip(it, an, a.force)
            ej.write_text(json.dumps(edits, indent=2))

    if not a.no_reel and not only:
        reel_items = [i for i in CAT["items"] if i.get("reel") and i["id"] in analyses]
        edits["_reel"] = build_reel(reel_items, analyses)
        ej.write_text(json.dumps(edits, indent=2))
