#!/usr/bin/env python
"""Edit the videos listed in presentation/catalog.json. Originals are only read, never changed or deleted.
  presentation/videos/<id>.mp4   trimmed (idle start/end removed, idle gaps > 3 s cut and marked on screen), long clips sped up (marked), H.264 + faststart
  presentation/posters/<id>.jpg  poster frame (caption-free source when there is one)
  presentation/edits.json        what was cut / sped up per clip (the HTML pages show it)
  presentation/videos/highlight_reel.mp4   title cards + the reel items in catalogue order
Idle = no joint moved more than 5 ticks within 0.3 s (films: *_joints.jsonl); other videos use camera frame differences.
Usage: scripts/presentation_edit.py [--only id1,id2] [--no-reel] [--reel-only] [--force]"""
import argparse, json, math, textwrap
from pathlib import Path
import av, cv2, numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]; P = ROOT / "presentation"; CAT = json.loads((P / "catalog.json").read_text())
(P / "videos").mkdir(exist_ok=True); (P / "posters").mkdir(exist_ok=True)
W, H, FPS = 640, 480, 20
COL = {"success": (46, 160, 67), "partial": (210, 150, 20), "failure": (200, 50, 50), "info": (60, 110, 200)}   # RGB
_fonts = {}
def font(sz, bold=False):
    k = (sz, bold)
    if k not in _fonts: _fonts[k] = ImageFont.truetype(f"/usr/share/fonts/truetype/dejavu/DejaVuSans{'-Bold' if bold else ''}.ttf", sz)
    return _fonts[k]

def probe(path):
    c = cv2.VideoCapture(str(path)); return (c.get(cv2.CAP_PROP_FPS) or 0), int(c.get(cv2.CAP_PROP_FRAME_COUNT))

def joint_motion(jpath):
    rows = [json.loads(l) for l in open(jpath) if l.strip()]
    T = np.array([r["t"] for r in rows], float); Q = np.array([[r[str(i)] for i in range(1, 9)] for r in rows], float)
    j = np.clip(np.searchsorted(T, T - 0.3), 0, len(T) - 1); d = np.abs(((Q - Q[j] + 2048) % 4096) - 2048).max(1)
    return T, d > 5

def frame_motion(path, fps):
    c = cv2.VideoCapture(str(path)); prev = None; T, M = [], []; k = 0
    while True:
        ok, im = c.read()
        if not ok: break
        g = cv2.GaussianBlur(cv2.cvtColor(cv2.resize(im, (160, 120)), cv2.COLOR_BGR2GRAY), (5, 5), 0)
        T.append(k / fps); M.append(prev is not None and (cv2.absdiff(g, prev) > 18).mean() > 0.004); prev = g; k += 1
    return np.array(T), np.array(M)

def keep_segments(T, moving, dur, pre=1.0, post=2.5, pad=0.6, gap=3.0):
    """Motion intervals -> kept segments; gaps of idle longer than `gap` s are cut (pad s of stillness kept on each side)."""
    idx = np.where(moving)[0]
    if len(idx) == 0: return [[0.0, dur]]
    iv = []; s = e = T[idx[0]]
    for k in idx[1:]:
        if T[k] - e > 0.5: iv.append([s, e]); s = T[k]
        e = T[k]
    iv.append([s, e]); segs = [[s - pad, e + pad] for s, e in iv]; segs[0][0] = iv[0][0] - pre; segs[-1][1] = iv[-1][1] + post
    out = []
    for s, e in segs:
        s, e = max(0.0, s), min(dur, e)
        if out and s - out[-1][1] < max(gap - 2 * pad, 2.5): out[-1][1] = max(out[-1][1], e)
        else: out.append([s, e])
    return out if sum(e - s for s, e in out) > 3 else [[0.0, dur]]

def nice_speed(d, target, cap):
    return 1.0 if d <= target else min(cap, math.ceil(d / target * 2) / 2)

_patch = {}
def label_patch(text, size=15, bold=True, bg=(0, 0, 0, 170), fg=(255, 255, 255, 255), padx=8, pady=4):
    k = (text, size, bold, bg, fg)
    if k not in _patch:
        f = font(size, bold); w = int(f.getlength(text)) + 2 * padx; h = size + 2 * pady + 4
        im = Image.new("RGBA", (w, h), (0, 0, 0, 0)); d = ImageDraw.Draw(im); d.rounded_rectangle((0, 0, w - 1, h - 1), radius=6, fill=bg); d.text((padx, pady), text, font=f, fill=fg)
        a = np.array(im); _patch[k] = (cv2.cvtColor(a[:, :, :3], cv2.COLOR_RGB2BGR).astype(np.float32), a[:, :, 3:4].astype(np.float32) / 255)
    return _patch[k]

def blend(img, patch, x, y):
    rgb, al = patch; h, w = al.shape[:2]; x = max(0, min(W - w, x)); y = max(0, min(H - h, y))
    roi = img[y:y + h, x:x + w].astype(np.float32); img[y:y + h, x:x + w] = (roi * (1 - al) + rgb * al).astype(np.uint8)

class Enc:
    def __init__(s, path):
        s.o = av.open(str(path), "w", options={"movflags": "+faststart"}); s.st = s.o.add_stream("libx264", rate=FPS)
        s.st.width, s.st.height, s.st.pix_fmt = W, H, "yuv420p"; s.st.options = {"crf": "24", "preset": "veryfast"}; s.n = 0
    def put(s, bgr):
        if bgr.shape[:2] != (H, W): bgr = cv2.resize(bgr, (W, H))
        for p in s.st.encode(av.VideoFrame.from_ndarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), format="rgb24")): s.o.mux(p)
        s.n += 1
    def close(s):
        for p in s.st.encode(): s.o.mux(p)
        s.o.close()

def plan_of(segs, speed):
    plan = []
    for si, (a, b) in enumerate(segs):
        for j in range(max(1, int(round((b - a) / speed * FPS)))): plan.append((a + j * speed / FPS, si))
    return plan

def frames(path, fps, plan):
    """Yield (frame, seg_index) for each planned source time, reading the file forward only."""
    c = cv2.VideoCapture(str(path)); cur_k, cur = -1, None
    for s, si in plan:
        k = int(round(s * fps))
        while cur_k < k:
            ok, im = c.read()
            if not ok: break
            cur, cur_k = im, cur_k + 1
        if cur is not None: yield cur.copy(), si

def analyse(it):
    src = ROOT / it["src"]; fps, n = probe(src)
    if fps <= 0 or n <= 0: return None
    dur = n / fps
    if it.get("no_trim"): segs = [[0.0, dur]]
    elif it.get("joints") and (ROOT / it["joints"]).exists(): T, M = joint_motion(ROOT / it["joints"]); segs = keep_segments(T, M, dur)
    else: T, M = frame_motion(src, fps); segs = keep_segments(T, M, dur)
    kept = sum(e - s for s, e in segs); return {"src": src, "fps": fps, "dur": dur, "segs": segs, "kept": kept}

def edit_clip(it, a, force):
    out = P / "videos" / f"{it['id']}.mp4"; speed = 1.0 if it.get("no_trim") else nice_speed(a["kept"], 40, 6)
    cuts = [round(a["segs"][i + 1][0] - a["segs"][i][1], 1) for i in range(len(a["segs"]) - 1)]
    lead, tail = round(a["segs"][0][0], 1), round(a["dur"] - a["segs"][-1][1], 1)
    info = {"src_dur": round(a["dur"], 1), "kept": round(a["kept"], 1), "speed": speed, "out_dur": round(a["kept"] / speed, 1), "cuts": cuts, "lead_cut": lead, "tail_cut": tail, "segs": [[round(s, 2), round(e, 2)] for s, e in a["segs"]]}
    if out.exists() and out.stat().st_size > 1000 and not force: return info
    plan = plan_of(a["segs"], speed); enc = Enc(out); last_si = 0; cut_until = -1; badge = label_patch(f"{speed:g}x speed") if speed > 1 else None
    for j, (im, si) in enumerate(frames(a["src"], a["fps"], plan)):
        if si != last_si: cut_until = j + int(1.2 * FPS); cut_txt = label_patch(f"- {cuts[si - 1]:.0f} s idle cut -", 14, bg=(120, 0, 0, 190)); last_si = si
        if badge: blend(im, badge, 6, H - 32)
        if j < cut_until: blend(im, cut_txt, W // 2 - 80, H - 32)
        enc.put(im)
    enc.close()
    poster_src = ROOT / it["clean"] if it.get("clean") and (ROOT / it["clean"]).exists() else a["src"]
    t = plan[int(len(plan) * 0.8)][0]; c = cv2.VideoCapture(str(poster_src)); c.set(cv2.CAP_PROP_POS_FRAMES, int(t * a["fps"])); ok, im = c.read()
    if ok: cv2.imwrite(str(P / "posters" / f"{it['id']}.jpg"), cv2.resize(im, (W, H)), [cv2.IMWRITE_JPEG_QUALITY, 82])
    print(f"{it['id']}: {a['dur']:.0f} s -> kept {a['kept']:.0f} s in {len(a['segs'])} part(s), {speed:g}x -> {a['kept'] / speed:.0f} s", flush=True)
    return info

def wrap(text, f, width):
    words, lines, cur = text.split(), [], ""
    for w in words:
        t = (cur + " " + w).strip()
        if f.getlength(t) <= width: cur = t
        else: lines.append(cur); cur = w
    return lines + [cur] if cur else lines

def card(title, sub="", kicker="", small=""):
    im = Image.new("RGB", (W, H), (15, 18, 25)); d = ImageDraw.Draw(im)
    for y in range(H): d.line([(0, y), (W, y)], fill=(15 + y // 24, 18 + y // 20, 25 + y // 12))
    d.rectangle((40, 150, 46, 330), fill=(106, 168, 255)); y = 150
    if kicker: d.text((64, y), kicker, font=font(16, True), fill=(106, 168, 255)); y += 30
    for l in wrap(title, font(30, True), 530): d.text((64, y), l, font=font(30, True), fill=(240, 243, 248)); y += 40
    y += 8
    for l in wrap(sub, font(18), 530): d.text((64, y), l, font=font(18), fill=(170, 180, 198)); y += 26
    if small: d.text((64, H - 50), small, font=font(13), fill=(120, 130, 150))
    return cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)

def put_card(enc, img, secs):
    n = int(secs * FPS); f = int(0.35 * FPS)
    for j in range(n): enc.put((img * min(1.0, (j + 1) / f, (n - j) / f)).astype(np.uint8))

def lower_third(it, speed):
    rgb = COL[it["outcome"]]; badge = label_patch(it["verdict"], 13, True, bg=rgb + (235,)); title = label_patch(it["title"], 16, True, bg=(0, 0, 0, 185))
    sp = label_patch(f"{speed:g}x", 13, True) if speed > 1 else None; return badge, title, sp

def reel(items, analyses):
    out = P / "videos" / "highlight_reel.mp4"; enc = Enc(out); secs = {s["id"]: s for s in CAT["sections"]}; order = [s["id"] for s in CAT["sections"]]; total = []
    intro = card("Teaching a desk robot real chores", "One real 8-servo arm, one table camera, a vision model and kinematics. Every run filmed and saved as a LeRobot episode, failures included.", "NORMACORE HACKATHON - BERLIN, 21-22 NOV 2026", "Highlight reel - clips trimmed and sped up, original timing shown in each clip")
    cv2.imwrite(str(P / "posters" / "reel.jpg"), intro, [cv2.IMWRITE_JPEG_QUALITY, 88]); put_card(enc, intro, 5)
    sec_no = 0; cur = None
    for it in sorted(items, key=lambda x: order.index(x["section"])):
        if it["section"] != cur:
            cur = it["section"]; sec_no += 1; s = secs[cur]; put_card(enc, card(s["title"], s["line"], f"PART {sec_no}"), 3)
        a = analyses[it["id"]]; speed = max(nice_speed(a["kept"], 40, 6), nice_speed(a["kept"], it.get("reel_max", 18), 8)); src = ROOT / it["clean"] if it.get("clean") and (ROOT / it["clean"]).exists() else a["src"]
        badge, title, sp = lower_third(it, speed); plan = plan_of(a["segs"], speed); nfade = int(0.3 * FPS)
        for j, (im, si) in enumerate(frames(src, a["fps"], plan)):
            g = cv2.GaussianBlur(im[H - 78:], (0, 0), 6); im[H - 78:] = (g * 0.55).astype(np.uint8)
            blend(im, title, 12, H - 70); blend(im, badge, 12, H - 38)
            if sp: blend(im, sp, W - 52, H - 38)
            if j < nfade: im = (im * ((j + 1) / nfade)).astype(np.uint8)
            enc.put(im)
        total.append((it["id"], round(len(plan) / FPS, 1)))
    ns = sum(1 for i in CAT["items"] if i["outcome"] == "success" and i["section"] not in ("sim", "appendix")); nall = sum(1 for i in CAT["items"] if i["section"] not in ("sim", "appendix", "task"))
    put_card(enc, card("What we learned", f"{nall} recorded real-robot runs, {ns} successes. A camera photo with the arm out of the way decides success - never the model's yes/no or the script's own 'done'.", "THE END", "Full gallery: presentation/index.html - slides: presentation/slides.html"), 5)
    enc.close(); print(f"highlight reel: {enc.n / FPS:.0f} s ->", out, flush=True); return {"reel_s": round(enc.n / FPS, 1), "parts": total}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--only", default=""); ap.add_argument("--no-reel", action="store_true"); ap.add_argument("--reel-only", action="store_true"); ap.add_argument("--force", action="store_true"); a = ap.parse_args()
    only = set(filter(None, a.only.split(","))); ej = P / "edits.json"; edits = json.loads(ej.read_text()) if ej.exists() else {}; analyses = {}
    for it in CAT["items"]:
        if only and it["id"] not in only and not (a.reel_only and it.get("reel")): continue
        if a.reel_only and not it.get("reel"): continue
        an = analyse(it)
        if an is None: print("SKIP (unreadable):", it["src"], flush=True); continue
        analyses[it["id"]] = an
        if not a.reel_only: edits[it["id"]] = edit_clip(it, an, a.force); ej.write_text(json.dumps(edits, indent=1))
    if not a.no_reel and not only: edits["_reel"] = reel([i for i in CAT["items"] if i.get("reel") and i["id"] in analyses], analyses); ej.write_text(json.dumps(edits, indent=1))
