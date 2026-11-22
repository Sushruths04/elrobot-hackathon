#!/usr/bin/env python
"""Final 1080p CRF-14 Hackathon Pitch Video Builder ('Autonomous House Helper').

Follows the exact 8-chapter pitch narrative, cutting idle gaps via joint-motion segments so every clip plays smoothly
at natural 1x-2x pacing directly from raw *_nocaption.mp4 camera feeds with MJPEG deblocking + CLAHE clarity.
"""
from __future__ import annotations

import io
import json
import math
import shutil
import wave
from pathlib import Path

import av
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from piper.voice import PiperVoice
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "presentation"
W, H, FPS, SR = 1920, 1080, 30, 48000
VW, VH = 1216, 912  # Large, razor-sharp 4:3 video window (64% of width, 84% of height)

_fonts: dict = {}


def font(sz: int, bold: bool = False):
    k = (sz, bold)
    if k not in _fonts:
        _fonts[k] = ImageFont.truetype(
            f"/usr/share/fonts/truetype/dejavu/DejaVuSans{'-Bold' if bold else ''}.ttf", sz
        )
    return _fonts[k]


def wrap_lines(text: str, f, max_w: int) -> list[str]:
    words, lines, cur = text.split(), [], ""
    for w in words:
        cand = (cur + " " + w).strip()
        if f.getlength(cand) <= max_w:
            cur = cand
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def polish_raw(bgr: np.ndarray) -> np.ndarray:
    clean = cv2.bilateralFilter(bgr, d=5, sigmaColor=16, sigmaSpace=16)
    lab = cv2.cvtColor(clean, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l_enh = cv2.createCLAHE(clipLimit=1.4, tileGridSize=(8, 8)).apply(l)
    clean = cv2.cvtColor(cv2.merge([l_enh, a, b]), cv2.COLOR_LAB2BGR)
    ih, iw = clean.shape[:2]
    scale = min(VW / iw, VH / ih)
    nw, nh = int(round(iw * scale)), int(round(ih * scale))
    up = cv2.resize(clean, (nw, nh), interpolation=cv2.INTER_LANCZOS4)
    blur = cv2.GaussianBlur(up, (0, 0), 0.9)
    sharp = cv2.addWeighted(up, 1.25, blur, -0.25, 0)
    if nw == VW and nh == VH:
        return sharp
    canvas = np.full((VH, VW, 3), (244, 243, 238), dtype=np.uint8)
    x0, y0 = (VW - nw) // 2, (VH - nh) // 2
    canvas[y0 : y0 + nh, x0 : x0 + nw] = sharp
    return canvas


def active_segments(jpath: Path | None, dur: float) -> list[tuple[float, float]]:
    if jpath is None or not jpath.exists():
        return [(0.0, dur)]
    rows = [json.loads(line) for line in open(jpath) if line.strip()]
    T = np.array([r["t"] for r in rows], float)
    Q = np.array([[r[str(i)] for i in range(1, 9)] for r in rows], float)
    j = np.clip(np.searchsorted(T, T - 0.3), 0, len(T) - 1)
    mov = np.where(np.abs(((Q - Q[j] + 2048) % 4096) - 2048).max(1) > 5)[0]
    if len(mov) == 0:
        return [(0.0, dur)]
    iv = []
    s = e = float(T[mov[0]])
    for k in mov[1:]:
        tk = float(T[k])
        if tk - e > 0.5:
            iv.append([s, e])
            s = tk
        e = tk
    iv.append([s, e])
    out = []
    for a, b in iv:
        a, b = max(0.0, a - 0.5), min(dur, b + 0.8)
        if out and a - out[-1][1] < 2.0:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(x[0], x[1]) for x in out] if sum(x[1] - x[0] for x in out) > 2.0 else [(0.0, dur)]


def load_active_frames(src_rel: str, joints_rel: str | None, n_out: int, clip_window: tuple[float, float] | None = None) -> list[np.ndarray]:
    src = ROOT / src_rel
    cap = cv2.VideoCapture(str(src))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    nf = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    dur = nf / fps
    if clip_window is not None:
        segs = [(max(0.0, clip_window[0]), min(dur, clip_window[1]))]
    else:
        segs = active_segments(ROOT / joints_rel if joints_rel else None, dur)

    # Collect frame indices belonging to active motion segments only (cuts all dead idle waits!)
    valid_fidxs = []
    for a_s, b_s in segs:
        f0 = max(0, int(round(a_s * fps)))
        f1 = min(nf - 1, int(round(b_s * fps)))
        valid_fidxs.extend(range(f0, f1 + 1))
    if not valid_fidxs:
        valid_fidxs = list(range(nf))

    pick_pos = np.linspace(0, len(valid_fidxs) - 1, n_out)
    target_set = sorted({valid_fidxs[int(round(p))] for p in pick_pos})
    frame_map = {}
    cur_f = 0
    t_ptr = 0
    while t_ptr < len(target_set):
        ok, im = cap.read()
        if not ok:
            break
        if cur_f == target_set[t_ptr]:
            frame_map[cur_f] = polish_raw(im)
            t_ptr += 1
        cur_f += 1
    cap.release()
    if not frame_map:
        return [np.full((VH, VW, 3), 245, dtype=np.uint8)] * n_out
    fallback = next(iter(frame_map.values()))
    return [frame_map.get(valid_fidxs[int(round(p))], fallback) for p in pick_pos]


def make_slide_canvas(step_idx: int, n_steps: int, chapter: str, title: str, bullets: list[str]) -> np.ndarray:
    im = Image.new("RGB", (W, H), (248, 247, 244))
    d = ImageDraw.Draw(im)

    # Clean light header bar
    d.rectangle((0, 0, W, 98), fill=(255, 255, 255))
    d.line([(0, 98), (W, 98)], fill=(226, 223, 215), width=2)
    d.rectangle((52, 26, 60, 72), fill=(21, 128, 61))
    d.text((78, 22), "AUTONOMOUS HOUSE HELPER", font=font(27, True), fill=(24, 24, 27))
    d.text((78, 56), "8-Servo Robot Arm  •  Vision, Tool Use, Digital Twin & RL", font=font(16, False), fill=(100, 96, 90))

    ptxt = f"STEP {step_idx} / {n_steps}"
    pw = int(font(17, True).getlength(ptxt)) + 34
    d.rounded_rectangle((W - 52 - pw, 30, W - 52, 68), radius=19, fill=(240, 253, 244), outline=(187, 247, 208), width=2)
    d.text((W - 52 - pw + 17, 39), ptxt, font=font(17, True), fill=(21, 128, 61))

    # Video card bezel (x=48..1272, y=122..1042; video inside at x=52, y=126 of size 1216x912)
    d.rounded_rectangle((48, 122, 52 + VW + 4, 126 + VH + 4), radius=14, fill=(255, 255, 255), outline=(214, 211, 203), width=3)

    # Right Pitch Points Card (x=1300..1872, y=122..1042)
    rx0, ry0, rx1, ry1 = 1300, 122, 1872, 1042
    d.rounded_rectangle((rx0, ry0, rx1, ry1), radius=16, fill=(255, 255, 255), outline=(224, 221, 213), width=2)

    d.text((rx0 + 32, ry0 + 36), chapter.upper(), font=font(16, True), fill=(21, 128, 61))
    y = ry0 + 70
    f_t = font(33, True)
    for ln in wrap_lines(title, f_t, rx1 - rx0 - 64):
        d.text((rx0 + 32, y), ln, font=f_t, fill=(24, 24, 27))
        y += 43

    y += 22
    d.line([(rx0 + 32, y), (rx1 - 32, y)], fill=(232, 229, 222), width=2)
    y += 28

    f_b = font(21, False)
    for bul in bullets:
        d.ellipse((rx0 + 34, y + 8, rx0 + 45, y + 19), fill=(21, 128, 61))
        for bl in wrap_lines(bul, f_b, rx1 - rx0 - 94):
            d.text((rx0 + 58, y), bl, font=f_b, fill=(68, 64, 60))
            y += 30
        y += 20

    return cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)


def make_bookend_slide(kicker: str, title: str, subtitle: str, tags: list[str]) -> np.ndarray:
    im = Image.new("RGB", (W, H), (248, 247, 244))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle((110, 130, W - 110, H - 130), radius=24, fill=(255, 255, 255), outline=(220, 217, 209), width=2)
    d.rectangle((170, 215, 178, 615), fill=(21, 128, 61))
    d.text((210, 212), kicker.upper(), font=font(22, True), fill=(21, 128, 61))
    d.text((210, 268), title, font=font(64, True), fill=(24, 24, 27))
    y = 368
    f_s = font(28, False)
    for ln in wrap_lines(subtitle, f_s, W - 440):
        d.text((210, y), ln, font=f_s, fill=(87, 83, 78))
        y += 41
    px, py = 210, H - 245
    fp = font(20, True)
    for tg in tags:
        pw = int(fp.getlength(tg)) + 34
        d.rounded_rectangle((px, py, px + pw, py + 46), radius=10, fill=(242, 240, 234), outline=(220, 217, 209), width=1)
        d.text((px + 17, py + 10), tg, font=fp, fill=(24, 24, 27))
        px += pw + 12
    return cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)


def synth_vo(voice: PiperVoice, text: str) -> np.ndarray:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        voice.synthesize(text, wf, length_scale=0.95, sentence_silence=0.20)
    buf.seek(0)
    with wave.open(buf, "rb") as rf:
        sr_in = rf.getframerate()
        pcm = np.frombuffer(rf.readframes(rf.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
    if sr_in != SR and len(pcm) > 0:
        g = math.gcd(SR, sr_in)
        pcm = resample_poly(pcm, SR // g, sr_in // g).astype(np.float32)
    pk = float(np.max(np.abs(pcm))) if len(pcm) else 1.0
    return pcm * (0.80 / pk) if pk > 1e-4 else pcm


def warm_music(total_samples: int) -> np.ndarray:
    t = np.arange(total_samples, dtype=np.float64) / SR
    chords = [[174.61, 220.0, 261.63, 329.63], [196.0, 246.94, 293.66, 392.0], [164.81, 196.0, 246.94, 293.66], [220.0, 261.63, 329.63, 392.0]]
    bar_s = 5.5
    L, R = np.zeros(total_samples), np.zeros(total_samples)
    for b in range(int(math.ceil((total_samples / SR) / bar_s))):
        i0, i1 = int(b * bar_s * SR), min(total_samples, int((b + 1) * bar_s * SR + 0.5 * SR))
        if i0 >= total_samples:
            break
        tb = t[i0:i1] - b * bar_s
        dur_b = (i1 - i0) / SR
        env = np.exp(-tb / (dur_b * 0.85)) * np.minimum(1.0, tb / 0.12) * np.minimum(1.0, (dur_b - tb) / 0.4)
        for fi, fq in enumerate(chords[b % 4]):
            L[i0:i1] += env * 0.065 * np.sin(2 * np.pi * fq * tb)
            R[i0:i1] += env * 0.065 * np.sin(2 * np.pi * (fq * 1.001) * tb + 0.3 * fi)
    return np.column_stack([L, R]).astype(np.float32)


def main() -> None:
    onnx_p = ROOT / "models/tts/voice_ryan.onnx"
    voice = PiperVoice.load(str(onnx_p), config_path=str(onnx_p) + ".json")

    # Exact 8-Chapter Pitch Order specified by the user!
    scenes = [
        {
            "src": "results/auto_pen_marker.mp4",
            "joints": None,
            "win": None,
            "chapter": "01 • Autonomous Desk Organisation",
            "title": "Autonomous Pen & Marker Pick-and-Place",
            "bullets": [
                "Autonomous camera + 7-DoF IK places markers & pens in ~23 s",
                "Gripper stall ticks sense thin blue pens (1397) vs thick markers (1570)",
                "Continuous reset-and-place loop documented across autonomous runs",
            ],
            "vo": "We start with autonomous desk organisation: locating pens and markers from the camera, sensing thin pens versus thick markers from gripper stall ticks, and placing them inside the white box in twenty-three seconds.",
            "min_s": 9.8,
        },
        {
            "src": "results/films/duster_magnet_caps_into_box_20261122_005058_nocaption.mp4",
            "joints": "results/films/duster_magnet_caps_into_box_20261122_005058_joints.jsonl",
            "win": None,
            "chapter": "02 • Precision Center-Magnet Pickup",
            "title": "Magnetic Duster Bottle-Cap Cleanup",
            "bullets": [
                "The magnet sits only in the dead center of the duster pad",
                "Robot aligns the exact duster center 1.2 cm over each bottle cap",
                "Snaps up both metal caps with zero pressing, checks table, and bins them",
            ],
            "vo": "Next, magnetic bottle-cap cleanup. Because the magnet sits only in the dead center of the duster, the robot precisely aligns that center twelve millimeters over each metal cap to snap them up, checks the table is clear, and drops them in the box.",
            "min_s": 13.5,
        },
        {
            "src": "results/films/tempo_guided_session_20261122_030809_nocaption.mp4",
            "joints": "results/films/tempo_guided_session_20261122_030809_joints.jsonl",
            "win": (0.0, 155.0),  # The crisp lid-open + tissue-extraction portion before the bean lanes!
            "chapter": "03 • Hero Deformable Manipulation",
            "title": "Chained Flip-Top Lid Open & Tissue Extraction",
            "bullets": [
                "Horizontal constant-height stroke catches the lip and pops the lid open",
                "Immediately indexes to the standing fold and pinches a single wet wipe",
                "Straight vertical pull extracts one thin sheet cleanly out of the pack",
            ],
            "vo": "Now our centerpiece skill: chained flip-top lid opening and tissue extraction. One smooth sideways stroke pops the hinged cap wide open, and the gripper immediately pinches the standing tip and pulls a single wet wipe straight out.",
            "min_s": 14.0,
        },
        {
            "src": "results/films/tempo_wipe_beans_with_tissue_20261122_024526_nocaption.mp4",
            "joints": "results/films/tempo_wipe_beans_with_tissue_20261122_024526_joints.jsonl",
            "win": None,
            "chapter": "04 • Cleaning Spilled Beans (Attempt 1)",
            "title": "Five-Lane Table-Clearing Sweep with Tissue",
            "bullets": [
                "Asked to clean spilled coffee beans using the extracted tissue",
                "Executes five parallel table-level wiping lanes with lifted returns",
                "Clears the target zone, though soft tissue scatters loose beans",
            ],
            "vo": "Then we asked the robot to clean spilled coffee beans. First, it used the tissue in a five-lane table-clearing sweep pattern across the spill zone.",
            "min_s": 10.5,
        },
        {
            "src": "results/films/tempo_guided_session_20261122_034731_nocaption.mp4",
            "joints": "results/films/tempo_guided_session_20261122_034731_joints.jsonl",
            "win": None,
            "chapter": "05 • Intelligent Tool Adaptation",
            "title": "Pushing the Entire Packet Box Sideways",
            "bullets": [
                "Smarter strategy: skip pulling a soft tissue for hard rolling beans",
                "Uses the rigid side of the wet-wipe packet itself as a wide plow",
                "Pushes the entire box 21 cm sideways to clear the beans in one stroke",
            ],
            "vo": "Then came the smart adaptation: instead of pulling a soft tissue to wipe rolling beans, it pushed the entire wet-wipe box sideways like a wide squeegee, sweeping the beans clean in one stroke.",
            "min_s": 12.0,
        },
        {
            "src": "results/teleop_demos/teleop_spoon_113019_ep0.mp4",
            "joints": None,
            "win": None,
            "chapter": "06 • Bowl-to-Cup Bean Transfer",
            "title": "Scooping Coffee Beans from Bowl into Cup",
            "bullets": [
                "One of our core target tasks: spooning loose beans from bowl to cup",
                "Collected 24 dual-camera 30.0 fps episodes + vision-shifted autonomous runs",
                "Grasps wooden spoon handle, scoops beans, and pours into the black cup",
            ],
            "vo": "One of our main manipulation goals was transferring coffee beans from a bowl into a cup with a wooden spoon, where we recorded twenty-four dual-camera episodes and ran vision-shifted scooping.",
            "min_s": 12.0,
        },
        {
            "src": "docs/twin/twin_replay_sidebyside.mp4",
            "joints": None,
            "win": None,
            "chapter": "07 • Real-to-Sim Digital Twin",
            "title": "Side-by-Side Real Robot & MuJoCo Twin",
            "bullets": [
                "Synchronised side-by-side replay of our real arm and MuJoCo 3.2.7 twin",
                "Matches real hardware trajectories within 6.79 mm mean error",
                "Validated across a 100-trial cup benchmark (94% autonomous grasp)",
            ],
            "vo": "We also built a physics-calibrated MuJoCo digital twin that mirrors our real arm side-by-side within seven millimeters and reaches ninety-four percent across a hundred cup-grasping trials.",
            "min_s": 11.5,
        },
        {
            "src": "results/pingpong_balance_hd.mp4",
            "joints": None,
            "win": None,
            "chapter": "08 • Dynamic Goal: 30 Hz Table-Tennis RL",
            "title": "RL Racket Ball Balancing & Hardware Torque Limit",
            "bullets": [
                "Main dynamic goal: balance & dribble a ping-pong ball on a JOOLA racket",
                "30 Hz PPO policy matched 1:1 to our sideways grip hits 95% balance in sim",
                "On real 12 V servos, the heavy outstretched racket exceeded holding torque",
            ],
            "vo": "Our ultimate dynamic goal was balancing a table-tennis ball. We trained a thirty-hertz reinforcement learning policy to ninety-five percent in the twin, though on the real arm the heavy outstretched racket exceeded servo holding torque.",
            "min_s": 12.5,
        },
    ]

    intro_vo = synth_vo(
        voice,
        "Meet Autonomous House Helper: an eight-servo robot arm that tidies desks, collects metal caps, opens wet wipes, sweeps spills, and explores dynamic reinforcement learning.",
    )
    outro_vo = synth_vo(
        voice,
        "Autonomous House Helper: real-world household manipulation, honest hardware insights, and calibrated simulation on desktop hardware.",
    )
    intro_s = max(4.5, len(intro_vo) / SR + 0.6)
    outro_s = max(4.2, len(outro_vo) / SR + 0.8)

    packs = []
    for sc in scenes:
        vo = synth_vo(voice, sc["vo"])
        packs.append((sc, vo, max(sc["min_s"], len(vo) / SR + 0.75)))

    tot_s = intro_s + sum(x[2] for x in packs) + outro_s
    tot_frames = int(round(tot_s * FPS))
    tot_samples = int(round(tot_frames / FPS * SR))

    music = warm_music(tot_samples)
    speech = np.zeros(tot_samples, dtype=np.float32)
    duck = np.ones(tot_samples, dtype=np.float32) * 0.42

    def put_vo(vo: np.ndarray, t_start: float):
        s0 = int((t_start + 0.25) * SR)
        s1 = min(tot_samples, s0 + len(vo))
        if s1 > s0:
            speech[s0:s1] += vo[: s1 - s0]
            duck[max(0, s0 - int(0.2 * SR)) : min(tot_samples, s1 + int(0.3 * SR))] = 0.12

    ct = 0.0
    put_vo(intro_vo, ct)
    ct += intro_s
    for _, vo, ds in packs:
        put_vo(vo, ct)
        ct += ds
    put_vo(outro_vo, ct)

    kw = int(0.15 * SR)
    dsmooth = np.convolve(duck, np.ones(kw) / kw, mode="same").astype(np.float32)
    stereo = np.vstack([
        np.clip(speech + music[:, 0] * dsmooth, -0.96, 0.96),
        np.clip(speech + music[:, 1] * dsmooth, -0.96, 0.96),
    ])

    out_main = P / "Autonomous_House_Helper_Presentation_1080p.mp4"
    container = av.open(str(out_main), mode="w", options={"movflags": "+faststart"})
    vs = container.add_stream("libx264", rate=FPS)
    vs.width, vs.height, vs.pix_fmt = W, H, "yuv420p"
    vs.options = {"crf": "14", "preset": "fast"}
    as_ = container.add_stream("aac", rate=SR)
    as_.layout = "stereo"
    as_.bit_rate = 192000

    def push(bgr: np.ndarray):
        vf = av.VideoFrame.from_ndarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), format="rgb24")
        for p in vs.encode(vf):
            container.mux(p)

    fc = 0
    paper = np.full((H, W, 3), (244, 247, 248), dtype=np.uint8)
    intro_im = make_bookend_slide(
        "NormaCore Robotics Hackathon  •  Berlin",
        "Autonomous House Helper",
        "From pen tidying, center-magnet cap pickup, and chained wet-wipe extraction to intelligent box-push bean sweeping, calibrated digital twins, and 30 Hz table-tennis RL.",
        ["1. Pen Tidy", "2. Center Magnet", "3. Lid & Wipe Pull", "4. Smart Box Sweep", "5. Spoon & 30 Hz RL"],
    )
    cv2.imwrite(str(P / "posters/youtube_pitch_95s.jpg"), intro_im, [cv2.IMWRITE_JPEG_QUALITY, 95])
    ni = int(round(intro_s * FPS))
    ff = int(0.35 * FPS)
    for j in range(ni):
        al = min(1.0, (j + 1) / ff, (ni - j) / ff)
        push((intro_im.astype(np.float32) * al + paper.astype(np.float32) * (1.0 - al)).astype(np.uint8))
        fc += 1

    for idx, (sc, _, ds) in enumerate(packs, 1):
        ns = int(round(ds * FPS))
        tmpl = make_slide_canvas(idx, len(packs), sc["chapter"], sc["title"], sc["bullets"])
        vframes = load_active_frames(sc["src"], sc["joints"], ns, clip_window=sc["win"])
        for j, vf_im in enumerate(vframes):
            canv = tmpl.copy()
            canv[126 : 126 + VH, 52 : 52 + VW] = vf_im
            pw = int(VW * ((j + 1) / max(1, ns)))
            cv2.rectangle(canv, (52, 126 + VH - 5), (52 + pw, 126 + VH), (61, 128, 21), -1)
            push(canv)
            fc += 1
        print(f"  step {idx}/{len(packs)} ({sc['title'][:36]}) encoded", flush=True)

    outro_im = make_bookend_slide(
        "Thank You",
        "Autonomous House Helper",
        "Real-robot household chores, emergent tool adaptation, dual-camera datasets, and calibrated physics simulation — built 100% locally.",
        ["Pen & Cap Cleanup", "Wet-Wipe Open & Pull", "Packet Squeegee Sweep", "MuJoCo Twin & 30 Hz RL"],
    )
    no = max(1, tot_frames - fc)
    for j in range(no):
        al = min(1.0, (j + 1) / ff, (no - j) / ff)
        push((outro_im.astype(np.float32) * al + paper.astype(np.float32) * (1.0 - al)).astype(np.uint8))
        fc += 1

    for p in vs.encode():
        container.mux(p)

    stereo = stereo[:, : int(round(fc / FPS * SR))]
    for s in range(0, stereo.shape[1], 1024):
        af = av.AudioFrame.from_ndarray(np.ascontiguousarray(stereo[:, s : s + 1024]), format="fltp", layout="stereo")
        af.sample_rate = SR
        for p in as_.encode(af):
            container.mux(p)
    for p in as_.encode():
        container.mux(p)
    container.close()

    shutil.copy(out_main, P / "videos/youtube_pitch_95s.mp4")
    shutil.copy(out_main, P / "videos/highlight_reel.mp4")
    print(f"FINAL PITCH VIDEO READY: {out_main} ({fc/FPS:.1f}s, {out_main.stat().st_size/1e6:.1f} MB)", flush=True)


if __name__ == "__main__":
    main()
