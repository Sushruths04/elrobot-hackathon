#!/usr/bin/env python
"""Render clean, CAPTION-FREE 1920x1080 @ 30fps YouTube videos with natural 22 kHz voiceover & subtle warm music.

Zero burned-in captions, zero side-panel text, zero badges on the footage:
  - Crisp 1440x1080 Lanczos4 robot video centered over a soft light-blurred ambient edge fill (no black bars, no text clutter).
  - Direct, natural voiceover covering all robot successes: opening the lid & pulling out tissues, packet-sweep & spoon bean cleaning,
    magnetic bottle-cap pickup, pen & marker pick-and-place, and the calibrated twin & 30 Hz racket ball balancing.
"""
from __future__ import annotations

import io
import math
import shutil
import wave
from pathlib import Path

import av
import cv2
import numpy as np
from piper.voice import PiperVoice
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "presentation"
VID_DIR = P / "videos"
POSTER_DIR = P / "posters"
VID_DIR.mkdir(parents=True, exist_ok=True)
POSTER_DIR.mkdir(parents=True, exist_ok=True)

W, H, FPS = 1920, 1080, 30
SR = 48000


def synth_voice_48k(voice: PiperVoice, text: str) -> np.ndarray:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        voice.synthesize(text, wf, length_scale=0.96, sentence_silence=0.22)
    buf.seek(0)
    with wave.open(buf, "rb") as rf:
        sr_in = rf.getframerate()
        pcm = np.frombuffer(rf.readframes(rf.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
    if sr_in != SR and len(pcm) > 0:
        g = math.gcd(SR, sr_in)
        pcm = resample_poly(pcm, SR // g, sr_in // g).astype(np.float32)
    pk = float(np.max(np.abs(pcm))) if len(pcm) else 1.0
    if pk > 1e-4:
        pcm = pcm * (0.78 / pk)
    return pcm


def generate_warm_bed(total_samples: int) -> np.ndarray:
    t = np.arange(total_samples, dtype=np.float64) / SR
    chords = [
        [174.61, 220.00, 261.63, 329.63],
        [196.00, 246.94, 293.66, 392.00],
        [164.81, 196.00, 246.94, 293.66],
        [220.00, 261.63, 329.63, 392.00],
    ]
    bar_s = 5.5
    left = np.zeros(total_samples, dtype=np.float64)
    right = np.zeros(total_samples, dtype=np.float64)
    n_bars = int(math.ceil((total_samples / SR) / bar_s))
    for b in range(n_bars):
        i0 = int(b * bar_s * SR)
        i1 = min(total_samples, int((b + 1) * bar_s * SR + 0.6 * SR))
        if i0 >= total_samples:
            break
        tb = t[i0:i1] - b * bar_s
        dur_b = (i1 - i0) / SR
        env = np.exp(-tb / (dur_b * 0.85)) * np.minimum(1.0, tb / 0.12) * np.minimum(1.0, (dur_b - tb) / 0.4)
        freqs = chords[b % len(chords)]
        sl = np.zeros_like(tb)
        sr_c = np.zeros_like(tb)
        for fi, fq in enumerate(freqs):
            sl += (0.28 / len(freqs)) * np.sin(2 * np.pi * fq * tb)
            sr_c += (0.28 / len(freqs)) * np.sin(2 * np.pi * (fq * 1.001) * tb + 0.3 * fi)
        left[i0:i1] += env * sl
        right[i0:i1] += env * sr_c
    fade = min(int(1.6 * SR), total_samples // 4)
    if fade > 0:
        ramp = np.linspace(0.0, 1.0, fade)
        left[:fade] *= ramp; right[:fade] *= ramp
        left[-fade:] *= ramp[::-1]; right[-fade:] *= ramp[::-1]
    return np.column_stack([left, right]).astype(np.float32)


def frame_to_clean_1080p(bgr: np.ndarray) -> np.ndarray:
    """Pure caption-free 1920x1080 frame: soft warm blurred backdrop + crisp 1440x1080 center video."""
    # Softly blurred, lightened warm background so edges match the room naturally with zero black bars and zero text!
    bg_small = cv2.resize(bgr, (240, 135), interpolation=cv2.INTER_AREA)
    bg_blur = cv2.GaussianBlur(bg_small, (25, 25), 10.0)
    bg_full = cv2.resize(bg_blur, (W, H), interpolation=cv2.INTER_LINEAR)
    bg_light = cv2.addWeighted(bg_full, 0.55, np.full_like(bg_full, 245), 0.45, 0)

    fg = cv2.resize(bgr, (1440, 1080), interpolation=cv2.INTER_LANCZOS4)
    x0 = (W - 1440) // 2  # 240
    bg_light[:, x0 : x0 + 1440] = fg
    # Subtle 1px crisp hairline border separating center frame from soft wings
    cv2.line(bg_light, (x0, 0), (x0, H), (215, 212, 205), 2)
    cv2.line(bg_light, (x0 + 1439, 0), (x0 + 1439, H), (215, 212, 205), 2)
    return bg_light


def read_clip_1080p(clip_path: Path, n_out_frames: int) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(clip_path))
    raw = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        raw.append(bgr)
    cap.release()
    if not raw:
        return [np.full((H, W, 3), 245, dtype=np.uint8)] * n_out_frames
    idxs = np.linspace(0, len(raw) - 1, n_out_frames)
    return [frame_to_clean_1080p(raw[int(round(i))]) for i in idxs]


def build_clean_cut(voice: PiperVoice, out_mp4: Path, scenes: list[dict]) -> float:
    print(f"Rendering caption-free 1080p {out_mp4.name} ({len(scenes)} scenes)...", flush=True)
    packs = []
    for sc in scenes:
        vo = synth_voice_48k(voice, sc["narration"])
        dur_s = max(sc.get("min_s", 7.0), len(vo) / SR + 0.80)
        packs.append((sc, vo, dur_s))

    total_dur_s = sum(p[2] for p in packs)
    total_frames = int(round(total_dur_s * FPS))
    total_samples = int(round(total_frames / FPS * SR))

    music = generate_warm_bed(total_samples)
    speech_track = np.zeros(total_samples, dtype=np.float32)
    duck_env = np.ones(total_samples, dtype=np.float32) * 0.14

    cur_t = 0.0
    for sc, vo, dur_s in packs:
        s0 = int((cur_t + 0.25) * SR)
        s1 = min(total_samples, s0 + len(vo))
        if s1 > s0:
            speech_track[s0:s1] += vo[: s1 - s0]
            d0 = max(0, s0 - int(0.2 * SR))
            d1 = min(total_samples, s1 + int(0.3 * SR))
            duck_env[d0:d1] = 0.045
        cur_t += dur_s

    k_win = int(0.15 * SR)
    duck_smooth = np.convolve(duck_env, np.ones(k_win) / k_win, mode="same").astype(np.float32)
    mix_l = np.clip(speech_track + music[:, 0] * duck_smooth, -0.96, 0.96)
    mix_r = np.clip(speech_track + music[:, 1] * duck_smooth, -0.96, 0.96)
    stereo_pcm = np.vstack([mix_l, mix_r])

    container = av.open(str(out_mp4), mode="w", options={"movflags": "+faststart"})
    v_stream = container.add_stream("libx264", rate=FPS)
    v_stream.width, v_stream.height, v_stream.pix_fmt = W, H, "yuv420p"
    v_stream.options = {"crf": "17", "preset": "fast"}

    a_stream = container.add_stream("aac", rate=SR)
    a_stream.layout = "stereo"
    a_stream.bit_rate = 192000

    f_cursor = 0
    prev_last_frame = None
    saved_poster = False

    for s_idx, (sc, _, dur_s) in enumerate(packs, 1):
        n_sc = int(round(dur_s * FPS))
        frames_1080 = read_clip_1080p(P / "videos" / f"{sc['clip_id']}.mp4", n_sc)
        if not saved_poster and len(frames_1080) > 15:
            cv2.imwrite(str(POSTER_DIR / f"{out_mp4.stem}.jpg"), frames_1080[15], [cv2.IMWRITE_JPEG_QUALITY, 92])
            saved_poster = True

        for j, fr in enumerate(frames_1080):
            if j < 8 and prev_last_frame is not None:
                alpha = (j + 1) / 9.0
                fr = cv2.addWeighted(fr, alpha, prev_last_frame, 1.0 - alpha, 0)
            vf = av.VideoFrame.from_ndarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB), format="rgb24")
            for pkt in v_stream.encode(vf):
                container.mux(pkt)
            f_cursor += 1
        prev_last_frame = frames_1080[-1]
        print(f"  [{out_mp4.stem}] clip {s_idx}/{len(packs)} ({sc['clip_id']}) encoded", flush=True)

    for pkt in v_stream.encode():
        container.mux(pkt)

    actual_samples = int(round(f_cursor / FPS * SR))
    stereo_pcm = stereo_pcm[:, :actual_samples]
    for start in range(0, stereo_pcm.shape[1], 1024):
        chunk = np.ascontiguousarray(stereo_pcm[:, start : start + 1024])
        af = av.AudioFrame.from_ndarray(chunk, format="fltp", layout="stereo")
        af.sample_rate = SR
        for pkt in a_stream.encode(af):
            container.mux(pkt)
    for pkt in a_stream.encode():
        container.mux(pkt)

    container.close()
    print(f"DONE {out_mp4.name}: {f_cursor/FPS:.1f}s ({out_mp4.stat().st_size/1e6:.1f} MB)", flush=True)
    return f_cursor / FPS


def main() -> None:
    onnx_p = ROOT / "models/tts/voice_ryan.onnx"
    if not onnx_p.exists():
        onnx_p = ROOT / "models/tts/voice.onnx"
    voice = PiperVoice.load(str(onnx_p), config_path=str(onnx_p) + ".json")

    pitch_scenes = [
        {
            "clip_id": "final_032056",
            "narration": "Using an eight-servo robot arm and a table camera, our system locates the wet-wipe pack, catches the flip-top lid in one sideways stroke, pinches the top fold, and pulls a single tissue straight out.",
            "min_s": 13.0,
        },
        {
            "clip_id": "guided_033615",
            "narration": "Pinching right at the standing edge and pulling vertically extracts one thin sheet cleanly while leaving the pack flat on the table.",
            "min_s": 9.5,
        },
        {
            "clip_id": "guided_034731",
            "narration": "To clean spilled coffee beans without scattering them, the arm uses the rigid side of the packet itself like a squeegee, clearing twenty-one centimeters of table in a single pass.",
            "min_s": 11.0,
        },
        {
            "clip_id": "duster_005058",
            "narration": "Next, the arm grasps a magnetic duster, hovers twelve millimeters above each metal bottle cap so the magnet snaps them up without pressing down, verifies the table is clear, and drops them into the white box.",
            "min_s": 13.0,
        },
        {
            "clip_id": "teleop_pick2",
            "narration": "For desk organisation, the robot picks up markers and thin pens from the table and places them neatly inside the organiser box, using gripper stall position to sense object thickness.",
            "min_s": 11.5,
        },
        {
            "clip_id": "teleop_spoon_113019",
            "narration": "With synchronized table and wrist cameras at thirty frames per second, the arm grasps a wooden spoon, scoops coffee beans out of a bowl, and pours them cleanly into the black cup.",
            "min_s": 12.0,
        },
        {
            "clip_id": "sim_cup_hd",
            "narration": "In our calibrated physics twin, wrist-camera ray alignment reaches ninety-four percent success across one hundred cup-grasping trials.",
            "min_s": 8.5,
        },
        {
            "clip_id": "sim_pingpong_balance",
            "narration": "And a thirty-hertz reinforcement learning controller matched to our sideways racket grip balances a ping-pong ball and recovers smoothly from nudges.",
            "min_s": 8.0,
        },
    ]

    full_scenes = [
        pitch_scenes[0],
        {
            "clip_id": "full_022202",
            "narration": "Moving horizontally at constant height catches the hinged cap lip cleanly without pushing down on the pack, popping the lid wide open before pulling out the sheet.",
            "min_s": 11.5,
        },
        pitch_scenes[1],
        pitch_scenes[2],
        {
            "clip_id": "wipe_024526",
            "narration": "For wider spills, five parallel table-level sweeping lanes with lifted returns clear over ninety-five percent of the work zone.",
            "min_s": 10.5,
        },
        pitch_scenes[5],
        {
            "clip_id": "spoon_match_135344",
            "narration": "When the bowl moves to a new spot, live camera detection locates the handle, shifts the trajectory, grasps under load, and scoops beans across to the cup.",
            "min_s": 11.5,
        },
        pitch_scenes[3],
        pitch_scenes[4],
        {
            "clip_id": "auto_pen_marker",
            "narration": "Across repeated autonomous pen and marker runs, each cycle finishes in twenty-three seconds from live camera detection to placement in the box.",
            "min_s": 9.5,
        },
        pitch_scenes[6],
        pitch_scenes[7],
        {
            "clip_id": "sim_pingpong_dribble",
            "narration": "Fine-tuning the same thirty-hertz controller for rhythmic upward strikes achieves twenty-two consecutive ball bounces on the racket.",
            "min_s": 5.5,
        },
    ]

    # Remove old .srt files since user requested no captions
    for old_srt in VID_DIR.glob("youtube_*.srt"):
        old_srt.unlink()

    pitch_p = VID_DIR / "youtube_pitch_95s.mp4"
    full_p = VID_DIR / "youtube_full_showcase.mp4"
    build_clean_cut(voice, pitch_p, pitch_scenes)
    build_clean_cut(voice, full_p, full_scenes)

    shutil.copy(pitch_p, VID_DIR / "highlight_reel.mp4")
    shutil.copy(POSTER_DIR / "youtube_pitch_95s.jpg", POSTER_DIR / "reel.jpg")
    print("ALL CAPTION-FREE YOUTUBE VIDEOS READY!", flush=True)


if __name__ == "__main__":
    main()
