#!/usr/bin/env python
"""Build a static showcase website of everything recorded: showcase/index.html (+ H.264 copies of the videos, result photos).  Serve: ./norma showcase   (http://127.0.0.1:8892)
Honest by design: every item says what it really is, including failures, and which runs were NOT filmed."""
import glob, html, json, os, shutil, sys
from pathlib import Path
import cv2
ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT / "showcase"; (OUT / "videos").mkdir(parents=True, exist_ok=True); (OUT / "images").mkdir(exist_ok=True)
def h264(src, dst):
    if dst.exists() and dst.stat().st_size > 1000: return
    import av
    cap = cv2.VideoCapture(str(src)); fps = cap.get(cv2.CAP_PROP_FPS) or 8; ok, im = cap.read()
    if not ok: return
    h, w = im.shape[:2]; out = av.open(str(dst), "w"); st = out.add_stream("libx264", rate=max(1, int(round(fps)))); st.width, st.height, st.pix_fmt = w, h, "yuv420p"; st.options = {"crf": "23"}
    while ok:
        for p in st.encode(av.VideoFrame.from_ndarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), format="rgb24")): out.mux(p)
        ok, im = cap.read()
    for p in st.encode(): out.mux(p)
    out.close()
def dur(p):
    c = cv2.VideoCapture(str(p)); fps = c.get(cv2.CAP_PROP_FPS) or 1; return c.get(cv2.CAP_PROP_FRAME_COUNT) / fps
VIDEOS = [  # (section, source, title, caption)
 ("Real arm - teleoperation demonstrations (leader arm drives the follower)", "demo_backup/teleop_pick2_success.mp4", "Demo take 2 - SUCCESS", "A person moves the leader arm: marker picked up and placed in the white box. Used as the good training episode (LeRobot episode 0)."),
 ("Real arm - teleoperation demonstrations (leader arm drives the follower)", "demo_backup/teleop_pick1_failure.mp4", "Demo take 1 - FAILURE (placed wrongly)", "Same task, placed wrongly. Kept as a labelled failure example, not for training as a success (episode 1)."),
 ("Real arm - an early autonomous attempt that went wrong", "run/recordings/20261121_201411_RUN_marker/front.mp4", "First autonomous marker run - the arm pushed the box", "The table mapping had x mirrored; the alignment loop moved the wrong way and the jaws pushed the white box. Found, explained and fixed the same evening (see results notes)."),
 ("Simulation (MuJoCo twin built by the other agent)", "results/success_example.mp4", "Simulation - successful pick", "From the 100-trial MuJoCo benchmark."),
 ("Simulation (MuJoCo twin built by the other agent)", "results/pingpong_balance.mp4", "Simulation - balance demo", "Another simulation experiment in the repo."),
]
IMAGES = [  # (section, file, caption)
 ("Real arm - autonomous pen / marker into the white box (about 23 s from seeing the pen)", "pick_result.jpg", "First fully autonomous run: pen in the box (model check and photo)."),
 ("Real arm - autonomous pen / marker into the white box (about 23 s from seeing the pen)", "pick_result2.jpg", "Later run result."),
 ("Real arm - autonomous pen / marker into the white box (about 23 s from seeing the pen)", "pick_loop_result.jpg", "Retry loop: placed in the box; a hand takes it out to reset."),
 ("Real arm - autonomous pen / marker into the white box (about 23 s from seeing the pen)", "pick_loop_result2.jpg", "Retry loop: the marker rests across the box (rim)."),
 ("Real arm - autonomous pen / marker into the white box (about 23 s from seeing the pen)", "thin_pen_result.jpg", "Thin blue pen: held with a firmer grip, placed across the box."),
 ("Real arm - magnetic duster collects metal bottle caps and goes into the box", "duster_scene2.jpg", "Setup: duster with a magnet in its centre, three metal caps, the white box."),
 ("Real arm - magnetic duster collects metal bottle caps and goes into the box", "duster_detect2.jpg", "Colour/shape detection of the duster (green) and caps (magenta)."),
 ("Real arm - magnetic duster collects metal bottle caps and goes into the box", "diag_duster_crop.jpg", "Diagnosis photo: the wrist-camera board rests on the old, thicker duster, so the jaws only caught its edge."),
 ("Real arm - magnetic duster collects metal bottle caps and goes into the box", "duster_cap_check.jpg", "An earlier run: the duster slipped out on a fast move and landed upside down with a cap stuck on it."),
 ("Real arm - magnetic duster collects metal bottle caps and goes into the box", "duster_end2.jpg", "Earlier run: duster in the box, one cap still on the table."),
 ("Real arm - magnetic duster collects metal bottle caps and goes into the box", "duster_result.jpg", "FINAL RUN: duster grasped first try, held over both caps (no pressing), camera check, then into the box. Both caps are gone from the table (on the magnet underneath)."),
 ("Real arm - LEGO / Duplo: put the red brick on top of the yellow one (not finished)", "new_task_scene.jpg", "The task: two red Duplo bricks and a yellow-orange one on the table. Goal: pick a red brick, place it on the yellow one, press so the studs snap."),
 ("Real arm - LEGO / Duplo: put the red brick on top of the yellow one (not finished)", "bricks_detect.jpg", "Colour detection of the bricks (green boxes = position and angle of each brick, found in milliseconds)."),
 ("Real arm - LEGO / Duplo: put the red brick on top of the yellow one (not finished)", "diag_brick_bottom_crop.jpg", "Diagnosis photo at the grasp height: the open jaws around the red brick (the aim was right after the calibration fix)."),
 ("Real arm - LEGO / Duplo: put the red brick on top of the yellow one (not finished)", "stack_result.jpg", "Attempt: the red brick ended up beside the yellow one."),
 ("Real arm - LEGO / Duplo: put the red brick on top of the yellow one (not finished)", "stack_result2.jpg", "Attempt: the red brick landed on the yellow one but off-centre (1.5-4 cm table-level position error). One earlier run placed it perfectly, then an early press knocked it off."),
 ("Real arm - LEGO / Duplo: put the red brick on top of the yellow one (not finished)", "loop_attempt1.jpg", "Retry loop after a first grasp (1933 ticks, a full hold): the bricks had left the picture, so the loop stopped."),
]
cards = {}
for sec, src, title, cap in VIDEOS:
    p = ROOT / src
    if not p.exists(): continue
    dst = OUT / "videos" / (Path(src).parent.name + "_" + Path(src).name if "recordings" in src else Path(src).name); h264(p, dst)
    cards.setdefault(sec, []).append(f'<figure><video controls preload="metadata" playsinline src="videos/{html.escape(dst.name)}"></video><figcaption><b>{html.escape(title)}</b> <span class="d">{dur(dst):.0f} s</span><br>{html.escape(cap)}</figcaption></figure>')
for sec, f, cap in IMAGES:
    p = ROOT / "results/images" / f
    if not p.exists(): continue
    shutil.copy(p, OUT / "images" / f); cards.setdefault(sec, []).append(f'<figure><a href="images/{f}"><img loading="lazy" src="images/{f}"></a><figcaption>{html.escape(cap)}</figcaption></figure>')
for f in sorted(glob.glob(str(ROOT / "results/films/*.mp4"))):                    # films made with scripts/film_task.py
    f = Path(f); dst = OUT / "videos" / f.name; shutil.copy(f, dst); last = ""; lg = f.with_suffix(".log")
    if lg.exists(): last = (lg.read_text().strip().splitlines() or [""])[-1]
    cards.setdefault("Real arm - FILMED autonomous runs", []).append(f'<figure><video controls preload="metadata" playsinline src="videos/{html.escape(dst.name)}"></video><figcaption><b>{html.escape(f.stem.replace("_", " "))}</b> <span class="d">{dur(dst):.0f} s</span><br>Live caption = the run\'s own log. Last line: {html.escape(last[:110])}</figcaption></figure>')
ds = ROOT / "datasets/elrobot_teleop/meta/info.json"; info = json.loads(ds.read_text()) if ds.exists() else {}
page = f"""<!doctype html><html lang=en><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1"><title>ElRobot Showcase</title>
<style>:root{{--bg:#0f1219;--card:#181d28;--tx:#e8ecf3;--mut:#93a0b5;--acc:#6aa8ff}}body{{margin:0;background:var(--bg);color:var(--tx);font:15px/1.5 system-ui}}header{{padding:28px 20px 10px;max-width:1200px;margin:auto}}h1{{margin:0 0 6px;font-size:26px}}h2{{font-size:17px;margin:34px 0 12px;color:var(--acc)}}main{{max-width:1200px;margin:auto;padding:0 20px 60px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:16px}}figure{{margin:0;background:var(--card);border-radius:10px;overflow:hidden}}video,img{{width:100%;display:block;background:#000}}figcaption{{padding:10px 12px;color:var(--mut);font-size:13px}}figcaption b{{color:var(--tx)}}.d{{color:var(--acc)}}.note{{background:var(--card);border-left:3px solid var(--acc);padding:10px 14px;border-radius:6px;color:var(--mut);margin:10px 0}}code{{color:var(--tx)}}</style>
<header><h1>ElRobot pick &amp; place - what we built</h1><div style="color:var(--mut)">NormaCore hackathon, Berlin, 21-22 Nov 2026. A camera + a vision model + kinematics move a real 8-servo arm; everything below was recorded on the real robot unless it says simulation.</div></header><main>
<div class=note><b>Honest note:</b> only the teleoperation demos and one early failed attempt were filmed. The autonomous pen runs, the duster-and-caps run and the brick stacking were documented with photos and logs (no video) - the photos are the evidence. A video of a full autonomous run can be recorded with <code>scripts/record_run.py</code>.</div>
""" + "".join(f"<h2>{html.escape(sec)}</h2><div class=grid>{''.join(c)}</div>" for sec, c in sorted(cards.items(), key=lambda kv: 0 if kv[0].startswith("Real arm - FILMED") else 1)) + f"""
<h2>Data for training</h2><div class=note>LeRobot dataset v3.0 (<code>datasets/elrobot_teleop</code>, kept on disk, not in git): {info.get('total_episodes','?')} episodes, {info.get('total_frames','?')} frames at {info.get('fps','?')} fps. state = follower joints, action = leader joints, one table camera (640x480). Episode 0 = success, episode 1 = labelled failure - train on the success only. Source video was 8 fps (frames repeated to 30 fps).</div></main></html>"""
(OUT / "index.html").write_text(page); shutil.make_archive(str(ROOT / "showcase_bundle"), "zip", OUT); print("built (open this file in a browser, no server needed):", OUT / "index.html", "|", sum(len(c) for c in cards.values()), "items |", round(sum(f.stat().st_size for f in (OUT / 'videos').glob('*')) / 1e6, 1), "MB of videos")
