#!/usr/bin/env python
"""Build clean, natural light-theme presentation/index.html and presentation/slides.html (100% local, zero dark/blue AI styling).

Organised into the exact 6 separated sections:
  1. Hero Workflow: Autonomous Wet-Wipe Lid Open & Sheet Extraction
  2. Intelligent Spill Cleaning: Packet Squeegee & Surface Sweeping
  3. Autonomous Magnetic Bottle-Cap Collection
  4. Autonomous Pen & Marker Pick-and-Place
  5. Teleoperation & 30 fps LeRobot Training Datasets (strictly separate from autonomous)
  6. Physics Digital Twin & 30 Hz Racket Ball Control
"""
from __future__ import annotations

import html
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "presentation"
CAT = json.loads((P / "catalog.json").read_text())
EDITS = json.loads((P / "edits.json").read_text()) if (P / "edits.json").exists() else {}
ITEMS = {i["id"]: i for i in CAT["items"]}
E = html.escape

PHOTOS = {
    "magnet_caps": [
        ("duster_detect2.jpg", "Colour & Shape Detection", "Live segmentation of the duster handle (green) and metal bottle caps (magenta) in milliseconds."),
        ("duster_result.jpg", "Clean Table Result", "Duster placed in the white box with both metal caps captured underneath on the magnet — zero caps left on the table."),
    ],
    "pen_marker": [
        ("pick_result.jpg", "Autonomous Marker into Box (~23 s)", "First autonomous marker run: detected in RGB, grasped via 7-DoF IK, and placed cleanly inside the box."),
        ("thin_pen_result.jpg", "Thin Blue Pen — Stall-Tick Sensing", "Gripper stall reading distinguishes thin pens (1397 ticks) from thick markers (1570 ticks) and empty closes (1322 ticks)."),
        ("pick_loop_result.jpg", "Continuous Autonomous Reset & Pick Loop", "Places the marker into the box repeatedly as a person resets it to new positions on the table."),
    ],
    "sim_rl": [
        ("twin_vs_real_bat.jpg", "1:1 Sideways Racket Grip Match", "Real follower holding the JOOLA bat sideways (left) matched directly in the MuJoCo 3.2.7 twin (right)."),
    ],
}


def copy_photos() -> None:
    (P / "images").mkdir(parents=True, exist_ok=True)
    for src_dir in (ROOT / "results/images", ROOT / "results"):
        for lst in PHOTOS.values():
            for fname, _, _ in lst:
                sp = src_dir / fname
                if sp.exists():
                    shutil.copy(sp, P / "images" / fname)


def dur_tag(it: dict) -> str:
    e = EDITS.get(it["id"])
    return f"{e['out_dur']:.0f} s" if e else "HD"


def badge_cls(it: dict) -> str:
    v = it.get("verdict", "").lower()
    if "teleop" in v or "taught" in v or "reference" in v:
        return "b amber"
    return "b ok"


def card_html(it: dict) -> str:
    v = f"videos/{it['id']}.mp4"
    poster = f"posters/{it['id']}.jpg"
    pa = f' poster="{poster}"' if (P / poster).exists() else ""
    return (
        f'<figure class="card">'
        f'<video controls preload="metadata" playsinline{pa} src="{v}"></video>'
        f'<figcaption>'
        f'<div class="cmeta"><span class="{badge_cls(it)}">{E(it["verdict"])}</span><span class="dur">{dur_tag(it)}</span></div>'
        f'<div class="ctitle">{E(it["title"])}</div>'
        f'<p>{E(it["note"])}</p>'
        f'</figcaption></figure>'
    )


def photo_html(fname: str, title: str, cap: str) -> str:
    if not (P / "images" / fname).exists():
        return ""
    return (
        f'<figure class="card">'
        f'<a href="images/{fname}" target="_blank"><img loading="lazy" src="images/{fname}" alt=""></a>'
        f'<figcaption>'
        f'<div class="cmeta"><span class="b ok">Autonomous Photo</span></div>'
        f'<div class="ctitle">{E(title)}</div>'
        f'<p>{E(cap)}</p>'
        f'</figcaption></figure>'
    )


PIPELINE_SVG = """<svg viewBox="0 0 1140 210" class="arch" xmlns="http://www.w3.org/2000/svg" font-family="system-ui,-apple-system,Segoe UI,Roboto,sans-serif">
<defs>
  <marker id="arw" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6.5" markerHeight="6.5" orient="auto"><path d="M0,1 L9,5 L0,9 z" fill="#18181b"/></marker>
</defs>
<g fill="#18181b">
  <rect x="10" y="20" width="200" height="84" rx="10" fill="#ffffff" stroke="#d6d3cd" stroke-width="1.5"/>
  <text x="110" y="48" text-anchor="middle" font-weight="700" font-size="15">1. Live RGB Camera</text>
  <text x="110" y="70" text-anchor="middle" fill="#57534e" font-size="12.5">30 fps table + wrist views</text>
  <text x="110" y="88" text-anchor="middle" fill="#78716c" font-size="11.5">Measured fresh before every move</text>

  <rect x="242" y="20" width="210" height="84" rx="10" fill="#ffffff" stroke="#d6d3cd" stroke-width="1.5"/>
  <text x="347" y="48" text-anchor="middle" font-weight="700" font-size="15">2. 3D Table Grounding</text>
  <text x="347" y="70" text-anchor="middle" fill="#57534e" font-size="12.5">Multi-height calibration planes</text>
  <text x="347" y="88" text-anchor="middle" fill="#78716c" font-size="11.5">&lt; 5 mm wrist ray alignment</text>

  <rect x="484" y="20" width="224" height="84" rx="10" fill="#f0fdf4" stroke="#15803d" stroke-width="1.6"/>
  <text x="596" y="48" text-anchor="middle" font-weight="700" font-size="15" fill="#14532d">3. Task Skill Primitives</text>
  <text x="596" y="70" text-anchor="middle" fill="#166534" font-size="12.5">Lip sweep • Tip pinch • Packet push</text>
  <text x="596" y="88" text-anchor="middle" fill="#15803d" font-size="11.5">Magnet hover • 7-DoF IK grasp</text>

  <rect x="740" y="20" width="185" height="84" rx="10" fill="#ffffff" stroke="#d6d3cd" stroke-width="1.5"/>
  <text x="832" y="48" text-anchor="middle" font-weight="700" font-size="15">4. 8-Servo Control</text>
  <text x="832" y="70" text-anchor="middle" fill="#57534e" font-size="12.5">22 ms bus writes • Stall sense</text>
  <text x="832" y="88" text-anchor="middle" fill="#78716c" font-size="11.5">Current &amp; height safety checks</text>

  <rect x="957" y="20" width="172" height="84" rx="10" fill="#ffffff" stroke="#d6d3cd" stroke-width="1.5"/>
  <text x="1043" y="48" text-anchor="middle" font-weight="700" font-size="15">5. Visual Check</text>
  <text x="1043" y="70" text-anchor="middle" fill="#57534e" font-size="12.5">Before/after frame diff</text>
  <text x="1043" y="88" text-anchor="middle" fill="#78716c" font-size="11.5">Automatic re-index &amp; retry</text>

  <rect x="90" y="132" width="440" height="62" rx="10" fill="#faf9f6" stroke="#e5e3dc"/>
  <text x="310" y="158" text-anchor="middle" font-weight="700" font-size="13.5">30 fps Dual-Camera LeRobot v3 Datasets</text>
  <text x="310" y="178" text-anchor="middle" fill="#57534e" font-size="12">Synchronised table + wrist streams • 39,000+ frames recorded</text>

  <rect x="590" y="132" width="460" height="62" rx="10" fill="#faf9f6" stroke="#e5e3dc"/>
  <text x="820" y="158" text-anchor="middle" font-weight="700" font-size="13.5">Calibrated MuJoCo 3.2.7 Physics Twin</text>
  <text x="820" y="178" text-anchor="middle" fill="#57534e" font-size="12">6.79 mm real-arm trajectory fit • 94% cup pick • 95% 30 Hz ball balance</text>
</g>
<g stroke="#18181b" stroke-width="1.6" fill="none" marker-end="url(#arw)">
  <path d="M210 62 H237"/><path d="M452 62 H479"/><path d="M708 62 H735"/><path d="M925 62 H952"/>
</g>
</svg>"""


BASE_CSS = """
:root{--bg:#f8f7f4;--card:#ffffff;--sub:#f2f0ea;--border:#e4e1d8;--tx:#18181b;--mut:#57534e;--light:#78716c;--ok:#15803d;--okbg:#dcfce7;--amb:#b45309;--ambbg:#fef3c7}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--tx);font:15px/1.58 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;-webkit-font-smoothing:antialiased}
a{color:var(--tx);text-decoration:underline;text-underline-offset:3px}a:hover{color:var(--ok)}
.b{display:inline-block;padding:2px 9px;border-radius:6px;font-size:11.5px;font-weight:700;letter-spacing:.02em;text-transform:uppercase}
.b.ok{background:var(--okbg);color:var(--ok)}
.b.amber{background:var(--ambbg);color:var(--amb)}
"""


def build_gallery() -> None:
    sections_html = []
    for s in CAT["sections"]:
        cards = [card_html(i) for i in CAT["items"] if i["section"] == s["id"]]
        cards += [photo_html(f, t, c) for f, t, c in PHOTOS.get(s["id"], [])]
        cards = [c for c in cards if c]
        if not cards:
            continue
        sections_html.append(
            f'<section id="{s["id"]}" class="sec-block">'
            f'<div class="sec-head"><h2>{E(s["title"])}</h2><p>{E(s["line"])}</p></div>'
            f'<div class="grid">{"".join(cards)}</div></section>'
        )

    nav_links = "".join(
        f'<a href="#{s["id"]}">{E(s["title"].split(":")[0])}</a>' for s in CAT["sections"]
    )
    has_full = (P / "videos/youtube_full_showcase.mp4").exists()
    full_btn = (
        '<button class="cut-btn" onclick="setCut(\'videos/youtube_full_showcase.mp4\',\'posters/youtube_full_showcase.jpg\',this)">'
        'Full Showcase (2m 50s, Voiced)</button>'
    ) if has_full else ""

    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Autonomous House Helper — ElRobot</title>
<style>{BASE_CSS}
header{{max-width:1260px;margin:auto;padding:40px 24px 18px;border-bottom:1px solid var(--border);display:flex;justify-content:space-between;align-items:flex-end;flex-wrap:wrap;gap:16px}}
.kicker{{color:var(--ok);font-weight:700;font-size:12.5px;letter-spacing:.08em;text-transform:uppercase}}
h1{{margin:6px 0 6px;font-size:34px;font-weight:800;letter-spacing:-.025em;color:var(--tx)}}
.sub{{color:var(--mut);font-size:16px;max-width:760px}}
.btn-deck{{background:var(--tx);color:#fff;padding:10px 20px;border-radius:8px;font-weight:600;font-size:14px;text-decoration:none}}
.btn-deck:hover{{background:#27272a;color:#fff}}
main{{max-width:1260px;margin:auto;padding:24px 24px 80px}}
.hero-grid{{display:grid;grid-template-columns:1.55fr 1fr;gap:24px;margin-bottom:28px;align-items:stretch}}
.player-card{{background:var(--card);border:1px solid var(--border);border-radius:14px;padding:14px;box-shadow:0 4px 18px rgba(0,0,0,.04)}}
.player-card video{{width:100%;aspect-ratio:16/9;border-radius:8px;background:#eae8e1;display:block}}
.cut-tabs{{display:flex;gap:10px;margin-top:12px;flex-wrap:wrap;align-items:center;justify-content:space-between}}
.tab-group{{display:flex;gap:8px;flex-wrap:wrap}}
.cut-btn{{background:var(--sub);color:var(--mut);border:1px solid var(--border);border-radius:8px;padding:7px 14px;cursor:pointer;font:600 13px inherit}}
.cut-btn.active{{background:var(--tx);color:#fff;border-color:var(--tx)}}
.dl-links{{font-size:12.5px;color:var(--light)}}
.summary-card{{background:var(--card);border:1px solid var(--border);border-radius:14px;padding:24px;display:flex;flex-direction:column;justify-content:space-between;box-shadow:0 4px 18px rgba(0,0,0,.04)}}
.kpi-list{{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin:16px 0}}
.kpi{{background:var(--sub);border-radius:10px;padding:12px 14px}}
.kpi b{{display:block;font-size:21px;color:var(--tx)}}
.kpi span{{font-size:12.5px;color:var(--mut)}}
.arch-card{{background:var(--card);border:1px solid var(--border);border-radius:14px;padding:20px 24px;margin-bottom:28px;box-shadow:0 4px 18px rgba(0,0,0,.03)}}
.arch-card h3{{margin:0 0 10px;font-size:16px;font-weight:700;color:var(--tx)}}
.arch{{width:100%;height:auto;display:block}}
nav.top-nav{{position:sticky;top:0;z-index:10;background:rgba(248,247,244,.94);backdrop-filter:blur(8px);border-bottom:1px solid var(--border);padding:12px 0;display:flex;gap:8px 12px;flex-wrap:wrap}}
nav.top-nav a{{text-decoration:none;font-weight:600;font-size:13px;color:var(--mut);padding:6px 12px;border-radius:8px;background:var(--card);border:1px solid var(--border)}}
nav.top-nav a:hover{{background:var(--tx);color:#fff;border-color:var(--tx)}}
.sec-block{{margin-top:44px}}
.sec-head h2{{margin:0 0 4px;font-size:22px;font-weight:800;letter-spacing:-.015em}}
.sec-head p{{margin:0 0 18px;color:var(--mut);max-width:880px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(350px,1fr));gap:20px}}
.card{{margin:0;background:var(--card);border:1px solid var(--border);border-radius:12px;overflow:hidden;display:flex;flex-direction:column;box-shadow:0 2px 10px rgba(0,0,0,.03)}}
.card video,.card img{{width:100%;aspect-ratio:4/3;object-fit:cover;display:block;background:#eae8e1}}
figcaption{{padding:14px 16px 16px;display:flex;flex-direction:column;gap:6px}}
.cmeta{{display:flex;justify-content:space-between;align-items:center}}
.dur{{font-size:12px;color:var(--light);font-weight:600}}
.ctitle{{font-weight:700;font-size:15.5px;color:var(--tx);line-height:1.35}}
figcaption p{{margin:0;color:var(--mut);font-size:13.5px;line-height:1.52}}
@media(max-width:960px){{.hero-grid{{grid-template-columns:1fr}}}}
</style></head>
<body>
<header>
  <div>
    <div class="kicker">NormaCore Robotics Hackathon  •  Berlin</div>
    <h1>Autonomous House Helper</h1>
    <div class="sub">An 8-servo robot arm and table camera that open flip-top wet-wipe packs, pull single tissues, sweep spilled coffee beans, collect bottle caps magnetically, scoop beans into a cup, and tidy pens.</div>
  </div>
  <a class="btn-deck" href="slides.html">Open Slide Deck &rarr;</a>
</header>
<main>
  <div class="hero-grid">
    <div class="player-card">
      <video id="heroVid" controls playsinline preload="metadata" poster="posters/youtube_pitch_95s.jpg" src="videos/youtube_pitch_95s.mp4"></video>
      <div class="cut-tabs">
        <div class="tab-group">
          <button class="cut-btn active" onclick="setCut('videos/youtube_pitch_95s.mp4','posters/youtube_pitch_95s.jpg',this)">Pitch Cut (1m 35s, Voiced)</button>
          {full_btn}
        </div>
        <div class="dl-links">
          YouTube MP4: <a href="videos/youtube_pitch_95s.mp4" download>Pitch</a> &middot; <a href="videos/youtube_full_showcase.mp4" download>Full</a>
        </div>
      </div>
    </div>
    <div class="summary-card">
      <div>
        <div class="kicker">What Works on the Real Robot</div>
        <h3 style="margin:6px 0 8px;font-size:20px">Six Focused Capabilities, Clearly Separated</h3>
        <p style="margin:0;color:var(--mut);font-size:14px">Autonomous runs are separated strictly from teleoperation datasets. Every clip below is clean 30 fps H.264 camera footage stored locally.</p>
        <div class="kpi-list">
          <div class="kpi"><b>Wet-Wipes</b><span>Flip-top open &amp; single-sheet pull</span></div>
          <div class="kpi"><b>Packet Squeegee</b><span>21 cm one-stroke bean sweep</span></div>
          <div class="kpi"><b>Magnet &amp; Pens</b><span>Zero-press cap pickup &amp; pen tidy</span></div>
          <div class="kpi"><b>Twin &amp; 30 Hz RL</b><span>94% cup pick &amp; 95% bat balance</span></div>
        </div>
      </div>
      <div style="font-size:12.5px;color:var(--light);border-top:1px solid var(--border);padding-top:10px">
        100% local offline bundle &middot; Clean 1080p 30 fps H.264 + AAC videos in <code>presentation/videos/</code>
      </div>
    </div>
  </div>

  <div class="arch-card">
    <h3>System Pipeline — From Live Camera Frame to Verified Physical Action</h3>
    {PIPELINE_SVG}
  </div>

  <nav class="top-nav">{nav_links}</nav>
  {"".join(sections_html)}
</main>
<script>
function setCut(src, poster, btn) {{
  const v = document.getElementById('heroVid');
  v.pause(); v.src = src; v.poster = poster; v.load(); v.play().catch(()=>{{}});
  document.querySelectorAll('.cut-btn').forEach(b => b.classList.toggle('active', b === btn));
}}
</script>
</body></html>"""
    (P / "index.html").write_text(page)


def vtile(iid: str) -> str:
    it = ITEMS[iid]
    pa = f' poster="posters/{iid}.jpg"' if (P / f"posters/{iid}.jpg").exists() else ""
    return (
        f'<figure class="vt">'
        f'<video muted loop playsinline controls preload="metadata"{pa} data-src="videos/{iid}.mp4"></video>'
        f'<figcaption><span class="{badge_cls(it)}">{E(it["verdict"])}</span> {E(it["title"])}</figcaption></figure>'
    )


def ptile(fname: str, label: str, title: str) -> str:
    return (
        f'<figure class="vt">'
        f'<img src="images/{fname}" alt="">'
        f'<figcaption><span class="b ok">{E(label)}</span> {E(title)}</figcaption></figure>'
    )


def slide(title: str, inner: str, kicker: str = "", cls: str = "") -> str:
    return (
        f'<section class="slide {cls}">'
        + (f'<div class="kick">{E(kicker)}</div>' if kicker else "")
        + (f"<h2>{title}</h2>" if title else "")
        + inner
        + "</section>"
    )


def build_slides() -> None:
    S = [
        slide(
            "",
            '<div class="hero">'
            '<div class="kick">NormaCore Robotics Hackathon  •  Berlin</div>'
            '<h1>Autonomous House Helper</h1>'
            '<p class="lead">One 8-servo robot arm and one table camera: opening flip-top wet-wipe packs, pulling single tissues,<br>sweeping spilled coffee beans, collecting bottle caps magnetically, scooping beans, and tidying pens.</p>'
            '<div class="pills"><span>1. Lid Open &amp; Tissue Pull</span><span>2. Coffee-Bean Cleaning</span><span>3. Magnetic Cap Pickup</span><span>4. Pen &amp; Marker Tidy</span><span>5. Physics Twin &amp; 30 Hz RL</span></div>'
            '</div>',
            cls="title",
        ),
        slide(
            "How It Works — Local Closed-Loop Pipeline",
            PIPELINE_SVG
            + '<ul class="cols">'
            '<li><b>Fresh RGB Grounding:</b> Object poses are measured right before motion via calibrated height planes (4, 6, 10 cm) — never hardcoded.</li>'
            '<li><b>Right Tool Mechanics:</b> Horizontal lip strokes open hinged caps; vertical pulls extract soft wipes; the rigid packet edge sweeps loose beans.</li>'
            '<li><b>Honest Visual Checks:</b> Thin wipes do not stall servo current, so before/after camera checks verify lid opening, sheet hold, and clean tables.</li>'
            '</ul>',
            "ARCHITECTURE",
        ),
        slide(
            "Step 1: Autonomous Pen &amp; Marker Pick-and-Place",
            '<div class="g3">' + vtile("auto_pen_marker") + ptile("pick_result.jpg", "23 s Run", "Autonomous Marker into Box") + ptile("thin_pen_result.jpg", "Stall Sense", "Thin Pen (1397) vs Marker (1570)") + '</div>'
            '<p class="note">Autonomous camera + 7-DoF IK runs (~23 s cycle; motor-8 stall ticks distinguish thin blue pens at 1397 ticks from thick markers at 1570 ticks).</p>',
            "01 • AUTONOMOUS DESK ORGANISATION",
        ),
        slide(
            "Step 2: Center-Magnet Duster Bottle-Cap Cleanup",
            '<div class="g3">' + vtile("duster_005058") + ptile("duster_detect2.jpg", "Detection", "Live Duster &amp; Cap Segmentation") + ptile("duster_result.jpg", "Clean Table", "Both Caps Collected &amp; Binned") + '</div>'
            '<p class="note">Because the magnet sits only in the dead center of the duster, the robot aligns that center 1.2 cm above each cap so they snap up without pressing, verifies 0 caps remain, and bins them.</p>',
            "02 • CENTER-MAGNET BOTTLE-CAP PICKUP",
        ),
        slide(
            "Step 3: Chained Flip-Top Lid Open &amp; Tissue Extraction",
            '<div class="g3">' + vtile("guided_030809") + vtile("final_032056") + vtile("full_022202") + '</div>'
            '<p class="note">A single horizontal stroke at z = 10 cm catches the flip-top lip and pops the cap wide open; the gripper immediately pinches the standing fold at z = 7.6 cm and pulls one tissue straight out.</p>',
            "03 • HERO DEFORMABLE MANIPULATION",
        ),
        slide(
            "Step 4: Spill Adaptation — 5-Lane Tissue Wipe vs. Smart Box Push &amp; Spoon Scoop",
            '<div class="g3">' + vtile("wipe_024526") + vtile("guided_034731") + vtile("teleop_spoon_113019") + '</div>'
            '<p class="note">First it wiped spilled beans in a 5-lane tissue pattern; then it adapted intelligently — pushing the rigid packet box sideways as a 21 cm squeegee! Plus dual-camera 30 fps bowl-to-cup spoon scooping.</p>',
            "04 • EMERGENT TOOL ADAPTATION &amp; BEAN TRANSFER",
        ),
        slide(
            "Step 5: Side-by-Side Digital Twin &amp; 30 Hz Table-Tennis RL",
            '<div class="g3">' + vtile("twin_sidebyside") + vtile("sim_cup_hd") + vtile("sim_pingpong_balance") + '</div>'
            '<p class="note">Side-by-side MuJoCo 3.2.7 twin (6.79 mm fit, 94% cup benchmark) and our main dynamic goal: 30 Hz RL racket ball balancing (95% in twin; real 12 V servos lacked holding torque for the outstretched bat).</p>',
            "05 • DIGITAL TWIN &amp; 30 HZ REINFORCEMENT LEARNING",
        ),
    ]

    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Autonomous House Helper — Slide Deck</title>
<style>{BASE_CSS}
html,body{{height:100%;overflow:hidden;background:#eae8e2}}
#stage{{position:absolute;left:50%;top:50%;width:1280px;height:720px;transform-origin:center}}
.slide{{position:absolute;inset:0;background:var(--bg);border:1px solid #d6d3cd;border-radius:14px;padding:44px 56px;display:none;flex-direction:column;overflow:hidden;box-shadow:0 12px 40px rgba(0,0,0,.08)}}
.slide.on{{display:flex}}
.kick{{color:var(--ok);font-weight:700;letter-spacing:.09em;font-size:12.5px;text-transform:uppercase;margin-bottom:6px}}
h2{{margin:0 0 16px;font-size:32px;font-weight:800;letter-spacing:-.02em;color:var(--tx)}}
h1{{font-size:56px;line-height:1.06;margin:10px 0 18px;font-weight:800;letter-spacing:-.03em;color:var(--tx)}}
.hero{{margin:auto 0}}.lead{{font-size:21px;color:var(--mut);line-height:1.52}}
.pills span{{display:inline-block;background:var(--card);border:1px solid var(--border);color:var(--tx);border-radius:8px;padding:8px 15px;margin:18px 10px 0 0;font-size:14px;font-weight:600}}
.vt{{margin:0;background:var(--card);border:1px solid var(--border);border-radius:12px;overflow:hidden;display:flex;flex-direction:column;min-height:0}}
.vt video,.vt img{{width:100%;aspect-ratio:4/3;object-fit:cover;background:#e5e3dc;display:block}}
.vt figcaption{{padding:10px 12px;font-size:13px;font-weight:600;color:var(--tx);display:flex;gap:8px;align-items:center}}
.g3{{display:grid;grid-template-columns:repeat(3,1fr);gap:18px;flex:1;min-height:0;align-items:center;align-content:center}}
.note{{color:var(--mut);font-size:16.5px;line-height:1.48;margin:14px 0 0;border-left:3px solid var(--ok);padding-left:14px}}
.arch{{width:100%;height:auto;margin:6px 0 14px}}
.cols{{display:grid;grid-template-columns:repeat(3,1fr);gap:18px;padding:0;list-style:none;font-size:15px;color:var(--mut);margin:0}}
.cols li{{background:var(--card);border:1px solid var(--border);padding:14px 16px;border-radius:10px}}
.cols b{{color:var(--tx)}}
#bar{{position:fixed;left:0;bottom:0;height:4px;background:var(--ok);transition:width .25s}}
#num{{position:fixed;right:18px;bottom:10px;color:var(--light);font-size:13px;font-weight:600}}
#help{{position:fixed;left:18px;bottom:10px;color:var(--light);font-size:12.5px}}
</style></head>
<body>
<div id="stage">{"".join(S)}</div><div id="bar"></div><div id="num"></div>
<div id="help">&larr; &rarr; / Space = navigate &middot; F = fullscreen &middot; <a href="index.html">Open Full Page</a></div>
<script>
const sl=[...document.querySelectorAll('.slide')];let k=Math.max(0,Math.min(sl.length-1,(parseInt(location.hash.slice(1))||1)-1));
function fit(){{const s=Math.min(innerWidth/1280,innerHeight/720)*0.96;document.getElementById('stage').style.transform=`translate(-50%,-50%) scale(${{s}})`}}
function show(n){{k=Math.max(0,Math.min(sl.length-1,n));sl.forEach((s,i)=>{{s.classList.toggle('on',i===k);s.querySelectorAll('video').forEach(v=>{{if(i===k){{if(v.dataset.src&&!v.src)v.src=v.dataset.src;if(v.hasAttribute('muted'))v.play().catch(()=>{{}})}}else v.pause()}})}});
document.getElementById('bar').style.width=((k+1)/sl.length*100)+'%';document.getElementById('num').textContent=(k+1)+' / '+sl.length;history.replaceState(null,'','#'+(k+1))}}
addEventListener('resize',fit);addEventListener('keydown',e=>{{if(['ArrowRight','PageDown',' '].includes(e.key)){{e.preventDefault();show(k+1)}}else if(['ArrowLeft','PageUp'].includes(e.key))show(k-1);else if(e.key==='f')(document.fullscreenElement?document.exitFullscreen():document.documentElement.requestFullscreen())}});
fit();show(k);
</script></body></html>"""
    (P / "slides.html").write_text(page)


if __name__ == "__main__":
    copy_photos()
    build_gallery()
    build_slides()
    print("Built natural light-theme presentation:", P / "index.html", "and", P / "slides.html", flush=True)
