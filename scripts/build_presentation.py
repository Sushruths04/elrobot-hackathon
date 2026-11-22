#!/usr/bin/env python
"""Build presentation/index.html (full gallery) and presentation/slides.html (slide deck) from presentation/catalog.json + presentation/edits.json.
Run scripts/presentation_edit.py first (it makes the edited videos, posters and the highlight reel). Everything is local and works offline (open the files in a browser)."""
import html, json, shutil
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]; P = ROOT / "presentation"; CAT = json.loads((P / "catalog.json").read_text())
EDITS = json.loads((P / "edits.json").read_text()) if (P / "edits.json").exists() else {}
SEC = {s["id"]: s for s in CAT["sections"]}; ITEMS = {i["id"]: i for i in CAT["items"]}; E = html.escape
REAL = [i for i in CAT["items"] if i["section"] not in ("task", "sim", "appendix")]
PHOTOS = {  # photo-only evidence (runs that were not filmed): section -> [(file in results/images, caption)]
 "marker": [("pick_result.jpg", "Autonomous pen run (not filmed): pen in the box, model check + photo."), ("thin_pen_result.jpg", "Thin blue pen: held with a firmer grip (1397 ticks), placed across the box."), ("pick_loop_result.jpg", "Retry loop: placed in the box; a hand takes it out to reset.")],
 "duster": [("duster_detect2.jpg", "Colour/shape detection of the duster (green) and the caps (magenta)."), ("duster_result.jpg", "Final photo of the successful run: duster in the box, both caps gone from the table (on the magnet underneath).")],
 "lego": [("new_task_scene.jpg", "The task: put a red Duplo brick on the yellow one."), ("bricks_detect.jpg", "Colour detection: position and angle of each brick in milliseconds."), ("stack_result2.jpg", "Attempt: on the yellow brick but off-centre (1.5-4 cm table-level error). Not finished.")],
}
LEGO = {"id": "lego", "title": "LEGO / Duplo stacking (photos only, not finished)", "line": "One run placed the brick perfectly, then a press knocked it off; later runs were 1.5-4 cm off."}
LABEL = {"success": "Success", "partial": "Partial", "failure": "Failure", "info": "Other"}

def copy_photos():
    (P / "images").mkdir(exist_ok=True)
    for lst in PHOTOS.values():
        for f, _ in lst:
            if (ROOT / "results/images" / f).exists(): shutil.copy(ROOT / "results/images" / f, P / "images" / f)

def edit_line(i):
    e = EDITS.get(i["id"])
    if not e: return ""
    parts = []
    if e["lead_cut"] >= 1 or e["tail_cut"] >= 1: parts.append(f"start {e['lead_cut']:.0f} s / end {e['tail_cut']:.0f} s trimmed")
    if e["cuts"]: parts.append(f"{len(e['cuts'])} idle gap{'s' if len(e['cuts']) > 1 else ''} cut ({sum(e['cuts']):.0f} s)")
    if e["speed"] > 1: parts.append(f"{e['speed']:g}x speed")
    return f"Original {e['src_dur']:.0f} s &rarr; {e['out_dur']:.0f} s" + (": " + ", ".join(parts) if parts else " (unchanged)")

def badge(i): return f'<span class="b {i["outcome"]}">{E(i["verdict"])}</span>'

def card(i):
    v = f'videos/{i["id"]}.mp4'; poster = f'posters/{i["id"]}.jpg'
    has_v = (P / v).exists(); pa = f' poster="{poster}"' if (P / poster).exists() else ""
    media = f'<video controls preload="none" playsinline{pa} src="{v}"></video>' if has_v else '<div class="nov">video could not be read</div>'
    return (f'<figure class="card" data-o="{i["outcome"]}">{media}<figcaption><div class="t">{E(i["title"])}</div><div>{badge(i)} <span class="by">verdict: {E(i["by"])}</span></div>'
            f'<p>{E(i["note"])}</p><div class="ed">{edit_line(i)} &middot; <a href="../{E(i["src"])}">original file</a></div></figcaption></figure>')

def photo(f, cap): return f'<figure class="card photo" data-o="info"><a href="images/{f}"><img loading="lazy" src="images/{f}" alt=""></a><figcaption><span class="b info">Photo (not filmed)</span><p>{E(cap)}</p></figcaption></figure>'

CSS = """:root{--bg:#0d1017;--card:#161b26;--card2:#1c2331;--tx:#e9edf4;--mut:#94a0b6;--acc:#6aa8ff;--ok:#2ea043;--part:#d29614;--bad:#c83232;--info:#3c6ec8}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--tx);font:15px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
a{color:var(--acc)}.b{display:inline-block;padding:1px 8px;border-radius:99px;font-size:12px;font-weight:600;color:#fff}.b.success{background:var(--ok)}.b.partial{background:var(--part)}.b.failure{background:var(--bad)}.b.info{background:var(--info)}"""

def gallery():
    secs = [s for s in CAT["sections"]]; secs.insert(3, LEGO); body = []
    c = Counter(i["outcome"] for i in REAL)
    for s in secs:
        cards = [card(i) for i in CAT["items"] if i["section"] == s["id"]] + [photo(f, cap) for f, cap in PHOTOS.get(s["id"], []) if (P / "images" / f).exists()]
        if not cards: continue
        body.append(f'<section id="{s["id"]}"><h2>{E(s["title"])}</h2><p class="line">{E(s["line"])}</p><div class="grid">{"".join(cards)}</div></section>')
    np_ = "".join(f"<li><code>{E(x['path'])}</code>: {E(x['why'])}</li>" for x in CAT["not_playable"]); dup = "".join(f"<li><code>{E(x['path'])}</code> = {E(x['same_as'])}</li>" for x in CAT["duplicates"])
    reel = EDITS.get("_reel", {}); reel_html = f'<video controls preload="metadata" playsinline poster="posters/reel.jpg" src="videos/highlight_reel.mp4"></video>' if (P / "videos/highlight_reel.mp4").exists() else ""
    nav = " ".join(f'<a href="#{s["id"]}">{E(s["title"].split(":")[0].split("(")[0].strip())}</a>' for s in secs)
    page = f"""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>ElRobot - every recorded run</title>
<style>{CSS}
header{{max-width:1240px;margin:auto;padding:30px 22px 6px}}h1{{margin:0 0 4px;font-size:28px}}.sub{{color:var(--mut)}}main{{max-width:1240px;margin:auto;padding:0 22px 70px}}
.top{{display:grid;grid-template-columns:1.35fr 1fr;gap:20px;margin:18px 0}}.top video{{width:100%;border-radius:12px;background:#000}}.panel{{background:var(--card);border-radius:12px;padding:16px 18px}}
.stats{{display:flex;gap:10px;flex-wrap:wrap;margin:8px 0 12px}}.stat{{background:var(--card2);border-radius:10px;padding:8px 12px;min-width:86px}}.stat b{{display:block;font-size:22px}}
nav.sec{{position:sticky;top:0;z-index:5;background:rgba(13,16,23,.94);backdrop-filter:blur(6px);padding:10px 0;display:flex;gap:6px 14px;flex-wrap:wrap;font-size:13px;border-bottom:1px solid #222a38}}
.filters{{display:flex;gap:8px;margin:14px 0 0;flex-wrap:wrap}}.filters button{{background:var(--card2);color:var(--tx);border:1px solid #2a3344;border-radius:99px;padding:4px 12px;cursor:pointer;font:inherit;font-size:13px}}.filters button.on{{background:var(--acc);color:#081020;border-color:var(--acc)}}
h2{{font-size:19px;margin:38px 0 2px;color:var(--acc)}}.line{{color:var(--mut);margin:0 0 12px}}.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(330px,1fr));gap:16px}}
.card{{margin:0;background:var(--card);border-radius:12px;overflow:hidden;display:flex;flex-direction:column}}.card video,.card img{{width:100%;aspect-ratio:4/3;display:block;background:#000;object-fit:cover}}
.nov{{aspect-ratio:4/3;display:grid;place-items:center;color:var(--mut);background:#000}}figcaption{{padding:10px 13px 12px;color:var(--mut);font-size:13px}}figcaption p{{margin:6px 0}}.t{{color:var(--tx);font-weight:600;font-size:14.5px;margin-bottom:4px}}
.by{{font-size:12px}}.ed{{font-size:12px;color:#7d889c;border-top:1px solid #232b3a;padding-top:6px;margin-top:6px}}.hide{{display:none}}ul.small{{color:var(--mut);font-size:13px}}code{{color:#cfd7e6;font-size:12px}}
@media(max-width:860px){{.top{{grid-template-columns:1fr}}}}</style>
<header><h1>ElRobot: every recorded run</h1><div class="sub">NormaCore hackathon, Berlin, 21-22 Nov 2026 &middot; real 8-servo arm, one table camera, a vision model and kinematics &middot; <a href="slides.html">slide deck</a></div></header>
<main><div class="top"><div>{reel_html}<div class="sub" style="font-size:13px;margin-top:6px">Highlight reel ({reel.get('reel_s', 0):.0f} s): title cards + the best runs and the key failures, trimmed and sped up.</div></div>
<div class="panel"><b>Real-robot runs in this gallery</b><div class="stats"><div class="stat"><b>{len(REAL)}</b>runs</div><div class="stat"><b style="color:var(--ok)">{c['success']}</b>success</div><div class="stat"><b style="color:var(--part)">{c['partial']}</b>partial</div><div class="stat"><b style="color:var(--bad)">{c['failure']}</b>failure</div></div>
<p style="margin:0 0 8px;color:var(--mut);font-size:13.5px">Each badge says <b>who decided</b>: <i>user</i> = the person at the table said so; <i>judged from video</i> = no human verdict was recorded, so the end frames were checked by eye for this page. The <code>OUTCOME.json</code> files of the datasets were not changed.</p>
<p style="margin:0;color:var(--mut);font-size:13.5px">Edits: idle start/end trimmed, idle gaps longer than 3 s cut (marked in the video), long clips sped up (marked). The burned-in <code>t = ...</code> clock still shows the original timing. Every card links to its untouched original.</p></div></div>
<nav class="sec">{nav}</nav><div class="filters"><button class="on" data-f="all">All</button><button data-f="success">Success</button><button data-f="partial">Partial</button><button data-f="failure">Failure</button><button data-f="info">Other / photos</button></div>
{''.join(body)}
<section><h2>Files that are not shown</h2><p class="line">Nothing was deleted. These exist on disk but cannot be played, or are copies of videos above.</p><ul class="small">{np_}{dup}</ul>
<p class="line">Built by <code>scripts/presentation_edit.py</code> (edits, posters, reel) and <code>scripts/build_presentation.py</code> (pages) from <code>presentation/catalog.json</code>.</p></section></main>
<script>document.querySelectorAll('.filters button').forEach(b=>b.onclick=()=>{{document.querySelectorAll('.filters button').forEach(x=>x.classList.toggle('on',x===b));const f=b.dataset.f;
document.querySelectorAll('.card').forEach(c=>c.classList.toggle('hide',f!=='all'&&c.dataset.o!==f));document.querySelectorAll('main section[id]').forEach(s=>s.classList.toggle('hide',!s.querySelector('.card:not(.hide)')))}});</script></html>"""
    (P / "index.html").write_text(page)

def vtile(iid, big=False):
    i = ITEMS[iid]; pa = f' poster="posters/{iid}.jpg"' if (P / f"posters/{iid}.jpg").exists() else ""
    return f'<figure class="vt{" big" if big else ""}"><video muted loop playsinline preload="none"{pa} data-src="videos/{iid}.mp4"></video><figcaption>{badge(i)} {E(i["title"])}</figcaption></figure>'

def slide(title, inner, kicker="", cls=""):
    return f'<section class="slide {cls}">' + (f'<div class="kick">{E(kicker)}</div>' if kicker else "") + (f"<h2>{title}</h2>" if title else "") + inner + "</section>"

def scoreboard():
    rows = []
    for s in CAT["sections"]:
        if s["id"] in ("task", "sim", "appendix"): continue
        c = Counter(i["outcome"] for i in REAL if i["section"] == s["id"])
        rows.append(f'<tr><td>{E(s["title"])}</td><td>{sum(c.values())}</td><td class="ok">{c["success"]}</td><td class="pa">{c["partial"]}</td><td class="ba">{c["failure"]}</td><td class="mu">{c["info"]}</td></tr>')
    c = Counter(i["outcome"] for i in REAL)
    rows.append(f'<tr class="tot"><td>All filmed real-robot runs</td><td>{len(REAL)}</td><td class="ok">{c["success"]}</td><td class="pa">{c["partial"]}</td><td class="ba">{c["failure"]}</td><td class="mu">{c["info"]}</td></tr>')
    return '<table class="score"><tr><th>Task</th><th>Runs</th><th>Success</th><th>Partial</th><th>Failure</th><th>Other</th></tr>' + "".join(rows) + "</table>"

ARCH = """<svg viewBox="0 0 1160 300" class="arch" xmlns="http://www.w3.org/2000/svg" font-family="system-ui,sans-serif">
<defs><marker id="ar" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0L10,5L0,10z" fill="#6aa8ff"/></marker></defs>
<g font-size="15" fill="#e9edf4">
<rect x="10" y="40" width="190" height="80" rx="12" fill="#1c2331" stroke="#3a4760"/><text x="105" y="74" text-anchor="middle" font-weight="700">Table camera</text><text x="105" y="98" text-anchor="middle" fill="#94a0b6" font-size="13">640x480, fresh photo</text>
<rect x="250" y="40" width="210" height="80" rx="12" fill="#1c2331" stroke="#3a4760"/><text x="355" y="70" text-anchor="middle" font-weight="700">Perception</text><text x="355" y="92" text-anchor="middle" fill="#94a0b6" font-size="13">vision model (median of 3)</text><text x="355" y="110" text-anchor="middle" fill="#94a0b6" font-size="13">+ colour detectors</text>
<rect x="510" y="40" width="200" height="80" rx="12" fill="#1c2331" stroke="#3a4760"/><text x="610" y="70" text-anchor="middle" font-weight="700">Pixel to table xy</text><text x="610" y="94" text-anchor="middle" fill="#94a0b6" font-size="13">calibration planes</text><text x="610" y="111" text-anchor="middle" fill="#94a0b6" font-size="13">z = 4 / 6 / 10 cm</text>
<rect x="760" y="40" width="180" height="80" rx="12" fill="#1c2331" stroke="#3a4760"/><text x="850" y="70" text-anchor="middle" font-weight="700">IK + path check</text><text x="850" y="94" text-anchor="middle" fill="#94a0b6" font-size="13">MuJoCo kinematics</text><text x="850" y="111" text-anchor="middle" fill="#94a0b6" font-size="13">tip never below 3 cm</text>
<rect x="985" y="40" width="165" height="80" rx="12" fill="#1c2331" stroke="#3a4760"/><text x="1067" y="70" text-anchor="middle" font-weight="700">armd daemon</text><text x="1067" y="94" text-anchor="middle" fill="#94a0b6" font-size="13">~25 ms per call</text><text x="1067" y="111" text-anchor="middle" fill="#94a0b6" font-size="13">8 servos via Station</text>
<rect x="760" y="190" width="390" height="80" rx="12" fill="#1a2a1f" stroke="#2e7a44"/><text x="955" y="222" text-anchor="middle" font-weight="700">Human: ARM button (10 min) + verdict</text><text x="955" y="246" text-anchor="middle" fill="#94a0b6" font-size="13">no move without ARM; success = person or clean photo</text>
<rect x="10" y="190" width="450" height="80" rx="12" fill="#1c2331" stroke="#3a4760"/><text x="235" y="222" text-anchor="middle" font-weight="700">film_task recorder</text><text x="235" y="246" text-anchor="middle" fill="#94a0b6" font-size="13">captioned video + joints -&gt; LeRobot v3 episode + OUTCOME.json</text>
<rect x="510" y="190" width="200" height="80" rx="12" fill="#1c2331" stroke="#3a4760"/><text x="610" y="222" text-anchor="middle" font-weight="700">Teach by hand</text><text x="610" y="246" text-anchor="middle" fill="#94a0b6" font-size="13">soft hold, then replay</text></g>
<g stroke="#6aa8ff" stroke-width="2.2" fill="none" marker-end="url(#ar)"><path d="M200 80H246"/><path d="M460 80H506"/><path d="M710 80H756"/><path d="M940 80H981"/><path d="M1067 186V124"/><path d="M1000 124 C 900 160, 520 160, 300 186"/><path d="M710 230H756"/></g></svg>"""

def slides():
    reel_ok = (P / "videos/highlight_reel.mp4").exists(); S = []
    S.append(slide("", f'<div class="hero"><div class="kick">NormaCore hackathon &middot; Berlin &middot; 21-22 Nov 2026</div><h1>Teaching a desk robot real chores</h1><p class="lead">One real 8-servo arm, one table camera, a vision model and kinematics.<br>Every run filmed and saved as a LeRobot episode &mdash; failures included.</p><div class="pills"><span>{len(REAL)} filmed real-robot runs</span><span>6 tasks</span><span>LeRobot v3 datasets</span><span>honest verdicts</span></div></div>', cls="title"))
    S.append(slide("How it works", ARCH + '<ul class="cols"><li><b>Look first:</b> a fresh photo right before every move; positions are measured live, never hard-coded.</li><li><b>Move safely:</b> IK path checks, slow near objects, current watchdog; nothing moves unless the human pressed ARM.</li><li><b>Record everything:</b> every task is filmed and saved as a training episode, trimmed to the task.</li></ul>', "SYSTEM"))
    S.append(slide("The task we worked towards", '<div class="two">' + vtile("human_task_demo", True) + '<div class="txt"><p>A person shows it once, by hand:</p><ol><li>open the flip-top cap of a tissue pack</li><li>pinch <b>one</b> tissue and pull it out</li><li>clear the coffee beans next to it</li></ol><p class="mu">Before that: warm-up tasks (marker into a box, a magnetic duster collecting metal caps) to get calibration, grasping and recording right.</p></div></div>', "PART 1"))
    S.append(slide("Warm-up: marker into the white box", '<div class="g3">' + vtile("teleop_pick2") + vtile("run_marker_first") + f'<figure class="vt"><img src="images/pick_result.jpg" alt=""><figcaption><span class="b success">Autonomous (photo)</span> Pen in the box ~23 s after seeing it</figcaption></figure></div><p class="note">The first autonomous run pushed the box: the table mapping had x mirrored. Found from the video, fixed with a calibration that uses the jaw tip itself. Afterwards: first-try holds, the gripper stall reading tells marker (1550-1590 ticks) from thin pen (1397) from nothing (1322).</p>', "PART 2"))
    S.append(slide("Magnetic duster collects the metal caps", '<div class="g2">' + vtile("duster_005058") + vtile("duster_005616") + '</div><p class="note">Hold the duster ~1.2 cm above each cap for 1 s &mdash; no pressing, the magnet takes the cap. A camera check counts the caps left before going to the box. The failure: the detector saw only one of two caps.</p>', "PART 3"))
    S.append(slide("Opening the flip-top cap", '<div class="split"><div class="g22">' + vtile("lid_push_013002") + vtile("lid_hand_013433") + vtile("lid_replay_014807") + vtile("full_022202") + '</div><div class="side"><ol><li><b>Push down:</b> failed &mdash; and the vision model said "open" while the arm hid a closed lid.</li><li><b>Hand-guided demo</b> (soft hold, no leader arm): a sideways lift opens it.</li><li><b>Replay</b> of that demo: lid semi-open (pack 2 cm off).</li><li><b>Autonomous sideways sweep</b> at one height: cap open on attempt 3.</li></ol></div></div>', "PART 4"))
    S.append(slide("Pinch and pull one tissue", '<div class="g2">' + vtile("pull_015349") + vtile("guided_033615") + '</div><p class="note">Too deep and the gripper lifts the whole pack (the script even printed "tissue pulled"). Pinch at the tissue\'s tip, ~8-9 cm, close fully, pull <b>straight up</b>. A thin sheet does not stall the gripper (1324-1334 ticks either way), so only the camera can confirm the hold.</p>', "PART 5"))
    S.append(slide("Clearing the coffee beans", '<div class="g2">' + vtile("wipe_024526") + vtile("guided_034731") + '</div><p class="note">Wiping with the held tissue spread the beans. Lanes at table level cleared the bean area but scattered the beans. Pushing the whole pack through them in one stroke was the user\'s favourite &mdash; saved as a demo.</p>', "PART 6"))
    S.append(slide("The whole task in one run", '<div class="g3">' + vtile("guided_030809") + vtile("final_032056") + vtile("final_032412") + '</div><p class="note">Guided by short chat commands, and then autonomous with a camera check after every stage: cap open (picture change), tissue held (camera), retry if not. The bean stage still needs work.</p>', "PART 7"))
    S.append(slide("Simulation twin", '<div class="two">' + vtile("sim_success", True) + '<div class="txt"><p>A MuJoCo twin of the arm and table ran a 100-trial pick benchmark before the real robot, and shares the kinematics used for IK and path checks on the real arm.</p><p class="mu">The twin cannot show servo sag, jaw contact or calibration error &mdash; those were found on the real arm (see the failures).</p></div></div>', "PART 8"))
    S.append(slide("Scoreboard (honest)", scoreboard() + '<p class="note">Only filmed real-robot runs are counted. "Judged from video" verdicts are marked as such in the gallery; nothing was promoted from the script\'s own "done" or from the vision model\'s yes/no.</p>', "RESULTS"))
    S.append(slide("What we learned", '<ul class="lessons"><li><b>Measure the object right before moving.</b> Ten pinches missed because the pack had moved 10 cm since the last photo.</li><li><b>Never trust a yes/no for physical success.</b> The model said "lid open" and "tissue held" when they were not; the script printed "tissue pulled" while holding the whole pack.</li><li><b>Show, then repeat.</b> A 30-second hand-guided demo beat hours of tuning a push; replays work if the object is within 1-2 cm.</li><li><b>Hold gently, never press.</b> Pressing down pushed the pack; the duster magnet works from a distance; slow, soft accelerations keep things in the gripper.</li><li><b>One clean move beats many small steps.</b> Go to the working height in one move, then sweep at that height.</li><li><b>Film and log everything.</b> The video found the mirrored calibration and the false "lid open".</li></ul>', "LESSONS"))
    S.append(slide("Next", '<ul class="lessons"><li>Train a policy on the recorded LeRobot episodes (successes only; failures kept as labelled negatives).</li><li>Grippy fingertip pads for flat tissues; a top-down pinch for the tip.</li><li>Bean collection: push to a corner instead of spreading; verify with a before/after photo.</li><li>Compensate the sag at long reach (the tip drops about 8 mm at low stretched poses).</li></ul>' + ('<div class="reelbox"><video controls playsinline preload="none" poster="posters/reel.jpg" src="videos/highlight_reel.mp4"></video><div>Highlight reel &middot; <a href="index.html">all runs in the gallery</a></div></div>' if reel_ok else ""), "OUTLOOK"))
    page = f"""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Teaching a desk robot real chores - slides</title>
<style>{CSS}
html,body{{height:100%;overflow:hidden;background:#05070b}}#stage{{position:absolute;left:50%;top:50%;width:1280px;height:720px;transform-origin:center;}}
.slide{{position:absolute;inset:0;background:radial-gradient(1200px 700px at 85% -10%,#1b2740 0,#0d1017 55%);border-radius:6px;padding:46px 60px;display:none;flex-direction:column;overflow:hidden}}.slide.on{{display:flex}}
.kick{{color:var(--acc);font-weight:700;letter-spacing:.08em;font-size:13px;text-transform:uppercase;margin-bottom:4px}}h2{{margin:0 0 18px;font-size:34px}}h1{{font-size:60px;line-height:1.05;margin:10px 0 18px}}
.hero{{margin:auto 0}}.lead{{font-size:24px;color:#c6cfdd;line-height:1.45}}.pills span{{display:inline-block;background:#1c2331;border:1px solid #2c3850;border-radius:99px;padding:6px 14px;margin:14px 10px 0 0;font-size:15px}}
.vt{{margin:0;background:var(--card);border-radius:12px;overflow:hidden;display:flex;flex-direction:column;min-height:0}}.vt video,.vt img{{width:100%;aspect-ratio:4/3;object-fit:cover;background:#000;display:block}}.vt figcaption{{padding:8px 12px;font-size:14px;color:#cfd7e6}}
.g2,.g3,.two{{display:grid;gap:18px;flex:1;min-height:0;align-items:start;align-content:center}}.g2{{grid-template-columns:repeat(2,1fr);padding:0 120px}}.g3{{grid-template-columns:repeat(3,1fr)}}.two{{grid-template-columns:1.25fr 1fr;align-items:center}}
.split{{display:grid;grid-template-columns:auto 1fr;gap:30px;flex:1;min-height:0;align-items:center}}.g22{{display:grid;grid-template-columns:repeat(2,300px);gap:14px}}.g22 .vt figcaption{{font-size:12.5px;padding:6px 10px}}.side{{font-size:19px;line-height:1.45;color:#c3ccda}}.side ol{{padding-left:22px;margin:0}}.side li{{margin-bottom:14px}}
.txt{{font-size:21px;line-height:1.5}}.txt ol{{padding-left:24px}}.mu{{color:var(--mut);font-size:17px}}.note{{color:#c3ccda;font-size:17.5px;line-height:1.45;margin:14px 0 0;border-left:3px solid var(--acc);padding-left:14px}}
.arch{{width:100%;height:auto;margin:6px 0 10px}}.cols{{display:grid;grid-template-columns:repeat(3,1fr);gap:22px;padding:0;list-style:none;font-size:17px;color:#c3ccda}}.cols li{{background:var(--card);padding:14px 16px;border-radius:12px}}
.score{{border-collapse:collapse;font-size:20px;width:100%}}.score th,.score td{{padding:10px 14px;border-bottom:1px solid #263044;text-align:right}}.score th:first-child,.score td:first-child{{text-align:left}}.score th{{color:var(--mut);font-weight:600;font-size:15px}}.ok{{color:#4cc76a}}.pa{{color:#e2ac35}}.ba{{color:#e46060}}.mu{{color:var(--mut)}}.tot td{{font-weight:700;border-top:2px solid #3a4760}}
.lessons{{font-size:21px;line-height:1.45;padding-left:22px;margin:0}}.lessons li{{margin:0 0 12px}}.reelbox{{display:flex;gap:18px;align-items:center;margin-top:8px;color:var(--mut)}}.reelbox video{{width:360px;border-radius:10px;background:#000}}
#bar{{position:fixed;left:0;bottom:0;height:4px;background:var(--acc);transition:width .25s}}#num{{position:fixed;right:14px;bottom:10px;color:#6c788e;font-size:13px}}#help{{position:fixed;left:14px;bottom:10px;color:#4c566a;font-size:12px}}</style>
<div id="stage">{''.join(S)}</div><div id="bar"></div><div id="num"></div><div id="help">&larr; &rarr; / space / click &middot; f = fullscreen &middot; <a href="index.html" style="color:#5b6a86">gallery</a></div>
<script>const sl=[...document.querySelectorAll('.slide')];let k=Math.max(0,Math.min(sl.length-1,(parseInt(location.hash.slice(1))||1)-1));
function fit(){{const s=Math.min(innerWidth/1280,innerHeight/720)*0.98;document.getElementById('stage').style.transform=`translate(-50%,-50%) scale(${{s}})`}}
function show(n){{k=Math.max(0,Math.min(sl.length-1,n));sl.forEach((s,i)=>{{s.classList.toggle('on',i===k);s.querySelectorAll('video').forEach(v=>{{if(i===k){{if(v.dataset.src&&!v.src)v.src=v.dataset.src;if(v.hasAttribute('muted'))v.play().catch(()=>{{}})}}else v.pause()}})}});
document.getElementById('bar').style.width=((k+1)/sl.length*100)+'%';document.getElementById('num').textContent=(k+1)+' / '+sl.length;history.replaceState(null,'','#'+(k+1))}}
addEventListener('resize',fit);addEventListener('keydown',e=>{{if(['ArrowRight','PageDown',' '].includes(e.key)){{e.preventDefault();show(k+1)}}else if(['ArrowLeft','PageUp'].includes(e.key))show(k-1);else if(e.key==='Home')show(0);else if(e.key==='End')show(sl.length-1);else if(e.key==='f')(document.fullscreenElement?document.exitFullscreen():document.documentElement.requestFullscreen())}});
document.getElementById('stage').addEventListener('click',e=>{{if(e.target.closest('video,a'))return;show(k+(e.clientX>innerWidth/2?1:-1))}});fit();show(k);</script></html>"""
    (P / "slides.html").write_text(page)

if __name__ == "__main__":
    copy_photos(); gallery(); slides(); print("built:", P / "index.html", "|", P / "slides.html")
