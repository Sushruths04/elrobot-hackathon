"""ONE process, ONE command: the whole task as a decision loop (EBIM lesson: build the loop, don't be the loop).
Claude (API) is used only for PERCEPTION from the RGB camera: where is X / is X held / is X on Y. Every decision and retry is in this file.
No object positions are hardcoded: they are measured live from the image and mapped with the camera calibration.
    .venv/bin/python scripts/autorun.py --backend sim --vision color --object "red block" --target "blue tray"
    ANTHROPIC_API_KEY=... .venv/bin/python scripts/autorun.py --backend station --vision claude --object "red block" --target "blue tray" --i-am-at-the-estop
Logs: run/autorun/<time>/ (every look image + trial.json)."""
import argparse, base64, json, os, re, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from normakit import vision
from normakit.arm import Arm

ap = argparse.ArgumentParser()
ap.add_argument("--backend", choices=["sim", "station"], default="sim"); ap.add_argument("--vision", choices=["color", "dark", "yolo", "claude"], default="color")
ap.add_argument("--object", default="red block"); ap.add_argument("--target", default="blue tray"); ap.add_argument("--cam")
ap.add_argument("--table-calib", default="calibration/table_front.json", help="pixel->robot table mapping (station backend)")
ap.add_argument("--yolo-classes", default="cup,bottle", help="COCO classes for --vision yolo (weights download on first use)")
ap.add_argument("--retries", type=int, default=2); ap.add_argument("--i-am-at-the-estop", action="store_true"); a = ap.parse_args()
if a.backend == "station" and not a.i_am_at_the_estop: sys.exit("refusing to move the real arm: add --i-am-at-the-estop once a person is at the table and the power plug is in reach")
if a.vision == "claude":
    import anthropic; client = anthropic.Anthropic(); MODEL = os.environ.get("NORMA_MODEL", "claude-sonnet-5-5")

out = ROOT / "run/autorun" / time.strftime("%Y%m%d_%H%M%S"); out.mkdir(parents=True, exist_ok=True)
log = {"object": a.object, "target": a.target, "vision": a.vision, "backend": a.backend, "steps": [], "claude_calls": 0}
t_start = time.time(); n_img = [0]
def step(name, **kw): log["steps"].append({"t": round(time.time() - t_start, 1), "step": name, **kw}); print(f"[{log['steps'][-1]['t']:6.1f}s] {name} {kw if kw else ''}", flush=True)

def ask(jpeg: bytes, prompt: str) -> dict:
    log["claude_calls"] += 1
    msg = client.messages.create(model=MODEL, max_tokens=300, messages=[{"role": "user", "content": [
        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": base64.b64encode(jpeg).decode()}},
        {"type": "text", "text": prompt + " Reply with ONLY one JSON object, no other text."}]}])
    m = re.search(r"\{.*\}", msg.content[0].text, re.S); return json.loads(m.group(0)) if m else {}

def look(arm, tag):
    jpg = arm.look(a.cam); n_img[0] += 1; (out / f"{n_img[0]:02d}_{tag}.jpg").write_bytes(jpg); return jpg

_DET = []
def locate(jpg, what):
    img = vision.decode(jpg); h, w = img.shape[:2]
    if a.vision == "dark":                                               # black object on the light table: no model needed
        d = vision.find_dark_object(img); return ((d["bottom_px"][0], d["bottom_px"][1]), "dark blob") if d else (None, "no dark object in the table region")
    if a.vision == "yolo":                                               # teammate's detector (src/normakit/detect.py); bottom_px = where the object touches the table
        from normakit.detect import Detector
        if not _DET: _DET.append(Detector(classes=a.yolo_classes.split(",")))
        label = next((c for c in a.yolo_classes.split(",") if c in what.lower()), a.yolo_classes.split(",")[0]); r = _DET[0].find(jpg, label)
        return ((r["bottom_px"][0], r["bottom_px"][1]), f"{label} {r['conf']:.2f}") if r.get("ok") else (None, r.get("error", "not found"))
    if a.vision == "color":
        for word in what.lower().split():
            if word in vision.NAMED_COLORS:
                p = vision.find_named(img, word)
                return (p, "color") if p else (None, f"no {word} pixels")
        return None, "no colour word in description"
    r = ask(jpg, f"This RGB camera image is {w}x{h} pixels. Find: {what}. Return {{\"found\": true/false, \"u\": <x pixel>, \"v\": <y pixel>, \"note\": \"<short>\"}} where (u,v) is the centre of that object.")
    return ((float(r["u"]), float(r["v"])), r.get("note", "")) if r.get("found") else (None, r.get("note", "not found"))

def verify(jpg, question):
    if a.vision != "claude": return None                       # colour mode cannot judge a grasp: reported as unverified, never guessed
    return bool(ask(jpg, f"{question} Return {{\"answer\": true/false, \"note\": \"<short>\"}}").get("answer"))

if a.backend == "sim":
    from normakit.sim_backend import SimBackend; be = SimBackend(); be.start(); calib = ROOT / "run/sim_H.json"
else:
    from normakit.station_backend import StationBackend; be = StationBackend("localhost:8888"); be.start(); calib = ROOT / a.table_calib
if not calib.exists(): sys.exit(f"no camera calibration at {calib} - run scripts/calibrate_camera.py first (positions are never hardcoded)")
arm = Arm(be, calib_path=calib); result = "failed"
try:
    for attempt in range(1, a.retries + 2):
        step("attempt", n=attempt); arm.home(); jpg = look(arm, f"a{attempt}_look")
        p, note = locate(jpg, a.object); step("locate_object", pixel=p, note=note)
        if p is None: continue
        r = arm.pick(*p); step("pick", **{k: v for k, v in r.items() if k in ("ok", "error", "grasped")})
        if not r.get("ok"): continue
        held = verify(look(arm, f"a{attempt}_held"), f"Is the {a.object} held in the robot gripper, lifted off the table?"); step("verify_held", held=held)
        if held is False: arm.open_gripper(); continue
        jpg = look(arm, f"a{attempt}_target"); q, note = locate(jpg, a.target); step("locate_target", pixel=q, note=note)
        if q is None: arm.open_gripper(); continue
        r = arm.place(*q); step("place", **{k: v for k, v in r.items() if k in ("ok", "error")}); arm.open_gripper(); arm.home()
        placed = verify(look(arm, f"a{attempt}_done"), f"Is the {a.object} now on or inside the {a.target}?"); step("verify_placed", placed=placed)
        result = "success" if placed else ("unverified" if placed is None else "failed")
        if placed is not False: break
finally:
    try: arm.home()
    except Exception as e: step("home_failed", error=str(e))
    log.update(result=result, seconds=round(time.time() - t_start, 1)); (out / "trial.json").write_text(json.dumps(log, indent=1))
    print(f"RESULT: {result}  ({log['seconds']} s, {log['claude_calls']} Claude calls)  logs: {out.relative_to(ROOT)}", flush=True); os._exit(0)
