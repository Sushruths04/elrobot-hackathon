#!/usr/bin/env python
"""Which vision model is best for the robot checks? Runs candidate models on LABELLED frames from this robot and prints accuracy + latency.
  tasks: (1) 'is the marker inside the white box?' yes/no on 3 frames;  (2) pointing at the marker / the box (pixel error vs hand-labelled truth) on 2 frames.
  .venv/bin/python scripts/vlm_bench.py                      # every provider that has a key; Nebius vision candidates auto-selected
  .venv/bin/python scripts/vlm_bench.py --nebius-models A,B  # explicit list         --repeat 2"""
import argparse, json, math, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from normakit import llm
ap = argparse.ArgumentParser(); ap.add_argument("--nebius-models", default=""); ap.add_argument("--anthropic-models", default=""); ap.add_argument("--repeat", type=int, default=1); ap.add_argument("--no-gemini", action="store_true"); a = ap.parse_args()
IN = 'Is the marker (pen) inside the white open box on the table? Reply {"inside": true or false}'
YN = [("run/after_take2.jpg", True), ("run/scene_now.jpg", False), ("run/after_push.jpg", False)]          # hand-labelled by looking at the pictures
PT = [("run/scene_now.jpg", "the marker (pen) lying on the table", (300, 254)), ("run/scene_now.jpg", "the white box", (205, 250)), ("run/after_push.jpg", "the white box", (75, 285))]
cands = []
if llm.have("gemini") and not a.no_gemini: cands.append(("gemini", "gemini-robotics-er-2-preview"))
if llm.have("nebius"):
    ms = [m for m in a.nebius_models.split(",") if m] or [m for m in llm.models("nebius") if any(k in m.lower() for k in ("vl", "vision", "gemma-3", "pixtral", "llava", "qwen3", "llama-3.2-11b", "llama-3.2-90b", "internvl", "phi-4-multimodal"))]
    print("Nebius vision candidates:", ms, flush=True); cands += [("nebius", m) for m in ms]
if llm.have("anthropic"):
    ms = [m for m in a.anthropic_models.split(",") if m] or [m for m in llm.models("anthropic") if any(k in m for k in ("sonnet", "haiku", "opus"))][-4:]
    print("Anthropic candidates:", ms, flush=True); cands += [("anthropic", m) for m in ms]
if not cands: sys.exit("no provider key found: save keys with the web page key box or ~/.nebius_key / ~/.anthropic_key / ~/.gemini_er_key")
rows = []
for prov, model in cands:
    ok = n = 0; lat = []; perr = []; fails = 0
    for path, truth in YN:
        jp = (ROOT / path).read_bytes()
        for _ in range(a.repeat):
            try: r = llm.ask(prov, model, jp, IN); ok += (bool(r.get("inside")) == truth); n += 1; lat.append(r["seconds"])
            except Exception as e: fails += 1; err = str(e)[:90]
    for path, what, truth in PT:
        jp = (ROOT / path).read_bytes()
        try:
            t = time.time(); pts = llm.point(prov, model, jp, what, 640, 480)
            if pts: perr.append(math.hypot(pts[0][1] - truth[0], pts[0][2] - truth[1])); lat.append(time.time() - t)
            else: fails += 1
        except Exception as e: fails += 1; err = str(e)[:90]
    rows.append({"provider": prov, "model": model, "yesno": f"{ok}/{n}", "point_err_px_mean": round(sum(perr) / len(perr), 1) if perr else None, "latency_s_median": round(sorted(lat)[len(lat) // 2], 2) if lat else None, "failures": fails, "last_error": err if fails else ""})
    print(rows[-1], flush=True)
rows.sort(key=lambda r: (-(int(r["yesno"].split("/")[0]) / max(1, int(r["yesno"].split("/")[1]))), r["point_err_px_mean"] if r["point_err_px_mean"] is not None else 1e9, r["latency_s_median"] or 1e9))
(ROOT / "run/vlm_bench.json").write_text(json.dumps(rows, indent=1)); print("\nRANKING (accuracy, then pointing error, then speed):"); [print(f"  {i+1}. {r['provider']}/{r['model']}  yes/no {r['yesno']}  point err {r['point_err_px_mm'] if False else r['point_err_px_mean']} px  {r['latency_s_median']} s") for i, r in enumerate(rows)]
