"""Gemini Robotics-ER 2 pointing (perception only): 'point to X' -> pixel. The key is read from $GEMINI_API_KEY or ~/.gemini_er_key and is
sent in the x-goog-api-key HEADER (never in a URL, never printed, never written to the repo)."""
from __future__ import annotations
import base64, json, os, time, urllib.request, urllib.error
from pathlib import Path
MODEL = os.environ.get("ER2_MODEL", "gemini-robotics-er-2-preview")
URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent"

def _key() -> str:
    k = os.environ.get("GEMINI_API_KEY")
    if not k:
        f = Path.home() / ".gemini_er_key"
        if f.exists(): k = f.read_text().strip()
    if not k: raise RuntimeError("no Gemini key: set GEMINI_API_KEY or put it in ~/.gemini_er_key (chmod 600)")
    return k

def _gemini_point(jpeg: bytes, what: str, width: int, height: int, timeout: float = 30.0, thinking_budget: int | None = 0) -> dict:
    """-> {"points": [(label, x_px, y_px), ...], "seconds": float}. ER models answer [y, x] normalised to 0-1000."""
    prompt = (f"Point to {what}. The answer should follow the json format: [{{\"point\": [y, x], \"label\": <label>}}, ...]. "
              "The points are in [y, x] format normalized to 0-1000. Point at the spot where the object touches the table.")
    cfg = {"temperature": 0, "responseMimeType": "application/json"}
    if thinking_budget is not None: cfg["thinkingConfig"] = {"thinkingBudget": thinking_budget}      # no 'thinking' = faster
    body = {"contents": [{"parts": [{"inline_data": {"mime_type": "image/jpeg", "data": base64.b64encode(jpeg).decode()}}, {"text": prompt}]}], "generationConfig": cfg}
    def call(b):
        req = urllib.request.Request(URL, data=json.dumps(b).encode(), headers={"Content-Type": "application/json", "x-goog-api-key": _key()})
        with urllib.request.urlopen(req, timeout=timeout) as r: return json.loads(r.read())
    t0 = time.time()
    try: out = call(body)
    except urllib.error.HTTPError as e:
        msg = e.read().decode(errors="replace")[:300]
        if e.code == 400 and "thinking" in msg.lower() and thinking_budget is not None:           # this model rejects the thinking switch: retry without
            body["generationConfig"].pop("thinkingConfig", None); out = call(body)
        else: raise RuntimeError(f"Gemini HTTP {e.code}: {msg}") from None
    text = out["candidates"][0]["content"]["parts"][0]["text"]; pts = []
    try:
        data = json.loads(text)
        for d in (data if isinstance(data, list) else data.get("points", [])):
            y, x = d["point"]; pts.append((d.get("label", ""), x / 1000.0 * width, y / 1000.0 * height))
    except Exception:                                  # model sometimes returns slightly malformed JSON: pull the [y, x] pairs out directly
        import re
        for m in re.finditer(r'\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]', text):
            pts.append(("", float(m.group(2)) / 1000.0 * width, float(m.group(1)) / 1000.0 * height))
    return {"points": pts, "seconds": round(time.time() - t0, 2)}


def _gemini_ask(jpeg: bytes, question: str, timeout: float = 30.0) -> dict:
    """Ask ER-2 a yes/no style question about the picture; it must answer one JSON object. -> parsed dict (+ 'seconds')."""
    body = {"contents": [{"parts": [{"inline_data": {"mime_type": "image/jpeg", "data": base64.b64encode(jpeg).decode()}}, {"text": question + " Answer ONLY with one JSON object."}]}],
            "generationConfig": {"temperature": 0, "responseMimeType": "application/json", "thinkingConfig": {"thinkingBudget": 0}}}
    def call(b):
        req = urllib.request.Request(URL, data=json.dumps(b).encode(), headers={"Content-Type": "application/json", "x-goog-api-key": _key()})
        with urllib.request.urlopen(req, timeout=timeout) as r: return json.loads(r.read())
    t0 = time.time()
    try: out = call(body)
    except urllib.error.HTTPError as e:
        if e.code == 400: body["generationConfig"].pop("thinkingConfig", None); out = call(body)
        else: raise RuntimeError(f"Gemini HTTP {e.code}") from None
    d = json.loads(out["candidates"][0]["content"]["parts"][0]["text"]); d = d[0] if isinstance(d, list) and d else d
    return {**(d if isinstance(d, dict) else {"answer": d}), "seconds": round(time.time() - t0, 2)}


def point(jpeg: bytes, what: str, width: int, height: int, timeout: float = 30.0, thinking_budget: int | None = 0) -> dict:
    """Pointing via the provider chosen in run/vlm.json (default Gemini ER-2; the benchmark winner is Nebius Nemotron-3.5-Super-VL)."""
    from . import llm; c = llm.config()
    if c["provider"] == "gemini": return _gemini_point(jpeg, what, width, height, timeout, thinking_budget)
    t0 = time.time(); pts = llm.point(c["provider"], c["model"], jpeg, what, width, height, timeout=max(timeout, 60), many=True); return {"points": pts, "seconds": round(time.time() - t0, 2)}

def ask(jpeg: bytes, question: str, timeout: float = 30.0) -> dict:
    from . import llm; c = llm.config()
    if c["provider"] == "gemini": return _gemini_ask(jpeg, question, timeout)
    return llm.ask(c["provider"], c["model"], jpeg, question, timeout=max(timeout, 60))
