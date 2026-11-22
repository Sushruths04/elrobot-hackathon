"""One small vision-LLM client for three providers (urllib only; keys from files or env, never printed, always sent in headers):
  gemini    -> Gemini Robotics-ER 2 (pointing)   key: $GEMINI_API_KEY or ~/.gemini_er_key
  nebius    -> Nebius Token Factory (OpenAI-compatible)  key: $NEBIUS_API_KEY or ~/.nebius_key
  anthropic -> Claude API                        key: $ANTHROPIC_API_KEY or ~/.anthropic_key
ask(provider, model, jpeg, question) -> dict (parsed JSON answer + 'seconds');  point(...) -> [(label, x_px, y_px)];  models(provider) -> [ids]"""
from __future__ import annotations
import base64, json, os, re, time, urllib.request, urllib.error
from pathlib import Path
KEYS = {"gemini": ("GEMINI_API_KEY", ".gemini_er_key"), "nebius": ("NEBIUS_API_KEY", ".nebius_key"), "anthropic": ("ANTHROPIC_API_KEY", ".anthropic_key")}
NEBIUS = os.environ.get("NEBIUS_BASE_URL", "https://api.tokenfactory.nebius.com/v1")
def key(provider):
    env, f = KEYS[provider]; k = os.environ.get(env)
    if not k and (Path.home() / f).exists(): k = (Path.home() / f).read_text().strip()
    if not k: raise RuntimeError(f"no {provider} key: put it in ~/{f} (chmod 600) or set ${env}")
    return k
def have(provider):
    try: key(provider); return True
    except Exception: return False
def _json_from(text):
    text = text.strip(); m = re.search(r"```(?:json)?\s*(.*?)```", text, re.S); text = m.group(1) if m else text
    for pat in (r"\{.*\}", r"\[.*\]"):
        m = re.search(pat, text, re.S)
        if m:
            try: return json.loads(m.group(0))
            except Exception: pass
    d = {m.group(1): (m.group(2) == "true" if m.group(2) in ("true", "false") else float(m.group(2))) for m in re.finditer(r'"(\w+)"\s*:\s*(true|false|-?\d+(?:\.\d+)?)', text)}   # malformed JSON: still recover booleans/numbers
    d["raw"] = text[:300]; return d
def _post(url, headers, body, timeout=60):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json", **headers})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r: return json.loads(r.read())
    except urllib.error.HTTPError as e: raise RuntimeError(f"{url.split('/')[2]} HTTP {e.code}: {e.read().decode(errors='replace')[:200]}") from None
def _get(url, headers, timeout=30):
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=timeout) as r: return json.loads(r.read())
def models(provider):
    if provider == "nebius": return sorted(m["id"] for m in _get(NEBIUS + "/models?verbose=true", {"Authorization": "Bearer " + key("nebius")}).get("data", []))
    if provider == "anthropic": return sorted(m["id"] for m in _get("https://api.anthropic.com/v1/models?limit=100", {"x-api-key": key("anthropic"), "anthropic-version": "2023-06-01"}).get("data", []))
    if provider == "gemini": return ["gemini-robotics-er-2-preview"]
    raise ValueError(provider)
def ask(provider, model, jpeg, question, max_tokens=300, timeout=60):
    t0 = time.time(); b64 = base64.b64encode(jpeg).decode(); q = question + " Answer ONLY with one JSON object."
    if provider == "gemini":
        o = _post(f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent", {"x-goog-api-key": key("gemini")}, {"contents": [{"parts": [{"inline_data": {"mime_type": "image/jpeg", "data": b64}}, {"text": q}]}], "generationConfig": {"temperature": 0, "responseMimeType": "application/json"}}, timeout)
        txt = o["candidates"][0]["content"]["parts"][0]["text"]
    elif provider == "nebius":
        o = _post(NEBIUS + "/chat/completions", {"Authorization": "Bearer " + key("nebius")}, {"model": model, "temperature": 0, "max_tokens": max_tokens, "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + b64}}, {"type": "text", "text": q}]}]}, timeout)
        msg = o["choices"][0]["message"]; txt = msg.get("content") or ""
        if not txt.strip():                                           # reasoning models can spend the whole budget thinking: retry once with a bigger budget
            o = _post(NEBIUS + "/chat/completions", {"Authorization": "Bearer " + key("nebius")}, {"model": model, "temperature": 0, "max_tokens": max_tokens * 4, "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + b64}}, {"type": "text", "text": q}]}]}, timeout)
            msg = o["choices"][0]["message"]; txt = msg.get("content") or msg.get("reasoning_content") or ""
    elif provider == "anthropic":
        o = _post("https://api.anthropic.com/v1/messages", {"x-api-key": key("anthropic"), "anthropic-version": "2023-06-01"}, {"model": model, "max_tokens": max_tokens, "temperature": 0, "messages": [{"role": "user", "content": [{"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}}, {"type": "text", "text": q}]}]}, timeout)
        txt = "".join(c.get("text", "") for c in o["content"])
    else: raise ValueError(provider)
    d = _json_from(txt); d = d[0] if isinstance(d, list) and d and isinstance(d[0], dict) else d
    return {**(d if isinstance(d, dict) else {"answer": d}), "seconds": round(time.time() - t0, 2)}
def point(provider, model, jpeg, what, width, height, timeout=60, many=False):
    """Pixel position(s) of `what` -> [(label, x_px, y_px)]. Non-Gemini models answer normalised 0-1000 coordinates (x right, y down)."""
    if provider == "gemini":
        from . import er2; return er2._gemini_point(jpeg, what, width, height)["points"]
    q = (f'Point to {what}. Reply {{"points": [{{"label": text, "x": number, "y": number}}, ...]}} with the centre of every instance (or end) asked for, '
         'x and y normalised to 0-1000 over the whole image (x to the right, y downwards).')
    d = ask(provider, model, jpeg, q, max_tokens=400, timeout=timeout); out = []
    for it in (d.get("points") if isinstance(d.get("points"), list) else [d]):
        try: out.append((str(it.get("label", "")), float(it["x"]) / 1000 * width, float(it["y"]) / 1000 * height))
        except Exception: pass
    return out if many else out[:1]
def config():
    try: return json.loads((Path(__file__).resolve().parents[2] / "run/vlm.json").read_text())
    except Exception: return {"provider": "gemini", "model": "gemini-robotics-er-2-preview"}
