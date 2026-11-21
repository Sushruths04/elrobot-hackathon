#!/usr/bin/env python
"""Local control interface (127.0.0.1 only): live cameras, arm state, cup detection (Gemini ER-2), plan, run the grab, STOP,
and scenario recording (video + joint log + events).   .venv/bin/python scripts/ui.py   ->  http://127.0.0.1:8890
Station must be running (./norma start). Moves happen only through scripts/grab_cup.py --go, started from the page after a safety tick."""
import argparse, json, os, signal, subprocess, sys, threading, time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs
import cv2
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from normakit import vision, er2
from normakit.core import norm_to_rad
from normakit.kin import Kin
from normakit.station_backend import StationBackend, front_camera, REG_GOAL_POS, REG_GOAL_SPEED, REG_ACC, REG_TORQUE
PY = str(ROOT / ".venv/bin/python"); REC = ROOT / "run/recordings"; REC.mkdir(parents=True, exist_ok=True); KEYFILE = Path.home() / ".gemini_er_key"
try: be = StationBackend("localhost:8888")
except Exception: be = StationBackend("localhost:8888", require_motors=False)
be.start(); time.sleep(2.0); kin = Kin(); H = vision.Homography.load(str(ROOT / "calibration/table_front.json")); LOCK = threading.Lock()
ARM = {"until": 0.0}
proc = {"p": None, "log": ROOT / "run/ui_run.log", "label": ""}

def cams():
    try: return list(be.cameras())
    except Exception: return []
def jpeg(cam):
    c = front_camera(be) if cam in ("front", "", None) else cam
    return be.frame_jpeg(c)
def hold():
    s = be.state(); be._sync(REG_GOAL_POS, {i: (s.joints[i].raw % 4096).to_bytes(2, "little") for i in range(1, 9) if i in s.joints})
def jog(joint, delta):
    if time.time() > ARM["until"]: return {"error": "motion is NOT armed - the human must press 'ARM motion' on the web page first"}
    """Move ONE joint by a small number of ticks (clamped to its calibrated range, max 150, slow). Everything else is first told to hold."""
    joint = int(joint); delta = max(-150, min(150, int(delta))); s = be.state()
    if joint not in s.joints: return {"error": f"no joint {joint}"}
    if proc["p"] and proc["p"].poll() is None: return {"error": "a run is active - press STOP first"}
    j = s.joints[joint]; span = (j.hi - j.lo) % 4096; t = (j.raw + delta) % 4096; rel = (t - j.lo) % 4096
    if rel > span + (4096 - span) / 2: rel -= 4096
    if rel < 0: t = j.lo
    elif rel > span: t = j.hi
    be._sync(REG_GOAL_POS, {i: (v.raw % 4096).to_bytes(2, "little") for i, v in s.joints.items()}); be._sync(REG_TORQUE, {i: b"\x01" for i in s.joints})
    be._sync(REG_ACC, {joint: bytes([40])}); be._sync(REG_GOAL_SPEED, {joint: (200).to_bytes(2, "little")}); be._sync(REG_GOAL_POS, {joint: int(t).to_bytes(2, "little")}); time.sleep(0.9); n = be.state().joints[joint]
    REC_S.event(f"jog j{joint} {delta:+d}: {j.raw} -> {n.raw} (asked {t}), {n.current_ma} mA"); return {"joint": joint, "before": j.raw, "asked": int(t), "after": n.raw, "moved": ((n.raw - j.raw + 2048) % 4096) - 2048, "current_mA": n.current_ma}
def state():
    s = be.state(); j = {i: {"raw": v.raw, "norm": round(v.norm, 3), "mA": v.current_ma} for i, v in s.joints.items()}
    try: ee = [round(float(x), 3) for x in kin.fk({i: norm_to_rad(i, v.norm) for i, v in s.joints.items() if i <= 7})]
    except Exception: ee = None
    p = proc["p"]; return {"joints": j, "ee": ee, "running": bool(p and p.poll() is None), "recording": REC_S.active, "label": proc["label"], "armed_s": max(0, int(ARM["until"] - time.time())), "key": KEYFILE.exists() or bool(os.environ.get("GEMINI_API_KEY")), "cams": cams()}

class Recorder:
    def __init__(self): self.active = False; self.events = []; self.dir = None
    def event(self, text):
        self.events.append({"t": round(time.time() - getattr(self, "t0", time.time()), 2), "text": text})
    def start(self, label):
        if self.active: return self.dir.name
        self.dir = REC / (datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + "".join(c if c.isalnum() else "_" for c in label)[:30]); self.dir.mkdir(); self.events = []; self.t0 = time.time(); self.active = True
        self.label = label; threading.Thread(target=self._loop, daemon=True).start(); return self.dir.name
    def _loop(self):
        vw = None; jl = open(self.dir / "joints.jsonl", "w"); n = 0
        while self.active:
            t = time.time()
            try:
                bgr = cv2.cvtColor(vision.decode(jpeg("front")), cv2.COLOR_RGB2BGR)
                if vw is None: vw = cv2.VideoWriter(str(self.dir / "front.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 8, (bgr.shape[1], bgr.shape[0]))
                vw.write(bgr); n += 1
            except Exception: pass
            try: jl.write(json.dumps({"t": round(t - self.t0, 3), **state()["joints"]}) + "\n")
            except Exception: pass
            time.sleep(max(0, 0.125 - (time.time() - t)))
        if vw: vw.release()
        jl.close(); (self.dir / "meta.json").write_text(json.dumps({"label": self.label, "frames": n, "events": self.events, "seconds": round(time.time() - self.t0, 1)}, indent=1))
    def stop(self):
        self.active = False; time.sleep(0.4)
REC_S = Recorder()

def detect():
    jpg = jpeg("front"); rgb = vision.decode(jpg); h, w = rgb.shape[:2]; out = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR); res = {}
    d = vision.find_dark_object(rgb, roi=(60, 185, 500, 470), min_area=300, max_area=30000)
    if d:
        u, v = d["bottom_px"]; X, Y = H(u, v); res["blob"] = {"px": [round(u), round(v)], "table_xy_m": [round(float(X), 3), round(float(Y), 3)]}
        x1, y1, x2, y2 = map(int, d["box"]); cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2); cv2.putText(out, "blob", (x1, y1 - 6), 0, 0.5, (0, 255, 0), 2)
    else: res["blob"] = None
    try:
        r = er2.point(jpg, "the black cup", w, h); res["er2"] = {"seconds": r["seconds"], "points": []}
        for lab, x, y in r["points"]:
            X, Y = H(x, y); res["er2"]["points"].append({"label": lab, "px": [round(x), round(y)], "table_xy_m": [round(float(X), 3), round(float(Y), 3)]}); cv2.circle(out, (int(x), int(y)), 9, (255, 0, 255), 3); cv2.putText(out, "ER-2 " + lab, (int(x) + 12, int(y)), 0, 0.5, (255, 0, 255), 2)
    except Exception as e: res["er2_error"] = str(e)[:160]
    cv2.imwrite(str(ROOT / "run/ui_detect.jpg"), out); REC_S.event("detect " + json.dumps(res)[:300]); return res

def start_run(go, vis, speed, rec, task='marker'):
    if go and time.time() > ARM["until"]: return {"error": "motion is NOT armed - the human must press 'ARM motion' on the web page first"}
    p = proc["p"]
    if p and p.poll() is None: return {"error": "a run is already active"}
    args = ([PY, str(ROOT / "scripts/grab_marker.py"), "--speed", str(int(speed))] if task == "marker" else [PY, str(ROOT / "scripts/grab_cup.py"), "--vision", vis, "--speed", str(int(speed))]) + (["--go"] if go else [])
    proc["label"] = ("RUN " if go else "PLAN ") + task; f = open(proc["log"], "w"); proc["p"] = subprocess.Popen(args, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
    if rec: REC_S.start(proc["label"])
    REC_S.event("started " + " ".join(args[2:]))
    def watch():
        proc["p"].wait(); REC_S.event(f"process ended, exit code {proc['p'].returncode}")
        if rec: time.sleep(1.5); REC_S.stop()
    threading.Thread(target=watch, daemon=True).start(); return {"ok": True}
def stop_run():
    p = proc["p"]
    if p and p.poll() is None: p.kill(); p.wait()
    try: hold()
    except Exception as e: return {"error": str(e)}
    REC_S.event("STOP pressed - holding present pose"); return {"ok": True}


import urllib.request as _ur
def armd(path, body=None, timeout=180):
    try:
        req = _ur.Request("http://127.0.0.1:8891" + path, data=None if body is None else json.dumps(body).encode(), method="GET" if body is None else "POST")
        with _ur.urlopen(req, timeout=timeout) as r: return json.loads(r.read())
    except Exception as e: return {"error": f"armd not reachable ({type(e).__name__}): start it with ./norma up"}
leader_proc = {"p": None}
def demo_record(on, name):
    if on:
        r = armd("/record", {"on": True, "name": name})
        if "error" in r: return r
        leader_proc["p"] = subprocess.Popen([PY, str(ROOT / "scripts/rec_leader.py"), name], cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL); REC_S.start(name); return {"recording": name, "streams": "follower 40Hz + leader 50Hz + front camera"}
    r = armd("/record", {"on": False}); (ROOT / "run/rec_leader_stop").write_text("1"); REC_S.stop(); return r
def demo_list():
    try: lab = json.loads((ROOT / "run/demos/episodes.json").read_text())
    except Exception: lab = {}
    names = sorted(p.stem for p in (ROOT / "run/demos").glob("*.json") if p.stem != "episodes")
    return [{"name": n, "success": lab.get(n, {}).get("success"), "note": lab.get(n, {}).get("note", "")} for n in names]
def demo_mark(name, success, note=""):
    p = ROOT / "run/demos/episodes.json"
    try: d = json.loads(p.read_text())
    except Exception: d = {}
    d[name] = {"success": success, "note": note, "marked": time.strftime("%F %T")}; p.write_text(json.dumps(d, indent=1)); return d[name]
def ask_gemini(question):
    return er2.ask(jpeg("front"), question)

PAGE = r"""<!doctype html><meta charset=utf-8><title>ElRobot control</title><meta name=viewport content="width=device-width,initial-scale=1">
<style>:root{--bg:#10131a;--card:#1a1f2b;--tx:#e8ecf3;--mut:#8b95a7;--ok:#3ecf8e;--bad:#ff5d5d;--acc:#6aa8ff}body{margin:0;background:var(--bg);color:var(--tx);font:14px system-ui}
.wrap{display:grid;grid-template-columns:minmax(320px,1.4fr) minmax(300px,1fr);gap:14px;padding:14px}@media(max-width:900px){.wrap{grid-template-columns:1fr}}
.card{background:var(--card);border-radius:10px;padding:12px;margin-bottom:12px}h3{margin:0 0 8px;font-size:13px;color:var(--mut);text-transform:uppercase;letter-spacing:.06em}
img{width:100%;border-radius:6px;background:#000}button{background:#2a3246;color:var(--tx);border:0;padding:9px 12px;border-radius:7px;cursor:pointer;margin:3px 3px 3px 0;font-size:14px}
button.go{background:#1f7a4f}button.stop{background:#c0392b;font-weight:700;font-size:17px;width:100%;padding:14px}button:disabled{opacity:.4}select,input{background:#0e1117;color:var(--tx);border:1px solid #333b4f;border-radius:6px;padding:6px}
pre{background:#0b0e14;border-radius:6px;padding:8px;max-height:230px;overflow:auto;font-size:12px;white-space:pre-wrap;margin:6px 0 0}td{padding:2px 8px 2px 0}.bar{height:6px;background:#0b0e14;border-radius:3px;width:120px}.bar i{display:block;height:6px;background:var(--acc);border-radius:3px}
.pill{padding:2px 8px;border-radius:10px;font-size:12px}.on{background:#1f7a4f}.off{background:#3a4156}</style>
<div class=wrap><div>
 <div class=card><h3>Table camera <select id=cam onchange=ref()></select> <label><input type=checkbox id=ov> show detection</label></h3><img id=img></div>
 <div class=card><h3>Output</h3><pre id=out>ready</pre></div>
 <div class=card><h3>Run log (live, from the script)</h3><pre id=lg style="max-height:340px">no run yet</pre></div>
</div><div>
 <div class=card><button class=stop onclick="post('/api/stop').then(show)">STOP (hold pose)</button><br><button id=armb onclick="post('/api/arm',{on:true}).then(show)" style="width:100%;margin-top:8px;background:#7a5b00">ARM motion for 10 min (only you press this)</button></div>
 <div class=card><h3>Status</h3><span class="pill off" id=pr>idle</span> <span class="pill off" id=pc>not recording</span> <span class="pill off" id=pk>ER-2 key?</span><div id=ee style="margin-top:6px;color:var(--mut)"></div></div>
 <div class=card><h3>Cup</h3><button onclick="act('detect')">Detect cup (Gemini ER-2)</button><div style="color:var(--mut);font-size:12px">Result in Output; picture with the marker: tick “show detection”.</div></div>
 <div class=card><h3>Small steps (one joint, slow)</h3>Step <select id=stp><option>20</option><option selected>50</option><option>100</option><option>150</option></select> ticks (4096 = full turn)<table id=jg></table>
  <div style="color:var(--mut);font-size:12px">Output shows how far it really moved and the current. "moved 0" = that joint did not respond.</div></div>
 <div class=card><h3>Demos (teach by showing -> replay)</h3><input id=dn placeholder="demo name" value="demo1" style="width:110px"> <button onclick="post('/api/demo/record',{on:true,name:$('dn').value}).then(show)">Record</button><button onclick="post('/api/demo/record',{on:false}).then(show)">Stop</button><br>
  <select id=dl></select> speed <select id=ds><option>0.5</option><option>1.0</option></select> <button class=go onclick="post('/api/demo/play',{name:$('dl').value,scale:+$('ds').value}).then(show)">REPLAY (needs ARM)</button>
  <button onclick="post('/api/demo/mark',{name:$('dl').value,success:true}).then(show)">mark OK</button><button onclick="post('/api/demo/mark',{name:$('dl').value,success:false}).then(show)">mark bad</button><br>
  <input id=aq placeholder="ask Gemini about the table camera" style="width:70%"> <button onclick="post('/api/ask',{question:$('aq').value}).then(show)">Ask</button></div>
 <div class=card><h3>Run</h3>
  Task <select id=task><option value=marker>marker -> box</option><option value=cup>cup (old)</option></select> Speed <input id=spd type=number value=250 step=50 style="width:70px">
  <label><input type=checkbox id=rec checked> record scenario</label><br>
  <button onclick="run(0)">Plan only (no motion)</button><br>
  <label><input type=checkbox id=safe> Hands clear, power plug in reach, arm supported</label><br>
  <button class=go onclick="run(1)">RUN (moves the arm)</button> <button class=go onclick="post('/api/proceed').then(show)" style="background:#b8860b">PROCEED (when asked)</button></div>
 <div class=card><h3>Record</h3><input id=lab placeholder="label" value="scenario" style="width:140px"> <button onclick="rec(1)">Start</button><button onclick="rec(0)">Stop</button><div id=recs></div></div>
 <div class=card><h3>Joints (follower)</h3><table id=jt></table></div>
 <div class=card><h3>API keys (saved to your home folder only, never shown or sent to chat)</h3><select id=kp><option value=gemini>Gemini ER-2</option><option value=nebius>Nebius</option><option value=anthropic>Anthropic (Claude)</option></select> <input id=key type=password placeholder="paste key here" style="width:55%"> <button onclick="saveKey()">Save</button></div>
</div></div>
<script>
const $=id=>document.getElementById(id),post=(u,b)=>fetch(u,{method:'POST',body:JSON.stringify(b||{})}).then(r=>r.json());
const show=j=>{$('out').textContent=JSON.stringify(j,null,1)};
function ref(){$('img').src=($('ov').checked?'/detect.jpg?':'/frame.jpg?cam='+$('cam').value+'&')+Date.now()}
function act(a){$('out').textContent='working...';post('/api/'+a).then(show).then(()=>{$('ov').checked=true})}
function run(go){if(go&&!$('safe').checked){alert('Tick the safety box first');return}post('/api/run',{go,task:$('task').value,speed:+$('spd').value,rec:$('rec').checked}); $('ov').checked=true.then(show)}
function jog(j,d){post('/api/jog',{joint:j,delta:d*(+$('stp').value)}).then(show)}
function rec(s){post('/api/record',{on:s,label:$('lab').value}).then(show)}
function saveKey(){post('/api/key',{provider:$('kp').value,key:$('key').value}).then(j=>{$('key').value='';show(j)})}
async function tick(){try{const s=await (await fetch('/api/state')).json();
 $('pr').textContent=s.running?('RUNNING '+s.label):'idle';$('pr').className='pill '+(s.running?'on':'off');
 $('pc').textContent=s.recording?'recording':'not recording';$('pc').className='pill '+(s.recording?'on':'off');
 $('armb').textContent=s.armed_s>0?('ARMED - '+s.armed_s+' s left (click to extend)'):'ARM motion for 10 min (only you press this)';$('armb').style.background=s.armed_s>0?'#1f7a4f':'#7a5b00';
 $('pk').textContent=s.key?'ER-2 key set':'ER-2 key missing';$('pk').className='pill '+(s.key?'on':'off');
 $('ee').textContent=s.ee?('gripper x='+s.ee[0]+' y='+s.ee[1]+' z='+s.ee[2]+' m'):'';
 if($('cam').options.length!=s.cams.length+1){$('cam').innerHTML='<option value=front>front (auto)</option>'+s.cams.map(c=>`<option>${c}</option>`).join('')}
 $('jt').innerHTML=Object.entries(s.joints).map(([i,j])=>`<tr><td>j${i}</td><td>${j.raw}</td><td><div class=bar><i style="width:${Math.max(0,Math.min(1,j.norm))*100}%"></i></div></td><td>${j.mA} mA</td></tr>`).join('');
 {const l=await (await fetch('/api/log')).text();const e=$('lg');const atEnd=e.scrollTop+e.clientHeight>=e.scrollHeight-30;e.textContent=l||'no run yet';if(atEnd)e.scrollTop=e.scrollHeight}
 const r=await (await fetch('/api/recs')).json();$('recs').innerHTML=r.map(x=>`<div><a style="color:var(--acc)" href="/rec/${x}/front.mp4">${x}</a></div>`).join('')}catch(e){}
 if(!$('ov').checked)ref()}
var rows='';for(var i=1;i<=8;i++){rows+='<tr><td>'+(i==8?'gripper':'j'+i)+'</td><td><button onclick="jog('+i+',-1)">&minus;</button><button onclick="jog('+i+',1)">+</button></td></tr>'}$('jg').innerHTML=rows;
function dlist(){fetch('/api/demo/list',{method:'POST',body:'{}'}).then(r=>r.json()).then(l=>{var sel=$('dl'),cur=sel.value;sel.innerHTML=l.map(x=>'<option value='+x.name+'>'+x.name+(x.success===true?' [OK]':x.success===false?' [bad]':'')+'</option>').join('');if(cur)sel.value=cur})}
setInterval(dlist,4000);dlist();
setInterval(tick,700);tick();$('ov').onchange=ref;
</script>"""

class H_(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def send(self, code, body, ctype="application/json"):
        b = body if isinstance(body, bytes) else (body if isinstance(body, str) else json.dumps(body)).encode()
        self.send_response(code); self.send_header("Content-Type", ctype); self.send_header("Content-Length", str(len(b))); self.send_header("Cache-Control", "no-store"); self.end_headers(); self.wfile.write(b)
    def do_GET(self):
        u = urlparse(self.path); q = parse_qs(u.query)
        try:
            if u.path == "/": return self.send(200, PAGE, "text/html; charset=utf-8")
            if u.path == "/frame.jpg": return self.send(200, jpeg(q.get("cam", ["front"])[0]), "image/jpeg")
            if u.path == "/detect.jpg":
                fs = [f for f in (ROOT / "run/ui_detect.jpg", ROOT / "run/grab_target.jpg") if f.exists()]; return self.send(200, max(fs, key=lambda f: f.stat().st_mtime).read_bytes(), "image/jpeg")
            if u.path == "/api/state": return self.send(200, state())
            if u.path == "/api/log": return self.send(200, "\n".join(l for l in proc["log"].read_text(errors="replace").splitlines() if not any(k in l for k in ("b3Warning", "axis of motion", "pybullet build", "warnings.warn", "RequestsDependency")))[-3000:], "text/plain")
            if u.path == "/api/recs": return self.send(200, sorted([d.name for d in REC.iterdir() if d.is_dir()], reverse=True)[:15])
            if u.path.startswith("/rec/"):
                f = (REC / u.path[5:]).resolve()
                if REC.resolve() in f.parents and f.is_file(): return self.send(200, f.read_bytes(), "video/mp4" if f.suffix == ".mp4" else "application/octet-stream")
            self.send(404, "not found", "text/plain")
        except Exception as e: self.send(503, f"{type(e).__name__}: {e}"[:200], "text/plain")
    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0) or 0); b = json.loads(self.rfile.read(n) or b"{}"); p = urlparse(self.path).path
        try:
            if p == "/api/detect": return self.send(200, detect())
            if p == "/api/jog": return self.send(200, jog(b.get("joint"), b.get("delta", 0)))
            if p == "/api/proceed": (ROOT / "run/proceed").write_text("1"); return self.send(200, {"ok": "proceed sent"})
            if p == "/api/arm": ARM["until"] = time.time() + (600 if b.get("on", True) else 0); (ROOT / "run/arm_until").write_text(str(ARM["until"])); return self.send(200, {"armed_for_s": 600 if b.get("on", True) else 0})
            if p == "/api/ask": return self.send(200, ask_gemini(b.get("question", "Describe the scene.")))
            if p == "/api/demo/record": return self.send(200, demo_record(bool(b.get("on", True)), b.get("name", "demo")))
            if p == "/api/demo/list": return self.send(200, demo_list())
            if p == "/api/demo/mark": return self.send(200, demo_mark(b["name"], b.get("success"), b.get("note", "")))
            if p == "/api/demo/play": return self.send(200, armd("/play", {"name": b["name"], "scale": b.get("scale", 0.5)}))
            if p == "/api/stop": return self.send(200, stop_run())
            if p == "/api/run": return self.send(200, start_run(bool(b.get("go")), b.get("vis", "er2"), b.get("speed", 250), bool(b.get("rec")), b.get("task", "marker")))
            if p == "/api/record":
                if b.get("on"): return self.send(200, {"recording": REC_S.start(b.get("label") or "scenario")})
                REC_S.stop(); return self.send(200, {"recording": False})
            if p == "/api/key":
                prov = b.get("provider", "gemini"); fn = {"gemini": ".gemini_er_key", "nebius": ".nebius_key", "anthropic": ".anthropic_key"}.get(prov)
                k = (b.get("key") or "").strip()
                if not fn or len(k) < 20: return self.send(400, {"error": "unknown provider or that does not look like a key"})
                kf = Path.home() / fn; old = os.umask(0o177); kf.write_text(k); os.umask(old); os.chmod(kf, 0o600); return self.send(200, {"ok": f"{prov} key saved to ~/{fn} (not shown again)"})
            self.send(404, {"error": "unknown"})
        except Exception as e: self.send(500, {"error": f"{type(e).__name__}: {e}"[:300]})

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--port", type=int, default=8890); a = ap.parse_args()
    print(f"ElRobot control interface: http://127.0.0.1:{a.port}  (local only)", flush=True); ThreadingHTTPServer(("127.0.0.1", a.port), H_).serve_forever()
