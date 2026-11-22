"""Claude-planned spoon grasp (2026-11-22, last attempt before the presentation): jaws ACROSS the handle (roll 1.8 -> 89 deg to the handle in the
MuJoCo jaw-axis check), at the spot where the handle filled the jaw view (tip 0.202/0.242), tips lowered to ~0.045 so the handle is inside the pads;
empty -> open, lift, shift across the handle by +1/-1/+2/-2 cm. Grip = jaws stopped above the empty value with load. Front+wrist photo per try."""
import json, sys, time, urllib.request
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
A = lambda p, b=None, t=60: json.loads(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8891" + p, data=None if b is None else json.dumps(b).encode(), method="GET" if b is None else "POST"), timeout=t).read())
log = lambda *m: print(*m, flush=True)
def snap(tag):
    for c in ("table", "wrist"):
        try: (ROOT / f"results/photos/claude_grasp_{tag}_{c}_{time.strftime('%H%M%S')}.jpg").write_bytes(urllib.request.urlopen(f"http://127.0.0.1:8893/frame.jpg?c={c}", timeout=5).read())
        except Exception: pass
def mv(x, y, z, sp=100):
    r = A("/move_tip", {"x": float(x), "y": float(y), "z": float(z), "roll": ROLL, "speed": sp, "timeout": 20}); return bool(r.get("ok")), r.get("error")
ROLL = 1.8; BASE = np.array([0.202, 0.242]); ACROSS = np.array([-0.35, -0.74]); ACROSS /= np.linalg.norm(ACROSS)
log("WAITING: arm must be enabled (ARM motion button)")
while A("/state")["armed_s"] < 90: time.sleep(1)
A("/mirror", {"on": False})
for n, off in enumerate((0.0, 0.01, -0.01, 0.02, -0.02), 1):
    xy = BASE + off * ACROSS; A("/gripper", {"to": "open"}); time.sleep(0.5)
    ok, err = mv(xy[0], xy[1], 0.11, 150); log(f"try {n}: offset {off * 100:+.0f} cm across -> above {np.round(xy, 3)} ok={ok} {err or ''}")
    if not ok: continue
    ok, err = mv(xy[0], xy[1], 0.040, 80); t = A("/state")["tip"]; log(f"  down: ok={ok} {err or ''} tip {np.round(t, 3)}")
    if not ok: continue
    time.sleep(0.4); snap(f"t{n}_open")
    A("/gripper", {"to": "close"}); time.sleep(0.8); j = A("/state")["joints"]["8"]; log(f"  jaws {j['raw']} ticks, {j['mA']} mA (empty = 1323)")
    snap(f"t{n}_closed")
    if j["raw"] > 1340 and j["mA"] > 30:
        ok, err = mv(t[0], t[1], t[2] + 0.06, 80); time.sleep(0.6); snap(f"t{n}_lifted"); j = A("/state")["joints"]["8"]
        log(f"  lifted 6 cm: ok={ok}, jaws {j['raw']} ticks {j['mA']} mA")
        if j["raw"] > 1340: log("DONE - SPOON GRASPED AND LIFTED"); sys.exit(0)
        log("  lost it while lifting")
    mv(t[0], t[1], 0.11, 150)
A("/gripper", {"to": "open"}); log("STOP - no grasp in 5 tries")
