"""Autonomous tissue task, quick version: (1) sweep the flip-top cap open sideways from the top with camera checks, retrying higher up to 3 times (tempo_open.py --z0),
(2) pinch the tissue TIP near the top and pull it out, putting the pack down and retrying if the whole pack comes up (tempo_pull.py).  Filmed + recorded via film_task.py tempo_full."""
import subprocess, sys, time
from pathlib import Path
R = Path(__file__).resolve().parents[1]; PY = str(R / ".venv/bin/python"); T0 = time.time(); log = lambda *m: print(f"[{time.time()-T0:5.1f}s]", *m, flush=True)
def run(name, extra):
    p = subprocess.run([PY, str(R / "scripts" / name)] + extra, cwd=R, capture_output=True, text=True); out = p.stdout
    for ln in out.splitlines():
        if ln.strip() and "arn" not in ln: print("   |", ln.strip().replace("DONE", "step done").replace("STOP", "step stop")[:170], flush=True)
    return p.returncode, out
log("WAITING: arm must be enabled (ARM motion button)")
ok = False
for k, z0 in enumerate((0.095, 0.088, 0.081, 0.074, 0.068), 1):
    log(f"cap attempt {k}: sweep from the top at z {z0}"); rc, out = run("tempo_open.py", ["--z0", str(z0)])
    if rc != 0: log(f"cap attempt {k} ended with code {rc}"); sys.exit(rc)
    if "RESULT: cap open" in out: ok = True; log("the camera says the cap is open"); break
    log("the camera says the cap is still closed - retrying")
if not ok: log("DONE - the cap would not open after 5 attempts"); sys.exit(0)
rc, out = run("tempo_pull.py", ["--zs", "0.060,0.066,0.072"])
log("DONE - cap opened; tissue pull finished, check run/tempo_pull_final_pulled.jpg" if rc == 0 else f"STOP: the tissue step ended with code {rc}")
