"""Real-arm run of the fine-tuned ACT spoon policy (models/act_spoon_5k via server/policy_server.py on :8000), filmed by film_task (act_spoon).
Gates: human ARM (>=120 s left), mirroring OFF, start at the demos' start pose (folded, path-checked by armd), then scripts/run_policy.py --backend station
(Guard: clamp, step limit, overcurrent e-stop) with --max-delta 0.03, 600 steps. Prints DONE/STOP for film_task."""
import json, os, subprocess, sys, time, urllib.request
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
A = lambda p, b=None, t=60: json.loads(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8891" + p, data=None if b is None else json.dumps(b).encode(), method="GET" if b is None else "POST"), timeout=t).read())
log = lambda *m: print(*m, flush=True)
h = json.loads(urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=5).read()); log("policy server:", h)
if h.get("cameras") != ["table", "wrist"]: log("STOP - server is not the two-camera checkpoint"); sys.exit(1)
log("WAITING: arm must be enabled (ARM motion button)")
while A("/state")["armed_s"] < 120: time.sleep(1)
log("mirroring:", A("/mirror", {"on": False}))
FOLD = {"1": 2094, "2": 1055, "3": 1554, "4": 3020, "5": 170, "6": 2836, "7": 3021, "8": 2766}      # demos' start pose
r = A("/move_joints", {"ticks": FOLD, "speed": 300, "wait": True, "timeout": 25}); log("start pose:", {k: r.get(k) for k in ("ok", "error", "worst_ticks_off")})
if not r.get("ok"): A("/stop", {}); log("STOP - could not reach the start pose (path check or timeout); not running the policy"); sys.exit(1)
time.sleep(1.0); log("RUNNING the ACT policy (600 steps at 10 Hz, max 0.03 per tick)")
p = subprocess.run([str(ROOT / ".venv/bin/python"), "-u", str(ROOT / "scripts/run_policy.py"), "--backend", "station", "--i-have-a-human", "--url", "http://127.0.0.1:8000",
                    "--task", "pick up the spoon, scoop beans and pour them into the cup", "--steps", "600", "--max-delta", "0.03"], cwd=ROOT)
A("/stop", {}); log("DONE - ACT policy run finished (exit %d); holding pose. Judge the video: grasp / scoop / beans in cup." % p.returncode)
