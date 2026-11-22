"""Fine-tuned ACT policy (hands) + Nebius VLM supervisor (eyes/judgement) for the spoon task. Filmed by film_task (act_vlm).
  loop: ACT runs on the real arm in 15 s segments (scripts/run_policy.py --backend station, Guard watchdogs);
        after each segment the VLM looks at a fresh front photo and names the stage:
        approaching | spoon_held | scooping | over_cup | spoon_in_cup | stuck | uncertain
  spoon_in_cup -> DONE.  stuck twice in a row -> home (folded start pose) and a fresh attempt (max 3).  uncertain -> continue.
  Gates: human ARM (>=150 s), mirroring OFF, start pose path-checked by armd. Every verdict + photo saved to results/photos/act_vlm_*."""
import json, subprocess, sys, time, urllib.request
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from normakit import llm
T0 = time.time(); log = lambda *m: print(f"[{time.time() - T0:5.1f}s]", *m, flush=True)
A = lambda p, b=None, t=60: json.loads(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8891" + p, data=None if b is None else json.dumps(b).encode(), method="GET" if b is None else "POST"), timeout=t).read())
TASK = "pick up the spoon, scoop beans and pour them into the cup"
FOLD = {"1": 2094, "2": 1055, "3": 1554, "4": 3020, "5": 170, "6": 2836, "7": 3021, "8": 2766}
STAGE_Q = ("Front camera of a robot arm doing this task: pick up the wooden spoon from the bowl of coffee beans, scoop beans, and put the spoon with the beans into the black cup. "
           "Which stage is visible now? Choose one: approaching (gripper not yet holding the spoon), spoon_held (spoon lifted in the gripper), scooping, over_cup "
           "(spoon held above the cup), spoon_in_cup (the spoon is inside the cup, task done), stuck (arm not progressing, spoon dropped, or something knocked over), uncertain. "
           'Reply as JSON: {"stage": "...", "reason": "short"}')
def front():
    for _ in range(5):
        try: return urllib.request.urlopen("http://127.0.0.1:8893/frame.jpg?c=table", timeout=5).read()
        except Exception: time.sleep(0.5)
    return None
IN_CUP_Q = ("Look carefully at the black cup on the right side of the table. Is the wooden spoon now standing or lying INSIDE the black cup (not in the white bowl, "
            "not held above it, not on the table)? Reply as JSON: {\"answer\": \"yes\" or \"no\", \"reason\": \"short\"}")
def judge(tag):
    """done = two separate yes answers on two fresh photos (a single VLM 'yes' is not trusted: it said 'spoon held' with the arm folded away)"""
    votes = []
    for k in range(2):
        jp = front()
        if jp is None: return "uncertain", "no fresh frame"
        (ROOT / f"results/photos/act_vlm_{tag}_{k}_{time.strftime('%H%M%S')}.jpg").write_bytes(jp)
        c = llm.config(); r = llm.ask(c["provider"], c["model"], jp, IN_CUP_Q, max_tokens=3000, timeout=120)
        ans = "uncertain" if r.get("incomplete") else str(r.get("answer", "uncertain")).lower(); votes.append((ans, str(r.get("reason", ""))[:90])); time.sleep(0.5)
    if all(v[0].startswith("yes") for v in votes): return "spoon_in_cup", votes[0][1]
    return "not_done", "; ".join(f"{v[0]}: {v[1]}" for v in votes)
def home():
    r = A("/move_joints", {"ticks": FOLD, "speed": 250, "wait": True, "timeout": 30}); return bool(r.get("ok")), r.get("error")
def act_segment(steps=150):
    p = subprocess.run([str(ROOT / ".venv/bin/python"), "-u", str(ROOT / "scripts/run_policy.py"), "--backend", "station", "--i-have-a-human", "--url", "http://127.0.0.1:8000",
                        "--task", TASK, "--steps", str(steps), "--max-delta", "0.03"], cwd=ROOT, capture_output=True, text=True)
    tail = [l for l in (p.stdout + p.stderr).splitlines() if l.strip() and "warn" not in l][-2:]; return p.returncode, " | ".join(tail)[:200]
h = json.loads(urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=5).read()); log("policy server:", h)
if h.get("cameras") != ["table", "wrist"]: log("STOP - server is not the two-camera checkpoint"); sys.exit(1)
log("WAITING: arm must be enabled (ARM motion button)")
while A("/state")["armed_s"] < 150: time.sleep(1)
A("/mirror", {"on": False})
for attempt in range(1, 4):
    ok, err = home(); log(f"attempt {attempt}: home -> ok={ok} {err or ''}")
    if not ok: A("/stop", {}); log("STOP - cannot reach the start pose"); sys.exit(1)
    stuck = 0
    for seg in range(1, 7):                                   # up to 6 x 15 s = 90 s per attempt
        if A("/state")["armed_s"] < 25: A("/stop", {}); log("STOP - ARM about to expire (press ARM to continue)"); sys.exit(1)
        t_before = A("/state")["tip"]; rc, tail = act_segment(); t_after = A("/state")["tip"]; moved = sum((x - y) ** 2 for x, y in zip(t_before, t_after)) ** 0.5
        log(f"  ACT segment {seg}: exit {rc}, gripper moved {moved * 100:.1f} cm | {tail}")
        if rc != 0: A("/stop", {}); log("STOP - policy run stopped by the guard/e-stop"); sys.exit(2)
        stage, why = judge(f"a{attempt}s{seg}"); log(f"  VLM stage: {stage} ({why})")
        if stage.startswith("spoon_in_cup"): A("/stop", {}); home(); log("DONE - VLM: spoon is in the cup"); sys.exit(0)
        stuck = stuck + 1 if moved < 0.01 else 0
        if stuck >= 2: log("  stuck twice - going home for a fresh attempt"); break
A("/stop", {}); home(); log("STOP - no success after 3 attempts (see the photos/verdicts)")
