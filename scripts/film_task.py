#!/usr/bin/env python
"""Film an autonomous task while it runs: table-camera video (~20 fps, live caption of the task's log lines + timer) + follower joints, saved as H.264 in results/films/ and added to the showcase.
    .venv/bin/python scripts/film_task.py pen       # pick_loop.py: pen/marker into the white box
    .venv/bin/python scripts/film_task.py duster    # duster_caps.py: magnetic duster over the caps, then into the box
Reset the scene first, press ARM; recording starts when the task really starts (after ARM) and ends 4 s after it finishes."""
import json, os, re, subprocess, sys, threading, time, urllib.request
from datetime import datetime
from pathlib import Path
import cv2, numpy as np
ROOT = Path(__file__).resolve().parents[1]
TASKS = {"pen": ("scripts/pick_loop.py", "logs/pick_loop.log", "pen_marker_into_box"), "duster": ("scripts/duster_caps.py", "logs/duster.log", "duster_magnet_caps_into_box"), "tempo": ("scripts/tempo_open.py", "logs/tempo.log", "tempo_open_lid_push"), "tempo_hand": ("scripts/tempo_hand_demo.py", "logs/tempo_hand.log", "tempo_open_lid_hand_demo"), "tempo_pinch": ("scripts/tempo_pinch_demo.py", "logs/tempo_hand.log", "tempo_pinch_tissue_hand_demo"), "tempo_replay": ("scripts/tempo_replay.py", "logs/tempo_replay.log", "tempo_open_lid_replay"), "tempo_pull": ("scripts/tempo_pull.py", "logs/tempo_pull.log", "tempo_pull_tissue"), "tempo_full": ("scripts/tempo_full.py", "logs/tempo_full.log", "tempo_open_and_pull_autonomous"), "tempo_fast": ("scripts/tempo_fast.py", "logs/tempo_fast.log", "tempo_fast_full_task"), "tempo_auto": ("scripts/tempo_auto.py", "logs/tempo_auto.log", "tempo_auto_attempts_full_task"), "tempo_guided": ("scripts/tempo_guided.py", "logs/tempo_guided.log", "tempo_guided_session"), "scissors": ("scripts/tempo_pinch_demo.py", "logs/tempo_hand.log", "scissors_hand_taught_orientation_and_cut"), "tempo_final": ("scripts/tempo_final.py", "logs/tempo_final.log", "tempo_final_autonomous_run"), "tempo_wipe": ("scripts/tempo_wipe.py", "logs/tempo_wipe.log", "tempo_wipe_beans_with_tissue")}
task = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] in TASKS else sys.exit("usage: film_task.py pen|duster|tempo|tempo_hand|tempo_pinch|tempo_replay|tempo_pull|tempo_full|tempo_fast|tempo_wipe|tempo_auto|tempo_guided|scissors|tempo_final"); script, logf, name = TASKS[task]
LOG = ROOT / logf; LOG.write_text(""); stamp = datetime.now().strftime("%Y%m%d_%H%M%S"); films = ROOT / "results/films"; films.mkdir(parents=True, exist_ok=True); tmp = films / f"{name}_{stamp}_raw.mp4"; rawv = films / f"{name}_{stamp}_nocaption.mp4"; final = films / f"{name}_{stamp}.mp4"
proc = subprocess.Popen([str(ROOT / ".venv/bin/python"), str(ROOT / script)], cwd=ROOT, stdout=open(LOG, "w"), stderr=subprocess.STDOUT)
def lines():
    try: return [l for l in LOG.read_text(errors="replace").splitlines() if l.startswith("[") and "WAITING" not in l and "Requests" not in l and "warn" not in l.lower()]
    except Exception: return []
def frame(): return cv2.imdecode(np.frombuffer(urllib.request.urlopen("http://127.0.0.1:8890/frame.jpg", timeout=10).read(), np.uint8), cv2.IMREAD_COLOR)
def state():
    try: return json.loads(urllib.request.urlopen("http://127.0.0.1:8891/state", timeout=5).read())
    except Exception: return None
print("task started, waiting for ARM / the task to begin; recording starts with its first log line", flush=True)
while proc.poll() is None and not lines(): time.sleep(0.2)
t0 = time.time(); vw = None; vraw = None; samples = []; jl = open(films / f"{name}_{stamp}_joints.jsonl", "w"); n = 0; done_at = None; last_j = 0; END = re.compile(r"DONE|STOP at|gave up|stopping|NOT ARMED|ABORT|could not|slipped|Traceback")
while time.time() - t0 < 400:
    t = time.time()
    try:
        im = frame(); im0 = im.copy(); L = lines()[-3:]; cv2.rectangle(im, (0, 0), (640, 16 + 15 * max(1, len(L))), (0, 0, 0), -1)
        for i, l in enumerate(L): cv2.putText(im, re.sub(r"\s+", " ", l)[:92], (4, 13 + 15 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(im, f"t = {t - t0:5.1f} s", (540, 472), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
        if vw is None: vw = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), 20, (im.shape[1], im.shape[0]))
        vw.write(im); n += 1
        if vraw is None: vraw = cv2.VideoWriter(str(rawv), cv2.VideoWriter_fourcc(*"mp4v"), 20, (im0.shape[1], im0.shape[0]))
        vraw.write(im0)
        if t - last_j > 0.05:
            st = state()
            if st: samples.append((t - t0, [st['joints'][str(i)]['norm'] for i in range(1, 9)], [st['joints'][str(i)]['goal_norm'] for i in range(1, 9)]))
            if st: jl.write(json.dumps({"t": round(t - t0, 2), "tip": st["tip"], **{k: v["raw"] for k, v in st["joints"].items()}}) + "\n")
            last_j = t
    except Exception: pass
    if done_at is None and (proc.poll() is not None or (lines() and END.search(lines()[-1]))): done_at = time.time()
    if done_at and time.time() - done_at > 4: break
    time.sleep(max(0, 0.05 - (time.time() - t)))
vw and vw.release(); vraw and vraw.release(); jl.close(); proc.poll() is None and proc.terminate()
import av
real_fps = max(5, int(round(n / max(time.time() - t0 - 4, 1)))); cap = cv2.VideoCapture(str(tmp)); out = av.open(str(final), "w"); st = out.add_stream("libx264", rate=real_fps); st.width, st.height, st.pix_fmt = 640, 480, "yuv420p"; st.options = {"crf": "22"}; ok, im = cap.read()
while ok:
    for p in st.encode(av.VideoFrame.from_ndarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), format="rgb24")): out.mux(p)
    ok, im = cap.read()
for p in st.encode(): out.mux(p)
out.close(); tmp.unlink(); (films / f"{name}_{stamp}_samples.json").write_text(json.dumps(samples))      # keep the raw samples in case the dataset writer fails
try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    TASKTXT = {"pen": "pick up the pen or marker and place it in the white box", "duster": "pick up the magnetic duster, hold it over the metal bottle caps so the magnet takes them, then place the duster in the white box", "tempo": "open the white flip-top lid of the tissue pack by pushing it with the gripper (then pinch and pull one tissue, then sweep the coffee beans)", "tempo_hand": "open the flip-top lid of the tissue pack by lifting it sideways with the closed gripper (human moves the follower by hand)", "tempo_pinch": "pinch one tissue from the open pack and pull it out (human moves the follower by hand, gripper opens and closes on command)", "tempo_replay": "replay of the recorded human-guided demo that opens the flip-top lid of the tissue pack", "tempo_pull": "pinch the tissue sticking out of the opened pack and pull it out", "tempo_final": "final autonomous run: open the cap with a sideways sweep, pinch and pull one tissue, sweep the coffee beans", "scissors": "scissors cutting tape: the human moves the follower by hand with the scissors in the open jaws to show the orientation and the cut", "tempo_guided": "guided session: the human gives the moves, the arm executes them (cap, tissue, beans)", "tempo_auto": "tissue task with attempts: open the cap, pinch and pull a tissue, sweep the coffee beans, camera-verified at each stage", "tempo_wipe": "wipe the coffee beans sideways with the tissue held in the gripper", "tempo_fast": "whole tissue task in one run: sweep the cap open, pinch and pull one tissue straight up, sweep the coffee beans with it", "tempo_full": "sweep the flip-top cap of the tissue pack open sideways, then pinch the tissue tip and pull one tissue out (autonomous)"}[task]
    droot = ROOT / "datasets/autonomous" / f"{name}_{stamp}"; names = [f"joint_{i}" for i in range(1, 9)]
    ds = LeRobotDataset.create(f"local/{name}_{stamp}", 30, {"observation.state": {"dtype": "float32", "shape": (8,), "names": names}, "action": {"dtype": "float32", "shape": (8,), "names": names}, "observation.images.table": {"dtype": "video", "shape": (480, 640, 3), "names": ["height", "width", "channels"]}}, root=droot, robot_type="elrobot", use_videos=True)
    T = np.array([x[0] for x in samples]); S = np.array([x[1] for x in samples], np.float32); G = np.array([x[2] for x in samples], np.float32); frames = []; cap = cv2.VideoCapture(str(rawv)); ok, im = cap.read()
    while ok: frames.append(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)); ok, im = cap.read()
    fps_v = len(frames) / max(T[-1], 1e-3)
    if task == 'tempo_hand': G = np.vstack([S[3:], np.repeat(S[-1:], 3, 0)])      # hand-guided: no leader, so the action is the follower's own next state
    nfr = int(T[-1] * 30)
    for k in range(nfr):
        t = k / 30; ds.add_frame({"observation.state": np.array([np.interp(t, T, S[:, j]) for j in range(8)], np.float32), "action": np.array([np.interp(t, T, G[:, j]) for j in range(8)], np.float32), "observation.images.table": frames[min(len(frames) - 1, int(t * fps_v))], "task": TASKTXT})
    ds.save_episode(); ds.finalize() if hasattr(ds, "finalize") else None
    (droot / "OUTCOME.json").write_text(json.dumps({"task": task, "last_log_line": lines()[-1] if lines() else "", "success": None, "note": "success to be confirmed by the human who watched the run"}, indent=1)); print(f"LeRobot episode saved: {droot} ({nfr} frames at 30 fps)", flush=True)
except Exception as e: print("LeRobot episode NOT saved:", type(e).__name__, str(e)[:160], flush=True)
(films / f"{name}_{stamp}.log").write_text("\n".join(lines()) + "\n")
print(f"saved {final} ({n} frames, {time.time() - t0:.0f} s)  | last log line: {lines()[-1] if lines() else '-'}", flush=True)
subprocess.run([str(ROOT / ".venv/bin/python"), str(ROOT / "scripts/build_showcase.py")], cwd=ROOT, capture_output=True)
