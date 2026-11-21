"""Record PAIRED poses of leader and follower (read-only, torque stays off) to fit a per-joint leader->follower mapping.
You put BOTH arms in the same physical pose by hand and hold ~2 s; a pair is saved when both arms are still.
Move one or two joints at a time (e.g. only the shoulder ~40 deg on both arms), so every joint gets several different values.
    .venv/bin/python scripts/pose_pairs.py                     ->  run/pose_pairs.json"""
import asyncio, json, os, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from normakit.busread import snapshot
B = json.loads((ROOT / "run/buses.json").read_text()); L, F = B["leader"], B["follower"]
async def main():
    snap, c, q = await snapshot(); hist = []; out = ROOT / "run/pose_pairs.json"; pairs = json.loads(out.read_text()) if out.exists() else []; last_msg = 0
    def say(m, every=0):
        nonlocal last_msg
        if time.time() - last_msg >= every: print(m, flush=True); last_msg = time.time()
    print("READY: put BOTH arms in the same pose, hold 2 s, then change one or two joints and repeat. Joints 1-7; gripper/trigger last.", flush=True)
    while True:
        snap, _, _ = await snapshot(client=c, queue=q); now = time.time()
        try: ls = {i: snap[L][i]["pos"] for i in range(1, 9)}; fs = {i: snap[F][i]["pos"] for i in range(1, 9)}
        except KeyError: say("waiting for all 8 motors on both arms...", 5); continue
        hist.append((now, ls, fs)); hist[:] = [h for h in hist if h[0] > now - 1.4]
        if now - hist[0][0] < 1.2: continue
        sp = max(max(abs(h[k][i] - hist[-1][k][i]) for h in hist) for k in (1, 2) for i in range(1, 9))
        if sp > 12: say("moving...", 8); continue                           # >12 ticks (~1 deg) of movement in the window
        if pairs:
            d = max(abs(ls[i] - pairs[-1]["leader"][str(i)]) for i in range(1, 9))
            if d < 60 and max(abs(fs[i] - pairs[-1]["follower"][str(i)]) for i in range(1, 9)) < 60: say("same pose as the last pair - change a joint", 8); continue
        pairs.append({"leader": {str(i): v for i, v in ls.items()}, "follower": {str(i): v for i, v in fs.items()}}); out.write_text(json.dumps(pairs, indent=1))
        sp = lambda arm, i: max(p[arm][str(i)] for p in pairs) - min(p[arm][str(i)] for p in pairs)
        cov = {i: min(sp("leader", i), sp("follower", i)) for i in range(1, 9)}      # a joint only counts when BOTH arms moved
        need = [f"j{i}" for i in range(1, 8) if cov[i] < 500]
        print(f"pair {len(pairs)} saved | joint spread on BOTH arms (min of leader/follower, need >=500): " + " ".join(f"j{i}:{cov[i]}" for i in range(1, 9)) + f" | still need movement in: {', '.join(need) or 'none'}", flush=True)
        if len(pairs) >= 10 and not need: print("ENOUGH: all joints 1-7 covered - you can stop", flush=True)
        await asyncio.sleep(1.0)
asyncio.run(main())
