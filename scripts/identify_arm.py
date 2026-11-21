"""Read-only. Move ONE arm by hand during the window; the bus whose joints change is the one you are holding (= leader).
    .venv/bin/python scripts/identify_arm.py [seconds]"""
import asyncio, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from normakit.busread import snapshot

async def main(secs):
    snap, c, q = await snapshot()
    lo = {b: {i: m["pos"] for i, m in ms.items()} for b, ms in snap.items()}
    travel = {b: 0 for b in snap}
    print(f">>> MOVE THE ARM YOU WILL HOLD (the leader) NOW for {secs}s ...", flush=True)
    t_end = asyncio.get_event_loop().time() + secs
    prev = lo
    while asyncio.get_event_loop().time() < t_end:
        snap, _, _ = await snapshot(client=c, queue=q)
        for b, ms in snap.items():
            for i, m in ms.items():
                d = abs(m["pos"] - prev[b][i]); d = min(d, 4096 - d)
                travel[b] += d if d < 1500 else 0
        prev = {b: {i: m["pos"] for i, m in ms.items()} for b, ms in snap.items()}
    print("total joint travel (ticks):", travel)
    leader = max(travel, key=travel.get)
    follower = [b for b in travel if b != leader][0]
    if travel[leader] < 300 or travel[leader] < 3 * max(travel[follower], 1):
        print("NOT CONCLUSIVE - move only one arm, more clearly, and retry."); sys.stdout.flush(); import os; os._exit(2)
    (ROOT / "run/buses.json").write_text(json.dumps({"follower": follower, "leader": leader}))
    print(f"LEADER = {leader}   FOLLOWER = {follower}  (saved to run/buses.json)")
    sys.stdout.flush(); import os; os._exit(0)
asyncio.run(main(float(sys.argv[1]) if len(sys.argv) > 1 else 10))
