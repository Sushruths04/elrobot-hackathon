"""Read-only 'match meter': shows how far each LEADER joint is from the FOLLOWER (in % of range), so you can pose the leader like the follower
BEFORE starting Station's mirroring (which jumps the follower to the leader's percentages). Prints MATCHED when every joint is within 6% for 2 s."""
import asyncio, json, os, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from normakit.busread import snapshot
B = json.loads((ROOT / "run/buses.json").read_text()); L, F = B["leader"], B["follower"]
async def main():
    s, c, q = await snapshot(); ok_since = None; last = 0; was = None
    while True:
        s, _, _ = await snapshot(client=c, queue=q)
        d = {i: (s[L][i]["norm"] - s[F][i]["norm"]) * 100 for i in range(1, 9)}; good = all(abs(v) < 6 for v in d.values())
        ok_since = (ok_since or time.time()) if good else None; matched = ok_since is not None and time.time() - ok_since > 2
        if matched != was or time.time() - last > 6:
            last = time.time(); was = matched
            bad = ", ".join(f"j{i} {'raise' if v < 0 else 'lower'} leader by {abs(v):.0f}%" for i, v in d.items() if abs(v) >= 6)
            print(("MATCHED - safe to start mirroring" if matched else f"NOT matched: {bad or 'hold still...'}"), flush=True)
        await asyncio.sleep(0.4)
asyncio.run(main())
