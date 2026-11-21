"""Live watcher for a manual calibration (read-only). Shows each joint's calibrated span as a % of the span the arm model expects,
announces when a joint crosses 85%, when a bus is saved (frozen) or reset, and prints a summary line every 15 s.
    .venv/bin/python scripts/watch_calib.py"""
import asyncio, json, os, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from normakit.busread import snapshot
EXPECT = {1: 2021, 2: 2100, 3: 2294, 4: 2284, 5: 3826, 6: 2048, 7: 3866, 8: 1435}      # ticks, from the URDF joint limits
role = {v: k.upper() for k, v in json.loads((ROOT / "run/buses.json").read_text()).items()}
async def main():
    snap, c, q = await snapshot(); last_sum = 0; over = {}; frozen = {}
    while True:
        snap, _, _ = await snapshot(client=c, queue=q)
        for bus, ms in snap.items():
            name = role.get(bus, bus); pct = {}
            for i, m in ms.items(): pct[i] = 100 * ((m["hi"] - m["lo"]) % 4096) / EXPECT[i]
            for i, p in pct.items():
                k = (bus, i)
                if p >= 85 and not over.get(k): print(f"{name} j{i} reached {p:.0f}% of its expected range", flush=True)
                over[k] = p >= 85
            fz = all(m["frozen"] for m in ms.values())
            if frozen.get(bus) is not None and fz != frozen[bus]: print(f"{name} calibration {'SAVED (frozen)' if fz else 'RESET / unfrozen'}", flush=True)
            frozen[bus] = fz
        if time.time() - last_sum > 15:
            last_sum = time.time()
            for bus, ms in snap.items():
                name = role.get(bus, bus); parts = [f"j{i} {100*((m['hi']-m['lo'])%4096)/EXPECT[i]:.0f}%" for i, m in sorted(ms.items())]
                short = [f"j{i}" for i, m in sorted(ms.items()) if 100*((m['hi']-m['lo'])%4096)/EXPECT[i] < 85]
                print(f"SUMMARY {name}: {' '.join(parts)} | {'frozen' if all(m['frozen'] for m in ms.values()) else 'NOT saved'} | still short: {', '.join(short) or 'none'}", flush=True)
asyncio.run(main())
