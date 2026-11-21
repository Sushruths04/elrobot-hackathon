"""Read-only: save each bus's calibration (min,max,midpoint,span,frozen) to calibration/<serial>_<role>_<time>.json
and refresh run/calibration_backup.json. Buses that are not fully frozen are saved with "complete": false."""
import asyncio, json, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from normakit.busread import snapshot
async def main():
    snap, _, _ = await snapshot()
    roles = json.loads((ROOT / "run/buses.json").read_text())
    role_of = {v: k for k, v in roles.items()}
    (ROOT / "calibration").mkdir(exist_ok=True)
    ts, allb = time.strftime("%Y%m%d_%H%M%S"), {"saved": time.strftime("%F %T"), "buses": {}}
    for bus, ms in snap.items():
        arcs = {}
        for i, m in sorted(ms.items()):
            lo, hi = m["lo"], m["hi"]; span = (hi - lo) % 4096
            arcs[i] = dict(min=lo, max=hi, midpoint=(lo + span // 2) & 4095, span=span, frozen=m["frozen"])
        complete = all(a["frozen"] and a["span"] >= 100 for a in arcs.values())
        rec = {"bus": bus, "role": role_of.get(bus, "?"), "saved": time.strftime("%F %T"), "complete": complete, "motors": arcs}
        out = ROOT / "calibration" / f"{bus}_{rec['role']}_{ts}.json"
        out.write_text(json.dumps(rec, indent=1)); print(("OK  " if complete else "PARTIAL ") + str(out.relative_to(ROOT)))
        allb["buses"][bus] = arcs
    (ROOT / "run/calibration_backup.json").write_text(json.dumps(allb, indent=1))
    sys.stdout.flush(); import os; os._exit(0)
asyncio.run(main())
