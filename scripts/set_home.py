"""Read-only. Save the CURRENT physical pose of an arm as its HOME pose (percent of each joint's calibrated range).
Pose the arm by hand first (torque is off). Usage:
    .venv/bin/python scripts/set_home.py leader      # hold the leader in the pose you want, then run
    .venv/bin/python scripts/set_home.py follower    # same for the follower
Teleop maps leader% -> follower%, so ONE pose (same %) is the matching home for both arms."""
import asyncio, json, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from normakit.busread import snapshot

role = sys.argv[1] if len(sys.argv) > 1 else "leader"
buses = json.loads((ROOT / "run/buses.json").read_text())
async def main():
    snap, _, _ = await snapshot()
    ms = snap[buses[role]]
    bad = [i for i, m in ms.items() if (m["hi"] - m["lo"]) % 4096 < 100]
    if bad: sys.exit(f"motors {bad} not calibrated")
    p = ROOT / "run/home.json"
    home = json.loads(p.read_text()) if p.exists() else {}
    home[role] = {"norm": {str(i): round(m["norm"], 4) for i, m in ms.items()},
                  "raw": {str(i): m["pos"] for i, m in ms.items()}, "bus": buses[role], "saved": time.strftime("%F %T")}
    p.write_text(json.dumps(home, indent=1))
    print(f"saved {role} home:", {i: f"{v*100:.0f}%" for i, v in home[role]["norm"].items()})
    sys.stdout.flush(); import os; os._exit(0)
asyncio.run(main())
