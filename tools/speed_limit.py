#!/usr/bin/env python3
"""
Cap servo speed + acceleration on a bus (ST3215 RAM registers Goal Speed 0x2E, Acc 0x29).

  python3 tools/speed_limit.py                     # all buses, speed 400, accel 20
  python3 tools/speed_limit.py --bus 5B8E114691 --speed 300 --accel 10
  python3 tools/speed_limit.py --show              # just print current values

Units: speed in steps/s (4096 steps = 360 deg -> 400 ~ 35 deg/s). accel 0..254 (lower = softer).

Applies to every position move that does NOT set its own speed (our tools, web UI joint control).
It is LOST on servo reboot / calibration reset / power cycle, and OVERWRITTEN by:
  - Station auto-calibration (hard-coded speed 365, accel 50)
  - Station web-UI mirroring (hard-coded up to 3300)  -> use the Python teleop example with --max-speed
Re-run after any of those.
"""
import argparse, asyncio, logging, struct, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import estop  # noqa: E402

GOAL_SPEED, ACC = 0x2E, 0x29


def show(buses):
    for bus, motors in buses.items():
        vals = " ".join(f"m{m}:spd={struct.unpack('<H', bytes(s)[GOAL_SPEED:GOAL_SPEED + 2])[0]},acc={bytes(s)[ACC]}"
                        for m, s in sorted(motors.items()))
        print(f"{bus}: {vals}")


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--bus")
    ap.add_argument("--speed", type=int, default=400)
    ap.add_argument("--accel", type=int, default=20)
    ap.add_argument("--show", action="store_true")
    a = ap.parse_args()
    if not (1 <= a.speed <= 3400 and 1 <= a.accel <= 254):
        sys.exit("speed must be 1..3400, accel 1..254 (0 means 'unlimited' on ST3215 - not allowed here)")

    client = await estop.new_station_client(a.host, logging.getLogger("speed"))
    buses = await estop.snapshot(client)
    if a.bus:
        buses = {k: v for k, v in buses.items() if k == a.bus}
    if a.show or not buses:
        show(buses) if buses else print("no buses found")
        return

    cmds = []
    for bus, motors in buses.items():
        for mid in sorted(motors):
            cmds.append(estop.st_cmd(bus, mid, GOAL_SPEED, a.speed.to_bytes(2, "little")))
            cmds.append(estop.st_cmd(bus, mid, ACC, bytes([a.accel])))
    await estop.send_commands(client, cmds)
    await asyncio.sleep(0.5)
    after = await estop.snapshot(client)
    show({b: after[b] for b in buses if b in after})


if __name__ == "__main__":
    asyncio.run(main())
