#!/usr/bin/env python3
"""
Software stop for the ElRobot arms via NormaCore Station.

  python3 tools/estop.py            # HOLD: stop teleop mirroring + freeze every motor at its current pose
  python3 tools/estop.py --limp     # LIMP: torque OFF on every motor (arm drops under gravity - support it!)
  python3 tools/estop.py --bus 5B8E114691   # only one bus

This is a convenience, NOT a safety device: it needs the Station + USB to work.
The real e-stop is the servo power supply plug - keep it within reach.
"""
import argparse, asyncio, logging, os, struct, sys, time
from pathlib import Path

_HERE = Path(__file__).resolve().parent.parent
# upstream source: $NORMA_CORE, else vendor/norma-core-src (Sushruth's layout), else repo/norma-core
ROOT = Path(os.environ.get("NORMA_CORE") or next(
    (p for p in (_HERE / "vendor" / "norma-core-src", _HERE / "repo" / "norma-core") if p.exists()),
    _HERE / "repo" / "norma-core"))
sys.path[:0] = [str(ROOT / "software" / "station" / "shared"), str(ROOT)]

from station_py import new_station_client, send_commands  # noqa: E402
from target.gen_python.protobuf.station import commands, drivers  # noqa: E402
from target.gen_python.protobuf.drivers.st3215 import st3215  # noqa: E402
from target.gen_python.protobuf.drivers.motors_mirroring import mirroring  # noqa: E402

PRESENT_POSITION, GOAL_POSITION, TORQUE_ENABLE = 0x38, 0x2A, 0x28


def raw_pos(state):
    b = bytes(state)
    p = struct.unpack("<H", b[PRESENT_POSITION:PRESENT_POSITION + 2])[0]
    return ((4096 - (p & 4095)) & 4095) if p & 0x8000 else p & 4095


def st_cmd(bus, motor, addr, value: bytes):
    c = st3215.Command(target_bus_serial=bus, write=st3215.ST3215WriteCommand(motor_id=motor, address=addr, value=value))
    return commands.DriverCommand(type=drivers.StationCommandType.STC_ST3215_COMMAND, body=c.encode())


def stop_mirror_cmd(bus):
    m = mirroring.Command(type=mirroring.CommandType.CT_STOP_MIRROR,
                          source=mirroring.MirroringBus(type=mirroring.BusType.MBT_ST3215, unique_id=bus))
    return commands.DriverCommand(type=drivers.StationCommandType.STC_MOTOR_MIRRORING_COMMAND, body=m.encode())


async def snapshot(client, timeout=3.0):
    q = asyncio.Queue()
    client.follow("st3215/inference", q)
    e = await asyncio.wait_for(q.get(), timeout)
    st = st3215.InferenceStateReader(memoryview(bytes(e.Data)))
    out = {}
    for b in st.get_buses() or []:
        bi = b.get_bus()
        if bi:
            out[bi.get_serial_number()] = {m.get_id(): m.get_state() for m in (b.get_motors() or []) if m.get_state()}
    return out


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--bus", help="only this bus serial (default: all)")
    ap.add_argument("--limp", action="store_true", help="torque OFF instead of holding pose")
    a = ap.parse_args()

    t0 = time.time()
    client = await new_station_client(a.host, logging.getLogger("estop"))
    buses = await snapshot(client)
    if a.bus:
        buses = {k: v for k, v in buses.items() if k == a.bus}
    if not buses:
        print("no buses found - CUT SERVO POWER if the arm is still moving")
        return

    cmds = []
    for bus, motors in buses.items():
        cmds.append(stop_mirror_cmd(bus))
        for mid, state in sorted(motors.items()):
            if a.limp:
                cmds.append(st_cmd(bus, mid, TORQUE_ENABLE, b"\x00"))
            else:
                cmds.append(st_cmd(bus, mid, GOAL_POSITION, raw_pos(state).to_bytes(2, "little")))
    for _ in range(2):  # send twice in case a packet is dropped
        await send_commands(client, cmds)
        await asyncio.sleep(0.05)
    mode = "TORQUE OFF" if a.limp else "HOLD current pose"
    print(f"{mode} sent to {len(buses)} bus(es): {', '.join(buses)}  ({(time.time() - t0) * 1000:.0f} ms)")


if __name__ == "__main__":
    asyncio.run(main())
