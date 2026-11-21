#!/usr/bin/env python3
"""
Calibration helper for the ElRobot arms via Station (no web UI needed - the web-UI calibration
page auto-resets the bus when you open it).

  python3 tools/calibrate.py watch  [--bus leader|follower|SN]   # READ-ONLY live spans vs reference
  python3 tools/calibrate.py reset  --bus ...   # stop mirror + ResetCalibration (torque OFF, offsets cleared)
  python3 tools/calibrate.py save   --bus ...   # FreezeCalibration (= web-UI Save): centre on 2048, write offsets
  python3 tools/calibrate.py auto   --bus ...   # AutoCalibrate (ARM MOVES into its hard stops!)
  python3 tools/calibrate.py stop   --bus ...   # StopAutoCalibrate

Manual recipe: reset -> watch -> move every joint by hand to BOTH hard stops (incl. gripper/trigger)
until every span is "ok" -> save -> watch shows SAVED.
Reference spans: calibration/5B8E114691_follower_*.json (Sushruth's auto-cal, matched URDF within ~1%).
Safety: AGENTS.md - a human must be at the table and agree before reset/auto.
"""
import argparse, asyncio, glob, json, logging, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import estop  # noqa: E402  (sets up repo paths)
from estop import new_station_client, send_commands, st3215, commands, drivers, raw_pos  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
BUSES = {"leader": "5B79016336", "follower": "5B8E114691"}
ROLE = {v: k for k, v in BUSES.items()}


def reference_spans():
    files = sorted(glob.glob(str(REPO / "calibration" / "5B8E114691_follower_*.json")))
    if not files:
        return {}
    return {int(k): v["span"] for k, v in json.load(open(files[-1]))["motors"].items()}


def span(lo, hi):
    return (hi - lo) % 4096


def drv(c):
    return commands.DriverCommand(type=drivers.StationCommandType.STC_ST3215_COMMAND, body=c.encode())


_queues = {}


async def read_state(client, timeout=3.0):
    # subscribe ONCE per client (a new follow() per call piles up subscriptions -> timeouts)
    q = _queues.get(id(client))
    if q is None:
        q = _queues[id(client)] = asyncio.Queue()
        client.follow("st3215/inference", q)
    e = await asyncio.wait_for(q.get(), timeout)
    while not q.empty():          # skip to the newest entry
        e = q.get_nowait()
    st = st3215.InferenceStateReader(memoryview(bytes(e.Data)))
    out = {}
    for b in st.get_buses() or []:
        bi = b.get_bus()
        if not bi:
            continue
        ac = b.get_auto_calibration()
        auto = None
        if ac:
            auto = (str(ac.get_status()), ac.get_current_step(), ac.get_total_steps(),
                    ac.get_current_phase(), ac.get_error_message())
        motors = {}
        for m in b.get_motors() or []:
            if m.get_state():
                motors[m.get_id()] = (raw_pos(m.get_state()), m.get_range_min(), m.get_range_max(), m.get_range_freezed())
        out[bi.get_serial_number()] = (motors, auto)
    return out


_prev = {}


def render(state, only, ref):
    lines = []
    for sn, (motors, auto) in sorted(state.items()):
        if only and sn != only:
            continue
        lines.append(f"bus {sn} ({ROLE.get(sn, '?')})" + (f"  auto-cal: {auto}" if auto and auto[1] else ""))
        for mid, (pos, lo, hi, frozen) in sorted(motors.items()):
            s, r = span(lo, hi), ref.get(mid)
            pct = f"{100 * s / r:4.0f}%" if r else "   -"
            flag = "SAVED" if frozen else ("ok" if r and s >= 0.95 * r else "keep moving")
            old = _prev.get((sn, mid), pos)
            moved = min((pos - old) % 4096, (old - pos) % 4096)
            _prev[(sn, mid)] = pos
            mark = "   <<< MOVING" if moved > 3 else ""
            lines.append(f"  m{mid}: pos={pos:4d} range=[{lo:4d},{hi:4d}] span={s:4d} ref={r or '-':>4} {pct}  {flag}{mark}")
    return "\n".join(lines)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["watch", "reset", "save", "auto", "stop"])
    ap.add_argument("--bus", help="leader | follower | bus serial")
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--once", action="store_true", help="watch: print once and exit")
    a = ap.parse_args()
    bus = BUSES.get(a.bus, a.bus)
    if a.action != "watch" and not bus:
        sys.exit("--bus is required for " + a.action)

    client = await new_station_client(a.host, logging.getLogger("calibrate"))
    if a.action == "watch":
        ref = reference_spans()
        while True:
            txt = render(await read_state(client), bus, ref)
            if a.once:
                print(txt); return
            print("\033[2J\033[H" + time.strftime("%H:%M:%S") + "  (Ctrl+C to quit)\n" + txt, flush=True)
            await asyncio.sleep(0.3)

    if a.action == "reset":
        cmds = [estop.stop_mirror_cmd(bus), drv(st3215.Command(target_bus_serial=bus,
                reset_calibration=st3215.ResetCalibrationCommand(reset=True)))]
    elif a.action == "save":
        cmds = [drv(st3215.Command(target_bus_serial=bus, freeze_calibration=st3215.FreezeCalibrationCommand(freeze=True)))]
    elif a.action == "auto":
        cmds = [estop.stop_mirror_cmd(bus), drv(st3215.Command(target_bus_serial=bus,
                auto_calibrate=st3215.AutoCalibrateCommand(calibrate=True)))]
    else:
        cmds = [drv(st3215.Command(target_bus_serial=bus, stop_auto_calibrate=st3215.StopAutoCalibrateCommand(stop=True)))]
    for c in cmds:
        await send_commands(client, [c])
    print(f"{a.action} sent to bus {bus} ({ROLE.get(bus, '?')})")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
