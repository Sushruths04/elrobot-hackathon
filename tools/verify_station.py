#!/usr/bin/env python3
"""
One-shot NormaCore Station hardware check (read-only by default).

  python3 tools/verify_station.py                 # full check: USB serial, Station, motors, cameras
  python3 tools/verify_station.py --watch         # live joint positions (move arm by hand, torque off)
  python3 tools/verify_station.py --move-test --bus <SERIAL> --motor 1 --delta 60
                                                  # tiny wiggle on ONE motor, then back (opt-in)

Finds upstream norma-core at vendor/norma-core-src or repo/norma-core (override with NORMA_CORE=/path).
"""
import argparse, asyncio, glob, logging, os, re, struct, subprocess, sys, time
from pathlib import Path

_HERE = Path(__file__).resolve().parent.parent
# upstream source: $NORMA_CORE, else vendor/norma-core-src (Sushruth's layout), else repo/norma-core
ROOT = Path(os.environ.get("NORMA_CORE") or next(
    (p for p in (_HERE / "vendor" / "norma-core-src", _HERE / "repo" / "norma-core") if p.exists()),
    _HERE / "repo" / "norma-core"))
sys.path.insert(0, str(ROOT / "software" / "station" / "shared"))
sys.path.insert(0, str(ROOT))

from station_py import new_station_client, send_commands  # noqa: E402
from target.gen_python.protobuf.station import commands, drivers  # noqa: E402
from target.gen_python.protobuf.drivers.st3215 import st3215  # noqa: E402
from target.gen_python.protobuf.drivers.usbvideo import usbvideo, frame  # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("verify")

POS, VOLT, TEMP, CURR, OFFSET = 0x38, 0x3E, 0x3F, 0x45, 0x1F
GOAL_POS = 0x2A
OK, BAD, WARN = "\033[32m✔\033[0m", "\033[31m✘\033[0m", "\033[33m!\033[0m"


def norm_pos(p):
    return ((4096 - (p & 4095)) & 4095) if p & 0x8000 else p & 4095


def parse_motor(b):
    b = bytes(b)
    g = lambda a, n: (struct.unpack("<H", b[a:a + 2])[0] if n == 2 else b[a]) if len(b) >= a + n else 0
    off = struct.unpack("<h", b[OFFSET:OFFSET + 2])[0] if len(b) >= OFFSET + 2 else 0
    raw = norm_pos(g(POS, 2))
    # while a range is NOT saved, Station tracks raw + offset (port_meta.rs); saved -> raw (see collect_motors)
    return dict(pos=(raw + off + 4096) % 4096, raw=raw, off=off, status=g(0x41, 1), torque=g(0x28, 1), volt=g(VOLT, 1) / 10, temp=g(TEMP, 1), cur=g(CURR, 2))


# ---------------------------------------------------------------- 1. USB serial
def check_usb():
    print("\n[1] USB serial devices")
    devs = [d for d in glob.glob("/dev/cu.*")
            if not re.search(r"Bluetooth|debug|wlan|Beats|JBL|Headphones", d)]
    for d in devs:
        print(f"   {OK} {d}")
    if not devs:
        print(f"   {BAD} no /dev/cu.* serial device -> servo boards not enumerated "
              "(data cable? board powered? try a hub port a camera uses)")
    try:
        sp = subprocess.run(["system_profiler", "SPUSBHostDataType"], capture_output=True, text=True, timeout=20).stdout
        hits = re.findall(r"\n\s*([^\n]*(?:CH34|USB Serial|QinHeng|CP210|FT232|USB Single Serial)[^\n]*):", sp, re.I)
        for h in sorted(set(hits)):
            print(f"   {OK} system_profiler: {h.strip()}")
        if devs == [] and hits:
            print(f"   {WARN} chip visible in system_profiler but no /dev/cu.* -> driver issue")
    except Exception as e:
        print(f"   {WARN} system_profiler failed: {e}")
    return len(devs)


# ---------------------------------------------------------------- 2+3. motors
async def collect_motors(client, seconds):
    q = asyncio.Queue()
    client.follow("st3215/inference", q)
    buses, n, t0 = {}, 0, time.time()
    while time.time() - t0 < seconds:
        try:
            e = await asyncio.wait_for(q.get(), timeout=0.5)
        except asyncio.TimeoutError:
            continue
        if e is None:
            break
        n += 1
        st = st3215.InferenceStateReader(memoryview(bytes(e.Data)))
        for bs in st.get_buses() or []:
            bi = bs.get_bus()
            if not bi:
                continue
            motors = {}
            for m in bs.get_motors() or []:
                s = m.get_state()
                d = dict(parse_motor(s) if s else {}, rmin=m.get_range_min(), rmax=m.get_range_max(), frozen=m.get_range_freezed())
                # Saved (frozen) ranges are in Present-Position space: the servo already applies its EEPROM
                # offset. Station adds the offset only while collecting an unsaved range (port_meta.rs).
                # Verified 2026-11-21 after follower auto-cal (m2 at 1%, m4 at 95% of range = last auto-cal moves).
                if d.get("raw") is not None and d["frozen"]:
                    d["pos"] = d["raw"]
                motors[m.get_id()] = d
            buses[bi.get_serial_number()] = dict(port=bi.get_port_name(), motors=motors,
                                                lat_ms=(time.time_ns() - bs.get_system_stamp_ns()) / 1e6)
    return n, buses


def report_motors(n, buses):
    print("\n[3] ST3215 buses / motors  (queue st3215/inference)")
    if n == 0:
        print(f"   {BAD} no inference entries received -> Station sees no servo bus (joints=0)")
        return
    if not buses:
        print(f"   {BAD} {n} entries but 0 buses")
        return
    for sn, b in buses.items():
        ms = b["motors"]
        flag = OK if len(ms) == 8 else WARN
        print(f"   {flag} bus SN={sn!r} port={b['port']}  motors={len(ms)} (expect 8 for ElRobot 7+1)  latency={b['lat_ms']:.1f}ms")
        for mid in sorted(ms):
            m = ms[mid]
            span = (m["rmax"] - m["rmin"]) % 4096
            if not m["frozen"]:
                cal = "NOT SAVED (range = only what was seen so far)"
            elif span < 300:
                cal = f"SAVED but span only {span} steps -> recalibrate"
            else:
                cal = "calibrated (saved)"
            inside = m["rmin"] <= m.get("pos", 0) <= m["rmax"] if m["rmin"] <= m["rmax"] else True
            if m["frozen"] and not inside:
                cal += "  ! pos OUTSIDE range"
            err = f"  ERROR status=0x{m.get('status', 0):02x}" if m.get('status', 0) & 0xFE else ""
            print(f"       motor {mid:2d}: pos={m.get('pos', 0):4d} tq={m.get('torque', 0)} cur={m.get('cur', 0):4d}mA "
                  f"temp={m.get('temp', 0):3d}C volt={m.get('volt', 0):4.1f}V range=[{m['rmin']},{m['rmax']}] {cal}{err}")
    if len(buses) == 2:
        print(f"   {OK} two buses -> label LEADER/FOLLOWER by unplugging one; use SN as target_bus_serial")
    else:
        print(f"   {WARN} {len(buses)} bus(es); expected 2 (leader + follower)")


# ---------------------------------------------------------------- 4. cameras
def video_queue_ids():
    ids = set()
    for d in glob.glob(str(Path.home() / "Downloads/station_data/*/usbvideo/*")) + glob.glob("./station_data/*/usbvideo/*"):
        if os.path.isdir(d):
            ids.add(os.path.basename(d))
    return sorted(ids)


async def check_cameras(client, seconds, outdir):
    print("\n[4] Cameras (queues usbvideo/<id>)")
    ids = video_queue_ids()
    if not ids:
        print(f"   {BAD} no usbvideo queues found in station_data")
        return
    outdir.mkdir(parents=True, exist_ok=True)

    async def one(qid):
        q = asyncio.Queue()
        client.follow(f"usbvideo/{qid}", q)
        frames, info, fmt, saved, sizes, t0 = 0, None, None, None, set(), time.time()
        while time.time() - t0 < seconds:
            try:
                e = await asyncio.wait_for(q.get(), timeout=0.5)
            except asyncio.TimeoutError:
                continue
            if e is None:
                break
            env = usbvideo.RxEnvelopeReader(memoryview(bytes(e.Data)))
            cam = env.get_camera()
            if cam and not info:
                info = f"{cam.get_manufacturer()} {cam.get_product()} (VID {cam.get_vendor_id():#06x} PID {cam.get_product_id():#06x})"
            if env.get_type() == usbvideo.RxEnvelopeType.ET_FRAMES:
                fp = env.get_frames()
                f = fp.get_format()
                fmt = (f.get_width(), f.get_height(), frame.FrameFormatKind(f.get_kind()).name)
                data = fp.get_frames_data() or []
                frames += len(data)
                if data:
                    sizes.add(len(data[-1]))
                if data and saved is None and f.get_kind() == frame.FrameFormatKind.FF_JPEG:
                    saved = outdir / f"cam_{qid[:8]}.jpg"
                    saved.write_bytes(bytes(data[-1]))
        return qid, frames, info, fmt, saved, sizes

    for qid, frames, info, fmt, saved, sizes in await asyncio.gather(*(one(i) for i in ids)):
        if frames:
            print(f"   {OK} {qid[:8]}  {frames / seconds:5.1f} fps  {fmt}  {info or ''}")
            if saved:
                print(f"       snapshot -> {saved}")
            if sizes and max(sizes) < 4000:
                print(f"   {WARN} JPEGs only {max(sizes)} B and constant -> frames are probably BLACK "
                      "(lens cap? low light? check snapshot / web UI)")
        else:
            print(f"   {WARN} {qid[:8]}  no live frames (stale queue from an unplugged camera?)")


# ---------------------------------------------------------------- watch / move
async def watch(client):
    print("Live positions (Ctrl+C to stop). Torque off, move each joint by hand.\n")
    while True:
        _, buses = await collect_motors(client, 0.3)
        line = []
        for sn, b in buses.items():
            line.append(f"{sn[-6:]}: " + " ".join(f"{i}:{m.get('pos', 0):4d}" for i, m in sorted(b["motors"].items())))
        print("\r" + "  |  ".join(line or ["no buses"]) + "     ", end="", flush=True)


def write_cmd(bus, motor, value):
    c = st3215.Command(target_bus_serial=bus, write=st3215.ST3215WriteCommand(
        motor_id=motor, address=GOAL_POS, value=int(value).to_bytes(2, "little")))
    return commands.DriverCommand(type=drivers.StationCommandType.STC_ST3215_COMMAND, body=c.encode())


async def move_test(client, bus, motor, delta):
    _, buses = await collect_motors(client, 2)
    if bus not in buses or motor not in buses[bus]["motors"]:
        print(f"{BAD} bus {bus!r} / motor {motor} not found. Buses: {list(buses)}")
        return
    m = buses[bus]["motors"][motor]
    if not (m["rmin"] < m["rmax"]):
        print(f"{BAD} motor {motor} not calibrated (range [{m['rmin']},{m['rmax']}]) - calibrate in web UI first")
        return
    start = m["pos"]
    target = max(m["rmin"], min(m["rmax"], start + delta))
    print(f"motor {motor}: {start} -> {target} -> {start}")
    for goal in (target, start):
        await send_commands(client, [write_cmd(bus, motor, goal)])
        await asyncio.sleep(1.5)
        _, b2 = await collect_motors(client, 0.5)
        now = b2.get(bus, {}).get("motors", {}).get(motor, {}).get("pos")
        print(f"   goal={goal} now={now} err={None if now is None else now - goal}")


# ---------------------------------------------------------------- main
async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--seconds", type=float, default=4)
    ap.add_argument("--watch", action="store_true")
    ap.add_argument("--move-test", action="store_true")
    ap.add_argument("--bus")
    ap.add_argument("--motor", type=int, default=1)
    ap.add_argument("--delta", type=int, default=60, help="steps (~0.088 deg/step); keep small")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent.parent / "snapshots"))
    a = ap.parse_args()

    if not (a.watch or a.move_test):
        check_usb()
    print("\n[2] Station TCP")
    try:
        client = await new_station_client(a.host, log)
        print(f"   {OK} connected to {a.host}:8888")
    except Exception as e:
        print(f"   {BAD} cannot connect ({e}) -> is `./station --web --tcp` running?")
        return

    if a.watch:
        return await watch(client)
    if a.move_test:
        if not a.bus:
            print("--move-test needs --bus <serial> (run without flags first to see serials)")
            return
        if abs(a.delta) > 200:
            print("refusing |delta| > 200 steps (~18 deg) for a smoke test")
            return
        return await move_test(client, a.bus, a.motor, a.delta)

    (n, buses), _ = await asyncio.gather(collect_motors(client, a.seconds),
                                         check_cameras(client, a.seconds, Path(a.out)))
    report_motors(n, buses)
    print()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print()
