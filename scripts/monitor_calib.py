"""Read-only monitor: prints calibration progress + temp/voltage/current/range for each bus. Ctrl+C to stop."""
import asyncio, logging, struct, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from normakit import station_backend as sb
from station_py import new_station_client
from target.gen_python.protobuf.drivers.st3215 import st3215 as pb

STATUS = {0: "IDLE", 1: "IN_PROGRESS", 2: "DONE", 3: "FAILED", 4: "STOPPED"}
async def main(host, every=3.0):
    c = await new_station_client(host, logging.getLogger("mon"))
    q = asyncio.Queue(); c.follow("st3215/inference", q)
    last = 0
    while True:
        entry = await q.get()
        if entry is None or time.time() - last < every: continue
        last = time.time()
        st = pb.InferenceStateReader(memoryview(bytes(entry.Data)))
        for bus in st.get_buses():
            ac = bus.get_auto_calibration()
            ms = sorted(bus.get_motors(), key=lambda m: m.get_id())
            info = []
            for m in ms:
                s = bytes(m.get_state())
                if len(s) < 0x47:
                    info.append(f"{m.get_id()}:no-data"); continue
                span = (m.get_range_max() - m.get_range_min()) % 4096
                info.append(f"{m.get_id()}:span{span} {s[0x3F]}C {struct.unpack('<H', s[0x45:0x47])[0]}mA")
            warn = " HOT!" if any(len(bytes(m.get_state())) > 0x3F and bytes(m.get_state())[0x3F] >= 60 for m in ms) else ""
            print(f"{time.strftime('%H:%M:%S')} {bus.get_bus().get_serial_number()} "
                  f"{STATUS.get(ac.get_status(), ac.get_status())} {ac.get_current_step()}/{ac.get_total_steps()} "
                  f"'{ac.get_current_phase()}' {ac.get_error_message()}{warn} | " + " ".join(info), flush=True)
asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else "localhost:8888"))
