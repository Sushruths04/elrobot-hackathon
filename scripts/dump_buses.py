"""Read-only: print every bus/motor Station sees (range, freeze flag, position, torque, voltage, temp, current)."""
import asyncio, logging, struct, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from normakit import station_backend as sb   # sets up sys.path for vendor protobufs
from station_py import new_station_client
from target.gen_python.protobuf.drivers.st3215 import st3215 as pb

async def main(host):
    c = await new_station_client(host, logging.getLogger("dump"))
    q = asyncio.Queue(); c.follow("st3215/inference", q)
    entry = None
    while entry is None:
        entry = await asyncio.wait_for(q.get(), 10)
    st = pb.InferenceStateReader(memoryview(bytes(entry.Data)))
    for bus in st.get_buses():
        print("BUS", bus.get_bus().get_serial_number(), "port", getattr(bus.get_bus(), "get_port_name", lambda: "?")())
        for m in sorted(bus.get_motors(), key=lambda m: m.get_id()):
            s = bytes(m.get_state())
            pos = sb._pos(s) if len(s) > 0x3A else -1
            u16 = lambda a: struct.unpack("<H", s[a:a+2])[0]
            print(f"  motor {m.get_id()}: range=[{m.get_range_min()},{m.get_range_max()}] frozen={m.get_range_freezed()} "
                  f"pos={pos} torque={s[0x28]} status=0x{s[0x41]:02x} load={u16(0x3C)} volt={s[0x3E]/10:.1f}V temp={s[0x3F]}C cur={u16(0x45)}")
    sys.stdout.flush(); import os; os._exit(0)
asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else "localhost:8888"))
