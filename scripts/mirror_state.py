"""Read-only: print who mirrors whom right now (Station inference/mirroring) + torque and error status per bus."""
import asyncio, logging, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from normakit import station_backend as sb
from station_py import new_station_client
from target.gen_python.protobuf.drivers.motors_mirroring import mirroring as mm
MODE = {0: "UNKNOWN", 1: "LEADER", 2: "FOLLOWER"}
async def main():
    c = await new_station_client("localhost:8888", logging.getLogger("m"))
    q = asyncio.Queue(); c.follow("inference/mirroring", q)
    e = await asyncio.wait_for(q.get(), 10)
    rx = mm.RxEnvelopeReader(memoryview(bytes(e.Data)))
    st = rx.get_state()
    print("modes:", [(b.get_id().get_unique_id(), MODE.get(b.get_mode(), b.get_mode())) for b in st.get_modes()])
    for m in st.get_mirroring():
        print("MIRROR", m.get_source().get_id().get_unique_id(), "->", [t.get_id().get_unique_id() for t in m.get_targets()])
    if not list(st.get_mirroring()): print("no active mirroring")
    sys.stdout.flush(); import os; os._exit(0)
asyncio.run(main())
