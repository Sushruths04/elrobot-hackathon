"""Read-only snapshot of every ST3215 bus Station sees (no motion, no torque)."""
from __future__ import annotations
import asyncio, logging
from . import station_backend as sb          # sets sys.path for vendor protobufs
from station_py import new_station_client
from target.gen_python.protobuf.drivers.st3215 import st3215 as pb


def norm_of(pos: int, lo: int, hi: int) -> float:
    span = (hi - lo) % 4096
    if not span:
        return 0.5
    rel = (pos - lo) % 4096
    if rel > span + (4096 - span) / 2:
        rel -= 4096
    return rel / span


async def snapshot(host="localhost:8888", client=None, queue=None):
    """-> {serial: {motor_id: {pos, lo, hi, norm, torque, temp, cur}}}"""
    if client is None:
        client = await new_station_client(host, logging.getLogger("busread"))
    q = queue or asyncio.Queue()
    if queue is None:
        client.follow("st3215/inference", q)
    entry = None
    while entry is None:
        entry = await asyncio.wait_for(q.get(), 10)
    st = pb.InferenceStateReader(memoryview(bytes(entry.Data)))
    out = {}
    for bus in st.get_buses():
        d = {}
        for m in bus.get_motors():
            s = bytes(m.get_state())
            if len(s) < 0x47: continue                       # truncated register read (motor did not answer this cycle): skip it
            lo, hi = m.get_range_min(), m.get_range_max()
            pos = sb._pos(s)
            d[m.get_id()] = dict(pos=pos, lo=lo, hi=hi, norm=norm_of(pos, lo, hi), torque=s[0x28], temp=s[0x3F], frozen=bool(m.get_range_freezed()), status=s[0x41],
                                 cur=int.from_bytes(s[0x45:0x47], "little"))
        out[bus.get_bus().get_serial_number()] = d
    return out, client, q
