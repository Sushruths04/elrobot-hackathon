"""Real-arm backend: talks to a running NormaCore Station over its TCP/NormFS port (8888).

UNTESTED against live hardware (written from the Station source, v0.1.0-beta.10).
First thing on Saturday: `python scripts/smoke_station.py` and fix any field-name drift.

Station facts used (all from the repo):
  * Port 8888 = NormFS over TCP (u64-LE length-prefixed protobuf). `station --web --tcp`.
  * Motor state:   follow queue `st3215/inference` -> InferenceState (full register dump per motor).
  * Camera frames: queue ids appear in `inference-states` entries with type QDT_USB_VIDEO_FRAMES;
                   frames are usbvideo.RxEnvelope{frames: FramesPack(JPEG or NCHW)}.
                   station.yaml `usb-video.resize_target: 0` => full-res (default 224px is too small for an LLM).
  * Commands:      enqueue StationCommandsPack on queue `commands`; ST3215 Command.sync_write
                   (address 0x2A goal position, 0x2E goal speed, 0x29 acceleration, 0x28 torque enable).
"""
from __future__ import annotations

import asyncio
import io
import logging
import os
import struct
import sys
import threading
import time
from pathlib import Path

from .core import ArmState, Backend, JointState, JOINT_IDS

ROOT = Path(__file__).resolve().parents[2] / "vendor/norma-core-src"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "software/station/shared"))

REG_TORQUE, REG_ACC, REG_GOAL_POS, REG_GOAL_SPEED = 0x28, 0x29, 0x2A, 0x2E
REG_PRES_POS, REG_TEMP, REG_CUR = 0x38, 0x3F, 0x45
QDT_USB_VIDEO_FRAMES = 21
MIN_CALIBRATED_SPAN = 100   # same threshold the Station web UI uses
log = logging.getLogger("normakit.station")


def _bus_from_file(role: str):
    """run/buses.json: {"follower": "<serial>", "leader": "<serial>"} (identified by supply voltage)."""
    import json
    try:
        return json.loads((Path(__file__).resolve().parents[2] / "run/buses.json").read_text()).get(role)
    except Exception:
        return None


def _rel(pos: int, lo: int, span: int) -> float:
    """Position as a fraction of the calibrated arc; slightly outside the arc gives <0 or >1 (not a huge wrapped value)."""
    rel = (pos - lo) % 4096
    if rel > span + (4096 - span) / 2:
        rel -= 4096
    return rel / span


def _pos(b: bytes) -> int:
    raw = struct.unpack("<H", b[REG_PRES_POS:REG_PRES_POS + 2])[0]
    if raw & 0x8000:
        return (4096 - (raw & 4095)) & 4095
    return raw & 4095


class StationBackend(Backend):
    name = "station"

    def __init__(self, host="localhost:8888", bus_serial: str | None = None, enable_torque=False, require_motors=True):
        self.host = host
        self.bus_serial = bus_serial or os.environ.get("NORMA_BUS_SERIAL") or _bus_from_file("follower")
        self.enable_torque = enable_torque
        self.require_motors = require_motors
        self._state: dict[int, JointState] = {}
        self._bus: str | None = None
        self._frames: dict[str, bytes] = {}
        self._cam_order: list[str] = []
        self._loop = asyncio.new_event_loop()
        self._client = None
        self._ready = threading.Event()
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ lifecycle
    def start(self):
        threading.Thread(target=self._run, daemon=True).start()
        if not self._ready.wait(15):
            raise RuntimeError("Station not reachable on %s (is `station --web --tcp` running?)" % self.host)
        t0 = time.time()
        while self.require_motors and time.time() - t0 < 10 and len(self._state) < 8:
            time.sleep(0.1)
        if self.require_motors and len(self._state) < 8:
            raise RuntimeError(f"Only saw motors {sorted(self._state)}; need 1..8 on the follower bus. "
                               "Set NORMA_BUS_SERIAL to pick the right bus.")
        if self.enable_torque:
            self._sync(REG_TORQUE, {i: b"\x01" for i in JOINT_IDS})

    def stop_backend(self):
        self._loop.call_soon_threadsafe(self._loop.stop)

    def _run(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._main())

    async def _main(self):
        from station_py import new_station_client
        self._client = await new_station_client(self.host, log)
        self._ready.set()
        asyncio.create_task(self._follow_motors())
        asyncio.create_task(self._follow_inference_states())
        await asyncio.Event().wait()

    # ------------------------------------------------------------------ readers
    async def _follow_motors(self):
        from target.gen_python.protobuf.drivers.st3215 import st3215 as pb
        q: asyncio.Queue = asyncio.Queue()
        self._client.follow("st3215/inference", q)
        while True:
            entry = await q.get()
            if entry is None:
                continue
            try:
                st = pb.InferenceStateReader(memoryview(bytes(entry.Data)))
                for bus in st.get_buses():
                    serial = bus.get_bus().get_serial_number()
                    motors = bus.get_motors()
                    ids = {m.get_id() for m in motors}
                    if self.bus_serial and serial != self.bus_serial:
                        continue
                    if not self.bus_serial and not set(JOINT_IDS) <= ids:
                        continue
                    self._bus = serial
                    with self._lock:
                        for m in motors:
                            raw_state = bytes(m.get_state())
                            if len(raw_state) < REG_CUR + 2 or m.get_id() not in JOINT_IDS:
                                continue
                            lo, hi = m.get_range_min(), m.get_range_max()
                            span = (hi - lo) % 4096          # Station ranges may wrap past 4095 -> 0
                            if span < MIN_CALIBRATED_SPAN:
                                continue  # not calibrated yet (live min/max only)
                            pos = _pos(raw_state)
                            goal = struct.unpack("<H", raw_state[REG_GOAL_POS:REG_GOAL_POS + 2])[0] & 4095
                            self._state[m.get_id()] = JointState(
                                id=m.get_id(), norm=_rel(pos, lo, span),
                                goal_norm=_rel(goal, lo, span),
                                current_ma=struct.unpack("<H", raw_state[REG_CUR:REG_CUR + 2])[0],
                                temp_c=raw_state[REG_TEMP], raw=pos, lo=lo, hi=hi)
            except Exception as e:  # keep streaming
                log.warning("motor parse: %s", e)

    async def _follow_inference_states(self):
        from target.gen_python.protobuf.station import inference as inf
        q: asyncio.Queue = asyncio.Queue()
        self._client.follow("inference-states", q)
        followed = set()
        while True:
            entry = await q.get()
            if entry is None:
                continue
            try:
                rx = inf.InferenceRxReader(memoryview(bytes(entry.Data)))
                for e in rx.get_entries():
                    if int(e.get_type()) == QDT_USB_VIDEO_FRAMES:
                        qid = e.get_queue()
                        qid = qid if "/" in qid else f"usbvideo/{qid}"
                        if qid not in followed:
                            followed.add(qid)
                            self._cam_order = sorted(followed)
                            asyncio.create_task(self._follow_camera(qid))
            except Exception as ex:
                log.warning("inference-states parse: %s", ex)

    async def _follow_camera(self, qid: str):
        from target.gen_python.protobuf.drivers.usbvideo import usbvideo as uv, frame as fr
        from PIL import Image
        q: asyncio.Queue = asyncio.Queue()
        self._client.follow(qid, q)
        while True:
            entry = await q.get()
            if entry is None:
                continue
            try:
                env = uv.RxEnvelopeReader(memoryview(bytes(entry.Data)))
                if int(env.get_type()) != 0:  # ET_FRAMES
                    continue
                fp = env.get_frames()
                fmt = fp.get_format()
                if int(fmt.get_kind()) == 1:  # JPEG
                    frames = fp.get_frames_data()
                    if frames:
                        self._frames[qid] = bytes(frames[-1])
                else:  # NCHW u8
                    w, h = fmt.get_width(), fmt.get_height()
                    lin = bytes(fp.get_linear_data())
                    n = len(lin) // (3 * w * h)
                    if n:
                        import numpy as np
                        arr = np.frombuffer(lin, np.uint8)[-3 * w * h:].reshape(3, h, w).transpose(1, 2, 0)
                        buf = io.BytesIO()
                        Image.fromarray(arr).save(buf, "JPEG", quality=88)
                        self._frames[qid] = buf.getvalue()
            except Exception as ex:
                log.warning("camera parse %s: %s", qid, ex)

    # ------------------------------------------------------------------ Backend API
    def state(self) -> ArmState:
        with self._lock:
            return ArmState(dict(self._state))

    def cameras(self):
        return [f"cam{i}" for i, _ in enumerate(self._cam_order)] + list(self._cam_order)

    def frame_jpeg(self, camera=None) -> bytes:
        order = self._cam_order
        if not order:
            raise RuntimeError("no camera queues discovered yet (is usb-video enabled in station.yaml?)")
        camera = camera or "cam0"
        qid = order[int(camera[3:])] if camera.startswith("cam") and camera[3:].isdigit() else camera
        if qid not in self._frames:
            raise RuntimeError(f"no frame yet for {qid}")
        return self._frames[qid]

    def _sync(self, address: int, values: dict[int, bytes]):
        from station_py import send_commands
        from target.gen_python.protobuf.station import commands, drivers
        from target.gen_python.protobuf.drivers.st3215 import st3215 as pb
        cmd = pb.Command(
            target_bus_serial=self._bus or "",
            sync_write=pb.ST3215SyncWriteCommand(
                address=address,
                motors=[pb.ST3215SyncWriteCommand_MotorWrite(motor_id=i, value=v) for i, v in values.items()]))
        dc = commands.DriverCommand(type=drivers.StationCommandType.STC_ST3215_COMMAND, body=cmd.encode())
        fut = asyncio.run_coroutine_threadsafe(send_commands(self._client, [dc]), self._loop)
        fut.result(5)

    def send_goals(self, goals_norm, speed=0.5):
        st = self.state()
        spd = int(150 + 1500 * speed)  # steps/s
        self._sync(REG_ACC, {i: bytes([30]) for i in goals_norm})
        self._sync(REG_GOAL_SPEED, {i: spd.to_bytes(2, "little") for i in goals_norm})
        vals = {}
        for i, g in goals_norm.items():
            j = st.joints[i]
            vals[i] = (int(round(j.lo + g * ((j.hi - j.lo) % 4096))) % 4096).to_bytes(2, "little")
        self._sync(REG_GOAL_POS, vals)

    def halt(self):
        st = self.state()
        self._sync(REG_GOAL_POS, {i: int(j.raw).to_bytes(2, "little") for i, j in st.joints.items()})
