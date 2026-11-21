"""Read / set UVC camera controls (exposure, brightness, gain...) straight over usbfs - no extra packages.
The cameras are KD-USB (2c7f:4a60): Camera Terminal id 1, Processing Unit id 2, VideoControl interface 0.
    .venv/bin/python scripts/uvc_ctl.py get                      # read current values of every camera
    .venv/bin/python scripts/uvc_ctl.py set ae_mode 8            # 8 = auto exposure (aperture priority), 1 = manual
    .venv/bin/python scripts/uvc_ctl.py set exposure 150         # exposure time, units of 100 us (manual mode)
    .venv/bin/python scripts/uvc_ctl.py set brightness -20       # signed
Settings are volatile: replugging the camera restores its defaults."""
import ctypes, fcntl, glob, os, struct, subprocess, sys
CT, PU, IFACE = 1, 2, 0
#           name: (entity, selector, size, signed)
CTRL = {"ae_mode": (CT, 0x02, 1, False), "ae_priority": (CT, 0x03, 1, False), "exposure": (CT, 0x04, 4, False),
        "backlight": (PU, 0x01, 2, False), "brightness": (PU, 0x02, 2, True), "contrast": (PU, 0x03, 2, False), "gain": (PU, 0x04, 2, False),
        "powerline": (PU, 0x05, 1, False), "hue": (PU, 0x06, 2, True), "saturation": (PU, 0x07, 2, False), "sharpness": (PU, 0x08, 2, False),
        "gamma": (PU, 0x09, 2, False), "wb_temp": (PU, 0x0A, 2, False), "wb_auto": (PU, 0x0B, 1, False)}
GET_CUR, GET_MIN, GET_MAX, GET_DEF, SET_CUR = 0x81, 0x82, 0x83, 0x87, 0x01
class Ctrl(ctypes.Structure):
    _fields_ = [("bRequestType", ctypes.c_uint8), ("bRequest", ctypes.c_uint8), ("wValue", ctypes.c_uint16), ("wIndex", ctypes.c_uint16),
                ("wLength", ctypes.c_uint16), ("timeout", ctypes.c_uint32), ("data", ctypes.c_void_p)]
USBDEVFS_CONTROL = 0xC0185500
def devices():
    out = subprocess.run(["lsusb"], capture_output=True, text=True).stdout
    return [f"/dev/bus/usb/{l.split()[1]}/{l.split()[3][:3]}" for l in out.splitlines() if "2c7f:4a60" in l]
def xfer(path, rtype, req, entity, sel, data=b"", length=0):
    fd = os.open(path, os.O_RDWR)
    try:
        buf = ctypes.create_string_buffer(data if rtype & 0x80 == 0 else bytes(length), max(length, len(data)))
        c = Ctrl(rtype, req, sel << 8, (entity << 8) | IFACE, max(length, len(data)), 1000, ctypes.cast(buf, ctypes.c_void_p).value)
        n = fcntl.ioctl(fd, USBDEVFS_CONTROL, c)
        return bytes(buf.raw[:n if isinstance(n, int) and n >= 0 else 0])
    finally: os.close(fd)
def get(path, name, req=GET_CUR):
    e, s, n, sg = CTRL[name]
    try: return int.from_bytes(xfer(path, 0xA1, req, e, s, length=n), "little", signed=sg)
    except OSError as ex: return f"ERR {ex.errno}"
def setv(path, name, v):
    e, s, n, sg = CTRL[name]; xfer(path, 0x21, SET_CUR, e, s, int(v).to_bytes(n, "little", signed=sg))
if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "get"
    for p in devices():
        if cmd == "get":
            print(p); [print(f"  {k:12s} cur={get(p,k)!s:>7}  min={get(p,k,GET_MIN)!s:>7}  max={get(p,k,GET_MAX)!s:>7}  default={get(p,k,GET_DEF)!s:>7}") for k in CTRL]
        elif cmd == "set":
            try: setv(p, sys.argv[2], sys.argv[3]); print(p, sys.argv[2], "=", sys.argv[3], "-> now", get(p, sys.argv[2]))
            except OSError as ex: print(p, "set failed:", ex)
