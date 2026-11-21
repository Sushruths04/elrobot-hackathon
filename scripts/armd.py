#!/usr/bin/env python
"""armd - resident arm daemon (the FAST path). One warm process: backend connected, both kinematics models loaded, IK warm (<1 ms).
Every call is a ~25 ms HTTP request; trajectories run ON THE SERVOS (per-joint speed proportional to distance so all joints arrive together),
so there is no host-side ramp loop. Moves require the human's ARM (UI button writes run/arm_until). Local only: http://127.0.0.1:8891
  GET  /state     joints, jaw tip, armed seconds           POST /move_joints {"ticks":{"1":2034,...},"speed":900,"wait":true}
  POST /move_tip  {"x":0.0,"y":0.27,"z":0.10,"roll":0,"speed":900}   POST /gripper {"to":"open"|"close"|ticks}   POST /home   POST /stop   GET /stats
  POST /record {"on":true,"name":"pick1"} ... {"on":false}   POST /play {"name":"pick1","scale":1.0}   GET /demos   (teleop demo -> replay on the follower)"""
import json, math, os, sys, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src")); os.chdir(ROOT)
from normakit.core import norm_to_rad, rad_to_norm
from normakit.kin import Kin
from normakit.station_backend import StationBackend, REG_GOAL_POS, REG_GOAL_SPEED, REG_ACC
J = list(range(1, 9)); wd = lambda a, b: ((a - b + 2048) % 4096) - 2048
SPEED_CAP = {1: 1200, 2: 700, 3: 700, 4: 800, 5: 1500, 6: 1500, 7: 1500, 8: 800}      # ticks/s per joint (4096 = one turn); loaded joints capped lower
ZMIN = 0.03; OPEN, CLOSED = 2650, 1321   # OPEN 2650 (norm 0.87): wide gap so a +-1 cm position error still straddles a 1.5 cm pen (2183 was the narrow demo value)
LOCK = threading.Lock(); STATS = {"calls": 0, "last_ms": {}}
be = StationBackend("localhost:8888"); be.start(); time.sleep(1.0); kin = Kin(); kin._ensure_mj()
_warm = {i: norm_to_rad(i, be.state().joints[i].norm) for i in range(1, 8)}; kin.ik_jaw_tip((0.0, 0.27, 0.10), roll_rad=0.0, seed_rad=_warm)   # warm-up
def armed(): 
    try: return max(0.0, float((ROOT / "run/arm_until").read_text()) - time.time())
    except Exception: return 0.0
def S(): return be.state()
def to_norm(st, i, t):
    j = st.joints[i]; span = (j.hi - j.lo) % 4096; rel = (t - j.lo) % 4096
    if rel > span + (4096 - span) / 2: rel -= 4096
    return rel / span
def to_ticks(st, i, n): j = st.joints[i]; return int(round(j.lo + n * ((j.hi - j.lo) % 4096))) % 4096
def q_of(st, ticks=None): return {i: norm_to_rad(i, to_norm(st, i, (ticks or {}).get(i, st.joints[i].raw))) for i in range(1, 8)}
def hold(): st = S(); be._sync(REG_GOAL_POS, {i: (st.joints[i].raw % 4096).to_bytes(2, "little") for i in J if i in st.joints})
def drive(target, speed, wait=True, tol=30, timeout=10.0):
    """target: {joint: ticks}. Servo-side trajectory: speed_i = speed * |d_i| / max|d|; one write each for acc, speeds, goals."""
    st = S(); d = {i: wd(t, st.joints[i].raw) for i, t in target.items()}; dmax = max(1, max(abs(v) for v in d.values())); t0 = time.time()
    sp = {i: int(max(40, min(SPEED_CAP[i], speed * abs(d[i]) / dmax))) for i in target}
    be._sync(REG_ACC, {i: bytes([60]) for i in target}); be._sync(REG_GOAL_SPEED, {i: v.to_bytes(2, "little") for i, v in sp.items()})
    be._sync(REG_GOAL_POS, {i: (int(t) % 4096).to_bytes(2, "little") for i, t in target.items()}); tw = time.time() - t0
    if not wait: return {"write_ms": round(tw * 1000), "speeds": sp, "worst_ticks_to_go": max(abs(v) for v in d.values())}
    last = {}; steady = 0
    while time.time() - t0 < timeout:
        st = S()
        if max(st.joints[i].current_ma for i in range(1, 8)) > 700: hold(); return {"error": "HIGH CURRENT - held", "write_ms": round(tw * 1000)}
        err = max(abs(wd(target[i], st.joints[i].raw)) for i in target); pos = {i: st.joints[i].raw for i in target}
        steady = steady + 1 if pos == last else 0; last = pos
        if err <= tol or (err <= 4 * tol and steady > 25): return {"ok": True, "arrived_in_ms": round((time.time() - t0) * 1000), "write_ms": round(tw * 1000), "worst_ticks_off": err}
        time.sleep(0.01)
    hold(); st = S(); return {"error": "timeout - held", "per_joint_err": {i: wd(target[i], st.joints[i].raw) for i in target}}
def path_min_z(st, target):
    low = 9
    for s in np.linspace(0, 1, 30): low = min(low, float(kin.jaw_tip_pose(q_of(st, {i: (st.joints[i].raw + s * wd(target.get(i, st.joints[i].raw), st.joints[i].raw)) % 4096 for i in range(1, 8)}))[0][2]))
    return low

DEMOS = ROOT / "run/demos"; DEMOS.mkdir(parents=True, exist_ok=True); REC = {"on": False, "name": None, "samples": [], "t0": 0.0}
def _rec_loop():
    while REC["on"]:
        t = time.time(); st = S(); REC["samples"].append((round(t - REC["t0"], 3), {i: st.joints[i].raw for i in J if i in st.joints})); time.sleep(max(0, 0.02 - (time.time() - t)))
def record(on, name):
    if on:
        if REC["on"]: return {"error": "already recording " + REC["name"]}
        REC.update(on=True, name=name, samples=[], t0=time.time()); threading.Thread(target=_rec_loop, daemon=True).start(); return {"recording": name}
    REC["on"] = False; time.sleep(0.1); smp = REC["samples"]; secs = smp[-1][0] if smp else 0; (DEMOS / f"{REC['name']}.json").write_text(json.dumps(smp))
    chg = sum(1 for a, b in zip(smp, smp[1:]) if a[1] != b[1]); return {"saved": f"run/demos/{REC['name']}.json", "samples": len(smp), "seconds": round(secs, 1), "state_changes": chg, "distinct_per_s": round(chg / max(secs, 1e-3), 1)}
def play(name, scale=1.0, low_ok=False):
    f = DEMOS / f"{name}.json"
    if not f.exists(): return {"error": f"no demo {name} (run/demos/)"}
    smp = json.loads(f.read_text()); T = np.array([x[0] for x in smp]); P = {i: np.array([x[1][str(i)] for x in smp], float) for i in J}; st = S()
    if not low_ok:
        low = min(float(kin.jaw_tip_pose(q_of(st, {i: int(P[i][k]) for i in range(1, 8)}))[0][2]) for k in range(0, len(smp), max(1, len(smp) // 60)))
        if low < 0.015: return {"error": f"demo takes the jaw tip to z={low:.3f}: pass low_ok only if the demo really touches the table"}
    start = {i: int(P[i][0]) for i in J}
    if all(i in start for i in range(1, 8)) and path_min_z(st, {i: start[i] for i in range(1, 8)}) < 0.06:      # straight joint-space transit would dip: go via home first
        hh = json.loads((ROOT / "run/home_follower.json").read_text())["raw"]; th = {int(k): int(v) for k, v in hh.items()}
        for grp, sp in (((1, 5, 6, 7), 500), ((2, 3, 4), 400)):
            rr = drive({i: th[i] for i in grp}, sp, True, tol=45, timeout=25)
            if "error" in rr: return {"error": "transit via home failed", "detail": rr}
    r0 = drive(start, 500, True, tol=40, timeout=15)
    if "error" in r0: return {"error": "could not reach the demo start pose", "detail": r0}
    be._sync(REG_ACC, {i: bytes([0]) for i in J}); be._sync(REG_GOAL_SPEED, {i: (1500).to_bytes(2, "little") for i in J}); t0 = time.time(); dur = T[-1] / scale; worst = 0
    while True:
        e = time.time() - t0
        if e >= dur: break
        tt = e * scale; k = max(0, min(len(T) - 2, int(np.searchsorted(T, tt) - 1))); f_ = (tt - T[k]) / max(T[k + 1] - T[k], 1e-6); goal = {}
        for i in J: a, b = P[i][k], P[i][k + 1]; goal[i] = int(round(a + max(0.0, min(1.0, f_)) * (((b - a + 2048) % 4096) - 2048))) % 4096
        be._sync(REG_GOAL_POS, goal); st = S()
        if max(st.joints[i].current_ma for i in range(1, 8)) > 700: hold(); return {"error": "HIGH CURRENT - held", "at_s": round(e, 2)}
        worst = max(worst, max(abs(wd(goal[i], st.joints[i].raw)) for i in J))
    be._sync(REG_GOAL_POS, {i: int(P[i][-1]) % 4096 for i in J}); time.sleep(0.4)
    return {"ok": True, "played_s": round(dur, 1), "worst_tracking_lag_ticks": int(worst), "final_err": max(abs(wd(int(P[i][-1]), S().joints[i].raw)) for i in J)}

def mirror(on):
    """Station leader->follower mirroring ON (teleop) / OFF (autonomous). While it is ON it rewrites the follower's goal+speed ~47x per second and fights every command."""
    import asyncio
    from station_py import send_commands
    from target.gen_python.protobuf.station import commands, drivers
    from target.gen_python.protobuf.drivers.motors_mirroring import mirroring as mm
    buses = json.loads((ROOT / "run/buses.json").read_text()); L = mm.MirroringBus(type=mm.BusType.MBT_ST3215, unique_id=buses["leader"]); F = mm.MirroringBus(type=mm.BusType.MBT_ST3215, unique_id=buses["follower"])
    cmd = mm.Command(type=mm.CommandType.CT_START_MIRROR if on else mm.CommandType.CT_STOP_MIRROR, source=L, targets=[F] if on else None)
    dc = commands.DriverCommand(type=drivers.StationCommandType.STC_MOTOR_MIRRORING_COMMAND, body=cmd.encode())
    asyncio.run_coroutine_threadsafe(send_commands(be._client, [dc]), be._loop).result(5)
    if not on: hold()
    return {"mirroring": "ON (leader drives follower)" if on else "OFF (follower is free for commands)"}
def api(path, b):
    t_all = time.time(); STATS["calls"] += 1
    if path == "/state":
        st = S(); tip = kin.jaw_tip_pose(q_of(st))[0]
        return {"joints": {i: {"raw": st.joints[i].raw, "mA": st.joints[i].current_ma} for i in sorted(st.joints)}, "tip": [round(float(x), 4) for x in tip], "armed_s": int(armed())}
    if path == "/stats": return STATS
    if path == "/ik":                                               # dry IK for several wrist rolls: position error mm, orientation error deg (no motion, no ARM needed)
        st = S(); out = {}
        for rl in b.get("rolls", [0.0]):
            q, pe, oe, tl = kin.ik_jaw_tip((b["x"], b["y"], b["z"]), roll_rad=float(rl), tilt_deg=b.get("tilt"), seed_rad=q_of(st)); out[str(round(float(rl), 3))] = {"pos_err_mm": round(pe * 1000, 1), "ori_err_deg": round(oe, 1), "tilt": round(tl), "j7_norm": round(rad_to_norm(7, q[7]), 2)}
        return {"ik": out}
    if path == "/record": return record(bool(b.get("on", True)), b.get("name", "demo"))
    if path == "/demos": return {"demos": sorted(p.stem for p in DEMOS.glob("*.json") if p.stem != "episodes")}
    if path == "/mirror": return mirror(bool(b.get("on", False)))
    if path == "/stop": hold(); return {"ok": "holding present pose"}
    if armed() <= 0: return {"error": "NOT ARMED: the human presses 'ARM motion' on the web page (http://127.0.0.1:8890)"}
    with LOCK:
        speed = float(b.get("speed", 600))
        if path == "/move_joints":
            tg = {int(k): int(v) for k, v in b["ticks"].items()}; st = S(); low = path_min_z(st, tg) if all(i in tg for i in range(1, 8)) else None
            if low is not None and low < ZMIN and not b.get("low_ok"): return {"error": f"path would take the jaw tip to z={low:.3f} < {ZMIN}"}
            return drive(tg, speed, b.get("wait", True), timeout=float(b.get("timeout", 10)))
        if path == "/move_tip":
            st = S(); t0 = time.time(); q, pe, oe, tilt = kin.ik_jaw_tip((b["x"], b["y"], b["z"]), roll_rad=float(b.get("roll", 0.0)), tilt_deg=b.get("tilt"), seed_rad=q_of(st)); tik = time.time() - t0
            if pe > 0.02: return {"error": f"IK cannot reach (err {pe*100:.1f} cm)", "ik_ms": round(tik * 1000)}
            tg = {i: to_ticks(st, i, rad_to_norm(i, q[i])) for i in range(1, 8)}; low = path_min_z(st, tg)
            if low < min(ZMIN, b["z"] - 0.005) and not b.get("low_ok"): return {"error": f"path would dip to z={low:.3f}"}
            r = drive(tg, speed, b.get("wait", True), timeout=float(b.get("timeout", 10))); r.update({"ik_ms": round(tik * 1000), "ik_err_mm": round(pe * 1000, 1), "ori_err_deg": round(oe, 1), "tilt": round(tilt)}); return r
        if path == "/gripper":
            to = b.get("to", "open"); tgt = OPEN if to == "open" else CLOSED if to == "close" else int(to)
            if to == "close":                                              # an object stops the jaws before the closed value: success = jaws steady (not moving) for 0.5 s
                be._sync(0x30, {8: (850).to_bytes(2, "little")})           # firmer grip (was 500): a thin pen slips otherwise
                be._sync(REG_ACC, {8: bytes([40])}); be._sync(REG_GOAL_SPEED, {8: (400).to_bytes(2, "little")}); be._sync(REG_GOAL_POS, {8: tgt.to_bytes(2, "little")}); t0 = time.time(); last = None; steady_since = None
                while time.time() - t0 < 6:
                    v = S().joints[8].raw
                    if last is not None and abs(v - last) <= 2: steady_since = steady_since or time.time()
                    else: steady_since = None
                    if steady_since and time.time() - steady_since > 0.5: break
                    last = v; time.sleep(0.03)
                g = S().joints[8]; return {"ok": True, "gripper_ticks": g.raw, "mA": g.current_ma, "closed_on_object": bool(g.raw > CLOSED + 80 or (g.raw > CLOSED + 40 and g.current_ma > 15)), "closed_in_ms": round((time.time() - t0) * 1000)}
            r = drive({8: tgt}, 500, True, tol=25, timeout=4.0); r["gripper_ticks"] = S().joints[8].raw; return r
        if path == "/play": return play(b["name"], float(b.get("scale", 1.0)), bool(b.get("low_ok")))
        if path == "/home":
            h = json.loads((ROOT / "run/home_follower.json").read_text())["raw"]; tg = {int(k): int(v) for k, v in h.items()}
            r1 = drive({i: tg[i] for i in (1, 5, 6, 7)}, speed, True); r2 = drive({i: tg[i] for i in (2, 3, 4)}, min(speed, 500), True); return {"wrist_first": r1, "arm": r2}
    return {"error": "unknown endpoint"}
class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def reply(self, code, obj): b = json.dumps(obj).encode(); self.send_response(code); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
    def go(self, b):
        t = time.time()
        try: r = api(self.path.split("?")[0], b)
        except Exception as e: r = {"error": f"{type(e).__name__}: {e}"[:200]}
        r["total_ms"] = round((time.time() - t) * 1000, 1); STATS["last_ms"][self.path] = r["total_ms"]; self.reply(200, r)
    def do_GET(self): self.go({})
    def do_POST(self): n = int(self.headers.get("Content-Length", 0) or 0); self.go(json.loads(self.rfile.read(n) or b"{}"))
if __name__ == "__main__":
    print("armd ready: http://127.0.0.1:8891  (state, move_joints, move_tip, gripper, home, stop, stats)", flush=True); ThreadingHTTPServer(("127.0.0.1", 8891), H).serve_forever()
