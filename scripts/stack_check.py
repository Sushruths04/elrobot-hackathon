#!/usr/bin/env python
"""Read-only health check of the whole stack (no motion): Station, buses, cameras, kinematics, ER-2, UI, MCP, sim. Writes run/stack_check.json.
    .venv/bin/python scripts/stack_check.py [--er2]"""
import asyncio, json, os, socket, subprocess, sys, time, urllib.request
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src")); os.chdir(ROOT)
rows = []
def rec(name, status, detail=""):
    rows.append({"check": name, "status": status, "detail": detail}); print(f"[{status:4}] {name}: {detail}", flush=True)
def port(p):
    s = socket.socket(); s.settimeout(1.5)
    try: s.connect(("127.0.0.1", p)); return True
    except Exception: return False
    finally: s.close()
rec("station process", "PASS" if subprocess.run(["pgrep", "-x", "station"], capture_output=True).returncode == 0 else "FAIL", "./norma start")
rec("station ports 8888/8889", "PASS" if port(8888) and port(8889) else "FAIL", f"tcp 8888={port(8888)} web 8889={port(8889)}")
buses = json.loads((ROOT / "run/buses.json").read_text())
try:
    from normakit.busread import snapshot
    snap, _, _ = asyncio.run(snapshot())
    for role, ser in buses.items():
        ms = snap.get(ser)
        if not ms: rec(f"bus {role} ({ser})", "FAIL", "not visible"); continue
        uncal = [i for i, m in ms.items() if (m["hi"] - m["lo"]) % 4096 < 100]; bad = [i for i, m in ms.items() if m.get("status", 0)]
        tq = sum(1 for m in ms.values() if m["torque"]); tmax = max(m["temp"] for m in ms.values())
        rec(f"bus {role} ({ser})", "PASS" if len(ms) == 8 and not uncal and not bad else "WARN", f"{len(ms)} motors, uncalibrated={uncal}, fault status={bad}, torque on {tq}/8, max temp {tmax} C")
except Exception as e: rec("bus snapshot", "FAIL", str(e)[:120])
be = None
try:
    from normakit.station_backend import StationBackend, front_camera
    be = StationBackend("localhost:8888"); be.start(); time.sleep(2.0); good = []
    for c in be.cameras():
        try: good.append((c, len(be.frame_jpeg(c))))
        except Exception: pass
    fc = front_camera(be); rec("cameras", "PASS" if good else "FAIL", f"{len(good)} streaming, front = {str(fc)[-6:]}")
except Exception as e: rec("backend/cameras", "FAIL", str(e)[:120])
try:
    import numpy as np
    from normakit.core import norm_to_rad
    from normakit.kin import Kin
    kin = Kin(); st = be.state(); q = {i: norm_to_rad(i, st.joints[i].norm) for i in range(1, 8)}
    pb = kin.fk(q); tip, R = kin.jaw_tip_pose(q); kin._ensure_mj(); ee = kin._mj_data.site_xpos[kin._mj_ee_sid].copy()
    rec("kinematics pybullet == MuJoCo (EE link)", "PASS" if np.linalg.norm(pb - ee) < 0.003 else "FAIL", f"diff {np.linalg.norm(pb-ee)*1000:.1f} mm; jaw tip {np.round(tip,3).tolist()}")
    errs = []
    for t in ((-0.02, 0.27, 0.05), (0.08, 0.28, 0.10), (0.0, 0.22, 0.12)):
        q2, pe, oe, _ = kin.ik_jaw_tip(t, roll_rad=0.0, seed_rad=q); errs.append(round(pe * 1000, 1))
    rec("IK round trip (jaw tip, 7-DoF)", "PASS" if max(errs) < 8 else "WARN", f"position error mm at 3 targets: {errs}")
except Exception as e: rec("kinematics", "FAIL", str(e)[:140])
mp = ROOT / "calibration/table_tip_front.json"
rec("table mapping pixel->robot xy", "PASS" if mp.exists() and json.loads(mp.read_text()).get("planes") else "WARN", "table_tip_front.json present" if mp.exists() else "only the OLD table_front.json: x is MIRRORED vs the kinematics (marker-run incident) -> run scripts/calib_tip_map.py --go or use taught poses")
have_key = (Path.home() / ".gemini_er_key").exists() or bool(os.environ.get("GEMINI_API_KEY"))
rec("Gemini ER-2 key", "PASS" if have_key else "FAIL", "set (value not shown)" if have_key else "put it in ~/.gemini_er_key (chmod 600)")
if have_key and "--er2" in sys.argv and be:
    try:
        from normakit import er2, vision
        jp = be.frame_jpeg(front_camera(be)); r = er2.ask(jp, 'Is a robot arm visible? Reply {"visible": true or false}'); rec("ER-2 call", "PASS", f"{r.get('seconds')} s, answer {r.get('visible')}")
    except Exception as e: rec("ER-2 call", "FAIL", str(e)[:120])
try:
    with urllib.request.urlopen("http://127.0.0.1:8890/api/state", timeout=4) as r: rec("web UI :8890", "PASS", f"http {r.status}")
except Exception: rec("web UI :8890", "WARN", "not running: .venv/bin/python scripts/ui.py   (or ./norma up)")
try:
    with urllib.request.urlopen("http://127.0.0.1:8891/state", timeout=4) as r: d = json.loads(r.read()); rec("armd :8891 (fast arm daemon)", "PASS", f"tip {d.get('tip')} armed {d.get('armed_s')} s")
except Exception: rec("armd :8891 (fast arm daemon)", "WARN", "not running: ./norma up")
try:
    from normakit import mcp_ui; names = [t.name for t in asyncio.run(mcp_ui.mcp.list_tools())]; rec("MCP elrobot-ui", "PASS", f"{len(names)} tools: {names}")
except Exception as e: rec("MCP elrobot-ui", "FAIL", str(e)[:100])
try:
    from normakit.sim_backend import SimBackend
    s = SimBackend(realtime=False); rec("pybullet sim backend", "PASS", f"{len(s.state().joints)} joints")
except Exception as e: rec("pybullet sim backend", "FAIL", str(e)[:100])
(ROOT / "run/stack_check.json").write_text(json.dumps({"time": time.strftime("%F %T"), "rows": rows}, indent=1))
n = {k: sum(r["status"] == k for r in rows) for k in ("PASS", "WARN", "FAIL")}; print(f"\nSUMMARY: {n}  -> run/stack_check.json", flush=True); os._exit(0)
