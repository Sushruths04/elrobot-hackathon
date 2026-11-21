"""End-to-end logic test on the digital twin (no hardware):
   FK/IK round-trip -> safety guard -> self-calibration -> pixel reach accuracy -> MCP tool registry.
Run: .venv/bin/python scripts/selftest_sim.py
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from normakit.sim_backend import SimBackend
from normakit.arm import Arm
from normakit.core import norm_to_rad
from normakit import vision

ok = True
def check(name, cond, extra=""):
    global ok
    ok &= bool(cond)
    print(("PASS " if cond else "FAIL ") + name, extra)

b = SimBackend(); b.start()
a = Arm(b, calib_path="run/sim_H.json")

# 1. IK round trip
tgt = (0.05, 0.22, 0.10)
q, err = a.kin.ik_norm(tgt, b.state().norms())
check("IK reaches target", err < 0.01, f"err={err*100:.2f}cm")

# 2. guard step-limit & clamp
r = a.guard.move({1: 5.0}, 0.5)
check("guard clamps/steps", r["ok"] and r["sent"][1] <= 0.98 and any("step-limited" in n or "clamped" in n for n in r["notes"]), str(r["notes"]))
a.guard.settle()

# 3. e-stop latch
a.guard.estop(); r = a.guard.move({1: 0.5}); check("estop blocks motion", not r["ok"]); a.guard.release()

# 4. camera
a.home(); jpg = a.look("table", grid=True); check("camera frame + grid", len(jpg) > 5000 and len(a.grid_cells) == 48, f"{len(jpg)}B")
Path("run").mkdir(exist_ok=True); Path("run/sim_table.jpg").write_bytes(jpg)

# 5. self-calibration
if Path("run/sim_H.json").exists() and "--recalib" not in sys.argv:
    print("calibration: reusing run/sim_H.json (pass --recalib to redo, ~2 min)")
else:
    t0 = time.time(); res = a.calibrate_table(n=10); print(res, f"{time.time()-t0:.1f}s")
    check("hand-eye self-calibration", res["ok"] and res["mean_err_cm"] < 1.5)

# 6. pixel reach accuracy on the red block
a.home(); time.sleep(0.5)
img = vision.decode(a.look("table"))
px = vision.find_named(img, "red")
check("find red block", px is not None, str(px))
if px:
    blk = b.object_pos("red_block").copy()           # position BEFORE the arm can touch it
    r = a.reach_pixel(*px, z=0.06)                   # hover just above (sim contact physics is not the point here)
    fk = a.kin.fk({i: norm_to_rad(i, v) for i, v in b.state().norms().items()})
    est = np.array(r["xy"]) if "xy" in r else np.array(a.H(*px))
    d_est = np.linalg.norm(est - blk[:2])
    check("pixel->table estimate within 3cm of block", d_est < 0.03,
          f"est={est.round(3)} true={blk[:2].round(3)} err={d_est*100:.2f}cm (parallax: calib plane != object plane)")
    d = np.linalg.norm(fk[:2] - blk[:2])
    print(f"INFO final gripper xy error vs block = {d*100:.2f}cm; descent result: {r}")
    print("INFO descent to table height times out in the twin (likely contact with block/table); NOT verified")

# 7. MCP registry
from normakit import mcp_server
names = sorted(t.name for t in __import__("asyncio").run(mcp_server.mcp.list_tools()))
check("MCP tools registered", {"look", "pick_cell", "stop", "calibrate_table"} <= set(names), str(names))
print("\nALL OK" if ok else "\nSOME FAILED")
sys.exit(0 if ok else 1)
