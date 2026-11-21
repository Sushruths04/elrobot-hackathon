"""Guarded real-robot diagnostic & ST3215 torque/PID helper for docs/REAL_RUNBOOK.md.

Modes:
  1) Read-only status & FK check (safe, no motion, no torque changes):
       .venv/bin/python scripts/servo_diag_and_fix.py status
  2) Safe limp / release (SUPPORT THE ARM BY HAND FIRST, then pass --yes):
       .venv/bin/python scripts/servo_diag_and_fix.py limp --yes
  3) Open-air +-60 tick test on joints 3 and 4 (arm clear of cup/table, hand on 12V plug, --yes):
       .venv/bin/python scripts/servo_diag_and_fix.py test-elbow --delta 60 --yes
  4) Apply calibrated ST3215 torque & PID settings (TorqueLimit=850, MaxTorque=850, OverloadTorque=85, P=36, I=12, D=20):
       .venv/bin/python scripts/servo_diag_and_fix.py apply-torque-fix --yes
  5) Single-joint direction check (+delta ticks and back, compare camera vs URDF FK):
       .venv/bin/python scripts/servo_diag_and_fix.py check-dir --joint 3 --delta 50 --yes
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from normakit.core import JOINT_IDS, norm_to_rad
from normakit.kin import Kin
from normakit.station_backend import (
    REG_ACC,
    REG_GOAL_POS,
    REG_GOAL_SPEED,
    REG_TORQUE,
    StationBackend,
    front_camera,
    wrist_camera,
)

REG_MAX_TORQUE = 0x10
REG_P_COEF = 0x15
REG_D_COEF = 0x16
REG_I_COEF = 0x17
REG_PROT_CURRENT = 0x1C
REG_PROT_TORQUE = 0x22
REG_OVERLOAD_TORQUE = 0x24
REG_TORQUE_LIMIT = 0x30
REG_LOCK = 0x37


def wd(a: int, b: int) -> int:
    return ((a - b + 2048) % 4096) - 2048


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["status", "limp", "test-elbow", "apply-torque-fix", "check-dir"])
    ap.add_argument("--joint", type=int, default=3)
    ap.add_argument("--delta", type=int, default=60)
    ap.add_argument("--yes", action="store_true", help="confirm human is at table with 12V plug in hand")
    a = ap.parse_args()

    if a.cmd != "status" and not a.yes:
        sys.exit("REFUSED: human must be at the table (hand on 12V plug / supporting arm). Re-run with --yes.")
    if abs(a.delta) > 90:
        sys.exit("REFUSED: --delta capped at 90 ticks (~7.9 deg) for safety.")

    be = StationBackend("localhost:8888")
    be.start()
    time.sleep(1.2)
    kin = Kin()
    st = be.state()

    def fk_tip(s):
        q = {i: norm_to_rad(i, s.joints[i].norm) for i in range(1, 8)}
        tip, _ = kin.jaw_tip_pose(q)
        ee = kin.fk(q)
        return ee, tip

    if a.cmd == "status":
        ee, tip = fk_tip(st)
        fcam = front_camera(be)
        wcam = wrist_camera(be)
        print(f"Cameras: front={fcam}, wrist={wcam}")
        print(f"URDF EE xyz (m)     : ({ee[0]:+.4f}, {ee[1]:+.4f}, {ee[2]:+.4f})")
        print(f"URDF jaw_tip xyz (m): ({tip[0]:+.4f}, {tip[1]:+.4f}, {tip[2]:+.4f})")
        for i in JOINT_IDS:
            j = st.joints[i]
            span = (j.hi - j.lo) % 4096
            print(f"  joint {i}: raw={j.raw:4d} range=[{j.lo:4d},{j.hi:4d}] span={span:4d} norm={j.norm:+.3f} cur={j.current_ma:3d}mA temp={j.temp_c}C")
        be.stop_backend()
        return

    if a.cmd == "limp":
        be._sync(REG_TORQUE, {i: b"\x00" for i in JOINT_IDS})
        time.sleep(0.2)
        print("TORQUE OFF on all 8 follower motors. Move the arm clear of the cup gently by hand.")
        be.stop_backend()
        return

    # Helper: always latch goal = present BEFORE enabling torque on ALL joints (never single-joint cycle!)
    def safe_torque_on(speed_tps: int = 220):
        s_now = be.state()
        be._sync(REG_GOAL_POS, {i: int(s_now.joints[i].raw).to_bytes(2, "little") for i in JOINT_IDS})
        time.sleep(0.05)
        be._sync(REG_TORQUE, {i: b"\x01" for i in JOINT_IDS})
        be._sync(REG_ACC, {i: bytes([30]) for i in JOINT_IDS})
        be._sync(REG_GOAL_SPEED, {i: int(speed_tps).to_bytes(2, "little") for i in JOINT_IDS})
        time.sleep(0.15)
        return be.state()

    if a.cmd == "apply-torque-fix":
        # Safe at rest: disable torque briefly? NO — only unlock EEPROM on motors at rest or write RAM TorqueLimit first!
        # Note: ST3215 allows RAM TorqueLimit (0x30) up to EEPROM MaxTorque (0x10). To raise EEPROM MaxTorque/PID safely,
        # motors must have torque=0 while unlocking EEPROM (0x37=0) -> warn human to support arm!
        print("NOTE: Unlocking EEPROM briefly turns torque OFF; ensure the arm is resting/supported.")
        be._sync(REG_TORQUE, {i: b"\x00" for i in JOINT_IDS})
        time.sleep(0.1)
        be._sync(REG_LOCK, {i: b"\x00" for i in JOINT_IDS})
        time.sleep(0.08)
        be._sync(REG_P_COEF, {i: bytes([36]) for i in JOINT_IDS})
        be._sync(REG_I_COEF, {i: bytes([12]) for i in JOINT_IDS})
        be._sync(REG_D_COEF, {i: bytes([20]) for i in JOINT_IDS})
        be._sync(REG_MAX_TORQUE, {i: (850).to_bytes(2, "little") for i in JOINT_IDS})
        be._sync(REG_PROT_CURRENT, {i: (500).to_bytes(2, "little") for i in JOINT_IDS})
        be._sync(REG_OVERLOAD_TORQUE, {i: bytes([85]) for i in JOINT_IDS})
        be._sync(REG_PROT_TORQUE, {i: bytes([55]) for i in JOINT_IDS})
        time.sleep(0.1)
        be._sync(REG_LOCK, {i: b"\x01" for i in JOINT_IDS})
        time.sleep(0.1)
        s_hold = safe_torque_on(speed_tps=220)
        be._sync(REG_TORQUE_LIMIT, {i: (850).to_bytes(2, "little") for i in JOINT_IDS})
        time.sleep(0.1)
        print("APPLIED calibrated ST3215 settings (P=36, I=12, D=20, MaxTorque=850, TorqueLimit=850, Overload=85%) and holding current pose:",
              {i: s_hold.joints[i].raw for i in JOINT_IDS})
        be.stop_backend()
        return

    if a.cmd == "test-elbow":
        s0 = safe_torque_on(speed_tps=220)
        be._sync(REG_TORQUE_LIMIT, {i: (850).to_bytes(2, "little") for i in JOINT_IDS})
        for jid in (3, 4):
            start_raw = be.state().joints[jid].raw
            for sign, tag in ((-1, f"-{a.delta} (up)"), (+1, f"+{a.delta} (down)"), (0, "return")):
                tgt = (start_raw + sign * a.delta) % 4096
                be._sync(REG_GOAL_POS, {jid: int(tgt).to_bytes(2, "little")})
                time.sleep(1.2)
                s_cur = be.state().joints[jid]
                moved = wd(s_cur.raw, start_raw)
                print(f"  joint {jid} cmd {tag:12s}: start={start_raw} target={tgt} actual={s_cur.raw} delta={moved:+4d} ticks cur={s_cur.current_ma}mA")
                if s_cur.current_ma > 700:
                    be.halt()
                    sys.exit("ABORT: current > 700 mA — holding pose.")
        be.halt()
        be.stop_backend()
        return

    if a.cmd == "check-dir":
        jid = a.joint
        fcam = front_camera(be)
        s0 = safe_torque_on(speed_tps=180)
        (ROOT / f"run/dir_j{jid}_before.jpg").write_bytes(be.frame_jpeg(fcam))
        ee0, tip0 = fk_tip(s0)
        r0 = s0.joints[jid].raw
        tgt = (r0 + a.delta) % 4096
        be._sync(REG_GOAL_POS, {jid: int(tgt).to_bytes(2, "little")})
        time.sleep(1.3)
        s1 = be.state()
        (ROOT / f"run/dir_j{jid}_plus.jpg").write_bytes(be.frame_jpeg(fcam))
        ee1, tip1 = fk_tip(s1)
        be._sync(REG_GOAL_POS, {jid: int(r0).to_bytes(2, "little")})
        time.sleep(1.3)
        be.halt()
        print(f"Joint {jid}: raw {r0} -> {s1.joints[jid].raw} ({wd(s1.joints[jid].raw, r0):+d} ticks)")
        print(f"  URDF predicted jaw_tip delta (mm): dx={(tip1[0]-tip0[0])*1000:+.1f}, dy={(tip1[1]-tip0[1])*1000:+.1f}, dz={(tip1[2]-tip0[2])*1000:+.1f}")
        print(f"  Saved run/dir_j{jid}_before.jpg and run/dir_j{jid}_plus.jpg — verify physical motion matches URDF delta.")
        be.stop_backend()


if __name__ == "__main__":
    main()
