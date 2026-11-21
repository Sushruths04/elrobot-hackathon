"""Render both real arms as the Station viewer would draw them (calibrated % -> URDF joint limits). Read-only.
Writes run/twin.png (leader left, follower right) and prints joint angles in degrees.
Mapping = Station viewer: joints 1-7 linear over URDF limits; gripper (8) flipped (upper - linear)."""
import asyncio, json, math, os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np, pybullet as p
from PIL import Image, ImageDraw
from normakit.busread import snapshot

URDF = ROOT / "vendor/norma-core-src/hardware/elrobot/simulation/elrobot_follower.urdf"
buses = json.loads((ROOT / "run/buses.json").read_text())

async def main():
    snap, _, _ = await snapshot()
    cid = p.connect(p.DIRECT)
    arms, report = {}, {}
    for k, (role, off) in enumerate([("leader", -0.35), ("follower", 0.35)]):
        a = p.loadURDF(str(URDF), basePosition=[off, 0, 0], useFixedBase=True)
        names = {p.getJointInfo(a, j)[1].decode(): j for j in range(p.getNumJoints(a))}
        ms = snap[buses[role]]
        rep = {}
        for i in range(1, 9):
            j = names[f"rev_motor_{i:02d}"]; lo, hi = p.getJointInfo(a, j)[8:10]
            n = min(1, max(0, ms[i]["norm"])); ang = lo + n * (hi - lo)
            if i == 8: ang = hi - ang
            p.resetJointState(a, j, ang); rep[i] = (round(ms[i]["norm"] * 100), round(math.degrees(ang)))
        for j in range(p.getNumJoints(a)):
            p.changeVisualShape(a, j, rgbaColor=[0.25, 0.5, 0.9, 1] if role == "leader" else [0.9, 0.55, 0.2, 1])
        report[role] = rep
    imgs = []
    for eye, tgt, label in [([0, -0.9, 0.35], [0, 0, 0.2], "front"), ([1.0, 0, 0.35], [0, 0, 0.2], "side")]:
        V = p.computeViewMatrix(eye, tgt, [0, 0, 1]); P = p.computeProjectionMatrixFOV(50, 1.0, 0.05, 3)
        w = p.getCameraImage(560, 560, V, P, renderer=p.ER_TINY_RENDERER)[2]
        im = Image.fromarray(np.reshape(w, (560, 560, 4))[:, :, :3].astype("uint8")); ImageDraw.Draw(im).text((8, 8), label, fill=(0, 0, 0)); imgs.append(im)
    out = Image.new("RGB", (1120, 560)); out.paste(imgs[0], (0, 0)); out.paste(imgs[1], (560, 0))
    d = ImageDraw.Draw(out); d.text((8, 540), "BLUE = leader (left in front view)   ORANGE = follower (right)", fill=(0, 0, 0))
    out.save(ROOT / "run/twin.png")
    print("joint: leader(%,deg) follower(%,deg)")
    for i in range(1, 9): print(i, report["leader"][i], report["follower"][i])
    sys.stdout.flush(); os._exit(0)
asyncio.run(main())
