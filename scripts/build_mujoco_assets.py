"""Generate clean, portable MuJoCo 3.x MJCF scene, convex-decomposed jaw & hollow-cup meshes,
textured table surface, and camera calibration files from the official ElRobot URDF and handover data."""
from __future__ import annotations

import json
import math
import os
import re
import struct
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import mujoco
import numpy as np
import pybullet as p

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from normakit.core import URDF_LIMITS, norm_to_rad
from normakit.kin import Kin

MJ_DIR = ROOT / "assets/mujoco"
HULL_DIR = MJ_DIR / "hulls"
URDF_PATH = ROOT / "vendor/norma-core-src/hardware/elrobot/simulation/elrobot_follower.urdf"
URDF_ASSETS = URDF_PATH.parent / "assets"


def load_stl_verts(path: Path) -> np.ndarray:
    data = path.read_bytes()
    ntri = struct.unpack("<I", data[80:84])[0]
    verts = np.zeros((ntri * 3, 3), dtype=np.float64)
    off = 84
    for i in range(ntri):
        vals = struct.unpack("<12fH", data[off : off + 50])
        verts[3 * i] = vals[3:6]
        verts[3 * i + 1] = vals[6:9]
        verts[3 * i + 2] = vals[9:12]
        off += 50
    return verts * 0.001


def stl_to_obj(stl_path: Path, obj_path: Path, scale: float = 0.001, offset=(0.0, 0.0, 0.0)) -> None:
    data = stl_path.read_bytes()
    ntri = struct.unpack("<I", data[80:84])[0]
    verts, faces, vmap = [], [], {}
    off = 84
    ox, oy, oz = offset
    for _ in range(ntri):
        vals = struct.unpack("<12fH", data[off : off + 50])
        off += 50
        tri_idx = []
        for k in range(3):
            key = (
                round(vals[3 + 3 * k] * scale + ox, 7),
                round(vals[4 + 3 * k] * scale + oy, 7),
                round(vals[5 + 3 * k] * scale + oz, 7),
            )
            if key not in vmap:
                vmap[key] = len(verts) + 1
                verts.append(key)
            tri_idx.append(vmap[key])
        if len(set(tri_idx)) == 3:
            faces.append(tri_idx)
    with obj_path.open("w") as f:
        for vx, vy, vz in verts:
            f.write(f"v {vx:.7f} {vy:.7f} {vz:.7f}\n")
        for a, b, c in faces:
            f.write(f"f {a} {b} {c}\n")


def write_binary_stl(out_stl: Path, V: np.ndarray, F: list[tuple[int, int, int]]) -> None:
    with out_stl.open("wb") as fp:
        fp.write(b"MJCF convex mesh".ljust(80, b"\0"))
        fp.write(struct.pack("<I", len(F)))
        for i0, i1, i2 in F:
            p0, p1, p2 = V[i0], V[i1], V[i2]
            n = np.cross(p1 - p0, p2 - p0)
            nn = np.linalg.norm(n)
            if nn > 1e-12:
                n = n / nn
            fp.write(
                struct.pack(
                    "<12fH",
                    n[0], n[1], n[2],
                    p0[0], p0[1], p0[2],
                    p1[0], p1[1], p1[2],
                    p2[0], p2[1], p2[2],
                    0,
                )
            )


def build_jaw_hulls() -> dict[str, list[str]]:
    HULL_DIR.mkdir(parents=True, exist_ok=True)
    for old in HULL_DIR.glob("jaw_*_hull_*.stl"):
        old.unlink()
    jaws = [
        ("jaw_01", URDF_ASSETS / "Gripper_Jaw_01_v1_1.stl", (-0.311498, -0.016199, -0.171401)),
        ("jaw_02", URDF_ASSETS / "Gripper_Jaw_02_v1_1.stl", (-0.314659, -0.059918, -0.251741)),
    ]
    out_names: dict[str, list[str]] = {}
    with tempfile.TemporaryDirectory() as td:
        td_p = Path(td)
        for name, stl_p, origin_off in jaws:
            in_obj = td_p / f"{name}_in.obj"
            out_obj = td_p / f"{name}_vhacd.obj"
            log_f = td_p / f"{name}.log"
            stl_to_obj(stl_p, in_obj, scale=0.001, offset=origin_off)
            p.vhacd(
                str(in_obj),
                str(out_obj),
                str(log_f),
                concavity=0.0015,
                alpha=0.05,
                beta=0.05,
                minVolumePerCH=0.00001,
                resolution=250000,
                maxNumVerticesPerCH=48,
            )
            parts, cur_v, cur_f, v_off = [], [], [], 0
            for line in out_obj.read_text().splitlines():
                if line.startswith(("o ", "g ")):
                    if cur_v and cur_f:
                        parts.append((np.array(cur_v, float), cur_f))
                        v_off += len(cur_v)
                        cur_v, cur_f = [], []
                elif line.startswith("v "):
                    cur_v.append([float(x) for x in line.split()[1:4]])
                elif line.startswith("f "):
                    idx = [int(tok.split("/")[0]) - 1 - v_off for tok in line.split()[1:]]
                    for k in range(1, len(idx) - 1):
                        cur_f.append((idx[0], idx[k], idx[k + 1]))
            if cur_v and cur_f:
                parts.append((np.array(cur_v, float), cur_f))
            names = []
            for V, F in parts:
                if len(V) < 4 or len(F) < 4:
                    continue
                fname = f"{name}_hull_{len(names):02d}.stl"
                write_binary_stl(HULL_DIR / fname, V, F)
                names.append(fname)
            out_names[name] = names
    return out_names


def build_hollow_cup_meshes(n_seg: int = 24) -> int:
    for old in HULL_DIR.glob("cup_*.stl"):
        old.unlink()
    r_top_o, r_top_i = 0.0355, 0.0323
    r_bot_o, r_bot_i = 0.0280, 0.0248
    h_cup, t_floor = 0.096, 0.004
    vis_V, vis_F = [], []
    for s in range(n_seg):
        a0 = 2 * math.pi * s / n_seg
        a1 = 2 * math.pi * (s + 1) / n_seg
        c0, s0 = math.cos(a0), math.sin(a0)
        c1, s1 = math.cos(a1), math.sin(a1)
        V8 = np.array(
            [
                [r_bot_i * c0, r_bot_i * s0, t_floor],
                [r_bot_o * c0, r_bot_o * s0, 0.0],
                [r_bot_o * c1, r_bot_o * s1, 0.0],
                [r_bot_i * c1, r_bot_i * s1, t_floor],
                [r_top_i * c0, r_top_i * s0, h_cup],
                [r_top_o * c0, r_top_o * s0, h_cup],
                [r_top_o * c1, r_top_o * s1, h_cup],
                [r_top_i * c1, r_top_i * s1, h_cup],
            ],
            dtype=float,
        )
        F12 = [
            (0, 2, 1), (0, 3, 2),
            (4, 5, 6), (4, 6, 7),
            (1, 2, 6), (1, 6, 5),
            (0, 7, 3), (0, 4, 7),
            (0, 1, 5), (0, 5, 4),
            (3, 6, 2), (3, 7, 6),
        ]
        write_binary_stl(HULL_DIR / f"cup_wall_{s:02d}.stl", V8, F12)
        base_idx = len(vis_V)
        vis_V.extend(V8.tolist())
        vis_F.extend([(base_idx + a, base_idx + b, base_idx + c) for a, b, c in F12[:8]])

    floor_V = []
    for z in [0.0, t_floor]:
        for s in range(n_seg):
            a = 2 * math.pi * s / n_seg
            floor_V.append([r_bot_o * math.cos(a), r_bot_o * math.sin(a), z])
    floor_V_arr = np.array(floor_V, float)
    floor_F = []
    for s in range(1, n_seg - 1):
        floor_F.append((0, s + 1, s))
        floor_F.append((n_seg, n_seg + s, n_seg + s + 1))
    for s in range(n_seg):
        sn = (s + 1) % n_seg
        floor_F.append((s, sn, n_seg + sn))
        floor_F.append((s, n_seg + sn, n_seg + s))
    write_binary_stl(HULL_DIR / "cup_floor.stl", floor_V_arr, floor_F)
    b_idx = len(vis_V)
    vis_V.extend(floor_V_arr.tolist())
    vis_F.extend([(b_idx + a, b_idx + b, b_idx + c) for a, b, c in floor_F])
    write_binary_stl(HULL_DIR / "cup_visual.stl", np.array(vis_V, float), vis_F)
    return n_seg


def build_table_texture(out_png: Path) -> None:
    rng = np.random.default_rng(7)
    H, W = 1024, 1024
    # Light white-grey woodgrain laminate matching real_front_cam.jpg (~210-228 RGB)
    base = np.full((H, W, 3), [216, 218, 220], dtype=np.float32)
    streaks = rng.normal(0.0, 4.5, size=(H // 16, W)).astype(np.float32)
    streaks = cv2.resize(streaks, (W, H), interpolation=cv2.INTER_CUBIC)
    streaks = cv2.GaussianBlur(streaks, (3, 41), 0)
    for c in range(3):
        base[:, :, c] += streaks
    # Subtle table seam line across middle like the two-leaf table in real_front_cam.jpg
    base[520:523, :, :] -= 22.0
    # Two yellow lane tape stripes matching real_front_cam.jpg (at x ~ +0.20 m and x ~ -0.14 m)
    # Table plane spans x in [-0.50, +0.50] -> u_tex = (x + 0.50) * 1024, y in [-0.15, +0.85] -> v_tex = (0.85 - y) * 1024
    for x_tape in [-0.15, 0.21]:
        u0 = int((x_tape + 0.50) * W)
        v0, v1 = int((0.85 - 0.52) * H), int((0.85 - 0.02) * H)
        base[v0:v1, u0 - 8 : u0 + 8] = [212, 198, 108]  # RGB muted yellow masking tape
    img = np.clip(base, 0, 255).astype(np.uint8)
    # BGR for cv2.imwrite
    cv2.imwrite(str(out_png), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def compute_front_camera_pose() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit exact 6-DoF front camera pose in robot base frame from board_frame_front.jpg + touch_points_front.json."""
    vj1 = load_stl_verts(URDF_ASSETS / "Gripper_Jaw_01_v1_1.stl") + np.array([-0.311498, -0.016199, -0.171401])
    vj2 = load_stl_verts(URDF_ASSETS / "Gripper_Jaw_02_v1_1.stl") + np.array([-0.314659, -0.059918, -0.251741])
    kin = Kin()

    def exact_tip(norms: dict) -> np.ndarray:
        q = {int(k): norm_to_rad(int(k), float(val)) for k, val in norms.items() if int(k) <= 7}
        n8 = float(norms["8"])
        q[8] = URDF_LIMITS[8][1] * (1.0 - n8)
        kin._set(q)
        p.resetJointState(kin.rid, 16, -0.0255 * (1.0 - n8), physicsClientId=kin.cid)
        p.resetJointState(kin.rid, 17, +0.0255 * (1.0 - n8), physicsClientId=kin.cid)
        ls16 = p.getLinkState(kin.rid, 16, computeForwardKinematics=True, physicsClientId=kin.cid)
        ls17 = p.getLinkState(kin.rid, 17, computeForwardKinematics=True, physicsClientId=kin.cid)
        w2 = (np.array(p.getMatrixFromQuaternion(ls16[5])).reshape(3, 3) @ vj2.T).T + np.array(ls16[4])
        w1 = (np.array(p.getMatrixFromQuaternion(ls17[5])).reshape(3, 3) @ vj1.T).T + np.array(ls17[4])
        both = np.vstack([w1, w2])
        return both[both[:, 2] < both[:, 2].min() + 0.010].mean(axis=0)

    tp = json.loads((ROOT / "calibration/touch_points_front.json").read_text())
    cam_cal = json.loads((ROOT / "calibration/camera_front.json").read_text())
    K = np.array(cam_cal["K"], float)
    dist = np.array(cam_cal["dist"], float)
    img = cv2.imread(str(ROOT / "calibration/board_frame_front.jpg"))
    ok, corners = cv2.findChessboardCornersSB(
        cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (9, 6), cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
    )
    assert ok
    flat = corners.reshape(-1, 2).astype(float)
    sq = 0.024  # A4-printed 9x6 pattern (10x7 outer border touched from behind robot: 6.1 mm fit residual)
    cols, rows = np.meshgrid(np.arange(9), np.arange(6))
    bxy_rh = np.stack([-cols.ravel() * sq, rows.ravel() * sq], axis=1)
    B_outer = np.array([[-9 * sq, 6 * sq], [1 * sq, 6 * sq], [1 * sq, -1 * sq], [-9 * sq, -1 * sq]])
    Y_tips = np.array(
        [exact_tip(tp[o]["norms"])[:2] for o in ["bottom-right", "bottom-left", "top-left", "top-right"]]
    )
    mx, my = B_outer.mean(0), Y_tips.mean(0)
    U, _, Vt = np.linalg.svd((Y_tips - my).T @ (B_outer - mx))
    D = np.diag([1.0, 1.0 if np.linalg.det(U @ Vt) > 0 else -1.0])
    Rm = U @ D @ Vt
    t = my - Rm @ mx
    pts3d = np.column_stack([(Rm @ bxy_rh.T).T + t, np.zeros(len(bxy_rh))])
    _, rvec, tvec = cv2.solvePnP(pts3d, flat, K, dist)
    R_cw, _ = cv2.Rodrigues(rvec)
    R_wc = R_cw.T  # columns: [X_right, Y_down, Z_forward] in world
    cam_pos = (-R_cw.T @ tvec).ravel()

    # Exact pinhole homography for the MuJoCo pinhole camera (z=0 table plane -> pixel -> robot xy)
    # P = K @ [r1_cw, r2_cw, tvec], so H_rob2px = K @ [R_cw[:,0], R_cw[:,1], tvec]
    H_r2p = K @ np.column_stack([R_cw[:, 0], R_cw[:, 1], tvec.ravel()])
    H_p2r = np.linalg.inv(H_r2p)
    H_p2r = H_p2r / H_p2r[2, 2]

    # Also compute undistorted/raw homography for real camera (180-swap fixed!)
    Hp2b, _ = cv2.findHomography(flat.astype(np.float32), bxy_rh.astype(np.float32), cv2.RANSAC, 0.5)
    M_b2r = np.eye(3)
    M_b2r[:2, :2] = Rm
    M_b2r[:2, 2] = t
    H_real_fixed = M_b2r @ Hp2b
    H_real_fixed = H_real_fixed / H_real_fixed[2, 2]
    err_mm = np.linalg.norm((Rm @ B_outer.T).T + t - Y_tips, axis=1) * 1000.0
    (ROOT / "calibration/table_front_180fix.json").write_text(
        json.dumps(
            {
                "cam": "front",
                "name": "front_180fix",
                "note": "Fixed 180-deg perspective swap (human stood behind robot) + outer 10x7 border + jaw-tip FK",
                "H_pixel_to_robot_xy_m": H_real_fixed.tolist(),
                "residual_mm": err_mm.round(2).tolist(),
                "mean_residual_mm": round(float(err_mm.mean()), 2),
            },
            indent=1,
        )
    )
    (ROOT / "calibration/table_mujoco_front.json").write_text(
        json.dumps(
            {
                "cam": "cam0",
                "name": "mujoco_front",
                "model": "pinhole_exact_from_camera_front_solvePnP",
                "H_pixel_to_robot_xy_m": H_p2r.tolist(),
                "camera_pos_world_m": cam_pos.round(5).tolist(),
            },
            indent=1,
        )
    )
    return cam_pos, R_wc, K


def build_mjcf(jaw_hulls: dict[str, list[str]], n_cup_seg: int, cam_pos: np.ndarray, R_wc: np.ndarray, K_front: np.ndarray) -> None:
    urdf_txt = URDF_PATH.read_text()
    urdf_clean = re.sub(r"<mimic[^/]*/>", "", urdf_txt)
    mj_hdr = (
        f'<mujoco><compiler meshdir="{URDF_ASSETS.resolve()}" strippath="true" '
        f'fusestatic="false" discardvisual="false" balanceinertia="true" boundinertia="1e-07"/></mujoco>'
    )
    urdf_clean = urdf_clean.replace('<robot name="Elrobot" >', f'<robot name="Elrobot" >\n  {mj_hdr}')
    m = mujoco.MjModel.from_xml_string(urdf_clean)
    tmp_xml = MJ_DIR / "_tmp_urdf.xml"
    mujoco.mj_saveLastXML(str(tmp_xml), m)
    root = ET.parse(tmp_xml).getroot()
    tmp_xml.unlink()

    # Clean compiler to use portable relative paths from assets/mujoco/
    comp = root.find("compiler")
    comp.attrib.clear()
    comp.set("angle", "radian")
    comp.set("boundinertia", "1e-07")
    comp.set("autolimits", "true")

    # Add option & visual settings
    opt = ET.Element("option", {
        "timestep": "0.002",
        "gravity": "0 0 -9.81",
        "integrator": "implicitfast",
        "cone": "elliptic",
        "impratio": "5",
    })
    vis = ET.Element("visual")
    ET.SubElement(vis, "global", {"offwidth": "640", "offheight": "480"})
    ET.SubElement(vis, "quality", {"shadowsize": "512", "offsamples": "0"})
    ET.SubElement(vis, "headlight", {"ambient": "0.30 0.30 0.30", "diffuse": "0.22 0.22 0.22", "specular": "0.08 0.08 0.08"})
    root.insert(1, opt)
    root.insert(2, vis)

    # Assets: rewrite URDF STL paths to relative ../../vendor/... and add hulls, textures, materials
    asset = root.find("asset")
    rel_urdf_assets = "../../vendor/norma-core-src/hardware/elrobot/simulation/assets"
    for mesh_el in asset.findall("mesh"):
        f = mesh_el.attrib["file"]
        mesh_el.set("file", f"{rel_urdf_assets}/{f}")

    for prefix, flist in jaw_hulls.items():
        for fn in flist:
            ET.SubElement(asset, "mesh", {"name": fn[:-4], "file": f"hulls/{fn}"})
    for s in range(n_cup_seg):
        ET.SubElement(asset, "mesh", {"name": f"cup_wall_{s:02d}", "file": f"hulls/cup_wall_{s:02d}.stl"})
    ET.SubElement(asset, "mesh", {"name": "cup_floor", "file": "hulls/cup_floor.stl"})
    ET.SubElement(asset, "mesh", {"name": "cup_visual", "file": "hulls/cup_visual.stl"})

    ET.SubElement(asset, "texture", {"name": "table_tex", "type": "2d", "file": "table_tex.png"})
    ET.SubElement(asset, "texture", {"name": "sky_tex", "type": "skybox", "builtin": "gradient", "rgb1": "0.78 0.80 0.82", "rgb2": "0.62 0.64 0.66", "width": "256", "height": "256"})
    ET.SubElement(asset, "material", {"name": "table_mat", "texture": "table_tex", "specular": "0.12", "shininess": "0.15", "reflectance": "0.02"})
    ET.SubElement(asset, "material", {"name": "cup_mat", "rgba": "0.09 0.09 0.10 1", "specular": "0.18", "shininess": "0.25"})
    ET.SubElement(asset, "material", {"name": "arm_pla_mat", "rgba": "0.14 0.14 0.15 1", "specular": "0.15", "shininess": "0.2"})
    ET.SubElement(asset, "material", {"name": "servo_mat", "rgba": "0.07 0.07 0.08 1", "specular": "0.22", "shininess": "0.3"})
    ET.SubElement(asset, "material", {"name": "bottle_glass_mat", "rgba": "0.84 0.86 0.84 0.48", "specular": "0.65", "shininess": "0.85"})
    ET.SubElement(asset, "material", {"name": "bottle_label_mat", "rgba": "0.12 0.12 0.13 1", "specular": "0.1", "shininess": "0.1"})
    ET.SubElement(asset, "material", {"name": "coaster_mat", "rgba": "0.14 0.32 0.86 1", "specular": "0.1", "shininess": "0.1"})

    # Calibrated joint damping/armature/frictionloss (ST3215 1:345 spur gear train)
    # Armature ~ 0.012 kg*m^2 on main joints, 0.004 on wrist/gripper; frictionloss calibrated to sag & stall data
    jnt_dyn = {
        "rev_motor_01": ("0.012", "0.35", "0.14"),
        "rev_motor_02": ("0.016", "0.55", "0.22"),
        "rev_motor_03": ("0.016", "0.60", "0.242"),
        "rev_motor_04": ("0.014", "0.42", "0.135"),
        "rev_motor_05": ("0.006", "0.18", "0.06"),
        "rev_motor_06": ("0.008", "0.22", "0.08"),
        "rev_motor_07": ("0.005", "0.15", "0.05"),
        "rev_motor_08": ("0.004", "0.12", "0.02"),
        "rev_motor_08_1": ("0.01", "1.5", "0.01"),
        "rev_motor_08_2": ("0.01", "1.5", "0.01"),
    }

    wb = root.find("worldbody")
    # Walk robot bodies: style visual geoms cleanly, give arm links contype=2 conaffinity=1, and replace jaw collision geoms with V-HACD hulls!
    for body in wb.iter("body"):
        bname = body.attrib.get("name", "")
        for jnt in body.findall("joint"):
            jn = jnt.attrib["name"]
            if jn in jnt_dyn:
                arm_v, dmp_v, fl_v = jnt_dyn[jn]
                jnt.set("armature", arm_v)
                jnt.set("damping", dmp_v)
                jnt.set("frictionloss", fl_v)

        geoms = list(body.findall("geom"))
        for idx_g, g in enumerate(geoms):
            is_vis = g.attrib.get("contype") == "0"
            mname = g.attrib.get("mesh", "")
            if is_vis:
                g.set("name", f"vis_{bname}_{idx_g}")
                g.attrib.pop("rgba", None)
                g.set("material", "servo_mat" if mname.startswith("ST3215") else "arm_pla_mat")
            else:
                if bname in ("Gripper_Jaw_01_v1_1", "Gripper_Jaw_02_v1_1"):
                    body.remove(g)  # Replace single non-convex STL hull with V-HACD convex decomposition!
                elif bname == "Gripper_Base_v1_1":
                    g.set("name", f"col_{bname}_{idx_g}")
                    g.set("group", "3")
                    g.set("contype", "2")
                    g.set("conaffinity", "1")
                else:
                    # Upper-arm servo/bracket housings stay high above table (enforced by FK checks); skip costly 25x15 mesh GJK
                    body.remove(g)

        if bname == "Gripper_Jaw_01_v1_1":
            for hi, fn in enumerate(jaw_hulls["jaw_01"]):
                ET.SubElement(body, "geom", {
                    "name": f"col_jaw01_{hi:02d}",
                    "type": "mesh",
                    "mesh": fn[:-4],
                    "group": "3",
                    "contype": "1",
                    "conaffinity": "3",
                    "condim": "4",
                    "friction": "1.05 0.03 0.002",
                    "solref": "0.004 1",
                    "solimp": "0.94 0.99 0.0008",
                })
        elif bname == "Gripper_Jaw_02_v1_1":
            for hi, fn in enumerate(jaw_hulls["jaw_02"]):
                ET.SubElement(body, "geom", {
                    "name": f"col_jaw02_{hi:02d}",
                    "type": "mesh",
                    "mesh": fn[:-4],
                    "group": "3",
                    "contype": "1",
                    "conaffinity": "3",
                    "condim": "4",
                    "friction": "1.05 0.03 0.002",
                    "solref": "0.004 1",
                    "solimp": "0.94 0.99 0.0008",
                })
        elif bname == "Gripper_Base_v1_1":
            # Add sites for EE (exact URDF Gripper_Base_v1_1 origin) and jaw_tip_site (midpoint between jaw tips)
            ET.SubElement(body, "site", {"name": "ee_site", "pos": "0 0 0", "size": "0.004", "group": "4"})
            ET.SubElement(body, "site", {"name": "jaw_tip_site", "pos": "0.0009 0.092 0.0", "size": "0.004", "group": "4"})
            # Add real wrist camera mount + PCB mass (~38 g, visible in real_front_cam.jpg) and calibrated wrist camera!
            wcam_cal = json.loads((ROOT / "calibration/camera_wrist.json").read_text())
            Kw = wcam_cal["K"]
            fx_w, fy_w, cx_w, cy_w = Kw[0][0], Kw[1][1], Kw[0][2], Kw[1][2]
            dx_w, dy_w = 320.0 - cx_w, cy_w - 240.0
            cam_mount = ET.SubElement(body, "body", {"name": "wrist_cam_mount", "pos": "0.0008 0.038 0.036"})
            ET.SubElement(cam_mount, "inertial", {"pos": "0 0 0", "mass": "0.038", "diaginertia": "6e-06 6e-06 4e-06"})
            ET.SubElement(cam_mount, "geom", {
                "name": "vis_wrist_cam_pcb",
                "type": "box",
                "size": "0.016 0.016 0.004",
                "contype": "0",
                "conaffinity": "0",
                "group": "1",
                "material": "servo_mat",
            })
            # Wrist camera looks forward along +Y_local, pitched 22 deg toward -Z_local so jaws appear at bottom of frame (like real_other_cam_cam4.jpg)
            pitch = math.radians(22.0)
            # MuJoCo camera +X = +X_local (right), +Y = cos(p)*Z_local + sin(p)*Y_local (up), so -Z (look) = cos(p)*Y_local - sin(p)*Z_local
            ET.SubElement(body, "camera", {
                "name": "wrist",
                "pos": "0.0008 0.036 0.040",
                "xyaxes": f"1 0 0 0 {math.sin(pitch):.6f} {math.cos(pitch):.6f}",
                "resolution": "640 480",
                "sensorsize": "0.0064 0.0048",
                "focalpixel": f"{fx_w:.3f} {fy_w:.3f}",
                "principalpixel": f"{dx_w:.3f} {dy_w:.3f}",
            })

    # Add overhead lighting, radiator/back-wall backdrop, textured table, front camera, hollow black cup, distractor bottle, and drop coaster!
    ET.SubElement(wb, "light", {"name": "ceil_main", "pos": "0.0 0.28 1.35", "dir": "0 0 -1", "diffuse": "0.48 0.48 0.47", "specular": "0.18 0.18 0.18", "castshadow": "true"})
    ET.SubElement(wb, "light", {"name": "ceil_fill", "pos": "0.25 0.55 1.10", "dir": "-0.2 -0.3 -0.9", "diffuse": "0.22 0.22 0.24", "specular": "0.06 0.06 0.06", "castshadow": "false"})

    # Table box top at z=0.000 (spanning x in [-0.50, +0.50], y in [-0.15, +0.85])
    ET.SubElement(wb, "geom", {
        "name": "table_top",
        "type": "box",
        "pos": "0.0 0.35 -0.015",
        "size": "0.50 0.50 0.015",
        "material": "table_mat",
        "contype": "1",
        "conaffinity": "3",
        "condim": "4",
        "friction": "0.72 0.01 0.001",
    })
    # Subtle back wall / radiator visual behind the robot (matching real_front_cam.jpg background)
    ET.SubElement(wb, "geom", {
        "name": "back_wall",
        "type": "box",
        "pos": "0.0 -0.22 0.25",
        "size": "0.65 0.02 0.35",
        "rgba": "0.78 0.79 0.78 1",
        "contype": "0",
        "conaffinity": "0",
    })

    # Calibrated front (table overview) camera
    fx_f, fy_f, cx_f, cy_f = K_front[0, 0], K_front[1, 1], K_front[0, 2], K_front[1, 2]
    dx_f, dy_f = 320.0 - cx_f, cy_f - 240.0
    r_right = R_wc[:, 0]
    r_up = -R_wc[:, 1]
    xyaxes_front = f"{r_right[0]:.6f} {r_right[1]:.6f} {r_right[2]:.6f} {r_up[0]:.6f} {r_up[1]:.6f} {r_up[2]:.6f}"
    ET.SubElement(wb, "camera", {
        "name": "front",
        "pos": f"{cam_pos[0]:.5f} {cam_pos[1]:.5f} {cam_pos[2]:.5f}",
        "xyaxes": xyaxes_front,
        "resolution": "640 480",
        "sensorsize": "0.0064 0.0048",
        "focalpixel": f"{fx_f:.3f} {fy_f:.3f}",
        "principalpixel": f"{dx_f:.3f} {dy_f:.3f}",
    })

    # Place target coaster (mocap body so benchmark can randomise target place position too!)
    coaster = ET.SubElement(wb, "body", {"name": "place_target", "mocap": "true", "pos": "-0.09 0.25 0.0008"})
    ET.SubElement(coaster, "geom", {
        "name": "place_coaster_geom",
        "type": "cylinder",
        "size": "0.052 0.0008",
        "material": "coaster_mat",
        "contype": "0",
        "conaffinity": "0",
    })

    # Hollow black cup (6-DoF freejoint, 24 convex wall sectors + floor + smooth visual mesh)
    cup = ET.SubElement(wb, "body", {"name": "black_cup", "pos": "0.06 0.24 0.001"})
    ET.SubElement(cup, "freejoint", {"name": "cup_freejoint"})
    ET.SubElement(cup, "inertial", {"pos": "0 0 0.041", "mass": "0.042", "diaginertia": "4.6e-05 4.6e-05 2.4e-05"})
    ET.SubElement(cup, "geom", {
        "name": "cup_vis",
        "type": "mesh",
        "mesh": "cup_visual",
        "material": "cup_mat",
        "contype": "0",
        "conaffinity": "0",
        "group": "1",
    })
    ET.SubElement(cup, "geom", {
        "name": "cup_col_floor",
        "type": "mesh",
        "mesh": "cup_floor",
        "group": "3",
        "contype": "1",
        "conaffinity": "3",
        "condim": "4",
        "friction": "0.85 0.02 0.002",
        "solref": "0.004 1",
        "solimp": "0.93 0.99 0.0008",
    })
    for s in range(n_cup_seg):
        ET.SubElement(cup, "geom", {
            "name": f"cup_col_w{s:02d}",
            "type": "mesh",
            "mesh": f"cup_wall_{s:02d}",
            "group": "3",
            "contype": "1",
            "conaffinity": "3",
            "condim": "4",
            "friction": "0.90 0.03 0.002",
            "solref": "0.004 1",
            "solimp": "0.94 0.99 0.0008",
        })

    # Distractor bottle (with dark neck band like after_grab_attempt.jpg!)
    bottle = ET.SubElement(wb, "body", {"name": "distractor_bottle", "pos": "-0.16 0.34 0.001"})
    ET.SubElement(bottle, "freejoint", {"name": "bottle_freejoint"})
    ET.SubElement(bottle, "inertial", {"pos": "0 0 0.085", "mass": "0.240", "diaginertia": "0.00085 0.00085 0.00016"})
    ET.SubElement(bottle, "geom", {
        "name": "bottle_body",
        "type": "cylinder",
        "pos": "0 0 0.062",
        "size": "0.032 0.062",
        "material": "bottle_glass_mat",
        "contype": "1",
        "conaffinity": "3",
        "condim": "4",
        "friction": "0.65 0.01 0.001",
    })
    ET.SubElement(bottle, "geom", {
        "name": "bottle_shoulder",
        "type": "ellipsoid",
        "pos": "0 0 0.124",
        "size": "0.032 0.032 0.024",
        "material": "bottle_glass_mat",
        "contype": "0",
        "conaffinity": "0",
    })
    ET.SubElement(bottle, "geom", {
        "name": "bottle_neck",
        "type": "cylinder",
        "pos": "0 0 0.165",
        "size": "0.013 0.028",
        "material": "bottle_glass_mat",
        "contype": "1",
        "conaffinity": "3",
    })
    ET.SubElement(bottle, "geom", {
        "name": "bottle_neck_label",
        "type": "cylinder",
        "pos": "0 0 0.155",
        "size": "0.014 0.010",
        "material": "bottle_label_mat",
        "contype": "0",
        "conaffinity": "0",
    })

    # Contact exclusions inside gripper mechanism so internal rack bars & servo housing never jam jaw closure
    contact = ET.SubElement(root, "contact")
    ET.SubElement(contact, "exclude", {"body1": "Gripper_Jaw_01_v1_1", "body2": "Gripper_Jaw_02_v1_1"})
    for jb in ("Gripper_Jaw_01_v1_1", "Gripper_Jaw_02_v1_1"):
        for pb in ("Gripper_Base_v1_1", "ST3215_8_v1_1", "Gripper_Gear_v1_1", "ST3215_7_v1_1", "wrist_cam_mount", "Joint_06_v1_1"):
            ET.SubElement(contact, "exclude", {"body1": pb, "body2": jb})

    # Equality mimic constraints coupling motor 8 gear (rev_motor_08) to both prismatic jaws
    ratio = 0.0255 / 2.2028  # 0.011576176 m/rad
    eq = ET.SubElement(root, "equality")
    ET.SubElement(eq, "joint", {
        "name": "mimic_jaw_02",
        "joint1": "rev_motor_08_1",
        "joint2": "rev_motor_08",
        "polycoef": f"0 {-ratio:.8f} 0 0 0",
        "solref": "0.002 1",
        "solimp": "0.98 0.998 0.0002",
    })
    ET.SubElement(eq, "joint", {
        "name": "mimic_jaw_01",
        "joint1": "rev_motor_08_2",
        "joint2": "rev_motor_08",
        "polycoef": f"0 {ratio:.8f} 0 0 0",
        "solref": "0.002 1",
        "solimp": "0.98 0.998 0.0002",
    })

    # 8 torque actuators (controlled at 500 Hz by our ST3215 12V servo model)
    act = ET.SubElement(root, "actuator")
    for i in range(1, 9):
        ET.SubElement(act, "motor", {
            "name": f"act_motor_{i:02d}",
            "joint": f"rev_motor_{i:02d}",
            "ctrlrange": "-2.94 2.94",
            "gear": "1",
        })

    ET.indent(root, space="  ")
    out_xml = MJ_DIR / "elrobot_scene.xml"
    out_xml.write_text(ET.tostring(root, encoding="unicode") + "\n")
    print("Wrote", out_xml.relative_to(ROOT))


def main() -> None:
    MJ_DIR.mkdir(parents=True, exist_ok=True)
    jaw_hulls = build_jaw_hulls()
    n_seg = build_hollow_cup_meshes(24)
    build_table_texture(MJ_DIR / "table_tex.png")
    cam_pos, R_wc, K_front = compute_front_camera_pose()
    build_mjcf(jaw_hulls, n_seg, cam_pos, R_wc, K_front)


if __name__ == "__main__":
    main()
