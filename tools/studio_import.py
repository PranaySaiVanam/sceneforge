"""Bring a studio's own motion capture into SceneForge.

Studio exports (Vicon Shogun, OptiTrack Motive, MotionBuilder / HumanIK, Xsens, Rokoko, Mixamo, Blender)
differ from the CMU files the library was first built from in four ways. This module handles all of them:

  joint names  Character1_Spine2, mixamorig:LeftForeArm, Bip01 L Hand, hand_l ... are mapped onto the
               20-joint SceneForge skeleton by the *structure* of the hierarchy (the hand is the shallowest
               "hand"/"wrist" joint on that side, its parent is the forearm, the chest is where both arms
               meet, the neck is the head's parent) plus left/right words. A JSON joint map can override
               any of it.
  units        detected from the performer's standing height (cm, mm, m or inches), or forced.
  up axis      Z-up files are turned Y-up.
  frame rate   any rate (60, 100, 119.88, 120 ...) is resampled to 30 fps by interpolation.

Use it two ways:

  python tools/studio_import.py inspect take.bvh
      prints the detected mapping, units, up axis, frame rate and height, and checks the result.

  python tools/studio_import.py add take.bvh --id sword_slash --category combat \\
         --tags "sword slash, overhead strike" --desc "overhead sword slash" [--trim "1.0:3.5 abs"] [--extra "weapon=sword strike=1"]
      copies the take into data/studio/, adds a catalog row (cmu column = "studio:<file>"), and rebuilds
      the library (clips already built are reused, so the CMU download is not needed).
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_motion_library import KEEP, OUT_FPS, forward_kinematics, parse_bvh  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
STUDIO_DIR = ROOT / "data" / "studio"
UNITS = {"m": 1.0, "cm": 0.01, "mm": 0.001, "in": 0.0254, "inch": 0.0254, "inches": 0.0254}
PREFIX = re.compile(r"^(character\d*|mixamorig\d*|bip\d*|skeleton\d*|rig|def|armature|actor\d*|cc_base|b)$", re.I)


def tokens(name: str) -> list[str]:
    """'Character1_LeftForeArm' -> ['left', 'fore', 'arm']; 'Bip01 L Hand' -> ['l', 'hand']; 'hand_l' -> ['hand', 'l']."""
    name = name.split(":")[-1].split("|")[-1]
    parts = re.findall(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+", name)
    out = [p.lower() for p in parts]
    while out and (PREFIX.match(out[0]) or out[0].isdigit()):
        out = out[1:]
    return [t for t in out if not t.isdigit()]


def side_of(name: str) -> str | None:
    t = tokens(name)
    if "left" in t or "l" in t or "lt" in t:
        return "Left"
    if "right" in t or "r" in t or "rt" in t:
        return "Right"
    return None


def has(name: str, *words: str) -> bool:
    j = "".join(tokens(name))
    return any(w in j for w in words)


def map_joints(joints, override: dict | None = None) -> dict[str, int]:
    """SceneForge joint name -> index into the BVH joint list."""
    names = [j.name for j in joints]
    parent = [j.parent for j in joints]
    depth = []
    for i, p in enumerate(parent):
        depth.append(0 if p < 0 else depth[p] + 1)
    children = {i: [k for k, p in enumerate(parent) if p == i] for i in range(len(joints))}

    def ancestors(i):
        out = []
        while i >= 0:
            out.append(i)
            i = parent[i]
        return out

    def lca(a, b):
        sa = set(ancestors(a))
        for x in ancestors(b):
            if x in sa:
                return x
        return 0

    def shallowest(pred):
        c = [i for i, n in enumerate(names) if not n.endswith("_End") and pred(n)]
        return min(c, key=lambda i: depth[i]) if c else None

    m: dict[str, int] = {}
    for side in ("Left", "Right"):
        hand = shallowest(lambda n, s=side: side_of(n) == s and has(n, "hand", "wrist") and not has(n, "thumb", "index", "finger", "pinky", "ring", "middle"))
        foot = shallowest(lambda n, s=side: side_of(n) == s and has(n, "foot", "ankle"))
        if hand is None or foot is None:
            raise ValueError(f"could not find the {side.lower()} hand/foot; pass a joint map (--map)")
        m[f"{side}Hand"], m[f"{side}ForeArm"], m[f"{side}Arm"] = hand, parent[hand], parent[parent[hand]]
        m[f"{side}Foot"], m[f"{side}Leg"], m[f"{side}UpLeg"] = foot, parent[foot], parent[parent[foot]]
        toe = next((k for k in children[foot] if has(names[k], "toe", "ball")), None)
        m[f"{side}ToeBase"] = toe if toe is not None else (children[foot][0] if children[foot] else foot)
    m["Hips"] = lca(m["LeftUpLeg"], m["RightUpLeg"])
    m["Spine1"] = lca(m["LeftArm"], m["RightArm"])  # the chest: where both arms hang from
    head = shallowest(lambda n: has(n, "head") and not has(n, "top", "end", "nub", "site"))
    if head is None:
        raise ValueError("could not find the head; pass a joint map (--map)")
    m["Head"] = head
    m["Neck1"] = parent[head] if parent[head] != m["Spine1"] else head
    chain = []  # joints strictly between the hips and the chest
    i = parent[m["Spine1"]]
    while i >= 0 and i != m["Hips"]:
        chain.append(i)
        i = parent[i]
    m["Spine"] = chain[0] if chain else m["Spine1"]
    tip = next((k for k in children[head] if names[k].endswith("_End") or has(names[k], "top", "end", "nub")), None)
    m["Head_End"] = tip if tip is not None else -1  # -1: synthesised above the head
    for k, v in (override or {}).items():
        m[k] = names.index(v)
    return m


def detect_up(pos: np.ndarray, m: dict[str, int]) -> str:
    """'y' or 'z': which axis the spine points along in the first frames."""
    v = (pos[:10, m["Head"]] - pos[:10, m["Hips"]]).mean(0)
    return "z" if abs(v[2]) > abs(v[1]) and abs(v[2]) > abs(v[0]) else "y"


def resample(pos: np.ndarray, src_fps: float) -> np.ndarray:
    if abs(src_fps - OUT_FPS) < 1e-6:
        return pos
    step = src_fps / OUT_FPS
    if abs(step - round(step)) < 1e-6:
        return pos[::int(round(step))]
    t = np.arange(0, len(pos) - 1, step)
    i0 = np.floor(t).astype(int)
    a = (t - i0)[:, None, None]
    return pos[i0] * (1 - a) + pos[np.minimum(i0 + 1, len(pos) - 1)] * a


def load(path: Path, units: str | None = None, up: str | None = None, fps: float | None = None,
         joint_map: dict | None = None, report: dict | None = None) -> np.ndarray:
    """A studio BVH as (F, 20, 3) positions in metres, Y up, 30 fps, in SceneForge joint order."""
    joints, data, frame_time = parse_bvh(path)
    src_fps = fps or round(1.0 / frame_time, 3)
    raw = forward_kinematics(joints, data)
    m = map_joints(joints, joint_map)
    axis = (up or detect_up(raw, m)).lower()
    if axis == "z":  # Z-up -> Y-up: (x, y, z) -> (x, z, -y)
        raw = np.stack([raw[..., 0], raw[..., 2], -raw[..., 1]], -1)
    keep = []
    for n in KEEP:
        if n == "Head_End" and m["Head_End"] < 0:
            h, nk = raw[:, m["Head"]], raw[:, m["Neck1"]]
            d = h - nk
            keep.append(h + d / np.maximum(np.linalg.norm(d, axis=-1, keepdims=True), 1e-9) * np.linalg.norm(d, axis=-1, keepdims=True).mean() * 1.1)
        else:
            keep.append(raw[:, m[n]])
    pos = np.stack(keep, 1)
    feet = [KEEP.index(n) for n in ("LeftFoot", "RightFoot", "LeftToeBase", "RightToeBase")]
    height_units = float(np.median(pos[:, KEEP.index("Head_End"), 1] - pos[:, feet, 1].min(1)))
    if units:
        scale = UNITS[units.lower()]
    else:  # pick the unit that makes the performer a plausible human height
        scale = min(UNITS.values(), key=lambda s: abs(height_units * s - 1.72))
        if not 1.2 <= height_units * scale <= 2.3:
            scale = 1.72 / height_units  # odd units: normalise to an average height
    pos = resample(pos * scale, src_fps)
    if report is not None:
        report.update({"frames": int(len(data)), "fps": src_fps, "up": axis, "scale_to_m": scale,
                       "height_m": round(height_units * scale, 3), "seconds": round(len(data) / src_fps, 2),
                       "mapping": {k: (joints[v].name if v >= 0 else "(synthesised above the head)") for k, v in m.items()}})
    return pos


def check(pos: np.ndarray) -> list[str]:
    """Sanity checks on an imported take; returns warnings."""
    k = {n: i for i, n in enumerate(KEEP)}
    warn = []
    hips_y = float(np.median(pos[:, k["Hips"], 1] - pos[:, [k["LeftFoot"], k["RightFoot"]], 1].min(1)))
    if not 0.7 <= hips_y <= 1.25:
        warn.append(f"hip height {hips_y:.2f} m looks wrong (units or up axis?)")
    left = pos[:5, k["LeftUpLeg"]] - pos[:5, k["RightUpLeg"]]
    arm = pos[:5, k["LeftArm"]] - pos[:5, k["RightArm"]]
    if float((left * arm).sum(-1).mean()) < 0:
        warn.append("left and right look swapped between legs and arms; check the joint map")
    for a, b in (("LeftUpLeg", "LeftLeg"), ("LeftArm", "LeftForeArm"), ("Spine1", "Neck1")):
        L = np.linalg.norm(pos[:, k[a]] - pos[:, k[b]], axis=-1)
        if L.std() > 0.02:
            warn.append(f"bone {a}-{b} changes length by {L.std() * 100:.1f} cm (mapping may skip a joint)")
    return warn


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("inspect", "add"):
        p = sub.add_parser(name)
        p.add_argument("bvh", type=Path)
        p.add_argument("--units", choices=sorted(UNITS))
        p.add_argument("--up", choices=["y", "z"])
        p.add_argument("--fps", type=float)
        p.add_argument("--map", type=Path, help="JSON {SceneForgeJoint: studioJointName} overrides")
        if name == "add":
            p.add_argument("--id", required=True)
            p.add_argument("--category", required=True)
            p.add_argument("--tags", required=True)
            p.add_argument("--desc", required=True)
            p.add_argument("--trim", default="")
            p.add_argument("--extra", default="", help="e.g. 'weapon=sword strike=1 tier=1'")
            p.add_argument("--no-build", action="store_true")
    a = ap.parse_args()
    jm = json.loads(a.map.read_text()) if a.map else None
    rep: dict = {}
    pos = load(a.bvh, a.units, a.up, a.fps, jm, rep)
    print(f"{a.bvh.name}: {rep['frames']} frames at {rep['fps']} fps ({rep['seconds']} s), "
          f"{rep['up'].upper()}-up, x{rep['scale_to_m']} to metres, performer {rep['height_m']} m")
    for k, v in rep["mapping"].items():
        print(f"  {k:<13} <- {v}")
    warns = check(pos)
    for w in warns:
        print("  WARNING:", w)
    print("  OK: looks like a clean humanoid take" if not warns else "")
    if a.cmd == "inspect":
        return
    STUDIO_DIR.mkdir(parents=True, exist_ok=True)
    dest = STUDIO_DIR / a.bvh.name
    if a.bvh.resolve() != dest.resolve():
        shutil.copy2(a.bvh, dest)
    opts = [f"units={a.units}"] if a.units else []
    opts += [f"up={a.up}"] if a.up else []
    opts += [f"fps={a.fps:g}"] if a.fps else []
    if a.map:
        shutil.copy2(a.map, STUDIO_DIR / a.map.name)
        opts.append(f"map={a.map.name}")
    extra = " ".join(x for x in [a.extra.strip(), "source=studio", *opts] if x)
    if "tier=" not in extra:
        extra += " tier=2"
    catalog = ROOT / "data" / "clip_catalog.tsv"
    with catalog.open() as fh:
        ids = {row["id"] for row in csv.DictReader(fh, delimiter="\t")}
    if a.id in ids:
        sys.exit(f"clip id '{a.id}' already exists in the catalog")
    with catalog.open("a") as fh:
        fh.write("\t".join([a.id, f"studio:{dest.name}", a.category, a.tags, a.desc, a.trim, extra]) + "\n")
    print(f"added '{a.id}' to {catalog.relative_to(ROOT)}")
    if not a.no_build:
        subprocess.run([sys.executable, str(ROOT / "tools" / "build_motion_library.py")], check=True)


if __name__ == "__main__":
    main()
