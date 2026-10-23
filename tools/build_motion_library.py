"""Build SceneForge's motion library from CMU BVH files.

Pipeline per clip:
  1. Parse the BVH hierarchy and motion channels.
  2. Run forward kinematics to get world-space joint positions.
  3. Resample 120 fps -> 30 fps and keep the most active window.
  4. Normalize root motion: start at origin, facing +Z, feet on the floor.
  5. Compute blocking metadata (end displacement, heading change, speed).

Output: data/motions/<id>.json (positions in millimetres, int-packed) and
data/motion_library.json (catalog + metadata used by the agent).

Usage:
  python tools/build_motion_library.py --bvh-root /path/to/cmu-mocap/data
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

UNIT_TO_M = 0.0745  # CMU BVH (Hahne release) units -> metres (calibrated: ~1.7 m standing height)
SRC_FPS = 120
OUT_FPS = 30

# Joints kept for rendering (subset of the 31-joint CMU skeleton) + head tip.
KEEP = [
    "Hips", "LeftUpLeg", "LeftLeg", "LeftFoot", "LeftToeBase",
    "RightUpLeg", "RightLeg", "RightFoot", "RightToeBase",
    "Spine", "Spine1", "Neck1", "Head", "Head_End",
    "LeftArm", "LeftForeArm", "LeftHand",
    "RightArm", "RightForeArm", "RightHand",
]
# Bones drawn by the viewer (parent, child) as indices into KEEP.
BONES = [
    ("Hips", "LeftUpLeg"), ("LeftUpLeg", "LeftLeg"), ("LeftLeg", "LeftFoot"), ("LeftFoot", "LeftToeBase"),
    ("Hips", "RightUpLeg"), ("RightUpLeg", "RightLeg"), ("RightLeg", "RightFoot"), ("RightFoot", "RightToeBase"),
    ("Hips", "Spine"), ("Spine", "Spine1"), ("Spine1", "Neck1"), ("Neck1", "Head"), ("Head", "Head_End"),
    ("Spine1", "LeftArm"), ("LeftArm", "LeftForeArm"), ("LeftForeArm", "LeftHand"),
    ("Spine1", "RightArm"), ("RightArm", "RightForeArm"), ("RightForeArm", "RightHand"),
]
# The Hahne BVH release stamps every file at 120 fps, but some later CMU
# sessions were captured at 60 fps. Verified by gait analysis: at 120 fps
# subject 143's "Walk" has a 1.6 Hz stride and 1.9 m/s pace (a jog); at 60 fps
# it is a normal 0.8 Hz / 0.9 m/s walk.
TRUE_FPS = {141: 60, 143: 60, 77: 60}
MAX_SECONDS = {"locomotion": 6.0, "idle": 6.0}
DEFAULT_MAX_SECONDS = 8.0


# ----------------------------------------------------------------------------
# BVH parsing
# ----------------------------------------------------------------------------
class Joint:
    def __init__(self, name: str, parent: int):
        self.name = name
        self.parent = parent
        self.offset = np.zeros(3)
        self.channels: list[str] = []
        self.chan_start = 0


def parse_bvh(path: Path):
    tokens = path.read_text().split()
    i = 0
    joints: list[Joint] = []
    stack: list[int] = []
    n_channels = 0
    while tokens[i] != "MOTION":
        t = tokens[i]
        if t in ("ROOT", "JOINT"):
            joints.append(Joint(tokens[i + 1], stack[-1] if stack else -1))
            i += 2
        elif t == "End":  # End Site -> synthetic leaf joint named <parent>_End
            joints.append(Joint(joints[stack[-1]].name + "_End", stack[-1]))
            i += 2
        elif t == "{":
            stack.append(len(joints) - 1)
            i += 1
        elif t == "}":
            stack.pop()
            i += 1
        elif t == "OFFSET":
            joints[stack[-1]].offset = np.array([float(x) for x in tokens[i + 1:i + 4]])
            i += 4
        elif t == "CHANNELS":
            n = int(tokens[i + 1])
            j = joints[stack[-1]]
            j.channels = tokens[i + 2:i + 2 + n]
            j.chan_start = n_channels
            n_channels += n
            i += 2 + n
        else:
            i += 1
    i += 1  # MOTION
    n_frames = int(tokens[i + 1])
    i += 2
    frame_time = float(tokens[i + 2])
    i += 3
    data = np.array(tokens[i:i + n_frames * n_channels], dtype=np.float64)
    data = data.reshape(n_frames, n_channels)
    return joints, data, frame_time


def rot(axis: str, deg: np.ndarray) -> np.ndarray:
    """Batched rotation matrices (F,3,3) about a principal axis."""
    r = np.radians(deg)
    c, s = np.cos(r), np.sin(r)
    o, z = np.ones_like(r), np.zeros_like(r)
    if axis == "X":
        m = [[o, z, z], [z, c, -s], [z, s, c]]
    elif axis == "Y":
        m = [[c, z, s], [z, o, z], [-s, z, c]]
    else:
        m = [[c, -s, z], [s, c, z], [z, z, o]]
    return np.stack([np.stack(row, -1) for row in m], -2)


def forward_kinematics(joints: list[Joint], data: np.ndarray) -> np.ndarray:
    """World positions (F, J, 3) for every joint, in BVH units."""
    F = data.shape[0]
    world_rot = [None] * len(joints)
    world_pos = np.zeros((F, len(joints), 3))
    for ji, j in enumerate(joints):
        local_rot = np.broadcast_to(np.eye(3), (F, 3, 3)).copy()
        local_pos = np.broadcast_to(j.offset, (F, 3)).copy()
        for k, ch in enumerate(j.channels):
            col = data[:, j.chan_start + k]
            if ch.endswith("position"):
                local_pos[:, "XYZ".index(ch[0])] += col
            else:  # rotations compose in listed order (intrinsic)
                local_rot = local_rot @ rot(ch[0], col)
        if j.parent < 0:
            world_rot[ji] = local_rot
            world_pos[:, ji] = local_pos
        else:
            pr = world_rot[j.parent]
            world_pos[:, ji] = world_pos[:, j.parent] + np.einsum("fij,fj->fi", pr, local_pos)
            world_rot[ji] = pr @ local_rot
    return world_pos


# ----------------------------------------------------------------------------
# Clip processing
# ----------------------------------------------------------------------------
def heading(pos: np.ndarray, idx: dict[str, int]) -> np.ndarray:
    """Per-frame yaw (radians, 0 = facing +Z) from the hip axis."""
    across = pos[:, idx["LeftUpLeg"]] - pos[:, idx["RightUpLeg"]]
    fx, fz = -across[:, 2], across[:, 0]
    return np.arctan2(fx, fz)


def most_active_window(pos: np.ndarray, n: int) -> int:
    if pos.shape[0] <= n:
        return 0
    vel = np.linalg.norm(np.diff(pos, axis=0), axis=-1).sum(-1)
    energy = np.convolve(vel, np.ones(n - 1), mode="valid")
    return int(np.argmax(energy))


def loop_window(pos: np.ndarray, idx: dict[str, int]) -> tuple[int, int]:
    """For locomotion: the straight, seamlessly loopable window.

    Scores every (start, length) window: reward net travel, penalise heading
    change and the pose mismatch between the first and last frame (so the
    cycle can repeat without a pop).
    """
    yaw = np.unwrap(heading(pos, idx))
    local = pos - pos[:, :1]  # root-relative pose
    best, best_score = (0, min(len(pos), 90)), -1e9
    for n in range(int(1.6 * OUT_FPS), int(3.6 * OUT_FPS), 2):
        for s in range(0, len(pos) - n, 2):
            e = s + n - 1
            travel = float(np.linalg.norm(pos[e, 0, [0, 2]] - pos[s, 0, [0, 2]]))
            dyaw = abs(float(yaw[e] - yaw[s]))
            wobble = float(np.ptp(yaw[s:e + 1]))
            mismatch = float(np.abs(local[e] - local[s]).mean())
            score = travel / (n / OUT_FPS) - 2.0 * dyaw - 0.8 * wobble - 12.0 * mismatch
            if score > best_score:
                best, best_score = (s, n), score
    return best


def wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def process_clip(bvh: Path, category: str, trim: str = ""):
    joints, data, frame_time = parse_bvh(bvh)
    subject = int(bvh.stem.split("_")[0])
    src_fps = TRUE_FPS.get(subject, round(1.0 / frame_time))
    step = max(1, round(src_fps / OUT_FPS))
    data = data[int(src_fps * 0.25)::step]  # drop the T-pose frame(s) at the start
    pos = forward_kinematics(joints, data) * UNIT_TO_M
    idx = {j.name: k for k, j in enumerate(joints)}
    pos = pos[:, [idx[n] for n in KEEP]]
    kidx = {n: k for k, n in enumerate(KEEP)}

    if category == "locomotion" and len(pos) > 2 * OUT_FPS:
        start, n = loop_window(pos, kidx)
        pos = pos[start:start + n]
    else:
        n_max = int(MAX_SECONDS.get(category, DEFAULT_MAX_SECONDS) * OUT_FPS)
        start = most_active_window(pos, n_max)
        pos = pos[start:start + n_max]
    if trim:  # "a:b" seconds within the chosen window, to start a clip right at its action
        a, _, b = trim.partition(":")
        pos = pos[int(float(a or 0) * OUT_FPS): int(float(b) * OUT_FPS) if b else None]
    return finalize(pos)


def finalize(pos: np.ndarray):
    """Normalize a (F, J, 3) clip in KEEP order and compute its blocking metadata."""
    kidx = {n: k for k, n in enumerate(KEEP)}
    # Normalize: root starts at origin facing +Z, feet on the floor.
    yaw = np.unwrap(heading(pos, kidx))
    yaw0 = float(np.median(yaw[: min(5, len(yaw))]))
    pos = pos - np.array([pos[0, 0, 0], 0.0, pos[0, 0, 2]])
    c, s = math.cos(-yaw0), math.sin(-yaw0)
    x, z = pos[..., 0].copy(), pos[..., 2].copy()
    pos[..., 0] = x * c + z * s
    pos[..., 2] = -x * s + z * c
    feet = [kidx[n] for n in ("LeftFoot", "LeftToeBase", "RightFoot", "RightToeBase")]
    floor = np.percentile(pos[:, feet, 1].min(axis=1), 5)
    pos[..., 1] -= floor

    yaw = np.unwrap(heading(pos, kidx))
    end = pos[-1, 0]
    dur = (len(pos) - 1) / OUT_FPS
    disp = math.hypot(end[0], end[2])
    meta = {
        "frames": int(len(pos)),
        "duration": round(dur, 3),
        "end_disp": [round(float(end[0]), 3), round(float(end[2]), 3)],
        "end_yaw": round(wrap(float(np.median(yaw[-5:]) - np.median(yaw[:5]))), 3),
        "avg_speed": round(disp / max(dur, 1e-6), 3),
        "height": round(float(np.percentile(pos[:, kidx["Head_End"], 1], 50)), 3),
    }
    packed = np.round(pos * 1000).astype(np.int32).reshape(len(pos), -1).tolist()
    return meta, packed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bvh-root", required=True, type=Path)
    ap.add_argument("--catalog", default=Path(__file__).resolve().parents[1] / "data" / "clip_catalog.tsv", type=Path)
    ap.add_argument("--out", default=Path(__file__).resolve().parents[1] / "data", type=Path)
    args = ap.parse_args()

    motions_dir = args.out / "motions"
    motions_dir.mkdir(parents=True, exist_ok=True)
    library = {"fps": OUT_FPS, "joints": KEEP,
               "bones": [[KEEP.index(a), KEEP.index(b)] for a, b in BONES],
               "source": "CMU Graphics Lab Motion Capture Database (mocap.cs.cmu.edu), BVH conversion by B. Hahne",
               "clips": []}
    reference = None  # a standing pose for keyframed clips (first frame of the first idle clip)
    with args.catalog.open() as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            if row["cmu"].startswith("keyframed:"):
                sys.path.insert(0, str(Path(__file__).resolve().parent))
                from synth_motion import GENERATORS  # tools/synth_motion.py
                gen = GENERATORS[row["cmu"].split(":", 1)[1]]
                meta, packed = finalize(gen(reference, KEEP, BONES))
                meta["source"] = "keyframed"
            else:
                subj = int(row["cmu"].split("_")[0])
                bvh = args.bvh_root / f"{subj:03d}" / f"{row['cmu']}.bvh"
                meta, packed = process_clip(bvh, row["category"], (row.get("trim") or "").strip())
                meta["source"] = "mocap"
                if reference is None and row["category"] == "idle":
                    reference = np.array(packed[0], dtype=float).reshape(-1, 3) / 1000
            (motions_dir / f"{row['id']}.json").write_text(
                json.dumps({"id": row["id"], "fps": OUT_FPS, "frames": packed}, separators=(",", ":")))
            library["clips"].append({
                "id": row["id"], "cmu": row["cmu"], "category": row["category"],
                "tags": [t.strip() for t in row["tags"].split(",") if t.strip()],
                "description": row["description"], **meta,
            })
            print(f"{row['id']:<14} {row['cmu']:<7} {meta['duration']:>5.1f}s "
                  f"disp={meta['end_disp']} yaw={meta['end_yaw']:+.2f} h={meta['height']:.2f}m")
    (args.out / "motion_library.json").write_text(json.dumps(library, indent=1))
    print(f"\n{len(library['clips'])} clips -> {args.out / 'motion_library.json'}")


if __name__ == "__main__":
    main()
