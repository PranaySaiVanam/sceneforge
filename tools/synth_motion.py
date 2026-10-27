"""Keyframed motions for actions the CMU database does not contain (for example floor pushups).

Each generator builds joint positions directly on the CMU skeleton's own bone lengths,
measured from a reference standing pose, so the result plays with the same mannequin,
blending and blocking as the mocap clips. Clips made here are tagged source="keyframed".
"""
from __future__ import annotations

import numpy as np

FPS = 30


def _bone_lengths(ref: np.ndarray, joints: list[str], bones: list[tuple[str, str]]) -> dict:
    j = {n: i for i, n in enumerate(joints)}
    return {(a, b): float(np.linalg.norm(ref[j[b]] - ref[j[a]])) for a, b in bones}


def _fix_lengths(pose: np.ndarray, joints, bones, lengths) -> np.ndarray:
    """Restore every bone's length along its current direction (parents before children)."""
    j = {n: i for i, n in enumerate(joints)}
    out = pose.copy()
    for a, b in bones:
        d = out[j[b]] - out[j[a]]
        n = np.linalg.norm(d)
        if n > 1e-6:
            out[j[b]] = out[j[a]] + d / n * lengths[(a, b)]
    return out


def _ease(u: float) -> float:
    return u * u * (3 - 2 * u)


def _two_bone(s: np.ndarray, h: np.ndarray, a: float, b: float, pole: np.ndarray) -> np.ndarray:
    """Elbow position for shoulder s, hand h, upper-arm a, forearm b, bending toward pole."""
    d_vec = h - s
    d = min(np.linalg.norm(d_vec), a + b - 1e-4)
    axis = d_vec / max(np.linalg.norm(d_vec), 1e-6)
    x = (a * a - b * b + d * d) / (2 * d)
    y = np.sqrt(max(a * a - x * x, 0.0))
    n = pole - axis * np.dot(pole, axis)
    n /= max(np.linalg.norm(n), 1e-6)
    return s + axis * x + n * y


def pushups(ref: np.ndarray, joints: list[str], bones: list[tuple[str, str]], reps: int = 5) -> np.ndarray:  # noqa: E501
    """Stand, get down into a plank, do `reps` pushups, stand back up. Returns (F, J, 3)."""
    j = {n: i for i, n in enumerate(joints)}
    L = _bone_lengths(ref, joints, bones)
    foot = L[("LeftFoot", "LeftToeBase")]
    shin = L[("LeftLeg", "LeftFoot")]
    thigh = L[("LeftUpLeg", "LeftLeg")]
    upper = L[("LeftArm", "LeftForeArm")]
    fore = L[("LeftForeArm", "LeftHand")]
    hip_half = abs(ref[j["LeftUpLeg"], 0] - ref[j["RightUpLeg"], 0]) / 2
    sh_half = abs(ref[j["LeftArm"], 0] - ref[j["RightArm"], 0]) / 2
    # distances along the body axis, measured on the standing reference (vertical axis)
    y = lambda n: float(ref[j[n], 1])  # noqa: E731
    up_hip = y("Hips") - y("LeftUpLeg")
    seg = {"Spine": y("Spine") - y("Hips"), "Spine1": y("Spine1") - y("Spine"),
           "Neck1": y("Neck1") - y("Spine1"), "Head": y("Head") - y("Neck1"), "Head_End": y("Head_End") - y("Head"),
           "Arm": y("LeftArm") - y("Spine1")}
    toe_y = 0.03
    to_hips = foot + shin + thigh + up_hip
    to_shoulder = to_hips + seg["Spine"] + seg["Spine1"] + seg["Arm"]
    h_up = (upper + fore) * 0.97 + toe_y
    h_down = 0.24
    z_toe = -to_hips * 0.98  # hips end up roughly above the start spot

    def plank(h: float) -> np.ndarray:
        """Plank with shoulders at height h, toes fixed, hands planted under the 'up' shoulders."""
        ang = np.arcsin((h - toe_y) / to_shoulder)
        axis = np.array([0.0, np.sin(ang), np.cos(ang)])
        at = lambda dist: np.array([0.0, toe_y, z_toe]) + axis * dist  # noqa: E731
        p = np.zeros((len(joints), 3))
        for side, sx in (("Left", 1.0), ("Right", -1.0)):
            lat = np.array([sx * hip_half, 0, 0])
            p[j[f"{side}ToeBase"]] = at(0) + lat
            p[j[f"{side}Foot"]] = at(foot) + lat
            p[j[f"{side}Leg"]] = at(foot + shin) + lat
            p[j[f"{side}UpLeg"]] = at(foot + shin + thigh) + lat
        p[j["Hips"]] = at(to_hips)
        d = to_hips
        for n in ("Spine", "Spine1"):
            d += seg[n]
            p[j[n]] = at(d)
        p[j["Neck1"]] = at(d + seg["Neck1"]) + np.array([0, 0.02, 0])
        head = at(d + seg["Neck1"] + seg["Head"]) + np.array([0, 0.05, 0])
        p[j["Head"]] = head
        p[j["Head_End"]] = head + (axis * 0.6 + np.array([0, 0.4, 0])) / np.linalg.norm(axis * 0.6 + np.array([0, 0.4, 0])) * seg["Head_End"]
        z_hand = z_toe + np.sqrt(to_shoulder ** 2 - (h_up - toe_y) ** 2)
        for side, sx in (("Left", 1.0), ("Right", -1.0)):
            s = at(to_shoulder) + np.array([sx * sh_half, 0, 0])
            hand = np.array([sx * (sh_half + 0.09), toe_y, z_hand])
            pole = np.array([sx * 0.45, 0.0, -0.9])  # elbows tucked back at about 45 degrees
            p[j[f"{side}Arm"]] = s
            p[j[f"{side}ForeArm"]] = _two_bone(s, hand, upper, fore, pole)
            p[j[f"{side}Hand"]] = hand
        return p

    stand = ref.copy()
    top = plank(h_up)
    frames = []

    def morph(a, b, secs):
        n = int(secs * FPS)
        for i in range(n):
            u = _ease(i / max(n - 1, 1))
            frames.append(_fix_lengths(a * (1 - u) + b * u, joints, bones, L))

    morph(stand, top, 1.0)
    for _ in range(reps):
        for i in range(int(0.65 * FPS)):  # lower
            frames.append(plank(h_up + (h_down - h_up) * _ease(i / (0.65 * FPS - 1))))
        frames += [plank(h_down)] * int(0.1 * FPS)
        for i in range(int(0.6 * FPS)):  # press
            frames.append(plank(h_down + (h_up - h_down) * _ease(i / (0.6 * FPS - 1))))
        frames += [top] * int(0.1 * FPS)
    morph(top, stand, 1.0)
    return np.stack(frames)


GENERATORS = {"pushups": pushups}


# =====================================================================================
# A small IK rig on the CMU skeleton for keyframed human actions (hit reactions, block,
# nod, crouch, button, door, salute, clap, point). Body-local coordinates: hips at the
# origin of the floor plan, facing +Z, +X is the character's left.
# =====================================================================================
def _rot(axis: str, a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _track(keys, t):
    """Smooth piecewise interpolation of (time, value) keys; values may be scalars or vectors."""
    if t <= keys[0][0]:
        return np.asarray(keys[0][1], dtype=float)
    for (t0, v0), (t1, v1) in zip(keys, keys[1:]):
        if t <= t1:
            u = _ease((t - t0) / max(t1 - t0, 1e-6))
            return np.asarray(v0, dtype=float) * (1 - u) + np.asarray(v1, dtype=float) * u
    return np.asarray(keys[-1][1], dtype=float)


class HumanRig:
    UPPER = ["Spine", "Spine1", "Neck1", "Head", "Head_End", "LeftArm", "LeftForeArm", "LeftHand",
             "RightArm", "RightForeArm", "RightHand"]

    def __init__(self, ref: np.ndarray, joints: list[str], bones):
        self.joints = joints
        self.j = {n: i for i, n in enumerate(joints)}
        L = _bone_lengths(ref, joints, bones)
        self.L = L
        j = self.j
        r = ref - np.array([ref[j["Hips"], 0], 0, ref[j["Hips"], 2]])
        n = np.zeros_like(r)
        # symmetric pelvis and spine
        hip_off = (np.abs(r[j["LeftUpLeg"]] - r[j["Hips"]]) + np.abs(r[j["RightUpLeg"]] - r[j["Hips"]])) / 2
        thigh = (L[("LeftUpLeg", "LeftLeg")] + L[("RightUpLeg", "RightLeg")]) / 2
        shin = (L[("LeftLeg", "LeftFoot")] + L[("RightLeg", "RightFoot")]) / 2
        foot_y = 0.08
        # hip sockets sit below and slightly in front of the Hips joint
        sock_dy = -hip_off[1]
        hip_y = foot_y + 0.985 * (thigh + shin)
        n[j["Hips"]] = [0, hip_y - sock_dy, 0]  # socket height = hip_y
        for side, sx in (("Left", 1), ("Right", -1)):
            hipj = n[j["Hips"]] + np.array([sx * hip_off[0], sock_dy, hip_off[2]])
            foot = np.array([sx * hip_off[0], foot_y, hip_off[2] - 0.02])
            n[j[f"{side}UpLeg"]] = hipj
            n[j[f"{side}Leg"]] = _two_bone(hipj, foot, thigh, shin, np.array([0, 0, 1.0]))
            n[j[f"{side}Foot"]] = foot
            toe_len = L[(f"{side}Foot", f"{side}ToeBase")]
            n[j[f"{side}ToeBase"]] = foot + np.array([0, -0.05, np.sqrt(max(toe_len ** 2 - 0.0025, 1e-4))])
        for a, b in (("Hips", "Spine"), ("Spine", "Spine1"), ("Spine1", "Neck1"), ("Neck1", "Head"), ("Head", "Head_End")):
            d = r[j[b]] - r[j[a]]
            d[0] = 0
            d = d / max(np.linalg.norm(d), 1e-6) * L[(a, b)]
            n[j[b]] = n[j[a]] + d
        sh = (np.abs(r[j["LeftArm"]] - r[j["Spine1"]]) + np.abs(r[j["RightArm"]] - r[j["Spine1"]])) / 2
        up = (L[("LeftArm", "LeftForeArm")] + L[("RightArm", "RightForeArm")]) / 2
        fo = (L[("LeftForeArm", "LeftHand")] + L[("RightForeArm", "RightHand")]) / 2
        for side, sx in (("Left", 1), ("Right", -1)):
            s = n[j["Spine1"]] + np.array([sx * sh[0], r[j[f"{side}Arm"], 1] - r[j["Spine1"], 1], 0.0])
            e = s + np.array([sx * 0.05, -up * 0.995, -0.03])
            h = e + np.array([sx * 0.02, -fo * 0.97, 0.10])
            n[j[f"{side}Arm"]] = s
            n[j[f"{side}ForeArm"]] = _fix(e, s, up)
            n[j[f"{side}Hand"]] = _fix(h, n[j[f"{side}ForeArm"]], fo)
        self.rest = n
        self.up, self.fo, self.thigh, self.shin = up, fo, thigh, shin

    def hand_rest(self, side: str) -> np.ndarray:
        return self.rest[self.j[f"{side}Hand"]].copy()

    def pose(self, root=(0, 0, 0), yaw=0.0, pitch=0.0, roll=0.0, twist=0.0, head_pitch=0.0, head_yaw=0.0,
             lhand=None, rhand=None, lfoot=None, rfoot=None) -> np.ndarray:
        j, p = self.j, self.rest.copy()
        hips = p[j["Hips"]].copy()
        Rt = _rot("y", twist) @ _rot("z", roll) @ _rot("x", pitch)
        for n in self.UPPER:
            p[j[n]] = hips + Rt @ (p[j[n]] - hips)
        Rh = Rt @ _rot("y", head_yaw) @ _rot("x", head_pitch) @ Rt.T
        neck = p[j["Neck1"]].copy()
        for n in ("Head", "Head_End"):
            p[j[n]] = neck + Rh @ (p[j[n]] - neck)
        # arms by IK in body space (before global yaw)
        for side, sx, tgt in (("Left", 1, lhand), ("Right", -1, rhand)):
            if tgt is None:
                continue
            s = p[j[f"{side}Arm"]]
            pole = Rt @ np.array([sx * 0.55, -0.55, -0.45])
            h = np.asarray(tgt, dtype=float)
            p[j[f"{side}ForeArm"]] = _two_bone(s, h, self.up, self.fo, pole)
            p[j[f"{side}Hand"]] = _fix(h, p[j[f"{side}ForeArm"]], self.fo)
        # global yaw and root offset; legs re-solved so feet stay planted unless targeted
        Rg = _rot("y", yaw)
        off = np.asarray(root, dtype=float)
        rest_feet = {s: (p[j[f"{s}Foot"]].copy(), p[j[f"{s}ToeBase"]].copy()) for s in ("Left", "Right")}
        out = (Rg @ (p - np.array([0, 0, 0])).T).T + off
        for side, tgt in (("Left", lfoot), ("Right", rfoot)):
            foot0, toe0 = rest_feet[side]
            if tgt is None:
                foot = Rg @ foot0 + np.array([off[0], 0, off[2]])
            else:
                foot = np.asarray(tgt, dtype=float)
            hipj = out[j[f"{side}UpLeg"]]
            out[j[f"{side}Leg"]] = _two_bone(hipj, foot, self.thigh, self.shin, Rg @ np.array([0, 0.1, 1.0]))
            out[j[f"{side}Foot"]] = _fix(foot, out[j[f"{side}Leg"]], self.shin)
            out[j[f"{side}ToeBase"]] = out[j[f"{side}Foot"]] + Rg @ (toe0 - foot0)
        if getattr(self, "gaze", False):
            # newer clips tip Head_End toward the gaze so the viewer can draw where the actor looks
            fwd = Rg @ Rt @ _rot("y", head_yaw) @ _rot("x", head_pitch) @ np.array([0, 0, 1.0])
            h, he = out[j["Head"]], out[j["Head_End"]]
            L = np.linalg.norm(he - h)
            d = (he - h) / max(L, 1e-6) + fwd * 0.3
            out[j["Head_End"]] = h + d / np.linalg.norm(d) * L
        return out


def _fix(target: np.ndarray, parent: np.ndarray, length: float) -> np.ndarray:
    d = target - parent
    return parent + d / max(np.linalg.norm(d), 1e-6) * length


def _human(ref, joints, bones, seconds: float, fn) -> np.ndarray:
    rig = HumanRig(ref, joints, bones)
    n = int(round(seconds * FPS)) + 1
    return np.stack([fn(rig, i / FPS) for i in range(n)])


def _arc(t, t0, t1, h):
    if t <= t0 or t >= t1:
        return 0.0
    return h * np.sin(np.pi * (t - t0) / (t1 - t0))


def hit_react(ref, joints, bones):
    def f(rig, t):
        lr, rr = rig.hand_rest("Left"), rig.hand_rest("Right")
        return rig.pose(
            root=_track([(0, (0, 0, 0)), (0.14, (0, -0.04, -0.14)), (0.55, (0, -0.06, -0.34)), (1.4, (0, 0, -0.36))], t),
            pitch=float(_track([(0, 0), (0.12, -0.42), (0.55, -0.12), (1.4, 0)], t)),
            twist=float(_track([(0, 0), (0.15, 0.25), (0.7, 0.05), (1.4, 0)], t)),
            head_pitch=float(_track([(0, 0), (0.1, -0.55), (0.6, -0.1), (1.4, 0)], t)),
            lhand=_track([(0, lr), (0.16, (0.38, 1.38, 0.05)), (0.9, lr), (1.4, lr)], t),
            rhand=_track([(0, rr), (0.16, (-0.40, 1.30, 0.10)), (0.9, rr), (1.4, rr)], t),
            rfoot=np.asarray(_track([(0, (-0.11, 0.08, 0.08)), (0.15, (-0.11, 0.08, 0.08)),
                                     (0.5, (-0.13, 0.08, -0.42)), (1.4, (-0.11, 0.08, -0.28))], t))
            + np.array([0, _arc(t, 0.15, 0.5, 0.12), 0]),
            lfoot=np.asarray(_track([(0, (0.11, 0.08, 0.08)), (0.6, (0.11, 0.08, 0.08)), (1.0, (0.11, 0.08, -0.28)),
                                     (1.4, (0.11, 0.08, -0.28))], t)) + np.array([0, _arc(t, 0.6, 1.0, 0.08), 0]))
    return _human(ref, joints, bones, 1.4, f)


def hit_gut(ref, joints, bones):
    def f(rig, t):
        lr, rr = rig.hand_rest("Left"), rig.hand_rest("Right")
        return rig.pose(
            root=_track([(0, (0, 0, 0)), (0.18, (0, -0.12, -0.16)), (0.9, (0, -0.1, -0.2)), (1.6, (0, 0, -0.2))], t),
            pitch=float(_track([(0, 0), (0.18, 0.6), (0.9, 0.45), (1.6, 0)], t)),
            head_pitch=float(_track([(0, 0), (0.2, 0.25), (1.6, 0)], t)),
            lhand=_track([(0, lr), (0.2, (0.08, 1.0, 0.2)), (1.0, (0.08, 1.0, 0.2)), (1.6, lr)], t),
            rhand=_track([(0, rr), (0.2, (-0.08, 0.98, 0.22)), (1.0, (-0.08, 0.98, 0.22)), (1.6, rr)], t))
    return _human(ref, joints, bones, 1.6, f)


def block(ref, joints, bones):
    def f(rig, t):
        lr, rr = rig.hand_rest("Left"), rig.hand_rest("Right")
        k = [(0, 0), (0.25, 1), (1.0, 1), (1.4, 0)]
        u = float(_track(k, t))
        return rig.pose(
            root=(0, -0.09 * u, -0.12 * u), pitch=0.14 * u, head_pitch=0.12 * u,
            lhand=lr * (1 - u) + np.array([0.06, 1.66, 0.30]) * u,
            rhand=rr * (1 - u) + np.array([-0.06, 1.60, 0.33]) * u,
            lfoot=(0.11, 0.08, 0.08), rfoot=(-0.11, 0.08, 0.08 - 0.32 * u))
    return _human(ref, joints, bones, 1.4, f)


def nod(ref, joints, bones):
    def f(rig, t):
        hp = 0.2 * max(0.0, np.sin(2 * np.pi * t / 1.0)) if t < 3.0 else 0.0
        return rig.pose(head_pitch=hp, roll=0.025 * np.sin(2 * np.pi * t / 3.0), root=(0, 0.004 * np.sin(2 * np.pi * t / 3.0), 0))
    return _human(ref, joints, bones, 3.0, f)


def crouch(ref, joints, bones):
    def f(rig, t):
        b = 0.012 * np.sin(2 * np.pi * t / 1.5)
        return rig.pose(root=(0, -0.46 + b, 0.06), pitch=0.38, head_pitch=-0.35,
                        lhand=(0.24, 0.66 + b, 0.40), rhand=(-0.24, 0.66 + b, 0.40),
                        lfoot=(0.17, 0.08, 0.12), rfoot=(-0.17, 0.08, -0.05))
    return _human(ref, joints, bones, 3.0, f)


def push_button(ref, joints, bones):
    def f(rig, t):
        rr = rig.hand_rest("Right")
        tgt = _track([(0, rr), (0.7, (-0.14, 1.22, 0.46)), (0.9, (-0.14, 1.22, 0.53)), (1.1, (-0.14, 1.22, 0.46)), (1.9, rr)], t)
        return rig.pose(rhand=tgt, pitch=float(_track([(0, 0), (0.8, 0.08), (1.9, 0)], t)),
                        head_pitch=float(_track([(0, 0), (0.6, 0.15), (1.5, 0)], t)))
    return _human(ref, joints, bones, 2.0, f)


def open_door(ref, joints, bones):
    def f(rig, t):
        rr = rig.hand_rest("Right")
        tgt = _track([(0, rr), (0.7, (-0.28, 1.02, 0.55)), (1.8, (-0.62, 1.02, 0.02)), (2.5, rr)], t)
        return rig.pose(rhand=tgt, root=_track([(0, (0, 0, 0)), (0.7, (0, 0, 0.05)), (1.8, (0.05, 0, -0.12)), (2.6, (0.05, 0, -0.12))], t),
                        twist=float(_track([(0, 0), (0.7, -0.1), (1.8, -0.4), (2.6, -0.1)], t)),
                        pitch=float(_track([(0, 0), (0.7, 0.1), (2.6, 0)], t)))
    return _human(ref, joints, bones, 2.6, f)


def salute(ref, joints, bones):
    def f(rig, t):
        rr = rig.hand_rest("Right")
        head = rig.rest[rig.j["Head"]]
        u = float(_track([(0, 0), (0.45, 1), (1.6, 1), (2.0, 0)], t))
        return rig.pose(rhand=rr * (1 - u) + (head + np.array([-0.08, 0.06, 0.11])) * u, pitch=-0.03 * u)
    return _human(ref, joints, bones, 2.0, f)


def clap(ref, joints, bones):
    def f(rig, t):
        gap = 0.04 + 0.16 * abs(np.sin(np.pi * 2.2 * t))
        u = float(_track([(0, 0), (0.3, 1), (2.6, 1), (3.0, 0)], t))
        lr, rr = rig.hand_rest("Left"), rig.hand_rest("Right")
        return rig.pose(lhand=lr * (1 - u) + np.array([gap / 2, 1.22, 0.34]) * u,
                        rhand=rr * (1 - u) + np.array([-gap / 2, 1.22, 0.34]) * u, head_pitch=-0.05)
    return _human(ref, joints, bones, 3.0, f)


def point_at(ref, joints, bones):
    def f(rig, t):
        rr = rig.hand_rest("Right")
        u = float(_track([(0, 0), (0.45, 1), (1.6, 1), (2.1, 0)], t))
        return rig.pose(rhand=rr * (1 - u) + np.array([-0.2, 1.48, 0.66]) * u, twist=-0.15 * u, head_pitch=-0.05 * u)
    return _human(ref, joints, bones, 2.1, f)


# =====================================================================================
# Quadrupeds on the same 20-joint format: hind legs use the leg joints, front legs the arm
# joints, the spine runs horizontally (Hips = pelvis, Spine1 = withers), Neck1 = poll,
# Head_End = nose. The viewer draws them with an animal mesh; the agents treat them like
# any other character (root motion, facing, approach distances).
# =====================================================================================
DIMS = {
    # hip_h, body, hw, sw, thigh, shank, cannon, upper_front, lower_front, neck, neck_up, head, head_down
    "dog": dict(hip_h=0.50, body=0.52, hw=0.075, sw=0.085, thigh=0.20, shank=0.20, cannon=0.13,
                uf=0.25, lf=0.27, neck=0.20, neck_up=0.85, head=0.22, head_down=0.15),
    "horse": dict(hip_h=1.48, body=1.05, hw=0.21, sw=0.19, thigh=0.50, shank=0.50, cannon=0.55,
                  uf=0.80, lf=0.66, neck=0.85, neck_up=0.95, head=0.62, head_down=0.85),
}
GAITS = {
    # speed m/s, stride Hz, duty factor, phase per leg (LH, RH, LF, RF), lift (fraction of leg), body bob, pitch
    "walk": dict(v=1.0, f=1.5, duty=0.62, ph=(0.0, 0.5, 0.25, 0.75), lift=0.18, bob=0.012, pitch=0.02),
    "trot": dict(v=2.6, f=2.1, duty=0.45, ph=(0.0, 0.5, 0.5, 0.0), lift=0.25, bob=0.02, pitch=0.02),
    "gallop": dict(v=6.0, f=2.4, duty=0.32, ph=(0.0, 0.1, 0.55, 0.65), lift=0.3, bob=0.04, pitch=0.09),
}
HORSE_SPEED = {"walk": (1.6, 0.95), "trot": (3.6, 1.35), "gallop": (9.0, 1.9)}


def quadruped(ref, joints, bones, kind: str = "dog", gait: str = "walk", cycles: str = "2"):
    j = {n: i for i, n in enumerate(joints)}
    d = DIMS[kind]
    if gait in ("idle", "sit"):
        return _quadruped_still(j, d, kind, gait, len(joints))
    g = dict(GAITS[gait])
    if kind == "horse":
        g["v"], g["f"] = HORSE_SPEED[gait]
    n_cycles = int(cycles)
    frames_per = int(round(FPS / g["f"]))
    f = FPS / frames_per  # exact integer frames per stride -> seamless loop
    v = g["v"]
    S = v / f
    F = frames_per * n_cycles
    leg_len_h = d["thigh"] + d["shank"] + d["cannon"]
    out = []
    for i in range(F + 1):
        t = i / FPS
        p = np.zeros((len(joints), 3))
        z = v * t
        bob = g["bob"] * (d["hip_h"]) * np.cos(4 * np.pi * f * t)
        pitch = g["pitch"] * np.sin(2 * np.pi * f * t)
        hips = np.array([0.0, d["hip_h"] + bob, z])
        fwd = np.array([0.0, np.sin(pitch), np.cos(pitch)])
        withers = hips + fwd * d["body"] + np.array([0, 0.04 * d["hip_h"], 0])
        p[j["Hips"]] = hips + np.array([0, 0.06 * d["hip_h"], 0])
        p[j["Spine"]] = (p[j["Hips"]] + withers) / 2 + np.array([0, 0.03 * d["hip_h"], 0])
        p[j["Spine1"]] = withers
        nb = withers + np.array([0, 0.02, 0.0])
        head_bob = 0.04 * d["neck"] * np.sin(4 * np.pi * f * t) if gait == "walk" else 0.0
        poll = nb + np.array([0, np.sin(d["neck_up"]) * d["neck"] + head_bob, np.cos(d["neck_up"]) * d["neck"]])
        p[j["Neck1"]] = poll
        hd = np.array([0, -np.sin(d["head_down"]), np.cos(d["head_down"])])
        p[j["Head"]] = poll + hd * d["head"] * 0.45
        p[j["Head_End"]] = poll + hd * d["head"]
        for leg, (side, sx), ph in zip(("hind", "hind", "front", "front"),
                                       (("Left", 1), ("Right", -1), ("Left", 1), ("Right", -1)), g["ph"]):
            u = (f * t + ph) % 1.0
            if leg == "hind":
                anchor = p[j["Hips"]] + np.array([sx * d["hw"], -0.06 * d["hip_h"], 0])
            else:
                anchor = withers + np.array([sx * d["sw"], -0.12 * d["hip_h"], 0.02])
            if u < g["duty"]:
                rel, lift, s = S * (g["duty"] / 2 - u), 0.0, 0.0
            else:
                s = (u - g["duty"]) / (1 - g["duty"])
                rel = S * (-g["duty"] / 2 + _ease(s) * g["duty"])
                lift = g["lift"] * leg_len_h * np.sin(np.pi * s)
            foot = np.array([sx * (d["hw"] if leg == "hind" else d["sw"]), 0.03 * d["hip_h"] + lift, anchor[2] + rel])
            if leg == "hind":
                fold = 0.25 + 0.9 * np.sin(np.pi * s)
                hock = foot + np.array([0, np.cos(fold) * d["cannon"], -np.sin(fold) * d["cannon"]])
                p[j[f"{side}UpLeg"]] = anchor
                p[j[f"{side}Leg"]] = _two_bone(anchor, hock, d["thigh"], d["shank"], np.array([0, -0.2, 1.0]))
                p[j[f"{side}Foot"]] = _fix(hock, p[j[f"{side}Leg"]], d["shank"])
                p[j[f"{side}ToeBase"]] = foot
            else:
                p[j[f"{side}Arm"]] = anchor
                p[j[f"{side}ForeArm"]] = _two_bone(anchor, foot, d["uf"], d["lf"], np.array([0, -0.1, -1.0]))
                p[j[f"{side}Hand"]] = _fix(foot, p[j[f"{side}ForeArm"]], d["lf"])
        out.append(p)
    return np.stack(out)


def _quadruped_still(j, d, kind, gait, nj, seconds=4.0):
    out = []
    for i in range(int(seconds * FPS) + 1):
        t = i / FPS
        p = np.zeros((nj, 3))
        breath = 0.006 * d["hip_h"] * np.sin(2 * np.pi * t / 2.0)
        look = 0.35 * np.sin(2 * np.pi * t / seconds)
        if gait == "sit":
            hips = np.array([0.0, d["hip_h"] * 0.45, -0.1 * d["body"]])
            withers = hips + np.array([0, d["body"] * 0.62, d["body"] * 0.72])
        else:
            hips = np.array([0.0, d["hip_h"] + breath, 0.0])
            withers = hips + np.array([0, 0.04 * d["hip_h"] + breath, d["body"]])
        p[j["Hips"]] = hips + np.array([0, 0.06 * d["hip_h"], 0])
        p[j["Spine"]] = (p[j["Hips"]] + withers) / 2 + np.array([0, 0.03 * d["hip_h"], 0])
        p[j["Spine1"]] = withers
        nd = np.array([np.sin(look) * np.cos(d["neck_up"]), np.sin(d["neck_up"]), np.cos(look) * np.cos(d["neck_up"])])
        poll = withers + np.array([0, 0.02, 0]) + nd * d["neck"]
        p[j["Neck1"]] = poll
        hd = np.array([np.sin(look) * np.cos(d["head_down"]), -np.sin(d["head_down"]), np.cos(look) * np.cos(d["head_down"])])
        p[j["Head"]] = poll + hd * d["head"] * 0.45
        p[j["Head_End"]] = poll + hd * d["head"]
        for side, sx in (("Left", 1), ("Right", -1)):
            a = p[j["Hips"]] + np.array([sx * d["hw"], -0.06 * d["hip_h"], 0])
            if gait == "sit":
                toe = np.array([sx * d["hw"] * 1.3, 0.03 * d["hip_h"], hips[2] + 0.75 * d["body"] * 0.5])
                hock = np.array([sx * d["hw"] * 1.3, 0.05 * d["hip_h"], hips[2] - 0.05])
            else:
                toe = np.array([sx * d["hw"], 0.03 * d["hip_h"], -0.02])
                hock = toe + np.array([0, np.cos(0.25) * d["cannon"], -np.sin(0.25) * d["cannon"]])
            p[j[f"{side}UpLeg"]] = a
            p[j[f"{side}Leg"]] = _two_bone(a, hock, d["thigh"], d["shank"], np.array([0, -0.2, 1.0]))
            p[j[f"{side}Foot"]] = _fix(hock, p[j[f"{side}Leg"]], d["shank"])
            p[j[f"{side}ToeBase"]] = toe
            s = withers + np.array([sx * d["sw"], -0.12 * d["hip_h"], 0.02])
            foot = np.array([sx * d["sw"], 0.03 * d["hip_h"], s[2] + (0.05 if gait == "sit" else 0)])
            p[j[f"{side}Arm"]] = s
            p[j[f"{side}ForeArm"]] = _two_bone(s, foot, d["uf"], d["lf"], np.array([0, -0.1, -1.0]))
            p[j[f"{side}Hand"]] = _fix(foot, p[j[f"{side}ForeArm"]], d["lf"])
        out.append(p)
    return np.stack(out)


def dog_lunge(ref, joints, bones):
    """Dog crouches, lunges forward and snaps, then recovers (keyframed on the walk skeleton)."""
    still = quadruped(ref, joints, bones, "dog", "idle")
    j = {n: i for i, n in enumerate(joints)}
    out = []
    for i in range(int(1.6 * FPS) + 1):
        t = i / FPS
        p = still[i % len(still)].copy()
        crouch = float(_track([(0, 0), (0.4, 1), (0.55, 0), (1.6, 0)], t))
        lunge = float(_track([(0, 0), (0.4, 0), (0.7, 1), (1.0, 1), (1.6, 0.6)], t))
        body = ["Hips", "Spine", "Spine1", "Neck1", "Head", "Head_End", "LeftUpLeg", "RightUpLeg", "LeftArm", "RightArm"]
        for n in body:
            p[j[n]] += np.array([0, -0.08 * crouch + 0.04 * lunge, 0.35 * lunge])
        for n in ("Neck1", "Head", "Head_End"):
            p[j[n]] += np.array([0, -0.06 * lunge, 0.08 * lunge])
        for side in ("Left", "Right"):
            for a, b, c in ((f"{side}UpLeg", f"{side}Leg", f"{side}Foot"), (f"{side}Arm", f"{side}ForeArm", f"{side}Hand")):
                if c.endswith("Hand"):
                    p[j[c]] += np.array([0, 0.05 * np.sin(np.pi * min(1, max(0, (t - 0.4) / 0.4))), 0.33 * lunge])
                    p[j[b]] = _two_bone(p[j[a]], p[j[c]], DIMS["dog"]["uf"], DIMS["dog"]["lf"], np.array([0, -0.1, -1.0]))
                    p[j[c]] = _fix(p[j[c]], p[j[b]], DIMS["dog"]["lf"])
                else:
                    p[j[f"{side}ToeBase"]] += np.array([0, 0, 0.2 * lunge])
                    hock = p[j[f"{side}ToeBase"]] + np.array([0, np.cos(0.4) * DIMS["dog"]["cannon"], -np.sin(0.4) * DIMS["dog"]["cannon"]])
                    p[j[b]] = _two_bone(p[j[a]], hock, DIMS["dog"]["thigh"], DIMS["dog"]["shank"], np.array([0, -0.2, 1.0]))
                    p[j[c]] = _fix(hock, p[j[b]], DIMS["dog"]["shank"])
        out.append(p)
    return np.stack(out)


GENERATORS.update({
    "hit_react": hit_react, "hit_gut": hit_gut, "block": block, "nod": nod, "crouch": crouch,
    "push_button": push_button, "open_door": open_door, "salute": salute, "clap": clap, "point_at": point_at,
    "quadruped": quadruped, "dog_lunge": dog_lunge,
})


# =====================================================================================
# Transitions, traversal and defeat states (keyframed on the same HumanRig)
# =====================================================================================
import json as _json  # noqa: E402
from pathlib import Path as _Path  # noqa: E402

MOTIONS_DIR = _Path(__file__).resolve().parents[1] / "data" / "motions"


def _feet0(rig):
    j = rig.j
    return {s: rig.rest[j[f"{s}Foot"]].copy() for s in ("Left", "Right")}


def _step(t, t0, t1, a, b, h):
    """Foot position stepping from a to b between t0 and t1 with an arc of height h."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if t <= t0:
        return a.copy()
    if t >= t1:
        return b.copy()
    u = _ease((t - t0) / (t1 - t0))
    return a * (1 - u) + b * u + np.array([0, _arc(t, t0, t1, h), 0])


def _steps(t, start, plan):
    """plan: list of (t0, t1, target, height); returns the foot position at t."""
    pos = np.asarray(start, float)
    for t0, t1, tgt, h in plan:
        if t < t0:
            return pos
        pos2 = _step(t, t0, t1, pos, tgt, h)
        if t <= t1:
            return pos2
        pos = np.asarray(tgt, float)
    return pos


def _shoulder(rig, side, pitch=0.0, roll=0.0, twist=0.0):
    j = rig.j
    hips = rig.rest[j["Hips"]]
    Rt = _rot("y", twist) @ _rot("z", roll) @ _rot("x", pitch)
    return hips + Rt @ (rig.rest[j[f"{side}Arm"]] - hips)


def run_start(ref, joints, bones):
    """Push-off from a standstill into a run: drop, lean, drive the first two steps."""
    def f(rig, t):
        F = _feet0(rig)
        lr, rr = rig.hand_rest("Left"), rig.hand_rest("Right")
        u = min(t / 0.6, 1.0)
        z = 0.8 * u ** 1.7
        return rig.pose(
            root=(0, float(_track([(0, 0), (0.14, -0.1), (0.6, -0.05)], t)), z),
            pitch=float(_track([(0, 0), (0.2, 0.42), (0.6, 0.3)], t)),
            head_pitch=float(_track([(0, 0), (0.2, -0.3), (0.6, -0.2)], t)),
            lhand=_track([(0, lr), (0.3, (0.2, 1.25, 0.35)), (0.6, (0.2, 1.15, 0.28))], t),
            rhand=_track([(0, rr), (0.3, (-0.2, 0.85, -0.3)), (0.6, (-0.2, 0.9, -0.25))], t),
            rfoot=_steps(t, F["Right"], [(0.12, 0.42, F["Right"] + [0, 0, 0.75], 0.2)]),
            lfoot=_steps(t, F["Left"], [(0.38, 0.6, F["Left"] + [0, 0.12, 0.85], 0.15)]))
    return _human(ref, joints, bones, 0.6, f)


def brake(ref, joints, bones):
    """Planting the feet to stop out of a run: the body pitches back and settles."""
    def f(rig, t):
        F = _feet0(rig)
        lr, rr = rig.hand_rest("Left"), rig.hand_rest("Right")
        u = min(t / 0.45, 1.0)
        z = 0.7 * (1 - (1 - u) ** 2)
        return rig.pose(
            root=(0, float(_track([(0, -0.04), (0.3, -0.14), (0.8, 0)], t)), z),
            pitch=float(_track([(0, 0.2), (0.25, -0.22), (0.55, 0.05), (0.8, 0)], t)),
            head_pitch=float(_track([(0, -0.1), (0.3, 0.12), (0.8, 0)], t)),
            lhand=_track([(0, (0.2, 1.05, 0.25)), (0.3, (0.3, 1.1, 0.3)), (0.8, lr)], t),
            rhand=_track([(0, (-0.2, 0.9, -0.2)), (0.3, (-0.3, 1.1, 0.25)), (0.8, rr)], t),
            lfoot=_steps(t, F["Left"] + [0, 0, 0.1], [(0.0, 0.14, F["Left"] + [0, 0, 0.62], 0.08)]),
            rfoot=_steps(t, F["Right"] + [0, 0, -0.35], [(0.3, 0.6, F["Right"] + [0, 0, 0.7], 0.08)]))
    return _human(ref, joints, bones, 0.8, f)


def turn(ref, joints, bones, degrees="90"):
    """In-place pivot with foot shuffles; positive = turn left (toward +X when facing +Z)."""
    a = np.radians(float(degrees))
    big = abs(a) > np.radians(100)
    dur = 0.85 if big else 0.6
    lead, trail = ("Left", "Right") if a > 0 else ("Right", "Left")

    def f(rig, t):
        F = _feet0(rig)
        R = lambda ang, p: _rot("y", ang) @ p  # noqa: E731
        if big:
            plans = {lead: [(0.04, 0.3, R(a * 0.5, F[lead]), 0.07), (0.5, 0.78, R(a, F[lead]), 0.06)],
                     trail: [(0.24, 0.55, R(a, F[trail]), 0.07)]}
        else:
            plans = {lead: [(0.04, 0.32, R(a, F[lead]), 0.07)], trail: [(0.26, 0.56, R(a, F[trail]), 0.06)]}
        yaw = a * _ease(min(t / (dur * 0.88), 1.0))
        lead_head = np.clip(a - yaw, -0.6, 0.6) * min(t / 0.12, 1.0)
        feet = {s: _steps(t, F[s], plans[s]) for s in ("Left", "Right")}
        return rig.pose(yaw=yaw, root=(0, -0.035 * np.sin(np.pi * min(t / dur, 1.0)), 0), head_yaw=float(lead_head),
                        lfoot=feet["Left"], rfoot=feet["Right"])
    return _human(ref, joints, bones, dur, f)


def bump(ref, joints, bones):
    """Walks into something solid: a jolt, hands up, a step back."""
    def f(rig, t):
        F = _feet0(rig)
        lr, rr = rig.hand_rest("Left"), rig.hand_rest("Right")
        return rig.pose(
            root=_track([(0, (0, 0, 0)), (0.2, (0, 0, 0.12)), (0.45, (0, -0.05, -0.14)), (1.0, (0, 0, -0.16))], t),
            pitch=float(_track([(0, 0.08), (0.2, 0.1), (0.32, -0.22), (0.7, 0.02), (1.0, 0)], t)),
            head_pitch=float(_track([(0, 0), (0.25, -0.35), (0.7, 0.05), (1.0, 0)], t)),
            lhand=_track([(0, lr), (0.22, (0.16, 1.32, 0.3)), (0.7, (0.2, 1.2, 0.22)), (1.0, lr)], t),
            rhand=_track([(0, rr), (0.22, (-0.16, 1.36, 0.3)), (0.7, (-0.2, 1.2, 0.22)), (1.0, rr)], t),
            lfoot=_steps(t, F["Left"], [(0.0, 0.2, F["Left"] + [0, 0, 0.12], 0.05)]),
            rfoot=_steps(t, F["Right"], [(0.32, 0.62, F["Right"] + [0, 0, -0.3], 0.07)]))
    return _human(ref, joints, bones, 1.0, f)


def startle(ref, joints, bones):
    """A sudden flinch at an unexpected sound: jolt, hunch, half-step back, guard up."""
    def f(rig, t):
        F = _feet0(rig)
        lr, rr = rig.hand_rest("Left"), rig.hand_rest("Right")
        return rig.pose(
            root=_track([(0, (0, 0, 0)), (0.08, (0, 0.04, 0)), (0.3, (0, -0.13, -0.12)), (1.0, (0, -0.1, -0.14)),
                         (1.4, (0, -0.04, -0.14))], t),
            pitch=float(_track([(0, 0), (0.08, -0.12), (0.3, 0.18), (1.4, 0.06)], t)),
            roll=float(_track([(0, 0), (0.3, 0.06), (1.4, 0)], t)),
            lhand=_track([(0, lr), (0.25, (0.14, 1.34, 0.2)), (1.0, (0.18, 1.25, 0.24)), (1.4, (0.24, 1.05, 0.12))], t),
            rhand=_track([(0, rr), (0.25, (-0.14, 1.38, 0.22)), (1.0, (-0.18, 1.28, 0.24)), (1.4, (-0.24, 1.05, 0.12))], t),
            lfoot=_steps(t, F["Left"], [(0.14, 0.4, F["Left"] + [0.04, 0, -0.28], 0.08)]),
            rfoot=F["Right"] + [0, 0, 0])
    return _human(ref, joints, bones, 1.4, f)


def step_over(ref, joints, bones):
    """Steps over a low curb at 0.75 m with high, careful steps (1.5 m forward)."""
    def f(rig, t):
        F = _feet0(rig)
        L, Rf = F["Left"], F["Right"]
        z = float(_track([(0, 0), (0.6, 0.3), (1.05, 0.8), (1.5, 1.3), (1.9, 1.5)], t))
        return rig.pose(
            root=(0, 0.03 * np.sin(np.pi * min(t / 1.9, 1.0)), z),
            pitch=float(_track([(0, 0.05), (0.5, 0.18), (1.4, 0.15), (1.9, 0)], t)),
            head_pitch=float(_track([(0, 0), (0.4, 0.35), (1.3, 0.25), (1.9, 0)], t)),
            lhand=_track([(0, rig.hand_rest("Left")), (0.6, (0.32, 1.0, 0.1)), (1.5, (0.3, 1.0, 0.12)), (1.9, rig.hand_rest("Left"))], t),
            rhand=_track([(0, rig.hand_rest("Right")), (0.6, (-0.32, 1.0, 0.1)), (1.5, (-0.3, 1.0, 0.12)), (1.9, rig.hand_rest("Right"))], t),
            rfoot=_steps(t, Rf, [(0.12, 0.55, Rf + [0, 0, 0.45], 0.08), (1.0, 1.48, Rf + [0, 0, 1.5], 0.34)]),
            lfoot=_steps(t, L, [(0.55, 1.05, L + [0, 0, 1.05], 0.34), (1.45, 1.85, L + [0, 0, 1.5], 0.06)]))
    return _human(ref, joints, bones, 1.9, f)


def duck_under(ref, joints, bones):
    """Ducks under a low beam: drops into a crouch, shuffles 2 m, rises on the far side."""
    def f(rig, t):
        F = _feet0(rig)
        L, Rf = F["Left"], F["Right"]
        z = float(_track([(0, 0), (0.4, 0.15), (2.0, 1.85), (2.4, 2.0)], t))
        low = float(_track([(0, 0), (0.45, 1), (1.9, 1), (2.4, 0)], t))
        plan_r = [(0.25 + 0.4 * i, 0.55 + 0.4 * i, Rf + [0, 0, min(0.42 * (i + 1), 2.0)], 0.07) for i in range(5)]
        plan_l = [(0.45 + 0.4 * i, 0.75 + 0.4 * i, L + [0, 0, min(0.42 * (i + 1) + 0.2, 2.0)], 0.07) for i in range(5)]
        return rig.pose(
            root=(0, -0.42 * low, z), pitch=0.55 * low, head_pitch=-0.3 * low,
            lhand=rig.hand_rest("Left") * (1 - low) + np.array([0.24, 0.7, 0.45]) * low,
            rhand=rig.hand_rest("Right") * (1 - low) + np.array([-0.24, 0.7, 0.45]) * low,
            rfoot=_steps(t, Rf, plan_r), lfoot=_steps(t, L, plan_l))
    return _human(ref, joints, bones, 2.4, f)


def squeeze(ref, joints, bones):
    """Shuffles sideways (toward +X) through a narrow gap, arms tucked, 1.6 m."""
    def f(rig, t):
        F = _feet0(rig)
        L, Rf = F["Left"], F["Right"]
        x = float(_track([(0, 0), (2.2, 1.6)], t))
        plan_l = [(0.1 + 0.5 * i, 0.4 + 0.5 * i, L + [0.4 * (i + 1), 0, 0], 0.06) for i in range(4)]
        plan_r = [(0.35 + 0.5 * i, 0.62 + 0.5 * i, Rf + [0.4 * (i + 1), 0, 0], 0.05) for i in range(4)]
        tuck = float(_track([(0, 0), (0.3, 1), (1.9, 1), (2.2, 0)], t))
        return rig.pose(
            root=(x, -0.03 * tuck, 0), pitch=-0.06 * tuck, head_yaw=0.5 * tuck,
            lhand=rig.hand_rest("Left") * (1 - tuck) + np.array([0.08, 1.28, 0.16]) * tuck,
            rhand=rig.hand_rest("Right") * (1 - tuck) + np.array([-0.06, 1.24, 0.18]) * tuck,
            lfoot=_steps(t, L, plan_l), rfoot=_steps(t, Rf, plan_r))
    return _human(ref, joints, bones, 2.2, f)


def collapse(ref, joints, bones):
    """A non-combat collapse (fainting / unseen force): knees buckle, the body folds back to the floor."""
    def f(rig, t):
        F = _feet0(rig)
        pitch = float(_track([(0, 0), (0.8, 0.3), (1.0, 0.2), (1.6, -1.52), (1.8, -1.45), (2.0, -1.55), (2.6, -1.55)], t))
        ry = float(_track([(0, 0), (0.8, -0.32), (1.0, -0.4), (1.6, -0.8), (1.8, -0.76), (2.0, -0.8), (2.6, -0.8)], t))
        rz = float(_track([(0, 0), (0.8, -0.05), (1.6, -0.3), (2.6, -0.3)], t))
        hips_y = rig.rest[rig.j["Hips"], 1]
        def hand(side, sx):
            sh = _shoulder(rig, side, pitch)
            rest = rig.hand_rest(side)
            down = sh + np.array([sx * 0.3, 0, 0.12]) + np.array([0, -1, 0]) * 0.0
            floor = np.array([sh[0] + sx * 0.3, 0.07 - ry, sh[2] + 0.18])  # beside the shoulders on the floor
            u = float(_track([(0, 0), (0.9, 0.1), (1.6, 1.0), (2.6, 1.0)], t))
            limp = np.array([sx * 0.22, hips_y - 0.15, 0.12])
            v = float(_track([(0, 0), (0.6, 1.0), (2.6, 1.0)], t))
            base = rest * (1 - v) + limp * v
            return base * (1 - u) + floor * u
        feet_z = float(_track([(0, 0), (0.9, 0.05), (1.6, 0.55), (2.6, 0.6)], t))
        return rig.pose(
            root=(0, ry, rz), pitch=pitch,
            roll=float(_track([(0, 0), (0.8, 0.08), (1.6, 0.04), (2.6, 0.04)], t)),
            head_pitch=float(_track([(0, 0), (0.7, 0.55), (1.5, 0.1), (1.8, -0.15), (2.6, 0.2)], t)),
            head_yaw=float(_track([(0, 0), (1.8, 0), (2.4, 0.5), (2.6, 0.5)], t)),
            lhand=hand("Left", 1), rhand=hand("Right", -1),
            lfoot=F["Left"] + [0.06 * feet_z, 0, feet_z], rfoot=F["Right"] + [-0.04 * feet_z, 0, feet_z * 0.85])
    return _human(ref, joints, bones, 2.6, f)


def _last_frame(src: str) -> np.ndarray:
    data = _json.loads((_Path(MOTIONS_DIR) / f"{src}.json").read_text())
    return np.array(data["frames"][-1], dtype=float).reshape(-1, 3) / 1000.0


def _ik_to(p, j, side, target, pole, kind="arm", L=None):
    if kind == "arm":
        a, b, c = f"{side}Arm", f"{side}ForeArm", f"{side}Hand"
    else:
        a, b, c = f"{side}UpLeg", f"{side}Leg", f"{side}Foot"
    l1 = np.linalg.norm(p[j[b]] - p[j[a]]) if L is None else L[0]
    l2 = np.linalg.norm(p[j[c]] - p[j[b]]) if L is None else L[1]
    tgt = np.asarray(target, float)
    mid = _two_bone(p[j[a]], tgt, l1, l2, pole)
    end = _fix(tgt, mid, l2)
    delta = end - p[j[c]]
    p[j[b]] = mid
    p[j[c]] = end
    if kind == "leg":
        p[j[f"{side}ToeBase"]] = p[j[f"{side}ToeBase"]] + delta
    return p


def down(ref, joints, bones, src="fall_back", style="still"):
    """On-ground idle continuing exactly from the last frame of `src` (a fall, collapse or lie-down).

    still: breathing only. groan: head rolls, a knee draws up, a hand holds the stomach.
    crawl: drags forward on the elbows (prone sources), 0.5 m per 2.4 s cycle.
    """
    j = {n: i for i, n in enumerate(joints)}
    base = _last_frame(src)
    up = np.array([0, 1.0, 0])
    chest = [j[n] for n in ("Spine1", "Neck1", "Head", "Head_End", "LeftArm", "LeftForeArm", "LeftHand",
                            "RightArm", "RightForeArm", "RightHand")]
    period = 2.4 if style == "crawl" else 4.0
    n = int(round(period * FPS)) + 1
    L = {s: (np.linalg.norm(base[j[f"{s}ForeArm"]] - base[j[f"{s}Arm"]]),
             np.linalg.norm(base[j[f"{s}Hand"]] - base[j[f"{s}ForeArm"]])) for s in ("Left", "Right")}
    LL = {s: (np.linalg.norm(base[j[f"{s}Leg"]] - base[j[f"{s}UpLeg"]]),
              np.linalg.norm(base[j[f"{s}Foot"]] - base[j[f"{s}Leg"]])) for s in ("Left", "Right")}
    fwd = base[j["Head"]] - base[j["Hips"]]
    fwd[1] = 0
    fwd /= max(np.linalg.norm(fwd), 1e-6)
    left = base[j["LeftArm"]] - base[j["RightArm"]]
    left[1] = 0
    left /= max(np.linalg.norm(left), 1e-6)
    out = []
    for i in range(n):
        t = i / FPS
        w = 2 * np.pi * t / period
        p = base.copy()
        breath = 0.012 * np.sin(2 * w if style != "crawl" else w)
        p[chest] += up * breath
        if style == "groan":
            neck = p[j["Neck1"]]
            axis = p[j["Neck1"]] - p[j["Spine1"]]
            axis /= max(np.linalg.norm(axis), 1e-6)
            ang = 0.35 * np.sin(w)
            K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
            R = np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K
            for nme in ("Head", "Head_End"):
                p[j[nme]] = neck + R @ (p[j[nme]] - neck)
            belly = (p[j["Spine"]] + p[j["Hips"]]) / 2 + up * 0.13
            p = _ik_to(p, j, "Right", belly + left * 0.02, up + left * -0.5, "arm", L["Right"])
            draw = 0.24 * (0.5 - 0.5 * np.cos(w))
            foot = base[j["LeftFoot"]] + (base[j["LeftUpLeg"]] - base[j["LeftFoot"]]) * np.array([1, 0, 1]) * draw / max(
                np.linalg.norm((base[j["LeftUpLeg"]] - base[j["LeftFoot"]]) * np.array([1, 0, 1])), 1e-6)
            p = _ik_to(p, j, "Left", foot, up, "leg", LL["Left"])
        elif style == "crawl":
            D = 0.5
            s = D * t / period + 0.04 * np.sin(2 * w)
            p += fwd * s
            p[[j["Neck1"], j["Head"], j["Head_End"]]] += up * 0.04
            for side, ph, sx in (("Left", 0.0, 1), ("Right", 0.5, -1)):
                u = (t / period + ph) % 1.0
                if u < 0.6:
                    rel, lift = 0.35 - 0.3 * u / 0.6, 0.0
                else:
                    v = (u - 0.6) / 0.4
                    rel, lift = 0.05 + 0.3 * _ease(v), 0.09 * np.sin(np.pi * v)
                sh = p[j[f"{side}Arm"]]
                tgt = sh + fwd * rel + left * sx * 0.14
                tgt[1] = 0.06 + lift
                p = _ik_to(p, j, side, tgt, up * 0.6 + left * sx * 0.6 - fwd * 0.4, "arm", L[side])
                push = 0.12 * max(0.0, np.sin(2 * np.pi * (t / period + ph)))
                hip = p[j[f"{side}UpLeg"]]
                foot = p[j[f"{side}Foot"]] + (hip - p[j[f"{side}Foot"]]) * np.array([1, 0, 1]) * push
                p = _ik_to(p, j, side, foot, left * sx + up * 0.3, "leg", LL[side])
        out.append(p)
    return np.stack(out)


GENERATORS.update({
    "run_start": run_start, "brake": brake, "turn": turn, "bump": bump, "startle": startle,
    "step_over": step_over, "duck_under": duck_under, "squeeze": squeeze, "collapse": collapse, "down": down,
})


# =====================================================================================
# Ambient life, pacing beats, stunts, vehicles and director utilities (gaze-marked)
# =====================================================================================
def _g(ref, joints, bones, seconds, fn):
    rig = HumanRig(ref, joints, bones)
    rig.gaze = True
    n = int(round(seconds * FPS)) + 1
    return np.stack([fn(rig, i / FPS) for i in range(n)])


def _S(t, period):
    return np.sin(2 * np.pi * t / period)


def _whole(p, R, pivot):
    return (R @ (p - pivot).T).T + pivot


def _ground(p, contact=0.07):
    p = p.copy()
    p[:, 1] += contact - p[:, 1].min()
    return p


def _foot_path(t, pos_fn, yaw_fn, cycle, phase, lat, h=0.08):
    """Foot on a walking path: stance 60% of the cycle at a plant point, swing 40% to the next."""
    u = t / cycle - phase
    k = np.floor(u)
    f = u - k

    def plant(kk):
        tm = (kk + phase + 0.3) * cycle
        y = yaw_fn(tm)
        return pos_fn(tm) + np.array([np.cos(y), 0, -np.sin(y)]) * lat + np.array([0, 0.08, 0])
    if f < 0.6:
        return plant(k)
    v = _ease((f - 0.6) / 0.4)
    return plant(k) * (1 - v) + plant(k + 1) * v + np.array([0, h * np.sin(np.pi * v), 0])


# ---------------------------------------------------------------- wallflower idles
def idle_phone(ref, joints, bones):
    def f(rig, t):
        tap = 0.008 * max(0, _S(t, 0.33))
        return rig.pose(root=(0.03 * _S(t, 6), 0, 0), roll=0.02 * _S(t, 6), head_pitch=0.45 + 0.03 * _S(t, 3),
                        lhand=(0.05, 1.1, 0.27), rhand=(-0.04, 1.12 + tap, 0.29))
    return _g(ref, joints, bones, 6.0, f)


def idle_watch(ref, joints, bones):
    def f(rig, t):
        lr = rig.hand_rest("Left")
        u = float(_track([(0, 0), (0.5, 0), (1.1, 1), (2.6, 1), (3.2, 0), (4.5, 0)], t))
        return rig.pose(root=(0.025 * _S(t, 4.5), 0, 0), lhand=lr * (1 - u) + np.array([0.02, 1.24, 0.3]) * u,
                        head_pitch=0.4 * u, head_yaw=0.22 * u, roll=0.015 * _S(t, 4.5))
    return _g(ref, joints, bones, 4.5, f)


def idle_arms_crossed(ref, joints, bones):
    def f(rig, t):
        return rig.pose(root=(0.035 * _S(t, 6), 0.004 * _S(t, 3), 0), roll=0.025 * _S(t, 6),
                        head_yaw=float(_track([(0, 0), (1.5, 0.3), (3, 0), (4.5, -0.25), (6, 0)], t)),
                        lhand=(-0.13, 1.16, 0.19), rhand=(0.13, 1.13, 0.16))
    return _g(ref, joints, bones, 6.0, f)


def idle_shift(ref, joints, bones):
    def f(rig, t):
        return rig.pose(root=(0.045 * _S(t, 6), -0.012 * abs(_S(t, 6)), 0), roll=0.035 * _S(t, 6),
                        head_yaw=float(_track([(0, 0), (1.2, 0.35), (2.6, 0.3), (3.6, -0.3), (5, -0.2), (6, 0)], t)),
                        head_pitch=0.04 * _S(t, 3))
    return _g(ref, joints, bones, 6.0, f)


# ---------------------------------------------------------------- background chatting
def _talk(rig, t):
    keys = [(0, (-0.18, 1.08, 0.3)), (0.5, (-0.3, 1.2, 0.36)), (1.0, (-0.14, 1.05, 0.26)), (1.5, (-0.26, 1.16, 0.34)),
            (2.0, (-0.2, 1.12, 0.3)), (2.6, (-0.32, 1.22, 0.3)), (3.2, (-0.16, 1.06, 0.28)), (4.0, (-0.18, 1.08, 0.3))]
    lk = [(0, (0.2, 1.0, 0.2)), (1.2, (0.24, 1.12, 0.3)), (2.4, (0.18, 1.02, 0.22)), (4.0, (0.2, 1.0, 0.2))]
    return dict(rhand=_track(keys, t % 4.0), lhand=_track(lk, t % 4.0),
                head_pitch=0.05 * _S(t, 0.4) - 0.03, head_yaw=0.1 * _S(t, 2.0), twist=0.06 * _S(t, 2.0))


def _listen(rig, t):
    return dict(lhand=(0.25, 1.0, -0.02), rhand=rig.hand_rest("Right"),
                head_pitch=0.12 * max(0.0, _S(t, 1.3)), head_yaw=0.05 * _S(t, 4.0), roll=0.03)


def _chat(first_talks):
    def gen(ref, joints, bones):
        def f(rig, t):
            talking = (t % 8.0 < 4.0) == first_talks
            a, b = _talk(rig, t), _listen(rig, t)
            # 0.4 s crossfade between talking and listening
            ph = t % 4.0
            w = min(1.0, ph / 0.4)
            cur, prev = (a, b) if talking else (b, a)
            kw = {}
            for k in set(cur) | set(prev):
                c, p_ = np.asarray(cur.get(k, 0.0), float), np.asarray(prev.get(k, 0.0), float)
                kw[k] = c * w + p_ * (1 - w)
            kw = {k: (float(v) if np.ndim(v) == 0 else v) for k, v in kw.items()}
            return rig.pose(root=(0.02 * _S(t, 8), 0, 0), **kw)
        return _g(ref, joints, bones, 8.0, f)
    return gen


# ---------------------------------------------------------------- prop loops
def wipe_counter(ref, joints, bones):
    def f(rig, t):
        a = 2 * np.pi * t / 2.0
        return rig.pose(root=(0, -0.03, 0), pitch=0.22, head_pitch=0.25, twist=0.05 * np.cos(a),
                        rhand=(-0.12 + 0.12 * np.cos(a), 0.99, 0.5 + 0.08 * np.sin(a)), lhand=(0.22, 0.99, 0.42))
    return _g(ref, joints, bones, 4.0, f)


def sweep(ref, joints, bones):
    def f(rig, t):
        x = 0.22 * _S(t, 1.5)
        return rig.pose(root=(0, -0.05, 0), pitch=0.22, twist=-0.18 * _S(t, 1.5), head_pitch=0.3,
                        rhand=(-0.08 + x * 0.4, 1.02, 0.18), lhand=(-0.02 + x, 0.74, 0.34))
    return _g(ref, joints, bones, 3.0, f)


def _seated(rig, F, **kw):
    return rig.pose(root=(0, -0.44, 0), lfoot=F["Left"] + [0.04, 0, 0.42], rfoot=F["Right"] + [-0.04, 0, 0.42], **kw)


def type_laptop(ref, joints, bones):
    def f(rig, t):
        F = _feet0(rig)
        a, b = 0.008 * max(0, _S(t, 0.2)), 0.008 * max(0, _S(t + 0.1, 0.2))
        return _seated(rig, F, pitch=0.12, head_pitch=0.3 + 0.03 * _S(t, 4), head_yaw=0.08 * _S(t, 4),
                       lhand=(0.12, 0.8 + 0.44 + a, 0.42), rhand=(-0.12, 0.8 + 0.44 + b, 0.42))
    return _g(ref, joints, bones, 4.0, f)


def read_book(ref, joints, bones):
    def f(rig, t):
        page = float(_track([(0, 0), (4.4, 0), (4.8, 1), (5.3, 0), (6, 0)], t))
        return rig.pose(root=(0.02 * _S(t, 6), 0, 0), head_pitch=0.42, head_yaw=0.06 * _S(t, 3),
                        lhand=(0.11, 1.1, 0.32), rhand=np.array([-0.11, 1.1, 0.32]) * (1 - page) + np.array([0.07, 1.16, 0.34]) * page)
    return _g(ref, joints, bones, 6.0, f)


def sing(ref, joints, bones):
    def f(rig, t):
        sweep_ = float(_track([(0, 0), (1.0, 1), (2.0, 0.4), (3.0, 1), (4.0, 0)], t))
        return rig.pose(roll=0.05 * _S(t, 2), root=(0.02 * _S(t, 2), 0, 0), head_pitch=-0.18 + 0.05 * _S(t, 1),
                        head_yaw=0.12 * _S(t, 4), lhand=(0.05, 1.3, 0.16),
                        rhand=np.array([-0.3, 1.18, 0.35]) * (1 - sweep_) + np.array([-0.5, 1.42, 0.42]) * sweep_)
    return _g(ref, joints, bones, 4.0, f)


# ---------------------------------------------------------------- pacing beats
def think(ref, joints, bones):
    def f(rig, t):
        F = _feet0(rig)
        rub = np.array([0.012 * np.cos(2 * np.pi * t / 0.8), 0.01 * np.sin(2 * np.pi * t / 0.8), 0])
        tap = 0.045 * max(0.0, _S(t, 0.5)) if (t % 3.0) > 1.0 else 0.0
        return rig.pose(head_pitch=0.12 + 0.05 * _S(t, 6), head_yaw=0.2 * _S(t, 6), twist=0.04 * _S(t, 6),
                        rhand=np.array([-0.02, 1.57, 0.15]) + rub, lhand=(-0.04, 1.12, 0.18),
                        rfoot=F["Right"] + [0, tap, 0.08])
    return _g(ref, joints, bones, 6.0, f)


def pace_circle(ref, joints, bones):
    T, r = 7.2, 0.9

    def pos(t):
        th = 2 * np.pi * t / T
        return np.array([r - r * np.cos(th), 0, r * np.sin(th)])

    def yaw(t):
        return 2 * np.pi * t / T

    def f(rig, t):
        y = yaw(t)
        p0 = pos(t)
        Rg = _rot("y", y)
        L = _foot_path(t, pos, yaw, 1.2, 0.0, 0.11)
        R = _foot_path(t, pos, yaw, 1.2, 0.5, -0.11)
        return rig.pose(root=(p0[0], -0.02 + 0.015 * abs(_S(t, 1.2)), p0[2]), yaw=y, head_pitch=0.2,
                        lhand=(-0.04, 1.12, 0.18), rhand=np.array([-0.02, 1.56, 0.15]), lfoot=L, rfoot=R)
    return _g(ref, joints, bones, T, f)


def flinch_shield(ref, joints, bones):
    def f(rig, t):
        F = _feet0(rig)
        u = float(_track([(0, 0), (0.15, 1), (1.1, 1), (1.7, 0.3)], t))
        return rig.pose(root=(0, -0.2 * u, -0.1 * u), pitch=0.32 * u, head_pitch=0.35 * u, head_yaw=0.45 * u,
                        twist=0.15 * u,
                        lhand=rig.hand_rest("Left") * (1 - u) + np.array([-0.06, 1.58, 0.26]) * u,
                        rhand=rig.hand_rest("Right") * (1 - u) + np.array([0.08, 1.52, 0.28]) * u,
                        lfoot=_steps(t, F["Left"], [(0.05, 0.3, F["Left"] + [0.05, 0, -0.25], 0.07)]))
    return _g(ref, joints, bones, 1.7, f)


def exhausted(ref, joints, bones):
    def f(rig, t):
        br = _S(t, 4.0 / 3)
        return rig.pose(root=(0, -0.26 + 0.02 * br, 0.03), pitch=0.85 + 0.05 * br, head_pitch=-0.35,
                        lhand=(0.13, 0.74 + 0.01 * br, 0.24), rhand=(-0.13, 0.74 + 0.01 * br, 0.24),
                        lfoot=_feet0(rig)["Left"] + [0.04, 0, 0.02], rfoot=_feet0(rig)["Right"] + [-0.04, 0, 0.02])
    return _g(ref, joints, bones, 4.0, f)


def wipe_sweat(ref, joints, bones):
    def f(rig, t):
        rr = rig.hand_rest("Right")
        tgt = _track([(0, rr), (0.6, (-0.14, 1.74, 0.12)), (1.2, (0.1, 1.74, 0.12)), (1.6, (0.0, 1.5, 0.2)), (2.6, rr)], t)
        br = _S(t, 1.0)
        return rig.pose(rhand=tgt, root=(0, 0.01 * br, 0), pitch=0.08 + 0.03 * br,
                        head_pitch=float(_track([(0, 0), (0.6, -0.15), (1.4, 0.05), (2.6, 0)], t)))
    return _g(ref, joints, bones, 2.6, f)


def wake_up(ref, joints, bones):
    """Groggy recovery from the collapse's floor pose: sit up, roll to a crouch, stand, shake the head."""
    def f(rig, t):
        F = _feet0(rig)
        k = lambda keys: float(_track(keys, t))  # noqa: E731
        pitch = k([(0, -1.55), (0.7, -1.55), (1.9, 0.35), (3.0, 0.55), (4.3, 0.05), (5.6, 0.02)])
        ry = k([(0, -0.8), (0.7, -0.8), (1.9, -0.72), (3.0, -0.45), (4.3, 0.0), (5.6, 0.0)])
        rz = k([(0, -0.3), (1.9, -0.3), (3.0, -0.05), (4.3, 0.0), (5.6, 0.0)])
        fz = k([(0, 0.6), (0.7, 0.6), (1.9, 0.35), (3.0, 0.08), (4.3, 0.0), (5.6, 0.0)])
        shake = 0.3 * np.sin(2 * np.pi * 2.2 * t) * np.exp(-(t - 4.6) ** 2 / 0.25) if t > 4.0 else 0.0
        head_yaw = k([(0, 0.5), (0.4, -0.2), (0.7, 0.3), (1.9, 0.0), (5.6, 0.0)]) + shake
        lr, rr = rig.hand_rest("Left"), rig.hand_rest("Right")
        sh_l, sh_r = _shoulder(rig, "Left", pitch), _shoulder(rig, "Right", pitch)
        floor_l = np.array([sh_l[0] + 0.3, 0.07 - ry, sh_l[2] + 0.18])
        floor_r = np.array([sh_r[0] - 0.3, 0.07 - ry, sh_r[2] + 0.18])
        push_l = np.array([0.32, 0.07 - ry, -0.05])
        push_r = np.array([-0.32, 0.07 - ry, -0.05])
        knee_l, knee_r = np.array([0.14, 0.75, 0.3]), np.array([-0.14, 0.75, 0.3])
        head_r = np.array([-0.05, 1.7, 0.16])
        lh = _track([(0, floor_l), (0.7, floor_l), (1.6, push_l), (2.4, push_l), (3.2, knee_l), (4.3, lr), (5.6, lr)], t)
        rh = _track([(0, floor_r), (0.7, floor_r), (1.6, push_r), (2.4, push_r), (3.2, knee_r), (4.1, rr), (4.5, head_r),
                     (5.2, head_r), (5.6, rr)], t)
        return rig.pose(root=(0, ry, rz), pitch=pitch, roll=k([(0, 0.04), (1.9, 0.0)]), head_yaw=head_yaw,
                        head_pitch=k([(0, 0.2), (1.9, 0.25), (4.3, 0.1), (5.6, 0)]), lhand=lh, rhand=rh,
                        lfoot=F["Left"] + [0.06 * fz, 0, fz], rfoot=F["Right"] + [-0.04 * fz, 0, fz * 0.85])
    return _g(ref, joints, bones, 5.6, f)


# ---------------------------------------------------------------- calibration & director utilities
def calib_pose(ref, joints, bones, kind="t"):
    def f(rig, t):
        out = {}
        for side, sx in (("Left", 1), ("Right", -1)):
            s = _shoulder(rig, side)
            L = (rig.up + rig.fo) * 0.97
            out["lhand" if side == "Left" else "rhand"] = s + (np.array([sx * L, 0, 0]) if kind == "t"
                                                               else np.array([sx * L * 0.72, -L * 0.69, 0]))
        return rig.pose(**out)
    return _g(ref, joints, bones, 2.0, f)


def standin_turn(ref, joints, bones):
    def f(rig, t):
        s_l, s_r = _shoulder(rig, "Left"), _shoulder(rig, "Right")
        L = (rig.up + rig.fo) * 0.97
        p = rig.pose(lhand=s_l + np.array([L * 0.5, -L * 0.85, 0]), rhand=s_r + np.array([-L * 0.5, -L * 0.85, 0]))
        hip = p[rig.j["Hips"]].copy()
        hip[1] = 0
        return _whole(p, _rot("y", 2 * np.pi * t / 8.0), hip)
    return _g(ref, joints, bones, 8.0, f)


def scale_ref(ref, joints, bones):
    def f(rig, t):
        u = float(_track([(0, 0), (0.6, 1), (3.4, 1), (4.0, 0)], t))
        s_l, s_r = _shoulder(rig, "Left"), _shoulder(rig, "Right")
        L = (rig.up + rig.fo) * 0.97
        return rig.pose(lhand=rig.hand_rest("Left") * (1 - u) + (s_l + np.array([L, 0, 0])) * u,
                        rhand=rig.hand_rest("Right") * (1 - u) + (s_r + np.array([-L, 0, 0])) * u)
    return _g(ref, joints, bones, 4.0, f)


EYELINE = [-1.3, -0.65, 0.0, 0.65, 1.3, 0.65, 0.0, -0.65, -1.3]


def eyeline_yaw(t):
    """Mechanical stepped pan: hold each angle, then move to the next in 0.35 s."""
    i = int(min(t, 8.999))
    u = min(max((t - i - 0.65) / 0.35, 0.0), 1.0)
    a, b = EYELINE[i], EYELINE[min(i + 1, len(EYELINE) - 1)]
    return a + (b - a) * _ease(u)


def eyeline_sweep(ref, joints, bones):
    def f(rig, t):
        y = eyeline_yaw(t)
        return rig.pose(head_yaw=float(np.clip(y, -1.25, 1.25)), twist=float(y - np.clip(y, -1.25, 1.25)))
    return _g(ref, joints, bones, 9.0, f)


# ---------------------------------------------------------------- continuity connectors
def settle(ref, joints, bones):
    """Loop-to-loop shuffle: the feet re-plant into a square stance before a seated / kneeling / prone loop."""
    def f(rig, t):
        F = _feet0(rig)
        return rig.pose(root=(0.02 * np.sin(np.pi * min(t / 0.6, 1)), 0, 0),
                        lfoot=_steps(t, F["Left"] + [0.03, 0, -0.12], [(0.0, 0.25, F["Left"], 0.05)]),
                        rfoot=_steps(t, F["Right"] + [-0.02, 0, 0.1], [(0.22, 0.5, F["Right"], 0.05)]))
    return _g(ref, joints, bones, 0.6, f)


def wind_down(ref, joints, bones):
    """Breathing settles over 8 s after a sprint or a fight: chest heave fades, posture straightens."""
    def f(rig, t):
        a = np.exp(-t / 3.0)
        br = np.sin(2 * np.pi * t / (0.9 + 0.35 * t / 8))
        hips = float(_track([(0, 1), (5.0, 1), (7.0, 0), (8, 0)], t))
        return rig.pose(root=(0, -0.05 * a + 0.025 * a * br, 0), pitch=0.3 * a + 0.05 * a * br,
                        head_pitch=-0.1 * a, lhand=rig.hand_rest("Left") * (1 - hips) + np.array([0.24, 1.0, -0.02]) * hips,
                        rhand=rig.hand_rest("Right") * (1 - hips) + np.array([-0.24, 1.0, -0.02]) * hips)
    return _g(ref, joints, bones, 8.0, f)


# ---------------------------------------------------------------- stunts
def stagger(ref, joints, bones, direction="back"):
    d = {"back": np.array([0, 0, -1.0]), "forward": np.array([0, 0, 1.0]), "left": np.array([1.0, 0, 0]),
         "right": np.array([-1.0, 0, 0])}[direction]

    def f(rig, t):
        F = _feet0(rig)
        u = float(_track([(0, 0), (0.9, 1), (1.6, 1)], t))
        lean = float(_track([(0, 0), (0.12, 1), (0.6, 0.4), (1.2, 0), (1.6, 0)], t))
        first, second = ("Right", "Left") if direction in ("back", "right") else ("Left", "Right")
        fl = {first: _steps(t, F[first], [(0.12, 0.38, F[first] + d * 0.42, 0.08)]),
              second: _steps(t, F[second], [(0.34, 0.66, F[second] + d * 0.72, 0.08)])}
        flail = float(_track([(0, 0), (0.15, 1), (0.7, 1), (1.3, 0)], t))
        return rig.pose(root=d * 0.72 * u, pitch=-0.3 * lean * d[2], roll=0.25 * lean * d[0],
                        head_pitch=-0.25 * lean * d[2],
                        lhand=rig.hand_rest("Left") * (1 - flail) + np.array([0.48, 1.35, 0.1]) * flail,
                        rhand=rig.hand_rest("Right") * (1 - flail) + np.array([-0.48, 1.32, 0.12]) * flail,
                        lfoot=fl["Left"], rfoot=fl["Right"])
    return _g(ref, joints, bones, 1.6, f)


def _tucked(rig):
    return rig.pose(root=(0, -0.45, 0), pitch=1.0, head_pitch=0.5, lhand=(0.14, 0.62, 0.3), rhand=(-0.14, 0.62, 0.3),
                    lfoot=(0.12, 0.26, 0.2), rfoot=(-0.12, 0.26, 0.2))


def _roll_frame(rig, angle, axis="x"):
    p = _tucked(rig)
    c = (p[rig.j["Hips"]] + p[rig.j["Spine1"]]) / 2
    return _ground(_whole(p, _rot(axis, angle), c))


def leap_roll(ref, joints, bones):
    """Running dive (leap of faith on the flat), arms forward, shoulder roll and back up: ~4.5 m."""
    def f(rig, t):
        F = _feet0(rig)
        if t < 0.5:  # two driving steps
            return rig.pose(root=(0, -0.05, 1.2 * (t / 0.5) ** 1.4), pitch=0.35,
                            lhand=(0.2, 1.2, 0.35), rhand=(-0.2, 0.9, -0.2),
                            rfoot=_steps(t, F["Right"], [(0.0, 0.25, F["Right"] + [0, 0, 0.7], 0.15)]),
                            lfoot=_steps(t, F["Left"], [(0.22, 0.48, F["Left"] + [0, 0, 1.3], 0.15)]))
        if t < 1.0:  # airborne dive
            u = (t - 0.5) / 0.5
            ext = rig.pose(lhand=(0.12, 2.05, 0.35), rhand=(-0.12, 2.05, 0.35), head_pitch=-0.3)
            c = ext[rig.j["Hips"]].copy()
            p = _whole(ext, _rot("x", 0.4 + 0.9 * u), c)
            p = p - np.array([0, c[1], 0]) + np.array([0, 0.75 + 0.55 * np.sin(np.pi * u) - 0.3 * u, 1.2 + 1.8 * u])
            if u > 0.7:  # tuck before the hands touch
                w = (u - 0.7) / 0.3
                q = _roll_frame(rig, 1.3) + np.array([0, 0, 3.0])
                p = p * (1 - w) + q * w
            return p
        if t < 1.8:  # forward roll
            u = (t - 1.0) / 0.8
            return _roll_frame(rig, 1.3 + (2 * np.pi - 1.3) * u) + np.array([0, 0, 3.0 + 1.3 * u])
        u = _ease(min((t - 1.8) / 1.0, 1.0))  # come up
        tuck = _roll_frame(rig, 2 * np.pi) + np.array([0, 0, 4.3])
        stand = rig.pose(root=(0, -0.45 * (1 - u), 4.3 + 0.3 * u), pitch=0.9 * (1 - u))
        return tuck * (1 - u) + stand * u
    return _g(ref, joints, bones, 2.8, f)


def high_fall(ref, joints, bones):
    """Drops in from ~3 m flailing, hard landing, forward roll, back on the feet."""
    def f(rig, t):
        if t < 0.8:
            y = 3.2 - 5.0 * t * t
            w = 2 * np.pi * 2.2 * t
            out = {}
            for side, sx, ph in (("Left", 1, 0), ("Right", -1, np.pi)):
                s = _shoulder(rig, side, -0.2)
                out["lhand" if side == "Left" else "rhand"] = s + np.array([sx * 0.25, 0.45 * np.cos(w + ph), 0.45 * np.sin(w + ph)])
            F = _feet0(rig)
            lf = F["Left"] + np.array([0, y + 0.2 + 0.2 * np.sin(w), 0.2 * np.cos(w)])
            rf = F["Right"] + np.array([0, y + 0.2 - 0.2 * np.sin(w), -0.2 * np.cos(w)])
            return rig.pose(root=(0, y, 0), pitch=-0.2, head_pitch=0.2, lfoot=lf, rfoot=rf, **out)
        if t < 1.0:  # hard landing, absorb into a crouch with hands down
            u = _ease((t - 0.8) / 0.2)
            return rig.pose(root=(0, -0.55 * u, 0.1 * u), pitch=0.65 * u,
                            lhand=np.array([0.22, 0.1 + 0.55 * u + 0.6 * (1 - u), 0.38]),
                            rhand=np.array([-0.22, 0.1 + 0.55 * u + 0.6 * (1 - u), 0.38]))
        if t < 1.8:
            u = (t - 1.0) / 0.8
            land = rig.pose(root=(0, -0.55, 0.1), pitch=0.65, lhand=(0.22, 0.65, 0.38), rhand=(-0.22, 0.65, 0.38))
            roll = _roll_frame(rig, 1.0 + (2 * np.pi - 1.0) * u) + np.array([0, 0, 0.6 + 1.0 * u])
            w = min(u / 0.15, 1.0)
            return land * (1 - w) + roll * w
        u = _ease(min((t - 1.8) / 1.0, 1.0))
        tuck = _roll_frame(rig, 2 * np.pi) + np.array([0, 0, 1.6])
        stand = rig.pose(root=(0, -0.45 * (1 - u), 1.6 + 0.2 * u), pitch=0.9 * (1 - u))
        return tuck * (1 - u) + stand * u
    return _g(ref, joints, bones, 2.8, f)


def blast_back(ref, joints, bones):
    """Explosion shockwave: lifted off the feet, flung backwards ~4 m, lands on the back and slides."""
    def f(rig, t):
        flung = rig.pose(lhand=(0.3, 1.65, 0.5), rhand=(-0.3, 1.6, 0.55), head_pitch=0.4,
                         lfoot=(0.14, 0.4, 0.45), rfoot=(-0.12, 0.3, 0.35))
        flat = rig.pose(lhand=(0.45, 1.3, -0.05), rhand=(-0.45, 1.3, 0.0), head_pitch=0.25, head_yaw=0.3)
        if t < 0.12:
            u = t / 0.12
            p = rig.pose(root=(0, -0.08 * u, -0.05 * u), pitch=-0.2 * u)
            return p
        if t < 0.95:
            u = (t - 0.12) / 0.83
            base = rig.pose(root=(0, -0.08, -0.05), pitch=-0.2) * (1 - min(u * 3, 1)) + flung * min(u * 3, 1)
            c = base[rig.j["Hips"]].copy()
            p = _whole(base, _rot("x", -1.5 * _ease(u)), c)
            y = 1.1 * np.sin(np.pi * min(u * 1.15, 1.0))
            p = p - np.array([0, c[1], 0]) + np.array([0, max(y, 0) + 0.2 + 0.7 * (1 - u), -3.6 * u])
            return _ground(p) if u > 0.85 else p
        u = min((t - 0.95) / 0.55, 1.0)
        c = flat[rig.j["Hips"]].copy()
        p = _ground(_whole(flung * (1 - u) + flat * u, _rot("x", -1.5), c))
        return p + np.array([0, 0, -3.6 - 0.7 * (1 - (1 - u) ** 2)])
    return _g(ref, joints, bones, 3.0, f)


def eject_roll(ref, joints, bones):
    """Bails out of a moving vehicle to the left (+X): dive, sideways roll along the ground, up to a knee."""
    def f(rig, t):
        if t < 0.2:
            u = t / 0.2
            return rig.pose(root=(0, -0.35 * u, 0), pitch=0.4 * u, lhand=(0.3, 1.0, 0.2), rhand=(-0.1, 1.0, 0.3))
        if t < 0.6:
            u = (t - 0.2) / 0.4
            p = _roll_frame(rig, 0.0)
            c = (p[rig.j["Hips"]] + p[rig.j["Spine1"]]) / 2
            p = _whole(p, _rot("z", -1.2 * u), c)
            return p + np.array([1.0 * u, 0.35 * np.sin(np.pi * u), 1.5 * u])
        if t < 1.7:
            u = (t - 0.6) / 1.1
            p = _tucked(rig)
            c = (p[rig.j["Hips"]] + p[rig.j["Spine1"]]) / 2
            p = _ground(_whole(p, _rot("z", -1.2 - (2 * np.pi - 1.2) * u), c))
            return p + np.array([1.0 + 1.2 * u, 0, 1.5 + 2.5 * (1 - (1 - u) ** 2)])
        u = _ease(min((t - 1.7) / 1.0, 1.0))
        tuck = _roll_frame(rig, 0.0) + np.array([2.2, 0, 4.0])
        stand = rig.pose(root=(2.2, -0.45 * (1 - u), 4.0 + 0.2 * u), pitch=0.9 * (1 - u))
        return tuck * (1 - u) + stand * u
    return _g(ref, joints, bones, 2.7, f)


# ---------------------------------------------------------------- vehicles (rigid-body "skeletons") and seated riders
def vehicle(ref, joints, bones, kind="car", speed="0"):
    """A car or bike as a 20-joint rigid body: Hips = chassis centre, Up/Legs = rear axle, Arms = front axle,
    Spine = nose (pitch reference). Root motion moves it `speed` m/s along +Z."""
    j = {n: i for i, n in enumerate(joints)}
    v = float(speed)
    half, length, h = (0.85, 1.35, 0.6) if kind == "car" else (0.16, 0.7, 0.55)
    secs = 1.0
    out = []
    for i in range(int(secs * FPS) + 1):
        t = i / FPS
        p = np.zeros((len(joints), 3))
        z = v * t
        buzz = (0.006 if kind == "car" else 0.004) * np.sin(2 * np.pi * 15 * t) * (1 if v == 0 else 0.6)
        base = np.array([0, h + buzz, z])
        for n in joints:
            p[j[n]] = base
        p[j["LeftUpLeg"]] = base + [half, -h + 0.32, -length]
        p[j["RightUpLeg"]] = base + [-half, -h + 0.32, -length]
        # tyre contact points sit on the floor (the library normalises clips so the feet touch y = 0)
        p[j["LeftLeg"]] = base + [half, -h + 0.32, -length]
        p[j["RightLeg"]] = base + [-half, -h + 0.32, -length]
        p[j["LeftFoot"]] = p[j["LeftToeBase"]] = base + [half, -h - buzz, -length]
        p[j["RightFoot"]] = p[j["RightToeBase"]] = base + [-half, -h - buzz, -length]
        p[j["LeftArm"]] = p[j["LeftForeArm"]] = p[j["LeftHand"]] = base + [half, -h + 0.32, length]
        p[j["RightArm"]] = p[j["RightForeArm"]] = p[j["RightHand"]] = base + [-half, -h + 0.32, length]
        p[j["Spine"]] = base + [0, 0, length]
        p[j["Spine1"]] = base + [0, 0.5, 0]
        p[j["Neck1"]] = base + [0, 0.5, 0.5]
        p[j["Head"]] = base + [0, 0.5, length]
        p[j["Head_End"]] = base + [0, 0.6, length]
        out.append(p)
    return np.stack(out)


def drive_steer(ref, joints, bones):
    """Seated driver, hands at ten-and-two; the viewer turns the wheel and leans with the vehicle's path."""
    def f(rig, t):
        F = _feet0(rig)
        return _seated(rig, F, pitch=-0.12, head_yaw=0.12 * _S(t, 5) * (abs(_S(t, 5)) > 0.8),
                       lhand=(0.19, 1.16, 0.42), rhand=(-0.19, 1.16, 0.42))
    return _g(ref, joints, bones, 5.0, f)


def passenger_idle(ref, joints, bones):
    def f(rig, t):
        F = _feet0(rig)
        bump = 0.025 * np.exp(-((t % 3.7) - 0.2) ** 2 / 0.01) - 0.02 * np.exp(-((t % 5.3) - 0.3) ** 2 / 0.008)
        look = float(_track([(0, 0), (1.5, 0), (2.2, 0.8), (3.8, 0.8), (4.4, -0.4), (5.6, -0.4), (6, 0)], t))
        return rig.pose(root=(0, -0.44 + bump, 0), pitch=-0.12 + bump * 2, head_yaw=look,
                        lfoot=F["Left"] + [0.04, 0, 0.42], rfoot=F["Right"] + [-0.04, 0, 0.42],
                        lhand=(0.14, 0.5 + 0.44, 0.3), rhand=(-0.14, 0.5 + 0.44, 0.3))
    return _g(ref, joints, bones, 6.0, f)


GENERATORS.update({
    "idle_phone": idle_phone, "idle_watch": idle_watch, "idle_arms_crossed": idle_arms_crossed, "idle_shift": idle_shift,
    "chat_a": _chat(True), "chat_b": _chat(False), "wipe_counter": wipe_counter, "sweep": sweep,
    "type_laptop": type_laptop, "read_book": read_book, "sing": sing, "think": think, "pace_circle": pace_circle,
    "flinch_shield": flinch_shield, "exhausted": exhausted, "wipe_sweat": wipe_sweat, "wake_up": wake_up,
    "calib_pose": calib_pose, "standin_turn": standin_turn, "scale_ref": scale_ref, "eyeline_sweep": eyeline_sweep,
    "settle": settle, "wind_down": wind_down, "stagger": stagger, "leap_roll": leap_roll, "high_fall": high_fall,
    "blast_back": blast_back, "eject_roll": eject_roll, "vehicle": vehicle, "drive_steer": drive_steer,
    "passenger_idle": passenger_idle,
})
