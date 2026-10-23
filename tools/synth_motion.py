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


def pushups(ref: np.ndarray, joints: list[str], bones: list[tuple[str, str]], reps: int = 5) -> np.ndarray:
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
