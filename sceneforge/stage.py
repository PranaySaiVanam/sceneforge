"""Stage geometry shared by the planners, blocking and (mirrored in JS) the viewer.

The capture volume is a 40 m x 40 m floor divided into 2 m grid cells: columns A-T run
along X (A at -X), rows 1-20 run along Z (row 1 at -Z, "north" is +Z). "Grid 4B" and
"B4" both mean column B, row 4. Points can also be given in metres from the centre as
"[x, y]", where y is the floor's second axis (Z in the viewer).
"""
from __future__ import annotations

import math
import re

CELL = 2.0
N = 20
HALF = CELL * N / 2  # 20 m
COLS = "ABCDEFGHIJKLMNOPQRST"
COMPASS = {"north": 0.0, "n": 0.0, "east": -math.pi / 2, "e": -math.pi / 2, "south": math.pi, "s": math.pi,
           "west": math.pi / 2, "w": math.pi / 2, "northeast": -math.pi / 4, "northwest": math.pi / 4,
           "southeast": -3 * math.pi / 4, "southwest": 3 * math.pi / 4}

_GRID = re.compile(r"\b(?:grid\s*|cell\s*)?(?:(\d{1,2})\s*([a-t])|([a-t])\s*(\d{1,2}))\b", re.I)
_POINT = re.compile(r"[\[(]\s*(-?\d+(?:\.\d+)?)\s*[, ]\s*(-?\d+(?:\.\d+)?)\s*[\])]")


def cell_center(col: str, row: int) -> tuple[float, float]:
    ci = COLS.index(col.upper())
    return -HALF + (ci + 0.5) * CELL, -HALF + (row - 0.5) * CELL


def parse_point(text: str, grid_hint: bool = False) -> tuple[float, float] | None:
    """'Grid 4B', 'B4', '[3, -2]', '(3,-2)' -> (x, z) in metres. Returns None if the text names no point.

    Bare cell names like 'B4' are only accepted when the text says 'grid'/'cell' or grid_hint is set,
    so names such as 'Agent 4' are never read as coordinates.
    """
    m = _POINT.search(text)
    if m:
        return float(m.group(1)), float(m.group(2))
    low = text.lower()
    explicit = "grid" in low or "cell" in low or grid_hint
    for m in _GRID.finditer(text):
        row, col = (m.group(1), m.group(2)) if m.group(1) else (m.group(4), m.group(3))
        row = int(row)
        if 1 <= row <= N and (explicit or re.fullmatch(r"\s*\d{1,2}[a-t]\s*|\s*[a-t]\d{1,2}\s*", text, re.I)):
            return cell_center(col, row)
    return None


def cell_name(x: float, z: float) -> str:
    ci = min(N - 1, max(0, int((x + HALF) // CELL)))
    ri = min(N - 1, max(0, int((z + HALF) // CELL)))
    return f"{ri + 1}{COLS[ci]}"
