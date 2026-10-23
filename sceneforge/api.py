"""HTTP API (framework-agnostic handlers) used by both the FastAPI app and the stdlib dev server."""
from __future__ import annotations

import os
from pathlib import Path

from .director import direct
from .library import get_library
from .llm import DEFAULT_MODELS, KEY_ENV, available_provider

ROOT = Path(__file__).resolve().parents[1]
FRONTEND = ROOT / "frontend"
MOTIONS = ROOT / "data" / "motions"
MAX_PROMPT = 2000


def health() -> dict:
    return {"status": "ok", "clips": len(get_library().clips)}


def providers() -> dict:
    return {"default": available_provider("auto"),
            "available": ["offline"] + [p for p, k in KEY_ENV.items() if os.getenv(k)],
            "models": DEFAULT_MODELS}


def library() -> dict:
    lib = get_library()
    return {"fps": lib.fps, "joints": lib.joints, "bones": lib.bones, "source": lib.source,
            "clips": [{k: getattr(c, k) for k in ("id", "cmu", "category", "tags", "description", "duration",
                                                   "frames", "end_disp", "end_yaw", "source", "height")}
                      for c in lib.clips.values()]}


def direct_scene(body: dict) -> tuple[int, dict]:
    prompt = str(body.get("prompt", "")).strip()
    if not prompt:
        return 400, {"error": "prompt is required"}
    if len(prompt) > MAX_PROMPT:
        return 400, {"error": f"prompt too long (max {MAX_PROMPT} characters)"}
    provider = str(body.get("provider", "auto"))
    return 200, direct(prompt, provider=provider)


def motion_path(clip_id: str) -> Path | None:
    if clip_id not in get_library().clips:  # whitelist: no path traversal
        return None
    p = MOTIONS / f"{clip_id}.json"
    return p if p.exists() else None
