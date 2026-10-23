"""Scene plan schema shared by the planners, the agent pipeline and the viewer."""
from __future__ import annotations

from typing import Literal, Optional, Union

from pydantic import BaseModel, Field

ShotType = Literal["wide", "medium", "close_up", "two_shot", "over_shoulder",
                   "low_angle", "high_angle", "tracking", "pov"]
CameraMove = Literal["static", "push_in", "pull_out", "orbit", "crane_up", "handheld"]
TimeOfDay = Literal["day", "dusk", "night"]

PALETTE = ["#ff6b4a", "#4ac6ff", "#ffd24a", "#9b7bff", "#4affa8", "#ff4aa8"]


class Setting(BaseModel):
    location: str = "mocap stage"
    time_of_day: TimeOfDay = "day"
    mood: str = "neutral"


class Character(BaseModel):
    id: str = Field(description="Short id such as A, B, C")
    name: str
    color: Optional[str] = None
    x: Optional[float] = None
    z: Optional[float] = None
    facing: Optional[Union[str, float]] = Field(None, description="Character id to face, or yaw in degrees")


class MoveTo(BaseModel):
    target: str
    distance: float = 1.0


class Beat(BaseModel):
    character: str
    start: float = Field(ge=0)
    duration: float = Field(gt=0, le=20)
    action: str = Field(description="What the character does, in plain words")
    clip: Optional[str] = Field(None, description="Motion clip id from the library")
    face: Optional[str] = Field(None, description="Character id to turn toward at the start of the beat")
    move_to: Optional[MoveTo] = None
    speed: float = Field(1.0, ge=0.4, le=2.0)


class Shot(BaseModel):
    start: float = Field(ge=0)
    duration: float = Field(gt=0)
    type: ShotType = "medium"
    subjects: list[str] = Field(default_factory=list)
    move: CameraMove = "static"
    lens_mm: float = Field(35, ge=12, le=135)
    note: str = ""


class ScenePlan(BaseModel):
    title: str = "Untitled scene"
    logline: str = ""
    setting: Setting = Setting()
    characters: list[Character]
    beats: list[Beat]
    shots: list[Shot] = Field(default_factory=list)
    unmatched: list[str] = Field(default_factory=list, description="Actions no clip in the library can perform")


# ---------------------------------------------------------------- resolved
class ResolvedBeat(Beat):
    clip: str
    confidence: float
    alternatives: list[str] = Field(default_factory=list)
    x: float = 0.0
    z: float = 0.0
    yaw: float = 0.0  # radians, 0 = facing +Z
    auto: bool = False  # inserted by the continuity pass (idle fill)


class ResolvedScene(BaseModel):
    title: str
    logline: str
    setting: Setting
    duration: float
    characters: list[Character]
    beats: list[ResolvedBeat]
    shots: list[Shot]
    unmatched: list[str] = Field(default_factory=list)


JSON_SCHEMA_HINT = """{
  "title": str, "logline": str,
  "setting": {"location": str, "time_of_day": "day"|"dusk"|"night", "mood": str},
  "characters": [{"id": "A", "name": str, "x": float|null, "z": float|null, "facing": "B"|null}],
  "beats": [{"character": "A", "start": sec, "duration": sec, "action": str, "clip": clip_id,
             "face": char_id|null, "move_to": {"target": char_id, "distance": metres}|null, "speed": 0.4-2.0}],
  "shots": [{"start": sec, "duration": sec,
             "type": "wide"|"medium"|"close_up"|"two_shot"|"over_shoulder"|"low_angle"|"high_angle"|"tracking"|"pov",
             "subjects": [char_id], "move": "static"|"push_in"|"pull_out"|"orbit"|"crane_up"|"handheld",
             "lens_mm": 12-135, "note": str}]
}"""
