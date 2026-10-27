"""Scene plan schema shared by the planners, the agent pipeline and the viewer."""
from __future__ import annotations

from typing import Literal, Optional, Union

from pydantic import BaseModel, Field

ShotType = Literal["wide", "medium", "close_up", "two_shot", "over_shoulder",
                   "low_angle", "high_angle", "tracking", "pov"]
CameraMove = Literal["static", "push_in", "pull_out", "orbit", "crane_up", "handheld"]
TimeOfDay = Literal["day", "dusk", "night"]

PALETTE = ["#ff6b4a", "#4ac6ff", "#ffd24a", "#9b7bff", "#4affa8", "#ff4aa8",
           "#ff9f43", "#54a0ff", "#c8d6e5", "#a3cb38",
           "#8e9aa6", "#a69a8e", "#8ea69a", "#9a8ea6", "#a6a08e", "#8ea0a6"]  # muted tones for extras
MAX_CHARACTERS = 16  # ten named actors plus background extras
MAX_CAST = 10
Kind = Literal["human", "dog", "horse", "car", "bike"]
ObjectType = Literal["van", "door", "chair", "ladder", "button", "box", "barbell", "crate",
                     "wall", "curb", "gap", "beam", "cup", "suitcase"]


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
    kind: Kind = "human"
    tail: bool = False
    scale: float = Field(1.0, ge=0.5, le=1.6)
    weapon: Optional[str] = Field(None, description="Weapon held at the start: gun, rifle, sword, mace, bow")
    extra: bool = False  # background extra (crowd)
    standin: bool = False  # drawn as a grey low-poly stand-in


class SceneObject(BaseModel):
    id: str
    type: ObjectType
    x: float = 0.0
    z: float = 0.0
    yaw: float = 0.0  # radians, 0 = facing +Z
    length: float = 0.0  # walls: length in metres along the object's local X
    oriented: bool = True  # False: blocking turns it across the first actor's path (obstacles with no Facing)


class Layer(BaseModel):
    """An overlay that runs on top of whatever the body is doing (command scripts).

    look: head and chest track a target (an actor id, "camera", or a stage point).
    grip: a hand pose holding a prop (cup, suitcase) while the body keeps walking or idling.
    """
    type: Literal["look", "grip", "mood", "fx"]
    character: str
    target: Optional[str] = None
    point: Optional[list[float]] = None
    prop: Optional[str] = None  # grip: cup | suitcase; fx: explosion
    mood: Optional[str] = None  # sad | angry | anxious | stealthy
    start: float = 0.0
    duration: float = 3.0
    snap: bool = False  # a sudden head turn (startle) instead of a smooth one
    line: Optional[int] = None
    line_mode: Optional[str] = None
    line_offset: float = 0.0
    until_line: Optional[int] = None  # a grip lasts until this line starts (put it down)


class MoveTo(BaseModel):
    target: str = ""  # a character id or object id; empty when `point` is given
    distance: float = 1.0
    point: Optional[list[float]] = Field(None, description="[x, z] stage point in metres")
    away: bool = False  # flee / retreat: move `distance` metres away from target
    slot: int = 0  # position in a group approaching the same target
    of: int = 1
    strafe: Optional[str] = None  # "left" | "right": sidestep `distance` m while facing `target`
    through: str = ""  # obstacle id to traverse (curb, gap, beam)
    phase: str = ""  # "near": walk up to the obstacle; "far": cross it


class Beat(BaseModel):
    character: str
    start: float = Field(ge=0)
    duration: float = Field(gt=0, le=20)
    action: str = Field(description="What the character does, in plain words")
    clip: Optional[str] = Field(None, description="Motion clip id from the library")
    face: Optional[str] = Field(None, description="Character id to turn toward at the start of the beat")
    move_to: Optional[MoveTo] = None
    speed: float = Field(1.0, ge=0.4, le=2.0)
    weapon: Optional[str] = Field(None, description="Weapon in hand from this beat on ('none' = put away)")
    ride: Optional[str] = Field(None, description="Horse id the character rides during this beat")
    say: Optional[str] = Field(None, description="Dialogue line shown as a subtitle")
    face_point: Optional[list[float]] = None
    target_object: Optional[str] = None  # object the beat uses (door, button, van, ladder...)
    line: Optional[int] = None  # command-script line this beat came from (screenplay timing)
    line_mode: Optional[str] = None  # seq | with | at
    line_offset: float = 0.0  # planned start minus the line's planned start
    ease_in: float = 0.0  # seconds of acceleration from rest (time warp; same end point)
    ease_out: float = 0.0  # seconds of deceleration into the stop
    teleport: bool = False  # jump straight to move_to.point (the viewer dissolves the actor out and in)
    seat: int = 0  # ride beats: 0 = driver / rider, 1 = front passenger, 2-3 = rear seats


class Shot(BaseModel):
    start: float = Field(ge=0)
    duration: float = Field(gt=0)
    type: ShotType = "medium"
    subjects: list[str] = Field(default_factory=list)
    move: CameraMove = "static"
    lens_mm: float = Field(35, ge=12, le=135)
    note: str = ""
    fixed: bool = False  # an exact time from a command script ([At=5s]): never moved


class ScenePlan(BaseModel):
    title: str = "Untitled scene"
    logline: str = ""
    setting: Setting = Setting()
    characters: list[Character]
    beats: list[Beat]
    shots: list[Shot] = Field(default_factory=list)
    objects: list[SceneObject] = Field(default_factory=list)
    layers: list[Layer] = Field(default_factory=list)
    extras_trim: bool = True  # background extras stop when the principal action ends
    unmatched: list[str] = Field(default_factory=list, description="Actions no clip in the library can perform")


# ---------------------------------------------------------------- resolved
class ResolvedBeat(Beat):
    duration: float = Field(gt=0)  # idle fills and held poses can run past the planner's 20 s per-beat cap
    clip: str
    confidence: float
    alternatives: list[str] = Field(default_factory=list)
    x: float = 0.0
    z: float = 0.0
    yaw: float = 0.0  # radians, 0 = facing +Z
    auto: bool = False  # inserted by the continuity pass (idle fill)
    scale: float = 1.0


class ResolvedScene(BaseModel):
    title: str
    logline: str
    setting: Setting
    duration: float
    characters: list[Character]
    beats: list[ResolvedBeat]
    shots: list[Shot]
    objects: list[SceneObject] = Field(default_factory=list)
    layers: list[Layer] = Field(default_factory=list)
    unmatched: list[str] = Field(default_factory=list)


JSON_SCHEMA_HINT = """{
  "title": str, "logline": str,
  "setting": {"location": str, "time_of_day": "day"|"dusk"|"night", "mood": str},
  "characters": [{"id": "A", "name": str, "x": float|null, "z": float|null, "facing": "B"|null,
                  "kind": "human"|"dog"|"horse", "tail": bool, "scale": 0.5-1.6, "weapon": str|null}],
  "objects": [{"id": "van1", "type": "van"|"door"|"chair"|"ladder"|"button"|"box"|"barbell"|"crate", "x": m, "z": m, "yaw": rad}],
  "beats": [{"character": "A", "start": sec, "duration": sec, "action": str, "clip": clip_id,
             "face": char_id|null, "move_to": {"target": char_or_object_id, "point": [x, z]|null, "away": bool,
             "distance": metres}|null, "speed": 0.4-2.0, "weapon": str|null, "ride": horse_id|null, "say": str|null}],
  "shots": [{"start": sec, "duration": sec,
             "type": "wide"|"medium"|"close_up"|"two_shot"|"over_shoulder"|"low_angle"|"high_angle"|"tracking"|"pov",
             "subjects": [char_id], "move": "static"|"push_in"|"pull_out"|"orbit"|"crane_up"|"handheld",
             "lens_mm": 12-135, "note": str}]
}"""
