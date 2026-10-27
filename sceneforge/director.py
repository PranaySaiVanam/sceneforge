"""SceneForge agent pipeline.

    prompt
      -> Director        (LLM planner with schema validation + repair; rule-based fallback)
      -> Motion Retriever (grounds every beat in a real mocap clip; confidence + alternatives)
      -> Continuity      (fixes references, overlaps, gaps; holds poses after falls)
      -> Blocking        (stage positions & facing, approach distances, root-motion integration)
      -> Cinematographer (contiguous shot coverage; fills gaps with director-style defaults)
      -> ResolvedScene + trace

Every step records what it did and every automatic fix, so the UI can show
the agent's reasoning and the evals can measure it.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

from .commands import is_command_script, plan_commands
from .library import MotionLibrary, get_library
from .llm import available_provider, plan_with_llm
from .planner_offline import plan_offline
from .schema import MAX_CHARACTERS, PALETTE, Character, Layer, ResolvedBeat, ResolvedScene, ScenePlan, Shot
from .stage import HALF, cell_name

MAX_SCENE = 120.0
HOLD_AFTER = {"fall_face", "fall_back", "crawl", "sit", "sit_stool",  # end pose is held, not reset to idle
              "sit_down", "sit_chair", "kneel", "crouch", "lie_down", "prone", "cower_low", "dog_sit", "ride",
              "collapse", "drive_steer", "passenger_idle"}
IDLE_OF = {"dog": "dog_idle", "horse": "horse_idle", "car": "car_idle", "bike": "bike_idle"}
LOW_POSES = {"sit_down", "sit_chair", "kneel", "crouch", "prone", "lie_down", "cower_low", "type_laptop"}  # settle first
WINDED_AFTER = {"brake", "run", "exhausted", "punch", "kick", "boxing", "punch_kick", "spin_kick", "karate_punch",
                "front_kick", "roundhouse", "side_kick", "sword", "mace_smash", "sword_duel", "leap_roll", "high_fall"}
SEAT = {  # vehicle-frame seat offsets (x = left, z = forward) for riders, by seat index
    "car": [(0.38, 0.25), (-0.38, 0.25), (0.38, -0.75), (-0.38, -0.75)],
    "bike": [(0.0, -0.1), (0.0, -0.55), (0.0, -0.55), (0.0, -0.55)],
}
# command scripts: an actor left on the ground keeps breathing / groaning instead of freezing
DOWN_IDLE = {"fall_back": "down_groan_fall_back", "fall_face": "down_groan_fall_face", "lie_down": "down_still_lie_down",
             "prone": "down_still_prone", "collapse": "down_still_collapse",
             "blast_back": "down_groan_blast_back"}
TURNS = ("turn_l90", "turn_r90", "turn_l135", "turn_r135", "turn_180")
TRAVERSE = {"step_over": 0.9, "duck_under": 1.0, "squeeze": 0.8}  # clip -> distance of the start mark before the obstacle
# solid footprints for collisions: (centre offset along facing, half-width along local X, half-depth along local Z)
FOOTPRINT = {"van": (0.1, 1.0, 2.3), "crate": (0.0, 0.45, 0.45), "box": (0.0, 0.3, 0.2)}
LOOPING = {"locomotion", "idle", "dance", "daily"}
# Clips that only read on camera if the partner is within reach (metres).
REACH = {"punch": 1.1, "kick": 1.3, "punch_kick": 1.2, "boxing": 1.4, "sword": 1.8, "spin_kick": 1.6,
         "handshake": 0.9, "high_five": 1.0, "push": 0.8, "angry": 1.8, "salsa": 1.0,
         "karate_punch": 1.1, "front_kick": 1.3, "roundhouse": 1.4, "side_kick": 1.4, "mace_smash": 1.6,
         "sword_duel": 1.9, "dog_attack": 0.9, "argue": 1.8}
HIT_REACTIONS = {"hit_react", "hit_gut", "fall_back", "fall_face", "block", "evade", "duck", "stumble"}
OBJECT_APPROACH = {"van": -2.7, "door": 0.0, "chair": 0.0, "ladder": 0.0, "button": 0.0, "box": 0.0,
                   "barbell": 0.0, "crate": 0.0, "wall": -0.8, "curb": 0.0, "gap": 0.0, "beam": 0.0}  # where on the object an actor stands (along its facing)


@dataclass
class Trace:
    steps: list[dict] = field(default_factory=list)

    def add(self, agent: str, summary: str, t0: float, **extra):
        self.steps.append({"agent": agent, "summary": summary, "ms": round((time.perf_counter() - t0) * 1000, 1),
                           **extra})


# --------------------------------------------------------------------------- 1
def step_director(prompt: str, provider: str, lib: MotionLibrary, trace: Trace) -> ScenePlan:
    t0 = time.perf_counter()
    if is_command_script(prompt):
        plan = plan_commands(prompt, lib)
        trace.add("Director", f"command script: {len(plan.characters)} actors, {len(plan.beats)} beats, "
                  f"{len(plan.objects)} set object(s)", t0, provider="commands")
        return plan
    chosen = available_provider(provider)
    if chosen != "offline":
        res = plan_with_llm(prompt, lib, chosen)
        if res.plan is not None:
            trace.add("Director", f"{res.provider}/{res.model} planned {len(res.plan.beats)} beats, "
                      f"{len(res.plan.shots)} shots in {res.attempts} attempt(s)", t0,
                      provider=res.provider, model=res.model, attempts=res.attempts,
                      tokens={"input": res.input_tokens, "output": res.output_tokens}, errors=res.errors)
            return res.plan
        plan = plan_offline(prompt, lib)
        trace.add("Director", f"{res.provider} failed ({res.errors[-1] if res.errors else 'unknown'}); "
                  f"fell back to rule-based planner", t0, provider="offline", fallback_from=res.provider,
                  errors=res.errors)
        return plan
    plan = plan_offline(prompt, lib)
    why = "no API key set" if provider in ("auto", None, "") else "offline mode selected"
    trace.add("Director", f"rule-based planner ({why}): {len(plan.characters)} characters, {len(plan.beats)} beats",
              t0, provider="offline")
    return plan


# --------------------------------------------------------------------------- 2
def step_retrieve(plan: ScenePlan, lib: MotionLibrary, trace: Trace) -> list[ResolvedBeat]:
    t0 = time.perf_counter()
    out, fixes = [], []
    kinds = {c.id: c.kind for c in plan.characters}
    for b in plan.beats:
        kind = kinds.get(b.character, "human")
        matches = lib.search(b.action, k=3, hint=b.clip, kind=kind)
        if b.clip in lib.clips and lib.clips[b.clip].kind != kind and not (b.clip == "ride" and kind == "human"):
            fixes.append(f"'{b.clip}' is a {lib.clips[b.clip].kind} clip but {b.character} is a {kind}")
            b.clip = None
        if b.clip in lib.clips:
            clip_id = b.clip
            conf = next((m.confidence for m in matches if m.clip.id == clip_id), 0.5)
        elif matches:
            clip_id, conf = matches[0].clip.id, matches[0].confidence
            if b.clip:
                fixes.append(f"unknown clip '{b.clip}' -> {clip_id}")
        else:
            clip_id, conf = IDLE_OF.get(kind, "idle_wait"), 0.0
            fixes.append(f"no motion for '{b.action}' -> idle")
            plan.unmatched.append(b.action)
        rb = ResolvedBeat(**{**b.model_dump(), "clip": clip_id}, confidence=round(conf, 2),
                          alternatives=[m.clip.id for m in matches if m.clip.id != clip_id][:2])
        object.__setattr__(rb, "_orig_start", b.start)
        out.append(rb)
    low = [b for b in out if b.confidence < 0.35]
    for a in plan.unmatched:
        fixes.append(f"no clip in the library performs '{a}', so it was left out (add one: docs 9.6)")
    keyframed = sorted({b.clip for b in out if lib.clips[b.clip].source == "keyframed"})
    if keyframed:
        fixes.append(f"keyframed, not mocap: {', '.join(keyframed)}")
    studio = sorted({b.clip for b in out if lib.clips[b.clip].source == "studio"})
    if studio:
        fixes.append(f"studio mocap: {', '.join(studio)}")
    trace.add("Motion Retriever", f"grounded {len(out)} beats in mocap clips"
              + (f"; {len(low)} low-confidence" if low else "")
              + (f"; {len(plan.unmatched)} action(s) not in the library" if plan.unmatched else ""), t0, fixes=fixes,
              matches=[{"action": b.action, "clip": b.clip, "confidence": b.confidence,
                        "alternatives": b.alternatives} for b in out])
    return out


# --------------------------------------------------------------------------- 3
def step_continuity(plan: ScenePlan, beats: list[ResolvedBeat], lib: MotionLibrary, trace: Trace, fill: bool = True):
    t0 = time.perf_counter()
    fixes = []
    chars = plan.characters[:MAX_CHARACTERS] or [Character(id="A", name="Performer")]
    ids = {c.id for c in chars}
    obj_ids = {o.id for o in plan.objects}
    kinds = {c.id: c.kind for c in chars}
    for i, c in enumerate(chars):
        c.color = c.color or PALETTE[i % len(PALETTE)]
    for b in beats:
        if b.character not in ids:
            fixes.append(f"beat '{b.action}' referenced unknown character {b.character} -> {chars[0].id}")
            b.character = chars[0].id
        if b.face and b.face not in ids or b.face == b.character:
            b.face = None
        if b.move_to and b.move_to.point is None and (b.move_to.target not in ids | obj_ids or b.move_to.target == b.character):
            b.move_to = None
        if b.ride and (b.ride not in ids or kinds.get(b.ride) not in ("horse", "car", "bike")):
            b.ride = None
    if not fill:  # command scripts: timing is resolved line by line in blocking, gaps filled afterwards
        for b in beats:
            b.scale = next((c.scale for c in chars if c.id == b.character), 1.0)
        duration = min(MAX_SCENE, max([b.start + b.duration for b in beats] + [4.0]) + 0.6)
        trace.add("Continuity", f"{len(fixes)} fix(es); screenplay timing (lines run in order)", t0, fixes=fixes)
        return chars, sorted(beats, key=lambda b: (b.start, b.character)), round(duration, 2)
    # per-character: sort, remove overlaps, fill gaps
    final: list[ResolvedBeat] = []
    for c in chars:
        mine = sorted([b for b in beats if b.character == c.id], key=lambda b: b.start)
        t = 0.0
        for b in mine:
            if b.start < t - 0.05:
                fixes.append(f"{c.name}: '{b.action}' overlapped previous beat; moved {t - b.start:.1f}s later")
                b.start = round(t, 2)
            if b.start > MAX_SCENE:
                fixes.append(f"{c.name}: dropped '{b.action}' (beyond {MAX_SCENE:.0f}s)")
                continue
            if b.start - t > 0.05:
                prev = final[-1] if final and final[-1].character == c.id else None
                if prev and prev.clip in HOLD_AFTER:
                    prev.duration = round(b.start - prev.start, 2)
                else:
                    final.append(_idle(c.id, t, b.start - t, c.kind))
            final.append(b)
            t = b.start + b.duration
        c._end = t  # type: ignore[attr-defined]
    duration = min(MAX_SCENE, max([b.start + b.duration for b in final] + [4.0]) + 0.6)
    for c in chars:  # hold or idle to the end of the scene
        mine = [b for b in final if b.character == c.id]
        end = mine[-1].start + mine[-1].duration if mine else 0.0
        if duration - end > 0.05:
            if mine and mine[-1].clip in HOLD_AFTER:
                mine[-1].duration = round(duration - mine[-1].start, 2)
            else:
                final.append(_idle(c.id, end, duration - end, c.kind))
    for b in final:
        b.scale = next((c.scale for c in chars if c.id == b.character), 1.0)
    final.sort(key=lambda b: (b.start, b.character))
    trace.add("Continuity", f"{len(fixes)} fix(es); idle/hold fill for {sum(b.auto for b in final)} gap(s); "
              f"scene {duration:.1f}s", t0, fixes=fixes)
    return chars, final, round(duration, 2)


def _idle(cid: str, start: float, dur: float, kind: str = "human", scale: float = 1.0) -> ResolvedBeat:
    return ResolvedBeat(character=cid, start=round(start, 2), duration=round(max(dur, 0.05), 2),
                        action="idle", clip=IDLE_OF.get(kind, "idle_wait"),
                        confidence=1.0, auto=True, scale=scale)


# --------------------------------------------------------------------------- 4
def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def step_blocking(chars: list[Character], beats: list[ResolvedBeat], lib: MotionLibrary, trace: Trace,
                  objects=(), layers: list[Layer] | None = None):
    """Assigns every beat a start position/heading by integrating mocap root motion."""
    t0 = time.perf_counter()
    n = len(chars)
    notes = []
    objs = {o.id: o for o in objects}
    state: dict[str, tuple[float, float, float]] = {}
    for i, c in enumerate(chars):
        if c.x is not None and c.z is not None:
            x, z = c.x, c.z
        elif n == 1:
            x, z = 0.0, 0.0
        elif n == 2:
            x, z = (-2.0 if i == 0 else 2.0), 0.0
        else:
            ang = 2 * math.pi * i / n
            r = 2.6 if n <= 4 else 0.62 * n
            x, z = r * math.sin(ang), r * math.cos(ang)
        state[c.id] = (x, z, 0.0)
    for c in chars:  # initial facing
        x, z, _ = state[c.id]
        if isinstance(c.facing, (int, float)):
            yaw = math.radians(c.facing)
        elif isinstance(c.facing, str) and c.facing in state and c.facing != c.id:
            tx, tz, _ = state[c.facing]
            yaw = math.atan2(tx - x, tz - z)
        elif n >= 2:
            yaw = math.atan2(-x, -z)  # face the centre of the stage
        else:
            yaw = 0.0
        state[c.id] = (x, z, yaw)

    placed: dict[str, list[ResolvedBeat]] = {c.id: [] for c in chars}

    def obj_point(oid: str) -> tuple[float, float]:
        o = objs[oid]
        off = OBJECT_APPROACH.get(o.type, 0.0)
        return o.x + math.sin(o.yaw) * off, o.z + math.cos(o.yaw) * off

    def pos_at(cid: str, t: float) -> tuple[float, float]:
        if cid in objs:
            return obj_point(cid)
        for b in reversed(placed.get(cid, [])):
            if b.start <= t + 1e-6:
                if b.ride:
                    return pos_at(b.ride, t)
                return _transform(lib, b, min(t - b.start, b.duration))[:2]
        return state[cid][:2] if cid in state else (0.0, 0.0)

    def yaw_at(cid: str, t: float) -> float:
        for b in reversed(placed.get(cid, [])):
            if b.start <= t + 1e-6:
                return _transform(lib, b, min(t - b.start, b.duration))[2]
        return state[cid][2] if cid in state else 0.0

    kinds = {c.id: c.kind for c in chars}

    def approach_clip(cid: str, far: bool) -> str:
        k = kinds.get(cid, "human")
        if k == "dog":
            return "dog_run" if far else "dog_walk"
        if k == "horse":
            return "horse_gallop" if far else "horse_walk"
        if k in ("car", "bike"):
            return f"{k}_fast" if far else f"{k}_slow"
        return "run" if far else "walk"

    # ---------------------------------------------------------------- command-script polish
    layers = layers if layers is not None else []
    command = any(b.line is not None for b in beats) or any(l.line is not None for l in layers)
    cross_side: dict = {}

    def solids(ignore: str = ""):
        out = []
        for o in objs.values():
            if o.id == ignore:
                continue
            if o.type == "wall":
                out.append((o.x, o.z, o.yaw, max(o.length, 0.5) / 2, 0.15, "the wall"))
            elif o.type == "gap":  # two panels either side of a 0.9 m opening
                c_, s_ = math.cos(o.yaw), math.sin(o.yaw)
                for sg in (-1, 1):
                    out.append((o.x + sg * 1.45 * c_, o.z - sg * 1.45 * s_, o.yaw, 1.0, 0.15, "the wall"))
            elif o.type in FOOTPRINT:
                off, hx, hz = FOOTPRINT[o.type]
                out.append((o.x + math.sin(o.yaw) * off, o.z + math.cos(o.yaw) * off, o.yaw, hx, hz, f"the {o.type}"))
        return out

    def collide(x, z, px, pz, ignore=""):
        """First solid thing (or the stage edge) on the straight path -> (stop_x, stop_z, what) or None."""
        dx, dz = px - x, pz - z
        L = math.hypot(dx, dz)
        if L < 1e-6:
            return None
        best, what = 1.0, None
        lim = HALF - 0.4
        for v0, d in ((x, dx), (z, dz)):
            if d and abs(v0 + d) > lim:
                tt = ((lim if d > 0 else -lim) - v0) / d
                if 0 <= tt < best:
                    best, what = tt, "the edge of the stage"
        pad = 0.35
        for cx, cz, yw, hx, hz, name in solids(ignore):
            X, Z = (math.cos(yw), -math.sin(yw)), (math.sin(yw), math.cos(yw))
            sx, sz = (x - cx) * X[0] + (z - cz) * X[1], (x - cx) * Z[0] + (z - cz) * Z[1]
            vx, vz = dx * X[0] + dz * X[1], dx * Z[0] + dz * Z[1]
            if abs(sx) <= hx + pad and abs(sz) <= hz + pad:
                continue  # already touching it: let them walk away
            lo, hi, ok = 0.0, 1.0, True
            for s0, v, h in ((sx, vx, hx + pad), (sz, vz, hz + pad)):
                if abs(v) < 1e-9:
                    ok = abs(s0) <= h
                else:
                    a_, b_ = sorted(((-h - s0) / v, (h - s0) / v))
                    lo, hi = max(lo, a_), min(hi, b_)
                    ok = ok and lo <= hi
                if not ok:
                    break
            if ok and lo < best:
                best, what = lo, name
        if what is None:
            return None
        stop = max(best * L - 0.15, 0.0)
        return x + dx / L * stop, z + dz / L * stop, what

    def polishable(b: ResolvedBeat) -> bool:
        return command and b.line is not None and kinds.get(b.character, "human") == "human" and not b.ride

    def shift(cid: str, after: float, delta: float, pending: list) -> None:
        for later in pending:
            if later.character == cid and later.start >= after - 1e-6:
                later.start = round(later.start + delta, 2)

    def insert(b: ResolvedBeat, clip: str, start: float, x: float, z: float, yaw: float, action: str) -> ResolvedBeat:
        nb = ResolvedBeat(character=b.character, start=round(start, 2), duration=round(lib.clips[clip].duration, 2),
                          action=action, clip=clip, confidence=1.0, line=b.line, line_mode=b.line_mode, scale=b.scale)
        nb.x, nb.z, nb.yaw = round(x, 3), round(z, 3), round(yaw, 4)
        beats.append(nb)
        placed[b.character].append(nb)
        state[b.character] = _transform(lib, nb, nb.duration)
        return nb

    def pivot(b: ResolvedBeat, x: float, z: float, cur: float, want: float, pending: list):
        """A foot-shuffle turn before the beat when the body has to swing round more than 60 degrees."""
        diff = _wrap(want - cur)
        prev = placed[b.character][-1] if placed[b.character] else None
        grounded = prev is not None and (_holds(prev.clip) or prev.clip.startswith(("get_up", "stand_up")))
        if abs(diff) < math.radians(60) or grounded or b.clip.startswith(("crawl", "down_")):
            return x, z, cur
        clip = min(TURNS, key=lambda c: abs(_wrap(lib.clips[c].end_yaw - diff)))
        nb = insert(b, clip, b.start, x, z, cur, f"pivots {'left' if diff > 0 else 'right'} {abs(math.degrees(diff)):.0f} deg")
        shift(b.character, b.start, nb.duration, pending)
        b.start = round(b.start + nb.duration, 2)
        notes.append(f"{b.character}: {clip} ({abs(math.degrees(diff)):.0f} deg) before '{b.action[:30]}'")
        return state[b.character]

    def from_rest(b: ResolvedBeat) -> bool:
        prev = placed[b.character][-1] if placed[b.character] else None
        return prev is None or not lib.clips[prev.clip].is_locomotion or prev.start + prev.duration < b.start - 0.1 \
            or prev.clip in TURNS

    def explode(lay: Layer, li: int) -> None:
        """Everyone reacts to an explosion by distance: blown off their feet, shielding, or a startled look."""
        ex, ez = lay.point
        te = lay.start
        for c in chars:
            if c.kind != "human":
                continue
            px, pz = pos_at(c.id, te)
            d = math.hypot(px - ex, pz - ez)
            mine = placed[c.id]
            if mine and mine[-1].ride:  # riding: a startled look, still in the seat
                layers.append(Layer(type="look", character=c.id, point=[ex, ez], start=te, duration=2.5, snap=True))
                continue
            interruptible = mine and (mine[-1].start < te or mine[-1].line is None)
            if interruptible and te < mine[-1].start + mine[-1].duration:
                cut = mine[-1]
                cut.duration = round(max(te - cut.start, 0.05), 2)
                state[c.id] = _transform(lib, cut, cut.duration)
            elif mine and mine[-1].start >= te:
                continue  # already doing something scripted after the blast
            px, pz, cy = state[c.id]
            face = math.atan2(ex - px, ez - pz) if d > 0.2 else cy
            clip = "blast_back" if d < 4.5 else "flinch_shield" if d < 10 else "startle"
            nb = ResolvedBeat(character=c.id, start=round(te + d / 60, 2), duration=round(lib.clips[clip].duration, 2),
                              action=f"reacts to the explosion ({d:.0f} m)", clip=clip, confidence=1.0, line=li,
                              line_mode="at", scale=c.scale)
            nb.x, nb.z, nb.yaw = round(px, 3), round(pz, 3), round(face if clip == "blast_back" else cy, 4)
            beats.append(nb)
            placed[c.id].append(nb)
            state[c.id] = _transform(lib, nb, nb.duration)
            if clip != "blast_back":
                layers.append(Layer(type="look", character=c.id, point=[ex, ez], start=nb.start, duration=2.5, snap=True))
        notes.append(f"explosion at {cell_name(ex, ez)} ({te:.1f}s): reactions by distance")

    def place_one(b: ResolvedBeat, pending: list) -> None:
        x, z, yaw = state[b.character]
        prev_beat = placed[b.character][-1] if placed[b.character] else None
        if b.teleport and b.move_to and b.move_to.point is not None:
            b.x, b.z, b.yaw = round(b.move_to.point[0], 3), round(b.move_to.point[1], 3), round(yaw, 4)
            placed[b.character].append(b)
            state[b.character] = _transform(lib, b, b.duration)
            notes.append(f"{b.character}: teleported to {cell_name(b.x, b.z)}")
            return
        if prev_beat is not None and prev_beat.ride and not b.ride:
            hx, hz = pos_at(prev_beat.ride, b.start)
            hy = yaw_at(prev_beat.ride, b.start)
            mk = kinds.get(prev_beat.ride, "horse")
            if mk == "horse":  # dismount: step down on the horse's left side
                x, z, yaw = hx + math.cos(hy) * 0.95, hz - math.sin(hy) * 0.95, hy - math.pi / 2
            else:  # out of a car / off a bike on the left; a bail-out keeps the vehicle's heading and momentum
                sx, sz = SEAT.get(mk, SEAT["car"])[min(prev_beat.seat, 3)]
                side = 1.0 if sx >= 0 else -1.0
                off = (1.25 if mk == "car" else 0.7) * side
                x, z = hx + math.cos(hy) * off + math.sin(hy) * sz, hz - math.sin(hy) * off + math.cos(hy) * sz
                yaw = hy if b.clip == "eject_roll" else hy + side * math.pi / 2
                if b.clip == "eject_roll" and side < 0:  # passenger side: the dive is mirrored by turning round
                    yaw = hy + math.pi
                b.x, b.z, b.yaw = round(x, 3), round(z, 3), round(yaw, 4)
                placed[b.character].append(b)
                state[b.character] = _transform(lib, b, b.duration)
                return
        if polishable(b) and b.clip in LOW_POSES and prev_beat is not None and lib.clips[prev_beat.clip].is_locomotion \
                and prev_beat.start + prev_beat.duration >= b.start - 0.1:
            # loop-to-loop shuffle: re-plant the feet before sitting / kneeling / going prone
            nb = insert(b, "settle", b.start, x, z, yaw, "settles the feet")
            shift(b.character, b.start, nb.duration, pending)
            b.start = round(b.start + nb.duration, 2)
            x, z, yaw = state[b.character]
        if b.ride:
            hx, hz = pos_at(b.ride, b.start)
            b.x, b.z, b.yaw = round(hx, 3), round(hz, 3), round(yaw_at(b.ride, b.start), 4)
            placed[b.character].append(b)
            state[b.character] = (b.x, b.z, b.yaw)
            return
        if b.move_to and b.move_to.through and b.move_to.through in objs:
            o = objs[b.move_to.through]
            ax, az = math.sin(o.yaw), math.cos(o.yaw)
            key = (b.character, o.id)
            if b.move_to.phase == "near":  # walk to the start mark on this actor's side of the obstacle
                if not o.oriented and math.hypot(o.x - x, o.z - z) > 0.3:
                    o.yaw = round(math.atan2(o.x - x, o.z - z), 4)  # laid across this actor's path
                    o.oriented = True
                    ax, az = math.sin(o.yaw), math.cos(o.yaw)
                side = 1.0 if (x - o.x) * ax + (z - o.z) * az >= 0 else -1.0
                cross_side[key] = side
                nxt = next((p for p in pending if p.character == b.character and p.move_to
                            and p.move_to.through == o.id), None)
                lead = TRAVERSE.get(nxt.clip if nxt else "", 0.9)
                b.move_to = b.move_to.model_copy(update={"point": [o.x + ax * side * lead, o.z + az * side * lead],
                                                         "distance": 0.05, "target": ""})
            else:  # cross it: the traversal clip's own root motion carries them to the far side
                side = cross_side.get(key, 1.0)
                ex, ez, _ = lib.sample_root(b.clip, lib.clips[b.clip].duration)
                want = math.atan2(-ax * side, -az * side) - math.atan2(ex, ez)
                if polishable(b):
                    x, z, yaw = pivot(b, x, z, yaw, want, pending)
                b.x, b.z, b.yaw = round(x, 3), round(z, 3), round(want, 4)
                placed[b.character].append(b)
                state[b.character] = _transform(lib, b, b.duration)
                return
        strafe_look = None
        if b.move_to and b.move_to.strafe:
            mv = b.move_to
            if mv.target:
                tx, tz = pos_at(mv.target, b.start)
                f = math.atan2(tx - x, tz - z)
                strafe_look = mv.target
            else:
                f = yaw
            sg = 1.0 if mv.strafe == "left" else -1.0
            b.move_to = mv.model_copy(update={"point": [x + sg * math.cos(f) * mv.distance, z - sg * math.sin(f) * mv.distance],
                                              "distance": 0.1, "target": "", "strafe": None})
        if b.move_to and (b.move_to.point is not None or b.move_to.away or b.move_to.of > 1):
            # resolve a point destination: a stage point, a spot away from someone, or a slot around a target
            mv = b.move_to
            if mv.away and mv.target:
                tx, tz = pos_at(mv.target, b.start)
                dx, dz = x - tx, z - tz
                d = math.hypot(dx, dz) or 1.0
                mv.point = [x + dx / d * mv.distance, z + dz / d * mv.distance]
                mv.distance = 0.2
                if b.face is None:
                    yaw = math.atan2(dx, dz)
            elif mv.of > 1 and mv.target and mv.point is None:
                tx, tz = pos_at(mv.target, b.start)
                base = math.atan2(x - tx, z - tz)
                spread = min(2 * math.pi, 0.75 * mv.of)
                ang = base + (mv.slot - (mv.of - 1) / 2) * spread / max(mv.of, 1)
                mv.point = [tx + math.sin(ang) * mv.distance, tz + math.cos(ang) * mv.distance]
                mv.distance = 0.2
            b.move_to = mv
            if b.move_to.point is not None and lib.clips[b.clip].is_locomotion:
                px, pz = b.move_to.point
                back = b.clip in ("walk_back", "walk_crouch_back")
                clip = lib.clips[b.clip]
                bumped = brake = None
                if polishable(b):
                    hit = collide(x, z, px, pz, ignore=b.move_to.through)
                    if hit:
                        px, pz, bumped = hit
                        b.move_to = b.move_to.model_copy(update={"point": [round(px, 3), round(pz, 3)], "distance": 0.0})
                    if math.hypot(px - x, pz - z) > 0.5:
                        ex0, ez0, _ = lib.sample_root(b.clip, max(b.duration, 1.0) * b.speed)
                        tr0 = math.atan2(ex0, ez0) if math.hypot(ex0, ez0) > 0.3 else (math.pi if back else 0.0)
                        rest = from_rest(b)
                        x, z, yaw = pivot(b, x, z, yaw, math.atan2(px - x, pz - z) - tr0, pending)
                        rest = rest or placed[b.character][-1].clip in TURNS
                        mps = math.hypot(*clip.end_disp) / max(clip.duration, 1e-3) * b.speed
                        far = math.hypot(px - x, pz - z)
                        if mps > 2.4 and far > 3.5 and not back:
                            if rest:  # explosive first steps out of a standstill
                                insert(b, "push_off", b.start, x, z, math.atan2(px - x, pz - z), "pushes off into a run")
                                d0 = lib.clips["push_off"].duration
                                shift(b.character, b.start, d0, pending)
                                b.start = round(b.start + d0, 2)
                                x, z, _ = state[b.character]
                            if not bumped:  # stop short by the brake's slide
                                ux, uz = (px - x) / far, (pz - z) / far
                                px, pz = px - ux * 0.7, pz - uz * 0.7
                                brake = math.atan2(ux, uz)
                            b.ease_out = 0.15
                        else:
                            b.ease_in = 0.35 if rest else 0.0
                            b.ease_out = 0.3
                yaw = math.atan2(px - x, pz - z) + (math.pi if back else 0.0)
                gap = max(math.hypot(px - x, pz - z) - b.move_to.distance, 0.0)
                speed = max(math.hypot(*clip.end_disp) / max(clip.duration, 1e-3), 0.2) * b.speed * b.scale
                old = b.duration
                b.duration = round(min(max(gap / speed, 0.4), 20.0), 2)
                delta = b.duration - old
                if abs(delta) > 0.3:
                    notes.append(f"{b.character}: '{b.action[:40]}' retimed {old:.1f}s -> {b.duration:.1f}s to arrive")
                if delta > 0 or (command and b.line is not None):  # command lines: the next step follows on
                    for later in pending:
                        if later.character == b.character and later.start >= b.start:
                            later.start = round(later.start + delta, 2)
                # aim the clip's own travel direction at the destination (a back-walk travels along -Z)
                ex, ez, _ = lib.sample_root(b.clip, b.duration * b.speed)
                travel = math.atan2(ex, ez) if math.hypot(ex, ez) > 0.3 else (math.pi if back else 0.0)
                yaw = math.atan2(px - x, pz - z) - travel
                if b.clip.startswith("crawl_"):  # on the belly: keep crawling the way the body already points
                    yaw = state[b.character][2]
                b.x, b.z, b.yaw = round(x, 3), round(z, 3), round(yaw, 4)
                placed[b.character].append(b)
                state[b.character] = _transform(lib, b, b.duration)
                end = b.start + b.duration
                if strafe_look:  # eyes stay locked on the target while the feet carry them sideways
                    layers.append(Layer(type="look", character=b.character, target=strafe_look, start=b.start,
                                        duration=b.duration))
                if brake is not None:
                    ex_, ez_, _ = state[b.character]
                    nb = insert(b, "brake", end, ex_, ez_, brake, "plants the feet and brakes")
                    shift(b.character, end, nb.duration, pending)
                elif bumped:
                    ex_, ez_, _ = state[b.character]
                    nb = insert(b, "bump", end, ex_, ez_, math.atan2(px - b.x, pz - b.z), f"bumps into {bumped}")
                    shift(b.character, end, nb.duration, pending)
                    b.ease_out = 0.0
                    notes.append(f"{b.character}: path ran into {bumped}; stopped at {cell_name(ex_, ez_)} with a bump")
                return
        if b.face_point and not b.face and not b.move_to:
            yaw = math.atan2(b.face_point[0] - x, b.face_point[1] - z)
        if b.face and b.clip in REACH and not getattr(b, "_engaged", False):
            tx, tz = pos_at(b.face, b.start)
            d = math.hypot(tx - x, tz - z)
            reach = REACH[b.clip]
            if d > reach + 0.5:  # out of reach: insert an approach and push the exchange back
                walk = approach_clip(b.character, d - reach > 4.5)
                c = lib.clips[walk]
                est = (d - reach) / max(math.hypot(*c.end_disp) / c.duration, 0.2)
                delta = round(min(max(est, 0.4), 12.0), 2)
                approach = ResolvedBeat(character=b.character, start=b.start, duration=delta,
                                        action=f"closes distance to {b.face}", clip=walk, confidence=1.0,
                                        move_to={"target": b.face, "distance": reach}, face=b.face,
                                        line=b.line, line_mode=b.line_mode, scale=b.scale)
                for later in pending:
                    if later.character in (b.character, b.face) and later.start >= b.start - 1e-6:
                        later.start = round(later.start + delta, 2)
                b.start = round(b.start + delta, 2)
                object.__setattr__(b, "_engaged", True)
                pending += [approach, b]
                beats.append(approach)
                notes.append(f"{b.character} was {d:.1f} m from {b.face} for '{b.clip}' (reach {reach} m): "
                             f"inserted {delta:.1f}s {walk} approach, shifted the exchange")
                return
        if b.move_to and polishable(b) and lib.clips[b.clip].is_locomotion and not b.move_to.away:
            tx, tz = pos_at(b.move_to.target, b.start)
            if math.hypot(tx - x, tz - z) - b.move_to.distance > 0.6:
                ex0, ez0, _ = lib.sample_root(b.clip, max(b.duration, 1.0) * b.speed)
                tr0 = math.atan2(ex0, ez0) if math.hypot(ex0, ez0) > 0.3 else 0.0
                rest = from_rest(b)
                x, z, yaw = pivot(b, x, z, yaw, math.atan2(tx - x, tz - z) - tr0, pending)
                b.ease_in = 0.35 if (rest or placed[b.character][-1].clip in TURNS) else 0.0
                b.ease_out = 0.3
        elif polishable(b) and not b.move_to and (b.face or b.face_point) and \
                lib.clips[b.clip].category not in ("combat", "reaction") and b.clip not in TURNS and b.clip not in TRAVERSE:
            if b.face:
                tx, tz = pos_at(b.face, b.start)
            else:
                tx, tz = b.face_point
            if math.hypot(tx - x, tz - z) > 0.3:
                x, z, yaw = pivot(b, x, z, yaw, math.atan2(tx - x, tz - z), pending)
        if b.face or b.move_to:
            tid = b.move_to.target if b.move_to else b.face
            tx, tz = pos_at(tid, b.start)
            if math.hypot(tx - x, tz - z) > 0.05:
                yaw = math.atan2(tx - x, tz - z)
            if b.move_to and lib.clips[b.clip].is_locomotion:
                clip = lib.clips[b.clip]
                gap = math.hypot(tx - x, tz - z) - b.move_to.distance
                speed = max(math.hypot(*clip.end_disp) / max(clip.duration, 1e-3), 0.2) * b.speed
                old = b.duration
                b.duration = round(min(max(gap / speed, 0.4), 12.0), 2)
                delta = b.duration - old
                if abs(delta) > 0.3:
                    notes.append(f"{b.character}: '{b.action}' retimed {old:.1f}s -> {b.duration:.1f}s "
                                 f"to stop {b.move_to.distance:.1f} m from {tid}")
                if delta > 0 or (command and b.line is not None):  # push (or pull) this character's later beats
                    for later in pending:
                        if later.character == b.character and later.start >= b.start:
                            later.start = round(later.start + delta, 2)
            if b.move_to:  # correct for the clip's own sideways drift so the approach lands on target
                ex, ez, _ = lib.sample_root(b.clip, b.duration * b.speed)
                if math.hypot(ex, ez) > 0.3:
                    yaw -= math.atan2(ex, ez)
            sy = lib.clips[b.clip].strike_yaw
            if b.face and sy is not None and not b.move_to:  # aim the strike, not the chest (side kicks)
                yaw -= sy
        b.x, b.z, b.yaw = round(x, 3), round(z, 3), round(yaw, 4)
        placed[b.character].append(b)
        state[b.character] = _transform(lib, b, b.duration)


    start_of: dict[int, float] = {}
    if command:
        unlined = sorted([b for b in beats if b.line is None], key=lambda b: (b.start, b.character))
        while unlined:  # default "waits" beats for actors with no lines of their own
            b = unlined.pop(0)
            place_one(b, unlined)
        # command script: screenplay timing. A line starts when the lines before it have really finished
        # (after travel is measured), [With=prev] lines start with the previous line, [At=] lines are fixed.
        lines = sorted({b.line for b in beats if b.line is not None} | {l.line for l in layers if l.line is not None})
        end_of = {}
        actor_free: dict[str, float] = {}
        for li in lines:
            group = [b for b in beats if b.line == li]
            lgroup = [l for l in layers if l.line == li]
            mode = (group or lgroup)[0].line_mode or "seq"
            planned = min(b.start - b.line_offset for b in group + lgroup)
            if mode in ("at", "bg"):
                ls = planned
            elif mode == "with" and start_of:
                ls = start_of[max(start_of)]
            else:
                ls = max(end_of.values(), default=0.0)
            start_of[li] = ls
            for b in group:
                b.start = round(ls + b.line_offset, 2)
            for cid in {b.character for b in group}:  # an actor still busy pushes their part of the line
                mine = sorted([b for b in group if b.character == cid], key=lambda b: b.start)
                lag = actor_free.get(cid, 0.0) - mine[0].start
                busy = placed[cid][-1] if placed.get(cid) else None
                hit = mine[0].clip in HIT_REACTIONS
                if lag > 0.05 and hit and busy is not None and lib.clips[busy.clip].is_locomotion:
                    # struck while walking: the walk is cut short where they are, and they react there
                    busy.duration = round(max(mine[0].start - busy.start, 0.3), 2)
                    state[cid] = _transform(lib, busy, busy.duration)
                    actor_free[cid] = busy.start + busy.duration
                    notes.append(f"{cid}: '{busy.action[:30]}' cut short at {mine[0].start:.1f}s by a hit")
                    lag = actor_free[cid] - mine[0].start
                if lag > 0.05:
                    for b in group:
                        if b.character == cid or b.face == cid:
                            b.start = round(b.start + lag, 2)
            pending = list(group)
            while pending:
                b = min(pending, key=lambda b: (b.start, b.character))
                pending.remove(b)
                place_one(b, pending)
            for lay in lgroup:
                lay.start = round(ls + lay.line_offset, 2)
                if lay.type == "fx" and lay.point:
                    explode(lay, li)
            line_beats = [b for b in beats if b.line == li or (b.line is None and b.start >= ls - 1e-6 and b not in group)]
            end_of[li] = ls if mode == "bg" else max([b.start + b.duration for b in line_beats + group] + [ls])
            for b in line_beats + group:
                actor_free[b.character] = max(actor_free.get(b.character, 0.0), b.start + b.duration)
    else:
        pending = list(beats)
        while pending:  # always place the earliest unplaced beat (retiming can reorder)
            b = min(pending, key=lambda b: (b.start, b.character))
            pending.remove(b)
            place_one(b, pending)

    for lay in layers:  # a held prop lasts until the line that puts it down
        if lay.until_line is not None and lay.until_line in start_of:
            lay.duration = round(max(start_of[lay.until_line] - lay.start, 0.3), 2)
    trace.add("Blocking", f"placed {len(chars)} character(s) on stage; {len(notes)} approach retime(s)", t0,
              fixes=notes)
    return beats


def ease_warp(t: float, D: float, ei: float, eo: float, r0: float = 0.35) -> float:
    """Clip time for beat time t when the beat accelerates from / decelerates to a slower rate.

    The playback rate ramps r0 -> k over ei seconds, holds k, and ramps back to r0 over the last eo
    seconds, with k chosen so the clip still reaches exactly D: same end point, no foot sliding.
    """
    if (ei <= 0 and eo <= 0) or D <= 0:
        return t
    if t <= 0:
        return 0.0
    if t >= D:
        return t
    ei, eo = min(ei, D / 3), min(eo, D / 3)
    k = (D - r0 * (ei + eo) / 2) / (D - (ei + eo) / 2)
    if ei > 0 and t < ei:
        return r0 * t + (k - r0) * t * t / (2 * ei)
    u1 = r0 * ei + (k - r0) * ei / 2 if ei > 0 else 0.0
    if t <= D - eo or eo <= 0:
        return u1 + k * (t - ei)
    s_ = t - (D - eo)
    return u1 + k * (D - eo - ei) + k * s_ - (k - r0) * s_ * s_ / (2 * eo)


def _transform(lib: MotionLibrary, b: ResolvedBeat, t: float) -> tuple[float, float, float]:
    """World (x, z, yaw) of the character's root at beat-local time t."""
    clip = lib.clips[b.clip]
    local_t = ease_warp(t, b.duration, b.ease_in, b.ease_out) * b.speed
    if clip.category not in LOOPING:
        local_t = min(local_t, clip.duration)
    lx, lz, lyaw = lib.sample_root(b.clip, local_t)
    if b.scale != 1.0:
        lx, lz = lx * b.scale, lz * b.scale
    c, s = math.cos(b.yaw), math.sin(b.yaw)
    return b.x + lx * c + lz * s, b.z - lx * s + lz * c, b.yaw + lyaw


# --------------------------------------------------------------------------- 5
def step_cinematographer(plan: ScenePlan, chars: list[Character], beats: list[ResolvedBeat], duration: float,
                         lib: MotionLibrary, trace: Trace) -> list[Shot]:
    t0 = time.perf_counter()
    ids = [c.id for c in chars]
    notes = []
    # Blocking may have shifted beats; move each director's shot with the beat it was anchored to.
    moved = sorted(((getattr(b, "_orig_start", b.start), b.start) for b in beats if not b.auto), key=lambda p: p[0])
    for s in plan.shots:
        near = [p for p in moved if abs(p[0] - s.start) < 0.31]
        if near and not s.fixed:  # command scripts' [At=5s] camera times are exact
            s.start = round(min(near, key=lambda p: abs(p[0] - s.start))[1], 2)
    explicit = sorted([s for s in plan.shots if s.start < duration], key=lambda s: s.start)
    for prev, s in zip(explicit, explicit[1:]):  # two notes on one moment: the later one takes the second half
        if s.start < prev.start + 1.5 and not (s.fixed and s.start > prev.start + 0.3):
            s.start = round(min(prev.start + max(2.0, (duration - prev.start) / 2), duration - 1.0), 2)
            notes.append(f"'{s.note[:30]}' shared a start with the previous shot; moved to {s.start:.1f}s")
    for s in explicit:
        s.subjects = [x for x in s.subjects if x in ids] or ids[:2]
    extras = {c.id for c in chars if c.extra}
    key = [b for b in beats if not b.auto and b.character not in extras]  # extras are background, never covered
    auto = []
    if not explicit or explicit[0].start > 0.5:
        first_key = explicit[0].start if explicit else duration
        auto = _auto_coverage(key, chars, 0.0, first_key, lib)
        if auto:
            notes.append(f"designed {len(auto)} shot(s) of coverage for 0-{first_key:.1f}s")
    shots = sorted(auto + explicit, key=lambda s: s.start)
    # contiguous: each shot runs to the next one (LLM-provided durations are respected when they leave no gap)
    for i, s in enumerate(shots):
        end = shots[i + 1].start if i + 1 < len(shots) else duration
        s.duration = round(max(end - s.start, 0.3), 2)
    # break up long shots so the edit breathes: an explicit note holds for ~4 s, then coverage resumes
    final: list[Shot] = []
    for s in shots:
        final.append(s)
        if s in explicit and s.duration > 6.0:
            window = [b for b in key if s.start + 3.5 <= b.start < s.start + s.duration - 1.2]
            final += _auto_coverage(window, chars, s.start + 3.5, s.start + s.duration, lib, opening=False)
        elif s.duration > 7.5 and s.move == "static" and s not in explicit:
            window = [b for b in key if s.start + 2.5 < b.start < s.start + s.duration - 1.5]
            final += _auto_coverage(window, chars, s.start + 2.5, s.start + s.duration, lib, opening=False)
    final.sort(key=lambda s: s.start)
    for i, s in enumerate(final):
        end = final[i + 1].start if i + 1 < len(final) else duration
        s.duration = round(max(end - s.start, 0.3), 2)
    trace.add("Cinematographer", f"{len(final)} shots ({len(explicit)} from the director's notes)", t0, fixes=notes)
    return final


def _auto_coverage(key: list[ResolvedBeat], chars: list[Character], start: float, end: float,
                   lib: MotionLibrary, opening: bool = True) -> list[Shot]:
    ids = [c.id for c in chars if not c.extra] or [c.id for c in chars]
    shots: list[Shot] = []
    if opening:
        shots.append(Shot(start=start, duration=1, type="wide", subjects=ids, move="push_in", lens_mm=28,
                          note="establishing"))
    last_cut = start if opening else start - 10
    for b in sorted(key, key=lambda b: b.start):
        if b.start < start + (2.2 if opening else 0) or b.start >= end - 1.2 or b.start - last_cut < 1.8:
            continue
        cat = lib.clips[b.clip].category
        if cat == "daily":
            st, mv, lens, subj = "medium", "push_in", 40, [b.character]
        elif cat in ("emotion",) or b.clip in ("laugh", "mope", "angry", "shrug"):
            st, mv, lens, subj = "close_up", "push_in", 50, [b.character]
        elif cat == "reaction":
            st, mv, lens, subj = "medium", "handheld", 35, [b.character]
        elif cat in ("acrobatic",) or b.clip in ("jump", "jump_twist", "spin_kick"):
            st, mv, lens, subj = "low_angle", "static", 24, [b.character]
        elif cat == "combat" and len(ids) > 2 and b.face:
            st, mv, lens, subj = "two_shot", "handheld", 35, [b.character, b.face]
        elif cat == "combat" and len(ids) >= 2:
            st, mv, lens, subj = "two_shot", "handheld", 35, ids[:2]
        elif lib.clips[b.clip].is_locomotion and b.duration > 2:
            st, mv, lens, subj = "tracking", "static", 35, [b.character]
        elif len(ids) >= 2 and (b.face or (b.move_to and b.move_to.target)):
            st, mv, lens, subj = "over_shoulder", "static", 35, [b.character, b.face or b.move_to.target]
        else:
            st, mv, lens, subj = "medium", "static", 35, [b.character]
        if shots and shots[-1].type == st and shots[-1].subjects == subj:
            continue
        shots.append(Shot(start=round(b.start, 2), duration=1, type=st, subjects=subj, move=mv, lens_mm=lens,
                          note=f"auto: {b.action[:40]}"))
        last_cut = b.start
    return shots


# --------------------------------------------------------------------------- run
def direct(prompt: str, provider: str = "auto", lib: MotionLibrary | None = None) -> dict:
    lib = lib or get_library()
    trace = Trace()
    t_all = time.perf_counter()
    plan = step_director(prompt, provider, lib, trace)
    beats = step_retrieve(plan, lib, trace)
    command_mode = any(b.line is not None for b in plan.beats) or any(l.line is not None for l in plan.layers)
    chars, beats, duration = step_continuity(plan, beats, lib, trace, fill=not command_mode)
    layers = [l.model_copy() for l in plan.layers]
    beats = step_blocking(chars, beats, lib, trace, plan.objects, layers)
    # retiming may have moved beats; re-run continuity fill on the blocked result
    chars, beats, duration = _refill(chars, beats, duration, lib, command_mode)
    ids = {c.id for c in chars}
    if command_mode and plan.extras_trim and any(c.extra for c in chars) and not all(c.extra for c in chars):
        # background extras stop when the principal action ends
        extra_ids = {c.id for c in chars if c.extra}
        end = max(b.start + b.duration for b in beats if b.character not in extra_ids and not b.auto) + 0.6
        if end < duration:
            beats = [b for b in beats if b.character not in extra_ids or b.start < end]
            for b in beats:
                if b.character in extra_ids and b.start + b.duration > end:
                    b.duration = round(end - b.start, 2)
            for b in beats:  # principals hold to the new end
                if b.character not in extra_ids and b.start + b.duration > end:
                    b.duration = round(max(end - b.start, 0.05), 2)
            beats = [b for b in beats if b.duration > 0.04]
            duration = round(end, 2)
    layers = [l for l in layers if (l.type == "fx" or l.character in ids) and l.start < duration
              and (l.target in ids | {"camera", None})]
    for l in layers:
        l.duration = round(max(0.2, min(l.duration, duration - l.start)), 2)
    shots = step_cinematographer(plan, chars, beats, duration, lib, trace)
    scene = ResolvedScene(title=plan.title, logline=plan.logline, setting=plan.setting, duration=duration,
                          characters=chars, beats=beats, shots=shots, objects=plan.objects, layers=layers,
                          unmatched=list(dict.fromkeys(plan.unmatched)))
    return {"scene": scene.model_dump(), "trace": trace.steps,
            "total_ms": round((time.perf_counter() - t_all) * 1000, 1),
            "provider": next((s.get("provider") for s in trace.steps if s["agent"] == "Director"), "offline")}


def _down_idle(lib, prev: ResolvedBeat, start: float, dur: float) -> ResolvedBeat:
    """Command scripts: someone left on the ground breathes, groans or crawls instead of freezing."""
    clip = DOWN_IDLE[prev.clip]
    x, z, yaw = _transform(lib, prev, prev.duration)
    b = ResolvedBeat(character=prev.character, start=round(start, 2), duration=round(max(dur, 0.05), 2),
                     action="stays down", clip=clip, confidence=1.0, auto=True, scale=prev.scale)
    b.x, b.z, b.yaw = round(x, 3), round(z, 3), round(yaw, 4)
    return b


def _wind(lib, prev: ResolvedBeat, start: float, dur: float, c):
    """Command scripts: after a sprint or a fight the breathing settles over ~8 s before a normal idle."""
    x, z, yaw = _transform(lib, prev, prev.duration)
    w = ResolvedBeat(character=prev.character, start=round(start, 2), duration=round(min(dur, 8.0), 2),
                     action="catches their breath", clip="wind_down", confidence=1.0, auto=True, scale=prev.scale)
    w.x, w.z, w.yaw = round(x, 3), round(z, 3), round(yaw, 4)
    out = [w]
    if dur > 8.05:
        i = _idle(c.id, start + 8.0, dur - 8.0, c.kind, c.scale)
        i.x, i.z, i.yaw = w.x, w.z, w.yaw
        out.append(i)
    return out


def _holds(clip: str) -> bool:
    return clip in HOLD_AFTER or clip.startswith(("down_", "crawl_"))


def _refill(chars, beats, duration, lib, command_mode: bool = False):
    """After blocking retimes approaches, close gaps per character with idles placed at the right spot."""
    out = []
    if command_mode:  # let falls finish their whole motion before the ground idle takes over
        for c in chars:
            mine = sorted([b for b in beats if b.character == c.id], key=lambda b: b.start)
            for a, b in zip(mine, mine[1:] + [None]):
                if a.clip in DOWN_IDLE:
                    full = lib.clips[a.clip].duration / max(a.speed, 0.4)
                    limit = (b.start - a.start) if b else full
                    a.duration = round(max(a.duration, min(full, limit)), 2)
    end_all = max(b.start + b.duration for b in beats)
    duration = min(MAX_SCENE, max(duration, end_all + 0.6))
    for c in chars:
        mine = sorted([b for b in beats if b.character == c.id and not (b.auto and b.clip in ("idle_wait", "dog_idle", "horse_idle"))],
                      key=lambda b: b.start)
        t = 0.0
        prev = None
        for b in mine:
            if b.start - t > 0.05:
                if command_mode and prev is not None and prev.clip in WINDED_AFTER and b.start - t >= 2.5 and c.kind == "human":
                    out.extend(_wind(lib, prev, t, b.start - t, c))
                elif command_mode and prev is not None and prev.clip in DOWN_IDLE:
                    out.append(_down_idle(lib, prev, t, b.start - t))
                elif prev is not None and _holds(prev.clip):
                    prev.duration = round(b.start - prev.start, 2)
                else:
                    idle = _idle(c.id, t, b.start - t, c.kind, c.scale)
                    x, z, yaw = _transform(lib, prev, prev.duration) if prev else (b.x, b.z, b.yaw)
                    idle.x, idle.z, idle.yaw = round(x, 3), round(z, 3), round(yaw if prev else b.yaw, 4)
                    out.append(idle)
            out.append(b)
            t = b.start + b.duration
            prev = b
        if duration - t > 0.05:
            if command_mode and prev is not None and prev.clip in WINDED_AFTER and duration - t >= 2.5 and c.kind == "human":
                out.extend(_wind(lib, prev, t, duration - t, c))
            elif command_mode and prev is not None and prev.clip in DOWN_IDLE:
                out.append(_down_idle(lib, prev, t, duration - t))
            elif prev is not None and _holds(prev.clip):
                prev.duration = round(duration - prev.start, 2)
            else:
                idle = _idle(c.id, t, duration - t, c.kind, c.scale)
                x, z, yaw = _transform(lib, prev, prev.duration) if prev else (0, 0, 0)
                idle.x, idle.z, idle.yaw = round(x, 3), round(z, 3), round(yaw, 4)
                out.append(idle)
    out.sort(key=lambda b: (b.start, b.character))
    return chars, out, round(duration, 2)
