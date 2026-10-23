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

from .library import MotionLibrary, get_library
from .llm import available_provider, plan_with_llm
from .planner_offline import plan_offline
from .schema import PALETTE, Character, ResolvedBeat, ResolvedScene, ScenePlan, Shot

MAX_SCENE = 45.0
HOLD_AFTER = {"fall_face", "fall_back", "crawl", "sit", "sit_stool"}  # end pose is held, not reset to idle
LOOPING = {"locomotion", "idle", "dance", "daily"}
# Clips that only read on camera if the partner is within reach (metres).
REACH = {"punch": 1.1, "kick": 1.3, "punch_kick": 1.2, "boxing": 1.4, "sword": 1.8, "spin_kick": 1.6,
         "handshake": 0.9, "high_five": 1.0, "push": 0.8, "angry": 1.8, "salsa": 1.0}


@dataclass
class Trace:
    steps: list[dict] = field(default_factory=list)

    def add(self, agent: str, summary: str, t0: float, **extra):
        self.steps.append({"agent": agent, "summary": summary, "ms": round((time.perf_counter() - t0) * 1000, 1),
                           **extra})


# --------------------------------------------------------------------------- 1
def step_director(prompt: str, provider: str, lib: MotionLibrary, trace: Trace) -> ScenePlan:
    t0 = time.perf_counter()
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
    for b in plan.beats:
        matches = lib.search(b.action, k=3, hint=b.clip)
        if b.clip in lib.clips:
            clip_id = b.clip
            conf = next((m.confidence for m in matches if m.clip.id == clip_id), 0.5)
        elif matches:
            clip_id, conf = matches[0].clip.id, matches[0].confidence
            if b.clip:
                fixes.append(f"unknown clip '{b.clip}' -> {clip_id}")
        else:
            clip_id, conf = "idle_wait", 0.0
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
    trace.add("Motion Retriever", f"grounded {len(out)} beats in mocap clips"
              + (f"; {len(low)} low-confidence" if low else "")
              + (f"; {len(plan.unmatched)} action(s) not in the library" if plan.unmatched else ""), t0, fixes=fixes,
              matches=[{"action": b.action, "clip": b.clip, "confidence": b.confidence,
                        "alternatives": b.alternatives} for b in out])
    return out


# --------------------------------------------------------------------------- 3
def step_continuity(plan: ScenePlan, beats: list[ResolvedBeat], lib: MotionLibrary, trace: Trace):
    t0 = time.perf_counter()
    fixes = []
    chars = plan.characters[:4] or [Character(id="A", name="Performer")]
    ids = {c.id for c in chars}
    for i, c in enumerate(chars):
        c.color = c.color or PALETTE[i % len(PALETTE)]
    for b in beats:
        if b.character not in ids:
            fixes.append(f"beat '{b.action}' referenced unknown character {b.character} -> {chars[0].id}")
            b.character = chars[0].id
        if b.face and b.face not in ids or b.face == b.character:
            b.face = None
        if b.move_to and (b.move_to.target not in ids or b.move_to.target == b.character):
            b.move_to = None
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
                    final.append(_idle(c.id, t, b.start - t))
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
                final.append(_idle(c.id, end, duration - end))
    final.sort(key=lambda b: (b.start, b.character))
    trace.add("Continuity", f"{len(fixes)} fix(es); idle/hold fill for {sum(b.auto for b in final)} gap(s); "
              f"scene {duration:.1f}s", t0, fixes=fixes)
    return chars, final, round(duration, 2)


def _idle(cid: str, start: float, dur: float) -> ResolvedBeat:
    return ResolvedBeat(character=cid, start=round(start, 2), duration=round(max(dur, 0.05), 2),
                        action="idle", clip="idle_wait", confidence=1.0, auto=True)


# --------------------------------------------------------------------------- 4
def step_blocking(chars: list[Character], beats: list[ResolvedBeat], lib: MotionLibrary, trace: Trace):
    """Assigns every beat a start position/heading by integrating mocap root motion."""
    t0 = time.perf_counter()
    n = len(chars)
    notes = []
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
            x, z = 2.6 * math.sin(ang), 2.6 * math.cos(ang)
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

    def pos_at(cid: str, t: float) -> tuple[float, float]:
        for b in reversed(placed[cid]):
            if b.start <= t + 1e-6:
                return _transform(lib, b, min(t - b.start, b.duration))[:2]
        return state[cid][:2]

    pending = list(beats)
    while pending:  # always place the earliest unplaced beat (retiming can reorder)
        b = min(pending, key=lambda b: (b.start, b.character))
        pending.remove(b)
        x, z, yaw = state[b.character]
        if b.face and b.clip in REACH and not getattr(b, "_engaged", False):
            tx, tz = pos_at(b.face, b.start)
            d = math.hypot(tx - x, tz - z)
            reach = REACH[b.clip]
            if d > reach + 0.5:  # out of reach: insert an approach and push the exchange back
                walk = "run" if d - reach > 4.5 else "walk"
                c = lib.clips[walk]
                est = (d - reach) / max(math.hypot(*c.end_disp) / c.duration, 0.2)
                delta = round(min(max(est, 0.4), 12.0), 2)
                approach = ResolvedBeat(character=b.character, start=b.start, duration=delta,
                                        action=f"closes distance to {b.face}", clip=walk, confidence=1.0,
                                        move_to={"target": b.face, "distance": reach}, face=b.face)
                for later in pending:
                    if later.character in (b.character, b.face) and later.start >= b.start - 1e-6:
                        later.start = round(later.start + delta, 2)
                b.start = round(b.start + delta, 2)
                object.__setattr__(b, "_engaged", True)
                pending += [approach, b]
                beats.append(approach)
                notes.append(f"{b.character} was {d:.1f} m from {b.face} for '{b.clip}' (reach {reach} m): "
                             f"inserted {delta:.1f}s {walk} approach, shifted the exchange")
                continue
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
                if delta > 0:  # push this character's later beats back
                    for later in pending:
                        if later.character == b.character and later.start >= b.start:
                            later.start = round(later.start + delta, 2)
            if b.move_to:  # correct for the clip's own sideways drift so the approach lands on target
                ex, ez, _ = lib.sample_root(b.clip, b.duration * b.speed)
                if math.hypot(ex, ez) > 0.3:
                    yaw -= math.atan2(ex, ez)
        b.x, b.z, b.yaw = round(x, 3), round(z, 3), round(yaw, 4)
        placed[b.character].append(b)
        state[b.character] = _transform(lib, b, b.duration)

    trace.add("Blocking", f"placed {len(chars)} character(s) on stage; {len(notes)} approach retime(s)", t0,
              fixes=notes)
    return beats


def _transform(lib: MotionLibrary, b: ResolvedBeat, t: float) -> tuple[float, float, float]:
    """World (x, z, yaw) of the character's root at beat-local time t."""
    clip = lib.clips[b.clip]
    local_t = t * b.speed
    if clip.category not in LOOPING:
        local_t = min(local_t, clip.duration)
    lx, lz, lyaw = lib.sample_root(b.clip, local_t)
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
        if near:
            s.start = round(min(near, key=lambda p: abs(p[0] - s.start))[1], 2)
    explicit = sorted([s for s in plan.shots if s.start < duration], key=lambda s: s.start)
    for prev, s in zip(explicit, explicit[1:]):  # two notes on one moment: the later one takes the second half
        if s.start < prev.start + 1.5:
            s.start = round(min(prev.start + max(2.0, (duration - prev.start) / 2), duration - 1.0), 2)
            notes.append(f"'{s.note[:30]}' shared a start with the previous shot; moved to {s.start:.1f}s")
    for s in explicit:
        s.subjects = [x for x in s.subjects if x in ids] or ids[:2]
    key = [b for b in beats if not b.auto]
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
    ids = [c.id for c in chars]
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
        elif cat == "combat" and len(ids) >= 2:
            st, mv, lens, subj = "two_shot", "handheld", 35, ids[:2]
        elif lib.clips[b.clip].is_locomotion and b.duration > 2:
            st, mv, lens, subj = "tracking", "static", 35, [b.character]
        elif len(ids) >= 2 and (b.face or b.move_to):
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
    chars, beats, duration = step_continuity(plan, beats, lib, trace)
    beats = step_blocking(chars, beats, lib, trace)
    # retiming may have moved beats; re-run continuity fill on the blocked result
    chars, beats, duration = _refill(chars, beats, duration, lib)
    shots = step_cinematographer(plan, chars, beats, duration, lib, trace)
    scene = ResolvedScene(title=plan.title, logline=plan.logline, setting=plan.setting, duration=duration,
                          characters=chars, beats=beats, shots=shots, unmatched=list(dict.fromkeys(plan.unmatched)))
    return {"scene": scene.model_dump(), "trace": trace.steps,
            "total_ms": round((time.perf_counter() - t_all) * 1000, 1),
            "provider": next((s.get("provider") for s in trace.steps if s["agent"] == "Director"), "offline")}


def _refill(chars, beats, duration, lib):
    """After blocking retimes approaches, close gaps per character with idles placed at the right spot."""
    out = []
    end_all = max(b.start + b.duration for b in beats)
    duration = min(MAX_SCENE, max(duration, end_all + 0.6))
    for c in chars:
        mine = sorted([b for b in beats if b.character == c.id and not (b.auto and b.clip == "idle_wait")],
                      key=lambda b: b.start)
        t = 0.0
        prev = None
        for b in mine:
            if b.start - t > 0.05:
                if prev is not None and prev.clip in HOLD_AFTER:
                    prev.duration = round(b.start - prev.start, 2)
                else:
                    idle = _idle(c.id, t, b.start - t)
                    x, z, yaw = _transform(lib, prev, prev.duration) if prev else (b.x, b.z, b.yaw)
                    idle.x, idle.z, idle.yaw = round(x, 3), round(z, 3), round(yaw if prev else b.yaw, 4)
                    out.append(idle)
            out.append(b)
            t = b.start + b.duration
            prev = b
        if duration - t > 0.05:
            if prev is not None and prev.clip in HOLD_AFTER:
                prev.duration = round(duration - prev.start, 2)
            else:
                idle = _idle(c.id, t, duration - t)
                x, z, yaw = _transform(lib, prev, prev.duration) if prev else (0, 0, 0)
                idle.x, idle.z, idle.yaw = round(x, 3), round(z, 3), round(yaw, 4)
                out.append(idle)
    out.sort(key=lambda b: (b.start, b.character))
    return chars, out, round(duration, 2)
