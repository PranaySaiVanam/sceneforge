"""Motion library: catalog, retrieval and root-motion sampling.

Retrieval is a hybrid lexical ranker built for short action phrases:
  * BM25 over tags + description (with light stemming and synonyms)
  * phrase bonus when a multi-word tag appears verbatim in the query
  * id bonus when the planner names a clip id directly (LLM "clip_hint")
It runs in microseconds, needs no API key, and is measured in evals/.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

SYNONYMS = {
    "attack": "punch strike", "attacks": "punch strike", "lunge": "punch lunge", "lunges": "punch lunge",
    "hits": "punch", "jab": "punch", "jabs": "punch", "swing": "punch", "swings": "punch",
    "dodge": "duck dodge", "dodges": "duck dodge", "evade": "duck", "evades": "duck",
    "stumble": "drunk stagger", "stumbles": "drunk stagger", "staggers": "drunk stagger",
    "jog": "run", "jogs": "run", "jogging": "run", "flips": "flip backflip", "flip": "flip backflip",
    "sprint": "run", "sprints": "run", "dash": "run", "dashes": "run", "races": "run", "flees": "run flee",
    "stroll": "walk", "strolls": "walk", "enter": "walk", "enters": "walk", "approach": "walk approach",
    "approaches": "walk approach", "heads": "walk", "steps": "walk step", "crosses": "walk",
    "collapses": "fall collapse", "drops": "fall", "tumbles": "fall", "dies": "fall die",
    "greets": "wave greet", "hello": "wave hello", "bye": "wave goodbye",
    "celebrates": "celebrate jump_twist", "cheers": "celebrate jump", "rejoices": "celebrate",
    "cries": "mope sad", "weeps": "mope sad", "sobs": "mope sad", "grieves": "mope grief",
    "yells": "angry shout", "shouts": "angry shout", "argues": "angry argue", "screams": "angry shout",
    "sits": "sit", "seated": "sit", "rests": "sit rest", "stands": "stand", "waits": "wait idle",
    "searches": "search look around", "investigates": "investigate look around", "explores": "explore look around",
    "lifts": "pick up lift", "grabs": "pick up grab", "carries": "carry pick up", "tosses": "throw toss",
    "shoves": "push shove", "pushes": "push", "climbs": "climb", "sweeps": "sweep", "drinks": "drink",
    "dances": "dance", "spins": "spin", "flips": "flip", "kicks": "kick", "punches": "punch",
    "fights": "fight", "spars": "boxing spar", "duels": "duel sword", "slashes": "sword slash",
    "laughs": "laugh", "giggles": "laugh", "shrugs": "shrug", "bows": "bow curtsey", "hops": "hop",
    "jumps": "jump", "leaps": "jump leap", "crawls": "crawl", "sneaks": "sneak", "creeps": "sneak creep",
    "tiptoes": "sneak tiptoe", "limps": "limp", "paces": "pace", "stretches": "stretch", "yawns": "yawn stretch",
    "retreats": "walk backwards retreat", "backs": "walk backwards back away", "circles": "walk sideways circle",
    "falls": "fall", "trips": "fall trip", "rises": "get up", "recovers": "get up recover", "wakes": "get up stretch",
    "tastes": "spoon soup taste", "taste": "spoon soup", "tasting": "spoon soup", "cooks": "cook", "cooking": "cook", "dines": "eat dinner", "stirs": "stir mix", "bakes": "bake knead",
    "serves": "serve set the table", "pours": "pour make coffee", "brews": "brew make coffee", "munches": "eat munch",
    "breakfast": "eat breakfast", "lunch": "eat lunch", "texts": "phone", "calls": "phone call",
    "looks": "look", "stops": "halt idle wait", "stop": "halt idle wait", "halts": "halt idle", "exhausted": "tired", "weary": "tired", "terrified": "flee run", "scared": "flee",
}
STOP = set("a an the and or of to in on at with his her their its is are was were be by for from as it "
           "into onto then than that this one other both them they he she him do does doing did done some few "
           "times time again more starts start begins begin keeps keep tries try goes go gets get two three four "
           "five six seven eight nine ten twenty".split())


def tokenize(text: str) -> list[str]:
    toks = re.findall(r"[a-z]+", text.lower().replace("-", " ").replace("_", " "))
    out = []
    for t in toks:
        if t in STOP:
            continue
        out.append(t)
        if t in SYNONYMS:
            out.extend(SYNONYMS[t].split())
    return [stem(t) for t in out]


def stem(t: str) -> str:
    if len(t) > 4 and t.endswith(("ches", "shes", "sses", "xes", "zes")):
        t = t[:-2]
    else:
        for suf in ("ing", "ed", "s"):
            if len(t) > len(suf) + 2 and t.endswith(suf) and not t.endswith("ss"):
                t = t[: -len(suf)]
                break
    if t.endswith("e") and len(t) > 4:  # pirouette/pirouett, dance/danc
        t = t[:-1]
    if len(t) > 3 and t[-1] == t[-2]:  # running -> runn -> run
        t = t[:-1]
    return t


@dataclass
class Clip:
    id: str
    cmu: str
    category: str
    tags: list[str]
    description: str
    frames: int
    duration: float
    end_disp: list[float]
    end_yaw: float
    avg_speed: float
    height: float
    source: str = "mocap"  # or "keyframed" for actions the dataset does not contain
    kind: str = "human"  # skeleton the clip is for: human | dog | horse
    weapon: str | None = None  # weapon the clip shows in the hands (gun, rifle, bow, mace, sword)
    prop: str | None = None  # set dressing the clip needs (box, chair, door, button)
    strike_yaw: float | None = None  # direction of the strike relative to the body at impact (radians)
    tier: int = 1  # 1 = original library, 2 = extended pack (wins only by a clear margin)
    gaze: int = 0  # 1: Head_End is tipped toward where the actor looks (the viewer draws the eyeline)
    tokens: list[str] = field(default_factory=list)

    @property
    def is_locomotion(self) -> bool:
        return self.category == "locomotion"


@dataclass
class Match:
    clip: Clip
    score: float
    confidence: float


class MotionLibrary:
    def __init__(self, data_dir: Path = DATA_DIR):
        lib = json.loads((data_dir / "motion_library.json").read_text())
        self.data_dir = data_dir
        self.fps: int = lib["fps"]
        self.joints: list[str] = lib["joints"]
        self.bones = lib["bones"]
        self.source = lib["source"]
        self.clips: dict[str, Clip] = {}
        for c in lib["clips"]:
            clip = Clip(**c)
            clip.tokens = tokenize(" ".join(clip.tags) + " " + clip.description)
            self.clips[clip.id] = clip
        # BM25 statistics per skeleton kind and tier, so new clips never change how the original ones rank
        self.stats: dict[tuple[str, int], tuple[float, dict[str, float]]] = {}
        for kind, tier in {(c.kind, c.tier) for c in self.clips.values()} | {(c.kind, 2) for c in self.clips.values()}:
            group = [c for c in self.clips.values() if c.kind == kind and c.tier <= tier]
            n = len(group)
            avgdl = sum(len(c.tokens) for c in group) / n
            df: dict[str, int] = {}
            for c in group:
                for t in set(c.tokens):
                    df[t] = df.get(t, 0) + 1
            self.stats[(kind, tier)] = (avgdl, {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()})
        self.avgdl, self.idf = self.stats[("human", 1)]
        self._root_cache: dict[str, list[tuple[float, float, float]]] = {}

    # ------------------------------------------------------------------ search
    EXT_MARGIN = 4.0
    EXT_MIN = 7.0  # an extended clip must also be a strong match on its own

    def _score(self, c: "Clip", q: list[str], ql: str, hint: str | None, stats) -> float:
        avgdl, idf = stats
        s = 0.0
        dl = len(c.tokens)
        for t in set(q):
            tf = c.tokens.count(t)
            if tf:
                s += idf.get(t, 0) * tf * 2.2 / (tf + 1.2 * (0.25 + 0.75 * dl / avgdl))
        for tag in c.tags:  # verbatim multi-word phrase bonus
            if " " in tag and f" {tag} " in ql:
                s += 3.0
        if hint and hint.strip().lower() == c.id:
            s += 8.0
        return s

    def search(self, query: str, k: int = 3, hint: str | None = None, kind: str = "human") -> list[Match]:
        """Ranks clips for an action phrase.

        Original (tier 1) clips are scored exactly as in the first release. Extended (tier 2) clips are scored
        against the whole library and take a slot only when they beat the best original clip by EXT_MARGIN,
        or when the planner names them; this keeps every existing scene's choices stable.
        """
        q = tokenize(query + " " + (hint or ""))
        ql = " " + query.lower() + " "
        core_stats = self.stats.get((kind, 1), self.stats.get((kind, 2)))
        all_stats = self.stats.get((kind, 2), core_stats)
        core = [(self._score(c, q, ql, hint, core_stats), c) for c in self.clips.values() if c.kind == kind and c.tier == 1]
        ext = [(self._score(c, q, ql, hint, all_stats), c) for c in self.clips.values() if c.kind == kind and c.tier == 2]
        core.sort(key=lambda x: -x[0])
        ext.sort(key=lambda x: -x[0])
        best_core = core[0][0] if core else 0.0
        scored = list(core)
        for s, c in ext:
            named = bool(hint and hint.strip().lower() == c.id)
            strong = s >= best_core + self.EXT_MARGIN and s >= self.EXT_MIN
            scored.append((s if (named or strong) else min(s, best_core - 0.01), c))
        scored.sort(key=lambda x: -x[0])
        top = [x for x in scored[:k] if x[0] > 0]
        if not top:
            return []
        total = sum(math.exp(s) for s, _ in scored[:8])
        return [Match(c, round(s, 3), round(math.exp(s) / total, 3)) for s, c in top]

    def catalog_for_prompt(self) -> str:
        lines = []
        for c in self.clips.values():
            extra = "".join([f", {c.kind} only" if c.kind != "human" else "", f", holds {c.weapon}" if c.weapon else ""])
            lines.append(f"{c.id} [{c.category}, {c.duration:.1f}s{', loops' if c.is_locomotion else ''}{extra}]: {c.description}")
        return "\n".join(lines)

    # ------------------------------------------------------------ root motion
    def root_track(self, clip_id: str) -> list[tuple[float, float, float]]:
        """Per-frame (x, z, yaw) of the hips in clip-local space."""
        if clip_id not in self._root_cache:
            data = json.loads((self.data_dir / "motions" / f"{clip_id}.json").read_text())
            li, ri = self.joints.index("LeftUpLeg"), self.joints.index("RightUpLeg")
            track = []
            for fr in data["frames"]:
                x, z = fr[0] / 1000, fr[2] / 1000
                ax, az = (fr[li * 3] - fr[ri * 3]), (fr[li * 3 + 2] - fr[ri * 3 + 2])
                track.append((x, z, math.atan2(-az, ax)))
            # unwrap yaw
            for i in range(1, len(track)):
                x, z, y = track[i]
                prev = track[i - 1][2]
                while y - prev > math.pi:
                    y -= 2 * math.pi
                while y - prev < -math.pi:
                    y += 2 * math.pi
                track[i] = (x, z, y)
            self._root_cache[clip_id] = track
        return self._root_cache[clip_id]

    def sample_root(self, clip_id: str, t: float) -> tuple[float, float, float]:
        """Root (x, z, yaw delta) at clip-local time t, looping with accumulated motion.

        Mirrors the viewer's playback so blocking computed here matches what
        is rendered.
        """
        clip = self.clips[clip_id]
        track = self.root_track(clip_id)
        L = clip.duration
        n_loops = int(t // L) if L > 0 else 0
        rem = t - n_loops * L
        ox = oz = 0.0
        oyaw = 0.0
        ex, ez = track[-1][0], track[-1][1]
        eyaw = track[-1][2] - track[0][2]  # same definition the viewer uses
        for _ in range(n_loops):
            c, s = math.cos(oyaw), math.sin(oyaw)
            ox += ex * c + ez * s
            oz += -ex * s + ez * c
            oyaw += eyaw
        f = min(rem * self.fps, len(track) - 1)
        i0 = int(f)
        i1 = min(i0 + 1, len(track) - 1)
        a = f - i0
        x = track[i0][0] * (1 - a) + track[i1][0] * a
        z = track[i0][1] * (1 - a) + track[i1][1] * a
        y = (track[i0][2] * (1 - a) + track[i1][2] * a) - track[0][2]
        c, s = math.cos(oyaw), math.sin(oyaw)
        return ox + x * c + z * s, oz - x * s + z * c, oyaw + y


@lru_cache(maxsize=1)
def get_library() -> MotionLibrary:
    return MotionLibrary()
