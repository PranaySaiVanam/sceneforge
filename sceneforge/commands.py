"""Director command language: `[Actor]: [Verb] [target] [Modifier=Value, ...]`.

Example script:

    Cast: Ravi, Bruno [Kind=Dog], Storm [Kind=Horse], 5 Zombies, Kira [Tail=Yes, Weapon=Bow]
    Set: Van at Grid 12K [Facing=South]
    Ravi: Move to Grid 4B [Speed=Fast, Style=Angry]
    Zombies: Approach Ravi [Speed=Slow]
    Ravi: Equip sword
    Ravi: Attack Zombie 1 [Weapon=Sword, Result=Fall]
    Kira: Shoot Zombie 2 [Weapon=Bow, With=prev]
    Camera: Wide shot, slow orbit [At=2s]

Rules
  * One line = one beat per actor. Lines run in order; [With=prev] runs a line together with the
    previous one, [At=5s] starts it at an absolute time, [Delay=0.5] adds a pause.
  * A group (`5 Zombies:`, `All:`, `Ravi, Priya:`) gives every member the command, with small
    start and speed variations so a crowd never moves in lockstep.
  * One `Move` verb covers walk, jog, run and stalk: Speed (Slow|Normal|Jog|Fast|Sprint or a number)
    and Style (Angry, Sad, Happy, Sneaky, Crouched, Injured, Drunk, Zombie, Scared, Proud, Old, Heavy,
    March, Carry) pick the clip. Dogs and horses get their own gaits.
  * Unknown verbs fall back to library retrieval, so plain English inside a line still works.
"""
from __future__ import annotations

import math
import re

from .library import MotionLibrary
from .schema import MAX_CAST, MAX_CHARACTERS, PALETTE, Beat, Character, Layer, MoveTo, SceneObject, ScenePlan, Setting, Shot
from .stage import COMPASS, parse_point

IDS = "ABCDEFGHIJKLMNOP"
NUMBERS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
           "ten": 10, "a pair of": 2, "a couple of": 2}
DOG_WORDS = r"dog|dogs|puppy|hound|wolf|wolves|k9|canine"
HORSE_WORDS = r"horse|horses|pony|stallion|mare|steed"
CAR_WORDS = r"car|cars|jeep|taxi|cab|sedan|suv|truck|van|vehicle|bus|pickup"
BIKE_WORDS = r"bike|bikes|motorbike|motorcycle|scooter|bicycle|moped"
MOUNTS = ("horse", "car", "bike")
MOODS = {"sad": "sad", "defeated": "sad", "depressed": "sad", "gloomy": "sad", "dejected": "sad", "upset": "sad",
         "angry": "angry", "aggressive": "angry", "furious": "angry", "mad": "angry", "enraged": "angry",
         "anxious": "anxious", "alert": "anxious", "nervous": "anxious", "tense": "anxious", "worried": "anxious",
         "jumpy": "anxious", "stealthy": "stealthy", "suspicious": "stealthy", "cautious": "stealthy",
         "careful": "stealthy", "sneaky": "stealthy", "neutral": None, "normal": None, "calm": None, "none": None}
MOOD_SPEED = {"sad": 0.8, "angry": 1.1, "anxious": 1.05, "stealthy": 0.75}
WANDER_CLIPS = ["walk", "walk_happy", "walk_quick", "walk_proud", "walk_old", "walk", "walk_sad"]
WALLFLOWER = ["idle_phone", "idle_shift", "idle_arms_crossed", "idle_watch"]
ACTIVITY_CLIPS = {"phone": "idle_phone", "read": "read_book", "sweep": "sweep", "wipe": "wipe_counter",
                  "type": "type_laptop", "think": "think", "sing": "sing", "cheer": "cheer", "dance": "salsa"}
DANCES = ["salsa", "chicken_dance", "ballet"]
FIGHT_POOL = ["punch", "kick", "boxing", "punch_kick", "karate_punch", "front_kick", "roundhouse"]
TAIL_WORDS = r"tailed|tail|alien|creature"
SPECIAL_LINES = ("cast", "camera", "set", "objects", "object", "scene", "setting", "props")

STYLE_CLIPS = {
    "angry": "walk_angry", "stern": "walk_angry", "sad": "walk_sad", "tired": "walk_sad", "happy": "walk_happy",
    "sneaky": "sneak", "sneak": "sneak", "stealth": "sneak", "stealthy": "sneak", "stalk": "sneak",
    "crouched": "walk_crouch", "crouch": "walk_crouch", "low": "walk_crouch", "injured": "walk_limp",
    "limp": "walk_limp", "wounded": "walk_limp", "hurt": "walk_limp", "drunk": "walk_drunk", "zombie": "walk_zombie",
    "scared": "walk_scared", "afraid": "walk_scared", "nervous": "walk_scared", "proud": "walk_proud",
    "confident": "walk_proud", "cool": "walk_proud", "old": "walk_old", "elderly": "walk_old", "heavy": "walk_heavy",
    "strong": "walk_heavy", "macho": "walk_heavy", "march": "march", "military": "march", "carry": "carry_walk",
    "sideways": "sneak_side",
}
SPEEDS = {"slow": 0.8, "slowly": 0.8, "normal": 1.0, "walk": 1.0, "jog": 1.4, "medium": 1.4, "brisk": 1.4,
          "quick": 1.4, "trot": 1.4, "fast": 2.0, "run": 2.0, "quickly": 2.0, "sprint": 2.5, "gallop": 2.5}
REACTIONS = {"hit": "hit_react", "head": "hit_react", "gut": "hit_gut", "stomach": "hit_gut", "block": "block",
             "dodge": "evade", "evade": "evade", "duck": "duck", "fall": "fall_back", "fall_back": "fall_back",
             "fall_face": "fall_face", "stumble": "stumble", "stagger": "stagger_back", "push": "stagger_back",
             "blast": "blast_back", "none": None}
WEAPON_WORDS = {"gun": "gun", "pistol": "gun", "handgun": "gun", "revolver": "gun", "rifle": "rifle",
                "shotgun": "rifle", "sword": "sword", "blade": "sword", "katana": "sword", "mace": "mace",
                "gadha": "mace", "gada": "mace", "club": "mace", "hammer": "mace", "axe": "mace", "bow": "bow",
                "arrow": "bow", "fists": "fists", "fist": "fists", "punch": "fists", "kick": "kick", "foot": "kick"}
ATTACK_CLIPS = {"fists": "punch", "kick": "kick", "sword": "sword", "mace": "mace_smash", "gun": "shoot_gun",
                "rifle": "shoot_rifle", "bow": "shoot_bow"}
RANGED = {"gun", "rifle", "bow"}
EMOTES = {"wave": "wave", "point": "point_at", "salute": "salute", "clap": "clap", "applaud": "clap", "bow": "curtsey",
          "curtsey": "curtsey", "shrug": "shrug", "laugh": "laugh", "cheer": "cheer", "nod": "nod", "cry": "mope",
          "mope": "mope", "argue": "argue", "dance": "chicken_dance", "celebrate": "cheer", "high five": "high_five",
          "handshake": "handshake", "shake hands": "handshake", "stretch": "stretch", "yawn": "stretch",
          "sing": "sing", "singing": "sing", "perform a song": "sing", "dancing": "chicken_dance"}
GRIP_PROPS = {"cup": "cup", "mug": "cup", "coffee": "cup", "glass": "cup", "drink": "cup", "tea": "cup",
              "suitcase": "suitcase", "briefcase": "suitcase", "case": "suitcase", "bag": "suitcase", "handbag": "suitcase"}
OBSTACLES = {  # verb pattern -> (object type, traversal clip)
    r"step(?:s)?\s+over|stride(?:s)?\s+over|hop(?:s)?\s+over|climb(?:s)?\s+over": ("curb", "step_over"),
    r"duck(?:s)?\s+under|stoop(?:s)?\s+under|crouch(?:es)?\s+under|go(?:es)?\s+under": ("beam", "duck_under"),
    r"squeeze(?:s)?\s+through|slip(?:s)?\s+through|shuffle(?:s)?\s+through|edge(?:s)?\s+through|go(?:es)?\s+through": ("gap", "squeeze"),
}
DOWN_STYLES = {"still": "still", "dead": "still", "motionless": "still", "unconscious": "still", "groan": "groan",
               "groaning": "groan", "writhe": "groan", "moan": "groan", "hurt": "groan", "crawl": "crawl", "crawling": "crawl"}
POSE_HOLDS = {"sit": "sit_down", "kneel": "kneel", "crouch": "crouch", "prone": "prone", "lie": "lie_down"}


def is_command_script(text: str) -> bool:
    """True when most non-empty lines look like `Who: do something` or carry [Modifiers]."""
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    if not lines:
        return False
    hits = sum(bool(re.match(r"^[\w .,&'-]{1,40}:\s*\S", l)) or "[" in l and "=" in l
               or bool(re.match(r"^(spawn|explosion|an explosion|teleport)\b", l, re.I)) for l in lines)
    return hits >= max(1, math.ceil(len(lines) * 0.6)) and (len(lines) >= 2 or "[" in text or re.match(r"^\s*[\w ]+:", text))


def _mods(text: str) -> tuple[str, dict[str, str], set[str]]:
    mods, flags = {}, set()
    for block in re.findall(r"\[(.*?)\]", text):
        for part in re.split(r"[;,]", block):
            part = part.strip()
            if not part:
                continue
            if "=" in part or ":" in part:
                k, v = re.split(r"[=:]", part, maxsplit=1)
                mods[k.strip().lower()] = v.strip()
            else:
                flags.add(part.lower())
    return re.sub(r"\[.*?\]", "", text).strip(), mods, flags


def _seconds(v: str | None) -> float | None:
    if not v:
        return None
    m = re.search(r"-?\d+(?:\.\d+)?", v)
    return float(m.group()) if m else None


IRREGULAR = {"wolves": "wolf", "men": "man", "women": "woman", "people": "person", "children": "child",
             "thieves": "thief", "knives": "knife", "elves": "elf", "dwarves": "dwarf", "mice": "mouse", "geese": "goose",
             "police": "police officer", "staff": "staff member", "crew": "crew member"}
IE_WORDS = {"zombies", "rookies", "hippies", "cookies", "movies", "goalies", "newbies", "yuppies", "bookies", "groupies",
            "techies", "aussies", "roadies", "kiddies"}


def singular(word: str) -> str:
    w = word.strip()
    lw = w.lower()
    try:
        from .planner_offline import PLURAL_TO_ROLE
        if lw in PLURAL_TO_ROLE:
            return PLURAL_TO_ROLE[lw]
    except ImportError:
        pass
    if lw in IRREGULAR:
        return IRREGULAR[lw]
    if lw in IE_WORDS:
        return lw[:-1]
    if lw.endswith("ies") and len(lw) > 4:
        return lw[:-3] + "y"
    if lw.endswith(("ches", "shes", "xes", "sses", "zes", "oes")):
        return lw[:-2]
    if lw.endswith("s") and not lw.endswith("ss"):
        return lw[:-1]
    return lw


class _Actor:
    def __init__(self, cid: str, name: str, kind: str = "human"):
        self.id, self.name, self.kind = cid, name, kind
        self.end = 0.0
        self.state = "stand"  # stand | sit | prone | lying | crouch | kneel | ride
        self.down: str | None = None  # clip that put them on the ground (picks the matching ground idle)
        self.grip: Layer | None = None
        self.mood: Layer | None = None
        self.mood_name: str | None = None
        self.seat = 0
        self.horse: str | None = None
        self.weapon: str | None = None
        self.group: str | None = None


class CommandPlanner:
    def __init__(self, lib: MotionLibrary):
        self.lib = lib
        self.actors: dict[str, _Actor] = {}
        self.order: list[str] = []
        self.chars: list[Character] = []
        self.objects: list[SceneObject] = []
        self.layers: list[Layer] = []
        self.beats: list[Beat] = []
        self.shots: list[Shot] = []
        self.unmatched: list[str] = []
        self.notes: list[str] = []
        self.setting = Setting()
        self.cursor = 0.0
        self.prev_start = 0.0
        self.line_idx = -1
        self.line_mode = "seq"
        self.line_start = 0.0
        self.extras_trim = True
        self.line_mood = None

    # ------------------------------------------------------------------ cast
    def _kind_of(self, name: str) -> str:
        low = name.lower()
        if re.search(rf"\b({DOG_WORDS})\b", low):
            return "dog"
        if re.search(rf"\b({HORSE_WORDS})\b", low):
            return "horse"
        if re.search(rf"\b({BIKE_WORDS})\b", low):
            return "bike"
        if re.search(rf"\b({CAR_WORDS})\b", low):
            return "car"
        return "human"

    def add_actor(self, name: str, mods: dict | None = None, flags: set | None = None, extra: bool = False) -> _Actor | None:
        mods, flags = mods or {}, flags or set()
        key = name.strip().lower()
        if key in self.actors:
            return self.actors[key]
        named = sum(not c.extra for c in self.chars)
        if len(self.chars) >= MAX_CHARACTERS or (not extra and named >= MAX_CAST):
            self.notes.append(f"'{name}' not added: the stage holds {MAX_CAST} actors plus extras")
            return None
        kind = mods.get("kind", "").lower() or self._kind_of(name)
        kind = "dog" if re.search(DOG_WORDS, kind) else "horse" if re.search(HORSE_WORDS, kind) else \
            "bike" if re.search(BIKE_WORDS, kind) else "car" if re.search(CAR_WORDS, kind) else "human"
        cid = IDS[len(self.chars)]
        a = _Actor(cid, name.strip(), kind)
        ch = Character(id=cid, name=name.strip().title() if name.islower() else name.strip(), kind=kind, extra=extra)
        if extra:
            ch.color = PALETTE[10 + sum(c.extra for c in self.chars) % 6]
        if mods.get("standin", "").lower() in ("yes", "true", "1") or "stand-in" in flags or "standin" in flags:
            ch.standin = True
        tail = mods.get("tail", "").lower() in ("yes", "true", "1") or "tail" in flags or \
            bool(re.search(rf"\b({TAIL_WORDS})\b", name.lower()))
        ch.tail = tail
        if "scale" in mods or tail:
            ch.scale = max(0.6, min(1.5, _seconds(mods.get("scale")) or (1.2 if tail else 1.0)))
        w = mods.get("weapon", "").lower()
        if w in WEAPON_WORDS and WEAPON_WORDS[w] not in ("fists", "kick"):
            ch.weapon = a.weapon = WEAPON_WORDS[w]
        loc = mods.get("at") or mods.get("start")
        if loc and parse_point(loc, True):
            ch.x, ch.z = parse_point(loc, True)
        if "facing" in mods and mods["facing"].lower() in COMPASS:
            ch.facing = math.degrees(COMPASS[mods["facing"].lower()])
        self.actors[key] = a
        self.order.append(key)
        self.chars.append(ch)
        return a

    def add_group(self, count: int, role: str, mods=None, flags=None) -> list[_Actor]:
        words = role.strip().split()
        role = " ".join(words[:-1] + [singular(words[-1])]) if words else role
        members = []
        for i in range(count):
            a = self.add_actor(f"{role.title()} {i + 1}", mods, flags)
            if a:
                a.group = role.lower()
                members.append(a)
        # a group given one start point forms a loose cluster around it instead of stacking
        chs = [c for c in self.chars if c.id in {m.id for m in members}]
        if len(chs) > 1 and chs[0].x is not None:
            cx, cz = chs[0].x, chs[0].z
            for i, c in enumerate(chs):
                ang = 2 * math.pi * i / len(chs)
                r = 0.6 + 0.22 * len(chs)
                c.x, c.z = round(cx + r * math.sin(ang), 2), round(cz + r * math.cos(ang), 2)
        return members

    def resolve(self, who: str, create: bool = True) -> list[_Actor]:
        """'All', 'Zombies', 'Zombie 3', 'Ravi, Priya', '5 Zombies', 'Actor 2' -> actors."""
        who = who.strip()
        low = who.lower()
        if low in ("all", "everyone", "everybody", "all actors"):
            return [self.actors[k] for k in self.order]
        m = re.fullmatch(r"(?:all\s+)?(\d+|" + "|".join(NUMBERS) + r")\s+(.+)", low)
        if m:
            n = int(m.group(1)) if m.group(1).isdigit() else NUMBERS[m.group(1)]
            role = who[m.start(2):]
            existing = self._group_members(role)
            return existing or self.add_group(min(n, MAX_CHARACTERS - len(self.chars)), role)
        m = re.fullmatch(r"all\s+(.+)", low)
        if m:
            return self._group_members(m.group(1))
        if "," in who or re.search(r"\s(?:and|&)\s", who):
            out = []
            for part in re.split(r",|\s(?:and|&)\s", who):
                if part.strip():
                    out += self.resolve(part, create)
            return out
        if low in self.actors and self.actors[low] is not None:
            return [self.actors[low]]
        group = self._group_members(who)
        if group:
            return group
        if create:
            a = self.add_actor(who)
            return [a] if a else []
        return []

    def _group_members(self, role: str) -> list[_Actor]:
        words = role.lower().strip().split()
        if not words:
            return []
        stem = " ".join(words[:-1] + [singular(words[-1])])
        return [self.actors[k] for k in self.order if self.actors[k].group in (stem, role.lower().strip())]

    def target(self, text: str, actor: _Actor | None = None):
        """-> ('char', id) | ('object', id) | ('point', (x, z)) | (None, None)."""
        t = re.sub(r"^(the|a|an|to|towards?|at|from|on|onto|with|into)\s+", "", text.strip(), flags=re.I).strip(" .")
        t = re.sub(r"^(the|a|an)\s+", "", t, flags=re.I).strip(" .")
        if not t:
            return None, None
        p = parse_point(t, grid_hint=bool(re.fullmatch(r"\d{1,2}[a-t]|[a-t]\d{1,2}", t, re.I)))
        if p:
            return "point", p
        low = t.lower()
        for o in self.objects:
            if low in (o.id.lower(), o.type) or low.startswith(o.type):
                return "object", o.id
        if low in self.actors and self.actors[low] is not None:
            return "char", self.actors[low].id
        for k in self.order:  # partial: "the guard" -> "Guard"
            if low == k or low.rstrip("s") == k:
                return "char", self.actors[k].id
        if re.search(r"\b(van|truck|vehicle)\b", low):
            return "object", self.ensure_object("van").id
        if low in ("center", "centre", "middle", "stage centre", "stage center"):
            return "point", (0.0, 0.0)
        if low in COMPASS or low.replace(" ", "") in COMPASS:
            ang = COMPASS[low.replace(" ", "")]
            return "point", (math.sin(ang) * 6, math.cos(ang) * 6)
        group = self._group_members(t)
        if group:
            return "char", group[0].id
        return None, None

    def ensure_object(self, typ: str, at: tuple[float, float] | None = None, yaw: float | None = None) -> SceneObject:
        for o in self.objects:
            if o.type == typ:
                return o
        x, z = at if at else ((0.0, 7.0) if typ == "van" else (0.0, 3.0))
        if yaw is None and typ in ("curb", "beam", "gap", "wall"):
            yaw = 0.0
        n = sum(o.type == typ for o in self.objects) + 1
        o = SceneObject(id=f"{typ}{n}", type=typ, x=x, z=z, yaw=yaw if yaw is not None else 0.0,
                        oriented=typ not in ("curb", "gap", "beam"))
        self.objects.append(o)
        return o

    # ------------------------------------------------------------------ clips
    def move_clip(self, actor: _Actor, speed: float, style: str | None, index: int = 0) -> tuple[str, float]:
        if actor.kind == "dog":
            return ("dog_run", 1.0) if speed >= 1.4 else ("dog_walk", speed)
        if actor.kind == "horse":
            return ("horse_gallop", 1.0) if speed >= 2.0 else ("horse_trot", 1.0) if speed >= 1.3 else ("horse_walk", speed)
        if actor.kind in ("car", "bike"):
            fast, slow = f"{actor.kind}_fast", f"{actor.kind}_slow"
            return (fast, min(2.0, max(0.6, speed / 2.0))) if speed >= 1.8 else (slow, max(0.5, min(2.0, speed)))
        if style:
            clip = STYLE_CLIPS.get(style)
            if clip == "walk_zombie" and index % 2 == 1:
                clip = "zombie_lurch"
            if clip:
                if speed >= 2.0 and style in ("scared", "afraid", "nervous", "zombie"):
                    return "run", min(1.2, speed / 2.0)
                return clip, max(0.6, min(1.3, speed if speed < 1.4 else 1.15))
        if speed >= 2.5:
            return "run", 1.2
        if speed >= 2.0:
            return "run", 1.0
        if speed >= 1.3:
            return "walk_quick", 1.0
        return "walk", max(0.6, speed)

    def idle_clip(self, actor: _Actor) -> str:
        return {"dog": "dog_idle", "horse": "horse_idle", "car": "car_idle", "bike": "bike_idle"}.get(actor.kind, "idle_wait")

    # ------------------------------------------------------------------ main
    def plan(self, text: str) -> ScenePlan:
        lines = [l.strip() for l in text.strip().splitlines() if l.strip() and not l.strip().startswith("#")]
        # cast and set lines first, so every reference resolves
        for line in lines:
            head, _, rest = line.partition(":")
            if head.strip().lower() == "cast":
                for entry in self._split_entries(rest):
                    body, mods, flags = _mods(entry)
                    m = re.fullmatch(r"(?:(\d+|" + "|".join(NUMBERS) + r")\s+)?(.+)", body.strip(), re.I)
                    if m and m.group(1):
                        n = int(m.group(1)) if m.group(1).isdigit() else NUMBERS[m.group(1).lower()]
                        self.add_group(n, m.group(2), mods, flags)
                    elif body.strip():
                        self.add_actor(body, mods, flags)
            elif head.strip().lower() in ("set", "objects", "object", "props"):
                for entry in self._split_entries(rest):
                    self._place_object(entry)
            elif head.strip().lower() in ("scene", "setting"):
                self._setting(rest)
        for line in lines:
            head, _, rest = line.partition(":")
            h = head.strip().lower()
            if h in ("cast", "set", "objects", "object", "props", "scene", "setting"):
                continue
            if h == "camera":
                self._camera(rest)
                continue
            if self._global_line(line):
                continue
            if not rest.strip():
                self.unmatched.append(line)
                continue
            self._command_line(head, rest)
        for a in (self.actors[k] for k in self.order):
            if a is not None and not any(b.character == a.id for b in self.beats):
                self.beats.append(Beat(character=a.id, start=0, duration=3, action="waits", clip=self.idle_clip(a)))
        title = next((l for l in lines if not l.lower().startswith(("cast", "set", "camera", "scene", "spawn", "explosion"))),
                     "Directed scene")
        return ScenePlan(title=re.sub(r"\[.*?\]", "", title)[:56].strip(), logline=" / ".join(lines)[:160],
                         setting=self.setting, characters=self.chars, beats=self.beats, shots=self.shots,
                         objects=self.objects, layers=self.layers, unmatched=self.unmatched,
                         extras_trim=self.extras_trim)

    @staticmethod
    def _split_entries(rest: str) -> list[str]:
        out, depth, cur = [], 0, ""
        for ch in rest:
            depth += ch == "["
            depth -= ch == "]"
            if ch in ",;" and depth == 0:
                out.append(cur)
                cur = ""
            else:
                cur += ch
        out.append(cur)
        return [e.strip() for e in out if e.strip()]

    def _place_object(self, entry: str):
        body, mods, _ = _mods(entry)
        m = re.match(r"(?:an?\s+|the\s+)?(van|truck|door|chair|ladder|button|panel|box|luggage|suitcase|crate|barbell|"
                     r"wall|fence|curb|kerb|step|log|gap|doorway|passage|beam|pipe|low ceiling|bar)\b(.*)", body.strip(), re.I)
        if not m:
            self.unmatched.append(entry)
            return
        typ = {"truck": "van", "panel": "button", "luggage": "box", "suitcase": "box", "fence": "wall", "kerb": "curb",
               "step": "curb", "log": "curb", "doorway": "gap", "passage": "gap", "pipe": "beam", "low ceiling": "beam",
               "bar": "beam"}.get(m.group(1).lower(), m.group(1).lower())
        p = parse_point(m.group(2) or mods.get("at", ""), True) or (0.0, 6.0)
        yaw = COMPASS.get(mods.get("facing", "").lower(), 0.0)
        n = sum(o.type == typ for o in self.objects) + 1
        length = _seconds(mods.get("length")) or (6.0 if typ == "wall" else 0.0)
        oriented = "facing" in mods or typ not in ("curb", "gap", "beam")
        self.objects.append(SceneObject(id=f"{typ}{n}", type=typ, x=p[0], z=p[1], yaw=yaw, length=length, oriented=oriented))

    def _setting(self, rest: str):
        low = rest.lower()
        tod = "night" if "night" in low else "dusk" if re.search("dusk|sunset|evening|dawn", low) else "day"
        self.setting = Setting(location=rest.split(",")[0].strip()[:40] or "mocap stage", time_of_day=tod,
                               mood=self.setting.mood)

    def _camera(self, rest: str):
        from .planner_offline import _camera_directive
        body, mods, _ = _mods(rest)
        cam = _camera_directive(body) or {"type": "medium", "move": None, "lens_mm": None}
        start = _seconds(mods.get("at"))
        if start is None:
            start = self.cursor
        subjects = []
        for key in ("on", "subject", "subjects"):
            if key in mods:
                subjects = [a.id for a in self.resolve(mods[key], create=False)]
        lens = _seconds(mods.get("lens")) or cam["lens_mm"]
        mv = (mods.get("move") or cam["move"] or "static").lower().replace(" ", "_").replace("-", "_")
        mv = {"pushin": "push_in", "push": "push_in", "pullout": "pull_out", "pull": "pull_out", "crane": "crane_up"}.get(mv, mv)
        if mv not in ("static", "push_in", "pull_out", "orbit", "crane_up", "handheld"):
            mv = "static"
        st = cam["type"] or ("wide" if mv in ("orbit", "crane_up", "pull_out") else "medium")
        self.shots.append(Shot(start=round(start, 2), duration=2.0, type=st, subjects=subjects, move=mv,
                               lens_mm=lens or {"close_up": 50, "wide": 28}.get(st, 35), note=body[:80],
                               fixed=_seconds(mods.get("at")) is not None))

    # ------------------------------------------------------------------ one command line
    def _command_line(self, who: str, rest: str):
        body, mods, flags = _mods(rest)
        actors = self.resolve(who)
        if not actors:
            self.unmatched.append(f"{who}: {rest}")
            return
        simul = mods.get("with", "").lower() in ("prev", "previous", "last", "above") or body.startswith("&") or \
            "with prev" in flags or "same time" in flags or re.match(r"^(meanwhile|while|as)\b", body, re.I)
        body = re.sub(r"^(&|meanwhile,?|while|as|then,?)\s*", "", body, flags=re.I)
        at = _seconds(mods.get("at"))
        start = at if at is not None else (self.prev_start if simul else self.cursor)
        start += _seconds(mods.get("delay")) or 0.0
        self.line_idx += 1
        self.line_mode = "at" if at is not None else "with" if simul else "seq"
        self.line_start = start - (_seconds(mods.get("delay")) or 0.0)
        self.line_mood = MOODS.get(mods.get("mood", "").lower()) if mods.get("mood") else None
        if self.line_mood:
            for a in actors:
                if a.kind == "human":
                    lay = self._layer(a, start, type="mood", mood=self.line_mood, duration=999.0)
                    lay.until_line = self.line_idx + 1
        parts = [p for p in re.split(r"\s*;\s*|\s+then\s+", body, flags=re.I) if p.strip()]
        line_end = start
        for part in parts:
            ends = []
            for i, a in enumerate(actors):
                jitter = (i % 4) * 0.15 if len(actors) > 2 else 0.0
                t0 = max(start + jitter, a.end) if not simul else max(start + jitter, a.end - 0.01)
                end = self._verb(a, part.strip(), mods, flags, t0, i, len(actors), actors)
                ends.append(end)
            start = max(ends) if ends else start
            line_end = max(line_end, start)
        self.prev_start = at if at is not None else (self.prev_start if simul else self.cursor)
        self.cursor = max(self.cursor, line_end)

    def _add(self, a: _Actor, start: float, dur: float, action: str, clip: str, **kw) -> float:
        dur = max(0.3, min(20.0, dur))
        b = Beat(character=a.id, start=round(start, 2), duration=round(dur, 2), action=action, clip=clip,
                 line=self.line_idx, line_mode=self.line_mode, line_offset=round(start - self.line_start, 3), **kw)
        self.beats.append(b)
        a.end = max(a.end, start + dur)
        return start + dur

    def _by_id(self, cid):
        return next((x for x in self.actors.values() if x is not None and x.id == cid), None)

    def _is_mount(self, cid) -> bool:
        x = self._by_id(cid)
        return x is not None and x.kind in MOUNTS

    def _loop_add(self, a, t0, total, text, clip, **kw) -> float:
        """A looping activity longer than one beat allows: back-to-back beats in whole-loop chunks."""
        L = self.lib.clips[clip].duration if clip in self.lib.clips else 4.0
        chunk = max(L, L * max(1, int(16.0 // max(L, 0.5))))
        t, left = t0, max(total, 0.3)
        while left > 0.05:
            d = min(chunk, left)
            t = self._add(a, t, d, text, clip, **kw)
            left -= d
            kw = {k: v for k, v in kw.items() if k in ("face",)}
        return t

    def _set_mood(self, a: _Actor, mood: str | None, t0: float) -> None:
        if a.mood is not None:
            a.mood.until_line = self.line_idx
            a.mood = None
        a.mood_name = mood
        if mood:
            a.mood = self._layer(a, t0, type="mood", mood=mood, duration=999.0)

    def _new_line(self, mode: str = "seq", at: float | None = None) -> float:
        self.line_idx += 1
        self.line_mode = mode
        start = at if at is not None else self.cursor
        self.line_start = start
        return start

    # ------------------------------------------------------------------ lines with no actor in front
    def _global_line(self, line: str) -> bool:
        body, mods, flags = _mods(line)
        low = body.lower().strip(" .:")
        m = re.match(r"^spawn\s+(?:an?\s+)?(?:(\d+|" + "|".join(NUMBERS) + r")\s+)?(crowd|extras?|background(?: actors)?|people|"
                     r"pedestrians|passers-?by|bystanders|patrons|customers|diners|workers|soldiers|shoppers|guests)\b", low)
        if m:
            self._spawn(m, mods)
            return True
        m = re.match(r"^(?:an?\s+)?(?:explosion|blast|bomb|grenade|big bang)(?:\s+(?:goes off|explodes|detonates))?"
                     r"(?:\s+(?:at|near|by|on|in)\s+(.+))?$", low)
        if m:
            at = _seconds(mods.get("at")) if re.fullmatch(r"\s*\d+(?:\.\d+)?\s*s?\s*", mods.get("at", "")) else None
            pt = parse_point(m.group(1) or "", True) or parse_point(mods.get("at", ""), True) or (0.0, 0.0)
            start = self._new_line("at" if at is not None else "seq", at)
            self.layers.append(Layer(type="fx", character="", prop="explosion", point=list(pt), start=round(start, 2),
                                     duration=4.0, line=self.line_idx, line_mode=self.line_mode, line_offset=0.0))
            self.prev_start = start
            self.cursor = max(self.cursor, start + 0.5)
            return True
        m = re.match(r"^teleport\s+(.+?)\s+(?:to|at)\s+(.+)$", body.strip(), re.I)
        if m:
            brackets = " ".join(f"[{b}]" for b in re.findall(r"\[(.*?)\]", line))
            self._command_line(m.group(1), f"teleport to {m.group(2)} {brackets}")
            return True
        return False

    def _spawn(self, m, mods) -> None:
        n = m.group(1)
        n = (int(n) if n.isdigit() else NUMBERS.get(n, 6)) if n else int(_seconds(mods.get("count")) or 6)
        role = m.group(2)
        generic = role in ("crowd", "extra", "extras", "people") or role.startswith("background")
        name = "Extra" if generic else singular(role).title()
        group = "crowd" if generic else singular(role)
        activity = (mods.get("activity") or mods.get("doing") or ("wander" if "path" in mods else "idle")).lower()
        total = _seconds(mods.get("for"))
        if total:
            self.extras_trim = False
        total = total or 40.0
        at_raw = mods.get("at", "")
        at = _seconds(at_raw) if re.fullmatch(r"\s*\d+(?:\.\d+)?\s*s?\s*", at_raw) else None
        spot = parse_point(at_raw, True) if at is None and at_raw else None
        spot = spot or parse_point(mods.get("near", ""), True) or (0.0, 0.0)
        start = self._new_line("bg", at)  # background: runs alongside the screenplay, never holds it up
        members = []
        for i in range(n):
            k = sum(1 for x in self.actors.values() if x is not None and x.group == group) + 1
            a = self.add_actor(f"{name} {k}", {}, set(), extra=True)
            if a is None:
                break
            a.group = group
            members.append(a)
        if not members:
            return
        if activity.startswith("wander") or activity in ("walk", "stroll", "path", "pass"):
            for i, a in enumerate(members):
                self._wander(a, start, mods, i, len(members), total)
        elif activity.startswith("chat") or activity in ("talk", "conversation", "converse"):
            for j in range(0, len(members), 2):
                pair = members[j:j + 2]
                cx, cz = spot[0] + (j // 2 % 3) * 2.6 - 2.6, spot[1] + (j // 6) * 2.6
                for i, a in enumerate(pair):
                    ch = next(c for c in self.chars if c.id == a.id)
                    if len(pair) == 2:
                        ch.x, ch.z = round(cx + (-0.55 if i == 0 else 0.55), 2), round(cz, 2)
                        ch.facing = pair[1 - i].id
                        self._loop_add(a, start + 0.2 * j, total, "chats", "chat_a" if i == 0 else "chat_b")
                    else:
                        ch.x, ch.z = round(cx, 2), round(cz + 1.0, 2)
                        self._loop_add(a, start, total, "loiters", "idle_phone")
        else:
            clip = next((v for k, v in ACTIVITY_CLIPS.items() if activity.startswith(k)), None)
            for i, a in enumerate(members):
                ch = next(c for c in self.chars if c.id == a.id)
                ang = 2.4 * i
                r = 1.4 + 0.45 * i
                ch.x, ch.z = round(spot[0] + r * math.sin(ang), 2), round(spot[1] + r * math.cos(ang), 2)
                c = clip or WALLFLOWER[i % len(WALLFLOWER)]
                if c == "salsa":
                    c = DANCES[i % len(DANCES)]
                self._loop_add(a, start + 0.25 * (i % 4), total, f"background: {activity}", c)

    def _wander(self, a: _Actor, start: float, mods, i: int, n: int, total: float) -> float:
        """Path-following extra: casual walks between the path's ends, pauses, random head turns."""
        path = re.split(r"\s+(?:to|-|->|and)\s+", mods.get("path", ""), flags=re.I)
        p1 = parse_point(path[0], True) if len(path) >= 2 else None
        p2 = parse_point(path[1], True) if len(path) >= 2 else None
        ch = next(c for c in self.chars if c.id == a.id)
        if not (p1 and p2):
            cx, cz = (ch.x or 0.0), (ch.z or 0.0)
            p1, p2 = (cx - 8.0, cz + 2.0 * (i % 3 - 1)), (cx + 8.0, cz + 2.0 * (i % 3 - 1))
        dx, dz = p2[0] - p1[0], p2[1] - p1[1]
        L = math.hypot(dx, dz) or 1.0
        nx, nz = -dz / L, dx / L  # lane offset across the path
        lane = (0.5 + 0.45 * (i % 3)) * (1 if i % 2 == 0 else -1)
        frac = (i + 0.5) / n
        ch.x, ch.z = round(p1[0] + dx * frac + nx * lane, 2), round(p1[1] + dz * frac + nz * lane, 2)
        ends = [(p2[0] + nx * lane, p2[1] + nz * lane), (p1[0] + nx * lane, p1[1] + nz * lane)]
        if i % 2:
            ends.reverse()
        clip = WANDER_CLIPS[i % len(WANDER_CLIPS)]
        sp = [0.9, 1.0, 1.1, 0.95, 1.05][i % 5]
        trips = max(1, math.ceil(total * 1.3 / L))
        t = start + 0.35 * (i % 5)
        for k in range(trips):
            dest = ends[k % 2]
            t = self._add(a, t, 3.0, "wanders", clip, move_to=MoveTo(point=[round(dest[0], 2), round(dest[1], 2)], distance=0.2),
                          speed=sp)
            if (i + k) % 3 == 0:  # pause at the end now and then
                t = self._add(a, t, 2.5 + (i % 3), "pauses", WALLFLOWER[(i + k) % len(WALLFLOWER)])
        h = abs(hash((a.id, "look"))) % 997
        for k in range(int(total // 4.5)):  # glances around while walking
            h = (h * 131 + 71) % 997
            ang = h / 997 * 2 * math.pi
            self.layers.append(Layer(type="look", character=a.id, point=[round(ch.x + 6 * math.sin(ang), 1),
                                                                         round(ch.z + 6 * math.cos(ang), 1)],
                                     start=round(start + 2.0 + 4.5 * k + (h % 10) / 10, 2), duration=1.6,
                                     line=self.line_idx, line_mode=self.line_mode,
                                     line_offset=round(2.0 + 4.5 * k + (h % 10) / 10, 2)))
        return t

    def _layer(self, a: _Actor, start: float, **kw) -> Layer:
        lay = Layer(character=a.id, start=round(start, 2), line=self.line_idx, line_mode=self.line_mode,
                    line_offset=round(start - self.line_start, 3), **kw)
        self.layers.append(lay)
        return lay

    def _look_target(self, text: str, a: _Actor, mods) -> dict:
        t = text.strip()
        if re.fullmatch(r"(?:the\s+)?(?:camera|cam|lens|audience)(?:\s*\d+)?", t, re.I):
            return {"target": "camera"}
        kind, tgt = self.target(t, a)
        if kind == "char":
            return {"target": tgt}
        if kind == "point":
            return {"point": list(tgt)}
        if kind == "object":
            o = next(o for o in self.objects if o.id == tgt)
            return {"point": [o.x, o.z]}
        return {}

    def _speed_style(self, mods, flags, default_speed=1.0):
        sp = mods.get("speed", "").lower()
        speed = SPEEDS.get(sp, None)
        if speed is None and sp:
            speed = _seconds(sp) or default_speed
        for f in flags:
            if f in SPEEDS and speed is None:
                speed = SPEEDS[f]
        style = mods.get("style", "").lower() or next((f for f in flags if f in STYLE_CLIPS), None)
        return speed or default_speed, style or None

    def _verb(self, a: _Actor, text: str, mods, flags, t0: float, idx: int, n: int, group) -> float:
        low = text.lower().strip(" .")
        dur_override = _seconds(mods.get("duration"))
        speed, style = self._speed_style(mods, flags)
        vary = [1.0, 0.94, 1.06, 0.97, 1.03][idx % 5] if n > 2 else 1.0

        # mounted riders: movement commands drive the horse
        if a.state == "ride" and a.horse and re.match(r"^(move|go|walk|run|ride|gallop|trot|head|approach|flee|drive|drives|speed)\b", low):
            horse = next(x for x in self.actors.values() if x is not None and x.id == a.horse)
            return self._verb(horse, text, mods, flags, max(t0, horse.end), idx, n, group)

        mood = self.line_mood or a.mood_name
        if mood:
            speed *= MOOD_SPEED.get(mood, 1.0)

        # ---- emotional state as an overlay: "Mood Sad", "Feel angry", "Become anxious"
        m = re.match(r"^(?:mood|feel|feels|feeling|become|becomes|get|gets|look|looks|act|acts|turn|turns|go|goes|grow|grows)"
                     r"\s+(\w+)$", low)
        if m and m.group(1) in MOODS and a.kind == "human":
            self._set_mood(a, MOODS[m.group(1)], t0)
            return t0

        # ---- teleport (the viewer dissolves the actor out and back in)
        m = re.match(r"^(?:teleport|teleports|warp|warps|blink|blinks|reappear|reappears)\s+(?:to|at|on|in)\s+(.+)$", low)
        if m:
            kind, tgt = self.target(text[m.start(1):].split("[")[0], a)
            pt = tgt if kind == "point" else None
            if kind == "object":
                o = next(o for o in self.objects if o.id == tgt)
                pt = (o.x, o.z)
            if kind == "char":
                ch = next(c for c in self.chars if c.id == tgt)
                pt = (ch.x or 0.0, ch.z or 0.0)
            if pt is None:
                self.unmatched.append(text)
                return t0
            if a.state == "ride":
                a.state, a.horse = "stand", None
            return self._add(a, t0, 0.8, text, self.idle_clip(a), move_to=MoveTo(point=[pt[0], pt[1]], distance=0.0),
                             teleport=True)

        # ---- crowd verbs
        if re.match(r"^(?:wander|wanders|stroll around|strolls around|mill around|mills around|walk around|walks around|"
                    r"pass by|passes by)\b", low) and a.kind == "human":
            return self._wander(a, t0, mods, idx, n, dur_override or _seconds(mods.get("for")) or 20.0)
        m = re.match(r"^(?:chat|chats|chatting|converse|converses|talk|talks|gossip|gossips|make small talk|small talk)"
                     r"(?:\s+(?:with|to)\s+(.+))?$", low)
        if m and a.kind == "human":
            total = dur_override or _seconds(mods.get("for")) or 8.0
            partner = None
            if m.group(1):
                kind, tgt = self.target(text[m.start(1):], a)
                partner = tgt if kind == "char" else None
            elif n > 1:
                partner = group[idx ^ 1].id if (idx ^ 1) < n else group[idx - 1].id
            end = self._loop_add(a, t0, total, text, "chat_a" if idx % 2 == 0 else "chat_b", face=partner)
            p = self._by_id(partner) if partner and m.group(1) else None
            if p is not None and p.kind == "human":  # the partner joins in, out of phase
                self._loop_add(p, max(t0, p.end), total, f"chats with {a.name}", "chat_b", face=a.id)
            return end
        m = re.match(r"^(?:fight|fights|brawl|brawls|spar|spars|scuffle|scuffles|fight each other)(?:\s+(?:with\s+)?(.+))?$", low)
        if m and a.kind == "human":
            partner = None
            if m.group(1) and m.group(1) not in ("each other", "it out"):
                kind, tgt = self.target(text[m.start(1):], a)
                partner = tgt if kind == "char" else None
            if partner is None and n > 1:
                partner = group[idx ^ 1].id if (idx ^ 1) < n else group[idx - 1].id
            t = t0
            for k in range(int(_seconds(mods.get("rounds")) or 2)):
                clip = FIGHT_POOL[(idx * 3 + k) % len(FIGHT_POOL)]
                t = self._add(a, t, 2.4, f"fights ({clip})", clip, face=partner)
            return t

        # ---- ambient loops: wallflower idles and prop work
        AMBIENT = [(r"check(?:s)? (?:the |his |her |their |a )?phone|look(?:s)? at (?:the |his |her |their |a )?phone|text(?:s|ing)?|scroll(?:s)?",
                    "idle_phone"),
                   (r"check(?:s)? (?:the |his |her |their |a )?(?:watch|time)|look(?:s)? at (?:the |his |her |their )?watch", "idle_watch"),
                   (r"cross(?:es)? (?:the |his |her |their )?arms|fold(?:s)? (?:the |his |her |their )?arms|arms crossed", "idle_arms_crossed"),
                   (r"loiter(?:s)?|hang(?:s)? around|shift(?:s)? weight|wait(?:s)? around|stand(?:s)? around|idle around", "idle_shift"),
                   (r"wipe(?:s)? (?:down )?(?:the |a )?(?:counter|table|bar|desk)", "wipe_counter"),
                   (r"sweep(?:s)?(?: the floor)?|sweeping", "sweep"),
                   (r"type(?:s)?|typing|use(?:s)? (?:a |the |his |her )?laptop|work(?:s)? on (?:a |the |his |her )?laptop", "type_laptop"),
                   (r"read(?:s)?(?: (?:a |the |his |her |their )?(?:book|newspaper|paper|novel))?|reading", "read_book")]
        for pat, clip in AMBIENT:
            if re.fullmatch(pat, low) and a.kind == "human":
                return self._loop_add(a, t0, dur_override or _seconds(mods.get("for")) or 6.0, text, clip)

        # ---- pacing and continuity beats
        if re.match(r"^(?:think|thinks|ponder|ponders|contemplate|contemplates|consider|considers|rub(?:s)? (?:the |his |her |their )?chin|"
                    r"tap(?:s)? (?:the |his |her |their )?foot|mull(?:s)? it over|wonder|wonders)\b", low) and a.kind == "human":
            clip = "pace_circle" if re.search(r"pace|circle", mods.get("style", "").lower() + " " + low) else "think"
            return self._loop_add(a, t0, dur_override or _seconds(mods.get("for")) or (7.2 if clip == "pace_circle" else 6.0), text, clip)
        if re.match(r"^(?:is exhausted|exhausted|catch(?:es)? (?:his |her |their |my )?breath|hands on knees|pant|pants|gasp|gasps|"
                    r"out of breath|doubles over panting)\b", low) and a.kind == "human":
            return self._loop_add(a, t0, dur_override or 4.0, text, "exhausted")
        if re.match(r"^(?:wipe|wipes|mop|mops)\s+(?:the\s+|his\s+|her\s+|their\s+)?(?:sweat|brow|forehead)", low):
            return self._add(a, t0, 2.6, text, "wipe_sweat")
        if re.match(r"^(?:wake up|wakes up|come to|comes to|recover|recovers|regain(?:s)? consciousness|come round|comes round|"
                    r"get up groggy|gets up groggy)\b", low) and a.kind == "human":
            if a.down == "collapse" or (a.state in ("lying", "prone") and a.down is None):
                a.state, a.down = "stand", None
                return self._add(a, t0, self.lib.clips["wake_up"].duration, text, "wake_up")
            if a.state in ("lying", "prone"):
                prev, a.state, a.down = a.state, "stand", None
                clip = "get_up_prone" if prev == "prone" else "get_up"
                t = self._add(a, t0, self.lib.clips[clip].duration, text, clip)
                return self._add(a, t, 2.6, "shakes it off", "wipe_sweat")
            return self._add(a, t0, 2.6, text, "wipe_sweat")
        if re.match(r"^(?:flinch|flinches|shield|shields|brace|braces|cover(?:s)? (?:his |her |their |my )?(?:face|head)|duck and cover)\b", low) \
                and a.kind == "human":
            return self._add(a, t0, 1.7, text, "flinch_shield")

        # ---- system / director utilities
        if re.match(r"^(?:t-?pose|a-?pose|calibrate|calibrates|calibration|reset|resets|reset pose|bind pose)\b", low) and a.kind == "human":
            return self._add(a, t0, dur_override or 2.0, text, "tpose" if re.match(r"^t", low) else "apose")
        if re.match(r"^(?:stand-in|stand in|standin|lighting check|light check|turntable)\b", low) and a.kind == "human":
            return self._loop_add(a, t0, dur_override or 8.0, text, "standin_turn")  # grey only during the turn
        m = re.match(r"^(?:scale ref(?:erence)?|height ref(?:erence)?|stand rigid|for scale|show scale|scale check)"
                     r"(?:\s+(?:next to|beside|by|near|at)\s+(.+))?$", low)
        if m and a.kind == "human":
            t = t0
            near = m.group(1) or mods.get("next", "")
            if near:
                kind, tgt = self.target(near, a)
                mv = self._move_to(kind, tgt, mods, idx, n, default_dist=1.2)
                if mv is not None:
                    t = self._add(a, t, 3.0, f"walks to {near}", "walk", move_to=mv)
            return self._add(a, t, dur_override or 4.0, text, "scale_ref")
        if re.match(r"^(?:eye-?line|eyeline|eye line|sight-?line|sight line)(?:\s+(?:calibration|check|sweep|pan))?\b", low) and a.kind == "human":
            return self._add(a, t0, 9.0, text, "eyeline_sweep")

        # ---- stunts
        m = re.match(r"^(?:stagger|staggers|get pushed|gets pushed|is pushed|get shoved|gets shoved|is shoved|lose(?:s)? balance)"
                     r"(?:\s+from\s+(?:the\s+)?(\w+))?", low)
        if m and a.kind == "human":
            frm = (m.group(1) or mods.get("from", "front")).lower()
            moved = {"front": "back", "ahead": "back", "back": "forward", "behind": "forward", "left": "right",
                     "right": "left"}.get(frm, "back")
            return self._add(a, t0, 1.6, text, f"stagger_{moved}")
        if re.match(r"^(?:leap of faith|leap|leaps|dive roll|dives forward|dive forward|diving roll|jump and roll|jumps and rolls)\b", low) \
                and a.kind == "human" and a.state != "ride":
            return self._add(a, t0, self.lib.clips["leap_roll"].duration, text, "leap_roll")
        if re.match(r"^(?:high fall|fall from (?:a )?height|falls from (?:a )?height|drop in|drops in|drop down|drops down|"
                    r"jump down|jumps down|fall(?:s)? from the roof|jump(?:s)? off the roof)\b", low) and a.kind == "human":
            land = mods.get("land") or mods.get("to")
            pt = parse_point(land, True) if land else None
            kw = {"move_to": MoveTo(point=[pt[0], pt[1]], distance=0.0), "teleport": True} if pt else {}
            return self._add(a, t0, self.lib.clips["high_fall"].duration, text, "high_fall", **kw)
        m = re.match(r"^(?:push|pushes|shove|shoves)\s+(.+)$", low)
        if m and self.target(text[m.start(1):], a)[0] == "char":
            return self._attack(a, text, low, m, mods, flags, t0, idx)

        # ---- layered look-at: the head (and a little of the chest) tracks while the body carries on
        m = re.match(r"^(?:look|looks|lookat|glance|glances|watch|watches|track|tracks|keep looking|eyes on|stare|stares)"
                     r"\s+(?:at\s+|on\s+|to(?:wards?)?\s+)?(?!around\b)(.+)$", low)
        if m and a.kind == "human":
            tgt = self._look_target(text[m.start(1):], a, mods)
            if tgt:
                dur = dur_override or _seconds(mods.get("for")) or (999.0 if re.search(r"keep|track|watch", low)
                                                                    or mods.get("for", "").lower() in ("all", "end") else 3.0)
                self._layer(a, t0, type="look", duration=dur, **tgt)
                return t0

        # ---- blind reaction: a sudden head snap and flinch toward a sound or an entrance
        m = re.match(r"^(?:startle|startles|is startled|startled|jump(?:s)? at|hear(?:s)?|react(?:s)? to (?:a |the )?"
                     r"(?:sound|noise|bang|shot|crash|footsteps?|entrance))\b\s*(.*)$", low)
        if m and a.kind == "human":
            src = mods.get("from") or re.sub(r"^.*?\b(?:from|at|by|near)\s+", "", m.group(1)) if (mods.get("from") or re.search(r"\b(from|at|by|near)\s", m.group(1))) else ""
            tgt = self._look_target(src, a, mods) if src else {}
            if not tgt:
                tgt = {"point": [0.0, 20.0]}  # the north edge of the stage
            self._layer(a, t0, type="look", duration=dur_override or 3.0, snap=True, **tgt)
            return self._add(a, t0, self.lib.clips["startle"].duration, text, "startle",
                             face_point=None)

        # ---- strafe: sidestep while the chest stays on a target
        m = re.match(r"^(?:strafe|strafes|sidestep|sidesteps|side step|side steps|shuffle|shuffles)\s+(left|right)"
                     r"(?:\s+(?:around|facing|while facing|watching|covering)\s+(.+))?$", low) or \
            re.match(r"^(?:strafe|strafes)(?:\s+(left|right))?(?:\s+(?:around|facing|while facing|watching|covering)\s+(.+))?$", low)
        if m and a.kind == "human":
            side = m.group(1) or mods.get("direction", "left").lower()
            facing = m.group(2) or mods.get("facing", "")
            kind, tgt = self.target(facing, a) if facing else (None, None)
            dist = _seconds(mods.get("distance")) or 3.0
            clip = "walk_side" if side == "left" else "walk_side_r"
            sp = 1.3 if speed >= 1.4 else 1.0
            return self._add(a, t0, dur_override or dist / (0.95 * sp), text, clip, speed=sp,
                             move_to=MoveTo(target=tgt if kind == "char" else "", strafe=side, distance=dist),
                             face=tgt if kind == "char" else None)

        # ---- obstacles: walk up to it, then cross it with the matching traversal motion
        for pat, (otype, tclip) in OBSTACLES.items():
            m = re.match(rf"^(?:{pat})\s+(?:the\s+|a\s+|an\s+)?(.+)$", low)
            if m and a.kind == "human":
                kind, tgt = self.target(m.group(1), a)
                obj = next((o for o in self.objects if o.id == tgt), None) if kind == "object" else None
                if obj is None or obj.type not in ("curb", "beam", "gap"):
                    obj = self.ensure_object(otype)
                clip = {"curb": "step_over", "beam": "duck_under", "gap": "squeeze"}[obj.type]
                walk, sp = self.move_clip(a, speed, style, idx)
                t = self._add(a, t0, 3.0, f"walks up to the {obj.type}", walk, speed=sp,
                              move_to=MoveTo(target=obj.id, through=obj.id, phase="near", distance=0.1))
                return self._add(a, t, self.lib.clips[clip].duration, text, clip,
                                 move_to=MoveTo(target=obj.id, through=obj.id, phase="far", distance=0.1))

        # ---- props held in one hand: the arm takes a grip pose while the body keeps moving
        m = re.match(r"^(?:hold|holds|pick up|picks up|grab|grabs|take|takes|carry|carries|sip|sips|drink|drinks)"
                     r"\s+(?:from\s+)?(?:a|an|the|his|her|their)?\s*(\w+)(?:\s+(?:to|towards?)\s+(.+))?$", low)
        if m and a.kind == "human" and m.group(1) in GRIP_PROPS:
            prop = GRIP_PROPS[m.group(1)]
            if a.grip is None or a.grip.prop != prop:
                a.grip = self._layer(a, t0, type="grip", prop=prop, duration=999.0)
            if m.group(2):
                kind, tgt = self.target(text[m.start(2):], a)
                clip, sp = self.move_clip(a, speed, style, idx)
                mv = self._move_to(kind, tgt, mods, idx, n, default_dist=0.4)
                if mv is not None:
                    return self._add(a, t0, dur_override or 3.0, text, clip, move_to=mv, speed=sp)
            if re.match(r"^(?:sip|sips|drink|drinks)", low):
                return self._add(a, t0, dur_override or 2.5, text, "drink" if "drink" in self.lib.clips else self.idle_clip(a))
            return self._add(a, t0, dur_override or 1.2, text, self.idle_clip(a))
        if a.grip is not None and re.match(r"^(?:put down|puts down|set down|sets down|drop|drops|let go of|release|releases|"
                                           r"leave|leaves|hand over|hands over)\b", low) and \
                (any(w in low for w in GRIP_PROPS) or not re.search(r"luggage|box|crate|load|weight", low)):
            a.grip.until_line = self.line_idx
            a.grip = None
            return self._add(a, t0, dur_override or 1.0, text, self.idle_clip(a))

        # ---- defeat and clean-up: collapse, staying down, crawling on the ground
        if re.match(r"^(?:collapse|collapses|faint|faints|pass out|passes out|swoon|swoons|keel over|keels over|"
                    r"drop dead|drops dead|die|dies|black out|blacks out|crumple|crumples)\b", low) and a.kind == "human":
            a.state, a.down = "lying", "collapse"
            return self._add(a, t0, self.lib.clips["collapse"].duration, text, "collapse")
        m = re.match(r"^(?:stay down|stays down|lie still|lies still|play dead|plays dead|groan|groans|moan|moans|writhe|"
                     r"writhes|stay on the ground|stays on the ground|crawl|crawls)\b(?:\s+(?:to|towards?)\s+(.+))?", low)
        if m and a.kind == "human" and (a.state in ("lying", "prone") or not low.startswith(("crawl", "play dead"))):
            style_d = DOWN_STYLES.get(mods.get("style", "").lower()) or next(
                (v for k, v in DOWN_STYLES.items() if k in low), "still")
            if a.state not in ("lying", "prone"):
                a.state, a.down = "lying", "lie_down"
                t0 = self._add(a, t0, self.lib.clips["lie_down"].duration, "lies down", "lie_down")
            src = a.down or ("prone" if a.state == "prone" else "lie_down")
            if style_d == "crawl" and f"crawl_{src}" in self.lib.clips:
                mv = None
                if m.group(1):
                    kind, tgt = self.target(text[m.start(1):], a)
                    mv = self._move_to(kind, tgt, mods, idx, n, default_dist=0.3)
                return self._add(a, t0, dur_override or 6.0, text, f"crawl_{src}", move_to=mv, speed=1.4)
            if style_d == "crawl":
                style_d = "groan"
            return self._add(a, t0, dur_override or 4.0, text, f"down_{style_d}_{src}")

        m = re.match(r"^(?:move|go|walk|run|jog|sprint|head|travel|ride|gallop|trot|stalk|sneak|march|crawl|drive|drives|speed)\s+(?:back\s+)?(?:to|towards?|into|over to|up to)\s+(.+)$", low)
        if m:
            verb = low.split()[0]
            if verb in ("speed",) and "speed" not in mods:
                speed = max(speed, 2.5)
            if verb in ("run", "sprint", "gallop") and "speed" not in mods:
                speed = max(speed, 2.0)
            if verb == "jog" and "speed" not in mods:
                speed = max(speed, 1.4)
            if verb in ("stalk", "sneak") and not style:
                style = "sneaky"
            if verb == "march" and not style:
                style = "march"
            if verb == "crawl":
                return self._add(a, t0, dur_override or 4, text, "crawl")
            kind, tgt = self.target(text[m.start(1):], a)
            clip, sp = self.move_clip(a, speed, style, idx)
            mv = self._move_to(kind, tgt, mods, idx, n, default_dist=0.25 if kind == "point" else 1.3)
            if mv is None:
                self.unmatched.append(text)
                return t0
            return self._add(a, t0, dur_override or 3.0, text, clip, move_to=mv, speed=round(min(2.0, sp * vary), 2))

        m = re.match(r"^(?:approach|approaches|close in on|walk up to|go up to|charge(?: at)?|chase|follow|surround)\s+(.+)$", low)
        if m:
            kind, tgt = self.target(text[m.start(1):], a)
            if low.startswith(("charge", "chase")) and "speed" not in mods:
                speed = 2.0
            clip, sp = self.move_clip(a, speed, style, idx)
            dist = _seconds(mods.get("distance")) or (2.2 if self._is_horse(tgt) else 1.2)
            if low.startswith("surround"):
                dist = max(dist, 1.6)
            mv = self._move_to(kind, tgt, mods, idx, n, default_dist=dist)
            if mv is None:
                self.unmatched.append(text)
                return t0
            return self._add(a, t0, dur_override or 3.0, text, clip, move_to=mv, speed=round(min(2.0, sp * vary), 2),
                             face=tgt if kind == "char" else None)

        m = re.match(r"^(?:retreat|retreats|back away|backs away|back off|step back|steps back)(?:\s+from\s+(.+))?$", low)
        if m:
            kind, tgt = self.target(text[m.start(1):], a) if m.group(1) else (None, None)
            dist = _seconds(mods.get("distance")) or 3.0
            if a.kind != "human":
                clip, sp = self.move_clip(a, speed, None)
            else:
                clip, sp = ("walk_crouch_back" if style in ("crouched", "crouch", "low") else "walk_back"), 1.0
            mv = MoveTo(target=tgt or "", away=True, distance=dist) if kind == "char" else None
            dur = dur_override or dist / 1.1
            return self._add(a, t0, dur, text, clip, move_to=mv, face=tgt if (kind == "char" and a.kind == "human") else None)

        m = re.match(r"^(?:flee|flees|run away|runs away|escape|escapes|bolt)(?:\s+from\s+(.+))?$", low)
        if m:
            kind, tgt = self.target(text[m.start(1):], a) if m.group(1) else (None, None)
            clip, sp = self.move_clip(a, 2.5, "scared" if a.kind == "human" else None, idx)
            dist = _seconds(mods.get("distance")) or 9.0
            mv = MoveTo(target=tgt, away=True, distance=dist) if kind == "char" else MoveTo(point=[0.0, -14.0], distance=0.3)
            return self._add(a, t0, dur_override or 3.0, text, clip, move_to=mv, speed=round(min(2.0, sp * vary), 2))

        m = re.match(r"^(?:turn to|turns to|turn towards?|face|faces|look at|looks at|turn to face)\s+(.+)$", low)
        if m:
            kind, tgt = self.target(text[m.start(1):], a)
            dirw = m.group(1).strip().replace(" ", "")
            fp = None
            if dirw in COMPASS:
                ang = COMPASS[dirw]
                fp = [math.sin(ang) * 50, math.cos(ang) * 50]
            elif kind == "point":
                fp = list(tgt)
            return self._add(a, t0, dur_override or 1.2, text, self.idle_clip(a),
                             face=tgt if kind == "char" else None, face_point=fp,
                             target_object=tgt if kind == "object" else None)

        MOUNT_VERBS = (r"mount|mounts|get on|gets on|climb on|jump on|get in|gets in|get into|gets into|hop in|hops in|"
                       r"climb in|climbs in|climb into|board|boards|enter|enters")
        if re.match(rf"^(?:{MOUNT_VERBS})\b", low) and a.kind == "human":
            rest = re.sub(rf"^(?:{MOUNT_VERBS})\s*", "", low)
            kind, tgt = self.target(rest or "horse", a)
            if kind != "char" or not self._is_mount(tgt):
                want = "car" if re.search(CAR_WORDS, rest) else "bike" if re.search(BIKE_WORDS, rest) else "horse"
                mounts = [x for x in self.actors.values() if x is not None and x.kind in MOUNTS]
                pick = [x for x in mounts if x.kind == want] or mounts
                tgt = pick[0].id if pick else self.add_actor({"car": "Car", "bike": "Bike"}.get(want, "Horse")).id
            horse = self._by_id(tgt)
            riders = [x for x in self.actors.values() if x is not None and x.horse == tgt and x.state == "ride"]
            seat_word = mods.get("seat", "").lower()
            seat = {"driver": 0, "front": 1, "passenger": 1, "shotgun": 1, "back": 2, "rear": 2}.get(seat_word, len(riders))
            t = self._add(a, t0, 1.6, f"walks to {horse.name}", self.move_clip(a, 1.0, None)[0],
                          move_to=MoveTo(target=tgt, distance=1.0 if horse.kind == "horse" else 1.6), face=tgt)
            a.state, a.horse, a.seat = "ride", tgt, seat
            horse.end = max(horse.end, t + 1.4)
            clip = "ride" if horse.kind in ("horse", "bike") else ("drive_steer" if seat == 0 else "passenger_idle")
            return self._add(a, t, 1.6, text, clip, ride=tgt, **({"seat": seat} if seat else {}))

        if re.match(r"^(?:dismount|dismounts|get off|gets off|jump off|climb down|get out|gets out|step out|steps out|exit|exits|"
                    r"jump out|jumps out|bail out|bails out|eject|ejects|dive out|dives out|leap out|leaps out)\b", low):
            if a.state != "ride":
                return t0
            horse = self._by_id(a.horse)
            a.state, a.horse = "stand", None
            if horse is not None and horse.kind in ("car", "bike") and re.match(r"^(?:jump|bail|eject|dive|leap)", low):
                return self._add(a, t0, self.lib.clips["eject_roll"].duration, text, "eject_roll")
            return self._add(a, t0, 1.6, text, "idle_wait")

        m = re.match(r"^(?:sit|sits)(?:\s+down)?(?:\s+on\s+(.+))?$", low)
        if m:
            if a.kind == "dog":
                a.state = "sit"
                return self._add(a, t0, dur_override or 3.0, text, "dog_sit")
            a.state = "sit"
            t = self._add(a, t0, 3.0, text, "sit_down")
            return self._add(a, t, dur_override or 2.0, "stays seated", "sit_chair") if dur_override else t

        if re.match(r"^(?:stand|stands|stand up|stands up|get up|gets up|rise|rises|get to (?:his|her|their) feet)$", low):
            prev, a.state, a.down = a.state, "stand", None
            if a.kind != "human":
                return self._add(a, t0, 2.0, text, self.idle_clip(a))
            clip = {"sit": "stand_up", "prone": "get_up_prone", "lying": "get_up"}.get(prev, "idle_wait")
            return self._add(a, t0, self.lib.clips[clip].duration if clip != "idle_wait" else 1.0, text, clip)

        if re.match(r"^(?:crouch|crouches|crouch down|take cover|takes cover|hunker down|get low)$", low):
            a.state = "crouch"
            return self._add(a, t0, dur_override or 3.0, text, "crouch")
        if re.match(r"^(?:kneel|kneels|kneel down|take a knee|on one knee)$", low):
            a.state = "kneel"
            return self._add(a, t0, dur_override or self.lib.clips["kneel"].duration, text, "kneel")
        if re.match(r"^(?:prone|go prone|goes prone|lie face down|lies face down|drop prone|hit the deck|get down)$", low):
            a.state, a.down = "prone", "prone"
            return self._add(a, t0, dur_override or self.lib.clips["prone"].duration, text, "prone")
        if re.match(r"^(?:lie down|lies down|lay down|lays down|lie on (?:his|her|their) back|play dead)$", low):
            a.state, a.down = "lying", "lie_down"
            return self._add(a, t0, dur_override or self.lib.clips["lie_down"].duration, text, "lie_down")

        m = re.match(r"^(?:pace|paces)(?:\s+between\s+(.+?)\s+and\s+(.+))?$", low)
        if m:
            p1 = parse_point(m.group(1), True) if m.group(1) else None
            p2 = parse_point(m.group(2), True) if m.group(2) else None
            times = int(_seconds(mods.get("times")) or 3)
            t = t0
            if p1 and p2:
                for k in range(times):
                    pt = p2 if k % 2 == 0 else p1
                    t = self._add(a, t, 3.0, f"paces to {pt}", "walk", move_to=MoveTo(point=list(pt), distance=0.2))
                return t
            return self._add(a, t0, dur_override or 6.0, text, "pace")

        m = re.match(r"^(?:say|says|tell|tells|shout|shouts|whisper|whispers)\s*[:\"“']?(.+?)[\"”']?$", text.strip(), re.I)
        if m and re.match(r"^(say|says|tell|tells|shout|shouts|whisper|whispers)\b", low):
            line = m.group(1).strip(" \"'“”")
            clip = "argue" if low.startswith("shout") else "idle_breathe"
            return self._add(a, t0, dur_override or max(2.0, len(line.split()) * 0.42), text, clip, say=line)

        # attacks and reactions
        m = re.match(r"^(?:attack|attacks|hit|hits|strike|strikes|punch|punches|kick|kicks|slash|slashes|stab|stabs|smash|smashes|"
                     r"shoot|shoots|fire at|fires at|takedown|take down|takes down|finish|finishes|finisher on|bite|bites|"
                     r"roundhouse|side kick|karate punch)\s+(.+)$", low)
        if m:
            return self._attack(a, text, low, m, mods, flags, t0, idx)

        if re.match(r"^(?:block|blocks|parry|parries)\b", low):
            return self._add(a, t0, dur_override or 1.4, text, "block")
        if re.match(r"^(?:dodge|dodges|evade|evades|sidestep)\b", low):
            return self._add(a, t0, dur_override or 2.0, text, "evade")
        if re.match(r"^(?:duck|ducks)\b", low):
            return self._add(a, t0, dur_override or 2.5, text, "duck")
        if re.match(r"^(?:react|reacts|get hit|gets hit|take a hit|takes a hit|hit reaction|recoil|recoils|flinch|flinches)\b", low):
            clip = "hit_gut" if re.search(r"gut|stomach|body", low + " " + mods.get("style", "")) else "hit_react"
            return self._add(a, t0, 1.5, text, clip)
        if re.match(r"^(?:stagger|staggers|stumble|stumbles)\b", low):
            return self._add(a, t0, 2.5, text, "stumble")
        if re.match(r"^(?:fall|falls|collapse|collapses|fall down|falls down)\b", low):
            a.state = "lying"
            a.down = "fall_back" if "back" in low else "fall_face"
            return self._add(a, t0, 3.0, text, a.down)

        # interact / use
        if re.match(r"^(?:open|opens)\s+(?:the\s+)?door", low):
            return self._add(a, t0, 2.6, text, "open_door")
        if re.match(r"^(?:push|pushes|press|presses|hit|hits)\s+(?:the\s+|a\s+)?(?:button|switch|panel|doorbell|keypad)", low):
            return self._add(a, t0, 2.0, text, "push_button")
        if re.match(r"^(?:climb|climbs)\s+(?:up\s+)?(?:the\s+|a\s+)?ladder", low):
            return self._add(a, t0, 5.0, text, "climb")
        m = re.match(r"^(?:equip|equips|draw|draws|unholster|unholsters|pull out|pulls out|arm|arms)(?:\s+(?:a|an|the|his|her|their))?\s*(\w+)?", low)
        if m and low.split()[0] in ("equip", "equips", "draw", "draws", "unholster", "unholsters", "pull", "pulls", "arm", "arms"):
            w = WEAPON_WORDS.get((m.group(1) or "").lower(), mods.get("weapon", "gun").lower())
            w = WEAPON_WORDS.get(w, w) if w not in ("fists", "kick") else "gun"
            a.weapon = w
            return self._add(a, t0, 2.4, text, "draw_weapon", weapon=w)
        if re.match(r"^(?:unequip|unequips|holster|holsters|sheathe|sheathes|put away|puts away|lower|lowers)\b", low):
            a.weapon = None
            return self._add(a, t0, 2.4, text, "holster", weapon="none")

        if re.match(r"^(?:lift|lifts)\s+(?:the\s+)?weights|^(?:weightlift|weight lift|bench press|deadlift)", low):
            return self._add(a, t0, dur_override or 5.0, text, "weights")
        m = re.match(r"^(?:load|loads|put|puts|place|places|stow|stows)\s+(?:the\s+)?(.+?)\s+(?:in|into|inside)\s+(?:the\s+)?(van|truck|vehicle|car)$", low)
        if m:
            van = self.ensure_object("van")
            t = t0
            if a.state != "carry":
                t = self._add(a, t, self.lib.clips["lift_heavy"].duration, f"lifts the {m.group(1)}", "lift_heavy")
            t = self._add(a, t, 3.0, f"carries the {m.group(1)} to the van", "carry_walk",
                          move_to=MoveTo(target=van.id, distance=0.7))
            a.state = "stand"
            return self._add(a, t, 3.2, text, "set_down_heavy", target_object=van.id)
        m = re.match(r"^(?:carry|carries)\s+(?:the\s+)?(.+?)\s+(?:to|towards?)\s+(.+)$", low)
        if m:
            t = t0
            if a.state != "carry":
                t = self._add(a, t, self.lib.clips["lift_heavy"].duration, f"lifts the {m.group(1)}", "lift_heavy")
            a.state = "carry"
            kind, tgt = self.target(text[m.start(2):], a)
            mv = self._move_to(kind, tgt, mods, idx, n, default_dist=0.7)
            return self._add(a, t, 3.0, text, "carry_walk", move_to=mv)
        if re.match(r"^(?:lift|lifts|pick up|picks up|heave|heaves)\s+(?:the\s+|a\s+)?(?:heavy\s+)?(?:luggage|suitcase|box|crate|bag|load|weight)", low):
            a.state = "carry"
            return self._add(a, t0, self.lib.clips["lift_heavy"].duration, text, "lift_heavy")
        if re.match(r"^(?:set down|sets down|put down|puts down|drop|drops)\b", low):
            a.state = "stand"
            return self._add(a, t0, 3.2, text, "set_down_heavy")
        if re.match(r"^(?:pick up|picks up|grab|grabs|take|takes)\b", low):
            return self._add(a, t0, 4.0, text, "pick_up")

        m = re.match(r"^(?:emote|gesture|do|does)\s+(?:a\s+|an\s+)?(.+)$", low)
        key = (m.group(1) if m else low).strip()
        for word, clip in EMOTES.items():
            if re.match(rf"^{word}s?\b", key) or key == word:
                if a.kind != "human":
                    break
                kind, tgt = self.target(re.sub(rf"^{word}s?\s*(at|to)?\s*", "", key), a)
                face = tgt if kind == "char" else None
                if clip == "chicken_dance" and n > 2:  # a dancing crowd gets a mix of dances
                    clip = DANCES[idx % len(DANCES)]
                d = self.lib.clips[clip].duration
                if clip in ("sing",) or self.lib.clips[clip].category == "dance":
                    return self._loop_add(a, t0, dur_override or _seconds(mods.get("for")) or 6.0, text, clip, face=face)
                return self._add(a, t0, dur_override or max(2.0, min(d, 4.0)), text, clip, face=face)
        if re.match(r"^(?:listen|listens|nod|nods)\b", low):
            kind, tgt = self.target(re.sub(r"^(?:listen|listens|nod|nods)\s*(?:to|at)?\s*", "", low), a)
            return self._add(a, t0, dur_override or 3.0, text, "nod", face=tgt if kind == "char" else None)
        if re.match(r"^(?:cower|cowers|shrink|shrinks|tremble|trembles)\b", low):
            return self._add(a, t0, dur_override or 3.0, text, "cower_low" if "low" in low + " ".join(flags) else "cower")
        if re.match(r"^(?:exclaim|exclaims|cheer|cheers|celebrate|celebrates)\b", low):
            return self._add(a, t0, dur_override or 3.5, text, "cheer")
        if re.match(r"^(?:argue|argues|gesticulate|gesticulates|rant|rants)\b", low):
            kind, tgt = self.target(re.sub(r"^\w+\s*(?:with|at)?\s*", "", low), a)
            return self._add(a, t0, dur_override or 4.0, text, "argue", face=tgt if kind == "char" else None)
        if re.match(r"^(?:wait|waits|idle|idles|hold|holds|stay|stays|stand still|pause|pauses)\b", low):
            return self._add(a, t0, dur_override or 2.0, text, self.idle_clip(a))
        m = re.match(r"^(?:start|starts|stand|stands|place|placed|positioned)\s+(?:at|on|in)\s+(.+)$", low)
        if m:
            p = parse_point(m.group(1), True)
            ch = next(c for c in self.chars if c.id == a.id)
            if p:
                ch.x, ch.z = p
                return t0
        # fall back to retrieval over the whole library for this actor's kind
        matches = self.lib.search(text, k=1, kind=a.kind)
        if matches and matches[0].score >= 2.0:
            clip = matches[0].clip
            d = 3.0 if clip.is_locomotion else max(1.5, min(clip.duration, 4.0))
            return self._add(a, t0, dur_override or d, text, clip.id)
        self.unmatched.append(f"{a.name}: {text}")
        if a.kind == "human" and a.state in ("stand", "carry"):  # couldn't parse: reset to the bind pose, visibly
            return self._add(a, t0, 1.5, f"reset to A-pose (couldn't parse: {text[:40]})", "apose")
        return t0

    def _is_horse(self, cid) -> bool:
        return any(x is not None and x.id == cid and x.kind == "horse" for x in self.actors.values())

    def _move_to(self, kind, tgt, mods, idx, n, default_dist) -> MoveTo | None:
        if kind == "point":
            spread = 0.0
            if n > 1:  # a group sent to one point forms a small cluster around it
                ang = 2 * math.pi * idx / n
                r = 0.7 + 0.25 * n
                return MoveTo(point=[tgt[0] + r * math.sin(ang), tgt[1] + r * math.cos(ang)], distance=0.2)
            return MoveTo(point=list(tgt), distance=max(spread, 0.2))
        if kind in ("char", "object"):
            return MoveTo(target=tgt, distance=_seconds(mods.get("distance")) or default_dist, slot=idx, of=n)
        return None

    def _attack(self, a: _Actor, text, low, m, mods, flags, t0, idx) -> float:
        verb = low.split()[0]
        target_text = text[m.start(1):]
        target_text = re.split(r"\s+with\s+", target_text, flags=re.I)[0]
        kind, tgt = self.target(target_text, a)
        wtxt = (mods.get("weapon") or "").lower()
        mw = re.search(r"\bwith\s+(?:a|an|the|his|her|their)?\s*(\w+)", low)
        if not wtxt and mw:
            wtxt = mw.group(1)
        weapon = WEAPON_WORDS.get(wtxt) if wtxt else None
        if weapon is None and verb in ("shoot", "shoots", "fire", "fires") and a.weapon in RANGED:
            weapon = a.weapon  # an archer shoots their bow, a rifleman their rifle
        if weapon is None:
            weapon = {"kick": "kick", "kicks": "kick", "roundhouse": "kick", "side": "kick", "punch": "fists",
                      "punches": "fists", "karate": "fists", "slash": "sword", "slashes": "sword", "stab": "sword",
                      "stabs": "sword", "smash": "mace", "smashes": "mace", "shoot": "gun", "shoots": "gun",
                      "fire": "gun", "fires": "gun"}.get(verb) or a.weapon or "fists"
        if verb in ("push", "pushes", "shove", "shoves"):
            clip, weapon = "push", "fists"
        elif a.kind == "dog":
            clip = "dog_attack"
        elif low.startswith("roundhouse"):
            clip = "roundhouse"
        elif low.startswith("side kick"):
            clip = "side_kick"
        elif low.startswith("karate"):
            clip = "karate_punch"
        else:
            clip = ATTACK_CLIPS.get(weapon, "punch")
        takedown = verb in ("takedown", "take", "takes") or "takedown" in low
        finisher = verb in ("finish", "finishes", "finisher")
        if finisher and weapon in ("fists", "kick"):
            clip, weapon = "mace_smash", a.weapon or "mace"
        if takedown and clip == "punch":
            clip = "punch_kick"
        cl = self.lib.clips[clip]
        dur = _seconds(mods.get("duration")) or (max(2.0, min(cl.duration, 3.2)) if clip not in ("shoot_gun", "shoot_rifle", "shoot_bow") else min(cl.duration, 4.0))
        kw = {}
        if weapon in ("gun", "rifle", "sword", "mace", "bow"):
            kw["weapon"] = weapon
            a.weapon = weapon
        end = self._add(a, t0, dur, text, clip, face=tgt if kind == "char" else None, **kw)
        if kind != "char":
            return end
        victim = next((x for x in self.actors.values() if x is not None and x.id == tgt), None)
        if victim is None or victim.kind == "horse":
            return end
        result = (mods.get("result") or "").lower() or next((f for f in flags if f in REACTIONS), "")
        if not result:
            result = "fall" if (takedown or finisher) else "stagger" if clip == "push" else "hit"
        if finisher:
            result = "none" if victim.state in ("lying", "prone") else "fall"
        react = REACTIONS.get(result, "hit_react")
        if victim.kind == "dog":
            react = "dog_idle" if react else None
        impact = {"shoot_bow": 2.6, "shoot_gun": 1.0, "shoot_rifle": 1.2}.get(clip, min(1.0, dur * 0.42))
        if react:
            rt = max(t0 + impact, victim.end)
            if result in ("fall", "fall_back", "fall_face") and not takedown and react.startswith("fall"):
                victim.state, victim.down = "lying", react
                end2 = self._add(victim, rt, 3.0, f"falls from {a.name}'s blow", react, face=a.id)
            elif takedown:
                rt2 = self._add(victim, rt, 1.0, f"is hit by {a.name}", "hit_react", face=a.id)
                victim.state, victim.down = "lying", "fall_back"
                end2 = self._add(victim, rt2, 3.0, "is taken down", "fall_back", face=a.id)
            else:
                end2 = self._add(victim, rt, self.lib.clips[react].duration if react != "dog_idle" else 1.5,
                                 f"reacts to {a.name}", react, face=a.id)
            return max(end, end2)
        return end


def plan_commands(text: str, lib: MotionLibrary) -> ScenePlan:
    return CommandPlanner(lib).plan(text)
