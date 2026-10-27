"""Rule-based scene planner: no API key needed.

Used as (1) the fallback when an LLM call fails and (2) the baseline the LLM
planner is compared against in evals/. Handles English scene descriptions:
character extraction (counts, roles, names, pronouns, "one ... the other"),
clause splitting with sequential vs simultaneous timing, interactions
(face / approach), adverbs, camera directions and setting.
"""
from __future__ import annotations

import re

from .library import Match, MotionLibrary, get_library
from .schema import Beat, Character, MoveTo, ScenePlan, Setting, Shot

ROLES = {
    "man": "men", "woman": "women", "boy": "boys", "girl": "girls", "kid": "kids", "child": "children",
    "person": "people", "fighter": "fighters", "boxer": "boxers", "soldier": "soldiers", "guard": "guards",
    "thief": "thieves", "detective": "detectives", "dancer": "dancers", "zombie": "zombies", "robot": "robots",
    "ninja": "ninjas", "knight": "knights", "warrior": "warriors", "hero": "heroes", "villain": "villains",
    "cop": "cops", "officer": "officers", "actor": "actors", "friend": "friends", "stranger": "strangers",
    "waiter": "waiters", "janitor": "janitors", "athlete": "athletes", "player": "players",
    "monster": "monsters", "spy": "spies", "agent": "agents", "king": "kings", "princess": "princesses",
    "student": "students", "teacher": "teachers", "doctor": "doctors", "survivor": "survivors",
    "gymnast": "gymnasts", "worker": "workers", "brother": "brothers", "sister": "sisters",
    "rival": "rivals", "partner": "partners", "duelist": "duelists", "samurai": "samurai", "pirate": "pirates",
    "assassin": "assassins", "champion": "champions", "challenger": "challengers", "old man": "old men",
    "coach": "coaches", "trainer": "trainers", "instructor": "instructors", "recruit": "recruits",
    "cadet": "cadets", "sergeant": "sergeants", "son": "sons", "daughter": "daughters", "wife": "wives", "husband": "husbands", "boss": "bosses",
    "chef": "chefs", "cook": "cooks", "customer": "customers", "diner": "diners", "guest": "guests",
    "grandmother": "grandmothers", "grandfather": "grandfathers", "mom": "moms", "dad": "dads", "baker": "bakers",
    "mother": "mothers", "father": "fathers", "intruder": "intruders", "kid": "kids", "teenager": "teenagers",
}
PLURAL_TO_ROLE = {v: k for k, v in ROLES.items()}
COUNTS = {"two": 2, "three": 3, "four": 4, "2": 2, "3": 3, "4": 4, "a pair of": 2, "a couple of": 2, "both": 2,
          "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9,
          "10": 10, "a dozen": 10, "dozen": 10, "many": 8, "several": 6, "a group of": 5, "a crowd of": 10,
          "a bunch of": 6, "lots of": 8}
MAX_CAST = 10  # the stage holds ten actors
NOT_NAMES = set("""The A An Then When While After Before As Meanwhile Finally Suddenly Both They He She It One Two Three
Four Five Six Seven Eight Nine Ten Dozen Several Many Lots Group Crowd Bunch Camera Close Wide Medium Slow Fast Cut Low High Over Tracking Track Pan Push Pull Orbit Crane Shot Scene Night Day
Dusk Dawn Sunset In On At With His Her Their Later Eventually Next Now Quickly Slowly Someone Everyone Nobody Angle
POV Extreme Establishing Handheld Static Dolly Zoom Aerial Drone Lens Frame Shaky Hindi Telugu English I We You Our My Fade End Start Open Opening Inside Outside Rain Sun""".split())
CAMERA_PATTERNS = [
    (r"\bover[- ]the[- ]shoulder\b|\bots\b", "over_shoulder", None),
    (r"\bextreme close[- ]?up\b|\bclose[- ]?up\b|\bcloseup\b|\bclose on\b", "close_up", None),
    (r"\btwo[- ]shot\b", "two_shot", None),
    (r"\bestablishing\b|\bwide\b|\bfull shot\b|\blong shot\b", "wide", None),
    (r"\bmedium\b|\bmid[- ]shot\b", "medium", None),
    (r"\blow[- ]angle\b|\bfrom below\b|\bworm'?s[- ]eye\b", "low_angle", None),
    (r"\bhigh[- ]angle\b|\bfrom above\b|\bbird'?s[- ]eye\b|\boverhead\b|\btop[- ]down\b", "high_angle", None),
    (r"\btracking\b|\bfollow(s|ing)? (him|her|them|the)\b|\bcamera follows\b", "tracking", None),
    (r"\bpov\b|\bpoint of view\b|\bfirst[- ]person\b", "pov", None),
    (r"\bpush[- ]?in\b|\bdolly in\b|\bzoom in\b|\bmove(s)? in\b|\bcreep(s)? in\b", None, "push_in"),
    (r"\bpull[- ]?(out|back)\b|\bdolly out\b|\bzoom out\b", None, "pull_out"),
    (r"\borbit(s|ing)?\b|\b360\b|\bcircl(e|es|ing) around them\b|\barc shot\b|\bspin(s)? around\b", None, "orbit"),
    (r"\bcrane\b|\bjib\b|\brise(s)? up\b|\bboom up\b", None, "crane_up"),
    (r"\bhandheld\b|\bshaky\b", None, "handheld"),
]
CAMERA_WORDS = re.compile(r"\b(camera|shot|close[- ]?up|closeup|angle|pov|dolly|zoom|orbit|crane|tracking|push[- ]?in|"
                          r"pull[- ]?(out|back)|cut to|frame|framing|lens|establishing|handheld|two[- ]shot)\b", re.I)
LENS = re.compile(r"\b(\d{2,3})\s?mm\b", re.I)
SIMULTANEOUS = re.compile(r"^(while|as|meanwhile|at the same time|simultaneously)\b", re.I)
APPROACH = re.compile(r"\b(to|toward|towards|up to|over to|approach(es|ing)?|at|chases?|runs? after|confronts?)\b", re.I)
LOCATIONS = ["warehouse", "street", "rooftop", "park", "stage", "alley", "forest", "arena", "ring", "office", "temple",
             "beach", "desert", "hallway", "corridor", "classroom", "studio", "parking lot", "bar", "kitchen",
             "dojo", "battlefield", "market", "restaurant", "dining room", "cafe", "bakery", "home", "station", "garden", "courtyard", "palace", "village", "city"]
MOODS = {"tense": "tense", "suspense": "tense", "eerie": "eerie", "creepy": "eerie", "happy": "joyful",
         "joyful": "joyful", "sad": "melancholy", "melancholy": "melancholy", "epic": "epic", "funny": "comic",
         "comic": "comic", "romantic": "romantic", "dramatic": "dramatic", "angry": "tense", "scary": "eerie"}
MIN_MATCH = 1.2


def _split_clauses(text: str) -> list[tuple[str, bool]]:
    """Returns (clause, simultaneous_with_previous)."""
    text = re.sub(r"\s+", " ", text.strip())
    sentences = re.split(r"(?<=[.!?;:])\s+|\n+", text)
    out: list[tuple[str, bool]] = []
    for sent in sentences:
        sent = sent.strip(" .!?;:")
        if not sent:
            continue
        parts = re.split(r",?\s+(?=(?:and then|then|while|as|before|after|until|meanwhile|but)\b)|,\s+(?=and\b)|,\s+", sent)
        for p in parts:
            p = p.strip(" ,")
            if not p:
                continue
            simul = bool(SIMULTANEOUS.match(p))
            p = re.sub(r"^(and then|then|and|while|as|meanwhile|but|before|after|until|at the same time|simultaneously)\s+",
                       "", p, flags=re.I)
            out.append((p, simul))
    return out


class _Cast:
    def __init__(self):
        self.chars: list[Character] = []
        self.roles: dict[str, list[str]] = {}  # role -> [ids]
        self.groups: list[list[str]] = []

    def add(self, name: str, role: str | None = None) -> str:
        cid = "ABCDEFGHIJ"[len(self.chars)]
        self.chars.append(Character(id=cid, name=name))
        if role:
            self.roles.setdefault(role, []).append(cid)
        return cid

    def by_name(self, name: str) -> str | None:
        for c in self.chars:
            if c.name.lower() == name.lower():
                return c.id
        return None


def _extract_cast(text: str) -> _Cast:
    cast = _Cast()
    low = text.lower()
    # 1) Counted groups: "two fighters", "a pair of dancers"
    role_alt = "|".join(sorted(map(re.escape, PLURAL_TO_ROLE), key=len, reverse=True))
    count_alt = "|".join(sorted(map(re.escape, (k for k in COUNTS if k != "both")), key=len, reverse=True))
    for m in re.finditer(rf"\b({count_alt}|\d{{1,3}})\s+(?:\w+\s+)?({role_alt})\b", low):
        n, role = COUNTS.get(m.group(1)) or int(m.group(1)), PLURAL_TO_ROLE[m.group(2)]
        if role in cast.roles or len(cast.chars) >= MAX_CAST or n < 1:
            continue
        ids = [cast.add(f"{role.title()} {i + 1}", role) for i in range(min(n, MAX_CAST - len(cast.chars)))]
        cast.groups.append(ids)
    # 2) Proper names (capitalised, not a stop word, not a known role word)
    for m in re.finditer(r"\b([A-Z][a-z]{2,})\b", text):
        w = m.group(1)
        if w in NOT_NAMES or w.lower() in ROLES or w.lower() in PLURAL_TO_ROLE or w.lower() in LOCATIONS:
            continue
        if not cast.by_name(w) and len(cast.chars) < MAX_CAST:
            cast.add(w)
    # 3) Singular roles: "a tired soldier", "the detective", "another thief"
    sing_alt = "|".join(sorted(map(re.escape, ROLES), key=len, reverse=True))
    starts = [(m.start(), "the", m.group(1)) for m in re.finditer(rf"(?:^|[.!?;]\s+)({sing_alt})\b", low)]
    for pos, art, role in starts:  # "Mom kneads dough." -> a character with no article
        if role not in cast.roles and len(cast.chars) < MAX_CAST:
            cast.add(role.title(), role)
    for m in re.finditer(rf"\b(a|an|the|another|one|second|her|his|their|my)\s+(?:[a-z]+\s+){{0,2}}?({sing_alt})\b", low):
        art, role = m.group(1), m.group(2)
        existing = cast.roles.get(role, [])
        if len(cast.chars) >= MAX_CAST:
            break
        bound = cast.roles.get("_bound", [])
        unbound = [i for g in cast.groups for i in g if i not in bound]
        if not existing and art == "the" and unbound:
            # "two boxers ... the champion ... the challenger": name members of the group
            gid = unbound[0]
            c = next(c for c in cast.chars if c.id == gid)
            c.name = role.title()
            cast.roles[role] = [gid]
            cast.roles.setdefault("_bound", []).append(gid)
        elif not existing:
            cast.add(role.title(), role)
        elif art in ("another", "second"):
            cast.add(f"{role.title()} {len(existing) + 1}", role)
    if not cast.chars:
        cast.add("Performer")
    # Rename lone role characters: "Fighter 1" only when there are several
    cast.roles.pop("_bound", None)
    for role, ids in cast.roles.items():
        if len(ids) == 1 and next(c for c in cast.chars if c.id == ids[0]).name[-1].isdigit():
            c = next(c for c in cast.chars if c.id == ids[0])
            c.name = role.title()
    return cast


def _refs_in(clause: str, cast: _Cast, last: str | None) -> list[tuple[int, list[str], str]]:
    """All character references in a clause as (position, [ids], kind).

    kind: "name" | "role" | "pair" | "subj_pron" | "obj_pron" | "group" | "special"
    """
    low = " " + clause.lower() + " "
    ids_all = [c.id for c in cast.chars]
    refs: list[tuple[int, list[str], str]] = []
    for c in cast.chars:
        for m in re.finditer(rf"\b{re.escape(c.name.lower())}\b", low):
            refs.append((m.start(), [c.id], "name"))
    for role, ids in cast.roles.items():
        plural = ROLES[role]
        for m in re.finditer(rf"\b(the|both|all|two|three|four|five|six|seven|eight|nine|ten|dozen|several|many|"
                             rf"group of|crowd of|bunch of|lots of|\d{{1,3}})\s+(?:\w+\s+)?{re.escape(plural)}\b", low):
            refs.append((m.start(), ids, "group"))
        for m in re.finditer(rf"(?:\b(?:the|a|an|her|his|their|my)\s+(?:[a-z]+\s+){{0,2}}?|^\s*){re.escape(role)}\b(?!s)", low):
            refs.append((m.start(), [ids[0]] if len(ids) == 1 else [last if last in ids else ids[0]], "role"))
    pairs = [g for g in cast.groups if len(g) >= 2] or ([ids_all[:2]] if len(ids_all) >= 2 else [])
    if pairs:
        g = pairs[0]
        for m in re.finditer(r"\b(the first|first one|one of them|the one|one)\b(?! another)", low):
            refs.append((m.start(), [g[0]], "pair"))
        for m in re.finditer(r"\b(the other(?: one)?|the second)\b", low):
            refs.append((m.start(), [g[1] if last != g[1] else g[0]], "pair"))
    for m in re.finditer(r"\b(he|she)\b", low):
        refs.append((m.start(), [last] if last else ids_all[:1], "subj_pron"))
    for m in re.finditer(r"\b(him|her)\b", low):
        refs.append((m.start(), [last] if last else ids_all[:1], "obj_pron"))
    for m in re.finditer(r"\b(they|both|everyone|together|all of them)\b", low):
        refs.append((m.start(), ids_all, "group"))
    for m in re.finditer(r"\b(winner|victor|loser|defeated one)\b", low):
        refs.append((m.start(), ["@" + m.group(1)], "special"))
    refs.sort(key=lambda r: r[0])
    seen, out = set(), []
    for pos, ids, kind in refs:
        if any(abs(pos - p) < 3 for p in seen):
            continue
        seen.add(pos)
        out.append((pos, ids, kind))
    return out


def _camera_directive(clause: str):
    low = clause.lower()
    shot_type, move = None, None
    for pat, st, mv in CAMERA_PATTERNS:
        if re.search(pat, low):
            shot_type = shot_type or st
            move = move or mv
    lens = LENS.search(low)
    if shot_type is None and move is None and not CAMERA_WORDS.search(low):
        return None
    return {"type": shot_type, "move": move, "lens_mm": float(lens.group(1)) if lens else None}


def _strip_camera(clause: str) -> str:
    """Remove the camera part so 'close-up on Ravi as he laughs' still yields an action."""
    c = re.sub(r"(?i)\b(slow|slowly|quick|fast|gentle|dramatic|extreme|dramatically)\b\s+(?=\w*\s*(push|pull|dolly|zoom|orbit|crane|close|wide|tracking|pan))", "", clause)
    c = re.sub(r"(?i)\b(camera\s+\w+|wide shot|wide|establishing shot|establishing|close[- ]?up|closeup|close on|two[- ]shot|"
               r"medium shot|medium|low[- ]angle( shot)?|high[- ]angle( shot)?|over[- ]the[- ]shoulder( shot)?|tracking shot|"
               r"pov( shot)?|push[- ]?in|pull[- ]?(out|back)|dolly (in|out)|zoom (in|out)|orbit(s|ing)?( around)?|crane( up| shot)?|"
               r"around (them|him|her|the \w+)|handheld|cut to|shot|\d{2,3}\s?mm( lens)?|lens|frames?|on|of)\b", " ", c)
    return re.sub(r"\s+", " ", c).strip(" ,:-")


def _without_cast(text: str, cast: _Cast) -> str:
    words = [re.escape(c.name.lower()) for c in cast.chars] + [re.escape(r) for r in cast.roles] + \
            [re.escape(ROLES[r]) for r in cast.roles]
    if not words:
        return text
    return re.sub(rf"\b({'|'.join(sorted(words, key=len, reverse=True))})\b", " ", text.lower())


REACTIONS = {"duck", "fall_face", "fall_back", "backflip"}
INTERACTIVE = {"combat", "social", "emotion"}


def _sub_clauses(text: str, cast: _Cast) -> list[tuple[str, bool]]:
    """Clauses, further split on 'and' when a new subject follows ('... and they shake hands')."""
    names = "|".join(re.escape(c.name) for c in cast.chars)
    subj = rf"(?:they|he|she|both|everyone|the|one|{names})" if names else r"(?:they|he|she|both|everyone|the|one)"
    out = []
    for clause, simul in _split_clauses(text):
        parts = re.split(rf"\s+and\s+(?={subj}\b)", clause, flags=re.I)
        for i, p in enumerate(parts):
            out.append((p, simul if i == 0 else False))
    return out


CROWD_POOLS = {  # extra clips mixed into a crowd doing the same thing
    "combat": ["punch", "kick", "boxing", "punch_kick", "karate_punch", "front_kick", "roundhouse", "spin_kick"],
    "dance": ["salsa", "chicken_dance", "dance", "dance_2", "breakdance", "twist"],
}
VARIED = {"combat", "dance"}  # a crowd doing these gets a mix; anything else (singing, waving) stays in unison


def plan_offline(prompt: str, lib: MotionLibrary | None = None) -> ScenePlan:
    lib = lib or get_library()
    cast = _extract_cast(prompt)
    low = prompt.lower()
    two = len(cast.chars) == 2
    setting = Setting(
        location=next((l for l in LOCATIONS if re.search(rf"\b{l}\b", low)), "mocap stage"),
        time_of_day="night" if re.search(r"\b(night|midnight|dark|moonlit)\b", low)
        else "dusk" if re.search(r"\b(dusk|sunset|evening|twilight|dawn|sunrise)\b", low) else "day",
        mood=next((v for k, v in MOODS.items() if re.search(rf"\b{k}\b", low)), "dramatic"),
    )
    beats: list[Beat] = []
    shots_raw: list[dict] = []
    last: str | None = cast.chars[0].id
    cursor = 0.0          # when the next sequential clause starts
    prev_start = 0.0      # start of the previous clause (for "while"/"as")
    prev_beat: Beat | None = None
    char_end: dict[str, float] = {}
    unmatched: list[str] = []

    def other_than(ids: list[str]) -> str | None:
        for c in reversed(beats):
            if c.character not in ids:
                return c.character
        return next((c.id for c in cast.chars if c.id not in ids), None)

    for clause, simul in _sub_clauses(prompt, cast):
        cam = _camera_directive(clause)
        action_text = _strip_camera(clause) if cam else clause
        refs = _refs_in(action_text, cast, last)
        subj_ids = refs[0][1] if refs and refs[0][0] < 12 else [last]
        if not refs:
            subj_ids = [last]
        if subj_ids[0].startswith("@"):
            subj_ids = [last]
        # object = first later reference to someone else; object pronouns never refer to the subject
        target = None
        for pos, ids, kind in refs[1:]:
            if ids[0].startswith("@"):
                continue
            if kind == "obj_pron" and ids[0] in subj_ids:
                ids = [other_than(subj_ids)]
            if len(ids) == 1 and ids[0] and ids[0] not in subj_ids:
                target = ids[0]
                break
        # compound verbs with the same subject: "circles and lunges" -> two beats
        actions = []
        for p in [p for p in re.split(r"\band\b", action_text) if p.strip()]:
            m = lib.search(_without_cast(p, cast), k=1)
            if m and m[0].score >= (2.0 if cam else MIN_MATCH):
                actions.append((p, m[0]))
            elif actions:
                actions[-1] = (actions[-1][0] + " and " + p, actions[-1][1])
        if cam:
            subjects = [i for r in refs for i in r[1] if not i.startswith("@")]
            if len(subjects) == 1:
                last = subjects[0]
            anchor = cursor
            if not actions:
                prev_start = cursor  # a following "as ..." clause starts with this shot
            if shots_raw and shots_raw[-1]["anchor"] == anchor and (shots_raw[-1]["type"] is None or cam["type"] is None):
                prev = shots_raw[-1]  # "Wide shot, orbit around them" -> one shot
                prev["type"] = prev["type"] or cam["type"]
                prev["move"] = prev["move"] or cam["move"]
                prev["subjects"] = prev["subjects"] or subjects
                prev["note"] += ", " + clause.strip()
            else:
                shots_raw.append({**cam, "anchor": anchor, "subjects": subjects,
                                  "special": [i for r in refs for i in r[1] if i.startswith("@")], "note": clause.strip()})
        if not actions:
            words = _without_cast(action_text, cast)
            if not cam and re.search(r"[a-z]{3,}", words) and not re.fullmatch(r"\W*(and|then|now|finally)?\W*", words):
                unmatched.append(clause.strip())
            continue
        first_clip = actions[0][1].clip
        if simul:
            start = prev_start
        elif prev_beat and first_clip.id in REACTIONS and prev_beat.character not in subj_ids:
            start = prev_beat.start + min(0.8, prev_beat.duration * 0.4)  # reaction overlaps the attack
        else:
            start = cursor
        clause_end = start
        crowd = len(subj_ids) >= 5  # groups of up to four keep their original (lockstep) behaviour
        if crowd and actions[0][1].clip.category not in VARIED and not actions[0][1].clip.is_locomotion:
            crowd = False
        variety = []
        if crowd:  # a crowd never moves in lockstep: vary the clip within the matched category
            top = actions[0][1]
            pool = [m for m in lib.search(_without_cast(actions[0][0], cast), k=8)
                    if m.clip.category == top.clip.category
                    and m.score >= top.score * (0.92 if top.clip.is_locomotion else 0.45)
                    and not re.search(r"side|back", m.clip.id)]
            seen = {m.clip.id for m in pool}
            for cid_ in CROWD_POOLS.get(top.clip.category, []):
                if cid_ in lib.clips and cid_ not in seen and lib.clips[cid_].tier == 1:
                    pool.append(Match(clip=lib.clips[cid_], score=top.score, confidence=top.confidence))
            variety = pool or [top]
        for si, subj in enumerate(subj_ids):
            t = start if simul else max(start, min(char_end.get(subj, 0.0), start + 0.01) if first_clip.id in REACTIONS else max(start, char_end.get(subj, 0.0)))
            tgt = target if target != subj else None
            if tgt is None and two and (len(subj_ids) > 1 or re.search(r"\b(each other|face|faces|facing|stare)\b", action_text.lower())):
                tgt = next(c.id for c in cast.chars if c.id != subj)
            if crowd:
                t += 0.15 * (si % 4)
            for ai, (text, match) in enumerate(actions):
                if crowd and ai == 0:
                    match = variety[si % len(variety)]
                    if match.clip.category == "combat":  # pair the crowd off: 1 v 2, 3 v 4, ...
                        partner = subj_ids[si ^ 1] if (si ^ 1) < len(subj_ids) else subj_ids[0]
                        tgt = partner if partner != subj else None
                clip = match.clip
                tl = text.lower()
                speed = 0.8 if re.search(r"\b(slow|slowly|carefully|cautiously|wearily)\b", tl) else \
                    1.2 if re.search(r"\b(quick|quickly|fast|suddenly|rapidly|furiously)\b", tl) else 1.0
                if clip.is_locomotion:
                    dur = 3.0
                elif clip.category == "dance":
                    dur = 6.0
                elif clip.category == "exercise":
                    dur = clip.duration if clip.source == "keyframed" else max(3.0, min(clip.duration / speed, 5.0))
                elif clip.category == "daily":
                    dur = max(3.0, min(clip.duration / speed, 5.0))
                elif clip.category == "combat":
                    dur = 2.5
                elif clip.id in ("fall_face", "fall_back"):
                    dur = min(clip.duration, 3.0)
                else:
                    dur = max(1.5, min(clip.duration / speed, 3.5))
                move_to = None
                if clip.is_locomotion and tgt and APPROACH.search(tl) and "away" not in tl \
                        and clip.id not in ("walk_back", "pace", "walk_side"):
                    move_to = MoveTo(target=tgt, distance=1.1 if clip.id != "run" else 1.4)
                interactive = clip.category in INTERACTIVE or clip.id in REACTIONS or clip.id in ("walk_back", "walk_side")
                face = tgt if (tgt and (interactive or move_to)) else None
                if face is None and two and interactive:
                    face = next(c.id for c in cast.chars if c.id != subj)
                prev_same = next((x for x in reversed(beats) if x.character == subj), None)
                if prev_same and prev_same.clip in ("fall_face", "fall_back") and clip.id == "pushups" \
                        and "drop" in prev_same.action.lower() and abs(prev_same.start + prev_same.duration - t) < 0.05:
                    # "drops and does pushups": the pushup clip already gets down to the floor
                    beats.remove(prev_same)
                    t = prev_same.start
                    text = prev_same.action + " and " + text
                if prev_same and prev_same.clip in ("sit", "sit_stool") and clip.category == "daily" \
                        and abs(prev_same.start + prev_same.duration - t) < 0.05:
                    # the eating/cooking captures are performed standing at a table: skip the sit
                    beats.remove(prev_same)
                    t = prev_same.start
                    text = prev_same.action + " and " + text
                if prev_same and prev_same.clip == clip.id and abs(prev_same.start + prev_same.duration - t) < 0.05:
                    prev_same.action += " and " + text.strip()  # "trips and falls" -> one fall
                    continue
                b = Beat(character=subj, start=round(t, 2), duration=round(dur, 2), action=text.strip(),
                         clip=clip.id, face=face, move_to=move_to, speed=speed)
                beats.append(b)
                prev_beat = b
                t += dur
            char_end[subj] = t
            clause_end = max(clause_end, t)
        if len(subj_ids) == 1:
            last = subj_ids[0]
        prev_start = start
        cursor = max(cursor, clause_end)

    if not beats:  # nothing recognised: an idle beat so the viewer still has a scene
        beats.append(Beat(character=cast.chars[0].id, start=0, duration=4, action="stands and waits", clip="idle_wait"))

    fallen = {b.character for b in beats if b.clip in ("fall_face", "fall_back")}
    shots: list[Shot] = []
    for s in shots_raw:
        if not any(b.start >= s["anchor"] - 0.01 for b in beats):
            s["anchor"] = beats[-1].start if beats else 0.0  # closing camera note -> covers the last beat
        subjects = list(dict.fromkeys(s["subjects"]))
        for w in s.get("special", []):
            if w in ("@winner", "@victor"):
                subjects += [c.id for c in cast.chars if c.id not in fallen][:1]
            else:
                subjects += sorted(fallen)[:1]
        mv = s["move"] or "static"
        st = s["type"] or ("wide" if mv in ("pull_out", "orbit", "crane_up") else "medium")
        shots.append(Shot(start=round(s["anchor"], 2), duration=2.0, type=st, subjects=subjects, move=mv,
                          lens_mm=s["lens_mm"] or {"close_up": 50, "wide": 28, "over_shoulder": 40}.get(st, 35),
                          note=s["note"][:80]))
    shots.sort(key=lambda s: s.start)

    first = re.split(r"[.!?;:,]", prompt.strip())[0]
    title = " ".join(first.split()[:7]).strip()
    title = title[:1].upper() + title[1:]
    return ScenePlan(title=title[:56] or "Untitled scene", logline=prompt.strip()[:160],
                     setting=setting, characters=cast.chars, beats=beats, shots=shots, unmatched=unmatched)
