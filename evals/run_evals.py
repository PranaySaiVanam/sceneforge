"""SceneForge evaluation suite.

  python evals/run_evals.py                      # rule-based planner
  python evals/run_evals.py --provider anthropic # LLM planner (needs ANTHROPIC_API_KEY)

Two benchmarks:
  1. Motion retrieval: 66 action phrases -> acceptable clip ids (top-1, top-3, MRR)
  2. Scene direction: 20 scene prompts with expected cast, actions and camera notes.
     Scored on plan validity, cast accuracy, action recall, camera-note recall,
     contact accuracy for interactions, shot coverage and actor collisions.
Writes evals/results/<provider>.json and evals/results/<provider>.md
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sceneforge.director import REACH, _transform, direct  # noqa: E402
from sceneforge.library import get_library  # noqa: E402
from sceneforge.schema import ResolvedBeat  # noqa: E402

HERE = Path(__file__).resolve().parent


def eval_retrieval(lib):
    rows = list(csv.DictReader((HERE / "retrieval_cases.tsv").open(), delimiter="\t"))
    top1 = top3 = mrr = 0.0
    misses = []
    for r in rows:
        ok = set(r["acceptable"].split(","))
        got = [m.clip.id for m in lib.search(r["query"], k=5)]
        rank = next((i + 1 for i, g in enumerate(got) if g in ok), None)
        top1 += rank == 1
        top3 += rank is not None and rank <= 3
        mrr += 1 / rank if rank else 0
        if rank != 1:
            misses.append({"query": r["query"], "expected": sorted(ok), "got": got[:3]})
    n = len(rows)
    return {"cases": n, "top1": round(top1 / n, 3), "top3": round(top3 / n, 3), "mrr": round(mrr / n, 3),
            "misses": misses}


def root_at(lib, beats, cid, t):
    mine = [b for b in beats if b.character == cid and b.start <= t + 1e-6]
    if not mine:
        return None
    b = max(mine, key=lambda b: b.start)
    return _transform(lib, b, min(t - b.start, b.duration))[:2]


def eval_scene(case, provider, lib):
    t0 = time.perf_counter()
    try:
        out = direct(case["prompt"], provider=provider)
    except Exception as e:  # a crash is a failed case, not a failed run
        return {"id": case["id"], "valid": False, "error": repr(e)}
    ms = (time.perf_counter() - t0) * 1000
    sc = out["scene"]
    beats = [ResolvedBeat(**b) for b in sc["beats"]]
    clips = {b.clip for b in beats if not b.auto}
    groups = case["actions"]
    hit = [g for g in groups if clips & set(g)]
    shot_terms = {s["type"] for s in sc["shots"]} | {s["move"] for s in sc["shots"]}
    cam_hit = [s for s in case["shots"] if s in shot_terms]
    # coverage: shots contiguous from 0 to duration
    shots = sorted(sc["shots"], key=lambda s: s["start"])
    gaps = sum(abs(a["start"] + a["duration"] - b["start"]) > 0.05 for a, b in zip(shots, shots[1:]))
    covered = bool(shots) and shots[0]["start"] < 0.05 and abs(shots[-1]["start"] + shots[-1]["duration"] - sc["duration"]) < 0.05 and gaps == 0
    # contact: interactive beats begin within reach of their partner
    contacts = []
    for b in beats:
        if b.clip in REACH and b.face:
            me, other = root_at(lib, beats, b.character, b.start + 0.05), root_at(lib, beats, b.face, b.start + 0.05)
            if me and other:
                contacts.append(math.dist(me, other) <= REACH[b.clip] + 0.45)
    # collisions: sampled every 0.1 s, roots closer than 0.35 m
    ids = [c["id"] for c in sc["characters"]]
    samples = coll = 0
    t = 0.0
    while t < sc["duration"] and len(ids) > 1:
        pos = [root_at(lib, beats, i, t) for i in ids]
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                if pos[i] and pos[j]:
                    samples += 1
                    coll += math.dist(pos[i], pos[j]) < 0.35
        t += 0.1
    fixes = sum(len(s.get("fixes", [])) for s in out["trace"])
    res = {
        "id": case["id"], "valid": True, "ms": round(ms, 1),
        "characters_ok": len(sc["characters"]) == case["characters"],
        "action_recall": round(len(hit) / len(groups), 3) if groups else 1.0,
        "missed_actions": [g for g in groups if g not in hit],
        "camera_recall": round(len(cam_hit) / len(case["shots"]), 3) if case["shots"] else None,
        "contact_rate": round(sum(contacts) / len(contacts), 3) if contacts else None,
        "covered": covered, "collision_rate": round(coll / samples, 4) if samples else 0.0,
        "time_of_day_ok": (sc["setting"]["time_of_day"] == case["time_of_day"]) if "time_of_day" in case else None,
        "duration": sc["duration"], "shots": len(sc["shots"]), "auto_fixes": fixes,
        "provider": out["provider"],
    }
    d = next(s for s in out["trace"] if s["agent"] == "Director")
    if d.get("tokens"):
        res["tokens"] = d["tokens"]
    return res


def mean(xs):
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 3) if xs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="offline")
    a = ap.parse_args()
    lib = get_library()
    retrieval = eval_retrieval(lib)
    cases = [json.loads(l) for l in (HERE / "scenes.jsonl").read_text().splitlines() if l.strip()]
    scenes = [eval_scene(c, a.provider, lib) for c in cases]
    ok = [s for s in scenes if s["valid"]]
    lat = sorted(s["ms"] for s in ok)
    summary = {
        "provider": a.provider,
        "retrieval": {k: retrieval[k] for k in ("cases", "top1", "top3", "mrr")},
        "scenes": {
            "cases": len(scenes),
            "valid_plan_rate": round(len(ok) / len(scenes), 3),
            "cast_accuracy": mean([float(s["characters_ok"]) for s in ok]),
            "action_recall": mean([s["action_recall"] for s in ok]),
            "camera_note_recall": mean([s["camera_recall"] for s in ok]),
            "contact_rate": mean([s["contact_rate"] for s in ok]),
            "shot_coverage": mean([float(s["covered"]) for s in ok]),
            "collision_rate": mean([s["collision_rate"] for s in ok]),
            "time_of_day_accuracy": mean([float(s["time_of_day_ok"]) for s in ok if s["time_of_day_ok"] is not None]),
            "latency_ms_p50": round(statistics.median(lat), 1) if lat else None,
            "latency_ms_p95": round(lat[min(len(lat) - 1, int(0.95 * len(lat)))], 1) if lat else None,
            "auto_fixes_per_scene": mean([s["auto_fixes"] for s in ok]),
        },
    }
    out_dir = HERE / "results"
    out_dir.mkdir(exist_ok=True)
    (out_dir / f"{a.provider}.json").write_text(json.dumps({"summary": summary, "retrieval": retrieval,
                                                            "scenes": scenes}, indent=1))
    md = [f"# SceneForge eval results ({a.provider})", "",
          "| Metric | Value |", "|---|---|"]
    for k, v in summary["retrieval"].items():
        md.append(f"| retrieval {k} | {v} |")
    for k, v in summary["scenes"].items():
        md.append(f"| scene {k} | {v} |")
    md += ["", "## Per scene", "", "| id | cast | actions | camera | contact | coverage | collisions | fixes | ms |",
           "|---|---|---|---|---|---|---|---|---|"]
    for s in scenes:
        if not s["valid"]:
            md.append(f"| {s['id']} | crashed: {s['error']} |||||||| ")
            continue
        md.append(f"| {s['id']} | {'ok' if s['characters_ok'] else 'wrong'} | {s['action_recall']} | "
                  f"{s['camera_recall'] if s['camera_recall'] is not None else '-'} | "
                  f"{s['contact_rate'] if s['contact_rate'] is not None else '-'} | {'ok' if s['covered'] else 'gap'} | "
                  f"{s['collision_rate']} | {s['auto_fixes']} | {s['ms']} |")
    md += ["", "## Retrieval misses (top-1)", ""]
    md += [f"- \"{m['query']}\": expected {', '.join(m['expected'])}; got {', '.join(m['got'])}" for m in retrieval["misses"]] or ["None."]
    (out_dir / f"{a.provider}.md").write_text("\n".join(md) + "\n")
    print(json.dumps(summary, indent=1))
    for s in scenes:
        if s["valid"] and (s["action_recall"] < 1 or not s["characters_ok"] or s["camera_recall"] not in (None, 1.0)):
            print("  weak:", s["id"], "cast_ok=", s["characters_ok"], "missed=", s["missed_actions"], "cam=", s["camera_recall"])
        elif not s["valid"]:
            print("  CRASH:", s["id"], s["error"])


if __name__ == "__main__":
    main()
