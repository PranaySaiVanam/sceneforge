"""Run: python -m unittest discover tests   (or: pytest)"""
import json
import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sceneforge import api  # noqa: E402
from sceneforge.director import REACH, _transform, direct  # noqa: E402
from sceneforge.library import get_library, stem  # noqa: E402
from sceneforge.llm import extract_json, plan_with_llm  # noqa: E402
from sceneforge.planner_offline import plan_offline  # noqa: E402
from sceneforge.schema import ResolvedBeat  # noqa: E402

FIGHT = "Two fighters circle each other. One lunges with a punch, the other dodges and falls. Slow push-in on the winner."


class LibraryTests(unittest.TestCase):
    def setUp(self):
        self.lib = get_library()

    def test_catalog_loaded(self):
        self.assertGreaterEqual(len(self.lib.clips), 50)
        self.assertEqual(self.lib.fps, 30)

    def test_speeds_are_human(self):
        walk = self.lib.clips["walk"]
        run = self.lib.clips["run"]
        self.assertTrue(0.8 < walk.avg_speed < 1.9, walk.avg_speed)   # fps correction for subject 143
        self.assertTrue(2.5 < run.avg_speed < 6.0, run.avg_speed)

    def test_retrieval(self):
        self.assertEqual(self.lib.search("throws a punch")[0].clip.id, "punch")
        self.assertEqual(self.lib.search("sneaks past the guard")[0].clip.id, "sneak")
        self.assertEqual(self.lib.search("anything", hint="cartwheel")[0].clip.id, "cartwheel")

    def test_stem(self):
        self.assertEqual(stem("pirouettes"), stem("pirouette"))
        self.assertEqual(stem("running"), "run")

    def test_loop_sampling_accumulates(self):
        clip = self.lib.clips["walk"]
        x1, z1, _ = self.lib.sample_root("walk", clip.duration)
        x2, z2, _ = self.lib.sample_root("walk", clip.duration * 2)
        self.assertGreater(math.hypot(x2, z2), math.hypot(x1, z1) * 1.8)


class PlannerTests(unittest.TestCase):
    def test_fight_cast_and_timing(self):
        p = plan_offline(FIGHT)
        self.assertEqual([c.name for c in p.characters], ["Fighter 1", "Fighter 2"])
        punch = next(b for b in p.beats if b.clip == "punch")
        duck = next(b for b in p.beats if b.clip == "duck")
        self.assertEqual(punch.character, "A")
        self.assertEqual(duck.character, "B")
        self.assertTrue(punch.start < duck.start < punch.start + punch.duration)  # reaction overlaps

    def test_pronoun_object(self):
        p = plan_offline("Ravi sneaks in while a guard paces. The guard runs toward him.")
        run = next(b for b in p.beats if b.clip == "run")
        self.assertEqual((run.character, run.move_to.target), ("B", "A"))

    def test_group_roles(self):
        p = plan_offline("Two boxers spar. The champion throws a punch and the challenger ducks.")
        self.assertEqual([c.name for c in p.characters], ["Champion", "Challenger"])

    def test_kitchen_scene(self):
        p = plan_offline("Mom kneads dough for chapati while her son sets the table, then eats soup.")
        self.assertEqual([c.name for c in p.characters], ["Mom", "Son"])
        self.assertEqual([b.clip for b in p.beats], ["knead", "set_glass", "eat_soup"])
        self.assertEqual(p.beats[1].start, 0)  # "while" -> simultaneous

    def test_sit_before_standing_meal_is_dropped(self):
        p = plan_offline("Priya sits down and eats dinner from a plate.")
        self.assertEqual([b.clip for b in p.beats], ["eat_dinner"])

    def test_pushups_are_keyframed_and_merge_the_drop(self):
        p = plan_offline("The recruit drops and does pushups.")
        self.assertEqual([b.clip for b in p.beats], ["pushups"])
        self.assertEqual(get_library().clips["pushups"].source, "keyframed")

    def test_unknown_action_is_reported_not_faked(self):
        sc = direct("Ravi rides a bicycle, then waves.")["scene"]
        self.assertEqual(sc["unmatched"], ["Ravi rides a bicycle"])
        self.assertEqual([b["clip"] for b in sc["beats"] if not b["auto"]], ["wave"])

    def test_camera_notes(self):
        p = plan_offline(FIGHT)
        self.assertEqual(p.shots[0].move, "push_in")


class PipelineTests(unittest.TestCase):
    def test_reach_fix_lands_punch(self):
        lib = get_library()
        out = direct(FIGHT, provider="offline")
        beats = [ResolvedBeat(**b) for b in out["scene"]["beats"]]
        punch = next(b for b in beats if b.clip == "punch")
        other = next(b for b in beats if b.character == "B" and b.start <= punch.start < b.start + b.duration)
        me = _transform(lib, punch, 0)[:2]
        them = _transform(lib, other, punch.start - other.start)[:2]
        self.assertLess(math.dist(me, them), REACH["punch"] + 0.4)
        blocking = next(s for s in out["trace"] if s["agent"] == "Blocking")
        self.assertTrue(any("inserted" in f for f in blocking["fixes"]))

    def test_shots_cover_scene(self):
        sc = direct(FIGHT, provider="offline")["scene"]
        shots = sorted(sc["shots"], key=lambda s: s["start"])
        self.assertAlmostEqual(shots[0]["start"], 0, places=2)
        for a, b in zip(shots, shots[1:]):
            self.assertAlmostEqual(a["start"] + a["duration"], b["start"], places=1)
        self.assertAlmostEqual(shots[-1]["start"] + shots[-1]["duration"], sc["duration"], places=1)

    def test_no_character_overlaps_itself(self):
        sc = direct("A janitor sweeps the hallway, stops, stretches and yawns, then drinks from a bottle.")["scene"]
        beats = sorted(sc["beats"], key=lambda b: b["start"])
        for a, b in zip(beats, beats[1:]):
            self.assertLessEqual(a["start"] + a["duration"], b["start"] + 0.06)


class LLMTests(unittest.TestCase):
    def test_extract_json_from_fenced_prose(self):
        self.assertEqual(extract_json('Sure!\n```json\n{"a": {"b": 1}}\n```'), {"a": {"b": 1}})

    def test_repair_loop(self):
        good = {"title": "t", "characters": [{"id": "A", "name": "Hero"}],
                "beats": [{"character": "A", "start": 0, "duration": 2, "action": "waves", "clip": "wave"}],
                "shots": [{"start": 0, "duration": 2, "type": "medium"}]}
        bad = dict(good, beats=[{"character": "A", "start": -1, "duration": 2, "action": "waves"}])
        replies = iter([json.dumps(bad), json.dumps(good)])
        seen = []

        def fake(provider, model, system, messages):
            seen.append(messages[-1]["content"])
            return next(replies), 100, 50

        res = plan_with_llm("hero waves", get_library(), "anthropic", "test-model", caller=fake)
        self.assertIsNotNone(res.plan)
        self.assertEqual(res.attempts, 2)
        self.assertIn("failed validation", seen[1])
        self.assertEqual(res.input_tokens, 200)

    def test_network_failure_falls_back(self):
        def boom(*a):
            raise RuntimeError("HTTP 401")
        res = plan_with_llm("x", get_library(), "anthropic", "m", caller=boom)
        self.assertIsNone(res.plan)
        self.assertEqual(res.attempts, 1)


class APITests(unittest.TestCase):
    def test_validation(self):
        self.assertEqual(api.direct_scene({"prompt": ""})[0], 400)
        self.assertEqual(api.direct_scene({"prompt": "x" * 5000})[0], 400)

    def test_motion_whitelist(self):
        self.assertIsNone(api.motion_path("../../app"))
        self.assertIsNotNone(api.motion_path("walk"))


if __name__ == "__main__":
    unittest.main()
