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
from sceneforge.commands import is_command_script, singular  # noqa: E402
from sceneforge.stage import cell_center, cell_name, parse_point  # noqa: E402

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


POLISH = {"turn_l90", "turn_r90", "turn_l135", "turn_r135", "turn_180", "push_off", "brake"}


def _beats(text, polish=False):
    r = direct(text)
    return r["scene"], [b for b in r["scene"]["beats"] if not b["auto"] and (polish or b["clip"] not in POLISH)]


class CommandTests(unittest.TestCase):
    def test_detects_command_scripts(self):
        self.assertTrue(is_command_script("Cast: Ravi\nRavi: Move to Grid 4B [Speed=Fast]"))
        self.assertFalse(is_command_script(FIGHT))

    def test_singular(self):
        self.assertEqual(singular("Zombies").lower(), "zombie")
        self.assertEqual(singular("guards"), "guard")

    def test_grid(self):
        self.assertEqual(parse_point("Grid 4B"), parse_point("B4"))
        x, z = cell_center("A", 1)
        self.assertEqual(parse_point(cell_name(x, z)), (x, z))
        self.assertEqual(parse_point("[3, -2]"), (3.0, -2.0))

    def test_move_modifiers_pick_style_and_speed(self):
        _, beats = _beats("Cast: Ravi\nRavi: Move to Grid 4B [Speed=Fast, Style=Angry]")
        self.assertIn(beats[0]["clip"], {"run", "walk_angry", "run_fast"})
        _, slow = _beats("Cast: Ravi\nRavi: Move to Grid 4B [Style=Sneaky]")
        self.assertNotEqual(slow[0]["clip"], "walk")

    def test_ten_actors_and_groups(self):
        scene, _ = _beats("Cast: 10 Zombies\nZombies: Approach [0, 0] [Style=Zombie]")
        self.assertEqual(len(scene["characters"]), 10)
        pos = {(round(b["x"], 1), round(b["z"], 1)) for b in scene["beats"] if b["start"] == 0}
        self.assertGreater(len(pos), 5)  # spread out, not stacked

    def test_bow_archer_shoots_bow(self):
        _, beats = _beats("Cast: Kira [Weapon=Bow], Zed\nKira: Shoot Zed")
        self.assertIn("shoot_bow", [b["clip"] for b in beats])

    def test_strike_gets_reaction(self):
        _, beats = _beats("Cast: Ravi, Bruno\nRavi: Attack Bruno [Result=Fall]")
        clips = {b["character"]: b["clip"] for b in beats}
        self.assertIn("fall_back", [b["clip"] for b in beats if b["character"] == "B"])

    def test_animals_and_riding(self):
        scene, beats = _beats("Cast: Ravi, Storm [Kind=Horse], Bruno [Kind=Dog]\nRavi: Mount Storm\nStorm: Move to Grid 10J [Speed=Fast]\nBruno: Move to Grid 10J [Speed=Fast]")
        kinds = {c["id"]: c["kind"] for c in scene["characters"]}
        self.assertEqual(kinds["B"], "horse")
        self.assertTrue(any(b.get("ride") == "B" for b in beats))
        self.assertTrue(all(b["clip"].startswith("dog_") for b in beats if b["character"] == "C"))

    def test_van_loading_uses_object(self):
        scene, beats = _beats("Cast: Ravi\nSet: Van at Grid 10K\nRavi: Load luggage into van")
        self.assertEqual(scene["objects"][0]["type"], "van")
        self.assertIn("set_down_heavy", [b["clip"] for b in beats])

    def test_lines_run_in_order(self):
        _, beats = _beats("Cast: Ravi, Bruno\nRavi: Wave\nBruno: Nod")
        a = next(b for b in beats if b["character"] == "A")
        b = next(b for b in beats if b["character"] == "B" and b["clip"] == "nod")
        self.assertGreaterEqual(b["start"], a["start"] + a["duration"] - 0.01)


class PolishTests(unittest.TestCase):
    """Transitions, layers, obstacles and ground states (command scripts only)."""

    def test_pivot_before_reversing(self):
        _, beats = _beats("Cast: Ravi [At=Grid 10J, Facing=North]\nRavi: Move to Grid 4J", polish=True)
        self.assertIn(beats[0]["clip"], {"turn_180", "turn_l135", "turn_r135"})

    def test_run_pushes_off_and_brakes(self):
        _, beats = _beats("Cast: Ravi [At=Grid 10C, Facing=East]\nRavi: Run to Grid 10M", polish=True)
        clips = [b["clip"] for b in beats]
        self.assertEqual(clips[clips.index("run") - 1], "push_off")
        self.assertEqual(clips[clips.index("run") + 1], "brake")

    def test_ease_keeps_end_point(self):
        from sceneforge.director import ease_warp
        self.assertAlmostEqual(ease_warp(3.0, 3.0, 0.35, 0.3), 3.0, places=6)
        self.assertLess(ease_warp(0.2, 3.0, 0.35, 0.3), 0.2)

    def test_strafe_keeps_eyes_on_target(self):
        scene, beats = _beats("Cast: Agent, Guard\nAgent: Strafe left facing Guard [Distance=3]")
        self.assertEqual(beats[0]["clip"], "walk_side")
        self.assertTrue(any(l["type"] == "look" and l["target"] == "B" for l in scene["layers"]))

    def test_look_at_is_a_layer_not_a_beat(self):
        scene, beats = _beats("Cast: Ravi, Kira\nRavi: Walk to Grid 3C\nKira: Look at Ravi [With=prev, For=4s]")
        look = [l for l in scene["layers"] if l["type"] == "look"][0]
        self.assertEqual((look["character"], look["target"], look["duration"]), ("B", "A", 4.0))
        self.assertFalse(any(b["character"] == "B" and b["start"] > 0 for b in beats))

    def test_obstacle_traversal(self):
        scene, beats = _beats("Cast: Ravi [At=Grid 10C]\nSet: Curb at Grid 10G\nRavi: Step over curb")
        self.assertEqual(beats[-1]["clip"], "step_over")
        curb = scene["objects"][0]
        a = next(b for b in scene["beats"] if b["clip"] == "step_over")
        # starts on the actor's side of the curb, about 0.9 m out
        self.assertAlmostEqual(math.hypot(a["x"] - curb["x"], a["z"] - curb["z"]), 0.9, delta=0.25)

    def test_wall_collision_bumps(self):
        scene, beats = _beats("Cast: Ravi [At=Grid 10C]\nSet: Wall at Grid 10H [Length=8, Facing=East]\nRavi: Walk to Grid 10M")
        self.assertIn("bump", [b["clip"] for b in beats])
        walk = next(b for b in beats if b["clip"] == "walk")
        self.assertLess(walk["x"] + 0.01, scene["objects"][0]["x"])

    def test_ground_idle_after_fall(self):
        scene, _ = _beats("Cast: Ravi, Bruno\nRavi: Attack Bruno [Result=Fall]\nRavi: Wave\nRavi: Wave")
        clips = [b["clip"] for b in scene["beats"] if b["character"] == "B"]
        self.assertTrue(any(c.startswith("down_") for c in clips))
        self.assertNotEqual(clips[-1], "idle_wait")

    def test_grip_lasts_until_put_down(self):
        scene, _ = _beats("Cast: Meera\nMeera: Hold cup\nMeera: Walk to Grid 5C\nMeera: Put down cup\nMeera: Wave")
        grip = [l for l in scene["layers"] if l["type"] == "grip"][0]
        put = next(b for b in scene["beats"] if b["action"] == "Put down cup")
        self.assertAlmostEqual(grip["start"] + grip["duration"], put["start"], delta=0.05)


class CrowdAndStuntTests(unittest.TestCase):
    """Update 3: crowds, ambient life, moods, stunts, vehicles, utilities."""

    def test_prose_crowds_get_ten_varied_actors(self):
        from collections import Counter
        for prompt in ("10 people fighting", "ten people dancing"):
            scene = direct(prompt)["scene"]
            self.assertEqual(len(scene["characters"]), 10)
            clips = Counter(b["clip"] for b in scene["beats"] if not b["auto"] and b["clip"] not in ("walk", "run"))
            self.assertGreater(len(clips), 1, prompt)

    def test_prose_singing(self):
        scene = direct("8 people singing")["scene"]
        self.assertEqual(len(scene["characters"]), 8)
        self.assertTrue(all(b["clip"] == "sing" for b in scene["beats"] if not b["auto"]))

    def test_spawned_crowd_is_background(self):
        scene, beats = _beats("Cast: Ravi\nSpawn Crowd [Count=6, Path=Grid 2A to Grid 2T]\nRavi: Wave")
        extras = [c for c in scene["characters"] if c["extra"]]
        self.assertEqual(len(extras), 6)
        wave = next(b for b in beats if b["clip"] == "wave")
        self.assertLess(wave["start"], 1.0)  # the crowd never holds up the screenplay
        self.assertTrue(any(l["type"] == "look" and l["character"] == extras[0]["id"] for l in scene["layers"]))

    def test_chatting_pair(self):
        _, beats = _beats("Cast: Ravi, Kira\nRavi: Chat with Kira [For=8s]")
        clips = {b["character"]: b["clip"] for b in beats}
        self.assertEqual((clips["A"], clips["B"]), ("chat_a", "chat_b"))

    def test_mood_overlay_and_pace(self):
        scene, beats = _beats("Cast: Ravi [At=Grid 10C, Facing=East]\nRavi: Mood Sad\nRavi: Walk to Grid 10M")
        self.assertEqual(scene["layers"][0]["mood"], "sad")
        walk = next(b for b in beats if b["clip"] == "walk")
        self.assertLess(walk["speed"], 1.0)

    def test_explosion_reactions_by_distance(self):
        scene, beats = _beats("Cast: Near [At=Grid 10J], Mid [At=Grid 10L], Far [At=Grid 10T]\nExplosion at Grid 10I")
        got = {b["character"]: b["clip"] for b in beats if "explosion" in b["action"]}
        self.assertEqual(got, {"A": "blast_back", "B": "flinch_shield", "C": "startle"})
        self.assertTrue(any(b["clip"] == "down_groan_blast_back" for b in scene["beats"]))

    def test_vehicle_seats_and_bail_out(self):
        scene, beats = _beats("Cast: Ravi, Kira, Jeep [Kind=Car]\nRavi: Get in Jeep\nKira: Get in Jeep\n"
                              "Jeep: Drive to Grid 4P [Speed=Fast]\nRavi: Jump out")
        rides = {b["character"]: (b["clip"], b["seat"]) for b in beats if b["ride"]}
        self.assertEqual(rides["A"], ("drive_steer", 0))
        self.assertEqual(rides["B"], ("passenger_idle", 1))
        self.assertIn("car_fast", [b["clip"] for b in beats])
        self.assertIn("eject_roll", [b["clip"] for b in beats])

    def test_teleport_jumps(self):
        _, beats = _beats("Cast: Ravi [At=Grid 1A]\nRavi: Teleport to Grid 10F")
        tp = next(b for b in beats if b["teleport"])
        self.assertEqual((tp["x"], tp["z"]), (-9.0, -1.0))

    def test_shuffle_before_sitting_and_wind_down_after_running(self):
        scene, beats = _beats("Cast: Ravi [At=Grid 10A, Facing=East]\nRavi: Walk to Grid 10E\nRavi: Sit", polish=True)
        clips = [b["clip"] for b in beats]
        self.assertEqual(clips[clips.index("sit_down") - 1], "settle")
        scene, _ = _beats("Cast: Ravi [At=Grid 10A, Facing=East], Kira\nRavi: Run to Grid 10Q\nKira: Wave\nKira: Wave\nKira: Wave")
        self.assertIn("wind_down", [b["clip"] for b in scene["beats"] if b["character"] == "A"])

    def test_director_utilities_and_stunts(self):
        for line, clip in (("T-Pose", "tpose"), ("Stand-in", "standin_turn"), ("Scale reference", "scale_ref"),
                           ("Eye-line calibration", "eyeline_sweep"), ("Stagger [From=Left]", "stagger_right"),
                           ("Leap of faith", "leap_roll"), ("High fall", "high_fall"), ("Think", "think"),
                           ("Exhausted", "exhausted"), ("Check phone", "idle_phone"), ("Type on laptop", "type_laptop")):
            _, beats = _beats(f"Cast: Ravi\nRavi: {line}")
            self.assertEqual(beats[-1]["clip"], clip, line)

    def test_unparsed_command_resets_to_a_pose(self):
        scene, beats = _beats("Cast: Ravi\nRavi: Juggle flaming torches")
        self.assertEqual(beats[-1]["clip"], "apose")
        self.assertTrue(scene["unmatched"])


class StudioImportTests(unittest.TestCase):
    """tools/studio_import.py: studio BVH (MotionBuilder names, centimetres, Z-up, 24 fps) -> SceneForge clip."""

    FIXTURE = Path(__file__).resolve().parent / "fixtures" / "studio_sample.bvh"

    def setUp(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
        import studio_import
        self.si = studio_import

    def test_name_tokens_cover_common_rigs(self):
        side, has = self.si.side_of, self.si.has
        for name, s in (("Character1_LeftHand", "Left"), ("mixamorig:RightForeArm", "Right"), ("Bip01 L Hand", "Left"),
                        ("hand_r", "Right"), ("LHand", "Left"), ("Skeleton_RightFoot", "Right")):
            self.assertEqual(side(name), s, name)
        self.assertTrue(has("Bip01 L Hand", "hand"))
        self.assertFalse(has("Character1_LeftHandIndex1", "hand") and not has("Character1_LeftHandIndex1", "index"))

    def test_mapping_units_axis_and_rate(self):
        rep = {}
        pos = self.si.load(self.FIXTURE, report=rep)
        self.assertEqual((rep["up"], rep["scale_to_m"]), ("z", 0.01))
        self.assertEqual(rep["mapping"]["Spine1"], "Character1_Spine2")  # the chest = where the arms meet
        self.assertEqual(rep["mapping"]["Neck1"], "Character1_Neck1")    # the head's parent
        self.assertEqual(pos.shape[1:], (20, 3))
        self.assertAlmostEqual(len(pos) / 30.0, 240 / 24.0, delta=0.1)  # 24 fps resampled to 30
        self.assertTrue(1.5 < rep["height_m"] < 2.0)
        self.assertEqual(self.si.check(pos), [])

    def test_builds_without_the_cmu_database(self):
        import shutil
        import subprocess
        import tempfile
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / "studio").mkdir()
            shutil.copy(self.FIXTURE, tmp / "studio" / "take.bvh")
            cat = tmp / "catalog.tsv"
            cat.write_text("id\tcmu\tcategory\ttags\tdescription\ttrim\textra\n"
                           "studio_jab\tstudio:take.bvh\tcombat\tstudio jab\tstudio take\t\tsource=studio tier=2\n")
            subprocess.run([sys.executable, str(root / "tools" / "build_motion_library.py"), "--catalog", str(cat),
                            "--out", str(tmp), "--studio-root", str(tmp / "studio")], check=True, capture_output=True)
            lib = json.loads((tmp / "motion_library.json").read_text())
            clip = lib["clips"][0]
            self.assertEqual((clip["id"], clip["source"]), ("studio_jab", "studio"))
            self.assertTrue(1.6 < clip["height"] < 2.0)
            self.assertTrue((tmp / "motions" / "studio_jab.json").exists())
