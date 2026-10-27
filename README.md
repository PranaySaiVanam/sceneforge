# SceneForge AI
**Live demo:** https://sceneforge-q5ib.onrender.com

**Describe a scene in plain words. Get a blocked, shot-listed 3D previs, performed by real motion-capture data.**

> "Two fighters circle each other. One lunges with a punch, the other dodges and falls. Slow push-in on the winner."

SceneForge turns that sentence into a playable animated scene in about 20 ms: two actors performing real CMU mocap clips, staged so the punch actually lands, cut together by a virtual cinematographer, with a shot list you can export.

![SceneForge directing a fight scene](docs/screenshot.png)

Previsualization ("previs") is how film, game and animation teams plan action before an expensive shoot. Today it means an artist hand-blocking proxy characters for days. SceneForge is a multi-agent system that does a first pass from a paragraph of text, so directors can iterate on staging and coverage in seconds.

---

## What it does

- **Natural-language direction.** Type or dictate (English, Hindi or Telugu via the Web Speech API) who is in the scene, what they do, and optionally how to shoot it.
- **Real motion, not generated motion.** Every action is grounded in one of 73 curated clips (72 real CMU captures plus a keyframed pushup, clearly labelled) from the CMU Graphics Lab Motion Capture Database, with root motion preserved, so walks travel and falls land on the floor.
- **Everyday life, not just action.** Eating (dinner, soup, a sandwich), cooking (chopping, slicing, mixing, kneading, making tea or coffee), setting the table, phone calls, typing and more, with tables, counters, plates and bowls placed automatically under the actor's hands.
- **Exercise.** Jumping jacks, squats, weight lifting, stretching and pushups. CMU has no floor pushups, so `tools/synth_motion.py` keyframes one on the CMU skeleton's own bone lengths (two-bone arm IK, planted hands, straight plank).
- **Honest about gaps.** An action no clip can perform is reported ("Not in the motion library yet: …") instead of being replaced by something wrong.
- **Physical staging.** Characters are blocked on a 10 x 10 m capture volume. Approaches stop at the right distance, reactions overlap the attack that caused them, and fallen characters stay down.
- **A virtual cinematographer.** 9 shot types (wide, medium, close-up, two-shot, over-the-shoulder, low/high angle, tracking, POV), 6 camera moves, real lens focal lengths, the 180-degree rule and occlusion-aware framing.
- **Explainable agents.** The UI shows each agent's decisions, retrieval confidence and every automatic fix.
- **Exports.** Scene plan (JSON), shot list (CSV) and a rendered video of the director's cut (WebM).

## Architecture

```mermaid
flowchart LR
    P[Scene description] --> D[Director agent<br/>Claude / Gemini / GPT<br/>or rule-based fallback]
    D -->|ScenePlan JSON<br/>validated + repaired| R[Motion Retriever<br/>BM25 + phrase + id signals<br/>over 73 motion clips]
    R --> C[Continuity agent<br/>references, overlaps,<br/>idle and hold fill]
    C --> B[Blocking agent<br/>root-motion integration,<br/>reach checks, approaches]
    B --> K[Cinematographer<br/>coverage, 180-degree rule,<br/>notes follow their beats]
    K --> S[Resolved scene + trace]
    S --> V[Three.js viewer<br/>playback, camera, timeline,<br/>video export]
```

| Agent | Responsibility | Example decision from the trace |
|---|---|---|
| Director | Prompt to structured `ScenePlan` (pydantic schema). LLM output is validated; validation errors are sent back for up to 2 repair attempts; any failure falls back to the rule-based planner. | `anthropic/claude-sonnet-5-5 planned 5 beats, 3 shots in 1 attempt` |
| Motion Retriever | Grounds each beat in a real clip. Verifies LLM-chosen clip ids and reports confidence and alternatives. | `the other dodges -> duck (0.98)` |
| Continuity | Fixes unknown character references, removes self-overlaps, fills gaps with idles, holds the final pose after falls and sits. | `idle/hold fill for 2 gaps` |
| Blocking | Integrates each clip's root motion to place actors. Detects interactions that are out of reach and inserts an approach, shifting the exchange (and the partner's reaction) later. | `A was 8.2 m from B for 'punch' (reach 1.1 m): inserted 1.8 s run approach` |
| Cinematographer | Turns director notes into shots anchored to beats (so they move when blocking retimes), and designs coverage where there are no notes. | `designed 3 shots of coverage for 0-9.1 s` |

The Python sampler (`library.sample_root`) and the JavaScript sampler (`frontend/motion.js`) implement the same looping root-motion maths, so what the agents plan is exactly what renders.

## Command scripts (director syntax)

Besides plain English, the director accepts a line-by-line script in `[Verb] + [Modifier]` form. One `Move` verb covers walking, running, sneaking and stalking through modifiers instead of separate commands.

```text
Cast: Ravi [Weapon=Sword], Kira [Tail=Yes, Weapon=Bow, At=Grid 9H], Storm [Kind=Horse], Bruno [Kind=Dog], 5 Zombies [At=Grid 14J]
Set: Van at Grid 12K [Facing=North]
Camera: Wide shot, slow push in [At=0s]
Zombies: Approach Ravi [Style=Zombie]
Kira: Shoot Zombie 1
Ravi: Attack Zombie 2 [Weapon=Sword, Result=Fall]
Ravi: Move to Grid 4B [Speed=Fast, Style=Angry]
Arjun: Mount Storm
Storm: Move to Grid 5C [Speed=Fast]
Bruno: Move to Grid 6C [Speed=Fast, With=prev]
```

- **Stage:** a 40 m × 40 m grid of 2 m cells, columns A–T and rows 1–20 (`Grid 4B` or `B4`), or a point in metres such as `[3, -2]`.
- **Cast:** up to 10 actors. Optional fields are `Kind=Human|Dog|Horse`, `Tail=Yes`, `Weapon=…`, `Scale=…` and `At=…`. Groups like `5 Zombies` are addressed as `Zombies` or `Zombie 3`.
- **Move:** `Move to`, `Approach`, `Retreat from` (backs away facing the target), `Flee from`, `Turn to`/`Face` and `Pace between A and B`, with `Speed=Slow|Jog|Fast|Sprint` and `Style=Angry|Sneaky|Crouched|Injured|Scared|Zombie|March|Proud|Old`.
- **Fight:** `Attack`, `Shoot`, `Takedown`, `Finish`, `Roundhouse`, `Side kick`, `Block`, `Dodge`, `React`, `Equip`/`Holster`, with `Weapon=Fists|Kick|Sword|Mace|Gun|Rifle|Bow`. `Result=Hit|Gut|Block|Dodge|Fall` schedules the target's reaction at the moment of impact.
- **Body:** `Sit`, `Stand`, `Crouch`, `Kneel`, `Prone`, `Lie down`, `Mount`/`Dismount`.
- **Dialogue:** `Say "…"` (subtitled), `Argue with`, `Listen`, `Nod`, `Cower`, `Cheer`, `Wave`, `Point at`, `Salute`, `Clap`.
- **Interact:** `Open door`, `Push button`, `Climb ladder`, `Pick up`, `Lift weights`, `Lift luggage`, `Carry … to`, `Load luggage into van`.
- **Timing:** lines run in screenplay order. `[With=prev]` runs a line alongside the previous one, and `[At=5s]` pins it to a time.

**Natural movement (command scripts).**

- **Start and stop:** runs push off from a standstill (`push_off`) and plant the feet to stop (`brake`). Walks ease in and out through a time warp that keeps the exact end point, so the feet never slide.
- **Pivots:** a direction change of more than 60° inserts a foot-shuffle turn first (`turn_l90`, `turn_r90`, `turn_l135`, `turn_r135`, `turn_180`).
- **Strafe:** a sidestep left or right keeps the chest on the target, and a look layer locks the eyes.

**Props and obstacles.**

- **Grip poses:** equipped pistols and rifles take a low-ready or port-arms grip while walking or idling, and `Hold cup` and `Hold suitcase` set one-hand grip poses through hand IK.
- **Traversal:** `Step over curb`, `Squeeze through gap` and `Duck under beam` walk to the obstacle's start mark and cross it with a matching motion. An obstacle with no `Facing` turns across the actor's path.
- **Collisions:** a path into a wall, a van, a crate or the stage edge stops short with a `bump`.

**Camera-aware layers.** `Look at Kira | Camera | Grid 1A [For=4s]` turns the head (and up to about 35° of chest) toward the target while the body carries on. `Startle [From=...]` snaps the head round with a flinch. When the director's camera has a wall in front of the action, the wall turns see-through for that shot.

**Defeat and clean-up.** `Faint` or `Collapse` plays a non-combat fall. Anyone left on the ground breathes (`down_still_*`) or groans (`down_groan_*`). Each ground idle is generated from the exact last frame of the fall that came before it, so there is no pop. `Crawl` moves a prone actor forward on the belly.

**Crowds and background life.**

- **Plain English:** crowds work up to 10 actors ("10 people fighting", "ten people dancing", "8 people singing"). A fighting or dancing crowd gets a mix of clips, and fighters pair off.
- **Spawned extras:** `Spawn Crowd [Count=6, Path=Grid 2A to Grid 2T]` adds extras that wander the path with casual walks, turns, pauses and random head turns. With `[Activity=Chat|Idle|Phone|Read|Sweep|Wipe|Type|Dance]` they stay at a spot instead. Extras run in the background: they never hold up the screenplay, they never get camera coverage, and they stop when the main action ends.
- **Ambient loops:**
  - Wallflower idles: phone, watch, arms crossed, shifting weight.
  - Two-person chat loops that swap talking and nodding.
  - Prop loops with their props: wiping a counter, sweeping, typing at a desk, reading.
  - `Sing`.

**Moods.** `Mood Sad | Angry | Anxious | Stealthy` lasts until it is changed, and `[Mood=...]` applies to one line. Each mood is an additive posture on top of walks, runs and idles, and also changes the pace:

- **Sad:** drooping head and shoulders.
- **Angry:** a forward lean, arms out and heavy steps.
- **Anxious:** head darts, tense shoulders and hands near the chest.
- **Stealthy:** a lowered centre of gravity and a scanning head.

**Pacing beats.** `Think [Style=Pace]`, `Exhausted`, `Wipe sweat`, `Flinch`, `Faint` and `Wake up` (a groggy recovery). Seated, kneeling and prone poses settle the feet first. After a sprint or a fight, the breathing winds down over about 8 seconds.

**Stunts.**

- **Staggers:** `Push Bruno` and `Stagger [From=Left]` stagger in four directions.
- **Leap and high fall:** `Leap of faith` is a dive and roll; `High fall [Land=Grid 8H]` flails, lands hard and rolls.
- **Explosions:** `Explosion at Grid 10M` throws nearby actors back off their feet, makes those at middle distance shield their faces, and startles everyone further away. It also adds a fireball, smoke, a scorch mark and camera shake.

**Vehicles.** `Cast: Jeep [Kind=Car], Moto [Kind=Bike]`, then `Get in Jeep [Seat=Passenger]` and `Drive to Grid 8P`.

- The driver steers, bikes and riders lean into turns, and passengers react to bumps and look around.
- `Jump out` bails out and rolls; `Get out` steps out normally.

**Director utilities.**

- **Calibration:** `T-Pose` and `A-Pose`. A line that can't be parsed resets the actor to an A-pose and is reported.
- **Teleport:** `Teleport to Grid 10F` dissolves the actor out and back in.
- **Stand-ins:** `Stand-in` turns a grey low-poly stand-in on a turntable; there is also a Stand-ins switch for the whole cast.
- **Scale:** `Scale reference next to Van` shows a measuring pole with the actor's height.
- **Eye-line:** `Eye-line calibration` pans the head mechanically and draws a sight-line that lights up when it crosses the lens.
- **Gaze and snapping:** every actor wears an eye visor that shows their gaze, feet are kept above the floor, and hands and feet snap onto ladder rungs.

New motions follow the same rule as the rest of the library. Mocap clips come from CMU, and anything synthesised (the dog and horse gaits, hit reactions, nods, button presses and door opening) is labelled `keyframed` in the agent trace. The new clips sit in a second retrieval tier, so they only win a plain-English query when they clearly match better than the original clips. All original scenes and retrieval results are unchanged.

## Bring in your studio's mocap

Studio captures go straight into the library. No CMU download is needed, because clips that were already built are reused.

```bash
# 1. Check a take: joint mapping, units, up axis, frame rate, performer height
python tools/studio_import.py inspect path/to/Take_012.bvh

# 2. Add it: copies it to data/studio/, writes the catalog row, rebuilds the library
python tools/studio_import.py add path/to/Take_012.bvh --id sword_overhead --category combat \
    --tags "sword slash, overhead strike, two-handed sword" --desc "overhead two-handed sword slash" \
    --trim "1.0:3.6 abs" --extra "weapon=sword strike=1"

# 3. Check nothing else moved, then push
python -m unittest discover tests
```

The importer handles the four ways studio exports differ from the CMU files:

- **Joint names.** It maps Vicon, Motive, MotionBuilder/HumanIK, Mixamo, Xsens, Rokoko and Blender names (for example `Character1_Spine2`, `mixamorig:LeftForeArm`, `Bip01 L Hand`, `hand_l`) onto the 20-joint skeleton. The mapping follows the hierarchy: the chest is where the two arms meet, the neck is the head's parent, and the forearm is the hand's parent. If a guess is wrong, pass a joint map with `--map` (see `docs/studio_joint_map.example.json`).
- **Units.** Centimetres, millimetres, metres and inches are detected from the performer's standing height. You can force them with `--units`.
- **Up axis.** Z-up exports are rotated to Y-up. You can force it with `--up`.
- **Frame rate.** Any rate is resampled to 30 fps.

Export your takes as **BVH**, retargeted to a standard humanoid. Convert FBX in MotionBuilder or Blender first.

A studio clip starts in retrieval tier 2, so it never changes how existing prompts resolve. Add `tier=1` to `--extra` if your captures should win plain-English matches outright.

Studio clips are labelled "studio mocap" in the agent trace and on the timeline. Once added, a clip can be used in a script by its id (`Ravi: Sword overhead`) or by its tags in plain English.

## Evaluation

`python evals/run_evals.py` runs two benchmarks. Results for the rule-based planner (no API key):

| Benchmark | Metric | Result |
|---|---|---|
| Motion retrieval (89 action phrases) | Top-1 / Top-3 / MRR | **0.99 / 1.00 / 0.99** |
| Scene direction (26 scenes) | Valid plan rate | **1.00** |
| | Cast accuracy | **1.00** |
| | Action recall | **0.98** |
| | Camera-note recall | **1.00** |
| | Contact accuracy (interactions start within reach) | **1.00** |
| | Shot coverage (contiguous, no gaps) | **1.00** |
| | Actor collision rate | 0.012 |
| | Latency p50 | 5.0 ms |

Run `python evals/run_evals.py --provider anthropic` (or `gemini`, `openai`) with a key set to compare an LLM Director against this baseline; results land in `evals/results/`.

## Engineering notes

A few problems that were only found by measuring:

- **Half the dataset played at double speed.** The BVH release stamps every file at 120 fps, but subjects 141 and 143 were captured at 60 fps. Their "walk" measured 2.6 m/s with a 1.6 Hz stride, which is a jog. Gait-frequency analysis confirmed the true rate; `tools/build_motion_library.py` now corrects it per subject.
- **Loopable locomotion.** Walk cycles are cut by searching every window for maximum straight-line travel, minimum heading change and minimum start/end pose mismatch, so a 2-second walk can loop for 10 seconds without popping.
- **Fights that never connected.** After "circling", fighters ended up 8 m apart and punched air. The Blocking agent now checks per-clip reach and inserts an approach, moving the partner's reaction with it. Contact accuracy went from failing to 100% on the eval set.
- **Camera notes drifting off their moment.** "Slow push-in on the winner" was anchored to a time that blocking later shifted. Shots now follow the beat they were attached to.

## Run it

```bash
# Option 1: zero dependencies beyond pydantic (stdlib server)
pip install pydantic
python devserver.py            # http://127.0.0.1:8000

# Option 2: production server
pip install -r requirements.txt
uvicorn app:app --port 8000

# Option 3: Docker
docker build -t sceneforge . && docker run -p 8000:8000 --env-file .env sceneforge
```

### Go live (free) on Render

1. Push this folder to a GitHub repository.
2. On [render.com](https://render.com), choose **New > Blueprint** and pick the repository. `render.yaml` sets everything up (Python, `pip install pydantic`, `python devserver.py`, health check).
3. Open the `onrender.com` link Render gives you. Free services sleep after 15 minutes idle and take about a minute to wake.

The public server allows 20 scenes per visitor per minute (`SCENEFORGE_RATE_LIMIT`). If you add an LLM API key on a public deployment, also set a spending limit with that provider.

Set `ANTHROPIC_API_KEY`, `GEMINI_API_KEY` or `OPENAI_API_KEY` (see `.env.example`) to use an LLM Director. Without a key the rule-based planner runs, so the demo always works. Hindi and Telugu prompts need an LLM.

```bash
python -m unittest discover tests      # 21 tests, includes a mocked LLM repair loop
python evals/run_evals.py              # benchmarks
```

### Rebuild the motion library (optional)

The processed library is committed in `data/`. To rebuild from the raw BVH files:

```bash
git clone --depth 1 --filter=blob:none --sparse https://github.com/una-dinosauria/cmu-mocap
pip install numpy
python tools/build_motion_library.py --bvh-root cmu-mocap/data
```

Clips, tags and trims live in `data/clip_catalog.tsv`; add a row to add a motion.

### Add a new kind of scene

1. Find takes in the CMU index (`cmu-mocap-index-text.txt`), for example `79_12 eating dinner`.
2. Add a row to `data/clip_catalog.tsv`: `id, cmu take, category, comma-separated tags, description, optional trim`.
3. Rebuild the library (above). The build prints each clip's duration, travel and head height so you can spot bad takes.
4. Optional: give it furniture or hand props in `frontend/props.js` (`PROP_SPECS`), and synonyms in `sceneforge/library.py`.
5. Add a phrase to `evals/retrieval_cases.tsv` and run the evals to check the new motion is found.

## API

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/direct` | `{"prompt": "...", "provider": "auto"}` returns `{scene, trace, total_ms, provider}` |
| `GET` | `/api/library` | Clip catalog, skeleton joints and bones |
| `GET` | `/api/providers` | Which planners are available on this server |
| `GET` | `/motions/{clip}.json` | Joint positions for one clip (30 fps, millimetres) |

## Project layout

```
sceneforge/    agents: director.py, planner_offline.py, llm.py, library.py, schema.py, api.py
frontend/      Three.js viewer: app.js, motion.js (playback), camera.js (cinematographer), props.js (set dressing)
tools/         build_motion_library.py (BVH parsing, FK, fps correction, loop windows)
data/          clip_catalog.tsv + processed motion library
evals/         retrieval_cases.tsv, scenes.jsonl, run_evals.py, results/
tests/         unit tests
```

## Limitations and roadmap

- Motions come from a fixed library; blending is a 0.28 s crossfade, so some transitions slide. Next: foot-locking IK and motion matching between clips.
- Mannequins are procedural proxies. Next: FBX/glTF export of the blocked scene so it opens in Maya, Blender or MotionBuilder.
- The rule-based planner handles English only. LLM planners handle other languages.
- Next: a Blender MCP server so an agent can apply the plan to a real rig; text-to-motion generation for actions outside the library.

## Credits

- Motion data: [CMU Graphics Lab Motion Capture Database](http://mocap.cs.cmu.edu), BVH conversion by Bruce Hahne. Check CMU's usage terms before redistributing the raw data.
- Rendering: [three.js](https://threejs.org) (MIT), vendored in `frontend/vendor`.

Built by [Pranay Sai Vanam](https://linkedin.com/in/pranay-sai-vanam).
