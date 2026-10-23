# SceneForge eval results (offline)

| Metric | Value |
|---|---|
| retrieval cases | 89 |
| retrieval top1 | 0.989 |
| retrieval top3 | 1.0 |
| retrieval mrr | 0.994 |
| scene cases | 26 |
| scene valid_plan_rate | 1.0 |
| scene cast_accuracy | 1.0 |
| scene action_recall | 0.987 |
| scene camera_note_recall | 1.0 |
| scene contact_rate | 1.0 |
| scene shot_coverage | 1.0 |
| scene collision_rate | 0.012 |
| scene time_of_day_accuracy | 1.0 |
| scene latency_ms_p50 | 5.0 |
| scene latency_ms_p95 | 9.8 |
| scene auto_fixes_per_scene | 1.654 |

## Per scene

| id | cast | actions | camera | contact | coverage | collisions | fixes | ms |
|---|---|---|---|---|---|---|---|---|
| fight | ok | 1.0 | 1.0 | 1.0 | ok | 0.0 | 2 | 14.0 |
| stealth | ok | 1.0 | 1.0 | - | ok | 0.0556 | 2 | 4.6 |
| greeting | ok | 1.0 | 1.0 | 1.0 | ok | 0.0203 | 2 | 6.5 |
| zombie | ok | 1.0 | 1.0 | - | ok | 0.0 | 2 | 5.3 |
| soldier | ok | 1.0 | 1.0 | - | ok | 0.0 | 2 | 8.0 |
| dancers | ok | 1.0 | 1.0 | - | ok | 0.0 | 0 | 6.1 |
| duel | ok | 1.0 | - | 1.0 | ok | 0.031 | 4 | 9.8 |
| chase | ok | 1.0 | 1.0 | - | ok | 0.0 | 2 | 4.1 |
| janitor | ok | 1.0 | - | - | ok | 0.0 | 1 | 6.1 |
| argument | ok | 1.0 | 1.0 | 1.0 | ok | 0.0457 | 2 | 5.0 |
| gymnast | ok | 1.0 | 1.0 | - | ok | 0.0 | 1 | 5.6 |
| worker | ok | 0.667 | 1.0 | - | ok | 0.0 | 1 | 4.2 |
| boxers | ok | 1.0 | 1.0 | 1.0 | ok | 0.072 | 3 | 6.3 |
| friends | ok | 1.0 | - | 1.0 | ok | 0.0432 | 3 | 2.9 |
| drunk | ok | 1.0 | - | - | ok | 0.0 | 1 | 3.7 |
| ladder | ok | 1.0 | - | - | ok | 0.0 | 1 | 2.7 |
| kid | ok | 1.0 | 1.0 | - | ok | 0.0 | 1 | 4.9 |
| ballet | ok | 1.0 | 1.0 | - | ok | 0.0 | 1 | 2.6 |
| ninja | ok | 1.0 | - | 1.0 | ok | 0.0 | 2 | 2.7 |
| waiter | ok | 1.0 | - | 1.0 | ok | 0.0361 | 2 | 2.3 |
| chef | ok | 1.0 | 1.0 | - | ok | 0.0 | 1 | 5.0 |
| dinner | ok | 1.0 | - | - | ok | 0.0 | 1 | 5.1 |
| family | ok | 1.0 | - | - | ok | 0.0 | 1 | 6.2 |
| morning | ok | 1.0 | 1.0 | - | ok | 0.0 | 1 | 4.8 |
| bootcamp | ok | 1.0 | - | 1.0 | ok | 0.0 | 3 | 4.8 |
| gym | ok | 1.0 | - | - | ok | 0.0 | 1 | 6.9 |

## Retrieval misses (top-1)

- "flips backwards": expected backflip; got walk_back, backflip, cartwheel
