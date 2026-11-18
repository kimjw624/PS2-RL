# Strike task (reduced arena) — improvements after the report

Filter smoothing, wall-aware backup, Phase-I variants.

Filter options (`tracker.filter_guidance` / `--guidance`, all off by default; the defaults reproduce the report):

| option | what it does |
|---|---|
| `lam_rise_max` | after a cut the blend comes back as a ramp: it may rise by at most this per step (it may always fall, and the ramp never blocks a certificate) |
| `prefer_committed_variant` | equal blends certified with several backups: keep the committed plan's backup |
| `lambda_grid` | the blends tried (default 1, 0.75, 0.5, 0.25, 0) |
| `escape_latch_on` / `_off` | below h_c = on, only blend 0 (the backup with its escape reflex) until h_c > off |
| `escape_variants` entries | `[speed ratio, angle]` or `[speed ratio, angle, trigger_off]`: a variant with its own reflex trigger, e.g. `[3.5, 60, 6]` flees from h_c < 6 m (early escape) |
| `fallback_zero_slack` | no new plan with the slack and no committed plan: accept a new plan certified on the safe set itself before declaring the certificate lost |
| `select_by` | `"lambda"` (default): the largest certified blend; `"deviation"`: the certified candidate with the smallest normalised correction ‖(u − u_nom)/scale‖ over all blends and backups |

Nominal options (`scripts/eval_strike_filter.py`; they change the pursuit that the nominal, or the residual striker's
base, follows; the filter is unchanged; all off by default):

| option | what it does |
|---|---|
| `--pursuit_detour R` | a chaser that blocks the line to the target is passed at radius R [m]: the direction turns to the edge of the cone of directions that pass within R of it (inside R: tangent to it) |
| `--pursuit_search 1` | look-ahead heading search: the smallest turn of the target direction whose ray is clear of obstacles and walls (`--pursuit_search_margin`, 1 m) and, with `--pursuit_detour R`, of the chasers; look-ahead `--pursuit_search_lookahead` (6 m), largest turn `--pursuit_search_max_turn` (120°) |
| `--pursuit_search_heading_weight w` | the search minimises \|turn\| + w \|angle to the velocity\| (VFH+-style): it keeps to one side of an obstacle instead of switching |
| `--pursuit_search_final d` | within d [m] of the target the pursuit flies straight (a target next to a wall or box has no free ray) |
| `--pursuit_search_wall_margin m` | clearance of the ray from the arena walls (default: `--pursuit_search_margin`) |

`scripts/eval_strike_filter.py` now also reports how smooth the applied command is: the filter's active share, kept/changed
switches per second, Σ|Δλ| per second, ‖Δu‖ relative to the nominal command's own changes, and backup switches per second.
These go in the table and in the per-episode records.

CPU check on 16 test scenes (seed-0 models of the report run):
- the 7 scenes of the paper's trajectory figures;
- the 5 scenes where the filter acted most;
- 4 random scenes.

| PS2 striker, final filter | strike | contact | time-up | ‖Δu‖ / nominal | kept/changed switches/s | backup switches/s |
|---|---|---|---|---|---|---|
| as in the report | 14 | 1 (scene 845) | 1 | 3.5 | 1.98 | 2.32 |
| `lam_rise_max` 0.05 | 15 | 1 | 0 | 1.9 | 0.76 | 0.60 |
| wall-aware backup (v2, `learned_alone`) | 15 | 0 | 1 | 3.9 | 2.10 | 1.54 |
| v2 + `lam_rise_max` 0.05 | 15 | 0 | 1 | 2.0 | 0.82 | 0.71 |

- No improvement: `prefer_committed_variant`, a 9-level `lambda_grid`, the escape latch.
- `slide_along` added a chaser contact.
- `best_effort_fallback` avoided scene 845 without a certificate.
- The v2 backup is the moving-chaser, arena-wall Phase I of the full-arena pipeline (`outputs/strike_phase1_full/td3_a0p2_seed0`). In scene 845 it avoids the wall pin, and it lost no certificate in the 16 scenes.

48-scene CPU study (time-ups, herding and the nominal detour): `results/strike_reduced/improvements_cpu48/README.md`.
GPU rounds (1000 scenes × seeds 0, 1): `results/strike_reduced/improvements_gpu/README.md`. With the heading search,
PS2 strikes 99.65 % with no contact (report filter: 97.6 %, 1 contact), the residual striker 99.4 % (94.4 %).
Recommended: PS2 + heading search + early escapes (`ps2_search_early`: 99.70 % strikes, no contact, no long loss of
the certificate); the final-approach / wall-margin options were rejected.
Lost certificates and the early-escape fix: `results/strike_reduced/lost_certificate/README.md`.

Paired comparison of any driver output (needs the training venv):

```bash
docker exec -it px4sitl bash -c "cd ~/ws_shared/PS2-RL && source docker_batch/landing_env.sh && \
  python scripts/compare_strike_improve.py outputs/strike_improve --out outputs/strike_improve/compare.md"
```

Full test set on the GPU, against the report's filter (seeds 0 and 1, about 3–6 min per variant and seed):

```bash
docker exec -it px4sitl bash -c "cd ~/ws_shared/PS2-RL && bash docker_batch/run_strike_improve.sh"
# -> outputs/strike_improve/summary.md; variants: VARIANTS="ps2_ref ps2_ramp ...", table: DEFS (see the script)
```

Compute time of one filter step for one vehicle, jit on a 2-core CPU:
- final filter (7 backups × 5 blends): 62 ms median (75 ms p95);
- one escape (2 × 5): 13 ms;
- 9 blends (7 × 9): 85 ms.

The control step is 20 ms, so the final filter needs a GPU or a cheaper search (for example, certify λ = 1 over the
backups first and stop at the first certificate) to run in real time on such a CPU.

Phase-I variants (train, then evaluate with the report's strikers):
- `configs/strike_final_p1_esc.json` puts the low-speed escape region in the training mix at weight 0.1;
- `P1_ARGS="--strike_arena_walls 1"` adds arena walls.

```bash
# Phase I with the low-speed escape region and arena walls (seed 0, about 11 min), then the report's strikers through it
docker exec -it px4sitl bash -c "cd ~/ws_shared/PS2-RL && \
  ROOT=outputs/strike_imp_walls_esc STAGES=1 SEEDS=0 P1_TASK=docker_batch/configs/strike_final_p1_esc.json \
    P1_ARGS='--strike_arena_walls 1' bash docker_batch/run_strike_final.sh && \
  cp -r outputs/strike_final/residual outputs/strike_final/ps2 outputs/strike_imp_walls_esc/ && \
  ROOT=outputs/strike_imp_walls_esc STAGES=4 SEEDS=0 RUNS='ps2_final ps2_guid1 residual_final' bash docker_batch/run_strike_final.sh"
# escape region only: drop P1_ARGS; walls only: drop P1_TASK
```

Separate training + blending vs one policy trained on a weighted sum. The residual striker is one point on that curve:
- it was trained without the filter on progress + strike bonus − 150 per contact;
- without the filter it scores 74.0 % strikes with 26.05 % contacts;
- with the final filter it scores 94.4 % strikes with 0 contacts.

More points come from larger contact penalties, each trained and then evaluated with and without the filter
(`residual_final` runs both):

```bash
docker exec -it px4sitl bash -c "cd ~/ws_shared/PS2-RL && for c in 400 1000 2500; do \
  ROOT=outputs/strike_ws_c\$c STAGES=2 SEEDS=0 EXTRA_ARGS=\"--env_crash_penalty \$c\" bash docker_batch/run_strike_final.sh; \
  cp -r outputs/strike_final/p1 outputs/strike_ws_c\$c/; \
  ROOT=outputs/strike_ws_c\$c STAGES=4 SEEDS=0 RUNS=residual_final bash docker_batch/run_strike_final.sh; done"
```
