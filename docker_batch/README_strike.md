# Strike task — final pipeline (the report)

The report's arena: 40 × 20 m, 3 tall obstacles, 1 free chaser (radius 1.5 m, 2.5 m/s, 2.5 m/s²; "Chaser strength" below),
a drifting target at x 32–37 m, 15 s episodes.
One driver trains every stage for every seed on the GPU (one run at a time), evaluates everything on 1000 test scenes and
writes the tables and figures:

```bash
docker exec -d px4sitl bash -c "bash ~/ws_shared/PS2-RL/docker_batch/run_strike_final.sh > ~/ws_shared/PS2-RL/outputs/strike_final.out 2>&1"
docker exec -it px4sitl bash -c "tail -F ~/ws_shared/PS2-RL/outputs/strike_final.out"
# parts: STAGES="1" (Phase I), "2" (residual striker), "3" (PS2), "4" (evaluation + report); SEEDS="0 1 2"; N=1000
```

| stage | task file | output (`outputs/strike_final/…`) |
|---|---|---|
| 1 Phase I backup | `configs/strike_final_p1.json` | `p1/td3_a0p2_seed<s>` (+ `p1/compare`) |
| 2 residual striker, no filter | `configs/strike_final_residual.json` | `residual/striker_rs0p1_s<s>` |
| 3 PS2 through the deployed filter | `configs/strike_final_ps2.json` | `ps2/striker_rs0p1_s<s>` |
| 4 evaluation (`run_strike_final_eval.sh`) | the PS2 task's env block and filter | `eval/s<s>/<run>/` |
| 5 report (`scripts/report_strike.py`) | — | `report/` (results, ablation, horizon, phase1 tables in md + tex; figures) |

The deployed filter is `tracker.filter_guidance` of `strike_final_ps2.json` (single source of truth for training and
evaluation). Evaluations (per seed): pursuit with no filter / static chaser balls / guidance with one escape / bCBF / the
deployed filter; the residual striker without and with the deployed filter; the PS2 striker without the filter, with the
deployed filter, with one escape, with the bCBF, and the ablations (hover-LQR backup, no committed chaser prediction, no
hover latch, static chaser balls). The `*_blind` runs (pursuit and PS2) use the same filter blind to the chaser
(obstacles only): what ignoring the chaser costs. Per-episode records (`strike_filter_episodes.json`) give the 10 s
numbers from the same runs (`report/horizon.md`). Each seed is evaluated with the chaser its striker was trained with
(read from its PS2 run's `configs.json`), whatever chaser the task file holds now; `ENV='{...}'` overrides it.

Report only (CPU is fine): `python scripts/report_strike.py --root outputs/strike_final --seeds 0 1 2`. Missing runs are
skipped (the "seeds x episodes" column says how many seeds a row has).

## Looking at specific episodes

Episode i is the same scene and perturbation in every evaluation of a root (same N and scene seed), so one episode can
be compared across methods:

```bash
# the episodes under PS2 without the filter, PS2 / pursuit / residual striker with the deployed filter, PS2 with the
# static chaser ball and with the bCBF: <ROOT>/episodes/s<seed>/episode<i>.png / .pdf / .gif and episodes.md
docker exec -it px4sitl bash -c "cd ~/ws_shared/PS2-RL && EPISODES='1 6 473 845' bash docker_batch/run_strike_episodes.sh"
# episodes whose outcome differs between runs, and a pick of interesting ones (contacts, changed outcomes, close calls)
python scripts/compare_strike_episodes.py --runs outputs/strike_final/eval/s0/pursuit_final outputs/strike_final/eval/s0/ps2_final \
    --out outputs/strike_final/compare_s0
```

`run_strike_episodes.sh` simulates all N episodes per method and records the chosen ones (`--record_episodes`), so they
are exactly the evaluated episodes (about 6 min per method on the laptop GPU). `EXACT=0` replays only the chosen
episodes (`--only_episodes`, fast); on another device or batch size the closed loop can differ in the last digits, and
borderline episodes (slow strikes near the time limit) can then end differently.

## Paper kit (tables, figures, episode figures and GIFs, methods document)

Everything for the paper from the final root, with Korean labels by default:

```bash
docker exec -it px4sitl bash -c "cd ~/ws_shared/PS2-RL && bash docker_batch/run_strike_paper.sh"
# tables, figures and quoted numbers only (seconds, CPU is fine):  STEPS=tables bash docker_batch/run_strike_paper.sh
# English labels: PAPER_LANG=en;  chosen episodes: EPISODES="845 392";  episodes picked automatically: N_PICK=8
# fast replay of only the chosen episodes (CPU possible): EXACT=0 (default 1: exactly the evaluated episodes)
# filter snapshots of other episodes (after the replays): STEPS=snap SNAP_EPISODES="845 994"
```

| output (`outputs/strike_final/paper/…`) | content |
|---|---|
| `tables/table1_10s`, `table1_15s` | main comparison at 10 s and 15 s: success, contact % (count), time-up, mean strike time |
| `tables/table2_final` | the final filter against the bCBF and one escape (seeds where every row was evaluated) |
| `tables/table3_ablation` | ablations of the final filter with the PS2 striker (common seeds) |
| `tables/table4_phase1` | Phase-I recovered fraction per region, learned backup vs hover LQR |
| `tables/table5_paired` | paired comparisons on the same scenes, exact McNemar test |
| `figures/fig_*.png` / `.pdf` | outcomes, strike time, chaser clearance, filter activity, training curves, ablation, Phase I |
| `numbers.md` | the numbers quoted in the text: rates with 95 % Clopper–Pearson intervals, counts, paired tests |
| `episode_picks.json` | representative episodes (filter saves, close evasion, the contact, time-ups, baselines failing) |
| `episodes/ep<i>/` | `compare2d` (png, gif), `signals`, `detail_<tag>`, `view3d`, `view3d_<tag>`, `anim3d_<tag>.gif`; `overview.png`, `overview_main.png` |
| `snapshots/` | the filter's check redrawn at chosen steps (`scripts/strike_filter_snapshot.py`): the seven backup plans on the top view and the 7 × 5 certification matrix (backup × blend) with the chosen cell |

Every table comes as `.md`, `.csv` (opens in Excel), `.tsv` (paste into an HWP table) and `.tex`. Korean labels need a
Korean font for matplotlib in the container (`.venv/bin/pip install koreanize-matplotlib`, or `apt-get install
fonts-nanum`); without one the figures fall back to English labels.

The methods document (Korean, paper style: model, safe set and the h_c derivation, base set, Phase I, backup family,
discrete filter with its algorithm, bCBF comparison, striker, PS2, evaluation protocol and statistics, results,
hyper-parameters, what differs from the draft) is `results/strike_paper/strike_methods_ko.tex`. It reads the tables and
figures above: `cd results/strike_paper && xelatex strike_methods_ko.tex && xelatex strike_methods_ko.tex` (XeLaTeX and the
Noto CJK fonts), or `xelatex "\def\paperdir{/path/to/paper}\input{strike_methods_ko.tex}"` for another output folder.

## Chaser strength

The Phase-I backup is trained against static balls around a 0.4 m, 2 m/s chaser (the `strike` block). That chaser hardly
matters: straight pursuit that ignores it is hit in 1.6 % of the runs. The chaser the drone meets is set in the `env`
block, without touching the backup or the striker's observation format:

| env key | meaning | final (residual and PS2 task files) |
|---|---|---|
| `actual_chaser_radius` | chaser radius [m] (contact at radius + r_drone) | 1.5 |
| `actual_chaser_speed` | top speed [m/s] | 2.5 |
| `chaser_accel` | acceleration limit [m/s²] | 2.5 |

The filter is built for this chaser: h_c, the pure-pursuit prediction and the escape speeds (2 v_c, 3.5 v_c) use these
values. The learned backup keeps its training model. Phase I is unchanged, so finished Phase-I runs stay valid. Eval
option `--filter_sees_chasers 0` gives the same filter blind to the chaser.

A new chaser after Phase I has finished: copy the Phase-I runs into a new root and run stages 2–4 there.

```bash
docker exec px4sitl bash -c "cd ~/ws_shared/PS2-RL && mkdir -p outputs/strike_final_r15v25 && cp -r outputs/strike_final/p1 outputs/strike_final_r15v25/"
docker exec -d px4sitl bash -c "cd ~/ws_shared/PS2-RL && ROOT=outputs/strike_final_r15v25 STAGES='2 3 4' bash docker_batch/run_strike_final.sh > outputs/strike_final_r15v25.out 2>&1"
```

How it was chosen: 64 test scenes (seed 9001), 15 s, the PS2 striker of the sparse-arena decision runs (trained with the
old chaser); "no filter" from 128 scenes. Chaser acceleration = top speed / 1 s in every row.

| chaser radius, speed | pursuit, no filter: chaser hits¹ | pursuit, filter blind to the chaser | pursuit, deployed filter: chaser hits | PS2, filter blind to the chaser | PS2, deployed filter: strike / chaser hits / time-up | PS2: most time without a certificate in one episode |
|---|---|---|---|---|---|---|
| 0.4 m, 2 m/s (old) | 0.000 | 0.016 | 0/64 | — | 1.000 / 0 / 0 (128 scenes) | 0.8 s |
| 1.0 m, 2.5 m/s | 0.088 | — | 2/64 | — | — | — |
| 1.0 m, 3 m/s | 0.136 | 0.172 | 2/64 | 0.031 | 0.906 / 0 / 0.094 | 1.6 s |
| **1.5 m, 2.5 m/s** | 0.197 | 0.266 | 3/64 | 0.016 | **0.891 / 0 / 0.109** | 1.4 s |
| 1.25 m, 3 m/s | 0.197 | — | — | — | 0.891 / 1 / 0.094 | 3.7 s |
| 1.5 m, 3 m/s | 0.333 | 0.312 | — | 0.062 | 0.906 / 1 / 0.078 | 3.4 s |
| 1.0 m, 4 m/s | 0.365 | — | — | 0.078 | 0.859 / 2 / 0.109 | 4.1 s |

¹ among the runs that did not hit an obstacle first.

- Speed costs the filter quadratically (the h_c offset v_c δ + v_c²/(2 a_e) and the braking distance of the escape
  (2 v_c)²/(2 a_e)), the radius only linearly. So the radius grows more than the speed.
- With the PS2 striker, every contact came after more than 3 s without a certificate: the chaser herds the drone against
  the wall or an obstacle.
- Straight pursuit through the deployed filter is hit in 2–3 of 64 scenes at every stronger chaser tried (0 with the old
  one). The filter predicts the chaser as a pure pursuer at v_c; the real chaser has momentum and can come closer than
  predicted when the drone turns back. The PS2 striker keeps out of those situations.
- 1.5 m, 2.5 m/s is the strongest threat tested where the PS2 striker with the deployed filter had no contact, with the
  most certificate margin of the stronger chasers.

```bash
# threat sweep without the filter (fast): chaser contact of a nominal that ignores the chaser
python scripts/sweep_strike_chaser.py --backup_run outputs/strike_final/p1/td3_a0p2_seed0 \
    --task docker_batch/configs/strike_final_ps2.json --nominal pursuit --radii 0.4 1.0 1.5 2.0 --speeds 2 2.5 3 4 \
    --accel_per_speed 1.0 --out outputs/strike_chaser_sweep
# one chaser through the deployed filter (the env keys override the task file); --filter_sees_chasers 0: blind filter
python scripts/eval_strike_filter.py --backup_run outputs/strike_final/p1/td3_a0p2_seed0 \
    --task docker_batch/configs/strike_final_ps2.json --env '{"actual_chaser_radius": 1.5, "actual_chaser_speed": 3, "chaser_accel": 3}' \
    --chaser_model guidance --guidance "$(python -c "import json; print(json.load(open('docker_batch/configs/strike_final_ps2.json'))['tracker']['filter_guidance'])")" \
    --nominal striker --striker_run outputs/strike_final/ps2/striker_rs0p1_s0 --modes learned --episodes 64 --out outputs/strike_chaser_r1p5v3
```

# Strike task — Phase I (safe-arrival backup)

Python only (no SITL): 4 g thrust box, body rates ±18 rad/s, dt 0.02 s. Task file `docker_batch/configs/strike.json`.

## What is trained

A backup policy that, from a moving drone among tall obstacles (cylinders, boxes) and chaser balls, reaches the
base set within T = 1.5 s without contact. The hover-LQR backup is the baseline.

- **Safe set:** z ≥ floor (1 m), obstacle distance ≥ r_d (0.3 m), outside every chaser ball (radius r_c + r_d + v_c T = 3.7 m,
  the chaser's reachable ball over the horizon, treated as static).
- **Base set (hover at the current point):** anchor a = p + D r (r = velocity and attitude), V_min(r) = rᵀ S r ≤ c_B = 50,
  and the hover ellipsoid around a (1.22 m wide, 0.63 m tall) clear of obstacles, balls and the floor. c_B is
  certified at load (input box 56.1, chart 102, sampled one-step decrease with wind 0.3 m/s²: 56.1).
- **Backup = hover LQR + learned residual** (`strike.backup_action: residual_lqr`): u = u_LQR(x) + Δu. The trainer's
  action penalty `action_smoothness_weight · mean((Δu/scale)²)` is the anchor to the LQR (default 0.5). Without an anchor
  the actor, started from the LQR, collapses within a few thousand updates: one 0.02 s action barely changes the
  outcome, so the critic's action gradient is weak and the actor follows its errors (a constant roll-rate bias, no
  braking). With anchor 0.5 a 400k-step CPU check went from 0.44 to 0.50 (weighted) without a dip.
- **Backup input (98):** z − floor, v, q; the 8 obstacle slots (closest-point vector, centre offset, half-sizes,
  cos 2ψ, sin 2ψ, rounding, present); 2 chaser slots (offset, radius, present). Slot order is fixed at the start of a
  rollout (nearest first).
- **Objective:** the safe-arrival value β^(steps to B) (TD3, critic-only warm-up of 20k updates). No disturbance.
- **Regions:** general (random), blocked (obstacle where the LQR ends up), chaser (ball next to where the LQR ends up),
  shell (slow states, sanity check). Mixture 0.4 / 0.2 / 0.2 / 0.2.

## Run (GPU)

```bash
docker exec -d px4sitl bash -c "bash ~/ws_shared/PS2-RL/docker_batch/run_strike_phase1.sh > ~/ws_shared/PS2-RL/outputs/strike_phase1.out 2>&1"
docker exec -it px4sitl bash -c "tail -F ~/ws_shared/PS2-RL/outputs/strike_phase1.out"
```

Two runs of 5 M steps (anchor 0.5 and 0.2, seed 0), then the comparison on CPU. Knobs: `ANCHORS`, `SEEDS`, `STEPS`,
`BACKBONE`, `TRAIN_ARGS` (e.g. `TRAIN_ARGS="--strike_T 1.0"`), `OUT`. One run's log:
`outputs/strike_phase1/logs/td3_a0p5_seed0.log`.

## Outputs (`outputs/strike_phase1/`)

- `td3_a<anchor>_seed<k>/strike_backup_policy_actor.pkl`: the backup (best validation checkpoint); `summary.json` has
  the learned and LQR recovered fraction per region on val and test.
- `compare/strike_phase1_table.md`: recovered fraction per region, LQR vs each run.
- `compare/strike_phase1_outcomes.png`, `strike_phase1_examples.png`, `strike_phase1_curves.png`.

## Checks

1. The log's `[base set]` line: c_B below every bound; otherwise training stops.
2. `[eval] LQR backup (val)`: about general 0.52–0.55, blocked 0.00, chaser 0.11–0.15, shell 1.00.
3. `shell` stays at 1.00 for the learned backup.
4. The `[eval val]` lines rise without a collapse (no drop of general below the LQR after the 20k-update warm-up).
5. The learned backup beats the LQR on general and chaser (`[done]` line, `compare/strike_phase1_table.md`).

Send back: `outputs/strike_phase1/` without `*_weights.pkl`, and `outputs/strike_phase1.out`.

# Strike task — Phase II stage 1 (vanilla striker, no filter)

Arena 40 × 20 m with 6–10 tall obstacles, 0–2 chasers pursuing at ≤ 2 m/s, a target drifting at 0.2–0.8 m/s
(`ps2rl/envs/strike_env.py`). Reward per step: 2 · (distance closed) − 0.05 − 0.05 · |Δu from hover|², +100 strike,
−150 crash / out of the arena; episode ≤ 10 s; γ = 0.995. SAC (`ps2rl/phase2_ps2/strike_tracker_trainer.py`), critic
reward scale 0.1, seeds 0 and 1 in parallel, 10 M steps each (about 1.1 h on the RTX 4050).

The first batch (10/21 night: progress 5, strike +50, crash −50, γ 0.99, `outputs/strike_tracker/`) learned to dive at the
target at 15–22 m/s and crash (best: strike 0.25, crash 0.74): crashing near the target still paid +100 in progress, and
γ 0.99 looks only ~2 s ahead. The task file now has the rebalanced reward above.

```bash
docker exec -d px4sitl bash -c "bash ~/ws_shared/PS2-RL/docker_batch/run_strike_tracker.sh > ~/ws_shared/PS2-RL/outputs/strike_tracker_v2.out 2>&1"
docker exec -it px4sitl bash -c "tail -F ~/ws_shared/PS2-RL/outputs/strike_tracker_v2.out"
```

Outputs: `outputs/strike_tracker_v2/striker_rs0p1_s<seed>/` (best / final weights, history, summary, configs) and logs in
`outputs/strike_tracker_v2/logs/`. The scene pool is built once (a few minutes on the CPU) and cached in the output folder.

# Strike task — Phase II stage 1b: striker without obstacles (goal reaching)

Task file `docker_batch/configs/strike_free.json`: no obstacles, no chasers, floor kept; start x 2–25 m, target x 8–37 m,
both |y| ≤ 8 m, at least 8 m apart; moving target as before; same reward as strike.json plus a mild penalty above the
backup's envelope (10 m/s, 45°). The striker's observation keeps its layout but hides the obstacle / chaser parts
(`obs_hide_scene`), so the same policy can later fly through the safety filter in the obstacle arena unchanged.

```bash
docker exec -d px4sitl bash -c "TASK=docker_batch/configs/strike_free.json OUT=outputs/strike_striker_free bash ~/ws_shared/PS2-RL/docker_batch/run_strike_tracker.sh > ~/ws_shared/PS2-RL/outputs/strike_striker_free.out 2>&1"
docker exec -it px4sitl bash -c "tail -F ~/ws_shared/PS2-RL/outputs/strike_striker_free.out"
```

Send back `outputs/strike_striker_free/` (without the scene pool .npz).

# Strike task — Phase II: residual striker

Task file `docker_batch/configs/strike_residual.json`. The policy outputs a correction du on top of the straight-line
pursuit controller at 6 m/s (`ps2rl/envs/strike_pursuit.py`), which alone strikes 100 % without obstacles:
|du| ≤ 0.1·a_max on thrust, 0.25·ω_max on the body rates. Sparse arena (3 obstacles, 1 free chaser at 2 m/s), scene
visible, no filter in training. Reward as in strike.json; the effort term is on du. The trainer warms up the critic for
200k steps before the actor moves (`actor_start_update`) and anchors the actor to du = 0 (`action_anchor_weight` 0.2),
as in Phase I. At step 0 the policy is the pursuit controller.

```bash
docker exec -d px4sitl bash -c "TASK=docker_batch/configs/strike_residual.json OUT=outputs/strike_striker_residual bash ~/ws_shared/PS2-RL/docker_batch/run_strike_tracker.sh > ~/ws_shared/PS2-RL/outputs/strike_striker_residual.out 2>&1"
docker exec -it px4sitl bash -c "tail -F ~/ws_shared/PS2-RL/outputs/strike_striker_residual.out"
```

Send back `outputs/strike_striker_residual/` (without the scene pool .npz).

Variant with guidance features in the observation (`strike_residual_guid.json`: line-of-sight direction, distance and
log distance, relative velocity, closing speed, line-of-sight rate, errors in the yaw frame, target height, and the
pursuit command being corrected; `obs_guidance`, `obs_pursuit_cruise`):

```bash
docker exec -d px4sitl bash -c "TASK=docker_batch/configs/strike_residual_guid.json OUT=outputs/strike_striker_residual_guid bash ~/ws_shared/PS2-RL/docker_batch/run_strike_tracker.sh > ~/ws_shared/PS2-RL/outputs/strike_striker_residual_guid.out 2>&1"
```

# Strike task — Phase II PS2: residual striker trained through the safety filter

Task file `docker_batch/configs/strike_residual_ps2.json` (`ps2rl/envs/strike_filtered_env.py`): each step
u_nom = pursuit + du goes through the discrete guidance filter (Phase-I backup `outputs/strike_phase1/td3_a0p2_seed0`)
before the plant, so the striker learns the routes the filter lets through. Reward as in strike_residual_guid.json minus
0.5·|(u − u_nom)/scale|² (the filter's correction: being held in front of an obstacle costs). Warm start from the
unfiltered residual striker `outputs/strike_striker_residual_guid/striker_rs0p1_s0/best_weights.pkl`; the critic
re-adapts for 100k steps before the actor moves; 3M steps; evaluations run through the filter. One seed (two runs with
the 135-input observation do not fit the 6 GB GPU together).

```bash
docker exec -d px4sitl bash -c "SEEDS=0 TASK=docker_batch/configs/strike_residual_ps2.json OUT=outputs/strike_striker_ps2 bash ~/ws_shared/PS2-RL/docker_batch/run_strike_tracker.sh > ~/ws_shared/PS2-RL/outputs/strike_striker_ps2.out 2>&1"
docker exec -it px4sitl bash -c "tail -F ~/ws_shared/PS2-RL/outputs/strike_striker_ps2.out"
```

Send back `outputs/strike_striker_ps2/` (without the scene pool .npz).

Variant with escape variants in the filter (`strike_residual_ps2_var.json`, tracker `filter_guidance`): the backup family
also contains escapes at 4 or 7 m/s, straight away from the chaser or 60° to either side; the filter applies the largest
certified blend over all of them. On the sparse test set (guidance-feature striker) this removed the chaser contacts
(0.016 → 0) and most certificate losses, at about 3.5× the filter cost. It continues from the first PS2 run
(`outputs/strike_striker_ps2/striker_rs0p1_s0/best_weights.pkl`).

```bash
docker exec -d px4sitl bash -c "SEEDS=0 TASK=docker_batch/configs/strike_residual_ps2_var.json OUT=outputs/strike_striker_ps2_var bash ~/ws_shared/PS2-RL/docker_batch/run_strike_tracker.sh > ~/ws_shared/PS2-RL/outputs/strike_striker_ps2_var.out 2>&1"
```

# Strike task — full arena (Phase I with moving chasers → residual striker → PS2)

Full arena: 6–10 obstacles, 0–2 free chasers at 2 m/s, arena walls and ceiling, 15 s episodes. One driver runs the three
stages back to back on the GPU (each stage skips runs that already have `summary.json`):

```bash
docker exec -d px4sitl bash -c "bash ~/ws_shared/PS2-RL/docker_batch/run_strike_full.sh > ~/ws_shared/PS2-RL/outputs/strike_full.out 2>&1"
docker exec -it px4sitl bash -c "tail -F ~/ws_shared/PS2-RL/outputs/strike_full.out"
# one stage only, e.g. the residual striker:  STAGES=2 bash .../run_strike_full.sh
```

1. **Phase I** (`configs/strike_full.json` → `outputs/strike_phase1_full/td3_a0p2_seed0`). The backup is trained on the
   sets the filter checks:
   - chasers move (pure pursuit at v_c) during every episode and evaluation rollout (`strike.chaser_set "guidance"`);
   - safe set: obstacles, floor, walls, ceiling and the escape margin
     h_c = |p − c| − (r_c + r_d) − s⁺δ − s⁺²/(2a_e) ≥ 0, s = v_c − e·v (instead of the static 6.7 m ball);
   - base set: hover clear of obstacles / walls whose hover ellipsoid keeps h_c ≥ `chaser_base_margin` (2 m = 1 s at
     v_c) from every chaser; the ellipsoid's size and speed scale with the state's own level V_min, so a slow hover
     only needs ≈ 3.5 m;
   - observation per chaser slot: offset c − p, h_c, closing speed s, present; plus the 5 wall / ceiling distances
     (`strike.arena_walls`); safe / base membership stay exact functions of the observation;
   - T = 3 s (the filter's horizon), five regions incl. `escape` (slow, chaser 0.8–3.5 m beyond contact);
   - backup = flee-then-hover heuristic + learned residual (`strike.backup_action "residual_flee"`, `actor_init "base_bc"`:
     du = 0 at the start). The heuristic (`strike_scene.flee_action`) hovers while the hover keeps its chaser margin
     + 1 m, else flies away from the most critical chaser at 6 m/s, pushed off obstacles and walls; without chasers it is
     the hover LQR. With the braking LQR as the base the backup never learned to flee (CPU probes: chaser / escape
     regions stayed at 0); the heuristic alone recovers ≈ 0.4–0.5 of those regions.
2. **Residual striker** (`configs/strike_full_residual.json` → `outputs/strike_striker_full_residual`): the
   strike_residual_guid recipe in the full arena, no filter (warm-up). Its strike block is the Phase-I one, so its
   observation matches the filter's backup.
3. **PS2** (`configs/strike_full_ps2.json` → `outputs/strike_striker_full_ps2`): the striker through the discrete filter
   with the stage-1 backup. Backup family: the learned backup by itself first (`guidance.learned_alone`, certified by its
   Phase-I criterion: base set reached within T inside the safe set), then the learned / hover-LQR bases with the escape
   reflex and the five escape variants; the largest certified blend is applied.

Evaluation (CPU):

```bash
JAX_PLATFORMS=cpu python scripts/eval_strike_filter.py --backup_run outputs/strike_phase1_full/td3_a0p2_seed0 \
    --task docker_batch/configs/strike_full.json --modes "none learned" --chaser_model guidance \
    --guidance '{"horizon": 3.0, "learned_alone": true, "escape_variants": [[2.0, 60], [2.0, -60], [3.5, 0], [3.5, 60], [3.5, -60]]}' \
    --nominal striker --striker_run outputs/strike_striker_full_ps2/striker_rs0p1_s0 --episodes 128 --gif 4 --gif_pick failures \
    --out outputs/strike_filter_eval_full
```

# Strike task — Phase I closed-loop evaluation (safety filter on a simple controller)

`ps2rl/cil/strike_filter.py`: each step keeps the largest blend λ·u_nom + (1−λ)·u_backup (λ = 1, .75, .5, .25, 0) whose
next state is recoverable (the composed backup - hover LQR inside the base set, the backup outside - reaches the base set
within T without contact, all arena obstacles, chaser balls at the chasers' current positions); otherwise it continues the
committed plan. The learned backup sees only what Phase I trained it on (obstacles within the region's dist_max, balls
within ball_dist_max).

```bash
# CPU, ~10 min for 128 scenes: straight-line pursuit at 6 m/s without the filter and through it (LQR / learned backup)
JAX_PLATFORMS=cpu python scripts/eval_strike_filter.py --backup_run outputs/strike_phase1/td3_a0p2_seed0 --out outputs/strike_filter_eval
# the trained striker as the nominal controller
JAX_PLATFORMS=cpu python scripts/eval_strike_filter.py --backup_run outputs/strike_phase1/td3_a0p2_seed0 \
    --nominal striker --striker_run outputs/strike_tracker/striker_rs0p1_s0 --out outputs/strike_filter_eval_striker
```

Options added for the chaser study (defaults keep the earlier behaviour): `strike.chaser_hold` (base set keeps the hover
point clear of a chaser's reach for T + hold) and the `escape` Phase-I region (5th mixture weight; slow states with a
chaser ball 0–2 m beyond contact).

Hover latch (on by default, `--hover_latch 0` for the old behaviour): once a certified plan reaches the base set the
filter holds that anchor and continues with the hover LQR at the fixed anchor (the controller the base-set certificate
covers). Re-anchoring every step let the anchor drift by millimetres, so a hover that entered B at zero margin dropped
out of it and the certificate was lost while hovering safely.

Chasers. A free pursuer's ball R = r_c + r_d + v_c T only covers its reach for T: wherever the drone stops, the chaser
arrives later, so the hover base set expires (`--chaser_model reach` checks the exact reach r_c + r_d + v_c τ and makes
the hover last `--reach_hold` s; it still expires). Guards make the ball exact: with `--chaser_leash L` (L ≤ v_c T) each
chaser pursues but stays within L of its post, and the filter gets the posts, so the ball around a post holds every point
the guard can ever reach and the hover base set is invariant again.

Guidance-law chaser set (`--chaser_model guidance`, `ps2rl/cil/strike_guidance.py`): the safe set keeps the escape margin
h_c = R − r_c − r_d − s⁺δ − s⁺²/(2a_e) ≥ 0 (s = v_c − opening speed), the backup is blended with an evasion reflex when
h_c is small, chasers are propagated as pure pursuers over a 3 s rollout; new plans keep 0.2 m of h_c and 0.15 m of
obstacle clearance in hand, and the hover LQR is tried as a second backup. Options as JSON: `--guidance '{"a_evade": 5}'`;
sparse scenes: `--env '{"obstacles_min": 3, "obstacles_max": 3, "chasers_active_min": 1, "chasers_active_max": 1}'`.

Minimal-correction backup-CBF filter (`--filter_type bcbf`, `ps2rl/cil/strike_bcbf.py`): the PS2-RL backup-CBF QP on
the strike sets - rows d/dt h(phi(tau_i, x)) >= -alpha h along the backup flow with the discrete sensitivity (exact
Jacobian of the Euler rollout, reduced to Q [f, g]), relative-time form, a base-set row at the end, per-action weights
(1, 0.1, 0.1, 1) - with the discrete filter as its safeguard. The chasers are part of the rows' state (pure pursuit in
the flow, h_c rows). Where the backup itself violates the rows the QP is skipped. Options as JSON: `--bcbf '{"alpha": 4}'`.

```bash
JAX_PLATFORMS=cpu python scripts/eval_strike_filter.py --backup_run outputs/strike_phase1/td3_a0p2_seed0 --modes learned \
    --chaser_model guidance --filter_type bcbf --env '{"obstacles_min": 3, "obstacles_max": 3, "chasers_active_min": 1, "chasers_active_max": 1}' \
    --gif 2 --gif_pick close_chaser --out outputs/strike_filter_eval_bcbf
```

```bash
# guards on a 3 m leash (= v_c T: the ball the backup was trained with), chasers at full speed
JAX_PLATFORMS=cpu python scripts/eval_strike_filter.py --backup_run outputs/strike_phase1/td3_a0p2_seed0 --modes learned \
    --chaser_leash 3.0 --out outputs/strike_filter_eval_guards
```
