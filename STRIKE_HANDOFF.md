# Strike task (reduced arena) — handoff

Branch `strike-reduced-arena`. Everything needed to train, evaluate and report the low-altitude strike task in the
reduced arena, plus reference models and the method write-up (`docs/strike/strike_method.pdf`).

**The conference report's run** (3 seeds, models, evaluations, tables, figures, trajectory figures) is in
`results/strike_reduced/final/`. Start with `results/strike_reduced/final/README.md`. The Korean methods document is
`results/strike_paper/strike_methods_ko.tex`.

## The task

A quadrotor (collective thrust and body-rate commands, 4 g thrust box, ±18 rad/s, dt 0.02 s) must strike a slowly
drifting target at low altitude among tall obstacles while a chaser pursues it.

- Arena: 40 × 20 m, floor 1 m, ceiling 10 m, **3 tall obstacles**, **1 free chaser** (≤ 2 m/s, ≤ 2 m/s²).
- Target: x 32–37 m, z 1.6–2.5 m, drifting at 0.2–0.8 m/s; a strike is a distance below 0.6 m.
- Episodes: **15 s**.
- Requirement: **no contact** (obstacle, floor, chaser) and a high strike rate.

## Method in one paragraph

The PS2-RL two-phase pipeline, adapted to obstacles and a pursuer:

- **Phase I** trains a safe-arrival backup (hover LQR + a learned residual, TD3) that reaches a certified hover base set.
- **Safety filter** (discrete backup-set check), applied to every command:
  - it tries blends λ·u_nom + (1 − λ)·u_backup, λ ∈ {1, .75, .5, .25, 0};
  - it rolls out a family of backups for 3 s against pure-pursuit chasers;
  - it applies the largest blend that stays in the safe set and ends in a safe terminal state.
- **Safe set:** obstacles, floor, walls, ceiling, and the guidance-law escape margin h_c ≥ 0.
- **Backup family:**
  - the learned backup blended with an analytic evasion reflex;
  - the hover LQR with the reflex;
  - five escape variants (speed × direction).
- **Committed plan:** it replays its own chaser prediction.
- **Striker:** a correction on a 6 m/s pursuit controller, trained with SAC, then trained **through the filter** (PS2).

Equations and plain-words explanations for every part are in `docs/strike/strike_method.pdf`. Section 12 covers the design analysis.

## Status and headline result

| what | result |
|---|---|
| Reference models (`checkpoints/strike_reduced`), deployed filter, 128 test scenes, 15 s | **strike 128/128, 0 contacts** (median strike 5.2 s, max 13.7 s) |
| Same episodes cut at 10 s | strike 0.938, 0 contacts; the 8 others are slow strikes (10.1–13.7 s) |
| Without the filter (same striker) | strike 0.625, contact 0.375 |
| bCBF (minimal-correction QP) with the same safeguard | strike 0.969, 0 contacts, slower |

The decision experiments behind the final configuration are in `results/strike_reduced/`.

The 3-seed run of the report (`docker_batch/run_strike_final.sh`, 1000 test scenes per seed, 15 s) is in
`results/strike_reduced/final/`:

| configuration | strike % | contact % (count) |
|---|---|---|
| residual striker, no filter | 74.0 ± 0.9 | 26.05 (521 / 2000) |
| residual striker + final filter | 94.4 ± 0.6 | 0.00 (0 / 2000) |
| PS2 striker + one escape (seed 0) | 95.7 | 1.70 (17 / 1000) |
| PS2 striker + final filter | 97.6 ± 0.8 | 0.05 (1 / 2000) |

## Where things are

| path | content |
|---|---|
| `ps2rl/envs/strike_scene.py` | model, safe set, hover base set + certificate, Phase-I observation |
| `ps2rl/envs/strike_env.py`, `strike_pursuit.py`, `strike_filtered_env.py` | arena environment, pursuit controller, environment with the filter in the loop (PS2) |
| `ps2rl/phase1_sa/strike_sa.py` | Phase-I regions, environment, evaluator, training run |
| `ps2rl/cil/strike_filter.py`, `strike_guidance.py` | discrete backup-set filter, guidance margin h_c, evasion reflex, escape variants |
| `ps2rl/cil/strike_bcbf.py` | minimal-correction backup-CBF QP (baseline) |
| `ps2rl/phase2_ps2/strike_tracker_trainer.py` | striker trainer (SAC; residual; PS2 through the filter) |
| `scripts/train_phase1_strike.py`, `compare_phase1_strike.py` | Phase I training and comparison with the hover LQR |
| `scripts/train_phase2_strike.py` | striker / PS2 training |
| `scripts/eval_strike_filter.py` | closed-loop evaluation (tables, per-episode records, GIFs) |
| `scripts/report_strike.py` | report tables (md + LaTeX) and figures from the final pipeline |
| `scripts/strike_paper.py`, `strike_paper_traj.py`, `strike_episode_viz.py`, `strike_filter_snapshot.py` | the report's tables, figures and numbers; trajectory figures of the four compared configurations; episode figures; the filter's check at one step |
| `scripts/compare_strike_episodes.py`, `plot_strike_episodes.py` | the same scenes across methods |
| `docker_batch/run_strike_paper.sh`, `run_strike_episodes.sh` | drivers for the above (`STEPS=tables` needs only the evaluation records) |
| `docker_batch/run_strike_final.sh`, `run_strike_final_eval.sh` | final pipeline driver (3 seeds), evaluation driver |
| `docker_batch/configs/strike_final_{p1,residual,ps2}.json` | final task files; `strike_final_ps2.json` holds the deployed filter (`tracker.filter_guidance`) |
| `docker_batch/configs/strike*.json` (others) | earlier sparse-arena runs reported in the write-up |
| `docker_batch/README_strike.md` | how to run every stage |
| `checkpoints/strike_reduced/` | reference models: Phase-I backup, residual striker, PS2 striker |
| `results/strike_reduced/` | decision experiments (tables, per-episode records); `final/`: the report's run |
| `results/strike_paper/` | Korean methods document (XeLaTeX) and the HWP equations |
| `docs/strike/` | method write-up (LaTeX source + PDF) |

The code also has options that are off by default and not used by any config in this branch. They come from parked studies (a larger arena, a stronger chaser):
- moving-chaser Phase I (`StrikeConfig.chaser_set "guidance"`);
- arena walls in the scene;
- a flee-then-hover backup;
- `learned_alone` filter variant;
- a chaser stronger than the Phase-I proxy (env `actual_chaser_radius`, `actual_chaser_speed`, `chaser_accel`); the
  filter is then built for it;
- a filter blind to the chaser (`eval_strike_filter.py --filter_sees_chasers 0`, the `*_blind` evaluation runs).

## Setup

Python 3.10 with CUDA JAX (`requirements.txt`). In the px4sitl container: `bash docker_batch/setup_train_venv.sh`. The
drivers source `docker_batch/landing_env.sh`, which activates `.venv` and checks the GPU.

## Commands

Reproduce the headline result with the reference models (a few minutes on the GPU; about 50 min on a 2-core CPU with `JAX_PLATFORMS=cpu`):

```bash
python scripts/eval_strike_filter.py --backup_run checkpoints/strike_reduced/phase1_td3_a0p2_seed0 \
  --task docker_batch/configs/strike_final_ps2.json --chaser_model guidance \
  --guidance '{"committed_prediction": true, "escape_variants": [[2.0, 60], [2.0, -60], [3.5, 0], [3.5, 60], [3.5, -60]]}' \
  --nominal striker --striker_run checkpoints/strike_reduced/ps2_striker_s0 --modes "none learned" \
  --episodes 128 --scene_seed 9001 --n_show 0 --gif 2 --gif_pick close_chaser --out outputs/strike_eval_reference
```

Full pipeline (3 seeds: Phase I, residual striker, PS2, 1000-scene evaluation of baselines and ablations, report).
It is resumable; `STAGES="2 3 4"` etc. skips stages that are done:

```bash
docker exec -d px4sitl bash -c "bash ~/ws_shared/PS2-RL/docker_batch/run_strike_final.sh > ~/ws_shared/PS2-RL/outputs/strike_final.out 2>&1"
# report only (CPU is fine)
python scripts/report_strike.py --root outputs/strike_final --seeds 0 1 2
```

Folders it creates under `outputs/strike_final/` (a run is finished when its folder has `summary.json`; evaluations have `strike_filter_episodes.json`):
- `p1/td3_a0p2_seed{0,1,2}`;
- `residual/striker_rs0p1_s{0,1,2}`;
- `ps2/striker_rs0p1_s{0,1,2}`;
- `eval/s{0,1,2}/<run>`;
- `report/`.

## Final configuration (and why)

- **Safe set:** h_c with a_e = 5 m/s², δ = 0.2 s. A static chaser ball gave 49 % contacts.
- **Base set:** hover ellipsoid clear (c_B = 50, certified) with the hover latch.
  - A plan must end either in a latched hover with the reflex off (h_c ≥ 2 m, at least ≈ 1 s of hold), or in an escape with its braking point clear.
- **Backup family:** 7 backups. Direction variants give safety; speed variants add strikes.
- **Committed prediction:** the outcomes are the same, lost-certificate episodes drop from 5.5 % to 3.9 %, and the continuation is an exact replay.
- **Episodes of 15 s:** the 10 s time-ups were slow strikes, not deadlocks.
- **PS2 through the deployed filter**, temperature frozen. In a larger-arena run the auto-tuned temperature diverged.
- **bCBF:** safe with the variant safeguard but slower, so it stays a baseline.
- **Moving-chaser Phase I:** equally safe but not better with this striker, so it is an optional ablation. Its config is not in this branch.

## Open items

1. Finish the seed-1 ablations and the seed-2 evaluation (`run_strike_final_eval.sh` skips the runs already done).
2. Train Phase I with arena walls: the one contact (s0 scene 845) is the drone pinned between an obstacle and a wall.
3. Paper checklist (write-up §12): baselines (add a Lagrangian safe-RL baseline), a robustness sweep (chaser faster
   than the assumed v_c, denser obstacles), filter compute time, and a precise statement of the guarantee.
4. Optional: bCBF hybrid (QP rows of the backup that the discrete check certifies best), finer λ grid, a_e sensitivity.
