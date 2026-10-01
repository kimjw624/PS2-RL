# Phase-II landing training: tracker and PS2 for each landing reference

## References

The default set is six candidates from `scripts/landing_reference_candidates.py`: `B2_-30`, `B2_-20`, `B2_-10`, `B3_-30`, `B3_-20`, `B3_-10`. The candidate grids are in `outputs/landing_ref_candidates/`; regenerate them with `python scripts/landing_reference_candidates.py`.

- **Common to all six:** each hovers at 2 m inside the cone (45°, r0 = 0.3 m) and descends for 4 s to a soft touchdown beyond the platform rim. The descent is near-vertical first; the horizontal motion follows s(t)², so it drifts toward the pad axis late. Peak speed is about 1.1–1.3 m/s and tilt ≤ 5°.
- **The ID:** the digit is where it lands (2 = 20 cm, 3 = 30 cm beyond the rim). The suffix is the angle of the start→touchdown line from vertical (−30, −20, −10°, toward the axis).
- **Saved as:** `ps2rl/envs/assets/quadrotor_landing_<id>_reference.{npz,png,_config.json}`, with `<id>` = `b2_m30`, …

| id | start h [m] | leaves the cone at z [m] | outside [s] | min h [m] |
|---|---|---|---|---|
| b2_m30 | +0.65 | 1.13 | 2.1 | −0.22 |
| b2_m20 | +1.07 | 0.52 | 1.6 | −0.20 |
| b2_m10 | +1.45 | 0.29 | 1.3 | −0.20 |
| b3_m30 | +0.55 | 1.32 | 2.3 | −0.32 |
| b3_m20 | +0.97 | 0.72 | 1.8 | −0.30 |
| b3_m10 | +1.35 | 0.43 | 1.5 | −0.30 |

The earlier `skid`, `nearside` and `cornercut` references are still in `ps2rl/envs/assets`. To train on them, pass `REFS="name:path ..."`.

## What runs (`docker_batch/run_landing_phase2_train.sh`)

| stage | what | output |
|---|---|---|
| B | **tracker** per reference: SAC without the filter, 8M steps. These are the colleague's landing-tracker settings (`outputs/landing/land_v2_*`): landing reward v2, 32 envs, start perturbation ±0.3 m / ±0.5 m/s / ±10°. | `outputs/landing_phase2/<ref>_tracker/` |
| C | **PS2** per reference: SAC through the landing CIL, warm-started from the tracker. Uses the paper's PS2 settings: no random warm-up, actor lr 5e-5, projected critic targets. Stops at 1.5M steps or after `PS2_HOURS` (3 h), whichever comes first. | `outputs/landing_phase2/<ref>_ps2/` |
| D | **evaluation** from the same 32 starts (the reference start plus reset-library starts): tracker alone, tracker + filter at deployment, PS2 | `outputs/landing_phase2/eval_<ref>/` |
| E | report | `outputs/landing_phase2/REPORT.md` |

- **Filter:** backup `floor_rec10_td3_seed0`, α = 10, α_floor = 20, `control_weights=1/0.1/0.1/1`, discrete sensitivities and the discrete-time safeguard. The QP is solved in float32. In float64 the networks would become float64 too, which is slow on consumer GPUs. Measured on these references, the float32 QP falls back to the backup action on 0.2–3 % of steps; that is safe, but those samples carry no gradient.
- **Starts:** each run keeps the start perturbations that lie inside the safe set (margin 2 cm) and are backup-recoverable. These go into `init_states.npz`, and every reset draws from it. PS2 reuses the tracker's library, so both policies see the same starts.
- **Environment:** the repo's quadrotor env with `reward_mode="landing_v2"`, which is the colleague's normalised Charbonnier reward. I checked it against his env: identical rewards on 64 random transitions. It also uses `safety_mode="landing"`, so the logged `hard_deck_*` margins are the cone/floor margins.

## Commands

```bash
docker exec -d px4sitl bash -c "bash ~/ws_shared/PS2-RL/docker_batch/run_landing_phase2_train.sh \
    > ~/ws_shared/PS2-RL/outputs/landing_phase2.out 2>&1"
docker exec -it px4sitl bash -c "tail -F ~/ws_shared/PS2-RL/outputs/landing_phase2.out"
```

The log streams every run's lines with a `[run]` prefix. Each progress line shows steps/s, elapsed time and the ETA. The batch is resumable: re-running it skips finished runs and evaluations.

**Duration.**
- Six references, with three runs on the GPU at once.
- **Trackers:** 8M steps each. The colleague's took about 21 min alone on his GPU; expect roughly 1–1.5 h for both waves.
- **PS2:** much slower per step, because every env step and every update solves the filter QP with a 100-step backup rollout per sample. I measured about 17 steps/s on a 2-core CPU but could not time the RTX 4050. `PS2_HOURS=3` caps each run, so two waves take about 6 h.
- **Total:** about 7–7.5 h.
- **Adjusting:** check the first PS2 progress lines (steps/s, ETA). Raise `PS2_STEPS` if the GPU is fast. Shorten with `PS2_HOURS=2` or fewer references (`REFS=...`). If three parallel runs are slower in total than two (watch steps/s), set `MAX_PARALLEL=2`.

Knobs:

| variable | default |
|---|---|
| `REFS` | the six `b*` references |
| `TRACKER_STEPS` | 8000000 |
| `PS2_STEPS` | 1500000 |
| `PS2_HOURS` | 3 |
| `MAX_PARALLEL` | 3 |
| `FILTER` | `alpha=10,alpha_floor=20,control_weights=1/0.1/0.1/1` |
| `SEED` | 0 |
| `STAGES` | `"B C D E"` (e.g. `"D E"` to re-evaluate) |
| `TRACKER_ARGS`, `PS2_ARGS`, `EVAL_ARGS` | extra flags for `scripts/train_phase2_landing.py` / `scripts/evaluate_landing_phase2_runs.py` |

## Afterwards: what to send back

```bash
cd ~/ws_shared/PS2-RL/outputs && zip -r landing_phase2_results.zip landing_phase2 landing_phase2.out \
    -x "landing_phase2/*/final_weights.pkl"
```

If the zip is over 30 MB, also add `-x "*.gif"` and send the `eval_*/landing_*.gif` files separately (in a second zip if needed).

## Single runs

```bash
python scripts/train_phase2_landing.py --reference <npz> --projection off --run_name <name>          # tracker
python scripts/train_phase2_landing.py --reference <npz> --projection on --run_name <name> \
    --warm_start_weights outputs/landing_phase2/<tracker>/best_weights.pkl \
    --init_library outputs/landing_phase2/<tracker>/init_states.npz [--wall_time_budget_hours 4]      # PS2
python scripts/evaluate_landing_phase2_runs.py --tracker_run <dir> --ps2_run <dir> --out <dir>
```

Every SAC hyperparameter can be passed as `--<name>`. The defaults are the `tracker` / `ps2` presets in `ps2rl/phase2_ps2/landing_ps2_entry.py`.
