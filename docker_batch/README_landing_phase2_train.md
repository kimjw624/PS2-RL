# Phase-II landing training: tracker and PS2 for two references

## References

Both come from the colleague's generator (`ps2rl/envs/assets/generate_quadrotor_landing_reference.py`, 45° cone, r0 = 0.3 m) and start inside the cone at (−2, 0, 2) m with h = +0.30 m.

| name | file | generator flags | what it does |
|---|---|---|---|
| `skid` | `quadrotor_landing_skid_reference.npz` | `--offset 0.5 --offset_dir_deg 30` | crosses the cone, skids out at t = 1.54 s (0.11 m up), lands 0.5 m from the centre on the far side |
| `nearside` | `quadrotor_landing_nearside_reference.npz` | `--offset 0.45 --offset_dir_deg 180 --vx0 -0.5` | the hand-drawn one: drops almost straight down, leaves through the near wall at t = 0.66 s (1.66 m up), stays outside it (min h −0.15 m) and lands 0.15 m beyond the near edge of the platform |

## What runs (`docker_batch/run_landing_phase2_train.sh`)

| stage | what | output |
|---|---|---|
| B | **tracker** per reference: SAC without the filter, 8M steps. These are the colleague's landing-tracker settings (`outputs/landing/land_v2_*`): landing reward v2, 32 envs, start perturbation ±0.3 m / ±0.5 m/s / ±10°. | `outputs/landing_phase2/<ref>_tracker/` |
| C | **PS2** per reference: SAC through the landing CIL, warm-started from the tracker. Uses the paper's PS2 settings: no random warm-up, actor lr 5e-5, projected critic targets. Stops at 1.5M steps or after `PS2_HOURS`, whichever comes first. | `outputs/landing_phase2/<ref>_ps2/` |
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
- The trackers took the colleague about 21 min for 8M steps on his GPU.
- PS2 is much slower per step: every env step and every update solves the filter QP, with a 100-step backup rollout per sample. I measured about 17 steps/s on this 2-core CPU but could not time the RTX 4050.
- `PS2_HOURS=4` therefore caps each PS2 run. Two run in parallel, so the whole batch takes at most about 5 h.
- Check the first PS2 progress lines (steps/s, ETA). If the GPU is fast, raise `PS2_STEPS`; if it is slow, the budget still ends the run cleanly, with a final evaluation and checkpoints.

Knobs:

| variable | default |
|---|---|
| `REFS` | `"skid:… nearside:…"` |
| `TRACKER_STEPS` | 8000000 |
| `PS2_STEPS` | 1500000 |
| `PS2_HOURS` | 4 |
| `MAX_PARALLEL` | 2 |
| `FILTER` | `alpha=10,alpha_floor=20,control_weights=1/0.1/0.1/1` |
| `SEED` | 0 |
| `STAGES` | `"B C D E"` (e.g. `"D E"` to re-evaluate) |
| `TRACKER_ARGS`, `PS2_ARGS`, `EVAL_ARGS` | extra flags for `scripts/train_phase2_landing.py` / `scripts/evaluate_landing_phase2_runs.py` |

## Afterwards: what to send back

```bash
cd ~/ws_shared/PS2-RL/outputs && zip -r landing_phase2_results.zip landing_phase2 landing_phase2.out \
    -x "landing_phase2/*/final_weights.pkl"
```

If the zip is over 30 MB, also add `-x "*.gif"` and send the two `eval_*/landing_*.gif` files separately.

## Single runs

```bash
python scripts/train_phase2_landing.py --reference <npz> --projection off --run_name <name>          # tracker
python scripts/train_phase2_landing.py --reference <npz> --projection on --run_name <name> \
    --warm_start_weights outputs/landing_phase2/<tracker>/best_weights.pkl \
    --init_library outputs/landing_phase2/<tracker>/init_states.npz [--wall_time_budget_hours 4]      # PS2
python scripts/evaluate_landing_phase2_runs.py --tracker_run <dir> --ps2_run <dir> --out <dir>
```

Every SAC hyperparameter can be passed as `--<name>`. The defaults are the `tracker` / `ps2` presets in `ps2rl/phase2_ps2/landing_ps2_entry.py`.
