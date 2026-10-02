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
| B | **trackers**: SAC without the filter, one run per reference and per seed in `TRACKER_SEEDS` | `outputs/landing_phase2/<ref>_tracker_s<seed>/` |
| S | **selection**: every checkpoint (best and final of every seed) is flown alone from the same 64 starts; the one with the smallest mean position error becomes the reference's tracker | `outputs/landing_phase2/<ref>_tracker/` (+ `selection.json`) |
| C | **PS2**: SAC through the landing CIL, warm-started from the selected tracker. It stops at `PS2_STEPS` or after `PS2_HOURS`, whichever comes first | `outputs/landing_phase2/<ref>_ps2/` |
| D | **evaluation** from the same 32 starts (the reference start plus reset-library starts): tracker alone, tracker + filter at deployment, PS2 | `outputs/landing_phase2/eval_<ref>/` |
| E | report, including the tracker selection table | `outputs/landing_phase2/REPORT.md` |

- **Filter:** backup `floor_rec10_td3_seed0`, α = 10, α_floor = 20, `control_weights=1/0.1/0.1/1`, discrete sensitivities and the discrete-time safeguard. The QP is solved in float32; where it fails, the layer falls back to the backup action (safe, no gradient).
- **Starts:** each run keeps the start perturbations that lie inside the safe set (margin 2 cm) and are backup-recoverable. These go into `init_states.npz`, and every reset draws from it. PS2 reuses the tracker's library, so both policies see the same starts.
- **Environment:** the repo's quadrotor env with `reward_mode="landing_v2"` (the colleague's normalised Charbonnier cost, floored at −50 per step) and `safety_mode="landing"`, so the logged `hard_deck_*` margins are the cone/floor margins. Logged returns are always in this reward's units, whatever the critics learn.

## Commands

```bash
docker exec -d px4sitl bash -c "bash ~/ws_shared/PS2-RL/docker_batch/run_landing_phase2_train.sh \
    > ~/ws_shared/PS2-RL/outputs/landing_phase2.out 2>&1"
docker exec -it px4sitl bash -c "tail -F ~/ws_shared/PS2-RL/outputs/landing_phase2.out"
```

Run a subset of stages with `-e STAGES="B S"` (e.g. `docker exec -d -e STAGES="B S" px4sitl bash -c "..."`). The batch is resumable: re-running it skips finished runs and evaluations.

**Results of the first batch are kept, not overwritten.** They are moved aside as `<name>_old_<timestamp>`: the single-seed `<ref>_tracker` runs (old recipe; they take no part in the selection, so every PS2 run starts from the same kind of tracker), a PS2 run older than its tracker's selection, and an evaluation older than its PS2 run.

## Watching the curves and stopping a run

While the batch runs it keeps `outputs/landing_phase2/training.html` up to date (every `CURVES_EVERY` = 30 s; the page reloads itself). `outputs` is on the shared volume, so open it on the host:

```bash
xdg-open ~/px4-ros-docker/ws_shared/PS2-RL/outputs/landing_phase2/training.html
```

- **Figures:** one per mode (`training_tracker.png`, `training_ps2.png`), a column per reference and a line per run. Top row: mean position error over an episode; bottom row: −return. Thick lines are evaluations (deterministic policy, full-length episodes), thin lines are training episodes.
- **Table:** step, steps/s, time left, last and best evaluation error of every run, and the command that stops it.
- **Stopping:** `touch outputs/landing_phase2/<run>/STOP` (host or container). The run finishes at its next chunk exactly as it would at the end: final evaluation, `final_weights.pkl`, `best_weights.pkl`, `summary.json`. The batch then starts the next run. The best weights are the evaluation with the highest return so far, so stopping late costs nothing.
- **Without the batch:** `python scripts/plot_landing_training.py --root outputs/landing_phase2 --watch 30`.

## Tracker recipe

The first batch used the settings of the colleague's landing trackers (`--preset tracker_v1`). Of those six runs, five lost their progress at least once and one never recovered. The logs show the same event each time: within 20–30 thousand env steps the critics' value of the policy's own action jumps from about −300 to the clip at +10000, although every reward is negative, and the policy follows it. Three things in the recipe make that possible and slow to undo:

1. **Scale.** The reward reaches −50 per step, so returns are in the thousands. Adam changes a critic's output at a rate set by the learning rate, not by the error, so the critics lag for millions of steps and need hundreds of thousands more to come back from an overshoot.
2. **Unbounded targets.** The bootstrapped target may exceed any return the reward can produce, so an overestimate feeds itself.
3. **Clipped predictions.** Past ±10000 the prediction's gradient is zero: a critic that gets there stays there, and the actor gets no gradient either.

The `tracker` preset keeps the reward, network and data rates and changes this:

| setting | `tracker_v1` | `tracker` |
|---|---|---|
| what the critics learn | r, in [−50, 0] | exp(r / 3), in (0, 1] (`--critic_reward exp`) |
| critic targets | clipped at ±10000 | bounded by the reward's range / (1 − γ) |
| critic predictions | clipped at ±10000 | not clipped |
| temperature α | auto-tuned, floor 0.1 | fixed at the same ratio to the reward range |
| learning rates, batch | 1e-4, 64 | 3e-4, 256 |
| episode end | time only | also when the position error exceeds 1.5 m (`--terminate_pos_error`) |

- **exp(r / 3):** an error-free step is worth 1 and a cost of 3 is worth 1/e, so most of the critics' range resolves small tracking errors. With r itself, most of the range separates crashes from flying.
- **Termination:** a policy that cannot fly yet stays near the reference instead of drifting tens of metres away, where the reward is saturated. A terminated episode is reported with the reward floor for the steps not flown, and evaluations always fly full episodes, so returns stay comparable.
- **`--critic_reward normalized`** (1 + r/50) is the milder alternative. `--preset tracker_v1` reproduces the old runs.

Evaluation position error (mean over the episode, deterministic policy), old recipe on the GPU against the CPU tests of the new one:

| recipe | run | steps trained | first ≤ 0.6 m | first ≤ 0.2 m | at the end |
|---|---|---|---|---|---|
| `tracker_v1` | b2_m10, seed 0 | 8M | 4.20M | never | 39 m (collapsed at 4.66M) |
| `tracker_v1` | b3_m20, seed 0 (the best of the six) | 8M | 3.75M | 5.80M | 0.10 m |
| normalized, lr 1e-4, batch 64 | b2_m10, seed 0 | 2M | 1.65M | – | 0.54 m |
| normalized, lr 3e-4, batch 256 | b2_m10, seed 0 | 1.5M | 1.05M | – | 0.72 m |
| `tracker` (exp, lr 3e-4, batch 256) | b2_m10, seed 0 | 1.5M | 0.35M | 0.85M | 0.09 m |
| `tracker` (exp, lr 3e-4, batch 256) | b2_m20, seed 1 | 1.2M | 0.60M | 0.90M | 0.13 m |

"At the end" is the median of the last three evaluations. So the new preset needs about 1.5M steps for what the old one's best run reached in 8M. These are two runs of at most 1.5M steps; the old collapses came at 3–5M steps, so the new recipe is not yet tested that far. `TRACKER_STEPS` defaults to 3M; the curves page shows whether that is enough.

Flown alone from 32 starts, the `b2_m10` test tracker follows the reference to 0.08 m, leaves the cone on every start and touches down beside the platform. Through the filter it never leaves the safe set (smallest margin +0.007 m) and ends hovering on the cone wall about 15 cm up, without touching down. Landing from there is what PS2 has to learn.

## PS2 settings

- **Preset `ps2`:** the paper's PS2 settings (warm start, no random warm-up, actor lr 5e-5), with 256 envs and an update of 512 samples every 64 env steps. `--preset ps2_paper` is the paper's 32-env setting.
- **Same critic reward as the tracker.** The critics are warm-started, so the run reads the transform from the tracker's `configs.json`.
- **Cached QP rows.** A state's QP rows (100-step backup rollout and its sensitivities) depend on the state only. They are computed once when the state is visited and stored with the transition; the actor update re-solves only the small QP. This is exact.
- **No projection of the critic-target actions** (the paper projects them).
- **Gradient guard.** When one sample's float32 QP fails, its backward pass returns NaN even though the sample falls back to the backup action. Before the fix that NaN turned the whole batch's actor gradient into zero; with 512 samples and a 0.6 % failure rate, about 95 % of the actor updates did nothing. The failed sample's gradient is now zero and the rest of the batch counts.
- **KKT solve.** qpax factorises a 5×5 matrix by QR at every interior-point iteration; `ps2rl/cil/qpax_batched.py` replaces it by an elementwise Cholesky. Each run times both at start-up and uses the faster (`[qp] ...` lines in its log).
- **Device (`PS2_DEVICE=auto`).** The projection is thousands of tiny QPs, which a GPU does not necessarily run faster than the CPU. Stage C first runs `MAX_PARALLEL` short PS2 runs on the GPU, then on the CPU, and trains on the faster; the log says which and by how much.

The first GPU batch ran PS2 at 22–25 steps/s per run. On a 2-core CPU the new path runs at about 115 steps/s. I have not been able to time it on the GPU.

## Duration

- **Trackers:** the first batch ran at 8000–9700 steps/s per run, three at a time, so 3M steps take about 6 min per run and the 18 runs about 35–40 min. Stop a run once its curve is flat; `touch outputs/landing_phase2/*_tracker_s*/STOP` stops the three that are running.
- **Selection:** a few minutes.
- **PS2:** at most `PS2_HOURS` per wave of `MAX_PARALLEL` runs, so two waves and about 6 h with the defaults, plus about 10 min for the device timing.

## Knobs

| variable | default |
|---|---|
| `REFS` | the six `b*` references |
| `TRACKER_SEEDS` | `"0 1 2"` |
| `TRACKER_STEPS` | 3000000 |
| `PS2_STEPS` | 1500000 |
| `PS2_HOURS` | 3 |
| `PS2_DEVICE` | `auto` (`gpu`, `cpu`) |
| `PS2_PROBE_STEPS` | 8192 |
| `MAX_PARALLEL` | 3 |
| `CURVES_EVERY` | 30 (seconds; 0 = off) |
| `FILTER` | `alpha=10,alpha_floor=20,control_weights=1/0.1/0.1/1` |
| `SEED` | 0 (PS2) |
| `STAGES` | `"B S C D E"` |
| `TRACKER_ARGS`, `PS2_ARGS`, `EVAL_ARGS` | extra flags for `scripts/train_phase2_landing.py` / `scripts/evaluate_landing_phase2_runs.py` |

## Afterwards: what to send back

```bash
cd ~/px4-ros-docker/ws_shared/PS2-RL/outputs && zip -r landing_phase2_results.zip landing_phase2 landing_phase2.out \
    -x "landing_phase2/*/final_weights.pkl" -x "landing_phase2/*_old_*/*" -x "landing_phase2/probe/*"
```

If the zip is over 30 MB, also add `-x "*.gif"` and send the `eval_*/landing_*.gif` files separately.

## Single runs

```bash
python scripts/train_phase2_landing.py --reference <npz> --projection off --run_name <name>          # tracker
python scripts/select_landing_tracker.py --candidates <run dir> ... --out <dir>                       # pick one
python scripts/train_phase2_landing.py --reference <npz> --projection on --run_name <name> \
    --warm_start_weights <tracker dir>/best_weights.pkl \
    --init_library <tracker dir>/init_states.npz [--wall_time_budget_hours 3]                         # PS2
python scripts/evaluate_landing_phase2_runs.py --tracker_run <dir> --ps2_run <dir> --out <dir>
```

Every SAC hyperparameter can be passed as `--<name>`. The defaults are the presets in `ps2rl/phase2_ps2/landing_ps2_entry.py`.
