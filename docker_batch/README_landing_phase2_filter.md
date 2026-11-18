# Landing Phase II: filtering a trained landing policy with the BCBF

This checks whether the landing control-invariant layer keeps a Phase-II landing policy inside the approach cone. Here the policy was trained **without** projection to follow a reference that deliberately misses the pad. The CIL uses the Phase-I safe-arrival backup.

Everything runs **inside the `px4sitl` container**, from `~/ws_shared/PS2-RL`, with the GPU venv from `README_landing_phase1.md` (`bash docker_batch/setup_train_venv.sh`).

## One command

```bash
# first time: import the colleague's archive (reference + run folders), then everything else
IMPORT=~/ws_shared/PS2-RL.tar bash docker_batch/run_landing_phase2_filter_eval.sh

# later runs (already imported); several Phase-II runs at once
RUNS="outputs/landing/land_v2_miss_s1 outputs/landing/land_v2_miss_s2 outputs/landing/land_v2_miss_s3" \
    bash docker_batch/run_landing_phase2_filter_eval.sh

# skip Phase-I training by using the shipped backup (trained for this reference's cone)
BACKUP=checkpoints/landing_phase1/land_v2_miss_td3_seed0 bash docker_batch/run_landing_phase2_filter_eval.sh
```

Detached, from the host:

```bash
docker exec -d px4sitl bash -c "bash ~/ws_shared/PS2-RL/docker_batch/run_landing_phase2_filter_eval.sh \
    > ~/ws_shared/PS2-RL/outputs/landing_phase2_filter.out 2>&1"
docker exec -it px4sitl tail -f ~/ws_shared/PS2-RL/outputs/landing_phase2_filter.out
```

## What the script does

1. **Import** (`import_phase2_landing_runs.sh`, only with `IMPORT=`). It copies the landing reference(s), their `_config.json`, and the generator into `ps2rl/envs/assets/`, and the run folders into `outputs/landing/<run>/` (gitignored).
   - Their `ps2rl/envs/quadrotor_env.py` is not copied. It only changes the training reward, and copying it would drop this fork's disturbance options.

2. **Derive the Phase-I config** (`scripts/landing_phase1_config_from_phase2.py`):

   | Phase-I setting | source |
   |---|---|
   | cone r0/θ/ε, pad | the reference's `_config.json`, i.e. the cone the reference was designed against |
   | dt, g, action box | the run's `configs.json` env block |
   | design region | widened so it covers the reference envelope plus the run's initial perturbation (speed margin = `init_v_range`) |
   | z_des, c_B | kept at 1.25 m / 12. The ground constraint binds for this cone too, so both stay certified. The script moves z_des up if a cone ever makes it uncertified. |

   The config is written to `outputs/landing_phase1_cfg/<reference>_<hash>.json`. Runs that share a cone, plant and region reuse one backup.

3. **Train the backup** (`train_phase1_landing.py --config_json ...`). It is skipped if `outputs/landing_phase1_p2/<name>_td3_seed0/` already exists. It takes about 15–20 min for 5M steps on the RTX 4050, and the Phase-I certificate check runs before training.

4. **Evaluate** (`scripts/evaluate_landing_phase2_filter.py`):
   - Two variants from the same starts: policy only, and policy + filter.
   - Episode 0 starts exactly on the reference; the others use the run's own reset distribution.
   - The filter runs through the trainer's deployment path (`build_ps2_action_fns_for` + `LANDING_BINDING`), with the QP in float64.
   - Every start is classified with a backup rollout: in C_N or not. The guarantee only covers starts in C_N, so the tables split them.

The evaluation refuses to run if the backup and the Phase-II task disagree on the cone, pad, dt, g or action box. `--allow_geometry_mismatch` overrides that, for experiments only.

5. **Reachability analysis** (`scripts/analyze_landing_filter_reach.py`), written to `<eval dir>/reach/`:
   - **Part A, C_N slices.** On the vertical plane through the pad axis, for three states: hovering, straight descent at the reference's peak sink rate, and the reference's own velocity/attitude.
   - **Part B, filter only.** The checkpoint's hover LQR, re-centred on targets on the pad plane (pad centre, r0/2, r0, and the reference touchdown), run with and without the filter.
   - **Part D, sensitivity.** Part B's landing tests repeated for other BCBF settings: `--variants "alpha=10;alpha=20;include_relative_time_term=false;backup_policy_mode=base_controller"`.
   - **Part C, attribution.** On the evaluated episodes: the vertical acceleration the policy requests vs what is applied, and which QP row binds.

To compare evaluations run with different filter settings:

```bash
for V in "alpha=10" "alpha=20" "include_relative_time_term=false"; do
    python scripts/evaluate_landing_phase2_filter.py --phase2_run outputs/landing/land_v2_miss_s1 \
        --backup checkpoints/landing_phase1/land_v2_miss_td3_seed0 --cbf_override "$V" --gif_cases nominal
done
python scripts/compare_landing_filter_evals.py \
    --dirs outputs/landing_phase2_filter/land_v2_miss_s1__land_v2_miss_td3_seed0* \
    --out outputs/landing_phase2_filter/compare_land_v2_miss_s1
```

## Knobs

| variable | default | |
|---|---|---|
| `IMPORT` | *(empty)* | PS2-RL.tar or an extracted folder |
| `RUNS` | `outputs/landing/land_v2_miss_s1` | space-separated Phase-II run dirs |
| `CHECKPOINT` | `best` | `best` or `final` weights of the Phase-II run |
| `BACKBONE` / `SEED` / `TOTAL_STEPS` | `td3` / `0` / `5000000` | Phase-I training |
| `P1_EXTRA_ARGS` | *(empty)* | e.g. `"--region_v_max 3.0"` |
| `BACKUP` | *(empty)* | existing Phase-I checkpoint; skips steps 2–3 |
| `EPISODES` | `64` | |
| `EVAL_EXTRA_ARGS` | *(empty)* | e.g. `"--gif_cases nominal,worst_unfiltered,most_filter_effort"`, `"--backup_policy_mode base_controller"` (LQR-only backup baseline), `"--alpha 2"` |

## Outputs: `outputs/landing_phase2_filter/<run>__<backup>/`

| file | content |
|---|---|
| `summary.md` | the table (starts in C_N / not in C_N): cone exits, min h, final pad distance, tracking error, intervention rate, slack, QP fallbacks |
| `summary.json` | everything above plus provenance (files, cone, QP settings, env init ranges) |
| `overview.png` | (a) side view of all episodes with the cone; (b) h_cone(t); (c) tracking error; (d) filter activity and slack |
| `case_<name>.png` | one episode: side and top view, h_cone(t), height and distance from the pad axis |
| `actuators_<name>.png` | thrust and body rates: policy only, policy proposal u_ref and applied u_safe of the filtered run; then ‖Δu‖ and slack |
| `distributions.png` | per-episode min h, final pad distance, tracking error |
| `trajectory_<name>.gif` | 3D view with the cone surface and pad, plus a side view. Both vehicles fly simultaneously (red: policy only, blue: filtered), with live h, tilt and filter status. |
| `episodes.npz` | trajectories, actions, h(t) and per-episode metrics |

Cases:
- `nominal`: episode 0.
- `worst_unfiltered`: the start in C_N where the policy alone goes furthest outside.
- `most_filter_effort`: the start in C_N with the largest mean ‖Δu‖.
- `not_in_CN`: a start outside C_N, if any.

## What the first run showed

Setup: `land_v2_miss_s1`, 64 starts, with the shipped backup trained for this reference's cone (r0 0.3 m, θ 45°, test μ_w = 0.731).

- **Safety.** 58 of the 64 starts are in C_N; they are exactly the starts inside the cone.
  - Policy alone: 100 % of those episodes leave the cone (min h −0.59 m).
  - With the filter: 0 % leave, slack ≤ 2e-4, no QP fallbacks. This holds for every filter setting tried.
- **No fly-away with a trained backup.** The filtered policy settles in a hover about 0.37 m above the pad plane, 0.33 m off the axis.
  - The policy keeps asking for about −8 m/s² of vertical acceleration (a dive toward its touchdown point, which is outside the cone). The filter applies ≈ 0.
  - The fly-away seen with a badly trained backup (a 60k-step test) happens because no start is in C_N there. The QP then trades slack for large, erratic actions, and the cone is open upward.
- **Reaching the pad is possible.** At rest, all of the cone's interior is in C_N, including every point over the pad.
  - With the default gain α = 4, a filtered LQR aiming at the pad centre stops 6.6 cm above it.
  - With α = 10 or 20, with the fixed-τ rows, or with the LQR-only backup, it lands exactly.
- **What holds the vehicle up.** It is the relative-time rows with α = 4, not C_N. At the hover where the policy stops, the binding row is the cone row 0.1 s ahead on the backup rollout.
  - The backup raises h there at 1.70 m/s, and α·h = 1.70 m/s. So hovering is barely allowed and descending is not.
  - Under fixed-τ rows, the policy's own action would pass unchanged at that state.
- **Trade-off.**

  | setting | median min h | ends from pad centre | tracking error | max slack |
  |---|---|---|---|---|
  | α = 4 | 0.30 m | 0.33 m | 0.34 m | – |
  | α = 10 | 0.08 m | 0.58 m | 0.26 m | – |
  | α = 20 | 0.02 m | 0.67 m | 0.23 m | 6e-4 |
  | fixed-τ rows | 0.002 m | 0.86 m | – | 0.04 |

  The fixed-τ rows ride the cone wall, but slack grows to 0.04: those rows no longer make the backup action a guaranteed feasible point.

## Reading the result

- **Safety.** Among starts in C_N, *episodes leaving the cone* must be 0 % for the filtered variant, with slack at solver tolerance and no QP fallbacks. That is the claim being tested.
- **Cost of safety.** The policy never saw the filter in training, so the filter moves it off its training distribution. Tracking error and final pad distance show how well the filtered policy still does its job. Training Phase II **through** the CIL (see `LANDING_HANDOFF.md`) is what removes this cost.
- **Starts outside C_N.** Episodes starting outside C_N include starts that are already outside the cone. The run's reset box of ±0.3 m around (−2, 0, 2) touches the cone boundary. These starts carry no guarantee; the QP uses slack there.
