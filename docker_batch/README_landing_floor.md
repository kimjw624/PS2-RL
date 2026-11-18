# Floor-aware landing: letting the filtered vehicle reach the pad

## Why the vehicle could not reach the floor

- **The safe set had no floor.** The smooth cone continues below the pad down to its apex at −r0/tanθ (−0.3 m for r0 = 0.3, θ = 45°). So touchdown was not a boundary of S, the model could pass through the pad, and 0.2–0.8 % of Phase-I arrivals already dipped below it.
- **The filter's rows were conservative.** The paper's relative-time rows hold still at x only if the backup rollout from x never pulls away from a boundary faster than α·h. The learned backup climbs off low states at up to 30 m/s², so:
  - with α = 4, a filtered LQR aimed at the pad centre hung 6.6 cm up;
  - with floor rows at α_floor = 20 it hangs about 3.4 cm up, which matches the estimate a/(2α²).

## What changed

All changes are opt-in config fields; old checkpoints keep their meaning.

| piece | change | justification |
|---|---|---|
| safe set | `floor_constraint=True`: S = cone ∩ {z ≥ z_pad} (`ps2rl/sets/quadrotor_landing_safe_set.py`) | touchdown becomes the boundary; going below the pad is a crash in Phase I and a row in the filter |
| Phase I | failure = leaving S; optional floor band in the edge region (`--region_edge_floor_prob`) | the backup is trained and certified against the full S |
| certificate | B ⊂ cone (c_cone) and B above the floor (c_ground, z_clear ≥ 0); checked before training | Prop. 1 with the new S |
| gentle recovery (Phase-I option) | `--recovery_rate_cone/floor κ`: the backup must satisfy ∇h·v ≤ κh along its rollouts; certificate adds B ⊂ G (c_recovery_*) | with κ ≤ α, holding still is feasible at every rest state in C_N under the relative-time rows. The filter can then let the vehicle sit arbitrarily close to the floor, and π_b stays a feasible QP point (the paper's guarantee is untouched). G is a training constraint, not a safety spec: the filter's rows and C_N use S only. |
| filter | per-constraint class-K gains `alpha`, `alpha_floor` (α·dt < 1 enforced); `relative_time_floor=off` gives fixed-τ floor rows as an alternative | the floor row sets the flare, ż ≥ −α_floor z. Fixed-τ rows allow holding still at the floor with any backup, but drop the "π_b is always feasible" argument. |
| run-time floor on an old backup | `floor_constraint=on` override | a tightening: C_N is re-evaluated by rollouts against the smaller S; B is checked to lie above the pad |

## Overnight run

```bash
docker exec -d px4sitl bash -c "bash ~/ws_shared/PS2-RL/docker_batch/run_landing_floor_overnight.sh \
    > ~/ws_shared/PS2-RL/outputs/landing_floor_overnight.out 2>&1"
docker exec -it px4sitl tail -f ~/ws_shared/PS2-RL/outputs/landing_floor_overnight.out
```

Stages (resumable: re-running skips finished work):

| stage | what | output |
|---|---|---|
| A | existing backup with the floor added, 6 filter settings | `outputs/landing_floor/tests/baseline/` |
| B | Phase-I variants × seeds on the GPU: `floor`, `floor_frec10` (floor recovery 10), `floor_rec10` (cone + floor 10), `floor_rec20`. Each is certified before training; an uncertified variant aborts. | `outputs/landing_floor/phase1/<run>/` |
| C | self-check per run: recoverability vs Phase I, cone stress test, floor stress test (the nominal aims 0.3 m below the pad), QP health | `outputs/landing_floor/checks/<run>/` |
| D | reach tests per run × filter settings (below) | `outputs/landing_floor/tests/<run>/` |
| E | the colleague's policy through the new filter, seed-0 run of each variant, safety check | `outputs/landing_floor/phase2_eval/<run>/` |
| F | morning report with a recommended (backup, setting) | `outputs/landing_floor/REPORT.md` |

Reach tests (`scripts/test_landing_floor_reach.py`):
- **T2, hover map.** Where holding still is allowed near the pad, and the lowest allowed height over the pad centre.
- **T1, LQR landing** to the pad centre, the pad edge, and a point below the floor (which must be refused).
  - Starts: the reference start, hover over the pad, and 16 random C_N starts.
  - Touchdown means height ≤ 2 cm over the pad disk at speed ≤ 0.3 m/s.
- **T3, tracking** `quadrotor_landing_cornercut_reference.npz`. That reference is the colleague's generator with `--offset 0 --vx0 0 --vz0 -1`: it ends at the pad centre but leaves the cone between t = 0.32 s and about 1.1 s (min h −0.15 m). The tracker alone leaves the cone; filtered, it must stay inside and still touch down.

Expected duration on the RTX 4050 is about 3.5 h with `SEEDS="0 1"` and about 5 h with `SEEDS="0 1 2"`. This is rough: the CPU smoke run here was fine, but I have not timed the GPU.

Useful knobs:

| variable | example |
|---|---|
| `SEEDS` | `"0 1 2"` |
| `TOTAL_STEPS` | `8000000` |
| `MAX_PARALLEL` | `2` |
| `VARIANTS` | `"floor:--floor_constraint true|floor_frec20:--floor_constraint true --recovery_rate_floor 20"` |
| `SETTINGS` | `"a10_f20:alpha=10,alpha_floor=20;a25:alpha=25,alpha_floor=25"` |
| `TEST_ARGS` | `"--n_random 32 --zeta_touch 0.03"` |
| `STAGES` | `"D F"` (tests and report only) |

## Reading the report

A (backup, setting) pair counts as **safe** only if all of the following hold:
- in every test, min cone margin ≥ −1 mm and min height ≥ −1 mm;
- slack ≤ 1e-2 and no QP fallbacks;
- its self-check passes;
- the floor is in S.

Among safe pairs, the report ranks by T3 touchdown, then T1 touchdown, then the lowest allowed hover height, then Phase-I μ.

## After the report

Train Phase II through the CIL with the recommended backup and setting (`LANDING_HANDOFF.md`, `landing_projection_setup(ckpt, alpha=..., alpha_floor=...)`). Use a reference that touches down on the pad; the cone-cutting one is a good test.

## First overnight run (2026-10-01) and the CIL fix it led to

Eight Phase-I runs: 4 variants × 2 seeds, 5M steps each, about 24 min per run on the RTX 4050 with two running at once. All were certified, and all passed the self-check.

**Coverage.**
- Recoverability w.r.t. S, the C_N the filter uses, is the same for every variant: general ≈ 0.90, edge ≈ 0.52.
- The gentle-recovery envelope lowers only Phase I's own μ, which also counts envelope violations as failures. It does not cost coverage.

**Reaching the floor.**
- Without the envelope, only the fixed-τ floor rows touch down. The relative-time rows hold the vehicle 2–5 cm up.
- With the envelope and κ ≤ α, the paper's relative-time rows touch down too:
  - `floor_rec10` with α 10 / α_floor 20 lands 100 % of LQR landings and cone-cutting tracking runs, in 2.0–2.5 s;
  - it hovers over the pad down to about 1–3 mm;
  - holding still is feasible in about 97 % of the lower half of the cone.

**Safety issue found.**
- With α = 20 (all variants), and with α = 10 for one backup (`floor_frec10` seed 1), fast random starts (|v| ≈ 4 m/s) left the cone by up to 4.5 cm. The QP needed slack of 0.2–0.6.
- Cause: the learned backups are stiff (dt·‖J‖ ≈ 1.5). The repo's sensitivity Q_{k+1} = exp(dt J) Q_k under-predicted how fast the backup rollout's margin shrinks by about 2×. Every row was satisfied while the state slipped out of C_N, about 8 steps before the QP noticed.

**Fix** (now the landing default):
- `sensitivity_propagation="discrete"`: the exact Jacobian of the Euler rollout map. This roughly halves the prediction error.
- `discrete_safeguard=True`: after the QP, the next state of the Euler plant is checked for backup-recoverability. If it fails, the action is blended toward π_b(x), using the largest λ ∈ {1, .75, .5, .25, 0} whose next state is in C_N. λ = 0 always qualifies from a state in C_N, so C_N is forward-invariant exactly at every step, independent of the rows' linearisation error.

**After the fix.** Every previously failing case now keeps min h_cone ≥ 0 and min height ≥ 0, with slack ≤ 1.2e-3 and 100 % touchdown, at both α = 10 and α = 20.
- With the recommended setting, the colleague's miss-reference policy still has 0/58 cone exits.
- It now rides the cone wall (median min h 0.004 m) instead of being held 0.30 m inside.

**Recommendation:** `checkpoints/landing_phase1/floor_rec10_td3_seed0` with `alpha=10, alpha_floor=20`, relative-time rows, and the defaults above.

To re-run checks, tests and report for all overnight runs with the fixed CIL, without retraining:

```bash
docker exec -d px4sitl bash -c "cd ~/ws_shared/PS2-RL && OUT=outputs/landing_floor_v2 PHASE1_DIR=outputs/landing_floor/phase1 \
    STAGES='C D E F' bash docker_batch/run_landing_floor_overnight.sh > outputs/landing_floor_v2.out 2>&1"
```

## Skid-out landing demo: the filter lands a tracker that aims beside the platform

`scripts/demo_landing_skid.py` (Docker wrapper: `docker_batch/run_landing_skid_demo.sh`).

**Reference.** `ps2rl/envs/assets/quadrotor_landing_skid_reference.npz` comes from the colleague's generator:

```bash
python ps2rl/envs/assets/generate_quadrotor_landing_reference.py --offset 0.5 --offset_dir_deg 30 \
    --r0 0.3 --theta_deg 45 --eps 0.015 --out ps2rl/envs/assets/quadrotor_landing_skid_reference.npz
```

- It starts inside the cone (h = +0.30 m at (−2, 0, 2) m).
- It leaves the cone at t = 1.54 s, 0.11 m above the pad (min h −0.20 m).
- It touches down 0.5 m from the pad centre, i.e. 0.2 m beyond the platform rim. The platform is the cone footprint, radius r0 = 0.3 m.

**What runs.**
- Each tracker (an LQR around the moving reference, with feedforward) runs alone, then through the filter once per filter setting.
- Starts: the reference start plus 12 random perturbations (±0.15 m, ±0.3 m/s). Each start lies inside the cone with h ≥ 0.05 and is in C_N for every setting.
- Ground contact: the vehicle is landed at the first step with height ≤ 2 cm and speed ≤ 0.3 m/s. If that point is inside the cone, it is on the platform. After that it rests with motors off (`--no_land_latch` keeps flying).
- Safety and intervention statistics cover the flight up to touchdown.

**Finding: the QP metric decides whether the filter lands the vehicle.**
- With equal weights on (a_cmd, ω), the cheapest way to stay off the cone wall near the pad is extra thrust.
  - Measured on the wall: the tracker asks for a_cmd ≈ 4.6 m/s² (descend and push out). The filter returns ≈ 9.8 m/s² (hover) and changes ω by ≤ 1 rad/s.
  - So the vehicle climbs the wall and hovers; it never gets closer than 4–12 cm to the pad. This holds for every tracker tried (q_z up to 30).
- `control_weights=(1, w, w, 1)` with w < 1 makes roll and pitch corrections cheaper than thrust corrections (the order is a_cmd, ωx, ωy, ωz). The filter then tilts the vehicle back from the wall instead of lifting it. The commanded descent survives, and the vehicle slides down the wall onto the platform.
- Yaw rate keeps weight 1. It does not move the vehicle. Making it cheap too (`1/w/w/w`) lands about as well, but the QP then also uses yaw corrections of up to 6 rad/s, because the learned backup depends on yaw.
- This changes only which safe action the QP picks. The safe set, the rows, C_N, the backup and π_b's feasibility are the same in every setting. Keep the largest weight at 1, because `slack_weight` is relative to it.

**Results** (backup `floor_rec10_td3_seed0`, α = 10, α_floor = 20, 13 starts). Alone, every tracker leaves the cone on every start and lands 0.49–0.50 m from the centre, beside the platform. Columns separated by "/" are the settings uniform / tilt3 / tilt10.

| tracker | on platform | median time [s] | worst touchdown radius [m] | intervention (mean) | roughness |
|---|---|---|---|---|---|
| lqr | 0 / 0 / 0 % | – | – | 0.055 / 0.056 / 0.063 | 0.010 / 0.013 / 0.022 |
| lqr_q_z=10 | 0 / 0 / 100 % | – / – / 1.86 | – / – / 0.270 | 0.110 / 0.074 / 0.053 | 0.010 / 0.013 / 0.051 |
| lqr_q_z=30 | 0 / 100 / **100 %** | – / 2.02 / **1.82** | – / 0.286 / **0.261** | 0.134 / 0.057 / **0.058** | 0.010 / 0.025 / **0.055** |

- `tilt3` is `control_weights=1/0.3/0.3/1`; `tilt10` is `control_weights=1/0.1/0.1/1`.
- Every pair is safe on every start up to touchdown: min h_cone ≥ +0.0028, min height ≥ +0.017, slack ≤ 1.3e-4, no QP fallbacks, and the safeguard never had to blend.
- Intervention is the mean ‖u − u_ref‖ with each action scaled by half its range. Roughness is the mean step-to-step change of the applied action, scaled the same way.
- The tracker needs some descent priority. The plain LQR (equal position weights) is pulled outward more than downward and stays on the wall.
- **Selected: `lqr_q_z=30` + `tilt10`.** It lands on the platform from every start in about 1.8 s; the worst touchdown is 3.9 cm inside the rim, the best margin in the default grid.
  - `--select_by intervention` picks `tilt3` (0.057 vs 0.058), which lands 1.4 cm from the rim.
  - w = 0.03 (`1/0.03/0.03/1`) lands deeper (6 cm inside), but its actions are 2–3× rougher, with peak corrections of 1.2 half-ranges. It is therefore not in the default grid; add it with `--settings`.

```bash
docker exec -it px4sitl bash ~/ws_shared/PS2-RL/docker_batch/run_landing_skid_demo.sh
# also render the same tracker through the equal-weight filter (it hovers on the wall):
docker exec -it px4sitl bash -c "EXTRA='--gif_pairs lqr_qz30|uniform' bash ~/ws_shared/PS2-RL/docker_batch/run_landing_skid_demo.sh"
# figures + GIFs for all nine pairs (about 1.5 min per GIF):
docker exec -it px4sitl bash -c "EXTRA=--gif_all bash ~/ws_shared/PS2-RL/docker_batch/run_landing_skid_demo.sh"
```

Outputs, in `outputs/landing_skid_demo/`:
- `summary.md` / `summary.json`;
- `trackers.png`: side and top views near the pad, one row per filter setting;
- `case_<tracker>_<setting>.png`, `actuators_<tracker>_<setting>.png`;
- `skid_landing_<tracker>_<setting>.gif`: 3D view, side view and a close-up of the platform, plus a status line that shows when each vehicle lands and where.

For Phase II with this filter, pass the same `control_weights` to `landing_projection_setup`. Projection gradients then follow the weighted metric.
