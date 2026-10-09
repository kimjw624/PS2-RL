# Landing: the Phase-I safe-arrival backup and the landing filter

This is what Phase II needs to train (or filter) a landing policy through the control-invariant
layer (CIL) with the **approach-cone safe set** and the **learned Phase-I safe-arrival backup**:

- a trained, certified backup: `checkpoints/landing_phase1/floor_rec10_td3_seed0/`;
- a landing backup-CBF layer with the same interface as the powerloop one
  (`ps2rl/cil/quadrotor_landing_backup_cbf.py`), built on the repo's shared QP engine;
- a Phase-II binding (`ps2rl/phase2_ps2/landing_ps2_binding.py`);
- a self-test (`scripts/check_landing_bcbf.py`) and reach tests (`scripts/test_landing_floor_reach.py`);
- the full Phase-I code, in case the geometry changes and Phase I has to be re-run.

The shipped backup is for a 45° cone over a pad of radius r0 = 0.3 m at the origin, with the pad
plane in the safe set (S = cone ∩ {z ≥ z_pad}). It was trained with the gentle-recovery envelope
(κ = 10 on cone and floor), which is what lets a filtered vehicle come down to the pad. Use it with

```python
landing_projection_setup(CKPT, alpha=10.0, alpha_floor=20.0)
```

Results for this backup: `results/landing_phase1/floor_rec10_td3_seed0/`.
Background on the floor and the envelope: `docker_batch/README_landing_floor.md`.

---

## 1. Verify (a few minutes on CPU)

```bash
JAX_PLATFORMS=cpu python scripts/check_landing_bcbf.py \
    --backup checkpoints/landing_phase1/floor_rec10_td3_seed0 \
    --cbf_override alpha=10 --cbf_override alpha_floor=20
```

Expected output:
```
[check 1] backup recoverability (fresh samples): general 0.758 (Phase I 0.785), edge 0.318 (Phase I 0.335), shell 1.000 (Phase I 1.000)  [+-0.060 at 95%]
[check 2b] nominal targets 0.30 m below the pad: unfiltered min height -4.190 m | filtered +0.0004 m, max slack 4.1e-04
[check 2] nominal controller targets 1.0 m outside the wall, 32 recoverable starts, 150 steps
  unfiltered: 100% leave the cone | filtered: 0.0% leave, min h_cone +0.0025 m
[check 3] max slack 6.04e-04 (steps with slack>1e-3: 0.00%), solver fallbacks 0 (0.0% of steps), filter active on 93% of steps, ...
[PASS] wrote outputs/check_landing_bcbf/check_landing_bcbf.json and ...png
```
- **Check 1** confirms the loaded backup is the trained one.
- **Check 2** drives a controller at a point 1 m outside the cone and confirms that the CIL keeps it inside.
- **Check 2b** drives a controller at a point 0.3 m below the pad and confirms that the CIL keeps it above the pad.
- **Check 3** confirms the QP is healthy.

The script exits non-zero on failure.

## 2. Use it in Phase II

### Minimal API
```python
from ps2rl.cil import quadrotor_landing_backup_cbf as lbcbf

CKPT = "checkpoints/landing_phase1/floor_rec10_td3_seed0"
# cone, pad, floor, base set, backup, dt: all from CKPT; the two gains are the filter setting
cbf_cfg = lbcbf.landing_bcbf_config_from_checkpoint(CKPT, alpha=10.0, alpha_floor=20.0)
projector = lbcbf.QuadrotorLandingBackupCBFProjector(cbf_cfg)
runtime = projector.runtime

# every environment step, batched: x (B, 10) physical state, u_raw (B, 4) policy action
u_safe, slack, used_solver, info = lbcbf.solve_backup_cbf_qp_batch_with_info(x, u_raw, cbf_cfg, runtime=runtime)
```
The QP solve is differentiable (qpax), so training *through* the layer works exactly as in
the powerloop Phase II.

### With the repo's Phase-II trainer (`ps2rl/phase2_ps2/quadrotor_ps2_trainer.py`)
The landing module exports the same names as `quadrotor_backup_cbf`, so the swap is:

| powerloop | landing |
|---|---|
| `from ps2rl.cil.quadrotor_backup_cbf import (QuadrotorBackupCBFProjector, backup_policy_batch, solve_backup_cbf_qp_batch, solve_backup_cbf_qp_batch_with_info)` | same names from `ps2rl.cil.quadrotor_landing_backup_cbf` (projector: `QuadrotorLandingBackupCBFProjector`) |
| `_BINDING = PS2SystemBinding(...)` | `from ps2rl.phase2_ps2.landing_ps2_binding import LANDING_BINDING` |
| `cbf_cfg = QuadrotorBCBFConfig(...)` (in `quadrotor_ps2_entry.py`) | `cbf_cfg = lbcbf.landing_bcbf_config_from_checkpoint(CKPT, alpha=10.0, alpha_floor=20.0)` |
| `projector = QuadrotorBackupCBFProjector(cbf_cfg)` | `projector = lbcbf.QuadrotorLandingBackupCBFProjector(cbf_cfg)` |
| ceiling checks (`hard_deck_value`, `z_max`) in the env and metrics | `lbcbf.safe_value(x, cbf_cfg)` (>= 0 is safe); `cone_value` and `floor_value` give the two parts |

Or in one call: `cbf_cfg, projector, proj_ops = landing_projection_setup(CKPT, alpha=10.0, alpha_floor=20.0)`
from `ps2rl.phase2_ps2.landing_ps2_binding`.

The landing config exposes every field the quadrotor Phase-II code reads (`a_cmd_min`,
`a_cmd_max`, `omega_max`, `action_scale`, `dt`, `num_steps`, `horizon`, `base_set_c`,
`backup_policy_mode`, `learned_backup_policy_path`, `num_qp_inequalities`, `solver_tol`,
`base_alpha`, `use_analytic_jacobian`).

### Environment requirements
- **Dynamics:** step the plant with `lbcbf.landing_step(x, u, cbf_cfg)`, or an identical
  explicit-Euler step at `cbf_cfg.dt` with quaternion re-normalisation. That is the model the
  backup was trained on and the certificate assumes.
- **Initial states:** keep only recoverable ones:
  `ok, _, _ = lbcbf.make_recoverability_fn(cbf_cfg)(x0_batch)`. Outside the recoverable set
  the QP may need slack, and there is no guarantee.
- **Pad frame:** if your pad is not at the checkpoint's pad position (origin), shift states with
  `lbcbf.to_pad_frame(x, pad_world, cbf_cfg)` before calling the layer.
- **Logging:** log `slack` and `used_solver` every step (see section 4).

## 3. Interface contract (must match)

| item | value |
|---|---|
| state `x` | `(p_x, p_y, p_z, v_x, v_y, v_z, q_w, q_x, q_y, q_z)`; world frame, z up; `q` body-to-world, scalar first |
| action `u` | `(a, omega_x, omega_y, omega_z)`: mass-normalised collective thrust [m/s^2], body rates [rad/s] |
| action box | from the checkpoint: `a in [0, 4g]`, `|omega_i| <= 18` rad/s |
| time step, integrator | from the checkpoint: `dt = 0.02` s, explicit Euler + quaternion normalisation; backup horizon `N = 100` (2 s) |
| backup actor input | the raw 10-D physical state in the pad frame, not normalised (the same `x` as above) |
| safe set | `h_cone(x) = r0 + tan(theta) (z - z_pad) - sqrt(|p_xy - p_pad|^2 + eps^2) >= 0` and `z - z_pad >= 0`; the geometry is in the checkpoint |

## 4. Filter settings and numerics

- **Gains.** `alpha` (cone rows) and `alpha_floor` (floor rows) are the class-K gains; `alpha·dt < 1`
  is enforced. The floor row sets the flare, ż ≥ −α_floor·z. The backup's envelope (κ = 10) makes
  holding still feasible anywhere in C_N when κ ≤ α, so use `alpha >= 10`; the tests here were run
  with `alpha=10, alpha_floor=20`. Other QP settings (`base_alpha=2`, `slack_weight=1e6`,
  `solver_tol=5e-4`) are read from `QuadrotorBCBFConfig`, not copied.
- **Discrete-time defaults.** `sensitivity_propagation="discrete"` uses the exact Jacobian of the
  Euler rollout, and `discrete_safeguard=True` checks the next state for backup-recoverability
  after the QP and, if needed, blends the action toward the backup's
  (λ ∈ {1, .75, .5, .25, 0}; λ = 0 always qualifies from a state in C_N). With the safeguard,
  C_N is forward-invariant at every step of the Euler plant.
- **QP precision.** `qp_solve_dtype` is `float32` by default. With `JAX_ENABLE_X64=1` and
  `qp_solve_dtype="float64"` only the 5-variable QP is solved in double precision; the rows stay
  in float32. The self-test uses float64 and has no solver fallbacks. In float32 it passes too, but the solver
  falls back on 4.5 % of the stress-test steps.
  - x64 also makes *default* dtypes float64 wherever arrays or network parameters are created
    without an explicit dtype. Check that the trainer creates its networks and buffers as float32.
  - Either way, monitor the fallback rate (`1 - used_solver`): on a fallback the engine applies the
    backup action, which is safe but gives the policy **no gradient** for that sample.
- **Slack.** `slack > 1e-3` means the backup constraints were relaxed at that step, and
  safety is not guaranteed there. It stays below 1e-3 in the self-test and the reach tests.

## 5. What you must not change without re-running Phase I

The base-set certificate and the backup policy are valid only for the checkpoint's cone and
pad geometry, `dt`, action box, dynamics model, hover height `z_des = 1.25` m, level
`c_B = 12` and LQR weights. `make_landing_backup_runtime` **refuses to run** if the config
disagrees with the checkpoint on any of these.

To change any of them:
```bash
# 1. certify the base set for the new geometry (prints the z_des sweet spot and c_B)
JAX_PLATFORMS=cpu python scripts/certify_landing_base_set.py --cone_theta_deg <deg> --cone_r0 <m> --cone_eps <m> --out outputs/cert.json
# 2. retrain the safe-arrival policy (GPU); in the px4sitl container see docker_batch/README_landing_floor.md
python scripts/train_phase1_landing.py --backbone td3 --seed 0 --cone_theta_deg <deg> --cone_r0 <m> --cone_eps <m> \
    --floor_constraint true --region_edge_floor_prob 0.3 --recovery_rate_cone 10 --recovery_rate_floor 10
# 3. point CKPT at outputs/landing_phase1/<run_name> and re-run scripts/check_landing_bcbf.py and scripts/test_landing_floor_reach.py
```

## 6. What the guarantee covers, and known limitations

- **Guarantee:** forward invariance of the backup-induced set C_N in the **simulation model**
  (Euler, instantaneous body-rate tracking), provided the start state is recoverable and the
  slack is zero.
- **Coverage near the wall is limited.** About half of the low-altitude edge region of the design
  region is not recoverable (section 7). Starts there carry no guarantee.
- **Attitude dynamics are not modelled.** The backup may command large tilts and rates before
  hand-off to the LQR, and the PX4 rate-loop lag is not in the model.
- **The certificate is numerical.** `c_Lyap` comes from an adversarial search, not a proof;
  the chosen level has a 0.77 margin.
- **The envelope is a training constraint, not a safety specification.** The filter's rows and
  C_N use S only.

## 7. Phase-I facts (`floor_rec10_td3_seed0`, TD3, seed 0, 5M steps)

Fraction of held-out initial states from which the backup reaches the base set within the horizon:

| region | without leaving S (what the filter's C_N uses) | also inside the gentle-recovery envelope (Phase I's own test metric) |
|---|---|---|
| general (altitude-uniform over the cone) | 0.898 | 0.785 |
| low-altitude cone edge | 0.516 | 0.335 |
| capture shell | 1.000 | 1.000 |
| weighted (1/3/0.5) | – | 0.509 |

Certificate bounds for c_B = 12: c_U 17.88, c_chart 23.18, c_cone 31.83, c_ground 18.71,
c_recovery_cone 17.58, c_recovery_floor 39.67; adversarial one-step Lyapunov ratio at c_B 0.967.

## 8. Files

```
ps2rl/cil/quadrotor_landing_backup_cbf.py      landing CIL (config from checkpoint, runtime, projector, helpers)   NEW
ps2rl/phase2_ps2/landing_ps2_binding.py        PS2SystemBinding for the landing trainer                            NEW
ps2rl/cil/backup_cbf.py                        + per-constraint rows, discrete sensitivities, qp_solve_dtype       MODIFIED
ps2rl/utils/policy.py                          + ActorConfig.activation (required to load the backup)              MODIFIED
ps2rl/utils/field_overrides.py                 KEY=VALUE overrides for the scripts                                 NEW
ps2rl/base_controller/quadrotor_landing_dlqr.py  9-D hover LQR above the pad                                       NEW
ps2rl/sets/quadrotor_cone_sets.py, quadrotor_landing_safe_set.py   smooth approach cone; cone and pad plane        NEW
ps2rl/sets/landing_certificate.py              base-set level bounds                                               NEW
ps2rl/envs/quadrotor_landing_config.py         landing parameters (single source of truth for Phase I)             NEW
ps2rl/phase1_sa/landing_design_region.py, quadrotor_landing_sa_env.py, quadrotor_landing_sa_trainer.py   Phase I NEW
ps2rl/phase1_sa/sa_trainer_core.py             + SAC backbone option                                               MODIFIED
ps2rl/evaluation/landing_filter_reach.py       helpers of the reach tests                                          NEW
scripts/check_landing_bcbf.py                  self-test (section 1)                                               NEW
scripts/test_landing_floor_reach.py, scripts/summarize_landing_floor.py   reach tests and their report             NEW
scripts/certify_landing_base_set.py, scripts/train_phase1_landing.py   Phase-I entry points                        NEW
docker_batch/                                  GPU tooling for Phase I in the px4sitl container                    NEW
checkpoints/landing_phase1/floor_rec10_td3_seed0/   the backup actor + its configs, summary, certificate and self-check   NEW
results/landing_phase1/floor_rec10_td3_seed0/  results for that backup                                             NEW
```
