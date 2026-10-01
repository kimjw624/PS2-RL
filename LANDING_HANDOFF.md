# Landing: using the Phase I safe-arrival backup in Phase II

> **Update (2026-10-01): use the floor-aware backup for landing.** For the 45° cone
> (r0 0.3 m, the colleague's reference) use `checkpoints/landing_phase1/floor_rec10_td3_seed0/`:
> - safe set is cone n floor;
> - gentle-recovery backup, κ = 10;
> - certified and self-checked.
>
> Use it with `landing_projection_setup(CKPT, alpha=10.0, alpha_floor=20.0)`. The CIL now
> defaults to exact discrete sensitivities and a discrete-time safeguard
> (`sensitivity_propagation="discrete"`, `discrete_safeguard=True`). The safeguard keeps C_N
> forward-invariant step by step. With this setting a filtered controller lands on the pad
> without leaving the cone. See `docker_batch/README_landing_floor.md`.
> `td3_seed0` below is the earlier 30° cone checkpoint.

This package gives Phase II everything it needs to train (or filter) a landing policy
through the control-invariant layer (CIL) with the **approach-cone safe set** and the
**learned Phase I safe-arrival backup**:

- a trained, certified backup (`checkpoints/landing_phase1/td3_seed0/`);
- a landing backup-CBF layer with the same interface as the powerloop one
  (`ps2rl/cil/quadrotor_landing_backup_cbf.py`), built on the repo's shared QP engine;
- a Phase II binding (`ps2rl/phase2_ps2/landing_ps2_binding.py`);
- a one-minute self-test (`scripts/check_landing_bcbf.py`);
- the full Phase I code, in case the geometry changes and Phase I has to be re-run.

Background (method, numbers, trajectory and actuator plots): `landing_phase1_report.pdf`.

---

## 1. Install

The package is based on `kimjw624/PS2-RL@main` (`9d6b990`). Pick one of the two options.

**A. Git (recommended).** From the root of your PS2-RL checkout:
```bash
git checkout -b landing-phase2
git am -3 /path/to/landing_phase2_handoff/patches/*.patch
```

**B. Copy files.** Copy the contents of `PS2-RL/` over your checkout:
```bash
cp -r /path/to/landing_phase2_handoff/PS2-RL/. /path/to/your/PS2-RL/
```
Five upstream files are **modified**; the rest are new. If your fork has changed these five
files, merge by hand. All five changes are additive, and the old behaviour is the default:

| file | change |
|---|---|
| `ps2rl/utils/policy.py` | `ActorConfig.activation` (default `"relu"`). **Required**: the backup actor uses ELU, and without this field it loads but computes wrong actions. |
| `ps2rl/cil/backup_cbf.py` | optional `BCBFSystem.qp_solve_dtype` (default `None` = unchanged) |
| `ps2rl/phase1_sa/sa_trainer_core.py` | optional SAC actor backbone for Phase I (TD3 path bit-identical) |
| `README.md`, `.gitignore` | docs; publishes `checkpoints/landing_phase1/` |

Dependencies are unchanged: `requirements.txt` (`jax==0.6.2`, `qpax==0.0.9`).

## 2. Verify (about 1 minute on CPU)

```bash
JAX_PLATFORMS=cpu python scripts/check_landing_bcbf.py \
    --backup checkpoints/landing_phase1/td3_seed0/landing_backup_policy_actor.pkl
```

Expected output (the numbers come from fresh samples, so small differences are normal):
```
[check 1] backup recoverability (fresh samples): general 0.967 (Phase I 0.960), edge 0.859 (Phase I 0.872), shell 1.000 (Phase I 1.000)
[check 2] ... unfiltered: 100% leave the cone | filtered: 0.0% leave, min h_cone +0.0079 m
[check 3] max slack 1.4e-03 ..., solver fallbacks 0 (0.0% of steps), ...
[PASS] wrote outputs/check_landing_bcbf/check_landing_bcbf.json and ...png
```
- **Check 1** confirms the loaded backup is the trained one.
- **Check 2** drives a controller at a point 1 m outside the cone and confirms that the CIL
  keeps it inside.
- **Check 3** confirms the QP is healthy.

The script exits non-zero on failure.

## 3. Use it in Phase II

### Minimal API
```python
from ps2rl.cil import quadrotor_landing_backup_cbf as lbcbf

CKPT = "checkpoints/landing_phase1/td3_seed0/landing_backup_policy_actor.pkl"
cbf_cfg = lbcbf.landing_bcbf_config_from_checkpoint(CKPT)      # cone, pad, base set, backup, dt: all from CKPT
projector = lbcbf.QuadrotorLandingBackupCBFProjector(cbf_cfg)
runtime = projector.runtime

# every environment step, batched: x (B, 10) physical state, u_raw (B, 4) policy action
u_safe, slack, used_solver, info = lbcbf.solve_backup_cbf_qp_batch_with_info(x, u_raw, cbf_cfg, runtime=runtime)
```
The QP solve is differentiable (qpax), so training *through* the layer works exactly as in
the powerloop Phase II.

### With the repo's Phase II trainer (`ps2rl/phase2_ps2/quadrotor_ps2_trainer.py`)
The landing module exports the same names as `quadrotor_backup_cbf`, so the swap is:

| powerloop | landing |
|---|---|
| `from ps2rl.cil.quadrotor_backup_cbf import (QuadrotorBackupCBFProjector, backup_policy_batch, solve_backup_cbf_qp_batch, solve_backup_cbf_qp_batch_with_info)` | same names from `ps2rl.cil.quadrotor_landing_backup_cbf` (projector: `QuadrotorLandingBackupCBFProjector`) |
| `_BINDING = PS2SystemBinding(...)` | `from ps2rl.phase2_ps2.landing_ps2_binding import LANDING_BINDING` |
| `cbf_cfg = QuadrotorBCBFConfig(...)` (in `quadrotor_ps2_entry.py`) | `cbf_cfg = lbcbf.landing_bcbf_config_from_checkpoint(CKPT)` |
| `projector = QuadrotorBackupCBFProjector(cbf_cfg)` | `projector = lbcbf.QuadrotorLandingBackupCBFProjector(cbf_cfg)` |
| ceiling checks (`hard_deck_value`, `z_max`) in the env and metrics | `lbcbf.cone_value(x, cbf_cfg)` (>= 0 is safe) |

Or in one call: `cbf_cfg, projector, proj_ops = landing_projection_setup(CKPT)` from
`ps2rl.phase2_ps2.landing_ps2_binding`.

The landing config exposes every field the quadrotor Phase II code reads (`a_cmd_min`,
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
- **Logging:** log `slack` and `used_solver` every step (see section 5).

## 4. Interface contract (must match)

| item | value |
|---|---|
| state `x` | `(p_x, p_y, p_z, v_x, v_y, v_z, q_w, q_x, q_y, q_z)`; world frame, z up; `q` body-to-world, scalar first |
| action `u` | `(a, omega_x, omega_y, omega_z)`: mass-normalised collective thrust [m/s^2], body rates [rad/s] |
| action box | from the checkpoint: `a in [0, 4g]`, `|omega_i| <= 18` rad/s |
| time step, integrator | from the checkpoint: `dt = 0.02` s, explicit Euler + quaternion normalisation; backup horizon `N = 100` (2 s) |
| backup actor input | the raw 10-D physical state in the pad frame, not normalised (the same `x` as above) |
| safe set | `h_cone(x) = r0 + tan(theta) (z - z_pad) - sqrt(|p_xy - p_pad|^2 + eps^2) >= 0`; the geometry is in the checkpoint |

## 5. Numerics you should know about

- **QP settings** default to the powerloop Phase II values (`alpha=4`, `base_alpha=2`,
  `slack_weight=1e6`, `solver_tol=5e-4`). They are read from `QuadrotorBCBFConfig`, not copied.
  Override them with `landing_bcbf_config_from_checkpoint(CKPT, alpha=..., ...)`.
- **float32 solver fallbacks.** In float32, qpax returns a non-finite solution on some landing
  QPs: about 10% of steps in the adversarial stress test, which is fewer than normal training
  would see. The engine then applies the backup action. That is still safe, but it is not
  minimally invasive and gives the policy **no gradient** for that sample.
  - **Fix:** `JAX_ENABLE_X64=1` plus
    `landing_bcbf_config_from_checkpoint(CKPT, qp_solve_dtype="float64")`. Only the
    5-variable QP is then solved in double precision; the rows stay in float32. Fallbacks
    dropped from 467 to 0 in the stress test at no measurable CPU cost.
  - **Caveat:** x64 also makes *default* dtypes float64 wherever arrays or network parameters
    are created without an explicit dtype. Check that the trainer creates its networks and
    buffers as float32, or accept the slowdown on GPU.
  - Either way, monitor the fallback rate (`1 - used_solver`).
- **Slack.** `slack > 1e-3` means the backup constraints were relaxed at that step, and
  safety is not guaranteed there. It stays near 1e-3 or below in the stress test.

## 6. What you must not change without re-running Phase I

The base-set certificate and the backup policy are valid only for the checkpoint's cone and
pad geometry, `dt`, action box, dynamics model, hover height `z_des = 1.25` m, level
`c_B = 12` and LQR weights. `make_landing_backup_runtime` **refuses to run** if the config
disagrees with the checkpoint on any of these.

To change any of them:
```bash
# 1. certify the base set for the new geometry (prints the z_des sweet spot and c_B)
JAX_PLATFORMS=cpu python scripts/certify_landing_base_set.py --cone_theta_deg <deg> --cone_r0 <m> --out outputs/cert.json
# 2. retrain the safe-arrival policy (GPU); in the px4sitl container see docker_batch/README_landing_phase1.md
python scripts/train_phase1_landing.py --backbone td3 --seed 0 --cone_theta_deg <deg> --cone_r0 <m> [--z_des ... --base_set_c ...]
# 3. point CKPT at outputs/landing_phase1/<run>/landing_backup_policy_actor.pkl and re-run scripts/check_landing_bcbf.py
```

## 7. What the guarantee covers, and known limitations

- **Guarantee:** forward invariance of the backup-induced set in the **simulation model**
  (Euler, instantaneous body-rate tracking), provided the start state is recoverable and the
  slack is zero.
- **No ground plane.** The cone continues below the pad. About 0.2–0.8% of backup
  recoveries dip up to about 3 cm below the pad plane. A ground row (`z >= z_pad`) is the
  planned next step, for both Phase I and the CIL.
- **Aggressive attitudes.** Before hand-off to the LQR, the backup reaches a median peak tilt
  of about 40°, with a 95th percentile of 68–81°. The PX4 rate-loop lag is not modelled.
- **Cone geometry is provisional** (`r0 = 0.5` m, `theta = 30°`) until the landing reference
  fixes it.
- **The certificate is numerical.** `c_Lyap` comes from an adversarial search, not a proof;
  the chosen level has a 0.77 margin.

## 8. Phase I facts (test split, `td3_seed0`)

| region | recoverability | base controller alone | viability ceiling |
|---|---|---|---|
| general (altitude-uniform over the cone) | 0.960 | 0.890 | 0.968 |
| low-altitude cone edge | 0.872 | 0.681 | 0.880 |
| capture shell | 1.000 | 1.000 | 1.000 |
| weighted (1/3/0.5) | 0.906 | 0.763 | 0.913 |

Every failure is a crash from a state where no admissible input can stay inside the cone;
there are no timeouts.

## 9. Files

```
PS2-RL/
  ps2rl/cil/quadrotor_landing_backup_cbf.py      landing CIL (config from checkpoint, runtime, projector, helpers)   NEW
  ps2rl/phase2_ps2/landing_ps2_binding.py        PS2SystemBinding for the landing trainer                            NEW
  ps2rl/cil/backup_cbf.py                        + optional qp_solve_dtype                                           MODIFIED
  ps2rl/utils/policy.py                          + ActorConfig.activation (required to load the backup)              MODIFIED
  ps2rl/base_controller/quadrotor_landing_dlqr.py  9-D hover LQR above the pad                                       NEW
  ps2rl/sets/quadrotor_cone_sets.py              smooth approach cone                                                NEW
  ps2rl/sets/landing_certificate.py              base-set level bounds                                               NEW
  ps2rl/envs/quadrotor_landing_config.py         landing parameters (single source of truth for Phase I)             NEW
  ps2rl/phase1_sa/landing_design_region.py, quadrotor_landing_sa_env.py, quadrotor_landing_sa_trainer.py   Phase I NEW
  ps2rl/phase1_sa/sa_trainer_core.py             + SAC backbone option                                               MODIFIED
  scripts/check_landing_bcbf.py                  self-test (section 2)                                               NEW
  scripts/certify_landing_base_set.py, scripts/train_phase1_landing.py   Phase I entry points                        NEW
  docker_batch/                                  GPU tooling for Phase I in the px4sitl container, and the report generator   NEW
  checkpoints/landing_phase1/td3_seed0/          the backup actor + its configs, summary and certificate             NEW
  checkpoints/landing_phase1/base_set_certificate.json   full certificate report (z_des sweep, containment, invariance)   NEW
patches/                                         the same changes as a git patch series
landing_phase1_report.pdf                        formal Phase I report
```
