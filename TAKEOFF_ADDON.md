# PS2-RL slow vertical-takeoff add-on

Drop-in files for training a PS2-RL policy on a slow vertical takeoff instead of
the powerloop. **No existing file in PS2-RL is modified** — this archive only
adds files. Unzip it over the repo root:

```bash
cd ~/ws_shared/PS2-RL
unzip ~/ps2rl-takeoff-addon.zip -d .
```

Files added:

```
ps2rl/envs/assets/generate_quadrotor_takeoff_reference.py   # reference generator
ps2rl/envs/assets/quadrotor_takeoff_deck_reference.*        # 0.5 -> 3.5 m (presses the deck)
ps2rl/envs/assets/quadrotor_takeoff_mild_reference.*        # 0.5 -> 3.2 m (gentle press)
ps2rl/envs/assets/quadrotor_takeoff_subdeck_reference.*     # 0.5 -> 2.5 m (CIL stays idle)
scripts/check_takeoff_qp_health.py                          # QP health check, no training
```

The three `.npz` bundles are pre-generated, so you can train immediately. They
are also reproducible with the generator (see below).

---

## Why a vertical takeoff

The safe set is a hard-deck ceiling `z <= z_max` (3.0 m), and the powerloop
reference deliberately violates it (`z_top = 3.5`). A slow climb toward 3.5 m is
the same safety test with the attitude dynamics removed: the CIL has exactly one
job, which is to stop the climb at the deck.

It is also far kinder to Gazebo SITL:

| | powerloop | slow takeoff |
|---|---|---|
| max \|omega\| | 10.83 rad/s (620 deg/s) | 0 |
| a_cmd | up to 23.31 m/s^2 (2.38 g) | 9.26 - 10.36 m/s^2 |
| PX4 throttle (1 kg quadratic fit) | ~0.311 | 0.174 - 0.187 |

The reference commands zero body rate, so the pitch rate-loop lag that dominates
the powerloop sim-to-SITL gap does not bind, and `MC_PITCHRATE_MAX` /
`MC_ROLLRATE_MAX` are never approached. The throttle also stays inside the
0.12-0.28 calibration bracket instead of extrapolating past it.

---

## 1. Regenerating the references (optional)

Run from the repo root, inside the Python 3.10 training venv.

```bash
A=ps2rl/envs/assets

# ceiling-pressing (the real test): 408 samples, 8.12 s, peak vz 1.0 m/s
python $A/generate_quadrotor_takeoff_reference.py \
  --z-start 0.5 --z-end 3.5 --climb-speed 1.0 --hold-start 0.5 --hold-end 2.0 \
  --output-bundle $A/quadrotor_takeoff_deck_reference.npz \
  --output-npy    $A/quadrotor_takeoff_deck_reference_legacy.npy \
  --output-figure $A/quadrotor_takeoff_deck_reference.png \
  --output-config-json $A/quadrotor_takeoff_deck_reference_config.json
```

Same command with `--z-end 3.2` (mild) or `--z-end 2.5` (sub-deck).

The profile is a C2 quintic S-curve between two hover holds. Because the
acceleration is purely vertical, the reference attitude is level (q = identity)
and `omega_cmd` is identically zero; the only nontrivial channel is
`a_cmd = g + z_ddot`. Keep the total duration short — episode length sets how
many episodes a step budget buys (408 steps/episode means 1.5M steps is ~3,700
episodes, against ~14,000 for the 106-step powerloop).

## 2. QP health check (before and after training)

```bash
python scripts/check_takeoff_qp_health.py \
  --reference_path ps2rl/envs/assets/quadrotor_takeoff_deck_reference.npz \
  --backup_policy_mode learned \
  --learned_backup_policy_path checkpoints/deployed_sa/quadrotor_sa/best_weights.pkl
```

Expected on the learned backup, float32:

```
  use_solver=False (fallbacks) : 0/412   ok
  q_saturated                  : 0/412   ok
  slack           max          : 2.511e-05   ok
  altitude        max          : 2.8306 m   (deck 3.00, margin +0.1694)
  SAFETY                       : held
```

`use_solver=False` is the flag worth watching and **nothing in the standard
training metrics surfaces it**. When it is False the QP result was discarded and
the backup policy action was used instead: the step is still safe, so
`safe_rate` still reads 1.0, but the CIL replaced your policy rather than
filtering it.

Add `--actor-z-cap inf` for the worst case (an actor that chases the reference
above the deck and never yields — roughly an untrained policy during warm-up),
and `--grid` for a 1521-state sweep near the deck.

## 3. Phase 2 training

Phase 1 does **not** need retraining. The learned safe-arrival backup in
`checkpoints/deployed_sa/quadrotor_sa` was trained against
`z_max=3.0, z_des=2.0, base_set_c=8.0, T=2.0, num_steps=100` — none of which
depend on the reference. Keep those values and it drops straight in.

```bash
A=ps2rl/envs/assets

python scripts/train_phase2.py --system quadrotor \
  --reference_path $A/quadrotor_takeoff_deck_reference.npz \
  --backup_policy_mode learned \
  --learned_backup_policy_path checkpoints/deployed_sa/quadrotor_sa/best_weights.pkl \
  --base_set_c 8.0 --z_max 3.0 --z_des 2.0 --horizon_T 2.0 --num_steps 100 \
  --env_dt 0.02 --seed 1 \
  --w_pos_xy 2.5 --w_pos_z 4.0 --w_vel 1.0 --w_att 4.0 \
  --w_ref_omega_x 0.05 --w_ref_omega_y 0.05 --w_ref_omega_z 0.02 \
  --w_control_a 0.01 --w_control_omega 0.01 \
  --actor_lr 1e-4 --critic_lr 5e-4 --alpha_lr 1e-4 --min_alpha 1e-1 --q_clip_abs 5e6 \
  --total_steps 1500000 --eval_episodes 10 --record_update_metrics \
  --not_terminate_on_violation --save_final_weights \
  --output_root takeoff_ps2 --output_dir learned_seed1
```

Notes on the settings:

- `w_ref_omega_*` are small but nonzero on purpose. Reference omega is
  identically zero here, so they act as rotation regularizers rather than
  tracking terms.
- The `--z-end 3.5` reference is *meant* to be untrackable at the top. Judge the
  run by `safe_rate = 1.0` and a smooth hold near 2.8 m, not by position RMSE.
- Early training is the risky regime. A random actor fights the filter exactly
  like an unlearnable controller does, which is what drives `delta_min_u_ref`
  (and then slack) up. Warm-start from a vanilla tracker
  (`--warm_start --warm_start_weights <ckpt>`) or keep `--start_steps` small,
  and watch slack over the first few thousand steps.

### If you switch to the analytic backup

Add `--pid-min-virtual-accel-z 2.0`.

With the default of `0.0`, the PID ceiling brake
(`pid_z_safety_gain`, `pid_ceiling_vz_gain`) can drive its virtual acceleration
to exactly zero during a slow climb into the deck. The commanded thrust
*direction* is then undefined — `desired_quaternion_from_virtual_accel`
normalizes a zero vector — and `jacfwd` through the backup flow NaNs every
constraint row, so the QP is discarded on ~15% of steps. Measured on the
ceiling-pressing reference, float32:

| `pid_min_virtual_accel_z` | fallbacks | z hold | max slack | proj_norm |
|---|---|---|---|---|
| 0.0 (default) | 62/412 | 2.430 | 5.3e-6 | 13.04 |
| 2.0 | 0/412 | 2.812 | 2.0e-5 | 3.47 |

This flag is **ignored** when `--backup_policy_mode=learned`: the analytic PID is
never constructed on that path (`ps2rl/cil/quadrotor_backup_cbf.py`, the
`mode == "learned"` branch).

### On precision

float64 fixes the analytic-backup NaN too (0/412 fallbacks), which is how the
cause was identified, but it is not the practical fix and is not needed for the
learned backup. The trainer hardcodes float32 in several places — the env build
in `quadrotor_ps2_entry.py`, the replay buffer, the obs cast in
`quadrotor_ps2_trainer.py` — so `JAX_ENABLE_X64=1` alone leaves states and the
QP in float32 anyway. `check_takeoff_qp_health.py --f64` runs the check itself in
float64 if you want to compare.

## 4. SITL deployment caveat

Training is clean, but the bridge is not a drop-in. `auto_start` runs
BOOT -> ARM -> TAKEOFF -> **LINEUP -> DASH** -> POLICY -> RECOVER -> LAND, and
lineup/dash exist to fly the vehicle 10 m out and enter the loop at speed. A
takeoff wants handover at hover, at `z_start`: `takeoff_altitude: 0.5`,
`lineup_distance: 0.0`, and the dash phase neutralized. Check against the actual
state machine in `dual_stage_rl/SITL_src/ps2rl_px4_bridge/bridge_node.py` before
flying.
