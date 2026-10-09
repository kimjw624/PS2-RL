#!/usr/bin/env python
"""Phase II for landing under a disturbance (UE-bCBF), through the repo's Phase-II trainer.

Same pattern as ``scripts/train_phase2_landing.py`` (nominal landing): the trainer
(``quadrotor_ps2_entry`` -> ``quadrotor_ps2_trainer`` -> ``ps2_trainer_core``) is reused and
these pieces are swapped in:

* CIL:      ``quadrotor_landing_ue_bcbf`` built from the UE Phase-I checkpoint
            (QuadrotorBCBFConfig / QuadrotorBackupCBFProjector in entry and trainer);
* binding:  ``make_landing_ue_binding`` (phys_dim 13 = x + d_hat; residual u_ref = u_nom + a);
* env:      ``build_quadrotor_landing_ue_env`` (sinusoidal disturbance, observer, d_hat and
            u_nom in the observation, floor in the safety check, UE-recoverable starts);
* checks:   the learned-backup check accepts the 13-D (x, d_hat) backup; in residual mode the
            actor's mean head starts at zero (= the nominal tracker).

Every other flag goes to ``quadrotor_ps2_entry`` (``--help`` there). Its powerloop CIL flags
(``--base_set_c``, ``--backup_policy_mode``, ...) are filled in and ignored: geometry, backup
and disturbance bounds come from the checkpoint.

    # the shipped run's settings, on the repo trainer
    JAX_ENABLE_X64=1 python scripts/train_phase2_landing_ue.py --qp_float64 --seed 0 --total_steps 5000000 \
        --update_every 16 --batch_size 256 --actor_lr 3e-4 --critic_lr 3e-4 --alpha_lr 3e-4 --min_alpha 1e-2 \
        --q_clip_abs 1e4 --start_steps 10000 --update_after 5000 --eval_every 250000 --eval_episodes 16 \
        --save_final_weights --run_tag land_ue_p2_res_s0

    # pure policy (no nominal tracker; initial mean thrust = g unless --no_hover_bias)
    ... --no_residual

Without a GPU this path is ~10x slower than ``train_phase2_landing_ue_cached.py`` (the
repo core rebuilds the CIL rows for every update sample).
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

DEFAULT_CKPT = "outputs/landing_phase1_ue/ue_floor_rec10_td3_h128_seed0"
DEFAULT_REF = "ps2rl/envs/assets/quadrotor_landing_cornercut_reference.npz"


def _split_args(argv):
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--landing_ue_ckpt", default=DEFAULT_CKPT, help="UE Phase-I checkpoint (dir or actor .pkl)")
    ap.add_argument("--qp_float64", action="store_true", help="solve the QP in float64 (needs JAX_ENABLE_X64=1)")
    ap.add_argument("--alpha_floor", type=float, default=20.0, help="class-K gain of the floor rows")
    ap.add_argument("--no_residual", action="store_true", help="pure policy instead of u_ref = u_nom + u_res")
    ap.add_argument("--no_hover_bias", action="store_true",
                    help="pure policy: keep the initial mean thrust at the box centre (2 g) instead of g")
    ap.add_argument("--res_thrust", type=float, default=9.81)
    ap.add_argument("--res_rate", type=float, default=8.0)
    # landing reward / tracker (see ps2rl.envs.quadrotor_landing_ue_env.LandingUEEnvConfig)
    for name in ("s_z", "w_z", "s_vz", "w_vz", "s_xy", "w_xy", "s_vxy", "w_vxy", "s_tilt", "w_tilt", "s_yaw",
                 "w_yaw", "s_wz", "w_wz", "s_wxy", "w_wxy", "w_thrust", "delta", "floor",
                 "trk_kp", "trk_kd", "trk_k_att", "trk_k_yaw"):
        ap.add_argument(f"--{name}", type=float, default=None)
    ap.add_argument("--bank_size", type=int, default=None)
    return ap.parse_known_args(argv)


def main(argv=None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    largs, rest = _split_args(argv)
    if largs.qp_float64 and os.environ.get("JAX_ENABLE_X64", "") not in ("1", "true", "True"):
        raise SystemExit("--qp_float64 needs JAX_ENABLE_X64=1 in the environment (set it before python starts)")

    import jax
    import jax.numpy as jnp
    import numpy as np

    from ps2rl.backup_policy.quadrotor_learned_backup import load_learned_quadrotor_backup_policy
    from ps2rl.cil import quadrotor_landing_ue_bcbf as uecbf
    from ps2rl.envs.quadrotor_landing_ue_env import LandingUEEnvConfig, build_quadrotor_landing_ue_env
    from ps2rl.phase2_ps2 import quadrotor_ps2_entry as entry
    from ps2rl.phase2_ps2 import quadrotor_ps2_trainer as trainer
    from ps2rl.phase2_ps2.landing_ue_ps2_binding import UE_PHYS_DIM, make_landing_ue_binding
    from ps2rl.utils.optim import adam_init

    ckpt = str(Path(largs.landing_ue_ckpt) if Path(largs.landing_ue_ckpt).is_absolute() else ROOT / largs.landing_ue_ckpt)
    residual = not largs.no_residual
    env_kw = {f.name: getattr(largs, f.name) for f in dataclasses.fields(LandingUEEnvConfig)
              if getattr(largs, f.name, None) is not None}
    ue_env_cfg = LandingUEEnvConfig(nominal_controller="tracker" if residual else "none", **env_kw)
    holder: dict = {}

    def ue_cbf_config(**kw):
        """Stands in for QuadrotorBCBFConfig(...) in the entry: keep the QP tuning, take the rest from CKPT."""
        over = {k: kw[k] for k in ("alpha", "base_alpha", "slack_weight", "solver_tol") if k in kw}
        over["alpha_floor"] = float(largs.alpha_floor)
        if largs.qp_float64:
            over["qp_solve_dtype"] = "float64"
        cfg = uecbf.ue_bcbf_config_from_checkpoint(ckpt, **over)
        holder["cbf_cfg"] = cfg
        return cfg

    def ue_env(env_cfg, dtype=jnp.float32):
        if "cbf_cfg" not in holder:
            raise RuntimeError("the UE CIL config must be built before the environment")
        if "env" not in holder:  # the trainer rebuilds the env for every evaluation; build once
            holder["env"] = build_quadrotor_landing_ue_env(env_cfg, holder["cbf_cfg"], ue_env_cfg, dtype=dtype)
        return holder["env"]

    def check_ue_backup(cbf_cfg, **_kw):
        learned = load_learned_quadrotor_backup_policy(cbf_cfg.learned_backup_policy_path)
        if int(learned.actor_cfg.obs_dim) != UE_PHYS_DIM:
            raise ValueError(f"UE backup must take (x, d_hat) (obs_dim {UE_PHYS_DIM}), got {learned.actor_cfg.obs_dim}")

    orig_init_state = trainer._init_state

    def init_state_residual(key, actor_cfg, critic_cfg, sac_cfg):
        state = orig_init_state(key, actor_cfg, critic_cfg, sac_cfg)
        if sac_cfg.warm_start:
            return state
        last = state["actor_params"]["layers"][-1]
        if residual:  # start exactly at the nominal tracker
            state["actor_params"]["layers"][-1] = {"w": last["w"].at[:, :4].set(0.0), "b": last["b"].at[:4].set(0.0)}
        elif not largs.no_hover_bias:  # pure policy: initial mean thrust = g (the tanh box is centred at 2 g)
            cfg = holder["cbf_cfg"]
            mid = 0.5 * (cfg.a_cmd_min + cfg.a_cmd_max)
            half = 0.5 * (cfg.a_cmd_max - cfg.a_cmd_min)
            state["actor_params"]["layers"][-1] = {"w": last["w"], "b": last["b"].at[0].set(float(np.arctanh((cfg.gravity - mid) / half)))}
        state["actor_opt"] = adam_init(state["actor_params"])
        return state

    orig_logger = entry._build_metric_logger

    def logger_with_ue_record(run_dir):
        Path(run_dir).mkdir(parents=True, exist_ok=True)
        (Path(run_dir) / "landing_ue.json").write_text(json.dumps(
            {"landing_ue_ckpt": ckpt, "env": ue_env_cfg.as_dict(), "residual": residual,
             "actor_box": {"low": [-largs.res_thrust] + [-largs.res_rate] * 3,
                           "high": [largs.res_thrust] + [largs.res_rate] * 3} if residual else None,
             "alpha_floor": float(largs.alpha_floor), "qp_float64": bool(largs.qp_float64)}, indent=2))
        return orig_logger(run_dir)

    # --- the swaps (same places as scripts/train_phase2_landing.py, plus the UE-specific ones) ---------------
    entry.QuadrotorBCBFConfig = ue_cbf_config
    entry.QuadrotorBackupCBFProjector = uecbf.QuadrotorLandingUEBackupCBFProjector
    entry.build_quadrotor_env = ue_env
    entry._build_metric_logger = logger_with_ue_record
    trainer.QuadrotorBackupCBFProjector = uecbf.QuadrotorLandingUEBackupCBFProjector
    trainer.build_quadrotor_env = ue_env
    trainer._BINDING = make_landing_ue_binding(residual=residual, res_thrust=largs.res_thrust, res_rate=largs.res_rate)
    trainer._PHYS_DIM = UE_PHYS_DIM
    trainer._validate_learned_backup_policy_compatibility = check_ue_backup
    trainer._init_state = init_state_residual

    # flags the powerloop entry requires (ignored: geometry/backup/disturbance come from the checkpoint)
    forced = ["--backup_policy_mode", "learned", "--learned_backup_policy_path", ckpt]
    defaults = {"--base_set_c": "12.0", "--z_max": "15.0", "--reference_path": DEFAULT_REF, "--alpha": "10.0",
                "--output_root": "landing_phase2_ue", "--env_max_steps_extra_sec": "0.5",
                "--init_px_range": "0.2", "--init_py_range": "0.2", "--init_pz_range": "0.2",
                "--init_v_range": "0.3", "--init_tilt_deg_range": "5.0", "--init_yaw_deg_range": "5.0"}
    for flag, val in defaults.items():
        if flag not in rest:
            forced += [flag, val]
    print(f"[landing ue] checkpoint: {ckpt} | residual={residual} | hover_bias={not residual and not largs.no_hover_bias} "
          f"| jax x64={jax.config.jax_enable_x64}")
    entry.main(rest + forced)


if __name__ == "__main__":
    main()
