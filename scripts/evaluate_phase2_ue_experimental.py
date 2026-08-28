#!/usr/bin/env python
"""A/B evaluate an existing Phase-2 quadrotor actor with experimental UE-bCBF.

This script does NOT retrain Phase 2.  It evaluates the same saved actor under
identical seeds/disturbance with

  1) the original PS2 backup-CBF projector, and/or
  2) the experimental UE projector.

The UE branch uses the online 3-D disturbance observer, a short vanilla warmup
(default 0.2 s), the exact-discrete disturbance-channel tube, 1.2x empirical
tube inflation, and quadratic terminal h_B tightening.

The purpose is to determine whether the UE idea is practically useful before
finishing the nonlinear tube proof.  Results must not be presented as a formal
nonlinear safety certificate.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from math import pi
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from ps2rl.cil.cil_policy import ActorConfig
from ps2rl.cil.quadrotor_backup_cbf import QuadrotorBCBFConfig, QuadrotorBackupCBFProjector
from ps2rl.cil.quadrotor_ue_bcbf_experimental import ExperimentalUEConfig, QuadrotorExperimentalUEProjector
from ps2rl.envs.quadrotor_env import QuadrotorEnvConfig, build_quadrotor_env, quadrotor_dynamics
from ps2rl.evaluation import quadrotor_vanilla_eval as eval_utils
from ps2rl.phase2_ps2.quadrotor_ps2_trainer import SACConfig
from ps2rl.uncertainty.quadrotor_disturbance_observer import (
    disturbance_estimate,
    disturbance_observer_predict,
    initialize_disturbance_observer,
    observer_error_bound,
)
from ps2rl.utils.policy import actor_mean_action


def _resolve_config_path(run_dir: Path) -> Path:
    for name in ("configs.json", "config.json"):
        p = run_dir / name
        if p.exists():
            return p
    raise FileNotFoundError(f"No configs.json/config.json under {run_dir}")


def _weights_path(run_dir: Path, checkpoint: str) -> Path:
    name = {"best": "best_weights.pkl", "final": "final_weights.pkl"}.get(checkpoint, checkpoint)
    p = run_dir / name
    if p.exists():
        return p
    if not str(name).endswith(".pkl"):
        p2 = run_dir / f"{name}_weights.pkl"
        if p2.exists():
            return p2
    raise FileNotFoundError(f"Checkpoint not found: {p}")


def _stats(values: list[float]) -> dict[str, float | int | None]:
    a = np.asarray(values, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {"count": 0, "mean": None, "p50": None, "p95": None, "max": None}
    return {
        "count": int(a.size),
        "mean": float(np.mean(a)),
        "p50": float(np.percentile(a, 50.0)),
        "p95": float(np.percentile(a, 95.0)),
        "max": float(np.max(a)),
    }


def _run_mode(
    *,
    mode: str,
    actor_params: Any,
    actor_cfg: ActorConfig,
    sac_cfg: SACConfig,
    env_cfg: QuadrotorEnvConfig,
    cbf_cfg: QuadrotorBCBFConfig,
    ue_cfg: ExperimentalUEConfig,
    episodes: int,
    seed: int,
    observer_warmup_sec: float,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    env_fns = build_quadrotor_env(env_cfg)
    vanilla = QuadrotorBackupCBFProjector(cbf_cfg)
    ue = QuadrotorExperimentalUEProjector(cbf_cfg, ue_cfg=ue_cfg, runtime=vanilla.runtime)

    action_low = jnp.asarray(
        [cbf_cfg.a_cmd_min, -cbf_cfg.omega_max, -cbf_cfg.omega_max, -cbf_cfg.omega_max],
        dtype=jnp.float32,
    )
    action_high = jnp.asarray(
        [cbf_cfg.a_cmd_max, cbf_cfg.omega_max, cbf_cfg.omega_max, cbf_cfg.omega_max],
        dtype=jnp.float32,
    )
    action_scale = jnp.asarray(
        [cbf_cfg.a_cmd_max, cbf_cfg.omega_max, cbf_cfg.omega_max, cbf_cfg.omega_max],
        dtype=jnp.float32,
    )

    @jax.jit
    def raw_actor(params, obs_actor):
        return actor_mean_action(
            params,
            obs_actor[None, :],
            action_scale,
            actor_cfg,
            action_low=action_low,
            action_high=action_high,
        )[0]

    vanilla_solve = jax.jit(vanilla.solve_single_with_info)
    ue_solve = ue.solve_single_with_info

    returns: list[float] = []
    min_margins: list[float] = []
    violation_free: list[float] = []
    slack_values: list[float] = []
    intervention_values: list[float] = []
    solver_values: list[float] = []
    observer_ratio_values: list[float] = []
    ue_radius_values: list[float] = []
    ue_terminal_robust_h_values: list[float] = []
    ue_min_robust_h_values: list[float] = []
    ue_terminal_margin_values: list[float] = []
    ue_max_rho_safe_values: list[float] = []
    ue_max_rho_safe_raw_values: list[float] = []
    ue_max_rho_safe_phi_raw_values: list[float] = []
    ue_max_rho_safe_theta_raw_values: list[float] = []
    ue_terminal_rho_values: list[float] = []
    ue_terminal_rho_raw_values: list[float] = []
    ue_terminal_rho_phi_raw_values: list[float] = []
    ue_terminal_rho_theta_raw_values: list[float] = []
    ue_terminal_rho_alignment_values: list[float] = []
    ue_terminal_grad_h_b_norm_values: list[float] = []
    ue_terminal_rho_actual_values: list[float] = []
    ue_terminal_rho_actual_over_bound_values: list[float] = []
    ue_terminal_error_direction_alignment_values: list[float] = []
    ue_max_safe_output_margin_values: list[float] = []
    ue_terminal_safe_output_margin_values: list[float] = []

    trace: dict[str, list[Any]] = {
        "episode": [],
        "step": [],
        "time": [],
        "z": [],
        "hard_deck_margin": [],
        "reward": [],
        "slack": [],
        "use_solver": [],
        "intervention_norm": [],
        "d_hat_x": [],
        "d_hat_y": [],
        "d_hat_z": [],
        "d_true_x": [],
        "d_true_y": [],
        "d_true_z": [],
        "observer_error_norm": [],
        "observer_e_bar": [],
        "observer_error_ratio": [],
        "ue_active": [],
        "ue_max_radius": [],
        "ue_terminal_radius": [],
        "ue_min_robust_safe_h": [],
        "ue_terminal_h_b": [],
        "ue_terminal_h_b_robust": [],
        "ue_terminal_quadratic_margin": [],
        "ue_max_rho_safe": [],
        "ue_max_rho_safe_raw": [],
        "ue_max_rho_safe_phi_raw": [],
        "ue_max_rho_safe_theta_raw": [],
        "ue_terminal_rho": [],
        "ue_terminal_rho_raw": [],
        "ue_terminal_rho_phi_raw": [],
        "ue_terminal_rho_theta_raw": [],
        "ue_terminal_rho_phi_theta_alignment": [],
        "ue_terminal_grad_h_b_norm": [],
        "ue_terminal_rho_actual": [],
        "ue_terminal_rho_actual_over_bound": [],
        "ue_terminal_error_direction_alignment": [],
        "ue_max_safe_output_margin": [],
        "ue_terminal_safe_output_margin": [],
    }

    warmup_steps = int(round(float(observer_warmup_sec) / float(env_cfg.dt)))
    delta_v = float(ue_cfg.delta_v)

    for ep in range(int(episodes)):
        key_ep = jax.random.PRNGKey(int(seed) + 1009 * ep)
        state, obs = env_fns.reset(key_ep)
        observer = initialize_disturbance_observer(state.x[3:6])
        elapsed = 0.0
        ep_return = 0.0
        ep_min_margin = float("inf")
        ep_safe = True

        for step_idx in range(int(env_cfg.max_steps)):
            obs_actor = jnp.asarray(obs[: actor_cfg.obs_dim], dtype=jnp.float32)
            raw = raw_actor(actor_params, obs_actor)
            x_now = jnp.asarray(state.x, dtype=jnp.float32)
            d_hat = disturbance_estimate(x_now[3:6], observer, lambda_gain=float(ue_cfg.observer_lambda))
            e_bar = observer_error_bound(
                elapsed,
                delta_d=float(ue_cfg.delta_d),
                delta_v=delta_v,
                lambda_gain=float(ue_cfg.observer_lambda),
                dtype=jnp.float32,
            )

            ue_active = mode == "ue" and step_idx >= warmup_steps
            if ue_active:
                safe, slack, use_solver, info = ue_solve(x_now, raw, d_hat, e_bar)
                max_radius = float(info["max_radius"])
                terminal_radius = float(info["terminal_radius"])
                min_robust_h = float(info["min_robust_safe_h"])
                terminal_h_b = float(info["terminal_h_b"])
                terminal_h_b_rob = float(info["terminal_h_b_robust"])
                terminal_quad = float(info["terminal_quadratic_margin"])
                max_rho_safe = float(info["max_rho_safe"])
                max_rho_safe_raw = float(info["max_rho_safe_raw"])
                max_rho_safe_phi_raw = float(info["max_rho_safe_phi_raw"])
                max_rho_safe_theta_raw = float(info["max_rho_safe_theta_raw"])
                terminal_rho = float(info["terminal_rho"])
                terminal_rho_raw = float(info["terminal_rho_raw"])
                terminal_rho_phi_raw = float(info["terminal_rho_phi_raw"])
                terminal_rho_theta_raw = float(info["terminal_rho_theta_raw"])
                terminal_rho_alignment = float(info["terminal_rho_phi_theta_alignment"])
                terminal_ue_error_direction = np.asarray(info["terminal_ue_error_direction"], dtype=float)
                terminal_grad_h_b_norm = float(info["terminal_grad_h_b_norm"])
                max_safe_output_margin = float(info["max_safe_output_margin"])
                terminal_safe_output_margin = float(info["terminal_safe_output_margin"])
            else:
                safe, slack, use_solver, _ = vanilla_solve(x_now, raw)
                max_radius = 0.0
                terminal_radius = 0.0
                min_robust_h = np.nan
                terminal_h_b = np.nan
                terminal_h_b_rob = np.nan
                terminal_quad = np.nan
                max_rho_safe = np.nan
                max_rho_safe_raw = np.nan
                max_rho_safe_phi_raw = np.nan
                max_rho_safe_theta_raw = np.nan
                terminal_rho = np.nan
                terminal_rho_raw = np.nan
                terminal_rho_phi_raw = np.nan
                terminal_rho_theta_raw = np.nan
                terminal_rho_alignment = np.nan
                terminal_ue_error_direction = np.full((3,), np.nan, dtype=float)
                terminal_grad_h_b_norm = np.nan
                max_safe_output_margin = np.nan
                terminal_safe_output_margin = np.nan

            safe = jnp.clip(safe, action_low, action_high)
            nominal_acc = quadrotor_dynamics(
                x_now,
                safe,
                env_cfg.gravity,
                env_cfg.a_cmd_min,
                env_cfg.a_cmd_max,
                env_cfg.omega_max,
            )[3:6]
            observer_next = disturbance_observer_predict(
                observer,
                nominal_acc,
                d_hat,
                dt=float(env_cfg.dt),
            )

            key_step = jax.random.fold_in(key_ep, step_idx + 1)
            state_next, next_obs_true, obs_out, rew, done, env_info = env_fns.step(state, safe, key_step)

            d_true = jnp.asarray(env_info.disturbance_accel, dtype=jnp.float32)
            observer_error_vec = np.asarray(d_true - d_hat, dtype=float)
            obs_err = float(np.linalg.norm(observer_error_vec))
            ebar_f = float(e_bar)
            obs_ratio = obs_err / max(ebar_f, 1e-9)

            if ue_active:
                direction_norm = float(np.linalg.norm(terminal_ue_error_direction))
                terminal_rho_actual = abs(float(np.dot(terminal_ue_error_direction, observer_error_vec)))
                # Theoretical terminal_rho_raw = e_bar * ||direction||.
                # Avoid reporting meaningless ratios after a bound has
                # numerically decayed to essentially zero (notably f=0).
                if terminal_rho_raw > 1.0e-8:
                    terminal_rho_actual_over_bound = terminal_rho_actual / terminal_rho_raw
                else:
                    terminal_rho_actual_over_bound = np.nan
                if direction_norm > 1.0e-12 and obs_err > 1.0e-12:
                    terminal_error_direction_alignment = terminal_rho_actual / (direction_norm * obs_err)
                else:
                    terminal_error_direction_alignment = np.nan
            else:
                terminal_rho_actual = np.nan
                terminal_rho_actual_over_bound = np.nan
                terminal_error_direction_alignment = np.nan

            margin = float(env_info.hard_deck_margin)
            rew_f = float(rew)
            ep_return += rew_f
            ep_min_margin = min(ep_min_margin, margin)
            ep_safe = ep_safe and bool(float(env_info.is_safe) >= 0.5)

            slack_values.append(float(slack))
            intervention_values.append(float(jnp.linalg.norm(raw - safe)))
            solver_values.append(float(bool(use_solver)))
            observer_ratio_values.append(obs_ratio)
            if ue_active:
                ue_radius_values.append(max_radius)
                ue_terminal_robust_h_values.append(terminal_h_b_rob)
                ue_min_robust_h_values.append(min_robust_h)
                ue_terminal_margin_values.append(terminal_quad)
                ue_max_rho_safe_values.append(max_rho_safe)
                ue_max_rho_safe_raw_values.append(max_rho_safe_raw)
                ue_max_rho_safe_phi_raw_values.append(max_rho_safe_phi_raw)
                ue_max_rho_safe_theta_raw_values.append(max_rho_safe_theta_raw)
                ue_terminal_rho_values.append(terminal_rho)
                ue_terminal_rho_raw_values.append(terminal_rho_raw)
                ue_terminal_rho_phi_raw_values.append(terminal_rho_phi_raw)
                ue_terminal_rho_theta_raw_values.append(terminal_rho_theta_raw)
                ue_terminal_rho_alignment_values.append(terminal_rho_alignment)
                ue_terminal_grad_h_b_norm_values.append(terminal_grad_h_b_norm)
                if np.isfinite(terminal_rho_actual):
                    ue_terminal_rho_actual_values.append(terminal_rho_actual)
                if np.isfinite(terminal_rho_actual_over_bound):
                    ue_terminal_rho_actual_over_bound_values.append(terminal_rho_actual_over_bound)
                if np.isfinite(terminal_error_direction_alignment):
                    ue_terminal_error_direction_alignment_values.append(terminal_error_direction_alignment)
                ue_max_safe_output_margin_values.append(max_safe_output_margin)
                ue_terminal_safe_output_margin_values.append(terminal_safe_output_margin)

            trace["episode"].append(ep)
            trace["step"].append(step_idx)
            trace["time"].append(step_idx * float(env_cfg.dt))
            trace["z"].append(float(state.x[2]))
            trace["hard_deck_margin"].append(margin)
            trace["reward"].append(rew_f)
            trace["slack"].append(float(slack))
            trace["use_solver"].append(float(bool(use_solver)))
            trace["intervention_norm"].append(float(jnp.linalg.norm(raw - safe)))
            trace["d_hat_x"].append(float(d_hat[0]))
            trace["d_hat_y"].append(float(d_hat[1]))
            trace["d_hat_z"].append(float(d_hat[2]))
            trace["d_true_x"].append(float(d_true[0]))
            trace["d_true_y"].append(float(d_true[1]))
            trace["d_true_z"].append(float(d_true[2]))
            trace["observer_error_norm"].append(obs_err)
            trace["observer_e_bar"].append(ebar_f)
            trace["observer_error_ratio"].append(obs_ratio)
            trace["ue_active"].append(float(ue_active))
            trace["ue_max_radius"].append(max_radius)
            trace["ue_terminal_radius"].append(terminal_radius)
            trace["ue_min_robust_safe_h"].append(min_robust_h)
            trace["ue_terminal_h_b"].append(terminal_h_b)
            trace["ue_terminal_h_b_robust"].append(terminal_h_b_rob)
            trace["ue_terminal_quadratic_margin"].append(terminal_quad)
            trace["ue_max_rho_safe"].append(max_rho_safe)
            trace["ue_max_rho_safe_raw"].append(max_rho_safe_raw)
            trace["ue_max_rho_safe_phi_raw"].append(max_rho_safe_phi_raw)
            trace["ue_max_rho_safe_theta_raw"].append(max_rho_safe_theta_raw)
            trace["ue_terminal_rho"].append(terminal_rho)
            trace["ue_terminal_rho_raw"].append(terminal_rho_raw)
            trace["ue_terminal_rho_phi_raw"].append(terminal_rho_phi_raw)
            trace["ue_terminal_rho_theta_raw"].append(terminal_rho_theta_raw)
            trace["ue_terminal_rho_phi_theta_alignment"].append(terminal_rho_alignment)
            trace["ue_terminal_grad_h_b_norm"].append(terminal_grad_h_b_norm)
            trace["ue_terminal_rho_actual"].append(terminal_rho_actual)
            trace["ue_terminal_rho_actual_over_bound"].append(terminal_rho_actual_over_bound)
            trace["ue_terminal_error_direction_alignment"].append(terminal_error_direction_alignment)
            trace["ue_max_safe_output_margin"].append(max_safe_output_margin)
            trace["ue_terminal_safe_output_margin"].append(terminal_safe_output_margin)

            observer = observer_next
            elapsed += float(env_cfg.dt)
            state = state_next
            obs = next_obs_true
            if bool(done):
                break

        returns.append(ep_return)
        min_margins.append(ep_min_margin)
        violation_free.append(1.0 if ep_safe else 0.0)
        print(
            f"[{mode}] episode {ep + 1}/{episodes}: return={ep_return:.3f} "
            f"min_hS={ep_min_margin:.5f} safe={ep_safe}"
        )

    summary = {
        "mode": mode,
        "episodes": int(episodes),
        "return": _stats(returns),
        "min_hard_deck_margin": _stats(min_margins),
        "violation_free_episode_rate": float(np.mean(violation_free)) if violation_free else 1.0,
        "slack": _stats(slack_values),
        "intervention_norm": _stats(intervention_values),
        "solver_use_rate": float(np.mean(solver_values)) if solver_values else 0.0,
        "observer_actual_over_bound": _stats(observer_ratio_values),
        "observer_bound_violation_count": int(np.sum(np.asarray(observer_ratio_values) > 1.0 + 1e-6)),
        "ue_max_radius": _stats(ue_radius_values),
        "ue_terminal_h_b_robust": _stats(ue_terminal_robust_h_values),
        "ue_min_robust_safe_h": _stats(ue_min_robust_h_values),
        "ue_terminal_quadratic_margin": _stats(ue_terminal_margin_values),
        "ue_max_rho_safe": _stats(ue_max_rho_safe_values),
        "ue_max_rho_safe_raw": _stats(ue_max_rho_safe_raw_values),
        "ue_max_rho_safe_phi_raw": _stats(ue_max_rho_safe_phi_raw_values),
        "ue_max_rho_safe_theta_raw": _stats(ue_max_rho_safe_theta_raw_values),
        "ue_terminal_rho": _stats(ue_terminal_rho_values),
        "ue_terminal_rho_raw": _stats(ue_terminal_rho_raw_values),
        "ue_terminal_rho_phi_raw": _stats(ue_terminal_rho_phi_raw_values),
        "ue_terminal_rho_theta_raw": _stats(ue_terminal_rho_theta_raw_values),
        "ue_terminal_rho_phi_theta_alignment": _stats(ue_terminal_rho_alignment_values),
        "ue_terminal_grad_h_b_norm": _stats(ue_terminal_grad_h_b_norm_values),
        "ue_terminal_rho_actual": _stats(ue_terminal_rho_actual_values),
        "ue_terminal_rho_actual_over_bound": _stats(ue_terminal_rho_actual_over_bound_values),
        "ue_terminal_error_direction_alignment": _stats(ue_terminal_error_direction_alignment_values),
        "ue_max_safe_output_margin": _stats(ue_max_safe_output_margin_values),
        "ue_terminal_safe_output_margin": _stats(ue_terminal_safe_output_margin_values),
    }
    arrays = {k: np.asarray(v) for k, v in trace.items()}
    return summary, arrays


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True, help="Existing Phase-2 run directory containing configs + weights")
    p.add_argument("--checkpoint", default="best", help="best, final, or explicit checkpoint filename")
    p.add_argument("--mode", choices=("both", "vanilla", "ue"), default="both")
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--seed", type=int, default=3500000)
    p.add_argument("--disturbance-amplitude", type=float, default=2.0)
    p.add_argument("--disturbance-frequency-hz", type=float, default=0.5)
    p.add_argument("--disturbance-phase", type=float, default=0.0)
    p.add_argument("--disturbance-direction-x", type=float, default=1.0)
    p.add_argument("--disturbance-direction-y", type=float, default=0.0)
    p.add_argument("--disturbance-direction-z", type=float, default=0.0)
    p.add_argument("--ue-horizon", type=float, default=1.0)
    p.add_argument("--ue-tube-scale", type=float, default=1.2)
    p.add_argument(
        "--ue-rho-scale",
        type=float,
        default=1.0,
        help="Global diagnostic scale on both UE rho terms; 0 disables both, 1 leaves per-row scales unchanged",
    )
    p.add_argument(
        "--ue-safe-rho-scale",
        type=float,
        default=1.0,
        help="Diagnostic scale on intermediate safe-set rho only; use 0 to disable rho_S",
    )
    p.add_argument(
        "--ue-terminal-rho-scale",
        type=float,
        default=1.0,
        help="Diagnostic scale on terminal base-set rho only; use 0 to disable rho_B",
    )
    p.add_argument("--ue-observer-lambda", type=float, default=20.0)
    p.add_argument("--ue-observer-warmup-sec", type=float, default=0.2)
    p.add_argument(
        "--ue-terminal-mode",
        choices=("quadratic", "nominal"),
        default="quadratic",
        help="quadratic = use generator-based robust terminal tightening; nominal = diagnostic ablation only",
    )
    p.add_argument("--output", default="outputs/phase2_ue_experimental.json")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    cfg_json = eval_utils._load_json(_resolve_config_path(run_dir))
    sac_cfg = eval_utils._dataclass_from_dict(SACConfig, cfg_json.get("sac", {}))
    env_cfg_saved = eval_utils._dataclass_from_dict(QuadrotorEnvConfig, cfg_json.get("env", {}))
    cbf_cfg_saved = eval_utils._dataclass_from_dict(QuadrotorBCBFConfig, cfg_json.get("cbf", {}))
    actor_params = eval_utils._load_actor_params(_weights_path(run_dir, str(args.checkpoint)))

    actor_obs_dim = eval_utils._infer_actor_obs_dim(actor_params)
    if actor_obs_dim is None:
        actor_obs_dim = int(build_quadrotor_env(env_cfg_saved).obs_dim)
    actor_cfg = ActorConfig(
        obs_dim=int(actor_obs_dim),
        action_dim=4,
        hidden_sizes=(int(sac_cfg.hidden_size), int(sac_cfg.hidden_size)),
    )

    amp = float(args.disturbance_amplitude)
    freq = float(args.disturbance_frequency_hz)
    delta_v = 2.0 * pi * freq * amp
    env_cfg = replace(
        env_cfg_saved,
        disturbance_mode="sinusoidal" if amp > 0.0 else "none",
        disturbance_amplitude=amp,
        disturbance_frequency_hz=freq,
        disturbance_phase=float(args.disturbance_phase),
        disturbance_direction_x=float(args.disturbance_direction_x),
        disturbance_direction_y=float(args.disturbance_direction_y),
        disturbance_direction_z=float(args.disturbance_direction_z),
        terminate_on_violation=False,
    )

    n_steps = int(round(float(args.ue_horizon) / float(cbf_cfg_saved.dt)))
    if n_steps <= 0:
        raise ValueError("--ue-horizon must produce at least one backup step")
    cbf_cfg = replace(
        cbf_cfg_saved,
        T=n_steps * float(cbf_cfg_saved.dt),
        num_steps=n_steps,
    )
    ue_cfg = ExperimentalUEConfig(
        delta_d=amp,
        delta_v=delta_v,
        observer_lambda=float(args.ue_observer_lambda),
        tube_scale=float(args.ue_tube_scale),
        rho_scale=float(args.ue_rho_scale),
        safe_rho_scale=float(args.ue_safe_rho_scale),
        terminal_rho_scale=float(args.ue_terminal_rho_scale),
        terminal_mode=str(args.ue_terminal_mode),
    )

    print("\nExperimental UE Phase-2 evaluation")
    print(f"run: {run_dir}")
    print(f"checkpoint: {args.checkpoint}")
    print(f"backup mode: {cbf_cfg.backup_policy_mode}")
    print(f"T={cbf_cfg.T:.3f}s N={cbf_cfg.num_steps} dt={cbf_cfg.dt:.3f}s")
    print(f"disturbance: delta_d={amp:.3f}, f={freq:.3f}Hz, delta_v={delta_v:.6f}")
    print(f"observer lambda={ue_cfg.observer_lambda:.3f}, warmup={args.ue_observer_warmup_sec:.3f}s")
    print(f"empirical tube scale={ue_cfg.tube_scale:.3f}")
    print(f"UE rho scale={ue_cfg.rho_scale:.3f}")
    print(f"UE safe rho scale={ue_cfg.safe_rho_scale:.3f}")
    print(f"UE terminal rho scale={ue_cfg.terminal_rho_scale:.3f}")
    print(f"terminal mode={ue_cfg.terminal_mode}")
    print("NOTE: experimental first-order tube; this is not yet a formal nonlinear certificate.\n")

    modes = ["vanilla", "ue"] if args.mode == "both" else [args.mode]
    report: dict[str, Any] = {
        "settings": {
            **vars(args),
            "run_dir": str(run_dir),
            "delta_v": delta_v,
            "cbf_dt": float(cbf_cfg.dt),
            "cbf_num_steps": int(cbf_cfg.num_steps),
        },
        "formal_status": "experimental Phase-2 A/B test; UE tube is empirical/first-order, not a formal nonlinear certificate",
        "results": {},
    }

    out = Path(args.output).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    for mode in modes:
        summary, arrays = _run_mode(
            mode=mode,
            actor_params=actor_params,
            actor_cfg=actor_cfg,
            sac_cfg=sac_cfg,
            env_cfg=env_cfg,
            cbf_cfg=cbf_cfg,
            ue_cfg=ue_cfg,
            episodes=int(args.episodes),
            seed=int(args.seed),
            observer_warmup_sec=float(args.ue_observer_warmup_sec),
        )
        report["results"][mode] = summary
        npz_path = out.with_name(f"{out.stem}_{mode}.npz")
        np.savez_compressed(npz_path, **arrays)
        print(f"saved trace: {npz_path}")

    out.write_text(json.dumps(report, indent=2))
    print(f"\nSaved summary: {out}")
    if "vanilla" in report["results"] and "ue" in report["results"]:
        v = report["results"]["vanilla"]
        u = report["results"]["ue"]
        print("\n=== A/B quick comparison ===")
        print(f"vanilla violation-free: {100.0 * v['violation_free_episode_rate']:.2f}%")
        print(f"UE      violation-free: {100.0 * u['violation_free_episode_rate']:.2f}%")
        print(f"vanilla return mean: {v['return']['mean']}")
        print(f"UE      return mean: {u['return']['mean']}")
        print(f"UE observer bound violations: {u['observer_bound_violation_count']}")
        print(f"UE max radius p50/p95/max: {u['ue_max_radius']['p50']} / {u['ue_max_radius']['p95']} / {u['ue_max_radius']['max']}")
        print(f"UE slack p50/p95/max: {u['slack']['p50']} / {u['slack']['p95']} / {u['slack']['max']}")
        print(f"UE max rho safe p50/p95/max: {u['ue_max_rho_safe']['p50']} / {u['ue_max_rho_safe']['p95']} / {u['ue_max_rho_safe']['max']}")
        print(f"UE terminal rho p50/p95/max: {u['ue_terminal_rho']['p50']} / {u['ue_terminal_rho']['p95']} / {u['ue_terminal_rho']['max']}")
        print(f"UE max safe-output margin p50/p95/max: {u['ue_max_safe_output_margin']['p50']} / {u['ue_max_safe_output_margin']['p95']} / {u['ue_max_safe_output_margin']['max']}")


if __name__ == "__main__":
    main()
