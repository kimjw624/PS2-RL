#!/usr/bin/env python3
"""Paired weak-disturbance evaluation for standard bCBF versus UE-bCBF.

This is an experimental, add-only evaluator.  It compares two frozen Phase-2
actors (the nominal warm-start policy and the UE-fine-tuned policy) through the
same standard and UE safety filters.  Every paired condition uses identical
initial-state PRNG keys, disturbance directions, amplitudes, frequencies, and
phases.

The UE implementation remains an empirical first-order tube for the nonlinear
quadrotor model; results from this script are empirical evidence, not a formal
nonlinear robustness certificate.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass, replace
from datetime import datetime
import hashlib
import json
from math import pi
from pathlib import Path
import pickle
import shlex
import subprocess
import sys
from typing import Any, Iterable

import jax
import jax.numpy as jnp
import numpy as np

from ps2rl.cil.quadrotor_backup_cbf import QuadrotorBCBFConfig, QuadrotorBackupCBFProjector
from ps2rl.cil.quadrotor_ue_bcbf_experimental import ExperimentalUEConfig
from ps2rl.envs.quadrotor_env import QuadrotorEnvConfig, build_quadrotor_env
from ps2rl.evaluation import quadrotor_vanilla_eval as eval_utils
from ps2rl.phase2_ps2.ps2_trainer_core import build_ps2_action_fns
from ps2rl.phase2_ps2.quadrotor_ps2_trainer import (
    SACConfig,
    _action_bounds_from_cbf_cfg,
    _build_action_fns,
    _build_ue_observer_env,
    _disable_backup_fallback,
    _make_ue_projection_ops,
    _ue_network_obs_fn,
    _ue_projection_obs_fn,
)
from ps2rl.utils.policy import ActorConfig
from ps2rl.utils.seed import make_prng_key


PROJECT_ROOT = Path(__file__).resolve().parents[1]
_PHYS_DIM = 10
_DIRECTION_ORDER = ("x", "y", "z", "xy", "xz", "yz", "xyz")
_DIRECTION_VECTORS = {
    "x": np.asarray([1.0, 0.0, 0.0]),
    "y": np.asarray([0.0, 1.0, 0.0]),
    "z": np.asarray([0.0, 0.0, 1.0]),
    "xy": np.asarray([1.0, 1.0, 0.0]) / np.sqrt(2.0),
    "xz": np.asarray([1.0, 0.0, 1.0]) / np.sqrt(2.0),
    "yz": np.asarray([0.0, 1.0, 1.0]) / np.sqrt(2.0),
    "xyz": np.asarray([1.0, 1.0, 1.0]) / np.sqrt(3.0),
}


@dataclass(frozen=True)
class PolicySpec:
    label: str
    run_dir: Path
    checkpoint: Path
    sac_cfg: SACConfig
    actor_params: Any


@dataclass(frozen=True)
class ConditionRuntime:
    label: str
    policy: PolicySpec
    filter_name: str
    eval_action_fn: Any


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Not JSON serializable: {type(value)!r}")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, default=_json_default)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _resolve_config_path(run_dir: Path) -> Path:
    for name in ("configs.json", "config.json"):
        path = run_dir / name
        if path.exists():
            return path
    raise FileNotFoundError(f"No configs.json/config.json under {run_dir}")


def _resolve_checkpoint(run_dir: Path, name: str) -> Path:
    aliases = {"best": "best_weights.pkl", "final": "final_weights.pkl"}
    candidate_name = aliases.get(name, name)
    candidates = [run_dir / candidate_name, run_dir / "checkpoints" / candidate_name]
    if not candidate_name.endswith(".pkl"):
        candidates.extend(
            [run_dir / f"{candidate_name}_weights.pkl", run_dir / "checkpoints" / f"{candidate_name}_weights.pkl"]
        )
    for path in candidates:
        if path.exists():
            return path.resolve()
    tried = "\n".join(f"  - {path}" for path in candidates)
    raise FileNotFoundError(f"Checkpoint {name!r} not found. Tried:\n{tried}")


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return payload


def _load_actor_params(path: Path) -> Any:
    # Checkpoints are trusted artifacts produced by this repository.
    with path.open("rb") as stream:
        payload = pickle.load(stream)
    if not isinstance(payload, dict) or "actor_params" not in payload:
        raise KeyError(f"Checkpoint does not contain actor_params: {path}")
    return jax.tree_util.tree_map(lambda value: jnp.asarray(value), payload["actor_params"])


def _load_policy(label: str, run_dir_raw: str, checkpoint_name: str) -> tuple[PolicySpec, dict[str, Any]]:
    run_dir = Path(run_dir_raw).expanduser().resolve()
    cfg_json = _load_json(_resolve_config_path(run_dir))
    sac_cfg = eval_utils._dataclass_from_dict(SACConfig, cfg_json.get("sac", {}))
    checkpoint = _resolve_checkpoint(run_dir, checkpoint_name)
    return (
        PolicySpec(
            label=label,
            run_dir=run_dir,
            checkpoint=checkpoint,
            sac_cfg=sac_cfg,
            actor_params=_load_actor_params(checkpoint),
        ),
        cfg_json,
    )


def _parse_csv_floats(raw: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in raw.split(",") if item.strip())
    if not values:
        raise ValueError("At least one phase is required")
    if not all(np.isfinite(value) for value in values):
        raise ValueError("All phases must be finite")
    return values


def _float_tag(value: float, digits: int = 2) -> str:
    return f"{float(value):.{digits}f}".replace("-", "m").replace(".", "p")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_metadata() -> dict[str, Any]:
    def run(*args: str) -> str:
        proc = subprocess.run(
            ["git", *args],
            cwd=PROJECT_ROOT,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return proc.stdout.strip()

    status = run("status", "--short")
    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "dirty": bool(status),
        "status_short": status.splitlines(),
    }


def _source_fingerprints() -> dict[str, dict[str, Any]]:
    """Fingerprint behavior-relevant source even when Git metadata is unavailable."""

    relative_paths = (
        "scripts/evaluate_quadrotor_ue_weak_paired.py",
        "ps2rl/cil/backup_cbf.py",
        "ps2rl/cil/quadrotor_backup_cbf.py",
        "ps2rl/cil/quadrotor_ue_bcbf_experimental.py",
        "ps2rl/envs/quadrotor_env.py",
        "ps2rl/phase2_ps2/ps2_trainer_core.py",
        "ps2rl/phase2_ps2/quadrotor_ps2_trainer.py",
    )
    output: dict[str, dict[str, Any]] = {}
    for relative in relative_paths:
        path = PROJECT_ROOT / relative
        output[relative] = {
            "exists": path.is_file(),
            "sha256": _sha256(path) if path.is_file() else "",
            "size_bytes": path.stat().st_size if path.is_file() else 0,
        }
    return output


def _direction_label(vector: np.ndarray) -> str:
    vector = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        return "unknown"
    unit = vector / norm
    distances = {name: float(np.linalg.norm(unit - expected)) for name, expected in _DIRECTION_VECTORS.items()}
    label = min(distances, key=distances.get)
    if distances[label] > 2e-5:
        raise RuntimeError(f"Unexpected axis_set7 direction: {unit.tolist()}")
    return label


def _physical_env_state(filter_name: str, state: Any) -> Any:
    return state.env_state if filter_name == "ue" else state


def _find_balanced_episode_seeds(
    env_cfg: QuadrotorEnvConfig,
    *,
    seed_start: int,
    repeats: int,
    search_limit: int,
) -> dict[str, list[int]]:
    finder_cfg = replace(env_cfg, disturbance_phase=0.0, disturbance_direction_mode="axis_set7")
    env_fns = build_quadrotor_env(finder_cfg)
    selected = {label: [] for label in _DIRECTION_ORDER}
    for candidate in range(int(seed_start), int(seed_start) + int(search_limit)):
        state, _ = env_fns.reset(make_prng_key(candidate))
        label = _direction_label(np.asarray(state.disturbance_direction))
        if len(selected[label]) < repeats:
            selected[label].append(candidate)
        if all(len(selected[label]) >= repeats for label in _DIRECTION_ORDER):
            return selected
    counts = {label: len(values) for label, values in selected.items()}
    raise RuntimeError(f"Could not collect {repeats} reset keys for every axis_set7 direction: {counts}")


def _actor_cfg(policy: PolicySpec, obs_dim: int, action_dim: int) -> ActorConfig:
    hidden = int(policy.sac_cfg.hidden_size)
    return ActorConfig(obs_dim=obs_dim, action_dim=action_dim, hidden_sizes=(hidden, hidden))


def _build_condition_runtimes(
    policies: Iterable[PolicySpec],
    env_cfg: QuadrotorEnvConfig,
    cbf_cfg: QuadrotorBCBFConfig,
    ue_cfg: ExperimentalUEConfig,
    observer_warmup_sec: float,
) -> list[ConditionRuntime]:
    base_env = build_quadrotor_env(env_cfg)
    base_obs_dim = int(base_env.obs_dim)
    action_dim = int(base_env.action_dim)
    action_scale = jnp.asarray(
        [cbf_cfg.a_cmd_max, cbf_cfg.omega_max, cbf_cfg.omega_max, cbf_cfg.omega_max], dtype=jnp.float32
    )
    action_low, action_high = _action_bounds_from_cbf_cfg(cbf_cfg)
    vanilla_projector = QuadrotorBackupCBFProjector(cbf_cfg)
    ue_projection_ops = _make_ue_projection_ops(
        cbf_cfg,
        ue_cfg,
        observer_warmup_sec=float(observer_warmup_sec),
        vanilla_projector=vanilla_projector,
    )

    conditions: list[ConditionRuntime] = []
    for policy in policies:
        eval_sac = replace(policy.sac_cfg, use_projection=True, project_actor_actions=True)
        actor_cfg = _actor_cfg(policy, base_obs_dim, action_dim)

        _, standard_eval = _build_action_fns(
            eval_sac,
            actor_cfg,
            cbf_cfg,
            action_scale,
            backup_runtime=vanilla_projector.runtime,
            return_solver_info=True,
        )
        conditions.append(
            ConditionRuntime(
                label=f"{policy.label}__standard",
                policy=policy,
                filter_name="standard",
                eval_action_fn=standard_eval,
            )
        )

        _, ue_eval = build_ps2_action_fns(
            eval_sac,
            actor_cfg,
            action_scale,
            action_low,
            action_high,
            ue_projection_ops,
            phys_dim=_PHYS_DIM,
            disable_backup_fallback=_disable_backup_fallback(eval_sac),
            return_solver_info=True,
            network_obs_fn=_ue_network_obs_fn(base_obs_dim),
            projection_obs_fn=_ue_projection_obs_fn(base_obs_dim),
        )
        conditions.append(
            ConditionRuntime(
                label=f"{policy.label}__ue",
                policy=policy,
                filter_name="ue",
                eval_action_fn=ue_eval,
            )
        )
    return conditions


def _safe_mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _rms(values: list[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    return float(np.sqrt(np.mean(np.square(array)))) if array.size else 0.0


def _scalar(value: Any) -> float:
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(f"Expected scalar value, got shape {array.shape}")
    return float(array.reshape(()))


def _bool_scalar(value: Any) -> bool:
    return bool(np.asarray(value).reshape(()))


def _all_finite_info(info: dict[str, Any]) -> bool:
    keys = ("inputs_finite", "z_finite", "q_mat_finite", "q_vec_finite", "g_finite", "h_finite")
    present = [key for key in keys if key in info]
    return all(_bool_scalar(info[key]) for key in present) if present else True


_QP_FINITE_KEYS = (
    "q_mat_finite",
    "q_vec_finite",
    "g_finite",
    "h_finite",
    "inputs_finite",
    "z_finite",
)


def _qp_step_diagnostics(info: dict[str, Any], use_solver: Any) -> dict[str, float]:
    """Convert one projector info tree into behavior-neutral scalar diagnostics."""

    used = _bool_scalar(use_solver)
    flags = {
        key: (_bool_scalar(info[key]) if key in info else True)
        for key in _QP_FINITE_KEYS
    }
    inputs_finite = flags["inputs_finite"]
    z_finite = flags["z_finite"]
    fallback = not used
    invalid_inputs = fallback and not inputs_finite
    nonfinite_solution = fallback and inputs_finite and not z_finite
    unclassified = fallback and not invalid_inputs and not nonfinite_solution
    return {
        "use_solver": float(used),
        "fallback": float(fallback),
        "fallback_invalid_inputs": float(invalid_inputs),
        "fallback_nonfinite_solution": float(nonfinite_solution),
        "fallback_unclassified": float(unclassified),
        **{key: float(value) for key, value in flags.items()},
        "q_saturated": float(_bool_scalar(info["q_saturated"])) if "q_saturated" in info else 0.0,
    }


def _rollout_episode(
    condition: ConditionRuntime,
    env_fns: Any,
    *,
    episode_seed: int,
    phase_deg: float,
    expected_direction: str,
    observer_warmup_sec: float,
    env_dt: float,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    episode_key = make_prng_key(int(episode_seed))
    state, obs = env_fns.reset(episode_key)
    physical_state = _physical_env_state(condition.filter_name, state)
    actual_direction = _direction_label(np.asarray(physical_state.disturbance_direction))
    if actual_direction != expected_direction:
        raise RuntimeError(
            f"Reset direction changed for seed {episode_seed}: expected {expected_direction}, got {actual_direction}"
        )

    total_return = 0.0
    safe_steps: list[float] = []
    margins: list[float] = []
    pos_errors: list[float] = []
    vel_errors: list[float] = []
    att_errors: list[float] = []
    pos_xyz_errors: list[float] = []
    interventions: list[float] = []
    slacks: list[float] = []
    solver_flags: list[float] = []
    finite_flags: list[float] = []
    qp_diagnostics: dict[str, list[float]] = {}
    fallback_step_indices: list[int] = []
    post_warmup_fallback_steps = 0
    post_warmup_steps = 0
    disturbance_norms: list[float] = []

    trajectory: dict[str, list[Any]] = {
        "time_sec": [],
        "state": [],
        "reference_state": [],
        "raw_action": [],
        "safe_action": [],
        "hard_deck_margin": [],
        "is_safe": [],
        "disturbance_accel": [],
        "intervention_norm": [],
        "slack": [],
        "use_solver": [],
        "qp_inputs_finite": [],
        "qp_z_finite": [],
        "qp_q_saturated": [],
        # 0=solver used, 1=invalid inputs, 2=non-finite solution,
        # 3=unclassified fallback.
        "qp_fallback_reason": [],
        "ue_filter_active": [],
    }

    done = False
    step_idx = 0
    while not done:
        obs_j = jnp.asarray(obs, dtype=jnp.float32)
        safe_action, raw_action, slack, use_solver, qp_info = condition.eval_action_fn(
            condition.policy.actor_params, obs_j
        )
        step_key = jax.random.fold_in(episode_key, step_idx)
        state, next_obs_true, _, reward, done_j, env_info = env_fns.step(state, safe_action, step_key)
        done = _bool_scalar(done_j)

        safe_np = np.asarray(safe_action, dtype=np.float64)
        raw_np = np.asarray(raw_action, dtype=np.float64)
        next_obs_np = np.asarray(next_obs_true, dtype=np.float64)
        ref_state_np = np.asarray(env_info.ref_state, dtype=np.float64)
        intervention = float(np.linalg.norm(safe_np - raw_np))
        margin = _scalar(env_info.hard_deck_margin)
        is_safe = _scalar(env_info.is_safe)
        disturbance = np.asarray(env_info.disturbance_accel, dtype=np.float64)

        total_return += _scalar(reward)
        safe_steps.append(is_safe)
        margins.append(margin)
        pos_errors.append(_scalar(env_info.pos_error_norm))
        vel_errors.append(_scalar(env_info.vel_error_norm))
        att_errors.append(_scalar(env_info.att_error_norm))
        pos_xyz_errors.append(float(np.linalg.norm(next_obs_np[:3] - ref_state_np[:3])))
        interventions.append(intervention)
        slacks.append(_scalar(slack))
        solver_flags.append(1.0 if _bool_scalar(use_solver) else 0.0)
        finite_flags.append(1.0 if _all_finite_info(qp_info) else 0.0)
        step_qp = _qp_step_diagnostics(qp_info, use_solver)
        for key, value in step_qp.items():
            qp_diagnostics.setdefault(key, []).append(value)
        if step_qp["fallback"] > 0.5:
            fallback_step_indices.append(step_idx)
        ue_filter_active = condition.filter_name != "ue" or step_idx * float(env_dt) >= float(observer_warmup_sec)
        if ue_filter_active:
            post_warmup_steps += 1
            post_warmup_fallback_steps += int(step_qp["fallback"] > 0.5)
        disturbance_norms.append(float(np.linalg.norm(disturbance)))

        trajectory["time_sec"].append(_scalar(env_info.ref_time_sec))
        trajectory["state"].append(next_obs_np[:10])
        trajectory["reference_state"].append(ref_state_np)
        trajectory["raw_action"].append(raw_np)
        trajectory["safe_action"].append(safe_np)
        trajectory["hard_deck_margin"].append(margin)
        trajectory["is_safe"].append(is_safe)
        trajectory["disturbance_accel"].append(disturbance)
        trajectory["intervention_norm"].append(intervention)
        trajectory["slack"].append(_scalar(slack))
        trajectory["use_solver"].append(step_qp["use_solver"])
        trajectory["qp_inputs_finite"].append(step_qp["inputs_finite"])
        trajectory["qp_z_finite"].append(step_qp["z_finite"])
        trajectory["qp_q_saturated"].append(step_qp["q_saturated"])
        reason = 0
        if step_qp["fallback_invalid_inputs"] > 0.5:
            reason = 1
        elif step_qp["fallback_nonfinite_solution"] > 0.5:
            reason = 2
        elif step_qp["fallback_unclassified"] > 0.5:
            reason = 3
        trajectory["qp_fallback_reason"].append(reason)
        trajectory["ue_filter_active"].append(float(ue_filter_active))

        obs = next_obs_true
        step_idx += 1

    violation_steps = int(sum(value < 0.5 for value in safe_steps))
    row = {
        "condition": condition.label,
        "policy": condition.policy.label,
        "filter": condition.filter_name,
        "episode_seed": int(episode_seed),
        "direction": actual_direction,
        "phase_deg": float(phase_deg),
        "steps": int(step_idx),
        "return": float(total_return),
        "safe_step_rate": _safe_mean(safe_steps),
        "violation_free": 1.0 if violation_steps == 0 else 0.0,
        "violation_steps": violation_steps,
        "hard_deck_margin_min": float(np.min(margins)),
        "hard_deck_margin_p05": float(np.percentile(margins, 5.0)),
        "pos_error_norm_rmse": _rms(pos_errors),
        "pos_xyz_rmse": _rms(pos_xyz_errors),
        "vel_error_norm_rmse": _rms(vel_errors),
        "att_error_norm_rmse": _rms(att_errors),
        "intervention_norm_mean": _safe_mean(interventions),
        "intervention_norm_max": float(np.max(interventions)),
        "slack_mean": _safe_mean(slacks),
        "slack_max": float(np.max(slacks)),
        "solver_rate": _safe_mean(solver_flags),
        "qp_finite_rate": _safe_mean(finite_flags),
        "qp_fallback_steps": int(sum(qp_diagnostics.get("fallback", []))),
        "qp_fallback_rate": _safe_mean(qp_diagnostics.get("fallback", [])),
        "qp_fallback_invalid_inputs_steps": int(sum(qp_diagnostics.get("fallback_invalid_inputs", []))),
        "qp_fallback_invalid_inputs_rate": _safe_mean(qp_diagnostics.get("fallback_invalid_inputs", [])),
        "qp_fallback_nonfinite_solution_steps": int(
            sum(qp_diagnostics.get("fallback_nonfinite_solution", []))
        ),
        "qp_fallback_nonfinite_solution_rate": _safe_mean(
            qp_diagnostics.get("fallback_nonfinite_solution", [])
        ),
        "qp_fallback_unclassified_steps": int(sum(qp_diagnostics.get("fallback_unclassified", []))),
        "qp_fallback_unclassified_rate": _safe_mean(qp_diagnostics.get("fallback_unclassified", [])),
        "qp_q_mat_finite_rate": _safe_mean(qp_diagnostics.get("q_mat_finite", [])),
        "qp_q_vec_finite_rate": _safe_mean(qp_diagnostics.get("q_vec_finite", [])),
        "qp_g_finite_rate": _safe_mean(qp_diagnostics.get("g_finite", [])),
        "qp_h_finite_rate": _safe_mean(qp_diagnostics.get("h_finite", [])),
        "qp_inputs_finite_rate": _safe_mean(qp_diagnostics.get("inputs_finite", [])),
        "qp_z_finite_rate": _safe_mean(qp_diagnostics.get("z_finite", [])),
        "qp_q_saturated_rate": _safe_mean(qp_diagnostics.get("q_saturated", [])),
        "qp_first_fallback_step": fallback_step_indices[0] if fallback_step_indices else -1,
        "qp_last_fallback_step": fallback_step_indices[-1] if fallback_step_indices else -1,
        "qp_post_warmup_steps": int(post_warmup_steps),
        "qp_post_warmup_fallback_steps": int(post_warmup_fallback_steps),
        "qp_post_warmup_fallback_rate": (
            float(post_warmup_fallback_steps) / float(post_warmup_steps) if post_warmup_steps else 0.0
        ),
        "disturbance_norm_max": float(np.max(disturbance_norms)),
    }
    trajectory_np = {key: np.asarray(values) for key, values in trajectory.items()}
    trajectory_np.update(
        {
            "episode_seed": np.asarray(int(episode_seed)),
            "phase_deg": np.asarray(float(phase_deg)),
            "direction": np.asarray(actual_direction),
            "condition": np.asarray(condition.label),
        }
    )
    return row, trajectory_np


def _summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["condition"]), []).append(row)

    summaries: list[dict[str, Any]] = []
    for condition in sorted(grouped):
        group = grouped[condition]
        values = lambda key: np.asarray([float(row[key]) for row in group], dtype=np.float64)
        summaries.append(
            {
                "condition": condition,
                "policy": group[0]["policy"],
                "filter": group[0]["filter"],
                "episodes": len(group),
                "violation_free_episode_rate": float(np.mean(values("violation_free"))),
                "safe_step_rate": float(np.mean(values("safe_step_rate"))),
                "total_violation_steps": int(np.sum(values("violation_steps"))),
                "hard_deck_margin_min": float(np.min(values("hard_deck_margin_min"))),
                "hard_deck_margin_p05_across_episodes": float(np.percentile(values("hard_deck_margin_min"), 5.0)),
                "return_mean": float(np.mean(values("return"))),
                "return_std": float(np.std(values("return"))),
                "pos_xyz_rmse_mean": float(np.mean(values("pos_xyz_rmse"))),
                "vel_error_norm_rmse_mean": float(np.mean(values("vel_error_norm_rmse"))),
                "att_error_norm_rmse_mean": float(np.mean(values("att_error_norm_rmse"))),
                "intervention_norm_mean": float(np.mean(values("intervention_norm_mean"))),
                "intervention_norm_max": float(np.max(values("intervention_norm_max"))),
                "slack_mean": float(np.mean(values("slack_mean"))),
                "solver_rate": float(np.mean(values("solver_rate"))),
                "qp_finite_rate": float(np.mean(values("qp_finite_rate"))),
                "qp_fallback_steps": int(np.sum(values("qp_fallback_steps"))),
                "qp_fallback_rate": float(np.mean(values("qp_fallback_rate"))),
                "qp_fallback_invalid_inputs_steps": int(
                    np.sum(values("qp_fallback_invalid_inputs_steps"))
                ),
                "qp_fallback_invalid_inputs_rate": float(
                    np.mean(values("qp_fallback_invalid_inputs_rate"))
                ),
                "qp_fallback_nonfinite_solution_steps": int(
                    np.sum(values("qp_fallback_nonfinite_solution_steps"))
                ),
                "qp_fallback_nonfinite_solution_rate": float(
                    np.mean(values("qp_fallback_nonfinite_solution_rate"))
                ),
                "qp_fallback_unclassified_steps": int(
                    np.sum(values("qp_fallback_unclassified_steps"))
                ),
                "qp_fallback_unclassified_rate": float(
                    np.mean(values("qp_fallback_unclassified_rate"))
                ),
                "qp_q_mat_finite_rate": float(np.mean(values("qp_q_mat_finite_rate"))),
                "qp_q_vec_finite_rate": float(np.mean(values("qp_q_vec_finite_rate"))),
                "qp_g_finite_rate": float(np.mean(values("qp_g_finite_rate"))),
                "qp_h_finite_rate": float(np.mean(values("qp_h_finite_rate"))),
                "qp_inputs_finite_rate": float(np.mean(values("qp_inputs_finite_rate"))),
                "qp_z_finite_rate": float(np.mean(values("qp_z_finite_rate"))),
                "qp_q_saturated_rate": float(np.mean(values("qp_q_saturated_rate"))),
                "qp_post_warmup_steps": int(np.sum(values("qp_post_warmup_steps"))),
                "qp_post_warmup_fallback_steps": int(
                    np.sum(values("qp_post_warmup_fallback_steps"))
                ),
                "qp_post_warmup_fallback_rate": (
                    float(np.sum(values("qp_post_warmup_fallback_steps")))
                    / float(np.sum(values("qp_post_warmup_steps")))
                    if np.sum(values("qp_post_warmup_steps")) > 0
                    else 0.0
                ),
            }
        )
    return summaries


def _paired_filter_differences(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    index = {
        (row["policy"], row["episode_seed"], row["direction"], row["phase_deg"], row["filter"]): row
        for row in rows
    }
    metrics = (
        "violation_free",
        "violation_steps",
        "hard_deck_margin_min",
        "return",
        "pos_xyz_rmse",
        "vel_error_norm_rmse",
        "att_error_norm_rmse",
        "intervention_norm_mean",
        "slack_mean",
    )
    output: list[dict[str, Any]] = []
    base_keys = sorted({key[:4] for key in index})
    for policy, seed, direction, phase_deg in base_keys:
        standard = index.get((policy, seed, direction, phase_deg, "standard"))
        ue = index.get((policy, seed, direction, phase_deg, "ue"))
        if standard is None or ue is None:
            continue
        row: dict[str, Any] = {
            "policy": policy,
            "episode_seed": seed,
            "direction": direction,
            "phase_deg": phase_deg,
        }
        for metric in metrics:
            row[f"ue_minus_standard__{metric}"] = float(ue[metric]) - float(standard[metric])
        output.append(row)
    return output


def _plot_summary(summary_rows: list[dict[str, Any]], output_path: Path, *, complete: bool) -> None:
    if not summary_rows:
        return
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Warning: matplotlib unavailable; skipping comparison plot: {exc}")
        return

    labels = [str(row["condition"]) for row in summary_rows]
    x = np.arange(len(labels))
    colors = ["#4C78A8" if row["filter"] == "standard" else "#F58518" for row in summary_rows]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    axes[0, 0].bar(x, [row["violation_free_episode_rate"] for row in summary_rows], color=colors)
    axes[0, 0].set_ylim(0.0, 1.05)
    axes[0, 0].set_title("Violation-free episode rate (higher is better)")

    axes[0, 1].bar(x, [row["hard_deck_margin_min"] for row in summary_rows], color=colors)
    axes[0, 1].axhline(0.0, color="black", linewidth=1.0)
    axes[0, 1].set_title("Worst hard-deck margin (higher is better)")

    axes[1, 0].bar(x, [row["pos_xyz_rmse_mean"] for row in summary_rows], color=colors)
    axes[1, 0].set_title("Mean position RMSE (lower is better)")

    axes[1, 1].bar(x, [row["intervention_norm_mean"] for row in summary_rows], color=colors)
    axes[1, 1].set_title("Mean filter intervention norm")

    for axis in axes.ravel():
        axis.set_xticks(x)
        axis.set_xticklabels(labels, rotation=20, ha="right")
        axis.grid(True, axis="y", alpha=0.25)
    status = "complete" if complete else "in progress"
    fig.suptitle(f"Weak-disturbance paired bCBF comparison ({status})")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def _save_progress(output_dir: Path, rows: list[dict[str, Any]], *, complete: bool) -> list[dict[str, Any]]:
    summary_rows = _summarize(rows)
    _write_csv(output_dir / "results" / "episodes.csv", rows)
    _write_csv(output_dir / "results" / "summary.csv", summary_rows)
    _write_json(output_dir / "results" / "summary.json", summary_rows)
    _write_csv(output_dir / "comparisons" / "ue_minus_standard_paired.csv", _paired_filter_differences(rows))
    _plot_summary(summary_rows, output_dir / "plots" / "evaluation_progress.png", complete=complete)
    if complete:
        _plot_summary(summary_rows, output_dir / "plots" / "comparison.png", complete=True)
    return summary_rows


def _prepare_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir:
        output_dir = Path(args.output_dir).expanduser().resolve()
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        tag = f"weak_A{_float_tag(args.disturbance_amplitude)}_f{_float_tag(args.disturbance_frequency_hz)}"
        if args.run_tag:
            tag += f"_{args.run_tag.strip()}"
        output_dir = Path(args.output_root).expanduser().resolve() / f"{stamp}_{tag}"
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory is non-empty: {output_dir}\nUse a new path or pass --overwrite.")
    for child in ("metadata", "scenarios", "results", "comparisons", "trajectories/worst_cases", "plots"):
        (output_dir / child).mkdir(parents=True, exist_ok=True)
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nominal-run-dir", required=True)
    parser.add_argument("--ue-run-dir", required=True, help="Completed UE fine-tuning run containing configs and weights")
    parser.add_argument("--nominal-checkpoint", default="best")
    parser.add_argument("--ue-checkpoint", default="final")
    parser.add_argument(
        "--output-root", default="outputs/ue_bcbf_evaluation/01_50k_policy_filter_ablation"
    )
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--run-tag", default="")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--disturbance-amplitude", type=float, default=0.5)
    parser.add_argument("--disturbance-frequency-hz", type=float, default=0.05)
    parser.add_argument("--phases-deg", default="0,90,180,270")
    parser.add_argument("--repeats-per-direction", type=int, default=1)
    parser.add_argument("--seed-start", type=int, default=3_500_000)
    parser.add_argument("--seed-search-limit", type=int, default=10_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.disturbance_amplitude < 0.0 or not np.isfinite(args.disturbance_amplitude):
        raise ValueError("--disturbance-amplitude must be nonnegative and finite")
    if args.disturbance_frequency_hz < 0.0 or not np.isfinite(args.disturbance_frequency_hz):
        raise ValueError("--disturbance-frequency-hz must be nonnegative and finite")
    if args.repeats_per_direction <= 0:
        raise ValueError("--repeats-per-direction must be positive")

    phases_deg = _parse_csv_floats(args.phases_deg)
    output_dir = _prepare_output_dir(args)
    nominal_policy, nominal_json = _load_policy(
        "nominal", args.nominal_run_dir, args.nominal_checkpoint
    )
    ue_policy, ue_json = _load_policy("ue50k", args.ue_run_dir, args.ue_checkpoint)
    if "ue" not in ue_json:
        raise KeyError(f"UE config is missing from {_resolve_config_path(ue_policy.run_dir)}")

    # Use one common environment and bCBF configuration for all four cells.
    # This isolates policy and filter effects.  The UE run is authoritative
    # because it records the horizon/configuration actually used for fine-tuning.
    env_cfg_saved = eval_utils._dataclass_from_dict(QuadrotorEnvConfig, ue_json.get("env", {}))
    cbf_cfg = eval_utils._dataclass_from_dict(QuadrotorBCBFConfig, ue_json.get("cbf", {}))
    ue_cfg = eval_utils._dataclass_from_dict(ExperimentalUEConfig, ue_json.get("ue", {}))
    observer_warmup_sec = float(ue_json.get("ue_observer_warmup_sec", 0.2))
    env_cfg = replace(
        env_cfg_saved,
        disturbance_mode="sinusoidal" if args.disturbance_amplitude > 0.0 else "none",
        disturbance_amplitude=float(args.disturbance_amplitude),
        disturbance_frequency_hz=float(args.disturbance_frequency_hz),
        disturbance_direction_mode="axis_set7",
        terminate_on_violation=False,
    )

    selected_seeds = _find_balanced_episode_seeds(
        env_cfg,
        seed_start=int(args.seed_start),
        repeats=int(args.repeats_per_direction),
        search_limit=int(args.seed_search_limit),
    )
    scenario_rows = [
        {
            "direction": direction,
            "repeat_index": repeat_index,
            "episode_seed": episode_seed,
            "phase_deg": phase_deg,
            "phase_rad": phase_deg * pi / 180.0,
        }
        for phase_deg in phases_deg
        for direction in _DIRECTION_ORDER
        for repeat_index, episode_seed in enumerate(selected_seeds[direction])
    ]
    _write_csv(output_dir / "scenarios" / "scenario_matrix.csv", scenario_rows)

    conditions = _build_condition_runtimes(
        (nominal_policy, ue_policy), env_cfg, cbf_cfg, ue_cfg, observer_warmup_sec
    )
    manifest = {
        "created_at": datetime.now().astimezone().isoformat(),
        "kind": "paired_weak_disturbance_standard_vs_ue_bcbf",
        "command": shlex.join([sys.executable, *sys.argv]),
        "git": _git_metadata(),
        "source_fingerprints": _source_fingerprints(),
        "output_dir": str(output_dir),
        "policies": {
            policy.label: {
                "run_dir": str(policy.run_dir),
                "checkpoint": str(policy.checkpoint),
                "checkpoint_sha256": _sha256(policy.checkpoint),
                "configured_training_steps": int(policy.sac_cfg.total_steps),
            }
            for policy in (nominal_policy, ue_policy)
        },
        "common_environment": asdict(env_cfg),
        "common_bcbf": asdict(cbf_cfg),
        "ue": asdict(ue_cfg),
        "ue_observer_warmup_sec": observer_warmup_sec,
        "phases_deg": list(phases_deg),
        "directions": list(_DIRECTION_ORDER),
        "repeats_per_direction": int(args.repeats_per_direction),
        "selected_episode_seeds": selected_seeds,
        "nominal_config_path": str(_resolve_config_path(nominal_policy.run_dir)),
        "ue_config_path": str(_resolve_config_path(ue_policy.run_dir)),
        "nominal_config_sha256": _sha256(_resolve_config_path(nominal_policy.run_dir)),
        "ue_config_sha256": _sha256(_resolve_config_path(ue_policy.run_dir)),
        "note": "Empirical first-order UE tube; not a formal nonlinear robustness certificate.",
    }
    _write_json(output_dir / "metadata" / "manifest.json", manifest)
    (output_dir / "metadata" / "command.txt").write_text(manifest["command"] + "\n", encoding="utf-8")

    print("Paired weak-disturbance evaluation")
    print(f"output: {output_dir}")
    print(f"episodes per condition: {len(scenario_rows)}")
    print("conditions: " + ", ".join(condition.label for condition in conditions))

    all_rows: list[dict[str, Any]] = []
    worst: dict[str, tuple[tuple[int, float], dict[str, np.ndarray]]] = {}
    total_episodes = len(conditions) * len(scenario_rows)
    completed = 0

    for phase_deg in phases_deg:
        phase_rad = float(phase_deg) * pi / 180.0
        phase_env_cfg = replace(env_cfg, disturbance_phase=phase_rad)
        standard_env = build_quadrotor_env(phase_env_cfg)
        ue_env, _ = _build_ue_observer_env(phase_env_cfg, ue_cfg)

        for condition in conditions:
            env_fns = ue_env if condition.filter_name == "ue" else standard_env
            for direction in _DIRECTION_ORDER:
                for episode_seed in selected_seeds[direction]:
                    row, trajectory = _rollout_episode(
                        condition,
                        env_fns,
                        episode_seed=episode_seed,
                        phase_deg=phase_deg,
                        expected_direction=direction,
                        observer_warmup_sec=observer_warmup_sec,
                        env_dt=float(env_cfg.dt),
                    )
                    all_rows.append(row)
                    completed += 1
                    rank = (int(row["violation_steps"]), -float(row["hard_deck_margin_min"]))
                    previous = worst.get(condition.label)
                    if previous is None or rank > previous[0]:
                        worst[condition.label] = (rank, trajectory)
                    print(
                        f"[{completed:03d}/{total_episodes:03d}] {condition.label} "
                        f"dir={direction} phase={phase_deg:g} seed={episode_seed} "
                        f"safe={row['violation_free']:.0f} margin={row['hard_deck_margin_min']:.4f} "
                        f"pos_rmse={row['pos_xyz_rmse']:.4f} "
                        f"fallback={row['qp_fallback_rate']:.3f}"
                    )
            _save_progress(output_dir, all_rows, complete=False)

    for condition_label, (_, trajectory) in worst.items():
        np.savez(output_dir / "trajectories" / "worst_cases" / f"{condition_label}.npz", **trajectory)
    summary_rows = _save_progress(output_dir, all_rows, complete=True)
    _write_json(
        output_dir / "results" / "run_complete.json",
        {"completed": True, "episodes": len(all_rows), "conditions": len(summary_rows)},
    )

    print("\nSummary")
    for row in summary_rows:
        print(
            f"{row['condition']}: violation_free={row['violation_free_episode_rate']:.3f} "
            f"min_margin={row['hard_deck_margin_min']:.4f} "
            f"pos_rmse={row['pos_xyz_rmse_mean']:.4f} "
            f"intervention={row['intervention_norm_mean']:.4f} "
            f"fallback={row['qp_fallback_rate']:.3f} "
            f"invalid_inputs={row['qp_fallback_invalid_inputs_rate']:.3f} "
            f"nonfinite_solution={row['qp_fallback_nonfinite_solution_rate']:.3f}"
        )
    print(f"\nSaved results to: {output_dir}")
    print(f"Comparison plot: {output_dir / 'plots' / 'comparison.png'}")


if __name__ == "__main__":
    main()
