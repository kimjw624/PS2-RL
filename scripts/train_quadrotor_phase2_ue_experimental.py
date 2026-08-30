#!/usr/bin/env python3
"""Experimental UE-bCBF Phase-2 fine-tuning for the quadrotor.

The run warm-starts from an existing Phase-2 checkpoint, injects a weak 3-D
sinusoidal translational disturbance, and trains through the full experimental
UE projector (including quadratic terminal tightening by default).

Artifacts are deliberately organized for later analysis while preserving the
root-level files expected by the existing PS2-RL evaluators.

The first-order UE tube remains empirical and is not yet a formal nonlinear
certificate.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime
import json
from math import pi
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
from typing import Any

import numpy as np

from ps2rl.cil.quadrotor_backup_cbf import QuadrotorBCBFConfig
from ps2rl.cil.quadrotor_ue_bcbf_experimental import ExperimentalUEConfig
from ps2rl.envs.quadrotor_env import QuadrotorEnvConfig
from ps2rl.evaluation import quadrotor_vanilla_eval as eval_utils
from ps2rl.phase2_ps2.quadrotor_ps2_trainer import SACConfig, run_ue_training
from ps2rl.plotting.plots import plot_quad_trajectory


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


def _float_tag(value: float, digits: int = 2) -> str:
    return f"{float(value):.{digits}f}".replace("-", "m").replace(".", "p")


def _steps_tag(steps: int) -> str:
    steps = int(steps)
    if steps % 1_000_000 == 0:
        return f"{steps // 1_000_000}m"
    if steps % 1_000 == 0:
        return f"{steps // 1_000}k"
    return str(steps)


def _direction_tag(mode: str) -> str:
    return {"axis_set7": "axis7", "random_unit": "rand3d", "fixed": "fixed"}.get(mode, mode)


def _default_run_name(args: argparse.Namespace) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    core = (
        f"{args.stage}_A{_float_tag(args.disturbance_amplitude)}"
        f"_f{_float_tag(args.disturbance_frequency_hz)}"
        f"_{_direction_tag(args.disturbance_direction_mode)}"
        f"_T{_float_tag(args.ue_horizon)}"
        f"_tube{_float_tag(args.ue_tube_scale)}"
        f"_seed{int(args.seed)}"
        f"_{_steps_tag(args.total_steps)}"
    )
    if args.run_tag.strip():
        core += f"_{args.run_tag.strip()}"
    return f"{stamp}_{core}"


def _prepare_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir:
        out = Path(args.output_dir).expanduser().resolve()
    else:
        out = Path(args.output_root).expanduser().resolve() / _default_run_name(args)
    if out.exists() and any(out.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Output directory already exists and is non-empty: {out}\n"
            "Use a new run name/output path, or pass --overwrite intentionally."
        )
    out.mkdir(parents=True, exist_ok=True)
    for child in ("checkpoints", "training", "evaluation/best", "evaluation/final", "plots", "metadata"):
        (out / child).mkdir(parents=True, exist_ok=True)
    return out


def _git_metadata(repo_root: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        try:
            proc = subprocess.run(
                ["git", *args],
                cwd=repo_root,
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
            return proc.stdout.strip()
        except Exception:
            return ""

    status = run("status", "--short")
    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "dirty": bool(status),
        "status_short": status.splitlines(),
    }


def _json_dump(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _save_npz(path: Path, payload: dict[str, Any]) -> None:
    np.savez(path, **{k: np.asarray(v) for k, v in payload.items()})


def _eval_without_trajectory(eval_stats: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in eval_stats.items() if k != "trajectory"}


def _plot_training_history(history: dict[str, Any], eval_every: int, output_path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Warning: matplotlib unavailable; skipping training plot: {exc}")
        return

    def arr(key: str) -> np.ndarray:
        return np.asarray(history.get(key, []), dtype=np.float64)

    fig, axes = plt.subplots(3, 2, figsize=(12, 11))
    ax = axes.ravel()

    ep_step = arr("step")
    ep_ret = arr("ep_return")
    if ep_step.size == ep_ret.size:
        ax[0].plot(ep_step, ep_ret, linewidth=1.0)
    ax[0].set_title("Episode Return")
    ax[0].set_xlabel("Environment step")
    ax[0].grid(True, alpha=0.3)

    ep_safe = arr("ep_safe_rate")
    ep_margin = arr("ep_hard_deck_margin_min")
    if ep_step.size == ep_safe.size:
        ax[1].plot(ep_step, ep_safe, label="safe rate", linewidth=1.0)
    if ep_step.size == ep_margin.size:
        ax[1].plot(ep_step, ep_margin, label="min hard-deck margin", linewidth=1.0)
    ax[1].set_title("Episode Safety")
    ax[1].set_xlabel("Environment step")
    ax[1].grid(True, alpha=0.3)
    if ax[1].lines:
        ax[1].legend()

    critic = arr("critic_loss")
    actor = arr("actor_loss")
    if critic.size:
        ax[2].plot(np.arange(critic.size), critic, label="critic", linewidth=1.0)
    if actor.size:
        ax[2].plot(np.arange(actor.size), actor, label="actor", linewidth=1.0)
    ax[2].set_title("SAC Losses")
    ax[2].set_xlabel("Logged update chunk")
    ax[2].grid(True, alpha=0.3)
    if ax[2].lines:
        ax[2].legend()

    alpha = arr("alpha")
    slack = arr("slack_mean")
    if alpha.size:
        ax[3].plot(np.arange(alpha.size), alpha, label="alpha", linewidth=1.0)
    if slack.size:
        ax[3].plot(np.arange(slack.size), slack, label="mean slack", linewidth=1.0)
    ax[3].set_title("Entropy / QP")
    ax[3].set_xlabel("Logged update chunk")
    ax[3].grid(True, alpha=0.3)
    if ax[3].lines:
        ax[3].legend()

    eval_ret = arr("eval_return_mean")
    eval_safe = arr("eval_violation_free_episode_rate")
    eval_step = arr("eval_step")
    eval_x = eval_step if eval_step.size == eval_ret.size else np.arange(eval_ret.size, dtype=np.float64) * float(eval_every)
    if eval_ret.size:
        ax[4].plot(eval_x, eval_ret, label="return", linewidth=1.0)
    if eval_safe.size == eval_ret.size:
        ax[4].plot(eval_x, eval_safe, label="violation-free rate", linewidth=1.0)
    ax[4].set_title("Periodic Evaluation")
    ax[4].set_xlabel("Environment step")
    ax[4].grid(True, alpha=0.3)
    if ax[4].lines:
        ax[4].legend()

    pos = arr("eval_pos_xyz_rmse_mean")
    vel = arr("eval_vel_xz_rmse_mean")
    pitch = arr("eval_pitch_rmse_deg_mean")
    n_eval = max(pos.size, vel.size, pitch.size)
    eval_x2 = eval_step if eval_step.size == n_eval else np.arange(n_eval, dtype=np.float64) * float(eval_every)
    if pos.size:
        ax[5].plot(eval_x2[: pos.size], pos, label="pos xyz RMSE", linewidth=1.0)
    if vel.size:
        ax[5].plot(eval_x2[: vel.size], vel, label="vel xz RMSE", linewidth=1.0)
    if pitch.size:
        ax[5].plot(eval_x2[: pitch.size], pitch, label="pitch RMSE [deg]", linewidth=1.0)
    ax[5].set_title("Evaluation Tracking")
    ax[5].set_xlabel("Environment step")
    ax[5].grid(True, alpha=0.3)
    if ax[5].lines:
        ax[5].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _save_eval_artifacts(
    run_dir: Path,
    label: str,
    eval_stats: dict[str, Any],
    *,
    z_max: float,
    dt: float,
) -> None:
    out = run_dir / "evaluation" / label
    out.mkdir(parents=True, exist_ok=True)
    trajectory = eval_stats.get("trajectory", {})
    _json_dump(out / "summary.json", _eval_without_trajectory(eval_stats))
    if trajectory:
        _save_npz(out / "trajectory.npz", trajectory)
        try:
            plot_quad_trajectory(
                trajectory,
                z_max=float(z_max),
                output_path=str(out / "trajectory.png"),
                dt=float(dt),
            )
        except Exception as exc:
            print(f"Warning: failed to plot {label} trajectory: {exc}")


def _organize_artifacts(
    run_dir: Path,
    result: dict[str, Any],
    *,
    eval_every: int,
    env_cfg: QuadrotorEnvConfig,
    manifest: dict[str, Any],
) -> None:
    history = result["history"]
    final_eval = result["eval"]
    best_eval_full = result.get("best_eval_full", final_eval)

    # Compatibility files used by the existing PS2-RL tooling.
    _save_npz(run_dir / "history.npz", history)
    _save_npz(run_dir / "eval_trajectory.npz", final_eval.get("trajectory", {}))
    _save_npz(run_dir / "best_eval_trajectory.npz", best_eval_full.get("trajectory", {}))

    # Structured copies for long-term comparison/analysis.
    _save_npz(run_dir / "training" / "history.npz", history)
    _json_dump(run_dir / "training" / "summary.json", result["summary"])
    _plot_training_history(history, eval_every=int(eval_every), output_path=run_dir / "plots" / "training_curves.png")

    _save_eval_artifacts(run_dir, "final", final_eval, z_max=env_cfg.z_max, dt=env_cfg.dt)
    _save_eval_artifacts(run_dir, "best", best_eval_full, z_max=env_cfg.z_max, dt=env_cfg.dt)

    best_plot = run_dir / "evaluation" / "best" / "trajectory.png"
    if best_plot.exists():
        shutil.copy2(best_plot, run_dir / "best_trajectory.png")

    for name in ("best_weights.pkl", "final_weights.pkl"):
        src = run_dir / name
        if src.exists():
            shutil.copy2(src, run_dir / "checkpoints" / name)

    _json_dump(run_dir / "metadata" / "run_manifest.json", manifest)
    (run_dir / "metadata" / "command.txt").write_text(manifest["command"] + "\n", encoding="utf-8")
    (run_dir / "README.txt").write_text(
        "UE Phase-2 run layout\n"
        "=====================\n"
        "Root-level configs.json, summary.json, best_weights.pkl and final_weights.pkl\n"
        "are retained for compatibility with existing PS2-RL evaluation scripts.\n\n"
        "checkpoints/       portable copies of best/final trained policies\n"
        "training/          history.npz and training summary\n"
        "evaluation/best/   best-checkpoint eval summary, trajectory and plot\n"
        "evaluation/final/  final-checkpoint eval summary, trajectory and plot\n"
        "plots/             training learning-curve figure\n"
        "metadata/          exact command, source checkpoint and git state\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True, help="Nominal/previous Phase-2 run used for warm start/configs")
    p.add_argument("--checkpoint", default="best", help="best, final, or a checkpoint filename")
    p.add_argument("--output-root", default="outputs/ue_phase2_finetune")
    p.add_argument("--output-dir", default="", help="Exact run directory override; otherwise an ordered name is generated")
    p.add_argument("--run-tag", default="", help="Optional short suffix for the generated run name")
    p.add_argument("--stage", choices=("smoke", "train"), default="smoke")
    p.add_argument("--overwrite", action="store_true", help="Allow writing into an existing non-empty output directory")

    p.add_argument("--total-steps", type=int, default=100_000)
    p.add_argument("--num-envs", type=int, default=8)
    p.add_argument("--steps-per-jit", type=int, default=32)
    p.add_argument("--seed", type=int, default=3_500_000)
    p.add_argument("--eval-every", type=int, default=10_000)
    p.add_argument("--eval-episodes", type=int, default=3)

    p.add_argument("--disturbance-amplitude", type=float, default=0.5)
    p.add_argument("--disturbance-frequency-hz", type=float, default=0.05)
    p.add_argument("--disturbance-phase", type=float, default=pi / 2.0)
    p.add_argument(
        "--disturbance-direction-mode",
        choices=("axis_set7", "random_unit", "fixed"),
        default="axis_set7",
        help="axis_set7 matches the x/y/z/xy/xz/yz/xyz diagnostic sweep",
    )

    p.add_argument("--ue-horizon", type=float, default=1.0)
    p.add_argument("--ue-tube-scale", type=float, default=1.2)
    p.add_argument("--ue-rho-scale", type=float, default=1.0)
    p.add_argument("--ue-safe-rho-scale", type=float, default=1.0)
    p.add_argument("--ue-terminal-rho-scale", type=float, default=1.0)
    p.add_argument("--ue-observer-lambda", type=float, default=20.0)
    p.add_argument("--ue-observer-warmup-sec", type=float, default=0.2)
    p.add_argument("--ue-terminal-mode", choices=("quadratic", "nominal"), default="quadratic")
    p.add_argument(
        "--ue-compute-radius-diagnostics",
        action="store_true",
        help=(
            "Compute expensive full-state spectral-norm radius diagnostics during training. "
            "Disabled by default; the actual UE QP support margins are unchanged."
        ),
    )

    target_group = p.add_mutually_exclusive_group()
    target_group.add_argument(
        "--project-target-actions",
        action="store_true",
        help="Force UE projection of SAC target actions.",
    )
    target_group.add_argument(
        "--no-project-target-actions",
        action="store_true",
        help="Disable target-action projection for a faster ablation.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    output_dir = _prepare_output_dir(args)

    cfg_json = eval_utils._load_json(_resolve_config_path(run_dir))
    sac_saved = eval_utils._dataclass_from_dict(SACConfig, cfg_json.get("sac", {}))
    env_saved = eval_utils._dataclass_from_dict(QuadrotorEnvConfig, cfg_json.get("env", {}))
    cbf_saved = eval_utils._dataclass_from_dict(QuadrotorBCBFConfig, cfg_json.get("cbf", {}))
    warm_weights = _weights_path(run_dir, str(args.checkpoint))

    target_projection = bool(sac_saved.project_target_actions)
    if args.project_target_actions:
        target_projection = True
    elif args.no_project_target_actions:
        target_projection = False

    # Fine-tune immediately from the existing policy. Replay is intentionally
    # rebuilt under the UE-filtered environment rather than reusing nominal
    # transitions.
    sac_cfg = replace(
        sac_saved,
        seed=int(args.seed),
        total_steps=int(args.total_steps),
        start_steps=0,
        num_envs=int(args.num_envs),
        steps_per_jit=int(args.steps_per_jit),
        eval_every=int(args.eval_every),
        eval_episodes=int(args.eval_episodes),
        warm_start=True,
        warm_start_weights=str(warm_weights),
        project_actor_actions=True,
        project_target_actions=target_projection,
    )

    amp = float(args.disturbance_amplitude)
    freq = float(args.disturbance_frequency_hz)
    delta_v = 2.0 * pi * freq * amp
    env_cfg = replace(
        env_saved,
        disturbance_mode="sinusoidal" if amp > 0.0 else "none",
        disturbance_amplitude=amp,
        disturbance_frequency_hz=freq,
        disturbance_phase=float(args.disturbance_phase),
        disturbance_direction_x=1.0,
        disturbance_direction_y=0.0,
        disturbance_direction_z=0.0,
        disturbance_direction_mode=str(args.disturbance_direction_mode),
        terminate_on_violation=False,
    )

    n_steps = int(round(float(args.ue_horizon) / float(cbf_saved.dt)))
    if n_steps <= 0:
        raise ValueError("--ue-horizon must produce at least one backup step")
    cbf_cfg = replace(cbf_saved, T=n_steps * float(cbf_saved.dt), num_steps=n_steps)

    ue_cfg = ExperimentalUEConfig(
        delta_d=amp,
        delta_v=delta_v,
        observer_lambda=float(args.ue_observer_lambda),
        tube_scale=float(args.ue_tube_scale),
        rho_scale=float(args.ue_rho_scale),
        safe_rho_scale=float(args.ue_safe_rho_scale),
        terminal_rho_scale=float(args.ue_terminal_rho_scale),
        terminal_mode=str(args.ue_terminal_mode),
        compute_radius_diagnostics=bool(args.ue_compute_radius_diagnostics),
    )

    repo_root = Path(__file__).resolve().parents[1]
    manifest = {
        "created_at": datetime.now().astimezone().isoformat(),
        "kind": "experimental_ue_phase2_finetune",
        "stage": str(args.stage),
        "output_dir": str(output_dir),
        "warm_start_run": str(run_dir),
        "warm_start_checkpoint": str(warm_weights),
        "command": shlex.join([sys.executable, *sys.argv]),
        "git": _git_metadata(repo_root),
        "design": {
            "disturbance_amplitude": amp,
            "disturbance_frequency_hz": freq,
            "delta_v": delta_v,
            "disturbance_direction_mode": env_cfg.disturbance_direction_mode,
            "ue_horizon_sec": cbf_cfg.T,
            "ue_num_steps": cbf_cfg.num_steps,
            "ue_tube_scale": ue_cfg.tube_scale,
            "ue_terminal_mode": ue_cfg.terminal_mode,
            "ue_rho_scale": ue_cfg.rho_scale,
            "ue_safe_rho_scale": ue_cfg.safe_rho_scale,
            "ue_terminal_rho_scale": ue_cfg.terminal_rho_scale,
            "observer_lambda": ue_cfg.observer_lambda,
            "observer_warmup_sec": float(args.ue_observer_warmup_sec),
            "compute_radius_diagnostics": ue_cfg.compute_radius_diagnostics,
            "project_actor_actions": sac_cfg.project_actor_actions,
            "project_target_actions": sac_cfg.project_target_actions,
        },
        "note": "Empirical first-order UE tube; not yet a formal nonlinear certificate.",
    }
    _json_dump(output_dir / "metadata" / "run_manifest_pretrain.json", manifest)

    print("\nExperimental UE Phase-2 fine-tuning")
    print(f"warm start: {warm_weights}")
    print(f"output: {output_dir}")
    print(f"steps={sac_cfg.total_steps} num_envs={sac_cfg.num_envs} steps_per_jit={sac_cfg.steps_per_jit}")
    print(f"disturbance: A={amp:.3f}, f={freq:.3f} Hz, delta_v={delta_v:.6f}")
    print(f"direction mode: {env_cfg.disturbance_direction_mode}")
    print(f"T={cbf_cfg.T:.3f}s N={cbf_cfg.num_steps} dt={cbf_cfg.dt:.3f}s")
    print(f"tube scale={ue_cfg.tube_scale:.3f} terminal mode={ue_cfg.terminal_mode}")
    print(f"rho scales={ue_cfg.rho_scale:.3f}/{ue_cfg.safe_rho_scale:.3f}/{ue_cfg.terminal_rho_scale:.3f}")
    print(f"observer lambda={ue_cfg.observer_lambda:.3f}, warmup={args.ue_observer_warmup_sec:.3f}s")
    print(f"project_actor_actions={sac_cfg.project_actor_actions}, project_target_actions={sac_cfg.project_target_actions}")
    print(f"full-radius diagnostics={ue_cfg.compute_radius_diagnostics}")
    print("NOTE: empirical first-order UE tube; not yet a formal nonlinear certificate.\n")

    result = run_ue_training(
        sac_cfg,
        env_cfg,
        cbf_cfg,
        ue_cfg,
        observer_warmup_sec=float(args.ue_observer_warmup_sec),
        output_dir=str(output_dir),
    )

    manifest["completed_at"] = datetime.now().astimezone().isoformat()
    manifest["summary"] = result["summary"]
    _organize_artifacts(
        output_dir,
        result,
        eval_every=sac_cfg.eval_every,
        env_cfg=env_cfg,
        manifest=manifest,
    )

    print("\n=== UE fine-tuning summary ===")
    print(json.dumps(result["summary"], indent=2))
    print(f"\nOrganized run saved to: {output_dir}")
    print("Key artifacts:")
    print(f"  best policy:       {output_dir / 'checkpoints' / 'best_weights.pkl'}")
    print(f"  final policy:      {output_dir / 'checkpoints' / 'final_weights.pkl'}")
    print(f"  training curves:   {output_dir / 'plots' / 'training_curves.png'}")
    print(f"  best trajectory:   {output_dir / 'evaluation' / 'best' / 'trajectory.png'}")
    print(f"  final trajectory:  {output_dir / 'evaluation' / 'final' / 'trajectory.png'}")
    print(f"  manifest:          {output_dir / 'metadata' / 'run_manifest.json'}")


if __name__ == "__main__":
    main()
