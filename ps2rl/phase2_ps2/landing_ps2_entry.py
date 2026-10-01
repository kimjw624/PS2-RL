"""Phase-II landing training: a tracker for a landing reference, alone or through the landing CIL.

Two modes, the same environment and the same starts:

  * ``--projection off``: a plain tracker (SAC, no safety layer), the way the landing trackers
    in ``outputs/landing/land_v2_*`` were trained (landing reward v2, 8M steps);
  * ``--projection on``: the PS2 policy, trained end-to-end through the landing control-invariant
    layer (BCBF-QP + discrete-time safeguard, Phase-I backup ``--backup``), usually warm-started
    from the tracker as in the paper.

Everything geometric comes from the Phase-I checkpoint (cone, pad, floor, base set, backup, dt,
action box) and is checked against the reference's ``_config.json``. The environment is the
repo's quadrotor env with ``reward_mode="landing_v2"`` and ``safety_mode="landing"`` (its safety
margin is the cone/floor value, so every logged ``hard_deck_*`` metric is the landing margin).

Starts: candidates are drawn around the reference start with the ``--init_*`` ranges and kept
only if they lie in the safe set with margin ``--init_start_margin`` and are backup-recoverable
(in C_N). The kept states form the run's reset library (``init_states.npz``); both modes use it,
so the tracker and the PS2 policy see the same start distribution, and every PS2 episode starts
where the CIL guarantee applies.

    python scripts/train_phase2_landing.py --reference ps2rl/envs/assets/quadrotor_landing_skid_reference.npz \\
        --projection off --run_name skid_tracker
    python scripts/train_phase2_landing.py --reference ps2rl/envs/assets/quadrotor_landing_skid_reference.npz \\
        --projection on --warm_start_weights outputs/landing_phase2/skid_tracker/best_weights.pkl --run_name skid_ps2
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# SAC presets (every value can be overridden on the command line):
#   "tracker"   the settings of the landing trackers in outputs/landing/land_v2_*;
#   "ps2_paper" the paper's quadrotor PS2 settings (warm start, no random warm-up, smaller learning
#               rates, projected critic targets), 32 envs, an update of 64 samples every 8 env steps;
#   "ps2"       the same, batched for a GPU: 256 envs, an update of 512 samples every 64 env steps.
#               Same samples-per-env-step ratio (8) and the same number of QP calls per vector step,
#               so each vector step costs about as long while it collects 8x the env steps. The
#               projection is latency-bound on a GPU (sequential 100-step backup rollouts and QP
#               iterations on tiny batches), which is why the 32-env setting ran at ~20 steps/s.
PRESETS: dict[str, dict[str, Any]] = {
    "tracker": dict(total_steps=8_000_000, start_steps=4_000, update_after=2_000, update_every=8, gradient_steps=1,
                    batch_size=64, replay_size=300_000, gamma=0.99, tau=0.005, actor_lr=1e-4, critic_lr=1e-4,
                    alpha_lr=1e-4, init_alpha=0.2, min_alpha=0.1, target_entropy=-4.0, max_grad_norm=5.0,
                    q_clip_abs=1e4, hidden_size=256, eval_every=50_000, eval_episodes=3, log_every=10_000,
                    best_weights_save_period=100_000, num_envs=32, steps_per_jit=128, project_target_actions=False),
    "ps2": dict(total_steps=1_500_000, start_steps=0, update_after=1_024, update_every=8, gradient_steps=1,
                batch_size=64, replay_size=300_000, gamma=0.99, tau=0.005, actor_lr=5e-5, critic_lr=1e-4,
                alpha_lr=5e-5, init_alpha=0.2, min_alpha=1e-2, target_entropy=-4.0, max_grad_norm=5.0,
                q_clip_abs=1e4, hidden_size=256, eval_every=25_000, eval_episodes=3, log_every=2_000,
                best_weights_save_period=50_000, num_envs=32, steps_per_jit=16, project_target_actions=True),
}
PRESETS["ps2_paper"] = PRESETS.pop("ps2")
PRESETS["ps2"] = dict(PRESETS["ps2_paper"], total_steps=3_000_000, update_after=4_096, update_every=64, batch_size=512,
                      replay_size=500_000, num_envs=256, steps_per_jit=4, log_every=8_192, eval_every=65_536,
                      best_weights_save_period=131_072)
DEFAULT_FILTER = "alpha=10,alpha_floor=20,control_weights=1/0.1/0.1/1"
REFERENCE_GEOMETRY = {"pad_x": "pad_x", "pad_y": "pad_y", "z_pad": "pad_z", "r0": "cone_r0",
                      "theta_deg": "cone_theta_deg", "eps": "cone_eps"}


def _bool(v: str) -> bool:
    low = str(v).strip().lower()
    if low not in ("1", "0", "true", "false", "yes", "no", "on", "off"):
        raise argparse.ArgumentTypeError(f"expected on/off, got {v!r}")
    return low in ("1", "true", "yes", "on")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--reference", required=True, help="landing reference .npz (with its _config.json next to it)")
    p.add_argument("--backup", default="checkpoints/landing_phase1/floor_rec10_td3_seed0", help="Phase-I checkpoint")
    p.add_argument("--projection", type=_bool, required=True, help="on: train through the landing CIL (PS2); off: tracker")
    p.add_argument("--filter", default=DEFAULT_FILTER,
                   help="landing CIL settings KEY=VAL,... (QuadrotorLandingBCBFConfig fields; tuples '/'-separated)")
    p.add_argument("--qp_solve_dtype", choices=("float32", "float64"), default="float32",
                   help="float64 needs JAX_ENABLE_X64=1, which also makes the networks float64 (slow on consumer GPUs)")
    p.add_argument("--preset", choices=tuple(PRESETS), default=None, help="SAC preset (default: ps2 if projection on)")
    p.add_argument("--warm_start_weights", default="", help="checkpoint to initialise actor/critics from")
    p.add_argument("--wall_time_budget_hours", type=float, default=0.0, help="stop training after this long (0 = off)")
    p.add_argument("--safeguard_in_updates", type=_bool, default=False,
                   help="also apply the discrete-time safeguard inside the SAC update's projections (actor/critic "
                        "targets). Off: updates use the QP projection only; executed actions and evaluation always "
                        "use QP + safeguard, so safety is unchanged and each update skips 5 extra backup rollouts.")
    for k, v in PRESETS["tracker"].items():
        if isinstance(v, bool):
            p.add_argument(f"--{k}", type=_bool, default=None)
        else:
            p.add_argument(f"--{k}", type=type(v), default=None)
    p.add_argument("--seed", type=int, default=0)
    # environment
    p.add_argument("--reward_mode", default="landing_v2", choices=("landing_v2", "trajectory_following"))
    p.add_argument("--max_steps_extra_sec", type=float, default=0.5, help="episode = reference + this much hold")
    p.add_argument("--terminate_on_violation", type=_bool, default=False)
    p.add_argument("--env_override", action="append", default=[], metavar="KEY=VAL",
                   help="any other QuadrotorEnvConfig field, e.g. landing_reward_w_z=2 (repeatable)")
    # starts
    p.add_argument("--init_px_range", type=float, default=0.3)
    p.add_argument("--init_py_range", type=float, default=0.3)
    p.add_argument("--init_pz_range", type=float, default=0.3)
    p.add_argument("--init_v_range", type=float, default=0.5)
    p.add_argument("--init_tilt_deg_range", type=float, default=10.0)
    p.add_argument("--init_yaw_deg_range", type=float, default=0.0)
    p.add_argument("--init_library_size", type=int, default=4096)
    p.add_argument("--init_start_margin", type=float, default=0.02, help="starts need safe-set margin >= this [m]")
    p.add_argument("--init_library", default="", help="use this reset library instead of building one")
    # output
    p.add_argument("--out_root", default="outputs/landing_phase2")
    p.add_argument("--run_name", required=True)
    return p.parse_args(argv)


def _resolve(path: str) -> Path:
    q = Path(path).expanduser()
    if q.is_absolute():
        return q
    for base in (Path.cwd(), PROJECT_ROOT):
        if (base / q).exists():
            return (base / q).resolve()
    return (PROJECT_ROOT / q).resolve()


def reference_geometry_mismatch(reference: Path, landing_cfg) -> list[str]:
    """Differences between the reference's design cone/pad and the backup's (empty = consistent)."""
    cfg_path = reference.with_name(reference.stem + "_config.json")
    if not cfg_path.exists():
        return [f"no {cfg_path.name} next to the reference; cannot check the cone geometry"]
    lc = json.loads(cfg_path.read_text()).get("landing_config", {})
    bad = []
    for k_ref, k_cfg in REFERENCE_GEOMETRY.items():
        if k_ref in lc and not math.isclose(float(lc[k_ref]), float(getattr(landing_cfg, k_cfg)), rel_tol=1e-6, abs_tol=1e-9):
            bad.append(f"{k_ref}: reference {lc[k_ref]} vs backup {getattr(landing_cfg, k_cfg)}")
    return bad


def sample_starts(ref_start: np.ndarray, n: int, ranges: dict[str, float], rng: np.random.Generator) -> np.ndarray:
    """Perturbations of the reference start, the same way the env samples them."""
    from scipy.spatial.transform import Rotation

    x = np.repeat(ref_start[None].astype(np.float64), n, axis=0)
    for i, k in enumerate(("init_px_range", "init_py_range", "init_pz_range")):
        x[:, i] += rng.uniform(-ranges[k], ranges[k], n)
    x[:, 3:6] += rng.uniform(-ranges["init_v_range"], ranges["init_v_range"], (n, 3))
    tilt = np.radians(ranges["init_tilt_deg_range"])
    yaw = np.radians(ranges["init_yaw_deg_range"])
    roll, pitch, yw = rng.uniform(-tilt, tilt, n), rng.uniform(-tilt, tilt, n), rng.uniform(-yaw, yaw, n)
    # env: q = q_perturb(roll, pitch, yaw; ZYX) * q0, scalar-first
    q_pert = Rotation.from_euler("ZYX", np.stack([yw, pitch, roll], 1))
    q0 = Rotation.from_quat(np.roll(ref_start[6:10], -1))
    q = (q_pert * q0).as_quat()  # scalar-last
    x[:, 6:10] = np.roll(q, 1, axis=1)
    x[:, 6:10] *= np.where(x[:, 6:7] < 0, -1.0, 1.0)
    return x


def build_init_library(cbf_cfg, runtime, reference: Path, args, out_path: Path) -> dict[str, Any]:
    import jax.numpy as jnp

    from ps2rl.cil import quadrotor_landing_backup_cbf as lbcbf

    ref = np.load(reference)["states"]
    ranges = {k: float(getattr(args, k)) for k in ("init_px_range", "init_py_range", "init_pz_range", "init_v_range",
                                                     "init_tilt_deg_range", "init_yaw_deg_range")}
    rng = np.random.default_rng(args.seed + 12345)
    rec_fn = lbcbf.make_recoverability_fn(cbf_cfg, runtime)
    keep, tried, chunk = [], 0, max(1024, args.init_library_size)
    while sum(len(k) for k in keep) < args.init_library_size and tried < 50 * args.init_library_size:
        cand = sample_starts(ref[0], chunk, ranges, rng)
        tried += chunk
        cj = jnp.asarray(cand, dtype=jnp.float32)
        margin = np.asarray(lbcbf.safe_value(cj, cbf_cfg)) if hasattr(lbcbf, "safe_value") else np.asarray(lbcbf.cone_value(cj, cbf_cfg))
        ok = margin >= args.init_start_margin
        ok &= np.asarray(rec_fn(cj)[0])
        keep.append(cand[ok])
    lib = np.concatenate(keep, axis=0)[: args.init_library_size]
    if len(lib) == 0:
        raise SystemExit("no start passed the safe-set / recoverability filter; check the reference start and the ranges")
    acc = len(np.concatenate(keep)) / max(1, tried)
    np.savez(out_path, states=lib.astype(np.float32), ranges=json.dumps(ranges), start_margin=args.init_start_margin,
             acceptance=acc)
    return {"path": str(out_path), "size": int(len(lib)), "acceptance": float(acc), "ranges": ranges}


def main(argv=None):
    args = parse_args(argv)
    import jax

    from ps2rl.cil import quadrotor_landing_backup_cbf as lbcbf
    from ps2rl.envs.quadrotor_env import QuadrotorEnvConfig
    from ps2rl.evaluation.landing_phase2_run import parse_field_overrides
    from ps2rl.phase2_ps2.landing_ps2_binding import LANDING_BINDING
    from ps2rl.phase2_ps2.ps2_trainer_core import SACConfig
    from ps2rl.phase2_ps2.quadrotor_ps2_trainer import run_training

    if args.qp_solve_dtype == "float64" and not jax.config.jax_enable_x64:
        raise SystemExit("--qp_solve_dtype float64 needs JAX_ENABLE_X64=1")
    reference = _resolve(args.reference)
    backup = _resolve(args.backup)
    run_dir = _resolve(args.out_root) / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    filt = parse_field_overrides([t for t in args.filter.split(",") if t.strip()], lbcbf.QuadrotorLandingBCBFConfig,
                                 extra={"floor_constraint": bool})
    cbf_cfg = lbcbf.landing_bcbf_config_from_checkpoint(str(backup), qp_solve_dtype=args.qp_solve_dtype, **filt)
    lc = cbf_cfg.landing
    bad = reference_geometry_mismatch(reference, lc)
    if bad:
        raise SystemExit("reference and backup disagree on the landing geometry:\n  " + "\n  ".join(bad))
    if not lc.floor_constraint:
        print("[warn] the backup's safe set has no floor; add floor_constraint=on to --filter to add the pad plane")
    projector = lbcbf.QuadrotorLandingBackupCBFProjector(cbf_cfg)

    # reset library (same seed/ranges/backup -> same library in both modes)
    if args.init_library:
        lib_info = {"path": str(_resolve(args.init_library)), "given": True}
    else:
        t0 = time.time()
        lib_info = build_init_library(cbf_cfg, projector.runtime, reference, args, run_dir / "init_states.npz")
        print(f"[starts] reset library: {lib_info['size']} states (acceptance {lib_info['acceptance']:.2f}, "
              f"{time.time() - t0:.0f} s) -> {lib_info['path']}", flush=True)

    env_extra = parse_field_overrides(args.env_override, QuadrotorEnvConfig) if args.env_override else {}
    env_cfg = QuadrotorEnvConfig(**{
        "dt": lc.dt, "max_steps_extra_sec": args.max_steps_extra_sec, "backup_horizon_T": lc.T,
        "gravity": lc.gravity, "a_cmd_min": lc.a_cmd_min, "a_cmd_max": lc.a_cmd_max, "omega_max": lc.omega_max,
        "z_max": 15.0, "terminate_on_violation": args.terminate_on_violation, "reward_mode": args.reward_mode,
        "reference_path": str(reference), "reference_dt": lc.dt,
        "safety_mode": "landing", "landing_pad_x": lc.pad_x, "landing_pad_y": lc.pad_y, "landing_pad_z": lc.pad_z,
        "landing_cone_r0": lc.cone_r0, "landing_cone_theta_deg": lc.cone_theta_deg, "landing_cone_eps": lc.cone_eps,
        "landing_floor": bool(lc.floor_constraint), "init_state_library": lib_info["path"],
        **env_extra,
    })

    preset = dict(PRESETS[args.preset or ("ps2" if args.projection else "tracker")])
    for k in preset:
        v = getattr(args, k)
        if v is not None:
            preset[k] = v
    warm = args.warm_start_weights.strip()
    sac_cfg = SACConfig(
        seed=args.seed, use_projection=bool(args.projection), project_actor_actions=bool(args.projection),
        warm_start=bool(warm), warm_start_weights=str(_resolve(warm)) if warm else "",
        wall_time_budget_sec=float(args.wall_time_budget_hours) * 3600.0, **preset,
    )
    meta = {
        "mode": "ps2" if args.projection else "tracker", "reference": str(reference), "backup": str(backup),
        "filter": filt, "qp_solve_dtype": args.qp_solve_dtype, "init_library": lib_info,
        "safeguard_in_updates": bool(args.safeguard_in_updates),
        "preset": args.preset or ("ps2" if args.projection else "tracker"), "warm_start_weights": sac_cfg.warm_start_weights,
        "command": "python scripts/train_phase2_landing.py " + " ".join(argv or __import__("sys").argv[1:]),
    }
    (run_dir / "landing_meta.json").write_text(json.dumps(meta, indent=2, default=str))

    metrics_path = run_dir / "metrics.jsonl"
    metrics_path.write_text("")  # a restarted run starts a fresh log
    t_start = time.time()

    def logger(step: int, metrics: dict[str, float]) -> None:
        rec = {"step": int(step), **{k: float(v) for k, v in metrics.items()}}
        with open(metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        el = time.time() - t_start
        rate = step / max(el, 1e-6)
        eta = (sac_cfg.total_steps - step) / max(rate, 1e-9)
        if sac_cfg.wall_time_budget_sec > 0:
            eta = min(eta, max(0.0, sac_cfg.wall_time_budget_sec - el))
        keys = ("episode/return", "episode/safe_rate", "episode/hard_deck_margin_min", "episode/pos_error_norm",
                "eval/return_mean", "eval/violation_free_episode_rate", "eval/hard_deck_margin_min",
                "train/slack_mean", "train/actor_use_solver_rate")
        txt = " ".join(f"{k.split('/')[0][:2]}/{k.split('/')[1]}={metrics[k]:.4g}" for k in keys if k in metrics)
        print(f"[progress] step {step:,}/{sac_cfg.total_steps:,} | {rate:.0f} steps/s | elapsed "
              f"{el / 3600:.2f} h, ETA {eta / 3600:.2f} h | {txt}", flush=True)

    print(f"[train] {meta['mode']} | ref {reference.name} | backup {backup.name} | filter {filt} | "
          f"{sac_cfg.total_steps:,} steps, {sac_cfg.num_envs} envs | projection={sac_cfg.use_projection} "
          f"(targets {sac_cfg.project_target_actions}) | warm start {sac_cfg.warm_start_weights or 'none'} | "
          f"episode {env_cfg.max_steps} steps | out {run_dir}", flush=True)
    update_cfg = cbf_cfg if args.safeguard_in_updates else replace(cbf_cfg, discrete_safeguard=False)
    result = run_training(sac_cfg, env_cfg, cbf_cfg, output_dir=str(run_dir), metric_logger=logger,
                          binding=LANDING_BINDING, make_projector=lambda c: projector, validate_learned_backup=False,
                          update_cbf_cfg=update_cfg)

    np.savez(run_dir / "history.npz", **{k: np.asarray(v, dtype=np.float64) for k, v in result["history"].items()})
    np.savez(run_dir / "eval_trajectory.npz", **{k: np.asarray(v) for k, v in result["eval"]["trajectory"].items()})
    configs = {"sac": asdict(sac_cfg), "env": asdict(env_cfg), "cbf": asdict(cbf_cfg), "landing_meta": meta}
    (run_dir / "configs.json").write_text(json.dumps(configs, indent=2, default=str))
    summ = result["summary"]
    print(f"[done] {run_dir} | {summ['total_steps']:,} steps in {summ['wall_time_sec'] / 3600:.2f} h | final eval: return "
          f"{summ.get('final_eval_return_mean', float('nan')):.1f}, violation-free episodes "
          f"{summ.get('final_eval_violation_free_episode_rate', float('nan')):.2f}, min margin "
          f"{summ.get('final_eval_hard_deck_margin_min', float('nan')):+.4f} m, pos error "
          f"{summ.get('final_eval_pos_error_norm_mean', float('nan')):.3f} m", flush=True)


if __name__ == "__main__":
    main()
