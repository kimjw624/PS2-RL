"""Does the landing BCBF filter keep a Phase-II landing policy inside the approach cone?

Runs a trained Phase-II landing policy (e.g. one trained without projection to track a
reference that deliberately misses the pad) twice from the same initial states:

  * policy only   : the actor's deterministic action, as trained;
  * policy + filter: the same action projected by the landing control-invariant layer
                     (BCBF-QP with the Phase-I safe-arrival backup).

Everything is read from files: the actor, env, observation and reference from the Phase-II
run's ``configs.json``; the cone, pad, base set and backup from the Phase-I checkpoint. The
script refuses to run if the two disagree on the cone/pad (the reference config) or the
plant/action box, unless ``--allow_geometry_mismatch``.

    python scripts/evaluate_landing_phase2_filter.py \\
        --phase2_run outputs/landing/land_v2_miss_s1 \\
        --backup outputs/landing_phase1_p2/land_v2_miss_td3_seed0 \\
        --num_episodes 64

Outputs (default ``outputs/landing_phase2_filter/<run>__<backup>/``): summary.json, summary.md,
episodes.npz, overview/case/actuator/distribution figures and a GIF per selected case.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--phase2_run", required=True, help="Phase-II run directory (configs.json + *_weights.pkl)")
    p.add_argument("--checkpoint", default="best", help="best | final | <file name in the run dir>")
    p.add_argument("--backup", required=True, help="Phase-I landing run dir or landing_backup_policy_actor.pkl")
    p.add_argument("--backup_policy_mode", default="learned", choices=("learned", "base_controller"),
                   help="'base_controller' uses the LQR everywhere as backup (baseline without the safe-arrival policy)")
    p.add_argument("--reference_config", default=None, help="override <reference>_config.json")
    p.add_argument("--allow_geometry_mismatch", action="store_true")
    p.add_argument("--num_episodes", type=int, default=64, help="episode 0 starts exactly on the reference")
    p.add_argument("--no_nominal", action="store_true", help="draw episode 0 from the reset distribution too")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--qp_solve_dtype", default=None, choices=("float32", "float64"),
                   help="default float64 (needs x64, enabled automatically unless JAX_ENABLE_X64=0)")
    p.add_argument("--alpha", type=float, default=None, help="override the BCBF class-K gain")
    p.add_argument("--cbf_override", action="append", default=[], metavar="KEY=VALUE",
                   help="override a landing BCBF/QP field (repeatable), e.g. alpha=10, "
                        "include_relative_time_term=false, base_alpha=2, slack_weight=1e6")
    p.add_argument("--out", default="")
    p.add_argument("--gif_cases", default="nominal,worst_unfiltered",
                   help="comma list of cases to animate (nominal, worst_unfiltered, most_filter_effort, "
                        "not_in_CN, or episode indices); 'none' to skip")
    p.add_argument("--gif_slowdown", type=float, default=0.25)
    p.add_argument("--gif_stride", type=int, default=1)
    p.add_argument("--gif_dpi", type=int, default=90)
    p.add_argument("--require_gpu", action="store_true")
    return p.parse_args(argv)


ARGS = parse_args()
if ARGS.qp_solve_dtype in (None, "float64"):
    os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from ps2rl.evaluation import landing_phase2_filter_eval as fe  # noqa: E402
from ps2rl.cil.quadrotor_landing_backup_cbf import QuadrotorLandingBCBFConfig  # noqa: E402
from ps2rl.evaluation.landing_phase2_run import (  # noqa: E402
    check_geometry_match, load_phase2_run, overrides_tag, parse_field_overrides)
from ps2rl.plotting import landing_phase2_plots as lp  # noqa: E402


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.bool_):
        return bool(o)
    return str(o)


def _pick_cases(ep: dict, recoverable: np.ndarray, nominal: bool) -> dict[str, int]:
    cases = {}
    if nominal:
        cases["nominal"] = 0
    pool = np.flatnonzero(recoverable)
    if pool.size:
        cases["worst_unfiltered"] = int(pool[np.argmin(ep["unfiltered"]["h_min"][pool])])
        cases["most_filter_effort"] = int(pool[np.argmax(ep["filtered"]["du_norm_mean"][pool])])
    bad = np.flatnonzero(~recoverable)
    if bad.size:
        cases["not_in_CN"] = int(bad[np.argmin(ep["filtered"]["h_min"][bad])])
    return cases


def _fmt(x, spec=".3f"):
    return "–" if x is None else format(x, spec)


def _markdown(summary: dict) -> str:
    def row(label, key, sub=None, spec=".3f", scale=1.0, subset="start_in_CN"):
        vals = []
        for v in fe.VARIANTS:
            s = summary["variants"][v][subset]
            if not s.get("episodes"):
                vals.append("–")
                continue
            x = s[key] if sub is None else s[key].get(sub)
            if x is None:
                vals.append("–")
            elif spec.endswith("d"):
                vals.append(format(int(round(x * scale)), spec))
            else:
                vals.append(format(x * scale, spec))
        return f"| {label} | " + " | ".join(vals) + " |"

    st = summary["starts"]
    lines = [
        f"# Landing BCBF filter on `{summary['setup']['phase2_run_name']}`",
        "",
        f"* policy: `{summary['setup']['weights_path']}`",
        f"* backup: `{summary['setup']['backup_checkpoint']}` ({summary['setup']['backup_policy_mode']}), "
        f"QP {summary['setup']['qp_solve_dtype']}, alpha={summary['setup']['alpha']}, "
        f"relative-time rows={summary['setup']['include_relative_time_term']}",
        f"* cone: r0={summary['setup']['cone']['r0']} m, θ={summary['setup']['cone']['theta_deg']}°, "
        f"ε={summary['setup']['cone']['eps']}; pad={summary['setup']['cone']['pad']}",
        f"* episodes: {st['episodes']} (episode 0 nominal: {st['nominal_first']}); start inside cone "
        f"{st['inside_cone']}/{st['episodes']}, start in C_N (backup-recoverable) {st['in_CN']}/{st['episodes']}",
        "",
        "Starts in C_N (the filter's guarantee applies):",
        "",
        "| metric | policy only | policy + filter |",
        "|---|---|---|",
        row("episodes", "episodes", spec="d"),
        row("episodes leaving the cone [%]", "cone_violation_episode_rate", spec=".1f", scale=100),
        row("states outside the cone [%]", "cone_violation_step_rate", spec=".2f", scale=100),
        row("worst min h_cone [m]", "h_min", "min", spec="+.3f"),
        row("median min h_cone [m]", "h_min", "median", spec="+.3f"),
        row("median first exit time [s]", "first_exit_time", "median", spec=".2f"),
        row("final distance to pad centre, median [m]", "final_pad_dist_xy", "median"),
        row("ends over the pad (≤ r0) [%]", "ends_over_pad_rate", spec=".1f", scale=100),
        row("final height, median [m]", "final_height", "median"),
        row("dips below the pad plane [%]", "below_pad_rate", spec=".1f", scale=100),
        row("mean tracking error, median [m]", "pos_err_mean", "median"),
        row("filter modifies the action [% of steps]", "intervention_rate", "mean", spec=".1f", scale=100),
        row("max QP slack", "slack_max", spec=".2e"),
        row("QP fallbacks (non-finite → backup) [steps]", "qp_fallback_steps", spec="d"),
        row("max tilt, p95 [deg]", "tilt_max_deg", "p95", spec=".1f"),
    ]
    if summary["variants"]["filtered"]["start_not_in_CN"].get("episodes"):
        lines += ["", "Starts outside C_N (no guarantee; reported for completeness):", "",
                  "| metric | policy only | policy + filter |", "|---|---|---|",
                  row("episodes", "episodes", spec="d", subset="start_not_in_CN"),
                  row("episodes leaving the cone [%]", "cone_violation_episode_rate", spec=".1f", scale=100,
                      subset="start_not_in_CN"),
                  row("worst min h_cone [m]", "h_min", "min", spec="+.3f", subset="start_not_in_CN"),
                  row("max QP slack", "slack_max", spec=".2e", subset="start_not_in_CN")]
    lines += ["", "Cases: " + ", ".join(f"{k} = episode {v}" for k, v in summary["cases"].items()), ""]
    return "\n".join(lines)


def main() -> None:
    args = ARGS
    print(f"[jax] backend={jax.default_backend()} x64={jax.config.jax_enable_x64}")
    if args.require_gpu and jax.default_backend() != "gpu":
        raise SystemExit("JAX is not on the GPU (see docker_batch/landing_env.sh)")

    run = load_phase2_run(args.phase2_run, repo_root=REPO, checkpoint=args.checkpoint,
                          reference_config=args.reference_config)
    qp_dtype = args.qp_solve_dtype or ("float64" if jax.config.jax_enable_x64 else "float32")
    tuning = parse_field_overrides(args.cbf_override, QuadrotorLandingBCBFConfig)
    if args.alpha is not None:
        tuning["alpha"] = args.alpha
    overrides = {"qp_solve_dtype": qp_dtype, "backup_policy_mode": args.backup_policy_mode, **tuning}
    setup = fe.build_setup(run, args.backup, **overrides)
    lc = setup.cbf_cfg.landing

    mismatch = check_geometry_match(run, lc)
    if run.reference_config is None:
        print("[warn] the reference has no _config.json; the cone comes from the backup checkpoint only")
    if mismatch:
        msg = "backup checkpoint and Phase-II task disagree:\n  " + "\n  ".join(mismatch)
        if not args.allow_geometry_mismatch:
            raise SystemExit(msg + "\nTrain a matching backup (docker_batch/run_landing_phase2_filter_eval.sh) "
                             "or pass --allow_geometry_mismatch to evaluate against the backup's cone anyway.")
        print("[warn] " + msg)

    backup_path = Path(args.backup)
    backup_name = (backup_path if backup_path.is_dir() else backup_path.parent).name
    tag = f"{run.run_dir.name}__{backup_name}" + ("" if args.backup_policy_mode == "learned" else "__lqr_backup")
    if tuning:
        tag += "__" + overrides_tag(tuning)
    out = Path(args.out) if args.out else REPO / "outputs" / "landing_phase2_filter" / tag
    out.mkdir(parents=True, exist_ok=True)
    print(f"[setup] policy {run.weights_path}\n[setup] backup {setup.cbf_cfg.learned_backup_policy_path} "
          f"({args.backup_policy_mode}), QP {qp_dtype}, alpha={setup.cbf_cfg.alpha}\n[setup] out {out}")

    nominal = not args.no_nominal
    keys, state0, obs0 = fe.initial_states(setup, args.num_episodes, args.seed, nominal_first=nominal)
    x0 = np.asarray(state0.x)
    starts = fe.classify_starts(setup, x0)
    rec_mask = starts["recoverable"].astype(bool)
    print(f"[starts] inside cone {int((starts['h0'] >= 0).sum())}/{len(x0)}, in C_N {int(rec_mask.sum())}/{len(x0)}")

    recs, h, ep, timing = {}, {}, {}, {}
    for v in fe.VARIANTS:
        t0 = time.time()
        recs[v] = fe.run_variant(setup, v, keys, state0, obs0, args.batch_size)
        timing[v] = time.time() - t0
        h[v] = fe.cone_values(setup, recs[v]["traj"])
        ep[v] = fe.episode_metrics(setup, recs[v], h[v])
        print(f"[{v}] {timing[v]:.1f} s; episodes leaving the cone: "
              f"{int(ep[v]['violated'][rec_mask].sum())}/{int(rec_mask.sum())} (start in C_N), "
              f"{int(ep[v]['violated'].sum())}/{len(x0)} (all)")

    cases = _pick_cases(ep, rec_mask, nominal)
    summary = {
        "setup": {
            "phase2_run": str(run.run_dir), "phase2_run_name": run.run_dir.name,
            "weights_path": str(run.weights_path), "reference_path": str(run.reference_path),
            "reference_config_path": str(run.reference_config_path) if run.reference_config_path else None,
            "backup_checkpoint": setup.cbf_cfg.learned_backup_policy_path,
            "backup_policy_mode": args.backup_policy_mode,
            "qp_solve_dtype": qp_dtype, "alpha": setup.cbf_cfg.alpha, "cbf_overrides": tuning,
            "include_relative_time_term": setup.cbf_cfg.include_relative_time_term,
            "base_alpha": setup.cbf_cfg.base_alpha,
            "slack_weight": setup.cbf_cfg.slack_weight, "solver_tol": setup.cbf_cfg.solver_tol,
            "landing_config": lc.as_dict(),
            "cone": {"r0": lc.cone_r0, "theta_deg": lc.cone_theta_deg, "eps": lc.cone_eps,
                     "pad": [lc.pad_x, lc.pad_y, lc.pad_z]},
            "geometry_mismatch": mismatch,
            "env": {k: getattr(run.env_cfg, k) for k in ("dt", "max_steps", "include_time_features",
                                                         "init_px_range", "init_py_range", "init_pz_range",
                                                         "init_v_range", "init_tilt_deg_range", "init_yaw_deg_range")},
            "actor": {"obs_dim": setup.actor_cfg.obs_dim, "hidden": list(setup.actor_cfg.hidden_sizes)},
            "jax_backend": jax.default_backend(), "timing_sec": timing, "seed": args.seed,
        },
        "starts": {"episodes": int(len(x0)), "nominal_first": nominal,
                   "inside_cone": int((starts["h0"] >= 0).sum()), "in_CN": int(rec_mask.sum()),
                   "h0": fe._stats(starts["h0"])},
        "variants": {
            v: {"all": fe.summarize(ep[v], np.ones_like(rec_mask), lc.cone_r0),
                "start_in_CN": fe.summarize(ep[v], rec_mask, lc.cone_r0),
                "start_not_in_CN": fe.summarize(ep[v], ~rec_mask, lc.cone_r0)}
            for v in fe.VARIANTS
        },
        "cases": cases,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=_json_default), encoding="utf-8")
    md = _markdown(summary)
    (out / "summary.md").write_text(md, encoding="utf-8")
    np.savez_compressed(
        out / "episodes.npz",
        x0=x0, **{f"start_{k}": v for k, v in starts.items()},
        **{f"{v}_{k}": a for v in fe.VARIANTS for k, a in ep[v].items()},
        **{f"{v}_traj": recs[v]["traj"] for v in fe.VARIANTS},
        **{f"{v}_u": recs[v]["u"] for v in fe.VARIANTS},
        **{f"{v}_h": h[v] for v in fe.VARIANTS},
        filtered_u_raw=recs["filtered"]["u_raw"], filtered_slack=recs["filtered"]["slack"],
        filtered_du_norm=recs["filtered"]["du_norm"],
    )

    ref = np.load(run.reference_path)
    scene = lp.Scene.build(lc, ref["states"], ref["t"] if "t" in ref else np.arange(len(ref["states"])) * run.env_cfg.dt,
                           setup.dt)
    base = f"{run.run_dir.name} ({args.checkpoint}) with backup {backup_name}"
    showcase = cases.get("nominal", cases.get("worst_unfiltered", 0))
    figs = {"overview": lp.plot_overview(scene, recs, h, showcase, out / "overview.png", base)}
    figs["distributions"] = lp.plot_distributions(ep, rec_mask, lc.cone_r0, out / "distributions.png", base)
    for name, k in cases.items():
        tag_c = f"episode {k} ({name}{'' if rec_mask[k] else ', start not in C_N'})"
        figs[f"case_{name}"] = lp.plot_case(scene, recs, h, k, out / f"case_{name}.png", f"{base} | {tag_c}")
        figs[f"actuators_{name}"] = lp.plot_actuators(scene, recs, k, setup.cbf_cfg.action_low,
                                                      setup.cbf_cfg.action_high, out / f"actuators_{name}.png",
                                                      f"{base} | {tag_c}", gravity=lc.gravity)
    gif_req = [c.strip() for c in args.gif_cases.split(",") if c.strip() and c.strip() != "none"]
    for c in gif_req:
        k = int(c) if c.isdigit() else cases.get(c)
        if k is None:
            print(f"[gif] case '{c}' not available (cases: {sorted(cases)})")
            continue
        name = c if not c.isdigit() else f"episode{k}"
        t0 = time.time()
        figs[f"gif_{name}"] = lp.save_gif(scene, recs, h, k, out / f"trajectory_{name}.gif",
                                          f"{base} | ep {k} ({name}{'' if rec_mask[k] else ', start not in C_N'})",
                                          slowdown=args.gif_slowdown, stride=args.gif_stride, dpi=args.gif_dpi)
        print(f"[gif] {figs[f'gif_{name}']} ({time.time() - t0:.0f} s)")
    summary["figures"] = figs
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=_json_default), encoding="utf-8")
    print("\n" + md)
    print(f"[done] {out}")


if __name__ == "__main__":
    main()
