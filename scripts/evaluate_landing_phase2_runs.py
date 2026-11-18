#!/usr/bin/env python
"""Evaluate Phase-II landing runs from the same starts:

  * ``tracker``          the tracker (trained without the filter) alone;
  * ``tracker+filter``   the same tracker through the landing filter at deployment;
  * ``ps2``              the PS2 policy (trained through the filter) with its filter, as trained.

Runs come from ``scripts/train_phase2_landing.py`` (``configs.json`` + ``landing_meta.json``).
The filter is rebuilt from the PS2 run's meta (or ``--filter``), the starts from the runs' reset
library (plus the reference start as episode 0), and each episode lasts the training episode
(reference + hold) plus ``--extra_hold``.

Ground contact as in ``scripts/demo_landing_skid.py``: landed at the first step with height
<= --zeta_touch and speed <= --v_touch; on the platform if the horizontal distance from the pad
centre is then <= r0. Safety and
intervention statistics cover the flight up to touchdown.

    python scripts/evaluate_landing_phase2_runs.py --tracker_run outputs/landing_phase2/skid_tracker \\
        --ps2_run outputs/landing_phase2/skid_ps2 --out outputs/landing_phase2/eval_skid
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tracker_run", required=True, help="run trained with --projection off")
    p.add_argument("--ps2_run", default="", help="run trained with --projection on (optional)")
    p.add_argument("--filter", default="", help="filter settings KEY=VAL,... (default: the PS2 run's, else the trainer default)")
    p.add_argument("--backup", default="", help="Phase-I checkpoint (default: the runs')")
    p.add_argument("--qp_solve_dtype", choices=("float32", "float64"), default="float32")
    p.add_argument("--weights", choices=("best", "final"), default="best")
    p.add_argument("--n_starts", type=int, default=32, help="episodes: the reference start + library starts")
    p.add_argument("--extra_hold", type=float, default=0.0, help="seconds beyond the training episode")
    p.add_argument("--zeta_touch", type=float, default=0.02)
    p.add_argument("--v_touch", type=float, default=0.3)
    p.add_argument("--no_land_latch", action="store_true")
    p.add_argument("--no_gif", action="store_true")
    p.add_argument("--gif_slowdown", type=float, default=0.25)
    p.add_argument("--gif_stride", type=int, default=1)
    p.add_argument("--gif_tail", type=float, default=0.6)
    p.add_argument("--gif_zoom_margin", type=float, default=0.25)
    p.add_argument("--title", default="", help="name used in titles (default: the reference file stem)")
    p.add_argument("--out", required=True)
    return p.parse_args(argv)


ARGS = parse_args()
if ARGS.qp_solve_dtype == "float64":
    os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from ps2rl.cil import quadrotor_landing_backup_cbf as L  # noqa: E402
from ps2rl.envs.quadrotor_env import QuadrotorEnvConfig, build_quadrotor_env  # noqa: E402
from ps2rl.evaluation import landing_filter_reach as R  # noqa: E402
from ps2rl.evaluation.landing_phase2_policy_eval import (  # noqa: E402
    evaluation_starts, load_landing_run as load_run, make_mean_policy as make_policy, rollout_policy as rollout)
from ps2rl.evaluation.landing_phase2_run import parse_field_overrides, resolve_repo_path  # noqa: E402
from ps2rl.phase2_ps2.landing_ps2_entry import DEFAULT_FILTER  # noqa: E402
from ps2rl.plotting import landing_phase2_plots as lp  # noqa: E402

VARIANTS = ("tracker", "tracker+filter", "ps2")
PLOT_KEY = {"tracker": "unfiltered", "tracker+filter": "filtered", "ps2": "ps2"}
LABELS = {"unfiltered": "tracker only", "filtered": "tracker + filter", "ps2": "PS2 (trained w/ filter)"}


def _jd(o):
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def _f(v, fmt=".3f"):
    return "–" if v is None else format(v, fmt)


def main():
    args = ARGS
    t_start = time.time()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    runs = {"tracker": load_run(Path(args.tracker_run), args.weights)}
    if args.ps2_run:
        runs["ps2"] = load_run(Path(args.ps2_run), args.weights)
    ref_path = Path(runs["tracker"]["env_cfg"].reference_path)
    for k, r in runs.items():
        if Path(r["env_cfg"].reference_path).resolve() != ref_path.resolve():
            raise SystemExit(f"runs track different references: {ref_path} vs {r['env_cfg'].reference_path}")
    src = runs.get("ps2", runs["tracker"])["meta"]
    filt_str = args.filter or ",".join(f"{k}={'/'.join(str(x) for x in v) if isinstance(v, (list, tuple)) else v}"
                                       for k, v in (src.get("filter") or {}).items()) or DEFAULT_FILTER
    filt = parse_field_overrides([t for t in filt_str.split(",") if t.strip()], L.QuadrotorLandingBCBFConfig,
                                 extra={"floor_constraint": bool})
    backup = args.backup or src["backup"]
    try:
        backup = str(resolve_repo_path(backup, repo_root=REPO))
    except FileNotFoundError:
        pass
    cfg = L.landing_bcbf_config_from_checkpoint(backup, qp_solve_dtype=args.qp_solve_dtype, **filt)
    lc = cfg.landing
    proj = L.QuadrotorLandingBackupCBFProjector(cfg)
    rec_fn = L.make_recoverability_fn(cfg, proj.runtime)
    pad = np.array([lc.pad_x, lc.pad_y, lc.pad_z])
    half = 0.5 * (np.asarray(cfg.action_high) - np.asarray(cfg.action_low))

    env_cfg = runs["tracker"]["env_cfg"]
    env = build_quadrotor_env(env_cfg)
    b = np.load(ref_path)
    ref_x = np.asarray(b["states"], np.float64)
    ref_t = np.asarray(b["t"]) if "t" in b else np.arange(len(ref_x)) * lc.dt
    steps = int(env_cfg.max_steps) + int(round(args.extra_hold / lc.dt))
    title_name = args.title or ref_path.stem.replace("quadrotor_landing_", "").replace("_reference", "")

    # starts: the reference start (if admissible) + library starts
    lib_path = env_cfg.init_state_library
    lib = np.load(lib_path)["states"] if lib_path else np.zeros((0, 10), np.float32)
    x0, ok0 = evaluation_starts(ref_x, lib, args.n_starts, cfg, rec_fn)
    rc = np.asarray(rec_fn(jnp.asarray(x0))[0])
    print(f"[eval] {title_name}: {len(x0)} starts ({'reference start + ' if ok0 else ''}library), all in C_N: {bool(rc.all())}; "
          f"{steps} steps; filter {filt}; weights {args.weights}", flush=True)

    k_ref = np.minimum(np.arange(steps + 1), len(ref_x) - 1)
    results, recs, hs, touch = {}, {}, {}, {}
    plan = [("tracker", "tracker", None), ("tracker+filter", "tracker", proj)] + ([("ps2", "ps2", proj)] if "ps2" in runs else [])
    for name, run_key, projector in plan:
        t0 = time.time()
        pol = make_policy(runs[run_key], cfg, env.obs_dim)
        res = rollout(pol, env.observe, cfg, projector, x0, steps)
        traj = res["traj"]
        h_raw = np.asarray(L.safe_value(jnp.asarray(traj), cfg))
        c = R.contact_index(traj, pad[2], args.zeta_touch, args.v_touch)
        on_pf = np.array([ci >= 0 and np.hypot(*(traj[e, ci, 0:2] - pad[:2])) <= lc.cone_r0 for e, ci in enumerate(c)])
        c_eval = np.full_like(c, -1) if args.no_land_latch else c
        fly_x = R.flight_mask(c_eval, steps + 1) | (np.arange(steps + 1)[None, :] == np.where(c_eval >= 0, c_eval, 0)[:, None])
        fly_u = R.flight_mask(c_eval, steps)
        rel = traj[..., 0:3] - pad
        rad = np.array([np.hypot(*rel[e, ci, 0:2]) if ci >= 0 else np.nan for e, ci in enumerate(c)])
        du = np.linalg.norm((res["u"] - res["u_raw"]) / half, axis=-1)
        err = np.linalg.norm(traj[:, :, 0:3] - ref_x[k_ref][None, :, 0:3], axis=-1)[:, : len(ref_x)]
        r = {
            "leaves_safe_set_rate": float((np.where(fly_x, h_raw, np.inf).min(1) < 0).mean()),
            "min_safe_margin": float(h_raw[fly_x].min()),
            "touchdown_rate": float((c >= 0).mean()),
            "touchdown_on_platform_rate": float(on_pf.mean()),
            "touchdown_beside_platform_rate": float(((c >= 0) & ~on_pf).mean()),
            "touchdown_time_median": float(np.median(c[c >= 0]) * lc.dt) if (c >= 0).any() else None,
            "touchdown_radial_median": float(np.nanmedian(rad)) if np.isfinite(rad).any() else None,
            "touchdown_radial_max_on_platform": float(np.nanmax(rad[on_pf])) if on_pf.any() else None,
            "touchdown_speed_max": float(max(np.linalg.norm(traj[e, ci, 3:6]) for e, ci in enumerate(c) if ci >= 0)) if (c >= 0).any() else None,
            "final_height_median": float(np.median(rel[:, -1, 2])),
            "tracking_error_mean": float(err.mean()),
            "max_slack": float(res["slack"][fly_u].max()),
            "qp_fallback_rate": float((~res["qp_used"])[fly_u].mean()) if projector is not None else 0.0,
            "safeguard_active_rate": float((res["safeguard_lambda"][fly_u] < 1.0).mean()),
            "intervention_mean": float(du[fly_u].mean()),
            "intervention_peak": float(du[fly_u].max()),
        }
        results[name] = r
        rec = {"traj": traj, "u": res["u"], "u_raw": res["u_raw"], "du_norm": du, "slack": res["slack"]}
        if not args.no_land_latch:
            rec = R.latch_after_touchdown(rec, c)
        key = PLOT_KEY[name]
        recs[key] = rec
        hs[key] = np.asarray(L.cone_value(jnp.asarray(rec["traj"]), cfg))
        touch[key] = (c, on_pf)
        print(f"  [{name}] {time.time() - t0:.0f}s | leaves S {100 * r['leaves_safe_set_rate']:.0f}% (min margin "
              f"{r['min_safe_margin']:+.4f}) | touchdown on platform {100 * r['touchdown_on_platform_rate']:.0f}%, beside "
              f"{100 * r['touchdown_beside_platform_rate']:.0f}% (median {_f(r['touchdown_time_median'], '.2f')} s) | tracking "
              f"error {r['tracking_error_mean']:.3f} m | slack {r['max_slack']:.1e}, fallbacks {100 * r['qp_fallback_rate']:.1f}%, "
              f"intervention {r['intervention_mean']:.3f}", flush=True)

    # figures
    scene = lp.Scene.build(lc, ref_x, ref_t, lc.dt)
    figs = {}
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))
    zz = np.linspace(0, float(ref_x[:, 2].max() - pad[2]) + 0.3, 200)
    rr = scene.radius(zz)
    axs[0].fill_betweenx(zz, -rr, rr, color="tab:green", alpha=0.1)
    axs[0].plot(rr, zz, color="tab:green")
    axs[0].plot(-rr, zz, color="tab:green")
    axs[0].plot([-lc.cone_r0, lc.cone_r0], [0, 0], color="tab:green", lw=5, solid_capstyle="butt", label="platform")
    s_ref, z_ref = scene.side(ref_x)
    axs[0].plot(s_ref, z_ref, "--", color="k", lw=1.4, label="reference")
    th = np.linspace(0, 2 * np.pi, 200)
    axs[1].plot(pad[0] + lc.cone_r0 * np.cos(th), pad[1] + lc.cone_r0 * np.sin(th), color="tab:green", lw=2, label="platform")
    axs[1].plot(ref_x[:, 0], ref_x[:, 1], "--", color="k", lw=1.4, label="reference")
    for key, rec in recs.items():
        col = lp.COLORS[key]
        for e in range(len(x0)):
            s, z = scene.side(rec["traj"][e])
            axs[0].plot(s, z, color=col, lw=1.8 if e == 0 else 0.6, alpha=1.0 if e == 0 else 0.35,
                        label=LABELS[key] if e == 0 else None)
            axs[1].plot(rec["traj"][e, :, 0], rec["traj"][e, :, 1], color=col, lw=1.8 if e == 0 else 0.6, alpha=1.0 if e == 0 else 0.35)
        c, _ = touch[key]
        for e, ci in enumerate(c):
            if ci >= 0:
                s, z = scene.side(rec["traj"][e, ci])
                axs[0].plot(s, z, "v", color=col, ms=5, mec="k", mew=0.4)
                axs[1].plot(rec["traj"][e, ci, 0], rec["traj"][e, ci, 1], "v", color=col, ms=5, mec="k", mew=0.4)
    axs[0].set_aspect("equal", adjustable="datalim")
    axs[0].set_xlabel("along the reference's miss direction [m]")
    axs[0].set_ylabel("height above pad [m]")
    axs[0].set_title(f"{title_name}: all starts (bold: reference start; ▼ touchdown)", fontsize=10)
    axs[0].legend(fontsize=8)
    axs[1].set_aspect("equal", adjustable="datalim")
    axs[1].set_xlabel("x [m]")
    axs[1].set_ylabel("y [m]")
    axs[1].set_title("top view", fontsize=10)
    for a in axs:
        a.grid(alpha=0.3)
    fig.tight_layout()
    figs["overview"] = str(out / "overview.png")
    fig.savefig(figs["overview"], dpi=130)
    plt.close(fig)

    title = f"{title_name} reference | backup {Path(backup).name} | filter {filt_str}"
    figs["case"] = lp.plot_case(scene, recs, hs, 0, out / "case.png", title, labels=LABELS)
    figs["actuators_tracker_filter"] = lp.plot_actuators(scene, {"unfiltered": recs["unfiltered"], "filtered": recs["filtered"]}, 0,
                                                         cfg.action_low, cfg.action_high, out / "actuators_tracker_filter.png",
                                                         title, gravity=lc.gravity, labels=LABELS)
    if "ps2" in recs:
        figs["actuators_ps2"] = lp.plot_actuators(scene, {"unfiltered": recs["unfiltered"], "filtered": recs["ps2"]}, 0,
                                                  cfg.action_low, cfg.action_high, out / "actuators_ps2.png", title,
                                                  gravity=lc.gravity, labels={"unfiltered": LABELS["unfiltered"],
                                                                              "filtered": LABELS["ps2"]})
    if not args.no_gif:
        td, ends = {}, []
        for key, (c, on_pf) in touch.items():
            if c[0] >= 0:
                td[key] = (int(c[0]), "LANDED on the platform" if on_pf[0] else "LANDED beside the platform")
                ends.append(int(c[0]))
        g_recs, g_h = recs, hs
        if not args.no_land_latch and len(td) == len(recs):
            n_end = min(steps + 1, max(ends) + 1 + int(round(args.gif_tail / lc.dt)))
            g_recs = {v: {k: (a[:, :n_end] if k == "traj" else a[:, : n_end - 1]) for k, a in rec.items()} for v, rec in recs.items()}
            g_h = {v: a[:, :n_end] for v, a in hs.items()}
        zoom = None
        if args.gif_zoom_margin > 0:
            mg = float(args.gif_zoom_margin)
            s_lo, s_hi = -(lc.cone_r0 + mg), max(float(np.hypot(*(ref_x[-1, 0:2] - pad[:2]))), lc.cone_r0) + mg
            zoom = {"s": (s_lo, s_hi), "z": (-0.03, 0.45 * (s_hi - s_lo))}
        t0 = time.time()
        figs["gif"] = lp.save_gif(scene, g_recs, g_h, 0, out / f"landing_{title_name}.gif", title, slowdown=args.gif_slowdown,
                                  stride=args.gif_stride, labels=LABELS, touchdown=td, zoom=zoom,
                                  xlabel="along the reference's miss direction [m]")
        print(f"[gif] {figs['gif']} ({time.time() - t0:.0f} s)", flush=True)

    summary = {"reference": str(ref_path), "backup": backup, "filter": filt, "qp_solve_dtype": args.qp_solve_dtype,
               "weights": args.weights, "starts": int(len(x0)), "starts_include_reference_start": ok0, "steps": steps,
               "dt": lc.dt, "zeta_touch": args.zeta_touch, "v_touch": args.v_touch, "land_latch": not args.no_land_latch,
               "runs": {k: str(r["dir"]) for k, r in runs.items()},
               "training": {k: {kk: r["summary"].get(kk) for kk in ("total_steps", "wall_time_sec", "steps_per_sec",
                                                                    "stopped_by_wall_budget", "best_eval_step")}
                            for k, r in runs.items()},
               "results": results, "figures": figs, "wall_time_sec": time.time() - t_start}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=_jd))
    md = [f"# Phase-II landing evaluation: {title_name}", "",
          f"Reference `{ref_path.name}`; backup `{Path(backup).name}`; filter `{filt_str}`; {len(x0)} starts "
          f"({'reference start + ' if ok0 else ''}reset library, all in C_N: {bool(rc.all())}); touchdown = height ≤ "
          f"{args.zeta_touch} m at speed ≤ {args.v_touch} m/s; {args.weights} weights.", "",
          "| policy | leaves safe set | min margin [m] | on platform | beside platform | median touchdown [s] | "
          "tracking error [m] | max slack | QP fallbacks | intervention |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for name, r in results.items():
        md.append(f"| {name} | {100 * r['leaves_safe_set_rate']:.0f} % | {r['min_safe_margin']:+.4f} | "
                  f"{100 * r['touchdown_on_platform_rate']:.0f} % | {100 * r['touchdown_beside_platform_rate']:.0f} % | "
                  f"{_f(r['touchdown_time_median'], '.2f')} | {r['tracking_error_mean']:.3f} | {r['max_slack']:.1e} | "
                  f"{100 * r['qp_fallback_rate']:.1f} % | {r['intervention_mean']:.3f} |")
    md += ["", "Training: " + "; ".join(
        f"{k} {_f((v.get('total_steps') or 0) / 1e6, '.2f')}M steps in {_f((v.get('wall_time_sec') or 0) / 3600, '.2f')} h"
        f"{' (stopped by the wall-time budget)' if v.get('stopped_by_wall_budget') else ''}"
        for k, v in summary["training"].items())]
    (out / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"[done] {out} ({(time.time() - t_start) / 60:.1f} min)")


if __name__ == "__main__":
    main()
