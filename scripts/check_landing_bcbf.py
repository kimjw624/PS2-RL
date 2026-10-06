"""Self-test of the landing control-invariant layer (run this before Phase-II training).

    JAX_PLATFORMS=cpu python scripts/check_landing_bcbf.py \
        --backup checkpoints/landing_phase1/floor_rec10_td3_seed0 \
        --cbf_override alpha=10 --cbf_override alpha_floor=20

Three checks, all configured from the checkpoint (nothing geometric is set here):

1. Backup recoverability: states drawn from the Phase-I design region are rolled out
   under the composed backup pi_b; the per-region fraction that reaches the base set
   without leaving the cone should match the Phase-I evaluation (within sampling error).
2. Safety filter stress test: from recoverable start states, a nominal controller drives
   toward a point *outside* the cone (the checkpoint's hover LQR re-centred on that
   point). Without the filter it leaves the cone; with the BCBF-QP it must not.
3. QP health: slack, solver fallbacks and the size of the intervention.

Writes ``<out>/check_landing_bcbf.json`` and ``<out>/check_landing_bcbf.png`` and exits
non-zero if the filtered rollouts leave the cone or need non-negligible slack.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")
# The QP is solved in float64 by default (rows stay float32); this needs x64 enabled
# before jax is imported. Set JAX_ENABLE_X64=0 and --qp_solve_dtype float32 to test the
# pure-float32 path the trainer uses by default.
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ps2rl.cil import quadrotor_landing_backup_cbf as L  # noqa: E402
from ps2rl.phase1_sa.landing_design_region import REGION_NAMES, LandingDesignRegionConfig, build_landing_sampler  # noqa: E402


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backup", required=True, help="landing_backup_policy_actor.pkl or its run directory")
    p.add_argument("--out", default="outputs/check_landing_bcbf")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n_recoverability", type=int, default=512, help="states per region for check 1")
    p.add_argument("--n_filter", type=int, default=32, help="start states for check 2")
    p.add_argument("--start_scale", type=float, default=0.0,
                   help="design-region scale for check-2 start states (0 = gentlest velocities/tilts of Phase I)")
    p.add_argument("--exit_distance", type=float, default=1.0, help="target lies this far outside the cone wall [m]")
    p.add_argument("--horizon_factor", type=float, default=1.5, help="check-2 rollout length in backup horizons")
    p.add_argument("--slack_tol", type=float, default=1e-2, help="max tolerated QP slack")
    p.add_argument("--h_tol", type=float, default=-1e-3, help="min tolerated cone margin of filtered rollouts [m]")
    for name in ("alpha", "base_alpha", "slack_weight", "solver_tol"):
        p.add_argument(f"--{name}", type=float, default=None, help="QP tuning override (default: powerloop Phase-II value)")
    p.add_argument("--cbf_override", action="append", default=[], metavar="KEY=VALUE",
                   help="landing BCBF override (repeatable), e.g. alpha_floor=20, relative_time_floor=off, floor_constraint=on")
    p.add_argument("--qp_solve_dtype", choices=("float32", "float64"), default=None,
                   help="dtype of the qpax solve (default float64 when JAX x64 is enabled, else float32)")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    overrides = {k: getattr(args, k) for k in ("alpha", "base_alpha", "slack_weight", "solver_tol") if getattr(args, k) is not None}
    overrides["qp_solve_dtype"] = args.qp_solve_dtype or ("float64" if jax.config.jax_enable_x64 else "float32")
    from ps2rl.utils.field_overrides import parse_field_overrides

    overrides.update(parse_field_overrides(args.cbf_override, L.QuadrotorLandingBCBFConfig, extra={"floor_constraint": bool}))
    cfg = L.landing_bcbf_config_from_checkpoint(args.backup, **overrides)
    md = L.read_checkpoint_metadata(args.backup)
    lc = cfg.landing
    print(f"[config] from {cfg.learned_backup_policy_path}")
    print(f"  cone r0={lc.cone_r0} theta={lc.cone_theta_deg} deg eps={lc.cone_eps} pad=({lc.pad_x},{lc.pad_y},{lc.pad_z}); "
          f"base set z_des={lc.z_des} c_B={lc.base_set_c}; dt={lc.dt} N={lc.num_steps}")
    print(f"  safe set: cone{' + floor' if lc.floor_constraint else ''}; per-constraint alpha {cfg.alpha_per_constraint}, "
          f"relative-time rows {cfg.relative_time_per_constraint or cfg.include_relative_time_term}; Phase-I recovery rates "
          f"cone {lc.recovery_rate_cone:g} / floor {lc.recovery_rate_floor:g}")
    print(f"  QP: alpha={cfg.alpha} base_alpha={cfg.base_alpha} slack_weight={cfg.slack_weight:g} solver_tol={cfg.solver_tol:g}; "
          f"{cfg.num_qp_inequalities} inequalities; solved in {cfg.qp_solve_dtype}")
    if "certificate" in md:
        print("  certificate (Phase I): " + ", ".join(f"{k}={v:.3f}" for k, v in md["certificate"].items() if isinstance(v, float)))

    projector = L.QuadrotorLandingBackupCBFProjector(cfg)
    rt = projector.runtime
    cone, base_set = L.landing_sets(cfg)
    ctrl = base_set.controller
    recover = L.make_recoverability_fn(cfg, rt)
    # Phase I evaluated membership with its own failure test (S n gentle-recovery envelope, if on)
    recover_p1 = L.make_recoverability_fn(cfg, rt, training_envelope=True)
    region_cfg = LandingDesignRegionConfig.from_dict(md.get("design_region", {}))
    sampler = build_landing_sampler(region_cfg, cone, base_set)
    key = jax.random.PRNGKey(args.seed)
    report: dict = {"checkpoint": cfg.learned_backup_policy_path, "landing": lc.as_dict(),
                    "qp": {k: getattr(cfg, k) for k in ("alpha", "base_alpha", "slack_weight", "solver_tol", "qp_solve_dtype")}}

    # ---------------------------------------------------------------- check 1
    rec = {}
    for i, reg in enumerate(REGION_NAMES):
        keys = jax.random.split(jax.random.fold_in(key, i), args.n_recoverability)
        xs, found = jax.vmap(lambda k: sampler["sample_region"](jnp.int32(i), k, jnp.float32(1.0)))(keys)
        hb, hf, _ = recover_p1(xs[np.asarray(found)])
        rec[reg] = float(np.mean(np.asarray(hb)))
        hb_s, _, _ = recover(xs[np.asarray(found)])
        report.setdefault("recoverability_wrt_S_only", {})[reg] = float(np.mean(np.asarray(hb_s)))
    half = float(np.sqrt(np.log(2 / 0.05) / (2 * args.n_recoverability)))
    ref = None
    summ_path = Path(cfg.learned_backup_policy_path).parent / "summary.json"
    if summ_path.exists():
        s = json.load(open(summ_path))
        ref = {r: s["test_at_best"][r]["m_hat"] for r in REGION_NAMES}
    print("[check 1] backup recoverability (fresh samples): " + ", ".join(
        f"{r} {rec[r]:.3f}" + (f" (Phase I {ref[r]:.3f})" if ref else "") for r in REGION_NAMES) + f"  [+-{half:.3f} at 95%]")
    ok1 = ref is None or all(abs(rec[r] - ref[r]) <= half + 0.02 for r in REGION_NAMES)
    report["recoverability"] = {"fresh": rec, "phase1_test": ref, "hoeffding_halfwidth": half, "ok": bool(ok1)}

    # ---------------------------------------------------------------- check 2
    keys = jax.random.split(jax.random.fold_in(key, 99), 8 * args.n_filter)
    cand, found = jax.vmap(lambda k: sampler["sample_region"](jnp.int32(0), k, jnp.float32(args.start_scale)))(keys)
    cand = np.asarray(cand)[np.asarray(found)]
    hb, _, _ = recover(jnp.asarray(cand))
    x0 = cand[np.asarray(hb)][: args.n_filter]
    if len(x0) < args.n_filter:
        print(f"[check 2] only {len(x0)} recoverable start states found")
    if len(x0) == 0:
        report["filter_stress_test"] = {"ok": False, "reason": "no recoverable start states"}
        with open(out / "check_landing_bcbf.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"[FAIL] no recoverable start states for the stress test; wrote {out / 'check_landing_bcbf.json'}")
        raise SystemExit(1)
    dp = x0[:, 0:2] - np.array([lc.pad_x, lc.pad_y])
    rhat = dp / np.maximum(np.linalg.norm(dp, axis=1, keepdims=True), 1e-9)
    zeta0 = x0[:, 2] - lc.pad_z
    r_exit = np.asarray(cone.radius_at(zeta0)) + args.exit_distance
    targets = np.concatenate([np.array([lc.pad_x, lc.pad_y]) + rhat * r_exit[:, None], x0[:, 2:3]], axis=1)
    target_offset = jnp.asarray(targets - np.asarray([lc.pad_x, lc.pad_y, lc.pad_z + lc.z_des]), dtype=x0.dtype)

    def nominal(x, off):
        """Checkpoint hover LQR re-centred on a target outside the cone (drives out of the cone)."""
        xs = x.at[0:3].add(-off)
        return ctrl.action(xs)

    nominal_b = jax.jit(jax.vmap(nominal))
    step_b = jax.jit(jax.vmap(lambda x, u: L.landing_step(x, u, cfg)))
    h_b = jax.jit(jax.vmap(lambda x: cone.value(x)))
    steps = int(round(args.horizon_factor * lc.num_steps))
    X_nom = [jnp.asarray(x0)]
    X_f = [jnp.asarray(x0)]
    slack_t, fallback_t, dev_t, solve_ms = [], [], [], []
    for k in range(steps):
        xn = X_nom[-1]
        X_nom.append(step_b(xn, nominal_b(xn, target_offset)))
        xf = X_f[-1]
        u_nom = nominal_b(xf, target_offset)
        t0 = time.time()
        u_safe, slack, used, _ = projector.solve_batch_with_info(xf, u_nom)
        jax.block_until_ready(u_safe)
        if k > 0:
            solve_ms.append(1e3 * (time.time() - t0))
        slack_t.append(np.asarray(slack))
        fallback_t.append(~np.asarray(used))
        dev_t.append(np.linalg.norm((np.asarray(u_safe) - np.asarray(u_nom)) / np.asarray(cfg.action_scale), axis=1))
        X_f.append(step_b(xf, u_safe))
    Xn, Xf = np.stack([np.asarray(x) for x in X_nom], 1), np.stack([np.asarray(x) for x in X_f], 1)
    hn = np.asarray(jax.vmap(h_b)(jnp.asarray(Xn)))
    hf_ = np.asarray(jax.vmap(h_b)(jnp.asarray(Xf)))
    slack_t, fallback_t, dev_t = np.stack(slack_t, 1), np.stack(fallback_t, 1), np.stack(dev_t, 1)
    viol_nom = float(np.mean(hn.min(1) < 0.0))
    viol_f = float(np.mean(hf_.min(1) < 0.0))
    res = {
        "n_start_states": int(len(x0)),
        "steps": steps,
        "unfiltered_leave_cone_frac": viol_nom,
        "filtered_leave_cone_frac": viol_f,
        "filtered_min_h": float(hf_.min()),
        "filtered_min_h_per_run_median": float(np.median(hf_.min(1))),
        "max_slack": float(slack_t.max()),
        "frac_steps_slack_gt_1e-3": float(np.mean(slack_t > 1e-3)),
        "solver_fallback_steps": int(fallback_t.sum()),
        "mean_intervention_norm": float(dev_t.mean()),
        "frac_steps_intervening": float(np.mean(dev_t > 1e-3)),
        "qp_batch_solve_ms_median": float(np.median(solve_ms)) if solve_ms else float("nan"),
    }
    ok2 = (hf_.min() >= args.h_tol) and (slack_t.max() <= args.slack_tol)
    res["ok"] = bool(ok2)
    report["filter_stress_test"] = res

    # ---------------------------------------------------------------- check 2b (floor)
    if lc.floor_constraint:
        depth = lc.cone_r0 / np.tan(np.radians(lc.cone_theta_deg))  # cone apex depth below the pad
        tgt_f = np.tile(np.array([lc.pad_x, lc.pad_y, lc.pad_z - depth]), (len(x0), 1))
        off_f = jnp.asarray(tgt_f - np.asarray([lc.pad_x, lc.pad_y, lc.pad_z + lc.z_des]), dtype=x0.dtype)
        xn_f, xf_f = jnp.asarray(x0), jnp.asarray(x0)
        zmin_n, zmin_f, slack_f = np.inf, np.inf, 0.0
        for _ in range(steps):
            xn_f = step_b(xn_f, nominal_b(xn_f, off_f))
            u_nom = nominal_b(xf_f, off_f)
            u_safe, slack, used, _ = projector.solve_batch_with_info(xf_f, u_nom)
            xf_f = step_b(xf_f, u_safe)
            zmin_n = min(zmin_n, float(np.min(np.asarray(xn_f)[:, 2] - lc.pad_z)))
            zmin_f = min(zmin_f, float(np.min(np.asarray(xf_f)[:, 2] - lc.pad_z)))
            slack_f = max(slack_f, float(np.max(np.asarray(slack))))
        ok2b = (zmin_f >= args.h_tol) and (slack_f <= args.slack_tol)
        report["floor_stress_test"] = {"target_depth": float(depth), "unfiltered_min_height": zmin_n,
                                       "filtered_min_height": zmin_f, "max_slack": slack_f, "ok": bool(ok2b)}
        print(f"[check 2b] nominal targets {depth:.2f} m below the pad: unfiltered min height {zmin_n:+.3f} m | "
              f"filtered {zmin_f:+.4f} m, max slack {slack_f:.1e}")
        ok2 = ok2 and ok2b
    print(f"[check 2] nominal controller targets {args.exit_distance} m outside the wall, {len(x0)} recoverable starts, {steps} steps")
    print(f"  unfiltered: {100 * viol_nom:.0f}% leave the cone | filtered: {100 * viol_f:.1f}% leave, min h_cone {hf_.min():+.4f} m")
    fb_frac = float(fallback_t.mean())
    res["solver_fallback_frac"] = fb_frac
    print(f"[check 3] max slack {slack_t.max():.2e} (steps with slack>1e-3: {100 * res['frac_steps_slack_gt_1e-3']:.2f}%), "
          f"solver fallbacks {int(fallback_t.sum())} ({100 * fb_frac:.1f}% of steps), filter active on "
          f"{100 * res['frac_steps_intervening']:.0f}% of steps, batch QP {res['qp_batch_solve_ms_median']:.1f} ms for {len(x0)} states")
    if fb_frac > 0.01:
        print("  note: a fallback applies the backup action (safe, but not minimally invasive and without gradient). "
              "Float32 qpax fails on a few percent of these QPs; --qp_solve_dtype float64 removes the fallbacks.")

    _plot(out / "check_landing_bcbf.png", cfg, cone, Xn, Xf, hn, hf_, slack_t, dev_t)
    with open(out / "check_landing_bcbf.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    verdict = "PASS" if (ok1 and ok2) else "FAIL"
    print(f"[{verdict}] wrote {out / 'check_landing_bcbf.json'} and {out / 'check_landing_bcbf.png'}")
    if verdict != "PASS":
        raise SystemExit(1)


def _plot(path, cfg, cone, Xn, Xf, hn, hf, slack, dev) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    blue, crit, ink2, grid = "#2a78d6", "#d03b3b", "#52514e", "#e4e3df"
    plt.rcParams.update({"font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                         "grid.color": grid, "axes.edgecolor": grid, "legend.frameon": False})
    lc = cfg.landing
    dt = lc.dt
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.2))
    ax = axes[0, 0]
    zmax = max(Xn[..., 2].max(), Xf[..., 2].max()) + 0.3
    zz = np.linspace(0.0, zmax, 100)
    ax.plot(np.asarray(cone.radius_at(zz)), zz + lc.pad_z, color="#0b0b0b", lw=1.1, label="cone boundary")

    def radial(X):
        return np.hypot(X[..., 0] - lc.pad_x, X[..., 1] - lc.pad_y)

    for i in range(Xn.shape[0]):
        ax.plot(radial(Xn[i]), Xn[i, :, 2], color=crit, lw=0.7, alpha=0.6, label="nominal (unfiltered)" if i == 0 else None)
        ax.plot(radial(Xf[i]), Xf[i, :, 2], color=blue, lw=0.9, label="BCBF-filtered" if i == 0 else None)
        ax.plot(radial(Xf[i, :1]), Xf[i, :1, 2], "o", ms=2.5, color=ink2)
    ax.set_xlim(0, max(radial(Xn).max(), 2.0) + 0.1)
    ax.set_xlabel("radial distance from the cone axis [m]")
    ax.set_ylabel("altitude [m]")
    ax.set_title("(a) paths in the radial plane", loc="left")
    ax.legend(loc="lower right")
    t = np.arange(hn.shape[1]) * dt
    ax = axes[0, 1]
    for i in range(hn.shape[0]):
        ax.plot(t, hn[i], color=crit, lw=0.6, alpha=0.5)
        ax.plot(t, hf[i], color=blue, lw=0.8)
    ax.axhline(0.0, color=ink2, lw=0.9)
    ax.set_ylim(min(-0.3, hf.min() - 0.05), max(hf.max(), 0.5) + 0.05)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("cone margin $h_{cone}$ [m]")
    ax.set_title("(b) safety margin (red: nominal, blue: filtered)", loc="left")
    ts = np.arange(slack.shape[1]) * dt
    ax = axes[1, 0]
    ax.semilogy(ts, np.maximum(slack.max(0), 1e-10), color=blue, lw=1.0)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("max QP slack over start states")
    ax.set_title("(c) QP slack (0 = constraints met exactly)", loc="left")
    ax = axes[1, 1]
    ax.plot(ts, dev.mean(0), color=blue, lw=1.2, label="mean")
    ax.fill_between(ts, dev.min(0), dev.max(0), color=blue, alpha=0.12, lw=0, label="min-max")
    ax.set_xlabel("time [s]")
    ax.set_ylabel(r"$\|u_{safe}-u_{nom}\|$ (normalised)")
    ax.set_title("(d) filter intervention", loc="left")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
