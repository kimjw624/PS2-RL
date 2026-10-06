"""Can the filtered vehicle reach the floor? Tests for one backup under several filter settings.

For each filter setting (``--settings``):

T2  hover map     holding still (level, v = 0, u = u*) near the pad: allowed by every BCBF row?
                  Gives the lowest height the filter lets the vehicle sit at over the pad centre.
T1  LQR landing   the checkpoint's hover LQR re-centred on a target, with and without the filter,
                  from the reference start (``--reference``), from hover over the pad and from random C_N starts
                  (design-region samples that are backup-recoverable under the setting's S).
                  Targets: pad centre on the floor (all starts), the pad edge (0.9 r0) and a point
                  below the floor at the cone-apex depth (fixed starts). Touchdown = height
                  <= --zeta_touch over the pad disk at speed <= --v_touch.
T3  tracking      (``--tracking_reference``) an LQR tracking a reference that ends on the pad but
                  cuts outside the cone on the way; unfiltered it leaves the cone, filtered it must
                  stay inside and still touch down.

Settings are ``name:KEY=VAL,KEY=VAL;name2:...`` over the landing BCBF fields (alpha, alpha_floor,
relative_time_floor, base_alpha, ...) plus ``floor_constraint=on`` (adds the floor to a cone-only
checkpoint's safe set; a tightening, B is checked to lie above the pad).

    python scripts/test_landing_floor_reach.py --backup checkpoints/landing_phase1/floor_rec10_td3_seed0 \\
        --settings "a10_f20:alpha=10,alpha_floor=20" \\
        --tracking_reference ps2rl/envs/assets/quadrotor_landing_cornercut_reference.npz --out outputs/floor_tests/x
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

DEFAULT_SETTINGS = ";".join([
    "as_trained:",
    "floor_a4:floor_constraint=on",
    "floor_a10:floor_constraint=on,alpha=10",
    "floor_a10_f20:floor_constraint=on,alpha=10,alpha_floor=20",
    "floor_a20:floor_constraint=on,alpha=20",
    "floor_a10_fixedtau:floor_constraint=on,alpha=10,relative_time_floor=off",
])


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backup", required=True)
    p.add_argument("--settings", default=DEFAULT_SETTINGS)
    p.add_argument("--reference", default=None,
                   help="landing reference .npz: its first state is an extra T1 start and it sets the side the tests look at")
    p.add_argument("--tracking_reference", default=None, help="reference .npz for T3")
    p.add_argument("--n_random", type=int, default=16, help="random C_N starts for T1")
    p.add_argument("--n_track", type=int, default=8, help="perturbed starts for T3 (besides the nominal)")
    p.add_argument("--horizon_factor", type=float, default=3.0, help="rollout length in backup horizons N")
    p.add_argument("--zeta_touch", type=float, default=0.02, help="touchdown height threshold [m]")
    p.add_argument("--v_touch", type=float, default=0.3, help="touchdown speed threshold [m/s]")
    p.add_argument("--hover_grid", default="41x26", help="S x Z points of the hover map")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--qp_solve_dtype", default="float64", choices=("float32", "float64"))
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
from ps2rl.evaluation import landing_filter_reach as R  # noqa: E402
from ps2rl.utils.field_overrides import parse_field_overrides  # noqa: E402
from ps2rl.phase1_sa.landing_design_region import LandingDesignRegionConfig, build_landing_sampler  # noqa: E402


def _jd(o):
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def parse_settings(text: str) -> list[tuple[str, dict]]:
    out = []
    for chunk in [c for c in text.split(";") if c.strip()]:
        name, _, body = chunk.partition(":")
        items = [t for t in body.split(",") if t.strip()]
        out.append((name.strip(), parse_field_overrides(items, L.QuadrotorLandingBCBFConfig,
                                                        extra={"floor_constraint": bool})))
    return out


def _stats(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return None
    return {"mean": float(x.mean()), "median": float(np.median(x)), "min": float(x.min()), "max": float(x.max())}


def summarize_runs(td: dict, res: dict, mask=None) -> dict:
    m = np.ones_like(td["touchdown"], dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    if not m.any():
        return {"n": 0}
    return {
        "n": int(m.sum()),
        "touchdown_rate": float(td["touchdown"][m].mean()),
        "touchdown_time": _stats(td["touchdown_time"][m]),
        "min_h_cone": float(td["min_h_cone"][m].min()),
        "min_height": float(td["min_height"][m].min()),
        "final_height": _stats(td["final_height"][m]),
        "final_radial": _stats(td["final_radial"][m]),
        "max_slack": float(res["slack"][m].max()),
        "qp_fallbacks": int((~res["qp_used"][m].astype(bool)).sum()),
        "safeguard_active_rate": float((res["safeguard_lambda"][m] < 1.0).mean()) if "safeguard_lambda" in res else None,
    }


def main():
    args = ARGS
    t_start = time.time()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"[jax] backend={jax.default_backend()} x64={jax.config.jax_enable_x64}")
    settings = parse_settings(args.settings)
    md = L.read_checkpoint_metadata(args.backup)
    lc0 = L.landing_config_from_checkpoint(args.backup)
    pad = np.array([lc0.pad_x, lc0.pad_y, lc0.pad_z])
    tan_t = float(np.tan(np.radians(lc0.cone_theta_deg)))
    ref = None
    if args.reference:
        ref = np.asarray(np.load(args.reference)["states"], np.float64)
    trk = None
    if args.tracking_reference:
        b = np.load(args.tracking_reference)
        trk = {"states": np.asarray(b["states"], np.float32),
               "u": np.concatenate([np.asarray(b["a_cmd"])[:, None], np.asarray(b["omega_cmd"])], 1).astype(np.float32)}
    d = R.side_direction(pad[:2], ref if ref is not None else (trk["states"] if trk is not None else None))
    steps = int(round(args.horizon_factor * lc0.num_steps))
    rng = np.random.default_rng(args.seed)

    # ---- shared starts
    hover_state = np.r_[pad[:2], pad[2] + lc0.z_des, 0, 0, 0, R.level_quaternion(1)[0]]
    fixed = {"hover over the pad": hover_state}
    if ref is not None:
        fixed = {"reference start": ref[0].copy(), **fixed}
    region_cfg = LandingDesignRegionConfig.from_dict(md.get("design_region", {}))
    cfg_tmp = L.landing_bcbf_config_from_checkpoint(args.backup, qp_solve_dtype=args.qp_solve_dtype)
    safe0, base0 = L.landing_sets(cfg_tmp)
    sampler = build_landing_sampler(region_cfg, safe0, base0)
    keys = jax.random.split(jax.random.PRNGKey(args.seed), 8 * max(args.n_random, 1))
    cand, found = jax.vmap(lambda k: sampler["sample_region"](jnp.int32(0), k, jnp.float32(1.0)))(keys)
    cand = np.asarray(cand)[np.asarray(found)]
    targets_fixed = {
        "pad centre": np.r_[pad[:2], pad[2]],
        "pad edge (0.9 r0)": np.r_[pad[:2] + 0.9 * lc0.cone_r0 * d, pad[2]],
        "below the floor (apex depth)": np.r_[pad[:2], pad[2] - lc0.cone_r0 / tan_t],
    }
    hover_z = lc0.z_des
    nx, nz = (int(t) for t in args.hover_grid.lower().split("x"))
    r_top = float(safe0.radius_at(hover_z))
    s_grid = np.linspace(-r_top, r_top, nx)
    z_grid = np.linspace(0.0, hover_z, nz)
    P = R.slice_positions(cfg_tmp, d, s_grid, z_grid)
    h_cone_grid = np.asarray(L.cone_value(jnp.asarray(np.concatenate([P, np.zeros_like(P), np.tile([1.0, 0, 0, 0], P.shape[:2] + (1,))], -1)), cfg_tmp))

    report = {"backup": str(args.backup), "landing_config": lc0.as_dict(), "design_region": region_cfg.as_dict(),
              "side_direction": d, "steps": steps, "dt": lc0.dt, "zeta_touch": args.zeta_touch, "v_touch": args.v_touch,
              "phase1_summary": None, "settings": {}}
    summ_path = Path(args.backup if Path(args.backup).is_dir() else Path(args.backup).parent) / "summary.json"
    if summ_path.exists():
        s = json.loads(summ_path.read_text())
        report["phase1_summary"] = {"test_mu_w": s.get("test_at_best", {}).get("mu_weighted"),
                                    **{f"test_{r}": s.get("test_at_best", {}).get(r, {}).get("m_hat") for r in ("general", "edge", "shell")}}

    hover_maps = {}
    for name, over in settings:
        t0 = time.time()
        cfg = L.landing_bcbf_config_from_checkpoint(args.backup, qp_solve_dtype=args.qp_solve_dtype, **over)
        proj = L.QuadrotorLandingBackupCBFProjector(cfg)
        rt = proj.runtime
        rec_fn = L.make_recoverability_fn(cfg, rt)
        entry = {"overrides": over, "floor_constraint": bool(cfg.landing.floor_constraint), "alpha": cfg.alpha,
                 "sensitivity_propagation": cfg.sensitivity_propagation, "discrete_safeguard": cfg.discrete_safeguard,
                 "alpha_per_constraint": cfg.alpha_per_constraint, "relative_time_per_constraint": cfg.relative_time_per_constraint}

        # T2 hover map
        hf = R.hover_feasibility(cfg, rt, P)
        axis = int(np.argmin(np.abs(s_grid)))
        ok_axis = np.flatnonzero(hf["feasible"][:, axis])
        inside = h_cone_grid >= 0
        low_half = inside & (P[..., 2] - pad[2] <= 0.5 * hover_z)
        entry["hover"] = {
            "lowest_feasible_height_on_axis": float(z_grid[ok_axis.min()]) if ok_axis.size else None,
            "feasible_fraction_inside_cone": float(hf["feasible"][inside].mean()),
            "feasible_fraction_lower_half": float(hf["feasible"][low_half].mean()) if low_half.any() else None,
        }
        hover_maps[name] = hf["feasible"]

        # T1 LQR landing
        rec_c, _, _ = map(np.asarray, rec_fn(jnp.asarray(cand)))
        rand = cand[rec_c][: args.n_random]
        x0_list, tg_list, tags = [], [], []
        for sname, xs in fixed.items():
            for tname, tg in targets_fixed.items():
                x0_list.append(xs)
                tg_list.append(tg)
                tags.append((sname, tname))
        for i, xs in enumerate(rand):
            x0_list.append(xs)
            tg_list.append(targets_fixed["pad centre"])
            tags.append(("random C_N start", "pad centre"))
        x0 = np.array(x0_list, dtype=np.float32)
        tg = np.array(tg_list)
        hov = np.array([lc0.pad_x, lc0.pad_y, lc0.pad_z + lc0.z_des])
        offs = jnp.asarray(tg - hov, dtype=jnp.float32)
        act_b = jax.jit(jax.vmap(R.recentred_lqr(cfg)))
        nominal = lambda x, k: act_b(x, offs)  # noqa: E731
        res_f = R.rollout_nominal(cfg, proj, x0, nominal, steps, filtered=True)
        res_u = R.rollout_nominal(cfg, proj, x0, nominal, steps, filtered=False)
        td_f = R.touchdown_metrics(cfg, res_f["traj"], zeta_touch=args.zeta_touch, v_touch=args.v_touch)
        td_u = R.touchdown_metrics(cfg, res_u["traj"], zeta_touch=args.zeta_touch, v_touch=args.v_touch)
        tags_arr = np.array([f"{a} -> {b}" for a, b in tags])
        rec1 = np.asarray(rec_fn(jnp.asarray(x0))[0]).astype(bool)  # the guarantee covers starts in C_N only
        entry["lqr"] = {
            "per_case_filtered": {t: summarize_runs(td_f, res_f, (tags_arr == t) & rec1) for t in dict.fromkeys(tags_arr)},
            "per_case_unfiltered_min_height": {t: float(td_u["min_height"][tags_arr == t].min()) for t in dict.fromkeys(tags_arr)},
            "random_starts_found": int(len(rand)),
            "fixed_starts_not_in_CN": sorted({a for (a, _), r in zip(tags, rec1) if not r and a != "random C_N start"}),
        }
        entry["lqr_centre_all"] = summarize_runs(td_f, res_f, np.array([b == "pad centre" for _, b in tags]) & rec1)

        # T3 tracking
        if trk is not None:
            T_ref = trk["states"].shape[0]
            ref_x = jnp.asarray(trk["states"])
            ref_u = jnp.asarray(trk["u"])
            act_t = jax.jit(jax.vmap(R.tracking_lqr(cfg), in_axes=(0, None, None)))

            def nominal_t(x, k):
                i = min(k, T_ref - 1)
                return act_t(x, ref_x[i], ref_u[i])

            pert = np.zeros((args.n_track + 1, 10), dtype=np.float32)
            pert[1:, 0:3] = rng.uniform(-0.1, 0.1, (args.n_track, 3))
            pert[1:, 3:6] = rng.uniform(-0.2, 0.2, (args.n_track, 3))
            x0t = np.asarray(trk["states"][0])[None] + pert
            rec_t, _, _ = map(np.asarray, rec_fn(jnp.asarray(x0t)))
            steps_t = max(steps, T_ref + int(lc0.num_steps))
            tr_f = R.rollout_nominal(cfg, proj, x0t, nominal_t, steps_t, filtered=True)
            tr_u = R.rollout_nominal(cfg, proj, x0t, nominal_t, steps_t, filtered=False)
            tdt_f = R.touchdown_metrics(cfg, tr_f["traj"], zeta_touch=args.zeta_touch, v_touch=args.v_touch)
            tdt_u = R.touchdown_metrics(cfg, tr_u["traj"], zeta_touch=args.zeta_touch, v_touch=args.v_touch)
            entry["tracking"] = {
                "starts": int(len(x0t)), "starts_in_CN": int(rec_t.sum()),
                "filtered_in_CN": summarize_runs(tdt_f, tr_f, rec_t),
                "unfiltered_leaves_cone_rate": float((tdt_u["min_h_cone"] < 0).mean()),
                "unfiltered_touchdown_rate": float(tdt_u["touchdown"].mean()),
            }
        report["settings"][name] = entry
        c = entry["lqr_centre_all"]
        print(f"[{name}] {time.time() - t0:.0f}s | hover lowest {entry['hover']['lowest_feasible_height_on_axis']} m, "
              f"feasible inside cone {entry['hover']['feasible_fraction_inside_cone']:.2f} | LQR->centre touchdown "
              f"{c['touchdown_rate']:.2f} (n={c['n']}), min h_cone {c['min_h_cone']:+.3f}, min height {c['min_height']:+.3f}, "
              f"slack {c['max_slack']:.1e}"
              + (f" | tracking: touchdown {entry['tracking']['filtered_in_CN'].get('touchdown_rate', float('nan')):.2f}, "
                 f"min h_cone {entry['tracking']['filtered_in_CN'].get('min_h_cone', float('nan')):+.3f} "
                 f"(unfiltered leaves cone {entry['tracking']['unfiltered_leaves_cone_rate']:.2f})" if trk is not None else ""))

        # per-setting figure
        fig, axs = plt.subplots(1, 3, figsize=(20, 6))
        zz = np.linspace(min(0.0, -lc0.cone_r0 / tan_t), hover_z + 0.3, 200)
        rr = np.asarray(safe0.radius_at(zz))
        for ax in axs[:2]:
            ax.fill_betweenx(zz, -rr, rr, color="tab:green", alpha=0.08)
            ax.plot(rr, zz, color="tab:green")
            ax.plot(-rr, zz, color="tab:green")
            ax.plot([-lc0.cone_r0, lc0.cone_r0], [0, 0], color="tab:green", lw=5, solid_capstyle="butt")
            ax.axhline(0.0, color="0.3", lw=0.8)
        cols = plt.cm.viridis(np.linspace(0, 0.9, len(targets_fixed)))
        for i, (sname, tname) in enumerate(tags[: len(fixed) * len(targets_fixed)]):
            k = list(targets_fixed).index(tname)
            rel = res_f["traj"][i][:, 0:3] - pad
            axs[0].plot(rel[:, 0:2] @ d, rel[:, 2], color=cols[k], lw=1.6 if sname == "hover over the pad" else 1.0,
                        ls="-" if sname == "hover over the pad" else "--",
                        label=f"{tname} ({sname})")
            tgr = tg[i] - pad
            axs[0].plot(tgr[0:2] @ d, tgr[2], "x", color=cols[k], ms=9, mew=2)
        axs[0].set_xlim(-1.2 * r_top, 1.2 * r_top)
        axs[0].set_ylim(min(-lc0.cone_r0 / tan_t, -0.05) - 0.05, hover_z + 0.3)
        axs[0].set_aspect("equal")
        axs[0].set_title("T1: filtered LQR to targets (x)", fontsize=10)
        axs[0].legend(fontsize=6)
        axs[0].contour(s_grid, z_grid, hf["feasible"].astype(float), levels=[0.5], colors="tab:orange", linewidths=1.0)
        if trk is not None:
            rel = trk["states"][:, 0:3] - pad
            axs[1].plot(rel[:, 0:2] @ d, rel[:, 2], "--", color="k", lw=1.2, label="reference")
            relu = tr_u["traj"][0][:, 0:3] - pad
            axs[1].plot(relu[:, 0:2] @ d, relu[:, 2], color="tab:red", lw=1.4, label="tracker alone")
            for j in range(len(x0t)):
                relf = tr_f["traj"][j][:, 0:3] - pad
                axs[1].plot(relf[:, 0:2] @ d, relf[:, 2], color="tab:blue", lw=1.8 if j == 0 else 0.6,
                            alpha=1.0 if j == 0 else 0.5, label="tracker + filter" if j == 0 else None)
            axs[1].set_aspect("equal", adjustable="datalim")
            axs[1].set_title("T3: tracking a reference that cuts outside the cone", fontsize=10)
            axs[1].legend(fontsize=7)
        t_ax = np.arange(res_f["traj"].shape[1]) * lc0.dt
        ci = [i for i, (_, b) in enumerate(tags) if b == "pad centre"]
        for i in ci:
            axs[2].plot(t_ax, res_f["traj"][i][:, 2] - pad[2], color="tab:blue", lw=0.8, alpha=0.6)
        if trk is not None:
            t_t = np.arange(tr_f["traj"].shape[1]) * lc0.dt
            axs[2].plot(t_t, tr_f["traj"][0][:, 2] - pad[2], color="tab:purple", lw=1.8, label="T3 tracking (filtered)")
        axs[2].axhline(args.zeta_touch, color="tab:orange", ls=":", label=f"touchdown height {args.zeta_touch} m")
        axs[2].axhline(0.0, color="k", lw=0.8)
        axs[2].set_yscale("symlog", linthresh=0.01)
        axs[2].set_xlabel("t [s]")
        axs[2].set_ylabel("height above pad [m] (symlog)")
        axs[2].set_title("T1 (to pad centre, all starts) and T3 heights", fontsize=10)
        axs[2].legend(fontsize=7)
        for ax in axs:
            ax.grid(alpha=0.3)
        fig.suptitle(f"{Path(args.backup).name} | setting {name}: {over or 'as trained'}")
        fig.tight_layout()
        fig.savefig(out / f"setting_{name}.png", dpi=120)
        plt.close(fig)

    # hover maps side by side
    n = len(hover_maps)
    fig, axs = plt.subplots(1, n, figsize=(3.6 * n, 3.6), squeeze=False)
    for ax, (name, fe) in zip(axs[0], hover_maps.items()):
        ax.pcolormesh(s_grid, z_grid, np.where(h_cone_grid >= 0, fe.astype(float), np.nan), cmap="RdYlGn",
                      vmin=0, vmax=1, shading="auto")
        zz = np.linspace(0, hover_z, 100)
        ax.plot(np.asarray(safe0.radius_at(zz)), zz, color="k", lw=0.8)
        ax.plot(-np.asarray(safe0.radius_at(zz)), zz, color="k", lw=0.8)
        low = report["settings"][name]["hover"]["lowest_feasible_height_on_axis"]
        ax.set_title(f"{name}\nlowest hover on axis: {low if low is None else f'{low:.3f} m'}", fontsize=8)
        ax.set_xlabel("along miss direction [m]", fontsize=8)
    axs[0, 0].set_ylabel("height above pad [m]")
    fig.suptitle("T2: where holding still is allowed by the filter (green)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "hover_maps.png", dpi=130)
    plt.close(fig)

    report["wall_time_sec"] = time.time() - t_start
    (out / "floor_tests.json").write_text(json.dumps(report, indent=2, default=_jd))
    print(f"[done] {out} ({report['wall_time_sec'] / 60:.1f} min)")


if __name__ == "__main__":
    main()
