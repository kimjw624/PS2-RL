"""Can the vehicle reach the pad through the landing filter, and why does a filtered policy climb away?

Part A  C_N slices. The filter keeps the state in the backup-recoverable set C_N, so C_N
        bounds where the vehicle can be, whatever the nominal controller wants. Slices on
        the vertical plane through the pad axis along the reference's miss direction, for
        (1) hover at rest, (2) straight descent at the reference's peak sink rate and
        (3) the reference's own velocity/attitude at each height.
Part B  Filter only. The checkpoint's hover LQR re-centred on targets on the pad plane
        (pad centre, r0/2, r0, the reference touchdown point), from the reference start
        and from hover over the pad, with and without the filter. If it cannot land where
        C_N allows, the filter is conservative; if it can, a policy that climbs away does
        so for its own reasons.
Part D  Sensitivity: Part B's landing tests repeated for other BCBF settings (``--variants``),
        e.g. a larger class-K gain alpha, the fixed-tau rows (no relative-time term), or the
        LQR-only backup, to see which design choice holds the vehicle up.
Part C  Attribution on a Phase-II evaluation (``--eval_dir`` from evaluate_landing_phase2_filter.py):
        per step, vertical acceleration requested by the policy vs applied, and which QP
        row binds (a cone row at backup node i, i.e. i*dt ahead along the backup rollout,
        or the terminal base-set row).

    python scripts/analyze_landing_filter_reach.py --backup outputs/landing_phase1_p2/<run> \\
        --phase2_run outputs/landing/land_v2_miss_s1 \\
        --eval_dir outputs/landing_phase2_filter/land_v2_miss_s1__<run>
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backup", required=True)
    p.add_argument("--phase2_run", default=None, help="for the reference (miss direction, start, sink rate)")
    p.add_argument("--eval_dir", default=None, help="output dir of evaluate_landing_phase2_filter.py (Part C)")
    p.add_argument("--out", default="")
    p.add_argument("--grid_s", type=int, default=121)
    p.add_argument("--grid_z", type=int, default=81)
    p.add_argument("--horizon_factor", type=float, default=3.0, help="LQR test length in backup horizons N")
    p.add_argument("--attribution_episodes", default="cases", help="'cases', 'all_in_CN' or comma list of indices")
    p.add_argument("--qp_solve_dtype", default="float64", choices=("float32", "float64"))
    p.add_argument("--cbf_override", action="append", default=[], metavar="KEY=VALUE",
                   help="BCBF/QP field override for the main analysis (repeatable)")
    p.add_argument("--variants", default="alpha=10;alpha=20;include_relative_time_term=false;backup_policy_mode=base_controller",
                   help="Part D: ';'-separated variants, each a ','-separated list of KEY=VALUE ('' to skip)")
    return p.parse_args(argv)


ARGS = parse_args()
if ARGS.qp_solve_dtype == "float64":
    os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from ps2rl.cil import quadrotor_landing_backup_cbf as lbcbf  # noqa: E402
from ps2rl.evaluation import landing_filter_reach as R  # noqa: E402
from ps2rl.evaluation.landing_phase2_run import load_phase2_run, overrides_tag, parse_field_overrides  # noqa: E402


def _jd(o):
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def _cone_lines(ax, cone_r, z):
    r = cone_r(z)
    ax.plot(-r, z, color="tab:green", lw=1.3)
    ax.plot(r, z, color="tab:green", lw=1.3)


def main():
    args = ARGS
    base_over = parse_field_overrides(args.cbf_override, lbcbf.QuadrotorLandingBCBFConfig, extra={"floor_constraint": bool})
    cfg = lbcbf.landing_bcbf_config_from_checkpoint(args.backup, qp_solve_dtype=args.qp_solve_dtype, **base_over)
    projector = lbcbf.QuadrotorLandingBackupCBFProjector(cfg)
    rt = projector.runtime
    lc = cfg.landing
    cone, base = lbcbf.landing_sets(cfg)
    pad = np.array([lc.pad_x, lc.pad_y, lc.pad_z])
    tan_t = float(np.tan(np.radians(lc.cone_theta_deg)))

    def cone_r(z):
        a = lc.cone_r0 + tan_t * np.asarray(z)
        return np.sqrt(np.maximum(a * a - lc.cone_eps ** 2, 0.0)) * (a > lc.cone_eps)

    ref = None
    if args.phase2_run:
        run = load_phase2_run(args.phase2_run, repo_root=REPO)
        ref = np.asarray(np.load(run.reference_path)["states"], dtype=np.float64)
    d = R.side_direction(pad[:2], ref)
    backup_name = Path(args.backup).name if Path(args.backup).is_dir() else Path(args.backup).parent.name
    out = Path(args.out) if args.out else (Path(args.eval_dir) / "reach" if args.eval_dir
                                           else REPO / "outputs" / "landing_filter_reach" / backup_name)
    out.mkdir(parents=True, exist_ok=True)
    summary: dict = {"backup": cfg.learned_backup_policy_path, "landing_config": lc.as_dict(),
                     "side_direction": d, "qp_solve_dtype": args.qp_solve_dtype, "cbf_overrides": base_over,
                     "alpha": cfg.alpha, "include_relative_time_term": cfg.include_relative_time_term}

    # ------------------------------------------------------------------ Part A
    ext = np.sqrt(np.diag(np.linalg.inv(base.controller.p_matrix_f64())) * lc.base_set_c)  # error-space extents
    z_top = max(lc.z_des + ext[2], float(ref[:, 2].max() - pad[2]) if ref is not None else 0.0) + 0.25
    z_apex = -lc.cone_r0 / tan_t
    s_max = 1.1 * float(cone_r(z_top))
    s_grid = np.linspace(-s_max, s_max, args.grid_s)
    z_grid = np.linspace(z_apex, z_top, args.grid_z)
    P = R.slice_positions(cfg, d, s_grid, z_grid)
    n = P.shape[0] * P.shape[1]
    yaw0 = 0.0
    fields = {"hover (v = 0, level)": (np.zeros_like(P), R.level_quaternion(n, yaw0).reshape(P.shape[:2] + (4,)))}
    if ref is not None:
        vz_sink = float(np.max(-ref[:, 5]))
        vfield = np.zeros_like(P)
        vfield[..., 2] = -vz_sink
        fields[f"straight descent at the reference's peak sink rate ({vz_sink:.2f} m/s)"] = (
            vfield, R.level_quaternion(n, yaw0).reshape(P.shape[:2] + (4,)))
        # reference velocity/attitude as a function of height (descending branch, monotone in z)
        zr = ref[:, 2] - pad[2]
        order = np.argsort(zr)
        vq = np.zeros(P.shape[:2] + (7,))
        for j in range(7):
            vq[..., j] = np.interp(P[..., 2] - pad[2], zr[order], ref[order, 3 + j])
        q = vq[..., 3:7] / np.linalg.norm(vq[..., 3:7], axis=-1, keepdims=True)
        fields["the reference's velocity and attitude at that height"] = (vq[..., 0:3], q)

    slices = {}
    for name, (v, q) in fields.items():
        slices[name] = R.cn_slice(cfg, rt, P, v, q)
        print(f"[A] {name}: {slices[name]['recoverable'].mean():.3f} of the slice recoverable")
    x_hover = np.concatenate([P, np.zeros_like(P), R.level_quaternion(n).reshape(P.shape[:2] + (4,))], axis=-1)
    in_b = R.base_set_value(cfg, x_hover) >= 0.0

    hov = slices["hover (v = 0, level)"]["recoverable"]
    iz0 = int(np.argmin(np.abs(z_grid - 0.0)))
    axis_col = int(np.argmin(np.abs(s_grid)))
    rec_axis = np.flatnonzero(hov[:, axis_col])
    pad_row = hov[iz0]
    summary["part_A"] = {
        name: {"recoverable_fraction_of_slice": float(sl["recoverable"].mean()),
               "recoverable_fraction_inside_cone": float(sl["recoverable"][sl["h"] >= 0].mean())}
        for name, sl in slices.items()
    }
    summary["part_A"]["hover_on_pad_centre_recoverable"] = bool(hov[iz0, axis_col])
    summary["part_A"]["hover_lowest_recoverable_height_on_axis"] = float(z_grid[rec_axis.min()]) if rec_axis.size else None
    summary["part_A"]["hover_recoverable_offsets_on_pad_plane"] = (
        [float(s_grid[pad_row].min()), float(s_grid[pad_row].max())] if pad_row.any() else None)
    summary["part_A"]["cone_radius_on_pad_plane"] = float(cone_r(0.0))
    summary["part_A"]["grid"] = {"s": [float(s_grid[0]), float(s_grid[-1]), len(s_grid)],
                                 "z": [float(z_grid[0]), float(z_grid[-1]), len(z_grid)]}

    cmap = ListedColormap(["#d62728", "#bdbdbd", "#2ca02c"])
    aspect = (z_grid[-1] - z_grid[0]) / (s_grid[-1] - s_grid[0])
    fig, axs = plt.subplots(1, len(slices), figsize=(6.2 * len(slices), 6.2 * aspect + 2.2), squeeze=False)
    for ax, (name, sl) in zip(axs[0], slices.items()):
        cat = np.where(sl["recoverable"], 2, np.where(sl["backup_leaves_cone"], 0, 1))
        ax.pcolormesh(s_grid, z_grid, cat, cmap=cmap, vmin=0, vmax=2, shading="auto", alpha=0.55)
        _cone_lines(ax, cone_r, np.linspace(0.0 if z_apex > 0 else z_apex, z_top, 200))
        ax.contour(s_grid, z_grid, in_b.astype(float), levels=[0.5], colors="k", linewidths=1.2)
        ax.plot([-lc.cone_r0, lc.cone_r0], [0, 0], color="tab:green", lw=5, solid_capstyle="butt")
        ax.axhline(0.0, color="0.3", lw=0.6)
        if ref is not None:
            rel = ref[:, 0:3] - pad
            ax.plot(rel[:, 0:2] @ d, rel[:, 2], "--", color="k", lw=1.4)
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("along miss direction [m]")
        ax.set_aspect("equal")
    axs[0, 0].set_ylabel("height above pad [m]")
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    handles = [Patch(color="#2ca02c", alpha=0.55, label="in C_N (backup reaches B safely)"),
               Patch(color="#bdbdbd", alpha=0.8, label="backup stays in cone but misses B within N"),
               Patch(color="#d62728", alpha=0.55, label="backup leaves the cone"),
               Line2D([], [], color="tab:green", lw=1.3, label="cone boundary / pad"),
               Line2D([], [], color="k", lw=1.2, label="base set B (v = 0 slice)"),
               Line2D([], [], color="k", ls="--", lw=1.4, label="reference")]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=9)
    fig.suptitle(f"Backup-recoverable set C_N on the vertical plane through the pad axis ({backup_name})")
    fig.tight_layout(rect=(0, 0.14, 1, 0.94))
    fig.savefig(out / "reach_cn_slices.png", dpi=140)
    plt.close(fig)

    # ------------------------------------------------------------------ Part B
    offsets = [0.0, 0.5 * lc.cone_r0, lc.cone_r0]
    if ref is not None:
        offsets.append(float((ref[-1, 0:2] - pad[:2]) @ d))
    offsets = sorted(set(round(o, 6) for o in offsets))
    targets = np.array([np.r_[pad[:2] + o * d, pad[2]] for o in offsets])
    starts = {"hover over the pad (base-set centre)": np.r_[pad[:2], pad[2] + lc.z_des, 0, 0, 0,
                                                             R.level_quaternion(1)[0]]}
    if ref is not None:
        starts = {"reference start": ref[0].copy(), **starts}
    x0 = np.array([s for s in starts.values() for _ in offsets])
    tg = np.array([t for _ in starts for t in targets])
    steps = int(round(args.horizon_factor * lc.num_steps))
    lq = R.lqr_landing(cfg, projector, x0, tg, steps)
    rows = []
    for i, (sname, oname) in enumerate([(s, o) for s in starts for o in offsets]):
        tf = lq["traj_filtered"][i]
        tn = lq["traj_unfiltered"][i]
        rel_f = tf[:, 0:3] - pad
        rows.append({
            "start": sname, "target_offset": oname,
            "target_inside_cone": bool(abs(oname) <= float(cone_r(0.0))),
            "unfiltered_final_height": float(tn[-1, 2] - pad[2]), "unfiltered_min_h": float(lq["h_unfiltered"][i].min()),
            "filtered_min_height": float(rel_f[:, 2].min()), "filtered_final_height": float(rel_f[-1, 2]),
            "filtered_final_offset": float(rel_f[-1, 0:2] @ d),
            "filtered_final_speed": float(np.linalg.norm(tf[-1, 3:6])),
            "filtered_min_h": float(lq["h_filtered"][i].min()),
            "max_slack": float(lq["slack"][i].max()), "qp_fallbacks": int((~lq["qp_used"][i]).sum()),
        })
        r = rows[-1]
        print(f"[B] {sname:>36} -> target {oname:+.2f} m: filtered min height {r['filtered_min_height']:+.3f}, "
              f"final (offset {r['filtered_final_offset']:+.3f}, height {r['filtered_final_height']:+.3f}), "
              f"min h {r['filtered_min_h']:+.4f}, slack {r['max_slack']:.1e}")
    summary["part_B"] = {"steps": steps, "dt": lc.dt, "runs": rows}

    fig, axs = plt.subplots(2, len(starts), figsize=(7.0 * len(starts), 10), squeeze=False)
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(offsets)))
    zmin_plot = min(z_apex, float(lq["traj_filtered"][..., 2].min() - pad[2]))
    for j, sname in enumerate(starts):
        ax = axs[0, j]
        zz = np.linspace(min(0.0, zmin_plot), z_top, 200)
        ax.fill_betweenx(zz, -cone_r(zz), cone_r(zz), color="tab:green", alpha=0.1)
        _cone_lines(ax, cone_r, zz)
        ax.contour(s_grid, z_grid, hov.astype(float), levels=[0.5], colors="tab:green", linestyles=":", linewidths=1.2)
        ax.contour(s_grid, z_grid, in_b.astype(float), levels=[0.5], colors="k", linewidths=1.0)
        ax.plot([-lc.cone_r0, lc.cone_r0], [0, 0], color="tab:green", lw=5, solid_capstyle="butt")
        ax2 = axs[1, j]
        tt = np.arange(steps + 1) * lc.dt
        for k, o in enumerate(offsets):
            i = j * len(offsets) + k
            for key, ls in (("traj_unfiltered", "--"), ("traj_filtered", "-")):
                rel = lq[key][i][:, 0:3] - pad
                ax.plot(rel[:, 0:2] @ d, rel[:, 2], ls, color=colors[k], lw=1.8 if ls == "-" else 1.0)
            ax.plot(o, 0.0, "x", color=colors[k], ms=10, mew=2)
            rel = lq["traj_filtered"][i][:, 0:3] - pad
            ax2.plot(tt, rel[:, 2], color=colors[k], lw=1.8, label=f"target {o:+.2f} m: height")
            ax2.plot(tt, rel[:, 0:2] @ d, color=colors[k], lw=1.0, ls="-.", label=f"target {o:+.2f} m: offset")
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_title(f"{sname}\nsolid: LQR + filter, dashed: LQR alone, x: target, dotted: C_N (hover) boundary",
                     fontsize=9)
        ax.set_xlabel("along miss direction [m]")
        ax.set_ylabel("height above pad [m]")
        ax.grid(alpha=0.3)
        ax2.axhline(0.0, color="k", lw=0.8)
        ax2.set_xlabel("t [s]")
        ax2.set_ylabel("[m]")
        ax2.set_title("filtered run: height (solid) and offset along miss direction (dash-dot)", fontsize=9)
        ax2.legend(fontsize=7, ncol=2)
        ax2.grid(alpha=0.3)
    fig.suptitle("Filter only: hover LQR re-centred on targets on the pad plane")
    fig.tight_layout()
    fig.savefig(out / "reach_lqr_landing.png", dpi=140)
    plt.close(fig)

    # ------------------------------------------------------------------ Part D
    variants = [v for v in args.variants.split(";") if v.strip()]
    if variants:
        base_label = (", ".join(args.cbf_override) or "as configured") + \
            f" (alpha={cfg.alpha:g}, relative-time rows {'on' if cfg.include_relative_time_term else 'off'})"
        sets = [("baseline", base_label, cfg, projector)]
        for v in variants:
            over = {**base_over, **parse_field_overrides([t for t in v.split(",") if t.strip()],
                                                        lbcbf.QuadrotorLandingBCBFConfig, extra={"floor_constraint": bool})}
            cfg_v = lbcbf.landing_bcbf_config_from_checkpoint(args.backup, qp_solve_dtype=args.qp_solve_dtype, **over)
            sets.append((v, v, cfg_v, lbcbf.QuadrotorLandingBackupCBFProjector(cfg_v)))
        d_offsets = sorted(set([0.0, float(offsets[-1])]))
        d_targets = np.array([np.r_[pad[:2] + o * d, pad[2]] for o in d_offsets])
        d_x0 = np.array([s_ for s_ in starts.values() for _ in d_offsets])
        d_tg = np.array([t for _ in starts for t in d_targets])
        d_rows, d_traj = [], {}
        for key, label, cfg_v, proj_v in sets:
            res = R.lqr_landing(cfg_v, proj_v, d_x0, d_tg, steps)
            d_traj[key] = res
            for i, (sname, o) in enumerate([(s_, o) for s_ in starts for o in d_offsets]):
                rel = res["traj_filtered"][i][:, 0:3] - pad
                d_rows.append({"variant": key, "label": label, "start": sname, "target_offset": o,
                               "filtered_min_height": float(rel[:, 2].min()),
                               "filtered_final_height": float(rel[-1, 2]),
                               "filtered_final_offset": float(rel[-1, 0:2] @ d),
                               "filtered_min_h": float(res["h_filtered"][i].min()),
                               "max_slack": float(res["slack"][i].max()),
                               "qp_fallbacks": int((~res["qp_used"][i]).sum())})
                r = d_rows[-1]
                print(f"[D] {key:<44} {sname[:20]:>20} -> {o:+.2f}: final height {r['filtered_final_height']:+.3f} "
                      f"offset {r['filtered_final_offset']:+.3f}, min h {r['filtered_min_h']:+.4f}, slack {r['max_slack']:.1e}, "
                      f"fallbacks {r['qp_fallbacks']}")
        summary["part_D"] = {"steps": steps, "runs": d_rows}

        fig, axs = plt.subplots(3, len(d_offsets), figsize=(7.0 * len(d_offsets), 16), squeeze=False)
        vcol = plt.cm.tab10(np.arange(len(sets)))
        sname0 = list(starts)[0]
        tt = np.arange(steps + 1) * lc.dt
        for k, o in enumerate(d_offsets):
            finals = np.array([(d_traj[key]["traj_filtered"][k][-1, 0:2] - pad[:2]) @ d for key, *_ in sets])
            fz = np.array([d_traj[key]["traj_filtered"][k][-1, 2] - pad[2] for key, *_ in sets])
            zoom_x = (min(finals.min(), 0.0, o) - 0.25, max(finals.max(), o, lc.cone_r0) + 0.25)
            zoom_z = (min(0.0, fz.min()) - 0.08, max(fz.max(), 0.0) + 0.35)
            for row, (xl, zl) in enumerate(((None, None), (zoom_x, zoom_z))):
                ax = axs[row, k]
                zz = np.linspace(min(0.0, z_apex), z_top, 400)
                ax.fill_betweenx(zz, -cone_r(zz), cone_r(zz), color="tab:green", alpha=0.08)
                _cone_lines(ax, cone_r, zz)
                ax.plot([-lc.cone_r0, lc.cone_r0], [0, 0], color="tab:green", lw=5, solid_capstyle="butt")
                for c_i, (key, label, _, _) in enumerate(sets):
                    rel = d_traj[key]["traj_filtered"][k][:, 0:3] - pad
                    ax.plot(rel[:, 0:2] @ d, rel[:, 2], color=vcol[c_i], lw=1.8, label=label)
                    ax.plot(rel[-1, 0:2] @ d, rel[-1, 2], "o", color=vcol[c_i], ms=6)
                rel = d_traj["baseline"]["traj_unfiltered"][k][:, 0:3] - pad
                ax.plot(rel[:, 0:2] @ d, rel[:, 2], "--", color="0.4", lw=1.0, label="LQR alone")
                ax.plot(o, 0.0, "x", color="k", ms=10, mew=2)
                if xl is not None:
                    ax.set_xlim(*xl)
                    ax.set_ylim(*zl)
                    ax.set_title("zoom near the pad", fontsize=10)
                else:
                    ax.set_title(f"from {sname0} to target {o:+.2f} m (x); o = final position", fontsize=10)
                    ax.legend(fontsize=7)
                ax.set_aspect("equal", adjustable="box" if xl is not None else "datalim")
                ax.set_xlabel("along miss direction [m]")
                ax.set_ylabel("height above pad [m]")
                ax.grid(alpha=0.3)
            ax = axs[2, k]
            for c_i, (key, label, _, _) in enumerate(sets):
                rel = d_traj[key]["traj_filtered"][k][:, 0:3] - pad
                ax.plot(tt, rel[:, 2], color=vcol[c_i], lw=1.8, label=f"{label}: height")
                ax.plot(tt, d_traj[key]["h_filtered"][k], color=vcol[c_i], lw=1.0, ls=":", label=f"{label}: h_cone")
            ax.axhline(0.0, color="k", lw=0.8)
            ax.set_xlabel("t [s]")
            ax.set_ylabel("[m]")
            ax.set_title("height (solid) and cone margin h (dotted)", fontsize=10)
            ax.legend(fontsize=6, ncol=2)
            ax.grid(alpha=0.3)
        fig.suptitle("Filtered hover-LQR landing for different BCBF settings")
        fig.tight_layout()
        fig.savefig(out / "reach_sensitivity.png", dpi=140)
        plt.close(fig)

    # ------------------------------------------------------------------ Part C
    if args.eval_dir:
        ev = np.load(Path(args.eval_dir) / "episodes.npz")
        es = json.loads((Path(args.eval_dir) / "summary.json").read_text())
        rec0 = ev["start_recoverable"].astype(bool)
        if args.attribution_episodes == "cases":
            eps = sorted(set(es["cases"].values()))
        elif args.attribution_episodes == "all_in_CN":
            eps = list(np.flatnonzero(rec0))
        else:
            eps = [int(e) for e in args.attribution_episodes.split(",")]
        tf = ev["filtered_traj"][eps]
        x = tf[:, :-1]
        u = ev["filtered_u"][eps]
        u_ref = ev["filtered_u_raw"][eps]
        slack = ev["filtered_slack"][eps]
        du = ev["filtered_du_norm"][eps]
        br = R.binding_rows(cfg, rt, x, u, slack)
        az_ref = R.vertical_accel(u_ref, x[..., 6:10], lc.gravity)
        az = R.vertical_accel(u, x[..., 6:10], lc.gravity)
        tu = ev["unfiltered_traj"][eps][:, :-1]
        az_unf = R.vertical_accel(ev["unfiltered_u"][eps], tu[..., 6:10], lc.gravity)
        active = du > 1e-3
        low, high = np.asarray(cfg.action_low), np.asarray(cfg.action_high)
        sat_ref = np.any(np.isclose(u_ref, low, atol=1e-3 * (high - low)) | np.isclose(u_ref, high, atol=1e-3 * (high - low)), axis=-1)
        sat_unf = np.any(np.isclose(ev["unfiltered_u"][eps], low, atol=1e-3 * (high - low))
                         | np.isclose(ev["unfiltered_u"][eps], high, atol=1e-3 * (high - low)), axis=-1)
        part_c = []
        for j, e in enumerate(eps):
            a = active[j]
            nodes = br["node"][j][a]
            part_c.append({
                "episode": int(e), "start_in_CN": bool(rec0[e]),
                "filter_active_fraction": float(a.mean()),
                "binding_terminal_row_fraction": float((nodes == -1).mean()) if a.any() else None,
                "binding_cone_node_time_median_s": float(np.median(nodes[nodes >= 0]) * lc.dt) if (nodes >= 0).any() else None,
                "mean_vertical_accel_requested_when_active": float(az_ref[j][a].mean()) if a.any() else None,
                "mean_vertical_accel_applied_when_active": float(az[j][a].mean()) if a.any() else None,
                "policy_saturated_fraction_filtered_run": float(sat_ref[j].mean()),
                "policy_saturated_fraction_unfiltered_run": float(sat_unf[j].mean()),
                "final_height": float(tf[j, -1, 2] - pad[2]),
                "max_height": float(tf[j, :, 2].max() - pad[2]),
            })
            p = part_c[-1]
            print(f"[C] ep {e}: filter active {p['filter_active_fraction']:.2f}, terminal-row binding "
                  f"{p['binding_terminal_row_fraction']}, policy saturated {p['policy_saturated_fraction_filtered_run']:.2f} "
                  f"(unfiltered run {p['policy_saturated_fraction_unfiltered_run']:.2f}), height final {p['final_height']:+.2f}")

            T = x.shape[1]
            t = np.arange(T) * lc.dt
            fig, axs = plt.subplots(4, 1, figsize=(11, 12), sharex=True)
            ax = axs[0]
            ax.plot(np.arange(T + 1) * lc.dt, tf[j, :, 2] - pad[2], color="tab:blue", lw=2, label="height, filtered")
            ax.plot(np.arange(T + 1) * lc.dt, ev["unfiltered_traj"][e, :, 2] - pad[2], color="tab:red", lw=1.5,
                    label="height, policy only")
            if ref is not None:
                ax.plot(np.arange(ref.shape[0]) * lc.dt, ref[:, 2] - pad[2], "--", color="k", lw=1, label="reference height")
            ax.plot(np.arange(T + 1) * lc.dt, ev["filtered_h"][e], color="tab:green", lw=1.2, label="h_cone, filtered")
            ax.axhline(0, color="0.4", lw=0.6)
            ax.set_ylabel("[m]")
            ax.legend(fontsize=8, ncol=2)
            ax.set_title(f"episode {e} ({'start in C_N' if rec0[e] else 'start NOT in C_N'})")
            ax = axs[1]
            ax.plot(t, az_unf[j], color="tab:red", lw=1.2, label="policy only: applied")
            ax.plot(t, az_ref[j], "--", color="tab:blue", lw=1.2, label="filtered run: requested by the policy")
            ax.plot(t, az[j], color="tab:blue", lw=1.8, label="filtered run: applied")
            ax.axhline(0, color="0.4", lw=0.6)
            ax.set_ylabel("vertical accel. [m/s²]")
            ax.legend(fontsize=8)
            ax = axs[2]
            nd = br["node"][j].astype(float) * lc.dt
            term = br["node"][j] == -1
            ax.scatter(t[a & ~term], nd[a & ~term], s=9, color="tab:purple", label="binding cone row: time ahead on the backup rollout")
            ax.scatter(t[a & term], np.full((a & term).sum(), lc.T), s=14, marker="s", color="tab:orange",
                       label=f"binding terminal base-set row (plotted at T = {lc.T:g} s)")
            ax.set_ylabel("binding row [s ahead]")
            ax.set_ylim(-0.1, lc.T + 0.2)
            ax.legend(fontsize=8)
            ax = axs[3]
            ax.plot(t, du[j], color="tab:purple", lw=1.4, label="‖u_safe − u_ref‖ (normalised)")
            ax.fill_between(t, 0, sat_ref[j] * du[j].max(), step="mid", color="0.6", alpha=0.3,
                            label="policy proposal at an action-box limit")
            ax.set_ylabel("intervention")
            ax.set_xlabel("t [s]")
            ax.legend(fontsize=8)
            for a_ in axs:
                a_.grid(alpha=0.3)
            fig.tight_layout()
            fig.savefig(out / f"reach_attribution_ep{e}.png", dpi=130)
            plt.close(fig)
        summary["part_C"] = part_c

    (out / "reach_summary.json").write_text(json.dumps(summary, indent=2, default=_jd))
    print(f"[done] {out}")


if __name__ == "__main__":
    main()
