#!/usr/bin/env python
"""Where can the backup save the vehicle? C_N maps of the cone-obstacle base-set designs.

    python scripts/cone_obstacle_coverage.py --certificate outputs/cone_obstacle/certificate.json \
        --tip_z 0.8,1.2,1.6 --out outputs/cone_obstacle/coverage

C_N is the set the backup-CBF filter can keep the vehicle in: states from which the base
controller (the backup policy without a learned Phase I) reaches B within the horizon
without leaving S. It is evaluated on an x-z grid through the funnel axis for a few
velocity cases, per base-set design and geometry. From the maps:

* coverage of S per region (approach corridor above the tip, under the tip),
* the reach radius: the smallest horizontal distance from the axis, below the cylinder
  top, of a C_N state - a target cylinder narrower than this cannot be touched with the
  filter on (for that velocity case).

Writes ``coverage.json`` and one figure per geometry.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cone_obstacle_common import DEFAULT_DESIGNS, design_config, load_certificate, make_config, parse_list, resolve  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from ps2rl.cil.quadrotor_cone_obstacle_cbf import ConeObstacleBCBFConfig, make_recoverability_fn  # noqa: E402
from ps2rl.sets.quadrotor_cone_obstacle_sets import obstacle_sets  # noqa: E402

VELOCITY_CASES = {"hover": (0.0, 0.0, 0.0), "+x 0.5": (0.5, 0.0, 0.0), "+x 1.0": (1.0, 0.0, 0.0), "down 0.5": (0.0, 0.0, -0.5)}


def grid(cfg, a):
    xs = np.arange(a.x_min, a.x_max + 1e-9, a.res)
    zs = np.arange(float(cfg.ground_z), float(cfg.tip_z) + a.z_above + 1e-9, a.res)
    xx, zz = np.meshgrid(xs, zs)
    return xs, zs, xx, zz


def states(xx, zz, v, y=0.0):
    n = xx.size
    x = np.zeros((n, 10), np.float32)
    x[:, 0], x[:, 1], x[:, 2] = xx.ravel(), y, zz.ravel()
    x[:, 3:6] = v
    x[:, 6] = 1.0
    return jnp.asarray(x)


def reach_radius(mask, xs, zs, cfg, top):
    """Smallest |x - axis| on the approach side (x <= axis) among C_N cells with z <= top."""
    sel = (zs[:, None] <= top + 1e-9) & (xs[None, :] <= float(cfg.axis_x) + 1e-9) & mask
    if not sel.any():
        return float("inf")
    return float(np.min(np.abs(np.broadcast_to(xs[None, :], mask.shape)[sel] - float(cfg.axis_x))))


def run_design(cfg_d, a, xs, zs, xx, zz):
    bcfg = ConeObstacleBCBFConfig(scenario=cfg_d, alpha=a.alpha, alpha_ground=a.alpha_ground)
    safe, base = obstacle_sets(cfg_d)
    rec = make_recoverability_fn(bcfg)
    out = {}
    for name, v in VELOCITY_CASES.items():
        x = states(xx, zz, v)
        hb, hf, tb = [], [], []
        for i in range(0, x.shape[0], a.chunk):
            r = rec(x[i:i + a.chunk])
            hb.append(np.asarray(r[0])); hf.append(np.asarray(r[1])); tb.append(np.asarray(r[2]))
        ok = np.concatenate(hb).reshape(xx.shape)
        in_s = np.asarray(jax.vmap(safe.contains)(x)).reshape(xx.shape)
        in_b = np.asarray(jax.vmap(base.contains)(x)).reshape(xx.shape)
        out[name] = {"cn": ok & in_s, "safe": in_s, "base": in_b, "t_arrive": np.concatenate(tb).reshape(xx.shape)}
    return out


def region_masks(cfg, xx, zz):
    tip = float(cfg.tip_z)
    return {"approach (x<-0.5, z>tip)": (xx < float(cfg.axis_x) - 0.5) & (zz > tip),
            "under the tip (z<tip)": zz < tip,
            "all": np.ones_like(xx, bool)}


def plot_geometry(path, cfg, designs, maps, xs, zs, radii, cases):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    from ps2rl.sets.quadrotor_cone_obstacle_sets import ConeObstacleSafeSet

    safe = ConeObstacleSafeSet.from_config(cfg)
    r = np.linspace(0, max(abs(xs[0]), abs(xs[-1])) + 0.1, 200)
    zw = np.asarray(safe.surface_z(r))
    n_r, n_c = len(designs), len(cases)
    fig, axes = plt.subplots(n_r, n_c, figsize=(3.2 * n_c, 1.9 * n_r + 0.6), sharex=True, sharey=True, squeeze=False)
    cmap = ListedColormap(["#ffffff", "#9ecae1", "#3182bd"])  # outside C_N / C_N / B
    ext = (xs[0] - 0.5 * (xs[1] - xs[0]), xs[-1] + 0.5 * (xs[1] - xs[0]), zs[0] - 0.5 * (zs[1] - zs[0]), zs[-1] + 0.5 * (zs[1] - zs[0]))
    for i, d in enumerate(designs):
        for j, case in enumerate(cases):
            ax = axes[i, j]
            m = maps[d][case]
            img = np.where(m["base"] & m["safe"], 2, np.where(m["cn"], 1, 0))
            ax.imshow(img, origin="lower", extent=ext, cmap=cmap, vmin=0, vmax=2, aspect="auto", interpolation="nearest")
            ax.fill_between(np.r_[-r[::-1], r] + cfg.axis_x, np.r_[zw[::-1], zw], zs[-1] + 1, color="#555555", alpha=0.85, lw=0)
            for rad in radii:
                ax.plot([cfg.axis_x - rad, cfg.axis_x - rad, cfg.axis_x + rad, cfg.axis_x + rad],
                        [cfg.ground_z, cfg.cylinder_top, cfg.cylinder_top, cfg.ground_z], color="#d62728", lw=0.6, alpha=0.6)
            ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
            if i == 0:
                ax.set_title(f"v = {case}", fontsize=9)
            if j == 0:
                ax.set_ylabel(d.replace("points:", "pt:").replace("approach_", "app_"), fontsize=7)
    fig.suptitle(f"C_N (light) and B (dark) with the LQR backups - tip {cfg.tip_z:.1f} m, theta {cfg.cone_theta_deg:g} deg, "
                 f"eps {cfg.cone_eps:g}\nred: target cylinders r = {', '.join(f'{v:g}' for v in radii)} m, top at "
                 f"{cfg.cylinder_top:.2f} m", fontsize=9)
    fig.supxlabel("x [m] (approach from -x)", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vehicle", default="docker_batch/configs/landing_x500_cone45_r0p3.json")
    ap.add_argument("--certificate", default="outputs/cone_obstacle/certificate.json")
    ap.add_argument("--designs", default=",".join(DEFAULT_DESIGNS))
    ap.add_argument("--tip_z", default="0.8,1.2,1.6")
    ap.add_argument("--theta_deg", default="45", help="comma list; every (tip, theta, eps) combination of the lists")
    ap.add_argument("--eps", default="0.25")
    ap.add_argument("--alpha", type=float, default=10.0)
    ap.add_argument("--alpha_ground", type=float, default=20.0)
    ap.add_argument("--res", type=float, default=0.05)
    ap.add_argument("--x_min", type=float, default=-3.5)
    ap.add_argument("--x_max", type=float, default=1.5)
    ap.add_argument("--z_above", type=float, default=2.0, help="grid height above the tip")
    ap.add_argument("--radii", default="0.1,0.25,0.5,0.75", help="target-cylinder radii drawn and evaluated")
    ap.add_argument("--chunk", type=int, default=4096)
    ap.add_argument("--plot_cases", default="hover,+x 1.0")
    ap.add_argument("--out", default="outputs/cone_obstacle/coverage")
    a = ap.parse_args(argv)

    cert = load_certificate(a.certificate)
    designs = [d for d in a.designs.split(",") if d.strip()]
    radii = parse_list(a.radii)
    out_dir = resolve(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {"args": vars(a), "geometries": []}
    t0 = time.time()
    for tip in parse_list(a.tip_z):
        for th in parse_list(a.theta_deg):
            for eps in parse_list(a.eps):
                cfg = make_config(a.vehicle, tip_z=tip, cone_theta_deg=th, cone_eps=eps)
                xs, zs, xx, zz = grid(cfg, a)
                regions = region_masks(cfg, xx, zz)
                g = {"tip_z": tip, "theta_deg": th, "eps": eps, "designs": {}}
                maps = {}
                for d in designs:
                    try:
                        cfg_d = design_config(cfg, d, cert)
                    except ValueError as exc:
                        print(f"[tip {tip:g} th {th:g} eps {eps:g}] {d}: skipped ({exc})", flush=True)
                        continue
                    t1 = time.time()
                    m = run_design(cfg_d, a, xs, zs, xx, zz)
                    maps[d] = m
                    rec = {"base_set_c": cfg_d.base_set_c, "coverage": {}, "reach_radius": {}}
                    for case, mm in m.items():
                        rec["coverage"][case] = {k: float((mm["cn"] & reg).sum() / max((mm["safe"] & reg).sum(), 1)) for k, reg in regions.items()}
                        rec["reach_radius"][case] = reach_radius(mm["cn"], xs, zs, cfg, cfg.cylinder_top)
                    g["designs"][d] = rec
                    cov = rec["coverage"]
                    print(f"[tip {tip:g} th {th:g} eps {eps:g}] {d:<38s} c={cfg_d.base_set_c:6.2f}  "
                          + "  ".join(f"{case}: app {100 * cov[case]['approach (x<-0.5, z>tip)']:5.1f}% under {100 * cov[case]['under the tip (z<tip)']:5.1f}% "
                                      f"r_reach {rec['reach_radius'][case]:.2f}" for case in VELOCITY_CASES)
                          + f"  ({time.time() - t1:.0f} s)", flush=True)
                report["geometries"].append(g)
                fig = out_dir / f"coverage_tip{tip:g}_th{th:g}_eps{eps:g}.png"
                plot_geometry(fig, cfg, list(maps), maps, xs, zs, radii, [c for c in a.plot_cases.split(",") if c])
                np.savez_compressed(out_dir / f"maps_tip{tip:g}_th{th:g}_eps{eps:g}.npz", xs=xs, zs=zs,
                                    **{f"{d}|{case}|{k}": mm[k] for d, m in maps.items() for case, mm in m.items() for k in ("cn", "base", "safe")})
                print(f"  -> {fig}", flush=True)
    report["wall_time_s"] = time.time() - t0
    (out_dir / "coverage.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {out_dir / 'coverage.json'} ({report['wall_time_s']:.0f} s)")


if __name__ == "__main__":
    main()
