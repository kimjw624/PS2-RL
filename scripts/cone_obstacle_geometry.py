#!/usr/bin/env python
"""Does the funnel stand between the start and the target cylinder? (geometry only)

    python scripts/cone_obstacle_geometry.py --out outputs/cone_obstacle/geometry

For starts on the approach side (above the tip, below the outer wall) and a target
cylinder of radius R from the ground up to H (<= tip), centred on the axis or shifted to
the far side (``--target_x``), per funnel shape (theta, eps):

* rim blocked   the straight segment to the nearest point of the cylinder (the near rim of
                its top) cuts the funnel - a straight-line policy would need the filter;
* hidden        no point of the cylinder (side or top) is visible from the start;
* detour        shortest straight distance to a visible point of the cylinder over the
                distance to the rim (>= 1; an upper bound of the true detour, which may
                hug the funnel instead).

Everything is in the vertical plane through the axis (the problem is axisymmetric). Writes
``geometry.json`` and a figure.
"""

from __future__ import annotations

import argparse
import itertools
import json
from math import radians, tan
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cone_obstacle_common import parse_list, resolve  # noqa: E402

import numpy as np  # noqa: E402


def h_cone(r, z, tip, tth, eps):
    return np.sqrt(r * r + eps * eps) - eps - tth * (z - tip)


def surface(r, tip, tth, eps):
    return tip + (np.sqrt(r * r + eps * eps) - eps) / tth


def visible(s, p, tip, tth, eps, n=200):
    """(..., 2) start (r, z) and (..., 2) target -> the segment stays outside the funnel.
    Segments between two points on the same side of the axis stay in the half-plane, so the
    signed x coordinate is used and r = |x|."""
    t = np.linspace(0.0, 1.0, n)
    pts = s[..., None, :] + t[:, None] * (p - s)[..., None, :]
    return (h_cone(np.abs(pts[..., 0]), pts[..., 1], tip, tth, eps) >= -1e-9).all(-1)


def analyse(tip, theta, eps, radius, top, starts, target_x=0.0, n_side=40, n_top=12):
    tth = tan(radians(theta))
    s = starts                                   # (N, 2) as (x < 0, z)
    xn = target_x - radius                       # near face of the cylinder
    if top > float(np.min(surface(np.abs(np.linspace(xn, target_x + radius, 64)), tip, tth, eps))) + 1e-9:
        return None                              # the cylinder would poke into the funnel
    rim = np.asarray([xn, top])
    d_rim = np.linalg.norm(s - rim, axis=1)
    vis_rim = visible(s, np.broadcast_to(rim, s.shape), tip, tth, eps)
    # cylinder boundary in the plane: near side (x = -R, z in [0, top]) and top (x in [-R, R], z = top)
    side = np.stack([np.full(n_side, xn), np.linspace(0.0, top, n_side)], 1)
    topf = np.stack([np.linspace(xn, target_x + radius, n_top), np.full(n_top, top)], 1)
    cyl = np.concatenate([side, topf], 0)        # (M, 2)
    vis = visible(s[:, None, :], np.broadcast_to(cyl[None], (s.shape[0],) + cyl.shape), tip, tth, eps)
    d = np.linalg.norm(s[:, None, :] - cyl[None], axis=2)
    d_vis = np.where(vis, d, np.inf).min(1)
    return {"rim_blocked": float(np.mean(~vis_rim)), "hidden": float(np.mean(~vis.any(1))),
            "detour_mean": float(np.nanmean(np.where(np.isfinite(d_vis), d_vis / d_rim, np.nan))),
            "detour_p90": float(np.nanquantile(np.where(np.isfinite(d_vis), d_vis / d_rim, np.nan), 0.9)),
            "rim_gap_to_funnel": float(surface(abs(xn), tip, tth, eps) - top)}


def sample_starts(rng, n, tip, theta, eps, r_min, r_max, dz_min, dz_max, wall_clearance):
    tth = tan(radians(theta))
    out = []
    while sum(len(o) for o in out) < n:
        r = rng.uniform(r_min, r_max, 4 * n)
        z = tip + rng.uniform(dz_min, dz_max, 4 * n)
        ok = z <= surface(r, tip, tth, eps) - wall_clearance
        out.append(np.stack([-r[ok], z[ok]], 1))
    return np.concatenate(out, 0)[:n]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tip_z", type=float, default=1.2)
    ap.add_argument("--theta_deg", default="30,45,60")
    ap.add_argument("--eps", default="0.05,0.25,0.5,1.0")
    ap.add_argument("--radii", default="0.1,0.25,0.5,0.75")
    ap.add_argument("--top_offsets", default="0,0.1,0.25", help="cylinder tops as depths below the tip")
    ap.add_argument("--target_x", default="0", help="cylinder centres along the approach direction (0: on the axis, > 0: far side)")
    ap.add_argument("--start_r", default="1.0,3.0", help="start distance range from the axis")
    ap.add_argument("--start_dz", default="0.2,2.5", help="start height range above the tip")
    ap.add_argument("--wall_clearance", type=float, default=0.15, help="starts at least this far below the outer wall")
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="outputs/cone_obstacle/geometry")
    a = ap.parse_args(argv)
    out = resolve(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    r_lo, r_hi = parse_list(a.start_r)
    dz_lo, dz_hi = parse_list(a.start_dz)
    rows = []
    for th, eps in itertools.product(parse_list(a.theta_deg), parse_list(a.eps)):
        starts = sample_starts(rng, a.n, a.tip_z, th, eps, r_lo, r_hi, dz_lo, dz_hi, a.wall_clearance)
        for tx, rad, off in itertools.product(parse_list(a.target_x), parse_list(a.radii), parse_list(a.top_offsets)):
            r = analyse(a.tip_z, th, eps, rad, a.tip_z - off, starts, target_x=tx)
            if r is None:
                continue
            rows.append({"theta_deg": th, "eps": eps, "target_x": tx, "radius": rad, "top_below_tip": off, **r})
            print(f"theta {th:4.0f}  eps {eps:4.2f}  x_c {tx:4.2f}  R {rad:4.2f}  top -{off:4.2f}: rim blocked {100 * r['rim_blocked']:5.1f} %  "
                  f"hidden {100 * r['hidden']:5.1f} %  detour mean {r['detour_mean']:.3f} p90 {r['detour_p90']:.3f}  "
                  f"(rim {r['rim_gap_to_funnel']:.3f} m below the funnel)", flush=True)
    (out / "geometry.json").write_text(json.dumps({"args": vars(a), "rows": rows}, indent=1) + "\n", encoding="utf-8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ths, epss = parse_list(a.theta_deg), parse_list(a.eps)
    fig, axes = plt.subplots(1, len(ths), figsize=(4.0 * len(ths), 3.3), sharey=True, squeeze=False)
    for j, th in enumerate(ths):
        ax = axes[0, j]
        for k, eps in enumerate(epss):
            sel = [r for r in rows if r["theta_deg"] == th and r["eps"] == eps and r["top_below_tip"] == parse_list(a.top_offsets)[0]
                   and r["target_x"] == parse_list(a.target_x)[0]]
            ax.plot([r["radius"] for r in sel], [100 * r["rim_blocked"] for r in sel], marker="o", color=plt.cm.viridis(k / max(1, len(epss) - 1)),
                    label=f"eps {eps:g}")
        ax.set_title(f"theta {th:g} deg (top at the tip)", fontsize=9)
        ax.set_xlabel("cylinder radius R [m]")
        ax.grid(alpha=0.3)
    axes[0, 0].set_ylabel("starts whose straight path to the rim\ncuts the funnel [%]")
    axes[0, -1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "geometry_blocked.png", dpi=110)
    print(f"wrote {out / 'geometry.json'}")


if __name__ == "__main__":
    main()
