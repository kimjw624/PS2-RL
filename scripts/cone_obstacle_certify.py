#!/usr/bin/env python
"""Certify the base-set designs of the cone-obstacle task and size them for each geometry.

    python scripts/cone_obstacle_certify.py --vehicle docker_batch/configs/landing_x500_cone45_r0p3.json \
        --tip_z 0.8,1.2,1.6 --out outputs/cone_obstacle/certificate.json

For the 9-D hover LQR (points), the paper's 7-D one (band) and the 6-D brake (anywhere)
it reports the level bounds that do not depend on where the base set sits: input
feasibility c_U, the quaternion chart c_chart and the adversarial one-step Lyapunov
decrease c_Lyap of the nonlinear Euler step. Then, per geometry, the geometric bound c_geo
(largest level whose ellipsoid stays outside the funnel and above the ground) for the band
at mid-gap and for candidate hover points, and the height of the brake set's stopping
transient at full energy c (a vehicle at rest is in B anywhere in S; a moving one needs
this much room under the tip).
Finally the brake set's invariance is checked by simulation from sampled states in B: its
rest point p_inf = p + L r is only constant for the linearised loop, so the drift of the
nonlinear loop is measured too.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cone_obstacle_common import ROOT, make_config, parse_list  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from ps2rl.envs.quadrotor_env import quadrotor_step_euler  # noqa: E402
from ps2rl.sets import landing_certificate as lcert  # noqa: E402
from ps2rl.sets.quadrotor_cone_obstacle_sets import (  # noqa: E402
    AnywhereBaseSet,
    BrakeDLQR,
    ConeObstacleSafeSet,
    band_controller,
    point_controller,
    position_metric,
)


def step_fn(cfg):
    w = cfg.omega_clip
    w = w if isinstance(w, float) else jnp.asarray(w, jnp.float32)
    return lambda x, u: quadrotor_step_euler(x, u, cfg.dt, cfg.gravity, cfg.a_cmd_min, cfg.a_cmd_max, w)


# ---------------------------------------------------------------- reduced-LQR Lyapunov checks
def _quat_from_phi(phi):
    half = -0.5 * phi
    qw = jnp.sqrt(jnp.clip(1.0 - jnp.sum(half * half, axis=-1, keepdims=True), 0.0, 1.0))
    return jnp.concatenate([qw, half], axis=-1)


def band_state_from_error(e, z_des):
    """7-D (zeta, v, phi) -> state at x = y = 0."""
    z = jnp.zeros(e.shape[:-1] + (2,), e.dtype)
    return jnp.concatenate([z, z_des + e[..., 0:1], e[..., 1:4], _quat_from_phi(e[..., 4:7])], axis=-1)


def brake_state_from_error(e, z_ref=5.0):
    """6-D (v, phi) -> state at (0, 0, z_ref) (the brake does not look at the position)."""
    p = jnp.zeros(e.shape[:-1] + (3,), e.dtype).at[..., 2].set(z_ref)
    return jnp.concatenate([p, e[..., 0:3], _quat_from_phi(e[..., 3:6])], axis=-1)


def reduced_lyap_worst(ctrl, step, level, state_from_error, n_starts=2048, iters=200, lr=0.05, seed=0):
    """Adversarial worst V(x+)/V(x) on {e^T P e = level} for a reduced-state LQR."""
    p = jnp.asarray(ctrl.p_matrix, jnp.float32)
    p64 = np.asarray(p, np.float64)
    dim = p64.shape[0]
    l_inv_t = jnp.asarray(np.linalg.inv(np.linalg.cholesky(0.5 * (p64 + p64.T))).T, jnp.float32)

    def ratio(u, lev):
        u = u / jnp.linalg.norm(u)
        e = jnp.sqrt(lev) * (l_inv_t @ u)
        x = state_from_error(e)
        e1 = ctrl.error_state(step(x, ctrl.action(x)))
        return (e1 @ p @ e1) / (e @ p @ e)

    vg = jax.jit(jax.vmap(jax.value_and_grad(ratio), in_axes=(0, None)))
    u = jnp.asarray(np.random.default_rng(seed).standard_normal((n_starts, dim)), jnp.float32)
    for _ in range(iters):
        _, g = vg(u, level)
        u = u + lr * g
        u = u / jnp.linalg.norm(u, axis=1, keepdims=True)
    v, _ = vg(u, level)
    return float(jnp.max(v))


def band_lyap_worst(ctrl, step, level, **kw):
    return reduced_lyap_worst(ctrl, step, level, lambda e: band_state_from_error(e, ctrl.z_des), **kw)


def bisect(pred, hi, iters=14):
    if pred(hi):
        return hi
    lo, up = 0.0, hi
    for _ in range(iters):
        mid = 0.5 * (lo + up)
        lo, up = (mid, up) if pred(mid) else (lo, mid)
    return lo


# ---------------------------------------------------------------- brake-set invariance
def anywhere_invariance(cfg, c, *, factor=1.0, n=4096, steps=300, seed=0):
    """Sampled states in B under the base controller: fraction leaving B (by more than a
    1 mm / 1e-3 slack), fraction leaving S, worst row violation, and the drift of the rest
    point p_inf of the nonlinear loop."""
    safe = ConeObstacleSafeSet.from_config(cfg)
    bset = AnywhereBaseSet(BrakeDLQR.from_config(cfg), point_controller(cfg, (cfg.axis_x, cfg.axis_y, cfg.tip_z)), c, safe,
                           clearance_factor=factor)
    step = step_fn(cfg)
    rng = np.random.default_rng(seed)
    m = 20 * n
    p = np.stack([rng.uniform(-3.0, 3.0, m), rng.uniform(-0.6, 0.6, m), rng.uniform(cfg.ground_z, cfg.tip_z + 2.5, m)], 1)
    v = rng.normal(0.0, 0.6, (m, 3))
    ang = rng.normal(0.0, 0.15, (m, 3))
    half = 0.5 * ang
    q = np.concatenate([np.sqrt(np.clip(1 - (half**2).sum(1, keepdims=True), 0, 1)), half], 1)
    x = jnp.asarray(np.concatenate([p, v, q], 1), jnp.float32)
    keep = np.asarray(jax.jit(jax.vmap(lambda s: bset.contains(s) & safe.contains(s)))(x))
    x = x[np.nonzero(keep)[0][:n]]

    @jax.jit
    def run(x0):
        p0, _ = bset.rest_point_and_energy(x0)

        def body(c_, _):
            xx, worst, bad_s, drift = c_
            xn = step(xx, bset.action(xx))
            pn, _ = bset.rest_point_and_energy(xn)
            return (xn, jnp.minimum(worst, bset.margin(xn)), bad_s | ~safe.contains(xn),
                    jnp.maximum(drift, jnp.linalg.norm(pn - p0))), None
        (_, worst, bs, drift), _ = jax.lax.scan(body, (x0, jnp.inf, False, 0.0), None, length=steps)
        return worst, bs, drift

    worst, bs, drift = (np.asarray(v) for v in jax.vmap(run)(x))
    return {"level": c, "clearance_factor": factor, "samples": int(x.shape[0]), "left_B": float(np.mean(worst < -1e-3)),
            "worst_row": float(worst.min()), "left_S": float(np.mean(bs)),
            "rest_point_drift": {"p50": float(np.median(drift)), "p99": float(np.quantile(drift, 0.99)), "max": float(drift.max())},
            "extents": list(bset.extents)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vehicle", default="docker_batch/configs/landing_x500_cone45_r0p3.json")
    ap.add_argument("--tip_z", default="0.8,1.2,1.6")
    ap.add_argument("--theta_deg", type=float, default=45.0)
    ap.add_argument("--eps", type=float, default=0.25)
    ap.add_argument("--anywhere_levels", default="1,2,4,8")
    ap.add_argument("--clearance_factors", default="1.0,-1",
                    help="clearance factors of the brake set to test (-1: the configured anywhere_clearance_factor)")
    ap.add_argument("--lyap_starts", type=int, default=2048)
    ap.add_argument("--lyap_iters", type=int, default=200)
    ap.add_argument("--out", default="outputs/cone_obstacle/certificate.json")
    a = ap.parse_args(argv)
    t0 = time.time()

    cfg0 = make_config(a.vehicle, cone_theta_deg=a.theta_deg, cone_eps=a.eps)
    step = step_fn(cfg0)
    out = {"vehicle": a.vehicle, "box": {"low": list(cfg0.action_low), "high": list(cfg0.action_high)}}

    # ------------- 9-D hover LQR (position-independent bounds)
    c9 = point_controller(cfg0, (0.0, 0.0, 1.0))
    p_inv9 = np.linalg.inv(c9.p_matrix_f64())
    c_u9, c_chart9 = float(c9.max_certified_level), float(lcert.c_chart_bound(p_inv9))
    hi9 = min(c_u9, c_chart9)
    c_lyap9 = lcert.c_lyap_adversarial(c9, step, c_hi=hi9, iters=14, ascent_iters=a.lyap_iters, n_starts=a.lyap_starts)
    m_h, m_v, blk = position_metric(c9)
    out["lqr9"] = {"c_U": c_u9, "c_chart": c_chart9, "c_Lyap": c_lyap9, "c_max": min(hi9, c_lyap9),
                   "half_extent_per_sqrt_c": {"horizontal": float(np.sqrt(blk[0, 0])), "vertical": float(np.sqrt(blk[2, 2]))},
                   "metric": {"m_h": m_h, "m_v": m_v}}
    print(f"[9-D] c_U {c_u9:.2f}  c_chart {c_chart9:.2f}  c_Lyap {c_lyap9:.2f}  -> c_max {min(hi9, c_lyap9):.2f}; "
          f"ellipsoid half-extent {np.sqrt(blk[0, 0]):.3f} m horizontal, {np.sqrt(blk[2, 2]):.3f} m vertical per sqrt(c)", flush=True)

    # ------------- 7-D band LQR (the paper's)
    c7 = band_controller(cfg0.replace(band_z=1.0))
    p_inv7 = np.linalg.inv(np.asarray(c7.p_matrix, np.float64))
    c_u7 = float(c7.max_certified_level)
    c_chart7 = float(4.0 / np.linalg.eigvalsh(p_inv7[4:7, 4:7]).max())
    hi7 = min(c_u7, c_chart7)
    c_lyap7 = bisect(lambda lev: band_lyap_worst(c7, step, lev, n_starts=a.lyap_starts, iters=a.lyap_iters) < 1.0, hi7)
    ext7 = float(np.sqrt(p_inv7[0, 0]))
    out["lqr7_band"] = {"c_U": c_u7, "c_chart": c_chart7, "c_Lyap": c_lyap7, "c_max": min(hi7, c_lyap7),
                        "vertical_half_extent_per_sqrt_c": ext7}
    print(f"[7-D band] c_U {c_u7:.2f}  c_chart {c_chart7:.2f}  c_Lyap {c_lyap7:.2f}  -> c_max {min(hi7, c_lyap7):.2f}; "
          f"vertical half-extent {ext7:.3f} m per sqrt(c)", flush=True)

    # ------------- 6-D brake LQR (anywhere)
    cb = BrakeDLQR.from_config(cfg0)
    p_inv6 = np.linalg.inv(cb.p_matrix_f64())
    c_u6 = float(cb.max_certified_level)
    c_chart6 = float(4.0 / np.linalg.eigvalsh(p_inv6[3:6, 3:6]).max())
    hi6 = min(c_u6, c_chart6)
    c_lyap6 = bisect(lambda lev: reduced_lyap_worst(cb, step, lev, brake_state_from_error,
                                                     n_starts=a.lyap_starts, iters=a.lyap_iters) < 1.0, hi6)
    l_gain = cb.rest_gain()
    w6 = l_gain @ p_inv6 @ l_gain.T
    out["lqr6_brake"] = {"c_U": c_u6, "c_chart": c_chart6, "c_Lyap": c_lyap6, "c_max": min(hi6, c_lyap6),
                         "stop_half_extent_per_sqrt_c": {"horizontal": float(np.sqrt(w6[0, 0])), "vertical": float(np.sqrt(w6[2, 2]))},
                         "rest_gain_L": l_gain.tolist()}
    print(f"[6-D brake] c_U {c_u6:.2f}  c_chart {c_chart6:.2f}  c_Lyap {c_lyap6:.2f}  -> c_max {min(hi6, c_lyap6):.2f}; "
          f"stopping-transient half-extent {np.sqrt(w6[0, 0]):.3f} m horizontal, {np.sqrt(w6[2, 2]):.3f} m vertical per sqrt(c)",
          flush=True)

    # ------------- geometry-dependent caps
    out["geometries"] = []
    for tip in parse_list(a.tip_z):
        cfg = cfg0.replace(tip_z=tip)
        safe = ConeObstacleSafeSet.from_config(cfg)
        gap = cfg.gap
        zb = cfg.band_height
        c_geo_band = min(tip - zb, zb - cfg.ground_z) ** 2 / ext7**2
        g = {"tip_z": tip, "gap": gap, "band": {"z": zb, "c_geo": c_geo_band, "c_max": min(out["lqr7_band"]["c_max"], c_geo_band)},
             "points": [], "anywhere_min_gap": {}}
        mid = cfg.ground_z + 0.5 * gap
        cands = {"approach_low": (-1.0, 0.0, mid), "approach_beside": (-2.0, 0.0, tip + 0.5),
                 "under_tip": (0.0, 0.0, mid), "far_low": (1.0, 0.0, mid)}
        for name, pt in cands.items():
            d = float(safe.m_distance(jnp.asarray(pt, jnp.float32), m_h, m_v))
            c_geo = min(d * d, (pt[2] - cfg.ground_z) ** 2 * m_v)
            g["points"].append({"name": name, "point": list(pt), "c_geo": c_geo, "c_max": min(out["lqr9"]["c_max"], c_geo)})
        for lev in parse_list(a.anywhere_levels):
            g["anywhere_min_gap"][str(lev)] = 2.0 * float(np.sqrt(lev)) * float(np.sqrt(w6[2, 2]))
        out["geometries"].append(g)
        pts = ", ".join(f"{p['name']} {p['c_max']:.1f}" for p in g["points"])
        print(f"[tip {tip:.1f} m] band at {zb:.2f} m: c_max {g['band']['c_max']:.2f} (geometry {c_geo_band:.2f}) | points: {pts}", flush=True)

    # ------------- brake-set invariance, at the middle geometry
    cfg_mid = cfg0.replace(tip_z=parse_list(a.tip_z)[len(parse_list(a.tip_z)) // 2])
    out["anywhere_invariance"] = []
    for lev in parse_list(a.anywhere_levels):
        if lev > min(out["lqr6_brake"]["c_max"], out["lqr9"]["c_max"]):
            continue
        for fac in parse_list(a.clearance_factors):
            fac = float(cfg_mid.anywhere_clearance_factor) if fac < 0 else fac
            r = anywhere_invariance(cfg_mid, lev, factor=fac)
            out["anywhere_invariance"].append(r)
            print(f"[anywhere c={lev:g} k={fac:g}] {r['samples']} states in B: {100 * r['left_B']:.2f} % leave B "
                  f"(worst row {r['worst_row']:.4f}), {100 * r['left_S']:.2f} % leave S within 6 s; rest-point drift "
                  f"p99 {r['rest_point_drift']['p99']:.4f} m, max {r['rest_point_drift']['max']:.4f} m", flush=True)

    out["wall_time_s"] = time.time() - t0
    p = ROOT / a.out if not Path(a.out).is_absolute() else Path(a.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {p} ({out['wall_time_s']:.0f} s)")


if __name__ == "__main__":
    main()
