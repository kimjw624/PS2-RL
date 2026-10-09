#!/usr/bin/env python
"""Filtered reach tests of the cone-obstacle task: can a simple nominal touch the target
cylinder under the funnel with the backup-CBF filter on, per base-set design?

    python scripts/cone_obstacle_reach.py --certificate outputs/cone_obstacle/certificate.json \
        --tip_z 0.8,1.2,1.6 --out outputs/cone_obstacle/reach

Nominal controllers (acceleration + body-rate commands from the 9-D hover LQR gain):

* ``center``     LQR toward the cylinder's centre (axis, half its height); the position
                 error is capped at ``--max_position_error`` like the SITL node does.
* ``top``        LQR toward the centre of the cylinder's top (axis, top height): the
                 straight path grazes the rounded tip, so the filter has to bend it.
* ``rim``        LQR toward the near edge of the cylinder's top (``--rim_radius`` from the
                 axis, top height): the shortest way to touch it; blocked only when the
                 funnel's rounded tip overhangs the rim.
* ``horizontal`` fly straight along the start direction at the start altitude (aim
                 ``--lead`` m ahead on that line): it runs into the funnel's outer wall,
                 and only the filter can make it slide down and under the tip.

Every design is also run without the filter (``none``) for the same nominals and starts.
The trajectories do not depend on the cylinder's size (touching only ends the task), so
one run per (geometry, design) scores every radius and top height: the first time with
r <= R and z <= H, with S never left before. Writes ``reach.json``, per-geometry figures
and the raw trajectories (``traj_*.npz``).
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cone_obstacle_common import DEFAULT_DESIGNS, design_config, load_certificate, make_config, parse_list, resolve  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from ps2rl.cil.quadrotor_cone_obstacle_cbf import (  # noqa: E402
    ConeObstacleBCBFConfig,
    get_cached_runtime,
    make_recoverability_fn,
    obstacle_step,
    solve_backup_cbf_qp_batch_with_info,
)
from ps2rl.sets.quadrotor_cone_obstacle_sets import ConeObstacleSafeSet, point_controller  # noqa: E402

NOMINALS = ("center", "top", "horizontal", "rim")


# ------------------------------------------------------------------ starts and nominals
def make_starts(cfg, a):
    """(B, 10) start states and (B, 7) nominal parameters [mode, aim/start xyz, direction xyz]."""
    xs, ps, meta = [], [], []
    cx, cy = float(cfg.axis_x), float(cfg.axis_y)
    tx, ty = cfg.target_center
    centre = (tx, ty, float(cfg.ground_z) + 0.5 * (cfg.cylinder_top - float(cfg.ground_z)))
    for nom, y0, dz0, v0 in itertools.product(a.nominals.split(","), parse_list(a.y0), parse_list(a.dz0), parse_list(a.v0)):
        p0 = (cx + a.x0, cy + y0, float(cfg.tip_z) + dz0)
        d = np.asarray([1.0, 0.0, 0.0]) if a.x0 < 0 else np.asarray([-1.0, 0.0, 0.0])
        x = np.zeros(10, np.float32)
        x[0:3], x[3:6], x[6] = p0, v0 * d, 1.0
        mode = NOMINALS.index(nom)
        rim = (tx - float(d[0]) * a.rim_radius, ty, cfg.cylinder_top)
        anchor = {"center": centre, "top": (tx, ty, cfg.cylinder_top), "horizontal": p0, "rim": rim}[nom]
        xs.append(x)
        ps.append(np.r_[mode, anchor, d].astype(np.float32))
        meta.append({"nominal": nom, "y0": y0, "dz0": dz0, "v0": v0, "start": list(p0)})
    return jnp.asarray(np.stack(xs)), jnp.asarray(np.stack(ps)), meta


def make_nominal(cfg, a):
    ctrl = point_controller(cfg, (cfg.axis_x, cfg.axis_y, 1.0))
    k = jnp.asarray(ctrl.k_matrix, jnp.float32)
    u_star = jnp.asarray(ctrl.u_star, jnp.float32)
    lo, hi = jnp.asarray(cfg.action_low, jnp.float32), jnp.asarray(cfg.action_high, jnp.float32)
    cap, lead = float(a.max_position_error), float(a.lead)

    def one(x, prm):
        mode, anchor, d = prm[0], prm[1:4], prm[4:7]
        p = x[0:3]
        s = jnp.dot(p - anchor, d)
        aim_h = anchor + (s + lead) * d                     # on the start line, lead ahead
        aim_h = aim_h.at[2].set(anchor[2])                  # at the start altitude
        aim = jnp.where(mode == NOMINALS.index("horizontal"), aim_h, anchor)
        dp = p - aim
        n = jnp.linalg.norm(dp)
        dp = dp * jnp.minimum(1.0, cap / jnp.maximum(n, 1e-9))
        e = ctrl.error_state(x)
        e = e.at[0:3].set(dp)
        return jnp.clip(u_star - k @ e, lo, hi)

    return jax.vmap(one)


# ------------------------------------------------------------------ simulation
def make_chunk(bcfg, nominal, filtered: bool, n_steps: int):
    rt = get_cached_runtime(bcfg)
    step_b = jax.vmap(lambda x, u: obstacle_step(x, u, bcfg))

    def chunk(xb, prm):
        def body(x, _):
            u_ref = nominal(x, prm)
            if filtered:
                u, slack, _, info = solve_backup_cbf_qp_batch_with_info(x, u_ref, bcfg, rt)
                lam = info["safeguard_lambda"].astype(jnp.float32)
                found = info["safeguard_found"]
                slack = jnp.asarray(slack, jnp.float32).reshape(x.shape[0], -1)[:, 0]
            else:
                u, lam, found, slack = u_ref, jnp.ones(x.shape[0]), jnp.ones(x.shape[0], bool), jnp.zeros(x.shape[0])
            xn = step_b(x, u.astype(x.dtype))
            return xn, (x, u, u_ref, lam, found, slack)

        return jax.lax.scan(body, xb, None, length=n_steps)

    return jax.jit(chunk)


def simulate(bcfg, nominal, filtered, x0, prm, steps, chunk_steps, label):
    fn = make_chunk(bcfg, nominal, filtered, chunk_steps)
    x = x0
    logs = []
    t0 = time.time()
    for i in range(0, steps, chunk_steps):
        x, out = fn(x, prm)
        logs.append([np.asarray(o) for o in out])
        if filtered and (i // chunk_steps) % max(1, (steps // chunk_steps) // 4) == 0:
            print(f"    {label}: t = {(i + chunk_steps) * bcfg.dt:4.1f} s ({time.time() - t0:.0f} s)", flush=True)
    keys = ("x", "u", "u_ref", "lam", "found", "slack")
    return {k: np.concatenate([lg[j] for lg in logs], axis=0).swapaxes(0, 1) for j, k in enumerate(keys)}


# ------------------------------------------------------------------ scoring
def score(cfg, traj, radii, tops, dt):
    safe = ConeObstacleSafeSet.from_config(cfg)
    x = traj["x"]                                      # (B, T, 10)
    h = np.asarray(safe.component_values(jnp.asarray(x)))  # (B, T, 2)
    hmin = h.min(-1)
    unsafe_t = np.where((hmin < 0).any(1), np.argmax(hmin < 0, axis=1), -1)
    tx, ty = cfg.target_center
    r = np.hypot(x[..., 0] - tx, x[..., 1] - ty)
    z = x[..., 2]
    res = []
    for b in range(x.shape[0]):
        rec = {"unsafe_step": int(unsafe_t[b]), "min_h_cone": float(h[b, :, 0].min()), "min_h_ground": float(h[b, :, 1].min()),
               "touch_time": {}, "speed_at_touch": {}}
        for rad in radii:
            for top in tops:
                hit = (r[b] <= rad) & (z[b] <= top) & (z[b] >= cfg.ground_z)
                k = int(np.argmax(hit)) if hit.any() else -1
                ok = k >= 0 and (unsafe_t[b] < 0 or unsafe_t[b] > k)
                key = f"R{rad:g}_H{top:g}"
                rec["touch_time"][key] = float(k * dt) if ok else None
                rec["speed_at_touch"][key] = float(np.linalg.norm(x[b, k, 3:6])) if ok else None
        dev = np.abs(traj["u"][b] - traj["u_ref"][b]).max(-1)
        rec["filter_active_frac"] = float((dev > 1e-3).mean())
        rec["safeguard_min_lambda"] = float(traj["lam"][b].min())
        rec["safeguard_not_found_steps"] = int((~traj["found"][b].astype(bool)).sum())
        rec["final_pos"] = x[b, -1, 0:3].tolist()
        rec["final_speed"] = float(np.linalg.norm(x[b, -1, 3:6]))
        res.append(rec)
    return res


def summarize(results, radii, tops):
    out = {}
    for rad in radii:
        for top in tops:
            key = f"R{rad:g}_H{top:g}"
            tt = [r["touch_time"][key] for r in results]
            ok = [t for t in tt if t is not None]
            out[key] = {"success": len(ok) / len(tt), "median_time": float(np.median(ok)) if ok else None}
    out["unsafe"] = float(np.mean([r["unsafe_step"] >= 0 for r in results]))
    return out


# ------------------------------------------------------------------ figures
def plot_geometry(path, cfg, runs, meta, radius):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    safe = ConeObstacleSafeSet.from_config(cfg)
    rr = np.linspace(0, 4.0, 200)
    zw = np.asarray(safe.surface_z(rr))
    names = list(runs)
    noms = sorted(set(m["nominal"] for m in meta), key=NOMINALS.index)
    fig, axes = plt.subplots(len(noms), len(names), figsize=(2.6 * len(names), 2.6 * len(noms) + 0.5), sharex=True, sharey=True,
                             squeeze=False)
    for j, name in enumerate(names):
        x = runs[name]["x"]
        for i, nom in enumerate(noms):
            ax = axes[i, j]
            ax.fill_between(np.r_[-rr[::-1], rr] + cfg.axis_x, np.r_[zw[::-1], zw], 10, color="#555555", alpha=0.85, lw=0)
            ax.axhline(cfg.ground_z, color="k", lw=0.8)
            tx = cfg.target_center[0]
            ax.plot([tx - radius, tx - radius, tx + radius, tx + radius],
                    [cfg.ground_z, cfg.cylinder_top, cfg.cylinder_top, cfg.ground_z], color="#d62728", lw=1.0)
            dzs = sorted(set(mm["dz0"] for mm in meta))
            vmax = max(1e-9, max(mm["v0"] for mm in meta))
            for b, m in enumerate(meta):
                if m["nominal"] != nom or abs(m["y0"]) > 1e-9:
                    continue
                c = plt.cm.viridis(0.15 + 0.7 * (m["v0"] / vmax))
                ls = ("-", "--", ":", "-.")[dzs.index(m["dz0"]) % 4]
                ax.plot(x[b, :, 0], x[b, :, 2], color=c, lw=1.0, ls=ls)
            ax.set_xlim(min(cfg.axis_x - 3.3, min(mm["start"][0] for mm in meta) - 0.2), cfg.target_center[0] + 1.3)
            ax.set_ylim(cfg.ground_z - 0.1, max(cfg.tip_z + 1.6, max(mm["start"][2] for mm in meta) + 0.3))
            if i == 0:
                ax.set_title(name.replace("points:", "pt:").replace("approach_", "app_"), fontsize=8)
            if j == 0:
                ax.set_ylabel(f"nominal: {nom}\nz [m]", fontsize=8)
    fig.suptitle(f"x-z trajectories (y0 = 0; colour: start speed, solid / dashed / dotted: start height, low to high) - tip {cfg.tip_z:g} m, "
                 f"theta {cfg.cone_theta_deg:g} deg; red: cylinder r = {radius:g} m", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=105)
    plt.close(fig)


def replot(out_dir, vehicle, radius, only=None):
    """Redraw the trajectory figures of a finished run from its reach.json and traj_*.npz."""
    rep = json.loads((out_dir / "reach.json").read_text(encoding="utf-8"))
    a = rep["args"]
    for g in rep["geometries"]:
        cfg = make_config(vehicle, tip_z=g["tip_z"], cone_theta_deg=g["theta_deg"], cone_eps=g["eps"],
                          target_offset_x=float(a.get("target_offset_x", 0.0)), target_height=g["tops"][0])
        tag = f"tip{g['tip_z']:g}_th{g['theta_deg']:g}_eps{g['eps']:g}" + (f"_tx{a['target_offset_x']:g}" if a.get("target_offset_x") else "")
        runs = {n: dict(np.load(out_dir / f"traj_{tag}_{n.replace(':', '-').replace('@', '_at_').replace('/', 'over')}.npz"))
                for n in g["runs"] if only is None or n in only}
        plot_geometry(out_dir / f"reach_{tag}.png", cfg, runs, g["starts"], radius)
        print(f"  -> {out_dir / f'reach_{tag}.png'}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vehicle", default="docker_batch/configs/landing_x500_cone45_r0p3.json")
    ap.add_argument("--certificate", default="outputs/cone_obstacle/certificate.json")
    ap.add_argument("--designs", default=",".join(DEFAULT_DESIGNS))
    ap.add_argument("--replot", default="", help="only redraw the figures of a finished run in this output folder")
    ap.add_argument("--unfiltered", default="true", help="also run every nominal without the filter")
    ap.add_argument("--tip_z", default="1.2")
    ap.add_argument("--theta_deg", default="45")
    ap.add_argument("--eps", default="0.25")
    ap.add_argument("--alpha", type=float, default=10.0)
    ap.add_argument("--alpha_ground", type=float, default=20.0)
    ap.add_argument("--nominals", default=",".join(NOMINALS))
    ap.add_argument("--x0", type=float, default=-3.0, help="start offset from the axis along x")
    ap.add_argument("--y0", default="0,0.25")
    ap.add_argument("--dz0", default="0.5,1.0", help="start heights above the tip")
    ap.add_argument("--v0", default="0,0.5,1.0", help="start speeds toward the axis")
    ap.add_argument("--max_position_error", type=float, default=1.0)
    ap.add_argument("--lead", type=float, default=1.0)
    ap.add_argument("--rim_radius", type=float, default=0.25, help="cylinder radius the rim nominal aims at")
    ap.add_argument("--duration", type=float, default=8.0)
    ap.add_argument("--chunk_steps", type=int, default=50)
    ap.add_argument("--radii", default="0.1,0.25,0.5,0.75")
    ap.add_argument("--top_offsets", default="0,0.1,0.25",
                    help="cylinder tops scored, as depths below the tip [m] (the first one sets the nominals' cylinder)")
    ap.add_argument("--target_offset_x", type=float, default=0.0, help="cylinder centre from the axis along the approach (> 0: far side)")
    ap.add_argument("--plot_radius", type=float, default=0.25)
    ap.add_argument("--out", default="outputs/cone_obstacle/reach")
    a = ap.parse_args(argv)
    if a.replot:
        replot(resolve(a.replot), a.vehicle, a.plot_radius, only=[d for d in a.designs.split(",") if d] if a.designs != ",".join(DEFAULT_DESIGNS) else None)
        return

    cert = load_certificate(a.certificate)
    designs = [d for d in a.designs.split(",") if d.strip()]
    radii = parse_list(a.radii)
    out_dir = resolve(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {"args": vars(a), "geometries": []}
    t_all = time.time()
    for tip, th, eps in itertools.product(parse_list(a.tip_z), parse_list(a.theta_deg), parse_list(a.eps)):
        offs = parse_list(a.top_offsets)
        cfg = make_config(a.vehicle, tip_z=tip, cone_theta_deg=th, cone_eps=eps, target_offset_x=a.target_offset_x,
                          target_height=-1.0 if offs[0] == 0 and a.target_offset_x == 0 else tip - offs[0])
        tops = [round(float(cfg.tip_z) - o, 6) for o in offs]
        x0, prm, meta = make_starts(cfg, a)
        nominal = make_nominal(cfg, a)
        steps = int(round(a.duration / cfg.dt))
        steps -= steps % a.chunk_steps
        tag = f"tip{tip:g}_th{th:g}_eps{eps:g}" + (f"_tx{a.target_offset_x:g}" if a.target_offset_x else "")
        g = {"tip_z": tip, "theta_deg": th, "eps": eps, "tops": tops, "starts": meta, "runs": {}}
        runs = {}
        todo = ([("none", None)] if a.unfiltered.lower() == "true" else []) + [(d, d) for d in designs]
        for name, spec in todo:
            try:
                cfg_d = design_config(cfg, spec, cert) if spec else cfg
            except ValueError as exc:
                print(f"[{tag}] {name}: skipped ({exc})", flush=True)
                continue
            bcfg = ConeObstacleBCBFConfig(scenario=cfg_d, alpha=a.alpha, alpha_ground=a.alpha_ground)
            t1 = time.time()
            traj = simulate(bcfg, nominal, spec is not None, x0, prm, steps, a.chunk_steps, f"{tag} {name}")
            res = score(cfg, traj, radii, tops, cfg.dt)
            if spec is not None:   # starts outside C_N cannot be protected by the filter at all
                rec0 = np.asarray(make_recoverability_fn(bcfg)(x0)[0])
                for r_, ok in zip(res, rec0):
                    r_["start_in_CN"] = bool(ok)
            summ = {nom: summarize([r for r, m in zip(res, meta) if m["nominal"] == nom], radii, tops) for nom in a.nominals.split(",")}
            g["runs"][name] = {"base_set_c": cfg_d.base_set_c if spec else None, "summary": summ, "episodes": res}
            runs[name] = traj
            line = "  ".join(f"{nom}: unsafe {100 * s['unsafe']:3.0f}% | " + " ".join(
                f"R{rad:g} {100 * s[f'R{rad:g}_H{tops[0]:g}']['success']:3.0f}%" for rad in radii) for nom, s in summ.items())
            n_out = sum(1 for r_ in res if r_.get("start_in_CN") is False)
            print(f"[{tag}] {name:<38s} {line}  ({time.time() - t1:.0f} s{f'; {n_out} starts outside C_N' if n_out else ''})", flush=True)
            np.savez_compressed(out_dir / f"traj_{tag}_{name.replace(':', '-').replace('@', '_at_').replace('/', 'over')}.npz",
                                **{k: v for k, v in traj.items()})
        report["geometries"].append(g)
        plot_geometry(out_dir / f"reach_{tag}.png", cfg, runs, meta, a.plot_radius)
        (out_dir / "reach.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
        print(f"  -> {out_dir / f'reach_{tag}.png'}", flush=True)
    report["wall_time_s"] = time.time() - t_all
    (out_dir / "reach.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {out_dir / 'reach.json'} ({report['wall_time_s']:.0f} s)")


if __name__ == "__main__":
    main()
