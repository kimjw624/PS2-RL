#!/usr/bin/env python
"""Candidate references for the cone-obstacle task: fly toward the funnel, into it, and drop
onto the target cylinder under the tip. The filter has to keep the vehicle out of the funnel.

Each reference starts in hover on the approach side (``--x0`` from the axis, ``--dz0`` above
the tip) and ends in hover inside the target cylinder, ``--end_depth`` below its top, on the
axis. Horizontal and vertical motion are separate smooth septic profiles (zero velocity,
acceleration and jerk at both ends); when the drop starts sets the shape:

  * S  straight line: both start together and end together;
  * E  early drop: the drop starts halfway through the horizontal approach;
  * M  mid drop:   ... at 75 % of it;
  * L  late drop:  only once the vehicle is over the axis (an L: over the tip, then down).

The later the drop, the deeper and longer the reference runs inside the funnel. Attitude,
thrust and body rates come from differential flatness (the landing generator), so every
candidate is consistent with the env model; peaks are checked against the vehicle's box.

Every candidate is flown twice in the CPU model with the same tracker (the hover-LQR gain
around the reference with feedforward, position error capped at ``--max_position_error``):
without the filter and through the backup-CBF filter with the chosen base set (default: the
9-D hover ellipsoid beside the funnel, ``points:approach_beside@2``, with its LQR as the backup).

    python scripts/cone_obstacle_reference_candidates.py --out outputs/cone_obstacle/ref_candidates   # grids
    python scripts/cone_obstacle_reference_candidates.py --save L3-1-f \\
        --save_path ps2rl/envs/assets/quadrotor_cone_obstacle_l3_1_f_reference.npz                  # one of them

IDs: <shape><|x0|>-<dz0>-<s|f> (slow / fast approach), e.g. L3-1-f = late drop, start 3 m from
the axis and 1 m above the tip, fast.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cone_obstacle_common import design_config, load_certificate, make_config, parse_list, resolve  # noqa: E402
from landing_reference_candidates import bundle_of, septic_unit  # noqa: E402

import numpy as np  # noqa: E402

from ps2rl.envs.assets import generate_quadrotor_landing_reference as gen  # noqa: E402

SHAPES = {"S": None, "E": 0.5, "M": 0.75, "L": 1.0}   # drop start as a fraction of the horizontal phase (S: straight)
SHAPE_NAMES = {"S": "straight line", "E": "early drop (from 50 % of the approach)",
               "M": "mid drop (from 75 % of the approach)", "L": "late drop (over the axis, then down)"}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--vehicle", default="docker_batch/configs/landing_x500_cone45_r0p3.json")
    p.add_argument("--certificate", default="outputs/cone_obstacle/certificate.json")
    p.add_argument("--design", default="points:approach_beside@2", help="base set (see cone_obstacle_common.design_config)")
    p.add_argument("--tip_z", type=float, default=1.2)
    p.add_argument("--theta_deg", type=float, default=45.0)
    p.add_argument("--eps", type=float, default=0.25)
    p.add_argument("--radius", type=float, default=0.25, help="target cylinder radius [m]")
    p.add_argument("--top", type=float, default=-1.0, help="target cylinder top [m] (< 0: the tip)")
    p.add_argument("--end_depth", type=float, default=0.3, help="reference end point below the cylinder top [m]")
    p.add_argument("--shapes", default="E,M,L,S")
    p.add_argument("--x0", default="3,2", help="start distances from the axis (approach from -x) [m]")
    p.add_argument("--dz0", default="0.5,1,1.5", help="start heights above the tip [m]")
    p.add_argument("--speeds", default="0.6,1.0", help="average horizontal speeds: slow, fast [m/s]")
    p.add_argument("--drop_speed", type=float, default=0.5, help="average vertical speed of the drop [m/s]")
    p.add_argument("--y0", type=float, default=0.0, help="lateral start offset [m] (0: through the axis)")
    p.add_argument("--dt", type=float, default=0.02)
    p.add_argument("--hold", type=float, default=3.0, help="seconds simulated after the reference ends")
    p.add_argument("--max_position_error", type=float, default=1.0, help="tracker position-error cap [m] (0 = off)")
    p.add_argument("--alpha", type=float, default=10.0)
    p.add_argument("--alpha_ground", type=float, default=20.0)
    p.add_argument("--no_sim", action="store_true", help="only build and plot the references")
    p.add_argument("--out", default="outputs/cone_obstacle/ref_candidates")
    p.add_argument("--save", default="", help="candidate id to write as a reference .npz")
    p.add_argument("--save_path", default="")
    return p.parse_args(argv)


# ------------------------------------------------------------------ references
def candidate_id(shape, x0, dz0, fast):
    return f"{shape}{x0:g}-{dz0:g}-{'f' if fast else 's'}"


def septic_window(t, t0, T):
    s, ds, dds, ddds = septic_unit(t - t0, T)
    inside = (t >= t0) & (t <= t0 + T)
    s = np.where(t < t0, 0.0, np.where(t > t0 + T, 1.0, s))
    return s, np.where(inside, ds, 0.0), np.where(inside, dds, 0.0), np.where(inside, ddds, 0.0)


def build_candidate(cfg, a, shape, x0, dz0, v_avg):
    ax, ay = float(cfg.axis_x), float(cfg.axis_y)
    p0 = np.array([ax - x0, ay + a.y0, float(cfg.tip_z) + dz0])
    p1 = np.array([ax, ay, cfg.cylinder_top - a.end_depth])
    dxy = np.linalg.norm(p1[:2] - p0[:2])
    dz = p0[2] - p1[2]
    t_h = dxy / v_avg
    if SHAPES[shape] is None:                # straight line: one profile for both
        t_v, t_d = t_h, 0.0
    else:
        t_v = dz / a.drop_speed
        t_d = SHAPES[shape] * t_h
    T = max(t_h, t_d + t_v)
    t = np.arange(0.0, T + 1e-9, a.dt)
    sh = septic_window(t, 0.0, t_h)
    sv = septic_window(t, t_d, t_v)
    P = np.zeros((len(t), 3)); V = np.zeros_like(P); A = np.zeros_like(P); J = np.zeros_like(P)
    for i in range(3):
        s = sh if i < 2 else sv
        d = p1[i] - p0[i]
        P[:, i], V[:, i], A[:, i], J[:, i] = p0[i] + d * s[0], d * s[1], d * s[2], d * s[3]
    Q, a_cmd, omega = gen.flat_outputs_to_states(P, V, A, J, 0.0)
    r = np.hypot(P[:, 0] - ax, P[:, 1] - ay)
    e = float(cfg.cone_eps)
    h = np.sqrt(r * r + e * e) - e - cfg.tan_theta * (P[:, 2] - float(cfg.tip_z))
    inside = np.flatnonzero(h < 0)
    tilt = np.degrees(np.arccos(np.clip(1 - 2 * (Q[:, 1] ** 2 + Q[:, 2] ** 2), -1, 1)))
    # how far above the funnel's underside the reference goes (vertical depth into the funnel)
    surf = float(cfg.tip_z) + (np.sqrt(r * r + e * e) - e) / cfg.tan_theta
    lo, hi = cfg.action_low, cfg.action_high
    feasible = bool(a_cmd.min() >= lo[0] and a_cmd.max() <= hi[0] and np.all(np.abs(omega[:, 0:2]) <= hi[1])
                    and np.all(np.abs(omega[:, 2]) <= hi[3]))
    return {
        "t": t, "P": P, "V": V, "Q": Q, "a_cmd": a_cmd, "omega": omega, "h": h,
        "shape": shape, "x0": x0, "dz0": dz0, "v_avg": v_avg, "duration": float(t[-1]),
        "start": p0.tolist(), "end": p1.tolist(), "min_h": float(h.min()),
        "depth_in_funnel": float(max(0.0, (P[:, 2] - surf).max())),
        "time_in_funnel": float(len(inside) * a.dt),
        "enter_time": float(t[inside[0]]) if len(inside) else None,
        "enter_r": float(r[inside[0]]) if len(inside) else None,
        "peak_speed": float(np.linalg.norm(V, axis=1).max()), "peak_tilt": float(tilt.max()),
        "a_range": (float(a_cmd.min()), float(a_cmd.max())), "peak_omega": float(np.abs(omega).max()),
        "feasible": feasible,
    }


# ------------------------------------------------------------------ closed loop
def simulate(cfg_d, cands, a):
    import jax
    import jax.numpy as jnp

    from ps2rl.cil.quadrotor_cone_obstacle_cbf import (ConeObstacleBCBFConfig, get_cached_runtime, make_recoverability_fn,
                                                        obstacle_step, solve_backup_cbf_qp_batch_with_info)
    from ps2rl.sets.quadrotor_cone_obstacle_sets import point_controller

    bcfg = ConeObstacleBCBFConfig(scenario=cfg_d, alpha=a.alpha, alpha_ground=a.alpha_ground)
    rt = get_cached_runtime(bcfg)
    ids = list(cands)
    n_ref = max(len(c["t"]) for c in cands.values())
    steps = n_ref + int(round(a.hold / a.dt))
    steps += (-steps) % 50
    xr = np.zeros((len(ids), steps, 10), np.float32)
    ur = np.zeros((len(ids), steps, 4), np.float32)
    for b, cid in enumerate(ids):
        c = cands[cid]
        x = np.concatenate([c["P"], c["V"], c["Q"]], 1)
        u = np.concatenate([c["a_cmd"][:, None], c["omega"]], 1)
        k = len(x)
        xr[b, :k], ur[b, :k] = x, u
        xr[b, k:], ur[b, k:] = x[-1], u[-1]
    ctrl = point_controller(cfg_d, (cfg_d.axis_x, cfg_d.axis_y, 1.0))
    kmat = jnp.asarray(ctrl.k_matrix, jnp.float32)
    lo, hi = jnp.asarray(cfg_d.action_low, jnp.float32), jnp.asarray(cfg_d.action_high, jnp.float32)
    cap = float(a.max_position_error)

    def track(x, x_ref, u_ref):
        de = ctrl.error_state(x) - ctrl.error_state(x_ref)
        n = jnp.linalg.norm(de[0:3])
        de = de.at[0:3].multiply(jnp.minimum(1.0, cap / jnp.maximum(n, 1e-9)) if cap > 0 else 1.0)
        return jnp.clip(u_ref - kmat @ de, lo, hi)

    track_b = jax.vmap(track)
    step_b = jax.vmap(lambda x, u: obstacle_step(x, u, bcfg))

    def make(filtered, n):
        def chunk(x, xr_c, ur_c):
            def body(xx, ref):
                x_ref, u_ref = ref
                u_nom = track_b(xx, x_ref, u_ref)
                if filtered:
                    u, slack, _, info = solve_backup_cbf_qp_batch_with_info(xx, u_nom, bcfg, rt)
                    lam = info["safeguard_lambda"].astype(jnp.float32)
                else:
                    u, lam = u_nom, jnp.ones(xx.shape[0])
                return step_b(xx, u.astype(xx.dtype)), (xx, u, u_nom, lam)
            return jax.lax.scan(body, x, (jnp.swapaxes(xr_c, 0, 1), jnp.swapaxes(ur_c, 0, 1)))
        return jax.jit(chunk)

    out = {}
    x0 = jnp.asarray(xr[:, 0])
    out["start_in_CN"] = np.asarray(make_recoverability_fn(bcfg)(x0)[0])
    for name, filtered in (("unfiltered", False), ("filtered", True)):
        fn, x, logs, t0 = make(filtered, 50), x0, [], time.time()
        for i in range(0, steps, 50):
            x, o = fn(x, jnp.asarray(xr[:, i:i + 50]), jnp.asarray(ur[:, i:i + 50]))
            logs.append([np.asarray(v) for v in o])
        out[name] = {k: np.concatenate([lg[j] for lg in logs], 0).swapaxes(0, 1) for j, k in enumerate(("x", "u", "u_nom", "lam"))}
        print(f"  {name}: {len(ids)} references x {steps} steps in {time.time() - t0:.0f} s", flush=True)
    out["ids"], out["x_ref"] = ids, xr
    return out


def score(cfg, sim, cands, a):
    res = {}
    e = float(cfg.cone_eps)
    for b, cid in enumerate(sim["ids"]):
        rec = {"start_in_CN": bool(sim["start_in_CN"][b])}
        for name in ("unfiltered", "filtered"):
            x = sim[name]["x"][b]
            r = np.hypot(x[:, 0] - cfg.axis_x, x[:, 1] - cfg.axis_y)
            h_c = np.sqrt(r * r + e * e) - e - cfg.tan_theta * (x[:, 2] - cfg.tip_z)
            h_g = x[:, 2] - cfg.ground_z
            bad = (h_c < 0) | (h_g < 0)
            k_bad = int(np.argmax(bad)) if bad.any() else -1
            hit = (r <= a.radius) & (x[:, 2] <= cfg.cylinder_top) & (x[:, 2] >= cfg.ground_z)
            k_hit = int(np.argmax(hit)) if hit.any() else -1
            ok = k_hit >= 0 and (k_bad < 0 or k_bad > k_hit)
            dev = np.linalg.norm(x[:, 0:3] - sim["x_ref"][b, :, 0:3], axis=1)
            rec[name] = {"unsafe": k_bad >= 0, "min_h_cone": float(h_c.min()), "min_h_ground": float(h_g.min()),
                         "touch_time": float(k_hit * a.dt) if ok else None,
                         "speed_at_touch": float(np.linalg.norm(x[k_hit, 3:6])) if ok else None,
                         "final_dist_to_end": float(np.linalg.norm(x[-1, 0:3] - np.asarray(cands[cid]["end"]))),
                         "max_dev_from_ref": float(dev.max()), "final_speed": float(np.linalg.norm(x[-1, 3:6]))}
            if name == "filtered":
                du = np.abs(sim[name]["u"][b] - sim[name]["u_nom"][b]).max(-1)
                rec[name]["filter_active_frac"] = float((du > 1e-3).mean())
                rec[name]["min_safeguard_lambda"] = float(sim[name]["lam"][b].min())
                rec[name]["stalled"] = bool(k_hit < 0 and rec[name]["final_speed"] < 0.1)
        res[cid] = rec
    return res


# ------------------------------------------------------------------ figures
def draw_scene(ax, cfg, base_pt, base_ext, xlim, ylim):
    from matplotlib.patches import Ellipse, Rectangle

    r = np.linspace(0, max(abs(xlim[0]), abs(xlim[1])) + 0.5, 300)
    zw = float(cfg.tip_z) + (np.sqrt(r * r + cfg.cone_eps**2) - cfg.cone_eps) / cfg.tan_theta
    ax.fill_between(np.r_[-r[::-1], r] + cfg.axis_x, np.r_[zw[::-1], zw], ylim[1] + 5, color="0.55", alpha=0.55, lw=0)
    ax.axhline(cfg.ground_z, color="k", lw=0.8)
    ax.add_patch(Rectangle((cfg.axis_x - cfg.target_radius, cfg.ground_z), 2 * cfg.target_radius, cfg.cylinder_top - cfg.ground_z,
                           fc="tab:red", ec="tab:red", alpha=0.15, lw=1.0))
    if base_pt is not None:
        ax.add_patch(Ellipse((base_pt[0], base_pt[2]), 2 * base_ext[0], 2 * base_ext[1], fc="tab:green", ec="tab:green", alpha=0.35))
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.grid(alpha=0.2)
    ax.tick_params(labelsize=7)


def plot_grids(out, cfg, cands, res, sim, a, base_pt, base_ext):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    shapes = [s for s in a.shapes.split(",") if s]
    x0s, dzs, vs = parse_list(a.x0), parse_list(a.dz0), parse_list(a.speeds)
    xlim = (cfg.axis_x - max(x0s) - 0.4, cfg.axis_x + 1.0)
    ylim = (cfg.ground_z - 0.1, cfg.tip_z + max(dzs) + 0.4)
    files = []
    idx = {cid: i for i, cid in enumerate(sim["ids"])} if sim else {}
    for shape in shapes:
        cols = [(x0, v) for x0 in x0s for v in vs]
        fig, axs = plt.subplots(len(dzs), len(cols), figsize=(3.5 * len(cols), 3.0 * len(dzs) + 0.9), squeeze=False,
                                constrained_layout=True)
        for i, dz0 in enumerate(dzs):
            for j, (x0, v) in enumerate(cols):
                cid = candidate_id(shape, x0, dz0, v == max(vs))
                c = cands[cid]
                ax = axs[i, j]
                draw_scene(ax, cfg, base_pt, base_ext, xlim, ylim)
                P = c["P"]
                inn = c["h"] < 0
                ax.plot(np.where(~inn, P[:, 0], np.nan), P[:, 2], "--", color="k", lw=1.3)
                ax.plot(np.where(inn, P[:, 0], np.nan), P[:, 2], "--", color="tab:red", lw=1.6)
                k = np.arange(0, len(P), int(round(1.0 / a.dt)))
                ax.plot(P[k, 0], P[k, 2], "o", color="0.25", ms=2.2)
                ax.plot(P[0, 0], P[0, 2], "o", color="tab:blue", ms=5)
                txt = (f"ref: {c['duration']:.1f} s, peak {c['peak_speed']:.1f} m/s, tilt {c['peak_tilt']:.0f}°, "
                       f"{'' if c['feasible'] else 'OUT OF BOX, '}in funnel {c['time_in_funnel']:.1f} s / {100 * c['depth_in_funnel']:.0f} cm")
                if sim:
                    b = idx[cid]
                    xu, xf = sim["unfiltered"]["x"][b], sim["filtered"]["x"][b]
                    ax.plot(xu[:, 0], xu[:, 2], color="tab:orange", lw=1.0, alpha=0.9)
                    ax.plot(xf[:, 0], xf[:, 2], color="tab:blue", lw=1.6)
                    rf, ru = res[cid]["filtered"], res[cid]["unfiltered"]
                    tt = "–" if rf["touch_time"] is None else f"{rf['touch_time']:.1f} s"
                    txt += (f"\nno filter: {'COLLIDES' if ru['unsafe'] else 'safe'} (min h {100 * ru['min_h_cone']:+.0f} cm) | "
                            f"filter: {'UNSAFE' if rf['unsafe'] else 'safe'}, touch {tt}"
                            f"{', stalls' if rf['stalled'] else ''}, active {100 * rf['filter_active_frac']:.0f} %"
                            f"{'' if res[cid]['start_in_CN'] else ', start not in C_N'}")
                ax.set_title(f"{cid}   (x0 {x0:g} m, +{dz0:g} m, {'fast' if v == max(vs) else 'slow'} {v:g} m/s)",
                             fontsize=9, fontweight="bold")
                ax.set_xlabel(txt, fontsize=6.3)
        fig.suptitle(f"Shape {shape}: {SHAPE_NAMES[shape]} — end {a.end_depth:g} m inside the cylinder (red, r {a.radius:g} m); "
                     f"tip {cfg.tip_z:g} m, θ {cfg.cone_theta_deg:g}°, ε {cfg.cone_eps:g}\n"
                     "dashed: reference (red where it is inside the funnel), dots every 1 s; orange: tracker without the filter; "
                     f"blue: tracker through the filter; green: base set ({a.design})", fontsize=9.5)
        f = out / f"candidates_{shape}.png"
        fig.savefig(f, dpi=105)
        plt.close(fig)
        files.append(str(f))
    return files


def main(argv=None):
    a = parse_args(argv)
    cert = load_certificate(a.certificate)
    cfg = make_config(a.vehicle, tip_z=a.tip_z, cone_theta_deg=a.theta_deg, cone_eps=a.eps, target_radius=a.radius,
                      target_height=a.top)
    cands = {}
    for shape, x0, dz0, v in itertools.product([s for s in a.shapes.split(",") if s], parse_list(a.x0), parse_list(a.dz0),
                                               parse_list(a.speeds)):
        cands[candidate_id(shape, x0, dz0, v == max(parse_list(a.speeds)))] = build_candidate(cfg, a, shape, x0, dz0, v)

    if a.save:
        if a.save not in cands:
            raise SystemExit(f"unknown candidate {a.save}; ids: {', '.join(cands)}")
        c = cands[a.save]
        path = resolve(a.save_path or f"ps2rl/envs/assets/quadrotor_cone_obstacle_{a.save.replace('-', '_').replace('.', 'p').lower()}_reference.npz")
        b = bundle_of(c)
        np.savez(path, **b)
        meta = {"cone_obstacle_reference": {"generator": "scripts/cone_obstacle_reference_candidates.py", "candidate": a.save,
                                            "shape": SHAPE_NAMES[c["shape"]], "x0": c["x0"], "dz0": c["dz0"], "v_avg": c["v_avg"],
                                            "drop_speed": a.drop_speed, "y0": a.y0, "dt": a.dt, "start": c["start"], "end": c["end"],
                                            "scenario": {k: v for k, v in cfg.as_dict().items() if not k.startswith("lqr_")}},
                "stats": {k: c[k] for k in ("duration", "min_h", "depth_in_funnel", "time_in_funnel", "enter_time", "enter_r",
                                            "peak_speed", "peak_tilt", "a_range", "peak_omega", "feasible")}}
        path.with_name(path.stem + "_config.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"saved {a.save} -> {path} (+ _config.json)")
        return

    out = resolve(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg_d = design_config(cfg, a.design, cert)
    base_pt, base_ext = None, None
    if cfg_d.base_set == "points" and len(cfg_d.hover_points) == 1:
        base_pt = cfg_d.hover_points[0]
        m_h, m_v = (float(cert["lqr9"]["metric"][k]) for k in ("m_h", "m_v"))
        base_ext = (np.sqrt(cfg_d.base_set_c / m_h), np.sqrt(cfg_d.base_set_c / m_v))
    sim, res = None, {}
    if not a.no_sim:
        print(f"{len(cands)} candidates; base set {a.design} (c = {cfg_d.base_set_c:.2f})", flush=True)
        sim = simulate(cfg_d, cands, a)
        res = score(cfg, sim, cands, a)
    files = plot_grids(out, cfg, cands, res, sim, a, base_pt, base_ext)

    rows = ["| id | shape | start x0 / +z [m] | speed [m/s] | duration [s] | in funnel [s] / depth [cm] | peak speed [m/s] / tilt [°] | "
            "in box | no filter | filter: safe | filter: touch [s] | filter: stalls | filter active | start in C_N |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for cid, c in cands.items():
        r = res.get(cid)
        if r:
            u, f = r["unfiltered"], r["filtered"]
            tail = (f"{'collides' if u['unsafe'] else 'safe'} | {'no' if f['unsafe'] else 'yes'} | "
                    f"{'–' if f['touch_time'] is None else format(f['touch_time'], '.1f')} | {'yes' if f['stalled'] else 'no'} | "
                    f"{100 * f['filter_active_frac']:.0f} % | {'yes' if r['start_in_CN'] else 'no'} |")
        else:
            tail = "– | – | – | – | – | – |"
        rows.append(f"| {cid} | {c['shape']} | {c['x0']:g} / {c['dz0']:g} | {c['v_avg']:g} | {c['duration']:.1f} | "
                    f"{c['time_in_funnel']:.1f} / {100 * c['depth_in_funnel']:.0f} | {c['peak_speed']:.2f} / {c['peak_tilt']:.0f} | "
                    f"{'yes' if c['feasible'] else 'NO'} | " + tail)
    (out / "candidates.md").write_text("\n".join(rows) + "\n", encoding="utf-8")
    meta = {cid: {**{k: v for k, v in c.items() if not isinstance(v, np.ndarray)}, **({"sim": res[cid]} if cid in res else {})}
            for cid, c in cands.items()}
    (out / "candidates.json").write_text(json.dumps({"args": vars(a), "base_set_c": cfg_d.base_set_c, "candidates": meta}, indent=1,
                                                    default=str) + "\n", encoding="utf-8")
    if sim:
        np.savez_compressed(out / "trajectories.npz", ids=np.asarray(sim["ids"]), x_ref=sim["x_ref"],
                            x_unfiltered=sim["unfiltered"]["x"], x_filtered=sim["filtered"]["x"])
    print("\n".join(rows))
    print("figures:", *files)


if __name__ == "__main__":
    main()
