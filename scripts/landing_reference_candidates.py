#!/usr/bin/env python
"""Candidate landing references that descend slowly toward a point just outside the platform.

Each reference starts in hover at height ``--z0`` and touches down softly (zero velocity, acceleration
and jerk at both ends) at ``offset`` beyond the platform rim. It descends along a line that makes an
angle ``phi`` with the vertical: phi > 0 drifts away from the pad axis as the vehicle descends, phi < 0
toward it, and phi = 0 is straight down onto the touchdown point. Because the cone narrows toward the
pad (45 deg) and the touchdown point lies outside it, every candidate leaves the cone on the way down:
where it leaves depends on phi and the offset, and how it moves horizontally depends on the shape.

  * shape "line":     straight line from the start to the touchdown point;
  * shape "vertical": the same start and end, but horizontal motion follows s(t)^2, so the vehicle
                      descends almost vertically first and drifts out late.

Kinematics are smooth septic profiles s(t); attitude, thrust and body rates come from differential
flatness (the colleague's generator), so every candidate is dynamically consistent with the env model.

    python scripts/landing_reference_candidates.py --out outputs/landing_ref_candidates      # grids
    python scripts/landing_reference_candidates.py --save B2_-20 \\
        --save_path ps2rl/envs/assets/quadrotor_landing_descent_reference.npz                # one of them
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from ps2rl.envs.assets import generate_quadrotor_landing_reference as gen  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--z0", type=float, default=2.0, help="start height above the pad [m]")
    p.add_argument("--duration", type=float, default=4.0, help="seconds from hover to touchdown")
    p.add_argument("--dt", type=float, default=0.02)
    p.add_argument("--offsets", default="0.1,0.2,0.3", help="touchdown distance beyond the platform rim [m]")
    p.add_argument("--angles", default="-30,-20,-10,0,10,20,30", help="descent angle from vertical [deg], + = away from the axis")
    p.add_argument("--shapes", default="line,vertical")
    p.add_argument("--miss_dir_deg", type=float, default=0.0, help="horizontal direction of the touchdown point")
    p.add_argument("--r0", type=float, default=0.3)
    p.add_argument("--theta_deg", type=float, default=45.0)
    p.add_argument("--eps", type=float, default=0.015)
    p.add_argument("--pad", default="0,0,0")
    p.add_argument("--backup", default="checkpoints/landing_phase1/floor_rec10_td3_seed0",
                   help="Phase-I checkpoint for the start-recoverability check ('' to skip)")
    p.add_argument("--out", default="outputs/landing_ref_candidates")
    p.add_argument("--save", default="", help="candidate id to write as a reference .npz")
    p.add_argument("--save_path", default="")
    return p.parse_args(argv)


def septic_unit(t: np.ndarray, T: float):
    """s(t) from 0 to 1 with zero velocity/acceleration/jerk at both ends, and its first three derivatives."""
    tau = np.clip(t / T, 0.0, 1.0)
    s = 35 * tau**4 - 84 * tau**5 + 70 * tau**6 - 20 * tau**7
    ds = (140 * tau**3 - 420 * tau**4 + 420 * tau**5 - 140 * tau**6) / T
    dds = (420 * tau**2 - 1680 * tau**3 + 2100 * tau**4 - 840 * tau**5) / T**2
    ddds = (840 * tau - 5040 * tau**2 + 8400 * tau**3 - 4200 * tau**4) / T**3
    return s, ds, dds, ddds


def shaped(s, ds, dds, ddds, shape: str):
    if shape == "line":
        return s, ds, dds, ddds
    if shape == "vertical":  # g = s^2
        return s * s, 2 * s * ds, 2 * (ds * ds + s * dds), 2 * (3 * ds * dds + s * ddds)
    raise ValueError(f"unknown shape {shape!r}")


def candidate_id(shape: str, offset_index: int, angle: float) -> str:
    return f"{'A' if shape == 'line' else 'B'}{offset_index + 1}_{'0' if angle == 0 else format(angle, '+g')}"


def build_candidate(args, offset: float, angle_deg: float, shape: str) -> dict:
    pad = np.array([float(v) for v in args.pad.split(",")])
    T, dt, z0 = args.duration, args.dt, args.z0
    t = np.linspace(0.0, T, int(round(T / dt)) + 1)
    d = np.array([np.cos(np.radians(args.miss_dir_deg)), np.sin(np.radians(args.miss_dir_deg)), 0.0])
    r_td = args.r0 + offset
    r_start = r_td - z0 * np.tan(np.radians(angle_deg))  # signed distance along d (negative: across the axis)
    p0 = pad + r_start * d + np.array([0.0, 0.0, z0])
    p1 = pad + r_td * d
    s, ds, dds, ddds = septic_unit(t, T)
    g, dg, ddg, dddg = shaped(s, ds, dds, ddds, shape)
    P = np.zeros((len(t), 3)); V = np.zeros_like(P); A = np.zeros_like(P); J = np.zeros_like(P)
    for i in range(2):
        P[:, i] = p0[i] + (p1[i] - p0[i]) * g
        V[:, i], A[:, i], J[:, i] = (p1[i] - p0[i]) * dg, (p1[i] - p0[i]) * ddg, (p1[i] - p0[i]) * dddg
    P[:, 2] = p0[2] + (p1[2] - p0[2]) * s
    V[:, 2], A[:, 2], J[:, 2] = (p1[2] - p0[2]) * ds, (p1[2] - p0[2]) * dds, (p1[2] - p0[2]) * ddds
    Q, a_cmd, omega = gen.flat_outputs_to_states(P, V, A, J, 0.0)
    h = gen.h_cone(P, pad[:2], pad[2], args.r0, args.theta_deg, args.eps)
    out = np.flatnonzero(h < 0)
    tilt = np.degrees(np.arccos(np.clip(1 - 2 * (Q[:, 1] ** 2 + Q[:, 2] ** 2), -1, 1)))
    return {
        "t": t, "P": P, "V": V, "Q": Q, "a_cmd": a_cmd, "omega": omega, "h": h,
        "offset": offset, "angle_deg": angle_deg, "shape": shape, "r_start": float(r_start), "r_touch": float(r_td),
        "h_start": float(h[0]), "min_h": float(h.min()),
        "exit_time": float(t[out[0]]) if len(out) else None,
        "exit_height": float(P[out[0], 2] - pad[2]) if len(out) else None,
        "time_outside": float(len(out) * dt),
        "peak_speed": float(np.linalg.norm(V, axis=1).max()),
        "peak_tilt": float(tilt.max()),
        "a_range": (float(a_cmd.min()), float(a_cmd.max())),
        "peak_omega": float(np.abs(omega).max()),
    }


def bundle_of(c: dict) -> dict:
    states = np.concatenate([c["P"], c["V"], c["Q"]], axis=1)
    rpy = np.array([gen.euler_zyx_from_rotmat(gen.quat_to_rotmat(q)) for q in c["Q"]])
    qn = np.linalg.norm(c["Q"], axis=1)
    qs = np.r_[0.0, np.linalg.norm(np.diff(c["Q"], axis=0), axis=1)]
    return dict(t=c["t"], states=states, a_cmd=c["a_cmd"], omega_cmd=c["omega"], roll=rpy[:, 0], pitch=rpy[:, 1],
                yaw=rpy[:, 2], quat_norm=qn, quat_step_norm=qs, max_a_cmd=np.array([c["a_cmd"].max()]),
                max_abs_omega=np.array([np.abs(c["omega"]).max()]), max_quat_step_norm=np.array([qs.max()]),
                max_quat_norm_error=np.array([np.abs(qn - 1).max()]))


def main():
    args = parse_args()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    offsets = [float(v) for v in args.offsets.split(",")]
    angles = [float(v) for v in args.angles.split(",")]
    shapes = [s.strip() for s in args.shapes.split(",")]
    cands = {}
    for shape in shapes:
        for oi, off in enumerate(offsets):
            for ang in angles:
                if shape != "line" and ang == 0:
                    continue  # straight down: both shapes coincide
                cands[candidate_id(shape, oi, ang)] = build_candidate(args, off, ang, shape)

    if args.save:
        if args.save not in cands:
            raise SystemExit(f"unknown candidate {args.save}; ids: {', '.join(cands)}")
        c = cands[args.save]
        path = Path(args.save_path or f"ps2rl/envs/assets/quadrotor_landing_{args.save.replace('+', 'p').replace('-', 'm').lower()}_reference.npz")
        b = bundle_of(c)
        np.savez(path, **b)
        pad = [float(v) for v in args.pad.split(",")]
        cfg = {"landing_config": {"generator": "scripts/landing_reference_candidates.py", "candidate": args.save,
                                  "shape": c["shape"], "angle_deg": c["angle_deg"], "offset_beyond_rim": c["offset"],
                                  "z0": args.z0, "duration": args.duration, "dt": args.dt, "miss_dir_deg": args.miss_dir_deg,
                                  "pad_x": pad[0], "pad_y": pad[1], "z_pad": pad[2], "r0": args.r0,
                                  "theta_deg": args.theta_deg, "eps": args.eps, "out": str(path)},
               "landing_trajectory_stats": {"num_points": int(len(c["t"])), "total_duration": float(c["t"][-1]),
                                            "h_start": c["h_start"], "min_h": c["min_h"], "exit_time": c["exit_time"],
                                            "exit_height": c["exit_height"], "peak_speed": c["peak_speed"],
                                            "peak_tilt_deg": c["peak_tilt"], "max_a_cmd": float(c["a_cmd"].max()),
                                            "max_abs_omega": c["peak_omega"]}}
        path.with_name(path.stem + "_config.json").write_text(json.dumps(cfg, indent=2))
        from types import SimpleNamespace

        gen.plot(b, SimpleNamespace(z0=args.z0, r0=args.r0, theta_deg=args.theta_deg, eps=args.eps, pad_x=pad[0], z_pad=pad[2]),
                 path.with_suffix(".png"))
        print(f"saved {args.save} -> {path} (+ _config.json, .png)")
        return

    # start recoverability under the Phase-I backup (hover start)
    rec = {}
    if args.backup:
        import jax.numpy as jnp

        from ps2rl.cil import quadrotor_landing_backup_cbf as L

        cfg = L.landing_bcbf_config_from_checkpoint(args.backup)
        lc = cfg.landing
        if not np.isclose([lc.cone_r0, lc.cone_theta_deg, lc.cone_eps], [args.r0, args.theta_deg, args.eps]).all():
            print(f"[warn] backup cone ({lc.cone_r0}, {lc.cone_theta_deg}, {lc.cone_eps}) differs from the candidates'")
        x0 = np.stack([np.concatenate([c["P"][0], c["V"][0], c["Q"][0]]) for c in cands.values()]).astype(np.float32)
        ok = np.asarray(L.make_recoverability_fn(cfg)(jnp.asarray(x0))[0])
        rec = dict(zip(cands, ok.tolist()))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pad = np.array([float(v) for v in args.pad.split(",")])
    tan = np.tan(np.radians(args.theta_deg))
    zz = np.linspace(0, args.z0 + 0.3, 100)
    rr = args.r0 + tan * zz
    d2 = np.array([np.cos(np.radians(args.miss_dir_deg)), np.sin(np.radians(args.miss_dir_deg))])
    s_all = np.concatenate([(c["P"][:, 0:2] - pad[:2]) @ d2 for c in cands.values()])
    xlim = (min(-args.r0 - 0.3, s_all.min() - 0.15), max(args.r0 + 0.5, s_all.max() + 0.15))
    figs = []
    for shape in shapes:
        cols = [a for a in angles if not (shape != "line" and a == 0)]
        fig, axs = plt.subplots(len(offsets), len(cols), figsize=(3.2 * len(cols), 3.7 * len(offsets)), squeeze=False,
                                constrained_layout=True)
        for oi, off in enumerate(offsets):
            for ci, ang in enumerate(cols):
                cid = candidate_id(shape, oi, ang)
                c = cands[cid]
                a = axs[oi, ci]
                a.fill_betweenx(zz, -rr, rr, color="tab:green", alpha=0.10, lw=0)
                a.plot(rr, zz, color="tab:green", lw=1)
                a.plot(-rr, zz, color="tab:green", lw=1)
                a.plot([-args.r0, args.r0], [0, 0], color="tab:green", lw=5, solid_capstyle="butt")
                s = (c["P"][:, 0:2] - pad[:2]) @ d2
                z = c["P"][:, 2] - pad[2]
                inside = c["h"] >= 0
                a.plot(np.where(inside, s, np.nan), z, color="k", lw=2)
                a.plot(np.where(~inside, s, np.nan), z, color="tab:red", lw=2)
                k = np.arange(0, len(s), int(round(0.5 / args.dt)))
                a.plot(s[k], z[k], "o", color="0.3", ms=2.5)  # every 0.5 s
                a.plot(s[0], z[0], "o", color="tab:blue", ms=6)
                a.plot(s[-1], z[-1], "x", color="tab:red", ms=8, mew=2)
                if c["exit_height"] is not None:
                    ie = np.flatnonzero(c["h"] < 0)[0]
                    a.plot(s[ie], z[ie], "D", color="tab:orange", ms=5)
                a.set_xlim(*xlim)
                a.set_ylim(-0.08, args.z0 + 0.25)
                a.set_aspect("equal")
                a.grid(alpha=0.25)
                a.tick_params(labelsize=7)
                ok = rec.get(cid)
                a.set_title(f"{cid}:  φ {ang:+g}°,  lands {100 * off:.0f} cm out", fontsize=9.5, fontweight="bold")
                txt = (f"leaves cone at z = {_fmt(c['exit_height'])} m (t = {_fmt(c['exit_time'], '.1f')} s), "
                       f"outside {c['time_outside']:.1f} s\n"
                       f"start h {c['h_start']:+.2f} m{'' if ok is None else (' (in C_N)' if ok else ' (NOT in C_N)')}, "
                       f"peak |v| {c['peak_speed']:.2f} m/s")
                a.set_xlabel(txt, fontsize=7)
            axs[oi, 0].set_ylabel("height above pad [m]", fontsize=8)
        name = {"line": "straight line from start to touchdown",
                "vertical": "near-vertical first, drifting out late (horizontal ∝ s²)"}[shape]
        fig.suptitle(f"Shape {'A' if shape == 'line' else 'B'}: {name} — hover start at {args.z0:g} m, "
                     f"{args.duration:g} s to a soft touchdown\nblack: inside the cone, red: outside; ◆ leaves the cone; "
                     f"dots every 0.5 s; ● start, ✕ touchdown; platform = green bar (r0 = {args.r0} m); "
                     f"x: distance from the pad axis toward the touchdown point [m]", fontsize=10)

        f = out / f"candidates_{'A' if shape == 'line' else 'B'}.png"
        fig.savefig(f, dpi=110)
        plt.close(fig)
        figs.append(str(f))

    rows = ["| id | shape | φ [deg] | beyond rim [cm] | start r [m] | start h [m] | leaves cone at z [m] | time outside [s] | "
            "min h [m] | peak speed [m/s] | peak tilt [deg] | start in C_N |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for cid, c in cands.items():
        rows.append(f"| {cid} | {c['shape']} | {c['angle_deg']:+g} | {100 * c['offset']:.0f} | {c['r_start']:+.2f} | "
                    f"{c['h_start']:+.2f} | {_fmt(c['exit_height'])} | {c['time_outside']:.1f} | {c['min_h']:+.2f} | "
                    f"{c['peak_speed']:.2f} | {c['peak_tilt']:.0f} | {rec.get(cid, '–')} |")
    (out / "candidates.md").write_text("\n".join(rows) + "\n")
    json.dump({cid: {k: v for k, v in c.items() if not isinstance(v, np.ndarray)} for cid, c in cands.items()},
              open(out / "candidates.json", "w"), indent=2, default=str)
    print("\n".join(rows))
    print("figures:", *figs)


def _fmt(v, f=".2f"):
    return "–" if v is None else format(v, f)


if __name__ == "__main__":
    main()
