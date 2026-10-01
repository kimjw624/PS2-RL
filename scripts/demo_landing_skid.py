"""Skid-out landing demo: the tracker heads for a spot beside the platform, the filter lands it on it.

The reference (``quadrotor_landing_skid_reference.npz`` by default; made with the colleague's
generator, ``--offset 0.5 --offset_dir_deg 30``) starts inside the approach cone, skids out of it
near the ground and touches down 0.5 m from the pad centre, i.e. 0.2 m beyond the platform rim
(the cone footprint, radius r0). Each candidate tracker (an LQR around the moving reference with
feedforward, possibly re-weighted) is run

  * alone:          it follows the reference out of the cone and lands beside the platform;
  * through the filter (landing BCBF-QP + discrete-time safeguard, Phase-I backup from --backup),
    once per filter setting: it must stay in the cone and above the floor at every step and
    touch down on the platform.

Ground contact: the pad plane extends beyond the platform. A vehicle is landed at the first step
with height <= --zeta_touch and speed <= --v_touch. Inside the cone at that height means on the
platform. From then on it rests there with motors off, the way a land detector disarms a real
vehicle. Safety and intervention statistics cover the flight up to that step;
``--no_land_latch`` keeps flying instead.

Why filter settings matter here: the QP picks the safe action closest to the tracker's. With
equal weights on (a_cmd, omega), the cheapest way to stay off the cone wall near the pad is
extra thrust, so the vehicle climbs the wall and hovers there; it never reaches the floor.
Making roll and pitch rates cheaper than thrust (``control_weights=1/w/w/1``) corrects through
tilt instead. The commanded descent survives, and the vehicle slides along the wall onto the
platform. Yaw rate keeps weight 1: it does not move the vehicle, and making it cheap only lets the
QP exploit the backup's yaw dependence. The safe set, the rows, C_N and the backup are identical
in every setting.

Starts: the reference's first state plus random perturbations, kept only if they lie inside the
cone with margin --start_margin and are backup-recoverable (in C_N).

The selected (tracker, setting) pair lands on the platform from every start, safely. Ties are
broken by --select_by: by default the pair whose worst touchdown is farthest inside the platform
rim. The script renders case/actuator figures and a GIF for it.

    python scripts/demo_landing_skid.py --backup checkpoints/landing_phase1/floor_rec10_td3_seed0
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
    p.add_argument("--backup", default="checkpoints/landing_phase1/floor_rec10_td3_seed0")
    p.add_argument("--reference", default="ps2rl/envs/assets/quadrotor_landing_skid_reference.npz")
    p.add_argument("--settings",
                   default="uniform:alpha=10,alpha_floor=20;"
                           "tilt3:alpha=10,alpha_floor=20,control_weights=1/0.3/0.3/1;"
                           "tilt10:alpha=10,alpha_floor=20,control_weights=1/0.1/0.1/1",
                   help="name:KEY=VAL,...;... filter settings (QuadrotorLandingBCBFConfig fields; tuples '/'-separated)")
    p.add_argument("--trackers", default="lqr:;lqr_qz10:lqr_q_z=10;lqr_qz30:lqr_q_z=30",
                   help="name:KEY=VAL,...;... LQR-weight overrides (QuadrotorLandingConfig fields, e.g. lqr_q_z=10)")
    p.add_argument("--n_starts", type=int, default=12, help="random perturbed starts besides the nominal one")
    p.add_argument("--perturb_pos", type=float, default=0.15)
    p.add_argument("--perturb_vel", type=float, default=0.3)
    p.add_argument("--start_margin", type=float, default=0.05, help="starts must have h_cone >= this [m]")
    p.add_argument("--hold_time", type=float, default=None, help="seconds after the reference ends (default: backup horizon T)")
    p.add_argument("--zeta_touch", type=float, default=0.02)
    p.add_argument("--v_touch", type=float, default=0.3)
    p.add_argument("--no_land_latch", action="store_true", help="keep flying after touchdown")
    p.add_argument("--select_by", choices=("rim_margin", "intervention", "time"), default="rim_margin",
                   help="among pairs that land on the platform from every start: largest worst-case distance inside "
                        "the platform rim / least filter intervention / fastest median touchdown")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gif_all", action="store_true", help="figures and GIF for every pair, not only the selected one")
    p.add_argument("--gif_pairs", default="", help="extra pairs to render, 'tracker|setting,...' (e.g. 'lqr_qz30|uniform')")
    p.add_argument("--gif_slowdown", type=float, default=0.25)
    p.add_argument("--gif_stride", type=int, default=1)
    p.add_argument("--gif_tail", type=float, default=0.6, help="seconds shown after the last touchdown in the GIF")
    p.add_argument("--gif_zoom_margin", type=float, default=0.25,
                   help="close-up panel: margin [m] around the platform and the reference touchdown (<= 0: no close-up)")
    p.add_argument("--out", default="outputs/landing_skid_demo")
    return p.parse_args(argv)


ARGS = parse_args()
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
from ps2rl.envs.quadrotor_landing_config import QuadrotorLandingConfig  # noqa: E402
from ps2rl.evaluation import landing_filter_reach as R  # noqa: E402
from ps2rl.evaluation.landing_phase2_run import parse_field_overrides  # noqa: E402
from ps2rl.plotting import landing_phase2_plots as lp  # noqa: E402

LABELS = {"unfiltered": "tracker only", "filtered": "tracker + BCBF filter"}


def _jd(o):
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def _named(spec: str, cls, extra=None) -> list[tuple[str, dict]]:
    out = []
    for chunk in [c for c in spec.split(";") if c.strip()]:
        name, _, body = chunk.partition(":")
        items = [t for t in body.split(",") if t.strip()]
        out.append((name.strip(), parse_field_overrides(items, cls, extra=extra) if items else {}))
    return out


def _f(v, fmt=".3f"):
    return "–" if v is None else format(v, fmt)


def contact_index(traj, pad_z, zeta_touch, v_touch):
    """First step with height <= zeta_touch and speed <= v_touch (-1: never)."""
    c = (traj[..., 2] - pad_z <= zeta_touch) & (np.linalg.norm(traj[..., 3:6], axis=-1) <= v_touch)
    return np.where(c.any(axis=-1), c.argmax(axis=-1), -1)


def latch(rec: dict, idx: np.ndarray) -> dict:
    """After touchdown the vehicle rests where it touched down, motors off (u = NaN in the plots)."""
    out = {k: np.array(v, copy=True) for k, v in rec.items()}
    for e, k in enumerate(idx):
        if k < 0:
            continue
        out["traj"][e, k + 1:] = out["traj"][e, k]
        out["traj"][e, k + 1:, 3:6] = 0.0
        for key in ("u", "u_raw"):
            if key in out:
                out[key][e, k:] = np.nan
        for key in ("du_norm", "slack"):
            if key in out:
                out[key][e, k:] = 0.0
    return out


def flight_mask(idx: np.ndarray, steps: int) -> np.ndarray:
    """(episodes, steps) True for the steps flown before touchdown (all steps if none)."""
    k = np.arange(steps)[None, :]
    end = np.where(idx >= 0, idx, steps)[:, None]
    return k < end


def main():
    args = ARGS
    t_start = time.time()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    settings = _named(args.settings, L.QuadrotorLandingBCBFConfig, extra={"floor_constraint": bool})
    trackers = _named(args.trackers, QuadrotorLandingConfig)

    # the first setting fixes geometry, starts and the unfiltered runs (geometry is the same in all)
    cfg0 = L.landing_bcbf_config_from_checkpoint(args.backup, qp_solve_dtype="float64", **settings[0][1])
    lc = cfg0.landing
    if not lc.floor_constraint:
        print("[warn] the backup's safe set has no floor; add floor_constraint=on to the settings")
    pad = np.array([lc.pad_x, lc.pad_y, lc.pad_z])
    half = 0.5 * (np.asarray(cfg0.action_high) - np.asarray(cfg0.action_low))

    b = np.load(args.reference)
    ref_x = np.asarray(b["states"], np.float32)
    ref_u = np.concatenate([np.asarray(b["a_cmd"])[:, None], np.asarray(b["omega_cmd"])], 1).astype(np.float32)
    ref_t = np.asarray(b["t"]) if "t" in b else np.arange(len(ref_x)) * lc.dt
    T_ref = ref_x.shape[0]
    hold = lc.T if args.hold_time is None else float(args.hold_time)
    steps = T_ref - 1 + int(round(hold / lc.dt))
    h_ref = np.asarray(L.cone_value(jnp.asarray(ref_x), cfg0))
    td_ref = ref_x[-1, 0:3] - pad
    print(f"[ref] {args.reference}: h(start) {h_ref[0]:+.3f}, min h {h_ref.min():+.3f}, touchdown "
          f"{np.hypot(*td_ref[:2]):.2f} m from the pad centre (platform radius r0 = {lc.cone_r0}); {steps} steps", flush=True)

    # starts: nominal + perturbations, inside the cone (with margin) and in C_N of every setting
    projs = {}
    for sname, sover in settings:
        cfg = L.landing_bcbf_config_from_checkpoint(args.backup, qp_solve_dtype="float64", **sover)
        if (cfg.landing.cone_r0, cfg.landing.floor_constraint) != (lc.cone_r0, lc.floor_constraint):
            raise SystemExit(f"setting '{sname}' changes the safe set; settings may only change the filter")
        proj = L.QuadrotorLandingBackupCBFProjector(cfg)
        projs[sname] = (cfg, proj, L.make_recoverability_fn(cfg, proj.runtime))
    rng = np.random.default_rng(args.seed)
    pert = np.zeros((args.n_starts * 4 + 1, 10), np.float32)
    pert[1:, 0:3] = rng.uniform(-args.perturb_pos, args.perturb_pos, (len(pert) - 1, 3))
    pert[1:, 3:6] = rng.uniform(-args.perturb_vel, args.perturb_vel, (len(pert) - 1, 3))
    cand = ref_x[0][None] + pert
    ok = np.asarray(L.cone_value(jnp.asarray(cand), cfg0)) >= args.start_margin
    for cfg, proj, rec_fn in projs.values():
        ok &= np.asarray(rec_fn(jnp.asarray(cand))[0])
    keep = np.flatnonzero(ok)
    if keep.size == 0 or keep[0] != 0:
        raise SystemExit("the reference's own start is not inside the cone / in C_N; check the reference and backup")
    x0 = cand[keep[: args.n_starts + 1]]
    print(f"[starts] {len(x0)} (nominal + {len(x0) - 1} perturbed; inside the cone with h >= {args.start_margin}, in C_N)", flush=True)

    ref_xj, ref_uj = jnp.asarray(ref_x), jnp.asarray(ref_u)
    k_ref = np.minimum(np.arange(steps + 1), T_ref - 1)
    trk, pairs, store = {}, {}, {}
    for tname, tover in trackers:
        act = jax.jit(jax.vmap(R.tracking_lqr(cfg0, tover or None), in_axes=(0, None, None)))

        def nominal(x, k, act=act):
            i = min(k, T_ref - 1)
            return act(x, ref_xj[i], ref_uj[i])

        t0 = time.time()
        res_u = R.rollout_nominal(cfg0, projs[settings[0][0]][1], x0, nominal, steps, filtered=False)
        c_u = contact_index(res_u["traj"], pad[2], args.zeta_touch, args.v_touch)
        h_u_raw = np.asarray(L.cone_value(jnp.asarray(res_u["traj"]), cfg0))
        on_pf_u = np.array([c >= 0 and h_u_raw[e, c] >= 0 for e, c in enumerate(c_u)])
        rad_u = np.array([np.hypot(*(res_u["traj"][e, c, 0:2] - pad[:2])) if c >= 0 else np.nan for e, c in enumerate(c_u)])
        err_u = np.linalg.norm(res_u["traj"][:, :, 0:3] - ref_x[k_ref][None, :, 0:3], axis=-1)
        rec_u = {"traj": res_u["traj"], "u": res_u["u"]}
        if not args.no_land_latch:
            rec_u = latch(rec_u, c_u)
        trk[tname] = {
            "lqr_overrides": tover,
            "leaves_cone_rate": float((h_u_raw.min(1) < 0).mean()),
            "min_h_cone": float(h_u_raw.min()),
            "touchdown_rate": float((c_u >= 0).mean()),
            "touchdown_on_platform_rate": float(on_pf_u.mean()),
            "touchdown_radial_median": float(np.nanmedian(rad_u)) if np.isfinite(rad_u).any() else None,
            "tracking_error_mean": float(err_u[:, :T_ref].mean()),
        }
        print(f"[{tname}] alone ({time.time() - t0:.0f}s): leaves the cone {trk[tname]['leaves_cone_rate']:.2f}, touches down "
              f"{trk[tname]['touchdown_rate']:.2f} at {_f(trk[tname]['touchdown_radial_median'], '.2f')} m from centre "
              f"(on the platform {trk[tname]['touchdown_on_platform_rate']:.2f}), tracking error {trk[tname]['tracking_error_mean']:.3f} m",
              flush=True)

        for sname, _ in settings:
            cfg, proj, _ = projs[sname]
            t0 = time.time()
            res_f = R.rollout_nominal(cfg, proj, x0, nominal, steps, filtered=True)
            h_raw = np.asarray(L.cone_value(jnp.asarray(res_f["traj"]), cfg))
            c_f = contact_index(res_f["traj"], pad[2], args.zeta_touch, args.v_touch)
            on_pf = np.array([c >= 0 and h_raw[e, c] >= 0 for e, c in enumerate(c_f)])
            c_eval = np.full_like(c_f, -1) if args.no_land_latch else c_f
            fly_x = flight_mask(c_eval, steps + 1)          # states up to and including touchdown
            fly_x |= np.arange(steps + 1)[None, :] == np.where(c_eval >= 0, c_eval, 0)[:, None]
            fly_u = flight_mask(c_eval, steps)              # actions flown before touchdown
            rel = res_f["traj"][..., 0:3] - pad
            du = np.linalg.norm((res_f["u"] - res_f["u_ref"]) / half, axis=-1)
            rad_f = np.array([np.hypot(*rel[e, c, 0:2]) if c >= 0 else np.nan for e, c in enumerate(c_f)])
            r = {
                "tracker": tname, "setting": sname,
                "touchdown_on_platform_rate": float(on_pf.mean()),
                "touchdown_time_median": float(np.median(c_f[on_pf]) * lc.dt) if on_pf.any() else None,
                "touchdown_radial_median": float(np.nanmedian(rad_f[on_pf])) if on_pf.any() else None,
                "touchdown_radial_max": float(np.nanmax(rad_f[on_pf])) if on_pf.any() else None,
                "touchdown_speed_max": float(max(np.linalg.norm(res_f["traj"][e, c, 3:6]) for e, c in enumerate(c_f) if c >= 0))
                if (c_f >= 0).any() else None,
                "min_h_cone": float(h_raw[fly_x].min()),
                "min_height": float(rel[..., 2][fly_x].min()),
                "max_slack": float(res_f["slack"][fly_u].max()),
                "qp_fallbacks": int((~res_f["qp_used"].astype(bool))[fly_u].sum()),
                "safeguard_active_rate": float((res_f["safeguard_lambda"][fly_u] < 1.0).mean()),
                "intervention_mean": float(du[fly_u].mean()),
                "intervention_peak": float(du[fly_u].max()),
                # mean step-to-step change of the applied action (scaled by half-range): chattering
                "action_roughness": float(np.linalg.norm(np.diff(res_f["u"], axis=1) / half, axis=-1)[fly_u[:, 1:]].mean()),
                "final_radial_max": float(np.linalg.norm(rel[:, -1, 0:2], axis=-1).max()),
            }
            r["safe"] = r["min_h_cone"] >= -1e-6 and r["min_height"] >= -1e-6 and r["max_slack"] <= 1e-2 and r["qp_fallbacks"] == 0
            key = f"{tname}|{sname}"
            pairs[key] = r
            rec_f = {"traj": res_f["traj"], "u": res_f["u"], "u_raw": res_f["u_ref"], "du_norm": du, "slack": res_f["slack"]}
            if not args.no_land_latch:
                rec_f = latch(rec_f, c_f)
            recs = {"unfiltered": rec_u, "filtered": rec_f}
            store[key] = {"recs": recs, "c_u": c_u, "c_f": c_f, "on_pf": on_pf, "on_pf_u": on_pf_u,
                          "h": {v: np.asarray(L.cone_value(jnp.asarray(recs[v]["traj"]), cfg)) for v in recs}}
            print(f"  [{tname} + {sname}] {time.time() - t0:.0f}s | on the platform {r['touchdown_on_platform_rate']:.2f} "
                  f"(median {_f(r['touchdown_time_median'], '.2f')} s, radial <= {_f(r['touchdown_radial_max'])} m) | min h_cone "
                  f"{r['min_h_cone']:+.4f}, min height {r['min_height']:+.4f}, slack {r['max_slack']:.1e}, fallbacks "
                  f"{r['qp_fallbacks']}, safeguard {100 * r['safeguard_active_rate']:.1f}%, intervention {r['intervention_mean']:.3f} "
                  f"(peak {r['intervention_peak']:.2f}), roughness {r['action_roughness']:.4f}"
                  f"{'' if r['safe'] else '  UNSAFE'}", flush=True)

    good = [k for k, r in pairs.items() if r["safe"] and r["touchdown_on_platform_rate"] == 1.0
            and trk[r["tracker"]]["leaves_cone_rate"] > 0]
    pool = good or [k for k, r in pairs.items() if r["safe"]] or list(pairs)
    for r in pairs.values():
        r["rim_margin_min"] = None if r["touchdown_radial_max"] is None else float(lc.cone_r0 - r["touchdown_radial_max"])

    def rank(k):
        r = pairs[k]
        crit = {"rim_margin": -(r["rim_margin_min"] if r["rim_margin_min"] is not None else -np.inf),
                "intervention": r["intervention_mean"],
                "time": r["touchdown_time_median"] if r["touchdown_time_median"] is not None else np.inf}[args.select_by]
        return (-r["touchdown_on_platform_rate"], crit, r["intervention_mean"], trk[r["tracker"]]["tracking_error_mean"])

    best = min(pool, key=rank)
    print(f"[select] {best} (by {args.select_by})" + ("" if good else " (no pair met every criterion; best effort)"), flush=True)

    # overview: one row per setting, side + top view of the nominal start
    scene = lp.Scene.build(lc, ref_x, ref_t, lc.dt)
    figs = {}
    fig, axs = plt.subplots(len(settings), 2, figsize=(15, 4.6 * len(settings)), squeeze=False)
    zmax = float(ref_x[:, 2].max() - pad[2])
    zz = np.linspace(0, zmax + 0.3, 200)
    rr = scene.radius(zz)
    th = np.linspace(0, 2 * np.pi, 200)
    cols = plt.cm.viridis(np.linspace(0, 0.85, len(trackers)))
    s_ref, z_ref = scene.side(ref_x)
    for row, (sname, sover) in enumerate(settings):
        a0, a1 = axs[row]
        a0.fill_betweenx(zz, -rr, rr, color="tab:green", alpha=0.1)
        a0.plot(rr, zz, color="tab:green")
        a0.plot(-rr, zz, color="tab:green")
        a0.plot([-lc.cone_r0, lc.cone_r0], [0, 0], color="tab:green", lw=5, solid_capstyle="butt", label="platform (cone footprint)")
        a0.plot(s_ref, z_ref, "--", color="k", lw=1.3, label="reference")
        a1.plot(lc.pad_x + lc.cone_r0 * np.cos(th), lc.pad_y + lc.cone_r0 * np.sin(th), color="tab:green", lw=2, label="platform")
        a1.plot(ref_x[:, 0], ref_x[:, 1], "--", color="k", lw=1.3, label="reference")
        for c, (tname, _) in zip(cols, trackers):
            st = store[f"{tname}|{sname}"]
            for v, ls in (("unfiltered", ":"), ("filtered", "-")):
                tr = st["recs"][v]["traj"][0]
                s, z = scene.side(tr)
                lab = f"{tname}: {LABELS[v]}" + (f" ({100 * pairs[f'{tname}|{sname}']['touchdown_on_platform_rate']:.0f}% on platform)"
                                                  if v == "filtered" else "")
                a0.plot(s, z, ls, color=c, lw=1.8 if v == "filtered" else 1.1, label=lab)
                a1.plot(tr[:, 0], tr[:, 1], ls, color=c, lw=1.8 if v == "filtered" else 1.1)
                cc = st["c_f" if v == "filtered" else "c_u"][0]
                if cc >= 0:
                    a0.plot(s[cc], z[cc], "v", color=c, ms=8, mec="k")
                    a1.plot(tr[cc, 0], tr[cc, 1], "v", color=c, ms=8, mec="k")
        a0.set_xlim(-0.8, 0.9)
        a0.set_ylim(-0.05, 0.6)
        a0.set_aspect("equal")
        a0.set_xlabel("along the skid direction [m]")
        a0.set_ylabel("height above pad [m]")
        a0.set_title(f"filter setting '{sname}' ({', '.join(f'{k}={v}' for k, v in sover.items())}) — side view near the pad", fontsize=9)
        a0.legend(fontsize=6.5, loc="upper left")
        a1.set_xlim(-0.6, 0.7)
        a1.set_ylim(-0.5, 0.6)
        a1.set_aspect("equal")
        a1.set_xlabel("x [m]")
        a1.set_ylabel("y [m]")
        a1.set_title("top view near the pad (▼ touchdown; dotted: tracker alone, solid: with filter)", fontsize=9)
        a1.legend(fontsize=7, loc="upper left")
        for a in (a0, a1):
            a.grid(alpha=0.3)
    fig.tight_layout()
    figs["trackers"] = str(out / "trackers.png")
    fig.savefig(figs["trackers"], dpi=120)
    plt.close(fig)

    zoom = None
    if args.gif_zoom_margin > 0:
        mg = float(args.gif_zoom_margin)
        s_lo, s_hi = -(lc.cone_r0 + mg), max(float(np.hypot(*td_ref[:2])), lc.cone_r0) + mg
        zoom = {"s": (s_lo, s_hi), "z": (-0.03, 0.45 * (s_hi - s_lo))}
    extra = [k.strip() for k in args.gif_pairs.split(",") if k.strip()]
    unknown = [k for k in extra if k not in pairs]
    if unknown:
        print(f"[warn] --gif_pairs: unknown pairs {unknown}; known: {list(pairs)}")
    for key in (list(pairs) if args.gif_all else [best] + [k for k in extra if k in pairs and k != best]):
        st = store[key]
        tname, sname = key.split("|")
        tag = f"{tname}_{sname}"
        title = f"skid-out reference | tracker {tname} | filter '{sname}' | backup {Path(args.backup).name}"
        figs[f"case_{tag}"] = lp.plot_case(scene, st["recs"], st["h"], 0, out / f"case_{tag}.png", title, labels=LABELS)
        figs[f"actuators_{tag}"] = lp.plot_actuators(scene, st["recs"], 0, cfg0.action_low, cfg0.action_high,
                                                     out / f"actuators_{tag}.png", title, gravity=lc.gravity, labels=LABELS)
        cu, cf = int(st["c_u"][0]), int(st["c_f"][0])
        td = {}
        if cu >= 0:
            td["unfiltered"] = (cu, "LANDED on the platform" if st["on_pf_u"][0] else "LANDED beside the platform")
        if cf >= 0:
            td["filtered"] = (cf, "LANDED on the platform" if st["on_pf"][0] else "LANDED beside the platform")
        recs, h = st["recs"], st["h"]
        if not args.no_land_latch and len(td) == 2:
            n_end = min(steps + 1, max(cu, cf) + 1 + int(round(args.gif_tail / lc.dt)))
            recs = {v: {k: (a[:, :n_end] if k == "traj" else a[:, : n_end - 1]) for k, a in recs[v].items()} for v in recs}
            h = {v: a[:, :n_end] for v, a in h.items()}
        t0 = time.time()
        figs[f"gif_{tag}"] = lp.save_gif(scene, recs, h, 0, out / f"skid_landing_{tag}.gif", title,
                                         slowdown=args.gif_slowdown, stride=args.gif_stride, labels=LABELS, touchdown=td,
                                         zoom=zoom, xlabel="along the skid direction [m]")
        print(f"[gif] {figs[f'gif_{tag}']} ({time.time() - t0:.0f} s)", flush=True)

    summary = {"backup": str(args.backup), "settings": dict(settings), "reference": str(args.reference),
               "reference_touchdown_offset": float(np.hypot(*td_ref[:2])), "platform_radius": lc.cone_r0,
               "starts": int(len(x0)), "steps": steps, "dt": lc.dt, "zeta_touch": args.zeta_touch, "v_touch": args.v_touch,
               "land_latch": not args.no_land_latch, "trackers": trk, "pairs": pairs, "selected": best, "select_by": args.select_by,
               "figures": figs, "wall_time_sec": time.time() - t_start}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=_jd))

    f = _f
    md = ["# Skid-out landing demo", "",
          f"Reference: starts inside the cone (h = {h_ref[0]:+.2f} m), leaves it (min h {h_ref.min():+.2f} m) and touches down "
          f"{np.hypot(*td_ref[:2]):.2f} m from the pad centre (platform radius {lc.cone_r0} m). Backup `{Path(args.backup).name}`; "
          f"{len(x0)} starts inside the cone and in C_N; touchdown = height ≤ {args.zeta_touch} m at speed ≤ {args.v_touch} m/s.", "",
          "## Trackers alone", "",
          "| tracker | leaves cone | touches down | at [m from centre] | on platform | tracking error [m] |", "|---|---|---|---|---|---|"]
    for n, r in trk.items():
        md.append(f"| {n} | {100 * r['leaves_cone_rate']:.0f} % | {100 * r['touchdown_rate']:.0f} % | "
                  f"{f(r['touchdown_radial_median'], '.2f')} | {100 * r['touchdown_on_platform_rate']:.0f} % | {r['tracking_error_mean']:.3f} |")
    md += ["", "## Tracker + filter", "",
           "| tracker | filter | on platform | median time [s] | radial ≤ [m] | inside rim ≥ [m] | min h_cone | min height | max slack | fallbacks | safeguard | intervention (peak) | roughness |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, r in pairs.items():
        md.append(f"| {r['tracker']}{' **(selected)**' if k == best else ''} | {r['setting']} | {100 * r['touchdown_on_platform_rate']:.0f} % | "
                  f"{f(r['touchdown_time_median'], '.2f')} | {f(r['touchdown_radial_max'], '.3f')} | {f(r['rim_margin_min'], '.3f')} | {r['min_h_cone']:+.4f} | "
                  f"{r['min_height']:+.4f} | {r['max_slack']:.1e} | {r['qp_fallbacks']} | {100 * r['safeguard_active_rate']:.1f} % | "
                  f"{r['intervention_mean']:.3f} ({r['intervention_peak']:.2f}) | {r['action_roughness']:.4f} |")
    md += ["", "Filter settings: " + "; ".join(f"`{n}` = {', '.join(f'{k}={v}' for k, v in o.items())}" for n, o in settings),
           "", f"Selected among pairs that land from every start by `{args.select_by}`; intervention = mean ‖u − u_ref‖ "
               "(each action scaled by half its range) over the flight; roughness = mean ‖u_{k+1} − u_k‖ of the applied action, same scaling."]
    (out / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"[done] {out} ({(time.time() - t_start) / 60:.1f} min)")


if __name__ == "__main__":
    main()
