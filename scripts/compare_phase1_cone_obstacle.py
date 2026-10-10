#!/usr/bin/env python
"""Compare the Phase-I backups for the cone-obstacle task and pick the one Phase II uses.

    python scripts/compare_phase1_cone_obstacle.py \
        --run sa_shrunk=outputs/cone_obstacle_phase1/sa_shrunk_seed0 \
        --run sa_plain=outputs/cone_obstacle_phase1/sa_plain_seed0

Candidates: the base set's LQR alone (always included, name ``lqr``) and every ``--run``
(Phase-I safe-arrival actors, blind to d_hat as in Phase II, LQR inside B). Each backup is
scored on the same states and the same d_hat draws:

* design-region held-out test sets (general / edge / shell, the Phase-I objective), and
* a reference tube around the Phase-II reference (``task['reference']``): reference states with
  position / velocity noise; samples that fall inside the funnel are moved just below the funnel
  wall with their velocity kept (where the filtered vehicle actually is when the reference
  runs into the funnel).

Two criteria: ``tight`` = the backup reaches the tube-tightened base set within N steps without
leaving the tightened safe set, under a frozen d_hat with |d_hat| up to delta_d + e_bar (exactly
what the Phase-II UE-bCBF filter checks; the size of its invariant set C_N), and ``nominal``
(d_hat = 0, no margins). The decision uses ``--metric`` (default: mean of the design-region
weighted tight rate and the reference-tube tight rate):

1. best safe-arrival backup - LQR < ``--lqr_margin``      -> LQR alone ("good enough");
2. otherwise a safe-arrival backup: the plain-set one, unless the shrunk-set one is better by at
   least ``--shrunk_margin`` ("only if there is a big difference").

Outputs (``--out``, default outputs/cone_obstacle_phase1/compare): compare.json,
chosen_backup.json (``--backup_from`` of scripts/train_phase2_cone_obstacle_ue.py), compare.png
(bars + recoverable-set slices of the filter's C_N).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from cone_obstacle_common import load_task, task_config  # noqa: E402
from ps2rl.cil import cone_obstacle_ue_bcbf as cu  # noqa: E402
from ps2rl.phase1_sa.cone_obstacle_design_region import REGION_NAMES, ConeObstacleDesignRegionConfig  # noqa: E402
from ps2rl.phase1_sa.cone_obstacle_ue_sa_env import build_cone_obstacle_ue_sa_env  # noqa: E402
from ps2rl.phase1_sa.cone_obstacle_ue_sa_trainer import eval_d_hats, make_blind_backup, make_lqr_backup, make_rollout  # noqa: E402
from ps2rl.phase1_sa.landing_design_region import heldout_sets  # noqa: E402
from ps2rl.uncertainty.cone_obstacle_ue_tube import UEConfig  # noqa: E402
from ps2rl.utils.policy import ActorConfig  # noqa: E402
from ps2rl.utils.quaternion import normalize_quaternion, quaternion_from_euler_zyx, quaternion_multiply  # noqa: E402


def reference_tube(ref_path: str, cfg, n: int, seed: int, p_noise: float, v_noise: float, tilt_deg: float,
                   wall_band: float) -> np.ndarray:
    """States around the reference; those inside the funnel are moved just below its wall."""
    p = Path(ref_path)
    ref = np.load(p if p.exists() else ROOT / p)
    rs = np.asarray(ref["states"], np.float64)
    rng = np.random.default_rng(seed)
    k = rng.integers(0, rs.shape[0], n)
    x = rs[k].copy()
    x[:, 0:3] += rng.uniform(-p_noise, p_noise, (n, 3))
    x[:, 3:6] += rng.uniform(-v_noise, v_noise, (n, 3))
    tilt = np.deg2rad(tilt_deg)
    rpy = rng.uniform(-1, 1, (n, 3)) * np.array([tilt, tilt, tilt])
    q = jax.vmap(lambda a, q0: normalize_quaternion(quaternion_multiply(quaternion_from_euler_zyx(a[0], a[1], a[2]), q0)))(
        jnp.asarray(rpy, jnp.float32), jnp.asarray(x[:, 6:10], jnp.float32))
    x[:, 6:10] = np.asarray(q)
    r = np.hypot(x[:, 0] - cfg.axis_x, x[:, 1] - cfg.axis_y)
    zs = cfg.tip_z + (np.sqrt(r ** 2 + cfg.cone_eps ** 2) - cfg.cone_eps) / cfg.tan_theta
    inside = x[:, 2] > zs
    x[inside, 2] = zs[inside] - rng.uniform(0.0, wall_band, int(inside.sum()))
    x[:, 2] = np.maximum(x[:, 2], cfg.floor_z + 0.05)
    return x.astype(np.float32)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", default="docker_batch/configs/cone_obstacle_ue.json")
    p.add_argument("--run", action="append", default=[], help="name=PHASE1_RUN_DIR (name containing 'shrunk' = trained "
                                                               "on the shrunk sets; or read from its metadata)")
    p.add_argument("--weights", default="best", choices=("best", "final"))
    p.add_argument("--metric", default="mean", choices=("mean", "design", "reference"))
    p.add_argument("--lqr_margin", type=float, default=0.02, help="SA must beat LQR by this much (absolute rate)")
    p.add_argument("--shrunk_margin", type=float, default=0.05, help="shrunk SA must beat plain SA by this much")
    p.add_argument("--n_reference", type=int, default=4096)
    p.add_argument("--ref_p_noise", type=float, default=0.3)
    p.add_argument("--ref_v_noise", type=float, default=0.5)
    p.add_argument("--ref_tilt_deg", type=float, default=10.0)
    p.add_argument("--ref_wall_band", type=float, default=0.3)
    p.add_argument("--map_speed", type=float, default=2.0, help="slice 2: horizontal speed toward the funnel axis [m/s]")
    p.add_argument("--map_res", type=int, default=61)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out", default="outputs/cone_obstacle_phase1/compare")
    p.add_argument("--no_plot", action="store_true")
    a = p.parse_args(argv)

    task = load_task(a.task)
    cfg = task_config(task)
    region_cfg = ConeObstacleDesignRegionConfig.from_dict(task.get("design_region", {}))
    ue = UEConfig.from_dict(task.get("ue", {}))
    env_fns = build_cone_obstacle_ue_sa_env(cfg, region_cfg, ue, tighten=True)
    n_steps, dt = int(cfg.num_steps), float(cfg.dt)

    # ---------------------------------------------------------------- candidates
    cands = {"lqr": {"backup_fn": make_lqr_backup(cfg, env_fns), "params": None, "path": "", "tighten": None}}
    for spec in a.run:
        name, run = spec.split("=", 1)
        run = Path(run)
        actor = pickle.load(open(run / cu.ACTOR_FILE, "rb"))
        md = actor["metadata"]
        trained = type(cfg).from_dict(md["cone_obstacle_config"])
        for k in ("tip_z", "cone_theta_deg", "cone_eps", "min_altitude", "base_set_c", "a_cmd_max", "omega_max"):
            if not np.isclose(float(getattr(trained, k)), float(getattr(cfg, k))):
                raise SystemExit(f"{run}: trained with {k}={getattr(trained, k)}, task has {getattr(cfg, k)}")
        ad = dict(actor["actor_cfg"])
        ad["hidden_sizes"] = tuple(ad["hidden_sizes"])
        acfg = ActorConfig(**ad)
        params = pickle.load(open(run / f"{a.weights}_weights.pkl", "rb"))["actor_params"] if a.weights == "final" \
            else actor["actor_params"]
        cands[name] = {"backup_fn": make_blind_backup(cfg, env_fns, acfg), "params": params, "path": str(run),
                       "tighten": bool(md.get("tighten", "shrunk" in name))}

    # ---------------------------------------------------------------- state sets (shared by every candidate)
    sets = heldout_sets(region_cfg, env_fns.sampler, split="test")
    xs = {nm: np.asarray(sets[nm]) for nm in REGION_NAMES}
    xs["reference"] = reference_tube(task["reference"], cfg, a.n_reference, a.seed, a.ref_p_noise, a.ref_v_noise,
                                     a.ref_tilt_deg, a.ref_wall_band)
    dhs = {nm: eval_d_hats(ue, xs[nm].shape[0], 9000 + i) for i, nm in enumerate(xs)}
    w = region_cfg.weights
    print(f"[sets] " + "  ".join(f"{k} {v.shape[0]}" for k, v in xs.items()) + f" | d_hat up to {ue.d_hat_max:.3f} m/s^2", flush=True)

    results = {}
    for name, c in cands.items():
        rb = jax.jit(jax.vmap(make_rollout(c["backup_fn"], env_fns, ue, n_steps, dt), in_axes=(None, 0, 0, None)))
        res = {}
        for nm, x0 in xs.items():
            hb_t, hf_t, th, _, _ = map(np.asarray, rb(c["params"], jnp.asarray(x0), jnp.asarray(dhs[nm]), True))
            hb_n, hf_n, _, _, _ = map(np.asarray, rb(c["params"], jnp.asarray(x0), jnp.zeros_like(jnp.asarray(dhs[nm])), False))
            res[nm] = {"n": int(x0.shape[0]), "tight": float(hb_t.mean()), "nominal": float(hb_n.mean()),
                       "tight_crash": float(hf_t.mean()), "nominal_crash": float(hf_n.mean()),
                       "mean_arrival_s": float(th[hb_t].mean() * dt) if hb_t.any() else float("nan")}
        den = sum(w[nm] for nm in REGION_NAMES)
        res["design_tight"] = sum(w[nm] * res[nm]["tight"] for nm in REGION_NAMES) / den
        res["design_nominal"] = sum(w[nm] * res[nm]["nominal"] for nm in REGION_NAMES) / den
        res["reference_tight"] = res["reference"]["tight"]
        res["reference_nominal"] = res["reference"]["nominal"]
        res["score"] = {"mean": 0.5 * (res["design_tight"] + res["reference_tight"]), "design": res["design_tight"],
                        "reference": res["reference_tight"]}[a.metric]
        results[name] = res
        print(f"[{name:10s}] tight: design {res['design_tight']:.3f} (general {res['general']['tight']:.3f} edge "
              f"{res['edge']['tight']:.3f} shell {res['shell']['tight']:.3f}) reference {res['reference_tight']:.3f} | nominal: "
              f"design {res['design_nominal']:.3f} reference {res['reference_nominal']:.3f} | score ({a.metric}) {res['score']:.3f}",
              flush=True)

    # ---------------------------------------------------------------- decision (the user's rule)
    sa = {k: v for k, v in results.items() if k != "lqr"}
    lqr = results["lqr"]["score"]
    if not sa:
        choice, why = "lqr", "no safe-arrival run given"
    else:
        best_sa = max(sa, key=lambda k: sa[k]["score"])
        gain = sa[best_sa]["score"] - lqr
        if gain < a.lqr_margin:
            choice, why = "lqr", f"best safe-arrival backup ({best_sa}) beats LQR by only {gain:+.3f} < {a.lqr_margin}"
        else:
            shrunk = [k for k in sa if cands[k]["tighten"]]
            plain = [k for k in sa if not cands[k]["tighten"]]
            if shrunk and plain:
                ks = max(shrunk, key=lambda k: sa[k]["score"])
                kp = max(plain, key=lambda k: sa[k]["score"])
                d = sa[ks]["score"] - sa[kp]["score"]
                if d >= a.shrunk_margin:
                    choice, why = ks, f"shrunk beats plain by {d:+.3f} >= {a.shrunk_margin} (and LQR by {sa[ks]['score'] - lqr:+.3f})"
                else:
                    choice, why = kp, (f"safe arrival beats LQR by {gain:+.3f} >= {a.lqr_margin}; shrunk vs plain {d:+.3f} "
                                       f"< {a.shrunk_margin}, so the plain-set one")
            else:
                choice, why = best_sa, f"safe arrival beats LQR by {gain:+.3f} >= {a.lqr_margin}"
    chosen = {"name": choice, "backup": "lqr" if choice == "lqr" else "learned", "backup_path": cands[choice]["path"],
              "reason": why, "metric": a.metric, "lqr_margin": a.lqr_margin, "shrunk_margin": a.shrunk_margin}
    print(f"[choice] {choice}: {why}", flush=True)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "compare.json").write_text(json.dumps({"task": a.task, "results": results, "choice": chosen,
                                                  "candidates": {k: {"path": v["path"], "tighten": v["tighten"]} for k, v in cands.items()},
                                                  "ue": ue.as_dict()}, indent=2, default=float))
    (out / "chosen_backup.json").write_text(json.dumps(chosen, indent=2))
    if not a.no_plot:
        plot(task, cands, results, chosen, a, out / "compare.png")
    print(f"[compare] wrote {out}/compare.json, chosen_backup.json" + ("" if a.no_plot else ", compare.png"))


def plot(task, cands, results, chosen, a, path):
    """Bars (tight / nominal per set) + slices of the filter's C_N (tightened, d_hat = 0) per backup."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfg = task_config(task)
    names = list(cands)
    cols = {"lqr": "#8a8986"}
    pal = ["#2a78d6", "#eb6834", "#4d9f61", "#8a5cc2"]
    for i, n in enumerate(k for k in names if k != "lqr"):
        cols[n] = pal[i % len(pal)]
    xs = np.linspace(cfg.axis_x - 3.6, cfg.axis_x + 1.0, a.map_res)
    zs = np.linspace(cfg.floor_z, cfg.tip_z + 2.0, int(a.map_res * 0.6))
    X, Z = np.meshgrid(xs, zs)
    slices = [("hover (v = 0)", np.zeros(3)), (f"moving toward the axis at {a.map_speed:g} m/s", np.array([a.map_speed, 0.0, 0.0]))]
    fig = plt.figure(figsize=(4.4 * len(names), 9.6))
    gs = fig.add_gridspec(3, len(names), height_ratios=[1.0, 0.9, 0.9])
    ax = fig.add_subplot(gs[0, :])
    sets = ["general", "edge", "shell", "reference"]
    width = 0.8 / len(names)
    for i, n in enumerate(names):
        vals = [results[n][s]["tight"] for s in sets]
        nom = [results[n][s]["nominal"] for s in sets]
        pos = np.arange(len(sets)) + (i - (len(names) - 1) / 2) * width
        ax.bar(pos, vals, width * 0.92, color=cols[n], label=f"{n} (tight)")
        ax.scatter(pos, nom, color="k", marker="_", s=120, zorder=4, label="nominal criterion" if i == 0 else None)
        for x_, v in zip(pos, vals):
            ax.text(x_, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(np.arange(len(sets)))
    ax.set_xticklabels(["design: general", "design: edge (wall/floor, inbound)", "design: shell", "reference tube (Phase II)"])
    ax.set_ylabel("recoverable fraction")
    ax.set_ylim(0, 1.08)
    ax.legend(ncol=len(names) + 1, fontsize=8, loc="upper left")
    ax.set_title(f"Phase-I backups: recoverable within T = {cfg.T:g} s (bars: UE-tightened sets, |d_hat| <= delta_d + e_bar; "
                 f"ticks: nominal sets, d_hat = 0)\nChoice: {chosen['name']} - {chosen['reason']}", fontsize=9.5, loc="left")
    for i, n in enumerate(names):
        c = cands[n]
        ccfg = cu.cone_ue_config_from_task(task, backup="lqr" if n == "lqr" else "learned", backup_path=c["path"])
        rt = cu.make_ue_runtime(ccfg)
        rec = cu.make_recoverability_fn_ue(ccfg, rt)
        for j, (title, v) in enumerate(slices):
            x = np.zeros((X.size, 10), np.float32)
            x[:, 0], x[:, 2], x[:, 1] = X.ravel(), Z.ravel(), cfg.axis_y
            x[:, 3:6] = v
            x[:, 6] = 1.0
            ok = np.asarray(rec(jnp.asarray(x), jnp.zeros((X.size, 3), jnp.float32))).reshape(X.shape)
            axm = fig.add_subplot(gs[1 + j, i])
            axm.contourf(X, Z, ok.astype(float), levels=[-0.5, 0.5, 1.5], colors=["#fcfcfb", cols[n]], alpha=0.55)
            rr = np.abs(xs - cfg.axis_x)
            axm.plot(xs, cfg.tip_z + (np.sqrt(rr ** 2 + cfg.cone_eps ** 2) - cfg.cone_eps) / cfg.tan_theta, color="#c4553f", lw=1.2)
            axm.axhline(cfg.floor_z, color="#c4553f", lw=1.0, ls="--")
            hp = cfg.hover_points[0]
            axm.plot(hp[0], hp[2], "k+", ms=8)
            axm.fill_between([cfg.axis_x - cfg.target_radius, cfg.axis_x + cfg.target_radius], cfg.floor_z, cfg.tip_z,
                             color="#3d3c39", alpha=0.8)
            axm.set_xlim(xs[0], xs[-1])
            axm.set_ylim(zs[0], zs[-1])
            axm.set_aspect("equal")
            axm.set_title(f"{n}: C_N, {title}  ({ok.mean():.0%} of slice)", fontsize=8.5, loc="left")
            axm.set_xlabel("x [m]", fontsize=8)
            if i == 0:
                axm.set_ylabel("z [m]", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
