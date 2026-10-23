#!/usr/bin/env python
"""Compare Phase-I strike backups with the hover-LQR backup on the held-out test scenes.

    python scripts/compare_phase1_strike.py --runs outputs/strike_phase1/td3_seed0 outputs/strike_phase1/td3_seed1 \
        --out outputs/strike_phase1/compare

Every run is evaluated on the test scenes of its own region config (same seeds as in training, so runs with the
same config share the scenes). Writes to --out:
  strike_phase1_table.md / .json   recovered fraction (C_N membership) per region: LQR and each run (+ mean over runs)
  strike_phase1_outcomes.png       outcome per region (recovered / contact / no base set within T), LQR vs learned
  strike_phase1_examples.png       scenes where the LQR fails and the learned backup recovers (and one where both fail)
  strike_phase1_curves.png         validation recovered fraction during training, LQR as dashed lines
  chosen_backup.json               the run with the best weighted recovered fraction (used for the two figures above)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from ps2rl.envs.strike_scene import StrikeConfig, anchor, build_model, split_scene  # noqa: E402
from ps2rl.phase1_sa.strike_sa import (  # noqa: E402
    REGION_NAMES, StrikeRegionConfig, heldout_sets, make_actor_backup, make_flee_backup, make_lqr_backup, make_rollout)
from ps2rl.utils.policy import _coerce_actor_cfg  # noqa: E402

INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
LQR_C, RL_C, CHASER, FAIL = "#c4553f", "#2a78d6", "#1baf7a", "#3d3c39"
OBS_FILL, OBS_EDGE = "#c9c7c0", "#6b6a65"


def load_run(run: Path):
    cfgs = json.loads((run / "configs.json").read_text())
    cfg, rc = StrikeConfig.from_dict(cfgs["strike"]), StrikeRegionConfig.from_dict(cfgs["region"])
    with open(run / "strike_backup_policy_actor.pkl", "rb") as f:
        ck = pickle.load(f)
    hist = json.loads((run / "history.json").read_text()) if (run / "history.json").exists() else {}
    return cfg, rc, ck["actor_params"], _coerce_actor_cfg(ck["actor_cfg"]), hist


def evaluate(m, backup, params, sets):
    roll = jax.jit(jax.vmap(make_rollout(m, backup, trajectory=True), in_axes=(None, 0, 0)))
    out = {}
    for name in REGION_NAMES:
        v = sets[name]
        ok, hf, k_end, xs = map(np.asarray, roll(params, jnp.asarray(v[:, :10]), jnp.asarray(v[:, 10:])))
        out[name] = {"ok": ok, "fail": hf, "k_end": k_end, "xs": xs}
    return out


def outline(o, n=90):
    cx, cy, bx, by, yaw, rr, _ = o
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    if bx == 0 and by == 0:
        return np.stack([cx + rr * np.cos(th), cy + rr * np.sin(th)], 1)
    c, s = np.cos(yaw), np.sin(yaw)
    pts = np.array([[-bx, -by], [bx, -by], [bx, by], [-bx, by]])
    return pts @ np.array([[c, s], [-s, c]]) + [cx, cy]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--out", default="")
    p.add_argument("--split", choices=("val", "test"), default="test")
    a = p.parse_args(argv)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Circle, Polygon

    runs = [Path(r) for r in a.runs]
    out = Path(a.out) if a.out else runs[0].parent / "compare"
    out.mkdir(parents=True, exist_ok=True)

    loaded = [load_run(r) for r in runs]
    cfg, rc = loaded[0][0], loaded[0][1]
    m = build_model(cfg)
    sets_cache = {}
    res_lqr, res_flee, res_rl, table = {}, {}, [], {"split": a.split, "regions": list(REGION_NAMES), "runs": {}}
    for run, (cfg_r, rc_r, params, actor_cfg, _) in zip(runs, loaded):
        m_r = build_model(cfg_r)
        key = (cfg_r, rc_r)
        if key not in sets_cache:
            sets_cache[key] = heldout_sets(m_r, rc_r, a.split)
            res_lqr[key] = evaluate(m_r, make_lqr_backup(m_r), None, sets_cache[key])
            if cfg_r.moving_chasers:          # moving chasers: the flee-then-hover heuristic as a second baseline
                res_flee[key] = evaluate(m_r, make_flee_backup(m_r), None, sets_cache[key])
        r = evaluate(m_r, make_actor_backup(m_r, actor_cfg), params, sets_cache[key])
        res_rl.append((run, key, r))
        w = rc_r.weights
        table["runs"][run.name] = {
            "learned": {n: float(r[n]["ok"].mean()) for n in REGION_NAMES},
            "lqr": {n: float(res_lqr[key][n]["ok"].mean()) for n in REGION_NAMES},
            "learned_crash": {n: float(r[n]["fail"].mean()) for n in REGION_NAMES},
            "lqr_crash": {n: float(res_lqr[key][n]["fail"].mean()) for n in REGION_NAMES},
            "n_per_region": {n: int(sets_cache[key][n].shape[0]) for n in REGION_NAMES},
        }
        t = table["runs"][run.name]
        t["learned_mu_weighted"] = float(sum(w[n] * t["learned"][n] for n in REGION_NAMES))
        t["lqr_mu_weighted"] = float(sum(w[n] * t["lqr"][n] for n in REGION_NAMES))
        if key in res_flee:
            t["flee"] = {n: float(res_flee[key][n]["ok"].mean()) for n in REGION_NAMES}
            t["flee_mu_weighted"] = float(sum(w[n] * t["flee"][n] for n in REGION_NAMES))

    # ---------------------------------------------------------------- table
    names = list(table["runs"])
    learned = np.array([[table["runs"][r]["learned"][n] for n in REGION_NAMES] for r in names])
    lqr0 = table["runs"][names[0]]["lqr"]
    n_reg = table["runs"][names[0]]["n_per_region"]
    hw = {n: float(np.sqrt(np.log(2 / 0.05) / (2 * n_reg[n]))) for n in REGION_NAMES}
    lines = [f"Recovered fraction (base set reached within T = {cfg.T} s, no contact), {a.split} scenes "
             f"({', '.join(f'{n} {n_reg[n]}' for n in REGION_NAMES)}); 95 % Hoeffding half-width per region "
             f"{', '.join(f'{hw[n]:.3f}' for n in REGION_NAMES)}.", "",
             "| backup | " + " | ".join(REGION_NAMES) + " | weighted |", "|---|" + "---|" * (len(REGION_NAMES) + 1),
             "| hover LQR | " + " | ".join(f"{lqr0[n]:.3f}" for n in REGION_NAMES)
             + f" | {table['runs'][names[0]]['lqr_mu_weighted']:.3f} |"]
    if "flee" in table["runs"][names[0]]:
        f0 = table["runs"][names[0]]
        lines.append("| flee-then-hover heuristic | " + " | ".join(f"{f0['flee'][n]:.3f}" for n in REGION_NAMES)
                     + f" | {f0['flee_mu_weighted']:.3f} |")
    for r in names:
        t = table["runs"][r]
        lines.append(f"| {r} | " + " | ".join(f"{t['learned'][n]:.3f}" for n in REGION_NAMES) + f" | {t['learned_mu_weighted']:.3f} |")
    sa_cfgs = [{k: v for k, v in json.loads((r / "configs.json").read_text())["sa"].items() if k != "seed"} for r in runs]
    same_recipe = all(c == sa_cfgs[0] for c in sa_cfgs) and all((l[0], l[1]) == (cfg, rc) for l in loaded)
    if len(names) > 1 and same_recipe:     # seeds of one recipe: mean over seeds
        mu = [table["runs"][r]["learned_mu_weighted"] for r in names]
        lines.append(f"| learned, mean ± std (n = {len(names)}) | "
                     + " | ".join(f"{learned[:, j].mean():.3f} ± {learned[:, j].std():.3f}" for j in range(len(REGION_NAMES)))
                     + f" | {np.mean(mu):.3f} ± {np.std(mu):.3f} |")
    (out / "strike_phase1_table.md").write_text("\n".join(lines) + "\n")
    (out / "strike_phase1_table.json").write_text(json.dumps(table, indent=2))
    print("\n".join(lines))

    plt.rcParams.update({"font.size": 9.5, "axes.titlesize": 10, "axes.titleweight": "bold"})
    # figures and the choice: the run with the best weighted recovered fraction on these scenes
    best = int(np.argmax([table["runs"][r.name]["learned_mu_weighted"] for r, _, _ in res_rl]))
    run0, key0, r0 = res_rl[best]
    choice = {"chosen_run": run0.name, "actor": str(run0 / "strike_backup_policy_actor.pkl"),
              "rule": f"highest weighted recovered fraction on the {a.split} scenes",
              "learned": table["runs"][run0.name]["learned"], "lqr": table["runs"][run0.name]["lqr"],
              "learned_mu_weighted": table["runs"][run0.name]["learned_mu_weighted"],
              "lqr_mu_weighted": table["runs"][run0.name]["lqr_mu_weighted"]}
    (out / "chosen_backup.json").write_text(json.dumps(choice, indent=2))
    print(f"chosen backup: {run0.name} (weighted {choice['learned_mu_weighted']:.3f} vs LQR {choice['lqr_mu_weighted']:.3f})")
    lq0 = res_lqr[key0]

    # ---------------------------------------------------------------- outcome bars
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    fig.patch.set_facecolor(SURF)
    ax.set_facecolor(SURF)
    y, labels = [], []
    for i, n in enumerate(REGION_NAMES):
        for j, (lab, rr_) in enumerate((("hover LQR", lq0[n]), (f"learned ({run0.name})", r0[n]))):
            yy = i * 2.6 + j
            ok, fl = rr_["ok"].mean(), rr_["fail"].mean()
            parts = [(ok, RL_C if j else LQR_C, "recovered"), (fl, FAIL, "contact"), (1 - ok - fl, "#c99a06", "no base set within T")]
            left = 0.0
            for val, col, _ in parts:
                ax.barh(yy, val, left=left, color=col, height=0.8)
                if val > 0.07:
                    ax.text(left + val / 2, yy, f"{val:.0%}", ha="center", va="center", color="white", fontsize=8.5, fontweight="bold")
                left += val
            y.append(yy)
            labels.append(f"{n}: {lab}")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel(f"fraction of {a.split} scenes")
    ax.grid(axis="x", color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    ax.legend(handles=[Line2D([], [], color=LQR_C, lw=6, label="recovered (LQR)"), Line2D([], [], color=RL_C, lw=6, label="recovered (learned)"),
                       Line2D([], [], color=FAIL, lw=6, label="contact"), Line2D([], [], color="#c99a06", lw=6, label="no base set within T")],
              loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=4, frameon=False, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(out / "strike_phase1_outcomes.png", dpi=150, facecolor=SURF)
    plt.close(fig)

    # ---------------------------------------------------------------- example scenes
    sets0 = sets_cache[key0]
    picks, rng = [], np.random.default_rng(0)
    for n in ("general", "blocked", "chaser"):
        idx = np.nonzero(r0[n]["ok"] & ~lq0[n]["ok"])[0]
        idx = rng.permutation(idx)          # a random sample of the improved scenes
        picks += [(n, int(k), "learned recovers, LQR fails") for k in idx[:2]]
    both = np.nonzero(~r0["blocked"]["ok"] & ~lq0["blocked"]["ok"])[0]
    if both.size:
        picks.append(("blocked", int(both[0]), "both fail"))
    picks = picks[:8]
    ncol = 4
    nrow = max(1, int(np.ceil(len(picks) / ncol)))
    fig_h = 4.5 * nrow + 1.2
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.3 * ncol, fig_h), squeeze=False)
    fig.patch.set_facecolor(SURF)
    R = cfg.ball_radius
    for axi, (n, k, verdict) in zip(axes.ravel(), picks):
        v = sets0[n][k]
        x0, scene = v[:10], v[10:]
        obst, balls = map(np.asarray, split_scene(cfg, jnp.asarray(scene)))
        axi.set_facecolor(SURF)
        axi.grid(color=GRID, lw=0.5)
        axi.set_axisbelow(True)
        for o in obst:
            if o[6] > 0.5:
                axi.add_patch(Polygon(outline(o), closed=True, fc=OBS_FILL, ec=OBS_EDGE, lw=0.9, zorder=3))
        for b in balls:
            if b[3] > 0.5:
                axi.add_patch(Circle(b[:2], R, fill=False, ec=CHASER, ls="--", lw=1.0, zorder=4))     # footprint
                rr_ = np.sqrt(max(R ** 2 - (x0[2] - b[2]) ** 2, 0.0))
                if rr_ > 0:                                                                        # slice at the drone's altitude
                    axi.add_patch(Circle(b[:2], rr_, fc=CHASER, alpha=0.15, ec="none", zorder=4))
                axi.add_patch(Circle(b[:2], cfg.chaser_radius, fc=CHASER, ec=INK, lw=0.7, zorder=6))
        for rr_, col in ((lq0[n], LQR_C), (r0[n], RL_C)):
            ke = int(rr_["k_end"][k])
            xs = rr_["xs"][k]
            axi.plot(xs[:ke + 1, 0], xs[:ke + 1, 1], color=col, lw=2.6, zorder=7, solid_capstyle="round")
            if rr_["ok"][k]:
                a_ = np.asarray(anchor(m, jnp.asarray(xs[ke])))
                axi.add_patch(Circle(a_[:2], m.lqr.rho_xy, fc=col, alpha=0.15, ec=col, lw=1.0, zorder=5))
                axi.plot(*a_[:2], "+", color=col, ms=8, mew=1.6, zorder=8)
            else:
                axi.plot(*xs[ke, :2], "x", color=col, ms=9, mew=2.2, zorder=9)
        axi.plot(0, 0, "o", color=INK, ms=6, zorder=10)
        vv = x0[3:5]
        axi.annotate("", xy=vv * 0.5, xytext=(0, 0), arrowprops=dict(arrowstyle="->", color=INK2, lw=1.0), zorder=6)
        lim = 9.0
        axi.set_xlim(-lim, lim)
        axi.set_ylim(-lim, lim)
        axi.set_aspect("equal")
        axi.tick_params(labelsize=7.5, colors=INK2)
        ball_z = ", ball z " + "/".join(f"{b[2]:.1f}" for b in balls if b[3] > 0.5) + " m" if (balls[:, 3] > 0.5).any() else ""
        axi.set_title(f"{n}: {verdict}\n{np.linalg.norm(x0[3:6]):.1f} m/s, z {x0[2]:.1f} m{ball_z}", loc="left", fontsize=9.3)
    for axi in axes.ravel()[len(picks):]:
        axi.axis("off")
    fig.legend(handles=[Line2D([], [], color=LQR_C, lw=2.6, label="hover-LQR backup"),
                        Line2D([], [], color=RL_C, lw=2.6, label=f"learned backup ({run0.name})"),
                        Line2D([], [], marker="o", ls="", ms=11, mfc=(0.5, 0.5, 0.5, 0.15), mec=INK2,
                               label=f"hover ellipsoid when the base set is reached (r = {m.lqr.rho_xy:.2f} m)"),
                        Line2D([], [], marker="x", ls="", ms=9, mew=2.2, color=INK2, label="contact / horizon over"),
                        Line2D([], [], marker="o", ls="", ms=11, mfc=(0.106, 0.686, 0.478, 0.15), mec=CHASER,
                               label=f"chaser ball (r = {R:.1f} m): footprint dashed, slice at the drone's altitude filled")],
               loc="lower center", ncol=3, frameon=False, fontsize=9)
    fig.suptitle(f"Phase I backups on {a.split} scenes (drone at the origin, T = {cfg.T} s)", x=0.01, ha="left", fontsize=12)
    fig.subplots_adjust(left=0.035, right=0.99, top=1.0 - 0.95 / fig_h, bottom=1.15 / fig_h, hspace=0.38, wspace=0.16)
    fig.savefig(out / "strike_phase1_examples.png", dpi=140, facecolor=SURF)
    plt.close(fig)

    # ---------------------------------------------------------------- training curves
    fig, axes = plt.subplots(1, len(REGION_NAMES) + 1, figsize=(4.0 * (len(REGION_NAMES) + 1), 3.6))
    fig.patch.set_facecolor(SURF)
    cols = ["#2a78d6", "#8a5cc2", "#1f9e89", "#c99a06", "#c4553f"]
    lqr_val = {}
    for run in runs:
        s = json.loads((run / "summary.json").read_text())
        lqr_val = s.get("lqr_val", lqr_val)
    for j, n in enumerate(list(REGION_NAMES) + ["mu_weighted"]):
        axi = axes[j]
        axi.set_facecolor(SURF)
        key_h = f"eval_m_{n}" if n != "mu_weighted" else "eval_mu_weighted"
        for c, (run, (_, _, _, _, hist)) in zip(cols, zip(runs, loaded)):
            vals = hist.get(key_h, [])
            if vals:
                steps = np.arange(1, len(vals) + 1) * json.loads((run / "configs.json").read_text())["sa"]["eval_every"] / 1e6
                axi.plot(steps, vals, "o-", ms=3, lw=1.5, color=c, label=run.name)
        if lqr_val and (n == "mu_weighted" or n in lqr_val):     # runs before the escape region have no escape entry
            ref = lqr_val[n]["m_hat"] if n != "mu_weighted" else lqr_val["mu_weighted"]
            axi.axhline(ref, color=LQR_C, ls="--", lw=1.3, label="hover LQR")
        axi.set_title(n if n != "mu_weighted" else "weighted", loc="left")
        axi.set_xlabel("env steps [M]")
        axi.set_ylim(-0.02, 1.02)
        axi.grid(color=GRID, lw=0.5)
    axes[0].set_ylabel("recovered fraction (val)")
    axes[-1].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out / "strike_phase1_curves.png", dpi=150, facecolor=SURF)
    plt.close(fig)
    print(f"wrote {out}/strike_phase1_table.md, strike_phase1_outcomes.png, strike_phase1_examples.png, strike_phase1_curves.png")


if __name__ == "__main__":
    main()
