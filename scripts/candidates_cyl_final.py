#!/usr/bin/env python
"""Fly every candidate layout of the deployment demo (docker_batch/configs/cyl_online_candidates.json) so that one can be
picked: small virtual cylinders, the start and the goal as close as the base set allows.

Each candidate is flown by two controllers on the realistic plant (no wind), the same episodes and plant draws:
  filter    the deployment nominal (go-to-goal with a heading search) + the filter
  straight  a straight-line nominal (no search) + the filter: the filter alone takes it round the cylinders

    python scripts/candidates_cyl_final.py --out results/cyl_online/candidates
    python scripts/candidates_cyl_final.py --only O1,S1 --episodes 5 --out /tmp/cand

Writes candidates.json (every metric), candidates_table.md, and one figure per case (candidates_one.png,
candidates_two.png): the top views of all candidates at the same scale, every episode drawn, with the stretches where
the filter overrode the nominal in episode 1 marked. Adopt one with scripts/eval_cyl_final.py --candidates O2,S3.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402

from eval_cyl_final import (GRID, INK, INK2, MUTED, SERIES, apply_candidate, goal_for, load_candidates,  # noqa: E402
                            parse_args as final_args, run_case, slalom_routes)
from ps2rl.online.model import load_backup  # noqa: E402

OVERRIDE = "#e34948"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backup", default=str(ROOT / "checkpoints/cyl_online/td3_w128_seed0"))
    p.add_argument("--candidates_file", default=str(ROOT / "docker_batch/configs/cyl_online_candidates.json"))
    p.add_argument("--only", default="", help="comma-separated candidate names (default: all)")
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    return p.parse_args(argv)


def metrics(m, r: dict, eps, kind: str) -> dict:
    """Rates and the shape of the flown paths, up to each episode's end (goal, contact or time limit)."""
    L = r["log"]
    X = L["x"]                                                           # (T, E, 10)
    t_end = np.asarray(r["per_episode"]["t_goal"])
    live = L["t"][:, None] <= np.where(t_end > 0, t_end, np.inf)[None, :]
    y = np.where(live, X[:, :, 1], np.nan)
    v = np.where(live, np.linalg.norm(X[:, :, 3:6], axis=2), np.nan)
    lam = np.where(live, L["lam"], np.nan)
    switches = np.sum((np.abs(np.diff(L["lam"], axis=0)) > 0.25) & live[1:], axis=0)
    out = {"success": r["success"], "contact": r["contact"], "floor": r["floor"], "timeout": r["timeout"],
           "t_goal": r["t_goal_mean"], "t_goal_max": float(np.max(t_end)) if np.all(t_end > 0) else float("nan"),
           "min_clear": r["min_clear"], "lam_mean": r["lam_mean"], "intervene": r["intervene_frac"],
           "lam_switches": float(np.mean(switches)), "overridden_episodes": float(np.mean(np.nanmin(lam, axis=0) < 0.999)),
           "y_min": float(np.nanmin(y)), "y_max": float(np.nanmax(y)), "v_max": float(np.nanmax(v)),
           "path_ratio": r["path_ratio"], "mode_none": r.get("mode_none", 0.0),
           "steps_p99": r.get("steps_p99", 0.0), "steps_max": r.get("steps_max", 0)}
    if kind == "two":
        out["routes"] = slalom_routes(r, eps.obst[0])
    else:
        o = eps.obst[0, 0]
        crossed = np.any(X[:, :, 0] >= o[0], axis=0)
        k = np.argmax(X[:, :, 0] >= o[0], axis=0)
        left = X[k, np.arange(X.shape[1]), 1] > o[1]
        out["routes"] = {"left": float(np.mean(crossed & left)), "right": float(np.mean(crossed & ~left)),
                         "not_past": float(np.mean(~crossed))}
    return out


def plot(m, runs: list, kind: str, path: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Circle

    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2,
                         "ytick.color": INK2, "axes.titlesize": 9.5, "axes.titleweight": "regular"})
    x_hi = max(float(rn["eps"].goal[0, 0]) for rn in runs) + 0.6
    y_lo = min(min(rn["res"][k]["y_min"] for k in ("filter", "straight")) for rn in runs) - 0.5
    y_hi = max(max(rn["res"][k]["y_max"] for k in ("filter", "straight")) for rn in runs) + 0.6
    for rn in runs:
        o = rn["eps"].obst[0]
        o = o[o[:, 3] > 0.5]
        y_lo = min(y_lo, float((o[:, 1] - o[:, 2] - m.r_drone - m.rho_xy).min()) - 0.15)
        y_hi = max(y_hi, float((o[:, 1] + o[:, 2] + m.r_drone + m.rho_xy).max()) + 0.15)
    xr, yr = (-0.6, x_hi), (y_lo, y_hi)
    ncol = 2
    nrow = int(np.ceil(len(runs) / ncol))
    w = 4.6
    h = w * (yr[1] - yr[0]) / (xr[1] - xr[0]) + 0.75
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * w, nrow * h + 0.9), squeeze=False, sharex=True, sharey=True)
    for i, rn in enumerate(runs):
        ax = axes[i // ncol, i % ncol]
        eps, logs, res = rn["eps"], rn["logs"], rn["res"]
        for o in eps.obst[0]:
            if o[3] < 0.5:
                continue
            ax.add_patch(Circle(o[0:2], o[2], color="#9a9893", zorder=3))
            ax.add_patch(Circle(o[0:2], o[2] + m.r_drone, fill=False, ls="--", lw=0.8, color=INK2, zorder=3))
            ax.add_patch(Circle(o[0:2], o[2] + m.r_drone + m.rho_xy, fill=False, ls=":", lw=0.8, color=MUTED, zorder=3))
        for name, col in (("straight", SERIES["blind"]), ("filter", SERIES["filter"])):
            L = logs[name]
            t_end = np.asarray(L["t_goal"])
            for e in range(L["x"].shape[1]):
                n = len(L["t"]) if t_end[e] <= 0 else int(np.searchsorted(L["t"], t_end[e])) + 1
                ax.plot(L["x"][:n, e, 0], L["x"][:n, e, 1], lw=0.8, alpha=0.45, color=col, zorder=2)
        for name, col in (("straight", SERIES["blind"]), ("filter", SERIES["filter"])):
            L = logs[name]
            n = len(L["t"]) if L["t_goal"][0] <= 0 else int(np.searchsorted(L["t"], L["t_goal"][0])) + 1
            X0, lam0 = L["x"][:n, 0], L["lam"][:n, 0]
            ov = lam0 < 0.999
            ax.plot(X0[:, 0], X0[:, 1], lw=1.6, color=col, zorder=4)
            ax.plot(X0[ov, 0], X0[ov, 1], ls="none", marker="o", ms=2.2, mew=0, color=OVERRIDE, zorder=5)
        g = eps.goal[0]
        ax.add_patch(Circle(g[0:2], rn["goal_radius"], fill=False, ls=":", lw=0.8, color=INK, zorder=3))
        ax.plot([0], [0], marker="o", color=INK, ms=4, zorder=6)
        ax.plot([g[0]], [g[1]], marker="*", color=INK, ms=9, zorder=6)
        ax.axhline(0.0, color=MUTED, lw=0.6, ls=(0, (1, 2)), zorder=1)
        f, s = res["filter"], res["straight"]
        route = ""
        if kind == "two":
            rt = f["routes"]
            route = f", weave {rt['weave']:.0%}"
        ax.set_title(f"{rn['name']}: course {g[0]:.1f} m, y up to {f['y_max']:.1f} m\n"
                     f"nominal + filter {f['success']:.0%} in {f['t_goal']:.1f} s{route} | straight {s['success']:.0%}"
                     + (f" in {s['t_goal']:.1f} s" if s["success"] > 0 else ""), loc="left", color=INK)
        ax.set_aspect("equal")
        ax.set_xlim(*xr)
        ax.set_ylim(*yr)
        ax.grid(color=GRID, lw=0.6)
        if i // ncol == nrow - 1:
            ax.set_xlabel("x [m]")
        if i % ncol == 0:
            ax.set_ylabel("y [m]")
    for j in range(len(runs), nrow * ncol):
        axes[j // ncol, j % ncol].axis("off")
    handles = [Line2D([], [], color=SERIES["filter"], lw=1.4, label="nominal + filter (20 episodes)"),
               Line2D([], [], color=SERIES["blind"], lw=1.4, label="straight line + filter"),
               Line2D([], [], color=OVERRIDE, ls="none", marker="o", ms=3, label="filter overriding (episode 1 of each)"),
               Line2D([], [], color=INK2, lw=0.8, ls="--", label="contact (cylinder + drone radius)"),
               Line2D([], [], color=MUTED, lw=0.8, ls=":", label="+ hover-set reach; goal: arrival circle")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=8)
    fig.suptitle(title, x=0.01, ha="left", color=INK, fontsize=10.5)
    fig.tight_layout(rect=(0, 0.07 if nrow > 1 else 0.12, 1, 0.97))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv=None) -> None:
    a = parse_args(argv)
    m, actor, _ = load_backup(a.backup)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cands = load_candidates(a.candidates_file)
    names = [n for n in (a.only.split(",") if a.only else cands) if n]
    base = final_args(["--out", str(out), "--backup", a.backup, "--episodes", str(a.episodes), "--seed", str(a.seed)])
    runs = {"one": [], "two": []}
    table = {}
    for name in names:
        fa, kind = apply_candidate(base, cands[name])
        r = run_case(m, actor, fa, kind, only=("filter", "blind"))
        eps = r["eps"]
        res = {"filter": metrics(m, r["filter"], eps, kind), "straight": metrics(m, r["blind"], eps, kind)}
        layout = {k: cands[name][k] for k in cands[name] if k != "about"}
        start_gap = (fa.obst_x if kind == "one" else fa.first_x) - fa.radius
        row = {"kind": kind, "layout": layout, "about": cands[name].get("about", ""), "goal_x": goal_for(fa, kind),
               "start_to_surface": start_gap, **res}
        table[name] = row
        logs = {k: {"t": r[b]["log"]["t"], "x": r[b]["log"]["x"], "lam": r[b]["log"]["lam"],
                    "t_goal": r[b]["per_episode"]["t_goal"]} for k, b in (("filter", "filter"), ("straight", "blind"))}
        runs[kind].append({"name": name, "eps": eps, "logs": logs, "res": res, "goal_radius": r["scene"]["goal_radius"]})
        f, s = res["filter"], res["straight"]
        print(f"[cand] {name} ({kind}) goal x {row['goal_x']:.1f}: success {f['success']:.2f} contact {f['contact']:.2f} "
              f"t {f['t_goal']:.1f} s y [{f['y_min']:.2f}, {f['y_max']:.2f}] min clear {f['min_clear']:.2f} lam "
              f"{f['lam_mean']:.2f} intervene {f['intervene']:.2f} overridden {f['overridden_episodes']:.2f} "
              f"routes {f['routes']} | straight: success {s['success']:.2f} contact {s['contact']:.2f} t {s['t_goal']:.1f} "
              f"min clear {s['min_clear']:.2f} routes {s['routes']}", flush=True)
    (out / "candidates.json").write_text(json.dumps({"args": vars(a), "candidates": table}, indent=2, default=float))
    lines = ["| name | course (goal x) | start to surface | success | time [s] | y range [m] | min clear [m] | "
             "filter overrides (episodes / steps) | route | straight + filter: success, time, min clear |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for name, row in table.items():
        f, s = row["filter"], row["straight"]
        rt = (f"weave {f['routes']['weave']:.0%}" if row["kind"] == "two" else f"left {f['routes']['left']:.0%}")
        lines.append(f"| {name} | {row['goal_x']:.1f} m | {row['start_to_surface']:.1f} m | {f['success']:.0%} | "
                     f"{f['t_goal']:.1f} | {f['y_min']:.2f} .. {f['y_max']:.2f} | {f['min_clear']:.2f} | "
                     f"{f['overridden_episodes']:.0%} / {f['intervene']:.0%} | {rt} | {s['success']:.0%}, "
                     f"{s['t_goal']:.1f} s, {s['min_clear']:.2f} m |")
    (out / "candidates_table.md").write_text("\n".join(lines) + "\n")
    bk = f"learned backup ({actor.width} wide), 2 blends, horizon 1 s"
    for kind, label in (("one", "one cylinder"), ("two", "slalom")):
        if runs[kind]:
            r0 = cands[runs[kind][0]["name"]]
            plot(m, runs[kind], kind, out / f"candidates_{kind}.png",
                 f"Candidate layouts, {label} (r {r0['radius']:g} m, virtual), start at rest, 1 m/s cruise; {bk}")
    print(f"[cand] wrote {out}/candidates.json, candidates_table.md, figures")


if __name__ == "__main__":
    main()
