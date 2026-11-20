#!/usr/bin/env python
"""Compare obstacle-only demo flights (SITL or hardware records of the ps2rl_cyl_online node) with the simulations of
the same course and controller: numpy (+ numba) and matplotlib, no JAX, no ROS.

For each flight the course, nominal and filter settings are read from its summary.json, and ``--episodes`` episodes are
simulated with them (``ps2rl.online.sim``: the realistic plant of the sweep, from rest at the hand-over point).

    python3 scripts/compare_cyl_flight_sim.py ~/ws_shared/experiments/cyl_online/2026*/ --out results/cyl_online/sitl

Writes ``compare.png`` (one top view per flight: the simulated paths and the flown path, coloured by lam) and
``compare.json`` / prints a table: time to the goal circle, minimum clearance, lam mean, intervention, lam switches,
largest y, for the flight against the simulations' mean (min .. max).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from ps2rl.online.filter import FilterConfig, OnlineFilter  # noqa: E402
from ps2rl.online.model import load_backup  # noqa: E402
from ps2rl.online.nominal import Nominal, NominalConfig  # noqa: E402
from ps2rl.online.sim import Episodes, PlantConfig, SceneConfig, run_episodes  # noqa: E402

LAMBDA_SETS = {2: (1.0, 0.0), 3: (1.0, 0.5, 0.0), 5: (1.0, 0.75, 0.5, 0.25, 0.0)}
ROOT = Path(__file__).resolve().parents[1]
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"


def load_flight(folder: Path) -> dict:
    rows = [r for r in csv.DictReader(open(folder / "flight.csv", encoding="utf-8")) if r["phase"] == "POLICY" and r["t_policy"]]
    if not rows:
        raise SystemExit(f"{folder}: no controller phase")
    col = lambda k: np.array([float(r[k]) for r in rows])  # noqa: E731
    d = {k: col(k) for k in ("t_policy", "px", "py", "lam", "clearance")}
    d["summary"] = json.loads((folder / "summary.json").read_text())
    d["name"] = folder.name
    return d


def nominal_of(rt: dict) -> tuple[str, dict]:
    """("planner" | "straight", the NominalConfig fields). Records before 2026-11-20 14:00 kept only the NominalConfig
    dict under "nominal"."""
    cfg = rt.get("nominal_config") or (rt["nominal"] if isinstance(rt.get("nominal"), dict) else {})
    name = rt["nominal"] if isinstance(rt.get("nominal"), str) else ("planner" if cfg.get("search", True) else "straight")
    return name, cfg


def simulate(flight: dict, episodes: int, seed: int) -> dict:
    s = flight["summary"]
    rt, course = s["runtime"], s["course"]
    nominal, ncfg = nominal_of(rt)
    backup = Path(rt["backup"])
    if not (backup / "backup.json").exists():                    # recorded on another machine: take the repo's copy
        backup = ROOT / "checkpoints" / "cyl_online" / backup.name
    m, actor, _ = load_backup(backup)
    z = float(rt["flight_height"])
    cyl = np.asarray(course["cylinders_xyr"], float)
    E = episodes
    x0 = np.zeros((E, 10)); x0[:, 2] = z; x0[:, 6] = 1.0
    obst = np.tile(np.c_[cyl, np.ones(len(cyl))][None], (E, 1, 1))
    goal = np.tile(np.asarray(course["goal"], float)[None], (E, 1))
    eps = Episodes(x0=x0, goal=goal, obst=obst)
    sc = SceneConfig(kind="one", z=z, sensing_range=float(rt["sensing_range"]), goal_radius=float(rt["goal_radius"]),
                     start_speed=(0.0, 0.0), episode_sec=40.0)
    nom = Nominal(m, NominalConfig(**ncfg) if ncfg else NominalConfig(cruise=float(rt["cruise"]), margin=float(rt["search_margin"]),
                                                                      search=nominal == "planner"))
    hold = int(rt.get("hold") or round(float(rt.get("control_period", m.dt)) / m.dt))    # records before 2026-11-20 15:00: 1
    fc = FilterConfig(lambdas=LAMBDA_SETS[int(rt["blends"])], backups=("learned",), horizon=float(rt["horizon"]),
                      period=int(rt["period"]), hold=hold)
    res = run_episodes(m, eps, sc, PlantConfig(), nom, OnlineFilter(m, actor, fc, nom), seed=seed, log=True,
                       observer_gain=float(rt.get("observer_gain", 2.0)), hold=hold)
    L = res["log"]
    X = L["x"]
    t_goal = np.asarray(res["per_episode"]["t_goal"])
    live = L["t"][:, None] <= np.where(t_goal > 0, t_goal, np.inf)[None, :]
    lam = np.where(live, L["lam"], np.nan)
    switches = np.sum((np.abs(np.diff(L["lam"], axis=0)) > 0.25) & live[1:], axis=0)
    return {"res": res, "X": X, "t": L["t"], "t_goal": t_goal, "lam_ep": np.nanmean(lam, axis=0), "switches": switches,
            "y_max": np.nanmax(np.where(live, X[:, :, 1], np.nan), axis=0), "m": m}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("flights", nargs="+")
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--out", default="", help="folder for compare.png / compare.json (default: next to the first flight)")
    a = p.parse_args(argv)
    flights = [load_flight(Path(f).expanduser()) for f in a.flights]
    out = Path(a.out).expanduser() if a.out else Path(a.flights[0]).expanduser().parent
    out.mkdir(parents=True, exist_ok=True)
    table = []
    sims = []
    for f in flights:
        sim = simulate(f, a.episodes, a.seed)
        sims.append(sim)
        c = f["summary"]["controller"]
        r = sim["res"]
        tg = sim["t_goal"][sim["t_goal"] > 0]
        row = {"flight": f["name"], "course": f["summary"]["runtime"]["course"], "nominal": nominal_of(f["summary"]["runtime"])[0],
               "outcome": f["summary"]["outcome"],
               "flight_goal_s": c.get("goal_entered_s"), "sim_goal_s": [float(tg.mean()), float(tg.min()), float(tg.max())] if len(tg) else None,
               "sim_success": r["success"], "sim_contact": r["contact"],
               "flight_min_clear": c.get("min_clearance_m"), "sim_min_clear": [float(np.mean(r["per_episode"]["min_clear"])), r["min_clear"]],
               "flight_lam": c.get("lam_mean"), "sim_lam": float(np.mean(sim["lam_ep"])),
               "flight_intervene": c.get("intervene_fraction"), "sim_intervene": r["intervene_frac"],
               "flight_switches": c.get("lam_switches"), "sim_switches": float(np.mean(sim["switches"])),
               "flight_y_max": (c.get("y_range_m") or [None, None])[1], "sim_y_max": float(np.nanmean(sim["y_max"]))}
        table.append(row)
        sg = row["sim_goal_s"]
        print(f"{f['name']}: {row['outcome']}")
        print(f"  goal circle: flight {row['flight_goal_s']:.2f} s, sim {sg[0]:.2f} s ({sg[1]:.2f}..{sg[2]:.2f}), sim success "
              f"{r['success']:.2f}, contact {r['contact']:.2f}")
        print(f"  min clearance: flight {row['flight_min_clear']:.2f} m, sim mean {row['sim_min_clear'][0]:.2f} (min "
              f"{row['sim_min_clear'][1]:.2f}); lam: flight {row['flight_lam']:.2f}, sim {row['sim_lam']:.2f}; intervention: "
              f"flight {row['flight_intervene']:.2f}, sim {row['sim_intervene']:.2f}; switches: flight {row['flight_switches']}, "
              f"sim {row['sim_switches']:.0f}; y max: flight {row['flight_y_max']:.2f}, sim {row['sim_y_max']:.2f} m")
    (out / "compare.json").write_text(json.dumps({"episodes": a.episodes, "seed": a.seed, "rows": table}, indent=2, default=float) + "\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.lines import Line2D
    from matplotlib.patches import Circle

    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2,
                         "ytick.color": INK2, "axes.titlesize": 9.5})
    cmap = LinearSegmentedColormap.from_list("lam", ["#e34948", "#eda100", "#2a78d6"])
    xs = np.concatenate([f["px"] for f in flights] + [[0.0]] + [np.asarray(f["summary"]["course"]["goal"])[0:1] for f in flights])
    ys = np.concatenate([f["py"] for f in flights] + [np.asarray(s["y_max"]) for s in sims])
    cyl_all = np.concatenate([np.asarray(f["summary"]["course"]["cylinders_xyr"], float) for f in flights])
    reach = cyl_all[:, 2] + sims[0]["m"].r_drone + sims[0]["m"].rho_xy
    xr = (-0.5, float(max(xs.max(), (cyl_all[:, 0] + reach).max())) + 0.7)
    yr = (float(min(ys.min(), (cyl_all[:, 1] - reach).min())) - 0.25, float(max(ys.max(), (cyl_all[:, 1] + reach).max())) + 0.35)
    n = len(flights)
    w = 5.2
    h = w * (yr[1] - yr[0]) / (xr[1] - xr[0]) + 0.9
    fig, axes = plt.subplots(n, 1, figsize=(w + 1.2, n * h + 0.6), squeeze=False)
    for i, (f, sim) in enumerate(zip(flights, sims)):
        ax = axes[i, 0]
        c = f["summary"]["course"]
        m = sim["m"]
        for o in np.asarray(c["cylinders_xyr"], float):
            ax.add_patch(Circle(o[0:2], o[2], color="#9a9893", zorder=3))
            ax.add_patch(Circle(o[0:2], o[2] + m.r_drone, fill=False, ls="--", lw=0.8, color=INK2, zorder=3))
            ax.add_patch(Circle(o[0:2], o[2] + m.r_drone + m.rho_xy, fill=False, ls=":", lw=0.8, color=MUTED, zorder=3))
        g = np.asarray(c["goal"])
        ax.plot([g[0]], [g[1]], marker="*", color=INK, ms=9, zorder=6)
        ax.add_patch(Circle(g[0:2], float(c.get("goal_radius", 0.5)), fill=False, ls=":", lw=0.8, color=INK, zorder=3))
        X, tg, t = sim["X"], sim["t_goal"], sim["t"]
        for e in range(X.shape[1]):
            k = len(t) if tg[e] <= 0 else int(np.searchsorted(t, tg[e])) + 1
            ax.plot(X[:k, e, 0], X[:k, e, 1], lw=0.8, alpha=0.35, color=MUTED, zorder=2)
        pts = np.c_[f["px"], f["py"]]
        lc = LineCollection(np.stack([pts[:-1], pts[1:]], axis=1), cmap=cmap, norm=plt.Normalize(0, 1), lw=2.2, zorder=4)
        lc.set_array(f["lam"][:-1])
        ax.add_collection(lc)
        ax.plot([0.0], [0.0], marker="o", color=INK, ms=4, zorder=6)
        row = table[i]
        sg = row["sim_goal_s"]
        ax.set_title(f"{row['course']}, {row['nominal']}: flight {row['flight_goal_s']:.1f} s to the goal circle, min clearance "
                     f"{row['flight_min_clear']:.2f} m, lam {row['flight_lam']:.2f}\n"
                     f"simulation ({a.episodes} episodes): {sg[0]:.1f} s ({sg[1]:.1f}..{sg[2]:.1f}), min clearance "
                     f"{row['sim_min_clear'][1]:.2f} m, lam {row['sim_lam']:.2f}", loc="left", color=INK)
        ax.set_xlim(*xr); ax.set_ylim(*yr); ax.set_aspect("equal")
        ax.grid(color=GRID, lw=0.6)
        ax.set_ylabel("y north [m]")
        if i == n - 1:
            ax.set_xlabel("x east of the take-off point [m]")
        if i == 0:
            ax.legend(handles=[Line2D([], [], color="#2a78d6", lw=2.2, label="flight (colour: lam, red = backup)"),
                               Line2D([], [], color=MUTED, lw=0.8, label="simulation"),
                               Line2D([], [], color=INK2, lw=0.8, ls="--", label="contact"),
                               Line2D([], [], color=MUTED, lw=0.8, ls=":", label="hover-set reach")],
                      fontsize=7.5, loc="upper left", frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(out / "compare.png", dpi=130)
    print(f"wrote {out / 'compare.png'}, {out / 'compare.json'}")


if __name__ == "__main__":
    main()
