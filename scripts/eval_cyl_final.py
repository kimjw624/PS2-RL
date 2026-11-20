#!/usr/bin/env python
"""The deployment test of the obstacle-only online filter: fly straight forward to a goal, slowly, on the realistic plant
(no wind), past small virtual cylinders (r 0.1 m). The defaults are the demo layouts chosen from the candidates
(docker_batch/configs/cyl_online_candidates.json):
  one (O3)     one cylinder 1.3 m ahead, its top touching the line (centre y -0.1: the vehicle goes round it on the
               left), the goal 2.0 m beyond it (x 3.3 m)
  two (S1)     the slalom: the first cylinder 1.1 m ahead, its top touching the line, the second 2.0 m further on at
               y 1.6 m, across the way back to the goal: the vehicle has to come back towards the line early and pass
               under it (a weave); the goal 1.2 m beyond the second (x 4.3 m)
1.1 m is the closest start the base set allows at r 0.1 (a new plan needs the start hover 1.0 m from the surface);
1.3 m leaves 0.2 m for the position hold before the hand-over. ``--goal_after`` sets both goal distances at once.
``--layout gap``: the side-by-side pair with a gap (the earlier layout). The earlier r 0.3 m course:
``--radius 0.3 --obst_x 1.5 --lateral 0 --first_x 1.5 --first_y -0.3 --second_dx 3.5 --second_y 1.5 --goal_after 5``.
numpy (+ numba if installed).

    python scripts/eval_cyl_final.py --backup checkpoints/cyl_online/td3_w128_seed0 --out results/cyl_online/final
    python scripts/eval_cyl_final.py --backup checkpoints/cyl_online/lqr --out /tmp/final_lqr     # hover-LQR backup
    python scripts/eval_cyl_final.py --candidates O2,S3 --out /tmp/final_o2_s3   # other layouts from the candidates
    python scripts/eval_cyl_final.py --spacing_sweep 2.0,2.5,3.0 --out /tmp/spacing
    python scripts/eval_cyl_final.py --layout gap --gap_sweep 2.0,2.5,3.0 --out /tmp/gap

For each case three controllers fly the same episodes (same plant draws):
  filter    the deployment nominal (go-to-goal, heading search keeping ``--search_margin`` from the surfaces) + the filter
  nominal   the same nominal alone (no filter)
  blind     a straight-line nominal (no search) + the filter: the filter alone takes it round the cylinder(s)

Writes summary.json (rates, times, clearances, filter statistics, per-decision rollout steps, the route taken in the
two-cylinder case) and figures: a top view per case (every episode, one coloured by the blend lam) and time series of
that episode (speed, lam, clearance).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from ps2rl.online.filter import MODES, FilterConfig, OnlineFilter  # noqa: E402
from ps2rl.online.model import load_backup, surface_distance  # noqa: E402
from ps2rl.online.nominal import Nominal, NominalConfig  # noqa: E402
from ps2rl.online.sim import PlantConfig, SceneConfig, make_scenes, run_episodes  # noqa: E402

LAMBDA_SETS = {2: (1.0, 0.0), 3: (1.0, 0.5, 0.0), 5: (1.0, 0.75, 0.5, 0.25, 0.0)}
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
SERIES = {"filter": "#2a78d6", "nominal": "#eb6834", "blind": "#1baf7a"}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backup", default=str(ROOT / "checkpoints/cyl_online/td3_w128_seed0"))
    p.add_argument("--blends", type=int, default=2, choices=(2, 3, 5))
    p.add_argument("--family", type=int, default=1, choices=(1, 2), help="1: the backup alone, 2: + the hover LQR")
    p.add_argument("--horizon", type=float, default=1.0)
    p.add_argument("--period", type=int, default=1)
    p.add_argument("--hold", type=int, default=1, help="model steps per command: 2 = a 40 ms control period")
    p.add_argument("--margin", type=float, default=0.1)
    p.add_argument("--fresh_margin", type=float, default=0.15)
    p.add_argument("--cruise", type=float, default=1.0)
    p.add_argument("--search_margin", type=float, default=1.0)
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--goal_after", type=float, default=None, help="the goal this far beyond the last cylinder [m], both "
                                                                  "cases (default: --goal_after_one / --goal_after_two)")
    p.add_argument("--goal_after_one", type=float, default=2.0, help="one cylinder: the goal this far beyond it [m]")
    p.add_argument("--goal_after_two", type=float, default=1.2, help="two cylinders: the goal this far beyond the last [m]")
    p.add_argument("--obst_x", type=float, default=1.3, help="one cylinder / gap layout: x of the cylinder(s) [m]")
    p.add_argument("--radius", type=float, default=0.1)
    p.add_argument("--lateral", type=float, default=-0.1, help="one cylinder: y of its centre [m] (-r: top on the line)")
    p.add_argument("--layout", default="slalom", choices=("slalom", "gap"), help="two-cylinder layout")
    p.add_argument("--first_x", type=float, default=1.1, help="slalom: x of the first cylinder [m]")
    p.add_argument("--first_y", type=float, default=-0.1, help="slalom: y of the first cylinder [m] (in the way)")
    p.add_argument("--second_dx", type=float, default=2.0, help="slalom: the second this far beyond the first [m]")
    p.add_argument("--second_y", type=float, default=1.6, help="slalom: y of the second [m] (across the way back)")
    p.add_argument("--gap", type=float, default=3.0, help="gap layout: surface-to-surface gap [m]")
    p.add_argument("--low_y", type=float, default=-0.15,
                   help="gap layout: y of the lower cylinder's centre [m] (-0.15: on the straight line)")
    p.add_argument("--centred", action="store_true", help="gap layout: gap centred on the line instead (no --low_y)")
    p.add_argument("--z", type=float, default=1.5)
    p.add_argument("--sensing", type=float, default=6.0)
    p.add_argument("--ideal", action="store_true", help="the filter's own model as the plant (no lag, delay, wind, noise)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gap_sweep", default="", help="gap layout: comma-separated gaps [m], the two-cylinder case per gap")
    p.add_argument("--start_sweep", default="", help="comma-separated distances [m] from the start to the first cylinder "
                                                     "(one: --obst_x, slalom: --first_x), both cases per distance")
    p.add_argument("--spacing_sweep", default="", help="slalom: comma-separated --second_dx values [m], the two-cylinder "
                                                       "case per spacing (how tight a weave still passes)")
    p.add_argument("--candidates", default="", help="comma-separated layout names from --candidates_file (one per case: "
                                                    "an O* name for one cylinder, an S* name for the slalom); overrides the "
                                                    "layout arguments and runs only the cases named")
    p.add_argument("--candidates_file", default=str(ROOT / "docker_batch/configs/cyl_online_candidates.json"))
    p.add_argument("--out", required=True)
    return p.parse_args(argv)


CANDIDATE_KEYS = {"one": ("radius", "obst_x", "lateral", "goal_after"),
                  "two": ("radius", "first_x", "first_y", "second_dx", "second_y", "goal_after")}


def load_candidates(path) -> dict:
    return json.loads(Path(path).read_text())["candidates"]


def apply_candidate(a, cand: dict):
    """A copy of the arguments with a candidate layout's fields set; returns (args, kind)."""
    b = argparse.Namespace(**vars(a))
    kind = cand["kind"]
    for k in CANDIDATE_KEYS[kind]:
        if k in cand:
            setattr(b, k, float(cand[k]))
    if kind == "two":
        b.layout = "slalom"
    return b, kind


def goal_after_for(a, kind: str) -> float:
    if a.goal_after is not None:
        return a.goal_after
    return a.goal_after_one if kind == "one" else a.goal_after_two


def goal_for(a, kind: str) -> float:
    last = a.obst_x if (kind == "one" or a.layout == "gap") else a.first_x + a.second_dx
    return last + goal_after_for(a, kind)


def run_case(m, actor, a, kind: str, only: tuple = ("filter", "nominal", "blind")) -> dict:
    sc = SceneConfig(kind=kind, goal_after=goal_after_for(a, kind), z=a.z, obst_x=a.obst_x, radius=a.radius, lateral=a.lateral, gap=a.gap,
                     low_y=None if a.centred else a.low_y, two_layout=a.layout, first_x=a.first_x, first_y=a.first_y,
                     second_dx=a.second_dx, second_y=a.second_y,
                     sensing_range=a.sensing, start_speed=(0.0, 0.0), episode_sec=40.0)
    eps = make_scenes(sc, a.episodes, seed=a.seed)
    backups = ("lqr",) if actor is None else (("learned",) if a.family == 1 else ("learned", "lqr"))
    fc = FilterConfig(lambdas=LAMBDA_SETS[a.blends], backups=backups, horizon=a.horizon, period=a.period, hold=a.hold,
                      margin=a.margin, fresh_margin=a.fresh_margin)
    plant = PlantConfig(ideal=a.ideal)
    deploy = NominalConfig(cruise=a.cruise, margin=a.search_margin)
    out = {}
    for name, nc, use_f in (("filter", deploy, True), ("nominal", deploy, False), ("blind", NominalConfig(cruise=a.cruise, search=False), True)):
        if name not in only:
            continue
        nom = Nominal(m, nc)
        f = OnlineFilter(m, actor, fc, nom) if use_f else None
        out[name] = run_episodes(m, eps, sc, plant, nom, f, seed=a.seed + 11, log=True, hold=a.hold)
    out["scene"] = sc.as_dict()
    out["filter_cfg"] = fc.as_dict()
    out["eps"] = eps
    return out


def routes(res_one: dict, obst: np.ndarray, x_line: float) -> dict:
    """Where each episode crossed the cylinders' line x = x_line: through the gap between the two cylinders, below the
    lower one, above the upper one (fractions; the rest never crossed)."""
    X = res_one["log"]["x"]
    o = obst[obst[:, 3] > 0.5]
    lo, hi = o[np.argmin(o[:, 1])], o[np.argmax(o[:, 1])]
    crossed = np.any(X[:, :, 0] >= x_line, axis=0)
    k = np.argmax(X[:, :, 0] >= x_line, axis=0)
    y = X[k, np.arange(X.shape[1]), 1]
    return {"gap": float(np.mean(crossed & (y > lo[1] + lo[2]) & (y < hi[1] - hi[2]))),
            "below": float(np.mean(crossed & (y <= lo[1] - lo[2]))), "above": float(np.mean(crossed & (y >= hi[1] + hi[2]))),
            "not_crossed": float(np.mean(~crossed))}


def slalom_routes(res_one: dict, obst: np.ndarray) -> dict:
    """Which side of each cylinder the episodes passed (first: above / below, second: above / below; the rest did not
    get past the second). "weave": above the first, under the second."""
    X = res_one["log"]["x"]
    c1, c2 = obst[0], obst[1]
    E = X.shape[1]

    def side(c):
        past = np.any(X[:, :, 0] >= c[0], axis=0)
        k = np.argmax(X[:, :, 0] >= c[0], axis=0)
        return past, X[k, np.arange(E), 1] > c[1]

    p1, a1 = side(c1)
    p2, a2 = side(c2)
    done = p1 & p2
    return {"weave": float(np.mean(done & a1 & ~a2)), "over_both": float(np.mean(done & a1 & a2)),
            "under_first": float(np.mean(done & ~a1)), "not_past": float(np.mean(~done))}


def plot_case(m, res: dict, kind: str, path: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.patches import Circle

    eps = res["eps"]
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2,
                         "ytick.color": INK2, "axes.titlesize": 10, "axes.titleweight": "regular"})
    obst = eps.obst[0]
    oy = obst[obst[:, 3] > 0.5]
    g = eps.goal[0]
    ring = oy[:, 2] + m.r_drone + m.rho_xy
    ys = np.concatenate([res[k]["log"]["x"][:, :, 1].ravel() for k in ("filter", "nominal", "blind")])
    ys = ys[np.isfinite(ys)]
    xr = (-0.6 if g[0] < 6.0 else -1.2, g[0] + 0.8)
    yr = (min(float(np.min(oy[:, 1] - ring)), float(ys.min())) - 0.35, max(float(np.max(oy[:, 1] + ring)), float(ys.max())) + 0.45)
    if g[0] >= 6.0:                                    # the long courses: at least +-2 m
        yr = (min(-2.0, yr[0]), max(2.0, yr[1]))
    top_h = float(np.clip(7.6 * (yr[1] - yr[0]) / (xr[1] - xr[0]), 2.6, 5.0))     # equal aspect in a 9 in wide figure
    fig, axes = plt.subplots(3, 1, figsize=(9.0, top_h + 5.2), gridspec_kw={"height_ratios": [top_h, 2.1, 2.1]})
    ax = axes[0]
    for o in obst:
        if o[3] < 0.5:
            continue
        ax.add_patch(Circle(o[0:2], o[2], color="#9a9893", zorder=3))
        ax.add_patch(Circle(o[0:2], o[2] + m.r_drone, fill=False, ls="--", lw=0.8, color=INK2, zorder=3))
        ax.add_patch(Circle(o[0:2], o[2] + m.r_drone + m.rho_xy, fill=False, ls=":", lw=0.8, color=MUTED, zorder=3))
    for name in ("nominal", "blind", "filter"):
        X = res[name]["log"]["x"]                       # (T, E, 10)
        for e in range(X.shape[1]):
            ax.plot(X[:, e, 0], X[:, e, 1], lw=0.8, alpha=0.5, color=SERIES[name], zorder=2,
                    label=None if e else {"filter": "nominal + filter", "nominal": "nominal alone",
                                          "blind": "straight line + filter"}[name])
    L = res["filter"]["log"]
    X0 = L["x"][:, 0]
    pts = X0[:, 0:2].reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("lam", ["#e34948", "#eda100", "#2a78d6"])   # 0 backup .. 1 nominal
    lc = LineCollection(segs, cmap=cmap, norm=plt.Normalize(0.0, 1.0), lw=2.4, zorder=4)
    lc.set_array(L["lam"][:len(segs), 0])
    ax.add_collection(lc)
    ax.plot([0], [0], marker="o", color=INK, ms=4)
    ax.plot([g[0]], [g[1]], marker="*", color=INK, ms=9)
    ax.add_patch(Circle(g[0:2], res["scene"]["goal_radius"], fill=False, ls=":", lw=0.8, color=INK, zorder=3))
    ax.text(g[0], g[1] - 0.18, "goal", ha="center", va="top", color=INK2, fontsize=8)
    ax.text(-0.15, 0.0, "start", ha="right", va="center", color=INK2, fontsize=8)
    ax.set_aspect("equal")
    ax.set_xlim(*xr)
    ax.set_ylim(*yr)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(color=GRID, lw=0.6)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16 * 4.0 / top_h), frameon=False, fontsize=8, ncol=3)
    cb = fig.colorbar(lc, ax=ax, fraction=0.025, pad=0.01)
    cb.set_label("lam, episode 1 (1 = nominal kept)", color=INK2)
    ax.set_title(title, color=INK, loc="left")
    # time series of episode 1 (filter), up to its end
    t_end = res["filter"]["per_episode"]["t_goal"][0]
    n = len(L["t"]) if t_end <= 0 else int(np.searchsorted(L["t"], t_end)) + 1
    t = L["t"][:n]
    X0 = X0[:n]
    v = np.linalg.norm(X0[:, 3:6], axis=1)
    lam = L["lam"][:n, 0]
    clear = surface_distance(X0[:, 0:2], np.repeat(eps.obst[0:1], len(t), 0)).min(axis=1) - m.r_drone
    ax = axes[1]
    ax.plot(t, v, lw=1.4, color=SERIES["filter"], label="speed [m/s]")
    ax.plot(t, lam, lw=1.0, color="#6250d6", label="blend lam")
    ax.set_ylabel("speed, lam")
    ax.set_ylim(-0.05, 1.3)
    ax.grid(color=GRID, lw=0.6)
    ax.legend(loc="upper right", frameon=False, fontsize=8, ncol=2)
    ax = axes[2]
    ax.plot(t, clear, lw=1.4, color=SERIES["filter"], label="distance to the nearest surface minus the drone radius")
    ax.axhline(0.0, color=INK, lw=0.8)
    ax.text(t[-1], 0.05, "contact", ha="right", va="bottom", color=INK2, fontsize=8)
    ax.set_ylabel("clearance [m]")
    ax.set_xlabel("time [s]")
    ax.set_ylim(-0.2, max(3.0, float(np.nanmin([clear.max(), 6.0]))))
    ax.grid(color=GRID, lw=0.6)
    ax.legend(loc="upper right", frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv=None) -> None:
    a = parse_args(argv)
    m, actor, meta = load_backup(a.backup)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.start_sweep:
        table = []
        for d in [float(v) for v in a.start_sweep.split(",") if v]:
            a.obst_x, a.first_x = d, d
            for kind in ("one", "two"):
                res = run_case(m, actor, a, kind, only=("filter", "blind"))
                r, rb = res["filter"], res["blind"]
                L = r["log"]
                t_end = np.asarray(r["per_episode"]["t_goal"])
                live = (L["t"][:, None] <= np.where(t_end > 0, t_end, np.inf)[None, :])
                flips = float(np.mean(np.sum((np.abs(np.diff(L["lam"], axis=0)) > 0.25) & live[1:], axis=0)))
                row = {"distance": d, "case": kind, "goal_x": goal_for(a, kind), "success": r["success"],
                       "contact": r["contact"], "floor": r["floor"], "timeout": r["timeout"], "t_goal": r["t_goal_mean"],
                       "min_clear": r["min_clear"], "lam_mean": r["lam_mean"], "intervene": r["intervene_frac"],
                       "lam_switches": flips, "mode_none": r["mode_none"], "straight_contact": rb["contact"],
                       "straight_floor": rb["floor"], "straight_min_clear": rb["min_clear"]}
                if kind == "two":
                    row["routes"] = (slalom_routes(r, res["eps"].obst[0]) if a.layout == "slalom"
                                     else routes(r, res["eps"].obst[0], a.obst_x))
                table.append(row)
                print(f"[start] {d:4.2f} m {kind}: success {r['success']:.2f} contact {r['contact']:.2f} timeout "
                      f"{r['timeout']:.2f} t {r['t_goal_mean']:.1f} s (goal x {row['goal_x']:g}) min clearance "
                      f"{r['min_clear']:.2f} lam {r['lam_mean']:.2f} intervene {r['intervene_frac']:.2f} switches {flips:.1f} "
                      f"no certificate {r['mode_none']:.3f}{' routes ' + str(row['routes']) if kind == 'two' else ''} | "
                      f"straight line + filter: contact {rb['contact']:.2f} min clearance {rb['min_clear']:.2f}", flush=True)
        (out / "start_sweep.json").write_text(json.dumps({"args": vars(a), "rows": table}, indent=2, default=float))
        return
    if a.spacing_sweep:
        if a.layout != "slalom":
            raise SystemExit("--spacing_sweep is for the slalom layout")
        table = []
        for dx in [float(v) for v in a.spacing_sweep.split(",") if v]:
            a.second_dx = dx
            res = run_case(m, actor, a, "two", only=("filter", "blind"))
            r, rb = res["filter"], res["blind"]
            L = r["log"]
            t_end = np.asarray(r["per_episode"]["t_goal"])
            live = (L["t"][:, None] <= np.where(t_end > 0, t_end, np.inf)[None, :])
            v = np.linalg.norm(L["x"][:, :, 3:6], axis=2)
            between = live & (L["x"][:, :, 0] > a.first_x) & (L["x"][:, :, 0] < a.first_x + dx)
            v_min = float(np.min(np.where(between, v, np.inf)))
            flips = float(np.mean(np.sum((np.abs(np.diff(L["lam"], axis=0)) > 0.25) & live[1:], axis=0)))
            rt = slalom_routes(r, res["eps"].obst[0])
            surf = float(np.hypot(dx, a.second_y - a.first_y) - 2.0 * a.radius)
            row = {"second_dx": dx, "surface_gap": surf, "success": r["success"], "contact": r["contact"],
                   "timeout": r["timeout"], "t_goal": r["t_goal_mean"], "min_clear": r["min_clear"], "lam_mean": r["lam_mean"],
                   "min_speed_between": v_min, "lam_switches": flips, "mode_none": r["mode_none"], "routes": rt,
                   "straight_contact": rb["contact"], "straight_floor": rb["floor"]}
            table.append(row)
            print(f"[spacing] second {dx:4.2f} m on (surfaces {surf:.2f} m apart): success {r['success']:.2f} contact "
                  f"{r['contact']:.2f} timeout {r['timeout']:.2f} t {r['t_goal_mean']:.1f} s, slowest between them {v_min:.2f} "
                  f"m/s, lam switches {flips:.1f} per episode, no certificate {r['mode_none']:.3f}, routes {rt} | straight "
                  f"line + filter: contact {rb['contact']:.2f}", flush=True)
        (out / "spacing_sweep.json").write_text(json.dumps({"args": vars(a), "rows": table}, indent=2, default=float))
        return
    if a.gap_sweep:
        if a.layout != "gap":
            raise SystemExit("--gap_sweep is for --layout gap")
        gaps = [float(v) for v in a.gap_sweep.split(",") if v]
        table = []
        need = 2.0 * (m.rho_xy + m.r_drone + a.margin + a.fresh_margin)
        print(f"[gap] a new plan needs the hover point {m.rho_xy + m.r_drone + a.margin + a.fresh_margin:.2f} m from each "
              f"surface: gaps below {need:.2f} m have no certifiable hover point between the cylinders")
        for gp in gaps:
            a.gap = gp
            res = run_case(m, actor, a, "two", only=("filter", "blind"))
            r = res["filter"]
            rb = res["blind"]
            L = r["log"]
            v = np.linalg.norm(L["x"][:, :, 3:6], axis=2)
            xin = (np.abs(L["x"][:, :, 0] - a.obst_x) < 1.0) & (L["t"][:, None] <= np.where(np.asarray(
                r["per_episode"]["t_goal"]) > 0, np.asarray(r["per_episode"]["t_goal"]), np.inf)[None, :])
            v_gap = float(np.min(np.where(xin, v, np.inf)))
            t_end = np.asarray(r["per_episode"]["t_goal"])
            live = (L["t"][1:, None] <= np.where(t_end > 0, t_end, np.inf)[None, :])          # before the goal
            flips = float(np.mean(np.sum((np.abs(np.diff(L["lam"], axis=0)) > 0.25) & live, axis=0)))
            rt = routes(r, res["eps"].obst[0], a.obst_x)
            through = rt["gap"]
            row = {"gap": gp, "success": r["success"], "contact": r["contact"], "timeout": r["timeout"],
                   "t_goal": r["t_goal_mean"], "min_clear": r["min_clear"], "lam_mean": r["lam_mean"],
                   "min_speed_in_gap": v_gap, "lam_switches": flips, "mode_none": r["mode_none"],
                   "straight_contact": rb["contact"], "straight_floor": rb["floor"], "through_gap": through,
                   "routes": rt}
            table.append(row)
            print(f"[gap] {gp:4.2f} m: success {r['success']:.2f} (through the gap {through:.2f}) contact {r['contact']:.2f} "
                  f"timeout {r['timeout']:.2f} "
                  f"t {r['t_goal_mean']:.1f} s, slowest in the gap {v_gap:.2f} m/s, lam switches {flips:.0f} per episode, "
                  f"no certificate {r['mode_none']:.3f}, routes {rt} | straight line + filter: contact {rb['contact']:.2f}",
                  flush=True)
        (out / "gap_sweep.json").write_text(json.dumps({"args": vars(a), "rows": table}, indent=2, default=float))
        return
    summary = {"backup": str(a.backup), "args": vars(a), "cases": {}}
    runs = [(a, kind, None) for kind in ("one", "two")]
    if a.candidates:
        cands = load_candidates(a.candidates_file)
        runs = [apply_candidate(a, cands[n]) + (n,) for n in a.candidates.split(",") if n]
        if len({kind for _, kind, _ in runs}) < len(runs):
            raise SystemExit("--candidates: at most one name per case (one O* and one S*)")
    base = a
    try:
        known = load_candidates(a.candidates_file)
    except (OSError, KeyError, ValueError):
        known = {}
    for a, kind, cname in runs:
        if cname is None:                       # the layout's name if it is one of the candidates
            cname = next((n for n, c in known.items() if c["kind"] == kind and (kind == "one" or a.layout == "slalom")
                          and all(abs(float(c[k]) - (goal_after_for(a, kind) if k == "goal_after" else getattr(a, k))) < 1e-9
                                  for k in CANDIDATE_KEYS[kind] if k in c)), None)
        res = run_case(m, actor, a, kind)
        case = {"scene": res["scene"], "filter_cfg": res["filter_cfg"]}
        for name in ("filter", "nominal", "blind"):
            r = {k: v for k, v in res[name].items() if k not in ("log", "per_episode")}
            case[name] = r
            ex = "" if name == "nominal" else (f" lam {r['lam_mean']:.2f} intervene {r['intervene_frac']:.2f} modes "
                                               + "/".join(f"{r['mode_' + k]:.2f}" for k in MODES)
                                               + f" steps {r['steps_mean']:.1f}/{r['steps_p99']:.0f}/{r['steps_max']}")
            if kind == "two":
                r["routes"] = (slalom_routes(res[name], res["eps"].obst[0]) if a.layout == "slalom"
                               else routes(res[name], res["eps"].obst[0], a.obst_x))
                ex += f" routes {r['routes']}"
            print(f"[final] {kind:3} {name:7}: success {r['success']:.2f} contact {r['contact']:.2f} floor {r['floor']:.2f} "
                  f"timeout {r['timeout']:.2f} t {r['t_goal_mean']:.1f} s min clearance {r['min_clear']:.2f} m{ex}", flush=True)
        if cname:
            case["candidate"] = cname
        summary["cases"][kind] = case
        two_label = (f"two cylinders, slalom: the second {a.second_dx:g} m on, across the way back" if a.layout == "slalom"
                     else f"two cylinders, {a.gap:g} m gap" + ("" if a.centred else ", the lower one on the line"))
        label = {"one": f"one cylinder (r {a.radius:g} m) on the line", "two": two_label}[kind]
        label = f"{cname + ': ' if cname else ''}{label}, goal x {goal_for(a, kind):g} m"
        bk = "hover LQR" if actor is None else f"learned backup ({actor.width} wide)"
        plot_case(m, res, kind, out / f"final_{kind}.png",
                  f"{label}, {a.cruise:g} m/s\n{bk}, {a.blends} blends, horizon {a.horizon:g} s, period {a.period}")
    a = base
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(f"[final] wrote {out}/summary.json, " + ", ".join(f"final_{k}.png" for k in summary["cases"]))


if __name__ == "__main__":
    main()
