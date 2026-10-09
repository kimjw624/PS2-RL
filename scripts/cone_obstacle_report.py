#!/usr/bin/env python
"""Summarise the cone-obstacle CPU study (coverage + reach tests) into tables and a figure.

    python scripts/cone_obstacle_report.py --coverage outputs/cone_obstacle/coverage/coverage.json \
        --reach outputs/cone_obstacle/reach/reach.json,outputs/cone_obstacle/reach_theta/reach.json \
        --out outputs/cone_obstacle/report

Writes ``report.md`` (markdown tables) and ``success_*.png`` (touch success vs. cylinder
radius and top depth, per base-set design and nominal).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cone_obstacle_common import resolve  # noqa: E402

import numpy as np  # noqa: E402


def geo_tag(g):
    return f"tip {g['tip_z']:g} m, theta {g['theta_deg']:g} deg, eps {g['eps']:g}"


def short(name):
    return name.replace("points:", "pt:").replace("approach_", "app_")


def coverage_tables(cov):
    lines = []
    for g in cov["geometries"]:
        lines += [f"\n### C_N coverage - {geo_tag(g)}\n",
                  "| design | c | hover: approach | hover: under tip | +x 1.0 m/s: approach | +x 1.0 m/s: under tip | down 0.5 m/s: under tip |",
                  "|---|---|---|---|---|---|---|"]
        for d, r in g["designs"].items():
            cv = r["coverage"]
            lines.append(f"| {short(d)} | {r['base_set_c']:.2f} | {100 * cv['hover']['approach (x<-0.5, z>tip)']:.1f} % | "
                         f"{100 * cv['hover']['under the tip (z<tip)']:.1f} % | {100 * cv['+x 1.0']['approach (x<-0.5, z>tip)']:.1f} % | "
                         f"{100 * cv['+x 1.0']['under the tip (z<tip)']:.1f} % | {100 * cv['down 0.5']['under the tip (z<tip)']:.1f} % |")
    return lines


def reach_tables(reach_list):
    lines = []
    for rep in reach_list:
        radii = [float(v) for v in str(rep["args"]["radii"]).split(",")]
        for g in rep["geometries"]:
            tops = g["tops"]
            noms = list(next(iter(g["runs"].values()))["summary"].keys())
            for nom in noms:
                lines += [f"\n### Touch success, nominal `{nom}` - {geo_tag(g)}\n",
                          "| design | unsafe | stalled | " + " | ".join(f"R {r:g} / top tip-{g['tip_z'] - t:.2f}" for t in tops for r in radii)
                          + f" | median time to touch (R {radii[min(1, len(radii) - 1)]:g}, first top) | filter active |",
                          "|---|---|---|" + "---|" * (len(tops) * len(radii)) + "---|---|"]
                for name, run in g["runs"].items():
                    s = run["summary"][nom]
                    eps = [e for e, m in zip(run["episodes"], g["starts"]) if m["nominal"] == nom]
                    cells = " | ".join(f"{100 * s[f'R{r:g}_H{t:g}']['success']:.0f}" for t in tops for r in radii)
                    mt = s.get(f"R{radii[min(1, len(radii) - 1)]:g}_H{tops[0]:g}", {}).get("median_time")
                    act = np.mean([e["filter_active_frac"] for e in eps])
                    big = f"R{radii[-1]:g}_H{tops[0]:g}"
                    stalled = sum(1 for e in eps if e["unsafe_step"] < 0 and e["final_speed"] < 0.1 and e["touch_time"][big] is None)
                    lines.append(f"| {short(name)} | {100 * s['unsafe']:.0f} % | {stalled}/{len(eps)} | {cells} | "
                                 f"{'-' if mt is None else f'{mt:.2f} s'} | {100 * act:.0f} % |")
    return lines


def success_figure(path, rep):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    radii = [float(v) for v in str(rep["args"]["radii"]).split(",")]
    geos = rep["geometries"]
    noms = [n for n in next(iter(geos[0]["runs"].values()))["summary"].keys() if n != "center"]
    fig, axes = plt.subplots(len(noms), len(geos), figsize=(3.6 * len(geos), 3.0 * len(noms) + 0.6), sharey=True, squeeze=False)
    for j, g in enumerate(geos):
        tops = g["tops"]
        names = list(g["runs"])
        for i, nom in enumerate(noms):
            ax = axes[i, j]
            for k, name in enumerate(names):
                s = g["runs"][name]["summary"][nom]
                for t, ls in zip(tops, ("-", "--", ":")):
                    ys = [100 * s[f"R{r:g}_H{t:g}"]["success"] for r in radii]
                    ax.plot(radii, np.asarray(ys) + 0.6 * (k - len(names) / 2), ls=ls, marker="o", ms=3,
                            color=plt.cm.tab10(k % 10), label=short(name) if t == tops[0] else None)
            ax.set_title(f"{nom} - {geo_tag(g)}", fontsize=8)
            ax.set_xlabel("cylinder radius R [m]", fontsize=8)
            if j == 0:
                ax.set_ylabel("touch success [%]", fontsize=8)
            ax.set_ylim(-5, 105)
            ax.grid(alpha=0.3)
    axes[0, -1].legend(fontsize=6, loc="lower right")
    deeper = ", ".join(f"{geos[0]['tip_z'] - t:.2f}" for t in geos[0]["tops"][1:])
    fig.suptitle(f"Touch success vs. cylinder radius (solid: first top, {geos[0]['tip_z'] - geos[0]['tops'][0]:.2f} m below the tip; "
                 f"dashed / dotted: {deeper} m below)", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--coverage", default="outputs/cone_obstacle/coverage/coverage.json")
    ap.add_argument("--reach", default="outputs/cone_obstacle/reach/reach.json")
    ap.add_argument("--out", default="outputs/cone_obstacle/report")
    a = ap.parse_args(argv)
    out = resolve(a.out)
    out.mkdir(parents=True, exist_ok=True)
    lines = ["# Cone-obstacle CPU study"]
    for c in [p for p in a.coverage.split(",") if p]:
        if resolve(c).exists():
            lines += coverage_tables(json.loads(resolve(c).read_text()))
    reps = [json.loads(resolve(r).read_text()) for r in a.reach.split(",") if r and resolve(r).exists()]
    lines += reach_tables(reps)
    for i, rep in enumerate(reps):
        success_figure(out / f"success_{i}.png", rep)
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
