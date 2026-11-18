#!/usr/bin/env python
"""Collect the Phase-II landing evaluations (``<root>/eval_<name>/summary.json``) into ``<root>/REPORT.md``."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _f(v, fmt=".3f"):
    return "–" if v is None else format(v, fmt)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True)
    p.add_argument("--names", nargs="+", required=True)
    args = p.parse_args(argv)
    root = Path(args.root)
    md = ["# Phase-II landing: tracker vs tracker + filter vs PS2", ""]
    rows, notes, selection, ps2_setup = [], [], [], {}
    for name in args.names:
        sel = root / f"{name}_tracker" / "selection.json"
        if sel.exists():
            d = json.loads(sel.read_text())
            cands = ", ".join(f"{Path(c['run']).name.rsplit('_', 1)[-1]}/{c['weights']} {c['tracking_error_mean']:.3f}" for c in d["candidates"])
            pick = d["selected"]
            selection.append(f"| {name} | {Path(pick['run']).name.rsplit('_', 1)[-1]} ({pick['weights']}) | {pick['tracking_error_mean']:.3f} | "
                             f"{pick['tracking_error_max']:.2f} | {100 * pick['leaves_cone_rate']:.0f} % | {cands} |")
        meta = root / f"{name}_ps2" / "landing_meta.json"
        if meta.exists():
            m = json.loads(meta.read_text())
            ps2_setup[name] = f"PS2 trained on {m.get('device', '?')}, KKT solve {m.get('qp_kkt', '?')}"
        f = root / f"eval_{name}" / "summary.json"
        if not f.exists():
            notes.append(f"- `{name}`: no evaluation ({f} missing)")
            continue
        s = json.loads(f.read_text())
        for pol, r in s["results"].items():
            rows.append(f"| {name} | {pol} | {100 * r['leaves_safe_set_rate']:.0f} % | {r['min_safe_margin']:+.4f} | "
                        f"{100 * r['touchdown_on_platform_rate']:.0f} % | {100 * r['touchdown_beside_platform_rate']:.0f} % | "
                        f"{_f(r['touchdown_time_median'], '.2f')} | {r['tracking_error_mean']:.3f} | {r['max_slack']:.1e} | "
                        f"{100 * r['qp_fallback_rate']:.1f} % | {r['intervention_mean']:.3f} |")
        tr = "; ".join(f"{k}: {_f((v.get('total_steps') or 0) / 1e6, '.2f')}M steps, {_f((v.get('wall_time_sec') or 0) / 3600, '.2f')} h, "
                       f"{_f(v.get('steps_per_sec'), '.0f')} steps/s{' (wall-time budget hit)' if v.get('stopped_by_wall_budget') else ''}"
                       for k, v in s.get("training", {}).items())
        notes.append(f"- `{name}`: {s['starts']} starts, filter `{s['filter']}`, backup `{Path(s['backup']).name}`. Training — {tr}. "
                     f"{ps2_setup.get(name, 'no PS2 run')}. "
                     f"Figures: `eval_{name}/` ({', '.join(Path(v).name for v in s['figures'].values())}).")
    md += ["| reference | policy | leaves safe set | min margin [m] | on platform | beside platform | median touchdown [s] | "
           "tracking error [m] | max slack | QP fallbacks | intervention |",
           "|---|---|---|---|---|---|---|---|---|---|---|"] + rows + ["", *notes, "",
           "Policies: `tracker` = trained without the filter, run alone; `tracker+filter` = the same tracker through the "
           "landing filter at deployment; `ps2` = trained through the filter (warm-started from the tracker), run with it. "
           "Touchdown = height ≤ ζ at speed ≤ v_touch; on the platform = within r0 of the pad centre. Safety/intervention "
           "statistics cover the flight up to touchdown."]
    if selection:
        md += ["", "## Tracker selection", "",
               "Every checkpoint of every tracker seed, run alone from the same starts; the smallest mean position error wins "
               "and is the warm start of PS2. A tracker that follows its reference leaves the cone (the reference does).", "",
               "| reference | selected (seed, checkpoint) | tracking error [m] | max [m] | starts leaving the cone | "
               "all candidates: seed/checkpoint error [m] |", "|---|---|---|---|---|---|"] + selection
    (root / "REPORT.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
