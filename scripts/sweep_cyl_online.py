#!/usr/bin/env python
"""Sweep the online filter's compute knobs against safety and progress (closed loop, realistic plant). numpy only.

    python scripts/sweep_cyl_online.py --stage all --out results/cyl_online/sweep.json
    python scripts/sweep_cyl_online.py --stage A --episodes 40 --out /tmp/sweep_a.json        # quicker

Knobs (FilterConfig): the backup (hover LQR or the Phase-I actor of width 32 / 64 / 128 / 256), the blends (2: 1, 0;
3: 1, .5, 0; 5: 1, .75, .5, .25, 0), the backup family (the learned backup alone, or + the hover LQR), the backup
rollout after the window (horizon), and the decision period (control steps per decision; the plan in between is
re-evaluated every step from the state).

Stages (one factor at a time around blends 3 / learned alone / period 1, then the combinations that matter; ``list``
runs the settings given by --settings, e.g. a confirmation with more episodes and another seed):
  A   backup x horizon (0.5, 0.75, 1.0, 1.5 s)
  B   blends x family, for widths 32 and 64 (horizon 1.0)
  C   period (1, 2, 5) x horizon (0.5, 1.0), width 64

Scenario sets, all with the realistic plant (sim.PlantConfig: delay, rate / thrust lag, gain errors, estimate noise; no
wind) and a disturbance observer (it absorbs the thrust-map error):
  stress  random scenes (2..4 cylinders), a short-sighted nominal (heading search keeping only 0.45 m from the surfaces,
          3 m look-ahead, 1.2 m/s) and 4 m sensing: the filter has to step in near every cylinder
  blind   the r 0.3 m deployment course (sim.R03_COURSE: from rest, the first cylinder 1.5 m ahead; "one": a cylinder
          on the line; "two": the slalom, the second cylinder 3.5 m on, across the way back) with a straight-line
          nominal (no search): the filter must not let the vehicle hit a cylinder
  final   the same course with the deployment nominal (heading search, 1.0 m margin, 1.0 m/s, 6 m sensing)
  (the demo layouts, r 0.1 m, are flown by scripts/eval_cyl_final.py)

Per setting and set: success / contact / timeout rates, minimum clearance, mean blend and intervention rate, how often
no certificate was found, and the rollout steps an onboard filter runs per decision (mean, 99th percentile, max); the
report (scripts/report_cyl_online.py) turns steps into milliseconds with the timing fit of bench_cyl_filter.py.
"""

from __future__ import annotations

import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from ps2rl.online.filter import FilterConfig, OnlineFilter  # noqa: E402
from ps2rl.online.model import load_backup  # noqa: E402
from ps2rl.online.nominal import Nominal, NominalConfig  # noqa: E402
from ps2rl.online.sim import R03_COURSE, PlantConfig, SceneConfig, make_scenes, run_episodes  # noqa: E402

LAMBDA_SETS = {2: (1.0, 0.0), 3: (1.0, 0.5, 0.0), 5: (1.0, 0.75, 0.5, 0.25, 0.0)}
WIDTHS = (32, 64, 128, 256)


def scenario_sets(n: int) -> dict:
    """name -> list of (label, SceneConfig, NominalConfig, n episodes)."""
    stress_nom = NominalConfig(cruise=1.2, margin=0.45, lookahead=3.0)
    deploy_nom = NominalConfig()
    straight = NominalConfig(search=False)
    return {
        "stress": [("random", SceneConfig(kind="random", sensing_range=4.0, start_speed=(0.0, 1.2)), stress_nom, n)],
        # the r 0.3 m course the sweep was run on (the demo layouts are smaller: scripts/eval_cyl_final.py)
        "blind": [("one", SceneConfig(kind="one", episode_sec=25.0, start_speed=(0.0, 0.0), **R03_COURSE), straight,
                   max(n // 4, 8)),
                  ("two", SceneConfig(kind="two", episode_sec=25.0, start_speed=(0.0, 0.0), **R03_COURSE), straight,
                   max(n // 4, 8))],
        "final": [("one", SceneConfig(kind="one", start_speed=(0.0, 0.0), **R03_COURSE), deploy_nom, max(n // 4, 8)),
                  ("two", SceneConfig(kind="two", start_speed=(0.0, 0.0), **R03_COURSE), deploy_nom, max(n // 4, 8))],
    }


def settings(stage: str) -> list[dict]:
    out = []
    if stage in ("A", "all"):
        for b in ("lqr",) + WIDTHS:
            for T in (0.5, 0.75, 1.0, 1.5):
                out.append({"backup": b, "blends": 3, "family": 1, "horizon": T, "period": 1})
    if stage in ("B", "all"):
        for b in (32, 64):
            for nl in (2, 3, 5):
                for fam in (1, 2):
                    out.append({"backup": b, "blends": nl, "family": fam, "horizon": 1.0, "period": 1})
    if stage in ("C", "all"):
        for P in (1, 2, 5):
            for T in (0.5, 1.0):
                out.append({"backup": 64, "blends": 3, "family": 1, "horizon": T, "period": P})
    seen, uniq = set(), []
    for s in out:
        k = tuple(sorted(s.items()))
        if k not in seen:
            seen.add(k)
            uniq.append(s)
    return uniq


def key(s: dict) -> str:
    return f"{s['backup']}|{s['blends']}|{s['family']}|{s['horizon']:g}|{s['period']}"


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backups", default=str(ROOT / "checkpoints/cyl_online"),
                   help="directory with the exports lqr/ and td3_w<width>_seed<seed>/")
    p.add_argument("--seed_tag", default="seed0")
    p.add_argument("--stage", default="all", choices=("A", "B", "C", "all", "list"))
    p.add_argument("--settings", default="", help="stage list: settings 'backup|blends|family|horizon|period' separated by "
                                                  "commas, e.g. '32|3|1|1.5|1,lqr|3|1|1|1' (a confirmation run)")
    p.add_argument("--episodes", type=int, default=100, help="stress episodes per setting (blind / final: a quarter each)")
    p.add_argument("--margin", type=float, default=0.1, help="filter margin added to the drone radius [m]")
    p.add_argument("--sets", default="stress,blind,final")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--backend", default="auto", choices=("auto", "numpy", "numba"),
                   help="rollout backend (identical decisions; numba is faster)")
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    root = Path(a.backups)
    models = {"lqr": load_backup(root / "lqr")}
    for w in WIDTHS:
        d = root / f"td3_w{w}_{a.seed_tag}"
        if (d / "backup.json").exists():
            models[w] = load_backup(d)
    out_path = Path(a.out)
    results = json.loads(out_path.read_text()) if out_path.exists() else {"runs": {}}
    results["meta"] = {"episodes": a.episodes, "margin": a.margin, "seed": a.seed, "sets": a.sets,
                       "plant": PlantConfig().__dict__, "backups": {str(k): str(root) for k in models}}
    sets = scenario_sets(a.episodes)
    if a.stage == "list":
        todo = []
        for item in a.settings.split(","):
            b, nl, fam, T, P = item.split("|")
            todo.append({"backup": b if b == "lqr" else int(b), "blends": int(nl), "family": int(fam), "horizon": float(T),
                         "period": int(P)})
    else:
        todo = settings(a.stage)
    todo = [s for s in todo if s["backup"] in models]
    print(f"[sweep] {len(todo)} settings x sets {a.sets} -> {out_path}", flush=True)
    for s in todo:
        k = key(s)
        if k in results["runs"]:
            continue
        m, actor, meta = models[s["backup"]]
        backups = ("lqr",) if s["backup"] == "lqr" else (("learned",) if s["family"] == 1 else ("learned", "lqr"))
        fc = FilterConfig(lambdas=LAMBDA_SETS[s["blends"]], backups=backups, horizon=s["horizon"], period=s["period"],
                          margin=a.margin, obstacle_range=float((meta.get("region") or {}).get("dist_max", 5.0)),
                          backend=a.backend)
        rec = {"setting": s, "filter": fc.as_dict(), "rows": fc.n_candidates + 1,
               "width": 0 if s["backup"] == "lqr" else int(s["backup"])}
        t0 = time.time()
        for name in a.sets.split(","):
            for label, sc, nc, n in sets[name]:
                nom = Nominal(m, nc)
                eps = make_scenes(sc, n, seed=a.seed + 101 * len(label))
                f = OnlineFilter(m, actor, fc, nom)
                r = run_episodes(m, eps, sc, PlantConfig(), nom, f, seed=a.seed + 7)
                r.pop("per_episode", None)
                rec[f"{name}/{label}"] = r
        rec["wall_s"] = time.time() - t0
        results["runs"][k] = rec
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=1, default=float))
        st = rec.get("stress/random", {})
        bl1, bl2 = rec.get("blind/one", {}), rec.get("blind/two", {})
        fi1, fi2 = rec.get("final/one", {}), rec.get("final/two", {})
        print(f"[sweep] {k:22} stress succ {st.get('success', np.nan):.2f} contact {st.get('contact', np.nan):.2f} "
              f"none {st.get('mode_none', np.nan):.3f} steps {st.get('steps_mean', np.nan):.1f}/{st.get('steps_p99', np.nan):.0f}"
              f"/{st.get('steps_max', -1)} | blind contact {bl1.get('contact', np.nan):.2f}/{bl2.get('contact', np.nan):.2f} "
              f"| final succ {fi1.get('success', np.nan):.2f}/{fi2.get('success', np.nan):.2f}"
              f" [{rec['wall_s']:.0f} s]", flush=True)


if __name__ == "__main__":
    main()
