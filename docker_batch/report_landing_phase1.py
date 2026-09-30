"""Generate the formal landing Phase-I report (PDF) from the comparison outputs.

    # inside the container, from ~/ws_shared/PS2-RL (CPU is fine, ~4 min):
    python docker_batch/report_landing_phase1.py --root outputs/landing_phase1_compare

    # choose the checkpoints explicitly
    python docker_batch/report_landing_phase1.py --root outputs/landing_phase1_compare \
        --run td3_seed0 --compare_run sac_seed2

Stages: data (simulations, baselines, viability, slices, case studies; cached in
``<out>/data.pkl``), figures (``<out>/figures``), tables and macros (``<out>/tables``,
``<out>/numbers.tex``), and the LaTeX document, compiled with latexmk when available.
Re-running reuses the cache unless ``--recompute`` is given.

Defaults: ``--run`` is the run with the best validation score; ``--compare_run`` is the
best-validation run of the other backbone.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from report_lib import report_data as rd  # noqa: E402
from report_lib import report_figures as rf  # noqa: E402
from report_lib import report_tables as rt  # noqa: E402

TEMPLATE = HERE / "report_lib" / "landing_phase1_report.tex"


def best_runs(root: Path) -> tuple[str, str]:
    runs = rd.list_runs(root)
    if not runs:
        raise SystemExit(f"no runs with summary.json under {root}")
    val = {r: json.load(open(root / r / "summary.json"))["best_val"]["mu_weighted"] for r in runs}
    run = max(runs, key=val.get)
    bb = json.load(open(root / run / "summary.json"))["sa_backbone"]
    others = [r for r in runs if json.load(open(root / r / "summary.json"))["sa_backbone"] != bb]
    return run, (max(others, key=val.get) if others else run)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=str, default="outputs/landing_phase1_compare")
    p.add_argument("--out", type=str, default="")
    p.add_argument("--run", type=str, default="")
    p.add_argument("--compare_run", type=str, default="")
    p.add_argument("--recompute", action="store_true")
    p.add_argument("--no_pdf", action="store_true")
    args = p.parse_args(argv)

    root = Path(args.root)
    out = Path(args.out) if args.out else root.parent / "landing_phase1_report"
    run_default, cmp_default = best_runs(root)
    run = args.run or run_default
    cmp_run = args.compare_run or cmp_default
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    def log(msg: str) -> None:
        print(f"[{time.time() - t0:6.1f}s] {msg}", flush=True)

    cache = out / "data.pkl"
    if cache.exists() and not args.recompute:
        data = rd.load(cache)
        if data.get("run") != run or data.get("compare_run") != cmp_run or data.get("root") != str(root):
            log("cache is for different runs; recomputing")
            data = None
    else:
        data = None
    if data is None:
        log(f"computing report data (run={run}, compare={cmp_run}, jax platform={os.environ.get('JAX_PLATFORMS')})")
        data = rd.compute(root, run, cmp_run, log=log)
        rd.save(data, cache)
    log("figures")
    rf.make_all(data, out / "figures")
    log("tables and macros")
    rt.write(data, out)
    tex = out / "landing_phase1_report.tex"
    shutil.copy(TEMPLATE, tex)
    if args.no_pdf:
        log(f"done (LaTeX source in {out})")
        return
    if shutil.which("latexmk") is None:
        log(f"latexmk not found; LaTeX source is in {out}. Compile with: cd {out} && latexmk -pdf landing_phase1_report.tex")
        return
    log("compiling PDF")
    r = subprocess.run(["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", tex.name],
                       cwd=out, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-4000:])
        raise SystemExit("latexmk failed; see the log above")
    log(f"done: {out / 'landing_phase1_report.pdf'}")


if __name__ == "__main__":
    main()
