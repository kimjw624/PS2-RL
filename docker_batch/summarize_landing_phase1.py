"""Aggregate landing Phase-I runs (SAC vs TD3 over seeds) into a table, a CSV and a figure.

    python docker_batch/summarize_landing_phase1.py --root outputs/landing_phase1_compare

Reads every ``<root>/<backbone>_seed<k>/summary.json`` (+ history.json/configs.json)
written by ``scripts/train_phase1_landing.py``. Model selection is on the val split,
reported numbers are the test split evaluated at the selected checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re

import numpy as np

REGIONS = ("general", "edge", "shell")
COLORS = {"sac": "#2a78d6", "td3": "#eb6834"}  # categorical slots 1-2 (validated palette)
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"


def load_runs(root: Path) -> list[dict]:
    runs = []
    for summ_path in sorted(root.glob("*/summary.json")):
        run_dir = summ_path.parent
        m = re.match(r"(?P<bb>[a-z0-9]+)_seed(?P<seed>\d+)$", run_dir.name)
        with open(summ_path, encoding="utf-8") as f:
            summ = json.load(f)
        bb = summ.get("sa_backbone", m.group("bb") if m else "?")
        seed = int(summ.get("seed", m.group("seed") if m else -1))
        hist, eval_every = {}, None
        if (run_dir / "history.json").exists():
            with open(run_dir / "history.json", encoding="utf-8") as f:
                hist = json.load(f)
        if (run_dir / "configs.json").exists():
            with open(run_dir / "configs.json", encoding="utf-8") as f:
                eval_every = int(json.load(f)["backup_ra"]["eval_every"])
        runs.append({"dir": run_dir, "backbone": bb, "seed": seed, "summary": summ, "history": hist, "eval_every": eval_every})
    return runs


def run_row(run: dict) -> dict:
    s = run["summary"]
    t = s.get("test_at_best", s.get("test_at_final"))
    row = {
        "backbone": run["backbone"],
        "seed": run["seed"],
        "backend": s.get("jax_backend", "?"),
        "wall_min": s["wall_time_sec"] / 60.0,
        "best_step": s["best_eval_step"],
        "val_mu_best": s["best_val"]["mu_weighted"],
        "test_mu": t["mu_weighted"],
        "test_crash": t["crash_rate"],
        "untrained_mu": s["untrained_val"]["mu_weighted"],
        "final_val_mu": s["final_val"]["mu_weighted"],
        "regression": s["best_val"]["mu_weighted"] - s["final_val"]["mu_weighted"],
    }
    for r in REGIONS:
        row[f"test_m_{r}"] = t[r]["m_hat"]
        row[f"test_m_{r}_hw"] = t[r]["hoeffding_halfwidth"]
    return row


def fmt(mean: float, std: float, n: int) -> str:
    return f"{mean:.3f}" if n < 2 else f"{mean:.3f} ± {std:.3f}"


def print_tables(rows: list[dict]) -> None:
    rows = sorted(rows, key=lambda r: (r["backbone"], r["seed"]))
    print("\nPer run (test split at the val-selected checkpoint):")
    hdr = f"{'backbone':8} {'seed':>4} {'mu_w':>6} {'general':>8} {'edge':>6} {'shell':>6} {'crash':>6} {'best@':>9} {'final val':>9} {'min':>6} {'dev':>4}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(
            f"{r['backbone']:8} {r['seed']:>4} {r['test_mu']:6.3f} {r['test_m_general']:8.3f} {r['test_m_edge']:6.3f} "
            f"{r['test_m_shell']:6.3f} {r['test_crash']:6.3f} {r['best_step']:>9} {r['final_val_mu']:9.3f} {r['wall_min']:6.1f} {r['backend']:>4}"
        )
    hw = rows[0]["test_m_edge_hw"] if rows else float("nan")
    print(f"(per-region Hoeffding half-width at 95%: ~{hw:.3f} for the edge set; seed-to-seed spread is usually larger)")

    print("\nBy backbone (mean ± std over seeds):")
    hdr = f"{'backbone':8} {'n':>2} {'test mu_w':>15} {'edge':>15} {'general':>15} {'shell':>15} {'crash':>15} {'best-final val':>15}"
    print(hdr)
    print("-" * len(hdr))
    for bb in sorted({r["backbone"] for r in rows}):
        g = [r for r in rows if r["backbone"] == bb]
        n = len(g)

        def ms(key: str) -> str:
            v = np.asarray([r[key] for r in g], dtype=float)
            return fmt(float(v.mean()), float(v.std(ddof=1)) if n > 1 else 0.0, n)

        print(
            f"{bb:8} {n:>2} {ms('test_mu'):>15} {ms('test_m_edge'):>15} {ms('test_m_general'):>15} "
            f"{ms('test_m_shell'):>15} {ms('test_crash'):>15} {ms('regression'):>15}"
        )
    print("'best-final val' = best val mu_w minus final val mu_w: large values mean training regressed late (collapse).")


def curves(runs: list[dict], key: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Per backbone: (steps, values[n_seeds, n_evals]) truncated to the shortest run; step 0 = untrained."""
    out = {}
    for bb in sorted({r["backbone"] for r in runs}):
        series, steps = [], None
        for r in runs:
            if r["backbone"] != bb or not r["history"].get(key) or not r["eval_every"]:
                continue
            untrained = r["summary"]["untrained_val"]
            v0 = untrained["mu_weighted"] if key == "eval_mu_weighted" else untrained[key.replace("eval_m_", "")]["m_hat"]
            series.append(np.asarray([v0, *r["history"][key]], dtype=float))
            steps = r["eval_every"]
        if not series:
            continue
        n = min(len(s) for s in series)
        vals = np.stack([s[:n] for s in series])
        out[bb] = (np.arange(n) * steps, vals)
    return out


def plot(runs: list[dict], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [("eval_mu_weighted", "Weighted recoverability μ̂ (val)"), ("eval_m_edge", "Low-altitude cone edge m̂_edge (val)")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), facecolor=SURFACE)
    for ax, (key, title) in zip(axes, panels):
        ax.set_facecolor(SURFACE)
        data = curves(runs, key)
        ends = []
        for bb, (x, v) in data.items():
            c = COLORS.get(bb, TEXT_2)
            mean = v.mean(axis=0)
            if v.shape[0] > 1:
                ax.fill_between(x / 1e6, v.min(axis=0), v.max(axis=0), color=c, alpha=0.14, linewidth=0)
            ax.plot(x / 1e6, mean, color=c, linewidth=2.0, label=f"{bb.upper()} (n={v.shape[0]})")
            ends.append((mean[-1], bb, x[-1] / 1e6))
        # End labels, nudged apart when the curves finish close together.
        placed: list[float] = []
        for y, bb, x_end in sorted(ends):
            yy = y
            for q in placed:
                if abs(yy - q) < 0.06:
                    yy = q + 0.06
            placed.append(yy)
            ax.annotate(f"{bb.upper()} {y:.3f}", (x_end, y), xytext=(x_end + 0.1, yy), textcoords="data", va="center",
                        fontsize=9, color=TEXT, annotation_clip=False)
        ax.set_title(title, fontsize=11, color=TEXT, loc="left")
        ax.set_xlabel("environment steps (millions)", fontsize=9, color=TEXT_2)
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, color=GRID, linewidth=0.8)
        ax.tick_params(colors=TEXT_2, labelsize=8.5)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.margins(x=0.12)
    axes[0].legend(frameon=False, fontsize=9, loc="lower right", labelcolor=TEXT)
    fig.text(0.01, 0.01, "line = mean over seeds, band = min–max across seeds", fontsize=8, color=TEXT_2)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=str, default="outputs/landing_phase1_compare")
    args = p.parse_args(argv)
    root = Path(args.root)
    runs = load_runs(root)
    if not runs:
        raise SystemExit(f"no finished runs (summary.json) under {root}")
    rows = [run_row(r) for r in runs]
    print_tables(rows)
    with open(root / "runs.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: (r["backbone"], r["seed"])))
    plot(runs, root / "learning_curves.png")
    print(f"\nwrote {root / 'runs.csv'} and {root / 'learning_curves.png'}")


if __name__ == "__main__":
    main()
