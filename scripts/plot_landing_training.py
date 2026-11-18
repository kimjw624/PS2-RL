#!/usr/bin/env python
"""Training curves of the Phase-II landing runs, as a page that refreshes itself.

Reads ``metrics.jsonl`` of every run under ``--root`` and writes

    <root>/training.html            open it in a browser; it reloads every --watch seconds
    <root>/training_<mode>.png      one figure per mode (tracker, ps2): a column per reference,
                                    a line per run (seed); position error and return

The page also lists every run (step, speed, last and best evaluation) with the command that stops
it: ``touch <run>/STOP`` makes the trainer finish at the next chunk exactly as it would at the end
(final evaluation, final and best weights, summary), so a batch carries on with the next stage.

    python scripts/plot_landing_training.py --root outputs/landing_phase2             # once
    python scripts/plot_landing_training.py --root outputs/landing_phase2 --watch 30  # keep refreshing
"""

from __future__ import annotations

import argparse
import fnmatch
import html
import json
import os
from pathlib import Path
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

STOP_FILE_NAME = "STOP"  # the same name as in ps2rl/phase2_ps2/landing_ps2_entry.py


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True, help="directory holding the run directories")
    p.add_argument("--watch", type=float, default=0.0, help="refresh every this many seconds (0: write once and exit)")
    p.add_argument("--exclude", nargs="*", default=["*_old_*"],
                   help="run directories to leave out (shell patterns); default: results the batch moved aside")
    p.add_argument("--stale_after", type=float, default=600.0,
                   help="a run without a summary whose log has not changed for this many seconds is shown as not updating")
    return p.parse_args(argv)


def read_metrics(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:  # a line being written
            continue
    return rows


def find_runs(root: Path, stale_after: float, exclude: list[str]) -> list[dict]:
    runs = []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        meta_f, met_f = d / "landing_meta.json", d / "metrics.jsonl"
        if any(fnmatch.fnmatch(d.name, pat) for pat in exclude):
            continue
        if not (meta_f.exists() and met_f.exists()) or (d / "selection.json").exists():
            continue  # not a run, or the copy of a selected tracker
        try:
            meta = json.loads(meta_f.read_text())
        except json.JSONDecodeError:
            continue
        rows = read_metrics(met_f)
        summary = json.loads((d / "summary.json").read_text()) if (d / "summary.json").exists() else None
        if summary is not None:
            status = ("stopped on request" if summary.get("stopped_by_request") else
                      "stopped at its time budget" if summary.get("stopped_by_wall_budget") else "finished")
        elif (d / STOP_FILE_NAME).exists():
            status = "stopping"
        elif time.time() - met_f.stat().st_mtime > stale_after:
            status = "not updating"
        else:
            status = "running"
        total = meta.get("total_steps")
        if total is None and (d / "configs.json").exists():  # runs of before the entry recorded it at start
            total = json.loads((d / "configs.json").read_text())["sac"].get("total_steps")
        runs.append({"dir": d, "name": d.name, "mode": meta.get("mode", "run"), "reference": Path(meta.get("reference", "")).stem,
                     "rows": rows, "status": status, "summary": summary, "total_steps": total, "meta": meta})
    # shorten the reference names by the leading / trailing words they all share
    refs = sorted({r["reference"] for r in runs})
    if len(refs) > 1:
        words = [x.split("_") for x in refs]
        pre = len(os.path.commonprefix(words))
        suf = len(os.path.commonprefix([w[::-1] for w in words]))
        for r in runs:
            w = r["reference"].split("_")
            r["reference"] = "_".join(w[pre:len(w) - suf]) or r["reference"]
    return runs


def series(rows: list[dict], key: str) -> tuple[np.ndarray, np.ndarray]:
    pts = [(r["step"], r[key]) for r in rows if key in r and np.isfinite(r[key])]
    if not pts:
        return np.zeros(0), np.zeros(0)
    a = np.asarray(pts, dtype=float)
    return a[:, 0], a[:, 1]


def eval_points(rows: list[dict], key: str) -> tuple[np.ndarray, np.ndarray]:
    """One point per evaluation (the log repeats the last evaluation until the next one)."""
    seen, pts = set(), []
    for r in rows:
        k = r.get("eval/last_step")
        if k is None or k in seen or key not in r or not np.isfinite(r[key]):
            continue
        seen.add(k)
        pts.append((k, r[key]))
    a = np.asarray(pts, dtype=float).reshape(-1, 2)
    return a[:, 0], a[:, 1]


def plot_mode(runs: list[dict], mode: str, out: Path) -> None:
    refs = sorted({r["reference"] for r in runs})
    fig, axes = plt.subplots(2, len(refs), figsize=(max(4.2 * len(refs), 6.0), 6.6), squeeze=False, sharex="col")
    for j, ref in enumerate(refs):
        group = [r for r in runs if r["reference"] == ref]
        for i, r in enumerate(group):
            c = f"C{i % 10}"
            label = f"{r['name']} ({r['status']})"
            s, v = series(r["rows"], "episode/pos_error_norm")
            axes[0, j].plot(s / 1e6, v, color=c, lw=0.6, alpha=0.35)
            s, v = eval_points(r["rows"], "eval/pos_error_norm_mean")
            axes[0, j].plot(s / 1e6, v, color=c, lw=1.6, marker="o", ms=2.5, label=label)
            s, v = series(r["rows"], "episode/return")
            axes[1, j].plot(s / 1e6, -v, color=c, lw=0.6, alpha=0.35)
            s, v = eval_points(r["rows"], "eval/return_mean")
            axes[1, j].plot(s / 1e6, -v, color=c, lw=1.6, marker="o", ms=2.5)
        axes[0, j].set_title(ref)
        if group:
            axes[0, j].legend(fontsize=7, loc="upper right")
        axes[1, j].set_xlabel("environment steps [million]")
        for ax in axes[:, j]:
            if any(np.any(np.asarray(line.get_ydata(), dtype=float) > 0.0) for line in ax.get_lines()):
                ax.set_yscale("log")  # needs something positive to show (a run that has not logged yet has nothing)
            ax.grid(True, which="both", alpha=0.25)
    axes[0, 0].set_ylabel("mean position error over an episode [m]")
    axes[1, 0].set_ylabel("−return (lower is better)")
    fig.suptitle(f"{mode}: evaluation (thick, deterministic policy) and training episodes (thin)   "
                 f"{time.strftime('%Y-%m-%d %H:%M:%S')}", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def table_row(r: dict, root: Path) -> str:
    rows = r["rows"]
    step = rows[-1]["step"] if rows else 0
    rate = rows[-1].get("perf/steps_per_sec") if rows else None
    s, v = eval_points(rows, "eval/pos_error_norm_mean")
    last = f"{v[-1]:.3f}" if len(v) else "–"
    best = f"{v.min():.3f} at {s[int(v.argmin())] / 1e6:.2f}M" if len(v) else "–"
    total = r["total_steps"]
    eta = "–"
    if r["status"] == "running" and rate and total:
        eta = f"{max(0.0, (total - step) / rate) / 3600:.2f} h"
    cmd = f"touch {root / r['name'] / STOP_FILE_NAME}" if r["status"] == "running" else ""
    cells = [r["name"], r["mode"], r["status"], f"{step / 1e6:.2f}M" + (f" / {total / 1e6:.2f}M" if total else ""),
             f"{rate:.0f}" if rate else "–", eta, last, best]
    return "<tr>" + "".join(f"<td>{html.escape(str(c))}</td>" for c in cells) + f"<td><code>{html.escape(cmd)}</code></td></tr>"


def write_page(root: Path, runs: list[dict], modes: list[str], refresh: float) -> None:
    stamp = int(time.time())
    head = "".join(f"<th>{h}</th>" for h in ("run", "mode", "status", "steps", "steps/s", "time left", "last eval error [m]",
                                             "best eval error [m]", "to stop it"))
    body = "\n".join(table_row(r, root) for r in runs)
    imgs = "\n".join(f'<h2>{html.escape(m)}</h2><img src="training_{m}.png?t={stamp}" style="max-width:100%">' for m in modes)
    meta = f'<meta http-equiv="refresh" content="{max(5, int(refresh))}">' if refresh > 0 else ""
    page = f"""<!doctype html><html><head><meta charset="utf-8">{meta}<title>landing training</title>
<style>body{{font-family:sans-serif;margin:16px}} table{{border-collapse:collapse;font-size:13px}}
td,th{{border:1px solid #bbb;padding:3px 8px;text-align:left}} code{{font-size:12px}}</style></head><body>
<h1>Phase-II landing training</h1>
<p>Updated {time.strftime('%Y-%m-%d %H:%M:%S')}{f' (reloads every {int(refresh)} s)' if refresh > 0 else ''}.
A stopped run finishes normally: final evaluation, final and best weights, summary. The best weights are the
evaluation with the highest return so far, so stopping late costs nothing.</p>
<table><tr>{head}</tr>
{body}
</table>
{imgs}
</body></html>"""
    tmp = root / "training.html.tmp"
    tmp.write_text(page, encoding="utf-8")
    tmp.replace(root / "training.html")


def refresh_once(root: Path, stale_after: float, refresh: float, exclude: list[str]) -> int:
    runs = find_runs(root, stale_after, exclude)
    modes = sorted({r["mode"] for r in runs}, key=lambda m: (m != "tracker", m))
    for m in modes:
        tmp = root / f"training_{m}.tmp.png"
        plot_mode([r for r in runs if r["mode"] == m], m, tmp)
        tmp.replace(root / f"training_{m}.png")
    write_page(root, runs, modes, refresh)
    return len(runs)


def main(argv=None):
    args = parse_args(argv)
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    while True:
        try:
            n = refresh_once(root, args.stale_after, args.watch, list(args.exclude))
        except Exception as e:  # a file caught mid-write; try again at the next refresh
            if args.watch <= 0:
                raise
            print(f"[curves] skipped a refresh: {e!r}", flush=True)
        else:
            if args.watch <= 0:
                print(f"[curves] {n} runs -> {root / 'training.html'}")
                return
        time.sleep(args.watch)


if __name__ == "__main__":
    main()
