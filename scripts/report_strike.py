#!/usr/bin/env python
"""Tables and figures for the strike report from the final pipeline (docker_batch/run_strike_final.sh).

Reads, for every seed s:
  <root>/p1/td3_a0p2_seed<s>/summary.json                      Phase-I backup (recovered fraction per region)
  <root>/residual/striker_rs0p1_s<s>/history.json               residual striker without the filter (training curve)
  <root>/ps2/striker_rs0p1_s<s>/history.json                    PS2 striker (training curve)
  <root>/eval/s<s>/<run>/strike_filter_episodes.json            closed-loop evaluations (scripts/eval_strike_filter.py)

Writes to <root>/report/:
  results.md / results.tex        main table: nominal x filter, strike / contact / time-up (mean +- std over seeds, pooled
                                  Clopper-Pearson 95 % interval), strike time, kept fraction of the nominal, lost certificates
  ablation.md / ablation.tex      the deployed filter with one component removed (PS2 striker)
  horizon.md                      the same rows with a 10 s time limit (from the per-episode records of the 15 s runs)
  phase1.md / phase1.tex          Phase-I recovered fraction per region, learned vs hover LQR
  fig_training.png                PS2 / residual training evaluations (mean and range over seeds)
  fig_outcomes.png                outcome bars of the main table

    python scripts/report_strike.py --root outputs/strike_final --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import beta

OUTCOMES = ("strike", "crash_obstacle", "crash_floor", "crash_chaser", "out_of_arena", "timeout")

# (group, filter label, eval run, mode); the run names are the ones docker_batch/run_strike_final_eval.sh writes
MAIN = [
    ("Pursuit (6 m/s)", "none", "pursuit_static", "none"),
    ("Pursuit (6 m/s)", "static chaser ball", "pursuit_static", "learned"),
    ("Pursuit (6 m/s)", "guidance set, 1 escape", "pursuit_guid1", "learned"),
    ("Pursuit (6 m/s)", "bCBF", "pursuit_bcbf", "learned"),
    ("Pursuit (6 m/s)", "ours", "pursuit_final", "learned"),
    ("Residual SAC, no PS2", "none", "residual_final", "none"),
    ("Residual SAC, no PS2", "ours", "residual_final", "learned"),
    ("PS2 (ours)", "none", "ps2_final", "none"),
    ("PS2 (ours)", "guidance set, 1 escape", "ps2_guid1", "learned"),
    ("PS2 (ours)", "bCBF", "ps2_bcbf", "learned"),
    ("PS2 (ours)", "ours", "ps2_final", "learned"),
]
ABLATION = [
    ("deployed filter", "ps2_final", "learned"),
    ("hover-LQR backup instead of the learned one", "ps2_final", "lqr"),
    ("no escape variants (1 escape)", "ps2_guid1", "learned"),
    ("no committed chaser prediction", "ps2_nocp", "learned"),
    ("no hover latch", "ps2_nolatch", "learned"),
    ("static chaser ball instead of h_c", "ps2_static", "learned"),
]


def load_episodes(root: Path, seed: int, run: str, mode: str):
    p = root / "eval" / f"s{seed}" / run / "strike_filter_episodes.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    if mode not in d["modes"]:
        return None
    e = {k: np.asarray(v) for k, v in d["modes"][mode].items()}
    e["dt"], e["max_steps"] = float(d["dt"]), int(d["max_steps"])
    return e


def truncate(e: dict, t_max: float | None) -> dict:
    """Outcomes with a shorter time limit: episodes still running at t_max become time-ups."""
    if t_max is None:
        return e
    k = int(round(t_max / e["dt"]))
    e = dict(e)
    late = e["end_step"] > k
    e["outcome"] = np.where(late, OUTCOMES.index("timeout"), e["outcome"])
    e["strike_step"] = np.where(late, 0, e["strike_step"])
    return e


def seed_metrics(e: dict) -> dict:
    o = e["outcome"]
    strike = o == 0
    contact = (o == 1) | (o == 2) | (o == 3)
    return {"n": int(o.size), "strike": int(strike.sum()), "contact": int(contact.sum()),
            "chaser": int((o == 3).sum()), "obstacle": int((o == 1).sum()), "floor": int((o == 2).sum()),
            "out": int((o == 4).sum()), "timeout": int((o == 5).sum()),
            "t_strike": float((e["strike_step"][strike] * e["dt"]).mean()) if strike.any() else float("nan"),
            "kept": float(np.mean(e["kept_fraction"])),
            "lost_cert": float(np.mean(e["no_cert_steps"] > 0))}


def cp_interval(k: int, n: int, a: float = 0.05) -> tuple[float, float]:
    lo = 0.0 if k == 0 else float(beta.ppf(a / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - a / 2, k + 1, n - k))
    return lo, hi


def aggregate(root: Path, seeds, run: str, mode: str, t_max: float | None = None):
    per = []
    for s in seeds:
        e = load_episodes(root, s, run, mode)
        if e is not None:
            per.append(seed_metrics(truncate(e, t_max)))
    if not per:
        return None
    n = sum(p["n"] for p in per)
    out = {"seeds": len(per), "n": n}
    for key in ("strike", "contact", "timeout", "chaser", "obstacle"):
        rates = np.array([p[key] / p["n"] for p in per])
        k = sum(p[key] for p in per)
        out[key] = (float(rates.mean()), float(rates.std()), k, cp_interval(k, n))
    for key in ("t_strike", "kept", "lost_cert"):
        v = np.array([p[key] for p in per])
        out[key] = (float(np.nanmean(v)), float(np.nanstd(v)))
    return out


def fmt_rate(v, tex=False):
    m, s, k, (lo, hi) = v
    pm = r"\pm" if tex else "±"
    if tex:
        return f"${m:.3f}{pm}{s:.3f}$ {{\\scriptsize [{lo:.3f}, {hi:.3f}]}}"
    return f"{m:.3f} {pm} {s:.3f} [{lo:.3f}, {hi:.3f}]"


def main_table(root, seeds, rows, t_max=None, tex=False, ablation=False):
    if tex:
        head = (r"\begin{tabular}{" + ("l" if ablation else "ll") + "cccccc}\\toprule" + "\n"
                + (r"variant" if ablation else r"nominal & filter")
                + r" & strike & contact & time-up & $t_{\rm strike}$ [s] & kept & lost cert.\\\midrule")
        lines = [head]
    else:
        cols = "| variant |" if ablation else "| nominal | filter |"
        lines = [cols + " strike | contact | time-up | strike time [s] | kept | lost certificate | seeds x episodes |",
                 "|---" * (6 + (1 if ablation else 2) + 1) + "|"]
    last_group = None
    for r in rows:
        if ablation:
            label, run, mode = r
            group = None
        else:
            group, label, run, mode = r
        a = aggregate(root, seeds, run, mode, t_max)
        if a is None:
            continue
        cells = [fmt_rate(a["strike"], tex), fmt_rate(a["contact"], tex), fmt_rate(a["timeout"], tex),
                 f"{a['t_strike'][0]:.2f}", f"{a['kept'][0]:.2f}", f"{a['lost_cert'][0]:.3f}"]
        if tex:
            if not ablation and last_group is not None and group != last_group:
                lines.append(r"\midrule")
            first = [label] if ablation else [group if group != last_group else "", label]
            lines.append(" & ".join(first + cells) + r"\\")
        else:
            first = [label] if ablation else [group, label]
            lines.append("| " + " | ".join(first + cells + [f"{a['seeds']} x {a['n'] // max(a['seeds'], 1)}"]) + " |")
        last_group = group
    if tex:
        lines.append(r"\bottomrule\end{tabular}")
    return "\n".join(lines)


def phase1_table(root, seeds, tex=False):
    regs = None
    learned, lqr = [], []
    for s in seeds:
        p = root / "p1" / f"td3_a0p2_seed{s}" / "summary.json"
        if not p.exists():
            continue
        sm = json.loads(p.read_text())
        t, l = sm["test_at_best"], sm["lqr_test"]
        regs = regs or [k for k in t if isinstance(t[k], dict) and "m_hat" in t[k]]
        learned.append([t[r]["m_hat"] for r in regs] + [t["mu_weighted"]])
        lqr.append([l[r]["m_hat"] for r in regs] + [l["mu_weighted"]])
    if not learned:
        return ""
    L, Q = np.array(learned), np.array(lqr)
    names = regs + ["weighted"]
    if tex:
        out = [r"\begin{tabular}{l" + "c" * len(names) + r"}\toprule", "backup & " + " & ".join(names) + r"\\\midrule",
               "hover LQR & " + " & ".join(f"{v:.3f}" for v in Q.mean(0)) + r"\\",
               f"learned ($n={len(L)}$ seeds) & " + " & ".join(f"${m:.3f}\\pm{s:.3f}$" for m, s in zip(L.mean(0), L.std(0)))
               + r"\\\bottomrule\end{tabular}"]
    else:
        out = ["| backup | " + " | ".join(names) + " |", "|---" * (len(names) + 1) + "|",
               "| hover LQR | " + " | ".join(f"{v:.3f}" for v in Q.mean(0)) + " |",
               f"| learned ({len(L)} seeds) | " + " | ".join(f"{m:.3f} ± {s:.3f}" for m, s in zip(L.mean(0), L.std(0))) + " |"]
    return "\n".join(out)


def training_figure(root, seeds, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    for ax, stage, title in ((axes[0], "residual", "residual striker, no filter"), (axes[1], "ps2", "PS2 striker, through the filter")):
        curves = []
        for s in seeds:
            p = root / stage / f"striker_rs0p1_s{s}" / "history.json"
            if p.exists():
                h = json.loads(p.read_text())
                curves.append((np.asarray(h["eval/step"]), np.asarray(h["eval/strike_rate"]), np.asarray(h["eval/crash_rate"])))
        if not curves:
            ax.set_visible(False)
            continue
        n = min(len(c[0]) for c in curves)
        x = curves[0][0][:n] / 1e6
        for j, (lab, col) in enumerate((("strike", "#2a78d6"), ("crash", "#c4553f"))):
            Y = np.stack([c[1 + j][:n] for c in curves])
            ax.plot(x, Y.mean(0), color=col, lw=1.8, label=f"{lab} (mean of {len(curves)})")
            ax.fill_between(x, Y.min(0), Y.max(0), color=col, alpha=0.18, lw=0)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("environment steps [M]")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="center right")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def outcome_figure(root, seeds, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = [(g, l, aggregate(root, seeds, r, m)) for g, l, r, m in MAIN]
    rows = [(g, l, a) for g, l, a in rows if a is not None]
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(10, 0.42 * len(rows) + 1.0))
    y = np.arange(len(rows))[::-1]
    left = np.zeros(len(rows))
    for key, col, lab in (("strike", "#2a78d6", "strike"), ("contact", "#3d3c39", "contact"), ("timeout", "#c99a06", "time-up")):
        v = np.array([a[key][0] for _, _, a in rows])
        ax.barh(y, v, left=left, color=col, label=lab, height=0.7)
        left = left + v
    ax.set_yticks(y)
    ax.set_yticklabels([f"{g}: {l}" for g, l, _ in rows], fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_xlabel("fraction of episodes (mean over seeds)")
    ax.legend(fontsize=8, ncol=3, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="outputs/strike_final")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--short_horizon", type=float, default=10.0, help="also report with this time limit [s] (horizon.md)")
    a = p.parse_args(argv)
    root = Path(a.root)
    out = root / "report"
    out.mkdir(parents=True, exist_ok=True)
    note = (f"Seeds {a.seeds}; rates are mean ± std over seeds, [pooled Clopper-Pearson 95 % interval]; contact = left the "
            "safe set (obstacle, floor, chaser); kept = mean fraction of the nominal command kept; lost certificate = "
            "fraction of episodes with at least one step without a certified plan.\n\n")
    (out / "results.md").write_text(note + main_table(root, a.seeds, MAIN) + "\n")
    (out / "results.tex").write_text(main_table(root, a.seeds, MAIN, tex=True) + "\n")
    (out / "ablation.md").write_text(note + main_table(root, a.seeds, ABLATION, ablation=True) + "\n")
    (out / "ablation.tex").write_text(main_table(root, a.seeds, ABLATION, tex=True, ablation=True) + "\n")
    (out / "horizon.md").write_text(f"Time limit {a.short_horizon:g} s (same episodes, cut short).\n\n" + note
                                    + main_table(root, a.seeds, MAIN, t_max=a.short_horizon) + "\n")
    (out / "phase1.md").write_text(phase1_table(root, a.seeds) + "\n")
    (out / "phase1.tex").write_text(phase1_table(root, a.seeds, tex=True) + "\n")
    training_figure(root, a.seeds, out / "fig_training.png")
    outcome_figure(root, a.seeds, out / "fig_outcomes.png")
    print((out / "results.md").read_text())
    print((out / "ablation.md").read_text())
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
