#!/usr/bin/env python
"""Report of the online-filter sweep: safety, progress and onboard compute per setting, and the recommended setting.

    python scripts/report_cyl_online.py --sweep results/cyl_online/sweep.json \\
        --bench results/cyl_online/bench_sandbox.json [--pi_bench bench_pi4.json] --out results/cyl_online

Compute: the worst-case decision time of a setting is the bench fit (scripts/bench_cyl_filter.py)
    ms = a + S (b + c R + e R F) + w (period - 1),   S = period + N rollout steps, R = rollout rows (candidates + 1),
    F = actor kFLOP per evaluation, w = one window step of the numpy path (the nominal simulated)
for each backend (numpy, numba). With a Raspberry Pi bench (--pi_bench) its own fit is used; without one the
sandbox fit is scaled by --pi_factor (default 4, an assumption until the Pi is measured).

Recommendation: among the settings with no contact and no floor strike in any scenario set and every deployment
episode reaching the goal, the highest stress success whose worst-case decision on the Pi fits
the budget: ``--budget_frac`` of the decision period (period x 20 ms; the rest is left for the ROS 2 bridge and the
control step), per backend; ties go to the cheaper setting.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

OBS_DIM = 48
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"


def kflop(width: int) -> float:
    return 0.0 if width == 0 else (OBS_DIM * width + width * width + 4 * width) * 2e-3


def predict(fit: dict, steps: int, rows: int, width: int, period: int) -> float:
    return (fit["a"] + steps * (fit["per_step"] + fit["per_step_row"] * rows + fit["per_step_row_kflop"] * rows * kflop(width))
            + fit.get("per_window_step", 0.0) * (period - 1))


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sweep", required=True)
    p.add_argument("--bench", required=True, help="bench JSON of the machine the sweep ran on")
    p.add_argument("--pi_bench", default="", help="bench JSON from the Raspberry Pi (optional)")
    p.add_argument("--pi_factor", type=float, default=4.0)
    p.add_argument("--budget_frac", type=float, default=0.5)
    p.add_argument("--dt", type=float, default=0.02)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    sweep = json.loads(Path(a.sweep).read_text())
    bench = json.loads(Path(a.bench).read_text())["info"]
    pi = json.loads(Path(a.pi_bench).read_text())["info"] if a.pi_bench else None
    backends = [b for b in ("numpy", "numba") if f"fit_worst_ms_{b}" in bench]
    rows_out = []
    for k, rec in sweep["runs"].items():
        s = rec["setting"]
        T, P = float(s["horizon"]), int(s["period"])
        N = int(round(T / a.dt))
        r = {"key": k, **s, "rows": rec["rows"], "width": rec["width"], "steps_worst": P + N}
        st = rec.get("stress/random", {})
        r.update({"stress_success": st.get("success"), "stress_contact": st.get("contact"), "stress_none": st.get("mode_none"),
                  "stress_t": st.get("t_goal_mean"), "stress_lam": st.get("lam_mean"), "stress_steps_mean": st.get("steps_mean"),
                  "stress_steps_p99": st.get("steps_p99"), "stress_min_clear": st.get("min_clear")})
        contacts = [rec[x]["contact"] for x in rec if "/" in x]
        r["any_contact"] = float(max(contacts)) if contacts else float("nan")
        r["final_success"] = min(rec.get("final/one", {}).get("success", 0.0), rec.get("final/two", {}).get("success", 0.0))
        r["blind_floor"] = max(rec.get(f"blind/{c}", {}).get("floor", 0.0) for c in ("one", "two"))
        r["stress_floor"] = st.get("floor", 0.0)
        r["final_t"] = np.mean([rec.get(f"final/{c}", {}).get("t_goal_mean", np.nan) for c in ("one", "two")])
        for be in backends:
            here = predict(bench[f"fit_worst_ms_{be}"], P + N, rec["rows"], rec["width"], P)
            r[f"{be}_ms_here"] = here
            r[f"{be}_ms_pi"] = (predict(pi[f"fit_worst_ms_{be}"], P + N, rec["rows"], rec["width"], P)
                                if pi and f"fit_worst_ms_{be}" in pi else a.pi_factor * here)
        r["budget_ms"] = a.budget_frac * P * a.dt * 1e3
        rows_out.append(r)
    rows_out.sort(key=lambda r: (str(r["backup"]), r["blends"], r["family"], r["horizon"], r["period"]))
    ok = [r for r in rows_out if r["any_contact"] == 0.0 and r["final_success"] >= 1.0 and r["stress_floor"] == 0.0
          and r["blind_floor"] == 0.0]
    rec = {}
    for be in backends:
        fit = [r for r in ok if r[f"{be}_ms_pi"] <= r["budget_ms"]]
        if fit:
            best = max(fit, key=lambda r: (round(r["stress_success"], 2), -r[f"{be}_ms_pi"]))
            rec[be] = best["key"]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pi_note = (f"Pi bench {a.pi_bench}" if pi else f"sandbox x {a.pi_factor:g} (assumed until the Pi is measured)")
    lines = ["| backup | blends | family | horizon [s] | period | rows | stress success | stress t [s] | mode none | "
             "contacts (all sets) | final | worst steps | " + " | ".join(f"{b} Pi ms" for b in backends)
             + " | budget ms |", "|" + "---|" * (12 + len(backends) + 1)]
    for r in rows_out:
        star = " **(rec. " + ", ".join(b for b, k in rec.items() if k == r["key"]) + ")**" if r["key"] in rec.values() else ""
        lines.append(f"| {r['backup']}{star} | {r['blends']} | {r['family']} | {r['horizon']:g} | {r['period']} | {r['rows']} | "
                     f"{r['stress_success']:.2f} | {r['stress_t']:.1f} | {r['stress_none']:.3f} | {r['any_contact']:.2f} | "
                     f"{r['final_success']:.2f} | {r['steps_worst']} | "
                     + " | ".join(f"{r[f'{b}_ms_pi']:.1f}" for b in backends) + f" | {r['budget_ms']:.0f} |")
    md = (f"# Online filter sweep\n\nCompute: {pi_note}. Budget: {a.budget_frac:g} x decision period.\n\n"
          + "\n".join(lines) + "\n\nRecommended: " + (", ".join(f"{b}: {k}" for b, k in rec.items()) or "none") + "\n")
    (out / "sweep_table.md").write_text(md)
    (out / "sweep_report.json").write_text(json.dumps({"rows": rows_out, "recommended": rec, "compute": pi_note}, indent=1,
                                                       default=float))
    print(md)
    # figure: stress success and Pi worst-case time vs horizon, one line per backup (blends 3, family 1, period 1)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    sub = [r for r in rows_out if r["blends"] == 3 and r["family"] == 1 and r["period"] == 1]
    order = ["lqr", 32, 64, 128, 256]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    plt.rcParams.update({"font.size": 9})
    be = "numba" if "numba" in backends else backends[0]
    for i, b in enumerate(order):
        rr = sorted([r for r in sub if r["backup"] == b], key=lambda r: r["horizon"])
        if not rr:
            continue
        lab = "hover LQR" if b == "lqr" else f"learned, width {b}"
        h = [r["horizon"] for r in rr]
        axes[0].plot(h, [r["stress_success"] for r in rr], marker="o", ms=4, lw=1.6, color=SERIES[i], label=lab)
        axes[1].plot(h, [r[f"{be}_ms_pi"] for r in rr], marker="o", ms=4, lw=1.6, color=SERIES[i], label=lab)
    if be == "numba" and "numpy" in backends:            # the same settings without numba, for comparison
        rr = sorted([r for r in sub if r["backup"] == 32], key=lambda r: r["horizon"])
        if rr:
            h = [r["horizon"] for r in rr]
            y = [r["numpy_ms_pi"] for r in rr]
            axes[1].plot(h, y, ls="--", lw=1.2, color=MUTED, marker="o", ms=3)
            axes[1].text(h[0], y[0] * 0.75, "learned, width 32, numpy", color=INK2, fontsize=8, va="top")
    axes[0].set_ylabel("stress success (no contacts anywhere)")
    axes[0].set_ylim(0, 1)
    axes[1].set_ylabel("worst-case decision on the Pi [ms]")
    axes[1].set_yscale("log")
    axes[1].axhline(0.5 * 20, color=MUTED, lw=0.8, ls=":")
    axes[1].text(1.5, 10.5, "half of a 50 Hz step", ha="right", va="bottom", color=MUTED, fontsize=8)
    for ax in axes:
        ax.set_xlabel("backup rollout horizon [s]")
        ax.grid(color=GRID, lw=0.6)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8, loc="upper right")
    fig.suptitle(f"3 blends, backup alone, decisions every step | solid: {be}, dashed: numpy | compute: {pi_note}",
                 fontsize=9, color=INK2)
    fig.tight_layout()
    fig.savefig(out / "sweep_horizon.png", dpi=150)


if __name__ == "__main__":
    main()
