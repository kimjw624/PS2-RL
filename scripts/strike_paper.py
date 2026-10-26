#!/usr/bin/env python
"""Paper tables, figures and key numbers for the strike task from the final pipeline's evaluations (CPU, seconds).

    python scripts/strike_paper.py --root outputs/strike_final --seeds 0 1 2 --lang ko

Reads <root>/eval/s<seed>/<run>/strike_filter_episodes.json (run_strike_final_eval.sh), the training histories and the
Phase-I summaries. Writes to <root>/paper (or --out):
  tables/table1_10s.*, table1_15s.*   main comparison (success / contact / time-up) at 10 s and 15 s
  tables/table2_final.*               the final filter against the bCBF and one escape: 10 s / 15 s, lost certificate
  tables/table3_ablation.*            the final filter with one component removed (PS2 striker)
  tables/table4_phase1.*              Phase-I recovered fraction per region
  tables/table5_paired.*              paired comparisons on the same scenes (exact McNemar test)
  (each as .md, .csv, .tex and .tsv; the .tsv pastes into an HWP / Excel table)
  figures/fig_outcomes, fig_strike_time, fig_safety, fig_filter, fig_training, fig_ablation, fig_phase1 (.png 300 dpi, .pdf)
  numbers.md                          the numbers quoted in the text (rates with 95 % intervals, counts, contact bounds)
  episode_picks.json                  representative episodes of seed --pick_seed (filter saves, evasion, failures, ...)
Rows whose runs are missing are left out; every row says how many seeds and episodes it has. Contact columns give the
pooled count. Tables 2, 3 and the ablation figure compare configurations on the seeds where all of them were evaluated.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
import sys
import warnings

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from report_strike import OUTCOMES, cp_interval, load_episodes, truncate  # noqa: E402

# ----------------------------------------------------------------------------------------------------------- labels
T = {
    "ko": {"pursuit": "기준 접근 제어", "residual": "잔차 정책", "ps2": "PS2 정책", "none": "필터 없음",
           "static": "구형 근사", "guid1": "단일 회피 필터", "bcbf": "bCBF + 이산 보조 필터", "final": "최종 필터",
           "blind": "추격자 미고려 필터", "lqr": "호버 LQR 회복 정책", "nocp": "추격자 예측 재생 없음",
           "nolatch": "호버 latch 없음", "config": "구성", "success": "성공 %", "contact": "접촉 %", "contact_n": "접촉 % (건수)", "timeup": "시간초과 %",
           "success10": "10초 성공 %", "success15": "15초 성공 %", "lostcert": "인증 상실 %", "tstrike": "평균 도달 시간 [s]",
           "kept": "명목 유지율", "n": "시드 × 장면", "strike": "성공", "contact_l": "접촉", "timeup_l": "시간초과",
           "out_l": "영역 이탈", "frac": "에피소드 비율", "time": "시간 [s]", "struck_by": "누적 성공률",
           "chaser_clear": "추격자 최소 여유 거리 [m]", "obst_clear": "장애물 최소 여유 거리 [m]", "ecdf": "누적 비율",
           "kept_l": r"에피소드 평균 명목 유지율 $\bar{\lambda}$", "episodes": "에피소드 수", "lost_l": "인증 상실 에피소드 %",
           "steps": "환경 스텝 [M]", "train_strike": "학습 평가 성공률", "residual_train": "잔차 정책 (필터 없이 학습)",
           "ps2_train": "PS2 정책 (필터 포함 학습)", "ckpt": "사용한 체크포인트", "seed": "시드", "region": "영역",
           "recovered": "회복 성공률", "general": "일반", "blocked": "정지 경로\n막힘", "chaser": "추격자\n근접",
           "shell": "호버\n근방", "escape": "저속\n탈출", "weighted": "가중 평균", "lqr_b": "호버 LQR", "learned_b": "학습 회복 정책",
           "deployed": "최종 필터 (전체 구성)", "variant": "변형", "limit10": "10초 제한", "limit15": "15초 제한",
           "abl_title": "구성 요소 제거 (PS2 정책, 15초)"},
    "en": {"pursuit": "Pursuit (6 m/s)", "residual": "Residual SAC", "ps2": "PS2", "none": "no filter",
           "static": "static chaser ball", "guid1": "1-escape filter", "bCBF": "bCBF", "bcbf": "bCBF + discrete safeguard",
           "final": "ours", "blind": "filter blind to the chaser", "lqr": "hover-LQR backup", "nocp": "no committed prediction",
           "nolatch": "no hover latch", "config": "configuration", "success": "success %", "contact": "contact %", "contact_n": "contact % (count)",
           "timeup": "time-up %", "success10": "success 10 s %", "success15": "success 15 s %", "lostcert": "lost cert. %",
           "tstrike": "mean strike time [s]", "kept": "kept", "n": "seeds x scenes", "strike": "strike", "contact_l": "contact",
           "timeup_l": "time-up", "out_l": "left arena", "frac": "fraction of episodes", "time": "time [s]",
           "struck_by": "struck by time t", "chaser_clear": "closest chaser clearance [m]",
           "obst_clear": "closest obstacle clearance [m]", "ecdf": "cumulative fraction", "kept_l": r"episode mean kept fraction $\bar{\lambda}$",
           "episodes": "episodes", "lost_l": "episodes with a lost certificate %", "steps": "environment steps [M]",
           "train_strike": "training evaluation success", "residual_train": "residual striker (no filter)",
           "ps2_train": "PS2 striker (through the filter)", "ckpt": "checkpoint used", "seed": "seed", "region": "region",
           "recovered": "recovered fraction", "general": "general", "blocked": "blocked", "chaser": "chaser", "shell": "shell",
           "escape": "escape", "weighted": "weighted", "lqr_b": "hover LQR", "learned_b": "learned backup",
           "deployed": "deployed filter", "variant": "variant", "limit10": "10 s limit", "limit15": "15 s limit",
           "abl_title": "ablation (PS2, 15 s)"},
}


def rows_main(L):
    """(label, run, mode, color, linestyle) of the main comparison (paper Table 1)."""
    return [
        (f"{L['pursuit']}", "pursuit_static", "none", "#b07d12", "--"),
        (f"{L['pursuit']} + {L['static']}", "pursuit_static", "learned", "#7a5195", "-"),
        (f"{L['pursuit']} + {L['guid1']}", "pursuit_guid1", "learned", "#e07b39", "-"),
        (f"{L['pursuit']} + bCBF", "pursuit_bcbf", "learned", "#1b8a6b", "-"),
        (f"{L['pursuit']} + {L['blind']}", "pursuit_blind", "learned", "#9a9a9a", "-"),
        (f"{L['pursuit']} + {L['final']}", "pursuit_final", "learned", "#b07d12", "-"),
        (f"{L['residual']}", "residual_final", "none", "#d1477a", "--"),
        (f"{L['residual']} + {L['final']}", "residual_final", "learned", "#d1477a", "-"),
        (f"{L['ps2']} ({L['none']})", "ps2_final", "none", "#2a78d6", "--"),
        (f"{L['ps2']} + {L['guid1']}", "ps2_guid1", "learned", "#e07b39", "-"),
        (f"{L['ps2']} + bCBF", "ps2_bcbf", "learned", "#1b8a6b", "-"),
        (f"{L['ps2']} + {L['blind']}", "ps2_blind", "learned", "#9a9a9a", "-"),
        (f"{L['ps2']} + {L['final']}", "ps2_final", "learned", "#2a78d6", "-"),
    ]


def rows_final(L):
    return [(L["final"] + (" (이산 다중 회복 + 계획 유지)" if L is T["ko"] else " (discrete, variants, committed plan)"),
             "ps2_final", "learned"),
            (L["bcbf"], "ps2_bcbf", "learned"),
            (L["guid1"], "ps2_guid1", "learned")]


def rows_ablation(L):
    return [(L["deployed"], "ps2_final", "learned"), (L["lqr"], "ps2_final", "lqr"), (L["guid1"], "ps2_guid1", "learned"),
            (L["nocp"], "ps2_nocp", "learned"), (L["nolatch"], "ps2_nolatch", "learned"), (L["static"], "ps2_static", "learned"),
            (L["blind"], "ps2_blind", "learned")]


def rows_paired(L):
    """(label, configuration A, configuration B, outcome) of the paired comparisons (same seeds, same scenes)."""
    fin, ps2, res, pur = L["final"], L["ps2"], L["residual"], L["pursuit"]
    return [(f"{ps2} + {fin} vs {res} + {fin}", ("ps2_final", "learned"), ("residual_final", "learned"), "strike"),
            (f"{ps2} + {fin} vs {pur} + {fin}", ("ps2_final", "learned"), ("pursuit_final", "learned"), "strike"),
            (f"{fin} vs {L['guid1']} (PS2)", ("ps2_final", "learned"), ("ps2_guid1", "learned"), "strike"),
            (f"{fin} vs {L['guid1']} (PS2)", ("ps2_final", "learned"), ("ps2_guid1", "learned"), "contact"),
            (f"{fin} vs bCBF (PS2)", ("ps2_final", "learned"), ("ps2_bcbf", "learned"), "strike"),
            (f"{fin} vs {L['lqr']} (PS2)", ("ps2_final", "learned"), ("ps2_final", "lqr"), "strike"),
            (f"{res} + {fin} vs {res}", ("residual_final", "learned"), ("residual_final", "none"), "contact")]


# ----------------------------------------------------------------------------------------------------------- data
def episodes(root, seeds, run, mode, t_max=None):
    """Per-seed episode records (truncated at t_max) of (run, mode)."""
    out = []
    for s in seeds:
        e = load_episodes(root, s, run, mode)
        if e is not None:
            p = root / "eval" / f"s{s}" / run / "strike_filter_episodes.json"
            raw = json.loads(p.read_text())["modes"][mode]
            if "min_obstacle_dist" in raw:
                e["min_obstacle_dist"] = np.asarray(raw["min_obstacle_dist"])
            out.append((s, truncate(e, t_max)))
    return out


def common_seeds(root, seeds, rows):
    """(rows that have data in some seed, the seeds in which every one of them has data). Comparisons between rows (the
    final filter, its alternatives, the ablations) use only those seeds, so every row sees the same scenes and strikers."""
    have = [r for r in rows if episodes(root, seeds, r[1], r[2])]
    common = [s for s in seeds if all(load_episodes(root, s, r[1], r[2]) is not None for r in have)]
    return have, common


def seeds_note(lang, used, available):
    """Caption suffix naming the seeds a comparison is restricted to (empty when it uses every seed that has data)."""
    if not used or list(used) == list(available):
        return ""
    names = ", ".join(f"s{s}" for s in used)
    return f" (공통 시드 {names})" if lang == "ko" else f" (common seeds {names})"


def paired(root, seeds, A, B, what="strike", t_max=None):
    """Paired comparison of configurations A and B on the same (seed, scene) episodes (every evaluation replays the same
    scenes and start perturbations): counts k_A, k_B of the outcome, discordant pairs b (A only) and c (B only), and the
    exact two-sided McNemar p-value 2 P[Bin(b + c, 1/2) <= min(b, c)]."""
    from scipy.stats import binom
    codes = {"strike": [0], "contact": [1, 2, 3], "timeout": [5]}[what]
    ka = kb = b = c = n = 0
    used = []
    for sd in seeds:
        ea, eb = load_episodes(root, sd, *A), load_episodes(root, sd, *B)
        if ea is None or eb is None or len(ea["outcome"]) != len(eb["outcome"]):
            continue
        xa = np.isin(truncate(ea, t_max)["outcome"], codes)
        xb = np.isin(truncate(eb, t_max)["outcome"], codes)
        ka, kb = ka + int(xa.sum()), kb + int(xb.sum())
        b, c, n = b + int((xa & ~xb).sum()), c + int((~xa & xb).sum()), n + int(xa.size)
        used.append(sd)
    if not used:
        return None
    m = b + c
    pv = 1.0 if m == 0 else float(min(1.0, 2.0 * binom.cdf(min(b, c), m, 0.5)))
    return {"kA": ka, "kB": kb, "b": b, "c": c, "n": n, "p": pv, "seeds": used}


def p_text(pv):
    return "< 1e-6" if pv < 1e-6 else f"{pv:.2g}"


def stats(per):
    """Rates (mean, std over seeds, pooled count, pooled CP interval) of strike / contact / time-up / out, and means."""
    if not per:
        return None
    n = sum(len(e["outcome"]) for _, e in per)
    res = {"seeds": len(per), "n": n, "n_per": n // len(per)}
    for key, codes in (("strike", [0]), ("contact", [1, 2, 3]), ("chaser", [3]), ("obstacle", [1, 2]), ("timeout", [5]),
                       ("out", [4])):
        k_s = [int(np.isin(e["outcome"], codes).sum()) for _, e in per]
        r_s = [k / len(e["outcome"]) for k, (_, e) in zip(k_s, per)]
        k = sum(k_s)
        res[key] = (float(np.mean(r_s)), float(np.std(r_s)), k, cp_interval(k, n))
    ts = np.concatenate([e["strike_step"][e["outcome"] == 0] * e["dt"] for _, e in per])
    res["t_strike"] = float(ts.mean()) if len(ts) else float("nan")
    res["t_strike_median"] = float(np.median(ts)) if len(ts) else float("nan")
    res["kept"] = float(np.mean([np.mean(e["kept_fraction"]) for _, e in per]))
    res["lost"] = float(np.mean([np.mean(e["no_cert_steps"] > 0) for _, e in per]))
    return res


def pct(v, d=1):
    return f"{100 * v:.{d}f}"


# ----------------------------------------------------------------------------------------------------------- tables
def write_table(path_stem: Path, header, rows, caption):
    """The same table as markdown, CSV, TSV (pastes into HWP / Excel) and LaTeX (booktabs)."""
    path_stem.parent.mkdir(parents=True, exist_ok=True)
    md = [f"**{caption}**", "", "| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    md += ["| " + " | ".join(r) + " |" for r in rows]
    path_stem.with_suffix(".md").write_text("\n".join(md) + "\n")
    with open(path_stem.with_suffix(".csv"), "w", newline="", encoding="utf-8-sig") as f:
        csv.writer(f).writerows([header] + rows)
    path_stem.with_suffix(".tsv").write_text("\n".join("\t".join(r) for r in [header] + rows) + "\n", encoding="utf-8")
    esc = lambda s: (s.replace("%", r"\%").replace("&", r"\&").replace("±", r"$\pm$").replace("×", r"$\times$")
                     .replace("< 1e-6", r"$<10^{-6}$"))
    esc_e = lambda s: re.sub(r"\b(\d(?:\.\d+)?)e-0*(\d+)\b", r"$\1\\times10^{-\2}$", esc(s))
    cap = re.sub(r"^(표\s*\d+[a-z]?\.|Table\s*\d+[a-z]?\.)\s*", "", caption)
    tex = [r"\begin{table}\centering\small", rf"\caption{{{esc(cap)}}}\label{{tab:{path_stem.name}}}",
           r"\resizebox{\ifdim\width>\linewidth\linewidth\else\width\fi}{!}{%",
           r"\begin{tabular}{l" + "c" * (len(header) - 1) + r"}\toprule", " & ".join(esc(h) for h in header) + r" \\\midrule"]
    tex += [" & ".join(esc_e(c) for c in r) + r" \\" for r in rows]
    tex += [r"\bottomrule\end{tabular}}\end{table}"]
    path_stem.with_suffix(".tex").write_text("\n".join(tex) + "\n")
    return "\n".join(md)


def rate_cell(v, ci=False):
    m, s, k, (lo, hi) = v
    cell = f"{pct(m)}" + (f" ± {pct(s)}" if s > 0 else "")
    return cell + (f" [{pct(lo)}, {pct(hi)}]" if ci else "")


def count_cell(v):
    """A rare-event rate (contacts): pooled percentage, two decimals below 1 %, with the pooled count in brackets."""
    m, s, k, _ = v
    return f"{pct(m, 2 if m < 0.01 else 1)} ({k})"


# ----------------------------------------------------------------------------------------------------------- figures
def setup_style(lang):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 300, "font.size": 8.5, "axes.titlesize": 9,
                         "axes.labelsize": 8.5, "legend.fontsize": 7.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
                         "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25,
                         "grid.linewidth": 0.5, "axes.axisbelow": True, "axes.linewidth": 0.7, "lines.linewidth": 1.4, "pdf.fonttype": 42,
                         "axes.unicode_minus": False})
    if lang != "ko":
        return lang
    have = {f.name for f in font_manager.fontManager.ttflist}
    # Korean-capable families (the Noto CJK collections register under their first face, JP/SC/TC, all with Hangul)
    for name in ("NanumGothic", "Noto Sans CJK KR", "Noto Sans KR", "Malgun Gothic", "AppleGothic", "UnDotum", "Baekmuk Gulim",
                 "Noto Sans CJK JP", "Noto Sans CJK SC", "Noto Sans CJK TC", "Source Han Sans K"):
        if name in have:
            plt.rcParams["font.family"] = [name, "DejaVu Sans"]
            return "ko"
    try:
        import koreanize_matplotlib  # noqa: F401  (bundles NanumGothic)
        plt.rcParams["axes.unicode_minus"] = False
        return "ko"
    except ImportError:
        warnings.warn("no Korean font found (apt-get install fonts-nanum, or pip install koreanize-matplotlib); "
                      "figures fall back to English labels")
        return "en"


def save(fig, out: Path, name: str):
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.png", bbox_inches="tight")
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)


def fig_outcomes(root, seeds, L, out):
    import matplotlib.pyplot as plt
    rows = [(lab, run, mode) for lab, run, mode, _c, _ls in rows_main(L) if episodes(root, seeds, run, mode)]
    if not rows:
        return
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 0.26 * len(rows) + 1.1), sharey=True)
    cols = (("strike", "#2a78d6", L["strike"]), ("contact", "#3d3c39", L["contact_l"]), ("timeout", "#d9a514", L["timeup_l"]),
            ("out", "#c9c7c0", L["out_l"]))
    y = np.arange(len(rows))[::-1]
    for ax, t_max, title in ((axes[0], 10.0, L["limit10"]), (axes[1], None, L["limit15"])):
        left = np.zeros(len(rows))
        for key, c, lab in cols:
            v = 100 * np.array([stats(episodes(root, seeds, r, m, t_max))[key][0] for _, r, m in rows])
            ax.barh(y, v, left=left, color=c, height=0.72, label=lab, edgecolor="white", linewidth=0.4)
            for yy, vv, ll in zip(y, v, left):
                if key == "strike" and vv > 12:
                    ax.text(ll + vv - 1, yy, f"{vv:.1f}", ha="right", va="center", fontsize=6.5, color="white")
            left += v
        ax.set_xlim(0, 100)
        ax.set_title(title)
        ax.set_xlabel(L["frac"] + " [%]")
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([lab for lab, _, _ in rows])
    h, l = axes[1].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.6, 0.0))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, out, "fig_outcomes")


def fig_strike_time(root, seeds, L, out):
    import matplotlib.pyplot as plt
    pick = [r for r in rows_main(L) if (r[1], r[2]) in {("ps2_final", "learned"), ("residual_final", "learned"),
                                                        ("pursuit_final", "learned"), ("ps2_bcbf", "learned"),
                                                        ("ps2_guid1", "learned"), ("ps2_final", "none")}]
    fig, ax = plt.subplots(figsize=(3.45, 2.5))
    t = np.linspace(0, 15, 301)
    for lab, run, mode, c, ls in pick:
        per = episodes(root, seeds, run, mode)
        if not per:
            continue
        curves = []
        for _, e in per:
            ts = np.where(e["outcome"] == 0, e["strike_step"] * e["dt"], np.inf)
            curves.append((ts[None, :] <= t[:, None]).mean(axis=1))
        C = np.array(curves)
        ax.plot(t, C.mean(0), color=c, ls=ls, lw=1.5 if run == "ps2_final" and mode == "learned" else 1.1, label=lab)
        if len(C) > 1:
            ax.fill_between(t, C.min(0), C.max(0), color=c, alpha=0.15, lw=0)
    ax.axvline(10, color="#555", lw=0.7, ls=":")
    ax.set_xlim(3, 15)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel(L["time"])
    ax.set_ylabel(L["struck_by"])
    ax.legend(loc="lower right", frameon=False, fontsize=6.5)
    save(fig, out, "fig_strike_time")


def chaser_contact_radius(root, seeds, default=0.7):
    """r_c + r_d of the evaluated chaser (evaluation metadata; the original chaser if absent)."""
    for s in seeds:
        d = root / "eval" / f"s{s}"
        for tb in (sorted(d.glob("*/strike_filter_table.json")) if d.is_dir() else []):
            ch = json.loads(tb.read_text()).get("chaser")
            if ch:
                return float(ch["actual_radius"]) + 0.3
    return default


def fig_safety(root, seeds, L, out, r_drone=0.3):
    import matplotlib.pyplot as plt
    r_contact_ch = chaser_contact_radius(root, seeds)
    pick = [r for r in rows_main(L) if (r[1], r[2]) in {("ps2_final", "none"), ("ps2_final", "learned"),
                                                        ("pursuit_final", "learned"), ("ps2_static", "learned"),
                                                        ("ps2_blind", "learned")}]
    have_obs = any("min_obstacle_dist" in e for r in pick for _, e in episodes(root, seeds, r[1], r[2]))
    fig, axes = plt.subplots(1, 2 if have_obs else 1, figsize=(7.0 if have_obs else 3.45, 2.5), squeeze=False)
    ax = axes[0, 0]
    for lab, run, mode, c, ls in pick:
        per = episodes(root, seeds, run, mode)
        if not per:
            continue
        d = np.concatenate([e["min_chaser_dist"] for _, e in per]) - r_contact_ch
        d = np.sort(np.minimum(d, 15.0))
        ax.plot(d, np.arange(1, len(d) + 1) / len(d), color=c, ls=ls, label=lab)
    ax.axvline(0, color="#c4302b", lw=0.8)
    ax.axvline(0.2, color="#555", lw=0.7, ls=":")
    ax.text(-0.04, 0.55, L["contact_l"], color="#c4302b", fontsize=6.5, rotation=90, ha="right", va="center",
            transform=ax.get_xaxis_transform())
    ax.text(0.26, 0.55, "0.2 m", color="#555", fontsize=6.5, rotation=90, ha="left", va="center",
            transform=ax.get_xaxis_transform())
    ax.set_xlim(-0.3, 8)
    ax.set_xlabel(L["chaser_clear"])
    ax.set_ylabel(L["ecdf"])
    ax.legend(frameon=False, fontsize=6.5, loc="lower right")
    if have_obs:
        ax = axes[0, 1]
        for lab, run, mode, c, ls in pick:
            per = [(s, e) for s, e in episodes(root, seeds, run, mode) if "min_obstacle_dist" in e]
            if not per:
                continue
            d = np.sort(np.concatenate([e["min_obstacle_dist"] for _, e in per]) - r_drone)
            ax.plot(d, np.arange(1, len(d) + 1) / len(d), color=c, ls=ls, label=lab)
        ax.axvline(0, color="#c4302b", lw=0.8)
        ax.set_xlim(-0.3, 6)
        ax.set_xlabel(L["obst_clear"])
    save(fig, out, "fig_safety")


def fig_filter(root, seeds, L, out):
    import matplotlib.pyplot as plt
    pick = [r for r in rows_main(L) if r[2] == "learned" and r[1] in ("pursuit_final", "residual_final", "ps2_final")]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.4), gridspec_kw={"width_ratios": [1.5, 1]})
    lost, names, colors = [], [], []
    for lab, run, mode, c, ls in pick:
        per = episodes(root, seeds, run, mode)
        if not per:
            continue
        k = np.sort(np.concatenate([e["kept_fraction"] for _, e in per]))
        axes[0].plot(k, np.arange(1, len(k) + 1) / len(k), color=c, lw=1.5, label=lab)
        lost.append(100 * np.mean(np.concatenate([e["no_cert_steps"] > 0 for _, e in per])))
        names.append(lab.split(" + ")[0])
        colors.append(c)
    axes[0].set_xlabel(L["kept_l"])
    axes[0].set_ylabel(L["ecdf"])
    axes[0].set_xlim(0.3, 1.0)
    axes[0].legend(frameon=False, fontsize=6.5, loc="upper left")
    x = np.arange(len(names))
    axes[1].bar(x, lost, color=colors, width=0.6)
    for xx, v in zip(x, lost):
        axes[1].text(xx, v + 0.05, f"{v:.1f}", ha="center", fontsize=7)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, fontsize=7)
    axes[1].set_ylabel(L["lost_l"])
    axes[1].grid(axis="x", visible=False)
    fig.tight_layout()
    save(fig, out, "fig_filter")


def fig_training(root, seeds, L, out):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.3))
    for ax, stage, title in ((axes[0], "residual", L["residual_train"]), (axes[1], "ps2", L["ps2_train"])):
        any_ = False
        for k, s in enumerate(seeds):
            p = root / stage / f"striker_rs0p1_s{s}" / "history.json"
            if not p.exists():
                continue
            any_ = True
            h = json.loads(p.read_text())
            x, y = np.asarray(h["eval/step"]) / 1e6, np.asarray(h["eval/strike_rate"])
            c = ["#2a78d6", "#d1477a", "#1b8a6b", "#b07d12"][k % 4]
            ax.plot(x, y, color=c, lw=1.2, label=f"{L['seed']} {s}")
            sm = root / stage / f"striker_rs0p1_s{s}" / "summary.json"
            best = json.loads(sm.read_text()).get("best_step") if sm.exists() else None
            if best is not None:
                j = int(np.argmin(np.abs(x * 1e6 - best)))
                ax.plot(x[j], y[j], "o", color=c, mec="#0b0b0b", ms=4.5, zorder=5)
        if not any_:
            ax.set_visible(False)
            continue
        ax.plot([], [], "o", color="#cccccc", mec="#0b0b0b", ms=4.5, label=L["ckpt"])
        ax.set_title(title)
        ax.set_xlabel(L["steps"])
        ax.set_ylim(-0.02, 1.02)
        ax.legend(frameon=False, fontsize=6.5, loc="lower left" if stage == "residual" else "lower right")
    axes[0].set_ylabel(L["train_strike"])
    save(fig, out, "fig_training")


def fig_ablation(root, seeds, L, out):
    import matplotlib.pyplot as plt
    rows, sc = common_seeds(root, seeds, rows_ablation(L))
    if not rows:
        return
    sc = sc or seeds
    st = [stats(episodes(root, sc, r, m)) for _, r, m in rows]
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 0.28 * len(rows) + 0.9), sharey=True)
    y = np.arange(len(rows))[::-1]
    for ax, key, c, ttl in ((axes[0], "strike", "#2a78d6", L["success"]), (axes[1], "contact", "#3d3c39", L["contact"]),
                            (axes[2], "timeout", "#d9a514", L["timeup"])):
        v = np.array([100 * x[key][0] for x in st])
        lo = np.array([100 * x[key][3][0] for x in st])
        hi = np.array([100 * x[key][3][1] for x in st])
        ax.barh(y, v, color=c, height=0.62)
        ax.errorbar(v, y, xerr=[v - lo, hi - v], fmt="none", ecolor="#0b0b0b", elinewidth=0.7, capsize=2)
        for yy, vv in zip(y, v):
            ax.text(vv + (hi.max() * 0.02 if key != "strike" else 1), yy + 0.28, f"{vv:.1f}", fontsize=6.3, va="center")
        ax.set_title(ttl, fontsize=8.5)
        ax.grid(axis="y", visible=False)
        if key == "strike":
            ax.set_xlim(min(50.0, float(lo.min()) - 5), 100)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([lab for lab, _, _ in rows])
    lang = "ko" if L is T["ko"] else "en"
    used_any = [s for s in seeds if any(load_episodes(root, s, r[1], r[2]) is not None for r in rows)]
    fig.suptitle(L["abl_title"] + seeds_note(lang, sc, used_any), fontsize=9, x=0.01, ha="left")
    fig.tight_layout()
    save(fig, out, "fig_ablation")


def phase1_values(root, seeds):
    regs, learned, lqr = None, [], []
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
        return None
    return (regs or []) + ["weighted"], np.array(learned), np.array(lqr)


def fig_phase1(root, seeds, L, out):
    import matplotlib.pyplot as plt
    v = phase1_values(root, seeds)
    if v is None:
        return
    names, Lr, Q = v
    fig, ax = plt.subplots(figsize=(3.45, 2.3))
    x = np.arange(len(names))
    q, lr = 100 * Q.mean(0), 100 * Lr.mean(0)
    ax.bar(x - 0.19, q, width=0.38, color="#9a9a9a", label=L["lqr_b"])
    ax.bar(x + 0.19, lr, width=0.38, color="#2a78d6", yerr=100 * Lr.std(0) if len(Lr) > 1 else None, capsize=2,
           error_kw={"elinewidth": 0.7}, label=f"{L['learned_b']} ({len(Lr)} {L['seed']})")
    for xx, a, b in zip(x, q, lr):
        ax.text(xx - 0.19, a + 1.5, f"{a:.0f}", ha="center", fontsize=5.8, color="#555")
        ax.text(xx + 0.19, b + 1.5, f"{b:.0f}", ha="center", fontsize=5.8, color="#1a4f8f")
    ax.set_xticks(x)
    ax.set_xticklabels([L.get(n, n) for n in names], fontsize=6.8)
    ax.set_ylabel(L["recovered"] + " [%]")
    ax.set_ylim(0, 112)
    ax.legend(frameon=False, fontsize=6.3, ncol=2, loc="upper left", bbox_to_anchor=(0, 1.04))
    ax.grid(axis="x", visible=False)
    save(fig, out, "fig_phase1")


# ----------------------------------------------------------------------------------------------------------- episodes
def pick_episodes(root, seed, n_per=1):
    """Representative episodes of one seed's evaluation (indices into its N scenes)."""
    def ep(run, mode):
        e = load_episodes(root, seed, run, mode)
        return None if e is None else e

    F, N = ep("ps2_final", "learned"), ep("ps2_final", "none")
    if F is None:
        return {}
    picks = {}
    s_ok = F["outcome"] == 0
    t_s = F["strike_step"] * F["dt"]

    def put(name, idx, order=None):
        idx = np.asarray(idx)
        if len(idx):
            if order is not None:
                idx = idx[np.argsort(order[idx])]
            picks[name] = [int(i) for i in idx[:n_per]]

    if N is not None:
        put("filter_saves_obstacle", np.nonzero((N["outcome"] == 1) & s_ok)[0], np.abs(t_s - np.median(t_s[s_ok])))
        put("filter_saves_chaser", np.nonzero((N["outcome"] == 3) & s_ok)[0])
    put("close_chaser_evasion", np.nonzero(s_ok)[0], F["min_chaser_dist"])
    put("ps2_contact", np.nonzero(np.isin(F["outcome"], [1, 2, 3]))[0])
    put("ps2_timeup", np.nonzero(F["outcome"] == 5)[0])
    put("typical", np.nonzero(s_ok)[0], np.abs(t_s - np.median(t_s[s_ok])))
    for run, mode, name, bad in (("pursuit_final", "learned", "ps2_beats_pursuit", [5]),
                                 ("residual_final", "learned", "ps2_beats_residual", [5]),
                                 ("ps2_static", "learned", "static_ball_fails", [1, 2, 3]),
                                 ("ps2_bcbf", "learned", "bcbf_fails", [1, 2, 3, 5])):
        e = ep(run, mode)
        if e is not None:
            put(name, np.nonzero(np.isin(e["outcome"], bad) & s_ok)[0], t_s)
    return picks


# ----------------------------------------------------------------------------------------------------------- main
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="outputs/strike_final")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--lang", choices=("ko", "en"), default="ko")
    p.add_argument("--pick_seed", type=int, default=0, help="seed whose evaluation the representative episodes come from")
    p.add_argument("--out", default="", help="output folder (default <root>/paper)")
    a = p.parse_args(argv)
    root = Path(a.root)
    out = Path(a.out) if a.out else root / "paper"
    lang = setup_style(a.lang)
    L = T[lang]
    Lt = T[a.lang]                                 # tables keep the requested language (no font needed)
    tab = out / "tables"

    # ---- tables
    texts = []
    for t_max, name, title in ((10.0, "table1_10s", "표 1. 10초 제한의 주요 구성 비교" if a.lang == "ko" else
                                "Table 1. Main comparison, 10 s limit"),
                               (None, "table1_15s", "표 1b. 15초 제한의 주요 구성 비교" if a.lang == "ko" else
                                "Table 1b. Main comparison, 15 s limit")):
        rows = []
        for lab, run, mode, _c, _ls in rows_main(Lt):
            st = stats(episodes(root, a.seeds, run, mode, t_max))
            if st:
                rows.append([lab, rate_cell(st["strike"]), count_cell(st["contact"]), rate_cell(st["timeout"]),
                             f"{st['t_strike']:.2f}", f"{st['seeds']} × {st['n_per']}"])
        if rows:
            texts.append(write_table(tab / name, [Lt["config"], Lt["success"], Lt["contact_n"], Lt["timeup"], Lt["tstrike"],
                                                  Lt["n"]], rows, title))
    rows = []
    have, sc2 = common_seeds(root, a.seeds, rows_final(Lt))
    sc2 = sc2 or a.seeds
    any2 = [s for s in a.seeds if any(load_episodes(root, s, r[1], r[2]) is not None for r in have)]
    for lab, run, mode in have:
        s10, s15 = stats(episodes(root, sc2, run, mode, 10.0)), stats(episodes(root, sc2, run, mode))
        if s15:
            rows.append([lab, rate_cell(s10["strike"]), rate_cell(s15["strike"]), count_cell(s15["contact"]),
                         pct(s15["lost"]), f"{s15['t_strike']:.2f}", f"{s15['kept']:.2f}", f"{s15['seeds']} × {s15['n_per']}"])
    if rows:
        texts.append(write_table(tab / "table2_final", [Lt["config"], Lt["success10"], Lt["success15"], Lt["contact_n"],
                                                        Lt["lostcert"], Lt["tstrike"], Lt["kept"], Lt["n"]], rows,
                                 ("표 2. 최종 구성의 성공률과 인증 상실률" if a.lang == "ko" else
                                  "Table 2. Final configuration: success and lost certificates") + seeds_note(a.lang, sc2, any2)))
    rows = []
    have, sc3 = common_seeds(root, a.seeds, rows_ablation(Lt))
    sc3 = sc3 or a.seeds
    any3 = [s for s in a.seeds if any(load_episodes(root, s, r[1], r[2]) is not None for r in have)]
    for lab, run, mode in have:
        st = stats(episodes(root, sc3, run, mode))
        if st:
            rows.append([lab, rate_cell(st["strike"]), count_cell(st["contact"]), rate_cell(st["timeout"]), pct(st["lost"]),
                         f"{st['seeds']} × {st['n_per']}"])
    if rows:
        texts.append(write_table(tab / "table3_ablation", [Lt["variant"], Lt["success"], Lt["contact_n"], Lt["timeup"],
                                                           Lt["lostcert"], Lt["n"]], rows,
                                 ("표 3. 구성 요소 제거 실험 (PS2 정책, 15초)" if a.lang == "ko" else
                                  "Table 3. Ablation (PS2 striker, 15 s)") + seeds_note(a.lang, sc3, any3)))
    v = phase1_values(root, a.seeds)
    if v is not None:
        names, Lr, Q = v
        rows = [[Lt["lqr_b"]] + [pct(x) for x in Q.mean(0)],
                [f"{Lt['learned_b']} ({len(Lr)})"] + [pct(m) + (f" ± {pct(s)}" if len(Lr) > 1 else "")
                                                      for m, s in zip(Lr.mean(0), Lr.std(0))]]
        texts.append(write_table(tab / "table4_phase1", [Lt["region"]] + [Lt.get(n, n).replace("\n", " ") for n in names], rows,
                                 "표 4. Phase I 회복 성공률 (%; 가중 평균은 앞 네 영역에 0.4, 0.2, 0.2, 0.2)" if a.lang == "ko"
                                 else "Table 4. Phase-I recovered fraction (%; weighted = 0.4, 0.2, 0.2, 0.2 over the first four)"))

    # ---- paired comparisons (same seeds and scenes; exact McNemar test)
    rows, pair_lines = [], []
    what_l = {"strike": Lt["strike"], "contact": Lt["contact_l"]}
    for lab, A, B, what in rows_paired(Lt):
        for t_max in ((10.0, None) if what == "strike" else (None,)):
            r = paired(root, a.seeds, A, B, what, t_max)
            if r is None:
                continue
            lim = "10 s" if t_max else "15 s"
            rows.append([lab, f"{what_l[what]} ({lim})", pct(r["kA"] / r["n"], 2 if what == "contact" else 1),
                         pct(r["kB"] / r["n"], 2 if what == "contact" else 1), str(r["b"]), str(r["c"]), p_text(r["p"]),
                         f"{len(r['seeds'])} × {r['n'] // len(r['seeds'])}"])
            pair_lines.append(f"- {lab}, {what_l[what]} {lim}: {r['kA']}/{r['n']} vs {r['kB']}/{r['n']}; "
                              f"{'A만' if a.lang == 'ko' else 'A only'} {r['b']}, {'B만' if a.lang == 'ko' else 'B only'} "
                              f"{r['c']}; McNemar p {p_text(r['p'])} (s{', s'.join(map(str, r['seeds']))})")
    if rows:
        hdr = (["비교 (A vs B)", "지표", "A %", "B %", "A만", "B만", "McNemar p", "시드 × 장면"] if a.lang == "ko" else
               ["comparison (A vs B)", "outcome", "A %", "B %", "A only", "B only", "McNemar p", "seeds x scenes"])
        texts.append(write_table(tab / "table5_paired", hdr, rows,
                                 "표 5. 같은 장면에서의 쌍 비교 (정확 McNemar 검정)" if a.lang == "ko" else
                                 "Table 5. Paired comparisons on the same scenes (exact McNemar test)"))

    # ---- key numbers
    num = ["# " + ("본문 인용 수치" if a.lang == "ko" else "Numbers quoted in the text"), ""]
    for lab, run, mode, _c, _ls in rows_main(Lt):
        s10, s15 = stats(episodes(root, a.seeds, run, mode, 10.0)), stats(episodes(root, a.seeds, run, mode))
        if not s15:
            continue
        k, n = s15["contact"][2], s15["n"]
        hi = s15["contact"][3][1]
        num.append(f"- **{lab}** ({s15['seeds']} × {s15['n_per']}): "
                   f"{'성공' if a.lang == 'ko' else 'success'} 10 s {pct(s10['strike'][0])} %, 15 s {pct(s15['strike'][0])} % "
                   f"[{pct(s15['strike'][3][0])}, {pct(s15['strike'][3][1])}]; "
                   f"{'접촉' if a.lang == 'ko' else 'contact'} {k}/{n} ({pct(s15['contact'][0], 2)} %, "
                   f"95 % CP {'상한' if a.lang == 'ko' else 'upper'} {pct(hi, 2)} %; {'추격자' if a.lang == 'ko' else 'chaser'} "
                   f"{s15['chaser'][2]}, {'장애물/바닥' if a.lang == 'ko' else 'obstacle/floor'} {s15['obstacle'][2]}); "
                   f"{'시간초과' if a.lang == 'ko' else 'time-up'} 10 s {pct(s10['timeout'][0])} %, 15 s {pct(s15['timeout'][0])} %; "
                   f"{'평균 도달' if a.lang == 'ko' else 'mean strike'} {s15['t_strike']:.2f} s "
                   f"({'중앙값' if a.lang == 'ko' else 'median'} {s15['t_strike_median']:.2f} s); "
                   f"{'명목 유지' if a.lang == 'ko' else 'kept'} {s15['kept']:.2f}; "
                   f"{'인증 상실' if a.lang == 'ko' else 'lost cert.'} {pct(s15['lost'])} %")
    num += ["", ("0/n 관측일 때 단측 95 % 상한: 1 − 0.05^(1/n); 양측 Clopper–Pearson 상한: 1 − 0.025^(1/n). "
                 "n = 1000: 0.30 % / 0.37 %; n = 3000: 0.10 % / 0.12 %.") if a.lang == "ko" else
            ("0/n observed: one-sided 95 % bound 1 − 0.05^(1/n); two-sided Clopper–Pearson 1 − 0.025^(1/n). "
             "n = 1000: 0.30 % / 0.37 %; n = 3000: 0.10 % / 0.12 %.")]
    if pair_lines:
        num += ["", "## " + ("같은 장면 쌍 비교 (정확 McNemar 검정)" if a.lang == "ko" else
                             "Paired comparisons on the same scenes (exact McNemar test)"), ""] + pair_lines
    (out / "numbers.md").write_text("\n".join(num) + "\n")
    picks = pick_episodes(root, a.pick_seed)
    (out / "episode_picks.json").write_text(json.dumps({"seed": a.pick_seed, "picks": picks}, indent=2))

    # ---- figures
    fig_dir = out / "figures"
    for f in (fig_outcomes, fig_strike_time, fig_safety, fig_filter, fig_training, fig_ablation, fig_phase1):
        try:
            f(root, a.seeds, L, fig_dir)
        except Exception as ex:                    # one broken figure must not stop the rest
            warnings.warn(f"{f.__name__}: {ex!r}")
    print("\n\n".join(texts))
    print((out / "numbers.md").read_text())
    print(f"episode picks (seed {a.pick_seed}): {json.dumps(picks)}")
    print(f"wrote {tab}, {fig_dir}, {out / 'numbers.md'}, {out / 'episode_picks.json'} (figure labels: {lang})")


if __name__ == "__main__":
    main()
