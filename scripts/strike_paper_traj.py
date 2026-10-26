#!/usr/bin/env python
"""Paper figures: trajectories of chosen episodes for the compared configurations (CPU; no simulation).

Each configuration is an evaluation replay (scripts/eval_strike_filter.py with --save_traj 1 and the same
--record_episodes / --only_episodes list; docker_batch/run_strike_paper.sh makes them), given as
"RUN_DIR:MODE:LABEL:KEY". Labels, outcome names and axis names follow the paper's terms (--terms ko: 드론, 목표 대상,
요격기, 장애물, 배치, 충돌, 성공, 시간 초과, 운용 영역 이탈, 혼합률, 회복 정책 번호, 요격기 여유도); sizes follow a
two-column A4 page (--col_cm, --page_cm).

    python scripts/strike_paper_traj.py --out outputs/strike_final/paper/traj4 \
        --config "R/residual:none:잔차 정책:res" --config "R/residual:learned:잔차 + 최종 필터:res_f" \
        --config "R/ps2_one:learned:PS2 + 단일 회피 필터:one" --config "R/ps2:learned:PS2 + 최종 필터:ours" \
        --grid 827 576 146 914 845 --episode 827 \
        --pair "필터 효과:res,res_f:146:clear,hc" --pair "회복 정책:one,ours:914:hc,variant" \
        --pair "PhaseII:res_f,ours:972:dist,lam"

Outputs (each .png at --dpi and .pdf):
  grid, grid_NAME  rows = episodes (배치), columns = configurations: top views with a shared crop per row
  episode_<i>_col  one episode, the configurations in 2 x 2 (column width); episode_<i>_page: in one row (page width)
  signals_<i>      one episode, every configuration on common time axes (target distance, clearance, h_c)
  pair_<name>_<i>  two configurations side by side with the signals chosen for the claim (clear | hc | dist | lam |
                   variant), column width
  episodes.json    the drawn outcome and time of every configuration and episode
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import strike_episode_viz as viz  # noqa: E402

# the draft's figure colours: navy (ours), brown (baselines), grey-blue (obstacles); one colour per configuration
COLORS = ["#7d8796", "#a46824", "#7a5aa6", "#185888", "#5f8f3a", "#d9a441"]
CHASER, TARGET = "#c0392b", "#2e8b57"
OBS_FILL, OBS_EDGE, OUTSIDE, HATCH = "#b4bcc8", "#6b7380", "#efefef", "#c4c4c4"
CORR_C, NOCERT_C = "#f2cf6b", "#ef8a7a"
INK, INK2 = "#1a1a1a", "#5a5a5a"

TERMS = {
    "ko": {"strike": "성공", "crash_obstacle": "충돌(장애물)", "crash_floor": "충돌(바닥)", "crash_chaser": "충돌(요격기)",
           "out_of_arena": "운용 영역 이탈", "timeout": "시간 초과", "episode": "배치",
           "drone": "드론 경로", "chaser": "요격기 경로", "target": "목표 대상 경로", "obstacle": "장애물",
           "outside": "운용 영역 밖", "start": "출발점", "tick": "1초 간격", "corr": "필터 수정 구간 (λ < 1)",
           "nocert": "인증 상실", "committed": "이전 계획 유지", "contact": "충돌", "t": "시간 [s]",
           "dist": "목표 대상까지\n거리 [m]", "clear": "최소 이격\n거리 [m]", "hc": "요격기 여유도\n$h_c$ [m]",
           "lam": "혼합률 λ", "eta": "정규화 수정량\n$\\|\\eta_k\\|$", "variant": "회복 정책\n번호 $j$", "r_s": "성공 반경 0.6 m", "hc_min": "$h_c$ = 0.2 m",
           "learned": "학습 기반", "lqr": "LQR 기반"},
    "en": {"strike": "strike", "crash_obstacle": "contact (obstacle)", "crash_floor": "contact (floor)",
           "crash_chaser": "contact (interceptor)", "out_of_arena": "leaves the arena", "timeout": "time up",
           "episode": "scene", "drone": "drone path", "chaser": "interceptor path", "target": "target path",
           "obstacle": "obstacle", "outside": "outside the arena", "start": "start", "tick": "1-s marks",
           "corr": "filter correction (λ < 1)", "nocert": "certificate lost", "committed": "committed plan",
           "contact": "contact", "t": "time [s]", "dist": "target\ndistance [m]", "clear": "min.\nclearance [m]",
           "hc": "interceptor\nmargin $h_c$ [m]", "lam": "blend λ", "eta": "normalised\ncorrection $\\|\\eta_k\\|$", "variant": "backup\nindex $j$",
           "r_s": "strike radius 0.6 m", "hc_min": "$h_c$ = 0.2 m", "learned": "learned", "lqr": "LQR"},
}
CM = 1 / 2.54


# ----------------------------------------------------------------------------------------------------------- style
def setup(terms, font_pt):
    lang = viz.setup_style("ko" if terms == "ko" else "en")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": font_pt, "axes.titlesize": font_pt, "axes.labelsize": font_pt,
                         "legend.fontsize": font_pt - 0.5, "xtick.labelsize": font_pt - 0.5,
                         "ytick.labelsize": font_pt - 0.5, "axes.grid": False, "lines.linewidth": 1.1,
                         "axes.linewidth": 0.6, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
                         "xtick.major.size": 2.0, "ytick.major.size": 2.0})
    if lang == "ko":
        for name in ("Noto Sans CJK KR", "NanumGothic", "Malgun Gothic"):
            from matplotlib import font_manager
            if name in {f.name for f in font_manager.fontManager.ttflist}:
                plt.rcParams["font.family"] = [name, "DejaVu Sans"]
                break
    return TERMS[terms if lang == "ko" else "en"]


def paper_variant(label: str, T) -> str:
    """'learned + escape (2 v_c, 60 deg)' -> '학습 기반 (2, +60°)' (the paper's notation: speed ratio, direction)."""
    import re
    m = re.match(r"(learned|hover LQR) \+ escape \(([\d.]+) v_c, ([+-]?[\d.]+) deg\)", label)
    if not m:
        return label
    ang = float(m.group(3))
    return (f"{T['learned'] if m.group(1) == 'learned' else T['lqr']} ({float(m.group(2)):g}, "
            + ("0°" if ang == 0 else f"{ang:+.0f}°") + ")")


def save(fig, path: Path, dpi):
    import matplotlib.pyplot as plt
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=dpi, facecolor="white")
    fig.savefig(path.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)


# ----------------------------------------------------------------------------------------------------------- data
class Config(viz.Panel):
    def __init__(self, spec: str, color: str):
        parts = spec.split(":")
        if len(parts) != 4:
            raise SystemExit(f"--config {spec!r}: expected RUN_DIR:MODE:LABEL:KEY")
        super().__init__(spec, color)
        self.color = color                       # also for the unfiltered run (viz.Panel greys it)
        self.key = parts[3]
        z = np.load(Path(parts[0]) / "strike_filter_traj.npz", allow_pickle=False)
        key = f"{self.mode}/correction"
        self.eta = z[key] if key in z.files else None

    def signals(self, i):
        s = super().signals(i)
        j, k = self.ep(i)
        s["eta"] = self.eta[:k, j] if self.eta is not None else np.zeros(k)
        return s

    def result(self, i, T):
        j, k = self.ep(i)
        return T[self.code(i)], k * self.dt

    def text(self, i, T):
        name, t = self.result(i, T)
        return f"{name}, {t:.1f} s" if self.code(i) != "timeout" else f"{name} ({t:.0f} s)"


def crop(cfgs, i, aspect, pad, min_w):
    """A box (x0, x1, y0, y1) holding every configuration's path, the interceptors and the target, with the axes'
    aspect (width / height), kept inside the arena plus a margin where possible."""
    pts = []
    for q in cfgs:
        j, k = q.ep(i)
        pts += [q.pos[:k, j, :2], q.tg[:k, j, :2]]
        for c in range(q.ch.shape[2]):
            if q.on[j, c] > 0.5:
                pts.append(q.ch[:k, j, c, :2])
    P = np.concatenate(pts)
    ax_, ay = cfgs[0].arena[0], cfgs[0].arena[1]
    lo = np.maximum(P.min(0) - pad, [-1.0, -ay - 1.0])
    hi = np.minimum(P.max(0) + pad, [ax_ + 1.0, ay + 1.0])
    w, h = max(hi[0] - lo[0], min_w), max(hi[1] - lo[1], min_w / aspect)
    if w / h < aspect:
        w = h * aspect
    else:
        h = w / aspect
    c = (lo + hi) / 2
    x0, y0 = c[0] - w / 2, c[1] - h / 2
    # slide the box back over the arena where it overhangs on one side only
    for lo_a, hi_a, v, s in ((-1.0, ax_ + 1.0, "x", w), (-ay - 1.0, ay + 1.0, "y", h)):
        v0 = x0 if v == "x" else y0
        if s <= hi_a - lo_a:
            v0 = min(max(v0, lo_a), hi_a - s)
        if v == "x":
            x0 = v0
        else:
            y0 = v0
    return x0, x0 + w, y0, y0 + h


def shade_outside(ax, A, big=50.0):
    """The arena boundary (walls) and the region outside it, hatched."""
    from matplotlib.patches import Rectangle
    for r in ((-big, -big, big, 2 * big + A[1]), (A[0], -big, big, 2 * big + A[1]), (-big, -big, 2 * big + A[0], big - A[1]),
              (-big, A[1], 2 * big + A[0], big)):
        ax.add_patch(Rectangle(r[:2], r[2], r[3], fc=OUTSIDE, ec=HATCH, hatch="////", lw=0, zorder=0.5))
    ax.plot([0, A[0], A[0], 0, 0], [-A[1], -A[1], A[1], A[1], -A[1]], color=INK2, lw=0.9, zorder=1)


def draw_top(ax, q: Config, i, box, T, ticks=True, tick_dt=1.0, corner="ll"):
    from matplotlib.patches import Circle, Polygon
    j, k = q.ep(i)
    x0, x1, y0, y1 = box
    ax.set_facecolor("white")
    shade_outside(ax, q.arena)
    for o in q.obst[j]:
        if o[6] > 0.5:
            ax.add_patch(Polygon(viz.outline(o), closed=True, fc=OBS_FILL, ec=OBS_EDGE, lw=0.6, zorder=2))
    # the filter's corrections under the path
    p = q.pos[:k, j]
    if q.mode != "none" and q.committed is not None:
        lam, com, noc = q.lam[:k, j], q.committed[:k, j] > 0.5, q.nocert[:k, j] > 0.5
        for mask, col in (((lam < 0.999) | com) & ~noc, CORR_C), (noc, NOCERT_C):
            if mask.any():
                m = mask.copy()
                m[1:] |= mask[:-1]                     # draw each flagged step's segment
                xs = np.where(m, p[:, 0], np.nan)
                ys = np.where(m, p[:, 1], np.nan)
                ax.plot(xs, ys, color=col, lw=4.2, solid_capstyle="round", alpha=0.9, zorder=3)
    step = max(int(round(tick_dt / q.dt)), 1)
    ax.plot(q.tg[:k, j, 0], q.tg[:k, j, 1], color=TARGET, lw=0.9, zorder=4)
    ax.add_patch(Circle(q.tg[k - 1, j, :2], q.r_s, fc=TARGET, alpha=0.25, ec=TARGET, lw=0.6, zorder=4))
    for c in range(q.ch.shape[2]):
        if q.on[j, c] > 0.5:
            ch = q.ch[:k, j, c]
            ax.plot(ch[:, 0], ch[:, 1], color=CHASER, lw=0.9, ls=(0, (3, 1.6)), zorder=4)
            ax.plot(ch[::step, 0], ch[::step, 1], "o", color=CHASER, ms=1.6, zorder=4)
            ax.add_patch(Circle(ch[-1, :2], q.r_c, fc=CHASER, alpha=0.35, ec=CHASER, lw=0.6, zorder=6))
    ax.plot(p[:, 0], p[:, 1], color=q.color, lw=1.4, zorder=5)
    ax.plot(p[::step, 0], p[::step, 1], "o", color=q.color, ms=1.8, zorder=5)
    ax.plot(*p[0, :2], "o", color=INK, ms=3.2, zorder=8)
    end_marker(ax, q.code(i), p[-1, :2], q.color)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.5)
        s.set_color(INK2)
    if ticks:
        w = x1 - x0
        d = 2 if w < 9 else (5 if w < 30 else 10)
        ax.set_xticks(np.arange(np.ceil(x0 / d) * d, x1 + 1e-6, d))
        ax.set_yticks(np.arange(np.ceil(y0 / d) * d, y1 + 1e-6, d))
        ax.tick_params(pad=1.2)
    else:
        ax.set_xticks([])
        ax.set_yticks([])
        scale_bar(ax, box, corner)


SB_W, SB_H = 0.34, 0.2          # the scale bar's corner region (fractions of the view)


def scale_corner(cfgs, i, box):
    """The corner of the view (ll, lr, ul, ur) with the least drawn content (paths, obstacles, outside the arena) for
    the scale bar; the same for every configuration of the episode."""
    x0, x1, y0, y1 = box
    w, h = x1 - x0, y1 - y0
    pts = []
    for q in cfgs:
        j, k = q.ep(i)
        pts += [q.pos[:k, j, :2], q.tg[:k, j, :2]]
        pts += [q.ch[:k, j, c, :2] for c in range(q.ch.shape[2]) if q.on[j, c] > 0.5]
    q = cfgs[0]
    j = q.ep(i)[0]
    pts += [viz.outline(o, 60) for o in q.obst[j] if o[6] > 0.5]
    for o in q.obst[j]:                                    # obstacle interiors
        if o[6] > 0.5:
            pts.append(np.array([o[:2]]).repeat(20, 0))
    P = np.concatenate(pts)
    gx, gy = np.meshgrid(np.linspace(0, 1, 9), np.linspace(0, 1, 6))
    best, score = "ll", None
    for c in ("ll", "lr", "ul", "ur"):
        cx0 = x0 if c[1] == "l" else x1 - SB_W * w
        cy0 = y0 if c[0] == "l" else y1 - SB_H * h
        n = np.sum((P[:, 0] >= cx0) & (P[:, 0] <= cx0 + SB_W * w) & (P[:, 1] >= cy0) & (P[:, 1] <= cy0 + SB_H * h))
        X, Y = cx0 + gx.ravel() * SB_W * w, cy0 + gy.ravel() * SB_H * h
        outside = np.mean((X < 0) | (X > q.arena[0]) | (Y < -q.arena[1]) | (Y > q.arena[1]))
        sc = n + 200 * outside
        if score is None or sc < score - 1e-9:
            best, score = c, sc
    return best


def scale_bar(ax, box, corner="ll"):
    x0, x1, y0, y1 = box
    w = x1 - x0
    d = 2 if w < 9 else (5 if w < 30 else 10)
    bx = x0 + 0.05 * w if corner[1] == "l" else x1 - 0.05 * w - d
    by = y0 + 0.07 * (y1 - y0) if corner[0] == "l" else y1 - 0.14 * (y1 - y0)
    ax.plot([bx, bx + d], [by, by], color=INK, lw=1.4, solid_capstyle="butt", zorder=9)
    ax.text(bx + d / 2, by + 0.025 * (y1 - y0), f"{d} m", ha="center", va="bottom", fontsize=5.8, zorder=9,
            bbox=dict(boxstyle="square,pad=0.08", fc="white", ec="none", alpha=0.8))


def end_marker(ax, code, xy, color, small=False):
    f = 0.7 if small else 1.0
    if code == "strike":
        ax.plot(*xy, "*", color="#f2b705", mec=INK, mew=0.5, ms=8.5 * f, zorder=10, clip_on=False)
    elif code == "timeout":
        ax.plot(*xy, "s", color="white", mec=INK, mew=0.8, ms=4.5 * f, zorder=10, clip_on=False)
    elif code == "out_of_arena":
        ax.plot(*xy, "D", color="white", mec=INK, mew=0.8, ms=4.5 * f, zorder=10, clip_on=False)
    else:
        ax.plot(*xy, "X", color="#d62728", mec=INK, mew=0.5, ms=7 * f, zorder=10, clip_on=False)


def legend_handles(T, cfgs=None, corr=True, minimal=False):
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    hs = []
    if cfgs:
        hs += [Line2D([], [], color=q.color, lw=1.4, label=q.label) for q in cfgs]
    hs += [Line2D([], [], color=CHASER, lw=0.9, ls=(0, (3, 1.6)), label=T["chaser"]),
           Line2D([], [], color=TARGET, lw=0.9, label=T["target"]),
           Patch(fc=OBS_FILL, ec=OBS_EDGE, lw=0.6, label=T["obstacle"])]
    if not minimal:
        hs += [Patch(fc=OUTSIDE, ec=HATCH, hatch="////", lw=0.4, label=T["outside"]),
               Line2D([], [], marker="o", ls="", color=INK, ms=3.2, label=T["start"]),
               Line2D([], [], marker="o", ls="", color=INK2, ms=1.8, label=T["tick"])]
    if corr:
        hs += [Line2D([], [], color=CORR_C, lw=4.2, label=T["corr"]), Line2D([], [], color=NOCERT_C, lw=4.2, label=T["nocert"])]
    hs += [Line2D([], [], marker="*", ls="", color="#f2b705", mec=INK, mew=0.5, ms=8.5, label=T["strike"]),
           Line2D([], [], marker="X", ls="", color="#d62728", mec=INK, mew=0.5, ms=7, label=T["contact"]),
           Line2D([], [], marker="s", ls="", color="white", mec=INK, mew=0.8, ms=4.5, label=T["timeout"])]
    return hs


# ----------------------------------------------------------------------------------------------------------- figures
def fig_grid(cfgs, eps, T, a, out, stem="grid"):
    """Rows = episodes, columns = configurations (page width)."""
    import matplotlib.pyplot as plt
    W = a.page_cm * CM
    left, right, top, hgap, vgap, title_h = 0.22, 0.04, 0.30, 0.06, 0.06, 0.15
    ncol, nrow = len(cfgs), len(eps)
    cw = (W - left - right - (ncol - 1) * hgap) / ncol
    ch = cw / a.aspect
    leg_h = 0.55
    H = top + nrow * (title_h + ch) + (nrow - 1) * vgap + leg_h
    fig = plt.figure(figsize=(W, H))
    letters = "abcdefghijklmnopqrstuvwxyz"
    for r, i in enumerate(eps):
        box = crop(cfgs, i, a.aspect, a.pad, a.min_w)
        corner = scale_corner(cfgs, i, box)
        y = H - top - r * (title_h + ch + vgap) - title_h - ch
        for c, q in enumerate(cfgs):
            x = left + c * (cw + hgap)
            ax = fig.add_axes((x / W, y / H, cw / W, ch / H))
            draw_top(ax, q, i, box, T, ticks=False, corner=corner)
            ax.set_title(q.text(i, T), fontsize=a.font - 0.5, pad=1.5, loc="center",
                         color="#b0281f" if q.code(i).startswith("crash") else INK)
        fig.text(0.08 / W, (y + ch / 2) / H, f"({letters[r]}) {T['episode']} {i}", rotation=90, ha="left", va="center",
                 fontsize=a.font, fontweight="bold")
    for c, q in enumerate(cfgs):
        x = left + c * (cw + hgap) + cw / 2
        fig.text(x / W, (H - 0.1) / H, q.label, ha="center", va="top", fontsize=a.font, fontweight="bold", color=q.color)
    fig.legend(handles=legend_handles(T), loc="lower center", ncol=6, frameon=False, bbox_to_anchor=(0.5, 0.0),
               handlelength=1.8, columnspacing=1.0, handletextpad=0.5)
    save(fig, out / stem, a.dpi)


def fig_episode(cfgs, i, T, a, out, cols, stem, width_cm):
    import matplotlib.pyplot as plt
    W = width_cm * CM
    n = len(cfgs)
    ncol = min(cols, n)
    nrow = int(np.ceil(n / ncol))
    left, right, top, hgap, vgap, title_h = 0.05, 0.05, 0.05, 0.07, 0.05, 0.27
    cw = (W - left - right - (ncol - 1) * hgap) / ncol
    ch = cw / a.aspect
    leg_rows = int(np.ceil(11 / (6 if ncol > 2 else 3)))
    leg_h = 0.13 * leg_rows + 0.08
    H = top + nrow * (title_h + ch) + (nrow - 1) * vgap + leg_h
    fig = plt.figure(figsize=(W, H))
    box = crop(cfgs, i, a.aspect, a.pad, a.min_w)
    corner = scale_corner(cfgs, i, box)
    letters = "abcdefgh"
    for k, q in enumerate(cfgs):
        r, c = divmod(k, ncol)
        x = left + c * (cw + hgap)
        y = H - top - r * (title_h + ch + vgap) - title_h - ch
        ax = fig.add_axes((x / W, y / H, cw / W, ch / H))
        draw_top(ax, q, i, box, T, ticks=False, corner=corner)
        ax.set_title(f"({letters[k]}) {q.label}\n{q.text(i, T)}", fontsize=a.font - 0.3, pad=1.5, loc="left",
                     linespacing=1.15)
    fig.legend(handles=legend_handles(T), loc="lower center", ncol=6 if ncol > 2 else 3, frameon=False,
               bbox_to_anchor=(0.5, 0.0), handlelength=1.6, columnspacing=0.8, handletextpad=0.4,
               fontsize=a.font - 1.0)
    save(fig, out / stem, a.dpi)


def plot_signal(ax, q: Config, i, key, T, lw=1.1, show_ref=True):
    s = q.signals(i)
    t = s["t"]
    y = {"dist": s["dist_t"], "clear": s["clear"], "hc": s["hc"], "lam": s["lam"], "eta": s["eta"]}[key]
    if key == "lam":
        ax.step(t, y, where="post", color=q.color, lw=lw * 0.9)
    elif key == "eta":
        ax.fill_between(t, 0, y, color=q.color, alpha=0.35, lw=0, step="post")
        ax.step(t, y, where="post", color=q.color, lw=lw * 0.6)
    else:
        ax.plot(t, y, color=q.color, lw=lw)
    if show_ref:
        if key == "dist":
            ax.axhline(q.r_s, color=INK2, lw=0.6, ls=":")
        elif key == "clear":
            ax.axhline(0, color="#d62728", lw=0.6)
        elif key == "hc":
            ax.axhline(0, color="#d62728", lw=0.6)
            ax.axhline(q.fresh_m, color=INK2, lw=0.6, ls=":")
    return q.code(i), t[-1], y[-1]


def variant_panel(ax, q: Config, i, n_var, offset=0.0):
    """The applied plan's backup index j at the steps the filter changed the command (lambda < 1, committed plan or
    no certificate); at lambda = 1 the index carries no meaning."""
    s = q.signals(i)
    corr = (s["lam"] < 0.999) | (s["status"] > 0)
    v = s["variant"].astype(float)
    ax.plot(s["t"], np.where(corr & (v >= 0), v + offset, np.nan), "s", color=q.color, ms=1.5, mew=0)
    ax.set_ylim(-0.6, n_var - 0.4)
    ax.set_yticks(range(n_var))


def signal_stack(fig, W, H, x0, x1, y_top, keys, cfgs, i, T, a, sig_h, sig_gap, n_var):
    """Signals of the configurations on common time axes, stacked downwards from y_top [in]; returns the axes."""
    import matplotlib.pyplot as plt
    t_end = max(q.ep(i)[1] * q.dt for q in cfgs)
    axes, y = [], y_top
    filtered = [q for q in cfgs if q.mode != "none"]
    for r, key in enumerate(keys):
        y -= sig_h + (sig_gap if r else 0.0)
        ax = fig.add_axes((x0 / W, y / H, (x1 - x0) / W, sig_h / H), sharex=axes[0] if axes else None)
        axes.append(ax)
        if key == "variant":
            offs = np.linspace(-0.13, 0.13, len(filtered)) if len(filtered) > 1 else [0.0]
            for q, o in zip(filtered, offs):
                variant_panel(ax, q, i, n_var, o)
        else:
            ends = [plot_signal(ax, q, i, key, T, lw=1.0, show_ref=(k == 0)) + (q.color,)
                    for k, q in enumerate(cfgs if key not in ("lam", "eta") else filtered)]
            if key == "hc":
                lo = min(np.nanmin(q.signals(i)["hc"]) for q in cfgs)
                ax.set_ylim(max(lo - 0.3, -1.0), a.hc_max)
            elif key == "clear":
                lo = min(np.nanmin(q.signals(i)["clear"]) for q in cfgs)
                ax.set_ylim(max(lo - 0.3, -0.5), None)
            elif key == "lam":
                ax.set_ylim(-0.08, 1.08)
                ax.set_yticks([0, 0.5, 1])
            elif key == "eta":
                ax.set_ylim(-0.04, None)
            lo, hi = ax.get_ylim()
            for code, te, ye, col in ends:                       # each run's outcome at its end (when inside the axes)
                if lo <= ye <= hi:
                    end_marker(ax, code, (te, ye), col, small=True)
        ax.set_ylabel(T[key], fontsize=a.font - 0.5, labelpad=2)
        ax.set_xlim(0, t_end + 0.2)
        ax.grid(True, lw=0.3, alpha=0.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if r < len(keys) - 1:
            plt.setp(ax.get_xticklabels(), visible=False)
    axes[-1].set_xlabel(T["t"], labelpad=1)
    return axes


def fig_pair(cfgs2, i, keys, T, a, out, stem, n_var, layout):
    """Two configurations: their top views (stacked at column width, or side by side at page width), then the chosen
    signals on common time axes."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    side = layout == "side"
    W = (a.page_cm if side else a.col_cm) * CM
    left, right, top, title_h, gap = 0.5, 0.06, 0.04, 0.27, 0.06
    if side:
        cw = (W - left - right - 0.15) / 2
    else:
        cw = W - left - right
    ch = cw / a.aspect
    sig_h, sig_gap = (0.62, 0.06) if side else (0.6, 0.06)
    hs = legend_handles(T, cfgs2, minimal=True)
    if "hc" in keys:
        hs.append(Line2D([], [], color=INK2, lw=0.6, ls=":", label=T["hc_min"]))
    if "dist" in keys:
        hs.append(Line2D([], [], color=INK2, lw=0.6, ls=":", label=T["r_s"]))
    ncol = 6 if side else 3
    leg_h = 0.14 * int(np.ceil(len(hs) / ncol)) + 0.06
    views_h = (title_h + ch) if side else 2 * (title_h + ch) + gap
    H = top + views_h + 0.1 + len(keys) * (sig_h + sig_gap) + 0.28 + leg_h
    fig = plt.figure(figsize=(W, H))
    box = crop(cfgs2, i, a.aspect, a.pad, a.min_w)
    corner = scale_corner(cfgs2, i, box)
    for k, q in enumerate(cfgs2):
        if side:
            x, y = left + k * (cw + 0.15), H - top - title_h - ch
        else:
            x, y = left, H - top - (k + 1) * (title_h + ch) - k * gap
        ax = fig.add_axes((x / W, y / H, cw / W, ch / H))
        draw_top(ax, q, i, box, T, ticks=False, corner=corner)
        ax.set_title(f"({'ab'[k]}) {q.label}: {q.text(i, T)}", fontsize=a.font - 0.3, pad=1.5, loc="left")
    signal_stack(fig, W, H, left, W - right, H - top - views_h - 0.1, keys, cfgs2, i, T, a, sig_h, sig_gap, n_var)
    fig.legend(handles=hs, loc="lower center", ncol=ncol, frameon=False, bbox_to_anchor=(0.5, 0.0), handlelength=1.6,
               columnspacing=0.8, handletextpad=0.4, fontsize=a.font - 1.0)
    save(fig, out / stem, a.dpi)


def fig_signals(cfgs, i, keys, T, a, out, n_var):
    """One episode, every configuration on common time axes (column width)."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    W = a.col_cm * CM
    left, right, top, sig_h, sig_gap, leg_h = 0.5, 0.06, 0.2, 0.62, 0.06, 0.36
    H = top + len(keys) * (sig_h + sig_gap) + 0.28 + leg_h
    fig = plt.figure(figsize=(W, H))
    axes = signal_stack(fig, W, H, left, W - right, H - top, keys, cfgs, i, T, a, sig_h, sig_gap, n_var)
    axes[0].set_title(f"{T['episode']} {i}", loc="left", fontsize=a.font, fontweight="bold", pad=2)
    hs = [Line2D([], [], color=q.color, lw=1.2, label=q.label) for q in cfgs]
    fig.legend(handles=hs, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.0), handlelength=1.6,
               columnspacing=1.0, fontsize=a.font - 0.8)
    save(fig, out / f"signals_{i}", a.dpi)


# ----------------------------------------------------------------------------------------------------------- main
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", action="append", required=True, help='"RUN_DIR:MODE:LABEL:KEY" (repeat; order = columns)')
    p.add_argument("--colors", nargs="*", default=None, help="one colour per --config (default: the paper palette)")
    p.add_argument("--grid", type=int, nargs="*", default=[], help="episodes of the grid figure (rows)")
    p.add_argument("--grid_spec", action="append", default=[], help='further grids "NAME:EP,EP,..." (grid_NAME)')
    p.add_argument("--episode", type=int, nargs="*", default=[], help="episodes drawn alone (2 x 2, one row, signals)")
    p.add_argument("--pair", action="append", default=[],
                   help='"NAME:KEY_A,KEY_B:EPISODE[,EPISODE...]:SIG[,SIG...]" with SIG in clear hc dist lam eta variant')
    p.add_argument("--signals", default="dist,clear,hc", help="signals of the per-episode signal figure")
    p.add_argument("--pair_layouts", nargs="*", default=["stack", "side"],
                   help="pair figures: stack (column width, views stacked) and/or side (page width, side by side)")
    p.add_argument("--terms", choices=("ko", "en"), default="ko")
    p.add_argument("--col_cm", type=float, default=8.0, help="column width [cm]")
    p.add_argument("--page_cm", type=float, default=17.0, help="page (two-column) width [cm]")
    p.add_argument("--aspect", type=float, default=1.9, help="width / height of the top views")
    p.add_argument("--pad", type=float, default=1.5, help="margin [m] around the drawn paths")
    p.add_argument("--min_w", type=float, default=10.0, help="smallest top-view width [m]")
    p.add_argument("--hc_max", type=float, default=6.0, help="upper limit of the h_c axes [m]")
    p.add_argument("--font", type=float, default=7.0, help="font size [pt]")
    p.add_argument("--dpi", type=int, default=600)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)

    T = setup(a.terms, a.font)
    cols = a.colors or COLORS
    if len(cols) < len(a.config):
        raise SystemExit("--colors: one colour per --config")
    cfgs = [Config(s, cols[k]) for k, s in enumerate(a.config)]
    by_key = {q.key: q for q in cfgs}
    grids = ([("grid", a.grid)] if a.grid else []) + [(f"grid_{g.split(':')[0]}", [int(e) for e in g.split(":")[1].split(",")])
                                                     for g in a.grid_spec]
    eps_all = sorted({e for _, g in grids for e in g} | set(a.episode)
                     | {int(e) for spec in a.pair for e in spec.split(":")[2].split(",")})
    for i in eps_all:
        missing = [q.key for q in cfgs if i not in q.idx]
        if missing:
            raise SystemExit(f"episode {i} is not in the replays of {missing}")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    labels = []
    n_var = 1
    for q in cfgs:
        if q.mode != "none" and len(q.variant_labels) > n_var:
            n_var = len(q.variant_labels)
            labels = [paper_variant(lb, T) for lb in q.variant_labels]

    for stem, g in grids:
        fig_grid(cfgs, g, T, a, out, stem)
    for i in a.episode:
        fig_episode(cfgs, i, T, a, out, 2, f"episode_{i}_col", a.col_cm)
        fig_episode(cfgs, i, T, a, out, len(cfgs), f"episode_{i}_page", a.page_cm)
        fig_signals(cfgs, i, a.signals.split(","), T, a, out, n_var)
    for spec in a.pair:
        name, keys, eps, sigs = spec.split(":")
        two = [by_key[k] for k in keys.split(",")]
        for e in eps.split(","):
            for lay in a.pair_layouts:
                fig_pair(two, int(e), sigs.split(","), T, a, out, f"pair_{name}_{e}_{'col' if lay == 'stack' else 'page'}",
                         n_var, lay)
    rec = {str(i): {q.key: {"label": q.label, "outcome": q.code(i), "outcome_text": q.result(i, T)[0],
                            "time_s": round(q.ep(i)[1] * q.dt, 2)} for q in cfgs} for i in eps_all}
    rec["backup_index_j"] = {str(k): lb for k, lb in enumerate(labels)}
    (out / "episodes.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False))
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
