#!/usr/bin/env python
"""Detailed figures of specific strike episodes from evaluation replays (CPU; no simulation).

Each replay is an output folder of scripts/eval_strike_filter.py run with --save_traj 1 and the same --record_episodes
(or --only_episodes) list (docker_batch/run_strike_paper.sh makes them). Panels are given as "RUN_DIR:MODE:LABEL[:TAG]".

    python scripts/strike_episode_viz.py --out outputs/strike_final/paper/episodes --lang ko \
        --panel "R/ps2:none:PS2 (필터 없음):ps2_none" --panel "R/ps2:learned:PS2 + 최종 필터:ps2_ours" \
        --panel "R/pursuit:learned:기준 접근 제어 + 최종 필터:pursuit_ours" --main ps2_ours --gif3d ps2_ours ps2_none

Per episode i (folder ep<i>/):
  compare2d.png/.pdf     top views of every panel (paths, chaser, target, outcome)
  compare2d.gif          the same, animated
  signals.png/.pdf       key signals of every panel on common time axes (target distance, clearance, chaser margin h_c)
  detail_<tag>.png/.pdf  one panel in detail: distances, clearances, h_c and reflex weight, speed / altitude / tilt,
                         kept fraction with the certificate status and the backup variant, nominal vs applied command
  view3d.png/.pdf        3-D view: obstacles as pillars, the panels' paths, the main panel's chaser and target
  view3d_<tag>.png       3-D view of one panel, its path colored by speed
  anim3d_<tag>.gif       3-D animation of one panel (attitude-aware quadrotor, chaser with its h_c = 0 sphere, target)
and overview.png/.pdf: every episode's top view with the panels' paths (paper figure; overview_main.png with only the
--overview_main panels).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import warnings

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

INK, INK2, SURF = "#0b0b0b", "#52514e", "#fcfcfb"
CHASER, TARGET, OBS_FILL, OBS_EDGE = "#1baf7a", "#eb6834", "#c9c7c0", "#6b6a65"
PALETTE = ["#2a78d6", "#b07d12", "#d1477a", "#1b8a6b", "#7a5195", "#e07b39", "#4f6bed"]
NONE_C = "#3d3c39"
STATUS_C = {"fresh": "#ffffff", "committed": "#f3d27a", "nocert": "#e8806f"}

TXT = {
    "ko": {"strike": "성공", "crash_obstacle": "장애물 접촉", "crash_floor": "바닥 접촉", "crash_chaser": "추격자 접촉",
           "out_of_arena": "영역 이탈", "timeout": "시간초과", "t": "시간 [s]", "dist_t": "표적 거리 [m]",
           "clear": "여유 거리 [m]", "obst": "장애물", "wall": "벽", "floor": "바닥", "ceil": "천장", "min": "최소",
           "hc": "추격자 여유도 $h_c$ [m]", "range": "추격자 거리 여유 $R-(r_c+r_d)$", "sigma": "회피 반사 가중치 σ",
           "speed": "속력 [m/s]", "alt": "고도 [m]", "tilt": "기울기 [°]", "kept": "명목 유지율 λ",
           "fresh": "새 인증", "committed": "이전 계획 유지", "nocert": "인증 없음", "variant": "선택된 회복 정책",
           "thrust": "추력 가속도 $a_T$ [m/s²]", "rate": "각속도 크기 $\\|\\omega\\|$ [rad/s]", "nominal": "명목",
           "applied": "실제", "episode": "에피소드", "chaser_path": "추격자 경로", "target_path": "표적 경로",
           "contact": "접촉", "timeup": "시간초과", "margin02": "새 계획 여유 0.2 m", "strike_r": "성공 반경 0.6 m",
           "fresh_margin": "새 계획 여유 0.15 m", "x": "x [m]", "y": "y [m]", "z": "z [m]",
           "hc_sphere": "점선 구: $h_c=0$ 경계", "speed_c": "속력 [m/s]"},
    "en": {"strike": "strike", "crash_obstacle": "hits obstacle", "crash_floor": "hits floor", "crash_chaser": "hit by chaser",
           "out_of_arena": "leaves arena", "timeout": "time up", "t": "time [s]", "dist_t": "target distance [m]",
           "clear": "clearance [m]", "obst": "obstacle", "wall": "wall", "floor": "floor", "ceil": "ceiling", "min": "min",
           "hc": "chaser margin $h_c$ [m]", "range": "chaser range margin $R-(r_c+r_d)$", "sigma": "reflex weight σ",
           "speed": "speed [m/s]", "alt": "altitude [m]", "tilt": "tilt [deg]", "kept": "kept fraction λ",
           "fresh": "new certificate", "committed": "committed plan", "nocert": "no certificate", "variant": "backup variant",
           "thrust": "thrust accel. $a_T$ [m/s²]", "rate": "body rate $\\|\\omega\\|$ [rad/s]", "nominal": "nominal",
           "applied": "applied", "episode": "episode", "chaser_path": "chaser path", "target_path": "target path",
           "contact": "contact", "timeup": "time up", "margin02": "new-plan margin 0.2 m", "strike_r": "strike radius 0.6 m",
           "fresh_margin": "new-plan margin 0.15 m", "x": "x [m]", "y": "y [m]", "z": "z [m]",
           "hc_sphere": "dashed sphere: $h_c = 0$", "speed_c": "speed [m/s]"},
}


# ----------------------------------------------------------------------------------------------------------- geometry
def outline(o, n=90):
    cx, cy, bx, by, yaw, rr = o[:6]
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    if bx == 0 and by == 0:
        return np.stack([cx + rr * np.cos(th), cy + rr * np.sin(th)], 1)
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[-bx, -by], [bx, -by], [bx, by], [-bx, by]]) @ np.array([[c, s], [-s, c]]) + [cx, cy]


def sdf_obstacles(pts, obst):
    """Signed distance of points (T, 2) to the present rounded rectangles (strike_env._np_sdf) -> (T,)."""
    o = obst[obst[:, 6] > 0.5]
    if len(o) == 0:
        return np.full(len(pts), 1e3)
    cx, cy, bx, by, yaw, rr = (o[:, i:i + 1] for i in range(6))
    c, s = np.cos(yaw), np.sin(yaw)
    dx, dy = pts[None, :, 0] - cx, pts[None, :, 1] - cy
    qx, qy = np.abs(c * dx + s * dy) - bx, np.abs(-s * dx + c * dy) - by
    d = np.hypot(np.maximum(qx, 0), np.maximum(qy, 0)) + np.minimum(np.maximum(qx, qy), 0) - rr
    return d.min(axis=0)


def quat_to_R(q):
    """Body-to-world rotation matrices (T, 3, 3) of quaternions (T, 4) in (w, x, y, z)."""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.stack([np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
                     np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
                     np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1)], 1)


# ----------------------------------------------------------------------------------------------------------- data
class Panel:
    def __init__(self, spec: str, color: str):
        parts = spec.split(":")
        if len(parts) < 3:
            raise SystemExit(f"--panel {spec!r}: expected RUN_DIR:MODE:LABEL[:TAG]")
        d, mode, label = parts[0], parts[1], parts[2]
        self.tag = parts[3] if len(parts) > 3 else f"{Path(d).name}_{mode}"
        z = np.load(Path(d) / "strike_filter_traj.npz", allow_pickle=False)
        if f"{mode}/pos" not in z.files:
            raise SystemExit(f"{d}: mode {mode!r} not in the replay (modes {list(z['modes'])})")
        self.label, self.mode = label, mode
        self.color = NONE_C if mode == "none" else color
        self.idx = {int(e): j for j, e in enumerate(z["episode_index"])}
        g = lambda k, default=None: z[k] if k in z.files else default
        self.obst, self.on = z["obst"], z["chaser_on"]
        self.pos, self.ch, self.tg = z[f"{mode}/pos"], z[f"{mode}/chaser"], z[f"{mode}/target"]
        self.lam, self.alive, self.vel = z[f"{mode}/lam"], z[f"{mode}/alive"], z[f"{mode}/vel"]
        self.outcome, self.end = z[f"{mode}/outcome"], z[f"{mode}/end_step"]
        self.quat = g(f"{mode}/quat")
        self.u, self.u_nom = g(f"{mode}/u"), g(f"{mode}/u_nom")
        self.nocert, self.committed = g(f"{mode}/nocert"), g(f"{mode}/committed")
        self.variant, self.held = g(f"{mode}/variant"), g(f"{mode}/held")
        self.names = [str(s) for s in z["outcomes"]]
        self.dt, self.arena = float(z["dt"]), z["arena"]
        self.r_d, self.r_s = float(z["r_drone"]), float(z["strike_radius"])
        self.r_c, self.v_c = float(z["chaser_radius"]), float(z["chaser_speed"])
        self.guidance = str(z["chaser_model"]) == "guidance"
        self.delay, self.a_e = float(z["delay"]), float(z["a_evade"])
        self.z_floor, self.z_max = float(g("z_floor", 1.0)), float(g("z_max", 10.0))
        self.t_on, self.t_off = float(g("trigger_on", 0.5)), float(g("trigger_off", 2.0))
        self.fresh_m, self.fresh_om = float(g("fresh_margin", 0.2)), float(g("fresh_obstacle_margin", 0.15))
        self.variant_labels = [str(s) for s in g("variant_labels", np.array(["-"]))]

    def ep(self, i):
        j = self.idx[i]
        return j, max(int(self.alive[:, j].sum()), 1)

    def code(self, i):
        return self.names[int(self.outcome[self.idx[i]])]

    def verdict(self, i, L):
        j, k = self.ep(i)
        return f"{L[self.code(i)]} ({k * self.dt:.1f} s)"

    def signals(self, i):
        """Per-step signals of episode i up to its end."""
        j, k = self.ep(i)
        p, v = self.pos[:k, j], self.vel[:k, j]
        sig = {"t": np.arange(k) * self.dt, "p": p, "v": v, "speed": np.linalg.norm(v, axis=1), "alt": p[:, 2],
               "dist_t": np.linalg.norm(p - self.tg[:k, j], axis=1),
               "obst": sdf_obstacles(p[:, :2], self.obst[j]) - self.r_d,
               "wall": np.min(np.stack([p[:, 0], self.arena[0] - p[:, 0], p[:, 1] + self.arena[1], self.arena[1] - p[:, 1]]),
                              axis=0) - self.r_d,
               "floor": p[:, 2] - self.z_floor, "ceil": self.z_max - p[:, 2] - self.r_d}
        hc, rng = [], []
        for c in range(self.ch.shape[2]):
            if self.on[j, c] > 0.5:
                rel = p - self.ch[:k, j, c]
                R = np.linalg.norm(rel, axis=1) + 1e-9
                e = rel / R[:, None]
                s = np.maximum(self.v_c - np.sum(e * v, axis=1), 0.0)
                hc.append(R - (self.r_c + self.r_d) - s * self.delay - s ** 2 / (2 * self.a_e))
                rng.append(R - (self.r_c + self.r_d))
        sig["hc"] = np.min(hc, axis=0) if hc else np.full(k, np.nan)
        sig["range"] = np.min(rng, axis=0) if rng else np.full(k, np.nan)
        sig["sigma"] = np.clip((self.t_off - sig["hc"]) / (self.t_off - self.t_on), 0, 1)
        sig["clear"] = np.min(np.stack([sig["obst"], sig["wall"], sig["floor"], sig["ceil"]]), axis=0)
        sig["lam"] = self.lam[:k, j]
        if self.quat is not None:
            q = self.quat[:k, j]
            sig["quat"] = q
            sig["tilt"] = np.degrees(np.arccos(np.clip(1 - 2 * (q[:, 1] ** 2 + q[:, 2] ** 2), -1, 1)))
        if self.u is not None:
            sig["u"], sig["u_nom"] = self.u[:k, j], self.u_nom[:k, j]
            sig["status"] = np.where(self.nocert[:k, j] > 0.5, 2, np.where(self.committed[:k, j] > 0.5, 1, 0))
            sig["variant"] = self.variant[:k, j]
        return sig


# ----------------------------------------------------------------------------------------------------------- style
def short_variant(label: str, lang: str) -> str:
    """'learned + escape (2 v_c, +60 deg)' -> '학습+회피 2v_c +60°' / 'learned+esc. 2v_c +60°'."""
    import re
    m = re.match(r"(learned|hover LQR) \+ escape \(([\d.]+) v_c, ([+-]?[\d.]+) deg\)", label)
    if not m:
        return label
    base = {"learned": ("학습", "learned"), "hover LQR": ("호버 LQR", "LQR")}[m.group(1)][0 if lang == "ko" else 1]
    ang = float(m.group(3))
    return f"{base}+{'회피' if lang == 'ko' else 'esc.'} {float(m.group(2)):g}$v_c$ {ang:+.0f}°".replace("+0°", "0°")


def setup_style(lang):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 300, "font.size": 8.5, "axes.titlesize": 9,
                         "axes.labelsize": 8.5, "legend.fontsize": 7.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
                         "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25,
                         "grid.linewidth": 0.5, "axes.linewidth": 0.7, "lines.linewidth": 1.4, "pdf.fonttype": 42,
                         "axes.unicode_minus": False, "mathtext.fontset": "dejavusans"})
    if lang != "ko":
        return lang
    have = {f.name for f in font_manager.fontManager.ttflist}
    for name in ("NanumGothic", "Noto Sans CJK KR", "Noto Sans KR", "Malgun Gothic", "AppleGothic", "UnDotum", "Baekmuk Gulim",
                 "Noto Sans CJK JP", "Noto Sans CJK SC", "Noto Sans CJK TC", "Source Han Sans K"):
        if name in have:
            plt.rcParams["font.family"] = [name, "DejaVu Sans"]
            return "ko"
    try:
        import koreanize_matplotlib  # noqa: F401
        plt.rcParams["axes.unicode_minus"] = False
        return "ko"
    except ImportError:
        warnings.warn("no Korean font (apt-get install fonts-nanum or pip install koreanize-matplotlib): English labels")
        return "en"


def save(fig, path: Path, pdf=True):
    import matplotlib.pyplot as plt
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight", facecolor=fig.get_facecolor())
    if pdf:
        fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def top_view(ax, q: Panel, j, grid=True):
    from matplotlib.patches import Polygon
    ax.set_facecolor(SURF)
    for o in q.obst[j]:
        if o[6] > 0.5:
            ax.add_patch(Polygon(outline(o), closed=True, fc=OBS_FILL, ec=OBS_EDGE, lw=0.8, zorder=2))
    ax.set_xlim(-1, q.arena[0] + 1)
    ax.set_ylim(-q.arena[1] - 0.6, q.arena[1] + 0.6)
    ax.set_aspect("equal")
    ax.plot([0, q.arena[0], q.arena[0], 0, 0], [-q.arena[1], -q.arena[1], q.arena[1], q.arena[1], -q.arena[1]],
            color=INK2, lw=0.6, zorder=1)
    ax.set_xticks([0, 10, 20, 30, 40])
    ax.set_yticks([-10, 0, 10])
    ax.grid(grid)


def top_grid(n, cols, arena, *, panel_w=3.2, title_h=0.3, tick_h=0.3, left=0.4, right=0.12, gap_w=0.4, top=0.4,
             bottom=0.5, ticks=True):
    """(figure, axes, (W, H)) for n top views of the arena in a grid, placed by hand. Equal-aspect axes keep their
    exact size, so a title never runs into the tick labels of the row above (tight_layout misjudges such grids)."""
    import matplotlib.pyplot as plt
    ncol = min(cols, n)
    nrow = int(np.ceil(n / ncol))
    ph = panel_w * (2.0 * arena[1] + 1.2) / (arena[0] + 2.0)     # top_view's y extent / x extent
    th, lw_ = (tick_h, left) if ticks else (0.06, 0.1)
    gw = gap_w if ticks else 0.15
    W = lw_ + ncol * panel_w + (ncol - 1) * gw + right
    H = top + nrow * (title_h + ph + th) + bottom
    fig = plt.figure(figsize=(W, H))
    axes = []
    for k in range(n):
        r, c = divmod(k, ncol)
        x0 = lw_ + c * (panel_w + gw)
        y0 = H - top - (r + 1) * (title_h + ph + th) + th
        axes.append(fig.add_axes((x0 / W, y0 / H, panel_w / W, ph / H)))
    return fig, axes, (W, H)


def end_marker(ax, q: Panel, i, p_end, three_d=False):
    code = q.code(i)
    mk, mc, ms = {"strike": ("*", "#f2b705", 13), "timeout": ("o", "#8a8a8a", 8)}.get(code, ("X", "#c4302b", 10))
    if three_d:
        ax.scatter(*p_end, marker=mk, s=ms ** 2, c=mc, edgecolors=INK, linewidths=0.6, zorder=10, depthshade=False)
    else:
        ax.plot(*p_end[:2], mk, color=mc, mec=INK, ms=ms, zorder=9)


# ----------------------------------------------------------------------------------------------------------- figures
def fig_compare2d(panels, i, L, out, cols):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Circle
    fig, axes, (W, H) = top_grid(len(panels), cols, panels[0].arena)
    fig.patch.set_facecolor("white")
    for ax, q in zip(axes, panels):
        j, k = q.ep(i)
        top_view(ax, q, j)
        ax.plot(q.tg[:k, j, 0], q.tg[:k, j, 1], color=TARGET, lw=1.2, zorder=4)
        ax.add_patch(Circle(q.tg[k - 1, j, :2], q.r_s, fc=TARGET, ec=INK, lw=0.5, zorder=6))
        for c in range(q.ch.shape[2]):
            if q.on[j, c] > 0.5:
                ax.plot(q.ch[:k, j, c, 0], q.ch[:k, j, c, 1], color=CHASER, lw=1.0, ls="--", zorder=4)
                ax.add_patch(Circle(q.ch[k - 1, j, c, :2], q.r_c, fc=CHASER, ec=INK, lw=0.5, zorder=6))
        ax.plot(q.pos[:k, j, 0], q.pos[:k, j, 1], color=q.color, lw=1.8, zorder=5)
        ax.plot(*q.pos[0, j, :2], "o", color=INK, ms=4, zorder=8)
        end_marker(ax, q, i, q.pos[k - 1, j])
        ax.set_title(f"{q.label}: {q.verdict(i, L)}", loc="left", fontsize=8.5, color=q.color, fontweight="bold")
    fig.legend(handles=[Line2D([], [], color=CHASER, lw=1.0, ls="--", label=L["chaser_path"]),
                        Line2D([], [], color=TARGET, lw=1.2, label=L["target_path"]),
                        Line2D([], [], marker="*", ls="", ms=11, color="#f2b705", mec=INK, label=L["strike"]),
                        Line2D([], [], marker="X", ls="", ms=9, color="#c4302b", mec=INK, label=L["contact"]),
                        Line2D([], [], marker="o", ls="", ms=7, color="#8a8a8a", mec=INK, label=L["timeup"])],
               loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 0.02 / H))
    fig.suptitle(f"{L['episode']} {i}", x=0.01, y=1 - 0.06 / H, ha="left", va="top", fontsize=10)
    save(fig, out / "compare2d")


def fig_signals(panels, i, L, out):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 1, figsize=(7.0, 4.6), sharex=True)
    for q in panels:
        s = q.signals(i)
        lw = 1.7 if q.mode != "none" and q is panels[0] else 1.1
        axes[0].plot(s["t"], s["dist_t"], color=q.color, lw=lw, label=q.label)
        axes[1].plot(s["t"], s["clear"], color=q.color, lw=lw)
        axes[2].plot(s["t"], s["hc"], color=q.color, lw=lw)
        for ax, key in zip(axes, ("dist_t", "clear", "hc")):
            end_marker(ax, q, i, np.array([s["t"][-1], s[key][-1]]))
    axes[0].axhline(panels[0].r_s, color=INK2, lw=0.7, ls=":")
    axes[0].set_ylabel(L["dist_t"])
    axes[0].set_yscale("log")
    axes[1].axhline(0, color="#c4302b", lw=0.8)
    axes[1].set_ylabel(f"{L['min']} {L['clear']}")
    axes[1].set_ylim(-0.3, None)
    axes[2].axhline(0, color="#c4302b", lw=0.8)
    axes[2].axhline(panels[0].fresh_m, color=INK2, lw=0.7, ls=":")
    axes[2].set_ylabel(L["hc"])
    axes[2].set_ylim(-0.5, 8)
    axes[2].set_xlabel(L["t"])
    h, lab = axes[0].get_legend_handles_labels()
    n_leg = int(np.ceil(len(h) / 3))
    fig.legend(h, lab, frameon=False, ncol=3, fontsize=7, loc="upper right", bbox_to_anchor=(0.99, 1.0))
    fig.suptitle(f"{L['episode']} {i}", x=0.01, y=0.99, ha="left", va="top", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.028 * n_leg))
    save(fig, out / "signals")


def _status_band(ax, t, s, ymin, ymax):
    """Shade the steps run on the committed plan (yellow) and without any certificate (red)."""
    st = s["status"]
    for code, key in ((1, "committed"), (2, "nocert")):
        on = st == code
        if on.any():
            ax.fill_between(t, ymin, ymax, where=on, color=STATUS_C[key], alpha=0.85, lw=0, step="post", zorder=0)


def _variant_strip(ax, t, s, q: Panel, lang):
    """The backup variant of the applied plan as a categorical strip, with a legend of the variants used."""
    from matplotlib.patches import Patch
    vv = s["variant"].astype(int)
    used = [int(k) for k in np.unique(vv[vv >= 0])]
    for k in used:
        on = vv == k
        ax.fill_between(t, 0, 1, where=on, color=PALETTE[k % len(PALETTE)], lw=0, step="post")
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.grid(False)
    labs = [short_variant(lb, lang) for lb in q.variant_labels]
    ax.legend(handles=[Patch(color=PALETTE[k % len(PALETTE)], label=labs[k] if k < len(labs) else str(k)) for k in used],
              frameon=False, ncol=min(4, max(len(used), 1)), fontsize=6.3, loc="upper left", bbox_to_anchor=(0, -0.25))


def fig_detail(q: Panel, i, L, out, lang="ko"):
    """Everything about one panel's episode: distances, clearances, h_c and the reflex weight, speed and altitude, the
    filter (kept fraction, certificate status, backup variant) and the nominal vs applied command."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    s = q.signals(i)
    t = s["t"]
    has_f = "status" in s and q.mode != "none"
    has_u = "u" in s
    ratios = [1, 1, 1.1, 0.9] + ([1, 0.22] if has_f else []) + ([1] if has_u else [])
    fig, axes = plt.subplots(len(ratios), 1, figsize=(7.0, 1.12 * sum(ratios) + 0.9), sharex=True,
                             gridspec_kw={"height_ratios": ratios})
    a = axes[0]
    a.plot(t, s["dist_t"], color=q.color)
    a.axhline(q.r_s, color=INK2, lw=0.7, ls=":", label=L["strike_r"])
    a.set_yscale("log")
    a.set_ylabel(L["dist_t"])
    a.legend(frameon=False, loc="lower left")
    a = axes[1]
    for key, c in (("obst", "#6b6a65"), ("wall", "#7a5195"), ("floor", "#b07d12"), ("ceil", "#9a9a9a")):
        a.plot(t, s[key], color=c, lw=0.9, label=L[key])
    a.plot(t, s["clear"], color=q.color, lw=1.8, label=L["min"])
    a.axhline(0, color="#c4302b", lw=0.8)
    a.axhline(q.fresh_om, color=INK2, lw=0.6, ls=":")
    a.set_ylim(-0.3, max(4.0, float(np.nanpercentile(s["clear"], 90)) + 0.5))
    a.set_ylabel(L["clear"])
    a.legend(frameon=False, ncol=5, fontsize=6.3, loc="upper right")
    a = axes[2]
    a.plot(t, s["range"], color=CHASER, lw=1.0, label=L["range"])
    a.plot(t, s["hc"], color=q.color, lw=1.8, label="$h_c$")
    a.axhline(0, color="#c4302b", lw=0.8)
    a.axhline(q.fresh_m, color=INK2, lw=0.6, ls=":", label=L["margin02"])
    a.set_ylim(-0.5, max(6.0, min(15.0, float(np.nanmax(s["range"])) + 0.5)))
    a.set_ylabel(L["hc"])
    a2 = a.twinx()
    a2.fill_between(t, 0, s["sigma"], color="#e07b39", alpha=0.2, lw=0, step="post")
    a2.set_ylim(0, 1.05)
    a2.set_ylabel(L["sigma"], color="#e07b39")
    a2.grid(False)
    a2.spines["right"].set_visible(True)
    a.legend(frameon=False, ncol=3, fontsize=6.3, loc="upper right")
    a = axes[3]
    a.plot(t, s["speed"], color=q.color, label=L["speed"])
    a.set_ylabel(L["speed"])
    a3 = a.twinx()
    a3.plot(t, s["alt"], color="#6b6a65", lw=1.0, ls="--", label=L["alt"])
    a3.set_ylabel(L["alt"])
    a3.grid(False)
    a3.spines["right"].set_visible(True)
    h1, l1 = a.get_legend_handles_labels()
    h2, l2 = a3.get_legend_handles_labels()
    a.legend(h1 + h2, l1 + l2, frameon=False, ncol=2, fontsize=6.3, loc="upper right")
    r = 4
    if has_f:
        a = axes[r]
        _status_band(a, t, s, -0.03, 1.05)
        a.step(t, s["lam"], where="post", color=q.color, lw=1.4, zorder=3)
        a.set_ylim(-0.03, 1.05)
        a.set_ylabel(L["kept"])
        a.legend(handles=[Patch(color=STATUS_C["committed"], label=L["committed"]),
                          Patch(color=STATUS_C["nocert"], label=L["nocert"])], frameon=False, ncol=2, fontsize=6.3,
                 loc="lower right", bbox_to_anchor=(1.0, 0.98), borderaxespad=0.0)
        _variant_strip(axes[r + 1], t, s, q, lang)
        axes[r + 1].set_ylabel(L["variant"], fontsize=6.5, rotation=0, ha="right", va="center")
        r += 2
    if has_u:
        a = axes[r]
        a.plot(t, s["u_nom"][:, 0], color=INK2, lw=0.9, ls="--", label=f"{L['nominal']} $a_T$")
        a.plot(t, s["u"][:, 0], color=q.color, lw=1.3, label=f"{L['applied']} $a_T$")
        a.set_ylabel(L["thrust"])
        a5 = a.twinx()
        a5.plot(t, np.linalg.norm(s["u_nom"][:, 1:], axis=1), color="#e07b39", lw=0.8, ls="--", alpha=0.8,
                label=f"{L['nominal']} $\\|\\omega\\|$")
        a5.plot(t, np.linalg.norm(s["u"][:, 1:], axis=1), color="#c4302b", lw=0.9, alpha=0.9,
                label=f"{L['applied']} $\\|\\omega\\|$")
        a5.set_ylabel(L["rate"])
        a5.grid(False)
        a5.spines["right"].set_visible(True)
        h1, l1 = a.get_legend_handles_labels()
        h2, l2 = a5.get_legend_handles_labels()
        a.legend(h1 + h2, l1 + l2, frameon=False, ncol=4, fontsize=6.3, loc="upper right")
    axes[-1].set_xlabel(L["t"])
    fig.suptitle(f"{L['episode']} {i}: {q.label} ({q.verdict(i, L)})", x=0.01, ha="left", fontsize=10)
    fig.tight_layout(h_pad=0.6)
    save(fig, out / f"detail_{q.tag}")


def fig_detail_compact(q: Panel, i, L, out, lang="ko"):
    """Paper version of the detail: margins (h_c, closest clearance), the filter (kept fraction, status, variant), speed."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    s = q.signals(i)
    t = s["t"]
    has_f = "status" in s and q.mode != "none"
    ratios = [1.2] + ([0.9, 0.2] if has_f else []) + [0.8]
    fig, axes = plt.subplots(len(ratios), 1, figsize=(3.45, 0.95 * sum(ratios) + 0.75), sharex=True,
                             gridspec_kw={"height_ratios": ratios})
    a = axes[0]
    a.plot(t, s["hc"], color=q.color, lw=1.5, label="$h_c$")
    a.plot(t, s["clear"], color="#6b6a65", lw=1.0, label=f"{L['min']} {L['clear'].split(' [')[0]}")
    a.axhline(0, color="#c4302b", lw=0.8)
    a.axhline(q.fresh_m, color=INK2, lw=0.6, ls=":")
    a.set_ylim(-0.4, 6)
    a.set_ylabel("[m]")
    a.legend(frameon=False, ncol=2, fontsize=6.3, loc="upper right")
    r = 1
    if has_f:
        a = axes[1]
        _status_band(a, t, s, -0.03, 1.05)
        a.step(t, s["lam"], where="post", color=q.color, lw=1.2, zorder=3)
        a.set_ylim(-0.03, 1.05)
        a.set_ylabel("λ")
        a.legend(handles=[Patch(color=STATUS_C["committed"], label=L["committed"]),
                          Patch(color=STATUS_C["nocert"], label=L["nocert"])], frameon=False, ncol=2, fontsize=6,
                 loc="lower right", bbox_to_anchor=(1.0, 0.98), borderaxespad=0.0)
        _variant_strip(axes[2], t, s, q, lang)
        r = 3
    a = axes[r]
    a.plot(t, s["speed"], color=q.color, lw=1.2)
    a.set_ylabel(L["speed"].split(" [")[0] + " [m/s]")
    a.set_xlabel(L["t"])
    fig.suptitle(f"{L['episode']} {i}: {q.label}", x=0.01, ha="left", fontsize=8.5)
    fig.tight_layout(h_pad=0.4)
    save(fig, out / f"detail_compact_{q.tag}")


def draw_pillars(ax, q: Panel, j, top=None):
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    z0, z1 = q.z_floor - 1.0, (top if top is not None else Z_TOP)
    for o in q.obst[j]:
        if o[6] < 0.5:
            continue
        xy = outline(o, 48)
        n = len(xy)
        sides = [[(*xy[a], z0), (*xy[(a + 1) % n], z0), (*xy[(a + 1) % n], z1), (*xy[a], z1)] for a in range(n)]
        pc = Poly3DCollection(sides, facecolor=OBS_FILL, edgecolor="none", alpha=0.22, zorder=1)
        ax.add_collection3d(pc)
        cap = Poly3DCollection([[(*p, z1) for p in xy]], facecolor="#b5b3ab", edgecolor=OBS_EDGE, linewidths=0.6, alpha=0.5,
                               zorder=1)
        ax.add_collection3d(cap)
        ax.plot(xy[:, 0], xy[:, 1], z0, color=OBS_EDGE, lw=0.6, zorder=1)


Z_TOP = 7.0          # drawn height of the obstacle pillars and the 3-D box [m] (flight altitude is 1.5-4 m)


def setup3d(ax, q: Panel, j, elev=32, azim=-58, zoom=1.12):
    ax.computed_zorder = False                    # draw in zorder: pillars first, paths and bodies on top
    ax.set_xlim(0, q.arena[0])
    ax.set_ylim(-q.arena[1], q.arena[1])
    ax.set_zlim(q.z_floor - 1.0, Z_TOP)
    ax.set_box_aspect((q.arena[0], 2 * q.arena[1], 2.0 * (Z_TOP - q.z_floor + 1.0)), zoom=zoom)
    ax.view_init(elev=elev, azim=azim)
    ax.set_xticks([0, 10, 20, 30, 40])
    ax.set_yticks([-10, -5, 0, 5, 10])
    ax.set_zticks([0, 2, 4, 6])
    ax.xaxis.pane.set_facecolor((0.98, 0.98, 0.97, 1.0))
    ax.yaxis.pane.set_facecolor((0.97, 0.97, 0.96, 1.0))
    ax.zaxis.pane.set_facecolor((0.95, 0.95, 0.94, 1.0))
    xs, ys = np.meshgrid([0, q.arena[0]], [-q.arena[1], q.arena[1]])
    ax.plot_surface(xs, ys, np.full_like(xs, q.z_floor, dtype=float), color="#e8e6df", alpha=0.35, linewidth=0, shade=False,
                    zorder=0)
    ax.set_xlabel("x [m]", labelpad=6)
    ax.set_ylabel("y [m]", labelpad=2)
    ax.set_zlabel("z [m]", labelpad=1)
    ax.tick_params(labelsize=6, pad=0)


def sphere(ax, c, r, color, alpha, n=14, wire=False):
    u, v = np.mgrid[0:2 * np.pi:complex(0, n), 0:np.pi:complex(0, n // 2 + 2)]
    x, y, z = c[0] + r * np.cos(u) * np.sin(v), c[1] + r * np.sin(u) * np.sin(v), c[2] + r * np.cos(v)
    if wire:
        return ax.plot_wireframe(x, y, z, color=color, alpha=alpha, linewidth=0.4, zorder=6)
    return ax.plot_surface(x, y, z, color=color, alpha=alpha, linewidth=0, shade=True, zorder=7)


def fig_view3d(panels, main: Panel, i, L, out):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    fig = plt.figure(figsize=(9.0, 4.2))
    ax = fig.add_axes((0.0, 0.0, 0.7, 0.94), projection="3d")      # legend in the right margin, clear of the scene
    j, k = main.ep(i)
    setup3d(ax, main, j)
    draw_pillars(ax, main, j)
    handles = []
    for q in panels:
        jq, kq = q.ep(i)
        p = q.pos[:kq, jq]
        ax.plot(p[:, 0], p[:, 1], p[:, 2], color=q.color, lw=2.0 if q is main else 1.3, zorder=5)
        ax.plot(p[:, 0], p[:, 1], np.full(kq, q.z_floor), color=q.color, lw=0.6, alpha=0.35, zorder=2)   # floor shadow
        end_marker(ax, q, i, p[-1], three_d=True)
        handles.append(Line2D([], [], color=q.color, lw=1.6, label=f"{q.label}: {q.verdict(i, L)}"))
    for c in range(main.ch.shape[2]):
        if main.on[j, c] > 0.5:
            cp = main.ch[:k, j, c]
            ax.plot(cp[:, 0], cp[:, 1], cp[:, 2], color=CHASER, lw=1.1, ls="--", zorder=4)
            sphere(ax, cp[-1], main.r_c, CHASER, 0.6)
    tp = main.tg[:k, j]
    ax.plot(tp[:, 0], tp[:, 1], tp[:, 2], color=TARGET, lw=1.2, zorder=4)
    sphere(ax, tp[-1], main.r_s, TARGET, 0.6)
    ax.scatter(*main.pos[0, j], c=INK, s=12, zorder=8)
    handles += [Line2D([], [], color=CHASER, lw=1.1, ls="--", label=f"{L['chaser_path']} ({main.label})"),
                Line2D([], [], color=TARGET, lw=1.2, label=L["target_path"])]
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(0.7, 0.5), frameon=False, fontsize=7)
    fig.suptitle(f"{L['episode']} {i}", x=0.02, y=0.98, ha="left", va="top", fontsize=10)
    save(fig, out / "view3d")


def fig_view3d_single(q: Panel, i, L, out):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Line3DCollection
    j, k = q.ep(i)
    s = q.signals(i)
    fig = plt.figure(figsize=(7.6, 4.2))
    ax = fig.add_axes((0.0, 0.0, 0.86, 0.93), projection="3d")
    setup3d(ax, q, j)
    draw_pillars(ax, q, j)
    p = s["p"]
    seg = np.stack([p[:-1], p[1:]], axis=1)
    lc = Line3DCollection(seg, cmap="viridis", linewidths=2.4, zorder=5)
    lc.set_array(s["speed"][:-1])
    lc.set_clim(0, max(8.0, float(s["speed"].max())))
    ax.add_collection3d(lc)
    ax.plot(p[:, 0], p[:, 1], np.full(k, q.z_floor), color=INK2, lw=0.6, alpha=0.35, zorder=2)
    for c in range(q.ch.shape[2]):
        if q.on[j, c] > 0.5:
            cp = q.ch[:k, j, c]
            ax.plot(cp[:, 0], cp[:, 1], cp[:, 2], color=CHASER, lw=1.1, ls="--", zorder=4)
            sphere(ax, cp[-1], q.r_c, CHASER, 0.6)
    tp = q.tg[:k, j]
    ax.plot(tp[:, 0], tp[:, 1], tp[:, 2], color=TARGET, lw=1.2, zorder=4)
    sphere(ax, tp[-1], q.r_s, TARGET, 0.6)
    end_marker(ax, q, i, p[-1], three_d=True)
    cb = fig.colorbar(lc, cax=fig.add_axes((0.9, 0.26, 0.016, 0.44)))   # own axes: clear of the z ticks
    cb.set_label(L["speed_c"])
    fig.suptitle(f"{L['episode']} {i}: {q.label} ({q.verdict(i, L)})", x=0.02, y=0.98, ha="left", va="top", fontsize=9.5)
    save(fig, out / f"view3d_{q.tag}", pdf=False)


def quad_arms(p, Rm, arm=0.55, scale=3.2):
    """Two crossed arms and the thrust axis of the quadrotor at p with attitude Rm (drawn larger than life)."""
    a = arm * scale
    bx, by, bz = Rm[:, 0], Rm[:, 1], Rm[:, 2]
    d1, d2 = (bx + by) / np.sqrt(2), (bx - by) / np.sqrt(2)
    return [np.stack([p - a * d1, p + a * d1]), np.stack([p - a * d2, p + a * d2]), np.stack([p, p + 0.9 * scale * bz])]


def anim3d(q: Panel, i, L, out, stride=5, fps=20):
    import matplotlib.pyplot as plt
    from matplotlib import animation
    j, k = q.ep(i)
    s = q.signals(i)
    p, Rq = s["p"], (quat_to_R(s["quat"]) if "quat" in s else None)
    fig = plt.figure(figsize=(7.2, 4.6))
    ax = fig.add_axes([0.0, 0.04, 1.0, 0.88], projection="3d")
    setup3d(ax, q, j, zoom=1.32)
    draw_pillars(ax, q, j)
    tp = q.tg[:k, j]
    chs = [c for c in range(q.ch.shape[2]) if q.on[j, c] > 0.5]
    trail, = ax.plot([], [], [], color=q.color, lw=1.9, zorder=5)
    shadow, = ax.plot([], [], [], color=INK2, lw=0.6, alpha=0.35, zorder=2)
    ctrails = [ax.plot([], [], [], color=CHASER, lw=1.0, ls="--", zorder=4)[0] for _ in chs]
    ttrail, = ax.plot([], [], [], color=TARGET, lw=1.0, zorder=4)
    arms = [ax.plot([], [], [], color=c, lw=lw, zorder=9)[0] for c, lw in ((INK, 2.2), (INK, 2.2), ("#c4302b", 1.5))]
    txt = fig.text(0.02, 0.025, "", fontsize=8.5, color=INK)
    fig.text(0.98, 0.025, L["hc_sphere"], fontsize=7.5, color=INK2, ha="right")
    fig.text(0.02, 0.955, f"{L['episode']} {i}: {q.label} ({q.verdict(i, L)})", fontsize=10, color=INK, ha="left")
    dyn = []
    frames = list(range(0, k, stride)) + [k - 1] * 12

    def draw(kk):
        for art in dyn:
            art.remove()
        dyn.clear()
        trail.set_data_3d(p[:kk + 1, 0], p[:kk + 1, 1], p[:kk + 1, 2])
        shadow.set_data_3d(p[:kk + 1, 0], p[:kk + 1, 1], np.full(kk + 1, q.z_floor))
        ttrail.set_data_3d(tp[:kk + 1, 0], tp[:kk + 1, 1], tp[:kk + 1, 2])
        for ln, c in zip(ctrails, chs):
            cp = q.ch[:kk + 1, j, c]
            ln.set_data_3d(cp[:, 0], cp[:, 1], cp[:, 2])
            dyn.append(sphere(ax, q.ch[kk, j, c], q.r_c, CHASER, 0.8, n=12))
            dyn.append(ax.scatter(*q.ch[kk, j, c], c=CHASER, s=28, edgecolors=INK, linewidths=0.5, zorder=8,
                                  depthshade=False))
            if q.guidance:
                rel = p[kk] - q.ch[kk, j, c]
                e = rel / max(np.linalg.norm(rel), 1e-6)
                sv = max(q.v_c - float(e @ s["v"][kk]), 0.0)
                dyn.append(sphere(ax, q.ch[kk, j, c], q.r_c + q.r_d + sv * q.delay + sv ** 2 / (2 * q.a_e), CHASER, 0.25,
                                  n=14, wire=True))
        dyn.append(sphere(ax, tp[kk], q.r_s, TARGET, 0.7, n=12))
        if Rq is not None:
            for ln, seg in zip(arms, quad_arms(p[kk], Rq[kk])):
                ln.set_data_3d(seg[:, 0], seg[:, 1], seg[:, 2])
        else:
            dyn.append(ax.scatter(*p[kk], c=q.color, s=25))
        ax.view_init(elev=24 + 6 * np.sin(np.pi * kk / max(k, 1)), azim=-72 + 40 * kk / max(k, 1))
        lam = "" if q.mode == "none" else f"   λ {s['lam'][kk]:.2f}"
        stt = ""
        if "status" in s and q.mode != "none":
            stt = "   " + (L["nocert"] if s["status"][kk] == 2 else L["committed"] if s["status"][kk] == 1 else L["fresh"])
        txt.set_text(f"t = {kk * q.dt:4.1f} s   $h_c$ = {s['hc'][kk]:5.2f} m   {L['speed'].split(' [')[0]} "
                     f"{s['speed'][kk]:4.1f} m/s{lam}{stt}")
        return []

    ani = animation.FuncAnimation(fig, draw, frames=frames, blit=False)
    ani.save(out / f"anim3d_{q.tag}.gif", writer=animation.PillowWriter(fps=fps), dpi=100)
    plt.close(fig)


def anim2d(panels, i, L, out, cols, stride=2, fps=25):
    import matplotlib.pyplot as plt
    from matplotlib import animation
    from matplotlib.patches import Circle
    fig, axes, (W, H) = top_grid(len(panels), cols, panels[0].arena, panel_w=3.3, title_h=0.26, top=0.34, bottom=0.08,
                                 ticks=False)
    fig.patch.set_facecolor("white")
    arts, k_ends = [], []
    for ax, q in zip(axes, panels):
        j, k = q.ep(i)
        s = q.signals(i)
        top_view(ax, q, j, grid=False)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():                 # the arena outline is the frame
            sp.set_visible(False)
        k_ends.append(k)
        ax.set_title(q.label, loc="left", fontsize=8, color=q.color, fontweight="bold")
        path, = ax.plot([], [], color=q.color, lw=1.6, zorder=5)
        drone = Circle((0, 0), q.r_d * 1.6, fc=q.color, ec=INK, lw=0.6, zorder=8)
        tgt = Circle((0, 0), q.r_s, fc=TARGET, ec=INK, lw=0.6, zorder=6)
        ax.add_patch(drone)
        ax.add_patch(tgt)
        rings = []
        for c in range(q.ch.shape[2]):
            on = q.on[j, c] > 0.5
            rg = Circle((0, 0), 0.0, fc=CHASER, alpha=0.12, ec=CHASER, ls="--", lw=0.7, zorder=3)
            cc = Circle((0, 0), q.r_c if on else 0.0, fc=CHASER, ec=INK, lw=0.5, zorder=7)
            ax.add_patch(rg)
            ax.add_patch(cc)
            rings.append((rg, cc, on, c))
        txt = ax.text(0.99, 0.03, "", transform=ax.transAxes, ha="right", va="bottom", fontsize=7, color=INK2)
        arts.append((q, j, k, s, path, drone, tgt, rings, txt))
    n_frames = max(k_ends)

    def draw(f):
        kk0 = min(f * stride, n_frames - 1)
        res = []
        for q, j, k, s, path, drone, tgt, rings, txt in arts:
            kk = min(kk0, k - 1)
            path.set_data(s["p"][:kk + 1, 0], s["p"][:kk + 1, 1])
            drone.center = tuple(s["p"][kk, :2])
            tgt.center = tuple(q.tg[kk, j, :2])
            for rg, cc, on, c in rings:
                cc.center = rg.center = tuple(q.ch[kk, j, c, :2])
                if on and q.guidance:
                    rel = s["p"][kk] - q.ch[kk, j, c]
                    e = rel / max(np.linalg.norm(rel), 1e-6)
                    sv = max(q.v_c - float(e @ s["v"][kk]), 0.0)
                    rg.set_radius(q.r_c + q.r_d + sv * q.delay + sv ** 2 / (2 * q.a_e))
            end = "" if kk0 < k - 1 else f"  ({L[q.code(i)]})"
            lam = "" if q.mode == "none" else f"  λ {s['lam'][kk]:.2f}"
            txt.set_text(f"t {kk * q.dt:4.1f} s{lam}{end}")
            res += [path, drone, tgt, txt] + [r for rg, cc, _, _ in rings for r in (rg, cc)]
        return res

    fig.suptitle(f"{L['episode']} {i}", x=0.01, y=1 - 0.05 / H, ha="left", va="top", fontsize=9)
    ani = animation.FuncAnimation(fig, draw, frames=int(np.ceil(n_frames / stride)) + 12, blit=True)
    ani.save(out / "compare2d.gif", writer=animation.PillowWriter(fps=fps), dpi=85)
    plt.close(fig)


def fig_overview(panels, eps, L, out, names=None, cols=3, stem="overview"):
    """Every episode's top view with the panels' paths (paper figure): the first panel's chaser and target."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Circle
    n_leg = len(panels) + 4
    rows_leg = int(np.ceil(n_leg / 4))
    fig, axes, (W, H) = top_grid(len(eps), cols, panels[0].arena, top=0.08, bottom=0.12 + 0.24 * rows_leg)
    ref = next((q for q in panels if q.mode != "none"), panels[0])
    for ax, i in zip(axes, eps):
        j, k = ref.ep(i)
        top_view(ax, ref, j)
        for c in range(ref.ch.shape[2]):
            if ref.on[j, c] > 0.5:
                ax.plot(ref.ch[:k, j, c, 0], ref.ch[:k, j, c, 1], color=CHASER, lw=0.9, ls="--", zorder=4)
                ax.add_patch(Circle(ref.ch[k - 1, j, c, :2], ref.r_c, fc=CHASER, ec=INK, lw=0.4, zorder=6))
        ax.plot(ref.tg[:k, j, 0], ref.tg[:k, j, 1], color=TARGET, lw=1.0, zorder=4)
        ax.add_patch(Circle(ref.tg[k - 1, j, :2], ref.r_s, fc=TARGET, ec=INK, lw=0.4, zorder=6))
        for q in panels:
            jq, kq = q.ep(i)
            ls = "--" if q.mode == "none" else "-"
            ax.plot(q.pos[:kq, jq, 0], q.pos[:kq, jq, 1], color=q.color, lw=1.5 if q is ref else 1.0, ls=ls, zorder=5)
            end_marker(ax, q, i, q.pos[kq - 1, jq])
        ax.plot(*ref.pos[0, j, :2], "o", color=INK, ms=3.5, zorder=8)
        ttl = f"({chr(97 + eps.index(i))}) {L['episode']} {i}" + (f": {names[i]}" if names and i in names else "")
        ax.set_title(ttl, loc="left", fontsize=8.5)
    hs = [Line2D([], [], color=q.color, lw=1.5, ls="--" if q.mode == "none" else "-", label=q.label) for q in panels]
    hs += [Line2D([], [], color=CHASER, lw=0.9, ls="--", label=L["chaser_path"]),
           Line2D([], [], marker="*", ls="", ms=10, color="#f2b705", mec=INK, label=L["strike"]),
           Line2D([], [], marker="X", ls="", ms=8, color="#c4302b", mec=INK, label=L["contact"]),
           Line2D([], [], marker="o", ls="", ms=6, color="#8a8a8a", mec=INK, label=L["timeup"])]
    fig.legend(handles=hs, loc="lower center", ncol=min(len(hs), 4), frameon=False, bbox_to_anchor=(0.5, 0.02 / H))
    save(fig, out / stem)


# ----------------------------------------------------------------------------------------------------------- main
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--panel", action="append", required=True, help='"RUN_DIR:MODE:LABEL[:TAG]" (repeat; panels in order)')
    p.add_argument("--episodes", type=int, nargs="*", default=None, help="episodes (default: all replayed)")
    p.add_argument("--names", default="", help="JSON {episode: caption} for the overview (e.g. from episode_picks.json)")
    p.add_argument("--main", default="", help="tag of the main panel (3-D overlay, chaser shown); default: first filtered")
    p.add_argument("--detail", nargs="*", default=None, help="tags with a detail plot (default: every panel)")
    p.add_argument("--gif3d", nargs="*", default=None, help="tags with a 3-D animation (default: the main panel)")
    p.add_argument("--what", default="compare2d,signals,detail,view3d,anim3d,anim2d,overview",
                   help="comma list of outputs")
    p.add_argument("--overview_main", nargs="*", default=None,
                   help="tags of a second, lighter overview (overview_main.png), e.g. ps2_ours ps2_none residual_ours")
    p.add_argument("--cols", type=int, default=3)
    p.add_argument("--stride3d", type=int, default=5, help="steps per 3-D animation frame")
    p.add_argument("--lang", choices=("ko", "en"), default="ko")
    p.add_argument("--out", default="outputs/strike_episodes_viz")
    a = p.parse_args(argv)
    lang = setup_style(a.lang)
    L = TXT[lang]
    what = set(a.what.split(","))
    out = Path(a.out)
    panels = [Panel(s, PALETTE[k % len(PALETTE)]) for k, s in enumerate(a.panel)]
    common = set.intersection(*[set(q.idx) for q in panels])
    eps = a.episodes if a.episodes else sorted(common)
    miss = [i for i in eps if i not in common]
    if miss:
        raise SystemExit(f"episodes {miss} are not in every replay (common: {sorted(common)})")
    main_p = next((q for q in panels if q.tag == a.main), None) or next((q for q in panels if q.mode != "none"), panels[0])
    det = [q for q in panels if a.detail is None or q.tag in a.detail]
    g3 = [q for q in panels if (q.tag in a.gif3d if a.gif3d is not None else q is main_p)]
    names = {int(k): v for k, v in json.loads(a.names).items()} if a.names else None
    summary = []
    for i in eps:
        o = out / f"ep{i}"
        o.mkdir(parents=True, exist_ok=True)
        for name, fn in (("compare2d", lambda: fig_compare2d(panels, i, L, o, a.cols)),
                         ("signals", lambda: fig_signals(panels, i, L, o)),
                         ("detail", lambda: [(fig_detail(q, i, L, o, lang), fig_detail_compact(q, i, L, o, lang))
                                             for q in det]),
                         ("view3d", lambda: (fig_view3d(panels, main_p, i, L, o), fig_view3d_single(main_p, i, L, o))),
                         ("anim3d", lambda: [anim3d(q, i, L, o, stride=a.stride3d) for q in g3]),
                         ("anim2d", lambda: anim2d(panels, i, L, o, a.cols))):
            if name in what:
                try:
                    fn()
                except Exception as ex:             # one broken figure must not stop the rest
                    warnings.warn(f"episode {i} {name}: {ex!r}")
        line = f"| {i} | " + ((names or {}).get(i, "") + " | " if names else "") + " | ".join(q.verdict(i, L) for q in panels) + " |"
        summary.append(line)
        print(f"[episode {i}] " + "; ".join(f"{q.label}: {q.verdict(i, L)}" for q in panels), flush=True)
    if "overview" in what and eps:
        fig_overview(panels, list(eps), L, out, names=names, cols=a.cols)
        if a.overview_main:
            sub = [q for q in panels if q.tag in a.overview_main]
            if sub:
                fig_overview(sub, list(eps), L, out, names=None, cols=a.cols, stem="overview_main")   # captions name
                # the selection reason, which may concern a method this lighter figure leaves out
    why = (["선택 이유(평가 기준)"] if lang == "ko" else ["picked as (evaluation)"]) if names else []
    md = ["| " + " | ".join([L["episode"]] + why + [q.label for q in panels]) + " |",
          "|---|" + "---|" * (len(panels) + len(why))] + summary
    if names:
        md += ["", "선택 이유는 원래 평가의 결과로 정한 것이다. 장면만 다시 실행한 경우(EXACT=0) 경계 사례는 결과가 다를 수 있다."
               if lang == "ko" else "The reasons come from the original evaluation; a replay of only these episodes (EXACT=0) "
               "can end borderline episodes differently."]
    (out / "episodes.md").write_text("\n".join(md) + "\n")
    print(f"wrote {out}/ep<i>/ and overview (labels: {lang})")


if __name__ == "__main__":
    main()
