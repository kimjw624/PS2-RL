"""Figure stage of the landing Phase-I report (vector PDFs for LaTeX).

Visual rules: one y-axis per panel, thin lines, recessive grid, a legend for every
multi-series panel, fixed categorical order (blue, orange, aqua, ...), status colours
(good / warning / critical) only for outcome states and always with a distinct marker.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, ListedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Ellipse, Patch  # noqa: E402

BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"
GOOD, WARN, CRIT = "#0ca30c", "#fab219", "#d03b3b"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
BAND = "#f0efec"
SEQ_BLUE = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6",
            "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
BACKBONE_COLOR = {"sac": BLUE, "td3": ORANGE}
TEXTWIDTH = 6.5

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["cmr10", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True,
    "axes.unicode_minus": False,
    "font.size": 8.5,
    "axes.titlesize": 9,
    "axes.labelsize": 8.5,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK2,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "lines.linewidth": 1.4,
    "lines.solid_capstyle": "round",
    "legend.frameon": False,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
    "pdf.fonttype": 42,
})


def pretty(run: str) -> str:
    """'td3_seed0' -> 'TD3, seed 0' (underscores would be parsed as math subscripts)."""
    if "_seed" in run:
        bb, seed = run.split("_seed")
        return f"{bb.upper()}, seed {seed}"
    return run.replace("_", " ")


def _title(ax, text: str) -> None:
    ax.set_title(text, loc="left", color=INK, fontsize=9)


def _save(fig, out: Path, name: str) -> str:
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.pdf")
    plt.close(fig)
    return f"{name}.pdf"


def _cone_r(cfg: dict, zeta: np.ndarray) -> np.ndarray:
    lin = cfg["cone_r0"] + np.tan(np.radians(cfg["cone_theta_deg"])) * zeta
    return np.sqrt(np.clip(lin**2 - cfg["cone_eps"] ** 2, 0.0, None))


def _radial(x: np.ndarray, cfg: dict) -> np.ndarray:
    return np.hypot(x[..., 0] - cfg["pad_x"], x[..., 1] - cfg["pad_y"])


def _draw_cone_rz(ax, cfg: dict, zmax: float, *, signed: bool = False, extents: dict | None = None) -> None:
    z = np.linspace(-0.05, zmax, 200)
    r = _cone_r(cfg, z)
    ax.plot(r, z, color=INK, lw=1.1)
    if signed:
        ax.plot(-r, z, color=INK, lw=1.1)
    ax.axhline(0.0, color=INK2, lw=0.8)
    ax.plot([-cfg["cone_r0"] if signed else 0.0, cfg["cone_r0"]], [0, 0], color=INK, lw=3.0, solid_capstyle="butt")
    if extents is not None:
        w = 2 * extents["dx"]
        ax.add_patch(Ellipse((0.0, cfg["z_des"]), w, 2 * extents["dz"], fill=True, fc=BAND, ec=INK2, lw=0.8, ls="--", zorder=0.5))


# ------------------------------------------------------------------- method figures
def fig_geometry(d: dict, out: Path) -> str:
    cfg, rc = d["cfg"], d["region_cfg"]
    ext = d["lqr"]["extents"]
    fig, ax = plt.subplots(figsize=(TEXTWIDTH, 3.0))
    zmax = rc["zeta_max"]
    zz = np.linspace(0, rc["edge_zeta_low"], 100)
    r_out = _cone_r(cfg, zz)
    lin_in = cfg["cone_r0"] + np.tan(np.radians(cfg["cone_theta_deg"])) * zz - rc["edge_margin"]
    r_in = np.sqrt(np.clip(lin_in**2 - cfg["cone_eps"] ** 2, 0, None))
    for sgn in (1, -1):
        ax.fill_betweenx(zz, sgn * r_in, sgn * r_out, color=ORANGE, alpha=0.10, lw=0)
    _draw_cone_rz(ax, cfg, zmax + 0.2, signed=True, extents=ext)
    colors = {"general": BLUE, "edge": ORANGE, "shell": AQUA}
    labels = {"general": "general (altitude-uniform)", "edge": "edge band", "shell": "capture shell"}
    rng = np.random.default_rng(0)
    for reg in ("general", "edge", "shell"):
        X = d["test_sets"][reg]
        sel = rng.choice(len(X), min(350, len(X)), replace=False)
        r = _radial(X[sel], cfg)
        sgn = np.sign(X[sel, 0] - cfg["pad_x"])
        ax.scatter(r * sgn, X[sel, 2] - cfg["pad_z"], s=4, color=colors[reg], alpha=0.8, lw=0, label=labels[reg], zorder=2)
    ax.axhline(zmax, color=MUTED, lw=0.7, ls=":")
    ax.text(0.5, zmax + 0.04, r"$\zeta_{\max}$ (top of $\Omega$)", color=INK, fontsize=7.5, ha="center",
            transform=ax.get_yaxis_transform())
    ax.text(0.0, -0.1, "pad", ha="center", va="top", fontsize=7.5, color=INK)
    ax.set_xlabel(r"signed horizontal distance from the cone axis  [m]")
    ax.set_ylabel(r"altitude above the pad $\zeta$  [m]")
    ax.set_aspect("equal")
    ax.set_ylim(-0.3, zmax + 0.25)
    handles, _ = ax.get_legend_handles_labels()
    handles += [Line2D([], [], color=INK, lw=1.1, label=r"cone boundary $h_{\mathrm{cone}}=0$"),
                Patch(fc=BAND, ec=INK2, ls="--", lw=0.8, label=r"$\mathcal{B}$, position projection"),
                Patch(fc=ORANGE, alpha=0.10, ec="none", label=r"edge band $0\leq h_{\mathrm{cone}}\leq\eta$, $\zeta\leq\zeta_{\mathrm{low}}$")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, markerscale=3, handletextpad=0.4)
    return _save(fig, out, "fig_geometry")


def fig_level_bounds(d: dict, out: Path) -> str:
    cert = d["certificate"]
    sweep = d["sweep"]
    z = np.array([r["z_des"] for r in sweep])
    fig, axes = plt.subplots(1, 2, figsize=(TEXTWIDTH, 3.0))
    ax = axes[0]
    series = [
        (np.array([r["c_cone_exact"] for r in sweep]), BLUE, r"$c_{\mathrm{cone}}$ (exact)", "-"),
        (np.array([r["c_cone_note"] for r in sweep]), BLUE, r"$c_{\mathrm{cone}}$ (corner bound)", "--"),
        (np.array([r["c_ground"] for r in sweep]), ORANGE, r"$c_{\mathrm{ground}}$", "-"),
    ]
    for y, c, lab, ls in series:
        ax.plot(z, y, color=c, ls=ls, label=lab)
    for val, lab, c in ((cert["c_u"], r"$c_{\mathcal{U}}$", AQUA), (cert["c_chart"], r"$c_{\mathrm{chart}}$", YELLOW),
                        (cert["c_lyap_adversarial"], r"$c_{\mathrm{Lyap}}$", MAGENTA)):
        ax.axhline(val, color=c, lw=1.2, label=lab)
    ax.axhline(d["cfg"]["base_set_c"], color=INK2, lw=0.9, ls=":")
    ax.text(2.45, d["cfg"]["base_set_c"] - 0.4, r"chosen $c_{\mathcal{B}}=12$", ha="right", va="top", fontsize=7.5, color=INK)
    ax.axvline(d["cfg"]["z_des"], color=INK2, lw=0.9, ls=":")
    ax.text(d["cfg"]["z_des"] + 0.05, 1.0, r"$z_{\mathrm{des}}=1.25$", fontsize=7.5, color=INK)
    ax.set_ylim(0, 32)
    ax.set_xlim(z[0], 2.5)
    ax.set_xlabel(r"$z_{\mathrm{des}}$  [m]")
    ax.set_ylabel("admissible level $c$")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=3, columnspacing=0.8, handlelength=1.6)
    _title(ax, "(a) level bounds vs. hover height")
    ax = axes[1]
    lv = d["lyap_curve"]
    ax.plot(lv["levels"], lv["worst"], color=BLUE, marker="o", ms=3.5, mec="white", mew=0.8)
    ax.axhline(1.0, color=INK2, lw=0.9)
    ax.axvline(d["cfg"]["base_set_c"], color=INK2, lw=0.9, ls=":")
    ax.axvline(cert["c_lyap_adversarial"], color=MAGENTA, lw=1.2)
    ax.text(d["cfg"]["base_set_c"] - 0.3, 1.02, r"$c_{\mathcal{B}}$", ha="right", fontsize=7.5, color=INK)
    ax.text(cert["c_lyap_adversarial"] + 0.25, 0.975, r"$c_{\mathrm{Lyap}}=%.2f$" % cert["c_lyap_adversarial"], fontsize=7.5, color=INK)
    ax.set_xlabel(r"level $c$ of the tested sublevel-set boundary")
    ax.set_ylabel(r"worst $V(x^+)/V(x)$ on $\{V=c\}$")
    ax.set_ylim(0.95, 1.04)
    _title(ax, "(b) adversarial one-step Lyapunov test")
    fig.tight_layout(w_pad=1.5)
    return _save(fig, out, "fig_level_bounds")


# ------------------------------------------------------------------- training figures
def _curves(d: dict, key: str):
    out = {}
    for r in d["runs"]:
        bb = d["summaries"][r]["sa_backbone"]
        h = d["histories"][r]
        ev = d["configs"][r]["backup_ra"]["eval_every"]
        u = d["summaries"][r]["untrained_val"]
        if key == "eval_mu_weighted":
            v0 = u["mu_weighted"]
        elif key == "eval_crash_rate":
            v0 = u["crash_rate"]
        else:
            v0 = u[key.replace("eval_m_", "")]["m_hat"]
        y = np.array([v0, *h[key]], dtype=float)
        out.setdefault(bb, []).append((np.arange(len(y)) * ev, y))
    return out


def _band_plot(ax, groups, *, label_end: bool = True, ylim=(0, 1)):
    ends = []
    for bb in sorted(groups):
        n = min(len(y) for _, y in groups[bb])
        x = groups[bb][0][0][:n] / 1e6
        Y = np.stack([y[:n] for _, y in groups[bb]])
        c = BACKBONE_COLOR.get(bb, INK2)
        if Y.shape[0] > 1:
            ax.fill_between(x, Y.min(0), Y.max(0), color=c, alpha=0.12, lw=0)
        ax.plot(x, Y.mean(0), color=c, lw=1.2, label=f"{bb.upper()} (mean of {Y.shape[0]})")
        ends.append((bb, x[-1], Y.mean(0)[-1]))
    if label_end and ends:
        ends.sort(key=lambda t: t[2])
        span = ylim[1] - ylim[0]
        placed = []
        for bb, x, y in ends:
            yy = y
            for p in placed:
                if abs(yy - p) < 0.07 * span:
                    yy = p + 0.07 * span
            placed.append(yy)
            ax.annotate(f"{bb.upper()} {y:.3f}", (x, y), xytext=(x + 0.08, yy), fontsize=7, color=INK, va="center")
    ax.set_ylim(*ylim)
    ax.set_xlim(0, None)
    ax.margins(x=0.2)


def fig_training(d: dict, out: Path) -> str:
    fig, axes = plt.subplots(2, 3, figsize=(TEXTWIDTH, 4.3), sharex=True)
    panels = [("eval_mu_weighted", r"(a) weighted recoverability $\hat\mu_w$"), ("eval_m_edge", r"(b) edge region $\hat m_{\mathrm{edge}}$"),
              ("eval_m_general", r"(c) general region $\hat m_{\mathrm{general}}$"), ("eval_m_shell", r"(d) capture shell $\hat m_{\mathrm{shell}}$"),
              ("eval_crash_rate", "(e) crash rate (all regions)")]
    for ax, (key, title) in zip(axes.flat, panels):
        _band_plot(ax, _curves(d, key))
        _title(ax, title)
    ax = axes.flat[5]
    for r in d["runs"]:
        bb = d["summaries"][r]["sa_backbone"]
        h = d["histories"][r]
        ax.plot(np.asarray(h["step"]) / 1e6, h["curriculum_scale"], color=BACKBONE_COLOR[bb], lw=0.9, alpha=0.8)
    ax.set_ylim(-0.03, 1.05)
    _title(ax, "(f) curriculum scale $s$ (each run)")
    for ax in axes[1]:
        ax.set_xlabel("environment steps  [millions]")
    axes[0, 0].legend(loc="lower right", fontsize=7)
    fig.tight_layout(h_pad=0.8, w_pad=0.8)
    return _save(fig, out, "fig_training")


def fig_training_diag(d: dict, out: Path) -> str:
    fig, axes = plt.subplots(1, 3, figsize=(TEXTWIDTH, 2.1))
    seen = set()
    for r in d["runs"]:
        bb = d["summaries"][r]["sa_backbone"]
        h = d["histories"][r]
        c = BACKBONE_COLOR[bb]
        lab = bb.upper() if bb not in seen else None
        seen.add(bb)
        x = np.asarray(h["step"]) / 1e6
        y = np.asarray(h["episode_success_within_n_rate"])
        k = 15
        ys = np.convolve(y, np.ones(k) / k, mode="valid")
        axes[0].plot(x[k - 1:], ys, color=c, lw=0.9, alpha=0.85, label=lab)
        cl = np.asarray(h["critic_loss"])
        xc = (np.arange(len(cl)) + 1) * 8192 / 1e6
        cls = np.convolve(cl, np.ones(k) / k, mode="valid")
        axes[1].plot(xc[k - 1:], cls, color=c, lw=0.9, alpha=0.85)
        if bb == "sac":
            a = np.asarray(h["alpha"])
            axes[2].plot((np.arange(len(a)) + 1) * 8192 / 1e6, a, color=c, lw=0.9)
    axes[0].set_ylim(0, 1)
    _title(axes[0], "(a) training arrivals within $N$")
    _title(axes[1], "(b) critic Huber loss")
    axes[1].set_yscale("log")
    _title(axes[2], r"(c) SAC temperature $\alpha$")
    axes[2].set_yscale("log")
    for ax in axes:
        ax.set_xlabel("steps  [millions]")
    axes[0].legend(loc="lower right")
    fig.tight_layout(w_pad=1.6)
    return _save(fig, out, "fig_training_diag")


# ------------------------------------------------------------------ results figures
def fig_slices(d: dict, out: Path) -> str:
    cfg = d["cfg"]
    ext = d["lqr"]["extents"]
    conds = d["slices"]["conds"]
    fig, axes = plt.subplots(2, 3, figsize=(TEXTWIDTH, 4.6), sharex=True, sharey=True)
    tmax = 1.2
    cmap = LinearSegmentedColormap.from_list("seqblue", SEQ_BLUE[1:])
    for row, key in enumerate(("learned", "lqr_only")):
        for col, sl in enumerate(d["slices"][key]):
            ax = axes[row, col]
            ax.grid(False)
            r, z, code, t = sl["r"], sl["z"], sl["code"], sl["t"]
            ext_ = [r[0], r[-1], z[0], z[-1]]
            crash = np.where(code == 0.0, 1.0, np.nan)
            inb = np.where(code == 2.0, 1.0, np.nan)
            im = ax.imshow(np.clip(t, 0, tmax), origin="lower", extent=ext_, aspect="auto", cmap=cmap, vmin=0, vmax=tmax, interpolation="nearest")
            ax.imshow(crash, origin="lower", extent=ext_, aspect="auto", cmap=ListedColormap([CRIT]), interpolation="nearest")
            ax.imshow(inb, origin="lower", extent=ext_, aspect="auto", cmap=ListedColormap(["#d9d8d3"]), interpolation="nearest")
            zz = np.linspace(0, z[-1], 100)
            ax.plot(_cone_r(cfg, zz), zz, color=INK, lw=1.0)
            ax.add_patch(Ellipse((0.0, cfg["z_des"]), 2 * ext["dx"], 2 * ext["dz"], fill=False, ec=INK, lw=0.7, ls="--"))
            ax.set_xlim(0, r[-1])
            ax.set_ylim(0, z[-1])
            if row == 0:
                _title(ax, conds[col][0].replace("descending", "down"))
            ax.text(0.97, 0.03, f"saved {100 * sl['frac_success']:.1f}%", transform=ax.transAxes, ha="right", fontsize=7,
                    color=INK, bbox=dict(fc="white", ec="none", pad=1.5, alpha=0.85))
    for ax in axes[1]:
        ax.set_xlabel(r"radial distance $\|\Delta p\|$  [m]")
    axes[0, 0].set_ylabel("learned $\\pi_b$\n" + r"altitude $\zeta$  [m]", color=INK)
    axes[1, 0].set_ylabel("base controller only\n" + r"altitude $\zeta$  [m]", color=INK)
    cax = fig.add_axes([0.905, 0.35, 0.013, 0.5])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("arrival time in $\\mathcal{B}$  [s]", color=INK2)
    cb.outline.set_visible(False)
    handles = [Patch(fc=CRIT, label="crash (leaves the cone)"), Patch(fc="#d9d8d3", label=r"already in $\mathcal{B}$"),
               Line2D([], [], color=INK, lw=0.7, ls="--", label=r"$\mathcal{B}$ position projection")]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.47, -0.01))
    fig.subplots_adjust(left=0.1, right=0.89, bottom=0.14, top=0.94, wspace=0.08, hspace=0.12)
    return _save(fig, out, "fig_slices")


def fig_fan(d: dict, out: Path) -> str:
    cfg = d["cfg"]
    ext = d["lqr"]["extents"]
    fig, axes = plt.subplots(1, 2, figsize=(TEXTWIDTH, 3.1), sharey=True)
    for ax, reg, title in ((axes[0], "general", "(a) general region, 60 random test states"),
                           (axes[1], "edge", "(b) edge band, 60 random test states")):
        _draw_cone_rz(ax, cfg, 3.1, extents=ext)
        n_ok = n_bad = 0
        for tr in d["fan"]:
            if tr["region"] != reg:
                continue
            x = tr["x"]
            hit = np.where(tr["inB"])[0]
            fail = np.where(tr["h"] < 0)[0]
            end = hit[0] if (tr["success"] and hit.size) else (fail[0] if fail.size else len(x) - 1)
            r = _radial(x[: end + 1], cfg)
            z = x[: end + 1, 2] - cfg["pad_z"]
            c = BLUE if tr["success"] else CRIT
            ax.plot(r, z, color=c, lw=0.7, alpha=0.8 if tr["success"] else 1.0, zorder=3 if not tr["success"] else 2)
            ax.plot(r[0], z[0], "o", ms=2.6, color=c, mec="white", mew=0.4, zorder=4)
            if tr["success"]:
                n_ok += 1
            else:
                n_bad += 1
                ax.plot(r[-1], z[-1], "x", ms=4, color=CRIT, mew=1.0, zorder=5)
        ax.set_xlim(0, 2.3)
        ax.set_ylim(-0.25, 3.1)
        ax.set_xlabel(r"radial distance $\|\Delta p\|$  [m]")
        _title(ax, title)
        ax.text(0.97, 0.97, f"saved {n_ok}, crashed {n_bad}", transform=ax.transAxes, ha="right", va="top", fontsize=7.5, color=INK)
    axes[0].set_ylabel(r"altitude $\zeta$  [m]")
    handles = [Line2D([], [], color=BLUE, lw=1.0, marker="o", ms=3, label="saved (path until entry into $\\mathcal{B}$)"),
               Line2D([], [], color=CRIT, lw=1.0, marker="x", ms=4, label="crashed (path until leaving the cone)")]
    fig.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.06, 1, 1), w_pad=0.6)
    return _save(fig, out, "fig_fan")


def fig_viability(d: dict, out: Path) -> str:
    fe = d["features"]["edge"]
    ok = d["per_state"][d["run"]]["edge"]["success"]
    v = d["viability"]["edge"]
    nonviable = np.zeros_like(ok)
    nonviable[v["shared_idx"][v["best_hmin"] < 0]] = True
    failed_viable = ~ok & ~nonviable
    fig, axes = plt.subplots(1, 2, figsize=(TEXTWIDTH, 2.7))
    groups = [(ok, GOOD, "o", 7, f"saved by the policy ({ok.sum()})"),
              (failed_viable, WARN, "^", 16, f"crashed, but a safe trajectory exists ({failed_viable.sum()})"),
              (nonviable, CRIT, "x", 14, f"crashed, no safe trajectory found ({nonviable.sum()})")]
    for ax, (kx, ky, lx, ly) in zip(axes, (("h", "v_out", r"initial margin $h_{\mathrm{cone}}(x_0)$  [m]", r"outward radial speed  [m/s]"),
                                          ("zeta", "vz", r"initial altitude $\zeta_0$  [m]", r"vertical speed $v_z$  [m/s]"))):
        for mask, c, mk, s, lab in groups:
            ax.scatter(fe[kx][mask], fe[ky][mask], s=s, c=c, marker=mk, lw=0.9 if mk == "x" else 0, alpha=0.75 if mk == "o" else 1.0,
                       label=lab, zorder=3 if mk != "o" else 2)
        ax.set_xlabel(lx)
        ax.set_ylabel(ly)
    _title(axes[0], "(a) edge test states: margin vs. outward speed")
    _title(axes[1], "(b) edge test states: altitude vs. vertical speed")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.06), markerscale=1.4)
    fig.tight_layout(rect=(0, 0.06, 1, 1), w_pad=1.0)
    return _save(fig, out, "fig_viability")


def fig_actuators(d: dict, out: Path) -> str:
    cfg = d["cfg"]
    names = [(d["compare_run"], BLUE, pretty(d["compare_run"])), (d["run"], ORANGE, pretty(d["run"])),
             ("lqr_only", AQUA, "base controller only")]
    fig, axes = plt.subplots(1, 3, figsize=(TEXTWIDTH, 2.2))
    bins_t = np.linspace(0, cfg["a_cmd_max"], 41)
    bins_w = np.linspace(0, cfg["omega_max"], 37)
    for name, c, lab in names:
        s = d["actuator_samples"][name]
        axes[0].hist(s["thrust"], bins=bins_t, histtype="step", color=c, lw=1.2, density=True, label=lab)
        axes[1].hist(np.linalg.norm(s["omega"][:, :2], axis=1), bins=bins_w * np.sqrt(2) / np.sqrt(2), histtype="step", color=c, lw=1.2, density=True)
        axes[2].hist(np.abs(s["omega"][:, 2]), bins=bins_w, histtype="step", color=c, lw=1.2, density=True)
    axes[0].axvline(cfg["gravity"], color=INK2, lw=0.8, ls=":")
    axes[0].text(cfg["gravity"] - 0.6, axes[0].get_ylim()[1] * 0.6, "hover $g$", fontsize=7, color=INK, ha="right")
    for ax in axes:
        ax.set_yscale("log")
    axes[0].set_xlabel(r"collective thrust $a$  [m/s$^2$]")
    axes[1].set_xlabel(r"$\|(\omega_x,\omega_y)\|$  [rad/s]")
    axes[2].set_xlabel(r"$|\omega_z|$  [rad/s]")
    axes[0].set_ylabel("density (log)")
    _title(axes[0], "(a) thrust")
    _title(axes[1], "(b) roll/pitch rate magnitude")
    _title(axes[2], "(c) yaw rate magnitude")
    axes[2].legend(handles=[Line2D([], [], color=c, lw=1.2, label=lab) for _, c, lab in names], loc="upper right", fontsize=6.8)
    fig.tight_layout(w_pad=0.6)
    return _save(fig, out, "fig_actuators")


# -------------------------------------------------------------------- case studies
def _entry_index(tr: dict, N: int) -> int | None:
    hit = np.where(tr["inB"][1:])[0]
    fail = np.where(tr["h"][1:] < 0)[0]
    tb = hit[0] + 1 if hit.size else None
    tf = fail[0] + 1 if fail.size else None
    if tb is not None and tb <= N and (tf is None or tb < tf):
        return tb
    return None


def _fail_index(tr: dict) -> int | None:
    fail = np.where(tr["h"][1:] < 0)[0]
    return int(fail[0] + 1) if fail.size else None


def fig_case(d: dict, case: dict, out: Path) -> str:
    cfg = d["cfg"]
    dt, N, c_b = cfg["dt"], cfg["num_steps"], cfg["base_set_c"]
    ext = d["lqr"]["extents"]
    main = case["main"]
    extra = case.get("compare") or case.get("opt")
    extra_label = None
    main_color, extra_color = BLUE, ORANGE
    if case["tag"] == "G":
        extra_label = pretty(d["compare_run"])
        main_label = pretty(d["run"])
        main_color, extra_color = BACKBONE_COLOR["td3"], BACKBONE_COLOR["sac"]
    elif case["tag"] == "H":
        extra_label = "base controller only"
        main_label = "learned $\\pi_b$"
    elif case["tag"] == "E":
        extra_label = "best optimised trajectory"
        main_label = "learned $\\pi_b$"
    else:
        main_label = "learned $\\pi_b$"
    fig, axes = plt.subplots(2, 3, figsize=(TEXTWIDTH, 4.2))
    t_main = np.arange(len(main["x"])) * dt
    tb = _entry_index(main, N)
    tf = _fail_index(main)
    end = tf if (tb is None and tf is not None) else len(main["x"]) - 1

    trajs = [(main, main_color, main_label, end)]
    if extra is not None:
        e_tb = _entry_index(extra, N) if "inB" in extra else None
        e_tf = _fail_index(extra)
        e_end = e_tf if (e_tb is None and e_tf is not None) else len(extra["x"]) - 1
        trajs.append((extra, extra_color, extra_label, e_end))

    short = max(e for _, _, _, e in trajs) < 20  # both trajectories end (crash) within 0.4 s
    t_view = (max(e for _, _, _, e in trajs) + 6) * dt if short else t_main[-1]

    # (a) side view
    ax = axes[0, 0]
    zmax = max(1.6, max(tr["x"][: e + 1, 2].max() for tr, _, _, e in trajs) + 0.25)
    _draw_cone_rz(ax, cfg, zmax, extents=ext)
    for tr, c, lab, e in trajs:
        r = _radial(tr["x"][: e + 1], cfg)
        z = tr["x"][: e + 1, 2] - cfg["pad_z"]
        ax.plot(r, z, color=c, lw=1.3, label=lab)
        ax.plot(r[0], z[0], "o", ms=4, color=c, mec="white", mew=0.6, zorder=5)
        k = _entry_index(tr, N) if "inB" in tr else None
        if k is not None:
            ax.plot(r[k], z[k], "s", ms=4, color=c, mec="white", mew=0.6, zorder=5)
        if e < len(tr["x"]) - 1 or (tr["h"][e] < 0):
            ax.plot(r[-1], z[-1], "x", ms=5, color=CRIT, mew=1.2, zorder=6)
    rmax = max(1.2, max(_radial(tr["x"][: e + 1], cfg).max() for tr, _, _, e in trajs) + 0.2)
    if short:
        rr = np.concatenate([_radial(tr["x"][: e + 1], cfg) for tr, _, _, e in trajs])
        zz_ = np.concatenate([tr["x"][: e + 1, 2] - cfg["pad_z"] for tr, _, _, e in trajs])
        pad_ = 0.12
        ax.set_xlim(rr.min() - pad_, rr.max() + pad_)
        ax.set_ylim(zz_.min() - pad_, zz_.max() + pad_)
    else:
        ax.set_xlim(0, rmax)
        ax.set_ylim(min(-0.15, min(tr["x"][: e + 1, 2].min() for tr, _, _, e in trajs) - 0.05), zmax)
    ax.set_xlabel(r"$\|\Delta p\|$  [m]")
    ax.set_ylabel(r"$\zeta$  [m]")
    _title(ax, "(a) side view (radial plane)")

    # (b) top view
    ax = axes[0, 1]
    th = np.linspace(0, 2 * np.pi, 200)
    z0 = main["x"][0, 2] - cfg["pad_z"]
    R0 = float(_cone_r(cfg, np.array([z0]))[0])
    ax.plot(R0 * np.cos(th) + cfg["pad_x"], R0 * np.sin(th) + cfg["pad_y"], color=INK, lw=0.9, ls="-")
    Rd = float(_cone_r(cfg, np.array([cfg["z_des"]]))[0])
    ax.plot(Rd * np.cos(th), Rd * np.sin(th), color=MUTED, lw=0.8, ls=":")
    ax.add_patch(Ellipse((cfg["pad_x"], cfg["pad_y"]), 2 * ext["dx"], 2 * ext["dy"], fc=BAND, ec=INK2, lw=0.7, ls="--", zorder=0.5))
    for tr, c, lab, e in trajs:
        ax.plot(tr["x"][: e + 1, 0], tr["x"][: e + 1, 1], color=c, lw=1.3)
        ax.plot(tr["x"][0, 0], tr["x"][0, 1], "o", ms=4, color=c, mec="white", mew=0.6, zorder=5)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("$x$  [m]")
    ax.set_ylabel("$y$  [m]")
    _title(ax, r"(b) top view (solid: cone at $\zeta_0$)")

    # (c) cone margin
    ax = axes[0, 2]
    for tr, c, lab, e in trajs:
        t = np.arange(len(tr["h"])) * dt
        ax.plot(t[: e + 1], tr["h"][: e + 1], color=c, lw=1.3)
    ax.axhline(0, color=CRIT, lw=0.9)
    ax.set_xlabel("time  [s]")
    ax.set_ylabel(r"$h_{\mathrm{cone}}$  [m]")
    _title(ax, "(c) safety margin")

    # (d) thrust
    ax = axes[1, 0]
    for tr, c, lab, e in trajs:
        t = np.arange(len(tr["u"])) * dt
        ax.plot(t[:e], tr["u"][:e, 0], color=c, lw=1.1, drawstyle="steps-post")
    ax.axhline(cfg["a_cmd_max"], color=INK2, lw=0.8, ls="--")
    ax.axhline(cfg["gravity"], color=MUTED, lw=0.8, ls=":")
    ax.text(0.99, cfg["a_cmd_max"] - 1.0, r"$a_{\max}=4g$", transform=ax.get_yaxis_transform(), ha="right", va="top", fontsize=7, color=INK)
    ax.text(0.99, cfg["gravity"] + 0.8, "$g$", transform=ax.get_yaxis_transform(), ha="right", fontsize=7, color=INK)
    ax.set_ylim(-1.5, cfg["a_cmd_max"] + 2)
    ax.set_xlabel("time  [s]")
    ax.set_ylabel(r"$a$  [m/s$^2$]")
    _title(ax, "(d) collective thrust command")

    # (e) body rates of the main policy (and the comparison, dashed, for G/H)
    ax = axes[1, 1]
    comps = [(0, BLUE, r"$\omega_x$"), (1, ORANGE, r"$\omega_y$"), (2, AQUA, r"$\omega_z$")]
    t = np.arange(len(main["u"])) * dt
    for j, c, lab in comps:
        ax.plot(t[:end], main["u"][:end, 1 + j], color=c, lw=1.0, drawstyle="steps-post", label=lab)
    if case["tag"] in ("G", "H") and extra is not None:
        te = np.arange(len(extra["u"])) * dt
        e_end = trajs[1][3]
        for j, c, lab in comps:
            ax.plot(te[:e_end], extra["u"][:e_end, 1 + j], color=c, lw=0.8, ls=(0, (2.5, 1.5)), drawstyle="steps-post")
    for s in (1, -1):
        ax.axhline(s * cfg["omega_max"], color=INK2, lw=0.8, ls="--")
    ax.set_ylim(-cfg["omega_max"] - 2, cfg["omega_max"] + 2)
    ax.set_xlabel("time  [s]")
    ax.set_ylabel(r"$\omega$  [rad/s]")
    _title(ax, "(e) body rates" + {"G": ", dashed: SAC", "H": ", dashed: base ctrl."}.get(case["tag"], ""))
    ax.legend(loc="upper right", ncol=3, handlelength=1.2, columnspacing=0.8, fontsize=7)

    # (f) Lyapunov level
    ax = axes[1, 2]
    for tr, c, lab, e in trajs:
        if "V" not in tr:
            continue
        tt = np.arange(len(tr["V"])) * dt
        ax.plot(tt[: e + 1], tr["V"][: e + 1] / c_b, color=c, lw=1.3)
    ax.axhline(1.0, color=INK2, lw=0.9)
    ax.text(0.99, 1.15, r"$\partial\mathcal{B}$", transform=ax.get_yaxis_transform(), ha="right", fontsize=7, color=INK)
    ax.set_yscale("log")
    lo, hi = ax.get_ylim()
    ax.set_ylim(max(lo, 1e-4), hi)
    ax.set_xlabel("time  [s]")
    ax.set_ylabel(r"$V(x)/c_{\mathcal{B}}$")
    _title(ax, "(f) base-set level")

    for ax in (axes[0, 2], axes[1, 0], axes[1, 1], axes[1, 2]):
        ax.set_xlim(0, t_view)
        if tb is not None:
            ax.axvspan(tb * dt, t_main[-1], color=BAND, lw=0, zorder=0)
            ax.axvline(tb * dt, color=INK2, lw=0.8, ls=":")
        if not short:
            ax.axvline(N * dt, color=MUTED, lw=0.7, ls=(0, (1, 2)))
    handles = [Line2D([], [], color=main_color, lw=1.3, label=main_label)]
    if extra is not None:
        handles.append(Line2D([], [], color=extra_color, lw=1.3, label=extra_label))
    handles.append(Line2D([], [], ls="", marker="o", ms=4, color=INK2, label="start"))
    any_entry = any(("inB" in tr) and (_entry_index(tr, N) is not None) for tr, _, _, _ in trajs)
    if any_entry:
        handles += [Line2D([], [], ls="", marker="s", ms=4, color=INK2, label=r"entry into $\mathcal{B}$ (hand-off to $\pi_{\mathcal{B}}$)"),
                    Patch(fc=BAND, ec="none", label=r"$\pi_{\mathcal{B}}$ (LQR) phase")]
    if tf is not None or (extra is not None and trajs[-1][3] < len(extra["x"]) - 1):
        handles.append(Line2D([], [], ls="", marker="x", ms=5, color=CRIT, mew=1.2, label="leaves the cone"))
    fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 4), bbox_to_anchor=(0.5, -0.045), fontsize=7)
    fig.tight_layout(rect=(0, 0.05, 1, 1), h_pad=0.9, w_pad=0.9)
    return _save(fig, out, f"fig_case_{case['tag']}")


def make_all(d: dict, out: Path) -> list[str]:
    names = [fig_geometry(d, out), fig_level_bounds(d, out), fig_training(d, out), fig_training_diag(d, out),
             fig_slices(d, out), fig_fan(d, out), fig_viability(d, out), fig_actuators(d, out)]
    for case in d["cases"]:
        names.append(fig_case(d, case, out))
    return names
