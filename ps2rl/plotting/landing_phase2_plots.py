"""Figures and a 3D GIF for the Phase-II landing filter evaluation.

All geometry (cone, pad) comes from the landing config of the backup checkpoint and the
reference from the Phase-II run; nothing is fixed here. The "side view" is the vertical
plane through the pad axis along the reference's miss direction (pad -> reference end,
falling back to pad -> reference start), so it shows the cone boundary the reference
crosses whatever the pad or miss direction are.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import animation  # noqa: E402
import numpy as np  # noqa: E402

COLORS = {"unfiltered": "tab:red", "filtered": "tab:blue", "reference": "k", "cone": "tab:green"}
LABELS = {"unfiltered": "policy only", "filtered": "policy + BCBF filter"}
ACTION_NAMES = ("a_cmd [m/s²]", "ω_x [rad/s]", "ω_y [rad/s]", "ω_z [rad/s]")


@dataclass(frozen=True)
class Scene:
    pad: np.ndarray  # (3,)
    r0: float
    tan_theta: float
    eps: float
    ref: np.ndarray  # (N, 10) reference states
    ref_t: np.ndarray  # (N,)
    dt: float
    direction: np.ndarray  # (2,) unit horizontal side-view direction

    @classmethod
    def build(cls, landing_cfg, ref_states: np.ndarray, ref_t: np.ndarray, dt: float) -> "Scene":
        pad = np.array([landing_cfg.pad_x, landing_cfg.pad_y, landing_cfg.pad_z], dtype=np.float64)
        d = None
        for p in (ref_states[-1, 0:2], ref_states[0, 0:2]):
            v = np.asarray(p, dtype=np.float64) - pad[:2]
            if np.linalg.norm(v) > 1e-6:
                d = v / np.linalg.norm(v)
                break
        if d is None:
            d = np.array([1.0, 0.0])
        return cls(pad, float(landing_cfg.cone_r0), float(np.tan(np.radians(landing_cfg.cone_theta_deg))),
                   float(landing_cfg.cone_eps), np.asarray(ref_states, np.float64), np.asarray(ref_t, np.float64),
                   float(dt), d)

    def radius(self, zeta: np.ndarray) -> np.ndarray:
        """Horizontal radius of {h = 0} at height zeta above the pad (smooth cone)."""
        a = self.r0 + self.tan_theta * np.asarray(zeta)
        return np.sqrt(np.maximum(a * a - self.eps ** 2, 0.0)) * (a > self.eps)

    def side(self, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        rel = np.asarray(p)[..., 0:3] - self.pad
        return rel[..., 0:2] @ self.direction, rel[..., 2]

    def radial(self, p: np.ndarray) -> np.ndarray:
        rel = np.asarray(p)[..., 0:2] - self.pad[:2]
        return np.linalg.norm(rel, axis=-1)


def _tilt_deg(q):
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    return np.degrees(np.arccos(np.clip(1.0 - 2.0 * (q[..., 1] ** 2 + q[..., 2] ** 2), -1.0, 1.0)))


def _rotmat(q):
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def _zeta_top(scene: Scene, trajs: list[np.ndarray]) -> float:
    zs = [scene.ref[:, 2] - scene.pad[2]] + [np.asarray(t)[..., 2] - scene.pad[2] for t in trajs]
    return float(max(np.nanmax(z) for z in zs)) + 0.3


def _draw_cone_side(ax, scene: Scene, zeta_top: float, zeta_bot: float = 0.0):
    z = np.linspace(zeta_bot, zeta_top, 200)
    r = scene.radius(z)
    ax.fill_betweenx(z, -r, r, color=COLORS["cone"], alpha=0.12, lw=0, label="approach cone (h ≥ 0)")
    ax.plot(-r, z, color=COLORS["cone"], lw=1.0)
    ax.plot(r, z, color=COLORS["cone"], lw=1.0)
    ax.plot([-scene.r0, scene.r0], [0.0, 0.0], color=COLORS["cone"], lw=5, solid_capstyle="butt", label="pad (radius r0)")
    ax.axhline(0.0, color="0.5", lw=0.6)


def _save(fig, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return str(path)


# ----------------------------------------------------------------- overview figure
def plot_overview(scene: Scene, recs: dict, h: dict, highlight: int, out: Path, title: str) -> str:
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    trajs = [recs[v]["traj"] for v in recs]
    zt = _zeta_top(scene, trajs)
    a = ax[0, 0]
    _draw_cone_side(a, scene, zt)
    s_ref, z_ref = scene.side(scene.ref)
    for v, rec in recs.items():
        s, z = scene.side(rec["traj"])
        for i in range(s.shape[0]):
            a.plot(s[i], z[i], color=COLORS[v], lw=0.6, alpha=0.25)
        a.plot(s[highlight], z[highlight], color=COLORS[v], lw=2.2, label=f"{LABELS[v]} (ep {highlight})")
    a.plot(s_ref, z_ref, "--", color=COLORS["reference"], lw=1.6, label="reference")
    a.plot(s_ref[-1], z_ref[-1], "x", color=COLORS["reference"], ms=9, mew=2)
    a.set_xlabel("horizontal distance from pad axis along the reference miss direction [m]")
    a.set_ylabel("height above pad [m]")
    a.set_title("(a) side view, all episodes")
    a.set_aspect("equal", adjustable="datalim")
    a.legend(fontsize=8, loc="upper left")
    a.grid(alpha=0.3)

    a = ax[0, 1]
    T1 = next(iter(h.values())).shape[1]
    t = np.arange(T1) * scene.dt
    for v, hv in h.items():
        for i in range(hv.shape[0]):
            a.plot(t, hv[i], color=COLORS[v], lw=0.6, alpha=0.25)
        a.plot(t, hv[highlight], color=COLORS[v], lw=2.2, label=LABELS[v])
    a.axhline(0.0, color="k", lw=1.0)
    a.axhspan(min(0.0, min(np.nanmin(hv) for hv in h.values())) - 0.05, 0.0, color="tab:red", alpha=0.07)
    a.set_xlabel("t [s]")
    a.set_ylabel("h_cone(x) [m]  (< 0: outside the cone)")
    a.set_title("(b) cone margin")
    a.legend(fontsize=8)
    a.grid(alpha=0.3)

    a = ax[1, 0]
    for v, rec in recs.items():
        e = np.where(rec["active"], rec["pos_err"], np.nan)
        tt = np.arange(1, e.shape[1] + 1) * scene.dt
        a.fill_between(tt, np.nanpercentile(e, 10, axis=0), np.nanpercentile(e, 90, axis=0), color=COLORS[v], alpha=0.18)
        a.plot(tt, np.nanmedian(e, axis=0), color=COLORS[v], lw=2, label=f"{LABELS[v]} (median, 10–90%)")
    a.set_xlabel("t [s]")
    a.set_ylabel("‖p − p_ref‖ [m]")
    a.set_title("(c) reference tracking error")
    a.legend(fontsize=8)
    a.grid(alpha=0.3)

    a = ax[1, 1]
    rec = recs["filtered"]
    act = rec["active"].astype(bool)
    tt = np.arange(rec["du_norm"].shape[1]) * scene.dt
    frac = np.where(act, rec["du_norm"] > 1e-3, False).sum(0) / np.maximum(act.sum(0), 1)
    a.plot(tt, frac, color="tab:purple", lw=2, label="fraction of episodes the filter modifies")
    a.plot(tt, np.nanmedian(np.where(act, rec["du_norm"], np.nan), axis=0), color="tab:blue", lw=1.5,
           label="median ‖Δu‖ (normalised by half-range)")
    a.set_xlabel("t [s]")
    a.set_ylabel("filter activity")
    a2 = a.twinx()
    a2.plot(tt, np.nanmax(np.where(act, rec["slack"], np.nan), axis=0), color="tab:orange", lw=1.0, label="max slack")
    a2.set_ylabel("QP slack (max over episodes)", color="tab:orange")
    a.set_title("(d) filter activity")
    h1, l1 = a.get_legend_handles_labels()
    h2, l2 = a2.get_legend_handles_labels()
    a.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper left")
    a.grid(alpha=0.3)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    return _save(fig, out)


# ----------------------------------------------------------------- case figures
def plot_case(scene: Scene, recs: dict, h: dict, ep: int, out: Path, title: str, labels: dict | None = None) -> str:
    lab = {**LABELS, **(labels or {})}
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    trajs = [recs[v]["traj"][ep] for v in recs]
    zt = _zeta_top(scene, trajs)
    a = ax[0, 0]
    _draw_cone_side(a, scene, zt)
    s_ref, z_ref = scene.side(scene.ref)
    a.plot(s_ref, z_ref, "--", color=COLORS["reference"], lw=1.6, label="reference")
    for v, rec in recs.items():
        s, z = scene.side(rec["traj"][ep])
        a.plot(s, z, color=COLORS[v], lw=2, label=lab[v])
        a.plot(s[0], z[0], "o", color=COLORS[v], ms=5)
        a.plot(s[-1], z[-1], "s", color=COLORS[v], ms=6)
    a.set_aspect("equal", adjustable="datalim")
    a.set_xlabel("along miss direction [m]")
    a.set_ylabel("height above pad [m]")
    a.set_title("(a) side view (o start, ■ end)")
    a.legend(fontsize=8, loc="upper left")
    a.grid(alpha=0.3)

    a = ax[0, 1]
    th = np.linspace(0, 2 * np.pi, 200)
    for zc, ls in ((0.0, "-"), (zt / 3.0, ":"), (2.0 * zt / 3.0, ":")):
        rc = float(scene.radius(zc))
        a.plot(scene.pad[0] + rc * np.cos(th), scene.pad[1] + rc * np.sin(th), ls, color=COLORS["cone"], lw=1.2,
               label=f"cone at {zc:.2f} m" if zc else "pad / cone at 0 m")
    a.plot(scene.ref[:, 0], scene.ref[:, 1], "--", color=COLORS["reference"], lw=1.6, label="reference")
    for v, rec in recs.items():
        tr = rec["traj"][ep]
        a.plot(tr[:, 0], tr[:, 1], color=COLORS[v], lw=2, label=lab[v])
    a.set_aspect("equal", adjustable="datalim")
    a.set_xlabel("x [m]")
    a.set_ylabel("y [m]")
    a.set_title("(b) top view")
    a.legend(fontsize=8)
    a.grid(alpha=0.3)

    a = ax[1, 0]
    t = np.arange(h["filtered"].shape[1]) * scene.dt
    for v in recs:
        a.plot(t, h[v][ep], color=COLORS[v], lw=2, label=lab[v])
    a.axhline(0.0, color="k", lw=1.0)
    a.set_xlabel("t [s]")
    a.set_ylabel("h_cone [m]")
    a.set_title("(c) cone margin")
    a.legend(fontsize=8)
    a.grid(alpha=0.3)

    a = ax[1, 1]
    for v, rec in recs.items():
        tr = rec["traj"][ep]
        a.plot(t, tr[:, 2] - scene.pad[2], color=COLORS[v], lw=2, label=f"height, {lab[v]}")
        a.plot(t, scene.radial(tr), color=COLORS[v], lw=1.2, ls="-.", label=f"radial distance, {lab[v]}")
    tr_ref = np.arange(scene.ref.shape[0]) * float(scene.ref_t[1] - scene.ref_t[0] if scene.ref_t.size > 1 else scene.dt)
    a.plot(tr_ref, scene.ref[:, 2] - scene.pad[2], "--", color="k", lw=1.2, label="reference height")
    a.plot(tr_ref, scene.radial(scene.ref), ":", color="k", lw=1.2, label="reference radial distance")
    a.set_xlabel("t [s]")
    a.set_ylabel("[m]")
    a.set_title("(d) height and distance from the pad axis")
    a.legend(fontsize=7, ncol=2)
    a.grid(alpha=0.3)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    return _save(fig, out)


def plot_actuators(scene: Scene, recs: dict, ep: int, low, high, out: Path, title: str, gravity: float, labels: dict | None = None) -> str:
    lab = {**LABELS, **(labels or {})}
    fig, ax = plt.subplots(5, 1, figsize=(11, 12), sharex=True)
    T = recs["filtered"]["u"].shape[1]
    t = np.arange(T) * scene.dt
    for j in range(4):
        a = ax[j]
        a.plot(t, recs["unfiltered"]["u"][ep, :, j], color=COLORS["unfiltered"], lw=1.4, label=f"{lab['unfiltered']}: applied")
        a.plot(t, recs["filtered"]["u_raw"][ep, :, j], "--", color=COLORS["filtered"], lw=1.2,
               label=f"{lab['filtered']}: proposal u_ref")
        a.plot(t, recs["filtered"]["u"][ep, :, j], color=COLORS["filtered"], lw=1.8, label=f"{lab['filtered']}: applied u_safe")
        a.axhline(low[j], color="0.4", ls=":", lw=1)
        a.axhline(high[j], color="0.4", ls=":", lw=1)
        if j == 0:
            a.axhline(gravity, color="0.6", ls="-.", lw=0.8, label="hover thrust g")
        a.set_ylabel(ACTION_NAMES[j])
        a.grid(alpha=0.3)
        pad = 0.08 * (high[j] - low[j])
        lo_d = min(np.nanmin(recs[v]["u"][ep, :, j]) for v in recs)
        hi_d = max(np.nanmax(recs[v]["u"][ep, :, j]) for v in recs)
        lo_d = min(lo_d, np.nanmin(recs["filtered"]["u_raw"][ep, :, j]))
        hi_d = max(hi_d, np.nanmax(recs["filtered"]["u_raw"][ep, :, j]))
        a.set_ylim(max(low[j] - pad, lo_d - pad), min(high[j] + pad, hi_d + pad))
    ax[0].legend(fontsize=8, ncol=3, loc="upper right")
    a = ax[4]
    a.plot(t, recs["filtered"]["du_norm"][ep], color="tab:purple", lw=1.6, label="‖u_safe − u_ref‖ (normalised)")
    a.set_ylabel("intervention")
    a2 = a.twinx()
    a2.plot(t, recs["filtered"]["slack"][ep], color="tab:orange", lw=1.0, label="QP slack")
    a2.set_ylabel("slack", color="tab:orange")
    h1, l1 = a.get_legend_handles_labels()
    h2, l2 = a2.get_legend_handles_labels()
    a.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper right")
    a.set_xlabel("t [s]")
    a.grid(alpha=0.3)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    return _save(fig, out)


def plot_distributions(ep_metrics: dict, recoverable: np.ndarray, r0: float, out: Path, title: str) -> str:
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
    specs = (("h_min", "min h_cone over the episode [m]", 0.0),
             ("final_pad_dist_xy", "final horizontal distance to pad centre [m]", r0),
             ("pos_err_mean", "mean tracking error [m]", None))
    for a, (key, label, line) in zip(ax, specs):
        vals = [ep_metrics[v][key] for v in ("unfiltered", "filtered")]
        lo = min(np.nanmin(x) for x in vals)
        hi = max(np.nanmax(x) for x in vals)
        bins = np.linspace(lo, hi if hi > lo else lo + 1e-3, 30)
        for v, x in zip(("unfiltered", "filtered"), vals):
            a.hist(x[recoverable], bins=bins, color=COLORS[v], alpha=0.55, label=f"{LABELS[v]} (start in C_N)")
            if (~recoverable).any():
                a.hist(x[~recoverable], bins=bins, color=COLORS[v], alpha=0.9, histtype="step", lw=1.5,
                       label=f"{LABELS[v]} (start not in C_N)")
        if line is not None:
            a.axvline(line, color="k", lw=1)
        a.set_xlabel(label)
        a.set_ylabel("episodes")
        a.grid(alpha=0.3)
    ax[0].legend(fontsize=7)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    return _save(fig, out)


# ----------------------------------------------------------------- GIF
def save_gif(scene: Scene, recs: dict, h: dict, ep: int, out: Path, title: str, *,
             slowdown: float = 0.25, stride: int = 1, trail: int = 60, dpi: int = 90, labels: dict | None = None,
             touchdown: dict | None = None, zoom: dict | None = None, xlabel: str = "along miss direction [m]") -> str:
    """3D view + side view (+ an optional close-up side view ``zoom={"s": (lo, hi), "z": (lo, hi)}``).

    ``touchdown`` maps a variant to a frame index, or to (index, message), from which the status
    line shows the message ("LANDED" by default).
    """
    lab = {**LABELS, **(labels or {})}
    touchdown = touchdown or {}
    wlab = max(len(v) for v in lab.values()) + 1
    trajs = {v: recs[v]["traj"][ep] for v in recs}
    du = recs["filtered"]["du_norm"][ep]
    n = next(iter(trajs.values())).shape[0]
    frames = list(range(0, n, max(1, stride)))
    if frames[-1] != n - 1:
        frames.append(n - 1)
    zt = _zeta_top(scene, list(trajs.values()))
    allp = np.concatenate([scene.ref[:, 0:3]] + [t[:, 0:3] for t in trajs.values()], axis=0)
    span = float(max(np.ptp(allp[:, 0]), np.ptp(allp[:, 1]), zt, 1.0))
    arm = 0.06 * span

    ncol = 3 if zoom else 2
    fig = plt.figure(figsize=(6.75 * ncol, 6.0))
    ax3 = fig.add_subplot(1, ncol, 1, projection="3d")
    ax2 = fig.add_subplot(1, ncol, 2)
    axz = fig.add_subplot(1, ncol, 3) if zoom else None

    # cone surface
    zz = np.linspace(0.0, zt, 18)
    th = np.linspace(0, 2 * np.pi, 48)
    Z, TH = np.meshgrid(zz, th)
    R = scene.radius(Z)
    ax3.plot_surface(scene.pad[0] + R * np.cos(TH), scene.pad[1] + R * np.sin(TH), scene.pad[2] + Z,
                     color=COLORS["cone"], alpha=0.12, linewidth=0, shade=False)
    ax3.plot_wireframe(scene.pad[0] + R * np.cos(TH), scene.pad[1] + R * np.sin(TH), scene.pad[2] + Z,
                       color=COLORS["cone"], alpha=0.25, linewidth=0.4, rstride=6, cstride=3)
    pr = np.linspace(0, scene.r0, 5)
    PR, PTH = np.meshgrid(pr, th)
    ax3.plot_surface(scene.pad[0] + PR * np.cos(PTH), scene.pad[1] + PR * np.sin(PTH), np.full_like(PR, scene.pad[2]),
                     color=COLORS["cone"], alpha=0.6, linewidth=0, shade=False)
    ax3.plot(scene.ref[:, 0], scene.ref[:, 1], scene.ref[:, 2], "--", color="k", lw=1.2, label="reference")
    c = allp.mean(axis=0)
    half = 0.55 * span
    ax3.set_xlim(c[0] - half, c[0] + half)
    ax3.set_ylim(c[1] - half, c[1] + half)
    ax3.set_zlim(min(scene.pad[2], allp[:, 2].min()) - 0.05, scene.pad[2] + zt)
    ax3.set_box_aspect((1.0, 1.0, (zt + 0.05) / (2 * half)))
    ax3.set_xlabel("x [m]")
    ax3.set_ylabel("y [m]")
    ax3.set_zlabel("z [m]")
    ax3.view_init(elev=20.0, azim=-60.0)

    _draw_cone_side(ax2, scene, zt, zeta_bot=min(0.0, allp[:, 2].min() - scene.pad[2]))
    s_ref, z_ref = scene.side(scene.ref)
    ax2.plot(s_ref, z_ref, "--", color="k", lw=1.2, label="reference")
    ax2.set_aspect("equal", adjustable="datalim")
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel("height above pad [m]")
    ax2.grid(alpha=0.3)
    if axz is not None:
        _draw_cone_side(axz, scene, max(zt, zoom["z"][1]), zeta_bot=min(0.0, zoom["z"][0]))
        axz.plot(s_ref, z_ref, "--", color="k", lw=1.2)
        axz.set_xlim(*zoom["s"])
        axz.set_ylim(*zoom["z"])
        axz.set_aspect("equal")
        axz.set_xlabel(xlabel)
        axz.set_title("close-up near the platform", fontsize=10)
        axz.grid(alpha=0.3)

    art = {}
    for v, tr in trajs.items():
        col = COLORS[v]
        art[v] = {
            "arm1": ax3.plot([], [], [], "-", color=col, lw=2.2)[0],
            "arm2": ax3.plot([], [], [], "-", color=col, lw=2.2)[0],
            "up": ax3.plot([], [], [], "-", color="0.2", lw=1.4)[0],
            "trail3": ax3.plot([], [], [], "-", color=col, lw=1.2, alpha=0.8, label=lab[v])[0],
            "trail2": ax2.plot([], [], "-", color=col, lw=1.6, label=lab[v])[0],
            "dot2": ax2.plot([], [], "o", color=col, ms=7)[0],
        }
        if axz is not None:
            art[v]["trailz"] = axz.plot([], [], "-", color=col, lw=1.8)[0]
            art[v]["dotz"] = axz.plot([], [], "o", color=col, ms=8)[0]
    ax3.legend(fontsize=8, loc="upper left")
    ax2.legend(fontsize=8, loc="lower right")
    fig.subplots_adjust(left=0.02, right=0.98, top=0.88, bottom=0.1, wspace=0.12 if not zoom else 0.18)
    ttl = fig.suptitle("", fontsize=11)
    status = ax2.text(0.02, 0.98, "", transform=ax2.transAxes, va="top", fontsize=8 if zoom else 9, family="monospace",
                      bbox=dict(boxstyle="round", fc="white", alpha=0.85))

    def update(fi):
        k = frames[fi]
        lines = []
        for v, tr in trajs.items():
            p = tr[k, 0:3]
            Rm = _rotmat(tr[k, 6:10])
            b1 = Rm @ np.array([1.0, 1.0, 0.0]) / np.sqrt(2) * arm
            b2 = Rm @ np.array([1.0, -1.0, 0.0]) / np.sqrt(2) * arm
            up = Rm @ np.array([0.0, 0.0, 1.0]) * arm
            for key, vec in (("arm1", b1), ("arm2", b2)):
                seg = np.stack([p - vec, p + vec])
                art[v][key].set_data(seg[:, 0], seg[:, 1])
                art[v][key].set_3d_properties(seg[:, 2])
            seg = np.stack([p, p + up])
            art[v]["up"].set_data(seg[:, 0], seg[:, 1])
            art[v]["up"].set_3d_properties(seg[:, 2])
            lo = max(0, k - trail)
            art[v]["trail3"].set_data(tr[lo:k + 1, 0], tr[lo:k + 1, 1])
            art[v]["trail3"].set_3d_properties(tr[lo:k + 1, 2])
            s, z = scene.side(tr[: k + 1])
            art[v]["trail2"].set_data(s, z)
            art[v]["dot2"].set_data([s[-1]], [z[-1]])
            if axz is not None:
                art[v]["trailz"].set_data(s, z)
                art[v]["dotz"].set_data([s[-1]], [z[-1]])
            td = touchdown.get(v)
            td, msg = td if isinstance(td, tuple) else (td, "LANDED")
            landed = f"  {msg}" if (td is not None and k >= td) else ""
            lines.append(f"{lab[v]:<{wlab}} h={h[v][ep, k]:+.2f} m  z={tr[k, 2] - scene.pad[2]:5.2f} m  tilt={_tilt_deg(tr[k, 6:10]):3.0f}°{landed}")
        kk = min(k, du.shape[0] - 1)
        lines.append(f"filter {'ACTIVE' if du[kk] > 1e-3 else 'idle  '}  ‖Δu‖={du[kk]:.2f}")
        status.set_text("\n".join(lines))
        ttl.set_text(f"{title}\nt={k * scene.dt:.2f}s | slowdown={slowdown:g}x")
        return []

    anim = animation.FuncAnimation(fig, update, frames=len(frames), blit=False)
    fps = max(1, int(round(1.0 / (scene.dt * max(1, stride) / slowdown))))
    out.parent.mkdir(parents=True, exist_ok=True)
    anim.save(str(out), writer=animation.PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)
    return str(out)


__all__ = ["Scene", "plot_actuators", "plot_case", "plot_distributions", "plot_overview", "save_gif"]
