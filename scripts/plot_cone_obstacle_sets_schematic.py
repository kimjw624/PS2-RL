#!/usr/bin/env python
"""Schematic of the outside-cone task's sets (slides / report).

    python scripts/plot_cone_obstacle_sets_schematic.py --out outputs/cone_obstacle_phase1/sets_schematic.png

Left: side view in the funnel's coordinates (r, z): keep-out (inside the funnel, below the floor),
target cylinder, the reference, the base set B (9-D hover ellipsoid, position slice at rest) and the
tube-tightened ("shrunk") sets at the end of the backup horizon. Right: the tube s(tau) and the
margins m_cone, m_floor, m_base along the horizon, for the per-step growth bound 1 (worst case) and
for the LQR's contraction rate inside B.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from cone_obstacle_common import load_task, task_config  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", default="docker_batch/configs/cone_obstacle_ue.json")
    p.add_argument("--out", default="outputs/cone_obstacle_phase1/sets_schematic.png")
    p.add_argument("--growth_in_b", type=float, default=-1.0, help="LQR contraction ||A_cl||_P (< 0: computed)")
    a = p.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse

    from ps2rl.sets.quadrotor_cone_obstacle_sets import obstacle_sets, position_metric
    from ps2rl.uncertainty.cone_obstacle_ue_tube import UEConfig, nominal_tube, tube_constants, tube_margins

    task = load_task(a.task)
    sc = task_config(task)
    ue = UEConfig.from_dict(task.get("ue", {}))
    safe, base = obstacle_sets(sc)
    tc = tube_constants(sc, base)
    ctrl = base.controller
    n, dt = int(sc.num_steps), float(sc.dt)
    if a.growth_in_b < 0:  # ||F||_P of the closed-loop LQR step at the hover point (d_hat = 0)
        import jax.numpy as jnp
        from ps2rl.phase1_sa.quadrotor_landing_ue_sa_env import landing_ue_step_fn
        from ps2rl.uncertainty.cone_obstacle_ue_tube import make_growth_fn
        plant = landing_ue_step_fn(sc)
        lo, hi = jnp.asarray(sc.action_low, jnp.float32), jnp.asarray(sc.action_high, jnp.float32)
        step = lambda x: plant(x, jnp.clip(ctrl.action(x), lo, hi), jnp.zeros(3, jnp.float32))
        x_h = jnp.asarray([*sc.hover_points[0], 0, 0, 0, 1, 0, 0, 0], jnp.float32)
        g_b = float(make_growth_fn(step, ctrl, tc)(x_h))
    else:
        g_b = float(a.growth_in_b)
    taus = np.arange(n + 1) * dt
    s1 = nominal_tube(n, 1.0, ue, tc, dt)
    sb = nominal_tube(n, g_b, ue, tc, dt)
    m1 = np.array([tube_margins(s, ue, tc) for s in s1], float)
    mb = np.array([tube_margins(s, ue, tc) for s in sb], float)

    tip, tt, eps = float(sc.tip_z), float(sc.tan_theta), float(sc.cone_eps)
    floor, R, top = float(sc.floor_z), float(sc.target_radius), float(sc.cylinder_top)
    hp = np.asarray(sc.hover_points[0], float)
    r_b = float(np.hypot(hp[0] - sc.axis_x, hp[1] - sc.axis_y))
    m_h, m_v, _ = position_metric(ctrl)
    c_b = float(sc.base_set_c)

    def surf(r, m=0.0):  # funnel surface shifted by a margin m on h
        return tip + (np.sqrt(r ** 2 + eps ** 2) - eps - m) / tt

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.6), gridspec_kw={"width_ratios": [1.35, 1.0]})
    ax = axes[0]
    rr = np.linspace(0, 3.6, 300)
    z_max = 3.2
    ax.fill_between(rr, surf(rr), z_max, color="#f3d9d3", lw=0)
    ax.plot(rr, surf(rr), color="#c4553f", lw=1.6, label=r"$h_{cone}=0$ (funnel wall)")
    ax.fill_between(rr, 0, floor, color="#f3d9d3", lw=0)
    ax.axhline(floor, color="#c4553f", lw=1.2, ls="--", label=r"$h_{floor}=0$ (z = %.1f m)" % floor)
    ax.axhline(0, color="k", lw=1.2)
    mc_end, mf_end, mb_end = m1[-1]
    ax.plot(rr, surf(rr, mc_end), color="#c4553f", lw=1.0, ls=":", label=r"tightened: $h_{cone}=m_{cone}(T)$ = %.2f m" % mc_end)
    ax.axhline(floor + mf_end, color="#c4553f", lw=1.0, ls=":", label=r"tightened: $h_{floor}=m_{floor}(T)$ = %.2f m" % mf_end)
    ax.fill_between([0, R], 0, top, color="#3d3c39", lw=0, label=f"target cylinder (R = {R:g} m, top = tip)")
    ref = np.load(ROOT / task["reference"]) if not Path(task["reference"]).exists() else np.load(task["reference"])
    rs = np.asarray(ref["states"])
    ax.plot(np.hypot(rs[:, 0] - sc.axis_x, rs[:, 1] - sc.axis_y), rs[:, 2], "k--", lw=1.3, label="reference L3-1-f")
    # base set: position slice at rest, V = e^T P e <= c  ->  m_h r^2 + m_v dz^2 <= c
    wh, wv = 2 * np.sqrt(c_b / m_h), 2 * np.sqrt(c_b / m_v)
    ax.add_patch(Ellipse((r_b, hp[2]), wh, wv, fc="#2a78d6", ec="#2a78d6", alpha=0.25, lw=1.2))
    for m_end, ls, tag in ((mb_end, ":", "worst case"), (mb[-1, 2], "--", "LQR contraction")):
        lvl = max(c_b - m_end, 0.0)
        if lvl > 0:
            ax.add_patch(Ellipse((r_b, hp[2]), 2 * np.sqrt(lvl / m_h), 2 * np.sqrt(lvl / m_v), fc="none", ec="#2a78d6",
                                 ls=ls, lw=1.3))
    ax.plot(*([r_b], [hp[2]]), "+", color="#2a78d6", ms=10)
    ax.text(r_b + wh / 2 + 0.04, hp[2] + 0.1, f"base set B\n(c = {c_b:g})", color="#2a78d6", fontsize=8.5, va="bottom")
    ax.text(r_b + wh / 2 + 0.04, hp[2] - 0.05,
            f"tightened at T:\n  dashed {max(c_b - mb[-1, 2], 0):.1f}\n  dotted {max(c_b - mb_end, 0):.1f}\n  (worst case)",
            color="#2a78d6", fontsize=7.5, va="top")
    ax.text(0.25, 2.85, "inside the funnel (keep-out)", color="#c4553f", fontsize=9)
    ax.text(2.2, 0.2, "below the floor (keep-out)", color="#c4553f", fontsize=9)
    ax.annotate("", xy=(0.0, tip), xytext=(0.6, tip - 0.35), arrowprops={"arrowstyle": "->", "color": "0.3"})
    ax.text(0.62, tip - 0.45, f"tip z = {tip:g} m, rounded by eps = {eps:g}", fontsize=8, color="0.3")
    ax.set_xlim(0, 3.6)
    ax.set_ylim(0, z_max)
    ax.set_aspect("equal")
    ax.set_xlabel("horizontal distance from the funnel axis r [m]")
    ax.set_ylabel("height z [m]")
    ax.set_title(r"Safe set $S=\{h_{cone}\geq 0\}\cap\{h_{floor}\geq 0\}$, base set $B$, target, reference", fontsize=10.5,
                 loc="left")
    ax.legend(fontsize=7.4, loc="upper right", framealpha=0.95)
    ax.grid(color="#e4e3df", lw=0.5)

    ax = axes[1]
    ax.plot(taus, m1[:, 2], color="#2a78d6", lw=1.8, label=r"$m_{base}$ (growth 1, worst case)")
    ax.plot(taus, mb[:, 2], color="#2a78d6", lw=1.2, ls="--", label=rf"$m_{{base}}$ (LQR contraction {g_b:.3f})")
    ax.axhline(c_b, color="#2a78d6", lw=0.8, ls=":")
    ax.text(1.95, c_b - 0.25, f"c_B = {c_b:g} (B empty if m_base > c_B)", color="#2a78d6", fontsize=8, ha="right", va="top")
    ax.set_xlabel(r"backup horizon $\tau$ [s]")
    ax.set_ylabel("base-set margin (level units)", color="#2a78d6")
    ax2 = ax.twinx()
    ax2.plot(taus, m1[:, 0], color="#c4553f", lw=1.8, label=r"$m_{cone}$ [m]")
    ax2.plot(taus, m1[:, 1], color="#4d9f61", lw=1.8, label=r"$m_{floor}$ [m]")
    ax2.plot(taus, tube_scale_s(s1, ue), color="0.4", lw=1.0, ls="-.", label=r"$s_{eff}(\tau)$ (P-metric)")
    ax2.set_ylabel("funnel / floor margin [m]")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper left")
    ax.set_title(r"Tube $s_{k+1}=\|F_k\|_P s_k + \Delta t\,\gamma\,q(\tau_k)$ and margins" + "\n"
                 + rf"$\delta_d$={ue.delta_d:g}, $\delta_v$={ue.delta_v:.3f}, $\bar e$={ue.e_bar:g}, $\gamma$={tc.gamma:.2f}, "
                 + rf"$L_{{cone}}$={tc.l_cone:.3f}, $L_{{floor}}$={tc.l_floor:.3f}", fontsize=10, loc="left")
    ax.grid(color="#e4e3df", lw=0.5)
    fig.tight_layout()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    print(f"growth in B {g_b:.4f}; end-of-horizon margins (growth 1): cone {m1[-1, 0]:.3f} m, floor {m1[-1, 1]:.3f} m, "
          f"base {m1[-1, 2]:.2f}; (contraction): cone {mb[-1, 0]:.3f}, floor {mb[-1, 1]:.3f}, base {mb[-1, 2]:.2f}")
    print(f"wrote {out}")


def tube_scale_s(s, ue):
    return np.asarray(s, float) * float(ue.tube_scale)


if __name__ == "__main__":
    main()
