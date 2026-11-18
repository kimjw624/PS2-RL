"""Why the certificate is lost, in one picture: the backup plans at a loss onset, the current family vs early escapes.

    python scripts/plot_lost_certificate.py REPLAY_DIR BACKUP_RUN --episode E --t T --out fig.png
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run", help="an eval_strike_filter.py --save_traj 1 output folder (strike_filter_traj.npz)")
    p.add_argument("--task", default=str(REPO / "docker_batch/configs/strike_final_ps2.json"),
                   help="task file whose tracker.filter_guidance is the filter under test")
    p.add_argument("backup_run", help="the Phase-I backup run the replay used")
    p.add_argument("--episode", type=int, required=True)
    p.add_argument("--t", type=float, required=True)
    p.add_argument("--early", default="[[3.5, 60, 6], [3.5, -60, 6]]")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    import jax
    import jax.numpy as jnp
    import strike_episode_viz as viz
    viz.setup_style("ko")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    from ps2rl.cil.strike_filter import backup_ranges, load_backup, make_filter, model_from_backup
    from ps2rl.cil.strike_guidance import GuidanceConfig, h_chasers
    task = json.load(open(a.task))
    g0 = json.loads(task["tracker"]["filter_guidance"])
    m = model_from_backup(a.backup_run)
    o_rng, b_rng = backup_ranges(a.backup_run)
    bk, prm = load_backup(m, "learned", a.backup_run)
    from ps2rl.envs.strike_env import StrikeEnvConfig
    ec = StrikeEnvConfig.from_dict(task.get("env", {}))
    arena = (0.0, ec.arena_x, ec.arena_y_half, ec.z_max)
    ge = dict(g0)
    ge["escape_variants"] = [list(v) for v in g0["escape_variants"]] + json.loads(a.early)
    f = make_filter(m, bk, prm, obstacle_range=o_rng, ball_range=b_rng, hover_latch=True, chaser_model="guidance",
                    guidance=GuidanceConfig.from_dict(ge), arena=arena)
    gc = GuidanceConfig.from_dict(ge)
    n_base = 2 + len(g0["escape_variants"])
    cand = jax.jit(f.candidate, static_argnums=(6,))
    q = viz.Panel(f"{a.run}:learned:x:x", "#000")
    j, k_end = q.ep(a.episode)
    t = int(round(a.t / q.dt))
    x = jnp.asarray(np.concatenate([q.pos[t, j], q.vel[t, j], q.quat[t, j]]), jnp.float32)
    obst, on, chp = jnp.asarray(q.obst[j]), jnp.asarray(q.on[j]), jnp.asarray(q.ch[t, j])
    comm = f.fresh_view(x, obst, chp, on)
    fig = plt.figure(figsize=(7.6, 5.0))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.25, 1], hspace=0.45, wspace=0.18)
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
    axh = fig.add_subplot(gs[1, :])
    lo = q.pos[t, j][:2] - 9.0
    hi = q.pos[t, j][:2] + 9.0
    titles = ["현재 회피 변형 7개", f"+ 조기 회피 2개 (h_c < 6 m 에서 시작)"]
    for pi, (ax, ks) in enumerate(zip(axes, [range(n_base), range(n_base, len(f.variant_labels))])):
        viz.top_view(ax, q, j)
        ax.plot(q.pos[:t + 1, j, 0], q.pos[:t + 1, j, 1], color="#185888", lw=1.2)
        for c in range(q.ch.shape[2]):
            if q.on[j, c] > 0.5:
                ax.plot(q.ch[:t + 1, j, c, 0], q.ch[:t + 1, j, c, 1], color="#c0392b", lw=0.8, ls="--")
                ax.add_patch(Circle(q.ch[t, j, c, :2], q.r_c, fc="#c0392b", alpha=0.5))
        for k in ks:
            r = cand(x, jnp.zeros(4), obst, chp, on, comm, k, jnp.float32(0.0), jnp.float32(gc.fresh_margin))
            ok = bool(r["path_ok"]) and bool(r["terminal_ok"])
            xs = np.asarray(r["states"])
            chs = np.asarray(r["chasers"])
            col = "#1baf7a" if ok else "#8c8b86"
            ax.plot(xs[:, 0], xs[:, 1], color=col, lw=1.6 if ok else 1.0, alpha=0.95)
            mg = np.asarray(r["margin"])
            bad = np.where(mg < 0)[0]
            if len(bad):
                ax.plot(xs[bad[0], 0], xs[bad[0], 1], "x", color="#e34948", ms=6, mew=1.6)
            hc = np.asarray(jax.vmap(lambda xx, cc: jnp.min(h_chasers(m, gc, xx[0:3], xx[3:6], cc, on)))(jnp.asarray(xs), jnp.asarray(chs)))
            axh.plot(np.arange(len(hc)) * q.dt, hc, color=col, lw=1.4 if ok else 0.9, ls="-" if pi == 1 else ":",
                     alpha=0.9)
            if k == ks[0]:
                for c in range(q.ch.shape[2]):
                    if q.on[j, c] > 0.5:
                        ax.plot(chs[:, c, 0], chs[:, c, 1], color="#c0392b", lw=0.8, alpha=0.6)
        ax.plot(*q.pos[t, j][:2], "o", color="#185888", ms=5)
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(max(lo[1], -10.5), min(hi[1], 10.5))
        ax.set_title(titles[pi], loc="left", fontsize=8.5)
    axh.axhline(gc.fresh_margin, color="#52514e", lw=0.7, ls=":")
    axh.axhline(0.0, color="#e34948", lw=0.7)
    axh.set_ylim(-2, 8)
    axh.set_xlabel("계획 안의 시간 [s]")
    axh.set_ylabel("$h_c$ [m]")
    axh.set_title("각 회피 계획의 $h_c$ (점선: 현재 변형, 실선: 조기 회피; 초록: 인증, 회색: 실패)", loc="left", fontsize=8)
    fig.suptitle(f"인증을 잃는 순간 (에피소드 {a.episode}, t = {a.t:.2f} s): 회색 계획은 요격기에 잡히거나(×) 막힘", x=0.02,
                 ha="left", fontsize=9)
    fig.savefig(a.out, dpi=160, bbox_inches="tight")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
