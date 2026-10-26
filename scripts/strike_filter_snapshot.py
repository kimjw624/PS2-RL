#!/usr/bin/env python
"""One step of the discrete safety filter, drawn from an evaluation replay (paper figure: what the filter checks).

For episode i of a replay (scripts/eval_strike_filter.py with --save_traj 1) and chosen times, the filter is rebuilt as
in the evaluation and its check of new candidates is rerun from the recorded state:
  left   top view around the vehicle: the seven backup variants' 3-s plans (lambda = 0) colored by verdict (certified,
         path condition violated, terminal condition violated), how each ends (hover ellipse or braking point), the
         chosen candidate's plan and its chaser prediction, the chaser's h_c = 0 distance;
  right  the certification matrix (backup variant x blend lambda) with the chosen cell.
The filter's committed memory is not recorded, so the check starts from a fresh memory (no held hover anchor): it is
exact while the vehicle is not hovering on a latched anchor (the script compares its choice with the recorded one).

    python scripts/strike_filter_snapshot.py --replay outputs/strike_final/paper/replays/s0/ps2 --mode learned \
        --episode 845 --times 5.6 7.0 --backup_run outputs/strike_final/p1/td3_a0p2_seed0 \
        --task docker_batch/configs/strike_final_ps2.json --guidance '{"committed_prediction": true, ...}' --lang ko \
        --out outputs/strike_final/paper/snapshots
Without --times: the first step where the filter cut the nominal command, the first step on the committed plan and the
first step without a certificate (those that occur). Output: snapshot_ep<i>_step<k>.png / .pdf (k = time / dt) and
snapshots_ep<i>.json (the certification matrices, the chosen and the recorded decision).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

TXT = {
    "ko": {"certified": "인증", "path_fail": "궤적 조건 위반", "term_fail": "종단 조건 위반", "chosen": "선택한 후보",
           "lam": "혼합률 $\\lambda$", "variant": "회복 정책", "none_cert": "새 인증 없음:\n이전 계획 유지", "none_lost": "새 인증 없음,\n이전 계획도 실패:\n인증 상실",
           "past": "지나온 경로",
           "chaser_pred": "추격자 예측", "hover_end": "호버 종단(타원체)", "brake_end": "제동 정지점", "hc0": "$h_c=0$ 거리",
           "episode": "에피소드", "pick": "선택", "matrix": "후보 35개의 인증 결과", "recorded": "평가 기록",
           "nominal": "명목 입력 방향"},
    "en": {"certified": "certified", "path_fail": "path condition fails", "term_fail": "terminal condition fails",
           "chosen": "chosen candidate", "lam": "blend $\\lambda$", "variant": "backup", "none_cert": "no new certificate:\ncommitted plan",
           "none_lost": "no new certificate,\ncommitted plan fails:\nno certificate",
           "past": "path so far", "chaser_pred": "chaser prediction", "hover_end": "hover end (ellipse)",
           "brake_end": "braking point", "hc0": "$h_c=0$ distance", "episode": "episode", "pick": "chosen",
           "matrix": "the 35 candidates", "recorded": "recorded", "nominal": "nominal direction"},
}
C_OK, C_PATH, C_TERM, C_PICK = "#1b8a6b", "#c4302b", "#e07b39", "#2a78d6"
# --terms paper: the paper's words (요격기, 배치, 인증 상실, 학습 기반 (2, +60°)) and the colours of its trajectory
# figures (scripts/strike_paper_traj.py: interceptor red, chosen plan navy)
PAPER_TXT = {"chaser_pred": "요격기 예측", "episode": "배치", "none_cert": "새 인증 없음:\n이전 계획 유지",
             "variant": "회복 정책 번호 $j$"}
PAPER_C = {"ok": "#1b8a6b", "path": "#8e44ad", "term": "#e07b39", "pick": "#185888", "chaser": "#c0392b"}


def build_filter(a):
    """The filter of the evaluation (scripts/eval_strike_filter.py) for the backup run, task, env overrides and guidance."""
    from ps2rl.cil.strike_filter import backup_ranges, load_backup, make_filter, model_from_backup, with_chaser
    from ps2rl.cil.strike_guidance import GuidanceConfig
    from ps2rl.envs.strike_env import StrikeEnvConfig, chaser_assumed
    from ps2rl.envs.strike_scene import load_strike_task
    m = model_from_backup(a.backup_run)
    task = load_strike_task(a.task)
    ec = StrikeEnvConfig.from_dict({**task.get("env", {}), **(json.loads(a.env) if a.env else {})})
    g = GuidanceConfig.from_dict(json.loads(a.guidance) if a.guidance else {})
    arena = (0.0, ec.arena_x, ec.arena_y_half, ec.z_max)
    m_f = with_chaser(m, *chaser_assumed(m, ec))
    o_rng, b_rng = backup_ranges(a.backup_run)
    bk, prm = load_backup(m, "lqr" if a.mode == "lqr" else "learned", a.backup_run if a.mode != "lqr" else None)
    filt = make_filter(m_f, bk, prm, obstacle_range=o_rng, ball_range=b_rng, chaser_model="guidance", guidance=g, arena=arena,
                       hover_latch=bool(a.hover_latch))
    return m_f, g, filt


def auto_times(q, j, k_end):
    """First nominal cut (fresh certificate with lambda < 1), first committed-plan step, first step without a certificate."""
    out = []
    lam, com, noc = q.lam[:k_end, j], q.committed[:k_end, j] > 0.5, q.nocert[:k_end, j] > 0.5
    for mask in ((lam < 0.999) & ~com & ~noc, com, noc):
        idx = np.nonzero(mask)[0]
        if len(idx):
            out.append(int(idx[0]))
    return sorted(set(out))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--replay", required=True, help="evaluation output folder with strike_filter_traj.npz")
    p.add_argument("--mode", default="learned", help="filter mode of the replay (learned | lqr)")
    p.add_argument("--episode", type=int, required=True)
    p.add_argument("--times", type=float, nargs="*", default=None, help="times [s] (default: picked automatically)")
    p.add_argument("--backup_run", required=True)
    p.add_argument("--task", default="docker_batch/configs/strike_final_ps2.json")
    p.add_argument("--env", default="", help="env overrides as JSON (the chaser the replay was run with)")
    p.add_argument("--guidance", default="", help="the filter's guidance options as JSON (the evaluation's --guidance)")
    p.add_argument("--hover_latch", type=int, default=1)
    p.add_argument("--window", type=float, default=0.0, help="half width [m] of the top view (0: fit the plans)")
    p.add_argument("--lang", choices=("ko", "en"), default="ko")
    p.add_argument("--terms", choices=("default", "paper"), default="default",
                   help="paper: the paper's terms and trajectory-figure colours (Korean)")
    p.add_argument("--out", default="outputs/strike_filter_snapshots")
    a = p.parse_args(argv)

    import strike_episode_viz as viz
    lang = viz.setup_style(a.lang)
    L, LV = TXT[lang], viz.TXT[lang]
    c_ok, c_path, c_term, c_pick, c_chaser = C_OK, C_PATH, C_TERM, C_PICK, viz.CHASER
    paper = a.terms == "paper" and lang == "ko"
    if paper:
        L, LV = {**L, **PAPER_TXT}, {**LV, "nocert": "인증 상실"}
        c_ok, c_path, c_term, c_pick, c_chaser = (PAPER_C[k] for k in ("ok", "path", "term", "pick", "chaser"))
    import jax
    import jax.numpy as jnp
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Circle, Patch, Rectangle

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    q = viz.Panel(f"{a.replay}:{a.mode}:-:snap", c_pick)
    if a.episode not in q.idx:
        raise SystemExit(f"episode {a.episode} is not in {a.replay} (episodes {sorted(q.idx)})")
    if q.quat is None or q.u_nom is None:
        raise SystemExit(f"{a.replay}: the replay lacks attitude / nominal commands (rerun eval_strike_filter.py --save_traj 1)")
    j, k_end = q.ep(a.episode)
    steps = ([int(round(t / q.dt)) for t in a.times] if a.times else auto_times(q, j, k_end))
    steps = [s for s in steps if 0 <= s < k_end]
    if not steps:
        raise SystemExit("no step to draw (times outside the episode, or the filter never acted)")

    m, g, filt = build_filter(a)
    lams = np.asarray(filt.lambdas, dtype=np.float32)
    labels = [viz.short_variant(lb, lang) for lb in filt.variant_labels]
    if paper:
        from strike_paper_traj import TERMS, paper_variant
        labels = [paper_variant(lb, TERMS["ko"]) for lb in filt.variant_labels]
    n_var = len(labels)
    obst = jnp.asarray(q.obst[j], jnp.float32)
    on = jnp.asarray(q.on[j], jnp.float32)
    fresh = jax.jit(filt.fresh_view)
    cand = [jax.jit(jax.vmap(lambda lam, x, u, o, c, cn, cm, k=k: filt.candidate(x, u, o, c, cn, cm, k, lam),
                             in_axes=(0, None, None, None, None, None, None))) for k in range(n_var)]
    rho_xy = float(m.lqr.rho_xy)
    records = []
    for s in steps:
        x = jnp.asarray(np.concatenate([q.pos[s, j], q.vel[s, j], q.quat[s, j]]), jnp.float32)
        chp = jnp.asarray(q.ch[s, j], jnp.float32)
        u_nom = jnp.asarray(q.u_nom[s, j], jnp.float32)
        comm = fresh(x, obst, chp, on)
        res = [jax.device_get(cand[k](jnp.asarray(lams), x, u_nom, obst, chp, on, comm)) for k in range(n_var)]
        path_ok = np.array([r["path_ok"] for r in res])            # (variants, lambdas)
        term_ok = np.array([r["terminal_ok"] for r in res])
        cert = path_ok & term_ok
        pick = None
        for li in range(len(lams)):                                 # largest blend, then the lowest variant index
            ks = np.nonzero(cert[:, li])[0]
            if len(ks):
                pick = (int(ks[0]), li)
                break
        rec_lam, rec_k = float(q.lam[s, j]), int(round(float(q.variant[s, j]))) if q.variant is not None else -1
        rec_status = "nocert" if q.nocert[s, j] > 0.5 else ("committed" if q.committed[s, j] > 0.5 else "fresh")
        records.append({"episode": a.episode, "t": s * q.dt, "step": s,
                        "certified": cert.astype(int).tolist(), "path_ok": path_ok.astype(int).tolist(),
                        "terminal_ok": term_ok.astype(int).tolist(), "lambdas": lams.tolist(), "variants": list(filt.variant_labels),
                        "chosen": None if pick is None else {"variant": pick[0], "lambda": float(lams[pick[1]])},
                        "recorded": {"status": rec_status, "lambda": rec_lam, "variant": rec_k}})
        print(f"[snapshot] episode {a.episode} t {s * q.dt:.2f} s: chosen "
              + ("none" if pick is None else f"variant {pick[0]} lambda {lams[pick[1]]:g}")
              + f"; recorded {rec_status}, lambda {rec_lam:g}, variant {rec_k}", flush=True)

        # ---- figure
        fig = plt.figure(figsize=(7.2, 3.6))
        ax = fig.add_axes((0.055, 0.22, 0.6, 0.69))
        axm = fig.add_axes((0.775, 0.29, 0.2, 0.56))
        viz.top_view(ax, q, j)
        if paper:                                                   # walls and the outside as in the trajectory figures
            from matplotlib.patches import Polygon
            from strike_paper_traj import OBS_EDGE, OBS_FILL, shade_outside
            for patch in list(ax.patches):
                patch.remove()
            shade_outside(ax, q.arena)
            for o in q.obst[j]:
                if o[6] > 0.5:
                    ax.add_patch(Polygon(viz.outline(o), closed=True, fc=OBS_FILL, ec=OBS_EDGE, lw=0.8, zorder=2))
        p0, v0 = np.asarray(q.pos[s, j]), np.asarray(q.vel[s, j])
        ax.plot(q.pos[:s + 1, j, 0], q.pos[:s + 1, j, 1], color="#8a8a8a", lw=1.0, zorder=4)
        pts = [p0[:2]]
        for k, r in enumerate(res):
            li0 = len(lams) - 1                                     # lambda = 0: the variant's own plan
            xs = np.concatenate([np.asarray(r["x1"][li0])[None], np.asarray(r["states"][li0])])
            col = c_ok if cert[k, li0] else (c_path if not path_ok[k, li0] else c_term)
            ax.plot(xs[:, 0], xs[:, 1], color=col, lw=1.0, alpha=0.9, zorder=5)
            end = xs[-1, :2]
            if bool(r["hover_end"][li0]):
                ax.add_patch(Circle(np.asarray(r["anchor_end"][li0])[:2], rho_xy, fc="none", ec=col, lw=0.6, ls=":", zorder=5))
            else:
                bp = np.asarray(r["brake_point"][li0])
                ax.plot([end[0], bp[0]], [end[1], bp[1]], color=col, lw=0.5, ls=":", zorder=5)
                ax.plot(*bp, marker="x", color=col, ms=4, mew=1.0, zorder=6)
            ax.text(end[0], end[1], f" {k}", fontsize=6, color=col, va="center", zorder=7)
            pts += [xs[:, :2].min(0), xs[:, :2].max(0)]
        if pick is not None:                                        # the chosen candidate's plan and chaser prediction
            r = res[pick[0]]
            xs = np.concatenate([np.asarray(r["x1"][pick[1]])[None], np.asarray(r["states"][pick[1]])])
            ax.plot(xs[:, 0], xs[:, 1], color=c_pick, lw=2.2, zorder=6)
            for c in range(q.ch.shape[2]):
                if q.on[j, c] > 0.5:
                    cp = np.asarray(r["chasers"][pick[1]])[:, c]
                    ax.plot(cp[:, 0], cp[:, 1], color=c_chaser, lw=1.0, ls="--", zorder=5)
                    pts += [cp[:, :2].min(0), cp[:, :2].max(0)]
        for c in range(q.ch.shape[2]):
            if q.on[j, c] > 0.5:
                cc = np.asarray(q.ch[s, j, c])
                rel = p0 - cc
                e = rel / max(np.linalg.norm(rel), 1e-6)
                sv = max(q.v_c - float(e @ v0), 0.0)
                ax.add_patch(Circle(cc[:2], q.r_c, fc=c_chaser, ec=viz.INK, lw=0.5, zorder=7))
                ax.add_patch(Circle(cc[:2], q.r_c + q.r_d + sv * q.delay + sv ** 2 / (2 * q.a_e), fc=c_chaser, alpha=0.08,
                                    ec=c_chaser, ls="--", lw=0.7, zorder=3))
                pts.append(cc[:2])
        un = np.asarray(q.u_nom[s, j])
        ax.plot(*p0[:2], "o", color=viz.INK, ms=4.5, zorder=8)
        if np.linalg.norm(v0[:2]) > 0.1:
            ax.annotate("", xy=p0[:2] + 0.5 * v0[:2], xytext=p0[:2], zorder=8,
                        arrowprops=dict(arrowstyle="-|>", color=viz.INK, lw=0.9))
        pts = np.array(pts)
        lo, hi = pts.min(0) - 2.0, pts.max(0) + 2.0
        if a.window > 0:
            lo, hi = p0[:2] - a.window, p0[:2] + a.window
        box = ax.get_position()
        aspect = (box.height * fig.get_figheight()) / (box.width * fig.get_figwidth())
        cx, cy = (lo + hi) / 2
        wx, wy = hi - lo
        if wy / wx > aspect:
            wx = wy / aspect
        else:
            wy = wx * aspect
        ax.set_xlim(cx - wx / 2, cx + wx / 2)
        ax.set_ylim(cy - wy / 2, cy + wy / 2)
        ax.set_xticks(np.arange(np.ceil((cx - wx / 2) / 5) * 5, cx + wx / 2, 5))
        ax.set_yticks(np.arange(np.ceil((cy - wy / 2) / 5) * 5, cy + wy / 2, 5))
        ax.set_xlabel(LV["x"])
        ax.set_ylabel(LV["y"])
        ttl = f"{L['episode']} {a.episode}, t = {s * q.dt:.1f} s"
        if pick is not None:
            ttl += f"  ({L['pick']}: $\\lambda^\\star$ = {lams[pick[1]]:g}, {labels[pick[0]]})"
        ax.set_title(ttl, loc="left", fontsize=8.5)
        # certification matrix
        img = np.where(cert, 2, np.where(~path_ok, 0, 1))
        from matplotlib.colors import ListedColormap
        axm.imshow(img, cmap=ListedColormap(["#f3b6ae", "#f6cf9f", "#a8dcc7"]), vmin=0, vmax=2, aspect="auto")
        axm.set_xticks(range(len(lams)))
        axm.set_xticklabels([f"{v:g}" for v in lams], fontsize=6.5)
        axm.set_yticks(range(n_var))
        axm.set_yticklabels([f"{k}: {lb}" for k, lb in enumerate(labels)], fontsize=6)
        axm.set_xlabel(L["lam"], fontsize=7.5)
        axm.set_title(L["matrix"], fontsize=7.5)
        axm.grid(False)
        for spine in axm.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.5)
        for k in range(n_var):
            for li in range(len(lams)):
                axm.text(li, k, "✓" if cert[k, li] else ("×" if not path_ok[k, li] else "T"), ha="center", va="center",
                         fontsize=6, color=viz.INK2, family="DejaVu Sans")
        if pick is not None:
            axm.add_patch(Rectangle((pick[1] - 0.5, pick[0] - 0.5), 1, 1, fc="none", ec=c_pick, lw=2.0))
        else:
            axm.text(0.5, 0.5, L["none_lost" if rec_status == "nocert" else "none_cert"], transform=axm.transAxes, ha="center", va="center", fontsize=7, color=c_path,
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=c_path, lw=0.6))
        axm.text(0.5, 1.13, f"{L['recorded']}: " + {"fresh": f"$\\lambda$ = {rec_lam:g}", "committed": LV["committed"],
                                                    "nocert": LV["nocert"]}[rec_status],
                 transform=axm.transAxes, ha="center", fontsize=6.5, color=viz.INK2)
        hs = [Line2D([], [], color=c_ok, lw=1.2, label=L["certified"]), Line2D([], [], color=c_path, lw=1.2, label=L["path_fail"]),
              Line2D([], [], color=c_term, lw=1.2, label=L["term_fail"]), Line2D([], [], color=c_pick, lw=2.2, label=L["chosen"]),
              Line2D([], [], color=c_chaser, lw=1.0, ls="--", label=L["chaser_pred"]),
              Line2D([], [], color="#8a8a8a", lw=1.0, label=L["past"]),
              Line2D([], [], marker="x", ls="", color=viz.INK2, ms=4, label=L["brake_end"]),
              Line2D([], [], marker="o", ls="", mfc="none", mec=viz.INK2, ms=6, label=L["hover_end"])]
        fig.legend(handles=hs, loc="lower center", ncol=4, frameon=False, fontsize=6.5, bbox_to_anchor=(0.36, 0.0))
        viz.save(fig, out / f"snapshot_ep{a.episode}_step{s:03d}")      # (no dot in the name: save() sets the suffix)
    (out / f"snapshots_ep{a.episode}.json").write_text(json.dumps(records, indent=1))
    print(f"wrote {len(steps)} snapshot(s) to {out}", flush=True)


if __name__ == "__main__":
    main()
