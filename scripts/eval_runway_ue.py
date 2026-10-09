"""Evaluate runway bird-chasing policies with and without the UE-bCBF safety layer, and draw them.

Every case runs on the same episodes (initial state, wind and bird draws come from --seed), so
the columns differ only in the policy and the filter:

    JAX_ENABLE_X64=1 python scripts/eval_runway_ue.py \
        --case "chaser alone=outputs/runway_phase2_ue/rwy_chase_vanilla_s0:none" \
        --case "chaser + CIL=outputs/runway_phase2_ue/rwy_chase_vanilla_s0:ue" \
        --case "Phase II + CIL=outputs/runway_phase2_ue/rwy_ps2_warm_s0:ue" \
        --episodes 128 --out outputs/runway_phase2_ue/eval_compare

MODE: ``none`` (policy executed as is; episodes are not cut at a violation, so the figure shows
where the chase would have gone) or ``ue`` (through the runway UE-bCBF CIL of the run's Phase-I
checkpoint). Writes eval.json, trajectories.npz, runway_overview.png (top and side views of
selected episodes, safety margins over time) and, with --gif, an animation of one episode.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
import pickle
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from ps2rl.cil import quadrotor_runway_ue_bcbf as rwcbf  # noqa: E402
from ps2rl.envs.quadrotor_runway_bird_env import RunwayBirdEnvConfig, build_runway_bird_env  # noqa: E402
from ps2rl.utils.policy import ActorConfig, actor_mean_action  # noqa: E402

INK, INK_2, GRID, SURF = "#0b0b0b", "#52514e", "#d9d8d3", "#fcfcfb"
STRIP = "#f3dcd2"
COLORS = ["#eb6834", "#2a78d6", "#1baf7a"]  # validated categorical slots (orange, blue, aqua)
BIRD = "#52514e"


def _load_run(run: Path):
    cfgs = json.loads((run / "configs.json").read_text())
    act = dict(cfgs["actor"])
    act["hidden_sizes"] = tuple(act["hidden_sizes"])
    env_d = {k: v for k, v in cfgs["env"].items() if k in {f.name for f in dataclasses.fields(RunwayBirdEnvConfig)}}
    return cfgs, ActorConfig(**act), RunwayBirdEnvConfig(**env_d)


def simulate(run: Path, mode: str, *, ckpt: str, episodes: int, seed: int, weights: str = "best"):
    cfgs, actor_cfg, env_cfg = _load_run(run)
    params = pickle.load(open(run / f"{weights}_weights.pkl", "rb"))["actor_params"]
    ps2 = cfgs["ps2"]
    cbf = rwcbf.runway_ue_bcbf_config_from_checkpoint(ckpt or cfgs["checkpoint"], alpha=float(ps2["alpha_cbf"]),
                                                      alpha_ceiling=float(ps2["alpha_ceiling"]),
                                                      qp_solve_dtype="float64" if jax.config.jax_enable_x64 else "float32")
    rt = rwcbf.get_cached_runtime(cbf)
    env_cfg = dataclasses.replace(env_cfg, terminate_on_unsafe=(mode != "none"), bank_seed=int(seed))
    env = build_runway_bird_env(env_cfg, cbf, rt, rwcbf.make_recoverability_fn(cbf, rt))
    low, high = jnp.asarray(cbf.action_low, jnp.float32), jnp.asarray(cbf.action_high, jnp.float32)
    scale = jnp.asarray(cbf.action_scale, jnp.float32)
    e_bar = jnp.asarray(float(cbf.ue.e_bar), jnp.float32)
    qp_dtype = jnp.float64 if jax.config.jax_enable_x64 else None

    def run_batch(key):
        es, obs = env.reset_batched(jax.random.split(key, episodes))

        def body(c, k):
            es, obs, alive = c
            raw = jnp.clip(actor_mean_action(params, obs, scale, actor_cfg, action_low=low, action_high=high), low, high)
            if mode == "ue":
                u, aux = jax.vmap(lambda x, d, uu: rwcbf.project_full(x, d, e_bar, uu, cbf, rt, qp_dtype))(obs[:, :10], obs[:, 10:13], raw)
                u = u.astype(jnp.float32)
                slack, lam = aux["slack"], aux["safeguard_lambda"]
            else:
                u, slack, lam = raw, jnp.zeros(episodes), jnp.ones(episodes)
            es2, obs_true, obs_out, rew, done, info = env.step_batched(es, u, jax.random.split(jax.random.fold_in(key, k), episodes))
            rec = {"p": info.p_true, "q": obs_true[:, 6:10], "bird": info.p_bird, "u": u, "raw": raw, "rew": rew,
                   "h_rwy": info.h_rwy,
                   "h_ceil": info.h_ceil, "dist": info.dist, "slack": slack, "lam": lam, "alive": alive}
            return (es2, obs_out, alive & ~done), rec

        _, tr = jax.lax.scan(body, (es, obs, jnp.ones((episodes,), bool)), jnp.arange(env.max_steps))
        return tr

    tr = jax.device_get(jax.jit(run_batch)(jax.random.PRNGKey(seed)))
    tr = {k: np.asarray(v) for k, v in tr.items()}
    al = tr["alive"].astype(bool)
    h_r = np.where(al, tr["h_rwy"], np.inf)
    h_c = np.where(al, tr["h_ceil"], np.inf)
    unsafe = ((h_r < 0) | (h_c < 0)).any(0)
    corr = np.linalg.norm((tr["u"] - tr["raw"]) / np.asarray(scale), axis=-1)
    res = {"episodes": episodes, "unsafe_rate": float(unsafe.mean()), "runway_incursion_rate": float((h_r < 0).any(0).mean()),
           "ceiling_excess_rate": float((h_c < 0).any(0).mean()),
           "max_runway_incursion_m": float(max(0.0, -h_r.min())), "max_ceiling_excess_m": float(max(0.0, -h_c.min())),
           "mean_dist_to_bird": float((tr["dist"] * al).sum() / al.sum()), "return_mean": float((tr["rew"] * al).sum(0).mean()),
           "cil_correction_mean": float((corr * al).sum() / al.sum()),
           "slack_gt_1e-3_rate": float(((tr["slack"] > 1e-3) * al).sum() / al.sum()),
           "safeguard_rate": float(((tr["lam"] < 1.0) * al).sum() / al.sum())}
    return res, tr, cbf


def _pick(trs, labels):
    """Episodes to draw: where the first case (usually the unfiltered chaser) goes deepest into the runway and
    highest above the ceiling."""
    t0 = trs[0]
    al = t0["alive"].astype(bool)
    inc = np.where(al, -t0["h_rwy"], -np.inf).max(0)
    exc = np.where(al, -t0["h_ceil"], -np.inf).max(0)
    return int(np.argmax(inc)), int(np.argmax(exc))


def plot(trs, labels, cbf, out: Path, results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    rc = cbf.runway
    y_e, z_m = float(rc.y_edge), float(rc.z_max)
    i_rwy, i_ceil = _pick(trs, labels)
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK_2, "axes.labelcolor": INK, "xtick.color": INK_2,
                         "ytick.color": INK_2, "axes.titlesize": 11, "axes.titleweight": "bold"})
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 10.5))
    fig.patch.set_facecolor(SURF)

    def draw_xy(ax, ep, title):
        ax.set_facecolor(SURF)
        ys = np.concatenate([np.ravel(t["p"][:, ep, 1]) for t in trs] + [trs[0]["bird"][:, ep, 1]])
        xs = np.concatenate([np.ravel(t["p"][:, ep, 0]) for t in trs] + [trs[0]["bird"][:, ep, 0]])
        y_hi = max(y_e + 3.0, float(np.nanmax(ys)) + 1.0)
        ax.axhspan(y_e, y_hi + 50, color=STRIP, lw=0, zorder=0)
        ax.axhline(y_e, color=INK, lw=1.4, zorder=2)
        ax.text(0.02, 0.97, "runway + protection strip (no entry)", transform=ax.transAxes, va="top", fontsize=9, color=INK)
        b = trs[0]["bird"][:, ep]
        ax.plot(b[:, 0], b[:, 1], color=BIRD, ls=(0, (4, 3)), lw=1.4, zorder=3)
        ax.annotate("", xy=b[-1, :2], xytext=b[-6, :2], arrowprops=dict(arrowstyle="-|>", color=BIRD, lw=1.2), zorder=3)
        for t, lab, col in zip(trs, labels, COLORS):
            al = t["alive"][:, ep].astype(bool)
            p = t["p"][al, ep]
            ax.plot(p[:, 0], p[:, 1], color=col, lw=2.0, zorder=4)
            ax.plot(p[-1, 0], p[-1, 1], "o", color=col, ms=6, mec=SURF, zorder=5)
        p0 = trs[0]["p"][0, ep]
        ax.plot(p0[0], p0[1], "s", color=INK, ms=6, zorder=6)
        ax.set_xlim(float(np.nanmin(xs)) - 1.5, float(np.nanmax(xs)) + 1.5)
        ax.set_ylim(float(np.nanmin(ys)) - 1.5, y_hi)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel("x along the runway [m]")
        ax.set_ylabel("y towards the runway [m]")
        ax.grid(color=GRID, lw=0.5)
        ax.set_title(title, loc="left", color=INK)

    def draw_yz(ax, ep, title):
        ax.set_facecolor(SURF)
        zs = np.concatenate([np.ravel(t["p"][:, ep, 2]) for t in trs] + [trs[0]["bird"][:, ep, 2]])
        ys = np.concatenate([np.ravel(t["p"][:, ep, 1]) for t in trs] + [trs[0]["bird"][:, ep, 1]])
        z_hi = max(z_m + 2.0, float(np.nanmax(zs)) + 1.0)
        ax.axhspan(z_m, z_hi + 50, color=STRIP, lw=0, zorder=0)
        ax.axvspan(y_e, 100, color=STRIP, lw=0, zorder=0)
        ax.axhline(z_m, color=INK, lw=1.4, zorder=2)
        ax.axvline(y_e, color=INK, lw=1.4, zorder=2)
        ax.text(0.02, 0.97, "above the airspace ceiling (no entry)", transform=ax.transAxes, va="top", fontsize=9, color=INK)
        b = trs[0]["bird"][:, ep]
        ax.plot(b[:, 1], b[:, 2], color=BIRD, ls=(0, (4, 3)), lw=1.4, zorder=3)
        ax.annotate("", xy=(b[-1, 1], b[-1, 2]), xytext=(b[-6, 1], b[-6, 2]),
                    arrowprops=dict(arrowstyle="-|>", color=BIRD, lw=1.2), zorder=3)
        for t, col in zip(trs, COLORS):
            al = t["alive"][:, ep].astype(bool)
            p = t["p"][al, ep]
            ax.plot(p[:, 1], p[:, 2], color=col, lw=2.0, zorder=4)
            ax.plot(p[-1, 1], p[-1, 2], "o", color=col, ms=6, mec=SURF, zorder=5)
        p0 = trs[0]["p"][0, ep]
        ax.plot(p0[1], p0[2], "s", color=INK, ms=6, zorder=6)
        ax.set_xlim(float(np.nanmin(ys)) - 1.5, max(y_e + 2.0, float(np.nanmax(ys)) + 1.0))
        ax.set_ylim(max(0.0, float(np.nanmin(zs)) - 1.0), z_hi)
        ax.set_xlabel("y towards the runway [m]")
        ax.set_ylabel("height z [m]")
        ax.grid(color=GRID, lw=0.5)
        ax.set_title(title, loc="left", color=INK)

    draw_xy(axes[0, 0], i_rwy, "Bird flies over the runway: top view")
    draw_yz(axes[0, 1], i_ceil, "Bird climbs through the ceiling: side view")
    dt = float(rc.dt)
    for ax, key, name, lim in ((axes[1, 0], "h_rwy", "distance to the runway strip  y_edge - y  [m]", None),
                               (axes[1, 1], "h_ceil", "distance below the ceiling  z_max - z  [m]", None)):
        ax.set_facecolor(SURF)
        for t, lab, col in zip(trs, labels, COLORS):
            al = t["alive"].astype(bool)
            h = np.where(al, t[key], np.nan)
            tt = np.arange(h.shape[0]) * dt
            ax.plot(tt, np.nanmin(h, axis=1), color=col, lw=2.0)
            ax.fill_between(tt, np.nanpercentile(h, 10, axis=1), np.nanpercentile(h, 90, axis=1), color=col, alpha=0.12, lw=0)
        ax.axhline(0.0, color=INK, lw=1.2)
        ax.axhspan(-100, 0, color=STRIP, lw=0, zorder=0)
        allv = np.concatenate([np.nanmin(np.where(t["alive"].astype(bool), t[key], np.nan), axis=1) for t in trs])
        ax.set_ylim(min(-0.5, float(np.nanmin(allv)) - 0.3), None)
        ax.set_xlabel("t [s]")
        ax.set_ylabel(name)
        ax.grid(color=GRID, lw=0.5)
        ax.set_title(("runway" if key == "h_rwy" else "ceiling") + " margin: worst episode (line), 10-90 % (band)",
                     loc="left", color=INK, fontsize=10)
    handles = [Line2D([], [], color=BIRD, ls=(0, (4, 3)), lw=1.4, label="bird")]
    for lab, col in zip(labels, COLORS):
        r = results[lab]
        handles.append(Line2D([], [], color=col, lw=2.0, label=f"{lab}: unsafe {r['unsafe_rate']:.0%} "
                              f"(runway {r['max_runway_incursion_m']:.1f} m, ceiling {r['max_ceiling_excess_m']:.1f} m), "
                              f"mean dist {r['mean_dist_to_bird']:.1f} m"))
    handles.append(Line2D([], [], color=INK, marker="s", ls="", ms=6, label="start"))
    fig.legend(handles=handles, loc="lower center", ncol=1, frameon=False, fontsize=9.5)
    n = trs[0]["alive"].shape[1]
    fig.suptitle(f"Bird deterrence next to a runway under wind (|d| <= {cbf.ue.delta_d} m/s^2): same {n} episodes for "
                 f"every policy", fontsize=12, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.04 + 0.025 * len(labels), 1, 0.97))
    fig.savefig(out / "runway_overview.png", dpi=140, facecolor=fig.get_facecolor())


def animate(trs, labels, cbf, out: Path, ep: int, stride: int = 2):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    rc = cbf.runway
    y_e = float(rc.y_edge)
    fig, ax = plt.subplots(figsize=(7.5, 6.0))
    fig.patch.set_facecolor(SURF)
    ax.set_facecolor(SURF)
    ys = np.concatenate([np.ravel(t["p"][:, ep, 1]) for t in trs] + [trs[0]["bird"][:, ep, 1]])
    xs = np.concatenate([np.ravel(t["p"][:, ep, 0]) for t in trs] + [trs[0]["bird"][:, ep, 0]])
    ax.axhspan(y_e, 200, color=STRIP, lw=0)
    ax.axhline(y_e, color=INK, lw=1.4)
    ax.set_xlim(float(np.nanmin(xs)) - 1.5, float(np.nanmax(xs)) + 1.5)
    ax.set_ylim(float(np.nanmin(ys)) - 1.5, max(y_e + 3.0, float(np.nanmax(ys)) + 1.0))
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(color=GRID, lw=0.5)
    ax.set_xlabel("x along the runway [m]")
    ax.set_ylabel("y towards the runway [m]")
    ax.text(0.02, 0.97, "runway + protection strip (no entry)", transform=ax.transAxes, va="top", fontsize=9)
    bird_line, = ax.plot([], [], color=BIRD, ls=(0, (4, 3)), lw=1.2)
    bird_pt, = ax.plot([], [], "v", color=BIRD, ms=9)
    lines = [ax.plot([], [], color=c, lw=2.0, label=l)[0] for l, c in zip(labels, COLORS)]
    pts = [ax.plot([], [], "o", color=c, ms=8, mec=SURF)[0] for c in COLORS[: len(trs)]]
    ax.legend(loc="lower right", fontsize=8.5, frameon=False)
    title = ax.set_title("", loc="left")
    n_t = trs[0]["p"].shape[0]

    def upd(f):
        k = min(f * stride, n_t - 1)
        b = trs[0]["bird"][: k + 1, ep]
        bird_line.set_data(b[:, 0], b[:, 1])
        bird_pt.set_data([b[-1, 0]], [b[-1, 1]])
        for t, ln, pt in zip(trs, lines, pts):
            last = int(min(k, t["alive"][:, ep].sum() - 1))
            p = t["p"][: last + 1, ep]
            ln.set_data(p[:, 0], p[:, 1])
            pt.set_data([p[-1, 0]], [p[-1, 1]])
        title.set_text(f"t = {k * float(rc.dt):.2f} s")
        return [bird_line, bird_pt, *lines, *pts, title]

    anim = FuncAnimation(fig, upd, frames=n_t // stride + 1, blit=False)
    anim.save(out / "runway_episode.gif", writer=PillowWriter(fps=25), dpi=90)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--case", action="append", required=True, help='"label=RUN_DIR:MODE" (MODE: none / ue)')
    p.add_argument("--ckpt", default="", help="runway UE Phase-I checkpoint (default: the one recorded in each run)")
    p.add_argument("--episodes", type=int, default=128)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--weights", default="best", choices=("best", "final"))
    p.add_argument("--out", required=True)
    p.add_argument("--gif", action="store_true")
    a = p.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    labels, trs, results = [], [], {}
    cbf = None
    for spec in a.case:
        label, rest = spec.split("=", 1)
        run, mode = rest.rsplit(":", 1)
        res, tr, cbf = simulate(Path(run), mode, ckpt=a.ckpt, episodes=a.episodes, seed=a.seed, weights=a.weights)
        print(f"[{label}] unsafe {res['unsafe_rate']:.3f} (runway {res['runway_incursion_rate']:.3f}, max {res['max_runway_incursion_m']:.2f} m; "
              f"ceiling {res['ceiling_excess_rate']:.3f}, max {res['max_ceiling_excess_m']:.2f} m) mean dist {res['mean_dist_to_bird']:.2f} m "
              f"return {res['return_mean']:.1f} corr {res['cil_correction_mean']:.3f} slack>1e-3 {res['slack_gt_1e-3_rate']:.4f} "
              f"safeguard {res['safeguard_rate']:.3f}", flush=True)
        labels.append(label)
        trs.append(tr)
        results[label] = {**res, "run": run, "mode": mode}
    (out / "eval.json").write_text(json.dumps({"seed": a.seed, "results": results}, indent=2))
    np.savez_compressed(out / "trajectories.npz", **{f"{i}_{k}": v for i, t in enumerate(trs) for k, v in t.items()})
    plot(trs, labels, cbf, out, results)
    if a.gif:
        animate(trs, labels, cbf, out, _pick(trs, labels)[0])
    print(f"[eval] wrote {out}/eval.json, trajectories.npz, runway_overview.png" + (", runway_episode.gif" if a.gif else ""))


if __name__ == "__main__":
    main()
