#!/usr/bin/env python
"""Closed-loop evaluation of the strike safety filter: a nominal controller in arena scenes (tall obstacles, pursuing
chasers, moving target), without the filter and through the filter with the hover-LQR backup and with the learned
Phase-I backup. Same scenes, perturbations and chaser behaviour in every mode.

    python scripts/eval_strike_filter.py --backup_run outputs/strike_phase1/td3_a0p2_seed0 --out outputs/strike_filter_eval
    # the trained striker as the nominal controller
    python scripts/eval_strike_filter.py --backup_run ... --nominal striker --striker_run outputs/strike_tracker/striker_rs0p1_s0

Nominal controllers: "pursuit" (flies straight at the target at --cruise m/s, ignores obstacles and chasers; a simple
velocity loop with thrust-axis and yaw control) or "striker" (a Phase-II policy, deterministic mean action).

Writes to --out: strike_filter_table.md / .json (strike, crash by type, timeout, no-certificate steps, mean kept
fraction of the nominal command), strike_filter_scenes.png (top views of a few scenes in each mode) and
strike_filter_<scene>.gif (one scene animated, the modes side by side).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from ps2rl.cil.strike_filter import backup_ranges, load_backup, make_filter, model_from_backup  # noqa: E402
from ps2rl.cil.strike_bcbf import StrikeBCBFConfig, make_strike_bcbf  # noqa: E402
from ps2rl.cil.strike_guidance import GuidanceConfig  # noqa: E402
from ps2rl.envs.strike_env import StrikeEnvConfig, action_bounds, build_strike_env, generate_scene_pool  # noqa: E402
from ps2rl.envs.strike_pursuit import make_pursuit  # noqa: E402
from ps2rl.envs.strike_scene import StrikeConfig, build_model, check_same_plant, load_strike_task  # noqa: E402
from ps2rl.utils.policy import _coerce_actor_cfg, actor_mean_action  # noqa: E402

OUTCOMES = ("strike", "crash_obstacle", "crash_floor", "crash_chaser", "out_of_arena", "timeout")
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
MODE_C = {"none": "#3d3c39", "lqr": "#c4553f", "learned": "#2a78d6"}
MODE_L = {"none": "no filter", "lqr": "filter, hover-LQR backup", "learned": "filter, learned backup"}
CHASER, TARGET, OBS_FILL, OBS_EDGE = "#1baf7a", "#eb6834", "#c9c7c0", "#6b6a65"


def pursuit_controller(m, cruise: float, kv: float = 2.5, katt: float = 10.0, kyaw: float = 3.0, tilt_max_deg: float = 50.0):
    """Batched straight-line pursuit (ps2rl.envs.strike_pursuit)."""
    return jax.vmap(make_pursuit(m, cruise, kv, katt, kyaw, tilt_max_deg))


def outline(o, n=90):
    cx, cy, bx, by, yaw, rr, _ = o
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    if bx == 0 and by == 0:
        return np.stack([cx + rr * np.cos(th), cy + rr * np.sin(th)], 1)
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[-bx, -by], [bx, -by], [bx, by], [-bx, by]]) @ np.array([[c, s], [-s, c]]) + [cx, cy]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backup_run", required=True, help="Phase-I run directory (learned backup; its strike config is used)")
    p.add_argument("--task", default="docker_batch/configs/strike.json", help="env block: arena, target, chasers")
    p.add_argument("--nominal", choices=("pursuit", "striker"), default="pursuit")
    p.add_argument("--striker_run", default="", help="Phase-II striker run directory (--nominal striker)")
    p.add_argument("--cruise", type=float, default=6.0, help="pursuit speed [m/s]")
    p.add_argument("--modes", default="none lqr learned")
    p.add_argument("--fresh_hold", type=float, default=None,
                   help="chaser hold time [s] for new plans (default: the backup's strike.chaser_hold); continuation uses 0")
    p.add_argument("--chaser_speed_scale", type=float, default=1.0,
                   help="chasers' actual speed / the assumed chaser_speed (0 = frozen); the filter keeps the assumed speed")
    p.add_argument("--chaser_model", choices=("static", "reach", "guidance"), default="static",
                   help="filter's chaser model: static ball R = r_c + r_d + v_c T, the exact reachable ball r_c + r_d + v_c tau, "
                        "or the guidance-law escape set h_c >= 0 with an evasion reflex (ps2rl/cil/strike_guidance.py)")
    p.add_argument("--filter_type", choices=("discrete", "bcbf"), default="discrete",
                   help="discrete: backup-set check over blends toward the backup; bcbf: minimal-correction backup-CBF QP "
                        "(sensitivity rows) with the discrete check as its safeguard (ps2rl/cil/strike_bcbf.py)")
    p.add_argument("--bcbf", default="", help="bCBF QP options as JSON, e.g. '{\"alpha\": 4, \"node_stride\": 3}' (StrikeBCBFConfig)")
    p.add_argument("--guidance", default="", help="guidance options as JSON, e.g. '{\"a_evade\": 5, \"horizon\": 3}' (GuidanceConfig)")
    p.add_argument("--env", default="", help="env overrides as JSON, e.g. '{\"obstacles_min\": 3, \"obstacles_max\": 3, "
                                             "\"chasers_active_min\": 1, \"chasers_active_max\": 1}'")
    p.add_argument("--reach_hold", type=float, default=None, help="reach model: guaranteed hover time after arrival [s] (default T)")
    p.add_argument("--hover_latch", type=int, default=1, help="1: hold the anchor once B is reached (fixed-anchor hover LQR); 0: re-anchor every step")
    p.add_argument("--chaser_leash", type=float, default=0.0,
                   help="> 0: guards - chasers stay within this distance of their start; the filter gets the posts (needs leash <= v_c T)")
    p.add_argument("--episodes", type=int, default=256)
    p.add_argument("--scene_seed", type=int, default=9001, help="seed of the evaluation scenes (not the training pool)")
    p.add_argument("--n_show", type=int, default=6, help="scenes drawn in the figure")
    p.add_argument("--gif", type=int, default=1, help="number of scenes animated")
    p.add_argument("--gif_pick", choices=("first", "close_chaser", "chaser_hit", "contact", "timeout", "failures"),
                   default="first",
                   help="scenes animated (outcomes of the last mode): the first ones, the strikes closest to a chaser, "
                        "episodes hit by a chaser, any contact, time-ups, or failures (half contacts, half time-ups)")
    p.add_argument("--save_episodes", type=int, default=1,
                   help="1: also write strike_filter_episodes.json (per-episode outcome, end / strike step, filter statistics)")
    p.add_argument("--out", default="outputs/strike_filter_eval")
    a = p.parse_args(argv)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import animation
    from matplotlib.lines import Line2D
    from matplotlib.patches import Circle, Polygon

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    m = model_from_backup(a.backup_run)          # the filter's model (its backup's scene format)
    cfg = m.cfg
    m_env = m                                     # the environment's model: the striker's observation format
    if a.nominal == "striker":
        _scfg = StrikeConfig.from_dict(json.loads((Path(a.striker_run) / "configs.json").read_text())["strike"])
        if _scfg != cfg:
            check_same_plant(_scfg, cfg, "striker run vs backup run")
            m_env = build_model(_scfg)
            print(f"[models] environment / striker observation from the striker run (chaser_set {_scfg.chaser_set}, walls "
                  f"{_scfg.arena_walls}); filter from the backup run (chaser_set {cfg.chaser_set}, walls {cfg.arena_walls})",
                  flush=True)
    task = load_strike_task(a.task)
    ec = StrikeEnvConfig.from_dict({**task.get("env", {}), **(json.loads(a.env) if a.env else {}),
                                    "chaser_speed_scale": a.chaser_speed_scale, "chaser_leash": a.chaser_leash})
    if a.nominal == "striker":           # the striker's observation settings (its network expects them)
        _sec = json.loads((Path(a.striker_run) / "configs.json").read_text()).get("env", {})
        ec = ec.replace(**{k: _sec[k] for k in ("obs_hide_scene", "obs_guidance", "obs_pursuit_cruise") if k in _sec})
    ec = ec.replace(residual_cruise=0.0)  # the environment receives full commands here (a residual is added below)
    gcfg = GuidanceConfig.from_dict(json.loads(a.guidance) if a.guidance else {})
    arena = (0.0, ec.arena_x, ec.arena_y_half, ec.z_max)
    if a.chaser_leash > cfg.chaser_speed * cfg.T + 1e-9:
        raise SystemExit(f"--chaser_leash {a.chaser_leash} > v_c T = {cfg.chaser_speed * cfg.T:g}: the filter's ball around the post would not cover the guard")
    ball_c = (lambda es: es.chaser_home) if a.chaser_leash > 0.0 else (lambda es: es.chaser_p)   # centres of the filter's balls
    n_ep = int(a.episodes)
    t0 = time.time()
    pool = generate_scene_pool(ec, max(ec.obstacles_max, cfg.n_obstacle_slots), n=n_ep, seed=a.scene_seed)
    print(f"[scenes] {n_ep} evaluation scenes (seed {a.scene_seed}) in {time.time() - t0:.0f} s", flush=True)
    env = build_strike_env(m_env, ec, pool=pool)

    if a.nominal == "pursuit":
        pur = pursuit_controller(m_env, a.cruise)
        nominal = lambda es, obs: pur(es.x, es.target_p, es.target_v)
        nom_label = f"straight-line pursuit at {a.cruise:g} m/s"
    else:
        sr = Path(a.striker_run)
        scfg = json.loads((sr / "configs.json").read_text())
        acfg = _coerce_actor_cfg(scfg["actor"])
        with open(sr / "best_weights.pkl", "rb") as f:
            sp = pickle.load(f)["actor_params"]
        sec = StrikeEnvConfig.from_dict(scfg.get("env", {}))
        s_lo, s_hi = action_bounds(m_env, sec)                     # residual striker: a correction on pursuit
        s_act = lambda obs: jnp.clip(actor_mean_action(sp, obs, m_env.scale, acfg, action_low=s_lo, action_high=s_hi), s_lo, s_hi)
        if sec.residual_cruise > 0.0:
            base = pursuit_controller(m_env, sec.residual_cruise)
            nominal = lambda es, obs: jnp.clip(base(es.x, es.target_p, es.target_v) + s_act(obs), m.low, m.high)
        else:
            nominal = lambda es, obs: s_act(obs)
        nom_label = f"striker {sr.name}" + (f" (residual on pursuit at {sec.residual_cruise:g} m/s)" if sec.residual_cruise > 0 else "")

    filters = {}
    o_rng, b_rng = backup_ranges(a.backup_run)
    print(f"[backup input] obstacles within {o_rng:g} m, chaser balls within {b_rng:g} m (the Phase-I training range)", flush=True)
    for mode in a.modes.split():
        if mode == "none":
            continue
        bk, prm = load_backup(m, mode, a.backup_run if mode == "learned" else None)
        if a.filter_type == "bcbf":
            f1 = make_strike_bcbf(m, bk, prm, chaser_model=a.chaser_model, guidance=gcfg, arena=arena,
                                  bcbf=StrikeBCBFConfig.from_dict(json.loads(a.bcbf) if a.bcbf else {}),
                                  obstacle_range=o_rng, ball_range=b_rng, hover_latch=bool(a.hover_latch))
        else:
            f1 = make_filter(m, bk, prm, obstacle_range=o_rng, ball_range=b_rng, fresh_hold=a.fresh_hold,
                             hover_latch=bool(a.hover_latch), chaser_model=a.chaser_model, reach_hold=a.reach_hold,
                             guidance=gcfg, arena=arena)
        filters[mode] = (jax.vmap(f1), jax.vmap(f1.fresh_view))
    n_tr = n_ep if a.gif_pick != "first" else max(int(a.n_show), int(a.gif))

    def run(mode, key):
        es, obs = env.reset_batched(jax.random.split(key, n_ep))
        z = jnp.zeros((n_ep,))
        acc0 = {"len": z, "t_strike": z, "lam": z, "no_cert": z, "committed": z, "min_chaser": z + 1e3, **{k: z for k in OUTCOMES}}
        comm0 = filters[mode][1](es.x, es.obst, ball_c(es), es.chaser_on) if mode != "none" else jnp.zeros((n_ep, 1))

        def body(c, k):
            es, obs, alive, acc, comm = c
            u_nom = nominal(es, obs)
            if mode == "none":
                u, lam, nocert, used_c = u_nom, jnp.ones((n_ep,)), jnp.zeros((n_ep,), bool), jnp.zeros((n_ep,), bool)
            else:
                fresh = filters[mode][1](es.x, es.obst, ball_c(es), es.chaser_on)
                comm = jnp.where((es.steps == 0)[:, None], fresh, comm)       # a new episode: commit its first view
                u, info, comm = filters[mode][0](es.x, u_nom, es.obst, ball_c(es), es.chaser_on, comm)
                lam, nocert, used_c = info.lam, info.no_certificate, info.used_committed
            es2, _, obs_out, _, done, info2 = env.step_batched(es, u, jax.random.split(jax.random.fold_in(key, k), n_ep))
            al = alive.astype(jnp.float32)
            acc = {"len": acc["len"] + al, "lam": acc["lam"] + al * lam, "no_cert": acc["no_cert"] + al * nocert,
                   "committed": acc["committed"] + al * used_c,
                   "t_strike": jnp.where(alive & info2.strike, (k + 1).astype(jnp.float32), acc["t_strike"]),
                   "min_chaser": jnp.where(alive, jnp.minimum(acc["min_chaser"], info2.min_chaser_dist), acc["min_chaser"]),
                   **{o: jnp.maximum(acc[o], al * getattr(info2, o).astype(jnp.float32)) for o in OUTCOMES}}
            rec = (es.x[:n_tr, 0:3], es.chaser_p[:n_tr], es.target_p[:n_tr], lam[:n_tr], alive[:n_tr], ball_c(es)[:n_tr],
                   es.x[:n_tr, 3:6])
            return (es2, obs_out, alive & ~done, acc, comm), rec

        (_, _, _, acc, _), rec = jax.lax.scan(body, (es, obs, jnp.ones((n_ep,), bool), acc0, comm0), jnp.arange(env.max_steps))
        return acc, rec, es

    key = jax.random.PRNGKey(123)
    results, recs, starts, accs, episodes_out = {}, {}, None, {}, {}
    for mode in a.modes.split():
        t0 = time.time()
        acc, rec, es0 = jax.jit(lambda k: run(mode, k))(key)
        acc, rec = jax.device_get((acc, rec))
        starts = jax.device_get(es0) if starts is None else starts
        L = np.maximum(acc["len"], 1)
        s = acc["strike"] > 0.5
        crash = (acc["crash_obstacle"] + acc["crash_floor"] + acc["crash_chaser"] + acc["out_of_arena"]) > 0.5
        contact = (acc["crash_obstacle"] + acc["crash_floor"] + acc["crash_chaser"]) > 0.5     # left the safe set
        results[mode] = {"strike_rate": float(s.mean()), "crash_rate": float(crash.mean()), "contact_rate": float(contact.mean()),
                         **{f"{o}_rate": float((acc[o] > 0.5).mean()) for o in OUTCOMES if o != "strike"},
                         "strike_time_mean": float((acc["t_strike"][s] * cfg.dt).mean()) if s.any() else float("nan"),
                         "kept_fraction_mean": float((acc["lam"] / L).mean()),
                         "no_certificate_step_rate": float(acc["no_cert"].sum() / L.sum()),
                         "episodes_with_no_certificate": float((acc["no_cert"] > 0).mean()),
                         "committed_view_step_rate": float(acc["committed"].sum() / L.sum()),
                         "wall_time_sec": time.time() - t0}
        recs[mode] = rec
        accs[mode] = acc
        if a.save_episodes:      # per-episode records (outcome code: index in OUTCOMES; steps in units of dt)
            code = np.where(s, 0, np.where(acc["crash_obstacle"] > 0.5, 1, np.where(acc["crash_floor"] > 0.5, 2,
                            np.where(acc["crash_chaser"] > 0.5, 3, np.where(acc["out_of_arena"] > 0.5, 4, 5)))))
            episodes_out[mode] = {"outcome": code.astype(int).tolist(), "end_step": acc["len"].astype(int).tolist(),
                                  "strike_step": acc["t_strike"].astype(int).tolist(),
                                  "no_cert_steps": acc["no_cert"].astype(int).tolist(),
                                  "committed_steps": acc["committed"].astype(int).tolist(),
                                  "kept_fraction": (acc["lam"] / L).round(4).tolist(),
                                  "min_chaser_dist": np.round(acc["min_chaser"], 3).tolist()}
        has_ch = np.asarray(starts.chaser_on).max(axis=1) > 0.5          # scenes with at least one chaser
        for tag, sel in (("no_chaser", ~has_ch), ("with_chaser", has_ch)):
            results[mode][f"{tag}_episodes"] = int(sel.sum())
            results[mode][f"{tag}_strike_rate"] = float(s[sel].mean()) if sel.any() else float("nan")
            results[mode][f"{tag}_contact_rate"] = float(contact[sel].mean()) if sel.any() else float("nan")
            results[mode][f"{tag}_no_certificate_episodes"] = float((acc["no_cert"][sel] > 0).mean()) if sel.any() else float("nan")
        r = results[mode]
        print(f"[{mode:7s}] strike {r['strike_rate']:.3f} (t {r['strike_time_mean']:.2f} s) contact {r['contact_rate']:.3f} "
              f"[obstacle {r['crash_obstacle_rate']:.3f} floor {r['crash_floor_rate']:.3f} chaser {r['crash_chaser_rate']:.3f} "
              f"out {r['out_of_arena_rate']:.3f}] timeout {r['timeout_rate']:.3f} | kept {r['kept_fraction_mean']:.2f} "
              f"no-certificate steps {r['no_certificate_step_rate']:.4f} ({r['episodes_with_no_certificate']:.3f} of episodes) "
              f"| {r['wall_time_sec']:.0f} s", flush=True)

    meta = {"nominal": nom_label, "backup_run": str(a.backup_run), "episodes": n_ep, "scene_seed": a.scene_seed,
            "chaser_hold": cfg.chaser_hold, "backup_input_ranges": [o_rng, b_rng], "chaser_speed_scale": a.chaser_speed_scale, "fresh_hold": a.fresh_hold,
            "T": cfg.T, "ball_radius": cfg.ball_radius, "chaser_model": a.chaser_model, "reach_hold": a.reach_hold,
            "hover_latch": bool(a.hover_latch), "chaser_leash": a.chaser_leash,
            "guidance": gcfg.as_dict() if a.chaser_model == "guidance" else None, "env_overrides": a.env,
            "filter_type": a.filter_type, "bcbf": a.bcbf, "results": results}
    (out / "strike_filter_table.json").write_text(json.dumps(meta, indent=2))
    if a.save_episodes:
        has_ch_l = (np.asarray(starts.chaser_on).max(axis=1) > 0.5).astype(int).tolist()
        (out / "strike_filter_episodes.json").write_text(json.dumps(
            {"outcomes": list(OUTCOMES), "dt": cfg.dt, "max_steps": int(env.max_steps), "has_chaser": has_ch_l,
             "modes": episodes_out}))
    lines = [f"Nominal: {nom_label}; {n_ep} arena scenes (seed {a.scene_seed}); filter: discrete backup-set check with a "
             f"committed plan, T = {cfg.T} s, "
             + (f"backup trained with moving chasers (h_c, hover margin {cfg.chaser_base_margin:g} m)"
                + (" and arena walls" if cfg.arena_walls else "") if cfg.moving_chasers
                else f"chaser balls r = {cfg.ball_radius:.1f} m, chaser hold {cfg.chaser_hold:g} s") + "; "
             f"chaser model {a.chaser_model}, hover latch {bool(a.hover_latch)}, "
             + (f"guards on a {a.chaser_leash:g} m leash (balls at the posts)." if a.chaser_leash > 0 else "free pursuit."), "",
             "Contact = left the safe set (obstacle, floor, chaser); leaving the arena and running out of time end the mission "
             "but are not contacts.", "",
             "| mode | strike | contact | obstacle | floor | chaser | out of arena | timeout | strike time [s] | kept | no-certificate episodes |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for mode, r in results.items():
        lines.append(f"| {MODE_L.get(mode, mode)} | {r['strike_rate']:.3f} | {r['contact_rate']:.3f} | {r['crash_obstacle_rate']:.3f} | "
                     f"{r['crash_floor_rate']:.3f} | {r['crash_chaser_rate']:.3f} | {r['out_of_arena_rate']:.3f} | "
                     f"{r['timeout_rate']:.3f} | {r['strike_time_mean']:.2f} | {r['kept_fraction_mean']:.2f} | "
                     f"{r['episodes_with_no_certificate']:.3f} |")
    lines += ["", "By scene type (strike / contact / episodes that lost the certificate):", "",
              "| mode | no chaser | with chasers |", "|---|---|---|"]
    for mode, r in results.items():
        cell = lambda t: (f"{r[t + '_strike_rate']:.3f} / {r[t + '_contact_rate']:.3f} / {r[t + '_no_certificate_episodes']:.3f} "
                          f"(n = {r[t + '_episodes']})")
        lines.append(f"| {MODE_L.get(mode, mode)} | {cell('no_chaser')} | {cell('with_chaser')} |")
    (out / "strike_filter_table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    # ------------------------------------------------------------------ scene figure
    modes = list(recs)
    n_show = int(a.n_show)
    if n_show <= 0 and int(a.gif) <= 0:
        print(f"wrote {out}/strike_filter_table.md", flush=True)
        return
    obst_all = np.asarray(starts.obst)
    short = {"strike": "strike", "crash_obstacle": "hits obstacle", "crash_floor": "hits floor", "crash_chaser": "hit by chaser",
             "out_of_arena": "leaves arena", "timeout": "time up"}

    def outcome(mode, i):
        for o in OUTCOMES:
            if accs[mode][o][i] > 0.5:
                return short[o]
        return "?"

    if n_show > 0:
        ncol = 2
        nrow = int(np.ceil(n_show / ncol))
        fig_h = 4.4 * nrow + 1.0
        fig, axes = plt.subplots(nrow, ncol, figsize=(16, fig_h), squeeze=False)
        fig.patch.set_facecolor(SURF)
        for ax in axes.ravel()[n_show:]:
            ax.axis("off")
        for i in range(n_show):
            ax = axes.ravel()[i]
            ax.set_facecolor(SURF)
            ax.grid(color=GRID, lw=0.5)
            ax.set_axisbelow(True)
            for o in obst_all[i]:
                if o[6] > 0.5:
                    ax.add_patch(Polygon(outline(o), closed=True, fc=OBS_FILL, ec=OBS_EDGE, lw=0.9, zorder=2))
            verdicts = []
            for mode in modes:
                pos, chp, tgp, lam, alive, bcp, vel = (np.asarray(v)[:, i] for v in recs[mode])
                k_end = int(alive.sum())
                ax.plot(pos[:k_end, 0], pos[:k_end, 1], color=MODE_C[mode], lw=2.0, zorder=5)
                acc_out = outcome(mode, i)
                ax.plot(*pos[k_end - 1, :2], "x" if acc_out != "strike" else "*", color=MODE_C[mode], ms=10, mew=2, zorder=7)
                if mode == modes[-1]:
                    for j in range(chp.shape[1]):
                        if np.asarray(starts.chaser_on)[i, j] > 0.5:
                            ax.plot(chp[:k_end, j, 0], chp[:k_end, j, 1], color=CHASER, lw=1.2, ls="--", zorder=4)
                            if a.chaser_model != "guidance":       # the guidance set has no fixed ball
                                ax.add_patch(Circle(bcp[k_end - 1, j, :2], cfg.ball_radius, fc=CHASER, alpha=0.08, ec=CHASER, ls="--", lw=0.7, zorder=3))
                            ax.add_patch(Circle(chp[k_end - 1, j, :2], cfg.chaser_radius, fc=CHASER, ec=INK, lw=0.6, zorder=6))
                    ax.plot(tgp[:k_end, 0], tgp[:k_end, 1], color=TARGET, lw=1.5, zorder=4)
                    ax.plot(*tgp[k_end - 1, :2], "o", color=TARGET, ms=8, mec=INK, zorder=6)
                verdicts.append(f"{ {'none': 'no filter', 'lqr': 'LQR backup', 'learned': 'learned backup'}.get(mode, mode)}: {acc_out} ({k_end * cfg.dt:.1f} s)")
            ax.plot(*pos[0, :2], "o", color=INK, ms=6, zorder=8)
            ax.set_xlim(-1, ec.arena_x + 1)
            ax.set_ylim(-ec.arena_y_half - 0.5, ec.arena_y_half + 0.5)
            ax.set_aspect("equal")
            ax.tick_params(labelsize=7.5, colors=INK2)
            ax.set_title(f"scene {i}\n" + "  |  ".join(verdicts), loc="left", fontsize=9)
        fig.legend(handles=[Line2D([], [], color=MODE_C[mo], lw=2, label=MODE_L.get(mo, mo)) for mo in modes]
                   + [Line2D([], [], color=CHASER, lw=1.2, ls="--", label="chaser path"), Line2D([], [], color=TARGET, lw=1.5, label="target path"),
                      Line2D([], [], marker="*", ls="", ms=10, color=INK2, label="strike"), Line2D([], [], marker="x", ls="", ms=9, color=INK2, label="crash / time up")],
                   loc="lower center", ncol=4, frameon=False, fontsize=9.5)
        fig.suptitle(f"Strike arena, nominal = {nom_label}: without the filter and through the filter (Phase-I backups)",
                     x=0.01, ha="left", fontsize=12)
        fig.subplots_adjust(left=0.03, right=0.99, top=1 - 0.75 / fig_h, bottom=0.8 / fig_h, hspace=0.35, wspace=0.08)
        fig.savefig(out / "strike_filter_scenes.png", dpi=130, facecolor=SURF)
        plt.close(fig)

    # ------------------------------------------------------------------ GIF (one scene, modes side by side)
    if a.gif_pick in ("contact", "timeout", "failures"):
        last = accs[modes[-1]]
        hit = np.nonzero((last["crash_obstacle"] + last["crash_floor"] + last["crash_chaser"]) > 0.5)[0]
        tup = np.nonzero(last["timeout"] > 0.5)[0]
        k = int(a.gif)
        if a.gif_pick == "contact":
            gif_scenes = [int(j) for j in hit[:k]]
        elif a.gif_pick == "timeout":
            gif_scenes = [int(j) for j in tup[:k]]
        else:
            n_hit = min(len(hit), (k + 1) // 2)
            gif_scenes = [int(j) for j in hit[:n_hit]] + [int(j) for j in tup[:k - n_hit]]
        print(f"[gif] {a.gif_pick}: scenes {gif_scenes} (contacts {list(map(int, hit[:12]))}, time-ups {list(map(int, tup[:12]))})",
              flush=True)
    elif a.gif_pick == "chaser_hit":
        gif_scenes = [int(j) for j in np.nonzero(accs[modes[-1]]["crash_chaser"] > 0.5)[0][: int(a.gif)]]
        print(f"[gif] episodes hit by a chaser: scenes {gif_scenes}", flush=True)
    elif a.gif_pick == "close_chaser":
        last = accs[modes[-1]]
        cand = np.nonzero(last["strike"] > 0.5)[0]
        gif_scenes = [int(j) for j in cand[np.argsort(last["min_chaser"][cand])][: int(a.gif)]]
        dists = ", ".join(f"{last['min_chaser'][j]:.2f}" for j in gif_scenes)
        print(f"[gif] strikes closest to a chaser: scenes {gif_scenes} (min distance {dists} m)", flush=True)
    else:
        gif_scenes = list(range(int(a.gif)))
    for i in gif_scenes:
        fig, axes = plt.subplots(len(modes), 1, figsize=(7.2, 3.5 * len(modes) + 0.4), squeeze=False)
        fig.patch.set_facecolor(SURF)
        arts = []
        k_ends = []
        for row, mode in enumerate(modes):
            ax = axes[row, 0]
            ax.set_facecolor(SURF)
            for o in obst_all[i]:
                if o[6] > 0.5:
                    ax.add_patch(Polygon(outline(o), closed=True, fc=OBS_FILL, ec=OBS_EDGE, lw=0.9, zorder=2))
            ax.set_xlim(-1, ec.arena_x + 1)
            ax.set_ylim(-ec.arena_y_half - 0.5, ec.arena_y_half + 0.5)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(MODE_L.get(mode, mode), loc="left", fontsize=10, color=MODE_C[mode], fontweight="bold")
            pos, chp, tgp, lam, alive, bcp, vel = (np.asarray(v)[:, i] for v in recs[mode])
            k_ends.append(int(alive.sum()))
            path, = ax.plot([], [], color=MODE_C[mode], lw=2.0, zorder=5)
            drone = Circle((0, 0), cfg.r_drone * 1.6, fc=MODE_C[mode], ec=INK, lw=0.8, zorder=8)
            tgt = Circle((0, 0), ec.strike_radius, fc=TARGET, ec=INK, lw=0.8, zorder=6)
            ax.add_patch(drone)
            ax.add_patch(tgt)
            balls, chs = [], []
            for j in range(chp.shape[1]):
                on = np.asarray(starts.chaser_on)[i, j] > 0.5
                b = Circle((0, 0), cfg.ball_radius if on else 0.0, fc=CHASER, alpha=0.12, ec=CHASER, ls="--", lw=0.8, zorder=3)
                c_ = Circle((0, 0), cfg.chaser_radius if on else 0.0, fc=CHASER, ec=INK, lw=0.6, zorder=7)
                ax.add_patch(b)
                ax.add_patch(c_)
                balls.append(b)
                chs.append(c_)
            txt = ax.text(0.99, 0.03, "", transform=ax.transAxes, ha="right", va="bottom", fontsize=9, color=INK2)
            arts.append((mode, path, drone, tgt, balls, chs, txt, pos, chp, tgp, lam, bcp, vel))
        n_frames = max(k_ends)
        stride = 2

        def draw(f):
            k = min(f * stride, n_frames - 1)
            out_ = []
            for (mode, path, drone, tgt, balls, chs, txt, pos, chp, tgp, lam, bcp, vel), ke in zip(arts, k_ends):
                kk = min(k, ke - 1)
                path.set_data(pos[:kk + 1, 0], pos[:kk + 1, 1])
                drone.center = tuple(pos[kk, :2])
                tgt.center = tuple(tgp[kk, :2])
                for j, (b, c_) in enumerate(zip(balls, chs)):
                    b.center, c_.center = tuple(bcp[kk, j, :2]), tuple(chp[kk, j, :2])
                    if a.chaser_model == "guidance" and b.get_radius() > 0.0:   # distance needed at the current closing speed
                        rel = pos[kk] - chp[kk, j]
                        e_ = rel / max(np.linalg.norm(rel), 1e-6)
                        s_ = max(cfg.chaser_speed - float(e_ @ vel[kk]), 0.0)
                        b.set_radius(cfg.chaser_radius + cfg.r_drone + s_ * gcfg.delay + s_ ** 2 / (2.0 * gcfg.a_evade))
                state = "" if k < ke - 1 else ("  (strike)" if np.linalg.norm(pos[ke - 1] - tgp[ke - 1]) < ec.strike_radius + 0.5 else "  (ended)")
                txt.set_text(f"t = {kk * cfg.dt:4.1f} s   nominal kept {lam[kk]:.2f}{state}" if mode != "none" else f"t = {kk * cfg.dt:4.1f} s{state}")
                out_ += [path, drone, tgt, txt, *balls, *chs]
            return out_

        sub = ("\ndashed circle: distance the drone needs from the chaser at its current closing speed (guidance set)"
               if a.chaser_model == "guidance" else "\nshaded: the filter's chaser ball")
        outs = "; ".join(f"{MODE_L.get(mo, mo)}: {outcome(mo, i)}" for mo in modes)
        fig.suptitle(f"Scene {i} ({outs}), nominal = {nom_label}{sub}", x=0.01, ha="left", fontsize=9)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        ani = animation.FuncAnimation(fig, draw, frames=int(np.ceil(n_frames / stride)) + 10, blit=True)
        ani.save(out / f"strike_filter_scene{i}.gif", writer=animation.PillowWriter(fps=25), dpi=80)
        plt.close(fig)
    print(f"wrote {out}/strike_filter_table.md, strike_filter_scenes.png, strike_filter_scene*.gif", flush=True)


if __name__ == "__main__":
    main()
