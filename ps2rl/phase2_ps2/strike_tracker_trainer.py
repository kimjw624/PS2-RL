"""Phase II stage 1 for the strike task: the vanilla striker (SAC, no safety filter).

Same SAC recipe as ``cone_obstacle_ue_ps2_trainer`` in vanilla mode (actor action executed unfiltered, hover-biased
start, critics on reward_scale * r, replay of the last transitions), on ``ps2rl.envs.strike_env``. Timeouts are not
terminal for the critic target (only strikes, crashes and leaving the arena are). Stage 2 (through the strike filter,
warm-started from these weights) comes with the filter.

``touch <run_dir>/STOP`` ends a run early; it finishes normally (final evaluation, weights, summary).
Best checkpoint: highest eval return (the return carries the strike bonus and the crash penalty).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import json
import pickle
import time
from pathlib import Path
from typing import Any, Dict

import jax
import jax.numpy as jnp
import numpy as np

from ps2rl.envs.strike_env import StrikeEnvConfig, action_bounds, build_strike_env, load_or_generate_pool
from ps2rl.envs.strike_scene import StrikeModel
from ps2rl.utils.networks import CriticConfig, init_q_params, q_value
from ps2rl.utils.optim import adam_init, adam_step, soft_update
from ps2rl.utils.policy import ActorConfig, actor_mean_action, init_actor_params, sample_actor_action


@dataclass(frozen=True)
class StrikeTrackerConfig:
    seed: int = 0
    total_steps: int = 10_000_000
    start_steps: int = 10_000
    update_after: int = 5_000
    update_every: int = 2
    batch_size: int = 256
    replay_size: int = 1_000_000
    gamma: float = 0.99
    tau: float = 0.005
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    init_alpha: float = 0.1
    min_alpha: float = 1e-3
    target_entropy: float = -4.0
    max_grad_norm: float = 5.0
    q_clip_abs: float = 1e4
    hidden_size: int = 256
    num_envs: int = 64
    steps_per_jit: int = 32
    eval_every: int = 250_000
    eval_episodes: int = 128
    log_every: int = 100_000
    reward_scale: float = 0.1
    actor_start_update: int = 0        # actor / alpha updates start here (critic warm-up from update_after)
    action_anchor_weight: float = 0.0  # actor loss + w * mean(((a - a_start) / half_range)^2): a_start = hover, or du = 0 for
                                       # a residual striker (the anchor that kept the Phase-I residual from collapsing)
    # safety filter in the loop (Phase II PS2; residual striker only): the Phase-I backup run the filter uses ("" = no
    # filter), the filter ("discrete" backup-set filter or "bcbf" QP + safeguard), its chaser model, and the reward
    # weight on the filter's correction |(u - u_nom) / scale|^2
    filter_backup_run: str = ""
    filter_type: str = "discrete"
    filter_chaser_model: str = "guidance"
    w_filter: float = 0.0
    filter_guidance: str = ""          # GuidanceConfig overrides as JSON, e.g. '{"escape_variants": [[2.0, 60], [3.5, 0]]}'
    # start actor, critics, targets and alpha from a saved run's weights (.pkl, e.g. the unfiltered residual striker's
    # best_weights.pkl); same observation and action settings required
    warm_start: str = ""

    @classmethod
    def from_dict(cls, d):
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})


def _replay_init(cap, obs_dim):
    z = lambda *s: jnp.zeros((cap, *s), jnp.float32)
    return {"data": {"obs": z(obs_dim), "act": z(4), "rew": z(), "next_obs": z(obs_dim), "term": z()},
            "ptr": jnp.int32(0), "size": jnp.int32(0)}


def _replay_add(rp, vals):
    cap = rp["data"]["obs"].shape[0]
    n = vals["obs"].shape[0]
    idx = (rp["ptr"] + jnp.arange(n, dtype=jnp.int32)) % cap
    data = {k: rp["data"][k].at[idx].set(vals[k].astype(jnp.float32)) for k in rp["data"]}
    return {"data": data, "ptr": (rp["ptr"] + n) % cap, "size": jnp.minimum(rp["size"] + n, cap)}


def _replay_sample(rp, batch, key):
    idx = jax.random.randint(key, (batch,), 0, jnp.maximum(rp["size"], 1))
    return {k: v[idx] for k, v in rp["data"].items()}


EP_FIELDS = ("strike", "crash_obstacle", "crash_floor", "crash_chaser", "out_of_arena", "timeout")


def _filtered_env(cfg: StrikeTrackerConfig, m: StrikeModel, ec: StrikeEnvConfig, pool: dict):
    """The strike environment with the safety filter in the loop (strike_filtered_env)."""
    from ps2rl.cil.strike_bcbf import make_strike_bcbf
    from ps2rl.cil.strike_filter import backup_ranges, load_backup, make_filter, model_from_backup
    from ps2rl.cil.strike_guidance import GuidanceConfig
    from ps2rl.envs.strike_filtered_env import build_filtered_strike_env
    from ps2rl.envs.strike_scene import check_same_plant

    # the filter's model comes from its backup run (the backup's scene format); the environment / striker observation
    # keeps the task's strike block. They must describe the same plant and chasers.
    m_b = model_from_backup(cfg.filter_backup_run)
    if m_b.cfg != m.cfg:
        check_same_plant(m.cfg, m_b.cfg, "task strike block vs filter backup run")
        print(f"[filter] backup scene format: chaser_set {m_b.cfg.chaser_set}, walls {m_b.cfg.arena_walls}, T {m_b.cfg.T} s "
              f"(striker observation: chaser_set {m.cfg.chaser_set}, walls {m.cfg.arena_walls})", flush=True)
    bk, prm = load_backup(m_b, "learned", cfg.filter_backup_run)
    o_rng, b_rng = backup_ranges(cfg.filter_backup_run)
    arena = (0.0, ec.arena_x, ec.arena_y_half, ec.z_max)
    gcfg = GuidanceConfig.from_dict(json.loads(cfg.filter_guidance) if cfg.filter_guidance else {})
    if cfg.filter_type == "bcbf":
        filt = make_strike_bcbf(m_b, bk, prm, chaser_model=cfg.filter_chaser_model, guidance=gcfg, arena=arena,
                                obstacle_range=o_rng, ball_range=b_rng)
    elif cfg.filter_type == "discrete":
        filt = make_filter(m_b, bk, prm, obstacle_range=o_rng, ball_range=b_rng, chaser_model=cfg.filter_chaser_model,
                           guidance=gcfg, arena=arena)
    else:
        raise ValueError(f"filter_type must be 'discrete' or 'bcbf', got {cfg.filter_type!r}")
    return build_filtered_strike_env(m, ec, filt, pool=pool, w_filter=cfg.w_filter)


def run_strike_tracker(cfg: StrikeTrackerConfig, m: StrikeModel, ec: StrikeEnvConfig, out_dir: Path,
                       task: dict | None = None, pool_cache: str | None = None) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    pool = load_or_generate_pool(ec, max(ec.obstacles_max, m.cfg.n_obstacle_slots), cache_dir=pool_cache)
    if cfg.filter_backup_run:
        env = _filtered_env(cfg, m, ec, pool)
        print(f"[filter] {cfg.filter_type} filter in the loop ({cfg.filter_chaser_model} chasers), backup "
              f"{cfg.filter_backup_run}, w_filter {cfg.w_filter}", flush=True)
    else:
        env = build_strike_env(m, ec, pool=pool)
    low, high = action_bounds(m, ec)                  # plant box, or the residual range
    scale = m.scale
    n_env = int(cfg.num_envs)
    actor_cfg = ActorConfig(obs_dim=env.obs_dim, action_dim=4, hidden_sizes=(cfg.hidden_size, cfg.hidden_size),
                            log_std_min=-5.0, log_std_max=2.0, activation="elu")
    critic_cfg = CriticConfig(obs_dim=env.obs_dim, act_dim=4, hidden_sizes=(cfg.hidden_size, cfg.hidden_size))
    key = jax.random.PRNGKey(int(cfg.seed))
    key, ka, k1, k2, ke = jax.random.split(key, 5)
    ap = init_actor_params(ka, actor_cfg)
    lo_, hi_ = np.asarray(low, np.float64), np.asarray(high, np.float64)       # start at hover (residual: at du = 0)
    start = np.zeros(4) if ec.residual_cruise > 0.0 else np.asarray(m.u_hover)
    pre = np.arctanh(np.clip((start - 0.5 * (lo_ + hi_)) / (0.5 * (hi_ - lo_)), -0.999, 0.999))
    last = ap["layers"][-1]
    ap["layers"][-1] = {"w": last["w"].at[:, :4].set(0.0), "b": last["b"].at[:4].set(jnp.asarray(pre, jnp.float32))}
    q1, q2 = init_q_params(k1, critic_cfg), init_q_params(k2, critic_cfg)
    log_alpha = jnp.asarray(np.log(cfg.init_alpha), jnp.float32)
    tq1 = tq2 = None
    if cfg.warm_start:
        with open(Path(cfg.warm_start).expanduser(), "rb") as f:
            ws = pickle.load(f)
        to_j = lambda t: jax.tree_util.tree_map(lambda a: jnp.asarray(a, jnp.float32), t)
        ap, q1, q2 = to_j(ws["actor_params"]), to_j(ws["q1_params"]), to_j(ws["q2_params"])
        tq1, tq2 = to_j(ws.get("target_q1_params", ws["q1_params"])), to_j(ws.get("target_q2_params", ws["q2_params"]))
        log_alpha = jnp.asarray(ws["log_alpha"], jnp.float32)
        print(f"[warm start] actor, critics and alpha from {cfg.warm_start}", flush=True)
    copy = lambda t: jax.tree_util.tree_map(jnp.copy, t)
    state = {"actor": ap, "q1": q1, "q2": q2, "tq1": copy(q1 if tq1 is None else tq1), "tq2": copy(q2 if tq2 is None else tq2),
             "log_alpha": log_alpha,
             "actor_opt": adam_init(ap), "q1_opt": adam_init(q1), "q2_opt": adam_init(q2), "alpha_opt": adam_init(log_alpha)}
    la_min = float(np.log(cfg.min_alpha))

    def clipq(v):
        return jnp.clip(jnp.nan_to_num(v, nan=0.0, posinf=cfg.q_clip_abs, neginf=-cfg.q_clip_abs), -cfg.q_clip_abs, cfg.q_clip_abs)

    def sanitize(g):
        g = jax.tree_util.tree_map(lambda t: jnp.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0), g)
        nrm = jnp.sqrt(sum(jnp.sum(t * t) for t in jax.tree_util.tree_leaves(g)) + 1e-12)
        return jax.tree_util.tree_map(lambda t: t * jnp.minimum(1.0, cfg.max_grad_norm / (nrm + 1e-6)), g), nrm

    a_start = jnp.asarray(start, jnp.float32)
    a_half = 0.5 * (jnp.asarray(high) - jnp.asarray(low))

    def update(st, b, key, gstep):
        kc, ka_ = jax.random.split(key)
        act_on = gstep >= cfg.actor_start_update
        alpha = jnp.exp(jnp.clip(st["log_alpha"], la_min, 5.0))
        nr, nlp, _ = sample_actor_action(st["actor"], b["next_obs"], kc, scale, actor_cfg, action_low=low, action_high=high)
        nr = jnp.clip(jnp.nan_to_num(nr), low, high)
        nlp = jnp.clip(jnp.nan_to_num(nlp), -20.0, 20.0)
        tq = jnp.minimum(q_value(st["tq1"], b["next_obs"], nr), q_value(st["tq2"], b["next_obs"], nr)) - alpha * nlp
        y = jax.lax.stop_gradient(clipq(cfg.reward_scale * b["rew"] + cfg.gamma * (1.0 - b["term"]) * clipq(tq)))

        def closs(p1, p2):
            return jnp.mean((clipq(q_value(p1, b["obs"], b["act"])) - y) ** 2) + jnp.mean((clipq(q_value(p2, b["obs"], b["act"])) - y) ** 2)

        cl, (g1, g2) = jax.value_and_grad(closs, argnums=(0, 1))(st["q1"], st["q2"])
        g1, _ = sanitize(g1)
        g2, _ = sanitize(g2)
        q1n, o1 = adam_step(st["q1"], g1, st["q1_opt"], cfg.critic_lr)
        q2n, o2 = adam_step(st["q2"], g2, st["q2_opt"], cfg.critic_lr)

        def aloss(p):
            raw, lp, _ = sample_actor_action(p, b["obs"], ka_, scale, actor_cfg, action_low=low, action_high=high)
            raw = jnp.clip(jnp.nan_to_num(raw), low, high)
            lp = jnp.clip(jnp.nan_to_num(lp), -20.0, 20.0)
            qpi = clipq(jnp.minimum(q_value(q1n, b["obs"], raw), q_value(q2n, b["obs"], raw)))
            anchor = cfg.action_anchor_weight * jnp.mean(jnp.sum(((raw - a_start) / a_half) ** 2, axis=-1) / 4.0)
            return jnp.mean(alpha * lp - qpi) + anchor, (lp, jnp.mean(qpi))

        (al, (lp, qpi)), ga = jax.value_and_grad(aloss, has_aux=True)(st["actor"])
        ga, gan = sanitize(ga)
        apn, aon = adam_step(st["actor"], ga, st["actor_opt"], cfg.actor_lr)
        lan, alo = adam_step(st["log_alpha"], -jnp.mean(jax.lax.stop_gradient(lp + cfg.target_entropy)), st["alpha_opt"], cfg.alpha_lr)
        keep = lambda new, old: jax.tree_util.tree_map(lambda a, o: jnp.where(act_on, a, o), new, old)   # critic warm-up
        apn, aon = keep(apn, st["actor"]), keep(aon, st["actor_opt"])
        lan, alo = keep(lan, st["log_alpha"]), keep(alo, st["alpha_opt"])
        new = {"actor": apn, "q1": q1n, "q2": q2n, "tq1": soft_update(st["tq1"], q1n, cfg.tau), "tq2": soft_update(st["tq2"], q2n, cfg.tau),
               "log_alpha": jnp.clip(lan, la_min, 5.0), "actor_opt": aon, "q1_opt": o1, "q2_opt": o2, "alpha_opt": alo}
        return new, {"critic_loss": cl, "actor_loss": al, "q_pi": qpi, "alpha": jnp.exp(new["log_alpha"]), "actor_gn": gan}

    metric_keys = ("critic_loss", "actor_loss", "q_pi", "alpha", "actor_gn")
    max_due = max(1, int(np.ceil(n_env / cfg.update_every)))

    def vec_step(c, _):
        st, rp, es, obs, key, gstep, ep_ret = c
        key, kp, kr, ks, ku = jax.random.split(key, 5)
        raw, _, _ = sample_actor_action(st["actor"], obs, kp, scale, actor_cfg, action_low=low, action_high=high)
        rnd = low + jax.random.uniform(kr, raw.shape, jnp.float32) * (high - low)
        u = jnp.clip(jnp.nan_to_num(jnp.where(gstep < cfg.start_steps, rnd, raw)), low, high)
        es, obs_true, obs_out, rew, done, info = env.step_batched(es, u, jax.random.split(ks, n_env))
        rp = _replay_add(rp, {"obs": obs, "act": u, "rew": rew, "next_obs": obs_true, "term": info.terminal.astype(jnp.float32)})
        g2 = gstep + n_env
        lo = jnp.maximum(gstep + 1, cfg.update_after)
        due = jnp.where(g2 >= lo, g2 // cfg.update_every - (lo - 1) // cfg.update_every, 0)
        due = jnp.minimum(jnp.where(rp["size"] > cfg.batch_size, due, 0), max_due)
        m0 = {k: jnp.asarray(0.0, jnp.float32) for k in metric_keys} | {"n_upd": jnp.asarray(0.0, jnp.float32)}

        def body(i, cc):
            def do(cc2):
                st2, k2, m2 = cc2
                k2, kb, kup = jax.random.split(k2, 3)
                st2, mm = update(st2, _replay_sample(rp, cfg.batch_size, kb), kup, gstep)
                return st2, k2, {k: m2[k] + mm[k] for k in metric_keys} | {"n_upd": m2["n_upd"] + 1.0}
            return jax.lax.cond(i < due, do, lambda z: z, cc)

        st, ku, mu = jax.lax.fori_loop(0, max_due, body, (st, ku, m0))
        ep_ret = ep_ret + rew
        d = info.episode_done.astype(jnp.float32)
        em = {k: jnp.sum(getattr(info, k).astype(jnp.float32)) for k in EP_FIELDS}
        em |= {"n_ep": jnp.sum(d), "ret": jnp.sum(d * ep_ret)}
        ep_ret = ep_ret * (1.0 - d)
        return (st, rp, es, obs_out, key, g2, ep_ret), (mu, em)

    def chunk(c):
        c, (mu, em) = jax.lax.scan(vec_step, c, None, length=int(cfg.steps_per_jit))
        return c, jax.tree_util.tree_map(jnp.sum, (mu, em))

    chunk_j = jax.jit(chunk, donate_argnums=0)

    n_eval = int(cfg.eval_episodes)

    def eval_run(actor_params, key):
        es, obs = env.reset_batched(jax.random.split(key, n_eval))
        z = jnp.zeros((n_eval,))
        acc0 = {"ret": z, "len": z, "t_strike": z, **{k: z for k in EP_FIELDS}, "min_obs": z + 1e3, "min_ch": z + 1e3, "speed": z}

        def body(c, k):
            es, obs, alive, acc = c
            u = jnp.clip(actor_mean_action(actor_params, obs, scale, actor_cfg, action_low=low, action_high=high), low, high)
            es2, _, obs_out, rew, done, info = env.step_batched(es, u, jax.random.split(jax.random.fold_in(key, k), n_eval))
            a = alive.astype(jnp.float32)
            acc = {"ret": acc["ret"] + a * rew, "len": acc["len"] + a, "speed": acc["speed"] + a * info.speed,
                   "t_strike": jnp.where(alive & info.strike, (k + 1).astype(jnp.float32), acc["t_strike"]),
                   "min_obs": jnp.where(alive, jnp.minimum(acc["min_obs"], info.min_obstacle_dist), acc["min_obs"]),
                   "min_ch": jnp.where(alive, jnp.minimum(acc["min_ch"], info.min_chaser_dist), acc["min_ch"]),
                   **{f: jnp.maximum(acc[f], a * getattr(info, f).astype(jnp.float32)) for f in EP_FIELDS}}
            return (es2, obs_out, alive & ~done, acc), None

        (_, _, _, acc), _ = jax.lax.scan(body, (es, obs, jnp.ones((n_eval,), bool), acc0), jnp.arange(env.max_steps))
        return acc

    eval_j = jax.jit(eval_run)

    def summarize(acc):
        a = jax.device_get(acc)
        s = a["strike"] > 0.5
        crash = (a["crash_obstacle"] + a["crash_floor"] + a["crash_chaser"] + a["out_of_arena"]) > 0.5
        return {"return_mean": float(a["ret"].mean()), "return_std": float(a["ret"].std()), "strike_rate": float(s.mean()),
                "crash_rate": float(crash.mean()), **{f"{f}_rate": float((a[f] > 0.5).mean()) for f in EP_FIELDS if f != "strike"},
                "strike_time_mean": float((a["t_strike"][s] * m.cfg.dt).mean()) if s.any() else float("nan"),
                "mean_speed": float((a["speed"] / np.maximum(a["len"], 1)).mean()),
                "min_obstacle_dist": float(a["min_obs"].min()), "min_chaser_dist": float(a["min_ch"].min())}

    history: Dict[str, list] = {}
    best = {"score": -np.inf, "step": 0}

    def save(tag, st):
        with open(out_dir / f"{tag}_weights.pkl", "wb") as f:
            pickle.dump(jax.device_get({"actor_params": st["actor"], "q1_params": st["q1"], "q2_params": st["q2"],
                                        "target_q1_params": st["tq1"], "target_q2_params": st["tq2"],
                                        "log_alpha": st["log_alpha"]}), f)

    def log_eval(step, st):
        ev = summarize(eval_j(st["actor"], jax.random.PRNGKey(777)))
        for k, v in ev.items():
            history.setdefault(f"eval/{k}", []).append(v)
        history.setdefault("eval/step", []).append(step)
        print(f"  [eval {step}] return {ev['return_mean']:.1f}+-{ev['return_std']:.1f} strike {ev['strike_rate']:.3f} "
              f"(t {ev['strike_time_mean']:.2f} s) crash {ev['crash_rate']:.3f} [obstacle {ev['crash_obstacle_rate']:.3f} "
              f"floor {ev['crash_floor_rate']:.3f} chaser {ev['crash_chaser_rate']:.3f} out {ev['out_of_arena_rate']:.3f}] "
              f"timeout {ev['timeout_rate']:.3f} speed {ev['mean_speed']:.1f} m/s", flush=True)
        if ev["return_mean"] > best["score"]:
            best.update(score=ev["return_mean"], step=step, eval=ev)
            save("best", st)
        return ev

    with open(out_dir / "configs.json", "w") as f:
        json.dump({"tracker": asdict(cfg), "env": ec.as_dict(), "strike": m.cfg.as_dict(), "actor": asdict(actor_cfg),
                   "obs_dim": env.obs_dim, "task": task or {}}, f, indent=2)
    es, obs = env.reset_batched(jax.random.split(ke, n_env))
    carry = (state, _replay_init(int(cfg.replay_size), env.obs_dim), es, obs, key, jnp.int32(0), jnp.zeros((n_env,)))
    carry = jax.tree_util.tree_map(jnp.copy, carry)    # distinct buffers: the carry is donated to the training chunk
    log_eval(0, carry[0])
    t0 = time.time()
    last_t, last_s, done_steps, next_eval, next_log, agg = t0, 0, 0, int(cfg.eval_every), int(cfg.log_every), None
    per_chunk = n_env * int(cfg.steps_per_jit)
    stop_file = out_dir / "STOP"
    while done_steps < int(cfg.total_steps):
        if stop_file.exists():
            print(f"[stop] {stop_file} found at step {done_steps}: finishing", flush=True)
            break
        carry, (mu, em) = chunk_j(carry)
        done_steps += per_chunk
        mu, em = jax.device_get((mu, em))
        agg = (mu, em) if agg is None else (jax.tree_util.tree_map(lambda a, b: a + b, agg[0], mu),
                                            jax.tree_util.tree_map(lambda a, b: a + b, agg[1], em))
        if done_steps >= next_log:
            now = time.time()
            sps = (done_steps - last_s) / max(now - last_t, 1e-6)
            last_t, last_s = now, done_steps
            mu_, em_ = agg
            nu, ne = max(float(mu_["n_upd"]), 1.0), max(float(em_["n_ep"]), 1.0)
            rec = {"step": done_steps, "sps": sps, **{f"train/{k}": float(mu_[k]) / nu for k in metric_keys},
                   "episode/return": float(em_["ret"]) / ne, **{f"episode/{k}": float(em_[k]) / ne for k in EP_FIELDS}}
            for k, v in rec.items():
                history.setdefault(k, []).append(v)
            print(f"step={done_steps} sps={sps:.0f} ep_ret={rec['episode/return']:.1f} strike={rec['episode/strike']:.3f} "
                  f"obstacle={rec['episode/crash_obstacle']:.3f} floor={rec['episode/crash_floor']:.3f} "
                  f"chaser={rec['episode/crash_chaser']:.3f} out={rec['episode/out_of_arena']:.3f} timeout={rec['episode/timeout']:.3f} "
                  f"| critic={rec['train/critic_loss']:.3g} q_pi={rec['train/q_pi']:.2f} alpha={rec['train/alpha']:.3g}", flush=True)
            agg, next_log = None, next_log + int(cfg.log_every)
        if done_steps >= next_eval:
            log_eval(done_steps, carry[0])
            save("final", carry[0])
            with open(out_dir / "history.json", "w") as f:
                json.dump(history, f)
            next_eval += int(cfg.eval_every)
    save("final", carry[0])
    final_ev = log_eval(done_steps, carry[0])
    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f)
    summary = {"total_steps": done_steps, "wall_time_sec": time.time() - t0, "best_step": best["step"],
               "best_eval": best.get("eval"), "final_eval": final_ev, "jax_backend": jax.default_backend()}
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    return summary


__all__ = ["EP_FIELDS", "StrikeTrackerConfig", "run_strike_tracker"]
