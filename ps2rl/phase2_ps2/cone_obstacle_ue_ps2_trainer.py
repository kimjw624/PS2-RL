"""Phase-II PS2 for the cone-obstacle task (SAC through the cone-obstacle UE-bCBF CIL).

The algorithm and the implementation are those of ``landing_ue_ps2_trainer`` (the recipe that
worked for the landing under a disturbance): SAC whose executed and actor actions go through the
CIL, the CIL rows cached in the replay buffer (they depend on (x, d_hat) only), the carry donated
so the buffer is updated in place, and the two-stage recipe

1. ``use_projection=False`` (+ env ``terminate_on_unsafe=False``): the vanilla tracker - same
   environment and observation, the actor's action executed unfiltered;
2. ``warm_start_weights``: PS2 through the UE-bCBF CIL, warm-started from (1).

Only the filter (``ps2rl.cil.cone_obstacle_ue_bcbf``: funnel + floor + hover ellipsoid beside the
funnel, LQR or Phase-I backup), the environment (``ps2rl.envs.cone_obstacle_ue_env``: random-direction
sinusoidal disturbance with amplitude/frequency drawn within delta_d / delta_v) and the evaluation
metrics differ. Two options from the runway trainer are available (defaults = landing recipe):
``warm_start_critic`` and ``actor_start_update``.

Evaluation: return, unsafe episodes (inside the funnel or below the floor), min h_cone / floor
margin, touched (the vehicle reached the target cylinder under the tip), first-touch time,
final distance to the reference's end point, CIL statistics. The best checkpoint maximises
return - 1e4 unsafe_rate - select_touch_weight (1 - touched_rate) (unsafe and touched terms only
with the CIL; the vanilla tracker is selected by return).
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

from ps2rl.cil import cone_obstacle_ue_bcbf as cu
from ps2rl.envs.cone_obstacle_ue_env import ConeObstacleUEEnvConfig, build_cone_obstacle_ue_env
from ps2rl.phase2_ps2.landing_ue_ps2_trainer import _load_warm_start, _no_cil_aux, replay_add, replay_init, replay_sample, select_rows
from ps2rl.utils.networks import CriticConfig, init_q_params, q_value
from ps2rl.utils.optim import adam_init, adam_step, soft_update
from ps2rl.utils.policy import ActorConfig, actor_mean_action, init_actor_params, sample_actor_action


@dataclass(frozen=True)
class ConeUEPS2Config:
    seed: int = 0
    total_steps: int = 3_000_000
    start_steps: int = 10_000
    update_after: int = 5_000
    update_every: int = 16
    batch_size: int = 256
    replay_size: int = 300_000
    gamma: float = 0.99
    tau: float = 0.005
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    init_hover_bias: bool = True      # start at the hover command (g, 0, 0, 0)
    init_alpha: float = 0.1
    min_alpha: float = 1e-2
    target_entropy: float = -4.0
    max_grad_norm: float = 5.0
    q_clip_abs: float = 1e4
    hidden_size: int = 256
    project_target_actions: bool = False
    num_envs: int = 32
    steps_per_jit: int = 32
    eval_every: int = 250_000
    eval_episodes: int = 64
    log_every: int = 50_176
    rows_keep: int = 32
    qp_float64: bool = True
    alpha_cbf: float = -1.0            # < 0: the task file's filter.alpha
    alpha_floor: float = -1.0          # < 0: the task file's filter.alpha_ground
    base_alpha: float = -1.0           # < 0: the task file's filter.base_alpha
    backup: str = "lqr"                # "lqr" | "learned" (Phase-I safe-arrival actor)
    backup_path: str = ""              # Phase-I run directory or actor file (backup = "learned")
    use_projection: bool = True
    warm_start_weights: str = ""
    warm_start_critic: bool = True     # landing recipe: actor, critics and alpha from the warm start
    actor_start_update: int = 0        # no actor (and alpha) updates before this many critic updates
    select_touch_weight: float = 1000.0

    @classmethod
    def from_dict(cls, d):
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})


def filter_config(cfg: ConeUEPS2Config, task: dict):
    """The UE-bCBF CIL of a run (task file + the run's gain overrides + backup choice)."""
    over = {}
    if cfg.alpha_cbf >= 0:
        over["alpha"] = float(cfg.alpha_cbf)
    if cfg.alpha_floor >= 0:
        over["alpha_ground"] = float(cfg.alpha_floor)
    if cfg.base_alpha >= 0:
        over["base_alpha"] = float(cfg.base_alpha)
    return cu.cone_ue_config_from_task(task, backup=cfg.backup, backup_path=cfg.backup_path, **over)


def summarize_touch(acc: Dict[str, Any], dt: float) -> Dict[str, float]:
    a = {k: np.asarray(v) for k, v in acc.items()}
    L = np.maximum(a["len"], 1.0)
    fx = a["final_x"]
    touched = a["touched"] > 0.5
    t_hit = np.where(touched, a["t_touch"] * dt, np.nan)
    return {"return_mean": float(a["ret"].mean()), "return_std": float(a["ret"].std()),
            "unsafe_rate": float(a["unsafe"].mean()), "min_h_cone": float(a["min_h"].min()),
            "min_z": float(a["min_z"].min()), "pos_err_mean": float((a["pos_err"] / L).mean()),
            "touched_rate": float(touched.mean()), "touch_time_mean": float(np.nanmean(t_hit)) if touched.any() else float("nan"),
            "final_dist_mean": float(a["final_dist"].mean()), "final_z_mean": float(fx[:, 2].mean()),
            "final_speed_mean": float(np.linalg.norm(fx[:, 3:6], axis=1).mean()),
            "cil_correction_mean": float((a["corr"] / L).mean()), "slack_gt_rate": float((a["slack_gt"] / L).mean()),
            "fallback_rate": float((a["fallback"] / L).mean()), "safeguard_rate": float((a["safeguard"] / L).mean()),
            "max_observer_error": float(a["d_err"].max())}


def run_cone_obstacle_ue_ps2(cfg: ConeUEPS2Config, task: dict, env_cfg: ConeObstacleUEEnvConfig, out_dir: Path,
                             task_path: str = "") -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    use_proj = bool(cfg.use_projection)
    if use_proj and cfg.qp_float64 and not jax.config.jax_enable_x64:
        raise SystemExit("qp_float64=True needs JAX_ENABLE_X64=1 in the environment (or pass --qp_float64 false)")
    qp_dtype = jnp.float64 if cfg.qp_float64 else None
    cbf_cfg = filter_config(cfg, task)
    rt = cu.make_ue_runtime(cbf_cfg)
    recover = cu.make_recoverability_fn_ue(cbf_cfg, rt)
    env = build_cone_obstacle_ue_env(env_cfg, cbf_cfg, rt, recover)
    sc = cbf_cfg.scenario
    e_bar = jnp.asarray(float(cbf_cfg.ue.e_bar), jnp.float32)
    low = jnp.asarray(cbf_cfg.action_low, jnp.float32)
    high = jnp.asarray(cbf_cfg.action_high, jnp.float32)
    scale = jnp.asarray(cbf_cfg.action_scale, jnp.float32)
    k_rows = min(int(cfg.rows_keep), int(cbf_cfg.num_backup_inequalities)) if use_proj else 0
    n_env = int(cfg.num_envs)
    actor_cfg = ActorConfig(obs_dim=env.obs_dim, action_dim=4, hidden_sizes=(cfg.hidden_size, cfg.hidden_size),
                            log_std_min=-5.0, log_std_max=2.0, activation="relu")
    critic_cfg = CriticConfig(obs_dim=env.obs_dim, act_dim=4, hidden_sizes=(cfg.hidden_size, cfg.hidden_size))
    key = jax.random.PRNGKey(int(cfg.seed))
    key, ka, k1, k2, ke = jax.random.split(key, 5)
    f32 = lambda t: jax.tree_util.tree_map(lambda v: jnp.asarray(v, jnp.float32), t)  # networks stay f32 under x64
    ap = f32(init_actor_params(ka, actor_cfg))
    warm = bool(cfg.warm_start_weights.strip())
    if cfg.init_hover_bias and not warm:
        lo_, hi_ = np.asarray(cbf_cfg.action_low, np.float64), np.asarray(cbf_cfg.action_high, np.float64)
        hover = np.asarray([float(cbf_cfg.gravity), 0.0, 0.0, 0.0])
        pre = np.arctanh(np.clip((hover - 0.5 * (lo_ + hi_)) / (0.5 * (hi_ - lo_)), -0.999, 0.999))
        last = ap["layers"][-1]
        ap["layers"][-1] = {"w": last["w"].at[:, :4].set(0.0), "b": last["b"].at[:4].set(jnp.asarray(pre, jnp.float32))}
    q1, q2 = f32(init_q_params(k1, critic_cfg)), f32(init_q_params(k2, critic_cfg))
    log_alpha = jnp.asarray(np.log(max(cfg.init_alpha, cfg.min_alpha)), jnp.float32)
    tq1, tq2 = q1, q2
    if warm:
        ap_w, q1_w, q2_w, tq1_w, tq2_w, la_w = _load_warm_start(cfg.warm_start_weights, ap, q1, q2, log_alpha)
        ap, log_alpha = ap_w, la_w
        if cfg.warm_start_critic:
            q1, q2, tq1, tq2 = q1_w, q2_w, tq1_w, tq2_w
        else:
            print("[warm start] critics re-initialized (actor and alpha from the warm start)", flush=True)
    copy = lambda t: jax.tree_util.tree_map(jnp.copy, t)  # distinct buffers: the carry is donated below
    state = {"actor": ap, "q1": q1, "q2": q2, "tq1": copy(tq1), "tq2": copy(tq2), "log_alpha": log_alpha,
             "actor_opt": adam_init(ap), "q1_opt": adam_init(q1), "q2_opt": adam_init(q2),
             "alpha_opt": adam_init(log_alpha), "n_upd": jnp.int32(0)}
    la_min = float(np.log(cfg.min_alpha))
    a_start = int(cfg.actor_start_update)

    def proj_full_batch(x, dh, u_ref):
        return jax.vmap(lambda xx, dd, uu: cu.project_full(xx, dd, e_bar, uu, cbf_cfg, rt, qp_dtype))(x, dh, u_ref)

    def qp_cached(a, b, u, ub):
        return jax.vmap(lambda aa, bb, uu, bk: cu.solve_qp_from_rows(aa, bb, uu, bk, cbf_cfg, rt, qp_dtype))(a, b, u, ub)

    def clipq(v):
        return jnp.clip(jnp.nan_to_num(v, nan=0.0, posinf=cfg.q_clip_abs, neginf=-cfg.q_clip_abs), -cfg.q_clip_abs, cfg.q_clip_abs)

    def sanitize(g):
        g = jax.tree_util.tree_map(lambda t: jnp.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0), g)
        n = jnp.sqrt(sum(jnp.sum(t * t) for t in jax.tree_util.tree_leaves(g)) + 1e-12)
        s = jnp.minimum(1.0, cfg.max_grad_norm / (n + 1e-6))
        return jax.tree_util.tree_map(lambda t: t * s, g), n

    def update(st, b, key):
        kc, ka_ = jax.random.split(key)
        alpha = jnp.exp(jnp.clip(st["log_alpha"], la_min, 5.0))
        nr, nlp, _ = sample_actor_action(st["actor"], b["next_obs"], kc, scale, actor_cfg, action_low=low, action_high=high)
        nr = jnp.clip(jnp.nan_to_num(nr), low, high).astype(jnp.float32)
        nlp = jnp.clip(jnp.nan_to_num(nlp), -20.0, 20.0)
        if use_proj and cfg.project_target_actions:
            ns, _, nused = qp_cached(b["next_rows_a"], b["next_rows_b"], nr, b["next_u_backup"])
            ns = jax.lax.stop_gradient(ns)
        else:
            ns, nused = nr, jnp.ones(nr.shape[0], bool)
        tq = jnp.minimum(q_value(st["tq1"], b["next_obs"], ns), q_value(st["tq2"], b["next_obs"], ns)) - alpha * nlp
        y = jax.lax.stop_gradient(clipq(b["rew"] + cfg.gamma * (1.0 - b["done"]) * clipq(tq)))

        def closs(p1, p2):
            return jnp.mean((clipq(q_value(p1, b["obs"], b["act"])) - y) ** 2) + jnp.mean(
                (clipq(q_value(p2, b["obs"], b["act"])) - y) ** 2)

        cl, (g1, g2) = jax.value_and_grad(closs, argnums=(0, 1))(st["q1"], st["q2"])
        g1, _ = sanitize(g1)
        g2, _ = sanitize(g2)
        q1n, o1 = adam_step(st["q1"], g1, st["q1_opt"], cfg.critic_lr)
        q2n, o2 = adam_step(st["q2"], g2, st["q2_opt"], cfg.critic_lr)

        def aloss(p):
            raw, lp, _ = sample_actor_action(p, b["obs"], ka_, scale, actor_cfg, action_low=low, action_high=high)
            raw = jnp.clip(jnp.nan_to_num(raw), low, high).astype(jnp.float32)
            lp = jnp.clip(jnp.nan_to_num(lp), -20.0, 20.0).astype(jnp.float32)
            if use_proj:
                safe, slack, used = qp_cached(b["rows_a"], b["rows_b"], raw, b["u_backup"])
            else:
                safe, used = raw, jnp.ones(raw.shape[0], bool)
            qpi = clipq(jnp.minimum(q_value(q1n, b["obs"], safe), q_value(q2n, b["obs"], safe)))
            return jnp.mean(alpha * lp - qpi), (lp, jnp.mean(qpi), jnp.mean(used.astype(jnp.float32)),
                                                jnp.mean(jnp.linalg.norm((safe - raw) / scale, axis=-1)))

        (al, (lp, qpi, used_rate, corr)), ga = jax.value_and_grad(aloss, has_aux=True)(st["actor"])
        ga, gan = sanitize(ga)
        apn, aon = adam_step(st["actor"], ga, st["actor_opt"], cfg.actor_lr)
        ag = -jnp.mean(jax.lax.stop_gradient(lp + cfg.target_entropy))
        lan, alo = adam_step(st["log_alpha"], ag, st["alpha_opt"], cfg.alpha_lr)
        lan = jnp.clip(lan, la_min, 5.0)
        go = st["n_upd"] >= a_start
        keep = lambda new, old: jax.tree_util.tree_map(lambda n_, o_: jnp.where(go, n_, o_), new, old)
        new = {"actor": keep(apn, st["actor"]), "q1": q1n, "q2": q2n, "tq1": soft_update(st["tq1"], q1n, cfg.tau),
               "tq2": soft_update(st["tq2"], q2n, cfg.tau), "log_alpha": keep(lan, st["log_alpha"]),
               "actor_opt": keep(aon, st["actor_opt"]), "q1_opt": o1, "q2_opt": o2, "alpha_opt": keep(alo, st["alpha_opt"]),
               "n_upd": st["n_upd"] + 1}
        m = {"critic_loss": cl, "actor_loss": al, "q_pi": qpi, "alpha": jnp.exp(new["log_alpha"]), "actor_qp_used": used_rate,
             "target_qp_used": jnp.mean(nused.astype(jnp.float32)), "actor_corr": corr, "actor_gn": gan,
             "rows_exact": jnp.mean(b["rows_exact"])}
        return new, m

    metric_keys = ("critic_loss", "actor_loss", "q_pi", "alpha", "actor_qp_used", "target_qp_used", "actor_corr",
                   "actor_gn", "rows_exact")
    ep_keys = ("completed_return", "completed_len", "completed_min_h", "completed_min_z", "completed_pos_err",
               "completed_final_dist", "completed_final_z", "completed_final_speed", "completed_unsafe", "completed_touched")
    step_keys = ("slack_max", "slack_gt", "fallback", "safeguard", "corr")
    max_due = max(1, int(np.ceil(n_env / cfg.update_every)))

    def vec_step(c, _):
        st, rp, es, obs, key, gstep = c
        key, kp, kr, ks, ku = jax.random.split(key, 5)
        raw_pol, _, _ = sample_actor_action(st["actor"], obs, kp, scale, actor_cfg, action_low=low, action_high=high)
        raw_rand = low + jax.random.uniform(kr, raw_pol.shape, jnp.float32) * (high - low)
        raw = jnp.clip(jnp.nan_to_num(jnp.where(gstep < cfg.start_steps, raw_rand, raw_pol.astype(jnp.float32))),
                       low, high).astype(jnp.float32)
        if use_proj:
            u, aux = proj_full_batch(obs[:, :10], obs[:, 10:13], raw)
            u = u.astype(jnp.float32)
        else:  # vanilla: execute the policy's action
            u, aux = raw, _no_cil_aux(n_env)
        es, obs_true, obs_out, rew, done, info = env.step_batched(es, u, jax.random.split(ks, n_env))
        if use_proj:
            pts = jnp.stack([raw, u, aux["u_backup"]], axis=1)
            ra, rb = jax.vmap(lambda a, b, pp: select_rows(a, b, pp, k_rows))(aux["a_rows"], aux["b_rows"], pts)
            u_red, s_red, _ = qp_cached(ra, rb, raw, aux["u_backup"])
            row_scale = jnp.maximum(1.0, jnp.max(jnp.abs(aux["a_rows"]), axis=-1))
            viol = (jnp.einsum("nri,ni->nr", aux["a_rows"], u_red) - aux["b_rows"] - s_red[:, None]) / row_scale
            ex = jnp.max(viol, axis=-1) <= 1e-3
        else:
            ra, rb, ex = jnp.zeros((n_env, 0, 4), jnp.float32), jnp.zeros((n_env, 0), jnp.float32), jnp.ones(n_env, bool)
        rp = replay_add(rp, {"obs": obs, "act": u, "rew": rew, "next_obs": obs_true, "done": done.astype(jnp.float32),
                             "rows_a": ra, "rows_b": rb, "rows_exact": ex.astype(jnp.float32), "u_backup": aux["u_backup"]})
        g2 = gstep + n_env
        lo = jnp.maximum(gstep + 1, cfg.update_after)
        due = jnp.where(g2 >= lo, g2 // cfg.update_every - (lo - 1) // cfg.update_every, 0)
        due = jnp.minimum(jnp.where(rp["size"] > cfg.batch_size + n_env, due, 0), max_due)
        m0 = {k: jnp.asarray(0.0, jnp.float32) for k in metric_keys}
        m0["n_upd"] = jnp.asarray(0.0, jnp.float32)

        def body(i, cc):
            st_, k_, m_ = cc

            def do(cc2):
                st2, k2, m2 = cc2
                k2, kb, kup = jax.random.split(k2, 3)
                b = replay_sample(rp, cfg.batch_size, n_env, kb)
                st2, mm = update(st2, b, kup)
                m2 = {k: m2[k] + jnp.asarray(mm[k], jnp.float32) for k in metric_keys} | {"n_upd": m2["n_upd"] + 1.0}
                return st2, k2, m2

            return jax.lax.cond(i < due, do, lambda z: z, (st_, k_, m_))

        st, ku, mu = jax.lax.fori_loop(0, max_due, body, (st, ku, m0))
        sm = {"slack_max": jnp.max(aux["slack"]), "slack_gt": jnp.sum(aux["slack"] > 1e-3).astype(jnp.float32),
              "fallback": jnp.sum(~aux["used_solver"]).astype(jnp.float32),
              "safeguard": jnp.sum(aux["safeguard_lambda"] < 1.0).astype(jnp.float32),
              "corr": jnp.sum(jnp.linalg.norm((u - raw) / scale, axis=-1))}
        em = {k: jnp.sum(getattr(info, k)) for k in ep_keys} | {"n_ep": jnp.sum(info.episode_done.astype(jnp.float32))}
        return (st, rp, es, obs_out, key, g2), (mu, sm, em)

    def chunk(c):
        c, (mu, sm, em) = jax.lax.scan(vec_step, c, None, length=int(cfg.steps_per_jit))
        mu = jax.tree_util.tree_map(jnp.sum, mu)
        sm = {"slack_max": jnp.max(sm["slack_max"])} | {k: jnp.sum(sm[k]) for k in step_keys if k != "slack_max"}
        em = jax.tree_util.tree_map(jnp.sum, em)
        return c, (mu, sm, em)

    chunk_j = jax.jit(chunk, donate_argnums=0)

    # ------------------------------------------------------------------------- evaluation
    n_eval = int(cfg.eval_episodes)

    def eval_run(actor_params, key):
        es, obs = env.reset_batched(jax.random.split(key, n_eval))
        alive = jnp.ones((n_eval,), bool)
        z = jnp.zeros((n_eval,))

        def body(c, k):
            es, obs, alive, acc = c
            raw = jnp.clip(actor_mean_action(actor_params, obs, scale, actor_cfg, action_low=low, action_high=high),
                           low, high).astype(jnp.float32)
            if use_proj:
                u, aux = proj_full_batch(obs[:, :10], obs[:, 10:13], raw)
                u = u.astype(jnp.float32)
            else:
                u, aux = raw, _no_cil_aux(n_eval)
            es2, obs_true, obs_out, rew, done, info = env.step_batched(es, u, jax.random.split(jax.random.fold_in(key, k), n_eval))
            a = alive.astype(jnp.float32)
            first = alive & (info.touch > 0.5) & (acc["touched"] < 0.5)
            acc = {
                "ret": acc["ret"] + a * rew, "len": acc["len"] + a,
                "unsafe": jnp.maximum(acc["unsafe"], a * (1.0 - info.safe)),
                "min_h": jnp.where(alive, jnp.minimum(acc["min_h"], info.h_cone), acc["min_h"]),
                "min_z": jnp.where(alive, jnp.minimum(acc["min_z"], info.z), acc["min_z"]),
                "pos_err": acc["pos_err"] + a * info.pos_err,
                "corr": acc["corr"] + a * jnp.linalg.norm((u - raw) / scale, axis=-1),
                "slack_gt": acc["slack_gt"] + a * (aux["slack"] > 1e-3),
                "fallback": acc["fallback"] + a * (~aux["used_solver"]),
                "safeguard": acc["safeguard"] + a * (aux["safeguard_lambda"] < 1.0),
                "final_x": jnp.where(alive[:, None], obs_true[:, 0:6], acc["final_x"]),
                "final_dist": jnp.where(alive, jnp.linalg.norm(obs_true[:, 0:3] - env.ref_states[-1, 0:3], axis=-1),
                                        acc["final_dist"]),
                "touched": jnp.maximum(acc["touched"], a * info.touch),
                "t_touch": jnp.where(first, (k + 1).astype(jnp.float32), acc["t_touch"]),
                "d_err": jnp.maximum(acc["d_err"], a * jnp.linalg.norm(info.d_true - info.d_hat, axis=-1)),
            }
            alive = alive & ~done
            return (es2, obs_out, alive, acc), (obs_true[:, :13], u, raw, info.d_true)

        acc0 = {"ret": z, "len": z, "unsafe": z, "min_h": z + jnp.inf, "min_z": z + jnp.inf, "pos_err": z, "corr": z,
                "slack_gt": z, "fallback": z, "safeguard": z, "final_x": jnp.zeros((n_eval, 6)), "final_dist": z,
                "touched": z, "t_touch": z, "d_err": z}
        (_, _, _, acc), traj = jax.lax.scan(body, (es, obs, alive, acc0), jnp.arange(env.max_steps))
        return acc, traj

    eval_j = jax.jit(eval_run)

    def summarize_eval(acc):
        return summarize_touch(acc, float(sc.dt))

    # ------------------------------------------------------------------------- loop
    rp = replay_init(int(cfg.replay_size), env.obs_dim, k_rows)
    es, obs = env.reset_batched(jax.random.split(ke, n_env))
    own = lambda t: jax.tree_util.tree_map(jnp.copy, t)
    carry = (own(state), {"data": rp["data"], "ptr": jnp.copy(rp["ptr"]), "size": jnp.copy(rp["size"])},
             own(es), jnp.copy(obs), jnp.copy(key), jnp.int32(0))
    history: Dict[str, list] = {}
    best = {"score": -np.inf, "step": 0}
    t0 = time.time()
    last_t, last_s = t0, 0
    next_eval, next_log = int(cfg.eval_every), int(cfg.log_every)
    agg = None
    per_chunk = n_env * int(cfg.steps_per_jit)
    total = int(cfg.total_steps)

    def save(tag, st):
        with open(out_dir / f"{tag}_weights.pkl", "wb") as f:
            pickle.dump(jax.device_get({"actor_params": st["actor"], "q1_params": st["q1"], "q2_params": st["q2"],
                                        "target_q1_params": st["tq1"], "target_q2_params": st["tq2"],
                                        "log_alpha": st["log_alpha"]}), f)

    def log_eval(step, st):
        acc, _ = eval_j(st["actor"], jax.random.PRNGKey(777))
        ev = summarize_eval(acc)
        for k, v in ev.items():
            history.setdefault(f"eval/{k}", []).append(v)
        history.setdefault("eval/step", []).append(step)
        print(f"  [eval {step}] return {ev['return_mean']:.1f}+-{ev['return_std']:.1f} unsafe {ev['unsafe_rate']:.3f} "
              f"touched {ev['touched_rate']:.2f} (t {ev['touch_time_mean']:.2f} s) final dist {ev['final_dist_mean']:.3f} "
              f"pos_err {ev['pos_err_mean']:.3f} min h {ev['min_h_cone']:.3f} min z {ev['min_z']:.3f} "
              f"corr {ev['cil_correction_mean']:.3f} slack>1e-3 {ev['slack_gt_rate']:.4f} fallback {ev['fallback_rate']:.4f} "
              f"safeguard {ev['safeguard_rate']:.3f}", flush=True)
        score = ev["return_mean"] - (1e4 * ev["unsafe_rate"] if use_proj else 0.0) \
            - float(cfg.select_touch_weight) * (1.0 - ev["touched_rate"]) * float(use_proj)
        if score > best["score"]:
            best.update(score=score, step=step, eval=ev)
            save("best", st)
        return ev

    with open(out_dir / "configs.json", "w") as f:
        json.dump({"ps2": asdict(cfg), "env": env_cfg.as_dict(), "actor": asdict(actor_cfg), "residual": False,
                   "task": task, "task_file": task_path,
                   "actor_box": {"low": np.asarray(low).tolist(), "high": np.asarray(high).tolist()},
                   "cbf": {"alpha": cbf_cfg.alpha, "alpha_floor": cbf_cfg.alpha_floor, "base_alpha": cbf_cfg.base_alpha,
                           "backup": cbf_cfg.backup, "backup_path": cbf_cfg.backup_path,
                           "slack_weight": cbf_cfg.slack_weight, "solver_tol": cbf_cfg.solver_tol,
                           "safeguard_lambdas": list(cbf_cfg.safeguard_lambdas), "rho_scale": cbf_cfg.rho_scale,
                           "num_rows": int(cbf_cfg.num_backup_inequalities), "rows_keep": k_rows}},
                  f, indent=2)
    log_eval(0, carry[0])
    chunk_c = chunk_j.lower(carry).compile()
    try:
        ma = chunk_c.memory_analysis()
        rp_gb = sum(v.size * v.dtype.itemsize for v in jax.tree_util.tree_leaves(carry[1])) / 1e9
        print(f"[memory] replay buffer {rp_gb:.2f} GB, training chunk needs ~{(ma.argument_size_in_bytes + ma.temp_size_in_bytes) / 1e9:.2f} GB",
              flush=True)
    except Exception:
        pass
    done_steps = 0
    while done_steps < total:
        carry, (mu, sm, em) = chunk_c(carry)
        done_steps += per_chunk
        mu, sm, em = jax.device_get((mu, sm, em))
        agg = {"mu": mu, "sm": sm, "em": em} if agg is None else {
            "mu": {k: agg["mu"][k] + mu[k] for k in mu},
            "sm": {k: (max(agg["sm"][k], sm[k]) if k == "slack_max" else agg["sm"][k] + sm[k]) for k in sm},
            "em": {k: agg["em"][k] + em[k] for k in em}}
        if done_steps >= next_log:
            now = time.time()
            sps = (done_steps - last_s) / max(now - last_t, 1e-6)
            last_t, last_s = now, done_steps
            nu = max(float(agg["mu"]["n_upd"]), 1.0)
            ne = max(float(agg["em"]["n_ep"]), 1.0)
            nst = float(per_chunk) * (cfg.log_every // per_chunk or 1)
            rec = {"step": done_steps, "sps": sps,
                   **{f"train/{k}": float(agg["mu"][k]) / nu for k in metric_keys},
                   **{f"episode/{k.replace('completed_', '')}": float(agg["em"][k]) / ne for k in ep_keys},
                   "cil/slack_max": float(agg["sm"]["slack_max"]), "cil/slack_gt_rate": float(agg["sm"]["slack_gt"]) / nst,
                   "cil/fallback_rate": float(agg["sm"]["fallback"]) / nst,
                   "cil/safeguard_rate": float(agg["sm"]["safeguard"]) / nst, "cil/corr": float(agg["sm"]["corr"]) / nst}
            for k, v in rec.items():
                history.setdefault(k, []).append(v)
            print(f"step={done_steps} sps={sps:.0f} ep_ret={rec['episode/return']:.1f} unsafe={rec['episode/unsafe']:.3f} "
                  f"touched={rec['episode/touched']:.3f} min_h={rec['episode/min_h']:.3f} final_dist={rec['episode/final_dist']:.3f} "
                  f"| critic={rec['train/critic_loss']:.3g} q_pi={rec['train/q_pi']:.1f} alpha={rec['train/alpha']:.3g} "
                  f"qp_used={rec['train/actor_qp_used']:.3f} rows_exact={rec['train/rows_exact']:.3f} "
                  f"| cil corr={rec['cil/corr']:.3f} slack>1e-3={rec['cil/slack_gt_rate']:.4f} "
                  f"fallback={rec['cil/fallback_rate']:.4f} safeguard={rec['cil/safeguard_rate']:.3f}", flush=True)
            agg = None
            next_log += int(cfg.log_every)
        if done_steps >= next_eval:
            log_eval(done_steps, carry[0])
            save("final", carry[0])
            with open(out_dir / "history.json", "w") as f:
                json.dump(history, f)
            next_eval += int(cfg.eval_every)
    save("final", carry[0])
    if history.get("eval/step") and history["eval/step"][-1] == done_steps:
        final_ev = {k.split("/", 1)[1]: v[-1] for k, v in history.items() if k.startswith("eval/") and k != "eval/step"}
    else:
        final_ev = log_eval(done_steps, carry[0])
    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f)
    summary = {"total_steps": done_steps, "wall_time_sec": time.time() - t0, "best_step": best["step"],
               "best_eval": best.get("eval"), "final_eval": final_ev, "backup": cbf_cfg.backup,
               "backup_path": cbf_cfg.backup_path, "ue_config": cbf_cfg.ue.as_dict(),
               "cone_obstacle_config": sc.as_dict()}
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    return {"summary": summary, "eval_fn": eval_j, "summarize": summarize_eval, "env": env, "cbf_cfg": cbf_cfg,
            "actor_cfg": actor_cfg}


__all__ = ["ConeUEPS2Config", "filter_config", "run_cone_obstacle_ue_ps2", "summarize_touch"]
