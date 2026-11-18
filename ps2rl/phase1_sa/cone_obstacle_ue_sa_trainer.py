"""Phase-I safe-arrival training for the cone-obstacle task (outside of an apex-down funnel).

The update, the collector and the training loop are the landing UE ones
(``quadrotor_landing_ue_sa_trainer``: discounted safe arrival, TD3, contraction regularizer,
design-region curriculum); they only see the environment through ``env_fns``. This module
supplies the cone-obstacle environment, a behaviour-cloning start from the base set's LQR, and
the evaluator.

Two training variants (``tighten``): on the tube-tightened ("shrunk") sets, or on the plain sets.
Both are trained without a disturbance (d_hat = 0). The evaluator reports, on held-out design-region
states, both criteria for the backup *as it will be used in Phase II* (actor blind to d_hat, plant
with a frozen d_hat):

* ``tight``   first hit of the tightened base set inside the tightened safe set, with |d_hat| up to
              delta_d + e_bar in the rollout dynamics - what the Phase-II UE-bCBF filter checks;
* ``nominal`` the plain BCBF criterion (d_hat = 0, no margins).

``m_hat`` / ``mu_weighted`` (used for model selection during training) follow the variant.
"""

from __future__ import annotations

from dataclasses import asdict
import json
from typing import Any, Callable, Dict

import jax
import jax.numpy as jnp
import numpy as np

from ps2rl.backup_policy.backup_policy import BackupPolicy
from ps2rl.phase1_sa.cone_obstacle_design_region import REGION_NAMES, ConeObstacleDesignRegionConfig
from ps2rl.phase1_sa.cone_obstacle_ue_sa_env import build_cone_obstacle_indicators, build_cone_obstacle_ue_sa_env
from ps2rl.phase1_sa.landing_design_region import heldout_sets
from ps2rl.phase1_sa.quadrotor_landing_sa_env import landing_action_box
from ps2rl.phase1_sa.quadrotor_landing_ue_sa_env import sample_d_hat
from ps2rl.phase1_sa.quadrotor_landing_ue_sa_trainer import (
    _EPISODE_HISTORY_PAIRS,
    _EPISODE_METRIC_FIELDS,
    _HISTORY_KEYS,
    LandingUESAConfig,
    _append_eval_history,
    _episode_bookkeeping,
    _eval_rank_key,
    build_ue_one_vec_step,
    build_ue_update_fn,
    make_composed_backup,
)
from ps2rl.phase1_sa.sa_critic import SafeArrivalCriticConfig, init_twin_q_params
from ps2rl.phase1_sa.sa_trainer_core import SALoopState, SASystemHooks, run_sa_training_loop, sa_replay_init, snapshot_sa_state
from ps2rl.uncertainty.cone_obstacle_ue_tube import UEConfig, make_growth_fn, tube_step
from ps2rl.utils.optim import adam_init, adam_step
from ps2rl.utils.policy import ActorConfig, actor_mean_action, init_actor_params
from ps2rl.utils.seed import make_prng_key

ConeObstacleUESAConfig = LandingUESAConfig  # same hyper-parameters


# ----------------------------------------------------------------------------- backups as used in Phase II
def make_blind_backup(cfg, env_fns, actor_cfg: ActorConfig) -> Callable:
    """(params, x, d_hat) -> composed backup with the actor blind to d_hat (it was trained with d_hat = 0)."""
    pi_b = make_composed_backup(cfg, env_fns, actor_cfg)
    return lambda params, x, d_hat: pi_b(params, x, jnp.zeros_like(d_hat))


def make_lqr_backup(cfg, env_fns) -> Callable:
    """(params, x, d_hat) -> the base set's LQR everywhere (params ignored)."""
    _, low, high = landing_action_box(cfg)
    ctrl = env_fns.base_set.controller
    return lambda params, x, d_hat: jnp.clip(ctrl.action(x), low, high)


def make_growth(backup_fn: Callable, env_fns) -> Callable:
    """(params, x, d_hat) -> ||F||_P of the frozen-estimate closed-loop step of ``backup_fn``."""
    ctrl = env_fns.base_set.controller
    plant = env_fns.plant_step

    def growth(params, x, d_hat):
        step = lambda z: plant(z, backup_fn(params, z, d_hat), d_hat)
        return make_growth_fn(step, ctrl, env_fns.tube)(x)

    return growth


def make_rollout(backup_fn: Callable, env_fns, ue: UEConfig, n_steps: int, dt: float) -> Callable:
    """(params, x0, d_hat, tight) -> (reached B, failed, arrival step, tube at the end, max growth rate)."""
    growth = make_growth(backup_fn, env_fns)
    plant, base_set, tc = env_fns.plant_step, env_fns.base_set, env_fns.tube
    nf_t, goal_t = build_cone_obstacle_indicators(env_fns.safe_set, base_set, ue, tc, tighten=True)
    nf_n, goal_n = build_cone_obstacle_indicators(env_fns.safe_set, base_set, ue, tc, tighten=False)

    def rollout(params, x0, d_hat, tight):
        not_failed = lambda x, s: jnp.where(tight, nf_t(x, s), nf_n(x, s))
        goal = lambda x, s: jnp.where(tight, goal_t(x, s), goal_n(x, s))

        def body(c, k):
            x, s, hb, hf, rmax = c
            g = growth(params, x, d_hat)
            xn = plant(x, backup_fn(params, x, d_hat), d_hat)
            sn = jnp.where(tight, tube_step(s, g, k * dt, ue, tc, dt), 0.0)
            act = ~(hb | hf)
            nf = act & ~not_failed(xn, sn)
            nb = act & ~nf & goal(xn, sn)
            rate = jnp.where(base_set.contains(x), -jnp.inf, jnp.log(jnp.maximum(g, 1e-6)) / dt)
            return (jnp.where(act, xn, x), jnp.where(act, sn, s), hb | nb, hf | nf, jnp.maximum(rmax, rate)), nb

        z = jnp.asarray(0.0)
        (_, s_end, hb, hf, rmax), hits = jax.lax.scan(
            body, (x0, z, base_set.contains(x0), ~not_failed(x0, z), -jnp.inf), jnp.arange(n_steps))
        return hb, hf, jnp.where(hb, jnp.argmax(hits) + 1, -1), s_end, rmax

    return rollout


def eval_d_hats(ue_eval: UEConfig, n: int, seed: int) -> np.ndarray:
    keys = jax.random.split(jax.random.PRNGKey(seed), n)
    return np.asarray(jax.vmap(lambda k: sample_d_hat(k, ue_eval))(keys))


def build_cone_obstacle_evaluator(cfg, region_cfg: ConeObstacleDesignRegionConfig, ue_eval: UEConfig, env_fns,
                                  backup_fn: Callable, *, tighten: bool, hoeffding_delta: float, d_hat_per_state: int = 1):
    n_steps, dt = int(cfg.num_steps), float(cfg.dt)
    rb = jax.jit(jax.vmap(make_rollout(backup_fn, env_fns, ue_eval, n_steps, dt), in_axes=(None, 0, 0, None)))
    sets = {split: heldout_sets(region_cfg, env_fns.sampler, split=split) for split in ("val", "test")}
    dh_sets = {split: {name: eval_d_hats(ue_eval, sets[split][name].shape[0] * int(d_hat_per_state), 4242 + 100 * off + i)
                       for i, name in enumerate(REGION_NAMES)} for split, off in (("val", 0), ("test", 1))}

    def run_eval(params, split: str) -> Dict[str, Any]:
        stats: Dict[str, Any] = {"split": split, "criterion": "tight" if tighten else "nominal"}
        acc = {"tight": 0.0, "nominal": 0.0}
        den = 0.0
        crash, rates = [], []
        w = region_cfg.weights
        for name in REGION_NAMES:
            x0 = np.repeat(sets[split][name], int(d_hat_per_state), axis=0)
            dh = dh_sets[split][name]
            hb, hf, th, s_end, rmax = map(np.asarray, rb(params, jnp.asarray(x0), jnp.asarray(dh), True))
            hb0, hf0, _, _, _ = map(np.asarray, rb(params, jnp.asarray(x0), jnp.zeros_like(jnp.asarray(dh)), False))
            m = int(x0.shape[0])
            arr = th[hb]
            crit_hb, crit_hf = (hb, hf) if tighten else (hb0, hf0)
            stats[name] = {
                "n": m, "m_hat": float(crit_hb.mean()), "tight_m_hat": float(hb.mean()), "nominal_m_hat": float(hb0.mean()),
                "hoeffding_halfwidth": float(np.sqrt(np.log(2.0 / hoeffding_delta) / (2.0 * max(m, 1)))),
                "crash_rate": float(crit_hf.mean()), "timeout_rate": float((~crit_hb & ~crit_hf).mean()),
                "mean_arrival_steps": float(arr.mean()) if arr.size else float("nan"),
                "tube_at_arrival_median": float(np.median(s_end[hb])) if hb.any() else float("nan"),
            }
            acc["tight"] += w[name] * stats[name]["tight_m_hat"]
            acc["nominal"] += w[name] * stats[name]["nominal_m_hat"]
            den += w[name]
            crash.append(crit_hf)
            rr = rmax[np.isfinite(rmax)]
            if rr.size:
                rates.append(rr)
        stats["tight_mu_weighted"] = acc["tight"] / max(den, 1e-12)
        stats["nominal_mu_weighted"] = acc["nominal"] / max(den, 1e-12)
        stats["mu_weighted"] = stats["tight_mu_weighted"] if tighten else stats["nominal_mu_weighted"]
        stats["crash_rate"] = float(np.concatenate(crash).mean())
        allr = np.concatenate(rates) if rates else np.array([np.nan])
        stats["rate_p90"] = float(np.percentile(allr, 90))
        stats["rate_p50"] = float(np.percentile(allr, 50))
        return stats

    return run_eval


# ----------------------------------------------------------------------------- actor start
def behaviour_clone_base_controller(params, env_fns, actor_cfg: ActorConfig, cfg, *, steps: int = 3000, batch: int = 512,
                                    lr: float = 1e-3, seed: int = 5):
    """Fit the actor's deterministic action to the base set's LQR on design-region states (d_hat = 0)."""
    scale, lo, hi = landing_action_box(cfg)
    ctrl = env_fns.base_set.controller
    samp = jax.vmap(lambda k: env_fns.sampler["sample"](k, jnp.asarray(1.0, jnp.float32)))

    def loss(p, o, y):
        a = actor_mean_action(p, o, scale, actor_cfg, action_low=lo, action_high=hi)
        return jnp.mean(jnp.square((a - y) / scale))

    @jax.jit
    def step(p, opt, k):
        x = samp(jax.random.split(k, batch))
        o = jnp.concatenate([x, jnp.zeros((batch, 3), x.dtype)], -1)
        l, g = jax.value_and_grad(loss)(p, o, jnp.clip(jax.vmap(ctrl.action)(x), lo, hi))
        p, opt = adam_step(p, g, opt, lr)
        return p, opt, l

    opt = adam_init(params)
    key = jax.random.PRNGKey(seed)
    l = jnp.nan
    for _ in range(int(steps)):
        key, k = jax.random.split(key)
        params, opt, l = step(params, opt, k)
    print(f"[actor init] behaviour-cloned the base LQR: {steps} steps, final normalized MSE {float(l):.2e}", flush=True)
    return params


# ----------------------------------------------------------------------------- training
def run_cone_obstacle_ue_sa_training(ra_cfg: LandingUESAConfig, cfg, region_cfg: ConeObstacleDesignRegionConfig,
                                     ue_train: UEConfig, ue_eval: UEConfig, *, tighten: bool,
                                     output_dir: str | None = None, actor_init: str = "lqr_bc") -> Dict[str, Any]:
    env_fns = build_cone_obstacle_ue_sa_env(cfg, region_cfg, ue_train, tighten=tighten,
                                            episode_max_steps=int(ra_cfg.episode_max_steps),
                                            terminate_on_goal=bool(ra_cfg.collector_terminate_on_goal))
    actor_cfg = ActorConfig(obs_dim=env_fns.actor_obs_dim, action_dim=4, hidden_sizes=(ra_cfg.hidden_size, ra_cfg.hidden_size),
                            log_std_min=ra_cfg.actor_log_std_min, log_std_max=ra_cfg.actor_log_std_max,
                            activation=ra_cfg.actor_activation)
    critic_cfg = SafeArrivalCriticConfig(obs_dim=env_fns.obs_dim, act_dim=4, hidden_sizes=(ra_cfg.hidden_size, ra_cfg.hidden_size))
    key = make_prng_key(ra_cfg.seed)
    key, k_a, k_c, k_env = jax.random.split(key, 4)
    actor_params = init_actor_params(k_a, actor_cfg)
    if actor_init == "lqr_bc":
        actor_params = behaviour_clone_base_controller(actor_params, env_fns, actor_cfg, cfg)
    elif actor_init != "random":
        raise ValueError(f"actor_init must be 'lqr_bc' or 'random', got {actor_init!r}")
    q1, q2 = init_twin_q_params(k_c, critic_cfg)
    state = {"actor_params": actor_params, "target_actor_params": actor_params, "q1_params": q1, "q2_params": q2,
             "target_q1_params": q1, "target_q2_params": q2, "actor_opt": adam_init(actor_params),
             "q1_opt": adam_init(q1), "q2_opt": adam_init(q2), "update_step": jnp.int32(0)}
    replay = sa_replay_init(ra_cfg.replay_size, env_fns.obs_dim, 4)
    update_fn = build_ue_update_fn(ra_cfg, actor_cfg, cfg, env_fns)
    env_keys = jax.random.split(k_env, int(ra_cfg.num_envs))
    s0 = jnp.asarray(float(ra_cfg.curriculum_start_scale), dtype=jnp.float32)
    env_state, obs = env_fns.reset_batched(env_keys, s0)
    window = int(max(1, ra_cfg.curriculum_window_episodes))
    loop_state = SALoopState(state=state, replay=replay, env_state=env_state, obs=obs, key=key, env_keys=env_keys,
                             global_step=jnp.int32(0), updates=jnp.int32(0), curriculum_scale=s0,
                             episode_count=jnp.int32(0), success_window=jnp.zeros((window,), jnp.float32),
                             success_window_size=jnp.int32(0), success_window_ptr=jnp.int32(0))
    blind = make_blind_backup(cfg, env_fns, actor_cfg)
    run_eval = build_cone_obstacle_evaluator(cfg, region_cfg, ue_eval, env_fns, blind, tighten=tighten,
                                             hoeffding_delta=float(ra_cfg.hoeffding_delta),
                                             d_hat_per_state=int(ra_cfg.eval_d_hat_per_state))
    hooks = SASystemHooks(episode_metric_fields=_EPISODE_METRIC_FIELDS, episode_bookkeeping=_episode_bookkeeping,
                          run_eval=run_eval, eval_rank_key=_eval_rank_key, append_eval_history=_append_eval_history,
                          episode_history_pairs=_EPISODE_HISTORY_PAIRS, history_keys=_HISTORY_KEYS)
    lqr_eval = build_cone_obstacle_evaluator(cfg, region_cfg, ue_eval, env_fns, make_lqr_backup(cfg, env_fns), tighten=tighten,
                                             hoeffding_delta=float(ra_cfg.hoeffding_delta),
                                             d_hat_per_state=int(ra_cfg.eval_d_hat_per_state))
    lqr_val = lqr_eval(state["actor_params"], "val")
    initial_eval = run_eval(state["actor_params"], "val")
    print(f"[eval] LQR backup: tight {lqr_val['tight_mu_weighted']:.3f} nominal {lqr_val['nominal_mu_weighted']:.3f} | "
          f"actor at start: tight {initial_eval['tight_mu_weighted']:.3f} nominal {initial_eval['nominal_mu_weighted']:.3f}",
          flush=True)
    one_vec_step = build_ue_one_vec_step(env_fns=env_fns, update_fn=update_fn, ra_cfg=ra_cfg, actor_cfg=actor_cfg, cfg=cfg,
                                         hooks=hooks)
    res = run_sa_training_loop(ra_cfg=ra_cfg, loop_state=loop_state, one_vec_step=one_vec_step, hooks=hooks)
    best_eval = res.best_eval_stats if res.best_eval_stats is not None else res.val_eval
    best_state = res.best_state if res.best_state is not None else snapshot_sa_state(res.loop_state.state)
    test_at_best = run_eval(best_state["actor_params"], "test")
    summary = {
        "training_objective": "discounted_safe_arrival_" + ("ue_tightened" if tighten else "nominal"),
        "task": "cone_obstacle_outside_funnel", "tighten": bool(tighten), "disturbance_in_training": False,
        "sa_backbone": "td3", "seed": int(ra_cfg.seed), "total_steps": int(ra_cfg.total_steps),
        "updates": int(jax.device_get(res.loop_state.updates)), "wall_time_sec": float(res.total_time),
        "jax_backend": jax.default_backend(),
        "final_curriculum_scale": float(jax.device_get(res.loop_state.curriculum_scale)),
        "tube_constants": env_fns.tube.as_dict(), "lqr_val": lqr_val, "lqr_test": lqr_eval(state["actor_params"], "test"),
        "untrained_val": initial_eval, "final_val": res.val_eval, "best_val": best_eval,
        "best_eval_step": int(res.best_eval_step), "test_at_best": test_at_best, "test_at_final": res.test_eval,
    }
    configs = {"cone_obstacle": cfg.as_dict(), "design_region": region_cfg.as_dict(), "ue_train": ue_train.as_dict(),
               "ue_eval": ue_eval.as_dict(), "tighten": bool(tighten), "backup_ra": asdict(ra_cfg),
               "actor": asdict(actor_cfg), "actor_init": actor_init, "actor_sees_d_hat": False}
    if output_dir is not None:
        with open(f"{output_dir}/summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=float)
        with open(f"{output_dir}/configs.json", "w", encoding="utf-8") as f:
            json.dump(configs, f, indent=2)
    return {"summary": summary, "history": res.history, "configs": configs,
            "final_state": snapshot_sa_state(res.loop_state.state), "best_state": best_state, "actor_cfg": actor_cfg,
            "tube": env_fns.tube, "env_fns": env_fns}


__all__ = ["ConeObstacleUESAConfig", "behaviour_clone_base_controller", "build_cone_obstacle_evaluator", "eval_d_hats",
           "make_blind_backup", "make_growth", "make_lqr_backup", "make_rollout", "run_cone_obstacle_ue_sa_training"]
