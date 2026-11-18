"""Phase-I safe-arrival training for landing under a bounded disturbance (UE-bCBF version).

Same objective and loop as ``quadrotor_landing_sa_trainer`` (discounted safe arrival,
TD3 backbone, design-region curriculum), with three changes:

1. Environment = frozen-estimate flow with d_hat in the dynamics; indicators tightened by
   the UE tube (``quadrotor_landing_ue_sa_env``). Critic input: (x, d_hat, tau/T, s);
   actor input: (x, d_hat).
2. Contraction regularizer on the actor (outside B):
       L_c = w_c * mean( relu( log||F(x, d_hat)||_P / dt - c_target ) ),
   where F is the Jacobian of the frozen-estimate closed-loop step in LQR error
   coordinates. The tube grows by ||F||_P per step; the critic sees the tube only as a
   state, so the actor's own Jacobian - which sets that growth - gets its gradient here.
3. Evaluation = first-hit membership in the *tightened* C_N with the exact tube of the
   evaluated actor, for held-out (x0, d_hat) pairs; the nominal C_N (d_hat = 0, no
   tightening) is reported alongside for comparison with the nominal Phase I.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import json
from typing import Any, Dict

import jax
import jax.numpy as jnp
import numpy as np

from ps2rl.backup_policy.backup_policy import BackupPolicy
from ps2rl.envs.quadrotor_landing_config import QuadrotorLandingConfig
from ps2rl.phase1_sa.landing_design_region import REGION_NAMES, LandingDesignRegionConfig, heldout_sets
from ps2rl.phase1_sa.quadrotor_landing_sa_env import landing_action_box
from ps2rl.phase1_sa.quadrotor_landing_ue_sa_env import (
    ACTOR_OBS_DIM,
    build_landing_ue_sa_env,
    build_ue_indicators,
    sample_d_hat,
)
from ps2rl.phase1_sa.sa_critic import SafeArrivalCriticConfig, init_twin_q_params, q_cont, q_full, q_full_from_flags
from ps2rl.phase1_sa.sa_trainer_core import (
    SALoopState,
    SASystemHooks,
    huber_loss,
    run_sa_training_loop,
    sa_replay_init,
    snapshot_sa_state,
    split_env_keys,
    update_curriculum_state,
    zero_sa_chunk_metrics,
)
from ps2rl.uncertainty.landing_ue_tube import LandingUEConfig, make_growth_fn, tube_margins, tube_step
from ps2rl.utils.optim import adam_init, adam_step, soft_update
from ps2rl.utils.policy import ActorConfig, actor_mean_action, init_actor_params
from ps2rl.utils.replay_buffer import jax_replay_add_batch, jax_replay_sample
from ps2rl.utils.seed import make_prng_key

_SAMPLE_FIELDS = ("obs", "act_raw", "next_obs_true", "goal_next", "fail_next")


@dataclass(frozen=True)
class LandingUESAConfig:
    seed: int = 0
    total_steps: int = 5_000_000
    start_steps: int = 5_000
    update_after: int = 2_000
    update_every: int = 8
    gradient_steps: int = 1
    batch_size: int = 256
    replay_size: int = 1_000_000

    beta: float = 0.99
    tau: float = 0.0025
    policy_delay: int = 2
    actor_lr: float = 1e-4
    critic_lr: float = 3e-4
    max_grad_norm: float = 5.0
    critic_huber_delta: float = 1.0
    action_smoothness_weight: float = 0.05

    hidden_size: int = 256
    actor_activation: str = "elu"
    actor_log_std_min: float = -5.0
    actor_log_std_max: float = 0.0

    exploration_std: float = 0.1
    exploration_clip: float = 0.25
    target_policy_noise_std: float = 0.0
    target_policy_noise_clip: float = 0.0

    # contraction regularizer (rate in 1/s, measured in the LQR P-metric)
    contraction_target: float = 1.0
    contraction_weight: float = 0.02

    episode_max_steps: int = 150
    eval_every: int = 100_000
    log_every: int = 20_000
    record_update_metrics: bool = True
    update_metric_every: int = 200
    num_envs: int = 64
    steps_per_jit: int = 64

    curriculum_start_scale: float = 0.0
    curriculum_increment: float = 0.1
    curriculum_success_threshold: float = 0.8
    curriculum_window_episodes: int = 200
    curriculum_min_episodes: int = 400

    use_handoff: bool = True
    collector_terminate_on_goal: bool = True
    hoeffding_delta: float = 0.05
    eval_d_hat_per_state: int = 1  # held-out d_hat draws per held-out x0
    # no actor updates before this many critic updates (0 = TD3 as usual). With an actor initialized
    # from a controller (runway: behaviour-cloned LQR) the critic first learns that actor's value.
    actor_start_update: int = 0
    # weight of an anchor to the base controller outside B: bc_weight * mean |(a - pi_B(x)) / scale|^2
    # (TD3+BC-style; 0 = off, the landing setting). Keeps a controller-initialized actor from being
    # pulled away by an immature critic; the actor still deviates where the critic's gradient is larger.
    bc_weight: float = 0.0

    @property
    def sa_backbone(self) -> str:  # read by the shared loop's log line
        return "td3"

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "LandingUESAConfig":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in payload.items() if k in names})


_EPISODE_METRIC_FIELDS = (
    "episode_len_sum",
    "episode_success_sum",
    "episode_success_within_n_sum",
    "episode_crash_sum",
    "episode_timeout_sum",
    "episode_min_h_sum",
    "episode_s_sum",
)
_EPISODE_HISTORY_PAIRS = (
    ("episode_len", "episode_len_sum"),
    ("episode_success_rate", "episode_success_sum"),
    ("episode_success_within_n_rate", "episode_success_within_n_sum"),
    ("episode_crash_rate", "episode_crash_sum"),
    ("episode_timeout_rate", "episode_timeout_sum"),
    ("episode_min_h", "episode_min_h_sum"),
    ("episode_final_tube", "episode_s_sum"),
)
_EVAL_KEYS = (
    tuple(f"eval_m_{r}" for r in REGION_NAMES)
    + tuple(f"eval_nominal_m_{r}" for r in REGION_NAMES)
    + ("eval_mu_weighted", "eval_nominal_mu_weighted", "eval_crash_rate", "eval_rate_p90")
)
_HISTORY_KEYS = (
    "step",
    "episode_idx",
    "curriculum_scale",
    "critic_loss",
    "actor_loss",
    "action_penalty",
    "target_mean",
    "q_pi_mean",
    "q1_grad_norm",
    "q2_grad_norm",
    "actor_grad_norm",
    "alpha",
    "entropy",
    *[k for k, _ in _EPISODE_HISTORY_PAIRS],
    *_EVAL_KEYS,
)


def _episode_bookkeeping(info: Any, beta_arr: jax.Array):
    done = info.episode_done.astype(jnp.float32)
    fields_ = {
        "episode_len_sum": jnp.sum(info.completed_len),
        "episode_success_sum": jnp.sum(info.completed_success),
        "episode_success_within_n_sum": jnp.sum(info.completed_success_within_n),
        "episode_crash_sum": jnp.sum(info.completed_crash),
        "episode_timeout_sum": jnp.sum(info.completed_timeout),
        "episode_min_h_sum": jnp.sum(info.completed_min_h * done),
        "episode_s_sum": jnp.sum(info.completed_s * done),
    }
    return info.completed_success_within_n, fields_


# ----------------------------------------------------------------------------- policy pieces
def make_composed_backup(cfg: QuadrotorLandingConfig, env_fns: Any, actor_cfg: ActorConfig):
    """(params, x, d_hat) -> pi_b(x, d_hat): LQR on B, actor (tanh-mean) outside."""
    scale, low, high = landing_action_box(cfg)
    base_set = env_fns.base_set

    def pi_b(params, x, d_hat):
        o = jnp.concatenate([x, d_hat])
        raw = actor_mean_action(params, o[None, :], scale, actor_cfg, action_low=low, action_high=high)[0]
        raw = jnp.clip(raw, low, high)
        return jnp.clip(BackupPolicy.select_action(x, raw, base_set), low, high)

    return pi_b


def make_param_growth_fn(cfg: QuadrotorLandingConfig, env_fns: Any, actor_cfg: ActorConfig):
    """(params, x, d_hat) -> ||F||_P of the frozen-estimate closed-loop step of pi_b."""
    pi_b = make_composed_backup(cfg, env_fns, actor_cfg)
    # error chart of the tube metric: the base controller's (landing), or a separate chart when
    # the environment provides one (runway: the metric needs p_y, the base set does not have it)
    ctrl = getattr(env_fns, "metric_chart", None) or env_fns.base_set.controller
    plant = env_fns.plant_step

    def growth(params, x, d_hat):
        step = lambda z: plant(z, pi_b(params, z, d_hat), d_hat)
        return make_growth_fn(step, ctrl, env_fns.tube)(x)

    return growth


# ----------------------------------------------------------------------------- update
def build_ue_update_fn(ra_cfg: LandingUESAConfig, actor_cfg: ActorConfig, cfg: QuadrotorLandingConfig, env_fns: Any):
    action_scale, action_low, action_high = landing_action_box(cfg)
    beta = jnp.asarray(float(ra_cfg.beta), dtype=jnp.float32)
    smooth_w = jnp.asarray(float(ra_cfg.action_smoothness_weight), dtype=jnp.float32)
    policy_delay = int(max(1, ra_cfg.policy_delay))
    dt = float(cfg.dt)
    c_tgt = float(ra_cfg.contraction_target)
    w_c = float(ra_cfg.contraction_weight)
    growth = make_param_growth_fn(cfg, env_fns, actor_cfg)
    goal_fn = jax.vmap(env_fns.goal_fn)
    fail_fn = jax.vmap(env_fns.fail_fn)
    handoff_fn = jax.vmap(lambda o: env_fns.base_set.contains(o[:10]))

    def tree_l2_norm(tree):
        return jnp.sqrt(sum(jnp.sum(jnp.square(x)) for x in jax.tree_util.tree_leaves(tree)) + 1e-12)

    def sanitize_grads(grads):
        grads = jax.tree_util.tree_map(lambda g: jnp.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0), grads)
        gn = tree_l2_norm(grads)
        if ra_cfg.max_grad_norm > 0.0:
            sc = jnp.minimum(1.0, ra_cfg.max_grad_norm / (gn + 1e-6))
            grads = jax.tree_util.tree_map(lambda g: g * sc, grads)
        return grads, gn

    def det_action(params, obs):
        raw = actor_mean_action(params, obs[..., :ACTOR_OBS_DIM], action_scale, actor_cfg,
                                action_low=action_low, action_high=action_high)
        return jnp.clip(jnp.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0), action_low, action_high)

    @jax.jit
    def update(state: Dict[str, Any], batch: Dict[str, jax.Array], key: jax.Array):
        key_next, key_tn = jax.random.split(key, 2)
        target_raw = det_action(state["target_actor_params"], batch["next_obs_true"])
        if ra_cfg.target_policy_noise_std > 0.0:
            n = jax.random.normal(key_tn, target_raw.shape) * (ra_cfg.target_policy_noise_std * action_scale)
            if ra_cfg.target_policy_noise_clip > 0.0:
                cm = ra_cfg.target_policy_noise_clip * action_scale
                n = jnp.clip(n, -cm, cm)
            target_raw = jnp.clip(target_raw + n, action_low, action_high)
        tq1 = q_full_from_flags(q_cont(state["target_q1_params"], batch["next_obs_true"], target_raw),
                                goal=batch["goal_next"], fail=batch["fail_next"])
        tq2 = q_full_from_flags(q_cont(state["target_q2_params"], batch["next_obs_true"], target_raw),
                                goal=batch["goal_next"], fail=batch["fail_next"])
        target_q = jnp.minimum(tq1, tq2)
        g_obs = goal_fn(batch["obs"]).astype(jnp.float32)
        f_obs = fail_fn(batch["obs"]).astype(jnp.float32)
        c_obs = jnp.clip(1.0 - g_obs - f_obs, 0.0, 1.0)
        y = jax.lax.stop_gradient(jnp.clip(g_obs + beta * c_obs * target_q, 0.0, 1.0))

        def critic_loss_fn(q1p, q2p):
            q1 = q_full(q1p, batch["obs"], batch["act_raw"], goal_fn, fail_fn)
            q2 = q_full(q2p, batch["obs"], batch["act_raw"], goal_fn, fail_fn)
            l1 = jnp.mean(huber_loss(q1 - y, ra_cfg.critic_huber_delta))
            l2 = jnp.mean(huber_loss(q2 - y, ra_cfg.critic_huber_delta))
            return l1 + l2, {"target_mean": jnp.mean(y)}

        (critic_loss, caux), (g1, g2) = jax.value_and_grad(critic_loss_fn, argnums=(0, 1), has_aux=True)(
            state["q1_params"], state["q2_params"])
        g1, gn1 = sanitize_grads(g1)
        g2, gn2 = sanitize_grads(g2)
        q1_params, q1_opt = adam_step(state["q1_params"], g1, state["q1_opt"], ra_cfg.critic_lr)
        q2_params, q2_opt = adam_step(state["q2_params"], g2, state["q2_opt"], ra_cfg.critic_lr)

        update_step = state["update_step"] + jnp.int32(1)
        actor_due = (update_step % jnp.int32(policy_delay)) == 0
        actor_frozen = update_step <= jnp.int32(ra_cfg.actor_start_update)  # never true with the default 0
        mask = (1.0 - handoff_fn(batch["obs"]).astype(jnp.float32)) if ra_cfg.use_handoff else jnp.ones(
            (batch["obs"].shape[0],), jnp.float32)
        denom = jnp.maximum(jnp.sum(mask), 1.0)
        xs = batch["obs"][:, :10]
        dh = batch["obs"][:, 10:13]
        if ra_cfg.bc_weight > 0.0:
            a_base = jax.lax.stop_gradient(jax.vmap(env_fns.base_set.controller.action)(xs))

        def actor_loss_fn(params):
            raw = det_action(params, batch["obs"])
            q1 = q_full(q1_params, batch["obs"], raw, goal_fn, fail_fn)
            q2 = q_full(q2_params, batch["obs"], raw, goal_fn, fail_fn)
            q_pi = jnp.sum(mask * jnp.minimum(q1, q2)) / denom
            pen = jnp.sum(mask * jnp.mean(jnp.square(raw / action_scale), axis=-1)) / denom
            loss = -q_pi + smooth_w * pen
            if ra_cfg.bc_weight > 0.0:
                bc = jnp.sum(mask * jnp.mean(jnp.square((raw - a_base) / action_scale), axis=-1)) / denom
                loss = loss + float(ra_cfg.bc_weight) * bc
            if w_c > 0.0:
                g = jax.vmap(growth, in_axes=(None, 0, 0))(params, xs, dh)
                rate = jnp.log(jnp.maximum(g, 1e-6)) / dt
                c_pen = jnp.sum(mask * jax.nn.relu(rate - c_tgt)) / denom
                loss = loss + w_c * c_pen
            else:
                rate = jnp.zeros_like(mask)
                c_pen = jnp.asarray(0.0)
            return loss, {"q_pi": q_pi, "pen": pen, "c_pen": c_pen,
                          "rate_mean": jnp.sum(mask * rate) / denom}

        def do_actor(_):
            (al, aux), ag = jax.value_and_grad(actor_loss_fn, has_aux=True)(state["actor_params"])
            ag, agn = sanitize_grads(ag)
            ap, ao = adam_step(state["actor_params"], ag, state["actor_opt"], ra_cfg.actor_lr)
            if ra_cfg.actor_start_update > 0:  # critic-only phase: keep the actor, still move the targets
                keep = lambda new, old: jax.tree_util.tree_map(lambda n, o: jnp.where(actor_frozen, o, n), new, old)
                ap, ao = keep(ap, state["actor_params"]), keep(ao, state["actor_opt"])
            return (ap, ao,
                    soft_update(state["target_actor_params"], ap, ra_cfg.tau),
                    soft_update(state["target_q1_params"], q1_params, ra_cfg.tau),
                    soft_update(state["target_q2_params"], q2_params, ra_cfg.tau),
                    al, aux["q_pi"], aux["pen"], agn, jnp.asarray(1.0), aux["c_pen"], aux["rate_mean"])

        def skip_actor(_):
            z = jnp.asarray(0.0)
            return (state["actor_params"], state["actor_opt"], state["target_actor_params"],
                    state["target_q1_params"], state["target_q2_params"], z, z, z, z, z, z, z)

        (ap, ao, tap, tq1p, tq2p, al, qpi, pen, agn, applied, c_pen, rate_mean) = jax.lax.cond(
            actor_due, do_actor, skip_actor, operand=None)
        new_state = {
            "actor_params": ap, "target_actor_params": tap,
            "q1_params": q1_params, "q2_params": q2_params,
            "target_q1_params": tq1p, "target_q2_params": tq2p,
            "actor_opt": ao, "q1_opt": q1_opt, "q2_opt": q2_opt, "update_step": update_step,
        }
        metrics = {
            "critic_loss": critic_loss, "actor_loss": al, "action_penalty": pen,
            "target_mean": caux["target_mean"], "q_pi_mean": qpi, "q1_grad_norm": gn1, "q2_grad_norm": gn2,
            "actor_grad_norm": agn, "actor_update_applied": applied,
            # the shared loop logs these two slots as alpha / entropy; here they carry the
            # contraction penalty and the mean P-metric growth rate [1/s] outside B
            "alpha": c_pen, "entropy": rate_mean,
        }
        return new_state, metrics, key_next

    return update


# ----------------------------------------------------------------------------- collection
def build_ue_one_vec_step(*, env_fns, update_fn, ra_cfg: LandingUESAConfig, actor_cfg: ActorConfig,
                          cfg: QuadrotorLandingConfig, hooks: SASystemHooks):
    action_scale, action_low, action_high = landing_action_box(cfg)
    growth = make_param_growth_fn(cfg, env_fns, actor_cfg)
    growth_b = jax.vmap(growth, in_axes=(None, 0, 0))
    num_envs = int(ra_cfg.num_envs)
    bs = jnp.int32(ra_cfg.batch_size)
    ua = jnp.int32(ra_cfg.update_after)
    ue_ = jnp.int32(max(1, ra_cfg.update_every))
    max_due = max(1, int(np.ceil(num_envs / max(1, ra_cfg.update_every))))
    beta_arr = jnp.asarray(float(ra_cfg.beta), dtype=jnp.float32)

    def collect(params, obs_b, key):
        k_r, k_n = jax.random.split(key)
        rnd = action_low + jax.random.uniform(k_r, obs_b.shape[:-1] + action_low.shape) * (action_high - action_low)
        raw = actor_mean_action(params, obs_b[..., :ACTOR_OBS_DIM], action_scale, actor_cfg,
                                action_low=action_low, action_high=action_high)
        raw = jnp.clip(jnp.nan_to_num(raw), action_low, action_high)
        noise = jax.random.normal(k_n, raw.shape) * (ra_cfg.exploration_std * action_scale)
        if ra_cfg.exploration_clip > 0.0:
            cm = ra_cfg.exploration_clip * action_scale
            noise = jnp.clip(noise, -cm, cm)
        return jnp.clip(raw + noise, action_low, action_high), rnd

    def one_vec_step(carry: SALoopState, _):
        key, k_c = jax.random.split(carry.key)
        noisy, rnd = collect(carry.state["actor_params"], carry.obs, k_c)
        act = jnp.where(carry.global_step < jnp.int32(ra_cfg.start_steps), rnd, noisy)
        g = growth_b(carry.state["actor_params"], carry.env_state.x, carry.env_state.d_hat)
        g = jnp.nan_to_num(g, nan=10.0, posinf=10.0)
        env_keys, step_keys = split_env_keys(carry.env_keys)
        env_state, obs_true, obs_out, done, info = env_fns.step_batched(carry.env_state, act, g, step_keys,
                                                                         carry.curriculum_scale)
        replay = jax_replay_add_batch(carry.replay, {
            "obs": carry.obs.astype(jnp.float32),
            "act_raw": act.astype(jnp.float32),
            "next_obs_true": obs_true.astype(jnp.float32),
            "goal_next": info.goal_next.astype(jnp.float32),
            "fail_next": info.fail_next.astype(jnp.float32),
            "done_rollout": done.astype(jnp.float32),
            "act_applied": info.applied_action.astype(jnp.float32),
        })
        prev = carry.global_step
        gstep = prev + jnp.int32(num_envs)
        lo = jnp.maximum(prev + 1, ua)
        due = jnp.where(gstep >= lo, gstep // ue_ - (lo - 1) // ue_, 0)
        due = jnp.minimum(jnp.where(replay.size >= bs, due, 0), max_due)

        def do_due(_, c):
            st, rp, k, upd, m = c

            def one(_, cc):
                st2, rp2, k2, upd2, m2 = cc
                k2, ks, ku = jax.random.split(k2, 3)
                batch = jax_replay_sample(rp2, ra_cfg.batch_size, ks, _SAMPLE_FIELDS)
                st2, um, _ = update_fn(st2, batch, ku)
                m2 = dict(m2)
                m2["update_count"] += 1.0
                m2["actor_update_count"] += um["actor_update_applied"]
                for k_sum, k_m in (("critic_loss_sum", "critic_loss"), ("actor_loss_sum", "actor_loss"),
                                   ("action_penalty_sum", "action_penalty"), ("target_mean_sum", "target_mean"),
                                   ("q_pi_mean_sum", "q_pi_mean"), ("q1_grad_norm_sum", "q1_grad_norm"),
                                   ("q2_grad_norm_sum", "q2_grad_norm"), ("actor_grad_norm_sum", "actor_grad_norm"),
                                   ("alpha_sum", "alpha"), ("entropy_sum", "entropy")):
                    m2[k_sum] = m2[k_sum] + um[k_m]
                return st2, rp2, k2, upd2 + 1, m2

            return jax.lax.fori_loop(0, ra_cfg.gradient_steps, one, c)

        state, replay, key, updates, m = jax.lax.fori_loop(
            0, due, do_due, (carry.state, replay, key, carry.updates, zero_sa_chunk_metrics(hooks.episode_metric_fields)))
        succ, ep_fields = hooks.episode_bookkeeping(info, beta_arr)
        cs, ec, sw, sws, swp = update_curriculum_state(carry.curriculum_scale, carry.episode_count, carry.success_window,
                                                       carry.success_window_size, carry.success_window_ptr, succ,
                                                       info.episode_done, ra_cfg)
        m = dict(m)
        m["episode_count"] = jnp.sum(info.episode_done.astype(jnp.float32))
        for f in hooks.episode_metric_fields:
            m[f] = ep_fields[f]
        return SALoopState(state=state, replay=replay, env_state=env_state, obs=obs_out, key=key, env_keys=env_keys,
                           global_step=gstep, updates=updates, curriculum_scale=cs, episode_count=ec,
                           success_window=sw, success_window_size=sws, success_window_ptr=swp), m

    return one_vec_step


# ----------------------------------------------------------------------------- evaluation
def build_ue_evaluator(cfg: QuadrotorLandingConfig, region_cfg: LandingDesignRegionConfig, ue: LandingUEConfig,
                       env_fns: Any, actor_cfg: ActorConfig, *, beta: float, hoeffding_delta: float,
                       d_hat_per_state: int = 1):
    """run_eval(params, split): tightened first-hit C_N membership with the exact tube."""
    pi_b = make_composed_backup(cfg, env_fns, actor_cfg)
    growth = make_param_growth_fn(cfg, env_fns, actor_cfg)
    plant = env_fns.plant_step
    cone, base_set, tc = env_fns.safe_set, env_fns.base_set, env_fns.tube
    not_failed, goal = build_ue_indicators(cone, base_set, ue, tc)
    n_steps = int(cfg.num_steps)
    dt = float(cfg.dt)
    sets = {split: heldout_sets(region_cfg, env_fns.sampler, split=split) for split in ("val", "test")}
    dh_sets = {}
    for split, off in (("val", 0), ("test", 1)):
        dh_sets[split] = {}
        for i, name in enumerate(REGION_NAMES):
            n = sets[split][name].shape[0] * int(d_hat_per_state)
            keys = jax.random.split(jax.random.PRNGKey(4242 + 100 * off + i), n)
            dh_sets[split][name] = np.asarray(jax.vmap(lambda k: sample_d_hat(k, ue))(keys))

    def rollout(params, x0, d_hat, tight):
        def body(c, k):
            x, s, hb, hf, rmax = c
            g = growth(params, x, d_hat)
            xn = plant(x, pi_b(params, x, d_hat), d_hat)
            sn = jnp.where(tight, tube_step(s, g, k * dt, ue, tc, dt), 0.0)
            act = ~(hb | hf)
            nf = act & ~not_failed(xn, sn)
            nb = act & ~nf & goal(xn, sn)
            rate = jnp.where(base_set.contains(x), -jnp.inf, jnp.log(jnp.maximum(g, 1e-6)) / dt)
            return (jnp.where(act, xn, x), jnp.where(act, sn, s), hb | nb, hf | nf, jnp.maximum(rmax, rate)), nb

        z = jnp.asarray(0.0)
        (_, s_end, hb, hf, rmax), hits = jax.lax.scan(
            body, (x0, z, base_set.contains(x0), ~not_failed(x0, z), -jnp.inf), jnp.arange(n_steps))
        t_hit = jnp.where(hb, jnp.argmax(hits) + 1, -1)
        return hb, hf, t_hit, s_end, rmax

    rb = jax.jit(jax.vmap(rollout, in_axes=(None, 0, 0, None)))

    def run_eval(params, split: str) -> Dict[str, Any]:
        stats: Dict[str, Any] = {"split": split}
        num = den = num_nom = 0.0
        crash, rates = [], []
        w = region_cfg.weights
        for name in REGION_NAMES:
            x0 = np.repeat(sets[split][name], int(d_hat_per_state), axis=0)
            dh = dh_sets[split][name]
            hb, hf, th, s_end, rmax = map(np.asarray, rb(params, jnp.asarray(x0), jnp.asarray(dh), True))
            hb0, _, _, _, _ = map(np.asarray, rb(params, jnp.asarray(x0), jnp.zeros_like(jnp.asarray(dh)), False))
            m = int(x0.shape[0])
            mh = float(hb.mean()) if m else float("nan")
            arr = th[hb]
            stats[name] = {
                "n": m, "m_hat": mh, "nominal_m_hat": float(hb0.mean()) if m else float("nan"),
                "hoeffding_halfwidth": float(np.sqrt(np.log(2.0 / hoeffding_delta) / (2.0 * max(m, 1)))),
                "crash_rate": float(hf.mean()) if m else float("nan"),
                "timeout_rate": float((~hb & ~hf).mean()) if m else float("nan"),
                "mean_arrival_steps": float(arr.mean()) if arr.size else float("nan"),
                "tube_at_arrival_median": float(np.median(s_end[hb])) if hb.any() else float("nan"),
            }
            num += w[name] * mh
            num_nom += w[name] * stats[name]["nominal_m_hat"]
            den += w[name]
            crash.append(hf)
            rr = rmax[np.isfinite(rmax)]
            if rr.size:
                rates.append(rr)
        stats["mu_weighted"] = num / max(den, 1e-12)
        stats["nominal_mu_weighted"] = num_nom / max(den, 1e-12)
        stats["crash_rate"] = float(np.concatenate(crash).mean())
        allr = np.concatenate(rates) if rates else np.array([np.nan])
        stats["rate_p90"] = float(np.percentile(allr, 90))
        stats["rate_p50"] = float(np.percentile(allr, 50))
        return stats

    return run_eval


def _eval_rank_key(stats):
    return (float(stats["mu_weighted"]), float(stats["edge"]["m_hat"]), float(stats["general"]["m_hat"]))


def _append_eval_history(history, stats):
    for name in REGION_NAMES:
        history[f"eval_m_{name}"].append(float(stats[name]["m_hat"]))
        history[f"eval_nominal_m_{name}"].append(float(stats[name]["nominal_m_hat"]))
    history["eval_mu_weighted"].append(float(stats["mu_weighted"]))
    history["eval_nominal_mu_weighted"].append(float(stats["nominal_mu_weighted"]))
    history["eval_crash_rate"].append(float(stats["crash_rate"]))
    history["eval_rate_p90"].append(float(stats["rate_p90"]))
    print(
        f"  [eval {stats['split']}] UE mu_w={stats['mu_weighted']:.3f} "
        + " ".join(f"m_{n}={stats[n]['m_hat']:.3f}" for n in REGION_NAMES)
        + f" | nominal mu_w={stats['nominal_mu_weighted']:.3f} "
        + " ".join(f"{n}={stats[n]['nominal_m_hat']:.3f}" for n in REGION_NAMES)
        + f" | crash={stats['crash_rate']:.3f} max-rate p50/p90={stats['rate_p50']:.1f}/{stats['rate_p90']:.1f} 1/s",
        flush=True,
    )


def run_landing_ue_sa_training(ra_cfg: LandingUESAConfig, cfg: QuadrotorLandingConfig,
                               region_cfg: LandingDesignRegionConfig, ue: LandingUEConfig, *,
                               output_dir: str | None = None) -> Dict[str, Any]:
    env_fns = build_landing_ue_sa_env(cfg, region_cfg, ue, episode_max_steps=int(ra_cfg.episode_max_steps),
                                      terminate_on_goal=bool(ra_cfg.collector_terminate_on_goal))
    actor_cfg = ActorConfig(obs_dim=env_fns.actor_obs_dim, action_dim=4,
                            hidden_sizes=(ra_cfg.hidden_size, ra_cfg.hidden_size),
                            log_std_min=ra_cfg.actor_log_std_min, log_std_max=ra_cfg.actor_log_std_max,
                            activation=ra_cfg.actor_activation)
    critic_cfg = SafeArrivalCriticConfig(obs_dim=env_fns.obs_dim, act_dim=4,
                                         hidden_sizes=(ra_cfg.hidden_size, ra_cfg.hidden_size))
    key = make_prng_key(ra_cfg.seed)
    key, k_a, k_c, k_env = jax.random.split(key, 4)
    actor_params = init_actor_params(k_a, actor_cfg)
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
    run_eval = build_ue_evaluator(cfg, region_cfg, ue, env_fns, actor_cfg, beta=float(ra_cfg.beta),
                                  hoeffding_delta=float(ra_cfg.hoeffding_delta),
                                  d_hat_per_state=int(ra_cfg.eval_d_hat_per_state))
    hooks = SASystemHooks(episode_metric_fields=_EPISODE_METRIC_FIELDS, episode_bookkeeping=_episode_bookkeeping,
                          run_eval=run_eval, eval_rank_key=_eval_rank_key, append_eval_history=_append_eval_history,
                          episode_history_pairs=_EPISODE_HISTORY_PAIRS, history_keys=_HISTORY_KEYS)
    initial_eval = run_eval(state["actor_params"], "val")
    one_vec_step = build_ue_one_vec_step(env_fns=env_fns, update_fn=update_fn, ra_cfg=ra_cfg, actor_cfg=actor_cfg,
                                         cfg=cfg, hooks=hooks)
    res = run_sa_training_loop(ra_cfg=ra_cfg, loop_state=loop_state, one_vec_step=one_vec_step, hooks=hooks)
    best_eval = res.best_eval_stats if res.best_eval_stats is not None else res.val_eval
    best_state = res.best_state if res.best_state is not None else snapshot_sa_state(res.loop_state.state)
    test_at_best = run_eval(best_state["actor_params"], "test")
    summary = {
        "training_objective": "discounted_safe_arrival_ue_tightened",
        "task": "approach_cone_landing_under_disturbance",
        "sa_backbone": "td3",
        "seed": int(ra_cfg.seed),
        "total_steps": int(ra_cfg.total_steps),
        "updates": int(jax.device_get(res.loop_state.updates)),
        "wall_time_sec": float(res.total_time),
        "jax_backend": jax.default_backend(),
        "final_curriculum_scale": float(jax.device_get(res.loop_state.curriculum_scale)),
        "tube_constants": env_fns.tube.as_dict(),
        "untrained_val": initial_eval,
        "final_val": res.val_eval,
        "best_val": best_eval,
        "best_eval_step": int(res.best_eval_step),
        "test_at_best": test_at_best,
        "test_at_final": res.test_eval,
    }
    configs = {"landing": cfg.as_dict(), "design_region": region_cfg.as_dict(), "ue": ue.as_dict(),
               "backup_ra": asdict(ra_cfg), "actor": asdict(actor_cfg)}
    if output_dir is not None:
        with open(f"{output_dir}/summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=float)
        with open(f"{output_dir}/configs.json", "w", encoding="utf-8") as f:
            json.dump(configs, f, indent=2)
    return {"summary": summary, "history": res.history, "configs": configs,
            "final_state": snapshot_sa_state(res.loop_state.state), "best_state": best_state, "actor_cfg": actor_cfg,
            "tube": env_fns.tube}


__all__ = [
    "LandingUESAConfig",
    "build_ue_evaluator",
    "build_ue_update_fn",
    "make_composed_backup",
    "make_param_growth_fn",
    "run_landing_ue_sa_training",
]
