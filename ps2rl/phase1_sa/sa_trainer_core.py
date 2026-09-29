"""System-agnostic core of the Phase-1 safe-arrival trainer.

The unicycle, quadrotor and landing Phase-1 trainers share this backbone. Two actor
updates are available through ``ra_cfg.sa_backbone`` (default ``"td3"``):

* ``"td3"`` - deterministic actor, fixed Gaussian exploration noise (paper setup).
* ``"sac"`` - tanh-Gaussian actor with an entropy bonus **in the actor loss only**.

Why the entropy stays out of the critic target: the safe-arrival value has the
indicator fixed point Q*(x) = beta^{d(x)} in [0, 1] (Prop. D.7). A soft Bellman
backup adds -alpha log pi to every continuation step, i.e. a per-step reward. With
positive entropy, a state that never reaches B collects ~ alpha H / (1 - beta) and can
outrank a late arrival; with negative entropy, failure (value 0, episode over) can
outrank a long safe arrival. Either way the ordering "arrive sooner > arrive later >
never arrive" that Phase I optimizes is lost. So the critic keeps the hard backup
    y = b(x) + beta c(x) min_i Q_i'(x', mu'(x')),
evaluated at the *deterministic mean* mu' of the target actor: that mean is the policy
pi_SA that pi_b deploys, so Q estimates exactly the quantity defining C_N(pi_b). The
actor maximizes E_{a ~ pi}[min_i Q_i(x, a)] + alpha H(pi(.|x)) with auto-tuned alpha;
the benefit over TD3 is state-dependent exploration (the policy narrows its noise
where Q is sharply curved, e.g. next to the cone boundary). Entropy is measured on
the normalized action in [-1, 1]^m so the target entropy is unit-free.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Callable, Dict, List, NamedTuple, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from ps2rl.utils.policy import (
    ActorConfig,
    actor_mean_action,
    init_actor_params,
    sample_actor_action,
)
from ps2rl.phase1_sa.sa_critic import (
    SafeArrivalCriticConfig,
    init_twin_q_params,
    q_cont,
    q_full,
    q_full_from_flags,
)
from ps2rl.utils.optim import adam_init, adam_step, soft_update
from ps2rl.utils.replay_buffer import (
    JaxReplayState,
    jax_replay_add_batch,
    jax_replay_init,
    jax_replay_sample,
)

SA_REPLAY_SAMPLE_FIELDS: Tuple[str, ...] = ("obs", "act_raw", "next_obs_true", "goal_next", "fail_next")

SA_UPDATE_METRIC_SUMS: Tuple[str, ...] = (
    "update_count",
    "actor_update_count",
    "critic_loss_sum",
    "actor_loss_sum",
    "action_penalty_sum",
    "target_mean_sum",
    "q_pi_mean_sum",
    "q1_grad_norm_sum",
    "q2_grad_norm_sum",
    "actor_grad_norm_sum",
    "alpha_sum",
    "entropy_sum",
)

SA_BACKBONES: Tuple[str, ...] = ("td3", "sac")


def sa_backbone(ra_cfg: Any) -> str:
    """Selected actor backbone (``td3`` unless ``ra_cfg.sa_backbone`` says otherwise)."""
    name = str(getattr(ra_cfg, "sa_backbone", "td3")).lower()
    if name not in SA_BACKBONES:
        raise ValueError(f"sa_backbone must be one of {SA_BACKBONES}, got '{name}'")
    return name


def sa_replay_init(capacity: int, obs_dim: int, act_dim: int) -> JaxReplayState:
    return jax_replay_init(
        capacity,
        {
            "obs": (obs_dim,),
            "act_raw": (act_dim,),
            "next_obs_true": (obs_dim,),
            "goal_next": (),
            "fail_next": (),
            "done_rollout": (),
            "act_applied": (act_dim,),
        },
    )


def summary_stats(x: np.ndarray) -> Dict[str, Any]:
    arr = np.asarray(x, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p95": None,
            "min": None,
            "max": None,
        }
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95.0)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def rate_mean(values: np.ndarray | List[float]) -> float:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return 0.0
    return float(np.mean(arr))


def snapshot_sa_state(state: Dict[str, Any]) -> Dict[str, Any]:
    snap = {
        "actor_params": state["actor_params"],
        "target_actor_params": state["target_actor_params"],
        "q1_params": state["q1_params"],
        "q2_params": state["q2_params"],
        "target_q1_params": state["target_q1_params"],
        "target_q2_params": state["target_q2_params"],
        "update_step": state["update_step"],
    }
    if "log_alpha" in state:
        snap["log_alpha"] = state["log_alpha"]
    return jax.device_get(snap)


def split_env_keys(env_keys: jax.Array) -> Tuple[jax.Array, jax.Array]:
    split = jax.vmap(lambda k: jax.random.split(k, 2))(env_keys)
    return split[:, 0], split[:, 1]


def huber_loss(error: jax.Array, delta: float) -> jax.Array:
    abs_err = jnp.abs(error)
    delta_arr = jnp.asarray(delta, dtype=error.dtype)
    quadratic = jnp.minimum(abs_err, delta_arr)
    linear = abs_err - quadratic
    return 0.5 * quadratic * quadratic + delta_arr * linear


def init_sa_state(
    key: jax.Array,
    actor_cfg: ActorConfig,
    critic_cfg: SafeArrivalCriticConfig,
    ra_cfg: Any = None,
) -> Dict[str, Any]:
    """Initial learner state. Passing ``ra_cfg`` with ``sa_backbone="sac"`` adds log-alpha."""
    k_actor, k_critic = jax.random.split(key, 2)
    actor_params = init_actor_params(k_actor, actor_cfg)
    q1_params, q2_params = init_twin_q_params(k_critic, critic_cfg)
    state = {
        "actor_params": actor_params,
        "target_actor_params": actor_params,
        "q1_params": q1_params,
        "q2_params": q2_params,
        "target_q1_params": q1_params,
        "target_q2_params": q2_params,
        "actor_opt": adam_init(actor_params),
        "q1_opt": adam_init(q1_params),
        "q2_opt": adam_init(q2_params),
        "update_step": jnp.int32(0),
    }
    if ra_cfg is not None and sa_backbone(ra_cfg) == "sac":
        log_alpha = jnp.asarray(np.log(float(getattr(ra_cfg, "sac_alpha_init", 0.01))), dtype=jnp.float32)
        state["log_alpha"] = log_alpha
        state["alpha_opt"] = adam_init(log_alpha)
    return state


def batch_safe_contains(contains_fn: Callable[[jax.Array], jax.Array]) -> Callable[[jax.Array], jax.Array]:
    def wrapped(x: jax.Array) -> jax.Array:
        x_arr = jnp.asarray(x)
        if x_arr.ndim == 1:
            return contains_fn(x_arr)
        return jax.vmap(contains_fn)(x_arr)

    return wrapped


def build_sa_update_fn(
    ra_cfg: Any,
    actor_cfg: ActorConfig,
    action_scale: jax.Array,
    action_low: jax.Array,
    action_high: jax.Array,
    goal_contains_fn: Callable[[jax.Array], jax.Array],
    fail_contains_fn: Callable[[jax.Array], jax.Array],
    handoff_contains_fn: Callable[[jax.Array], jax.Array],
):
    beta = jnp.asarray(float(ra_cfg.beta), dtype=jnp.float32)
    action_smoothness_weight = jnp.asarray(float(ra_cfg.action_smoothness_weight), dtype=jnp.float32)
    policy_delay = int(max(1, ra_cfg.policy_delay))
    backbone = sa_backbone(ra_cfg)
    use_sac = backbone == "sac"
    # log|d a / d a_normalized|: converts log pi(a) into the normalized [-1, 1]^m density.
    log_half_range_sum = jnp.sum(jnp.log(0.5 * (jnp.asarray(action_high) - jnp.asarray(action_low))))
    target_entropy = jnp.asarray(
        float(getattr(ra_cfg, "sac_target_entropy", -float(actor_cfg.action_dim))), dtype=jnp.float32
    )
    learn_alpha = bool(getattr(ra_cfg, "sac_learn_alpha", True))
    alpha_lr = float(getattr(ra_cfg, "sac_alpha_lr", 3e-4))
    log_alpha_min = float(np.log(float(getattr(ra_cfg, "sac_alpha_min", 1e-4))))
    log_alpha_max = float(np.log(float(getattr(ra_cfg, "sac_alpha_max", 0.05))))

    def tree_l2_norm(tree) -> jax.Array:
        leaves = jax.tree_util.tree_leaves(tree)
        sq = jnp.asarray(0.0, dtype=jnp.float32)
        for x in leaves:
            sq = sq + jnp.sum(jnp.square(x))
        return jnp.sqrt(sq + 1e-12)

    def sanitize_grads(grads):
        grads = jax.tree_util.tree_map(lambda g: jnp.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0), grads)
        grad_norm = tree_l2_norm(grads)
        if ra_cfg.max_grad_norm > 0.0:
            scale = jnp.minimum(1.0, ra_cfg.max_grad_norm / (grad_norm + 1e-6))
            grads = jax.tree_util.tree_map(lambda g: g * scale, grads)
        return grads, grad_norm

    def deterministic_action(params, obs: jax.Array) -> jax.Array:
        raw = actor_mean_action(
            params,
            obs,
            action_scale,
            actor_cfg,
            action_low=action_low,
            action_high=action_high,
        )
        return jnp.clip(jnp.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0), action_low, action_high)

    @jax.jit
    def update(state: Dict[str, Any], batch: Dict[str, jax.Array], key: jax.Array):
        key_next, key_target_noise = jax.random.split(key, 2)
        # Derived separately so the TD3 key stream is exactly the original one.
        key_actor = jax.random.fold_in(key, 7)

        target_raw = deterministic_action(state["target_actor_params"], batch["next_obs_true"])
        if ra_cfg.target_policy_noise_std > 0.0:
            target_noise = jax.random.normal(key_target_noise, shape=target_raw.shape, dtype=target_raw.dtype)
            target_noise = target_noise * (jnp.asarray(ra_cfg.target_policy_noise_std, dtype=target_raw.dtype) * action_scale)
            if ra_cfg.target_policy_noise_clip > 0.0:
                clip_mag = jnp.asarray(ra_cfg.target_policy_noise_clip, dtype=target_raw.dtype) * action_scale
                target_noise = jnp.clip(target_noise, -clip_mag, clip_mag)
            target_raw = jnp.clip(target_raw + target_noise, action_low, action_high)

        # The safe-arrival target must use the true pre-reset successor from the
        # collector, never the auto-reset observation used to continue rollouts.
        target_q1 = q_full_from_flags(
            q_cont(state["target_q1_params"], batch["next_obs_true"], target_raw),
            goal=batch["goal_next"],
            fail=batch["fail_next"],
        )
        target_q2 = q_full_from_flags(
            q_cont(state["target_q2_params"], batch["next_obs_true"], target_raw),
            goal=batch["goal_next"],
            fail=batch["fail_next"],
        )
        target_q = jnp.minimum(target_q1, target_q2)

        g_obs = goal_contains_fn(batch["obs"]).astype(jnp.float32)
        f_obs = fail_contains_fn(batch["obs"]).astype(jnp.float32)
        c_obs = jnp.clip(1.0 - g_obs - f_obs, 0.0, 1.0)
        y = jnp.clip(g_obs + beta * c_obs * target_q, 0.0, 1.0)
        y_stop = jax.lax.stop_gradient(y)

        def critic_loss_fn(q1_params, q2_params):
            q1_pred = q_full(q1_params, batch["obs"], batch["act_raw"], goal_contains_fn, fail_contains_fn)
            q2_pred = q_full(q2_params, batch["obs"], batch["act_raw"], goal_contains_fn, fail_contains_fn)
            q1_loss = jnp.mean(huber_loss(q1_pred - y_stop, ra_cfg.critic_huber_delta))
            q2_loss = jnp.mean(huber_loss(q2_pred - y_stop, ra_cfg.critic_huber_delta))
            aux = {
                "q1_pred_mean": jnp.mean(q1_pred),
                "q2_pred_mean": jnp.mean(q2_pred),
                "target_mean": jnp.mean(y_stop),
            }
            return q1_loss + q2_loss, aux

        (critic_loss, critic_aux), (q1_grads, q2_grads) = jax.value_and_grad(
            critic_loss_fn,
            argnums=(0, 1),
            has_aux=True,
        )(state["q1_params"], state["q2_params"])
        q1_grads, q1_grad_norm = sanitize_grads(q1_grads)
        q2_grads, q2_grad_norm = sanitize_grads(q2_grads)
        q1_params, q1_opt = adam_step(state["q1_params"], q1_grads, state["q1_opt"], ra_cfg.critic_lr)
        q2_params, q2_opt = adam_step(state["q2_params"], q2_grads, state["q2_opt"], ra_cfg.critic_lr)

        update_step = state["update_step"] + jnp.int32(1)
        actor_due = (update_step % jnp.int32(policy_delay)) == jnp.int32(0)

        def handoff_mask(q_like: jax.Array) -> jax.Array:
            if ra_cfg.use_handoff:
                return 1.0 - handoff_contains_fn(batch["obs"]).astype(jnp.float32)
            return jnp.ones_like(q_like, dtype=jnp.float32)

        def td3_actor_loss_fn(actor_params, _log_alpha):
            raw = deterministic_action(actor_params, batch["obs"])
            q1_pi = q_full(q1_params, batch["obs"], raw, goal_contains_fn, fail_contains_fn)
            q2_pi = q_full(q2_params, batch["obs"], raw, goal_contains_fn, fail_contains_fn)
            q_pi = jnp.minimum(q1_pi, q2_pi)
            mask = handoff_mask(q_pi)
            denom = jnp.maximum(jnp.sum(mask), 1.0)
            q_masked_mean = jnp.sum(mask * q_pi) / denom
            raw_normed = raw / action_scale
            act_penalty = jnp.mean(jnp.square(raw_normed), axis=-1)
            act_penalty_mean = jnp.sum(mask * act_penalty) / denom
            loss = -q_masked_mean + action_smoothness_weight * act_penalty_mean
            return loss, {
                "q_pi_mean": q_masked_mean,
                "action_penalty": act_penalty_mean,
                "mask_mean": jnp.mean(mask),
                "logp_mean": jnp.asarray(0.0, dtype=jnp.float32),
            }

        def sac_actor_loss_fn(actor_params, log_alpha):
            act, logp, mean_act = sample_actor_action(
                actor_params,
                batch["obs"],
                key_actor,
                action_scale,
                actor_cfg,
                action_low=action_low,
                action_high=action_high,
            )
            logp_n = logp + log_half_range_sum
            q1_pi = q_full(q1_params, batch["obs"], act, goal_contains_fn, fail_contains_fn)
            q2_pi = q_full(q2_params, batch["obs"], act, goal_contains_fn, fail_contains_fn)
            q_pi = jnp.minimum(q1_pi, q2_pi)
            mask = handoff_mask(q_pi)
            denom = jnp.maximum(jnp.sum(mask), 1.0)
            q_masked_mean = jnp.sum(mask * q_pi) / denom
            logp_mean = jnp.sum(mask * logp_n) / denom
            alpha = jax.lax.stop_gradient(jnp.exp(log_alpha))
            act_penalty = jnp.mean(jnp.square(mean_act / action_scale), axis=-1)
            act_penalty_mean = jnp.sum(mask * act_penalty) / denom
            loss = alpha * logp_mean - q_masked_mean + action_smoothness_weight * act_penalty_mean
            return loss, {
                "q_pi_mean": q_masked_mean,
                "action_penalty": act_penalty_mean,
                "mask_mean": jnp.mean(mask),
                "logp_mean": logp_mean,
            }

        actor_loss_fn = sac_actor_loss_fn if use_sac else td3_actor_loss_fn

        def do_actor_update(_):
            log_alpha = state["log_alpha"] if use_sac else jnp.asarray(0.0, dtype=jnp.float32)
            (actor_loss, actor_aux), actor_grads = jax.value_and_grad(actor_loss_fn, has_aux=True)(
                state["actor_params"], log_alpha
            )
            actor_grads, actor_grad_norm = sanitize_grads(actor_grads)
            actor_params, actor_opt = adam_step(state["actor_params"], actor_grads, state["actor_opt"], ra_cfg.actor_lr)
            if use_sac and learn_alpha:
                # J(alpha) = -log_alpha * (log pi + H_target): alpha grows when entropy < target.
                ent_gap = jax.lax.stop_gradient(actor_aux["logp_mean"] + target_entropy)
                alpha_grad = -ent_gap
                log_alpha_new, alpha_opt = adam_step(log_alpha, alpha_grad, state["alpha_opt"], alpha_lr)
                log_alpha_new = jnp.clip(log_alpha_new, log_alpha_min, log_alpha_max)
            elif use_sac:
                log_alpha_new, alpha_opt = log_alpha, state["alpha_opt"]
            else:
                log_alpha_new, alpha_opt = log_alpha, None
            target_actor_params = soft_update(state["target_actor_params"], actor_params, ra_cfg.tau)
            target_q1_params = soft_update(state["target_q1_params"], q1_params, ra_cfg.tau)
            target_q2_params = soft_update(state["target_q2_params"], q2_params, ra_cfg.tau)
            return (
                actor_params,
                actor_opt,
                target_actor_params,
                target_q1_params,
                target_q2_params,
                actor_loss,
                actor_aux["q_pi_mean"],
                actor_aux["action_penalty"],
                actor_aux["mask_mean"],
                actor_grad_norm,
                jnp.asarray(1.0, dtype=jnp.float32),
                log_alpha_new,
                alpha_opt,
                -actor_aux["logp_mean"],
            )

        def skip_actor_update(_):
            zero = jnp.asarray(0.0, dtype=jnp.float32)
            return (
                state["actor_params"],
                state["actor_opt"],
                state["target_actor_params"],
                state["target_q1_params"],
                state["target_q2_params"],
                zero,
                zero,
                zero,
                zero,
                zero,
                zero,
                state["log_alpha"] if use_sac else zero,
                state["alpha_opt"] if use_sac else None,
                zero,
            )

        (
            actor_params,
            actor_opt,
            target_actor_params,
            target_q1_params,
            target_q2_params,
            actor_loss,
            q_pi_mean,
            action_penalty,
            actor_mask_mean,
            actor_grad_norm,
            actor_update_applied,
            log_alpha_out,
            alpha_opt_out,
            entropy_est,
        ) = jax.lax.cond(actor_due, do_actor_update, skip_actor_update, operand=None)

        new_state = {
            "actor_params": actor_params,
            "target_actor_params": target_actor_params,
            "q1_params": q1_params,
            "q2_params": q2_params,
            "target_q1_params": target_q1_params,
            "target_q2_params": target_q2_params,
            "actor_opt": actor_opt,
            "q1_opt": q1_opt,
            "q2_opt": q2_opt,
            "update_step": update_step,
        }
        if use_sac:
            new_state["log_alpha"] = log_alpha_out
            new_state["alpha_opt"] = alpha_opt_out
        metrics = {
            "critic_loss": critic_loss,
            "actor_loss": actor_loss,
            "action_penalty": action_penalty,
            "target_mean": critic_aux["target_mean"],
            "q1_pred_mean": critic_aux["q1_pred_mean"],
            "q2_pred_mean": critic_aux["q2_pred_mean"],
            "q_pi_mean": q_pi_mean,
            "actor_mask_mean": actor_mask_mean,
            "q1_grad_norm": q1_grad_norm,
            "q2_grad_norm": q2_grad_norm,
            "actor_grad_norm": actor_grad_norm,
            "actor_update_applied": actor_update_applied,
            "alpha": jnp.exp(log_alpha_out) * actor_update_applied if use_sac else jnp.asarray(0.0, dtype=jnp.float32),
            "entropy": entropy_est,
        }
        return new_state, metrics, key_next

    return update


def build_sa_action_fns(
    ra_cfg: Any,
    actor_cfg: ActorConfig,
    action_scale: jax.Array,
    action_low: jax.Array,
    action_high: jax.Array,
) -> Tuple[
    Callable[[Any, jax.Array], jax.Array],
    Callable[[Any, jax.Array, jax.Array], Tuple[jax.Array, jax.Array, jax.Array]],
]:
    @jax.jit
    def eval_action_batch(params, obs_b: jax.Array) -> jax.Array:
        raw = actor_mean_action(
            params,
            obs_b,
            action_scale,
            actor_cfg,
            action_low=action_low,
            action_high=action_high,
        )
        return jnp.clip(jnp.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0), action_low, action_high)

    use_sac = sa_backbone(ra_cfg) == "sac"

    @jax.jit
    def collect_action_batch(params, obs_b: jax.Array, key: jax.Array):
        key_random, key_noise = jax.random.split(key, 2)
        if use_sac:
            # Exploration comes from the policy itself; no extra Gaussian noise.
            random_raw = action_low + jax.random.uniform(
                key_random,
                shape=obs_b.shape[:-1] + action_low.shape,
                dtype=jnp.float32,
            ) * (action_high - action_low)
            sampled, _, mean_raw = sample_actor_action(
                params, obs_b, key_noise, action_scale, actor_cfg, action_low=action_low, action_high=action_high
            )
            sampled = jnp.clip(jnp.nan_to_num(sampled, nan=0.0, posinf=0.0, neginf=0.0), action_low, action_high)
            mean_raw = jnp.clip(jnp.nan_to_num(mean_raw, nan=0.0, posinf=0.0, neginf=0.0), action_low, action_high)
            return mean_raw, sampled, random_raw
        random_raw = action_low + jax.random.uniform(
            key_random,
            shape=obs_b.shape[:-1] + action_low.shape,
            dtype=jnp.float32,
        ) * (action_high - action_low)
        raw = eval_action_batch(params, obs_b)
        noise = jax.random.normal(key_noise, shape=raw.shape, dtype=raw.dtype)
        noise = noise * (jnp.asarray(ra_cfg.exploration_std, dtype=raw.dtype) * action_scale)
        if ra_cfg.exploration_clip > 0.0:
            clip_mag = jnp.asarray(ra_cfg.exploration_clip, dtype=raw.dtype) * action_scale
            noise = jnp.clip(noise, -clip_mag, clip_mag)
        noisy_raw = jnp.clip(raw + noise, action_low, action_high)
        return raw, noisy_raw, random_raw

    return eval_action_batch, collect_action_batch


def update_curriculum_state(
    curriculum_scale: jax.Array,
    episode_count: jax.Array,
    success_window: jax.Array,
    success_window_size: jax.Array,
    success_window_ptr: jax.Array,
    completed_success: jax.Array,
    completed_done: jax.Array,
    ra_cfg: Any,
) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    threshold = jnp.asarray(float(ra_cfg.curriculum_success_threshold), dtype=jnp.float32)
    increment = jnp.asarray(float(ra_cfg.curriculum_increment), dtype=jnp.float32)
    min_episodes = jnp.int32(ra_cfg.curriculum_min_episodes)
    window = int(max(1, ra_cfg.curriculum_window_episodes))

    def body(i: int, carry):
        scale_i, ep_count_i, buf_i, buf_size_i, buf_ptr_i = carry
        done_i = completed_done[i]
        success_i = completed_success[i]

        def on_done(inner):
            scale_j, ep_count_j, buf_j, buf_size_j, buf_ptr_j = inner
            buf_j = buf_j.at[buf_ptr_j].set(success_i)
            buf_ptr_j = (buf_ptr_j + jnp.int32(1)) % jnp.int32(window)
            buf_size_j = jnp.minimum(buf_size_j + jnp.int32(1), jnp.int32(window))
            ep_count_j = ep_count_j + jnp.int32(1)
            ready = (ep_count_j >= min_episodes) & (buf_size_j >= jnp.int32(window))
            mean_success = jnp.where(
                buf_size_j > 0,
                jnp.sum(buf_j) / jnp.maximum(buf_size_j.astype(jnp.float32), 1.0),
                jnp.asarray(0.0, dtype=jnp.float32),
            )
            promote = ready & (mean_success >= threshold) & (scale_j < jnp.asarray(1.0, dtype=jnp.float32))
            scale_j = jnp.where(promote, jnp.minimum(jnp.asarray(1.0, dtype=jnp.float32), scale_j + increment), scale_j)
            buf_j = jnp.where(promote, jnp.zeros_like(buf_j), buf_j)
            buf_size_j = jnp.where(promote, jnp.int32(0), buf_size_j)
            buf_ptr_j = jnp.where(promote, jnp.int32(0), buf_ptr_j)
            return scale_j, ep_count_j, buf_j, buf_size_j, buf_ptr_j

        return jax.lax.cond(done_i, on_done, lambda inner: inner, carry)

    return jax.lax.fori_loop(
        0,
        completed_done.shape[0],
        body,
        (curriculum_scale, episode_count, success_window, success_window_size, success_window_ptr),
    )


class SALoopState(NamedTuple):
    state: Dict[str, Any]
    replay: JaxReplayState
    env_state: Any
    obs: jax.Array
    key: jax.Array
    env_keys: jax.Array
    global_step: jax.Array
    updates: jax.Array
    curriculum_scale: jax.Array
    episode_count: jax.Array
    success_window: jax.Array
    success_window_size: jax.Array
    success_window_ptr: jax.Array


@dataclass(frozen=True)
class SASystemHooks:
    """Per-system behavior injected into the shared Phase-1 training loop."""

    episode_metric_fields: Tuple[str, ...]
    episode_bookkeeping: Callable[[Any, jax.Array], Tuple[jax.Array, Dict[str, jax.Array]]]
    run_eval: Callable[[Any, str], Dict[str, Any]]
    eval_rank_key: Callable[[Dict[str, Any]], Tuple[float, ...]]
    append_eval_history: Callable[[Dict[str, List[float]], Dict[str, Any]], None]
    episode_history_pairs: Tuple[Tuple[str, str], ...]
    history_keys: Tuple[str, ...]


def zero_sa_chunk_metrics(episode_metric_fields: Sequence[str]) -> Dict[str, jax.Array]:
    zf = jnp.asarray(0.0, dtype=jnp.float32)
    zi = jnp.int32(0)
    metrics: Dict[str, jax.Array] = {"global_step": zi, "updates": zi, "episode_count": zf}
    for key in SA_UPDATE_METRIC_SUMS:
        metrics[key] = zf
    for key in episode_metric_fields:
        metrics[key] = zf
    return metrics


def build_sa_one_vec_step(
    *,
    env_fns: Any,
    collect_action_batch_fn: Callable[..., Tuple[jax.Array, jax.Array, jax.Array]],
    update_fn: Callable[..., Tuple[Dict[str, Any], Dict[str, jax.Array], jax.Array]],
    ra_cfg: Any,
    hooks: SASystemHooks,
):
    num_envs = int(ra_cfg.num_envs)
    batch_size_i32 = jnp.int32(ra_cfg.batch_size)
    update_after_i32 = jnp.int32(ra_cfg.update_after)
    update_every_i32 = jnp.int32(max(1, ra_cfg.update_every))
    num_envs_i32 = jnp.int32(num_envs)
    max_due_per_vec_step = max(1, int(np.ceil(float(num_envs) / float(max(1, ra_cfg.update_every)))))
    beta_arr = jnp.asarray(float(ra_cfg.beta), dtype=jnp.float32)

    def one_vec_step(carry: SALoopState, _):
        state_i = carry.state
        replay_i = carry.replay
        env_state_i = carry.env_state
        obs_i = carry.obs
        key_i = carry.key
        env_keys_i = carry.env_keys
        global_step_i = carry.global_step
        updates_i = carry.updates
        curriculum_scale_i = carry.curriculum_scale
        episode_count_i = carry.episode_count
        success_window_i = carry.success_window
        success_window_size_i = carry.success_window_size
        success_window_ptr_i = carry.success_window_ptr

        key_i, key_collect = jax.random.split(key_i, 2)
        _, noisy_raw_i, random_raw_i = collect_action_batch_fn(state_i["actor_params"], obs_i, key_collect)
        use_random = global_step_i < jnp.int32(ra_cfg.start_steps)
        act_raw_i = jnp.where(use_random, random_raw_i, noisy_raw_i)

        env_keys_i, step_keys = split_env_keys(env_keys_i)
        env_state_i, next_obs_true_i, next_obs_out_i, done_rollout_i, info_i = env_fns.step_batched(
            env_state_i,
            act_raw_i,
            step_keys,
            curriculum_scale_i,
        )
        replay_i = jax_replay_add_batch(
            replay_i,
            {
                "obs": obs_i.astype(jnp.float32),
                "act_raw": act_raw_i.astype(jnp.float32),
                "next_obs_true": next_obs_true_i.astype(jnp.float32),
                "goal_next": info_i.goal_next.astype(jnp.float32),
                "fail_next": info_i.fail_next.astype(jnp.float32),
                "done_rollout": done_rollout_i.astype(jnp.float32),
                "act_applied": info_i.applied_action.astype(jnp.float32),
            },
        )
        obs_i = next_obs_out_i

        prev_step = global_step_i
        global_step_i = global_step_i + num_envs_i32

        replay_ready = replay_i.size >= batch_size_i32
        lo = jnp.maximum(prev_step + jnp.int32(1), update_after_i32)
        hi = global_step_i
        due_raw = jnp.where(
            hi >= lo,
            hi // update_every_i32 - (lo - jnp.int32(1)) // update_every_i32,
            jnp.int32(0),
        )
        due = jnp.where(replay_ready, due_raw, jnp.int32(0))
        due = jnp.minimum(due, jnp.int32(max_due_per_vec_step))

        def do_due_update(_, upd_carry):
            state_u, replay_u, key_u, updates_u, metrics_u = upd_carry

            def do_one_grad(_, grad_carry):
                state_g, replay_g, key_g, updates_g, metrics_g = grad_carry
                key_g, key_sample, key_upd = jax.random.split(key_g, 3)
                batch = jax_replay_sample(replay_g, ra_cfg.batch_size, key_sample, SA_REPLAY_SAMPLE_FIELDS)
                state_g, upd_metrics, _ = update_fn(state_g, batch, key_upd)
                updates_g = updates_g + jnp.int32(1)
                metrics_g = dict(metrics_g)
                metrics_g["update_count"] = metrics_g["update_count"] + 1.0
                metrics_g["actor_update_count"] = metrics_g["actor_update_count"] + upd_metrics["actor_update_applied"]
                metrics_g["critic_loss_sum"] = metrics_g["critic_loss_sum"] + upd_metrics["critic_loss"]
                metrics_g["actor_loss_sum"] = metrics_g["actor_loss_sum"] + upd_metrics["actor_loss"]
                metrics_g["action_penalty_sum"] = metrics_g["action_penalty_sum"] + upd_metrics["action_penalty"]
                metrics_g["target_mean_sum"] = metrics_g["target_mean_sum"] + upd_metrics["target_mean"]
                metrics_g["q_pi_mean_sum"] = metrics_g["q_pi_mean_sum"] + upd_metrics["q_pi_mean"]
                metrics_g["q1_grad_norm_sum"] = metrics_g["q1_grad_norm_sum"] + upd_metrics["q1_grad_norm"]
                metrics_g["q2_grad_norm_sum"] = metrics_g["q2_grad_norm_sum"] + upd_metrics["q2_grad_norm"]
                metrics_g["actor_grad_norm_sum"] = metrics_g["actor_grad_norm_sum"] + upd_metrics["actor_grad_norm"]
                metrics_g["alpha_sum"] = metrics_g["alpha_sum"] + upd_metrics["alpha"]
                metrics_g["entropy_sum"] = metrics_g["entropy_sum"] + upd_metrics["entropy"]
                return state_g, replay_g, key_g, updates_g, metrics_g

            return jax.lax.fori_loop(0, ra_cfg.gradient_steps, do_one_grad, upd_carry)

        state_i, replay_i, key_i, updates_i, step_metrics = jax.lax.fori_loop(
            0,
            due,
            do_due_update,
            (state_i, replay_i, key_i, updates_i, zero_sa_chunk_metrics(hooks.episode_metric_fields)),
        )

        success_signal_i, episode_fields_i = hooks.episode_bookkeeping(info_i, beta_arr)

        curriculum_scale_i, episode_count_i, success_window_i, success_window_size_i, success_window_ptr_i = update_curriculum_state(
            curriculum_scale_i,
            episode_count_i,
            success_window_i,
            success_window_size_i,
            success_window_ptr_i,
            success_signal_i,
            info_i.episode_done,
            ra_cfg,
        )

        step_metrics = dict(step_metrics)
        step_metrics["episode_count"] = jnp.sum(info_i.episode_done.astype(jnp.float32))
        for field_name in hooks.episode_metric_fields:
            step_metrics[field_name] = episode_fields_i[field_name]

        return SALoopState(
            state=state_i,
            replay=replay_i,
            env_state=env_state_i,
            obs=obs_i,
            key=key_i,
            env_keys=env_keys_i,
            global_step=global_step_i,
            updates=updates_i,
            curriculum_scale=curriculum_scale_i,
            episode_count=episode_count_i,
            success_window=success_window_i,
            success_window_size=success_window_size_i,
            success_window_ptr=success_window_ptr_i,
        ), step_metrics

    return one_vec_step


class SATrainingLoopResult(NamedTuple):
    loop_state: SALoopState
    history: Dict[str, List[float]]
    best_eval_stats: Dict[str, Any] | None
    best_eval_step: int
    best_state: Dict[str, Any] | None
    val_eval: Dict[str, Any]
    test_eval: Dict[str, Any]
    total_time: float


def run_sa_training_loop(
    *,
    ra_cfg: Any,
    loop_state: SALoopState,
    one_vec_step,
    hooks: SASystemHooks,
) -> SATrainingLoopResult:
    num_envs = int(ra_cfg.num_envs)
    steps_per_jit = int(ra_cfg.steps_per_jit)
    total_steps = int(ra_cfg.total_steps)
    total_vec_steps = (total_steps + num_envs - 1) // num_envs if total_steps > 0 else 0

    chunk_fn_cache: Dict[int, Callable[[SALoopState], Tuple[SALoopState, Dict[str, jax.Array]]]] = {}

    def get_chunk_fn(length: int):
        if length not in chunk_fn_cache:

            @jax.jit
            def _chunk_fn(loop_s: SALoopState):
                loop_s, seq_metrics = jax.lax.scan(one_vec_step, loop_s, xs=None, length=length)
                chunk = jax.tree_util.tree_map(lambda x: jnp.sum(x, axis=0), seq_metrics)
                chunk["global_step"] = loop_s.global_step
                chunk["updates"] = loop_s.updates
                return loop_s, chunk

            chunk_fn_cache[length] = _chunk_fn
        return chunk_fn_cache[length]

    history: Dict[str, List[float]] = {key: [] for key in hooks.history_keys}

    best_eval_stats: Dict[str, Any] | None = None
    best_eval_step = 0
    best_eval_score: Tuple[float, ...] | None = None
    best_state: Dict[str, Any] | None = None
    latest_chunk_metrics: Dict[str, float] | None = None

    t0 = time.time()
    last_log_t = t0
    next_eval_step = int(ra_cfg.eval_every)
    next_log_step = int(ra_cfg.log_every)

    vec_done = 0
    while vec_done < total_vec_steps:
        this_vec = min(steps_per_jit, total_vec_steps - vec_done)
        loop_state, chunk = get_chunk_fn(this_vec)(loop_state)
        vec_done += this_vec

        chunk_host = jax.device_get(chunk)
        step_now = int(min(total_steps, int(chunk_host["global_step"])))
        current_curriculum = float(jax.device_get(loop_state.curriculum_scale))
        current_episode_count = int(jax.device_get(loop_state.episode_count))
        upd_count = float(chunk_host["update_count"])
        ep_count = float(chunk_host["episode_count"])

        if upd_count > 0.0 and ra_cfg.record_update_metrics:
            inv_upd = 1.0 / max(1.0, upd_count)
            critic_loss = float(chunk_host["critic_loss_sum"] * inv_upd)
            actor_loss = float(chunk_host["actor_loss_sum"] * inv_upd)
            action_penalty = float(chunk_host["action_penalty_sum"] * inv_upd)
            target_mean = float(chunk_host["target_mean_sum"] * inv_upd)
            q_pi_mean = float(chunk_host["q_pi_mean_sum"] * inv_upd)
            q1_grad_norm = float(chunk_host["q1_grad_norm_sum"] * inv_upd)
            q2_grad_norm = float(chunk_host["q2_grad_norm_sum"] * inv_upd)
            actor_grad_norm = float(chunk_host["actor_grad_norm_sum"] * inv_upd)
            history["critic_loss"].append(critic_loss)
            history["actor_loss"].append(actor_loss)
            history["action_penalty"].append(action_penalty)
            history["target_mean"].append(target_mean)
            history["q_pi_mean"].append(q_pi_mean)
            history["q1_grad_norm"].append(q1_grad_norm)
            history["q2_grad_norm"].append(q2_grad_norm)
            history["actor_grad_norm"].append(actor_grad_norm)
            n_actor = max(1.0, float(chunk_host["actor_update_count"]))
            alpha_mean = float(chunk_host["alpha_sum"]) / n_actor
            entropy_mean = float(chunk_host["entropy_sum"]) / n_actor
            if "alpha" in history:
                history["alpha"].append(alpha_mean)
            if "entropy" in history:
                history["entropy"].append(entropy_mean)
            latest_chunk_metrics = {
                "critic_loss": critic_loss,
                "actor_loss": actor_loss,
                "action_penalty": action_penalty,
                "target_mean": target_mean,
                "q_pi_mean": q_pi_mean,
                "q1_grad_norm": q1_grad_norm,
                "actor_grad_norm": actor_grad_norm,
                "alpha": alpha_mean,
                "entropy": entropy_mean,
            }

        if ep_count > 0.0:
            inv_ep = 1.0 / max(1.0, ep_count)
            history["step"].append(float(step_now))
            history["episode_idx"].append(float(current_episode_count))
            history["curriculum_scale"].append(current_curriculum)
            for hist_key, metric_key in hooks.episode_history_pairs:
                history[hist_key].append(float(chunk_host[metric_key] * inv_ep))

        while ra_cfg.eval_every > 0 and step_now >= next_eval_step:
            eval_stats = hooks.run_eval(loop_state.state["actor_params"], "val")
            hooks.append_eval_history(history, eval_stats)

            eval_score = hooks.eval_rank_key(eval_stats)
            if best_eval_score is None or eval_score > best_eval_score:
                best_eval_score = eval_score
                best_eval_step = next_eval_step
                best_eval_stats = {k: v for k, v in eval_stats.items() if k != "trajectory"}
                best_state = snapshot_sa_state(loop_state.state)
            next_eval_step += int(ra_cfg.eval_every)

        if ra_cfg.log_every > 0 and step_now >= next_log_step:
            now = time.time()
            step_rate = max(1, step_now - (next_log_step - int(ra_cfg.log_every))) / max(1e-6, now - last_log_t)
            last_log_t = now
            print(
                f"step={step_now} replay={int(jax.device_get(loop_state.replay.size))} updates={int(chunk_host['updates'])} "
                f"eps/sec={step_rate:.1f} curriculum={current_curriculum:.2f}"
            )
            if latest_chunk_metrics is not None:
                print(
                    "  "
                    f"critic={latest_chunk_metrics['critic_loss']:.4f} "
                    f"actor={latest_chunk_metrics['actor_loss']:.4f} "
                    f"act_pen={latest_chunk_metrics['action_penalty']:.4f} "
                    f"target={latest_chunk_metrics['target_mean']:.4f} "
                    f"q_pi={latest_chunk_metrics['q_pi_mean']:.4f} "
                    f"q1_gn={latest_chunk_metrics['q1_grad_norm']:.3f} "
                    f"actor_gn={latest_chunk_metrics['actor_grad_norm']:.3f}"
                    + (
                        f" alpha={latest_chunk_metrics['alpha']:.4g} H={latest_chunk_metrics['entropy']:.2f}"
                        if sa_backbone(ra_cfg) == "sac"
                        else ""
                    )
                )
            next_log_step += int(ra_cfg.log_every)

    total_time = time.time() - t0
    val_eval = hooks.run_eval(loop_state.state["actor_params"], "val")
    test_eval = hooks.run_eval(loop_state.state["actor_params"], "test")
    final_eval_score = hooks.eval_rank_key(val_eval)
    if best_eval_score is None or final_eval_score > best_eval_score:
        best_eval_score = final_eval_score
        best_eval_step = int(ra_cfg.total_steps)
        best_eval_stats = {k: v for k, v in val_eval.items() if k != "trajectory"}
        best_state = snapshot_sa_state(loop_state.state)

    return SATrainingLoopResult(
        loop_state=loop_state,
        history=history,
        best_eval_stats=best_eval_stats,
        best_eval_step=best_eval_step,
        best_state=best_state,
        val_eval=val_eval,
        test_eval=test_eval,
        total_time=total_time,
    )


__all__ = [
    "SA_BACKBONES",
    "SA_REPLAY_SAMPLE_FIELDS",
    "SA_UPDATE_METRIC_SUMS",
    "SALoopState",
    "SASystemHooks",
    "SATrainingLoopResult",
    "batch_safe_contains",
    "build_sa_action_fns",
    "build_sa_one_vec_step",
    "build_sa_update_fn",
    "huber_loss",
    "init_sa_state",
    "rate_mean",
    "run_sa_training_loop",
    "sa_backbone",
    "sa_replay_init",
    "snapshot_sa_state",
    "split_env_keys",
    "summary_stats",
    "update_curriculum_state",
    "zero_sa_chunk_metrics",
]
