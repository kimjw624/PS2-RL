"""Phase-I safe-arrival training for the approach-cone landing task.

Reuses the shared ``sa_trainer_core`` loop (TD3 or SAC actor backbone) with the
landing environment, the analytic design-region sampler and region-stratified
held-out evaluation of the recoverability m_r = mu_r(C_N(pi_b)) (landing note,
Sec. 10).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import json
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from ps2rl.backup_policy.backup_policy import BackupPolicy
from ps2rl.envs.quadrotor_landing_config import QuadrotorLandingConfig
from ps2rl.phase1_sa.landing_design_region import REGION_NAMES, LandingDesignRegionConfig, heldout_sets
from ps2rl.phase1_sa.quadrotor_landing_sa_env import build_landing_sa_env, landing_action_box, landing_step_fn
from ps2rl.phase1_sa.sa_critic import SafeArrivalCriticConfig
from ps2rl.phase1_sa.sa_trainer_core import (
    SALoopState,
    SASystemHooks,
    batch_safe_contains,
    build_sa_action_fns,
    build_sa_one_vec_step,
    build_sa_update_fn,
    init_sa_state,
    run_sa_training_loop,
    sa_backbone,
    sa_replay_init,
    snapshot_sa_state,
)
from ps2rl.utils.policy import ActorConfig, actor_mean_action
from ps2rl.utils.seed import make_prng_key


@dataclass(frozen=True)
class LandingSAConfig:
    sa_backbone: str = "sac"
    seed: int = 0
    total_steps: int = 5_000_000
    start_steps: int = 5_000
    update_after: int = 2_000
    update_every: int = 8
    gradient_steps: int = 1
    batch_size: int = 256
    replay_size: int = 1_000_000

    beta: float = 0.99  # beta^N = 0.366 >= v_min = 0.3 for N = 100 (note Eq. (13))
    tau: float = 0.0025
    policy_delay: int = 2
    actor_lr: float = 1e-4
    critic_lr: float = 3e-4
    max_grad_norm: float = 5.0
    critic_huber_delta: float = 1.0
    action_smoothness_weight: float = 0.05

    hidden_size: int = 256
    actor_activation: str = "elu"  # C^1 so pi_b stays differentiable for the CIL (note Remark 6)
    actor_log_std_min: float = -5.0
    actor_log_std_max: float = 0.0

    # TD3 exploration (ignored by SAC)
    exploration_std: float = 0.1
    exploration_clip: float = 0.25
    target_policy_noise_std: float = 0.0
    target_policy_noise_clip: float = 0.0

    # SAC (ignored by TD3); entropy is measured on the normalized action in [-1, 1]^4
    sac_alpha_init: float = 0.01
    sac_alpha_min: float = 1e-4
    sac_alpha_max: float = 0.05
    sac_alpha_lr: float = 3e-4
    sac_target_entropy: float = -4.0
    sac_learn_alpha: bool = True

    episode_max_steps: int = 150  # k_max > N (note Sec. 9)
    eval_every: int = 50_000
    log_every: int = 10_000
    record_update_metrics: bool = True
    update_metric_every: int = 200
    num_envs: int = 64
    steps_per_jit: int = 128

    curriculum_start_scale: float = 0.0
    curriculum_increment: float = 0.1
    curriculum_success_threshold: float = 0.8
    curriculum_window_episodes: int = 200
    curriculum_min_episodes: int = 400

    use_handoff: bool = True
    collector_terminate_on_goal: bool = True
    hoeffding_delta: float = 0.05

    def __post_init__(self) -> None:
        sa_backbone(self)  # validates the name

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "LandingSAConfig":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in payload.items() if k in names})


_EPISODE_METRIC_FIELDS = (
    "episode_len_sum",
    "episode_success_sum",
    "episode_success_within_n_sum",
    "episode_crash_sum",
    "episode_timeout_sum",
    "episode_min_h_sum",
)
_EPISODE_HISTORY_PAIRS = (
    ("episode_len", "episode_len_sum"),
    ("episode_success_rate", "episode_success_sum"),
    ("episode_success_within_n_rate", "episode_success_within_n_sum"),
    ("episode_crash_rate", "episode_crash_sum"),
    ("episode_timeout_rate", "episode_timeout_sum"),
    ("episode_min_h", "episode_min_h_sum"),
)
_EVAL_KEYS = tuple(f"eval_m_{r}" for r in REGION_NAMES) + ("eval_mu_weighted", "eval_crash_rate")
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


def _episode_bookkeeping(info: Any, beta_arr: jax.Array) -> tuple[jax.Array, dict[str, jax.Array]]:
    done = info.episode_done.astype(jnp.float32)
    fields_ = {
        "episode_len_sum": jnp.sum(info.completed_len),
        "episode_success_sum": jnp.sum(info.completed_success),
        "episode_success_within_n_sum": jnp.sum(info.completed_success_within_n),
        "episode_crash_sum": jnp.sum(info.completed_crash),
        "episode_timeout_sum": jnp.sum(info.completed_timeout),
        "episode_min_h_sum": jnp.sum(info.completed_min_h * done),
    }
    # Curriculum promotes on arrival within the backup horizon N (membership in C_N).
    return info.completed_success_within_n, fields_


def build_landing_evaluator(
    cfg: QuadrotorLandingConfig,
    region_cfg: LandingDesignRegionConfig,
    env_fns: Any,
    actor_cfg: ActorConfig,
    *,
    beta: float,
    hoeffding_delta: float,
):
    """Return run_eval(actor_params, split) -> stats, evaluating 1[x0 in C_N(pi_b)] per region."""
    scale, low, high = landing_action_box(cfg)
    step_plant = landing_step_fn(cfg)
    cone, base_set = env_fns.safe_set, env_fns.base_set
    n_steps = int(cfg.num_steps)
    sets = {split: heldout_sets(region_cfg, env_fns.sampler, split=split) for split in ("val", "test")}
    weights = region_cfg.weights

    def rollout(params, x0):
        # First-hit semantics (Lemma 4): x0 in C_N  <=>  tau_B <= N and tau_B < tau_F.
        def body(carry, _):
            x, hit_b, hit_f = carry
            raw = jnp.clip(actor_mean_action(params, x, scale, actor_cfg, action_low=low, action_high=high), low, high)
            u = BackupPolicy.select_action(x, raw, base_set)
            xn = step_plant(x, u)
            active = ~(hit_b | hit_f)
            new_f = active & ~cone.training_contains(xn)
            new_b = active & ~new_f & base_set.contains(xn)
            return (jnp.where(active, xn, x), hit_b | new_b, hit_f | new_f), new_b

        (_, hit_b, hit_f), hits = jax.lax.scan(body, (x0, jnp.asarray(False), jnp.asarray(False)), None, length=n_steps)
        t_hit = jnp.where(hit_b, jnp.argmax(hits) + 1, -1)
        return hit_b, hit_f, t_hit

    rollout_batch = jax.jit(jax.vmap(rollout, in_axes=(None, 0)))

    def run_eval(actor_params, split: str) -> dict[str, Any]:
        stats: dict[str, Any] = {"split": split}
        num, den = 0.0, 0.0
        crash_all = []
        for name in REGION_NAMES:
            x0 = sets[split][name]
            hit_b, hit_f, t_hit = map(np.asarray, rollout_batch(actor_params, jnp.asarray(x0)))
            m = int(x0.shape[0])
            m_hat = float(hit_b.mean()) if m else float("nan")
            half = float(np.sqrt(np.log(2.0 / hoeffding_delta) / (2.0 * max(m, 1))))
            arr = t_hit[hit_b]
            stats[name] = {
                "n": m,
                "m_hat": m_hat,
                "hoeffding_halfwidth": half,
                "crash_rate": float(hit_f.mean()) if m else float("nan"),
                "timeout_rate": float((~hit_b & ~hit_f).mean()) if m else float("nan"),
                "mean_arrival_steps": float(arr.mean()) if arr.size else float("nan"),
                "mean_discounted_score": float(np.where(hit_b, beta ** np.maximum(t_hit, 0), 0.0).mean()) if m else float("nan"),
            }
            num += weights[name] * m_hat
            den += weights[name]
            crash_all.append(hit_f)
        stats["mu_weighted"] = num / max(den, 1e-12)
        stats["crash_rate"] = float(np.concatenate(crash_all).mean())
        return stats

    return run_eval


def _eval_rank_key(stats: dict[str, Any]) -> tuple[float, ...]:
    return (float(stats["mu_weighted"]), float(stats["edge"]["m_hat"]), float(stats["general"]["m_hat"]))


def _append_eval_history(history: dict[str, list[float]], stats: dict[str, Any]) -> None:
    for name in REGION_NAMES:
        history[f"eval_m_{name}"].append(float(stats[name]["m_hat"]))
    history["eval_mu_weighted"].append(float(stats["mu_weighted"]))
    history["eval_crash_rate"].append(float(stats["crash_rate"]))
    print(
        f"  [eval {stats['split']}] mu_w={stats['mu_weighted']:.3f} "
        + " ".join(f"m_{n}={stats[n]['m_hat']:.3f}" for n in REGION_NAMES)
        + f" crash={stats['crash_rate']:.3f}"
    )


def run_landing_sa_training(
    ra_cfg: LandingSAConfig,
    cfg: QuadrotorLandingConfig,
    region_cfg: LandingDesignRegionConfig,
    *,
    output_dir: str | None = None,
) -> dict[str, Any]:
    if not (0.0 < float(ra_cfg.beta) < 1.0):
        raise ValueError(f"beta must lie in (0, 1), got {ra_cfg.beta}")
    num_envs = int(ra_cfg.num_envs)
    env_fns = build_landing_sa_env(
        cfg,
        region_cfg,
        episode_max_steps=int(ra_cfg.episode_max_steps),
        terminate_on_goal=bool(ra_cfg.collector_terminate_on_goal),
    )
    action_scale, action_low, action_high = landing_action_box(cfg)
    actor_cfg = ActorConfig(
        obs_dim=env_fns.obs_dim,
        action_dim=env_fns.action_dim,
        hidden_sizes=(ra_cfg.hidden_size, ra_cfg.hidden_size),
        log_std_min=ra_cfg.actor_log_std_min,
        log_std_max=ra_cfg.actor_log_std_max,
        activation=ra_cfg.actor_activation,
    )
    critic_cfg = SafeArrivalCriticConfig(
        obs_dim=env_fns.obs_dim, act_dim=env_fns.action_dim, hidden_sizes=(ra_cfg.hidden_size, ra_cfg.hidden_size)
    )
    cone, base_set = env_fns.safe_set, env_fns.base_set
    goal_fn = batch_safe_contains(lambda x: base_set.contains(x))
    fail_fn = batch_safe_contains(lambda x: jnp.logical_not(cone.training_contains(x)))

    key = make_prng_key(ra_cfg.seed)
    key, k_state, k_env = jax.random.split(key, 3)
    state = init_sa_state(k_state, actor_cfg, critic_cfg, ra_cfg)
    replay = sa_replay_init(ra_cfg.replay_size, env_fns.obs_dim, env_fns.action_dim)
    update_fn = build_sa_update_fn(ra_cfg, actor_cfg, action_scale, action_low, action_high, goal_fn, fail_fn, goal_fn)
    _, collect_fn = build_sa_action_fns(ra_cfg, actor_cfg, action_scale, action_low, action_high)

    env_keys = jax.random.split(k_env, num_envs)
    s0 = jnp.asarray(float(ra_cfg.curriculum_start_scale), dtype=jnp.float32)
    env_state, obs = env_fns.reset_batched(env_keys, s0)
    window = int(max(1, ra_cfg.curriculum_window_episodes))
    loop_state = SALoopState(
        state=state,
        replay=replay,
        env_state=env_state,
        obs=obs,
        key=key,
        env_keys=env_keys,
        global_step=jnp.int32(0),
        updates=jnp.int32(0),
        curriculum_scale=s0,
        episode_count=jnp.int32(0),
        success_window=jnp.zeros((window,), dtype=jnp.float32),
        success_window_size=jnp.int32(0),
        success_window_ptr=jnp.int32(0),
    )

    run_eval = build_landing_evaluator(
        cfg, region_cfg, env_fns, actor_cfg, beta=float(ra_cfg.beta), hoeffding_delta=float(ra_cfg.hoeffding_delta)
    )
    hooks = SASystemHooks(
        episode_metric_fields=_EPISODE_METRIC_FIELDS,
        episode_bookkeeping=_episode_bookkeeping,
        run_eval=run_eval,
        eval_rank_key=_eval_rank_key,
        append_eval_history=_append_eval_history,
        episode_history_pairs=_EPISODE_HISTORY_PAIRS,
        history_keys=_HISTORY_KEYS,
    )
    initial_eval = run_eval(state["actor_params"], "val")
    one_vec_step = build_sa_one_vec_step(
        env_fns=env_fns, collect_action_batch_fn=collect_fn, update_fn=update_fn, ra_cfg=ra_cfg, hooks=hooks
    )
    res = run_sa_training_loop(ra_cfg=ra_cfg, loop_state=loop_state, one_vec_step=one_vec_step, hooks=hooks)

    best_eval = res.best_eval_stats if res.best_eval_stats is not None else res.val_eval
    best_state = res.best_state if res.best_state is not None else snapshot_sa_state(res.loop_state.state)
    # Model selection on val, reporting on test: evaluate the selected checkpoint on test.
    test_at_best = run_eval(best_state["actor_params"], "test")
    summary = {
        "training_objective": "discounted_safe_arrival",
        "task": "approach_cone_landing",
        "sa_backbone": sa_backbone(ra_cfg),
        "seed": int(ra_cfg.seed),
        "total_steps": int(ra_cfg.total_steps),
        "updates": int(jax.device_get(res.loop_state.updates)),
        "wall_time_sec": float(res.total_time),
        "jax_backend": jax.default_backend(),
        "final_curriculum_scale": float(jax.device_get(res.loop_state.curriculum_scale)),
        "untrained_val": initial_eval,
        "final_val": res.val_eval,
        "best_val": best_eval,
        "best_eval_step": int(res.best_eval_step),
        "test_at_best": test_at_best,
        "test_at_final": res.test_eval,
    }
    result = {
        "summary": summary,
        "history": res.history,
        "configs": {
            "landing": cfg.as_dict(),
            "design_region": region_cfg.as_dict(),
            "backup_ra": asdict(ra_cfg),
            "actor": asdict(actor_cfg),
        },
        "final_state": snapshot_sa_state(res.loop_state.state),
        "best_state": best_state,
        "actor_cfg": actor_cfg,
    }
    if output_dir is not None:
        with open(f"{output_dir}/summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=float)
        with open(f"{output_dir}/configs.json", "w", encoding="utf-8") as f:
            json.dump(result["configs"], f, indent=2)
    return result


__all__ = ["LandingSAConfig", "build_landing_evaluator", "run_landing_sa_training"]
