"""SAC tracker for the slow reference (``ps2rl.envs.slow_track_env``), for SITL and hardware.

The algorithm and its settings are those of the cone-obstacle stage-1 tracker that worked
(``cone_obstacle_ue_ps2_trainer`` with ``use_projection=false``; 2.2 cm position error):

* the critics learn ``reward_scale * r`` with ``reward_scale = 0.01`` (unscaled returns stalled there);
* SAC, lr 3e-4, batch 256, 64 envs, an update every 2 env steps, replay 1M, gamma 0.99, tau 0.005,
  temperature auto-tuned from 0.1 with floor 0.01, target entropy -4;
* actor's last layer initialised at the hover command (g, 0, 0, 0), or at a zero correction in residual mode;
* episodes end when the vehicle loses the reference (with the discounted floor reward), so a policy
  that cannot fly yet does not collect data far away.

Not used (they failed): the landing ``tracker_v1`` preset (unscaled critic targets clipped at +-1e4,
collapses after 3-5M steps) and unscaled critic rewards.

Differences: by default the actor outputs a bounded correction to a PD + observer base controller
(``policy_mode="residual"``, the residual-striker pattern of this repo; see ``slow_track_env``), no CIL code path, timeouts are not terminal for the critic, and the best checkpoint is
chosen by the *robust evaluation* (``EVAL_SCENARIOS``: every episode starts at the hand-over and flies the
whole reference, on ideal, nominal, randomised, slow-actuator, thrust +-10 %, windy and offset-start
plants), not by return.

``touch <run_dir>/STOP`` ends a run at its next chunk; it finishes normally.
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

from ps2rl.envs.slow_track_env import SlowTrackEnvConfig, obs_layout, build_slow_track_env, load_reference
from ps2rl.utils.networks import CriticConfig, init_q_params, q_value
from ps2rl.utils.optim import adam_init, adam_step, soft_update
from ps2rl.utils.policy import ActorConfig, actor_mean_action, init_actor_params, sample_actor_action
from ps2rl.utils.quaternion import normalize_quaternion, quaternion_from_euler_zyx, quaternion_multiply


@dataclass(frozen=True)
class SlowTrackSACConfig:
    seed: int = 0
    total_steps: int = 3_000_000
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
    # 1e-2 (the cone tracker's floor) pins the temperature here: the slow reference's per-step costs are small,
    # so the entropy bonus outweighs cm-level tracking and the policy stays noisy (CPU check: 0.4-0.6 m at
    # 0.8M steps, alpha stuck at 0.01); with 1e-4 the auto-tuning can go below it
    min_alpha: float = 1e-4
    target_entropy: float = -4.0
    max_grad_norm: float = 5.0
    q_clip_abs: float = 1e4
    hidden_size: int = 256
    reward_scale: float = 0.01
    init_hover_bias: bool = True
    num_envs: int = 64
    steps_per_jit: int = 32
    eval_every: int = 100_352
    eval_episodes: int = 16          # per evaluation scenario
    log_every: int = 50_176
    warm_start_weights: str = ""
    # critic warm-up: no actor / temperature update before this many critic updates (the residual striker's
    # recipe: the zero-correction actor is kept until the critics know what a correction is worth)
    actor_start_update: int = 0
    # CAPS smoothness regularisation of the actor's mean action (Mysore et al., ICRA 2021), in units of the
    # normalised action (box mapped to [-1, 1]): caps_temporal * |mu(s) - mu(s')|^2 (consecutive states of the
    # replay) + caps_spatial * |mu(s) - mu(s + noise)|^2, noise ~ N(0, caps_sigma^2) on every observation entry.
    # 0 = off (the cone-tracker recipe). The first run (4.9M steps) chattered: command change 0.7 of the
    # half-range per step, errors 0.12-0.38 m.
    caps_temporal: float = 0.0
    caps_spatial: float = 0.0
    caps_sigma: float = 0.02
    # Behaviour cloning of the PD controller (the env's base_action, rebuilt from the observation) as a warm start
    # for the standalone policy, as Phase I warm-starts from the LQR: actor loss += w * |(mu(s) - u_pd(s)) / box|^2,
    # with w = bc_weight for the first bc_hold actor updates, then decaying linearly to 0 over bc_decay updates.
    # The policy flies without the PD; the PD only shapes its start. 0 = off.
    bc_weight: float = 0.0
    bc_hold: int = 100_000
    bc_decay: int = 400_000
    bc_min: float = 0.0           # the weight never decays below this (v3_bc lost everything once it reached 0)

    @classmethod
    def from_dict(cls, d):
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})


# --------------------------------------------------------------------------- evaluation scenarios
# Every scenario starts at the reference start (the hand-over) and flies the whole reference + 2 s.
# Plant fields given here override a draw from the training distribution; "start" sets the initial offset.
EVAL_SCENARIOS: Dict[str, Dict[str, Any]] = {
    "ideal": dict(delay=0, tau_rate=0.005, tau_thrust=0.005, thrust_gain=1.0, rate_gain=1.0, d_amp=0.0, wind=0.0,
                  noise_scale=0.0, start="exact"),
    "nominal": dict(delay=1, tau_rate=0.04, tau_thrust=0.04, thrust_gain=1.0, rate_gain=1.0, d_amp=0.0, wind=0.0,
                    noise_scale=1.0, start="rest"),
    "train": dict(start="random"),
    "slow_actuators": dict(delay=2, tau_rate=0.08, tau_thrust=0.06, start="rest"),
    "thrust_low": dict(thrust_gain=0.9, start="rest"),
    "thrust_high": dict(thrust_gain=1.1, start="rest"),
    "windy": dict(d_amp=0.5, wind=0.3, start="rest"),
    "offset_start": dict(start="offset"),
}


def make_eval_fn(env, env_cfg: SlowTrackEnvConfig, actor_cfg: ActorConfig, low, high, scale, n_ep: int,
                 extra_sec: float = 2.0):
    """Returns ``run(actor_params, key, scenario) -> dict of per-step arrays`` (jitted per scenario)."""
    n_ref = env.n_ref
    n_steps = n_ref + int(round(extra_sec / env_cfg.dt))
    rs0 = env.ref_states[0]
    f32 = jnp.float32

    def plant_for(key, sc):
        pl = env.sample_plant(key)
        rep = {}
        for k in ("delay", "tau_rate", "tau_thrust", "thrust_gain", "rate_gain", "d_amp", "noise_scale"):
            if k in sc:
                rep[k] = jnp.asarray(sc[k], jnp.int32 if k == "delay" else f32)
        if "wind" in sc:
            wdir = jax.random.uniform(jax.random.fold_in(key, 5), (), f32, 0.0, 2.0 * np.pi)
            rep["wind"] = sc["wind"] * jnp.stack([jnp.cos(wdir), jnp.sin(wdir), 0.0]).astype(f32)
        return pl._replace(**rep)

    def start_for(key, sc):
        k = jax.random.split(key, 4)
        mode = sc.get("start", "rest")
        if mode == "exact":
            return rs0
        if mode == "rest":       # what PX4 leaves us: a hover with small errors
            dp = jax.random.uniform(k[0], (3,), f32, -0.05, 0.05)
            rpy = jax.random.uniform(k[1], (3,), f32, -1.0, 1.0) * jnp.deg2rad(jnp.asarray([2.0, 2.0, 3.0], f32))
            dv = jax.random.uniform(k[2], (3,), f32, -0.05, 0.05)
        elif mode == "offset":   # 0.5 m away horizontally, 0.2 m in height
            a = jax.random.uniform(k[0], (), f32, 0.0, 2.0 * np.pi)
            dp = jnp.stack([0.5 * jnp.cos(a), 0.5 * jnp.sin(a), jnp.where(jax.random.uniform(k[3]) < 0.5, -0.2, 0.2)])
            rpy = jax.random.uniform(k[1], (3,), f32, -1.0, 1.0) * jnp.deg2rad(jnp.asarray([3.0, 3.0, 10.0], f32))
            dv = jnp.zeros(3, f32)
        else:                    # the training start distribution at the reference start
            dp = jax.random.uniform(k[0], (3,), f32, -1.0, 1.0) * env_cfg.init_p_start * jnp.asarray([1.0, 1.0, 0.5], f32)
            rpy = jax.random.uniform(k[1], (3,), f32, -1.0, 1.0) * jnp.deg2rad(
                jnp.asarray([env_cfg.init_tilt_deg, env_cfg.init_tilt_deg, env_cfg.init_yaw_deg], f32))
            dv = jax.random.uniform(k[2], (3,), f32, -env_cfg.init_v, env_cfg.init_v)
        q = normalize_quaternion(quaternion_multiply(quaternion_from_euler_zyx(rpy[0], rpy[1], rpy[2]),
                                                     normalize_quaternion(rs0[6:10])))
        return jnp.concatenate([rs0[0:3] + dp, rs0[3:6] + dv, q])

    def build(sc):
        def run(actor_params, key):
            keys = jax.random.split(key, n_ep)

            def one_reset(kk):
                k1, k2, k3 = jax.random.split(kk, 3)
                st, obs = env.reset_with(k3, 0, start_for(k2, sc), plant_for(k1, sc))
                return st._replace(max_steps=jnp.int32(n_steps + 10)), obs

            st, obs = jax.vmap(one_reset)(keys)

            def body(c, k):
                st, obs, alive = c
                u = jnp.clip(actor_mean_action(actor_params, obs, scale, actor_cfg, action_low=low, action_high=high),
                             low, high)
                st2, obs_t, obs_o, rew, done, info = env.step_batched(st, u, jax.vmap(lambda kk: jax.random.fold_in(kk, k))(keys))
                out = (info.x, info.u_cmd, info.pos_err, info.tilt_deg, rew, alive, info.terminal & alive,
                       info.d_eff, info.d_hat)
                return (st2, obs_o, alive & ~done), out

            _, out = jax.lax.scan(body, (st, obs, jnp.ones((n_ep,), bool)), jnp.arange(n_steps))
            names = ("x", "u", "pos_err", "tilt", "rew", "alive", "lost", "d_eff", "d_hat")
            return {n: jnp.swapaxes(o, 0, 1) for n, o in zip(names, out)}

        return jax.jit(run)

    fns = {name: build(sc) for name, sc in EVAL_SCENARIOS.items()}
    return fns, n_steps


def summarize_scenario(tr: Dict[str, np.ndarray], half: np.ndarray, rest_start: int) -> Dict[str, float]:
    al = tr["alive"].astype(bool)
    pe = np.where(al, tr["pos_err"], np.nan)
    u = tr["u"]
    du = np.linalg.norm(np.diff(u, axis=1) / half, axis=-1)
    du = np.where(al[:, 1:], du, np.nan)
    lost = tr["lost"].any(axis=1)
    ok = ~lost
    settled = pe[:, rest_start:] if pe.shape[1] > rest_start else pe
    return {
        "lost_rate": float(lost.mean()),
        "pos_err_mean": float(np.nanmean(pe)),
        "pos_err_p95": float(np.nanpercentile(pe, 95)),
        "pos_err_max": float(np.nanmax(pe)),
        "pos_err_after_5s": float(np.nanmean(settled)) if np.isfinite(settled).any() else float("nan"),
        "final_pos_err": float(np.nanmean(np.where(ok, pe[:, -1], np.nan))) if ok.any() else float("nan"),
        "tilt_max_deg": float(np.nanmax(np.where(al, tr["tilt"], np.nan))),
        "thrust_min": float(np.min(u[..., 0])), "thrust_max": float(np.max(u[..., 0])),
        "rate_xy_max": float(np.max(np.abs(u[..., 1:3]))), "rate_z_max": float(np.max(np.abs(u[..., 3]))),
        "du_mean": float(np.nanmean(du)),
        "observer_err_mean": float(np.nanmean(np.where(al[..., None], np.abs(tr["d_eff"] - tr["d_hat"]), np.nan))),
    }


def robust_score(ev: Dict[str, Dict[str, float]]) -> float:
    """Lower is better: mean over scenarios of the mean position error, + 1 m per lost episode fraction."""
    errs = [v["pos_err_mean"] + 1.0 * v["lost_rate"] for v in ev.values()]
    return float(np.mean(errs))


# --------------------------------------------------------------------------- PD from the observation (for BC)
def make_pd_from_obs(env_cfg: SlowTrackEnvConfig):
    """The env's base controller computed from a direct-mode observation alone (same numbers as ``base_action``)."""
    from ps2rl.utils.quaternion import normalize_quaternion as nq, rotation_matrix_from_quaternion, thrust_axis_world
    g = float(env_cfg.gravity)
    e3 = jnp.asarray([0.0, 0.0, 1.0], jnp.float32)
    lo = jnp.asarray(env_cfg.action_low, jnp.float32)
    hi = jnp.asarray(env_cfg.action_high, jnp.float32)

    def pd(o):
        ep = o[0:3] * env_cfg.obs_pos_scale
        ev = o[3:6] * env_cfg.obs_vel_scale
        q, qr = nq(o[6:10]), nq(o[10:14])
        ra = (o[20] + 1.0) * g
        d_hat = o[27:30]
        a_ref = thrust_axis_world(qr) * ra - g * e3
        f = a_ref - env_cfg.base_kp * ep - env_cfg.base_kd * ev - d_hat + g * e3
        R = rotation_matrix_from_quaternion(q)
        zd = f / jnp.maximum(jnp.linalg.norm(f), 1e-6)
        wb = R.T @ jnp.cross(R[:, 2], zd) * env_cfg.base_katt
        yaw = jnp.arctan2(R[1, 0], R[0, 0])
        yaw_ref = jnp.arctan2(2.0 * (qr[0] * qr[3] + qr[1] * qr[2]), 1.0 - 2.0 * (qr[2] ** 2 + qr[3] ** 2))
        dyaw = jnp.arctan2(jnp.sin(yaw - yaw_ref), jnp.cos(yaw - yaw_ref))
        return jnp.clip(jnp.stack([jnp.dot(f, R[:, 2]), wb[0], wb[1], -env_cfg.base_kyaw * dyaw]), lo, hi)

    return pd


# --------------------------------------------------------------------------- training
def run_slow_track_sac(cfg: SlowTrackSACConfig, env_cfg: SlowTrackEnvConfig, out_dir: Path) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    env = build_slow_track_env(env_cfg)
    # the actor's box: the correction box in residual mode, the command box in direct mode
    low = jnp.asarray(env_cfg.policy_low, jnp.float32)
    high = jnp.asarray(env_cfg.policy_high, jnp.float32)
    scale = 0.5 * (high - low)
    mid = 0.5 * (high + low)
    if cfg.bc_weight > 0.0 and env_cfg.policy_mode != "direct":
        raise ValueError("bc_weight needs policy_mode direct")
    pd_from_obs = make_pd_from_obs(env_cfg)
    cmd_half = 0.5 * (np.asarray(env_cfg.action_high) - np.asarray(env_cfg.action_low))
    n_env = int(cfg.num_envs)
    actor_cfg = ActorConfig(obs_dim=env.obs_dim, action_dim=4, hidden_sizes=(cfg.hidden_size, cfg.hidden_size),
                            log_std_min=-5.0, log_std_max=2.0, activation="relu")
    critic_cfg = CriticConfig(obs_dim=env.obs_dim, act_dim=4, hidden_sizes=(cfg.hidden_size, cfg.hidden_size))
    key = jax.random.PRNGKey(int(cfg.seed))
    key, ka, k1, k2, ke = jax.random.split(key, 5)
    ap = init_actor_params(ka, actor_cfg)
    if cfg.init_hover_bias and not cfg.warm_start_weights:
        lo_, hi_ = np.asarray(low, np.float64), np.asarray(high, np.float64)
        hover = np.zeros(4) if env_cfg.policy_mode == "residual" else np.asarray([env_cfg.gravity, 0.0, 0.0, 0.0])
        pre = np.arctanh(np.clip((hover - 0.5 * (lo_ + hi_)) / (0.5 * (hi_ - lo_)), -0.999, 0.999))
        last = ap["layers"][-1]
        ap["layers"][-1] = {"w": last["w"].at[:, :4].set(0.0), "b": last["b"].at[:4].set(jnp.asarray(pre, jnp.float32))}
    q1, q2 = init_q_params(k1, critic_cfg), init_q_params(k2, critic_cfg)
    log_alpha = jnp.asarray(np.log(max(cfg.init_alpha, cfg.min_alpha)), jnp.float32)
    if cfg.warm_start_weights:
        with open(Path(cfg.warm_start_weights), "rb") as f:
            w = pickle.load(f)
        ap, q1, q2 = w["actor_params"], w["q1_params"], w["q2_params"]
        log_alpha = jnp.asarray(w["log_alpha"], jnp.float32)
    copy = lambda t: jax.tree_util.tree_map(jnp.copy, t)
    state = {"actor": ap, "q1": q1, "q2": q2, "tq1": copy(q1), "tq2": copy(q2), "log_alpha": log_alpha,
             "actor_opt": adam_init(ap), "q1_opt": adam_init(q1), "q2_opt": adam_init(q2),
             "alpha_opt": adam_init(log_alpha), "n_upd": jnp.int32(0)}
    la_min = float(np.log(cfg.min_alpha))
    rscale = float(cfg.reward_scale)

    def clipq(v):
        return jnp.clip(jnp.nan_to_num(v, nan=0.0, posinf=cfg.q_clip_abs, neginf=-cfg.q_clip_abs),
                        -cfg.q_clip_abs, cfg.q_clip_abs)

    def sanitize(g):
        g = jax.tree_util.tree_map(lambda t: jnp.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0), g)
        n = jnp.sqrt(sum(jnp.sum(t * t) for t in jax.tree_util.tree_leaves(g)) + 1e-12)
        s = jnp.minimum(1.0, cfg.max_grad_norm / (n + 1e-6))
        return jax.tree_util.tree_map(lambda t: t * s, g), n

    def update(st, b, key):
        kc, ka_ = jax.random.split(key)
        alpha = jnp.exp(jnp.clip(st["log_alpha"], la_min, 5.0))
        na, nlp, _ = sample_actor_action(st["actor"], b["next_obs"], kc, scale, actor_cfg, action_low=low, action_high=high)
        na = jnp.clip(jnp.nan_to_num(na), low, high)
        nlp = jnp.clip(jnp.nan_to_num(nlp), -20.0, 20.0)
        tq = jnp.minimum(q_value(st["tq1"], b["next_obs"], na), q_value(st["tq2"], b["next_obs"], na)) - alpha * nlp
        y = jax.lax.stop_gradient(clipq(rscale * b["rew"] + cfg.gamma * (1.0 - b["terminal"]) * clipq(tq)))

        def closs(p1, p2):
            return (jnp.mean((clipq(q_value(p1, b["obs"], b["act"])) - y) ** 2)
                    + jnp.mean((clipq(q_value(p2, b["obs"], b["act"])) - y) ** 2))

        cl, (g1, g2) = jax.value_and_grad(closs, argnums=(0, 1))(st["q1"], st["q2"])
        g1, _ = sanitize(g1)
        g2, _ = sanitize(g2)
        q1n, o1 = adam_step(st["q1"], g1, st["q1_opt"], cfg.critic_lr)
        q2n, o2 = adam_step(st["q2"], g2, st["q2_opt"], cfg.critic_lr)

        n_act = jnp.maximum(st["n_upd"] - int(cfg.actor_start_update), 0).astype(jnp.float32)
        w_bc = cfg.bc_min + (cfg.bc_weight - cfg.bc_min) * jnp.clip(1.0 - (n_act - cfg.bc_hold) / max(cfg.bc_decay, 1), 0.0, 1.0)

        def aloss(p):
            a, lp, _ = sample_actor_action(p, b["obs"], ka_, scale, actor_cfg, action_low=low, action_high=high)
            a = jnp.clip(jnp.nan_to_num(a), low, high)
            lp = jnp.clip(jnp.nan_to_num(lp), -20.0, 20.0)
            qpi = clipq(jnp.minimum(q_value(q1n, b["obs"], a), q_value(q2n, b["obs"], a)))
            loss = jnp.mean(alpha * lp - qpi)
            if cfg.bc_weight > 0.0:
                m_bc = (actor_mean_action(p, b["obs"], scale, actor_cfg, action_low=low, action_high=high) - mid) / scale
                t_bc = (jax.vmap(pd_from_obs)(b["obs"]) - mid) / scale
                loss = loss + w_bc * jnp.mean(jnp.sum((m_bc - t_bc) ** 2, axis=-1))
            if cfg.caps_temporal > 0.0 or cfg.caps_spatial > 0.0:
                mu = lambda o: (actor_mean_action(p, o, scale, actor_cfg, action_low=low, action_high=high) - mid) / scale
                m0 = mu(b["obs"])
                if cfg.caps_temporal > 0.0:
                    loss = loss + cfg.caps_temporal * jnp.mean(jnp.sum((m0 - mu(b["next_obs"])) ** 2, axis=-1))
                if cfg.caps_spatial > 0.0:
                    noisy = b["obs"] + cfg.caps_sigma * jax.random.normal(jax.random.fold_in(ka_, 1), b["obs"].shape)
                    loss = loss + cfg.caps_spatial * jnp.mean(jnp.sum((m0 - mu(noisy)) ** 2, axis=-1))
            return loss, (lp, jnp.mean(qpi))

        (al, (lp, qpi)), ga = jax.value_and_grad(aloss, has_aux=True)(st["actor"])
        ga, gan = sanitize(ga)
        apn, aon = adam_step(st["actor"], ga, st["actor_opt"], cfg.actor_lr)
        ag = -jnp.mean(jax.lax.stop_gradient(lp + cfg.target_entropy))
        lan, alo = adam_step(st["log_alpha"], ag, st["alpha_opt"], cfg.alpha_lr)
        lan = jnp.clip(lan, la_min, 5.0)
        go = st["n_upd"] >= int(cfg.actor_start_update)
        keep = lambda new_, old_: jax.tree_util.tree_map(lambda n_, o_: jnp.where(go, n_, o_), new_, old_)
        new = {"actor": keep(apn, st["actor"]), "q1": q1n, "q2": q2n, "tq1": soft_update(st["tq1"], q1n, cfg.tau),
               "tq2": soft_update(st["tq2"], q2n, cfg.tau), "log_alpha": keep(lan, st["log_alpha"]),
               "actor_opt": keep(aon, st["actor_opt"]), "q1_opt": o1, "q2_opt": o2,
               "alpha_opt": keep(alo, st["alpha_opt"]), "n_upd": st["n_upd"] + 1}
        return new, {"critic_loss": cl, "actor_loss": al, "q_pi": qpi, "alpha": jnp.exp(lan), "actor_gn": gan, "w_bc": w_bc}

    metric_keys = ("critic_loss", "actor_loss", "q_pi", "alpha", "actor_gn", "w_bc")
    ep_keys = ("completed_return", "completed_len", "completed_pos_err", "completed_max_pos_err", "completed_lost",
               "completed_du")
    cap = int(cfg.replay_size)
    od = env.obs_dim
    zf = lambda *s: jnp.zeros((cap, *s), jnp.float32)
    rp0 = {"data": {"obs": zf(od), "act": zf(4), "rew": zf(), "next_obs": zf(od), "terminal": zf()},
           "ptr": jnp.int32(0), "size": jnp.int32(0)}
    max_due = max(1, int(np.ceil(n_env / cfg.update_every)))

    def vec_step(c, _):
        st, rp, es, obs, key, gstep = c
        key, kp, kr, ks, ku = jax.random.split(key, 5)
        a_pol, _, _ = sample_actor_action(st["actor"], obs, kp, scale, actor_cfg, action_low=low, action_high=high)
        a_rand = low + jax.random.uniform(kr, a_pol.shape, jnp.float32) * (high - low)
        u = jnp.clip(jnp.nan_to_num(jnp.where(gstep < cfg.start_steps, a_rand, a_pol)), low, high)
        es, obs_true, obs_out, rew, done, info = env.step_batched(es, u, jax.random.split(ks, n_env))
        idx = (rp["ptr"] + jnp.arange(n_env, dtype=jnp.int32)) % cap
        vals = {"obs": obs, "act": u, "rew": rew, "next_obs": obs_true, "terminal": info.terminal.astype(jnp.float32)}
        rp = {"data": {k: rp["data"][k].at[idx].set(vals[k].astype(jnp.float32)) for k in rp["data"]},
              "ptr": (rp["ptr"] + n_env) % cap, "size": jnp.minimum(rp["size"] + n_env, cap)}
        g2 = gstep + n_env
        lo = jnp.maximum(gstep + 1, cfg.update_after)
        due = jnp.where(g2 >= lo, g2 // cfg.update_every - (lo - 1) // cfg.update_every, 0)
        due = jnp.minimum(jnp.where(rp["size"] > cfg.batch_size, due, 0), max_due)
        m0 = {k: jnp.asarray(0.0, jnp.float32) for k in metric_keys} | {"n_upd": jnp.asarray(0.0, jnp.float32)}

        def body(i, cc):
            st_, k_, m_ = cc

            def do(cc2):
                st2, k2, m2 = cc2
                k2, kb, kup = jax.random.split(k2, 3)
                bi = jax.random.randint(kb, (cfg.batch_size,), 0, jnp.maximum(rp["size"], 1))
                b = {k: v[bi] for k, v in rp["data"].items()}
                st2, mm = update(st2, b, kup)
                return st2, k2, {k: m2[k] + mm[k] for k in metric_keys} | {"n_upd": m2["n_upd"] + 1.0}

            return jax.lax.cond(i < due, do, lambda z: z, (st_, k_, m_))

        st, ku, mu = jax.lax.fori_loop(0, max_due, body, (st, ku, m0))
        em = {k: jnp.sum(getattr(info, k)) for k in ep_keys} | {"n_ep": jnp.sum(info.episode_done.astype(jnp.float32))}
        return (st, rp, es, obs_out, key, g2), (mu, em)

    def chunk(c):
        c, (mu, em) = jax.lax.scan(vec_step, c, None, length=int(cfg.steps_per_jit))
        return c, (jax.tree_util.tree_map(jnp.sum, mu), jax.tree_util.tree_map(jnp.sum, em))

    chunk_j = jax.jit(chunk, donate_argnums=0)

    # ------------------------------------------------------------------ evaluation
    eval_fns, n_eval_steps = make_eval_fn(env, env_cfg, actor_cfg, low, high, scale, int(cfg.eval_episodes))
    half = cmd_half
    rest_start = int(round(5.0 / env_cfg.dt))

    def evaluate(actor_params, seed=777):
        ev = {}
        for i, (name, fn) in enumerate(eval_fns.items()):
            tr = jax.device_get(fn(actor_params, jax.random.PRNGKey(seed + i)))
            ev[name] = summarize_scenario(tr, half, rest_start)
        return ev

    history: Dict[str, list] = {}
    best = {"score": np.inf, "step": 0}

    def save(tag, st):
        with open(out_dir / f"{tag}_weights.pkl", "wb") as f:
            pickle.dump(jax.device_get({"actor_params": st["actor"], "q1_params": st["q1"], "q2_params": st["q2"],
                                        "target_q1_params": st["tq1"], "target_q2_params": st["tq2"],
                                        "log_alpha": st["log_alpha"]}), f)

    def log_eval(step, st):
        t0 = time.time()
        ev = evaluate(st["actor"])
        score = robust_score(ev)
        history.setdefault("eval/step", []).append(step)
        history.setdefault("eval/score", []).append(score)
        for name, m in ev.items():
            for k, v in m.items():
                history.setdefault(f"eval/{name}/{k}", []).append(v)
        def fmt(n, m):
            lost = "" if m["lost_rate"] == 0 else f" (lost {m['lost_rate']:.2f})"
            return f"{n} {m['pos_err_mean']:.3f}{lost}"

        line = "  ".join(fmt(n, m) for n, m in ev.items())
        print(f"  [eval {step}] score {score:.4f} | mean pos err per scenario [m]: {line}  ({time.time() - t0:.0f} s)",
              flush=True)
        if score < best["score"]:
            best.update(score=score, step=step, eval=ev)
            save("best", st)
        return ev

    with open(out_dir / "configs.json", "w") as f:
        json.dump({"sac": asdict(cfg), "env": env_cfg.as_dict(), "actor": asdict(actor_cfg),
                   "obs_layout": [list(x) for x in obs_layout(env_cfg)], "obs_dim": env.obs_dim,
                   "action_box": {"low": list(env_cfg.action_low), "high": list(env_cfg.action_high)},
                   "policy_box": {"low": np.asarray(low).tolist(), "high": np.asarray(high).tolist()},
                   "policy_mode": env_cfg.policy_mode,
                   "reference": load_reference(env_cfg.reference_path)["path"],
                   "eval_scenarios": EVAL_SCENARIOS}, f, indent=2)

    es, obs = env.reset_batched(jax.random.split(ke, n_env))
    own = lambda t: jax.tree_util.tree_map(lambda v: jnp.array(v, copy=True), t)  # distinct buffers (donated)
    carry = own((state, rp0, es, obs, key, jnp.int32(0)))
    log_eval(0, carry[0])
    t0 = time.time()
    last_t, last_s = t0, 0
    next_eval, next_log = int(cfg.eval_every), int(cfg.log_every)
    agg = None
    per_chunk = n_env * int(cfg.steps_per_jit)
    total = int(cfg.total_steps)
    done_steps = 0
    stop_file = out_dir / "STOP"
    while done_steps < total:
        if stop_file.exists():
            print(f"[stop] {stop_file} found at step {done_steps}: finishing", flush=True)
            break
        carry, (mu, em) = chunk_j(carry)
        done_steps += per_chunk
        mu, em = jax.device_get((mu, em))
        agg = {"mu": mu, "em": em} if agg is None else {"mu": {k: agg["mu"][k] + mu[k] for k in mu},
                                                         "em": {k: agg["em"][k] + em[k] for k in em}}
        if done_steps >= next_log:
            now = time.time()
            sps = (done_steps - last_s) / max(now - last_t, 1e-6)
            last_t, last_s = now, done_steps
            nu = max(float(agg["mu"]["n_upd"]), 1.0)
            ne = max(float(agg["em"]["n_ep"]), 1.0)
            rec = {"step": done_steps, "sps": sps, **{f"train/{k}": float(agg["mu"][k]) / nu for k in metric_keys},
                   **{f"episode/{k.replace('completed_', '')}": float(agg["em"][k]) / ne for k in ep_keys},
                   "episode/count": ne}
            for k, v in rec.items():
                history.setdefault(k, []).append(v)
            print(f"step={done_steps} sps={sps:.0f} ep_ret={rec['episode/return']:.1f} len={rec['episode/len']:.0f} "
                  f"pos_err={rec['episode/pos_err']:.3f} lost={rec['episode/lost']:.3f} du={rec['episode/du']:.3f} | "
                  f"critic={rec['train/critic_loss']:.3g} q_pi={rec['train/q_pi']:.2f} alpha={rec['train/alpha']:.3g}",
                  flush=True)
            agg = None
            next_log += int(cfg.log_every)
        if done_steps >= next_eval:
            log_eval(done_steps, carry[0])
            save("final", carry[0])
            (out_dir / "history.json").write_text(json.dumps(history))
            next_eval += int(cfg.eval_every)
    save("final", carry[0])
    final_ev = log_eval(done_steps, carry[0]) if history["eval/step"][-1] != done_steps else best.get("eval")
    (out_dir / "history.json").write_text(json.dumps(history))
    summary = {"total_steps": done_steps, "wall_time_sec": time.time() - t0, "best_step": best["step"],
               "best_score": best["score"], "best_eval": best.get("eval"), "final_eval": final_ev,
               "selection": "lowest robust_score (mean position error over EVAL_SCENARIOS + 1 m x lost rate)"}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    return summary


__all__ = ["EVAL_SCENARIOS", "SlowTrackSACConfig", "make_eval_fn", "robust_score", "run_slow_track_sac",
           "summarize_scenario"]
