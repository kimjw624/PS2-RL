#!/usr/bin/env python
# usage: python scripts/diag_strike_filter_loss.py <Phase-I run dir> [episodes]
"""At the first certificate loss of each episode: why does every candidate fail? For the backup's own plan from the current
state (lam = 0, fresh view) and for the committed continuation, report the first failing condition along the rollout:
contact (obstacle / floor / chaser ball) or no base set within N, with the base-set condition that binds at the end
(V_min, obstacle clearance, chaser clearance, floor clearance). Also the closest chaser's distance and closing speed."""
import sys
import numpy as np, jax, jax.numpy as jnp
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).resolve().parent))
from ps2rl.cil.strike_filter import backup_ranges, full_margins, load_backup, make_filter, model_from_backup
from ps2rl.envs.strike_env import StrikeEnvConfig, build_strike_env, generate_scene_pool
from ps2rl.envs.strike_scene import (load_strike_task, plant_step, lqr_action, obstacle_distances, relative_scene, attitude_chart,
                                     ball_distances)
from eval_strike_filter import pursuit_controller

U = sys.argv[1]
n = int(sys.argv[2]) if len(sys.argv) > 2 else 32
m = model_from_backup(U); cfg = m.cfg
ec = StrikeEnvConfig.from_dict(load_strike_task(str(Path(__file__).resolve().parents[1] / "docker_batch/configs/strike.json"))["env"])
pool = generate_scene_pool(ec, 10, n=n, seed=9001); env = build_strike_env(m, ec, pool=pool)
bk, prm = load_backup(m, "learned", U)
o_r, b_r = backup_ranges(U)
f1 = make_filter(m, bk, prm, obstacle_range=o_r, ball_range=b_r)
fv = jax.jit(jax.vmap(f1)); fresh = jax.jit(jax.vmap(f1.fresh_view))
pur = jax.jit(pursuit_controller(m, 6.0))
N = cfg.num_steps


def base_terms(x, obst, balls, hold):
    """(V_min slack, obstacle clearance slack, chaser clearance slack, floor clearance slack) at x."""
    o_rel, b_rel = relative_scene(x, obst, balls)
    r = jnp.concatenate([x[3:6], attitude_chart(x[6:10])])
    off = m.D @ r
    R_base = cfg.ball_radius + cfg.chaser_speed * (cfg.chaser_hold if hold is None else hold)
    return jnp.stack([cfg.c_B - r @ m.S @ r, jnp.min(obstacle_distances(off[0:2], o_rel)) - m.lqr.rho_xy - cfg.r_drone,
                      jnp.min(ball_distances(off, b_rel, R_base)) - m.lqr.rho, (x[2] - cfg.z_floor) + off[2] - m.lqr.z_ext])


def why(x0, scene, obst, balls, hold):
    """Composed backup from x0: first failure along the rollout."""
    x = x0
    for k in range(N + 1):
        s, b = full_margins(m, x, obst, balls, hold)
        if s < 0:
            o_rel, b_rel = relative_scene(x, obst, balls)
            d_o = float(jnp.min(obstacle_distances(jnp.zeros(2), o_rel))) - cfg.r_drone
            d_b = float(jnp.min(ball_distances(jnp.zeros(3), b_rel, cfg.ball_radius)))
            d_f = float(x[2] - cfg.z_floor)
            kind = ["obstacle", "chaser ball", "floor"][int(np.argmin([d_o, d_b, d_f]))]
            return f"contact with {kind} at step {k}"
        if b >= 0:
            return f"reaches B at step {k}"
        u = lqr_action(m, x) if b >= 0 else bk(prm, x, scene)
        x = plant_step(m, x, u)
    t = np.asarray(base_terms(x, obst, balls, hold))
    names = ["V_min", "obstacle clearance", "chaser clearance", "floor clearance"]
    return "no B within N; at the end " + ", ".join(f"{nm} {v:+.2f}" for nm, v in zip(names, t) if v < 0)


es, obs = env.reset_batched(jax.random.split(jax.random.PRNGKey(123), n))
comm = fresh(es.x, es.obst, es.chaser_p, es.chaser_on)
done_report = np.zeros(n, bool); alive = np.ones(n, bool)
for k in range(500):
    u_nom = pur(es.x, es.target_p, es.target_v)
    comm_before = comm
    u, info, comm = fv(es.x, u_nom, es.obst, es.chaser_p, es.chaser_on, comm)
    nc = np.asarray(info.no_certificate)
    for i in np.nonzero(nc & alive & ~done_report)[0]:
        done_report[i] = True
        x, obst = es.x[i], es.obst[i]
        balls = jnp.concatenate([es.chaser_p[i], es.chaser_on[i][:, None]], axis=1)
        sc_f = f1.backup_view(f1.fresh_view(x, obst, es.chaser_p[i], es.chaser_on[i]))
        on = np.asarray(es.chaser_on[i]) > 0.5
        rel = np.asarray(es.chaser_p[i] - x[None, :3]); dist = np.linalg.norm(rel, axis=1)
        closing = -np.sum(rel * (np.asarray(es.chaser_v[i]) - np.asarray(x[3:6])[None]), axis=1) / np.maximum(dist, 1e-9)
        j = int(np.argmin(np.where(on, dist, 1e9))) if on.any() else -1
        ch = f"nearest chaser {dist[j]:.2f} m, closing {closing[j]:+.2f} m/s" if j >= 0 else "no chaser"
        xb = plant_step(m, x, bk(prm, x, sc_f) if float(full_margins(m, x, obst, balls)[1]) < 0 else lqr_action(m, x))
        print(f"ep {i:2d} t {k * 0.02:.2f} s | speed {float(jnp.linalg.norm(x[3:6])):.1f} m/s, obstacle {float(jnp.min(obstacle_distances(x[0:2], obst))):.2f} m, {ch}")
        print(f"      backup's own plan (fresh view): {why(xb, sc_f, obst, balls, None)}")
        print(f"      committed continuation:         {why(plant_step(m, x, info.u_backup[i]), f1.backup_view(comm_before[i]), obst, balls, 0.0)}")
    es, _, obs, _, done, inf = env.step_batched(es, u, jax.random.split(jax.random.PRNGKey(1000 + k), n))
    alive &= ~np.asarray(done)
    if not alive.any():
        break
print(f"{done_report.sum()} of {n} episodes lost the certificate")
