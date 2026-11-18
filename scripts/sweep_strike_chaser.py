#!/usr/bin/env python
"""Chaser threat sweep: how dangerous the chaser is for a nominal controller that ignores it (no safety filter), over a
grid of chaser radius, top speed and acceleration, on the evaluation scenes. Fast (no filter): use it to pick a chaser
that matters, then check the deployed filter at the chosen settings with scripts/eval_strike_filter.py (env overrides
actual_chaser_radius / actual_chaser_speed / chaser_accel; the filter then assumes the same chaser).

    python scripts/sweep_strike_chaser.py --backup_run checkpoints/strike_reduced/phase1_td3_a0p2_seed0 \
        --task docker_batch/configs/strike_final_ps2.json --nominal striker --striker_run checkpoints/strike_reduced/ps2_striker_s0 \
        --radii 0.4 0.7 1.0 1.5 --speeds 2 3 4 5 --accel_per_speed 1.0 --out outputs/strike_chaser_sweep

Writes <out>/chaser_sweep.md / .json: per (radius, speed, accel) the strike rate, the chaser-contact rate, the other
contacts, the time-ups, and the chaser-contact rate among the episodes that did not end at an obstacle / the floor first.
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

from ps2rl.cil.strike_filter import model_from_backup  # noqa: E402
from ps2rl.envs.strike_env import StrikeEnvConfig, action_bounds, build_strike_env, generate_scene_pool  # noqa: E402
from ps2rl.envs.strike_pursuit import make_pursuit  # noqa: E402
from ps2rl.envs.strike_scene import StrikeConfig, build_model, check_same_plant, load_strike_task  # noqa: E402
from ps2rl.utils.policy import _coerce_actor_cfg, actor_mean_action  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backup_run", required=True, help="Phase-I run (its strike config: plant and the trained chaser)")
    p.add_argument("--task", default="docker_batch/configs/strike_final_ps2.json", help="env block: arena, scenes, episode length")
    p.add_argument("--env", default="", help="env overrides as JSON")
    p.add_argument("--nominal", choices=("pursuit", "striker"), default="pursuit")
    p.add_argument("--striker_run", default="")
    p.add_argument("--cruise", type=float, default=6.0, help="pursuit speed [m/s] (--nominal pursuit)")
    p.add_argument("--radii", type=float, nargs="+", default=[0.4, 0.7, 1.0, 1.5], help="chaser radii [m]")
    p.add_argument("--speeds", type=float, nargs="+", default=[2.0, 3.0, 4.0, 5.0], help="chaser top speeds [m/s]")
    p.add_argument("--accels", type=float, nargs="+", default=[2.0], help="chaser accelerations [m/s^2] (unless --accel_per_speed)")
    p.add_argument("--accel_per_speed", type=float, default=0.0,
                   help="> 0: acceleration = max(min(--accels), this * speed) [1/s], one value per speed")
    p.add_argument("--episodes", type=int, default=128)
    p.add_argument("--scene_seed", type=int, default=9001)
    p.add_argument("--out", default="outputs/strike_chaser_sweep")
    a = p.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    m = model_from_backup(a.backup_run)
    task = load_strike_task(a.task)
    ec0 = StrikeEnvConfig.from_dict({**task.get("env", {}), **(json.loads(a.env) if a.env else {})})
    m_env = m
    if a.nominal == "striker":
        sc = json.loads((Path(a.striker_run) / "configs.json").read_text())
        scfg = StrikeConfig.from_dict(sc["strike"])
        if scfg != m.cfg:
            check_same_plant(scfg, m.cfg, "striker run vs backup run")
            m_env = build_model(scfg)
        ec0 = ec0.replace(**{k: sc["env"][k] for k in ("obs_hide_scene", "obs_guidance", "obs_pursuit_cruise") if k in sc["env"]})
        acfg = _coerce_actor_cfg(sc["actor"])
        with open(Path(a.striker_run) / "best_weights.pkl", "rb") as f:
            sp = pickle.load(f)["actor_params"]
        sec = StrikeEnvConfig.from_dict(sc["env"])
        s_lo, s_hi = action_bounds(m_env, sec)
        base = jax.vmap(make_pursuit(m_env, sec.residual_cruise)) if sec.residual_cruise > 0.0 else None
        act = lambda obs: jnp.clip(actor_mean_action(sp, obs, m_env.scale, acfg, action_low=s_lo, action_high=s_hi), s_lo, s_hi)
        nominal = ((lambda es, obs: jnp.clip(base(es.x, es.target_p, es.target_v) + act(obs), m_env.low, m_env.high))
                   if base is not None else (lambda es, obs: act(obs)))
        label = f"striker {Path(a.striker_run).name}"
    else:
        pur = jax.vmap(make_pursuit(m_env, a.cruise))
        nominal = lambda es, obs: pur(es.x, es.target_p, es.target_v)
        label = f"pursuit at {a.cruise:g} m/s"
    ec0 = ec0.replace(residual_cruise=0.0)
    n = int(a.episodes)
    pool = generate_scene_pool(ec0, max(ec0.obstacles_max, m.cfg.n_obstacle_slots), n=n, seed=a.scene_seed)
    key = jax.random.PRNGKey(123)

    grid = []
    for r in a.radii:
        for v in a.speeds:
            accs = [max(min(a.accels), a.accel_per_speed * v)] if a.accel_per_speed > 0 else a.accels
            grid += [(float(r), float(v), float(acc)) for acc in accs]

    rows = []
    for r, v, acc in grid:
        t0 = time.time()
        ec = ec0.replace(actual_chaser_radius=r, actual_chaser_speed=v, chaser_accel=acc)
        env = build_strike_env(m_env, ec, pool=pool)

        def run(k):
            es, obs = env.reset_batched(jax.random.split(k, n))
            z = jnp.zeros((n,))
            acc0 = {"strike": z, "crash_obstacle": z, "crash_floor": z, "crash_chaser": z, "out_of_arena": z, "timeout": z,
                    "min_chaser": z + 1e3}

            def body(c, i):
                es, obs, alive, ac = c
                es2, _, obs2, _, done, info = env.step_batched(es, nominal(es, obs), jax.random.split(jax.random.fold_in(k, i), n))
                al = alive.astype(jnp.float32)
                ac = {kk: (jnp.maximum(ac[kk], al * getattr(info, kk).astype(jnp.float32)) if kk != "min_chaser" else
                           jnp.where(alive, jnp.minimum(ac[kk], info.min_chaser_dist), ac[kk])) for kk in ac}
                return (es2, obs2, alive & ~done, ac), None

            (_, _, _, ac), _ = jax.lax.scan(body, (es, obs, jnp.ones((n,), bool), acc0), jnp.arange(env.max_steps))
            return ac

        ac = jax.device_get(jax.jit(run)(key))
        s, ch, ob, fl, ou, to = (np.asarray(ac[k]) > 0.5 for k in ("strike", "crash_chaser", "crash_obstacle", "crash_floor",
                                                                    "out_of_arena", "timeout"))
        not_static = ~(ob | fl)
        row = {"radius": r, "speed": v, "accel": acc, "strike": float(s.mean()), "chaser_contact": float(ch.mean()),
               "obstacle_contact": float(ob.mean()), "floor_contact": float(fl.mean()), "out": float(ou.mean()),
               "timeout": float(to.mean()),
               "chaser_contact_given_no_static_crash": float(ch[not_static].mean()) if not_static.any() else float("nan"),
               "min_chaser_dist_median": float(np.median(np.asarray(ac["min_chaser"]))),
               "seconds": time.time() - t0}
        rows.append(row)
        print(f"r {r:4.2f} m  v {v:4.2f} m/s  a {acc:4.2f} m/s^2 | strike {row['strike']:.3f} chaser {row['chaser_contact']:.3f} "
              f"(of runs without an obstacle/floor crash: {row['chaser_contact_given_no_static_crash']:.3f}) obstacle "
              f"{row['obstacle_contact']:.3f} floor {row['floor_contact']:.3f} time-up {row['timeout']:.3f} | median min "
              f"chaser distance {row['min_chaser_dist_median']:.2f} m | {row['seconds']:.0f} s", flush=True)

    lines = [f"Chaser threat sweep: nominal {label}, no safety filter, {n} scenes (seed {a.scene_seed}), "
             f"{ec0.episode_seconds:g} s episodes. Chaser contact | given = among episodes without an obstacle / floor crash.", "",
             "| radius [m] | speed [m/s] | accel [m/s²] | strike | chaser contact | chaser given no static crash | obstacle | floor | time-up | median min distance [m] |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['radius']:g} | {r['speed']:g} | {r['accel']:g} | {r['strike']:.3f} | {r['chaser_contact']:.3f} | "
                     f"{r['chaser_contact_given_no_static_crash']:.3f} | {r['obstacle_contact']:.3f} | {r['floor_contact']:.3f} | "
                     f"{r['timeout']:.3f} | {r['min_chaser_dist_median']:.2f} |")
    (out / "chaser_sweep.md").write_text("\n".join(lines) + "\n")
    (out / "chaser_sweep.json").write_text(json.dumps({"nominal": label, "episodes": n, "scene_seed": a.scene_seed,
                                                       "task": a.task, "rows": rows}, indent=2))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
