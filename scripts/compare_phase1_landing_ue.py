"""Score backups on the same held-out (x0, d_hat) pairs under the UE-tightened C_N test.

    python scripts/compare_phase1_landing_ue.py --ue outputs/landing_phase1_ue/ue_floor_rec10_td3_h128_seed0 \
        --nominal checkpoints/landing_phase1/floor_rec10_td3_seed0

Both backups are rolled out under the frozen-estimate flow (E_d d_hat on v_dot); the nominal
backup simply ignores d_hat. Indicators, tube and design-region sets are those of the UE run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from ps2rl.backup_policy.backup_policy import BackupPolicy  # noqa: E402
from ps2rl.backup_policy.quadrotor_learned_backup import load_learned_quadrotor_backup_policy  # noqa: E402
from ps2rl.envs.quadrotor_landing_config import QuadrotorLandingConfig  # noqa: E402
from ps2rl.phase1_sa.landing_design_region import REGION_NAMES, LandingDesignRegionConfig, heldout_sets  # noqa: E402
from ps2rl.phase1_sa.quadrotor_landing_sa_env import landing_action_box  # noqa: E402
from ps2rl.phase1_sa.quadrotor_landing_ue_sa_env import build_landing_ue_sa_env, build_ue_indicators, sample_d_hat  # noqa: E402
from ps2rl.uncertainty.landing_ue_tube import LandingUEConfig, make_growth_fn, tube_step  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--ue", required=True)
    p.add_argument("--nominal", default="checkpoints/landing_phase1/floor_rec10_td3_seed0")
    p.add_argument("--split", default="test")
    a = p.parse_args(argv)
    c = json.loads((Path(a.ue) / "configs.json").read_text())
    cfg = QuadrotorLandingConfig.from_dict(c["landing"])
    region = LandingDesignRegionConfig.from_dict(c["design_region"])
    ue = LandingUEConfig.from_dict(c["ue"])
    env = build_landing_ue_sa_env(cfg, region, ue)
    not_failed, goal = build_ue_indicators(env.safe_set, env.base_set, ue, env.tube)
    _, low, high = landing_action_box(cfg)
    sets = heldout_sets(region, env.sampler, split=a.split)
    n_steps, dt = int(cfg.num_steps), float(cfg.dt)
    policies = {}
    for name, path in (("ue", Path(a.ue) / "landing_backup_policy_actor.pkl"),
                       ("nominal", Path(a.nominal) / "landing_backup_policy_actor.pkl")):
        lp = load_learned_quadrotor_backup_policy(path)
        d_in = int(lp.actor_cfg.obs_dim) == 13

        def pi_b(x, dh, lp=lp, d_in=d_in):
            o = jnp.concatenate([x, dh]) if d_in else x
            raw = jnp.clip(lp.action_single(o), low, high)
            return jnp.clip(BackupPolicy.select_action(x, raw, env.base_set), low, high)

        policies[name] = pi_b
    out = {}
    for name, pi_b in policies.items():
        def rollout(x0, dh, tight, pi_b=pi_b):
            step = lambda z: env.plant_step(z, pi_b(z, dh), dh)
            growth = make_growth_fn(step, env.base_set.controller, env.tube)

            def body(cc, k):
                x, s, hb, hf = cc
                g = growth(x)
                xn = step(x)
                sn = jnp.where(tight, tube_step(s, g, k * dt, ue, env.tube, dt), 0.0)
                act = ~(hb | hf)
                nf = act & ~not_failed(xn, sn)
                nb = act & ~nf & goal(xn, sn)
                return (jnp.where(act, xn, x), jnp.where(act, sn, s), hb | nb, hf | nf), None

            z = jnp.asarray(0.0)
            (_, _, hb, _), _ = jax.lax.scan(body, (x0, z, env.base_set.contains(x0), ~not_failed(x0, z)), jnp.arange(n_steps))
            return hb

        rb = jax.jit(jax.vmap(rollout, in_axes=(0, 0, None)))
        res = {}
        for i, r in enumerate(REGION_NAMES):
            x0 = jnp.asarray(sets[r])
            keys = jax.random.split(jax.random.PRNGKey(4242 + 100 * (a.split == "test") + i), x0.shape[0])
            dh = jax.vmap(lambda k: sample_d_hat(k, ue))(keys)
            res[r] = {"tightened_with_d_hat": float(np.mean(rb(x0, dh, True))),
                      "untightened_d_hat_0": float(np.mean(rb(x0, jnp.zeros_like(dh), False)))}
        w = region.weights
        for kk in ("tightened_with_d_hat", "untightened_d_hat_0"):
            res[f"mu_w_{kk}"] = sum(w[r] * res[r][kk] for r in REGION_NAMES) / sum(w.values())
        out[name] = res
        print(name, json.dumps(res))
    (Path(a.ue) / "compare_backups.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
