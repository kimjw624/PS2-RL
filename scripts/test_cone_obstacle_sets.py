#!/usr/bin/env python
"""Self-checks of the cone-obstacle sets and filter (CPU, ~1 min).

    python scripts/test_cone_obstacle_sets.py [--certificate outputs/cone_obstacle/certificate.json]

1. config: round trip, target-cylinder fields and their validation
2. safe set: analytic gradients against autodiff
3. every design: sampled states in B lie in S (B subset of S)
4. "anywhere": a level vehicle at rest is in B wherever it is clear of the funnel and the ground
5. the filter (QP + discrete safeguard) returns finite, in-box commands
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cone_obstacle_common import design_config, load_certificate, make_config, resolve  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from ps2rl.cil.quadrotor_cone_obstacle_cbf import ConeObstacleBCBFConfig, solve_backup_cbf_qp_batch_with_info  # noqa: E402
from ps2rl.envs.quadrotor_cone_obstacle_config import ConeObstacleConfig  # noqa: E402
from ps2rl.sets.quadrotor_cone_obstacle_sets import obstacle_sets  # noqa: E402

DESIGNS = ("band@max", "points:approach_beside@max", "points:approach_beside+under_tip@max", "anywhere@2", "anywhere@8")


def random_states(rng, n, tip, v_std=0.6, att_std=0.15):
    p = np.stack([rng.uniform(-3.0, 3.0, n), rng.uniform(-0.6, 0.6, n), rng.uniform(0.0, tip + 2.5, n)], 1)
    v = rng.normal(0.0, v_std, (n, 3))
    half = 0.5 * rng.normal(0.0, att_std, (n, 3))
    q = np.concatenate([np.sqrt(np.clip(1 - (half**2).sum(1, keepdims=True), 0, 1)), half], 1)
    return jnp.asarray(np.concatenate([p, v, q], 1), jnp.float32)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vehicle", default="docker_batch/configs/landing_x500_cone45_r0p3.json")
    ap.add_argument("--certificate", default="outputs/cone_obstacle/certificate.json")
    a = ap.parse_args(argv)
    if not resolve(a.certificate).exists():
        sys.exit(f"run scripts/cone_obstacle_certify.py first (no {a.certificate})")
    cert = load_certificate(a.certificate)
    cfg = make_config(a.vehicle, tip_z=1.2)
    rng = np.random.default_rng(0)

    # 1. config
    assert ConeObstacleConfig.from_dict(cfg.as_dict()) == cfg
    assert cfg.cylinder_top == cfg.tip_z and cfg.replace(target_height=0.9).cylinder_top == 0.9
    assert cfg.touches_target((0.1, 0.0, 1.0)) and not cfg.touches_target((0.1, 0.0, 1.3))
    for bad in ({"target_height": 1.5}, {"anywhere_clearance_factor": 0.9}, {"target_radius": -0.1}):
        try:
            cfg.replace(**bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{bad} should be rejected")
    print("1. config ok")

    # 2. safe-set gradients
    safe, _ = obstacle_sets(cfg)
    x = random_states(rng, 256, cfg.tip_z)
    vals, grads = jax.vmap(safe.values_and_grads)(x)
    auto = jax.vmap(jax.jacfwd(lambda s: safe.component_values(s)))(x)
    assert np.allclose(np.asarray(vals), np.asarray(jax.vmap(safe.component_values)(x)), atol=1e-5)
    assert np.allclose(np.asarray(grads), np.asarray(auto), atol=1e-4)
    print("2. safe-set gradients ok")

    # 3. B subset of S
    x = random_states(rng, 200_000, cfg.tip_z, v_std=0.4, att_std=0.08)
    for d in DESIGNS:
        cd = design_config(cfg, d, cert)
        s, b = obstacle_sets(cd)
        in_b = np.asarray(jax.vmap(b.contains)(x))
        in_s = np.asarray(jax.vmap(s.contains)(x))
        assert in_b.sum() > 100, f"{d}: too few samples in B ({in_b.sum()})"
        assert not (in_b & ~in_s).any(), f"{d}: {(in_b & ~in_s).sum()} states in B outside S"
        print(f"3. {d:<38s} B subset of S on {in_b.sum()} samples in B")

    # 4. "anywhere": at rest wherever clear
    _, b = obstacle_sets(design_config(cfg, "anywhere@2", cert))
    xs, zs = np.meshgrid(np.linspace(-3.0, 3.0, 61), np.linspace(0.02, cfg.tip_z + 2.0, 41))
    rest = np.zeros((xs.size, 10), np.float32)
    rest[:, 0], rest[:, 2], rest[:, 6] = xs.ravel(), zs.ravel(), 1.0
    rest = jnp.asarray(rest)
    clear = np.asarray(jax.vmap(safe.value)(rest)) > 0.02
    in_b = np.asarray(jax.vmap(b.contains)(rest))
    assert in_b[clear].all(), f"{(~in_b[clear]).sum()} clear rest states outside B"
    print(f"4. anywhere: all {clear.sum()} clear rest states are in B")

    # 5. filter
    for d in ("band@max", "anywhere@2"):
        bcfg = ConeObstacleBCBFConfig(scenario=design_config(cfg, d, cert), alpha=10.0, alpha_ground=20.0)
        xb = jnp.asarray([[-0.8, 0.0, 1.5, 0.8, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [-2.0, 0.2, 2.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]], jnp.float32)
        ur = jnp.asarray([[cfg.gravity, 0.0, 0.5, 0.0], [cfg.gravity + 1.0, 0.2, 0.2, 0.0]], jnp.float32)
        u, _, _, info = solve_backup_cbf_qp_batch_with_info(xb, ur, bcfg)
        u = np.asarray(u)
        assert np.isfinite(u).all()
        assert (u >= np.asarray(cfg.action_low) - 1e-4).all() and (u <= np.asarray(cfg.action_high) + 1e-4).all()
        print(f"5. filter {d}: u = {np.round(u, 3).tolist()}, safeguard lambda = {np.asarray(info['safeguard_lambda']).tolist()}")
    print("all checks passed")


if __name__ == "__main__":
    main()
