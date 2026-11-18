#!/usr/bin/env python
"""Recoverability of a landing backup on its design region, next to the hover LQR alone.

Uses Phase I's own evaluation (held-out test states per region; a start counts if the backup
reaches the base set within the backup horizon without leaving the training safe set), so the
numbers are the ones the trainer reports. The hover LQR alone, i.e. the base controller used
everywhere, needs no training: it is the baseline a learned backup has to beat to be worth using.

    # baseline of a design, before any training
    python scripts/evaluate_landing_backup_regions.py --config_json docker_batch/configs/landing_x500_cone45_r0p3.json

    # trained backups against that baseline (each on its own design region)
    python scripts/evaluate_landing_backup_regions.py --backup outputs/landing_x500/phase1/x500_rec10_td3_seed0 \\
        --out outputs/landing_x500/regions.json

``--region_override KEY=VAL`` evaluates on a changed design region (e.g. ``yaw_max_deg=45``);
``--by_yaw`` adds the recoverability per start-yaw bin.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ps2rl.backup_policy.quadrotor_learned_backup import load_learned_quadrotor_backup_policy
from ps2rl.cil import quadrotor_landing_backup_cbf as L
from ps2rl.envs.quadrotor_landing_config import QuadrotorLandingConfig
from ps2rl.evaluation.landing_phase2_run import parse_field_overrides
from ps2rl.phase1_sa.landing_design_region import REGION_NAMES, LandingDesignRegionConfig, heldout_sets
from ps2rl.phase1_sa.quadrotor_landing_sa_env import build_landing_sa_env
from ps2rl.phase1_sa.quadrotor_landing_sa_trainer import LandingSAConfig, build_landing_evaluator


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config_json", default="", help="Phase-I config (landing/region blocks): evaluates the hover LQR alone")
    p.add_argument("--backup", nargs="*", default=[], help="trained backups (run directories or actor .pkl files)")
    p.add_argument("--region_override", nargs="*", default=[], help="KEY=VAL changes of the design region for the evaluation")
    p.add_argument("--split", default="test", choices=("val", "test"))
    p.add_argument("--by_yaw", default="", help="bin edges of |start yaw| in deg, e.g. 0,10,20,30,45")
    p.add_argument("--out", default="")
    return p.parse_args(argv)


def evaluate(cfg: QuadrotorLandingConfig, region: LandingDesignRegionConfig, learned, split: str, yaw_edges: list[float]) -> dict:
    env = build_landing_sa_env(cfg, region)
    sa = LandingSAConfig()
    ctrl = env.base_set.controller
    out = {}
    policies = {"hover LQR alone": (None, None, lambda params, x: ctrl.action(x))}
    if learned is not None:
        policies = {"learned backup": (learned.actor_params, learned.actor_cfg, None), **policies}
    for name, (params, actor_cfg, raw_fn) in policies.items():
        run_eval = build_landing_evaluator(cfg, region, env, actor_cfg, beta=sa.beta, hoeffding_delta=sa.hoeffding_delta,
                                           raw_action_fn=raw_fn)
        out[name] = run_eval(params, split)
    if yaw_edges:  # per-yaw recoverability, from the same rollouts split by the start state
        import jax
        import jax.numpy as jnp
        from ps2rl.backup_policy.backup_policy import BackupPolicy
        from ps2rl.phase1_sa.quadrotor_landing_sa_env import landing_action_box, landing_step_fn
        from ps2rl.utils.policy import actor_mean_action

        scale, low, high = landing_action_box(cfg)
        step, safe, base = landing_step_fn(cfg), env.safe_set, env.base_set
        sets = heldout_sets(region, env.sampler, split=split)
        for name, (params, actor_cfg, raw_fn) in policies.items():
            raw = raw_fn or (lambda prm, x, c=actor_cfg: actor_mean_action(prm, x, scale, c, action_low=low, action_high=high))

            def rollout(x0, raw=raw, params=params):
                def body(c, _):
                    x, hb, hf = c
                    xn = step(x, BackupPolicy.select_action(x, jnp.clip(raw(params, x), low, high), base))
                    act = ~(hb | hf)
                    nf = act & ~safe.training_contains(xn)
                    return (jnp.where(act, xn, x), hb | (act & ~nf & base.contains(xn)), hf | nf), None
                (_, hb, _), _ = jax.lax.scan(body, (x0, False, False), None, length=int(cfg.num_steps))
                return hb

            ro = jax.jit(jax.vmap(rollout))
            for reg in REGION_NAMES:
                x0 = np.asarray(sets[reg])
                q = x0[:, 6:10]
                yaw = np.abs(np.degrees(np.arctan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]), 1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2))))
                hit = np.asarray(ro(jnp.asarray(x0)))
                out[name][reg]["by_abs_yaw_deg"] = {
                    f"{a:g}-{b:g}": (float(hit[(yaw >= a) & (yaw < b)].mean()) if ((yaw >= a) & (yaw < b)).any() else None)
                    for a, b in zip(yaw_edges[:-1], yaw_edges[1:])}
    return out


def show(title: str, cfg: QuadrotorLandingConfig, region: LandingDesignRegionConfig, res: dict) -> None:
    print(f"\n{title}")
    print(f"  action box thrust [{cfg.a_cmd_min:g}, {cfg.a_cmd_max:g}] m/s^2, rates {', '.join(f'{w:g}' for w in cfg.omega_limits)} rad/s; "
          f"c_B {cfg.base_set_c:g} at z_des {cfg.z_des:g}; region v_max {region.v_max:g} m/s, tilt {region.tilt_max_deg:g} deg, "
          f"yaw {region.yaw_max_deg:g} deg, height {region.zeta_max:g} m")
    print(f"  {'':18s} {'weighted':>9s} " + " ".join(f"{r:>9s}" for r in REGION_NAMES) + "   crash")
    for name, st in res.items():
        print(f"  {name:18s} {st['mu_weighted']:9.3f} " + " ".join(f"{st[r]['m_hat']:9.3f}" for r in REGION_NAMES) + f"   {st['crash_rate']:.3f}")
        for r in REGION_NAMES:
            if "by_abs_yaw_deg" in st[r]:
                print(f"    {r:8s} by |yaw|: " + "  ".join(f"{k}: {'–' if v is None else f'{v:.2f}'}" for k, v in st[r]["by_abs_yaw_deg"].items()))
    if "learned backup" in res:
        d = res["learned backup"]["mu_weighted"] - res["hover LQR alone"]["mu_weighted"]
        print(f"  learned - LQR alone (weighted): {d:+.3f}" + ("" if d > 0 else "   <- the learned backup does not beat the base controller here"))


def main(argv=None):
    args = parse_args(argv)
    if not args.config_json and not args.backup:
        raise SystemExit("give --config_json and/or --backup")
    yaw_edges = [float(v) for v in args.by_yaw.split(",")] if args.by_yaw else []
    over = parse_field_overrides(args.region_override, LandingDesignRegionConfig) if args.region_override else {}
    report = {}
    if args.config_json:
        d = json.loads(Path(args.config_json).read_text(encoding="utf-8"))
        cfg = QuadrotorLandingConfig().replace(**d.get("landing", {}))
        region = LandingDesignRegionConfig.from_dict({**d.get("region", d.get("design_region", {})), **over})
        res = evaluate(cfg, region, None, args.split, yaw_edges)
        show(args.config_json, cfg, region, res)
        report[args.config_json] = {"landing": cfg.as_dict(), "region": region.as_dict(), "results": res}
    for b in args.backup:
        cfg = L.landing_config_from_checkpoint(b)
        region = LandingDesignRegionConfig.from_dict({**L.read_checkpoint_metadata(b).get("design_region", {}), **over})
        actor = Path(b) / "landing_backup_policy_actor.pkl" if Path(b).is_dir() else Path(b)
        res = evaluate(cfg, region, load_learned_quadrotor_backup_policy(actor), args.split, yaw_edges)
        show(b, cfg, region, res)
        report[b] = {"landing": cfg.as_dict(), "region": region.as_dict(), "results": res}
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
