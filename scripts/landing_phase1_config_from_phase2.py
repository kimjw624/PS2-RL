"""Derive the Phase-I landing config that matches a Phase-II landing run.

The safe-arrival backup has to be trained for the same cone, pad, plant and action box as
the Phase-II task it will protect. This script reads them from the Phase-II run instead of
retyping them:

  * cone and pad   <- ``<reference>_config.json`` (written by generate_quadrotor_landing_reference.py)
  * dt, gravity, action box <- the run's ``configs.json`` env block
  * design region  <- widened to cover the reference envelope + the run's initial perturbation

and checks the base set (z_des, c_B) against the new cone (the Lyapunov part is
z-independent and is re-checked by ``train_phase1_landing.py``). If the configured z_des is
not certified for the new cone, the lowest certified z_des on a 5 cm grid is used.

    python scripts/landing_phase1_config_from_phase2.py \\
        --phase2_run outputs/landing/land_v2_miss_s1 --out outputs/landing_phase1_cfg/land_v2_miss.json
    python scripts/train_phase1_landing.py --config_json outputs/landing_phase1_cfg/land_v2_miss.json --backbone td3
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from ps2rl.base_controller.quadrotor_landing_dlqr import QuadrotorLandingDLQR
from ps2rl.envs.quadrotor_landing_config import QuadrotorLandingConfig
from ps2rl.evaluation.landing_phase2_run import load_phase2_run, phase1_overrides, reference_geometry
from ps2rl.phase1_sa.landing_design_region import LandingDesignRegionConfig
from ps2rl.sets import landing_certificate as lc
from ps2rl.sets.quadrotor_cone_sets import QuadrotorConeSafeSet


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--phase2_run", required=True, help="Phase-II run directory (has configs.json)")
    p.add_argument("--reference_config", default=None,
                   help="override <reference>_config.json (cone/pad the reference was designed for)")
    p.add_argument("--speed_margin", type=float, default=None,
                   help="design-region speed margin over the reference peak (default: the run's init_v_range)")
    p.add_argument("--z_search_max", type=float, default=4.0)
    p.add_argument("--out", required=True)
    for name in ("cone_r0", "cone_theta_deg", "cone_eps", "pad_x", "pad_y", "pad_z"):
        p.add_argument(f"--{name}", type=float, default=None, help="explicit geometry (if the reference has no config)")
    return p.parse_args(argv)


def certify_z(cfg: QuadrotorLandingConfig, z_max: float) -> tuple[QuadrotorLandingConfig, dict]:
    ctrl = QuadrotorLandingDLQR.from_config(cfg)
    p_inv = np.linalg.inv(ctrl.p_matrix_f64())
    c_b = float(cfg.base_set_c)
    c_u, c_chart = float(ctrl.max_certified_level), lc.c_chart_bound(p_inv)
    if c_b > min(c_u, c_chart) + 1e-9:
        raise SystemExit(f"c_B={c_b} exceeds c_U={c_u:.3f}/c_chart={c_chart:.3f}; lower base_set_c")

    def bounds(z):
        cone = QuadrotorConeSafeSet.from_config(cfg.replace(z_des=z))
        return (lc.c_cone_exact(p_inv, cone, z_des=z, c_hi=4.0 * c_b),
                lc.c_ground_bound(p_inv, z_des=z, z_clear=cfg.z_clear))

    z = float(cfg.z_des)
    c_cone, c_ground = bounds(z)
    moved = False
    while min(c_cone, c_ground) < c_b - 1e-9:
        z = round(z + 0.05, 4)
        if z > z_max:
            raise SystemExit(f"no certified z_des <= {z_max} for c_B={c_b}")
        c_cone, c_ground = bounds(z)
        moved = True
    info = {"z_des": z, "moved_from": float(cfg.z_des) if moved else None, "c_B": c_b,
            "c_U": c_u, "c_chart": c_chart, "c_cone": c_cone, "c_ground": c_ground}
    return cfg.replace(z_des=z), info


def main(argv=None) -> None:
    args = parse_args(argv)
    run = load_phase2_run(args.phase2_run, repo_root=REPO, reference_config=args.reference_config)
    explicit = {k: getattr(args, k) for k in ("cone_r0", "cone_theta_deg", "cone_eps", "pad_x", "pad_y", "pad_z")
                if getattr(args, k) is not None}
    geom = reference_geometry(run)
    if geom is None and len(explicit) < 6:
        raise SystemExit(f"{run.reference_path} has no _config.json; pass --reference_config or all six geometry flags")
    geom = {**(geom or {}), **explicit}

    out = phase1_overrides(run, geometry=geom, v_margin=args.speed_margin)
    cfg = QuadrotorLandingConfig().replace(**out["landing"])
    cfg, cert = certify_z(cfg, args.z_search_max)
    out["landing"]["z_des"] = cfg.z_des
    out["provenance"]["base_set_check"] = cert
    region = LandingDesignRegionConfig.from_dict(out["region"])

    cone = QuadrotorConeSafeSet.from_config(cfg)
    ref = np.load(run.reference_path)["states"]
    import jax.numpy as jnp
    h_ref = np.asarray(cone.value(jnp.asarray(ref, dtype=jnp.float32)))
    out["provenance"]["reference_cone_margin"] = {
        "h_start": float(h_ref[0]), "h_min": float(h_ref.min()),
        "steps_outside": int((h_ref < 0).sum()), "steps_total": int(h_ref.size),
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"Phase-II run      : {run.run_dir}")
    print(f"reference         : {run.reference_path}")
    print(f"reference config  : {run.reference_config_path}")
    print("landing overrides : " + ", ".join(f"{k}={v:g}" for k, v in out["landing"].items()))
    for k, why in out["provenance"]["region_reasons"].items():
        print(f"  region.{k:<13}: {why}")
    print(f"base set          : z_des={cert['z_des']} c_B={cert['c_B']}  c_U={cert['c_U']:.2f} "
          f"c_chart={cert['c_chart']:.2f} c_cone={cert['c_cone']:.2f} c_ground={cert['c_ground']:.2f}"
          + (f"  (moved from z_des={cert['moved_from']})" if cert["moved_from"] is not None else ""))
    m = out["provenance"]["reference_cone_margin"]
    print(f"reference vs cone : h(start)={m['h_start']:+.3f}  min h={m['h_min']:+.3f}  "
          f"outside {m['steps_outside']}/{m['steps_total']} samples")
    if m["h_start"] < 0:
        print("  !! the reference starts outside the cone: the filter cannot protect episodes that start there")
    print(f"region            : zeta<={region.zeta_max} |v|<={region.v_max} tilt<={region.tilt_max_deg} "
          f"yaw<={region.yaw_max_deg}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
