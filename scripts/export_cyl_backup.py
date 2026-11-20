#!/usr/bin/env python
"""Export a Phase-I backup of the obstacle-only online task to the numpy format the onboard filter loads.

    JAX_PLATFORMS=cpu python scripts/export_cyl_backup.py --run outputs/cyl_online_phase1/td3_w64_seed0 \\
        --out checkpoints/cyl_online/td3_w64_seed0
    JAX_PLATFORMS=cpu python scripts/export_cyl_backup.py --lqr docker_batch/configs/cyl_online_p1.json \\
        --out checkpoints/cyl_online/lqr                       # hover-LQR backup only (no network)

Writes ``backup.json`` (hover LQR, base-set extents, action box, actor layout, the run's summary and certificate) and
``backup_weights.npz`` (actor layers; the log-std head is dropped), then checks the export against the JAX code it
came from (``ps2rl.envs.strike_scene`` / ``strike_sa.make_actor_backup``) on random states and scenes: observation,
backup action, plant step and the safe / base margins must agree to float32 precision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import pickle
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from ps2rl.online.model import Actor, CylModel, load_backup, save_backup  # noqa: E402


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_against_jax(export_dir: Path, n: int = 512, seed: int = 0) -> dict:
    """Largest numpy - JAX differences on random states / scenes (see the module)."""
    import jax
    import jax.numpy as jnp

    from ps2rl.envs.strike_scene import StrikeConfig, build_model, make_obs, margins, plant_step
    from ps2rl.online import model as om
    from ps2rl.phase1_sa.strike_sa import make_actor_backup
    from ps2rl.utils.policy import ActorConfig

    m_np, actor_np, meta = load_backup(export_dir)
    cfg = StrikeConfig.from_dict(m_np.strike)
    m = build_model(cfg)
    rng = np.random.default_rng(seed)
    K = cfg.n_obstacle_slots
    tilt = rng.uniform(0, 0.4, n)
    ax = rng.uniform(-np.pi, np.pi, n)
    yaw = rng.uniform(-0.2, 0.2, n)
    rv = np.stack([tilt * np.cos(ax), tilt * np.sin(ax), yaw], 1)
    ang = np.linalg.norm(rv, axis=1, keepdims=True) + 1e-12
    q = np.concatenate([np.cos(ang / 2), np.sin(ang / 2) * rv / ang], 1)
    x = np.concatenate([rng.uniform(-2, 2, (n, 2)), rng.uniform(0.6, 3.0, (n, 1)), rng.uniform(-2, 2, (n, 3)), q], 1)
    obst = np.zeros((n, K, 4))
    obst[:, :, 0:2] = x[:, None, 0:2] + rng.uniform(-5, 5, (n, K, 2))
    obst[:, :, 2] = rng.uniform(0.15, 0.6, (n, K))
    obst[:, :, 3] = rng.uniform(0, 1, (n, K)) < 0.7
    view = om.local_view(m_np, x, obst)
    u = np.concatenate([rng.uniform(4.0, 16.0, (n, 1)), rng.uniform(-3, 3, (n, 3))], 1)

    def scene_of(v):                                   # strike scene rows (cx, cy, bx, by, yaw, rr, present)
        s = np.zeros((v.shape[0], K, 7))
        s[:, :, 0:2], s[:, :, 5], s[:, :, 6] = v[:, :, 0:2], v[:, :, 2], v[:, :, 3]
        return s.reshape(v.shape[0], -1).astype(np.float32)

    sc = scene_of(view)
    xj, sj, uj = jnp.asarray(x, jnp.float32), jnp.asarray(sc), jnp.asarray(u, jnp.float32)
    out = {}
    o_j = np.asarray(jax.vmap(lambda a, b: make_obs(m, a, b))(xj, sj))
    out["observation"] = float(np.abs(o_j - om.observation(m_np, x, view)).max())
    xn_j = np.asarray(jax.vmap(lambda a, b: plant_step(m, a, b))(xj, uj))
    out["plant_step"] = float(np.abs(xn_j - om.step(m_np, x, u)).max())
    s_j, b_j = map(np.asarray, jax.vmap(lambda a, b: margins(m, a, b))(xj, sj))
    s_n, b_n = om.margins(m_np, x, view)
    out["safe_margin"] = float(np.abs(s_j - s_n).max())
    out["base_margin"] = float(np.abs(np.minimum(s_j, b_j) - b_n).max())
    from ps2rl.envs.strike_scene import lqr_action
    out["lqr_action"] = float(np.abs(np.asarray(jax.vmap(lambda a: lqr_action(m, a))(xj)) - om.lqr_action(m_np, x)).max())
    if actor_np is not None:
        run = Path(meta["run"])
        run = run if run.is_absolute() else ROOT / run
        with open(run / "strike_backup_policy_actor.pkl", "rb") as f:
            ck = pickle.load(f)
        acfg = ck["actor_cfg"]
        acfg = acfg if isinstance(acfg, ActorConfig) else ActorConfig(**acfg)
        back = make_actor_backup(m, acfg)
        ub_j = np.asarray(jax.vmap(lambda a, b: back(ck["actor_params"], a, b))(xj, sj))
        out["backup_action"] = float(np.abs(ub_j - actor_np(x, view)).max())
    return out


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--run", help="Phase-I run directory (scripts/train_phase1_strike.py output)")
    src.add_argument("--lqr", help="task file: export the hover-LQR backup only (no network)")
    p.add_argument("--out", required=True)
    p.add_argument("--no_check", action="store_true")
    a = p.parse_args(argv)
    out = Path(a.out)
    if a.run:
        run = Path(a.run).resolve()
        cfgs = json.loads((run / "configs.json").read_text())
        m = CylModel.from_strike_config(cfgs["strike"])
        with open(run / "strike_backup_policy_actor.pkl", "rb") as f:
            ck = pickle.load(f)
        acfg = ck["actor_cfg"]
        act = acfg.get("activation", "relu") if isinstance(acfg, dict) else getattr(acfg, "activation", "relu")
        layers = [(np.asarray(l["w"]), np.asarray(l["b"])) for l in ck["actor_params"]["layers"]]
        actor = Actor(m, layers, act, np.asarray(ck["action_low"]), np.asarray(ck["action_high"]))
        summary = json.loads((run / "summary.json").read_text())
        cert = json.loads((run / "certificate.json").read_text()) if (run / "certificate.json").exists() else None
        try:
            run_ref = str(run.relative_to(ROOT))           # inside the repository: a path that works on any clone
        except ValueError:
            run_ref = str(run)
        meta = {"run": run_ref, "kind": "learned", "region": cfgs.get("region"), "sa": cfgs.get("sa"),
                "certificate": cert, "best_eval_step": summary.get("best_eval_step"),
                "test_at_best": summary.get("test_at_best"), "lqr_test": summary.get("lqr_test"),
                "source_sha256": _sha(run / "strike_backup_policy_actor.pkl")}
    else:
        task = json.loads(Path(a.lqr).read_text())
        m = CylModel.from_strike_config(task["strike"])
        actor = None
        meta = {"kind": "lqr", "task_file": str(a.lqr)}
    save_backup(out, m, actor, meta)
    m2, actor2, _ = load_backup(out)
    width = "hover LQR only" if actor2 is None else f"actor {actor2.width} wide, {actor2.flops / 1e3:.1f} kFLOP per state"
    print(f"[export] {out}: {width}; base set c_B {m2.c_B:g}, hover ellipsoid {m2.rho_xy:.2f} m / {m2.z_ext:.2f} m, "
          f"box thrust {m2.low[0]:.2f}..{m2.high[0]:.2f} m/s^2, rates {m2.high[1]:.2f}/{m2.high[3]:.2f} rad/s")
    if not a.no_check:
        diffs = check_against_jax(out)
        worst = max(diffs.values())
        print("[check] numpy vs JAX max |diff|: " + " ".join(f"{k} {v:.1e}" for k, v in diffs.items())
              + ("  OK" if worst < 2e-3 else "  MISMATCH"))
        meta2 = json.loads((out / "backup.json").read_text())
        meta2["meta"]["twin_check"] = diffs
        (out / "backup.json").write_text(json.dumps(meta2, indent=2))
        if worst >= 2e-3:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
