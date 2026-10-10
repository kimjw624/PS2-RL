#!/usr/bin/env python
"""Evaluate a Phase-II cone-obstacle policy under the disturbance, through different safety layers.

    JAX_ENABLE_X64=1 python scripts/eval_phase2_cone_obstacle_ue.py \
        --run outputs/cone_obstacle_phase2_ue/cone_ue_vanilla_s0 --modes none,ue --episodes 128
    JAX_ENABLE_X64=1 python scripts/eval_phase2_cone_obstacle_ue.py \
        --run outputs/cone_obstacle_phase2_ue/cone_ue_ps2_s0 --modes ue --episodes 128

Same policy, same initial states and disturbance draws (``--seed``), executed through
  ue       the cone-obstacle UE-bCBF CIL (tube, observer term) with the run's backup (or ``--backup``)
  nominal  the plain cone-obstacle BCBF (LQR backup, no disturbance model)
  none     no filter
Reports safety (episodes entering the funnel or going below the floor, min h_cone, min floor
margin), the task (touched = reached the target cylinder under the funnel tip, first-touch
time, final distance to the reference's end point), CIL statistics; writes eval.json and
trajectories.npz (read by ``scripts/plot_cone_obstacle_ue_progress.py``).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
import pickle
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from cone_obstacle_common import task_config  # noqa: E402
from ps2rl.cil import cone_obstacle_ue_bcbf as cu  # noqa: E402
from ps2rl.cil import quadrotor_cone_obstacle_cbf as ncbf  # noqa: E402
from ps2rl.envs.cone_obstacle_ue_env import ConeObstacleUEEnvConfig, build_cone_obstacle_ue_env  # noqa: E402
from ps2rl.phase2_ps2.cone_obstacle_ue_ps2_trainer import ConeUEPS2Config, filter_config  # noqa: E402
from ps2rl.utils.policy import ActorConfig, actor_mean_action  # noqa: E402

MODES = ("ue", "nominal", "none")


def load_run(run: Path, backup: str = "", backup_path: str = "") -> dict:
    cfgs = json.loads((run / "configs.json").read_text())
    ps2 = dict(cfgs["ps2"])
    if backup:
        ps2["backup"] = backup
        ps2["backup_path"] = backup_path
    act_d = dict(cfgs["actor"])
    act_d["hidden_sizes"] = tuple(act_d["hidden_sizes"])
    return {"ps2": ConeUEPS2Config.from_dict(ps2), "env_cfg": ConeObstacleUEEnvConfig(**cfgs["env"]),
            "actor_cfg": ActorConfig(**act_d), "task": cfgs["task"], "task_file": cfgs.get("task_file", "")}


def outcome(tr: dict, dt: float, end_p: np.ndarray, scale: np.ndarray) -> dict:
    al = tr["alive"].astype(bool)
    alf = al.astype(float)
    h = np.where(al, tr["h"], np.inf)
    z = np.where(al, tr["z"], np.inf)
    unsafe = ((h < 0) | (z < 0)).any(0)
    touch = (tr["touch"] > 0.5) & al
    touched = touch.any(0)
    t_touch = np.where(touched, np.argmax(touch, axis=0) + 1, 0) * dt
    last = np.maximum(al.sum(0).astype(int) - 1, 0)
    fx = tr["x"][last, np.arange(al.shape[1])]
    dist = np.linalg.norm(fx[:, 0:3] - end_p[None], axis=1)
    corr = np.linalg.norm((tr["u"] - tr["raw"]) / scale, axis=-1)
    return {"unsafe": unsafe, "touched": touched, "t_touch": t_touch, "final_dist": dist, "final_x": fx,
            "min_h": h.min(0), "min_z": z.min(0), "return": (tr["rew"] * alf).sum(0), "corr": corr}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True)
    p.add_argument("--weights", default="best", choices=("best", "final"))
    p.add_argument("--episodes", type=int, default=128)
    p.add_argument("--modes", default="ue,none", help=f"comma list of {MODES}")
    p.add_argument("--backup", default="", choices=("", "lqr", "learned"), help="override the run's backup for 'ue'")
    p.add_argument("--backup_path", default="")
    p.add_argument("--disturbance_mode", default="", choices=("", "random", "fixed"), help="override the run's")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--out", default="")
    a = p.parse_args(argv)
    run = Path(a.run)
    rc = load_run(run, a.backup, a.backup_path)
    params = pickle.load(open(run / f"{a.weights}_weights.pkl", "rb"))["actor_params"]
    actor_cfg = rc["actor_cfg"]
    cbf = filter_config(rc["ps2"], rc["task"])
    rt = cu.make_ue_runtime(cbf)
    env_cfg = dataclasses.replace(rc["env_cfg"], bank_seed=int(a.seed))
    if a.disturbance_mode:
        env_cfg = dataclasses.replace(env_cfg, disturbance_mode=a.disturbance_mode)
    env = build_cone_obstacle_ue_env(env_cfg, cbf, rt, cu.make_recoverability_fn_ue(cbf, rt))
    sc = cbf.scenario
    e_bar = jnp.asarray(float(cbf.ue.e_bar), jnp.float32)
    low, high = jnp.asarray(cbf.action_low, jnp.float32), jnp.asarray(cbf.action_high, jnp.float32)
    scale = jnp.asarray(cbf.action_scale, jnp.float32)
    qp_dtype = jnp.float64 if (rc["ps2"].qp_float64 and jax.config.jax_enable_x64) else None
    nom = ncbf.ConeObstacleBCBFConfig(scenario=task_config(rc["task"]), alpha=cbf.alpha, alpha_ground=cbf.alpha_ground,
                                      base_alpha=cbf.base_alpha)
    nom_rt = ncbf.get_cached_runtime(nom)

    def policy(obs):
        return jnp.clip(actor_mean_action(params, obs, scale, actor_cfg, action_low=low, action_high=high), low, high).astype(jnp.float32)

    def filt(mode, obs, raw):
        x, dh = obs[:, :10], obs[:, 10:13]
        if mode == "ue":
            u, aux = jax.vmap(lambda xx, dd, uu: cu.project_full(xx, dd, e_bar, uu, cbf, rt, qp_dtype))(x, dh, raw)
            return u, aux["slack"], aux["used_solver"], aux["safeguard_lambda"]
        if mode == "nominal":
            u, slack, used, info = ncbf.solve_backup_cbf_qp_batch_with_info(x, raw, nom, nom_rt)
            return u, slack, used, info.get("safeguard_lambda", jnp.ones_like(slack))
        one = jnp.ones((raw.shape[0],))
        return raw, 0.0 * one, one > 0, one

    def episode_batch(mode, key):
        n = a.episodes
        es, obs = env.reset_batched(jax.random.split(key, n))

        def body(c, k):
            es, obs, alive = c
            raw = policy(obs)
            u, slack, used, lam = filt(mode, obs, raw)
            u = u.astype(jnp.float32)
            es2, obs_true, obs_out, rew, done, info = env.step_batched(es, u, jax.random.split(jax.random.fold_in(key, k), n))
            rec = {"x": obs_true[:, :10], "d_hat": info.d_hat, "d": info.d_true, "u": u, "raw": raw, "rew": rew,
                   "h": info.h_cone, "z": info.z, "touch": info.touch, "slack": slack, "used": used, "lam": lam,
                   "alive": alive}
            return (es2, obs_out, alive & ~done), rec

        _, tr = jax.lax.scan(body, (es, obs, jnp.ones((n,), bool)), jnp.arange(env.max_steps))
        return jax.device_get(tr)

    out = Path(a.out) if a.out else run / "eval"
    out.mkdir(parents=True, exist_ok=True)
    results, trajs = {}, {}
    key = jax.random.PRNGKey(a.seed)
    end_p = np.asarray(env.ref_states)[-1, 0:3]
    for mode in [m.strip() for m in a.modes.split(",") if m.strip()]:
        if mode not in MODES:
            raise SystemExit(f"unknown mode {mode!r}; choose from {MODES}")
        tr = jax.jit(lambda k, m=mode: episode_batch(m, k))(key)
        tr = {k: np.asarray(v) for k, v in tr.items()}
        o = outcome(tr, float(sc.dt), end_p, np.asarray(scale))
        al = tr["alive"].astype(float)
        touched = o["touched"]
        res = {
            "episodes": int(al.shape[1]), "unsafe_rate": float(o["unsafe"].mean()),
            "min_h_cone": float(o["min_h"].min()), "min_floor_margin": float(o["min_z"].min()),
            "funnel_violation_depth_max": float(max(0.0, -o["min_h"].min())),
            "touched_rate": float(touched.mean()), "touched_and_safe_rate": float((touched & ~o["unsafe"]).mean()),
            "touch_time_mean": float(o["t_touch"][touched].mean()) if touched.any() else None,
            "final_dist_mean": float(o["final_dist"].mean()), "return_mean": float(o["return"].mean()),
            "cil_correction_mean": float((o["corr"] * al).sum() / al.sum()),
            "slack_gt_1e-3_rate": float(((tr["slack"] > 1e-3) * al).sum() / al.sum()),
            "fallback_rate": float(((~tr["used"].astype(bool)) * al).sum() / al.sum()),
            "safeguard_rate": float(((tr["lam"] < 1.0) * al).sum() / al.sum()),
            "max_observer_error": float(np.max(np.linalg.norm(tr["d"] - tr["d_hat"], axis=-1) * al)),
            "max_disturbance": float(np.max(np.linalg.norm(tr["d"], axis=-1) * al)),
        }
        results[mode] = res
        trajs[mode] = tr
        tt = f"{res['touch_time_mean']:.2f} s" if res["touch_time_mean"] is not None else "-"
        print(f"[{mode:7s}] unsafe {res['unsafe_rate']:.3f} (min h {res['min_h_cone']:+.3f} m, min floor margin "
              f"{res['min_floor_margin']:+.3f} m) touched {res['touched_rate']:.2f} (safe {res['touched_and_safe_rate']:.2f}, "
              f"t {tt}) final dist {res['final_dist_mean']:.3f} return {res['return_mean']:.1f} corr "
              f"{res['cil_correction_mean']:.3f} slack>1e-3 {res['slack_gt_1e-3_rate']:.4f} fallback {res['fallback_rate']:.4f} "
              f"safeguard {res['safeguard_rate']:.3f} | max |d| {res['max_disturbance']:.3f} max obs err "
              f"{res['max_observer_error']:.4f}", flush=True)
    meta = {"run": str(run), "weights": a.weights, "seed": a.seed, "task": rc["task"], "task_file": rc["task_file"],
            "backup": cbf.backup, "backup_path": cbf.backup_path, "ue": cbf.ue.as_dict(), "dt": float(sc.dt),
            "env": env_cfg.as_dict(), "reference_states": np.asarray(env.ref_states).tolist(), "results": results}
    (out / "eval.json").write_text(json.dumps(meta, indent=2, default=float))
    np.savez_compressed(out / "trajectories.npz", **{f"{m}__{k}": v for m, tr in trajs.items() for k, v in tr.items()})
    print(f"[eval] wrote {out}/eval.json, trajectories.npz")


if __name__ == "__main__":
    main()
