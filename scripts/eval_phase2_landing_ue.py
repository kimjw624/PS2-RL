"""Evaluate a Phase-II landing policy under the disturbance, with three safety layers.

    JAX_ENABLE_X64=1 python scripts/eval_phase2_landing_ue.py --run outputs/landing_phase2_ue/land_ue_p2_res_s0 --episodes 128

Works on runs of either Phase-II trainer (repo-structured or cached).

Same policy, same initial states and disturbance draws, executed through
  ue       the UE-bCBF landing CIL (disturbance-aware backup, tube, observer term) - what it was trained with
  nominal  the nominal landing CIL with the nominal Phase-I backup (no disturbance model)
  none     no filter
Reports safety (episodes leaving the cone or going below the pad, min h_cone, min z),
landing (final distance / height / speed, landed = z < 0.1 m, inside r0, |v| < 0.5 m/s),
filter statistics, and saves eval.json + plots (x-z side view with the cone, h_cone(t), z(t)).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
import pickle
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from ps2rl.cil import quadrotor_landing_backup_cbf as lbcbf  # noqa: E402
from ps2rl.cil import quadrotor_landing_ue_bcbf as uecbf  # noqa: E402
from ps2rl.envs.quadrotor_landing_ue_env import (  # noqa: E402
    NOMINAL_DIM,
    OBS_DIM,
    LandingUEEnvConfig,
    build_landing_ue_env,
    landing_ue_env_config_from_quadrotor,
)
from ps2rl.utils.policy import ActorConfig, actor_mean_action  # noqa: E402


def load_run_config(run: Path) -> dict:
    """Run settings from either trainer.

    * repo trainer (``train_phase2_landing_ue.py``): configs.json {sac, env, cbf} + landing_ue.json
    * cached trainer (``train_phase2_landing_ue_cached.py``): configs.json {ps2, env, actor, residual, actor_box, cbf}
    """
    cfgs = json.loads((run / "configs.json").read_text())
    if (run / "landing_ue.json").exists():
        ue = json.loads((run / "landing_ue.json").read_text())
        env_q = SimpleNamespace(**cfgs["env"])
        env_cfg = landing_ue_env_config_from_quadrotor(env_q, LandingUEEnvConfig(**ue["env"]))
        obs_dim = OBS_DIM + (NOMINAL_DIM if ue["residual"] else 0)
        hid = int(cfgs["sac"]["hidden_size"])
        return {"actor_cfg": ActorConfig(obs_dim=obs_dim, action_dim=4, hidden_sizes=(hid, hid)),
                "env_cfg": env_cfg, "residual": bool(ue["residual"]), "actor_box": ue["actor_box"],
                "alpha": float(cfgs["cbf"]["alpha"]), "alpha_floor": float(ue["alpha_floor"]),
                "qp_float64": bool(ue["qp_float64"]), "ckpt": ue["landing_ue_ckpt"]}
    ps2, act_d = cfgs["ps2"], dict(cfgs["actor"])
    act_d["hidden_sizes"] = tuple(act_d["hidden_sizes"])
    return {"actor_cfg": ActorConfig(**act_d), "env_cfg": LandingUEEnvConfig(**cfgs["env"]),
            "residual": bool(cfgs.get("residual", False)), "actor_box": cfgs.get("actor_box"),
            "alpha": float(ps2["alpha_cbf"]), "alpha_floor": float(ps2["alpha_floor"]),
            "qp_float64": bool(ps2.get("qp_float64")),
            "ckpt": cfgs.get("checkpoint") or (json.loads((run / "summary.json").read_text()).get("checkpoint", "")
                                               if (run / "summary.json").exists() else "")}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True)
    p.add_argument("--ckpt", default="", help="UE Phase-I checkpoint (default: the one recorded in the run)")
    p.add_argument("--nominal_ckpt", default="checkpoints/landing_phase1/floor_rec10_td3_seed0")
    p.add_argument("--weights", default="best", choices=("best", "final"))
    p.add_argument("--episodes", type=int, default=128)
    p.add_argument("--modes", default="ue,nominal,none,tracker_ue",
                   help="ue / nominal / none: the trained policy through that filter; tracker_ue: nominal tracker "
                        "alone through the UE filter (residual runs only)")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--out", default="")
    a = p.parse_args(argv)
    run = Path(a.run)
    rc = load_run_config(run)
    actor_cfg = rc["actor_cfg"]
    params = pickle.load(open(run / f"{a.weights}_weights.pkl", "rb"))["actor_params"]
    ckpt = a.ckpt or rc["ckpt"]

    cbf = uecbf.ue_bcbf_config_from_checkpoint(ckpt, alpha=rc["alpha"], alpha_floor=rc["alpha_floor"])
    rt = uecbf.make_ue_runtime(cbf)
    env_cfg = dataclasses.replace(rc["env_cfg"], bank_seed=int(a.seed))
    env = build_landing_ue_env(env_cfg, cbf, rt, uecbf.make_recoverability_fn_ue(cbf, rt))
    e_bar = jnp.asarray(float(cbf.ue.e_bar), jnp.float32)
    low, high = jnp.asarray(cbf.action_low, jnp.float32), jnp.asarray(cbf.action_high, jnp.float32)
    scale = jnp.asarray(cbf.action_scale, jnp.float32)
    qp_dtype = jnp.float64 if rc["qp_float64"] else None
    residual = rc["residual"]
    a_low = jnp.asarray(rc["actor_box"]["low"], jnp.float32) if residual else low
    a_high = jnp.asarray(rc["actor_box"]["high"], jnp.float32) if residual else high

    def policy(obs, mode):
        if mode == "tracker":  # the nominal controller alone (residual = 0)
            return jnp.clip(obs[:, -4:], low, high).astype(jnp.float32)
        a = actor_mean_action(params, obs, a_high if residual else scale, actor_cfg, action_low=a_low, action_high=a_high)
        u_ref = obs[:, -4:] + a if residual else a
        return jnp.clip(u_ref, low, high).astype(jnp.float32)

    nom_cfg = lbcbf.landing_bcbf_config_from_checkpoint(a.nominal_ckpt, alpha=rc["alpha"], alpha_floor=rc["alpha_floor"])
    nom_rt = lbcbf.get_cached_runtime(nom_cfg)
    geo_ue, geo_nom = cbf.landing.as_dict(), nom_cfg.landing.as_dict()
    same_geo = all(np.isclose(float(geo_ue[k]), float(geo_nom[k])) for k in lbcbf.CERTIFIED_FIELDS
                   if k not in ("recovery_rate_cone", "recovery_rate_floor"))

    def filt(mode, obs, raw):
        x, dh = obs[:, :10], obs[:, 10:13]
        if mode == "ue":
            u, aux = jax.vmap(lambda xx, dd, uu: uecbf.project_full(xx, dd, e_bar, uu, cbf, rt, qp_dtype))(x, dh, raw)
            return u, aux["slack"], aux["used_solver"], aux["safeguard_lambda"]
        if mode == "nominal":
            u, slack, used, info = lbcbf.solve_backup_cbf_qp_batch_with_info(x, raw, nom_cfg, nom_rt)
            return u, slack, used, info.get("safeguard_lambda", jnp.ones_like(slack))
        one = jnp.ones((raw.shape[0],))
        return raw, 0.0 * one, one > 0, one

    def episode_batch(mode, key):
        n = a.episodes
        es, obs = env.reset_batched(jax.random.split(key, n))

        def body(c, k):
            es, obs, alive = c
            raw = policy(obs, "tracker" if mode.startswith("tracker") else "policy")
            u, slack, used, lam = filt(mode.replace("tracker_", "").replace("tracker", "ue"), obs, raw)
            u = u.astype(jnp.float32)
            es2, obs_true, obs_out, rew, done, info = env.step_batched(es, u, jax.random.split(jax.random.fold_in(key, k), n))
            rec = {"x": obs_true[:, :10], "d_hat": info.d_hat, "d": info.d_true, "u": u, "raw": raw, "rew": rew,
                   "h": info.h_cone, "z": info.z, "slack": slack, "used": used, "lam": lam, "alive": alive}
            return (es2, obs_out, alive & ~done), rec

        _, tr = jax.lax.scan(body, (es, obs, jnp.ones((n,), bool)), jnp.arange(env.max_steps))
        return jax.device_get(tr)

    out = Path(a.out) if a.out else run / "eval"
    out.mkdir(parents=True, exist_ok=True)
    results, trajs = {}, {}
    key = jax.random.PRNGKey(a.seed)
    for mode in [m.strip() for m in a.modes.split(",") if m.strip()]:
        if mode.startswith("tracker") and not residual:
            continue
        if mode == "nominal" and not same_geo:
            print("[eval] nominal checkpoint has a different geometry; skipping 'nominal'")
            continue
        tr = jax.jit(lambda k: episode_batch(mode, k))(key) if False else episode_batch(mode, key)
        al = tr["alive"].astype(float)  # (T, n)
        L = np.maximum(al.sum(0), 1)
        h = np.where(tr["alive"], tr["h"], np.inf)
        z = np.where(tr["alive"], tr["z"], np.inf)
        unsafe = ((h < 0) | (z < 0)).any(0)
        last = np.maximum(al.sum(0).astype(int) - 1, 0)
        fx = tr["x"][last, np.arange(al.shape[1])]
        dist = np.sqrt(fx[:, 0] ** 2 + fx[:, 1] ** 2)
        speed = np.linalg.norm(fx[:, 3:6], axis=1)
        landed = (fx[:, 2] < 0.10) & (dist < float(cbf.landing.cone_r0)) & (speed < 0.5) & ~unsafe
        corr = np.linalg.norm((tr["u"] - tr["raw"]) / np.asarray(scale), axis=-1)
        res = {
            "episodes": int(al.shape[1]), "unsafe_rate": float(unsafe.mean()),
            "min_h_cone": float(h.min()), "min_z": float(z.min()),
            "h_violation_depth_max": float(max(0.0, -h.min())),
            "landed_rate": float(landed.mean()), "final_dist_mean": float(dist.mean()),
            "final_z_mean": float(fx[:, 2].mean()), "final_speed_mean": float(speed.mean()),
            "return_mean": float((tr["rew"] * al).sum(0).mean()),
            "cil_correction_mean": float((corr * al).sum() / al.sum()),
            "slack_gt_1e-3_rate": float(((tr["slack"] > 1e-3) * al).sum() / al.sum()),
            "fallback_rate": float(((~tr["used"].astype(bool)) * al).sum() / al.sum()),
            "safeguard_rate": float(((tr["lam"] < 1.0) * al).sum() / al.sum()),
            "max_observer_error": float(np.max(np.linalg.norm(tr["d"] - tr["d_hat"], axis=-1) * al)),
        }
        results[mode] = res
        trajs[mode] = tr
        print(f"[{mode:7s}] unsafe {res['unsafe_rate']:.3f} (min h {res['min_h_cone']:+.3f} m, min z {res['min_z']:+.3f} m) "
              f"landed {res['landed_rate']:.2f} final dist {res['final_dist_mean']:.3f} z {res['final_z_mean']:.3f} "
              f"|v| {res['final_speed_mean']:.2f} return {res['return_mean']:.1f} corr {res['cil_correction_mean']:.3f} "
              f"slack>1e-3 {res['slack_gt_1e-3_rate']:.4f} fallback {res['fallback_rate']:.4f} "
              f"safeguard {res['safeguard_rate']:.3f}", flush=True)
    (out / "eval.json").write_text(json.dumps({"run": str(run), "weights": a.weights, "ckpt": ckpt,
                                               "nominal_ckpt": a.nominal_ckpt, "ue": cbf.ue.as_dict(),
                                               "results": results}, indent=2))
    np.savez_compressed(out / "trajectories.npz", **{f"{m}_{k}": v for m, tr in trajs.items() for k, v in tr.items()})
    _plot(trajs, cbf, env, out)
    print(f"[eval] wrote {out}/eval.json, trajectories.npz, landing_ue_eval.png")


def _plot(trajs, cbf, env, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lc = cbf.landing
    tt = np.tan(np.deg2rad(lc.cone_theta_deg))
    modes = list(trajs)
    fig, axes = plt.subplots(2, len(modes), figsize=(5.2 * len(modes), 8.5), squeeze=False)
    ref = np.asarray(env.ref_states)
    zz = np.linspace(0, 2.6, 50)
    colors = {"ue": "tab:blue", "nominal": "tab:orange", "none": "tab:red", "tracker_ue": "tab:gray"}
    for j, m in enumerate(modes):
        tr = trajs[m]
        al = tr["alive"]
        ax = axes[0, j]
        r = lc.cone_r0 + tt * zz
        ax.fill_betweenx(zz, -r, r, color="0.92", label="cone (safe)")
        ax.plot(-r, zz, "k-", lw=0.8)
        ax.plot(r, zz, "k-", lw=0.8)
        ax.axhline(0, color="k", lw=1.2)
        ax.plot(ref[:, 0], ref[:, 2], "k--", lw=1.2, label="reference")
        for i in range(min(al.shape[1], 24)):
            T = int(al[:, i].sum())
            ax.plot(tr["x"][:T, i, 0], tr["x"][:T, i, 2], color=colors.get(m, "C0"), alpha=0.5, lw=0.8)
        ax.set_xlim(-2.6, 1.6)
        ax.set_ylim(-0.1, 2.6)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("z [m]")
        ax.set_title(f"{m}: side view (24 episodes)")
        ax.legend(loc="upper right", fontsize=8)
        ax2 = axes[1, j]
        t = np.arange(al.shape[0]) * lc.dt
        h = np.where(al, tr["h"], np.nan)
        zt = np.where(al, tr["z"], np.nan)
        ax2.plot(t, np.nanmin(h, axis=1), color="tab:purple", label="min over episodes h_cone")
        ax2.plot(t, np.nanmin(zt, axis=1), color="tab:green", label="min over episodes z")
        ax2.axhline(0, color="k", lw=1)
        ax2.set_xlabel("t [s]")
        ax2.set_ylabel("[m]")
        ax2.set_ylim(-0.3, 2.2)
        ax2.legend(fontsize=8)
        ax2.set_title(f"{m}: worst-case barrier values")
    fig.suptitle(f"Landing under disturbance |d| <= {cbf.ue.delta_d} m/s^2, {cbf.ue.frequency_hz} Hz (same policy, same draws)")
    fig.tight_layout()
    fig.savefig(out / "landing_ue_eval.png", dpi=130)


if __name__ == "__main__":
    main()
