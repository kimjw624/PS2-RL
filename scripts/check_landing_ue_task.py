#!/usr/bin/env python
"""Pre-training checks of a UE landing task file (backup, disturbance model, reference) - no RL involved.

    JAX_ENABLE_X64=1 python scripts/check_landing_ue_task.py --task docker_batch/configs/landing_ue_x500.json
    # another reference or disturbance with the same backup:
    JAX_ENABLE_X64=1 python scripts/check_landing_ue_task.py --task docker_batch/configs/landing_ue_x500.json \
        --reference ps2rl/envs/assets/quadrotor_landing_deepdip_x500_reference.npz --ue '{"frequency_hz": 0.05}'

1. **Can the filter let the vehicle reach the pad?** UE recoverability (tube-tightened first-hit C_N,
   the set the filter keeps the vehicle in) of level states over the pad centre, at rest and descending,
   with frozen estimates d_hat of size delta_d + e_bar in several directions. If the states just above
   the pad are not in C_N, no policy can land through the filter: the backup cannot reach the
   tightened base set from there within its horizon.
2. **What does the reference ask for, and do the filters still land it?** The environment's geometric
   reference tracker (PD + feed-forward + d_hat compensation) flies the task's episodes - same starts and
   disturbance draws - through no filter, the nominal landing filter and the UE filter: how far it leaves
   the cone, and whether it lands (z < 0.1 m inside r0 at |v| < 0.5 m/s at the end of the episode).

A task that fails (1), or whose tracker does not land through the UE filter in (2), will not train a
landing policy either; the usual fix is a slower disturbance rate delta_v (the tube over the 2 s horizon
grows with e_bar + delta_v tau, not with delta_d).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from ps2rl.cil import quadrotor_landing_backup_cbf as lbcbf  # noqa: E402
from ps2rl.cil import quadrotor_landing_ue_bcbf as uecbf  # noqa: E402
from ps2rl.envs.quadrotor_landing_ue_env import LandingUEEnvConfig, build_landing_ue_env  # noqa: E402
from ps2rl.uncertainty.landing_ue_tube import tube_constants, nominal_tube, tube_margins  # noqa: E402


def _resolve(path: str) -> Path:
    q = Path(path).expanduser()
    return q if q.is_absolute() or q.exists() else ROOT / path


def _floats(text: str) -> list[float]:
    return [float(v) for v in text.split(",") if v.strip()]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", required=True, help="landing UE task file (backup, ue, filter, env)")
    p.add_argument("--reference", default="", help="reference npz instead of the task's")
    p.add_argument("--ue", default="", help='JSON object overriding fields of the task\'s "ue" section, e.g. \'{"frequency_hz": 0.05}\'')
    p.add_argument("--heights", default="0.02,0.05,0.1,0.2,0.3,0.5,0.75,1.0", help="pad check: heights above the pad [m]")
    p.add_argument("--descent", default="0,0.3", help="pad check: descent speeds [m/s]")
    p.add_argument("--episodes", type=int, default=32, help="tracker check: episodes per mode (0 = skip)")
    p.add_argument("--modes", default="none,nominal,ue")
    p.add_argument("--extra_sec", type=float, default=-1.0, help="tracker check: episode tail after the reference (< 0: the task's)")
    p.add_argument("--seed", type=int, default=3)
    a = p.parse_args(argv)

    task = json.loads(_resolve(a.task).read_text(encoding="utf-8"))
    ue = uecbf.load_ue_config(task)
    if a.ue:
        over = json.loads(a.ue)
        if "delta_d" in over or "frequency_hz" in over:      # re-derive the dependent bounds
            over = {"delta_v": -1.0, "d_hat_max": -1.0, **over}
        ue = ue.replace(**over)
    filt = task.get("filter", {})
    alpha, alpha_floor = float(filt.get("alpha_cbf", 10.0)), float(filt.get("alpha_floor", 20.0))
    backup = str(_resolve(task["backup"]))
    cbf = uecbf.ue_bcbf_config_from_checkpoint(backup, alpha=alpha, alpha_floor=alpha_floor, ue=ue)
    rt = uecbf.make_ue_runtime(cbf)
    rec = uecbf.make_recoverability_fn_ue(cbf, rt)
    lc = cbf.landing
    tc = tube_constants(lc)
    s = nominal_tube(int(lc.num_steps), 1.0, ue, tc, float(lc.dt))
    mc, mf, mb = (float(np.asarray(v)) for v in tube_margins(s[-1], ue, tc))
    print(f"[task] {a.task}: backup {Path(backup).name} ({'plain' if uecbf.is_plain_backup(backup) else 'UE'}); box "
          f"{lc.a_cmd_min:g}-{lc.a_cmd_max:g} m/s^2, rates {lc.omega_max:g} rad/s; c_B {lc.base_set_c:g}")
    print(f"[ue] delta_d {ue.delta_d:g} m/s^2, frequency {ue.frequency_hz:g} Hz -> delta_v {ue.delta_v:.4f} m/s^3, e_bar {ue.e_bar:g}, "
          f"tube x{ue.tube_scale:g}; at the end of the {lc.num_steps * lc.dt:g} s horizon (no contraction): m_base {mb:.2f} of "
          f"c_B {lc.base_set_c:g} (level left {lc.base_set_c - mb:.2f}), cone {mc:.3f} m, floor {mf:.3f} m")

    # ---------------------------------------------------------------- 1. states over the pad
    dirs = np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, -1]], float)
    d_mag = float(ue.delta_d) + float(ue.e_bar)
    heights, descents = _floats(a.heights), _floats(a.descent)
    print(f"[pad] UE-recoverable fraction of level states over the pad centre (d_hat 0 and {d_mag:.2f} m/s^2 along +-x, +y, +-z):")
    print("        " + "".join(f"{h:>7.2f}" for h in heights) + "   <- height above the pad [m]")
    lowest_ok = {}
    for vd in descents:
        xs, ds = [], []
        for h in heights:
            for d in dirs:
                xs.append([lc.pad_x, lc.pad_y, lc.pad_z + h, 0.0, 0.0, -abs(vd), 1.0, 0.0, 0.0, 0.0])
                ds.append(d_mag * d)
        ok = np.asarray(rec(jnp.asarray(np.array(xs), jnp.float32), jnp.asarray(np.array(ds), jnp.float32)))
        frac = ok.reshape(len(heights), len(dirs)).mean(1)
        print(f"  v_z {-abs(vd):+.1f}" + "".join(f"{f:>7.2f}" for f in frac))
        full = [h for h, f in zip(heights, frac) if f >= 1.0]
        lowest_ok[vd] = min(full) if full else None
    for vd, h in lowest_ok.items():
        print(f"  descending at {abs(vd):.1f} m/s: every tested estimate recoverable from "
              + (f"{h:.2f} m up" if h is not None else "no tested height") + " over the pad")

    # ---------------------------------------------------------------- 2. the reference tracker through the filters
    if a.episodes <= 0:
        return
    env_over = {k: v for k, v in task.get("env", {}).items() if not k.startswith("_")}
    env_over.update(nominal_controller="tracker", terminate_on_unsafe=False, terminate_pos_error=0.0,
                    bank_size=max(512, 4 * a.episodes), bank_seed=7)
    if a.reference:
        env_over["reference_path"] = a.reference
    if a.extra_sec >= 0.0:
        env_over["extra_sec"] = a.extra_sec
    env_cfg = LandingUEEnvConfig(**env_over)
    env = build_landing_ue_env(env_cfg, cbf, rt, rec)
    nom = lbcbf.landing_bcbf_config_from_checkpoint(backup, alpha=alpha, alpha_floor=alpha_floor)
    nrt = lbcbf.get_cached_runtime(nom)
    e_bar = jnp.asarray(float(ue.e_bar), jnp.float32)
    low, high = jnp.asarray(cbf.action_low, jnp.float32), jnp.asarray(cbf.action_high, jnp.float32)
    scale = np.asarray(cbf.action_scale)
    n = int(a.episodes)
    print(f"[tracker] geometric reference tracker on {Path(env_cfg.reference_path).name}, {n} episodes "
          f"({env.max_steps * lc.dt:.1f} s), same starts and disturbance draws in every mode:")

    def run(mode, key):
        es, obs = env.reset_batched(jax.random.split(key, n))

        def body(c, k):
            es, obs = c
            raw = jnp.clip(obs[:, -4:], low, high).astype(jnp.float32)
            x, dh = obs[:, :10], obs[:, 10:13]
            if mode == "ue":
                u, _ = jax.vmap(lambda xx, dd, uu: uecbf.project_full(xx, dd, e_bar, uu, cbf, rt, jnp.float64))(x, dh, raw)
            elif mode == "nominal":
                u, *_ = lbcbf.solve_backup_cbf_qp_batch_with_info(x, raw, nom, nrt)
            else:
                u = raw
            u = u.astype(jnp.float32)
            es2, ot, oo, _, _, info = env.step_batched(es, u, jax.random.split(jax.random.fold_in(key, k), n))
            return (es2, oo), {"x": ot[:, :10], "h": info.h_cone, "z": info.z, "c": jnp.linalg.norm((u - raw) / scale, axis=-1)}

        _, tr = jax.lax.scan(body, (es, obs), jnp.arange(env.max_steps))
        return jax.device_get(tr)

    for mode in [m.strip() for m in a.modes.split(",") if m.strip()]:
        tr = run(mode, jax.random.PRNGKey(a.seed))
        fx = tr["x"][-1]
        dist = np.hypot(fx[:, 0] - lc.pad_x, fx[:, 1] - lc.pad_y)
        sp = np.linalg.norm(fx[:, 3:6], axis=1)
        unsafe = ((tr["h"] < 0) | (tr["z"] < 0)).any(0)
        landed = (fx[:, 2] - lc.pad_z < 0.1) & (dist < float(lc.cone_r0)) & (sp < 0.5) & ~unsafe
        print(f"  [{mode:7s}] unsafe {unsafe.mean():.2f}  deepest outside the cone {max(0.0, -float(tr['h'].min())):.3f} "
              f"(min h {float(tr['h'].min()):+.3f}, min z {float(tr['z'].min()):+.3f})  landed {landed.mean():.2f}  final distance "
              f"{dist.mean():.3f} m, height {fx[:, 2].mean() - lc.pad_z:.3f} m, |v| {sp.mean():.2f}  filter correction {tr['c'].mean():.3f}",
              flush=True)
        jax.clear_caches()


if __name__ == "__main__":
    main()
