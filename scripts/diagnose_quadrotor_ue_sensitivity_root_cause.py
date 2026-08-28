#!/usr/bin/env python
"""Root-cause diagnostic for the remaining UE-bCBF sensitivity mismatch.

This script is intentionally diagnostic only.  It does not modify PS2-RL,
the backup controller, or the QP.

It separates four possible causes of the large finite-difference errors seen
after Step 21:

1. Tangent-chart/projection error.
   Compare the projected exact-discrete F9 against the Jacobian of the SAME
   discrete map written directly in local right-multiplicative quaternion
   coordinates.

2. Sensitivity-recursion error.
   Compare recursively propagated Phi/Theta against JAX autodiff of the whole
   final-state rollout.  These should agree essentially to numerical precision
   if the recurrence is implemented correctly.

3. Finite-difference/nonsmooth-policy error.
   Sweep finite-difference epsilon and compare against exact autodiff.  The
   saved Phase-I actor uses ReLU hidden activations, so nearby activation-boundary
   crossings can make a finite-difference secant disagree strongly with the
   local derivative even when the derivative propagation code is correct.

4. ReLU hypothesis.
   Measure hidden-unit preactivation margins and repeat the one-step tangent FD
   test with a softplus surrogate using the SAME actor weights.  A large
   improvement for softplus strongly implicates ReLU nonsmoothness rather than
   quaternion geometry.

Run this before changing the QP or retraining the backup policy.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
from typing import Any

sys.path.append(str(Path(__file__).resolve().parents[1]))

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from analyze_quadrotor_lhb_methods import make_dhat_samples, sample_ps2_domain
from ps2rl.backup_policy.quadrotor_learned_backup import load_learned_quadrotor_backup_policy
from ps2rl.cil.quadrotor_backup_cbf import make_backup_runtime
from ps2rl.cil.quadrotor_ue_discrete_sensitivity import (
    discrete_backup_step,
    discrete_tangent_jacobians,
)
from ps2rl.cil.quadrotor_ue_rollout import (
    rollout_estimated_disturbance_backup_flow_and_sensitivities,
)
from ps2rl.evaluation.quadrotor_trace_reset_lib import QuadrotorResetLibrary


def _jsonable(v: Any) -> Any:
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, Path):
        return str(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (np.floating, float)):
        x = float(v)
        return x if np.isfinite(x) else None
    if isinstance(v, (np.integer, int)):
        return int(v)
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    return v


def _stats(x: np.ndarray | list[float]) -> dict[str, Any]:
    a = np.asarray(x, dtype=np.float64).reshape(-1)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {"count": 0}
    return {
        "count": int(a.size),
        "min": float(np.min(a)),
        "p50": float(np.percentile(a, 50.0)),
        "p90": float(np.percentile(a, 90.0)),
        "p95": float(np.percentile(a, 95.0)),
        "p99": float(np.percentile(a, 99.0)),
        "max": float(np.max(a)),
        "mean": float(np.mean(a)),
    }


def _print(label: str, rec: dict[str, Any]) -> None:
    if not rec.get("count"):
        print(f"{label}: no samples")
        return
    print(
        f"{label}: n={rec['count']} p50={rec['p50']:.6g} "
        f"p95={rec['p95']:.6g} p99={rec['p99']:.6g} max={rec['max']:.6g}"
    )


def _parse_csv(text: str) -> list[float]:
    out = [float(v.strip()) for v in str(text).split(",") if v.strip()]
    if not out:
        raise ValueError("expected a non-empty comma-separated list")
    return out


def _quat_mul(a: jax.Array, b: jax.Array) -> jax.Array:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return jnp.asarray(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dtype=a.dtype,
    )


def _quat_conj(q: jax.Array) -> jax.Array:
    return jnp.concatenate([q[:1], -q[1:]])


def _qnorm(q: jax.Array) -> jax.Array:
    return q / jnp.maximum(jnp.linalg.norm(q), jnp.asarray(1e-12, dtype=q.dtype))


def _quat_exp(dtheta: jax.Array) -> jax.Array:
    a = jnp.linalg.norm(dtheta)
    def small(_: None) -> jax.Array:
        return _qnorm(jnp.concatenate([jnp.ones((1,), dtype=dtheta.dtype), 0.5 * dtheta]))
    def normal(_: None) -> jax.Array:
        h = 0.5 * a
        return jnp.concatenate([jnp.cos(h)[None], jnp.sin(h) * dtheta / a])
    return jax.lax.cond(a < jnp.asarray(1e-8, dtype=dtheta.dtype), small, normal, operand=None)


def _boxplus(x: jax.Array, dz: jax.Array) -> jax.Array:
    q = _qnorm(x[6:10])
    q1 = _qnorm(_quat_mul(q, _quat_exp(dz[6:9])))
    return jnp.concatenate([x[:3] + dz[:3], x[3:6] + dz[3:6], q1])


def _boxminus(x_ref: jax.Array, x: jax.Array) -> jax.Array:
    q0 = _qnorm(x_ref[6:10])
    q1 = _qnorm(x[6:10])
    qe = _qnorm(_quat_mul(_quat_conj(q0), q1))
    qe = jnp.where(qe[0] < 0.0, -qe, qe)
    v = qe[1:4]
    vn = jnp.linalg.norm(v)
    def small(_: None) -> jax.Array:
        return 2.0 * v
    def normal(_: None) -> jax.Array:
        ang = 2.0 * jnp.arctan2(vn, jnp.maximum(qe[0], jnp.asarray(0.0, dtype=qe.dtype)))
        return ang * v / vn
    dth = jax.lax.cond(vn < jnp.asarray(1e-8, dtype=vn.dtype), small, normal, operand=None)
    return jnp.concatenate([x[:3] - x_ref[:3], x[3:6] - x_ref[3:6], dth])


def _base_values(runtime: Any, xs: np.ndarray) -> np.ndarray:
    fn = jax.jit(jax.vmap(lambda x: runtime.base_set_values_fn(x)[0]))
    return np.asarray(jax.device_get(fn(jnp.asarray(xs))), dtype=np.float64)


def _choose_outside(states: np.ndarray, runtime: Any, count: int, margin: float, seed: int) -> np.ndarray:
    hb = _base_values(runtime, states)
    idx = np.flatnonzero(hb <= -abs(float(margin)))
    if idx.size == 0:
        raise RuntimeError("no sufficiently outside-B states for sensitivity diagnostic")
    rng = np.random.default_rng(seed)
    take = min(int(count), int(idx.size))
    sel = rng.choice(idx, size=take, replace=False)
    return np.asarray(states[sel], dtype=np.float64)


def _chart_identity_test(xs: np.ndarray, *, eps_values: list[float], seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    dirs = rng.normal(size=(len(xs), 9))
    dirs /= np.maximum(np.linalg.norm(dirs, axis=1, keepdims=True), 1e-12)
    out = {}
    for eps in eps_values:
        vals = []
        for x, z in zip(xs, dirs):
            xj = jnp.asarray(x)
            zj = jnp.asarray(z)
            got = np.asarray(jax.device_get(_boxminus(xj, _boxplus(xj, float(eps) * zj)))) / float(eps)
            vals.append(np.linalg.norm(got - z) / max(np.linalg.norm(z), 1e-12))
        out[f"{eps:g}"] = _stats(vals)
    return out


def _direct_chart_f9(x: jax.Array, d: jax.Array, dt: float, runtime: Any) -> jax.Array:
    x1 = discrete_backup_step(x, d, dt, runtime)
    zero = jnp.zeros((9,), dtype=x.dtype)
    def local_map(dz: jax.Array) -> jax.Array:
        xp = _boxplus(x, dz)
        yp = discrete_backup_step(xp, d, dt, runtime)
        return _boxminus(x1, yp)
    return jax.jacfwd(local_map)(zero)


def _f9_projection_test(xs: np.ndarray, ds: np.ndarray, dt: float, runtime: Any) -> dict[str, Any]:
    def one(x, d):
        _, fproj, _ = discrete_tangent_jacobians(x, d, dt, runtime)
        fchart = _direct_chart_f9(x, d, dt, runtime)
        return fproj, fchart
    fn = jax.jit(jax.vmap(one, in_axes=(0, 0)))
    fp, fc = fn(jnp.asarray(xs), jnp.asarray(ds))
    fp = np.asarray(jax.device_get(fp), dtype=np.float64)
    fc = np.asarray(jax.device_get(fc), dtype=np.float64)
    den = np.maximum(np.linalg.norm(fc.reshape((len(xs), -1)), axis=1), 1e-12)
    rel = np.linalg.norm((fp - fc).reshape((len(xs), -1)), axis=1) / den
    return {"relative_fro_error_projected_vs_direct_chart_ad": _stats(rel)}


def _f9_fd_sweep(xs: np.ndarray, ds: np.ndarray, dt: float, runtime: Any, eps_values: list[float], seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    dirs = rng.normal(size=(len(xs), 9))
    dirs /= np.maximum(np.linalg.norm(dirs, axis=1, keepdims=True), 1e-12)

    def pred_one(x, d):
        _, f9, _ = discrete_tangent_jacobians(x, d, dt, runtime)
        return f9
    f9 = np.asarray(jax.device_get(jax.jit(jax.vmap(pred_one, in_axes=(0, 0)))(jnp.asarray(xs), jnp.asarray(ds))), dtype=np.float64)
    pred = np.einsum("nij,nj->ni", f9, dirs)

    out = {}
    for eps in eps_values:
        errs = []
        for i, (x, d, z) in enumerate(zip(xs, ds, dirs)):
            xj = jnp.asarray(x)
            dj = jnp.asarray(d)
            zj = jnp.asarray(z)
            x1 = discrete_backup_step(xj, dj, dt, runtime)
            yp = discrete_backup_step(_boxplus(xj, float(eps) * zj), dj, dt, runtime)
            ym = discrete_backup_step(_boxplus(xj, -float(eps) * zj), dj, dt, runtime)
            fp = np.asarray(jax.device_get(_boxminus(x1, yp)), dtype=np.float64)
            fm = np.asarray(jax.device_get(_boxminus(x1, ym)), dtype=np.float64)
            fd = (fp - fm) / (2.0 * float(eps))
            errs.append(np.linalg.norm(fd - pred[i]) / max(np.linalg.norm(fd), 1e-12))
        out[f"{eps:g}"] = _stats(errs)
    return out


def _final_state(x0: jax.Array, d: jax.Array, n_steps: int, dt: float, runtime: Any) -> jax.Array:
    def step(x, _):
        return discrete_backup_step(x, d, dt, runtime), None
    xf, _ = jax.lax.scan(step, x0, xs=None, length=int(n_steps))
    return xf


def _chain_rule_test(xs: np.ndarray, ds: np.ndarray, cfg: Any, runtime: Any, horizon: float) -> dict[str, Any]:
    n = int(round(float(horizon) / float(cfg.dt)))
    test_cfg = replace(cfg, T=n * float(cfg.dt), num_steps=n)

    def one(x, d):
        roll = rollout_estimated_disturbance_backup_flow_and_sensitivities(x, d, test_cfg, runtime)
        phi_rec = roll[1][-1]
        th_rec = roll[2][-1]
        phi_ad = jax.jacfwd(lambda xx: _final_state(xx, d, n, float(cfg.dt), runtime))(x)
        th_ad = jax.jacfwd(lambda dd: _final_state(x, dd, n, float(cfg.dt), runtime))(d)
        return phi_rec, phi_ad, th_rec, th_ad

    fn = jax.jit(jax.vmap(one, in_axes=(0, 0)))
    pr, pa, tr, ta = fn(jnp.asarray(xs), jnp.asarray(ds))
    pr, pa, tr, ta = [np.asarray(jax.device_get(v), dtype=np.float64) for v in (pr, pa, tr, ta)]
    ep = np.linalg.norm((pr - pa).reshape((len(xs), -1)), axis=1) / np.maximum(np.linalg.norm(pa.reshape((len(xs), -1)), axis=1), 1e-12)
    et = np.linalg.norm((tr - ta).reshape((len(xs), -1)), axis=1) / np.maximum(np.linalg.norm(ta.reshape((len(xs), -1)), axis=1), 1e-12)
    return {
        "phi_recursive_vs_whole_rollout_ad": _stats(ep),
        "theta_recursive_vs_whole_rollout_ad": _stats(et),
    }


def _theta_fd_sweep(xs: np.ndarray, ds: np.ndarray, cfg: Any, runtime: Any, horizon: float, eps_values: list[float]) -> dict[str, Any]:
    n = int(round(float(horizon) / float(cfg.dt)))
    eye = np.eye(3, dtype=np.float64)
    out = {}
    for eps in eps_values:
        rels = []
        for x, d in zip(xs, ds):
            xj, dj = jnp.asarray(x), jnp.asarray(d)
            ad = np.asarray(jax.device_get(jax.jacfwd(lambda dd: _final_state(xj, dd, n, float(cfg.dt), runtime))(dj)), dtype=np.float64)
            cols = []
            for j in range(3):
                dp = jnp.asarray(d + float(eps) * eye[j])
                dm = jnp.asarray(d - float(eps) * eye[j])
                xp = np.asarray(jax.device_get(_final_state(xj, dp, n, float(cfg.dt), runtime)), dtype=np.float64)
                xm = np.asarray(jax.device_get(_final_state(xj, dm, n, float(cfg.dt), runtime)), dtype=np.float64)
                cols.append((xp - xm) / (2.0 * float(eps)))
            fd = np.stack(cols, axis=1)
            rels.append(np.linalg.norm(fd - ad) / max(np.linalg.norm(fd), 1e-12))
        out[f"{eps:g}"] = _stats(rels)
    return out


def _relu_margin_report(learned: Any, xs: np.ndarray) -> dict[str, Any]:
    h = np.asarray(xs, dtype=np.float64)
    per_layer = []
    all_abs = []
    layers = learned.actor_params["layers"]
    for i, layer in enumerate(layers[:-1]):
        w = np.asarray(layer["w"], dtype=np.float64)
        b = np.asarray(layer["b"], dtype=np.float64)
        z = h @ w + b
        az = np.abs(z)
        per_layer.append(_stats(np.min(az, axis=1)))
        all_abs.append(az.reshape(-1))
        h = np.maximum(z, 0.0)
    flat = np.concatenate(all_abs) if all_abs else np.zeros((0,), dtype=np.float64)
    thresholds = [1e-5, 1e-4, 1e-3, 1e-2, 5e-2]
    return {
        "minimum_abs_preactivation_per_layer": per_layer,
        "fraction_hidden_preactivations_below": {f"{t:g}": float(np.mean(flat < t)) for t in thresholds},
        "total_hidden_preactivations": int(flat.size),
    }


def _softplus_actor(learned: Any, beta: float):
    layers = learned.actor_params["layers"]
    lo = jnp.asarray(learned.action_low)
    hi = jnp.asarray(learned.action_high)
    def action(x: jax.Array) -> jax.Array:
        h = x
        for i, layer in enumerate(layers):
            h = h @ jnp.asarray(layer["w"], dtype=h.dtype) + jnp.asarray(layer["b"], dtype=h.dtype)
            if i != len(layers) - 1:
                h = jax.nn.softplus(float(beta) * h) / float(beta)
        mean, _ = jnp.split(h, 2, axis=-1)
        mid = 0.5 * (hi + lo)
        half = 0.5 * (hi - lo)
        return jnp.asarray(mid, dtype=mean.dtype) + jnp.asarray(half, dtype=mean.dtype) * jnp.tanh(mean)
    return action


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reset-library", required=True)
    p.add_argument("--learned-backup-policy-path", required=True)
    p.add_argument("--domain-source", choices=("paper", "library"), default="paper")
    p.add_argument("--samples-per-region", type=int, default=512)
    p.add_argument("--samples", type=int, default=64)
    p.add_argument("--outside-margin", type=float, default=1.0)
    p.add_argument("--disturbance-bound", type=float, default=2.0)
    p.add_argument("--dhat-random", type=int, default=6)
    p.add_argument("--horizon", type=float, default=0.4)
    p.add_argument("--fd-eps-values", default="0.01,0.003,0.001,0.0003,0.0001")
    p.add_argument("--softplus-beta", type=float, default=20.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", default="outputs/ue_sensitivity_root_cause.json")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    eps_values = _parse_csv(args.fd_eps_values)

    lib = QuadrotorResetLibrary.load(Path(args.reset_library).expanduser().resolve())
    cfg = replace(
        lib.cbf_cfg,
        backup_policy_mode="learned",
        learned_backup_policy_path=str(args.learned_backup_policy_path),
    )
    runtime = make_backup_runtime(cfg)
    learned = load_learned_quadrotor_backup_policy(str(args.learned_backup_policy_path))
    domain, _ = sample_ps2_domain(
        lib,
        samples_per_region=int(args.samples_per_region),
        seed=int(args.seed),
        domain_source=str(args.domain_source),
    )
    xs = _choose_outside(domain, runtime, int(args.samples), float(args.outside_margin), int(args.seed) + 1)
    dhats = make_dhat_samples(float(args.disturbance_bound), int(args.dhat_random), int(args.seed) + 2)
    rng = np.random.default_rng(int(args.seed) + 3)
    ds = dhats[rng.integers(0, len(dhats), size=len(xs))].astype(np.float64)

    print("\n=== A. Quaternion chart identity ===")
    chart = _chart_identity_test(xs, eps_values=eps_values, seed=int(args.seed) + 4)
    for k, v in chart.items():
        _print(f"chart identity eps={k}", v)

    print("\n=== B. Projected F9 vs direct chart autodiff ===")
    fproj = _f9_projection_test(xs, ds, float(cfg.dt), runtime)
    _print("F9 projected vs direct-chart AD", fproj["relative_fro_error_projected_vs_direct_chart_ad"])

    print("\n=== C. One-step F9 finite-difference epsilon sweep (ReLU backup) ===")
    fdf = _f9_fd_sweep(xs, ds, float(cfg.dt), runtime, eps_values, int(args.seed) + 5)
    for k, v in fdf.items():
        _print(f"ReLU F9 FD eps={k}", v)

    chain_take = min(len(xs), 32)
    print("\n=== D. Recursive Phi/Theta vs whole-rollout autodiff ===")
    chain = _chain_rule_test(xs[:chain_take], ds[:chain_take], cfg, runtime, float(args.horizon))
    _print("Phi recursion vs whole AD", chain["phi_recursive_vs_whole_rollout_ad"])
    _print("Theta recursion vs whole AD", chain["theta_recursive_vs_whole_rollout_ad"])

    print("\n=== E. Theta finite-difference epsilon sweep vs whole-rollout autodiff ===")
    theta_take = min(chain_take, 16)
    tfd = _theta_fd_sweep(xs[:theta_take], ds[:theta_take], cfg, runtime, float(args.horizon), eps_values)
    for k, v in tfd.items():
        _print(f"Theta FD eps={k}", v)

    print("\n=== F. ReLU activation-boundary margins ===")
    relu_margin = _relu_margin_report(learned, xs)
    for i, rec in enumerate(relu_margin["minimum_abs_preactivation_per_layer"]):
        _print(f"layer {i} min |preactivation| per state", rec)
    print("fraction hidden preactivations below thresholds:", relu_margin["fraction_hidden_preactivations_below"])

    print("\n=== G. Same-weight softplus one-step F9 finite-difference sweep ===")
    soft_rt = replace(runtime, backup_policy_fn=_softplus_actor(learned, float(args.softplus_beta)))
    soft_fdf = _f9_fd_sweep(xs, ds, float(cfg.dt), soft_rt, eps_values, int(args.seed) + 5)
    for k, v in soft_fdf.items():
        _print(f"softplus F9 FD eps={k}", v)

    results = {
        "settings": {
            "samples": int(len(xs)),
            "outside_margin": float(args.outside_margin),
            "dt": float(cfg.dt),
            "horizon": float(args.horizon),
            "fd_eps_values": eps_values,
            "softplus_beta": float(args.softplus_beta),
        },
        "chart_identity": chart,
        "projected_f9_vs_direct_chart_ad": fproj,
        "relu_f9_fd_sweep": fdf,
        "recursive_vs_whole_rollout_ad": chain,
        "theta_fd_sweep": tfd,
        "relu_activation_margins": relu_margin,
        "softplus_f9_fd_sweep": soft_fdf,
        "interpretation": {
            "if_projected_vs_chart_is_small": "9-D projection/tangent formula is correct; previous FD mismatch is not a quaternion-projection bug.",
            "if_recursive_vs_whole_ad_is_small": "Phi/Theta recurrence is correct; previous Theta FD mismatch comes from nonsmoothness and/or finite-difference crossing, not chain-rule plumbing.",
            "if_softplus_fd_is_much_smaller_than_relu": "ReLU activation-boundary crossings are a primary cause of the finite-difference mismatch; a smooth backup or explicit hybrid/piecewise analysis is needed for a clean UE certificate.",
        },
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(_jsonable(results), indent=2), encoding="utf-8")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
