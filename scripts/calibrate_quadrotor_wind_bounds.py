#!/usr/bin/env python
"""Empirically calibrate UE-bCBF acceleration bounds from a Dryden-shaped wind model.

This utility is deliberately separate from the controller.  It generates low-altitude
Dryden-shaped translational gust velocities, converts wind velocity to an equivalent
world-frame acceleration disturbance with a simple isotropic quadratic-drag model,
and reports empirical ||d|| and ||d_dot|| statistics.

Important: Dryden turbulence is Gaussian and therefore not hard bounded.  The reported
maxima/percentiles are calibration data, not formal UE-bCBF bounds.  A hard bound must
still be selected explicitly, e.g. from a percentile plus engineering safety factor or
from a clipped/bounded gust envelope used by the deployment model.

Low-altitude MIL-F-8785C shape parameters used here:
    L_w = h
    L_u = L_v = h / (0.177 + 0.000823 h)^1.2
    sigma_w = 0.1 W20
    sigma_u = sigma_v = sigma_w / (0.177 + 0.000823 h)^0.4
with h in ft and W20 the wind speed at 20 ft.  The filter outputs are rescaled to
these target standard deviations, so the code focuses on the Dryden spectral shape
rather than a particular continuous-white-noise normalization convention.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy import signal

KNOT_TO_MS = 0.5144444444444445
FT_TO_M = 0.3048


def _jsonable(v: Any) -> Any:
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (np.floating, float)):
        x = float(v)
        return x if np.isfinite(x) else None
    if isinstance(v, (np.integer, int)):
        return int(v)
    return v


def _stats(x: np.ndarray) -> dict[str, float | int]:
    a = np.asarray(x, dtype=np.float64).reshape(-1)
    a = a[np.isfinite(a)]
    return {
        "count": int(a.size),
        "mean": float(np.mean(a)),
        "p50": float(np.percentile(a, 50.0)),
        "p90": float(np.percentile(a, 90.0)),
        "p95": float(np.percentile(a, 95.0)),
        "p99": float(np.percentile(a, 99.0)),
        "p99_9": float(np.percentile(a, 99.9)),
        "max": float(np.max(a)),
    }


def _parse_csv(text: str) -> list[float]:
    vals = [float(v.strip()) for v in str(text).split(",") if v.strip()]
    if not vals:
        raise ValueError("expected non-empty comma-separated list")
    return vals


def _dryden_low_altitude_params(altitude_m: float, w20_knots: float) -> dict[str, float]:
    h_ft = max(float(altitude_m) / FT_TO_M, 1.0)
    a = 0.177 + 0.000823 * h_ft
    lw_ft = h_ft
    lu_ft = h_ft / (a**1.2)
    lv_ft = lu_ft
    w20_ms = float(w20_knots) * KNOT_TO_MS
    sigma_w = 0.1 * w20_ms
    sigma_u = sigma_w / (a**0.4)
    sigma_v = sigma_u
    return {
        "h_ft": h_ft,
        "L_u_m": lu_ft * FT_TO_M,
        "L_v_m": lv_ft * FT_TO_M,
        "L_w_m": lw_ft * FT_TO_M,
        "sigma_u_ms": sigma_u,
        "sigma_v_ms": sigma_v,
        "sigma_w_ms": sigma_w,
        "W20_ms": w20_ms,
    }


def _shape_filter(kind: str, length_m: float, advection_speed_ms: float, dt: float):
    tau = max(float(length_m) / max(float(advection_speed_ms), 0.1), 1e-4)
    if kind == "u":
        num = [1.0]
        den = [tau, 1.0]
    else:
        num = [np.sqrt(3.0) * tau, 1.0]
        den = [tau * tau, 2.0 * tau, 1.0]
    a, b, c, d = signal.tf2ss(num, den)
    ad, bd, cd, dd, _ = signal.cont2discrete((a, b, c, d), float(dt), method="bilinear")
    return ad, bd, cd, dd


def _filter_noise(rng: np.random.Generator, n: int, filt, target_sigma: float, warmup: int) -> np.ndarray:
    ad, bd, cd, dd = filt
    x = np.zeros((ad.shape[0],), dtype=np.float64)
    out = np.zeros((n + warmup,), dtype=np.float64)
    noise = rng.normal(size=n + warmup)
    for k, w in enumerate(noise):
        out[k] = float((cd @ x).item() + dd.reshape(-1)[0] * w)
        x = ad @ x + bd.reshape(-1) * w
    y = out[warmup:]
    s = float(np.std(y))
    if s > 1e-12:
        y = y * (float(target_sigma) / s)
    return y


def _simulate_one(
    *,
    rng: np.random.Generator,
    dt: float,
    duration: float,
    warmup: float,
    altitude_m: float,
    w20_knots: float,
    mass_kg: float,
    cda_m2: float,
    rho: float,
    mean_wind_scale: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    prm = _dryden_low_altitude_params(altitude_m, w20_knots)
    n = int(round(float(duration) / float(dt)))
    nw = int(round(float(warmup) / float(dt)))
    vadv = max(prm["W20_ms"], 0.5)
    fu = _shape_filter("u", prm["L_u_m"], vadv, dt)
    fv = _shape_filter("v", prm["L_v_m"], vadv, dt)
    fw = _shape_filter("w", prm["L_w_m"], vadv, dt)
    ug = _filter_noise(rng, n, fu, prm["sigma_u_ms"], nw)
    vg = _filter_noise(rng, n, fv, prm["sigma_v_ms"], nw)
    wg = _filter_noise(rng, n, fw, prm["sigma_w_ms"], nw)

    mean = np.array([float(mean_wind_scale) * prm["W20_ms"], 0.0, 0.0], dtype=np.float64)
    wind = np.stack([ug, vg, wg], axis=1) + mean[None, :]
    speed = np.linalg.norm(wind, axis=1)
    coeff = 0.5 * float(rho) * float(cda_m2) / float(mass_kg)
    disturbance = coeff * speed[:, None] * wind
    ddot = np.diff(disturbance, axis=0) / float(dt)
    return disturbance, ddot, prm


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--w20-knots", default="15,30,45")
    p.add_argument("--altitude-m", type=float, default=2.0)
    p.add_argument("--mass-kg", type=float, default=2.0)
    p.add_argument("--cda-m2", default="0.05,0.10,0.15")
    p.add_argument("--air-density", type=float, default=1.225)
    p.add_argument("--mean-wind-scale", type=float, default=1.0)
    p.add_argument("--dt", type=float, default=0.02)
    p.add_argument("--duration", type=float, default=120.0)
    p.add_argument("--warmup", type=float, default=10.0)
    p.add_argument("--realizations", type=int, default=16)
    p.add_argument("--safety-factor", type=float, default=1.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", default="outputs/ue_wind_bound_calibration.json")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.mass_kg <= 0 or args.dt <= 0 or args.duration <= 0 or args.realizations <= 0:
        raise SystemExit("mass, dt, duration, and realizations must be positive")
    w20s = _parse_csv(args.w20_knots)
    cdas = _parse_csv(args.cda_m2)
    report: dict[str, Any] = {"settings": vars(args), "cases": {}}

    print("\n=== Dryden-shaped wind -> acceleration-bound calibration ===")
    print("NOTE: stochastic Dryden outputs are not hard bounded; use these as calibration statistics only.")
    for w20 in w20s:
        for cda in cdas:
            dnorms, vnorms = [], []
            prm = None
            for r in range(int(args.realizations)):
                rng = np.random.default_rng(int(args.seed) + 10007 * r + int(round(100 * w20)) + int(round(1000 * cda)))
                d, dv, prm = _simulate_one(
                    rng=rng, dt=float(args.dt), duration=float(args.duration), warmup=float(args.warmup),
                    altitude_m=float(args.altitude_m), w20_knots=float(w20), mass_kg=float(args.mass_kg),
                    cda_m2=float(cda), rho=float(args.air_density), mean_wind_scale=float(args.mean_wind_scale),
                )
                dnorms.append(np.linalg.norm(d, axis=1))
                vnorms.append(np.linalg.norm(dv, axis=1))
            dn = np.concatenate(dnorms)
            vn = np.concatenate(vnorms)
            sd, sv = _stats(dn), _stats(vn)
            rec_d = float(args.safety_factor) * sd["p99_9"]
            rec_v = float(args.safety_factor) * sv["p99_9"]
            f_eq = rec_v / (2.0 * np.pi * rec_d) if rec_d > 1e-12 else np.nan
            key = f"W20_{w20:g}kt_CdA_{cda:g}"
            report["cases"][key] = {
                "dryden_parameters": prm,
                "disturbance_norm_ms2": sd,
                "disturbance_rate_norm_ms3": sv,
                "calibration_envelope": {
                    "safety_factor_x_p99_9": float(args.safety_factor),
                    "delta_d_candidate_ms2": rec_d,
                    "delta_v_candidate_ms3": rec_v,
                    "equivalent_sinusoid_frequency_hz": f_eq,
                },
            }
            print(
                f"W20={w20:5.1f} kt CdA={cda:.3f} m^2 | "
                f"p99.9 ||d||={sd['p99_9']:.3f}, p99.9 ||d_dot||={sv['p99_9']:.3f} | "
                f"x{args.safety_factor:g} candidates: delta_d={rec_d:.3f}, delta_v={rec_v:.3f}, f_eq={f_eq:.3f} Hz"
            )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(_jsonable(report), indent=2))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
