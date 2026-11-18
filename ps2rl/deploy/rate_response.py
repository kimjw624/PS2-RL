"""Fit of a vehicle's roll/pitch rate response (delay + first-order lag) from a flight record."""

from __future__ import annotations

import numpy as np


def fit_rate_response(t: np.ndarray, cmd: np.ndarray, meas: np.ndarray, *, max_delay: float = 0.10,
                      max_tau: float = 0.30) -> dict | None:
    """Fit the measured roll/pitch rates to a delayed first-order response of the commanded ones.

    A command issued at t_j acts from t_j + delay on (until the next one acts); the rate follows it
    as d omega/dt = (command - omega) / tau. Each control interval is predicted from the measured rate
    at its start (one step ahead, the real sample times), and (delay, tau) is picked on a grid
    (5 ms x 4 ms) by the smallest squared error. delay + tau ("effective delay") is roughly what
    ``predict_ahead`` should be. None when there are too few steps or the commands hardly vary."""
    t = np.asarray(t, np.float64)
    cmd = np.asarray(cmd, np.float64)[:, 0:2]
    meas = np.asarray(meas, np.float64)[:, 0:2]
    n = len(t)
    ok = np.all(np.isfinite(meas), axis=1) & np.all(np.isfinite(cmd), axis=1)
    if n < 30 or ok.sum() < 30 or np.std(cmd[ok]) < 0.05 or np.any(np.diff(t) <= 0):
        return None
    k = np.arange(n - 1)
    y = meas[k + 1]
    best = None
    for d in np.arange(0.0, max_delay + 1e-9, 0.005):
        j0 = np.searchsorted(t, t[k] - d + 1e-9, side="right") - 1          # acting at the start of interval k
        j1 = np.searchsorted(t, t[k + 1] - 1e-6 - d + 1e-9, side="right") - 1  # acting at its end
        valid = (j0 >= 0) & (j1 - j0 <= 2) & ok[k] & ok[k + 1]
        if valid.sum() < 20:
            continue
        start = t[k].copy()
        cur = cmd[np.clip(j0, 0, None)].copy()
        parts = []
        for step in (1, 2):                    # up to two new commands take effect inside one interval
            jj = j0 + step
            has = valid & (jj <= j1)
            b = np.where(has, t[np.clip(jj, 0, n - 1)] + d, start)
            parts.append((np.maximum(b - start, 0.0), cur.copy()))
            start = b
            cur = np.where(has[:, None], cmd[np.clip(jj, 0, n - 1)], cur)
        parts.append((np.maximum(t[k + 1] - start, 0.0), cur))
        for tau in np.r_[0.0, np.arange(0.004, max_tau + 1e-9, 0.004)]:
            w = meas[k].copy()
            for length, c in parts:
                e = np.where(length > 0, np.exp(-length / tau) if tau > 0 else 0.0, 1.0)
                w = c + (w - c) * e[:, None]
            res = float(np.mean((y - w)[valid] ** 2))
            if best is None or res < best[0]:
                best = (res, float(d), float(tau), int(valid.sum()))
    if best is None:
        return None
    res, d, tau, used = best
    return {"delay_s": d, "tau_s": tau, "effective_delay_s": d + tau, "rms_rad_s": float(np.sqrt(res)), "steps": used}


__all__ = ["fit_rate_response"]
