"""Compiled rollouts for the online filter (optional: needs numba; ``FilterConfig.backend = "numba"``).

The same arithmetic as the numpy path of ``ps2rl.online.filter`` (``model.step``, ``margin_parts`` / ``combine``,
``lqr_action``, ``observation``, ``Actor``), written as plain loops and compiled by numba: every rollout runs on its
own and stops as soon as it reaches the base set or leaves the safe set, so a decision costs the steps its rollouts
actually need, without numpy's per-call overhead. Checked against the numpy path by ``scripts/check_cyl_online_fast.py``
(identical decisions; states to float rounding).

    kernel(...)   period 1: the blend step u = lam u_nom + (1 - lam) pi(x0) and the backup's first hit of the base set
                  within N steps, for every candidate row; period > 1: the numpy path runs the window (it simulates the
                  nominal) and the kernel the backup from the window's end (``first_step = False``)

Command hold (``hold`` model steps per command, a control period of hold x dt): every command - the blend at x0 and
each backup command - is computed from the state at a command time and held for ``hold`` model steps (the autopilot
holds the last rate setpoint); the safe set is checked at every model step, the base set (the hand-over to the hover
LQR) only at command times. hold = 1 is the original per-step kernel.
"""

from __future__ import annotations

import numpy as np

try:
    import numba as nb
    HAVE_NUMBA = True
except Exception:  # pragma: no cover - numba is optional
    nb = None
    HAVE_NUMBA = False

ABSENT = 1.0e3


def _njit(f):
    return nb.njit(cache=True, fastmath=False)(f) if HAVE_NUMBA else f


@_njit
def _parts(x, ob, d, g, z_floor, c_B, z_ext, S, DT, r):
    """Fills r = (v, phi) - r_eq(d); returns (h, dp, lv, da, az) as in model.margin_parts."""
    sgn = -2.0 if x[6] < 0.0 else 2.0
    r[0] = x[3]; r[1] = x[4]; r[2] = x[5]
    r[3] = sgn * x[7] - d[1] / g
    r[4] = sgn * x[8] + d[0] / g
    r[5] = sgn * x[9]
    lv = 0.0
    for i in range(6):
        acc = 0.0
        for j in range(6):
            acc += S[i, j] * r[j]
        lv += acc * r[i]
    ax = x[0]; ay = x[1]; az = x[2]
    for j in range(6):
        ax += r[j] * DT[j, 0]
        ay += r[j] * DT[j, 1]
        az += r[j] * DT[j, 2]
    dp = ABSENT
    da = ABSENT
    for k in range(ob.shape[0]):
        if ob[k, 3] > 0.5:
            ex = ob[k, 0] - x[0]; ey = ob[k, 1] - x[1]
            v = np.sqrt(ex * ex + ey * ey + 1e-12) - ob[k, 2]
            if v < dp:
                dp = v
            ex = ob[k, 0] - ax; ey = ob[k, 1] - ay
            v = np.sqrt(ex * ex + ey * ey + 1e-12) - ob[k, 2]
            if v < da:
                da = v
    return x[2] - z_floor, dp, c_B - lv, da, az - z_ext - z_floor


@_njit
def _combine(h, dp, lv, da, az, rd, rho_xy):
    safe = min(h, dp - rd)
    base = min(safe, min(min(lv, da - rho_xy - rd), az))
    return safe, base


@_njit
def _clip(u, lo, hi):
    for i in range(4):
        if u[i] < lo[i]:
            u[i] = lo[i]
        elif u[i] > hi[i]:
            u[i] = hi[i]


@_njit
def _lqr(r, d, g, KtT, lo, hi, u):
    u[0] = g - d[2]; u[1] = 0.0; u[2] = 0.0; u[3] = 0.0
    for j in range(6):
        for i in range(4):
            u[i] -= r[j] * KtT[j, i]
    _clip(u, lo, hi)


@_njit
def _actor(x, view, r, d, g, z_floor, KtT, lo, hi, W0, B0, W1, B1, W2, B2, relu, mid, half, residual_lqr,
           o, h0, h1, u):
    """u = clip(base + mid + half tanh(MLP(obs))) with base the hover LQR about the push's equilibrium (residual_lqr)
    or the equilibrium input."""
    n_slots = view.shape[0]
    o[:] = 0.0
    o[0] = x[2] - z_floor
    o[1] = x[3]; o[2] = x[4]; o[3] = x[5]
    o[4] = x[6]; o[5] = x[7]; o[6] = x[8]; o[7] = x[9]
    for k in range(n_slots):
        if view[k, 3] > 0.5:
            b = 8 + 10 * k
            rx = view[k, 0] - x[0]; ry = view[k, 1] - x[1]
            n = np.sqrt(rx * rx + ry * ry + 1e-12)
            dd = n - view[k, 2]
            o[b] = dd / n * rx; o[b + 1] = dd / n * ry
            o[b + 2] = rx; o[b + 3] = ry
            o[b + 6] = 1.0
            o[b + 8] = view[k, 2]
            o[b + 9] = 1.0
    nh0 = W0.shape[1]
    for j in range(nh0):
        h0[j] = B0[j]
    for i in range(o.shape[0]):
        oi = o[i]
        if oi != 0.0:
            for j in range(nh0):
                h0[j] += oi * W0[i, j]
    for j in range(nh0):
        if relu:
            h0[j] = h0[j] if h0[j] > 0.0 else 0.0
        else:
            h0[j] = h0[j] if h0[j] > 0.0 else np.expm1(h0[j])
    nh1 = W1.shape[1]
    for j in range(nh1):
        h1[j] = B1[j]
    for i in range(nh0):
        hi_ = h0[i]
        if hi_ != 0.0:
            for j in range(nh1):
                h1[j] += hi_ * W1[i, j]
    for j in range(nh1):
        if relu:
            h1[j] = h1[j] if h1[j] > 0.0 else 0.0
        else:
            h1[j] = h1[j] if h1[j] > 0.0 else np.expm1(h1[j])
    if residual_lqr:
        _lqr(r, d, g, KtT, lo, hi, u)
    else:
        u[0] = g - d[2]; u[1] = 0.0; u[2] = 0.0; u[3] = 0.0
    for a in range(4):
        z = B2[a]
        for i in range(nh1):
            z += h1[i] * W2[i, a]
        du = mid[a] + half[a] * np.tanh(z)
        if du != du:            # nan_to_num
            du = 0.0
        u[a] += du
    _clip(u, lo, hi)


@_njit
def _step(x, u, d, dt, g, lo, hi, out):
    a = min(max(u[0], lo[0]), hi[0])
    wx = min(max(u[1], lo[1]), hi[1])
    wy = min(max(u[2], lo[2]), hi[2])
    wz = min(max(u[3], lo[3]), hi[3])
    qw = x[6]; qx = x[7]; qy = x[8]; qz = x[9]
    out[0] = x[0] + dt * x[3]
    out[1] = x[1] + dt * x[4]
    out[2] = x[2] + dt * x[5]
    hd = dt * a
    out[3] = x[3] + hd * 2.0 * (qx * qz + qw * qy) + dt * d[0]
    out[4] = x[4] + hd * 2.0 * (qy * qz - qw * qx) + dt * d[1]
    out[5] = x[5] + hd * (1.0 - 2.0 * (qx * qx + qy * qy)) - dt * g + dt * d[2]
    hh = 0.5 * dt
    nw = qw + hh * (-qx * wx - qy * wy - qz * wz)
    nx = qx + hh * (qw * wx - qz * wy + qy * wz)
    ny = qy + hh * (qz * wx + qw * wy - qx * wz)
    nz = qz + hh * (-qy * wx + qx * wy + qw * wz)
    n = np.sqrt(nw * nw + nx * nx + ny * ny + nz * nz)
    out[6] = nw / n; out[7] = nx / n; out[8] = ny / n; out[9] = nz / n


@_njit
def kernel(X0, first_step, full, lam, u_nom, learned, views, obst, D, extra, switch_extra, N, fail_in, hit_in,
           dt, g, lo, hi, z_floor, r_d, c_B, rho_xy, z_ext, S, DT, KtT,
           W0, B0, W1, B1, W2, B2, relu, mid, half, residual_lqr,
           X_P, b_P, hit_out, fail_out, used_out, hold):
    """Per row: (first_step) the blend step from X0, then the composed backup until the base set (with the row's
    clearance ``extra``) is hit or the safe set is left, at most N steps. Without first_step X0 is the window's end and
    fail_in / hit_in its verdicts. Writes the state after the first step (X_P), its base margin (b_P), the verdicts
    and the backup steps used. ``full``: run all N steps even after the verdict and evaluate the actor at every step (timing of the worst
    case only; the verdicts are unchanged)."""
    R = X0.shape[0]
    obs_dim = W0.shape[0]
    o = np.zeros(obs_dim)
    h0 = np.zeros(W0.shape[1])
    h1 = np.zeros(W1.shape[1])
    r = np.zeros(6)
    u = np.zeros(4)
    ub = np.zeros(4)
    x = np.zeros(10)
    xn = np.zeros(10)
    for i in range(R):
        x[:] = X0[i]
        d = D[i]
        rd_sw = r_d + switch_extra
        rd = r_d + extra[i]
        if first_step:
            h, dp, lv, da, az = _parts(x, obst[i], d, g, z_floor, c_B, z_ext, S, DT, r)
            sw = _combine(h, dp, lv, da, az, rd_sw, rho_xy)[1] >= 0.0
            if learned[i] and not sw:
                _actor(x, views[i], r, d, g, z_floor, KtT, lo, hi, W0, B0, W1, B1, W2, B2, relu, mid, half,
                       residual_lqr, o, h0, h1, ub)
            else:
                _lqr(r, d, g, KtT, lo, hi, ub)
            for a in range(4):
                u[a] = lam[i] * u_nom[i, a] + (1.0 - lam[i]) * ub[a]
            _clip(u, lo, hi)
            fail = False
            for k in range(hold):                       # the blend is held for one control period
                _step(x, u, d, dt, g, lo, hi, xn)
                x[:] = xn
                h, dp, lv, da, az = _parts(x, obst[i], d, g, z_floor, c_B, z_ext, S, DT, r)
                s, b = _combine(h, dp, lv, da, az, rd, rho_xy)
                if s < 0.0:
                    fail = True
            X_P[i] = x
            b_P[i] = b
            hit = (b >= 0.0) and not fail
        else:
            fail = fail_in[i]
            hit = hit_in[i] and not fail
        used = 0
        if full or not (fail or hit):
            done = fail or hit
            h, dp, lv, da, az = _parts(x, obst[i], d, g, z_floor, c_B, z_ext, S, DT, r)
            for j in range(N):
                if j % hold == 0:                             # a command time: the backup's next command
                    sw = _combine(h, dp, lv, da, az, rd_sw, rho_xy)[1] >= 0.0
                    if learned[i] and (full or not sw):       # full (timing): the actor at every command
                        _actor(x, views[i], r, d, g, z_floor, KtT, lo, hi, W0, B0, W1, B1, W2, B2, relu, mid, half,
                               residual_lqr, o, h0, h1, ub)
                    if not learned[i] or sw:
                        _lqr(r, d, g, KtT, lo, hi, ub)
                _step(x, ub, d, dt, g, lo, hi, xn)
                x[:] = xn
                used += 1
                h, dp, lv, da, az = _parts(x, obst[i], d, g, z_floor, c_B, z_ext, S, DT, r)
                if done:
                    continue
                s, b = _combine(h, dp, lv, da, az, rd, rho_xy)
                if s < 0.0:
                    fail = True
                    done = True
                elif b >= 0.0 and (j + 1) % hold == 0:       # the hand-over happens at a command time
                    hit = True
                    done = True
                if done and not full:
                    break
        hit_out[i] = hit and not fail
        fail_out[i] = fail
        used_out[i] = used


def actor_arrays(actor, obs_dim: int):
    """(W0, B0, W1, B1, W2, B2, relu, mid, half) for the kernel; dummies without an actor."""
    if actor is None:
        z = np.zeros((obs_dim, 1)); z1 = np.zeros(1)
        return z, z1, np.zeros((1, 1)), z1, np.zeros((1, 4)), np.zeros(4), True, np.zeros(4), np.zeros(4)
    if len(actor.layers) != 3:
        raise ValueError("the compiled path supports actors with two hidden layers")
    (W0, B0), (W1, B1), (W2, B2) = [(np.ascontiguousarray(w, np.float64), np.asarray(b, np.float64)) for w, b in actor.layers]
    return W0, B0, W1, B1, W2, B2, actor.activation == "relu", np.asarray(actor.mid, float), np.asarray(actor.half, float)


__all__ = ["HAVE_NUMBA", "actor_arrays", "kernel"]
