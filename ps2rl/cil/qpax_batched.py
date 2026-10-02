"""Batch-friendly KKT linear algebra for qpax, for the small inequality-only QPs of the BCBF layer.

qpax's primal-dual interior point method factorises the reduced KKT matrix

    H = Q + G^T diag(z / s) G        (n x n, symmetric positive definite; n = action_dim + 1)

with ``jnp.linalg.qr`` at every iteration. Under ``vmap`` that is a batched LAPACK/cuSOLVER call
on thousands of tiny matrices, which several backends execute matrix by matrix. This module
replaces it, for QPs without equality constraints, by a Jacobi-scaled Cholesky factorisation
written out in elementwise operations (n is static and small), which vectorises over the batch
like any other arithmetic. The algorithm (Mehrotra predictor-corrector, relaxation and the
implicit derivative) is still qpax's; only the three linear-algebra helpers are swapped:

    qpax.pdip.initialize, qpax.pdip.factorize_kkt, qpax.pdip.qr_solve

``install()`` / ``uninstall()`` switch globally (functions traced before the switch keep the
implementation they were traced with). ``choose_backend`` times both on the current device.
"""

from __future__ import annotations

import time
from typing import Any, Callable

import jax
import jax.numpy as jnp
import qpax
import qpax.diff_qp as _diff
import qpax.pdip as _pdip
import qpax.pdip_relaxed as _relaxed

_ORIGINAL = {"initialize": _pdip.initialize, "factorize_kkt": _pdip.factorize_kkt, "qr_solve": _pdip.qr_solve}
_STATE = {"installed": False}


class _Chol:
    """Jacobi-scaled Cholesky factor of an SPD matrix: H = D^-1 L L^T D^-1 with D = diag(1/sqrt(h_ii))."""

    def __init__(self, rows: list[list[Any]], scale: jax.Array):
        self.rows = rows      # rows[i][j] (j <= i): entries of L, scalars
        self.scale = scale    # (n,)


def _factor(h: jax.Array) -> _Chol:
    n = h.shape[0]
    d = 1.0 / jnp.sqrt(jnp.diagonal(h))
    hs = h * d[:, None] * d[None, :]
    rows: list[list[Any]] = [[None] * n for _ in range(n)]
    for j in range(n):
        acc = hs[j, j]
        for k in range(j):
            acc = acc - rows[j][k] * rows[j][k]
        ljj = jnp.sqrt(acc)  # a non-positive pivot gives NaN; the layer's finiteness check then falls back
        rows[j][j] = ljj
        for i in range(j + 1, n):
            acc = hs[i, j]
            for k in range(j):
                acc = acc - rows[i][k] * rows[j][k]
            rows[i][j] = acc / ljj
    return _Chol(rows, d)


def _solve(f: _Chol, rhs: jax.Array) -> jax.Array:
    n = len(f.rows)
    scale = f.scale if rhs.ndim == 1 else f.scale[:, None]
    b = rhs * scale
    y: list[Any] = [None] * n
    for i in range(n):  # L y = b
        acc = b[i]
        for k in range(i):
            acc = acc - f.rows[i][k] * y[k]
        y[i] = acc / f.rows[i][i]
    x: list[Any] = [None] * n
    for i in reversed(range(n)):  # L^T x = y
        acc = y[i]
        for k in range(i + 1, n):
            acc = acc - f.rows[k][i] * x[k]
        x[i] = acc / f.rows[i][i]
    return jnp.stack(x, axis=0) * scale


def _require_no_equalities(a: jax.Array) -> None:
    if a.shape[0] != 0:
        raise ValueError("ps2rl.cil.qpax_batched supports QPs without equality constraints only; call uninstall()")


def _qr_solve(factor, rhs):
    if isinstance(factor, _Chol):
        return _solve(factor, rhs)
    if factor is None:  # the (empty) equality block
        return rhs
    return _ORIGINAL["qr_solve"](factor, rhs)


def _initialize(Q, q, A, b, G, h):
    _require_no_equalities(A)
    f = _factor(Q + G.T @ G)
    x = _solve(f, -q + G.T @ h)
    z = G @ x - h
    alpha_p = -jnp.min(-z)
    s = jnp.where(alpha_p < 0, -z, -z + (1 + alpha_p))
    alpha_d = -jnp.min(z)
    z = jnp.where(alpha_d >= 0, z + (1 + alpha_d), z)
    return x, s, z, jnp.zeros((0,), dtype=x.dtype)


def _factorize_kkt(Q, G, A, s, z):
    _require_no_equalities(A)
    p_inv_vec = z / s
    return p_inv_vec, _factor(Q + G.T @ (G.T * p_inv_vec).T), None


def install() -> None:
    """Use the elementwise Cholesky for every qpax solve traced from now on."""
    _pdip.initialize, _pdip.factorize_kkt, _pdip.qr_solve = _initialize, _factorize_kkt, _qr_solve
    _relaxed.factorize_kkt = _factorize_kkt
    _diff.factorize_kkt = _factorize_kkt
    _STATE["installed"] = True


def uninstall() -> None:
    """Back to qpax's QR-based helpers."""
    _pdip.initialize, _pdip.factorize_kkt, _pdip.qr_solve = (_ORIGINAL[k] for k in ("initialize", "factorize_kkt", "qr_solve"))
    _relaxed.factorize_kkt = _ORIGINAL["factorize_kkt"]
    _diff.factorize_kkt = _ORIGINAL["factorize_kkt"]
    _STATE["installed"] = False


def installed() -> bool:
    return bool(_STATE["installed"])


def choose_backend(make_solve_fn: Callable[[], Callable[[], Any]], *, repeats: int = 5, log: Callable[[str], None] = print) -> str:
    """Time a representative solve with both backends on this device and install the faster one.

    ``make_solve_fn()`` must build (trace afresh) and return a zero-argument function that runs the
    solve to completion, e.g. a jitted batched projection with its inputs bound. Returns the name
    of the backend left installed: "cholesky" or "qr".
    """
    times = {}
    for name, switch in (("qr", uninstall), ("cholesky", install)):
        switch()
        fn = make_solve_fn()
        jax.block_until_ready(fn())  # compile
        t0 = time.perf_counter()
        for _ in range(repeats):
            jax.block_until_ready(fn())
        times[name] = (time.perf_counter() - t0) / repeats
    best = min(times, key=times.get)
    (install if best == "cholesky" else uninstall)()
    log(f"[qp] KKT solve on {jax.default_backend()}: qr {1e3 * times['qr']:.1f} ms, cholesky {1e3 * times['cholesky']:.1f} ms "
        f"-> using {best}")
    return best


__all__ = ["choose_backend", "install", "installed", "uninstall", "qpax"]
