"""Base-set level certification for the landing task (landing note, Sec. 4.5).

For the LQR ellipsoid B_c = {x : e(x)^T P e(x) <= c}, every support quantity scales
with sqrt(c) (note Eq. (8)). The level bounds are:

  c_U      input feasibility (already computed by ``DiscreteLQR``)
  c_chart  ||phi|| < 2 on B_c, so the quaternion chart stays valid
  c_cone   B_c subset of the cone. Two versions:
             * ``note``  : the corner bound of Definition 2 (radius and altitude
                           extremes taken simultaneously -> conservative)
             * ``exact`` : min of h_cone over the position ellipsoid. h_cone is
                           concave, so the minimum sits on the ellipsoid surface; we
                           evaluate it on a dense S^2 grid and bisect on c
  c_ground base set stays z_clear above the pad plane (not in the note; keeps the
           hover ellipsoid out of ground effect). With the floor in the safe set this is
           also what makes B a subset of S (any z_clear >= 0 suffices for that).
  c_rec    (only with the Phase-I gentle-recovery envelope) B_c subset of
           G = {grad_p h_j . v <= kappa_j h_j}: exact for the floor (linear), a sufficient
           bound for the cone. Needed because the BCBF rows after the backup's arrival in
           B are evaluated along pi_B inside B.
  c_Lyap   one-step decrease V(F(x, pi_B(x))) < V(x) of the *nonlinear* Euler step,
           checked on dense samples (a numerical check, not a proof - note Remark 2)

P, K and c_U do not depend on z_des (hover linearization is translation invariant),
so only c_cone and c_ground move with z_des. That is what makes the "sweet spot"
well defined: the lowest z_des at which the cone and ground stop being binding.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable

import numpy as np

from ps2rl.base_controller.quadrotor_landing_dlqr import IDX_ATT, IDX_POS, IDX_VEL, IDX_Z, QuadrotorLandingDLQR
from ps2rl.sets.quadrotor_cone_sets import QuadrotorConeSafeSet


@dataclass(frozen=True)
class LevelBounds:
    z_des: float
    c_u: float
    c_chart: float
    c_lyap: float
    c_cone_exact: float
    c_cone_note: float
    c_ground: float

    @property
    def c_bar(self) -> float:
        return float(min(self.c_u, self.c_chart, self.c_lyap, self.c_cone_exact, self.c_ground))

    @property
    def binding(self) -> str:
        vals = {
            "input (c_U)": self.c_u,
            "chart": self.c_chart,
            "Lyapunov": self.c_lyap,
            "cone": self.c_cone_exact,
            "ground": self.c_ground,
        }
        return min(vals, key=vals.get)

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["c_bar"] = self.c_bar
        d["binding"] = self.binding
        return d


# ----------------------------------------------------------------- supports
def support_radii(p_inv: np.ndarray, c: float) -> dict[str, float]:
    """Largest deviations over {e^T P e <= c} (note Eq. (8))."""
    rho = {}
    rho["xy"] = float(np.sqrt(c * np.linalg.eigvalsh(p_inv[0:2, 0:2]).max()))
    rho["z"] = float(np.sqrt(c * p_inv[IDX_Z, IDX_Z]))
    rho["v"] = float(np.sqrt(c * np.linalg.eigvalsh(p_inv[3:6, 3:6]).max()))
    rho["att"] = float(np.sqrt(c * np.linalg.eigvalsh(p_inv[IDX_ATT, IDX_ATT]).max()))
    for i, name in enumerate(["dx", "dy", "dz", "vx", "vy", "vz", "phix", "phiy", "phiz"]):
        rho[name] = float(np.sqrt(c * p_inv[i, i]))
    return rho


def c_chart_bound(p_inv: np.ndarray, *, phi_max: float = 2.0) -> float:
    lam = float(np.linalg.eigvalsh(p_inv[IDX_ATT, IDX_ATT]).max())
    return phi_max**2 / lam


def c_ground_bound(p_inv: np.ndarray, *, z_des: float, z_clear: float) -> float:
    if z_des <= z_clear:
        return 0.0
    return (z_des - z_clear) ** 2 / float(p_inv[IDX_Z, IDX_Z])


def _sphere_grid(n_polar: int = 360, n_azim: int = 720) -> np.ndarray:
    t = np.linspace(0.0, np.pi, n_polar)
    p = np.linspace(0.0, 2.0 * np.pi, n_azim, endpoint=False)
    tt, pp = np.meshgrid(t, p, indexing="ij")
    return np.stack([np.sin(tt) * np.cos(pp), np.sin(tt) * np.sin(pp), np.cos(tt)], axis=-1).reshape(-1, 3)


def cone_min_over_base_set(
    p_inv: np.ndarray, cone: QuadrotorConeSafeSet, *, z_des: float, c: float, grid: np.ndarray | None = None
) -> float:
    """min h_cone over the position projection of B_c (exact up to grid resolution)."""
    if grid is None:
        grid = _sphere_grid()
    l_pp = np.linalg.cholesky(p_inv[IDX_POS, IDX_POS])
    e_p = np.sqrt(c) * grid @ l_pp.T  # surface of {e_p^T (P^-1_pp)^-1 e_p <= c}
    dx, dy, dz = e_p[:, 0], e_p[:, 1], e_p[:, 2]
    zeta = z_des + dz
    h = cone.r0 + cone.tan_theta * zeta - np.sqrt(dx * dx + dy * dy + cone.eps**2)
    return float(h.min())


def _bisect_largest(pred: Callable[[float], bool], hi: float, *, iters: int = 60) -> float:
    """Largest c in (0, hi] with pred(c) True, assuming pred is monotone (True then False)."""
    if pred(hi):
        return hi
    lo, up = 0.0, hi
    for _ in range(iters):
        mid = 0.5 * (lo + up)
        if pred(mid):
            lo = mid
        else:
            up = mid
    return lo


def c_cone_exact(p_inv: np.ndarray, cone: QuadrotorConeSafeSet, *, z_des: float, c_hi: float) -> float:
    grid = _sphere_grid()
    return _bisect_largest(
        lambda c: cone_min_over_base_set(p_inv, cone, z_des=z_des, c=c, grid=grid) >= 0.0, c_hi
    )


def c_cone_note(p_inv: np.ndarray, cone: QuadrotorConeSafeSet, *, z_des: float, c_hi: float) -> float:
    def ok(c: float) -> bool:
        rho = support_radii(p_inv, c)
        return cone.r0 + cone.tan_theta * (z_des - rho["z"]) - np.sqrt(rho["xy"] ** 2 + cone.eps**2) >= 0.0

    return _bisect_largest(ok, c_hi)


# --------------------------------------------------------- gentle recovery
def c_recovery_floor(p_inv: np.ndarray, *, z_des: float, kappa: float) -> float:
    """Largest c with v_z <= kappa * zeta on all of B_c (exact: a linear function on an ellipsoid).

    max_{e^T P e <= c} (e_vz - kappa e_zeta) = sqrt(c w^T P^-1 w) must not exceed kappa z_des.
    """
    if kappa <= 0.0:
        return float("inf")
    w = np.zeros(p_inv.shape[0])
    w[IDX_Z] = -kappa
    w[IDX_VEL.start + 2] = 1.0
    return float((kappa * z_des) ** 2 / (w @ p_inv @ w))


def c_recovery_cone(
    p_inv: np.ndarray, cone: QuadrotorConeSafeSet, *, z_des: float, kappa: float, c_hi: float
) -> float:
    """Largest c (sufficient condition) with grad_p h_cone . v <= kappa h_cone on B_c.

    |grad_p h_cone| <= sqrt(1 + tan^2 theta) everywhere and |v| <= rho_v(c) on B_c, so
    sqrt(1 + tan^2) rho_v(c) <= kappa min_{B_c} h_cone is sufficient.
    """
    if kappa <= 0.0:
        return float("inf")
    grid = _sphere_grid(180, 360)
    gmax = float(np.sqrt(1.0 + cone.tan_theta**2))

    def ok(c: float) -> bool:
        rho_v = support_radii(p_inv, c)["v"]
        return gmax * rho_v <= kappa * cone_min_over_base_set(p_inv, cone, z_des=z_des, c=c, grid=grid)

    return _bisect_largest(ok, c_hi)


# ---------------------------------------------------------------- Lyapunov
def lyapunov_ratio_samples(
    controller: QuadrotorLandingDLQR,
    step_fn: Callable[[Any, Any], Any],
    *,
    c_max: float,
    n_dirs: int = 20000,
    n_levels: int = 40,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (levels, worst V(x+)/V(x) over directions at each level).

    ``step_fn(x_batch, u_batch) -> x_next_batch`` is the plant's discrete step (the
    same Euler step the env uses). Levels are geometric from 1e-4 c_max to c_max.
    """
    import jax
    import jax.numpy as jnp

    rng = np.random.default_rng(seed)
    p = controller.p_matrix_f64()
    l_inv_t = np.linalg.inv(np.linalg.cholesky(p)).T  # e = sqrt(level) * L^-T u  => e^T P e = level
    u = rng.standard_normal((n_dirs, 9))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    dirs = u @ l_inv_t.T
    levels = np.geomspace(1e-4 * c_max, c_max, n_levels)
    p_j = jnp.asarray(p)

    @jax.jit
    def ratio(e_batch):
        x = controller.state_from_error(e_batch)
        uu = jax.vmap(controller.action)(x)
        x_next = step_fn(x, uu)
        e_next = controller.error_state(x_next)
        v0 = jnp.einsum("bi,ij,bj->b", e_batch, p_j, e_batch)
        v1 = jnp.einsum("bi,ij,bj->b", e_next, p_j, e_next)
        return v1 / v0

    worst = np.empty_like(levels)
    for i, lev in enumerate(levels):
        e = np.sqrt(lev) * dirs
        worst[i] = float(np.max(np.asarray(ratio(jnp.asarray(e)))))
    return levels, worst


def lyapunov_worst_adversarial(
    controller: QuadrotorLandingDLQR,
    step_single: Callable[[Any, Any], Any],
    *,
    level: float,
    n_starts: int = 4096,
    iters: int = 300,
    lr: float = 0.05,
    seed: int = 0,
) -> tuple[float, np.ndarray]:
    """Projected gradient ascent of V(x+)/V(x) on the level set {e^T P e = level}.

    Random directions in 9-D rarely hit the worst case (large tilt plus falling), so
    the ascent is what ``c_lyap`` should be based on. Returns (worst ratio, worst e).
    """
    import jax
    import jax.numpy as jnp

    p = jnp.asarray(controller.p_matrix_f64())
    l_inv_t = jnp.asarray(np.linalg.inv(np.linalg.cholesky(np.asarray(p))).T)

    def ratio_u(u, lev):
        u = u / jnp.linalg.norm(u)
        e = jnp.sqrt(lev) * (l_inv_t @ u)
        x = controller.state_from_error(e)
        x_next = step_single(x, controller.action(x))
        e_next = controller.error_state(x_next)
        return (e_next @ p @ e_next) / (e @ p @ e)

    val_grad = jax.jit(jax.vmap(jax.value_and_grad(ratio_u), in_axes=(0, None)))
    rng = np.random.default_rng(seed)
    u = jnp.asarray(rng.standard_normal((n_starts, 9)))
    for _ in range(iters):
        _, g = val_grad(u, level)
        u = u + lr * g
        u = u / jnp.linalg.norm(u, axis=1, keepdims=True)
    v, _ = val_grad(u, level)
    i = int(jnp.argmax(v))
    e_worst = np.sqrt(level) * (np.asarray(l_inv_t) @ (np.asarray(u[i]) / np.linalg.norm(u[i])))
    return float(v[i]), e_worst


def c_lyap_adversarial(
    controller: QuadrotorLandingDLQR,
    step_single: Callable[[Any, Any], Any],
    *,
    c_hi: float,
    iters: int = 12,
    **kwargs: Any,
) -> float:
    """Bisect the largest level whose adversarial worst ratio stays below 1.

    Assumes the violation is monotone in the level (true in every sweep we ran).
    """
    return _bisect_largest(
        lambda lev: lyapunov_worst_adversarial(controller, step_single, level=lev, **kwargs)[0] < 1.0,
        c_hi,
        iters=iters,
    )


def c_lyap_from_samples(levels: np.ndarray, worst_ratio: np.ndarray, *, tol: float = 0.0) -> float:
    """Largest sampled level up to which every tested level decreases V."""
    ok = worst_ratio < 1.0 - tol
    if ok.all():
        return float(levels[-1])
    first_bad = int(np.argmin(ok))
    return float(levels[first_bad - 1]) if first_bad > 0 else 0.0


def level_bounds(
    controller: QuadrotorLandingDLQR,
    cone: QuadrotorConeSafeSet,
    *,
    z_des: float,
    z_clear: float,
    c_lyap: float,
    c_hi: float | None = None,
) -> LevelBounds:
    p_inv = np.linalg.inv(controller.p_matrix_f64())
    c_u = float(controller.max_certified_level)
    hi = float(c_hi if c_hi is not None else 10.0 * c_u)
    return LevelBounds(
        z_des=float(z_des),
        c_u=c_u,
        c_chart=c_chart_bound(p_inv),
        c_lyap=float(c_lyap),
        c_cone_exact=c_cone_exact(p_inv, cone, z_des=z_des, c_hi=hi),
        c_cone_note=c_cone_note(p_inv, cone, z_des=z_des, c_hi=hi),
        c_ground=c_ground_bound(p_inv, z_des=z_des, z_clear=z_clear),
    )


def sweet_spot_z_des(
    controller_factory: Callable[[float], QuadrotorLandingDLQR],
    cone: QuadrotorConeSafeSet,
    *,
    z_grid: np.ndarray,
    z_clear: float,
    c_lyap: float,
) -> tuple[float, list[LevelBounds]]:
    """Lowest z_des at which neither the cone nor the ground is the binding level bound."""
    rows = [level_bounds(controller_factory(float(z)), cone, z_des=float(z), z_clear=z_clear, c_lyap=c_lyap) for z in z_grid]
    for row in rows:
        c_rest = min(row.c_u, row.c_chart, row.c_lyap)
        if row.c_cone_exact >= c_rest and row.c_ground >= c_rest:
            return row.z_des, rows
    return float("nan"), rows


__all__ = [
    "LevelBounds",
    "c_chart_bound",
    "c_cone_exact",
    "c_cone_note",
    "c_ground_bound",
    "c_lyap_adversarial",
    "c_lyap_from_samples",
    "cone_min_over_base_set",
    "level_bounds",
    "lyapunov_ratio_samples",
    "lyapunov_worst_adversarial",
    "support_radii",
    "sweet_spot_z_des",
]
