"""Runway task sets: S (runway keep-out half-space + ceiling) and B (retreat-LQR ellipsoid).

    h_rwy(x)  = y_edge - p_y     (>= 0: on the drone's side of the runway protection strip)
    h_ceil(x) = z_max  - p_z     (>= 0: below the airspace ceiling)

Both are linear in the state, so the barrier gradients are constant. Same interface as the
landing safe set (``component_values``, ``value``, ``values_and_grads``, ``contains``) so the
Phase-I and Phase-II code paths can use either.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from ps2rl.base_controller.quadrotor_retreat_dlqr import QuadrotorRetreatDLQR
from ps2rl.sets.base_sets import EllipsoidBaseSet

IDX_RWY = 0
IDX_CEIL = 1
NAMES = ("runway", "ceiling")


@dataclass(frozen=True)
class RunwaySafeSet:
    y_edge: float
    z_max: float

    @classmethod
    def from_config(cls, cfg) -> "RunwaySafeSet":
        return cls(y_edge=float(cfg.y_edge), z_max=float(cfg.z_max))

    @property
    def num_constraints(self) -> int:
        return 2

    def component_values(self, x: jax.Array) -> jax.Array:
        x = jnp.asarray(x)
        return jnp.stack([self.y_edge - x[..., 1], self.z_max - x[..., 2]], axis=-1)

    def value(self, x: jax.Array) -> jax.Array:
        return jnp.min(self.component_values(x), axis=-1)

    def contains(self, x: jax.Array) -> jax.Array:
        return self.value(x) >= 0.0

    def values_and_grads(self, x: jax.Array):
        x = jnp.asarray(x)
        grads = jnp.zeros((2, 10), x.dtype).at[0, 1].set(-1.0).at[1, 2].set(-1.0)
        return self.component_values(x), grads


def build_runway_sets(cfg) -> tuple[RunwaySafeSet, EllipsoidBaseSet]:
    ctrl = QuadrotorRetreatDLQR.from_config(cfg)
    return RunwaySafeSet.from_config(cfg), EllipsoidBaseSet(ctrl, float(cfg.base_set_c),
                                                            smooth_gain=float(cfg.base_set_smooth_gain))


def base_set_extents(cfg) -> dict[str, float]:
    """Half-widths of B along single error coordinates: sqrt(c_B (P^-1)_ii)."""
    ctrl = QuadrotorRetreatDLQR.from_config(cfg)
    pi = np.linalg.inv(ctrl.p_matrix_f64())
    c = float(cfg.base_set_c)
    ext = np.sqrt(c * np.diag(pi))
    tilt = float(np.sqrt(c * np.linalg.eigvalsh(pi[4:6, 4:6]).max()))
    return {"z": float(ext[0]), "v_x": float(ext[1]), "v_y": float(ext[2]), "v_z": float(ext[3]),
            "phi_tilt": tilt, "phi_yaw": float(ext[6]), "phi_norm": float(np.sqrt(c * np.linalg.eigvalsh(pi[4:7, 4:7]).max()))}


__all__ = ["IDX_CEIL", "IDX_RWY", "NAMES", "RunwaySafeSet", "base_set_extents", "build_runway_sets"]
