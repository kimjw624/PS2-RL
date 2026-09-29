"""Approach-cone safe set for the landing task (landing note, Sec. 3).

    h_cone(x) = r0 + tan(theta) * zeta - sqrt(||dp||^2 + eps^2),   S = {h_cone >= 0},

with dp = p_xy - p_pad and zeta = z - z_pad. The eps-smoothing matters beyond
aesthetics: the unsmoothed cone is non-differentiable on its axis, which is exactly
where the base-set equilibrium sits, and the CIL differentiates h along backup
rollouts. The one-sided (square-root) form has no spurious safe region below the
apex, unlike (r0 + tan(theta) zeta)^2 - ||dp||^2 (landing note, Remark 1).
"""

from __future__ import annotations

from dataclasses import dataclass
from math import radians, tan
from typing import Any

import jax
import jax.numpy as jnp

from ps2rl.sets.safe_sets import SafeSet


@dataclass(frozen=True)
class QuadrotorConeSafeSet(SafeSet):
    r0: float = 0.5
    theta_deg: float = 30.0
    eps: float = 0.025
    pad_x: float = 0.0
    pad_y: float = 0.0
    pad_z: float = 0.0

    def __post_init__(self) -> None:
        if not (0.0 < float(self.eps) < float(self.r0)):
            raise ValueError(f"cone needs 0 < eps < r0, got eps={self.eps}, r0={self.r0}")
        if not (0.0 < float(self.theta_deg) < 90.0):
            raise ValueError(f"cone half-angle must lie in (0, 90) deg, got {self.theta_deg}")

    @classmethod
    def from_config(cls, cfg: Any) -> "QuadrotorConeSafeSet":
        return cls(
            r0=float(cfg.cone_r0),
            theta_deg=float(cfg.cone_theta_deg),
            eps=float(cfg.cone_eps),
            pad_x=float(getattr(cfg, "pad_x", 0.0)),
            pad_y=float(getattr(cfg, "pad_y", 0.0)),
            pad_z=float(getattr(cfg, "pad_z", 0.0)),
        )

    @property
    def tan_theta(self) -> float:
        return tan(radians(float(self.theta_deg)))

    @property
    def num_constraints(self) -> int:
        return 1

    def radius_at(self, zeta: jax.Array | float) -> jax.Array:
        """Cross-section radius R_eps(zeta) = sqrt((r0 + tan(theta) zeta)^2 - eps^2) (0 below)."""
        lin = jnp.asarray(self.r0) + self.tan_theta * jnp.asarray(zeta)
        return jnp.sqrt(jnp.clip(lin * lin - self.eps**2, 0.0, None)) * (lin >= self.eps)

    def value(self, x: jax.Array) -> jax.Array:
        x_arr = jnp.asarray(x)
        dx = x_arr[..., 0] - self.pad_x
        dy = x_arr[..., 1] - self.pad_y
        zeta = x_arr[..., 2] - self.pad_z
        return self.r0 + self.tan_theta * zeta - jnp.sqrt(dx * dx + dy * dy + self.eps**2)

    def values_and_grads(self, x: jax.Array) -> tuple[jax.Array, jax.Array]:
        x_arr = jnp.asarray(x)
        dx = x_arr[0] - self.pad_x
        dy = x_arr[1] - self.pad_y
        s = jnp.sqrt(dx * dx + dy * dy + self.eps**2)
        val = self.r0 + self.tan_theta * (x_arr[2] - self.pad_z) - s
        grad = (
            jnp.zeros((10,), dtype=x_arr.dtype)
            .at[0]
            .set(-dx / s)
            .at[1]
            .set(-dy / s)
            .at[2]
            .set(self.tan_theta)
        )
        return jnp.stack([val]), jnp.stack([grad])

    def contains(self, x: jax.Array) -> jax.Array:
        return self.value(x) >= 0.0


__all__ = ["QuadrotorConeSafeSet"]
