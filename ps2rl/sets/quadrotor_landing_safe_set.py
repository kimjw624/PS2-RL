"""Landing safe set: approach cone, optionally intersected with the pad plane (floor).

    S = {h_cone(x) >= 0}                                  (floor_constraint = False)
    S = {h_cone(x) >= 0} n {h_floor(x) = z - z_pad >= 0}  (floor_constraint = True)

Without the floor the smooth cone continues below the pad down to its apex at
zeta = -r0 / tan(theta), so "reaching the floor" is not a boundary of S and the model
may pass through the pad. With the floor, touchdown is the boundary of S: the filter
enforces d/dt h_floor >= -alpha_floor h_floor, i.e. an exponential flare, and Phase I
counts going below the pad as a crash exactly like leaving the cone.

Gentle-recovery envelope (Phase I only, optional)
-------------------------------------------------
With the relative-time BCBF rows (the paper's form), holding still at x is feasible iff
along the backup rollout from x every barrier satisfies  dh_j/dtau <= alpha_j h_j,
i.e. the backup never pulls away from a boundary faster than the filter's class-K rate.
A backup that climbs off the floor at full thrust violates this near the floor, so the
filter cannot let the vehicle hover low (it stops about a/(2 alpha^2) above the floor for
a climb acceleration a). ``recovery_rate_*`` > 0 adds

    G_j = {x : grad_p h_j(x) . v <= kappa_j h_j(x)}

to the Phase-I failure condition (``training_contains``), so the learned backup satisfies
it along its rollouts. It is not part of the safety specification: the CIL rows and the
C_N membership used by Phase II are built from S only. With kappa_j <= alpha_j in the CIL,
holding still is feasible at every rest state in C_N(S n G).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp

from ps2rl.sets.quadrotor_cone_sets import QuadrotorConeSafeSet
from ps2rl.sets.safe_sets import SafeSet

CONSTRAINT_NAMES = ("cone", "floor")


@dataclass(frozen=True)
class QuadrotorLandingSafeSet(SafeSet):
    cone: QuadrotorConeSafeSet
    floor: bool = False
    recovery_rate_cone: float = 0.0
    recovery_rate_floor: float = 0.0

    def __post_init__(self) -> None:
        for name in ("recovery_rate_cone", "recovery_rate_floor"):
            if float(getattr(self, name)) < 0.0:
                raise ValueError(f"{name} must be >= 0 (0 disables it), got {getattr(self, name)}")
        if float(self.recovery_rate_floor) > 0.0 and not self.floor:
            raise ValueError("recovery_rate_floor > 0 needs floor_constraint=True")

    @classmethod
    def from_config(cls, cfg: Any) -> "QuadrotorLandingSafeSet":
        return cls(
            cone=QuadrotorConeSafeSet.from_config(cfg),
            floor=bool(getattr(cfg, "floor_constraint", False)),
            recovery_rate_cone=float(getattr(cfg, "recovery_rate_cone", 0.0)),
            recovery_rate_floor=float(getattr(cfg, "recovery_rate_floor", 0.0)),
        )

    # --- cone geometry, forwarded (the design-region sampler reads these) -------------
    @property
    def pad_x(self) -> float:
        return self.cone.pad_x

    @property
    def pad_y(self) -> float:
        return self.cone.pad_y

    @property
    def pad_z(self) -> float:
        return self.cone.pad_z

    @property
    def r0(self) -> float:
        return self.cone.r0

    @property
    def eps(self) -> float:
        return self.cone.eps

    @property
    def tan_theta(self) -> float:
        return self.cone.tan_theta

    def radius_at(self, zeta):
        return self.cone.radius_at(zeta)

    # --- constraints -------------------------------------------------------------------
    @property
    def num_constraints(self) -> int:
        return 2 if self.floor else 1

    @property
    def constraint_names(self) -> tuple[str, ...]:
        return CONSTRAINT_NAMES[: self.num_constraints]

    def floor_value(self, x: jax.Array) -> jax.Array:
        return jnp.asarray(x)[..., 2] - self.pad_z

    def component_values(self, x: jax.Array) -> jax.Array:
        """(..., num_constraints) barrier values [h_cone, (h_floor)]; batched."""
        vals = [self.cone.value(x)]
        if self.floor:
            vals.append(self.floor_value(x))
        return jnp.stack(vals, axis=-1)

    def value(self, x: jax.Array) -> jax.Array:
        """min_j h_j(x) (>= 0 inside S); batched over leading axes."""
        return jnp.min(self.component_values(x), axis=-1)

    def values_and_grads(self, x: jax.Array) -> tuple[jax.Array, jax.Array]:
        h_c, g_c = self.cone.values_and_grads(x)
        if not self.floor:
            return h_c, g_c
        x_arr = jnp.asarray(x)
        g_f = jnp.zeros((1, x_arr.shape[-1]), dtype=x_arr.dtype).at[0, 2].set(1.0)
        return jnp.concatenate([h_c, (x_arr[2] - self.pad_z)[None]]), jnp.concatenate([g_c, g_f], axis=0)

    def contains(self, x: jax.Array) -> jax.Array:
        return self.value(x) >= 0.0

    # --- Phase-I gentle-recovery envelope --------------------------------------------
    @property
    def recovery_enabled(self) -> bool:
        return float(self.recovery_rate_cone) > 0.0 or float(self.recovery_rate_floor) > 0.0

    def recovery_margins(self, x: jax.Array) -> jax.Array:
        """(..., num_constraints) kappa_j h_j - grad_p h_j . v  (>= 0 satisfies G_j; +inf if off)."""
        x_arr = jnp.asarray(x)
        v = x_arr[..., 3:6]
        dx = x_arr[..., 0] - self.pad_x
        dy = x_arr[..., 1] - self.pad_y
        s = jnp.sqrt(dx * dx + dy * dy + self.eps**2)
        dh_c = -(dx / s) * v[..., 0] - (dy / s) * v[..., 1] + self.tan_theta * v[..., 2]
        inf = jnp.asarray(jnp.inf, dtype=x_arr.dtype)
        k_c = float(self.recovery_rate_cone)
        m = [jnp.where(k_c > 0.0, k_c * self.cone.value(x_arr) - dh_c, inf)]
        if self.floor:
            k_f = float(self.recovery_rate_floor)
            m.append(jnp.where(k_f > 0.0, k_f * self.floor_value(x_arr) - v[..., 2], inf))
        return jnp.stack(m, axis=-1)

    def training_contains(self, x: jax.Array) -> jax.Array:
        """Phase-I 'not failed': x in S and (if enabled) the gentle-recovery envelope G."""
        ok = self.contains(x)
        if self.recovery_enabled:
            ok &= jnp.all(self.recovery_margins(x) >= 0.0, axis=-1)
        return ok


__all__ = ["CONSTRAINT_NAMES", "QuadrotorLandingSafeSet"]
