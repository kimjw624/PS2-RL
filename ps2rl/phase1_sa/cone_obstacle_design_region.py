"""Design region Omega and reference measure for the cone-obstacle Phase I (same structure as landing / runway).

Omega = {x in S : x_min <= p_x - axis_x <= x_max, |p_y - axis_y| <= y_max, p_z <= tip_z + z_above,
         ||v|| <= v_max, tilt <= tilt_max, |yaw| <= yaw_max}

S is the outside of the funnel above the floor. Regions (all rejection-sampled to Omega minus B):

* ``general`` - positions uniform in the box, velocity uniform in a ball, body z-axis uniform on
  the tilt cap, yaw uniform; velocity / tilt magnitudes ramp with the curriculum from
  ``general_scale_min`` to 1;
* ``edge`` - close to a barrier with the velocity turned towards it (prob. ``edge_toward_prob``):
  just under the funnel's outer wall (0 <= h_cone <= edge_margin_cone), or (prob.
  ``edge_floor_prob``) just above the floor (0 <= z - floor <= edge_margin_floor). A drone that
  tracks a reference into the funnel and is pushed off by the filter ends up here;
* ``shell`` - the capture shell c_B < V <= kappa c_B of the base set's hover LQR.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from math import radians, tan
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np

from ps2rl.phase1_sa.landing_design_region import _ball, _tilt_yaw_quaternion, tilt_of, yaw_of

REGION_NAMES = ("general", "edge", "shell")


@dataclass(frozen=True)
class ConeObstacleDesignRegionConfig:
    x_min: float = -3.5          # relative to the funnel axis (the approach comes from -x)
    x_max: float = 1.0
    y_max: float = 0.75
    z_above_tip: float = 2.2
    v_max: float = 3.0
    tilt_max_deg: float = 35.0
    yaw_max_deg: float = 30.0

    edge_margin_cone: float = 0.4
    edge_margin_floor: float = 0.3
    edge_toward_prob: float = 0.7
    edge_floor_prob: float = 0.3
    shell_kappa: float = 1.8
    general_scale_min: float = 0.35

    mix_general_low: float = 0.45
    mix_edge_low: float = 0.1
    mix_shell_low: float = 0.45
    mix_general_high: float = 0.4
    mix_edge_high: float = 0.45
    mix_shell_high: float = 0.15
    weight_general: float = 1.0
    weight_edge: float = 3.0
    weight_shell: float = 0.5
    max_resample_tries: int = 64

    heldout_general: int = 1024
    heldout_edge: int = 1024
    heldout_shell: int = 512
    heldout_seed: int = 1234

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ConeObstacleDesignRegionConfig":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in payload.items() if k in names})

    @property
    def mix_low(self) -> tuple[float, float, float]:
        return (self.mix_general_low, self.mix_edge_low, self.mix_shell_low)

    @property
    def mix_high(self) -> tuple[float, float, float]:
        return (self.mix_general_high, self.mix_edge_high, self.mix_shell_high)

    @property
    def weights(self) -> dict[str, float]:
        return {"general": self.weight_general, "edge": self.weight_edge, "shell": self.weight_shell}

    @property
    def heldout_sizes(self) -> dict[str, int]:
        return {"general": self.heldout_general, "edge": self.heldout_edge, "shell": self.heldout_shell}


def build_cone_obstacle_sampler(region_cfg: ConeObstacleDesignRegionConfig, scen, safe_set, base_set, *,
                                dtype=jnp.float32) -> dict[str, Callable]:
    """``scen`` is the ConeObstacleConfig (geometry and floor)."""
    ctrl = base_set.controller
    c_b = float(base_set.base_set_c)
    kappa = float(region_cfg.shell_kappa)
    ax, ay = float(scen.axis_x), float(scen.axis_y)
    x_lo, x_hi = ax + float(region_cfg.x_min), ax + float(region_cfg.x_max)
    y_max = float(region_cfg.y_max)
    z_lo, z_hi = float(scen.floor_z), float(scen.tip_z) + float(region_cfg.z_above_tip)
    tip, eps, tt = float(scen.tip_z), float(scen.cone_eps), tan(radians(float(scen.cone_theta_deg)))
    v_max = float(region_cfg.v_max)
    tilt_max = float(np.deg2rad(region_cfg.tilt_max_deg))
    yaw_max = float(np.deg2rad(region_cfg.yaw_max_deg))
    m_c, m_f = float(region_cfg.edge_margin_cone), float(region_cfg.edge_margin_floor)
    p_tow, p_floor = float(region_cfg.edge_toward_prob), float(region_cfg.edge_floor_prob)
    gmin = float(region_cfg.general_scale_min)
    tries = int(max(1, region_cfg.max_resample_tries))
    l_inv_t = jnp.asarray(np.linalg.inv(np.linalg.cholesky(ctrl.p_matrix_f64())).T, dtype=dtype)
    low_mix = jnp.asarray(region_cfg.mix_low, dtype=dtype)
    high_mix = jnp.asarray(region_cfg.mix_high, dtype=dtype)

    def in_omega(x):
        ok = safe_set.contains(x)
        ok &= (x[0] >= x_lo) & (x[0] <= x_hi) & (jnp.abs(x[1] - ay) <= y_max) & (x[2] <= z_hi)
        ok &= jnp.linalg.norm(x[3:6]) <= v_max + 1e-6
        ok &= tilt_of(x[6:10]) <= tilt_max + 1e-6
        ok &= jnp.abs(yaw_of(x[6:10])) <= yaw_max + 1e-6
        return ok

    def accept(x):
        return in_omega(x) & jnp.logical_not(base_set.contains(x))

    def assemble(p, v, q):
        return jnp.concatenate([p, v, q]).astype(dtype)

    def uniform_xy(key):
        kx, ky = jax.random.split(key)
        return (jax.random.uniform(kx, (), dtype=dtype, minval=x_lo, maxval=x_hi),
                jax.random.uniform(ky, (), dtype=dtype, minval=ay - y_max, maxval=ay + y_max))

    def draw_general(key, scale):
        m = gmin + (1.0 - gmin) * jnp.clip(scale, 0.0, 1.0)
        k_xy, k_z, k_v, k_q = jax.random.split(key, 4)
        px, py = uniform_xy(k_xy)
        pz = jax.random.uniform(k_z, (), dtype=dtype, minval=z_lo, maxval=z_hi)
        return assemble(jnp.stack([px, py, pz]), _ball(k_v, m * v_max, 3, dtype),
                        _tilt_yaw_quaternion(k_q, m * tilt_max, yaw_max, dtype))

    def draw_edge(key, _scale):
        k_sel, k_xy, k_h, k_v, k_o, k_q = jax.random.split(key, 6)
        v = _ball(k_v, v_max, 3, dtype)
        q = _tilt_yaw_quaternion(k_q, tilt_max, yaw_max, dtype)
        toward = jax.random.uniform(k_o, (), dtype=dtype) < p_tow
        h = jax.random.uniform(k_h, (), dtype=dtype, minval=0.0, maxval=1.0)
        px, py = uniform_xy(k_xy)
        # just under the funnel wall: h_cone = rho - eps - tan(th) (z - tip) in [0, m_c]
        dx, dy = px - ax, py - ay
        rho = jnp.sqrt(dx * dx + dy * dy + eps * eps)
        z_w = tip + (rho - eps - h * m_c) / tt
        n_in = jnp.stack([-dx / rho, -dy / rho, jnp.asarray(tt, dtype)])          # towards the wall (h decreasing)
        n_in = n_in / jnp.linalg.norm(n_in)
        vn = jnp.dot(v, n_in)
        v_w = jnp.where(toward & (vn < 0.0), v - 2.0 * vn * n_in, v)
        x_wall = assemble(jnp.stack([px, py, z_w]), v_w, q)
        # just above the floor, moving down
        z_f = z_lo + h * m_f
        v_f = v.at[2].set(jnp.where(toward, -jnp.abs(v[2]), v[2]))
        x_floor = assemble(jnp.stack([px, py, z_f]), v_f, q)
        return jnp.where(jax.random.uniform(k_sel, (), dtype=dtype) < p_floor, x_floor, x_wall)

    def draw_shell(key, _scale):
        k_l, k_u = jax.random.split(key)
        level = jax.random.uniform(k_l, (), dtype=dtype, minval=c_b, maxval=kappa * c_b)
        u = jax.random.normal(k_u, (9,), dtype=dtype)
        u = u / jnp.maximum(jnp.linalg.norm(u), 1e-12)
        e = jnp.sqrt(level) * (l_inv_t @ u)
        return ctrl.state_from_error(e).astype(dtype)

    drawers = (draw_general, draw_edge, draw_shell)

    def _rejection(draw, key, scale):
        def cond(c):
            i, found, _, _ = c
            return (~found) & (i < tries)

        def body(c):
            i, _, best, k = c
            k, kd = jax.random.split(k)
            cand = draw(kd, scale)
            ok = accept(cand)
            return i + 1, ok, jnp.where(ok, cand, best), k

        k0, kf = jax.random.split(key)
        first = draw(kf, scale)
        _, found, x, _ = jax.lax.while_loop(cond, body, (jnp.int32(0), accept(first), first, k0))
        return x, found

    def sample_region(idx, key, scale):
        return jax.lax.switch(idx, [lambda k, s, d=d: _rejection(d, k, s) for d in drawers], key, scale)

    def mixture_weights(scale):
        s = jnp.clip(jnp.asarray(scale, dtype=dtype), 0.0, 1.0)
        w = (1.0 - s) * low_mix + s * high_mix
        return w / jnp.sum(w)

    def sample(key, scale):
        k_reg, k_x = jax.random.split(key)
        idx = jax.random.categorical(k_reg, jnp.log(jnp.maximum(mixture_weights(scale), 1e-12)))
        x, found = sample_region(idx.astype(jnp.int32), k_x, scale)
        x_fb, _ = _rejection(draw_general, jax.random.fold_in(k_x, 7), scale)
        return jnp.where(found, x, x_fb)

    return {"sample": sample, "sample_region": sample_region, "mixture_weights": mixture_weights, "accept": accept,
            "in_omega": in_omega}


__all__ = ["ConeObstacleDesignRegionConfig", "REGION_NAMES", "build_cone_obstacle_sampler"]
