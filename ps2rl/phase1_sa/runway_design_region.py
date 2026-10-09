"""Design region Omega and reference measure for runway Phase I (same structure as landing).

Omega = {x in S : y_min <= p_y, z_min <= p_z, ||v|| <= v_max, tilt <= tilt_max, |yaw| <= yaw_max}
(p_x is always 0: nothing depends on it.)

Regions (all rejection-sampled to Omega \\ B):

* ``general`` - p_y, p_z uniform over the operating band, velocity uniform in a ball, body
  z-axis uniform on the tilt cap, yaw uniform; velocity/tilt magnitudes ramp with the
  curriculum from ``general_scale_min`` to 1;
* ``edge`` - close to a barrier with the velocity turned towards it (prob.
  ``edge_outward_prob``): the runway band 0 <= h_rwy <= edge_margin_y, or (prob.
  ``edge_ceiling_prob``) the ceiling band 0 <= h_ceil <= edge_margin_z. These are the states
  where a drone chasing a bird ends up, and the ones that decide recoverability;
* ``shell`` - the capture shell c_B < V <= kappa c_B of the retreat LQR, p_y uniform.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np

from ps2rl.phase1_sa.landing_design_region import _ball, _tilt_yaw_quaternion, tilt_of, yaw_of

REGION_NAMES = ("general", "edge", "shell")


@dataclass(frozen=True)
class RunwayDesignRegionConfig:
    y_min: float = -12.0
    z_min: float = 2.5
    v_max: float = 5.0
    tilt_max_deg: float = 35.0
    yaw_max_deg: float = 45.0

    edge_margin_y: float = 1.5
    edge_margin_z: float = 1.0
    edge_outward_prob: float = 0.7
    edge_ceiling_prob: float = 0.4
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
    def from_dict(cls, payload: dict[str, Any]) -> "RunwayDesignRegionConfig":
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


def build_runway_sampler(region_cfg: RunwayDesignRegionConfig, safe_set, base_set, *, dtype=jnp.float32) -> dict[str, Callable]:
    ctrl = base_set.controller
    c_b = float(base_set.base_set_c)
    kappa = float(region_cfg.shell_kappa)
    y_min, y_edge = float(region_cfg.y_min), float(safe_set.y_edge)
    z_min, z_max = float(region_cfg.z_min), float(safe_set.z_max)
    v_max = float(region_cfg.v_max)
    tilt_max = float(np.deg2rad(region_cfg.tilt_max_deg))
    yaw_max = float(np.deg2rad(region_cfg.yaw_max_deg))
    m_y, m_z = float(region_cfg.edge_margin_y), float(region_cfg.edge_margin_z)
    p_out, p_ceil = float(region_cfg.edge_outward_prob), float(region_cfg.edge_ceiling_prob)
    gmin = float(region_cfg.general_scale_min)
    tries = int(max(1, region_cfg.max_resample_tries))
    l_inv_t = jnp.asarray(np.linalg.inv(np.linalg.cholesky(ctrl.p_matrix_f64())).T, dtype=dtype)
    low_mix = jnp.asarray(region_cfg.mix_low, dtype=dtype)
    high_mix = jnp.asarray(region_cfg.mix_high, dtype=dtype)

    def in_omega(x):
        ok = safe_set.contains(x)
        ok &= (x[1] >= y_min) & (x[2] >= z_min)
        ok &= jnp.linalg.norm(x[3:6]) <= v_max + 1e-6
        ok &= tilt_of(x[6:10]) <= tilt_max + 1e-6
        ok &= jnp.abs(yaw_of(x[6:10])) <= yaw_max + 1e-6
        return ok

    def accept(x):
        return in_omega(x) & jnp.logical_not(base_set.contains(x))

    def assemble(y, z, v, q):
        return jnp.concatenate([jnp.stack([jnp.zeros_like(y), y, z]), v, q]).astype(dtype)

    def draw_general(key, scale):
        m = gmin + (1.0 - gmin) * jnp.clip(scale, 0.0, 1.0)
        k_y, k_z, k_v, k_q = jax.random.split(key, 4)
        y = jax.random.uniform(k_y, (), dtype=dtype, minval=y_min, maxval=y_edge)
        z = jax.random.uniform(k_z, (), dtype=dtype, minval=z_min, maxval=z_max)
        return assemble(y, z, _ball(k_v, m * v_max, 3, dtype), _tilt_yaw_quaternion(k_q, m * tilt_max, yaw_max, dtype))

    def draw_edge(key, _scale):
        k_sel, k_y, k_z, k_h, k_v, k_o, k_q = jax.random.split(key, 7)
        v = _ball(k_v, v_max, 3, dtype)
        q = _tilt_yaw_quaternion(k_q, tilt_max, yaw_max, dtype)
        toward = jax.random.uniform(k_o, (), dtype=dtype) < p_out
        h = jax.random.uniform(k_h, (), dtype=dtype, minval=0.0, maxval=1.0)
        # runway band: velocity towards the runway (+y)
        y_r = y_edge - h * m_y
        z_r = jax.random.uniform(k_z, (), dtype=dtype, minval=z_min, maxval=z_max)
        v_r = v.at[1].set(jnp.where(toward, jnp.abs(v[1]), v[1]))
        x_rwy = assemble(y_r, z_r, v_r, q)
        # ceiling band: velocity up (+z)
        y_c = jax.random.uniform(k_y, (), dtype=dtype, minval=y_min, maxval=y_edge)
        z_c = z_max - h * m_z
        v_c = v.at[2].set(jnp.where(toward, jnp.abs(v[2]), v[2]))
        x_ceil = assemble(y_c, z_c, v_c, q)
        return jnp.where(jax.random.uniform(k_sel, (), dtype=dtype) < p_ceil, x_ceil, x_rwy)

    def draw_shell(key, _scale):
        k_l, k_u, k_y = jax.random.split(key, 3)
        level = jax.random.uniform(k_l, (), dtype=dtype, minval=c_b, maxval=kappa * c_b)
        u = jax.random.normal(k_u, (7,), dtype=dtype)
        u = u / jnp.maximum(jnp.linalg.norm(u), 1e-12)
        e = jnp.sqrt(level) * (l_inv_t @ u)
        x = ctrl.state_from_error(e).astype(dtype)
        return x.at[1].set(jax.random.uniform(k_y, (), dtype=dtype, minval=y_min, maxval=y_edge))

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


__all__ = ["REGION_NAMES", "RunwayDesignRegionConfig", "build_runway_sampler"]
