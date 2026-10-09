"""Design region Omega and reference measure mu for landing Phase I (landing note, Sec. 6).

Omega = {x in S : zeta_min <= zeta <= zeta_max, ||v|| <= v_max, tilt(q) <= tilt_max, |yaw| <= yaw_max}

Three regions, all sampled analytically (no trace library is needed):

* ``general`` - the whole of Omega \\ B, **uniform per altitude slice**: zeta ~ U[zeta_min,
  zeta_max], then dp uniform (by area) on the cone cross-section disk of radius
  R_eps(zeta); velocity uniform in the ball; body z-axis uniform on the tilt cap; yaw
  uniform. Uniform in *volume* would put most samples high up where the cone is wide
  and arrival is easy; per-slice uniformity gives every altitude equal mass, so the
  narrow low-altitude part - where recoverability is decided - is not starved.
* ``edge`` - the low-altitude cone boundary R_edge: 0 <= h_cone <= edge_margin,
  zeta <= edge_zeta_low, with the radial velocity turned outward with probability
  ``edge_outward_prob`` (the states that limit landing performance).
* ``shell`` - the capture shell R_cap = {x in Omega : c_B < V(x) <= kappa c_B}: level
  uniform in (c_B, kappa c_B], direction uniform in the P-metric.

The reference tube R_tube of the note needs a landing reference, which does not exist
yet; it can be added as a fourth region later.

Every region is rejection-sampled to Omega \\ B (and S). Training draws from a
curriculum mixture; the general region stays in the mix at every curriculum scale
(Remark 4: optimality is needed on all of H, and rollouts leave Omega), and its
velocity/tilt magnitudes ramp from ``general_scale_min`` to 1 while positions are
always sampled over the full cross-section.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np

from ps2rl.base_controller.quadrotor_landing_dlqr import QuadrotorLandingDLQR
from ps2rl.sets.base_sets import EllipsoidBaseSet
from ps2rl.sets.quadrotor_cone_sets import QuadrotorConeSafeSet
from ps2rl.sets.quadrotor_landing_safe_set import QuadrotorLandingSafeSet
from ps2rl.utils.quaternion import normalize_quaternion_batch, quaternion_multiply_batch

REGION_NAMES = ("general", "edge", "shell")


@dataclass(frozen=True)
class LandingDesignRegionConfig:
    zeta_min: float = 0.0
    zeta_max: float = 3.0
    v_max: float = 2.0
    tilt_max_deg: float = 35.0
    yaw_max_deg: float = 45.0

    edge_margin: float = 0.25
    edge_zeta_low: float = 1.0
    edge_outward_prob: float = 0.7
    # with the floor in the safe set: fraction of edge draws taken from the floor band
    # (zeta <= zeta_min + edge_margin, anywhere over the cone cross-section, velocity
    # pointing down with prob. edge_outward_prob). 0 keeps the cone-wall-only edge region.
    edge_floor_prob: float = 0.0

    shell_kappa: float = 1.8

    general_scale_min: float = 0.35

    # curriculum mixture over (general, edge, shell): (1 - s) * low + s * high
    mix_general_low: float = 0.45
    mix_edge_low: float = 0.10
    mix_shell_low: float = 0.45
    mix_general_high: float = 0.40
    mix_edge_high: float = 0.45
    mix_shell_high: float = 0.15

    # reference-measure weights w_r for mu = sum_r w_r mu_r (evaluation score)
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
    def from_dict(cls, payload: dict[str, Any]) -> "LandingDesignRegionConfig":
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


def _tilt_yaw_quaternion(key: jax.Array, tilt_max: jax.Array, yaw_max: jax.Array, dtype) -> jax.Array:
    """Body z-axis uniform on the cap {tilt <= tilt_max}; yaw uniform in [-yaw_max, yaw_max]."""
    k_cos, k_az, k_yaw = jax.random.split(key, 3)
    cos_t = jax.random.uniform(k_cos, (), dtype=dtype, minval=jnp.cos(tilt_max), maxval=1.0)
    tilt = jnp.arccos(jnp.clip(cos_t, -1.0, 1.0))
    az = jax.random.uniform(k_az, (), dtype=dtype, minval=0.0, maxval=2.0 * jnp.pi)
    yaw = jax.random.uniform(k_yaw, (), dtype=dtype, minval=-yaw_max, maxval=yaw_max)
    q_tilt = jnp.stack([jnp.cos(0.5 * tilt), jnp.sin(0.5 * tilt) * jnp.cos(az), jnp.sin(0.5 * tilt) * jnp.sin(az), 0.0 * tilt])
    q_yaw = jnp.stack([jnp.cos(0.5 * yaw), 0.0 * yaw, 0.0 * yaw, jnp.sin(0.5 * yaw)])
    return normalize_quaternion_batch(quaternion_multiply_batch(q_yaw, q_tilt))


def _ball(key: jax.Array, radius: jax.Array, dim: int, dtype) -> jax.Array:
    k_dir, k_rad = jax.random.split(key)
    d = jax.random.normal(k_dir, (dim,), dtype=dtype)
    d = d / jnp.maximum(jnp.linalg.norm(d), 1e-12)
    r = radius * jax.random.uniform(k_rad, (), dtype=dtype) ** (1.0 / dim)
    return r * d


def tilt_of(q: jax.Array) -> jax.Array:
    """Angle between the body z-axis and e3."""
    qn = normalize_quaternion_batch(q)
    # (R e3)_z = 1 - 2 (q_x^2 + q_y^2)
    return jnp.arccos(jnp.clip(1.0 - 2.0 * (qn[..., 1] ** 2 + qn[..., 2] ** 2), -1.0, 1.0))


def yaw_of(q: jax.Array) -> jax.Array:
    qn = normalize_quaternion_batch(q)
    w, x, y, z = qn[..., 0], qn[..., 1], qn[..., 2], qn[..., 3]
    return jnp.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def build_landing_sampler(
    region_cfg: LandingDesignRegionConfig,
    cone: QuadrotorConeSafeSet | QuadrotorLandingSafeSet,
    base_set: EllipsoidBaseSet,
    *,
    dtype=jnp.float32,
) -> dict[str, Callable]:
    """Return JAX samplers. Every sampler maps (key, scale) -> x in R^10 (single state).

    ``scale`` in [0, 1] only affects the general region's velocity/tilt magnitudes.
    ``sample(key, curriculum_scale)`` draws a region from the curriculum mixture.
    """
    ctrl: QuadrotorLandingDLQR = base_set.controller  # type: ignore[assignment]
    c_b = float(base_set.base_set_c)
    kappa = float(region_cfg.shell_kappa)
    zeta_min = float(region_cfg.zeta_min)
    zeta_max = float(region_cfg.zeta_max)
    v_max = float(region_cfg.v_max)
    tilt_max = float(np.deg2rad(region_cfg.tilt_max_deg))
    yaw_max = float(np.deg2rad(region_cfg.yaw_max_deg))
    edge_margin = float(region_cfg.edge_margin)
    edge_zeta_low = float(region_cfg.edge_zeta_low)
    p_out = float(region_cfg.edge_outward_prob)
    p_floor = float(region_cfg.edge_floor_prob) if bool(getattr(cone, "floor", False)) else 0.0
    gmin = float(region_cfg.general_scale_min)
    tries = int(max(1, region_cfg.max_resample_tries))
    pad = jnp.asarray([cone.pad_x, cone.pad_y, cone.pad_z], dtype=dtype)
    l_inv_t = jnp.asarray(np.linalg.inv(np.linalg.cholesky(ctrl.p_matrix_f64())).T, dtype=dtype)
    low_mix = jnp.asarray(region_cfg.mix_low, dtype=dtype)
    high_mix = jnp.asarray(region_cfg.mix_high, dtype=dtype)

    def in_omega(x: jax.Array) -> jax.Array:
        zeta = x[2] - pad[2]
        ok = cone.contains(x)
        ok &= (zeta >= zeta_min) & (zeta <= zeta_max)
        ok &= jnp.linalg.norm(x[3:6]) <= v_max + 1e-6
        ok &= tilt_of(x[6:10]) <= tilt_max + 1e-6
        ok &= jnp.abs(yaw_of(x[6:10])) <= yaw_max + 1e-6
        return ok

    def accept(x: jax.Array) -> jax.Array:
        return in_omega(x) & jnp.logical_not(base_set.contains(x))

    def _assemble(dp: jax.Array, zeta: jax.Array, v: jax.Array, q: jax.Array) -> jax.Array:
        pos = jnp.stack([dp[0] + pad[0], dp[1] + pad[1], zeta + pad[2]])
        return jnp.concatenate([pos, v, q]).astype(dtype)

    def draw_general(key: jax.Array, scale: jax.Array) -> jax.Array:
        m = gmin + (1.0 - gmin) * jnp.clip(scale, 0.0, 1.0)
        k_z, k_r, k_a, k_v, k_q = jax.random.split(key, 5)
        zeta = jax.random.uniform(k_z, (), dtype=dtype, minval=zeta_min, maxval=zeta_max)
        rad = cone.radius_at(zeta) * jnp.sqrt(jax.random.uniform(k_r, (), dtype=dtype))
        ang = jax.random.uniform(k_a, (), dtype=dtype, minval=0.0, maxval=2.0 * jnp.pi)
        dp = rad * jnp.stack([jnp.cos(ang), jnp.sin(ang)])
        v = _ball(k_v, m * v_max, 3, dtype)
        q = _tilt_yaw_quaternion(k_q, m * tilt_max, yaw_max, dtype)
        return _assemble(dp, zeta, v, q)

    def draw_edge(key: jax.Array, _scale: jax.Array) -> jax.Array:
        k_z, k_h, k_a, k_v, k_o, k_q = jax.random.split(key, 6)
        zeta = jax.random.uniform(k_z, (), dtype=dtype, minval=zeta_min, maxval=min(edge_zeta_low, zeta_max))
        h = jax.random.uniform(k_h, (), dtype=dtype, minval=0.0, maxval=edge_margin)
        lin = cone.r0 + cone.tan_theta * zeta - h
        rad = jnp.sqrt(jnp.clip(lin * lin - cone.eps**2, 0.0, None))
        ang = jax.random.uniform(k_a, (), dtype=dtype, minval=0.0, maxval=2.0 * jnp.pi)
        radial = jnp.stack([jnp.cos(ang), jnp.sin(ang)])
        dp = rad * radial
        v = _ball(k_v, v_max, 3, dtype)
        v_r = jnp.dot(v[0:2], radial)
        flip = jax.random.uniform(k_o, (), dtype=dtype) < p_out
        v_xy = jnp.where(flip & (v_r < 0.0), v[0:2] - 2.0 * v_r * radial, v[0:2])
        v = jnp.concatenate([v_xy, v[2:3]])
        q = _tilt_yaw_quaternion(k_q, tilt_max, yaw_max, dtype)
        x_wall = _assemble(dp, zeta, v, q)
        if p_floor <= 0.0:
            return x_wall
        # floor band (separate key stream, so the wall draws are unchanged)
        k_sel, k_fz, k_fr, k_fo = jax.random.split(jax.random.fold_in(key, 101), 4)
        zf = jax.random.uniform(k_fz, (), dtype=dtype, minval=zeta_min, maxval=min(zeta_min + edge_margin, zeta_max))
        dpf = cone.radius_at(zf) * jnp.sqrt(jax.random.uniform(k_fr, (), dtype=dtype)) * radial
        down = jax.random.uniform(k_fo, (), dtype=dtype) < p_out
        vf = v.at[2].set(jnp.where(down, -jnp.abs(v[2]), v[2]))
        x_floor = _assemble(dpf, zf, vf, q)
        return jnp.where(jax.random.uniform(k_sel, (), dtype=dtype) < p_floor, x_floor, x_wall)

    def draw_shell(key: jax.Array, _scale: jax.Array) -> jax.Array:
        k_l, k_u = jax.random.split(key)
        level = jax.random.uniform(k_l, (), dtype=dtype, minval=c_b, maxval=kappa * c_b)
        u = jax.random.normal(k_u, (9,), dtype=dtype)
        u = u / jnp.maximum(jnp.linalg.norm(u), 1e-12)
        e = jnp.sqrt(level) * (l_inv_t @ u)
        return ctrl.state_from_error(e).astype(dtype)

    drawers = (draw_general, draw_edge, draw_shell)

    def _rejection(draw: Callable, key: jax.Array, scale: jax.Array) -> tuple[jax.Array, jax.Array]:
        # while_loop (not fori_loop): stops at the first accepted candidate. Under vmap
        # it runs until every lane has accepted, which is 1-3 iterations here instead of
        # always ``tries``. The key sequence matches a fixed-trial loop, so the accepted
        # sample is identical - only the wasted draws are gone (matters on GPU, where the
        # auto-reset sample is computed for every env at every step).
        def cond(carry):
            i, found, _, _ = carry
            return (~found) & (i < tries)

        def body(carry):
            i, _, best, k = carry
            k, k_draw = jax.random.split(k)
            cand = draw(k_draw, scale)
            ok = accept(cand)
            return i + 1, ok, jnp.where(ok, cand, best), k

        k0, k_first = jax.random.split(key)
        first = draw(k_first, scale)
        _, found, x, _ = jax.lax.while_loop(cond, body, (jnp.int32(0), accept(first), first, k0))
        return x, found

    def sample_region(region_idx: jax.Array, key: jax.Array, scale: jax.Array) -> tuple[jax.Array, jax.Array]:
        branches = [lambda k, s, d=d: _rejection(d, k, s) for d in drawers]
        return jax.lax.switch(region_idx, branches, key, scale)

    def mixture_weights(curriculum_scale: jax.Array) -> jax.Array:
        s = jnp.clip(jnp.asarray(curriculum_scale, dtype=dtype), 0.0, 1.0)
        w = (1.0 - s) * low_mix + s * high_mix
        return w / jnp.sum(w)

    def sample(key: jax.Array, curriculum_scale: jax.Array) -> jax.Array:
        k_reg, k_x = jax.random.split(key)
        idx = jax.random.categorical(k_reg, jnp.log(jnp.maximum(mixture_weights(curriculum_scale), 1e-12)))
        x, found = sample_region(idx.astype(jnp.int32), k_x, curriculum_scale)
        # Fallback (practically never taken): a general-region draw at the same scale.
        x_fb, _ = _rejection(draw_general, jax.random.fold_in(k_x, 7), curriculum_scale)
        return jnp.where(found, x, x_fb)

    return {
        "sample": sample,
        "sample_region": sample_region,
        "mixture_weights": mixture_weights,
        "accept": accept,
        "in_omega": in_omega,
    }


def heldout_sets(
    region_cfg: LandingDesignRegionConfig,
    sampler: dict[str, Callable],
    *,
    split: str,
) -> dict[str, np.ndarray]:
    """Fixed i.i.d. evaluation states per region at full scale (s = 1).

    i.i.d. (not quasi-random) so the per-region Hoeffding bound of the note (Sec. 10)
    applies to the reported recoverabilities.
    """
    offsets = {"val": 0, "test": 1}
    if split not in offsets:
        raise ValueError(f"split must be 'val' or 'test', got {split}")
    base_key = jax.random.PRNGKey(int(region_cfg.heldout_seed) + 1000 * offsets[split])
    out: dict[str, np.ndarray] = {}
    one = jnp.asarray(1.0, dtype=jnp.float32)
    for idx, name in enumerate(REGION_NAMES):
        n = int(region_cfg.heldout_sizes[name])
        keys = jax.random.split(jax.random.fold_in(base_key, idx), n)
        xs, found = jax.jit(jax.vmap(lambda k: sampler["sample_region"](jnp.int32(idx), k, one)))(keys)
        xs, found = np.asarray(xs), np.asarray(found)
        out[name] = xs[found]
    return out


__all__ = [
    "LandingDesignRegionConfig",
    "REGION_NAMES",
    "build_landing_sampler",
    "heldout_sets",
    "tilt_of",
    "yaw_of",
]
