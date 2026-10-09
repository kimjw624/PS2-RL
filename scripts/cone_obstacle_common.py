"""Shared helpers of the cone-obstacle CPU studies (certificates, coverage, reach tests)."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from ps2rl.envs.quadrotor_cone_obstacle_config import ConeObstacleConfig  # noqa: E402

VEHICLE_KEYS = ("a_cmd_min", "a_cmd_max", "omega_max", "omega_z_max")


def vehicle_fields(path: str) -> dict[str, Any]:
    """Action box and hover-LQR weights from a landing config file (its "landing" section):
    the vehicle's limits and the LQR designed for them carry over unchanged."""
    payload = json.loads((ROOT / path if not Path(path).is_absolute() else Path(path)).read_text(encoding="utf-8"))
    landing = payload.get("landing", payload)
    return {k: v for k, v in landing.items() if k in VEHICLE_KEYS or k.startswith("lqr_")}


def make_config(vehicle: str, **overrides: Any) -> ConeObstacleConfig:
    return ConeObstacleConfig(**{**vehicle_fields(vehicle), **overrides})


def parse_list(text: str, cast=float) -> list:
    return [cast(t) for t in str(text).split(",") if t.strip()]


def level_state(p, v=(0.0, 0.0, 0.0)) -> np.ndarray:
    return np.asarray([*p, *v, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)


def resolve(path: str) -> Path:
    return ROOT / path if not Path(path).is_absolute() else Path(path)


# ------------------------------------------------------------------ base-set designs
# Named hover points, relative to the geometry: (x offset from the axis [m], height mode,
# height value). Height modes: "mid" = middle of the ground-tip gap (+ value), "tip" =
# tip height + value. The approach comes from -x.
NAMED_POINTS = {
    "approach_low": (-1.0, "mid", 0.0),       # under the outer wall, on the approach side
    "approach_beside": (-2.0, "tip", 0.5),    # beside the funnel at approach altitude
    "under_tip": (0.0, "mid", 0.0),           # under the tip, inside the target cylinder
    "far_low": (1.0, "mid", 0.0),             # under the wall, far side
}

DEFAULT_DESIGNS = (
    "band@max",                     # PS2-RL paper: 7-D hover band at mid-gap, largest certified level
    "band@max/4",                   # the same band, half the vertical extent
    "points:approach_beside@max",   # one ellipsoid beside the funnel (approach side)
    "points:approach_beside@2",     # ... smaller
    "points:approach_low@max",      # one ellipsoid under the wall (approach side)
    "points:under_tip@max",         # one ellipsoid under the tip
    "points:approach_beside+under_tip@max",
    "anywhere@2",                   # brake: stop wherever clear
    "anywhere@8",
)


def load_certificate(path: str) -> dict[str, Any]:
    return json.loads(resolve(path).read_text(encoding="utf-8"))


def point_xyz(cfg: ConeObstacleConfig, name: str) -> tuple[float, float, float]:
    dx, mode, val = NAMED_POINTS[name]
    z = (float(cfg.ground_z) + 0.5 * cfg.gap if mode == "mid" else float(cfg.tip_z)) + float(val)
    return (float(cfg.axis_x) + float(dx), float(cfg.axis_y), z)


def geometric_cap(cfg: ConeObstacleConfig, kind: str, points: tuple, cert: dict[str, Any]) -> float:
    """Largest level whose base set stays in S (inf when the design has no such cap)."""
    import jax.numpy as jnp
    from ps2rl.sets.quadrotor_cone_obstacle_sets import ConeObstacleSafeSet

    if kind == "band":
        ext = float(cert["lqr7_band"]["vertical_half_extent_per_sqrt_c"])
        zb = cfg.band_height
        return min(float(cfg.tip_z) - zb, zb - float(cfg.ground_z)) ** 2 / ext**2
    if kind == "points":
        safe = ConeObstacleSafeSet.from_config(cfg)
        m_h, m_v = (float(cert["lqr9"]["metric"][k]) for k in ("m_h", "m_v"))
        caps = []
        for pt in points:
            d = float(safe.m_distance(jnp.asarray(pt, jnp.float32), m_h, m_v))
            caps.append(min(d * d, (pt[2] - float(cfg.ground_z)) ** 2 * m_v))
        return min(caps)
    return float("inf")   # anywhere: the clearance scales with the remaining energy


def lqr_cap(kind: str, cert: dict[str, Any]) -> float:
    return float(cert[{"band": "lqr7_band", "points": "lqr9", "anywhere": "lqr6_brake"}[kind]]["c_max"])


def design_config(cfg: ConeObstacleConfig, spec: str, cert: dict[str, Any], cap_safety: float = 0.98) -> ConeObstacleConfig:
    """``<kind>[:<point>+<point>...]@<level>``; level: a number, ``max`` or ``max/<k>``
    (``max`` = cap_safety x min(certified LQR level, geometric cap))."""
    head, _, level = spec.partition("@")
    kind, _, pts = head.partition(":")
    points = tuple(point_xyz(cfg, n) for n in pts.split("+")) if pts else ()
    cap = cap_safety * min(lqr_cap(kind, cert), geometric_cap(cfg, kind, points, cert))
    level = level or "max"
    if level.startswith("max"):
        c = cap / (float(level.split("/")[1]) if "/" in level else 1.0)
    else:
        c = float(level)
        if c > cap:
            raise ValueError(f"{spec}: level {c} exceeds the cap {cap:.3f} at tip_z={cfg.tip_z}")
    extra = {"hover_points": points} if kind == "points" else {}
    return cfg.replace(base_set=kind, base_set_c=c, **extra)


__all__ = ["DEFAULT_DESIGNS", "NAMED_POINTS", "ROOT", "design_config", "geometric_cap", "level_state", "load_certificate",
           "lqr_cap", "make_config", "parse_list", "point_xyz", "resolve", "vehicle_fields"]
