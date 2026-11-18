"""Strike task among tall obstacles and slow chasers: scene, safe set, re-anchored hover base set, hover-LQR
backup and the backup's observation.

Coordinates are relative to the drone's position at the start of a backup rollout (world-aligned axes).

Scene (one flat vector, ``scene_dim`` numbers):
  obstacles  (n_obstacle_slots, 7): cx, cy, bx, by, yaw, rr, present. Tall rounded rectangles in the horizontal plane
             (cylinder: bx = by = 0, rr = radius; box: rr = 0, half-sizes bx, by, rotated by yaw).
  balls      (n_chaser_slots, 4): cx, cy, cz, present. The ball a chaser can reach within the backup horizon T,
             treated as a static obstacle: radius R = r_c + r_d + v_c T (drone radius included).

Safe set:  z - z_floor >= 0,   sdf_O(p_xy) >= r_d,   |p - c_j| >= R.

Base set (hover at the current point): with r = (v, phi) and the hover LQR level V = (p - a, r)' P (p - a, r),
the anchor a = p + D r, D = P_pp^-1 P_pr, minimizes V over the hover point; V_min(r) = r' S r with
S = P_rr - P_rp P_pp^-1 P_pr. A state is in B when V_min <= c_B and the hover ellipsoid {V <= c_B} around a is
clear of everything:
    sdf_O(a_xy) >= rho_xy + r_d,   |a - c_j| >= R + rho,   a_z - z_ext >= z_floor,
with rho_xy / rho / z_ext the ellipsoid's horizontal / 3-D / vertical position extents. Once B is reached the anchor
is frozen and the hover LQR at it keeps the state in {V <= c_B} (c_B below the certificate's bounds).

LQR backup (the baseline): u = u* - K~ r, K~ = K_r - K_p D, i.e. the hover LQR re-anchored at every step - it brakes
to a stop wherever it is, without a goal point.

Observation of the learned backup (``obs_dim`` numbers):
  z - z_floor, v (3), q (4);
  per obstacle slot: closest-point vector (2), centre offset (2), half-sizes (2), cos 2yaw, sin 2yaw, rounding, present;
  per chaser slot: centre offset (3), ball radius, present.
The slots keep the order the scene had at the rollout start (``sort_scene``: nearest obstacle first), so the policy
input stays continuous along a rollout. Safe-set and base-set membership are exact functions of the observation
(``margins_from_obs``) as long as every obstacle of the scene has a slot.

Moving chasers (``chaser_set = "guidance"``, the full-arena Phase I): the chasers pursue the drone at v_c during the
rollout (``chaser_pursuit_step``, the scene's chaser centres move) and the chaser part of the sets is the guidance-law
escape margin instead of the static ball:
  safe set   h_c = |p - c| - (r_c + r_d) - s+ delta - s+^2 / (2 a_e) >= 0,  s = v_c - e'v,  e = (p - c) / |p - c|
             (the drone can still out-accelerate a pursuer closing at v_c; it depends on the drone's velocity);
  base set   the hovering drone keeps h_c >= chaser_base_margin everywhere in its hover ellipsoid: with the state's level
             l = V_min(r) <= c_B the hover LQR keeps |p - a| <= rho sqrt(l / c_B) and |v| <= v_ext sqrt(l / c_B), so
             |a - c| - rho_l - (r_c + r_d) - s_B delta - s_B^2 / (2 a_e) >= chaser_base_margin,  s_B = v_c + v_l.
             A pursuer closes at most v_c, so the hover stays safe for chaser_base_margin / v_c seconds (the filter
             re-plans every step).
  observation per chaser slot: centre offset c - p (3), h_c, closing speed s, present (6 numbers).
Arena walls (``arena_walls``): 5 more scene numbers (x_lo, x_hi, y_lo, y_hi, z_hi; far = absent), the safe set keeps r_d
from them, the base set the hover ellipsoid, and the observation ends with the 5 distances (capped at wall_obs_cap).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
import json
from pathlib import Path
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import scipy.linalg as sla

from ps2rl.envs.quadrotor_env import quadrotor_step_euler
from ps2rl.utils.quaternion import normalize_quaternion, rotation_matrix_from_quaternion

Array = jax.Array

OBST_FIELDS = 7   # cx, cy, bx, by, yaw, rr, present
BALL_FIELDS = 4   # cx, cy, cz, present
EGO_FEATS = 8     # z - z_floor, v, q
OBST_FEATS = 10   # closest-point vector, centre offset, half-sizes, cos 2yaw, sin 2yaw, rounding, present
BALL_FEATS = 5    # centre offset, radius, present
BALL_FEATS_GUIDANCE = 6   # moving chasers: centre offset, escape margin h_c, closing speed, present
WALL_FIELDS = 5   # arena walls: x_lo, x_hi, y_lo, y_hi, z_hi (scene) / the five distances (observation)
_ABSENT = 1.0e3   # distance reported for an empty slot


@dataclass(frozen=True)
class StrikeConfig:
    # vehicle (thrust-acceleration and body-rate box) and integration step
    dt: float = 0.02
    gravity: float = 9.81
    a_cmd_min: float = 0.0
    a_cmd_max: float = 39.24
    omega_max: float = 18.0
    # backup horizon [s]
    T: float = 1.5
    # floor height, drone radius, chasers (radius, speed limit)
    z_floor: float = 1.0
    r_drone: float = 0.3
    chaser_radius: float = 0.4
    chaser_speed: float = 2.0
    # hover LQR on e = (p - a, v, phi) and the base-set level
    lqr_q: tuple = (2.0, 2.0, 8.0, 0.01, 0.01, 0.5, 8.0, 8.0, 2.0)
    lqr_r: tuple = (0.06, 0.15, 0.15, 0.12)
    c_B: float = 50.0
    # moderate wind bound [m/s^2] used by the base-set certificate (one-step decrease under the worst wind)
    wind_accel_max: float = 0.3
    # hold time [s]: the base set keeps the hover point clear of a chaser's reach for T + chaser_hold (anchor clearance
    # R + v_c * chaser_hold + rho; the safe set keeps R). A certified plan then survives chaser_hold seconds of chaser
    # approach, time for the filter to find a plan that moves further away. 0 = the original base set.
    chaser_hold: float = 0.0
    # observation slots
    n_obstacle_slots: int = 8
    n_chaser_slots: int = 2
    # what the learned backup outputs: "residual_lqr" u = u_LQR(x) + du (the trainer's action penalty |du/scale|^2 then
    # anchors it to the LQR), "hover" u = u_hover + du (penalty pulls towards hover), or "residual_flee" (moving chasers)
    # u = u_flee(x, scene) + du: the flee-then-hover heuristic (``flee_action``) is the base and the anchor
    backup_action: str = "residual_lqr"
    # chaser part of the safe / base set (module docstring): "ball" - each chaser a static ball of radius ball_radius
    # around its position at the rollout start (the original); "guidance" - moving chasers (pure pursuit at
    # chaser_speed), escape margin h_c >= 0 with evasive acceleration chaser_a_evade [m/s^2] after chaser_delay [s], base
    # set: the hover keeps h_c >= chaser_base_margin [m] (safe for chaser_base_margin / chaser_speed seconds)
    chaser_set: str = "ball"
    chaser_a_evade: float = 5.0
    chaser_delay: float = 0.2
    chaser_base_margin: float = 2.0
    # arena walls in the scene and the observation (module docstring)
    arena_walls: bool = False
    wall_obs_cap: float = 20.0
    # flee-then-hover heuristic (``flee_action``; the base of backup_action "residual_flee"): the hover LQR while the hover
    # keeps the base set's chaser margin plus flee_switch_margin [m], else a velocity loop at flee_speed [m/s] away from the
    # most critical chaser, pushed away from obstacles / walls closer than flee_push_dist beyond the hover clearance, at
    # least flee_z_min above the floor and flee_ceiling_margin below the ceiling; loop gains and tilt limit
    flee_speed: float = 6.0
    flee_switch_margin: float = 1.0
    flee_push_dist: float = 1.5
    flee_z_min: float = 1.0
    flee_ceiling_margin: float = 3.0
    flee_k_vel: float = 2.0
    flee_k_att: float = 10.0
    flee_k_yaw: float = 3.0
    flee_tilt_max_deg: float = 45.0

    def __post_init__(self) -> None:
        if self.backup_action not in ("residual_lqr", "hover", "residual_flee"):
            raise ValueError(f"backup_action must be 'residual_lqr', 'hover' or 'residual_flee', got {self.backup_action!r}")
        if self.chaser_set not in ("ball", "guidance"):
            raise ValueError(f"chaser_set must be 'ball' or 'guidance', got {self.chaser_set!r}")
        if self.backup_action == "residual_flee" and self.chaser_set != "guidance":
            raise ValueError("backup_action 'residual_flee' needs moving chasers (chaser_set 'guidance')")
        object.__setattr__(self, "lqr_q", tuple(float(v) for v in self.lqr_q))
        object.__setattr__(self, "lqr_r", tuple(float(v) for v in self.lqr_r))
        if len(self.lqr_q) != 9 or len(self.lqr_r) != 4:
            raise ValueError("lqr_q needs 9 weights (p, v, phi) and lqr_r 4 (thrust, body rates)")

    @property
    def num_steps(self) -> int:
        return int(round(self.T / self.dt))

    @property
    def ball_radius(self) -> float:
        return float(self.chaser_radius + self.r_drone + self.chaser_speed * self.T)

    @property
    def action_low(self) -> tuple[float, float, float, float]:
        w = float(self.omega_max)
        return (float(self.a_cmd_min), -w, -w, -w)

    @property
    def action_high(self) -> tuple[float, float, float, float]:
        w = float(self.omega_max)
        return (float(self.a_cmd_max), w, w, w)

    @property
    def action_scale(self) -> tuple[float, float, float, float]:
        w = float(self.omega_max)
        return (float(self.a_cmd_max), w, w, w)

    @property
    def moving_chasers(self) -> bool:
        return self.chaser_set == "guidance"

    @property
    def contact_radius(self) -> float:
        return float(self.chaser_radius + self.r_drone)

    @property
    def ball_feats(self) -> int:
        return BALL_FEATS_GUIDANCE if self.moving_chasers else BALL_FEATS

    @property
    def wall_fields(self) -> int:
        return WALL_FIELDS if self.arena_walls else 0

    @property
    def scene_dim(self) -> int:
        return self.n_obstacle_slots * OBST_FIELDS + self.n_chaser_slots * BALL_FIELDS + self.wall_fields

    @property
    def obs_dim(self) -> int:
        return EGO_FEATS + self.n_obstacle_slots * OBST_FEATS + self.n_chaser_slots * self.ball_feats + self.wall_fields

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["lqr_q"], d["lqr_r"] = list(self.lqr_q), list(self.lqr_r)
        return d

    def replace(self, **kw) -> "StrikeConfig":
        return replace(self, **kw)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "StrikeConfig":
        names = {f.name for f in fields(cls)}
        unknown = set(payload) - names
        if unknown:
            raise ValueError(f"unknown strike config keys {sorted(unknown)}")
        return cls(**payload)


# fields the plant, contacts and chasers depend on: an environment and a safety filter built from two strike configs (e.g.
# the striker's observation format and a backup trained with another scene format) must agree on these
PLANT_FIELDS = ("dt", "gravity", "a_cmd_min", "a_cmd_max", "omega_max", "z_floor", "r_drone", "chaser_radius", "chaser_speed",
                "n_chaser_slots")


def check_same_plant(a: StrikeConfig, b: StrikeConfig, what: str = "") -> None:
    diff = {f: (getattr(a, f), getattr(b, f)) for f in PLANT_FIELDS if getattr(a, f) != getattr(b, f)}
    if diff:
        raise ValueError(f"{what}: the two strike configs describe different plants / chasers: {diff}")


def load_strike_task(path: str | Path) -> dict[str, Any]:
    """Task file: {"strike": {...}, "region": {...}, "sa": {...}, "env": {...}, "tracker": {...}} (blocks optional)."""
    task = json.loads(Path(path).read_text(encoding="utf-8"))
    unknown = set(task) - {"strike", "region", "sa", "env", "tracker", "notes"}
    if unknown:
        raise ValueError(f"{path}: unknown blocks {sorted(unknown)}")
    return task


# ----------------------------------------------------------------------------- hover LQR (numpy, float64)
@dataclass(frozen=True)
class HoverAnchorLQR:
    P: np.ndarray
    K: np.ndarray
    S: np.ndarray      # V_min(r) = r' S r
    D: np.ndarray      # anchor a = p + D r
    Kt: np.ndarray     # re-anchored control u = u* - Kt r
    rho_xy: float      # horizontal position extent of {V <= c_B}
    rho: float         # 3-D position extent
    z_ext: float       # vertical position extent
    v_ext: float = 0.0  # 3-D velocity extent of {V <= c_B}

    def as_dict(self) -> dict[str, Any]:
        return {"rho_xy": self.rho_xy, "rho": self.rho, "z_ext": self.z_ext, "v_ext": self.v_ext,
                "D": self.D.tolist(), "Kt": self.Kt.tolist()}


def hover_linearization(gravity: float) -> tuple[np.ndarray, np.ndarray]:
    """Continuous (A, B) about hover in e = (p, v, phi), phi = small-angle attitude (chart of ``attitude_chart``)."""
    A = np.zeros((9, 9))
    A[0:3, 3:6] = np.eye(3)
    A[3, 7] = gravity       # v_x' = g phi_y
    A[4, 6] = -gravity      # v_y' = -g phi_x
    B = np.zeros((9, 4))
    B[5, 0] = 1.0           # v_z' = a_cmd - g
    B[6:9, 1:4] = np.eye(3)  # phi' = omega
    return A, B


def design_hover_lqr(cfg: StrikeConfig) -> HoverAnchorLQR:
    A, B = hover_linearization(cfg.gravity)
    Ad, Bd = np.eye(9) + cfg.dt * A, cfg.dt * B        # explicit Euler: the Jacobian of the sampled plant step
    Q, R = np.diag(cfg.lqr_q), np.diag(cfg.lqr_r)
    P = sla.solve_discrete_are(Ad, Bd, Q, R)
    K = np.linalg.solve(R + Bd.T @ P @ Bd, Bd.T @ P @ Ad)
    Ppp, Ppr, Prr = P[:3, :3], P[:3, 3:], P[3:, 3:]
    D = np.linalg.solve(Ppp, Ppr)
    S = Prr - Ppr.T @ D
    Kt = K[:, 3:] - K[:, :3] @ D
    Pi = np.linalg.inv(P)
    c = float(cfg.c_B)
    return HoverAnchorLQR(P=P, K=K, S=S, D=D, Kt=Kt,
                          rho_xy=float(np.sqrt(c * np.linalg.eigvalsh(Pi[0:2, 0:2]).max())),
                          rho=float(np.sqrt(c * np.linalg.eigvalsh(Pi[0:3, 0:3]).max())),
                          z_ext=float(np.sqrt(c * Pi[2, 2])),
                          v_ext=float(np.sqrt(c * np.linalg.eigvalsh(Pi[3:6, 3:6]).max())))


# ----------------------------------------------------------------------------- base-set certificate (numpy)
def _np_chart(q: np.ndarray) -> np.ndarray:
    s = np.where(q[..., :1] < 0, -1.0, 1.0)
    return 2.0 * s * q[..., 1:]


def _np_step(x: np.ndarray, u: np.ndarray, cfg: StrikeConfig, d: np.ndarray | None = None) -> np.ndarray:
    """Batched explicit-Euler plant step (same model as ``plant_step``), optional acceleration disturbance d."""
    a = np.clip(u[:, 0], cfg.a_cmd_min, cfg.a_cmd_max)
    w = np.clip(u[:, 1:], -cfg.omega_max, cfg.omega_max)
    qw, qx, qy, qz = x[:, 6], x[:, 7], x[:, 8], x[:, 9]
    e3 = np.stack([2 * (qx * qz + qw * qy), 2 * (qy * qz - qw * qx), 1 - 2 * (qx * qx + qy * qy)], 1)
    vdot = e3 * a[:, None] - np.array([0.0, 0.0, cfg.gravity]) + (0.0 if d is None else d)
    qd = 0.5 * np.stack([-qx * w[:, 0] - qy * w[:, 1] - qz * w[:, 2],
                         qw * w[:, 0] - qz * w[:, 1] + qy * w[:, 2],
                         qz * w[:, 0] + qw * w[:, 1] - qx * w[:, 2],
                         -qy * w[:, 0] + qx * w[:, 1] + qw * w[:, 2]], 1)
    q2 = x[:, 6:] + cfg.dt * qd
    q2 /= np.linalg.norm(q2, axis=1, keepdims=True)
    return np.concatenate([x[:, :3] + cfg.dt * x[:, 3:6], x[:, 3:6] + cfg.dt * vdot, q2], 1)


def _lyap_ratio(cfg: StrikeConfig, lqr: HoverAnchorLQR, c: float, wind: float, n: int, seed: int) -> float:
    """Largest sampled V+/V on the level set {V = c} of the hover LQR at a fixed anchor (worst wind direction)."""
    P, K = lqr.P, lqr.K
    w, U = np.linalg.eigh(P)
    Pmh = U @ np.diag(w ** -0.5) @ U.T
    z = np.random.default_rng(seed).normal(size=(n, 9))
    e = np.sqrt(c) * (z / np.linalg.norm(z, axis=1, keepdims=True)) @ Pmh.T
    e = e[np.linalg.norm(e[:, 6:9], axis=1) < 2.0]               # inside the attitude chart
    phi = e[:, 6:9]
    sh = np.linalg.norm(phi, axis=1, keepdims=True) / 2.0       # |q_vec| = |phi| / 2
    q = np.concatenate([np.sqrt(np.maximum(1.0 - sh ** 2, 0.0)), phi / 2.0], 1)
    x = np.concatenate([e[:, :3], e[:, 3:6], q], 1)
    u = np.array([cfg.gravity, 0.0, 0.0, 0.0]) - e @ K.T
    xn = _np_step(x, u, cfg)
    en = np.concatenate([xn[:, :6], _np_chart(xn[:, 6:])], 1)
    if wind > 0.0:
        g = (en @ P)[:, 3:6]
        d = wind * g / np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-12)
        xn = _np_step(x, u, cfg, d)
        en = np.concatenate([xn[:, :6], _np_chart(xn[:, 6:])], 1)
    return float(np.max(np.einsum("ni,ij,nj->n", en, P, en) / np.einsum("ni,ij,nj->n", e, P, e)))


def certify_base_set(cfg: StrikeConfig, lqr: HoverAnchorLQR | None = None, *, n: int = 4000, seed: int = 0) -> dict[str, Any]:
    """Bounds on the base-set level c: input box (c_U), attitude chart (c_chart) and sampled nonlinear one-step decrease
    of the hover LQR without and with the worst moderate wind (c_Lyap). ``ok`` when c_B is below all of them."""
    lqr = lqr or design_hover_lqr(cfg)
    Pi = np.linalg.inv(lqr.P)
    margins = np.array([min(cfg.gravity - cfg.a_cmd_min, cfg.a_cmd_max - cfg.gravity)] + [cfg.omega_max] * 3)
    c_u = float(min(margins[i] ** 2 / (lqr.K[i] @ Pi @ lqr.K[i]) for i in range(4)))
    c_chart = float(4.0 / np.linalg.eigvalsh(Pi[6:9, 6:9]).max())
    hi = min(c_u, c_chart)

    def c_lyap(wind: float) -> float:
        if _lyap_ratio(cfg, lqr, hi, wind, n, seed) < 1.0:
            return hi
        lo_, hi_ = 0.0, hi
        for _ in range(14):
            mid = 0.5 * (lo_ + hi_)
            lo_, hi_ = (mid, hi_) if _lyap_ratio(cfg, lqr, mid, wind, n, seed) < 1.0 else (lo_, mid)
        return lo_

    c_l0, c_lw = c_lyap(0.0), c_lyap(float(cfg.wind_accel_max))
    Si = np.linalg.inv(lqr.S)
    c = float(cfg.c_B)
    return {
        "c_B": c, "c_U": c_u, "c_chart": c_chart, "c_lyap": c_l0, "c_lyap_wind": c_lw,
        "wind_accel_max": float(cfg.wind_accel_max),
        "ratio_at_c_B": _lyap_ratio(cfg, lqr, c, 0.0, n, seed + 1),
        "ratio_at_c_B_wind": _lyap_ratio(cfg, lqr, c, float(cfg.wind_accel_max), n, seed + 1),
        "rho_xy": lqr.rho_xy, "rho": lqr.rho, "z_ext": lqr.z_ext,
        "vxy_max_in_B": float(np.sqrt(c * np.linalg.eigvalsh(Si[0:2, 0:2]).max())),
        "vz_max_in_B": float(np.sqrt(c * Si[2, 2])),
        "tilt_max_in_B_deg": float(np.degrees(np.sqrt(c * np.linalg.eigvalsh(Si[3:5, 3:5]).max()))),
        "ok": bool(c <= min(c_u, c_chart, c_lw)),
    }


# ----------------------------------------------------------------------------- JAX model
class StrikeModel(NamedTuple):
    """Everything the JAX functions need (arrays as float32 constants).

    low / high: the plant's action box. The learned backup outputs du in [d_low, d_high] on top of a base action
    (``backup_base_action``: the LQR for backup_action "residual_lqr", hover for "hover"); the safe-arrival trainer's
    action penalty |du / scale|^2 then pulls towards that base action."""
    cfg: StrikeConfig
    lqr: HoverAnchorLQR
    S: Array
    D: Array
    Kt: Array
    low: Array
    high: Array
    scale: Array
    u_hover: Array
    d_low: Array
    d_high: Array


def build_model(cfg: StrikeConfig) -> StrikeModel:
    lqr = design_hover_lqr(cfg)
    f = lambda a: jnp.asarray(a, dtype=jnp.float32)
    u_hover = np.array([cfg.gravity, 0.0, 0.0, 0.0])
    if cfg.backup_action in ("residual_lqr", "residual_flee"):     # any plant action is reachable from any base action
        d_low, d_high = -np.asarray(cfg.action_scale), np.asarray(cfg.action_scale)
    else:
        d_low, d_high = np.asarray(cfg.action_low) - u_hover, np.asarray(cfg.action_high) - u_hover
    return StrikeModel(cfg=cfg, lqr=lqr, S=f(lqr.S), D=f(lqr.D), Kt=f(lqr.Kt), low=f(cfg.action_low),
                       high=f(cfg.action_high), scale=f(cfg.action_scale), u_hover=f(u_hover), d_low=f(d_low), d_high=f(d_high))


def attitude_chart(q: Array) -> Array:
    """phi = 2 sgn(q_w) q_xyz (small-angle rotation vector of the attitude; one-to-one for |phi| < 2)."""
    return 2.0 * jnp.where(q[0] < 0.0, -1.0, 1.0) * q[1:4]


def plant_step(m: StrikeModel, x: Array, u: Array) -> Array:
    cfg = m.cfg
    xn = quadrotor_step_euler(x, u, cfg.dt, cfg.gravity, cfg.a_cmd_min, cfg.a_cmd_max, cfg.omega_max)
    return xn.at[6:10].set(normalize_quaternion(xn[6:10]))


def lqr_action(m: StrikeModel, x: Array) -> Array:
    """The re-anchored hover LQR: brake to a stop wherever the vehicle is."""
    r = jnp.concatenate([x[3:6], attitude_chart(x[6:10])])
    u = jnp.array([m.cfg.gravity, 0.0, 0.0, 0.0], dtype=x.dtype) - m.Kt @ r
    return jnp.clip(u, m.low, m.high)


def backup_base_action(m: StrikeModel, x: Array, scene: Array | None = None) -> Array:
    """The action the learned backup's output du is added to (the flee heuristic needs the scene)."""
    if m.cfg.backup_action == "residual_lqr":
        return lqr_action(m, x)
    if m.cfg.backup_action == "residual_flee":
        if scene is None:
            raise ValueError("backup_action 'residual_flee' needs the scene")
        return flee_action(m, x, scene)
    return m.u_hover


def velocity_track_action(m: StrikeModel, x: Array, v_des: Array) -> Array:
    """Thrust + body rates that track a velocity (thrust-axis attitude loop, yaw to zero; flee_* gains / tilt limit)."""
    cfg = m.cfg
    v, q = x[3:6], x[6:10]
    a = cfg.flee_k_vel * (v_des - v)
    amax = cfg.gravity * np.tan(np.radians(cfg.flee_tilt_max_deg))
    axy = a[0:2] * jnp.minimum(1.0, amax / (jnp.linalg.norm(a[0:2]) + 1e-9))
    a_des = jnp.concatenate([axy, a[2:3] + cfg.gravity])
    Rm = rotation_matrix_from_quaternion(q)
    zb = Rm[:, 2]
    zd = a_des / jnp.maximum(jnp.linalg.norm(a_des), 1e-6)
    w = cfg.flee_k_att * (Rm.T @ jnp.cross(zb, zd))
    w = w.at[2].add(-cfg.flee_k_yaw * jnp.arctan2(Rm[1, 0], Rm[0, 0]))
    return jnp.clip(jnp.concatenate([jnp.dot(a_des, zb)[None], w]), m.low, m.high)


def flee_action(m: StrikeModel, x: Array, scene: Array) -> Array:
    """Flee-then-hover heuristic for moving chasers (StrikeConfig flee_*): the hover LQR while the hover keeps the base
    set's chaser margin plus flee_switch_margin, else a velocity loop at flee_speed away from the most critical chaser
    (least h_c), pushed away from obstacles and walls closer than flee_push_dist beyond the hover clearance. Without
    chasers it is the hover LQR."""
    cfg = m.cfg
    obst, balls = split_scene(cfg, scene)
    o_rel, b_rel = relative_scene(x, obst, balls)
    on = b_rel[:, 3] > 0.5
    r = jnp.concatenate([x[3:6], attitude_chart(x[6:10])])
    off = m.D @ r
    hover_ok = (jnp.min(jnp.where(on, hover_chaser_margin(m, b_rel[:, 0:3] - off[None, :], r @ m.S @ r), _ABSENT))
                >= cfg.flee_switch_margin)
    h_c, _ = escape_margin(cfg, b_rel[:, 0:3], x[3:6])
    j = jnp.argmin(jnp.where(on, h_c, 1e9))
    away = -b_rel[j, 0:2] / (jnp.linalg.norm(b_rel[j, 0:2]) + 1e-6)
    clear = float(m.lqr.rho_xy + cfg.r_drone)
    d_o, grad_o = jax.value_and_grad(lambda q_: jnp.min(obstacle_distances(q_, o_rel)))(jnp.zeros(2, x.dtype))
    push_o = jnp.clip((cfg.flee_push_dist - (d_o - clear)) / cfg.flee_push_dist, 0.0, 1.0)
    want = away + 2.0 * push_o * grad_o / (jnp.linalg.norm(grad_o) + 1e-9)
    walls = scene_walls(cfg, scene)
    z_top = jnp.asarray(jnp.inf, x.dtype)
    if walls is not None:
        gaps = wall_gaps(relative_walls(x, walls), jnp.zeros(3, x.dtype))
        push_w = jnp.clip((cfg.flee_push_dist - (gaps[0:4] - clear)) / cfg.flee_push_dist, 0.0, 1.0)
        n_w = jnp.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]], x.dtype)
        want = want + 2.0 * (push_w[:, None] * n_w).sum(axis=0)
        z_top = walls[4] - cfg.flee_ceiling_margin
    dirn = want / (jnp.linalg.norm(want) + 1e-6)
    z_ref = jnp.clip(x[2], cfg.z_floor + cfg.flee_z_min, z_top)
    u_flee = velocity_track_action(m, x, jnp.concatenate([cfg.flee_speed * dirn, (1.5 * (z_ref - x[2]))[None]]))
    return jnp.where(hover_ok, lqr_action(m, x), u_flee)


def anchor(m: StrikeModel, x: Array) -> Array:
    r = jnp.concatenate([x[3:6], attitude_chart(x[6:10])])
    return x[0:3] + m.D @ r


def split_scene(cfg: StrikeConfig, scene: Array) -> tuple[Array, Array]:
    """(obstacles, chaser balls); the arena walls (if any) are ``scene_walls``."""
    k = cfg.n_obstacle_slots * OBST_FIELDS
    kb = k + cfg.n_chaser_slots * BALL_FIELDS
    return scene[:k].reshape(cfg.n_obstacle_slots, OBST_FIELDS), scene[k:kb].reshape(cfg.n_chaser_slots, BALL_FIELDS)


def scene_walls(cfg: StrikeConfig, scene: Array) -> Array | None:
    """(x_lo, x_hi, y_lo, y_hi, z_hi) of a scene with arena walls, else None."""
    return scene[cfg.scene_dim - WALL_FIELDS:cfg.scene_dim] if cfg.arena_walls else None


def join_scene(obst: Array, balls: Array, walls: Array | None = None) -> Array:
    parts = [obst.reshape(-1), balls.reshape(-1)] + ([] if walls is None else [jnp.asarray(walls, obst.dtype).reshape(-1)])
    return jnp.concatenate(parts)


def arena_walls(arena: tuple) -> Array:
    """Scene walls of the strike arena (x_min, x_max, y_half, z_max): (x_min, x_max, -y_half, y_half, z_max)."""
    return jnp.asarray([arena[0], arena[1], -arena[2], arena[2], arena[3]], jnp.float32)


def relative_walls(x: Array, walls: Array) -> Array:
    return walls - jnp.stack([x[0], x[0], x[1], x[1], x[2]])


def wall_gaps(walls_rel: Array, point: Array) -> Array:
    """(5,) distances from a point (relative to the drone, 3-D) to the four walls and the ceiling."""
    return jnp.stack([point[0] - walls_rel[0], walls_rel[1] - point[0], point[1] - walls_rel[2], walls_rel[3] - point[1],
                      walls_rel[4] - point[2]])


def escape_margin(cfg: StrikeConfig, rel_c: Array, v: Array) -> tuple[Array, Array]:
    """Guidance-law escape margin of the drone (velocity v) against chasers at rel_c = c - p (C, 3): (h_c, s) each (C,)."""
    R = jnp.sqrt(jnp.sum(rel_c ** 2, axis=-1) + 1e-12)
    e = -rel_c / R[..., None]                       # line of sight from the chaser to the drone
    s = jnp.maximum(cfg.chaser_speed - e @ v, 0.0)
    return R - cfg.contact_radius - s * cfg.chaser_delay - s ** 2 / (2.0 * cfg.chaser_a_evade), s


def hover_chaser_margin(m: StrikeModel, rel_ca: Array, level: Array) -> Array:
    """(C,) base-set chaser margin of a hover at anchor a (rel_ca = c - a) entered at level l = V <= c_B: the worst h_c
    over its hover ellipsoid {V <= l} minus chaser_base_margin (module docstring)."""
    cfg, lqr = m.cfg, m.lqr
    f = jnp.sqrt(jnp.clip(level / cfg.c_B, 0.0, 1.0))
    s_b = cfg.chaser_speed + lqr.v_ext * f
    dist = jnp.sqrt(jnp.sum(rel_ca ** 2, axis=-1) + 1e-12)
    return (dist - lqr.rho * f - cfg.contact_radius - s_b * cfg.chaser_delay - s_b ** 2 / (2.0 * cfg.chaser_a_evade)
            - cfg.chaser_base_margin)


def chaser_pursuit_step(cfg: StrikeConfig, chaser_p: Array, p_drone: Array) -> Array:
    """One step of pure pursuit at v_c toward the drone (the worst case for the range), above the floor."""
    rel = p_drone[None, :] - chaser_p
    d = jnp.sqrt(jnp.sum(rel ** 2, axis=1, keepdims=True) + 1e-12)
    cp = chaser_p + jnp.minimum(cfg.chaser_speed * cfg.dt, d) * rel / d
    return cp.at[:, 2].set(jnp.maximum(cp[:, 2], cfg.z_floor + cfg.chaser_radius))


def pursue_scene(m: StrikeModel, scene: Array, p_drone: Array) -> Array:
    """The scene one step later: with moving chasers (chaser_set "guidance") their centres pursue the drone at
    p_drone (its position before the step); static balls otherwise (scene unchanged)."""
    cfg = m.cfg
    if not cfg.moving_chasers:
        return scene
    obst, balls = split_scene(cfg, scene)
    balls = balls.at[:, 0:3].set(chaser_pursuit_step(cfg, balls[:, 0:3], p_drone))
    return join_scene(obst, balls, scene_walls(cfg, scene))


def rrect_sdf(p_rel: Array, bx: Array, by: Array, yaw: Array, rr: Array) -> Array:
    """Signed distance of a point (relative to the centre) to a rounded rectangle."""
    c, s = jnp.cos(yaw), jnp.sin(yaw)
    lx = c * p_rel[0] + s * p_rel[1]
    ly = -s * p_rel[0] + c * p_rel[1]
    qx, qy = jnp.abs(lx) - bx, jnp.abs(ly) - by
    out = jnp.sqrt(jnp.maximum(qx, 0.0) ** 2 + jnp.maximum(qy, 0.0) ** 2 + 1e-12)
    return out + jnp.minimum(jnp.maximum(qx, qy), 0.0) - rr


def obstacle_distances(point: Array, obst_rel: Array) -> Array:
    """(K,) signed distance of a horizontal point (relative to the drone) to each slot (absent -> large)."""
    d = jax.vmap(lambda o: rrect_sdf(point - o[0:2], o[2], o[3], o[4], o[5]))(obst_rel)
    return jnp.where(obst_rel[:, 6] > 0.5, d, _ABSENT)


def ball_distances(point: Array, balls_rel: Array, radius: float) -> Array:
    d = jnp.sqrt(jnp.sum((point[None, :] - balls_rel[:, 0:3]) ** 2, axis=-1) + 1e-12) - radius
    return jnp.where(balls_rel[:, 3] > 0.5, d, _ABSENT)


def margins_rel(m: StrikeModel, h: Array, v: Array, q: Array, obst_rel: Array, balls_rel: Array,
                hold: float | None = None, walls_rel: Array | None = None) -> tuple[Array, Array]:
    """(safe margin [m], base margin) from the drone-relative scene; >= 0 means inside the set.

    obst_rel / balls_rel carry centres relative to the drone; h = z - z_floor. hold: the base set's chaser hold time
    for static balls (default StrikeConfig.chaser_hold; 0 checks the hover point against the true reach over T).
    walls_rel: arena walls relative to the drone (``relative_walls``; None = no walls). The chaser terms follow
    StrikeConfig.chaser_set (static ball or the moving chasers' escape margin)."""
    cfg, lqr = m.cfg, m.lqr
    zero2, zero3 = jnp.zeros(2, h.dtype), jnp.zeros(3, h.dtype)
    r = jnp.concatenate([v, attitude_chart(q)])
    off = m.D @ r                                                   # a - p
    level = r @ m.S @ r
    on = balls_rel[:, 3] > 0.5
    if cfg.moving_chasers:
        h_c, _ = escape_margin(cfg, balls_rel[:, 0:3], v)
        ch_safe = jnp.min(jnp.where(on, h_c, _ABSENT))
        ch_base = jnp.min(jnp.where(on, hover_chaser_margin(m, balls_rel[:, 0:3] - off[None, :], level), _ABSENT))
    else:
        R = cfg.ball_radius
        R_base = R + cfg.chaser_speed * (cfg.chaser_hold if hold is None else hold)
        ch_safe = jnp.min(ball_distances(zero3, balls_rel, R))
        ch_base = jnp.min(ball_distances(off, balls_rel, R_base)) - lqr.rho
    safe = jnp.minimum(jnp.minimum(h, jnp.min(obstacle_distances(zero2, obst_rel)) - cfg.r_drone), ch_safe)
    base = jnp.minimum(
        jnp.minimum(cfg.c_B - level, jnp.min(obstacle_distances(off[0:2], obst_rel)) - lqr.rho_xy - cfg.r_drone),
        jnp.minimum(ch_base, h + off[2] - lqr.z_ext))
    if walls_rel is not None:
        safe = jnp.minimum(safe, jnp.min(wall_gaps(walls_rel, zero3)) - cfg.r_drone)
        g = wall_gaps(walls_rel, off)
        base = jnp.minimum(base, jnp.minimum(jnp.min(g[0:4]) - lqr.rho_xy, g[4] - lqr.z_ext) - cfg.r_drone)
    return safe, base


def relative_scene(x: Array, obst: Array, balls: Array) -> tuple[Array, Array]:
    return obst.at[:, 0:2].add(-x[0:2]), balls.at[:, 0:3].add(-x[0:3])


def margins(m: StrikeModel, x: Array, scene: Array) -> tuple[Array, Array]:
    obst, balls = split_scene(m.cfg, scene)
    o_rel, b_rel = relative_scene(x, obst, balls)
    walls = scene_walls(m.cfg, scene)
    w_rel = None if walls is None else relative_walls(x, walls)
    return margins_rel(m, x[2] - m.cfg.z_floor, x[3:6], x[6:10], o_rel, b_rel, walls_rel=w_rel)


def is_safe(m: StrikeModel, x: Array, scene: Array) -> Array:
    return margins(m, x, scene)[0] >= 0.0


def in_base_set(m: StrikeModel, x: Array, scene: Array) -> Array:
    s, b = margins(m, x, scene)
    return (s >= 0.0) & (b >= 0.0)


def sort_scene(m: StrikeModel, x: Array, scene: Array) -> Array:
    """Obstacles nearest first (signed distance from the drone), chasers nearest first; empty slots last."""
    obst, balls = split_scene(m.cfg, scene)
    o_rel, b_rel = relative_scene(x, obst, balls)
    d_o = obstacle_distances(jnp.zeros(2, x.dtype), o_rel)
    d_b = ball_distances(jnp.zeros(3, x.dtype), b_rel, m.cfg.ball_radius)
    return join_scene(obst[jnp.argsort(d_o)], balls[jnp.argsort(d_b)], scene_walls(m.cfg, scene))


def make_obs(m: StrikeModel, x: Array, scene: Array) -> Array:
    cfg = m.cfg
    obst, balls = split_scene(cfg, scene)
    o_rel, b_rel = relative_scene(x, obst, balls)
    walls = scene_walls(cfg, scene)

    def obstacle_feat(o):
        f = lambda p: rrect_sdf(p - o[0:2], o[2], o[3], o[4], o[5])
        zero = jnp.zeros(2, x.dtype)
        d, g = jax.value_and_grad(f)(zero)
        cp = -d * g                                                    # vector from the drone to the closest point
        feat = jnp.concatenate([cp, o[0:2], o[2:4], jnp.cos(2.0 * o[4])[None], jnp.sin(2.0 * o[4])[None], o[5:7]])
        return jnp.where(o[6] > 0.5, feat, jnp.zeros_like(feat))

    def ball_feat(b):
        if cfg.moving_chasers:
            h_c, s = escape_margin(cfg, b[None, 0:3], x[3:6])
            feat = jnp.concatenate([b[0:3], h_c, s, b[3:4]])
        else:
            feat = jnp.concatenate([b[0:3], jnp.asarray([cfg.ball_radius], x.dtype), b[3:4]])
        return jnp.where(b[3] > 0.5, feat, jnp.zeros_like(feat))

    ego = jnp.concatenate([(x[2] - cfg.z_floor)[None], x[3:6], x[6:10]])
    parts = [ego, jax.vmap(obstacle_feat)(o_rel).reshape(-1), jax.vmap(ball_feat)(b_rel).reshape(-1)]
    if walls is not None:
        parts.append(jnp.minimum(wall_gaps(relative_walls(x, walls), jnp.zeros(3, x.dtype)), cfg.wall_obs_cap))
    return jnp.concatenate(parts).astype(jnp.float32)


def margins_from_obs(m: StrikeModel, o: Array) -> tuple[Array, Array]:
    """Same margins as ``margins`` read back from an observation (exact when every obstacle has a slot; capped wall
    distances only ever under-report walls farther than wall_obs_cap, which no margin is near)."""
    cfg = m.cfg
    k0 = EGO_FEATS
    k1 = k0 + cfg.n_obstacle_slots * OBST_FEATS
    k2 = k1 + cfg.n_chaser_slots * cfg.ball_feats
    of = o[k0:k1].reshape(cfg.n_obstacle_slots, OBST_FEATS)
    bf = o[k1:k2].reshape(cfg.n_chaser_slots, cfg.ball_feats)
    yaw = 0.5 * jnp.arctan2(of[:, 7], of[:, 6])
    o_rel = jnp.stack([of[:, 2], of[:, 3], of[:, 4], of[:, 5], yaw, of[:, 8], of[:, 9]], axis=1)
    b_rel = jnp.concatenate([bf[:, 0:3], bf[:, -1:]], axis=1)
    w_rel = None
    if cfg.arena_walls:
        d = o[k2:k2 + WALL_FIELDS]
        w_rel = jnp.stack([-d[0], d[1], -d[2], d[3], d[4]])
    return margins_rel(m, o[0], o[1:4], o[4:8], o_rel, b_rel, walls_rel=w_rel)


def goal_from_obs(m: StrikeModel, o: Array) -> Array:
    s, b = margins_from_obs(m, o)
    return (s >= 0.0) & (b >= 0.0)


def fail_from_obs(m: StrikeModel, o: Array) -> Array:
    return margins_from_obs(m, o)[0] < 0.0


__all__ = [
    "BALL_FEATS", "BALL_FEATS_GUIDANCE", "BALL_FIELDS", "EGO_FEATS", "OBST_FEATS", "OBST_FIELDS", "PLANT_FIELDS", "WALL_FIELDS",
    "HoverAnchorLQR", "StrikeConfig", "StrikeModel",
    "anchor", "arena_walls", "attitude_chart", "backup_base_action", "build_model", "certify_base_set", "check_same_plant",
    "chaser_pursuit_step", "design_hover_lqr", "escape_margin", "fail_from_obs", "flee_action", "goal_from_obs",
    "hover_chaser_margin", "velocity_track_action",
    "in_base_set", "is_safe", "join_scene", "load_strike_task", "lqr_action", "make_obs", "margins", "margins_from_obs",
    "margins_rel", "obstacle_distances", "plant_step", "pursue_scene", "relative_scene", "relative_walls", "rrect_sdf",
    "scene_walls", "sort_scene", "split_scene", "wall_gaps",
]
