"""Plant, hover LQR, safe / base set and the learned backup for tall cylinders, in numpy (batched).

numpy twins of ``ps2rl.envs.strike_scene`` for the obstacle-only task (n_chaser_slots = 0, cylinders: bx = by = yaw = 0),
checked against it by ``scripts/check_cyl_online_twin.py``. Everything takes a leading batch axis.

  state x (10)            p (3), v (3), q = (w, x, y, z) (4); world axes, z up
  command u (4)           collective thrust acceleration along body z [m/s^2], body rates (3) [rad/s]
  obstacles (M, 4)        cx, cy, radius, present (tall cylinders, absolute positions)
  safe set                z - z_floor >= 0 and |p_xy - c| - r >= r_d (+ margin)
  base set (hover)        r = (v, phi), level r' S r <= c_B, anchor a = p + D r with the hover ellipsoid clear:
                          |a_xy - c| - r >= rho_xy + r_d (+ margin), a_z - z_ext >= z_floor
  backup observation      z - z_floor, v, q, then per slot (n_slots, nearest first): closest-point vector (2), centre
                          offset (2), half-sizes (2, zero), cos 2 yaw (1), sin 2 yaw (0), radius, present

Disturbance (``d``, (..., 3) acceleration [m/s^2], e.g. a disturbance observer's estimate; None = 0): the model adds it
to the acceleration and the hover controllers / base set use the hover equilibrium that cancels it - thrust axis along
g e3 - d, i.e. phi_eq = (d_y, -d_x, 0) / g and thrust g - d_z - so the hover LQR holds its anchor in a steady push
instead of drifting (the re-anchored LQR has no position or integral feedback), and the base set (level, anchor) is
taken about that equilibrium. d = 0 gives the Phase-I sets and controllers exactly.

The constants (hover LQR, base-set extents, action box) come from an exported backup (``load_backup``) or, on a
development machine with scipy, from a strike config (``CylModel.from_strike_config``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

import numpy as np

EGO_FEATS = 8
OBST_FEATS = 10
ABSENT = 1.0e3
FORMAT = "ps2rl-cyl-online-backup/1"


@dataclass
class CylModel:
    dt: float
    gravity: float
    low: np.ndarray            # (4,) action box
    high: np.ndarray
    z_floor: float
    r_drone: float
    c_B: float
    T: float
    n_slots: int
    K: np.ndarray              # (4, 9) hover LQR on e = (p - a, v, phi)
    P: np.ndarray              # (9, 9)
    S: np.ndarray              # (6, 6) level V_min(r) = r' S r
    D: np.ndarray              # (3, 6) anchor a = p + D r
    Kt: np.ndarray             # (4, 6) re-anchored LQR u = u* - Kt r
    rho_xy: float
    rho: float
    z_ext: float
    backup_action: str = "residual_lqr"
    strike: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        for k in ("low", "high", "K", "P", "S", "D", "Kt"):
            setattr(self, k, np.asarray(getattr(self, k), np.float64))
        self.u_hover = np.array([self.gravity, 0.0, 0.0, 0.0])
        self.omega_box = self.high[1:].copy()
        self.scale = np.concatenate([[self.high[0]], self.high[1:]])
        self.KtT = np.ascontiguousarray(self.Kt.T)
        self.DT = np.ascontiguousarray(self.D.T)
        self.KT = np.ascontiguousarray(self.K.T)

    @property
    def num_steps(self) -> int:
        return int(round(self.T / self.dt))

    @property
    def obs_dim(self) -> int:
        return EGO_FEATS + self.n_slots * OBST_FEATS

    # --------------------------------------------------------------------- construction / export
    @classmethod
    def from_strike_config(cls, strike: dict[str, Any]) -> "CylModel":
        """Design the hover LQR from a strike config (needs scipy; the same design as strike_scene.design_hover_lqr)."""
        import scipy.linalg as sla
        s = dict(strike)
        if int(s.get("n_chaser_slots", 2)) != 0:
            raise ValueError("the online model is for the obstacle-only task (strike n_chaser_slots = 0)")
        dt, g = float(s.get("dt", 0.02)), float(s.get("gravity", 9.81))
        A = np.zeros((9, 9)); A[0:3, 3:6] = np.eye(3); A[3, 7] = g; A[4, 6] = -g
        B = np.zeros((9, 4)); B[5, 0] = 1.0; B[6:9, 1:4] = np.eye(3)
        Ad, Bd = np.eye(9) + dt * A, dt * B
        Q, R = np.diag(s["lqr_q"]), np.diag(s["lqr_r"])
        P = sla.solve_discrete_are(Ad, Bd, Q, R)
        K = np.linalg.solve(R + Bd.T @ P @ Bd, Bd.T @ P @ Ad)
        D = np.linalg.solve(P[:3, :3], P[:3, 3:])
        S = P[3:, 3:] - P[:3, 3:].T @ D
        Kt = K[:, 3:] - K[:, :3] @ D
        Pi = np.linalg.inv(P)
        c = float(s["c_B"])
        w = float(s.get("omega_max", 18.0))
        wz = float(s.get("omega_z_max", 0.0)) or w
        return cls(dt=dt, gravity=g, low=np.array([float(s.get("a_cmd_min", 0.0)), -w, -w, -wz]),
                   high=np.array([float(s.get("a_cmd_max", 39.24)), w, w, wz]), z_floor=float(s.get("z_floor", 1.0)),
                   r_drone=float(s.get("r_drone", 0.3)), c_B=c, T=float(s.get("T", 1.5)),
                   n_slots=int(s.get("n_obstacle_slots", 8)), K=K, P=P, S=S, D=D, Kt=Kt,
                   rho_xy=float(np.sqrt(c * np.linalg.eigvalsh(Pi[0:2, 0:2]).max())),
                   rho=float(np.sqrt(c * np.linalg.eigvalsh(Pi[0:3, 0:3]).max())), z_ext=float(np.sqrt(c * Pi[2, 2])),
                   backup_action=str(s.get("backup_action", "residual_lqr")), strike=s)

    def constants(self) -> dict[str, Any]:
        return {"dt": self.dt, "gravity": self.gravity, "low": self.low.tolist(), "high": self.high.tolist(),
                "z_floor": self.z_floor, "r_drone": self.r_drone, "c_B": self.c_B, "T": self.T, "n_slots": self.n_slots,
                "K": self.K.tolist(), "P": self.P.tolist(), "S": self.S.tolist(), "D": self.D.tolist(), "Kt": self.Kt.tolist(),
                "rho_xy": self.rho_xy, "rho": self.rho, "z_ext": self.z_ext, "backup_action": self.backup_action,
                "strike": self.strike}

    @classmethod
    def from_constants(cls, d: dict[str, Any]) -> "CylModel":
        return cls(**{k: d[k] for k in ("dt", "gravity", "low", "high", "z_floor", "r_drone", "c_B", "T", "n_slots", "K",
                                         "P", "S", "D", "Kt", "rho_xy", "rho", "z_ext")},
                   backup_action=d.get("backup_action", "residual_lqr"), strike=d.get("strike", {}))


# ------------------------------------------------------------------------- plant and hover LQR
def chart(q: np.ndarray) -> np.ndarray:
    """phi = 2 sgn(q_w) q_xyz (small-angle rotation vector)."""
    return np.where(q[..., :1] < 0.0, -2.0, 2.0) * q[..., 1:4]


def body_z(q: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.stack([2.0 * (qx * qz + qw * qy), 2.0 * (qy * qz - qw * qx), 1.0 - 2.0 * (qx * qx + qy * qy)], -1)


def step(m: CylModel, x: np.ndarray, u: np.ndarray, accel: np.ndarray | None = None) -> np.ndarray:
    """Explicit-Euler step of the model (the plant the filter certifies with), command clipped to the box."""
    a = np.clip(u[..., 0], m.low[0], m.high[0])
    w = np.clip(u[..., 1:4], m.low[1:], m.high[1:])
    q = x[..., 6:10]
    qw, qx, qy, qz = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    wx, wy, wz = w[..., 0], w[..., 1], w[..., 2]
    out = np.empty(np.broadcast_shapes(x.shape, u.shape[:-1] + (10,)))
    out[..., 0:3] = x[..., 0:3] + m.dt * x[..., 3:6]
    hd = m.dt * a
    out[..., 3] = x[..., 3] + hd * 2.0 * (qx * qz + qw * qy)
    out[..., 4] = x[..., 4] + hd * 2.0 * (qy * qz - qw * qx)
    out[..., 5] = x[..., 5] + hd * (1.0 - 2.0 * (qx * qx + qy * qy)) - m.dt * m.gravity
    if accel is not None:
        out[..., 3:6] += m.dt * accel
    h = 0.5 * m.dt
    out[..., 6] = qw + h * (-qx * wx - qy * wy - qz * wz)
    out[..., 7] = qx + h * (qw * wx - qz * wy + qy * wz)
    out[..., 8] = qy + h * (qz * wx + qw * wy - qx * wz)
    out[..., 9] = qz + h * (-qy * wx + qx * wy + qw * wz)
    out[..., 6:10] /= np.sqrt(np.sum(out[..., 6:10] ** 2, axis=-1, keepdims=True))
    return out


def equilibrium(m: CylModel, d: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    """(r_eq (..., 6), u_eq (..., 4)) of the hover that cancels the push d (module docstring); d None: (0, u_hover)."""
    if d is None:
        return np.zeros(6), m.u_hover
    r_eq = np.zeros(d.shape[:-1] + (6,))
    r_eq[..., 3] = d[..., 1] / m.gravity
    r_eq[..., 4] = -d[..., 0] / m.gravity
    u_eq = np.zeros(d.shape[:-1] + (4,))
    u_eq[..., 0] = m.gravity - d[..., 2]
    return r_eq, u_eq


def rvec(x: np.ndarray, d: np.ndarray | None = None, m: CylModel | None = None) -> np.ndarray:
    """r = (v, phi) - r_eq(d)."""
    r = np.concatenate([x[..., 3:6], chart(x[..., 6:10])], axis=-1)
    if d is not None:
        r[..., 3] -= d[..., 1] / m.gravity
        r[..., 4] += d[..., 0] / m.gravity
    return r


def lqr_action(m: CylModel, x: np.ndarray, r: np.ndarray | None = None, d: np.ndarray | None = None) -> np.ndarray:
    """Re-anchored hover LQR (brakes to a stop wherever it is); r = rvec(x, d) if given."""
    r = rvec(x, d, m) if r is None else r
    return np.clip(equilibrium(m, d)[1] - r @ m.KtT, m.low, m.high)


def held_error(m: CylModel, x: np.ndarray, a0: np.ndarray, d: np.ndarray | None = None) -> np.ndarray:
    return np.concatenate([x[..., 0:3] - a0, rvec(x, d, m)], axis=-1)


def held_action(m: CylModel, x: np.ndarray, a0: np.ndarray, d: np.ndarray | None = None) -> np.ndarray:
    """Hover LQR at a fixed anchor a0 (the controller the base-set certificate is about)."""
    return np.clip(equilibrium(m, d)[1] - held_error(m, x, a0, d) @ m.KT, m.low, m.high)


def held_level(m: CylModel, x: np.ndarray, a0: np.ndarray, d: np.ndarray | None = None) -> np.ndarray:
    e = held_error(m, x, a0, d)
    return np.einsum("...i,ij,...j->...", e, m.P, e)


def anchor(m: CylModel, x: np.ndarray, r: np.ndarray | None = None, d: np.ndarray | None = None) -> np.ndarray:
    r = rvec(x, d, m) if r is None else r
    return x[..., 0:3] + r @ m.DT


# ------------------------------------------------------------------------- sets
def surface_distance(pxy: np.ndarray, obst: np.ndarray) -> np.ndarray:
    """(..., M) horizontal distance from points (..., 2) to the cylinders' surfaces (absent: ABSENT)."""
    d = np.sqrt(np.sum((obst[..., 0:2] - pxy[..., None, :]) ** 2, axis=-1) + 1e-12) - obst[..., 2]
    return np.where(obst[..., 3] > 0.5, d, ABSENT)


def _min_last(a: np.ndarray) -> np.ndarray:
    return a.min(axis=-1) if a.shape[-1] else np.full(a.shape[:-1], ABSENT)


def margin_parts(m: CylModel, x: np.ndarray, obst: np.ndarray, d: np.ndarray | None = None) -> tuple:
    """The terms of the safe and base margins before the clearance is chosen (``combine``): height above the floor,
    the nearest surface from p, c_B - level, the nearest surface from the anchor, anchor height above the floor - z_ext."""
    r = rvec(x, d, m)
    a = x[..., 0:3] + r @ m.DT
    level = np.einsum("...i,ij,...j->...", r, m.S, r)
    return (x[..., 2] - m.z_floor, _min_last(surface_distance(x[..., 0:2], obst)), m.c_B - level,
            _min_last(surface_distance(a[..., 0:2], obst)), a[..., 2] - m.z_ext - m.z_floor)


def combine(m: CylModel, parts: tuple, extra=0.0) -> tuple[np.ndarray, np.ndarray]:
    """(safe margin, base-set margin) from ``margin_parts`` with the clearance r_d + extra (extra: scalar or per row)."""
    h, dp, lv, da, az = parts
    rd = m.r_drone + extra
    safe = np.minimum(h, dp - rd)
    return safe, np.minimum(safe, np.minimum(np.minimum(lv, da - m.rho_xy - rd), az))


def margins(m: CylModel, x: np.ndarray, obst: np.ndarray, extra=0.0,
            d: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(safe margin [m], base-set margin) of states x against the cylinders obst; >= 0 inside. The base-set margin
    includes the safe margin (>= 0 means in B: strike_scene.in_base_set). ``extra`` adds clearance to the drone radius
    (the filter's margin for model error); 0 gives the sets of strike_scene.margins. d: the push (base set about the
    hover equilibrium that cancels it)."""
    return combine(m, margin_parts(m, x, obst, d), extra)


def in_base(m: CylModel, x: np.ndarray, obst: np.ndarray, extra: float = 0.0, d: np.ndarray | None = None) -> np.ndarray:
    return margins(m, x, obst, extra, d)[1] >= 0.0


def local_view(m: CylModel, x: np.ndarray, obst: np.ndarray, obstacle_range: float = np.inf) -> np.ndarray:
    """(..., n_slots, 4) the backup's scene: the nearest cylinders (surface distance), nearest first, those farther than
    obstacle_range marked absent (strike_filter.local_scene)."""
    d = surface_distance(x[..., 0:2], obst)
    k = m.n_slots
    if obst.shape[-2] < k:                                     # pad with absent slots
        pad = np.zeros(obst.shape[:-2] + (k - obst.shape[-2], 4))
        obst = np.concatenate([obst, pad], axis=-2)
        d = np.concatenate([d, np.full(pad.shape[:-1], ABSENT)], axis=-1)
    idx = np.argsort(d, axis=-1, kind="stable")[..., :k]
    view = np.take_along_axis(obst, idx[..., None], axis=-2).copy()
    dk = np.take_along_axis(d, idx, axis=-1)
    view[..., 3] = np.where(dk <= obstacle_range, view[..., 3], 0.0)
    return view


def observation(m: CylModel, x: np.ndarray, view: np.ndarray) -> np.ndarray:
    """(..., obs_dim) the learned backup's input (strike_scene.make_obs for cylinders, no chaser slots)."""
    shp = np.broadcast_shapes(x.shape[:-1], view.shape[:-2])
    o = np.zeros(shp + (m.obs_dim,))
    o[..., 0] = x[..., 2] - m.z_floor
    o[..., 1:4] = x[..., 3:6]
    o[..., 4:8] = x[..., 6:10]
    rel = view[..., 0:2] - x[..., None, 0:2]                    # centre offset (..., K, 2)
    n = np.sqrt(np.sum(rel ** 2, axis=-1) + 1e-12)
    d = n - view[..., 2]
    on = view[..., 3]
    f = o[..., EGO_FEATS:].reshape(shp + (m.n_slots, OBST_FEATS))
    f[..., 0:2] = (d / n * on)[..., None] * rel
    f[..., 2:4] = rel * on[..., None]
    f[..., 6] = on
    f[..., 8] = view[..., 2] * on
    f[..., 9] = on
    return o


class DisturbanceObserver:
    """Acceleration-disturbance estimate d_hat = L (v - xi), xi' = a_cmd z_b - g e3 + d_hat (the slow-track observer),
    driven by the state estimate and the command applied from it; |d_hat| capped at d_max. Per control step:
    ``d = obs.estimate(x)`` with the new estimate, then ``obs.advance(x, u)`` with the command sent."""

    def __init__(self, m: CylModel, gain: float = 2.0, d_max: float = 1.0) -> None:
        self.m, self.L, self.d_max = m, float(gain), float(d_max)
        self.xi = None
        self.d = None

    def reset(self, v: np.ndarray) -> None:
        self.xi = np.array(v, dtype=float, copy=True)
        self.d = np.zeros_like(self.xi)

    def estimate(self, x: np.ndarray) -> np.ndarray:
        if self.xi is None:
            self.reset(x[..., 3:6])
        d = self.L * (x[..., 3:6] - self.xi)
        n = np.sqrt(np.sum(d ** 2, axis=-1, keepdims=True))
        self.d = d * np.minimum(1.0, self.d_max / np.maximum(n, 1e-9))
        return self.d

    def advance(self, x: np.ndarray, u: np.ndarray, dt: float | None = None) -> None:
        """xi over dt (default the model step; a held command: the control period) with the command u from x."""
        m = self.m
        acc = body_z(x[..., 6:10]) * np.clip(u[..., 0:1], m.low[0], m.high[0]) - np.array([0.0, 0.0, m.gravity])
        self.xi = self.xi + (m.dt if dt is None else dt) * (acc + self.d)


# ------------------------------------------------------------------------- learned backup
class Actor:
    """Deterministic action of a Phase-I actor: base action (hover LQR) + tanh-squashed mean residual."""

    def __init__(self, m: CylModel, layers: list[tuple[np.ndarray, np.ndarray]], activation: str, d_low, d_high,
                 dtype=np.float64) -> None:
        if activation not in ("relu", "elu"):
            raise ValueError(f"actor activation must be relu or elu, got {activation!r}")
        self.m = m
        self.dtype = dtype
        w_last, b_last = layers[-1]
        if w_last.shape[1] == 8:                                   # mean and log-std heads: keep the mean
            layers = layers[:-1] + [(w_last[:, :4], b_last[:4])]
        self.layers = [(np.ascontiguousarray(w, dtype), np.asarray(b, dtype)) for w, b in layers]
        self.activation = activation
        self.mid = 0.5 * (np.asarray(d_high) + np.asarray(d_low))
        self.half = 0.5 * (np.asarray(d_high) - np.asarray(d_low))
        self.width = int(self.layers[0][0].shape[1])

    def residual(self, o: np.ndarray) -> np.ndarray:
        h = o.astype(self.dtype, copy=False)
        n = len(self.layers)
        for i, (w, b) in enumerate(self.layers):
            h = h @ w
            h += b
            if i != n - 1:
                if self.activation == "relu":
                    np.maximum(h, 0.0, out=h)
                else:
                    h = np.where(h > 0.0, h, np.expm1(np.minimum(h, 0.0)))
        return self.mid + self.half * np.tanh(h)

    def __call__(self, x: np.ndarray, view: np.ndarray, r: np.ndarray | None = None, d: np.ndarray | None = None) -> np.ndarray:
        """Backup action; d: the push, cancelled by the base action (the hover LQR about its equilibrium)."""
        base = lqr_action(self.m, x, r, d) if self.m.backup_action == "residual_lqr" else equilibrium(self.m, d)[1]
        return np.clip(base + np.nan_to_num(self.residual(observation(self.m, x, view))), self.m.low, self.m.high)

    @property
    def flops(self) -> int:
        return int(sum(2 * w.shape[0] * w.shape[1] for w, _ in self.layers))


def random_actor(m: CylModel, width: int, seed: int = 0, activation: str = "relu") -> Actor:
    """An actor of the given width with random weights (timing only)."""
    rng = np.random.default_rng(seed)
    sizes = [m.obs_dim, width, width, 4]
    layers = [(rng.normal(size=(a, b)) / np.sqrt(a) * (1e-2 if i == 2 else 1.0), np.zeros(b))
              for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:]))]
    return Actor(m, layers, activation, -m.scale, m.scale)


# ------------------------------------------------------------------------- exported backups
def save_backup(path: str | Path, m: CylModel, actor: Actor | None, meta: dict[str, Any]) -> None:
    """``<path>/backup.json`` (constants, actor layout, provenance) and ``<path>/backup_weights.npz`` (no pickle)."""
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    payload = {"format": FORMAT, "model": m.constants(), "meta": meta, "actor": None}
    if actor is not None:
        payload["actor"] = {"activation": actor.activation, "d_low": (actor.mid - actor.half).tolist(),
                            "d_high": (actor.mid + actor.half).tolist(), "width": actor.width,
                            "layers": [[int(w.shape[0]), int(w.shape[1])] for w, _ in actor.layers]}
        arrays = {}
        for i, (w, b) in enumerate(actor.layers):
            arrays[f"w{i}"], arrays[f"b{i}"] = w.astype(np.float64), b.astype(np.float64)
        np.savez(out / "backup_weights.npz", **arrays)
    (out / "backup.json").write_text(json.dumps(payload, indent=2))


def load_backup(path: str | Path, dtype=np.float64) -> tuple[CylModel, Actor | None, dict[str, Any]]:
    p = Path(path).expanduser()
    payload = json.loads((p / "backup.json").read_text())
    if payload.get("format") != FORMAT:
        raise ValueError(f"{p}: not a {FORMAT} export")
    m = CylModel.from_constants(payload["model"])
    actor = None
    if payload.get("actor"):
        a = payload["actor"]
        z = np.load(p / "backup_weights.npz")
        layers = [(z[f"w{i}"], z[f"b{i}"]) for i in range(len(a["layers"]))]
        actor = Actor(m, layers, a["activation"], a["d_low"], a["d_high"], dtype=dtype)
    return m, actor, payload.get("meta", {})


__all__ = ["ABSENT", "Actor", "CylModel", "DisturbanceObserver", "EGO_FEATS", "OBST_FEATS", "anchor", "body_z", "chart",
           "combine", "margin_parts",
           "equilibrium", "held_action", "held_error", "held_level", "in_base", "load_backup", "local_view", "lqr_action", "margins", "observation", "random_actor",
           "rvec", "save_backup", "step", "surface_distance"]
