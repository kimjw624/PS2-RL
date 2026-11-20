"""Closed-loop episodes for the obstacle-only online filter, batched over episodes (numpy).

Scenes: tall cylinders between a start and a goal; the controller knows a cylinder once the vehicle has come within
``sensing_range`` of its surface (360 deg range sensor; the known set only grows). The filter and the nominal see only
the known cylinders; contacts are judged against the true ones.

Plant ("realistic", per-episode randomized; ``PlantConfig.ideal`` turns every effect off and leaves the filter's own
model): command delay of 0..delay_max control steps, first-order body-rate and thrust lags, thrust and rate gain errors,
and state-estimate noise. Integrated with ``substeps`` per control step. The filter certifies with the ideal model plus
``FilterConfig.margin``. A constant horizontal wind plus a sinusoidal gust can be added (``wind``, ``gust``; off by
default: wind is not considered for now), and with it a disturbance observer (``run_episodes(observer_gain=...)``).

Scenario sets:
  random   ``n_obst`` cylinders of radius r_min..r_max placed between x_obst (a band across the straight line from the
           start to the goal), at least ``gap_min`` apart (surface to surface, so the vehicle can pass), start at rest
           or cruising
  final    the deployment demo, as short as it can be: straight forward from rest at height z to a goal on the line
           beyond the last cylinder, with small virtual cylinders (r 0.1 m). The defaults are the chosen demo layouts
           (docker_batch/configs/cyl_online_candidates.json):
             "one" (O3)          one cylinder ``obst_x`` = 1.3 m ahead at y = ``lateral`` = -0.1 (its top touches the
                                 line, so the vehicle goes round it on the left; its surface 1.2 m away: the start
                                 hover is in the base set with 0.2 m to spare), the goal ``goal_after_one`` = 2.0 m
                                 beyond it (x 3.3 m)
             "two", "slalom" (S1, default)  the first at (first_x, first_y) = (1.1, -0.1), on the line (the start hover
                                 exactly at the base set's limit), the second ``second_dx`` = 2.0 m further on at
                                 y = ``second_y`` = 1.6, across the path the vehicle takes back to the goal after the
                                 first: it has to come back towards the line early and pass under it (a weave); the
                                 goal ``goal_after_two`` = 1.2 m beyond the second (x 4.3 m)
             "two", "gap"        side by side at x = obst_x, the lower one at y = low_y (None: the gap centred on the
                                 line), a surface gap ``gap`` between them
           ``goal_after`` (if set) overrides both goal distances. The earlier course (r 0.3 m, the first cylinder 1.5 m
           ahead, the slalom's second 3.5 m on, the goal 5 m beyond) is ``R03_COURSE``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import time

import numpy as np

from ps2rl.online.filter import MODES, OnlineFilter
from ps2rl.online.model import CylModel, DisturbanceObserver, surface_distance
from ps2rl.online.nominal import Nominal


@dataclass
class PlantConfig:
    ideal: bool = False
    substeps: int = 2
    delay_max: int = 2            # control steps (0..delay_max, per episode)
    tau_rate: tuple = (0.03, 0.08)
    tau_thrust: tuple = (0.02, 0.06)
    thrust_gain: tuple = (0.92, 1.08)
    rate_gain: tuple = (0.95, 1.05)
    wind: float = 0.0             # constant horizontal push, magnitude up to this [m/s^2] (0: not considered for now)
    gust: float = 0.0             # sinusoidal push amplitude [m/s^2]
    gust_freq: tuple = (0.05, 0.5)
    noise_p: float = 0.02         # estimate noise (standard deviations) [m]
    noise_v: float = 0.03         # [m/s]
    noise_att: float = 0.01       # [rad]


@dataclass
class SceneConfig:
    kind: str = "random"          # "random" | "one" | "two"
    goal_x: float = 14.0          # random scenes (one / two: goal_after)
    goal_after: float | None = None   # one / two: the goal this far beyond the last cylinder [m], both cases (0: goal_x)
    goal_after_one: float = 2.0   # ... when goal_after is None: one cylinder (O3)
    goal_after_two: float = 1.2   # ... two cylinders (S1)
    z: float = 1.5
    start_speed: tuple = (0.0, 1.0)
    sensing_range: float = 6.0
    episode_sec: float = 40.0
    goal_radius: float = 0.5
    # random scenes
    n_obst: tuple = (2, 4)
    r_min: float = 0.2
    r_max: float = 0.5
    x_obst: tuple = (4.0, 11.0)
    y_half: float = 2.5
    gap_min: float = 2.0
    # final test
    obst_x: float = 1.3           # "one" / "gap": x of the cylinder(s) [m]
    radius: float = 0.1
    two_layout: str = "slalom"    # "two": "slalom" (the second blocks the way back) or "gap" (side by side)
    first_x: float = 1.1          # "slalom": x of the first cylinder [m]
    first_y: float = -0.1         # "slalom": y of the first cylinder's centre (-r: its top on the line, avoided on the left)
    second_dx: float = 2.0        # "slalom": the second cylinder this far beyond the first [m]
    second_y: float = 1.6         # "slalom": ... at this y (across the way back to the goal after the first) [m]
    low_y: float | None = -0.15   # "gap": y of the lower cylinder's centre; None: gap centred
    gap: float = 3.0              # "gap": surface-to-surface gap between the two cylinders
    lateral: float = -0.1         # "one": y of the cylinder's centre (-r: its top on the line, avoided on the left)

    def as_dict(self) -> dict:
        return asdict(self)

    def goal_distance(self) -> float:
        """How far beyond the last cylinder the goal is ("one" / "two")."""
        if self.goal_after is not None:
            return float(self.goal_after)
        return self.goal_after_one if self.kind == "one" else self.goal_after_two


# the earlier deployment course (r 0.3 m; the sweep's "blind" / "final" sets and the start / spacing sweeps ran on it)
R03_COURSE = dict(radius=0.3, obst_x=1.5, lateral=0.0, first_x=1.5, first_y=-0.3, second_dx=3.5, second_y=1.5,
                  goal_after=5.0)


@dataclass
class Episodes:
    x0: np.ndarray         # (E, 10)
    goal: np.ndarray       # (E, 3)
    obst: np.ndarray       # (E, M, 4)


def make_scenes(cfg: SceneConfig, n: int, seed: int = 0) -> Episodes:
    rng = np.random.default_rng(seed)
    x0 = np.zeros((n, 10))
    x0[:, 2] = cfg.z
    x0[:, 3] = rng.uniform(*cfg.start_speed, n)
    x0[:, 6] = 1.0
    last = {"one": cfg.obst_x, "two": cfg.first_x + cfg.second_dx if cfg.two_layout == "slalom" else cfg.obst_x}.get(cfg.kind)
    ga = cfg.goal_distance()
    gx = last + ga if (last is not None and ga > 0.0) else cfg.goal_x
    goal = np.tile([gx, 0.0, cfg.z], (n, 1))
    if cfg.kind == "one":
        obst = np.tile([[cfg.obst_x, cfg.lateral, cfg.radius, 1.0]], (n, 1, 1))
    elif cfg.kind == "two" and cfg.two_layout == "slalom":
        y1 = float(cfg.first_y)
        obst = np.tile([[cfg.first_x, y1, cfg.radius, 1.0], [cfg.first_x + cfg.second_dx, cfg.second_y, cfg.radius, 1.0]],
                       (n, 1, 1))
    elif cfg.kind == "two" and cfg.two_layout == "gap":
        lo = -(0.5 * cfg.gap + cfg.radius) if cfg.low_y is None else float(cfg.low_y)
        hi = lo + 2.0 * cfg.radius + cfg.gap
        obst = np.tile([[cfg.obst_x, hi, cfg.radius, 1.0], [cfg.obst_x, lo, cfg.radius, 1.0]], (n, 1, 1))
    elif cfg.kind == "random":
        M = int(cfg.n_obst[1])
        obst = np.zeros((n, M, 4))
        for e in range(n):
            k = int(rng.integers(cfg.n_obst[0], cfg.n_obst[1] + 1))
            placed = []
            for _ in range(500):
                if len(placed) == k:
                    break
                r = rng.uniform(cfg.r_min, cfg.r_max)
                c = np.array([rng.uniform(*cfg.x_obst), rng.uniform(-cfg.y_half, cfg.y_half)])
                if all(np.linalg.norm(c - q[0:2]) - r - q[2] >= cfg.gap_min for q in placed):
                    placed.append(np.array([c[0], c[1], r, 1.0]))
            obst[e, :len(placed)] = np.asarray(placed)
    else:
        raise ValueError(f"scene kind must be random, one or two, got {cfg.kind!r}")
    return Episodes(x0=x0, goal=goal, obst=obst)


def _quat_step(q: np.ndarray, w: np.ndarray, h: float) -> np.ndarray:
    qw, qx, qy, qz = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    wx, wy, wz = w[:, 0], w[:, 1], w[:, 2]
    qn = q + 0.5 * h * np.stack([-qx * wx - qy * wy - qz * wz, qw * wx - qz * wy + qy * wz,
                                 qz * wx + qw * wy - qx * wz, -qy * wx + qx * wy + qw * wz], 1)
    return qn / np.linalg.norm(qn, axis=1, keepdims=True)


class Plant:
    """The realistic vehicle (see the module); state x (E, 10) plus actual rates and thrust."""

    def __init__(self, m: CylModel, cfg: PlantConfig, x0: np.ndarray, seed: int = 0) -> None:
        self.m, self.cfg = m, cfg
        E = x0.shape[0]
        rng = np.random.default_rng(seed + 7919)
        self.rng = rng
        self.x = x0.copy()
        self.w = np.zeros((E, 3))
        self.a = np.full(E, m.gravity)
        ideal = cfg.ideal
        u = lambda lo_hi: rng.uniform(*lo_hi, E)
        self.delay = np.zeros(E, int) if ideal else rng.integers(0, cfg.delay_max + 1, E)
        self.tau_w = None if ideal else u(cfg.tau_rate)
        self.tau_a = None if ideal else u(cfg.tau_thrust)
        self.g_a = np.ones(E) if ideal else u(cfg.thrust_gain)
        self.g_w = np.ones(E) if ideal else u(cfg.rate_gain)
        th = rng.uniform(-np.pi, np.pi, E)
        mag = 0.0 if ideal else rng.uniform(0.0, cfg.wind, E)
        self.wind = np.stack([mag * np.cos(th), mag * np.sin(th), np.zeros(E)], 1)
        th2 = rng.uniform(-np.pi, np.pi, E)
        self.gust_dir = np.stack([np.cos(th2), np.sin(th2), np.zeros(E)], 1)
        self.gust_amp = 0.0 if ideal else cfg.gust
        self.gust_f = u(cfg.gust_freq)
        self.gust_ph = rng.uniform(0, 2 * np.pi, E)
        self.buf = np.tile(np.array([m.gravity, 0, 0, 0.0]), (E, int(cfg.delay_max) + 1, 1))
        self.t = 0.0

    def estimate(self) -> np.ndarray:
        if self.cfg.ideal:
            return self.x.copy()
        E, c = self.x.shape[0], self.cfg
        x = self.x.copy()
        x[:, 0:3] += c.noise_p * self.rng.normal(size=(E, 3))
        x[:, 3:6] += c.noise_v * self.rng.normal(size=(E, 3))
        dq = 0.5 * c.noise_att * self.rng.normal(size=(E, 3))
        q = x[:, 6:10]
        x[:, 6:10] = _quat_step(q, 2.0 * dq, 1.0)            # small random rotation
        return x

    def apply(self, u: np.ndarray, active: np.ndarray) -> None:
        m, c = self.m, self.cfg
        if c.ideal:
            from ps2rl.online.model import step
            self.x = np.where(active[:, None], step(m, self.x, u), self.x)
            self.t += m.dt
            return
        self.buf = np.roll(self.buf, 1, axis=1)
        self.buf[:, 0] = u
        ud = self.buf[np.arange(u.shape[0]), self.delay]
        a_cmd = np.clip(ud[:, 0], m.low[0], m.high[0]) * self.g_a
        w_cmd = np.clip(ud[:, 1:4], m.low[1:], m.high[1:]) * self.g_w[:, None]
        h = m.dt / c.substeps
        x = self.x.copy()
        for _ in range(c.substeps):
            self.w += (h / self.tau_w)[:, None] * (w_cmd - self.w)
            self.a += (h / self.tau_a) * (a_cmd - self.a)
            q = x[:, 6:10]
            qw, qx, qy, qz = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
            zb = np.stack([2 * (qx * qz + qw * qy), 2 * (qy * qz - qw * qx), 1 - 2 * (qx * qx + qy * qy)], 1)
            push = self.wind + self.gust_amp * np.sin(2 * np.pi * self.gust_f * self.t + self.gust_ph)[:, None] * self.gust_dir
            acc = zb * self.a[:, None] - np.array([0, 0, m.gravity]) + push
            x[:, 0:3] += h * x[:, 3:6] + 0.5 * h * h * acc
            x[:, 3:6] += h * acc
            x[:, 6:10] = _quat_step(x[:, 6:10], self.w, h)
            self.t += h
        self.x = np.where(active[:, None], x, self.x)


@dataclass
class EpisodeLog:
    t: list = field(default_factory=list)
    x: list = field(default_factory=list)
    u: list = field(default_factory=list)
    u_nom: list = field(default_factory=list)
    lam: list = field(default_factory=list)
    mode: list = field(default_factory=list)
    known: list = field(default_factory=list)


def run_episodes(m: CylModel, eps: Episodes, scene: SceneConfig, plant_cfg: PlantConfig, nominal: Nominal,
                 filt: OnlineFilter | None, *, seed: int = 0, log: bool = False, observer_gain: float = 2.0,
                 hold: int | None = None) -> dict:
    """Run every episode to the goal, a contact or the time limit. filt None: the nominal alone. observer_gain > 0: a
    disturbance observer feeds its push estimate to the filter and the nominal (0: no estimate). hold: model steps per
    command (default: the filter's ``hold``, else 1) - the controller runs at command times only (estimate, sensing,
    decision, command, observer update over the control period) and the plant holds the command in between, as on the
    vehicle with a control period of hold x dt."""
    E, M = eps.obst.shape[0], eps.obst.shape[1]
    plant = Plant(m, plant_cfg, eps.x0, seed)
    n_steps = int(round(scene.episode_sec / m.dt))
    present = eps.obst[:, :, 3] > 0.5
    known = np.zeros((E, M), bool)
    active = np.ones(E, bool)
    contact = np.zeros(E, bool)
    floor = np.zeros(E, bool)
    reached = np.zeros(E, bool)
    t_goal = np.full(E, np.nan)
    min_clear = np.full(E, np.inf)
    lam_sum, lam_n, intervene = np.zeros(E), np.zeros(E), np.zeros(E)
    mode_count = np.zeros((E, len(MODES)))
    dec_time, dec_steps = [], []
    steps_v = []                                           # per vehicle and decision: rollout steps an onboard filter runs
    path = np.zeros(E)
    P = filt.cfg.period if filt is not None else 1
    H = int(hold if hold is not None else (filt.cfg.hold if filt is not None else 1))
    if filt is not None and int(filt.cfg.hold) != H:
        raise ValueError(f"hold {H} differs from the filter's {filt.cfg.hold}")
    L = EpisodeLog() if log else None
    plan = None
    obs_d = DisturbanceObserver(m, gain=observer_gain) if observer_gain > 0.0 else None
    for t in range(n_steps):
        xt = plant.x
        if t % H == 0:                                     # a command time
            c = t // H
            d_true = surface_distance(xt[:, 0:2], eps.obst)
            known |= present & (d_true <= scene.sensing_range)
            ok_obst = eps.obst.copy()
            ok_obst[:, :, 3] = (present & known).astype(float)
            xe = plant.estimate()
            d_hat = obs_d.estimate(xe) if obs_d is not None else None
            if filt is not None:
                if t == 0:
                    filt.reset(xe, ok_obst, eps.goal)
                if c % P == 0:
                    u_n0 = nominal(xe, eps.goal, ok_obst, d_hat)
                    plan = filt.decide(xe, ok_obst, eps.goal, u_n0, d_hat)
                    dec_time.append(plan.stats["time"])
                    dec_steps.append(plan.stats["steps"])
                    steps_v.append(plan.stats["steps_vehicle"][active])
                    mode_count[np.arange(E), plan.mode] += active
                g_p, o_p, d_p = filt.plan_nominal_inputs()
                u_nom = nominal(xe, g_p, o_p, d_p)
                u = filt.command(xe, u_nom)
                lam_now = np.where(plan.held, 0.0, plan.lam)
            else:
                u_nom = nominal(xe, eps.goal, ok_obst, d_hat)
                u = u_nom
                lam_now = np.ones(E)
            if obs_d is not None:
                obs_d.advance(xe, u, H * m.dt)
        lam_sum += lam_now * active
        lam_n += active
        intervene += (np.abs(u - u_nom).max(axis=1) > 1e-6) * active
        if log:
            L.t.append(t * m.dt); L.x.append(xt.copy()); L.u.append(u.copy()); L.u_nom.append(u_nom.copy())
            L.lam.append(lam_now.copy()); L.mode.append(None if plan is None else plan.mode.copy()); L.known.append(known.copy())
        p_before = plant.x[:, 0:3].copy()
        plant.apply(u, active)
        xt = plant.x
        path += np.linalg.norm(xt[:, 0:3] - p_before, axis=1) * active
        d_true = surface_distance(xt[:, 0:2], eps.obst)
        clear = d_true.min(axis=1, initial=1e3) - m.r_drone
        min_clear = np.where(active, np.minimum(min_clear, clear), min_clear)
        hit = active & (clear < 0.0)
        fl = active & (xt[:, 2] < 0.05)
        at_goal = active & (np.linalg.norm(xt[:, 0:2] - eps.goal[:, 0:2], axis=1) < scene.goal_radius)
        contact |= hit
        floor |= fl
        t_goal = np.where(at_goal & ~hit, (t + 1) * m.dt, t_goal)
        reached |= at_goal & ~hit
        active &= ~(hit | fl | at_goal)
        if not active.any():
            break
    res = {
        "n": E, "success": float(np.mean(reached & ~contact & ~floor)), "contact": float(np.mean(contact)),
        "floor": float(np.mean(floor)), "timeout": float(np.mean(~reached & ~contact & ~floor)),
        "t_goal_mean": float(np.nanmean(t_goal)) if reached.any() else float("nan"),
        "min_clear": float(np.min(min_clear)), "min_clear_p05": float(np.percentile(min_clear, 5)),
        "lam_mean": float(np.sum(lam_sum) / max(np.sum(lam_n), 1)),
        "intervene_frac": float(np.sum(intervene) / max(np.sum(lam_n), 1)),
        "path_ratio": float(np.mean(path[reached] / np.linalg.norm(eps.goal[reached, 0:2] - eps.x0[reached, 0:2], axis=1)))
        if reached.any() else float("nan"),
        "per_episode": {"success": (reached & ~contact & ~floor).tolist(), "contact": contact.tolist(),
                        "t_goal": np.nan_to_num(t_goal, nan=-1.0).tolist(), "min_clear": min_clear.tolist()},
    }
    if filt is not None:
        tot = mode_count.sum()
        res.update({f"mode_{k}": float(mode_count[:, i].sum() / max(tot, 1)) for i, k in enumerate(MODES)})
        sv = np.concatenate(steps_v) if steps_v else np.zeros(1)
        res["decisions"] = int(sv.size)
        res["steps_mean"] = float(sv.mean())
        res["steps_p99"] = float(np.percentile(sv, 99))
        res["steps_max"] = int(sv.max())
        res["batch_time_s"] = float(np.sum(dec_time))
    if log:
        res["log"] = {k: np.asarray(v) if k != "mode" else v for k, v in asdict(L).items()}
    return res


__all__ = ["EpisodeLog", "Episodes", "Plant", "PlantConfig", "R03_COURSE", "SceneConfig", "make_scenes", "run_episodes"]
