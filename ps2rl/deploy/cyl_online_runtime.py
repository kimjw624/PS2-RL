"""Online controller for the obstacle-only demo: the go-to-goal nominal through the online backup filter.

One state in, one command out, for the 50 Hz control loop of the flight node, with exactly the objects the
simulations used (``ps2rl.online``: ``Nominal``, ``OnlineFilter`` with the exported Phase-I backup, the disturbance
observer) and the same order of calls per control step as ``ps2rl.online.sim.run_episodes``:

    d = observer estimate (x)
    known cylinders <- those within sensing_range of x (the known set only grows)
    every ``period`` steps: plan = filter.decide(x, known, goal, nominal(x, goal, known, d), d)
    u = filter.command(x, nominal(x, plan's goal, plan's cylinders, plan's d))
    observer advance (x, u)

The cylinders are virtual: a course from ``docker_batch/configs/cyl_online_candidates.json`` (O1..O4, S1..S4; the demo
layouts are O3 and S1) or given explicitly (``obstacles`` "x,y,r;x,y,r", ``goal`` "x,y").

Task frame (the flight logic converts the vehicle's local frame to it): origin at the take-off point on the ground,
x east (the heading the flight logic holds), z up. The hand-over point is ``flight_height`` above the origin, the
course runs along +x at that height, the filter's floor is the model's z_floor above the origin. No JAX, no ROS.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import time

import numpy as np

from ps2rl.deploy.landing_runtime import ControlOutput
from ps2rl.online.filter import MODES, FilterConfig, OnlineFilter
from ps2rl.online.model import ABSENT, body_z, load_backup, margins, surface_distance
from ps2rl.online.nominal import Nominal, NominalConfig

LAMBDA_SETS = {2: (1.0, 0.0), 3: (1.0, 0.5, 0.0), 5: (1.0, 0.75, 0.5, 0.25, 0.0)}
NOMINALS = ("planner", "straight")
_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class CylOnlineRuntimeConfig:
    backup: str = "checkpoints/cyl_online/td3_w128_seed0"   # exported Phase-I backup (scripts/export_cyl_backup.py)
    course: str = "S1"                    # a layout of candidates_file (O1..O4, S1..S4) or "custom"
    candidates_file: str = "docker_batch/configs/cyl_online_candidates.json"
    obstacles: str = ""                   # "x,y,r;x,y,r" in the task frame [m]: replaces the course's cylinders
    goal: str = ""                        # "x,y" in the task frame [m]: replaces the course's goal
    nominal: str = "planner"              # planner: go-to-goal with the heading search; straight: no search (the filter
                                          # alone takes it round the cylinders)
    search_margin: float = 1.0            # planner: ray clearance from the cylinders' surfaces [m]
    cruise: float = 1.0                   # [m/s]
    blends: int = 2                       # 2: lam in (1, 0); 3: (1, 0.5, 0); 5
    horizon: float = 1.0                  # backup rollout [s]
    period: int = 1                       # control steps per decision
    backend: str = "auto"                 # numpy | numba | auto (numba if installed)
    sensing_range: float = 6.0            # a cylinder is known once its surface is this close [m]
    flight_height: float = 1.5            # hand-over point and course height above the take-off point [m]
    observer_gain: float = 2.0            # disturbance observer (0: off); it absorbs the thrust-map error
    observer_max: float = 1.0             # cap on the estimate [m/s^2]
    goal_radius: float = 0.5              # "reached": horizontal distance to the goal (as in the simulations) [m]
    control_period: float = 0.02          # one command per control period, held by the autopilot in between [s]: a
                                          # multiple of the model step (0.02 s); 0.04 = 25 Hz. The filter's rollouts
                                          # hold every command for as long (FilterConfig.hold)
    process: bool = False                 # run this controller in its own process (make_runtime): no GIL shared with
                                          # the ROS node's threads
    worker_cpu: int = -1                  # process: pin it to this CPU core (-1: no pinning)


def _resolve(path: str, base: Path | None) -> Path:
    p = Path(path).expanduser()
    if p.is_absolute():
        return p
    for root in (base, _ROOT):
        if root is not None and (root / p).exists():
            return root / p
    return (base or _ROOT) / p


def course_layout(cfg: CylOnlineRuntimeConfig, base: Path | None) -> tuple[np.ndarray, np.ndarray, str]:
    """(cylinders (M, 4) cx, cy, r, present; goal (3,); description) in the task frame."""
    z = float(cfg.flight_height)
    obst, goal, label = None, None, cfg.course
    if cfg.course and cfg.course.lower() != "custom":
        path = _resolve(cfg.candidates_file, base)
        cands = json.loads(path.read_text())["candidates"]
        if cfg.course not in cands:
            raise ValueError(f"course '{cfg.course}' is not in {path} (have {', '.join(cands)})")
        c = cands[cfg.course]
        r = float(c["radius"])
        if c["kind"] == "one":
            cyl = [(float(c["obst_x"]), float(c.get("lateral", 0.0)), r)]
        else:
            x1 = float(c["first_x"])
            cyl = [(x1, float(c["first_y"]), r), (x1 + float(c["second_dx"]), float(c["second_y"]), r)]
        obst = np.array([[*o, 1.0] for o in cyl])
        goal = np.array([max(o[0] for o in cyl) + float(c["goal_after"]), 0.0, z])
        label = f"{cfg.course}: {c.get('about', '')}"
    if cfg.obstacles:
        cyl = [tuple(float(v) for v in item.split(",")) for item in cfg.obstacles.split(";") if item.strip()]
        if any(len(o) != 3 for o in cyl):
            raise ValueError(f"obstacles must be 'x,y,r;x,y,r', got '{cfg.obstacles}'")
        obst = np.array([[*o, 1.0] for o in cyl])
        label = f"custom cylinders {cfg.obstacles}"
    if cfg.goal:
        g = [float(v) for v in cfg.goal.split(",")]
        if len(g) != 2:
            raise ValueError(f"goal must be 'x,y', got '{cfg.goal}'")
        goal = np.array([g[0], g[1], z])
    if obst is None or goal is None:
        raise ValueError("no course: give course (a candidate name) or both obstacles and goal")
    return obst, goal, label


class CylOnlineRuntime:
    """The controller in this process. ``make_runtime`` returns it, or a proxy to it in a worker process."""

    def __init__(self, cfg: CylOnlineRuntimeConfig, *, ps2rl_path: str | None = None, logger=None) -> None:
        if cfg.nominal not in NOMINALS:
            raise ValueError(f"nominal must be one of {NOMINALS}, got {cfg.nominal!r}")
        if int(cfg.blends) not in LAMBDA_SETS:
            raise ValueError(f"blends must be one of {sorted(LAMBDA_SETS)}, got {cfg.blends}")
        self.cfg = cfg
        self._log = logger
        base = Path(ps2rl_path).expanduser() if ps2rl_path else None
        backup = _resolve(cfg.backup, base)
        if not (backup / "backup.json").exists():
            raise FileNotFoundError(f"backup '{backup}' has no backup.json (export one with scripts/export_cyl_backup.py)")
        self.m, self.actor, self.meta = load_backup(backup)
        m = self.m
        self.obst, self.goal, self.course_label = course_layout(cfg, base)
        self.nominal = Nominal(m, NominalConfig(cruise=float(cfg.cruise), margin=float(cfg.search_margin),
                                                search=cfg.nominal == "planner"))
        self.hold = int(round(float(cfg.control_period) / m.dt))
        if self.hold < 1 or abs(self.hold * m.dt - float(cfg.control_period)) > 1e-9:
            raise ValueError(f"control_period {cfg.control_period} must be a multiple of the model step {m.dt}")
        self.fcfg = FilterConfig(lambdas=LAMBDA_SETS[int(cfg.blends)], backups=("learned",), horizon=float(cfg.horizon),
                                 period=int(cfg.period), hold=self.hold, backend=cfg.backend)
        self.filter = OnlineFilter(m, self.actor, self.fcfg, self.nominal)
        self.backend = "numba" if self.filter.use_numba else "numpy"
        # the flight logic's interface (shared with the landing / slow-track runtimes)
        self.dt = float(self.hold * m.dt)                       # the control period
        self.r_drone, self.rho_xy, self.z_floor = float(m.r_drone), float(m.rho_xy), float(m.z_floor)
        self.gravity = float(m.gravity)
        self.action_low = np.asarray(m.low, np.float64)
        self.action_high = np.asarray(m.high, np.float64)
        self.pad = np.zeros(3)                                   # the take-off point (task-frame origin)
        self.start_state = np.array([0.0, 0.0, float(cfg.flight_height), 0, 0, 0, 1, 0, 0, 0], np.float64)
        self.hover_position = self.start_state[0:3].copy()
        self.target = self.goal.copy()
        self.filter_overrides: dict = {}
        self.label = (f"{self.course_label}; nominal {cfg.nominal}"
                      + (f" (search margin {cfg.search_margin:g} m)" if cfg.nominal == "planner" else "")
                      + f"; backup {backup.name} (width {self.actor.width if self.actor else '-'}), {cfg.blends} blends, "
                        f"horizon {cfg.horizon:g} s, period {cfg.period}, control period {1e3 * self.dt:.0f} ms, {self.backend}")
        self.reset()
        t0 = time.perf_counter()
        self._warm_up()
        self._info(f"cyl-online runtime: {self.label}")
        self._info(f"cylinders (x, y, r) {np.round(self.obst[:, 0:3], 2).tolist()}, goal {np.round(self.goal[0:2], 2).tolist()} "
                   f"at {cfg.flight_height:g} m; command box {self.action_low.tolist()} .. {self.action_high.tolist()}; "
                   f"warm-up {1e3 * (time.perf_counter() - t0):.0f} ms")
        start_clear = float(surface_distance(self.start_state[0:2][None], self.obst[None])[0].min()) - m.r_drone
        need = m.rho_xy + m.r_drone + self.fcfg.margin + self.fcfg.fresh_margin
        dmin = float(surface_distance(self.start_state[0:2][None], self.obst[None])[0].min())
        self._info(f"start: {dmin:.2f} m from the nearest surface (a fresh plan from a hover needs {need:.2f} m; "
                   f"clearance minus the drone radius {start_clear:.2f} m)")
        if self.backend == "numpy":
            self._info("numpy backend (numba is not installed for this python): about 4-15 ms per control step with the "
                       "planner, up to ~30 ms while the filter slides a straight-line run round a cylinder (50 Hz leaves "
                       "20 ms); numba takes ~1 ms")

    def describe(self) -> dict:
        """What the flight records keep about the controller (also sent by the worker of a process runtime)."""
        return {"label": self.label, "backend": self.backend, "course_label": self.course_label,
                "cylinders_xyr": self.obst[:, 0:3].tolist(), "goal": self.goal.tolist(), "r_drone": self.r_drone,
                "rho_xy": self.rho_xy, "z_floor": self.z_floor, "dt": self.dt, "hold": self.hold,
                "filter": self.fcfg.as_dict(), "nominal_config": self.nominal.cfg.as_dict(), "backup_meta": self.meta}

    def _info(self, msg: str) -> None:
        (self._log.info if self._log is not None else print)(msg)

    # ------------------------------------------------------------------ state
    def reset(self) -> None:
        self.known = np.zeros(len(self.obst), bool)
        self.k = 0
        self.started = False
        self.plan = None
        self.xi: np.ndarray | None = None
        self.d_hat = np.zeros(3)
        self.goal_entered_t: float | None = None

    def _warm_up(self) -> None:
        """One decision on the course from the start (compiles the numba kernel before the flight needs it)."""
        x = self.start_state[None].copy()
        f = OnlineFilter(self.m, self.actor, self.fcfg, self.nominal)
        obst = self.obst[None].copy()
        goal = self.goal[None]
        f.reset(x, obst, goal)
        f.decide(x, obst, goal, self.nominal(x, goal, obst, None), None)

    # ------------------------------------------------------------------ disturbance observer
    def reset_observer(self, x: np.ndarray) -> None:
        self.xi = np.asarray(x[3:6], np.float64).copy()
        self.d_hat = np.zeros(3)

    def _estimate(self, x: np.ndarray) -> np.ndarray:
        if self.cfg.observer_gain <= 0.0:
            return np.zeros(3)
        if self.xi is None:
            self.reset_observer(x)
        d = self.cfg.observer_gain * (np.asarray(x[3:6]) - self.xi)
        n = float(np.linalg.norm(d))
        self.d_hat = d * min(1.0, self.cfg.observer_max / max(n, 1e-9))
        return self.d_hat

    def _advance(self, x: np.ndarray, a_thrust: float, dt: float) -> None:
        """xi over dt with the thrust a held (after a command: the control period)."""
        if self.cfg.observer_gain <= 0.0 or self.xi is None:
            return
        a = float(np.clip(a_thrust, self.action_low[0], self.action_high[0]))
        acc = body_z(np.asarray(x[6:10], np.float64)) * a - np.array([0.0, 0.0, self.gravity])
        self.xi = self.xi + dt * (acc + self.d_hat)

    def observe(self, x: np.ndarray, a_thrust: float, dt: float) -> None:
        """Before the hand-over: run the observer on the thrust the autopilot reports (warm at the hand-over)."""
        if self.xi is None:
            self.reset_observer(x)
            return
        if not (np.isfinite(a_thrust) and 0.0 < dt < 0.2):
            return
        self._estimate(x)
        self._advance(x, a_thrust, dt)

    # ------------------------------------------------------------------ sets (the flight logic's start checks)
    def clearance(self, x: np.ndarray) -> float:
        """Horizontal distance to the nearest (true) cylinder surface minus the drone radius [m] (< 0: contact)."""
        d = surface_distance(np.asarray(x[0:2])[None], self.obst[None])[0]
        return float(d.min(initial=ABSENT)) - self.m.r_drone

    def margins(self, x: np.ndarray) -> tuple[float, float]:
        """(clearance from the cylinders, height above the take-off point)."""
        return self.clearance(x), float(x[2] - self.pad[2])

    def recoverable(self, x: np.ndarray) -> bool:
        """The filter has a certified plan from x (a fresh one, or the backup as the continuation): what the first control
        step will need. Checked on a scratch filter; the flight's own filter starts at the hand-over."""
        X = np.asarray(x, np.float64)[None]
        obst = self.obst[None].copy()
        known = surface_distance(X[:, 0:2], obst) <= self.cfg.sensing_range
        obst[..., 3] = known.astype(float)
        f = OnlineFilter(self.m, self.actor, self.fcfg, self.nominal)
        goal = self.goal[None]
        d = self.d_hat[None]
        f.reset(X, obst, goal)
        plan = f.decide(X, obst, goal, self.nominal(X, goal, obst, d), d)
        return MODES[int(plan.mode[0])] in ("fresh", "committed")

    def nominal_goal(self, t: float) -> np.ndarray:
        return self.goal.copy()

    # ------------------------------------------------------------------ one control step
    def act(self, x: np.ndarray, t: float, *, a_meas: float | None = None, omega_meas=None,
            dt_obs: float | None = None) -> ControlOutput:
        """Command for state x (task frame, z up) at time t since the hand-over."""
        t0 = time.perf_counter()
        m = self.m
        X = np.asarray(x, np.float64)[None]
        d = self._estimate(X[0])[None]
        d_true = surface_distance(X[:, 0:2], self.obst[None])[0]
        self.known |= d_true <= self.cfg.sensing_range
        obst = self.obst[None].copy()
        obst[0, :, 3] = self.known.astype(float)
        goal = self.goal[None]
        if not self.started:
            self.filter.reset(X, obst, goal)
            self.started = True
        decided = self.k % int(self.fcfg.period) == 0
        u_n0 = self.nominal(X, goal, obst, d) if decided else None
        if decided:
            self.plan = self.filter.decide(X, obst, goal, u_n0, d)
        g_p, o_p, d_p = self.filter.plan_nominal_inputs()
        if decided and np.array_equal(g_p, goal) and np.array_equal(o_p, obst) and np.array_equal(d_p, d):
            u_nom = u_n0                       # a fresh plan: the plan's nominal inputs are this step's (the same call)
        else:
            u_nom = self.nominal(X, g_p, o_p, d_p)
        u = self.filter.command(X, u_nom)[0]
        self._advance(X[0], float(u[0]), self.dt)
        self.k += 1
        plan = self.plan
        lam = 0.0 if bool(plan.held[0]) else float(plan.lam[0])
        clear = float(d_true.min(initial=ABSENT)) - m.r_drone
        in_base = bool(margins(m, X, obst, self.fcfg.margin, d)[1][0] >= 0.0)
        dist_goal = float(np.linalg.norm(X[0, 0:2] - self.goal[0:2]))
        if self.goal_entered_t is None and dist_goal < self.cfg.goal_radius:
            self.goal_entered_t = float(t)
        info = {"d_hat": self.d_hat.copy(), "lam": lam, "mode": MODES[int(plan.mode[0])] if decided else "between",
                "n_cert": int(plan.stats["n_cert"][0]) if decided else -1,
                "steps": int(plan.stats["steps_vehicle"][0]) if decided else 0,
                "decide_ms": 1e3 * float(plan.stats["time"]) if decided else 0.0,
                "clearance": clear, "known": int(self.known.sum()), "dist_goal": dist_goal,
                "intervene": bool(np.abs(u - u_nom[0]).max() > 1e-6), "goal": self.goal.copy()}
        return ControlOutput(u=u, u_nominal=u_nom[0].copy(), slack=0.0, qp_used=True, safeguard_lambda=lam, h_cone=clear,
                             height=float(X[0, 2] - self.pad[2]), in_base_set=in_base,
                             latency_s=time.perf_counter() - t0, info=info)


    def close(self) -> None:
        pass


def make_runtime(cfg: CylOnlineRuntimeConfig, *, ps2rl_path: str | None = None, logger=None):
    """The controller: in this process, or (cfg.process) in a worker process behind a proxy with the same interface."""
    if cfg.process:
        from ps2rl.deploy.cyl_online_worker import RuntimeProcess
        return RuntimeProcess(cfg, ps2rl_path=ps2rl_path, logger=logger)
    return CylOnlineRuntime(cfg, ps2rl_path=ps2rl_path, logger=logger)


__all__ = ["CylOnlineRuntime", "CylOnlineRuntimeConfig", "LAMBDA_SETS", "NOMINALS", "course_layout", "make_runtime"]
