"""Online controller for the outside-cone task: a trained Phase-II policy passed through a safety filter.

One state in, one action out, for a 50 Hz control loop. It is the single-state form of what
``scripts/eval_phase2_cone_obstacle_ue.py`` runs in batches, so a flight can be compared with the
evaluation directly:

    policy    the actor of a Phase-II run (``scripts/train_phase2_cone_obstacle_ue.py``): the stage-1
              tracker or the PS2 policy; 29-D observation [x, d_hat, x_ref(k), omega_ref(k), t, sin, cos]
    observer  d_hat = lambda (v - xi),  xi_{k+1} = xi_k + dt (a_nom(x_k, u_k) + d_hat_k), as in training
    filter    "ue"       the cone-obstacle UE-bCBF (tube, observer term) with the run's or a given backup
              "nominal"  the plain cone-obstacle bCBF (LQR backup, no disturbance model)
              "none"     the policy alone (comparison flights; the funnel is virtual)

Positions are in the task frame of the run (funnel axis at (axis_x, axis_y), ground at ground_z, z up).
Before the hand-over the observer can be run on the measured thrust (``observe``), so that the
estimate of the model mismatch is warm when the policy starts. Nothing here depends on ROS or PX4;
``ps2rl/deploy/cone_obstacle_flight.py`` holds the flight logic.

JAX's environment (platform, threads, x64) must be set before JAX is imported, so JAX is imported
inside ``ConeObstacleRuntime.__init__``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
import os
from pathlib import Path
import pickle
import sys
import time

import numpy as np

from ps2rl.deploy.landing_runtime import ControlOutput

FILTERS = ("ue", "nominal", "none")
OBSERVERS = ("measured", "commanded", "off")
_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class ConeObstacleRuntimeConfig:
    run: str                          # Phase-II run directory (configs.json + <weights>_weights.pkl)
    weights: str = "best"             # "best" or "final"
    filter: str = "ue"                # "ue" | "nominal" | "none"
    backup_from: str = ""             # chosen_backup.json of compare_phase1_cone_obstacle.py (default: the run's own backup)
    qp_solve_dtype: str = "float64"
    jax_platform: str = "cpu"
    single_thread: bool = True
    # disturbance observer during the policy episode:
    #   "commanded"  xi advanced with the commanded action (the training form). Actuator lag and delay then
    #                look like a disturbance to the observer, and the policy and filter react to it
    #   "measured"   xi advanced with the measured thrust acceleration and the measured attitude (needs a_meas):
    #                d_hat then estimates the external force only. Each interval is integrated once it has ended,
    #                with the mean of the thrust measured at its two ends (the thrust measured at the start of an
    #                interval is the previous command's, not the one acting during it)
    #   "off"        d_hat = 0 (no disturbance information)
    observer: str = "measured"
    # delay compensation [s] (0 = off): the policy and the filter act on the state predicted this far ahead
    # with the model, the disturbance estimate and the commands still on their way (a Smith predictor for a
    # pure transport delay; an actuator lag is roughly a delay of its time constant). The first segment uses
    # the measured thrust and body rates when given. Set it to the command-to-effect delay of the vehicle
    # (computation + transport + the autopilot's control cycle).
    predict_ahead: float = 0.0


def _resolve(path: str, base: Path | None = None) -> str:
    """A path as given, or relative to the PS2-RL root (where run files record their paths from)."""
    if not path:
        return ""
    p = Path(path).expanduser()
    if p.is_absolute() or p.exists():
        return str(p.resolve())
    for root in ([base] if base is not None else []) + [_ROOT]:
        if (root / p).exists():
            return str((root / p).resolve())
    return str(p)


def _set_env(cfg: ConeObstacleRuntimeConfig) -> None:
    os.environ.setdefault("JAX_PLATFORMS", cfg.jax_platform)
    if cfg.single_thread and cfg.jax_platform == "cpu":
        os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")
        for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            os.environ.setdefault(var, "1")
    if cfg.qp_solve_dtype == "float64":
        os.environ.setdefault("JAX_ENABLE_X64", "1")


class ConeObstacleRuntime:
    def __init__(self, cfg: ConeObstacleRuntimeConfig, *, ps2rl_path: str | None = None, logger=None) -> None:
        if cfg.filter not in FILTERS:
            raise ValueError(f"filter must be one of {FILTERS}, got '{cfg.filter}'")
        if cfg.observer not in OBSERVERS:
            raise ValueError(f"observer must be one of {OBSERVERS}, got '{cfg.observer}'")
        self.cfg = cfg
        self._log = logger
        _set_env(cfg)
        if ps2rl_path:
            p = str(Path(ps2rl_path).expanduser().resolve())
            if p not in sys.path:
                sys.path.insert(0, p)
        base = Path(ps2rl_path).expanduser().resolve() if ps2rl_path else None

        import jax
        import jax.numpy as jnp

        sys.path.insert(0, str(_ROOT / "scripts"))
        from cone_obstacle_common import task_config
        from ps2rl.cil import cone_obstacle_ue_bcbf as cu
        from ps2rl.cil import quadrotor_cone_obstacle_cbf as ncbf
        from ps2rl.envs.cone_obstacle_ue_env import ConeObstacleUEEnvConfig
        from ps2rl.envs.quadrotor_env import quadrotor_dynamics, quadrotor_step_euler
        from ps2rl.phase2_ps2.cone_obstacle_ue_ps2_trainer import ConeUEPS2Config, filter_config
        from ps2rl.sets.quadrotor_cone_obstacle_sets import obstacle_sets
        from ps2rl.utils.policy import ActorConfig, actor_mean_action

        if cfg.qp_solve_dtype == "float64" and not jax.config.jax_enable_x64:
            raise RuntimeError("qp_solve_dtype=float64 needs JAX_ENABLE_X64=1 before JAX is first imported")
        self._jax, self._jnp = jax, jnp
        run = Path(_resolve(cfg.run, base))
        if not (run / "configs.json").exists():
            raise FileNotFoundError(f"{run}/configs.json not found (a Phase-II run of train_phase2_cone_obstacle_ue.py)")
        cfgs = json.loads((run / "configs.json").read_text())
        ps2 = ConeUEPS2Config.from_dict(cfgs["ps2"])
        if cfg.backup_from:
            ch = json.loads(Path(_resolve(cfg.backup_from, base)).read_text())
            ps2 = replace(ps2, backup=ch["backup"], backup_path=ch.get("backup_path", ""))
        if ps2.backup == "learned":
            ps2 = replace(ps2, backup_path=_resolve(ps2.backup_path, base))
        self.task = cfgs["task"]
        env_cfg = ConeObstacleUEEnvConfig(**cfgs["env"])
        self.ue_cbf = filter_config(ps2, self.task)
        self.backup = (ps2.backup, ps2.backup_path)
        sc = self.ue_cbf.scenario
        self.scenario = sc
        self.dt, self.gravity = float(sc.dt), float(sc.gravity)
        self.action_low = np.asarray(self.ue_cbf.action_low, np.float64)
        self.action_high = np.asarray(self.ue_cbf.action_high, np.float64)
        self.hover_action = np.array([self.gravity, 0.0, 0.0, 0.0])
        lam = float(self.ue_cbf.ue.observer_lambda)
        self.observer_lambda = lam

        ad = dict(cfgs["actor"])
        ad["hidden_sizes"] = tuple(ad["hidden_sizes"])
        actor_cfg = ActorConfig(**ad)
        with open(run / f"{cfg.weights}_weights.pkl", "rb") as f:
            params = jax.tree_util.tree_map(lambda v: jnp.asarray(v, jnp.float32), pickle.load(f)["actor_params"])

        ref = np.load(_resolve(env_cfg.reference_path, base))
        self.reference_states = np.asarray(ref["states"], np.float64)
        self._ref_omega = np.asarray(ref["omega_cmd"], np.float64)
        t_ref = np.asarray(ref["t"], np.float64)
        if abs(float(t_ref[1] - t_ref[0]) - self.dt) > 1e-9:
            raise ValueError("reference dt differs from the scenario dt")
        self.reference_duration = float(t_ref[-1])
        self.episode_steps = int(round((float(t_ref[-1]) + float(env_cfg.extra_sec)) / self.dt))
        self.episode_duration = self.episode_steps * self.dt
        self.start_state = self.reference_states[0].copy()
        self.hover_position = self.start_state[0:3].copy()
        self.target = self.reference_states[-1, 0:3].copy()
        # anchor of the flight frames: the ground point below the start (the flight logic puts the vehicle's
        # ground point here, so the funnel axis lies at axis - start horizontally from where it took off)
        self.pad = np.array([self.start_state[0], self.start_state[1], float(sc.ground_z)])
        self.axis = np.array([float(sc.axis_x), float(sc.axis_y)])
        tx, ty = sc.target_center
        self._target_xy = np.array([float(tx), float(ty)])

        ref_states = jnp.asarray(self.reference_states, jnp.float32)
        ref_omega = jnp.asarray(self._ref_omega, jnp.float32)
        n_ref = int(ref_states.shape[0])
        t_total = float(t_ref[-1])
        f32 = jnp.float32
        low, high = jnp.asarray(self.action_low, f32), jnp.asarray(self.action_high, f32)
        scale = jnp.asarray(self.ue_cbf.action_scale, f32)
        safe, base_set = obstacle_sets(sc)
        qp_dtype = jnp.float64 if cfg.qp_solve_dtype == "float64" else None
        e_bar = jnp.asarray(float(self.ue_cbf.ue.e_bar), f32)

        rt = cu.make_ue_runtime(self.ue_cbf)
        self._rec_ue = cu.make_recoverability_fn_ue(self.ue_cbf, rt)
        nom = ncbf.ConeObstacleBCBFConfig(scenario=task_config(self.task), alpha=self.ue_cbf.alpha,
                                          alpha_ground=self.ue_cbf.alpha_ground, base_alpha=self.ue_cbf.base_alpha,
                                          qp_solve_dtype=cfg.qp_solve_dtype)
        nom_rt = ncbf.get_cached_runtime(nom) if cfg.filter == "nominal" else None
        self._rec_nom = ncbf.make_recoverability_fn(nom) if cfg.filter == "nominal" else None

        def filt(x, dh, u_ref):
            if cfg.filter == "ue":
                u, aux = cu.project_full(x, dh, e_bar, u_ref, self.ue_cbf, rt, qp_dtype)
                return u, aux["slack"], aux["used_solver"], aux["safeguard_lambda"]
            if cfg.filter == "nominal":
                u, slack, used, info = ncbf.solve_backup_cbf_qp_batch_with_info(x[None], u_ref[None], nom, nom_rt)
                lam_ = info.get("safeguard_lambda", jnp.ones((1,), f32))
                return u[0], slack[0], used[0], lam_[0]
            return u_ref, jnp.asarray(0.0, f32), jnp.asarray(True), jnp.asarray(1.0, f32)

        observer_on = cfg.observer != "off"

        def step(x, x_ctl, xi, k):
            # x: the measured state (observer); x_ctl: the state the policy and the filter act on (x, or x predicted)
            dh = lam * (x[3:6] - xi) * (1.0 if observer_on else 0.0)
            t = k.astype(f32) * self.dt
            ph = 2.0 * jnp.pi * jnp.clip(t / t_total, 0.0, 1.0)
            i = jnp.clip(k, 0, n_ref - 1)
            obs = jnp.concatenate([x_ctl, dh, ref_states[i], ref_omega[i], jnp.stack([t, jnp.sin(ph), jnp.cos(ph)])]).astype(f32)
            raw = jnp.clip(actor_mean_action(params, obs[None], scale, actor_cfg, action_low=low, action_high=high)[0], low, high)
            u, slack, used, lam_ = filt(x_ctl, dh, raw)
            u = jnp.clip(u.astype(f32), low, high)
            xdot = quadrotor_dynamics(x, u, self.gravity, sc.a_cmd_min, sc.a_cmd_max, sc.omega_max)
            xi_next = xi + self.dt * (xdot[3:6] + dh)          # the training form ("commanded")
            comp = safe.component_values(x)
            return u, raw, slack, used, lam_, comp[0], comp[1], base_set.contains(x), dh, xi_next

        def observe(x_prev, xi, dh_prev, a_bar, dt_obs):
            # the interval that has just ended, integrated with the measured thrust (attitude at its start, as in training)
            u = jnp.asarray([0.0, 0.0, 0.0, 0.0], f32).at[0].set(a_bar)
            xdot = quadrotor_dynamics(x_prev, u, self.gravity, sc.a_cmd_min, sc.a_cmd_max, sc.omega_max)
            return xi + dt_obs * (xdot[3:6] + dh_prev)

        def predict(x, u, h, dh):
            xn = quadrotor_step_euler(x, u, h, self.gravity, sc.a_cmd_min, sc.a_cmd_max, sc.omega_max)
            return xn.at[0:3].add(0.5 * h * h * dh).at[3:6].add(h * dh)

        self._step = jax.jit(step)
        self._observe = jax.jit(observe)
        self._predict_seg = jax.jit(predict)
        self._pend: dict | None = None             # the observer interval not integrated yet (measured form)
        self._cmd_hist: list[tuple[float, np.ndarray]] = []   # (time issued, command) for the predictor
        self._margins = jax.jit(lambda x: safe.component_values(x))
        self.xi: np.ndarray | None = None
        self.d_hat = np.zeros(3)
        self.filter_overrides = {"filter": cfg.filter, "alpha": self.ue_cbf.alpha, "alpha_ground": self.ue_cbf.alpha_ground,
                                 "base_alpha": self.ue_cbf.base_alpha, "backup": ps2.backup, "backup_path": ps2.backup_path}
        self._warm_latency = self.warmup()

    # ------------------------------------------------------------------ observer
    def reset_observer(self, x: np.ndarray, d_hat: np.ndarray | None = None) -> None:
        """Start the observer at state x with estimate d_hat (default 0); forgets the command history."""
        dh = np.zeros(3) if d_hat is None else np.asarray(d_hat, np.float64)
        self.xi = np.asarray(x, np.float64)[3:6] - dh / self.observer_lambda
        self.d_hat = dh.copy()
        self._pend = None
        self._cmd_hist = []

    def reset(self) -> None:
        """Forget the observer and the command history (a new flight)."""
        self.xi = None
        self.d_hat = np.zeros(3)
        self._pend = None
        self._cmd_hist = []

    def _advance_measured(self, x: np.ndarray, a_thrust: float | None, dt_obs: float | None) -> None:
        """Integrate the interval since the previous sample with the measured thrust, then d_hat at x."""
        p = self._pend
        if p is not None:
            a_end = a_thrust if a_thrust is not None else (p["u_thrust"] if p["u_thrust"] is not None else p["a"])
            a_start = p["a"] if p["a"] is not None else a_end
            if a_end is not None and a_start is not None:
                h = self.dt if dt_obs is None else float(np.clip(dt_obs, 0.0, 0.1))
                jnp = self._jnp
                xi = self._observe(jnp.asarray(np.asarray(p["x"], np.float32)), jnp.asarray(self.xi, np.float32),
                                   jnp.asarray(p["dh"], np.float32), float(0.5 * (a_start + a_end)), h)
                self.xi = np.asarray(xi, np.float64)
        on = 1.0 if self.cfg.observer != "off" else 0.0
        self.d_hat = on * self.observer_lambda * (np.asarray(x, np.float64)[3:6] - self.xi)
        self._pend = {"x": np.asarray(x, np.float64).copy(), "a": a_thrust, "dh": self.d_hat.copy(), "u_thrust": None}

    def observe(self, x: np.ndarray, a_thrust: float, dt_obs: float) -> np.ndarray:
        """Advance the observer with the measured thrust acceleration (before the hand-over)."""
        if self.xi is None:
            self.reset_observer(x)
        a = float(a_thrust) if a_thrust is not None and np.isfinite(a_thrust) else None
        self._advance_measured(x, a, dt_obs)
        return self.d_hat.copy()

    def predict(self, x: np.ndarray, t: float, a_meas: float | None = None, omega_meas=None) -> np.ndarray:
        """The state predict_ahead seconds after t (see ``predict_ahead``)."""
        h = float(self.cfg.predict_ahead)
        if h <= 0.0:
            return np.asarray(x, np.float64)
        jnp = self._jnp
        hist = self._cmd_hist
        before = [u for s, u in hist if s <= t - h + 1e-9]
        u_now = (before[-1] if before else (hist[0][1] if hist else self.hover_action)).copy()
        if a_meas is not None and np.isfinite(a_meas):
            u_now[0] = float(a_meas)
        if omega_meas is not None and np.all(np.isfinite(omega_meas)):
            u_now[1:4] = np.asarray(omega_meas, np.float64)
        segs, tau, u_cur = [], float(t), u_now
        for s, u in hist:
            if t - h + 1e-9 < s < t:                       # issued, not in effect yet: in effect from s + h
                segs.append((s + h - tau, u_cur))
                tau, u_cur = s + h, u
        segs.append((t + h - tau, u_cur))
        xp = jnp.asarray(np.asarray(x, np.float32))
        dh = jnp.asarray(self.d_hat, np.float32)
        for length, u in segs:
            if length > 1e-6:
                xp = self._predict_seg(xp, jnp.asarray(np.asarray(u, np.float32)), float(length), dh)
        return np.asarray(xp, np.float64)

    # ------------------------------------------------------------------ helpers
    def reference_at(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        i = int(min(max(round(float(t) / self.dt), 0), len(self.reference_states) - 1))
        return self.reference_states[i].copy(), np.r_[self.gravity, self._ref_omega[i]]

    def nominal_goal(self, t: float | None) -> np.ndarray:
        return self.reference_at(1e9 if t is None else t)[0][0:3]

    def margins(self, x: np.ndarray) -> tuple[float, float]:
        """(funnel barrier h_cone, height above the floor) without running the filter."""
        h = np.asarray(self._margins(self._jnp.asarray(np.asarray(x, np.float32))))
        return float(h[0]), float(h[1])

    def touches(self, x: np.ndarray) -> bool:
        sc = self.scenario
        r2 = float(np.sum((np.asarray(x[0:2], np.float64) - self._target_xy) ** 2))
        return bool(r2 <= sc.target_radius ** 2 and sc.floor_z <= float(x[2]) <= sc.cylinder_top)

    def recoverable(self, x: np.ndarray) -> bool:
        """Is x in the filter's C_N (UE: with the current estimate d_hat)?"""
        jnp = self._jnp
        xj = jnp.asarray(np.asarray(x, np.float32))[None]
        if self._rec_nom is not None:
            rec, _, _ = self._rec_nom(xj)
            return bool(np.asarray(rec)[0])
        return bool(np.asarray(self._rec_ue(xj, jnp.asarray(self.d_hat, np.float32)[None]))[0])

    # ------------------------------------------------------------------ online call
    def act(self, x: np.ndarray, t: float = 0.0, a_meas: float | None = None, omega_meas=None,
            dt_obs: float | None = None) -> ControlOutput:
        """One control step at time t since the hand-over; advances the observer (see ``observer``).

        a_meas / omega_meas: measured thrust acceleration and body rates (optional); dt_obs: time since the
        previous observer sample (default: the control step)."""
        jnp = self._jnp
        started = time.perf_counter()
        x = np.asarray(x, np.float64)
        if self.xi is None:
            self.reset_observer(x)
        a_ok = a_meas is not None and bool(np.isfinite(a_meas))
        measured = self.cfg.observer == "measured"
        if measured or self._pend is not None:   # (a pending interval from before the hand-over is always closed)
            self._advance_measured(x, float(a_meas) if a_ok else None, dt_obs)
        x_ctl = self.predict(x, t, a_meas if a_ok else None, omega_meas) if self.cfg.predict_ahead > 0 else x
        k = int(round(float(t) / self.dt))
        out = self._step(jnp.asarray(x.astype(np.float32)), jnp.asarray(x_ctl.astype(np.float32)), jnp.asarray(self.xi, np.float32),
                         jnp.asarray(k, jnp.int32))
        u, raw, slack, used, lam, h_cone, z_floor, in_b, dh, xi_next = (np.asarray(v) for v in out)
        u = u.astype(np.float64)
        self.d_hat = dh.astype(np.float64)
        if measured:
            self._pend["u_thrust"] = float(u[0])
        else:
            self.xi = xi_next.astype(np.float64)
            self._pend = None
        if self.cfg.predict_ahead > 0:
            self._cmd_hist.append((float(t), u.copy()))
            keep = int(np.ceil(self.cfg.predict_ahead / self.dt)) + 3
            del self._cmd_hist[:-keep]
        info = {"d_hat": self.d_hat.copy(), "touch": self.touches(x)}
        if self.cfg.predict_ahead > 0:
            info["x_pred"] = x_ctl
        return ControlOutput(
            u=u, u_nominal=raw.astype(np.float64), slack=float(slack), qp_used=bool(used),
            safeguard_lambda=float(lam), h_cone=float(h_cone), height=float(z_floor), in_base_set=bool(in_b),
            latency_s=time.perf_counter() - started, info=info,
        )

    def warmup(self, iterations: int = 8) -> float:
        """Compile before the vehicle is in the air; returns the steady latency in seconds."""
        x0 = self.start_state.copy()
        self.reset_observer(x0)
        self.recoverable(x0)
        self.observe(x0, self.gravity, self.dt)
        lat = [self.act(x0, i * self.dt, a_meas=self.gravity, omega_meas=np.zeros(3)).latency_s for i in range(iterations)]
        self.reset()
        steady = float(min(lat[1:]))
        msg = (f"cone-obstacle runtime ready | policy {Path(self.cfg.run).name} ({self.cfg.weights}) | filter {self.cfg.filter} | "
               f"observer {self.cfg.observer}" + (f" | delay compensation {self.cfg.predict_ahead * 1e3:.0f} ms" if self.cfg.predict_ahead > 0 else "")
               + (f" (backup {self.backup[0]} {Path(self.backup[1]).name if self.backup[1] else ''})" if self.cfg.filter == "ue" else "")
               + f" | {self.cfg.qp_solve_dtype} QP | action box {self.action_low.tolist()} .. {self.action_high.tolist()} | "
               f"episode {self.episode_duration:.2f} s | dt {self.dt:g} s | warm latency {steady * 1e3:.1f} ms")
        (self._log.info if self._log is not None else print)(msg)
        if steady > 0.8 * self.dt:
            warn = f"control-step latency {steady * 1e3:.1f} ms is close to the {self.dt * 1e3:.0f} ms control period"
            (self._log.warn if self._log is not None else print)(warn)
        return steady


__all__ = ["ConeObstacleRuntime", "ConeObstacleRuntimeConfig", "FILTERS", "OBSERVERS"]
