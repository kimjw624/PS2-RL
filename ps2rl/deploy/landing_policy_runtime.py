"""Online landing controller with a trained Phase-II landing policy as the nominal controller.

One state in, one action out, for the 50 Hz loop of the landing flight (``landing_flight.py``):

    policy    a Phase-II landing run (``train_phase2_landing_ue_cached.py``: configs.json with
              ps2 / env / actor, <weights>_weights.pkl); its 29-D observation
              [x, d_hat, x_ref(k), omega_ref(k), t, sin, cos] is rebuilt here from the state, a
              disturbance observer and the run's reference
    filter    the landing BCBF-QP with the discrete-time safeguard, built from a Phase-I landing
              checkpoint (cone, floor, base set, safe-arrival backup) - the same filter as
              ``landing_runtime.LandingRuntime``; the checkpoint's geometry must match the run's
    observer  d_hat = lambda (v - xi); "measured" integrates each interval, once it has ended, with
              the mean of the measured thrust at its two ends; "commanded" is the training form;
              "off" gives d_hat = 0
    predict   optional delay compensation: the policy and the filter act on the state predicted
              ``predict_ahead`` seconds ahead with the model, d_hat and the commands still on
              their way (the first segment uses the measured rates and thrust)

The interface is that of ``LandingRuntime`` (``act``, ``recoverable``, ``margins``,
``nominal_goal``, ``start_state``, ...), with ``act`` also taking the measured thrust and rates.
JAX's environment (platform, threads, x64) must be set before JAX is imported, so JAX is imported
in ``__init__``.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import pickle
import sys
import time

import numpy as np

from ps2rl.deploy.landing_runtime import ControlOutput

OBSERVERS = ("measured", "commanded", "off")
_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class LandingPolicyRuntimeConfig:
    run: str                                     # Phase-II landing run directory (the policy)
    backup: str                                  # Phase-I landing checkpoint for the filter (safe-arrival backup)
    weights: str = "best"                        # "best" or "final"
    filter: str = "alpha=10,alpha_floor=20"      # landing BCBF config overrides, KEY=VAL separated by ','
    use_filter: bool = True                      # False: the policy alone (comparison only)
    qp_solve_dtype: str = "float64"
    observer: str = "measured"                   # measured / commanded / off (see the module docstring)
    observer_lambda: float = 0.0                 # 0: the run's (summary.json ue_config), else 20
    predict_ahead: float = 0.0                   # delay compensation [s]; 0 = off
    # after the training episode (reference + extra time) the time features stop at its last step,
    # so the policy keeps seeing the end of the reference (the pad) instead of times it never saw
    clamp_time: bool = True
    jax_platform: str = "cpu"
    single_thread: bool = True


def _resolve(path: str, base: Path | None = None) -> str:
    if not path:
        return ""
    p = Path(path).expanduser()
    if p.is_absolute() or p.exists():
        return str(p.resolve())
    for root in ([base] if base is not None else []) + [_ROOT]:
        if (root / p).exists():
            return str((root / p).resolve())
    return str(p)


def _set_env(cfg: LandingPolicyRuntimeConfig) -> None:
    os.environ.setdefault("JAX_PLATFORMS", cfg.jax_platform)
    if cfg.single_thread and cfg.jax_platform == "cpu":
        os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")
        for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            os.environ.setdefault(var, "1")
    if cfg.qp_solve_dtype == "float64":
        os.environ.setdefault("JAX_ENABLE_X64", "1")


class LandingPolicyRuntime:
    def __init__(self, cfg: LandingPolicyRuntimeConfig, *, ps2rl_path: str | None = None, logger=None) -> None:
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

        from ps2rl.cil import quadrotor_landing_backup_cbf as lbcbf
        from ps2rl.envs.quadrotor_env import quadrotor_dynamics, quadrotor_step_euler
        from ps2rl.evaluation.landing_phase2_run import parse_field_overrides
        from ps2rl.utils.policy import ActorConfig, actor_mean_action

        if cfg.qp_solve_dtype == "float64" and not jax.config.jax_enable_x64:
            raise RuntimeError("qp_solve_dtype=float64 needs JAX_ENABLE_X64=1 before JAX is first imported")
        self._jax, self._jnp = jax, jnp

        # ---- the policy run
        run = Path(_resolve(cfg.run, base))
        if not (run / "configs.json").exists():
            raise FileNotFoundError(f"{run}/configs.json not found (a Phase-II landing run)")
        cfgs = json.loads((run / "configs.json").read_text())
        if bool(cfgs.get("residual", False)):
            raise ValueError("residual Phase-II runs (policy = nominal tracker + residual) are not supported here")
        env = cfgs["env"]
        ad = dict(cfgs["actor"])
        ad["hidden_sizes"] = tuple(ad["hidden_sizes"])
        actor_cfg = ActorConfig(**ad)
        if int(actor_cfg.obs_dim) != 29:
            raise ValueError(f"expected the 29-D landing observation, the run's actor takes {actor_cfg.obs_dim}")
        with open(run / f"{cfg.weights}_weights.pkl", "rb") as f:
            params = jax.tree_util.tree_map(lambda v: jnp.asarray(v, jnp.float32), pickle.load(f)["actor_params"])
        summary = json.loads((run / "summary.json").read_text()) if (run / "summary.json").exists() else {}
        lam = float(cfg.observer_lambda) or float(summary.get("ue_config", {}).get("observer_lambda", 20.0))
        self.observer_lambda = lam

        # ---- the filter (Phase-I landing checkpoint)
        items = [t for t in cfg.filter.split(",") if t.strip()]
        overrides = parse_field_overrides(items, lbcbf.QuadrotorLandingBCBFConfig, extra={"floor_constraint": bool})
        self.bcbf = lbcbf.landing_bcbf_config_from_checkpoint(_resolve(cfg.backup, base), qp_solve_dtype=cfg.qp_solve_dtype,
                                                               **overrides)
        self.filter_overrides = overrides
        lc = self.bcbf.landing
        self.landing = lc
        run_geo = summary.get("landing_config", {})
        for k in ("cone_r0", "cone_theta_deg", "cone_eps", "pad_x", "pad_y", "pad_z", "a_cmd_max", "omega_max", "dt"):
            if k in run_geo and not np.isclose(float(run_geo[k]), float(getattr(lc, k))):
                raise ValueError(f"the backup's {k} = {getattr(lc, k)} differs from the policy run's {run_geo[k]}")
        self.dt = float(lc.dt)
        self.gravity = float(lc.gravity)
        self.pad = np.array([lc.pad_x, lc.pad_y, lc.pad_z], dtype=np.float64)
        self.action_low = np.asarray(self.bcbf.action_low, dtype=np.float64)
        self.action_high = np.asarray(self.bcbf.action_high, dtype=np.float64)
        self.hover_action = np.array([lc.gravity, 0.0, 0.0, 0.0], dtype=np.float64)
        self.hover_position = self.pad + np.array([0.0, 0.0, lc.z_des])
        self.backup = cfg.backup

        # ---- the reference
        ref = np.load(_resolve(env["reference_path"], base))
        self.reference_states = np.asarray(ref["states"], np.float64)
        self._ref_omega = np.asarray(ref["omega_cmd"], np.float64)
        t_ref = np.asarray(ref["t"], np.float64)
        if abs(float(t_ref[1] - t_ref[0]) - self.dt) > 1e-9:
            raise ValueError("reference dt differs from the checkpoint dt")
        self.reference_duration = float(t_ref[-1])
        self.episode_steps = int(round((float(t_ref[-1]) + float(env.get("extra_sec", 0.0))) / self.dt))
        self.episode_duration = self.episode_steps * self.dt
        self.start_state = self.reference_states[0].copy()
        self.target = self.reference_states[-1, 0:3].copy()
        self.reference_path = str(env["reference_path"])

        projector = lbcbf.QuadrotorLandingBackupCBFProjector(self.bcbf)
        safe, base_set = lbcbf.landing_sets(self.bcbf)
        self._rec_fn = lbcbf.make_recoverability_fn(self.bcbf, projector.runtime)
        f32 = jnp.float32
        low, high = jnp.asarray(self.action_low, f32), jnp.asarray(self.action_high, f32)
        scale = jnp.asarray(self.bcbf.action_scale, f32)
        ref_states = jnp.asarray(self.reference_states, f32)
        ref_omega = jnp.asarray(self._ref_omega, f32)
        n_ref = int(ref_states.shape[0])
        t_total = float(t_ref[-1])
        k_max = self.episode_steps - 1
        observer_on = cfg.observer != "off"
        use_filter = bool(cfg.use_filter)
        sc_lo, sc_hi, w_max = float(lc.a_cmd_min), float(lc.a_cmd_max), float(lc.omega_max)

        def step(x, x_ctl, xi, k):
            dh = lam * (x[3:6] - xi) * (1.0 if observer_on else 0.0)
            kk = jnp.minimum(k, k_max) if cfg.clamp_time else k
            t = kk.astype(f32) * self.dt
            ph = 2.0 * jnp.pi * jnp.clip(t / t_total, 0.0, 1.0)
            i = jnp.clip(kk, 0, n_ref - 1)
            obs = jnp.concatenate([x_ctl, dh, ref_states[i], ref_omega[i], jnp.stack([t, jnp.sin(ph), jnp.cos(ph)])]).astype(f32)
            raw = jnp.clip(actor_mean_action(params, obs[None], scale, actor_cfg, action_low=low, action_high=high)[0], low, high)
            if use_filter:
                u, slack, used, info = lbcbf.solve_backup_cbf_qp_single_with_info(x_ctl, raw, self.bcbf, projector.runtime)
                lam_ = info.get("safeguard_lambda", jnp.asarray(1.0, f32))
            else:
                u, slack, used, lam_ = raw, jnp.asarray(0.0, f32), jnp.asarray(True), jnp.asarray(1.0, f32)
            u = jnp.clip(u.astype(f32), low, high)
            xdot = quadrotor_dynamics(x, u, self.gravity, sc_lo, sc_hi, w_max)
            xi_next = xi + self.dt * (xdot[3:6] + dh)          # the training form ("commanded")
            comp = safe.component_values(x)
            return u, raw, slack, used, lam_, comp[0], x[2] - lc.pad_z, base_set.contains(x), dh, xi_next

        def observe(x_prev, xi, dh_prev, a_bar, dt_obs):
            u = jnp.asarray([0.0, 0.0, 0.0, 0.0], f32).at[0].set(a_bar)
            xdot = quadrotor_dynamics(x_prev, u, self.gravity, sc_lo, sc_hi, w_max)
            return xi + dt_obs * (xdot[3:6] + dh_prev)

        def predict(x, u, h, dh):
            xn = quadrotor_step_euler(x, u, h, self.gravity, sc_lo, sc_hi, w_max)
            return xn.at[0:3].add(0.5 * h * h * dh).at[3:6].add(h * dh)

        self._step = jax.jit(step)
        self._observe = jax.jit(observe)
        self._predict_seg = jax.jit(predict)
        self.xi: np.ndarray | None = None
        self.d_hat = np.zeros(3)
        self._pend: dict | None = None
        self._cmd_hist: list[tuple[float, np.ndarray]] = []
        self._warm_latency = self.warmup()

    # ------------------------------------------------------------------ observer
    def reset(self) -> None:
        """Forget the observer and the command history (a new flight)."""
        self.xi = None
        self.d_hat = np.zeros(3)
        self._pend = None
        self._cmd_hist = []

    def reset_observer(self, x: np.ndarray, d_hat: np.ndarray | None = None) -> None:
        dh = np.zeros(3) if d_hat is None else np.asarray(d_hat, np.float64)
        self.xi = np.asarray(x, np.float64)[3:6] - dh / self.observer_lambda
        self.d_hat = dh.copy()
        self._pend = None
        self._cmd_hist = []

    def _advance_measured(self, x: np.ndarray, a_thrust: float | None, dt_obs: float | None) -> None:
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
        """The state predict_ahead seconds after t."""
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
            if t - h + 1e-9 < s < t:
                segs.append((s + h - tau, u_cur))
                tau, u_cur = s + h, u
        segs.append((t + h - tau, u_cur))
        xp = jnp.asarray(np.asarray(x, np.float32))
        dh = jnp.asarray(self.d_hat, np.float32)
        for length, u in segs:
            if length > 1e-6:
                xp = self._predict_seg(xp, jnp.asarray(np.asarray(u, np.float32)), float(length), dh)
        return np.asarray(xp, np.float64)

    # ------------------------------------------------------------------ helpers (LandingRuntime interface)
    def reference_at(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        i = int(min(max(round(float(t) / self.dt), 0), len(self.reference_states) - 1))
        return self.reference_states[i].copy(), np.r_[self.gravity, self._ref_omega[i]]

    def nominal_goal(self, t: float | None) -> np.ndarray:
        return self.reference_at(1e9 if t is None else t)[0][0:3]

    def recoverable(self, x: np.ndarray) -> bool:
        rec, _, _ = self._rec_fn(self._jnp.asarray(np.asarray(x, dtype=np.float32))[None])
        return bool(np.asarray(rec)[0])

    def margins(self, x: np.ndarray) -> tuple[float, float]:
        lc = self.landing
        dx, dy = float(x[0]) - lc.pad_x, float(x[1]) - lc.pad_y
        tan_t = float(np.tan(np.radians(lc.cone_theta_deg)))
        zeta = float(x[2]) - lc.pad_z
        return lc.cone_r0 + tan_t * zeta - float(np.sqrt(dx * dx + dy * dy + lc.cone_eps**2)), zeta

    # ------------------------------------------------------------------ online call
    def act(self, x: np.ndarray, t: float = 0.0, a_meas: float | None = None, omega_meas=None,
            dt_obs: float | None = None) -> ControlOutput:
        """One control step at time t since the hand-over; advances the observer."""
        jnp = self._jnp
        started = time.perf_counter()
        x = np.asarray(x, np.float64)
        if self.xi is None:
            self.reset_observer(x)
        a_ok = a_meas is not None and bool(np.isfinite(a_meas))
        measured = self.cfg.observer == "measured"
        if measured or self._pend is not None:
            self._advance_measured(x, float(a_meas) if a_ok else None, dt_obs)
        x_ctl = self.predict(x, t, a_meas if a_ok else None, omega_meas) if self.cfg.predict_ahead > 0 else x
        k = int(round(float(t) / self.dt))
        out = self._step(jnp.asarray(x.astype(np.float32)), jnp.asarray(x_ctl.astype(np.float32)), jnp.asarray(self.xi, np.float32),
                         jnp.asarray(k, jnp.int32))
        u, raw, slack, used, lam, h_cone, height, in_b, dh, xi_next = (np.asarray(v) for v in out)
        u = u.astype(np.float64)
        self.d_hat = dh.astype(np.float64)
        if measured:
            self._pend["u_thrust"] = float(u[0])
        else:
            self.xi = xi_next.astype(np.float64)
            self._pend = None
        if self.cfg.predict_ahead > 0:
            self._cmd_hist.append((float(t), u.copy()))
            del self._cmd_hist[:-(int(np.ceil(self.cfg.predict_ahead / self.dt)) + 3)]
        return ControlOutput(
            u=u, u_nominal=raw.astype(np.float64), slack=float(slack), qp_used=bool(used),
            safeguard_lambda=float(lam), h_cone=float(h_cone), height=float(height), in_base_set=bool(in_b),
            latency_s=time.perf_counter() - started, info={"d_hat": self.d_hat.copy()},
        )

    def warmup(self, iterations: int = 8) -> float:
        x0 = self.start_state.copy()
        self.reset_observer(x0)
        self.recoverable(x0)
        self.observe(x0, self.gravity, self.dt)
        lat = [self.act(x0, i * self.dt, a_meas=self.gravity, omega_meas=np.zeros(3)).latency_s for i in range(iterations)]
        self.reset()
        steady = float(min(lat[1:]))
        msg = (f"landing policy runtime ready | policy {Path(self.cfg.run).name} ({self.cfg.weights}) | backup "
               f"{Path(self.cfg.backup).name} | filter {'on' if self.cfg.use_filter else 'OFF'} {self.filter_overrides} "
               f"({self.cfg.qp_solve_dtype} QP) | observer {self.cfg.observer} (lambda {self.observer_lambda:g})"
               + (f" | delay compensation {self.cfg.predict_ahead * 1e3:.0f} ms" if self.cfg.predict_ahead > 0 else "")
               + f" | episode {self.episode_duration:.2f} s | action box {self.action_low.tolist()} .. {self.action_high.tolist()} | "
               f"warm latency {steady * 1e3:.1f} ms")
        (self._log.info if self._log is not None else print)(msg)
        if steady > 0.8 * self.dt:
            warn = f"control-step latency {steady * 1e3:.1f} ms is close to the {self.dt * 1e3:.0f} ms control period"
            (self._log.warn if self._log is not None else print)(warn)
        return steady


__all__ = ["LandingPolicyRuntime", "LandingPolicyRuntimeConfig", "OBSERVERS"]
