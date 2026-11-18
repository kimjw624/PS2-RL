"""Online landing controller: a nominal controller passed through the landing safety filter.

One state in, one action out, for a 50 Hz control loop. It is the single-state form of what
``scripts/test_landing_floor_reach.py`` runs in batches, so a flight can be compared with those
tests directly:

    nominal "lqr_target"       the backup checkpoint's hover LQR re-centred on a target point
                               (reach test T1)
    nominal "track_reference"  the same LQR gain around a reference trajectory with feedforward
                               (reach test T3)
    filter                     the landing BCBF-QP with the discrete-time safeguard, built from
                               the Phase-I backup checkpoint (cone, floor, base set, backup)

Positions are in the checkpoint's frame: the pad centre is at (pad_x, pad_y, pad_z), z up.
Nothing here depends on ROS or PX4; ``ps2rl/deploy/landing_flight.py`` holds the flight logic and
the bridge package only converts frames and publishes.

Environment variables (JAX platform, thread count, x64) must be set before JAX is imported, so
this module imports JAX inside ``LandingRuntime.__init__``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

NOMINALS = ("lqr_target", "track_reference")
TARGETS = ("pad_centre", "apex")


@dataclass
class LandingRuntimeConfig:
    backup: str                                  # Phase-I run directory or actor .pkl
    # filter settings, KEY=VAL separated by ',' over the landing BCBF config fields
    # (tuples '/'-separated), as in the reach tests
    filter: str = "alpha=10,alpha_floor=20"
    use_filter: bool = True                      # False: the nominal controller alone (comparison only)
    qp_solve_dtype: str = "float64"              # float64 avoids the rare non-finite float32 QP solutions
    nominal: str = "lqr_target"
    # target of "lqr_target", relative to the pad centre: "pad_centre", "apex" (the cone's apex,
    # r0 / tan(theta) below the pad) or "x,y,z" in metres. A target below the pad keeps pushing
    # down at the floor, so a thrust-map error cannot leave the vehicle hovering just above it.
    target: str = "pad_centre"
    # "lqr_target": the LQR sees a position error of at most this length [m] (it chases a point that
    # far along the straight line to the target). Its speed is then about that many m/s instead of
    # growing with the distance. 0: the plain re-centred LQR of the reach tests, which from 2 m away
    # first cuts thrust to the minimum and exceeds the speed range the backup was trained on.
    max_position_error: float = 0.0
    reference_path: str = ""                     # for "track_reference": .npz with states, a_cmd, omega_cmd
    jax_platform: str = "cpu"
    single_thread: bool = True


@dataclass
class ControlOutput:
    u: np.ndarray               # applied action [a_cmd, wx, wy, wz] (z-up body frame)
    u_nominal: np.ndarray       # nominal action before the filter
    slack: float                # QP slack (> 0: a row was relaxed)
    qp_used: bool               # False: the QP gave no finite solution and the backup action was applied
    safeguard_lambda: float     # 1: QP action accepted; < 1: blended toward the backup action
    h_cone: float               # cone barrier at the current state (>= 0 inside)
    height: float               # height above the pad plane
    in_base_set: bool
    latency_s: float
    info: dict[str, Any] = field(default_factory=dict)


def _set_env(cfg: LandingRuntimeConfig) -> None:
    os.environ.setdefault("JAX_PLATFORMS", cfg.jax_platform)
    if cfg.single_thread and cfg.jax_platform == "cpu":
        # many tiny sequential ops: threads cost more than they give, and they compete with the simulator
        os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")
        for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            os.environ.setdefault(var, "1")
    if cfg.qp_solve_dtype == "float64":
        os.environ.setdefault("JAX_ENABLE_X64", "1")


class LandingRuntime:
    def __init__(self, cfg: LandingRuntimeConfig, *, ps2rl_path: str | None = None, logger=None) -> None:
        if cfg.nominal not in NOMINALS:
            raise ValueError(f"nominal must be one of {NOMINALS}, got '{cfg.nominal}'")
        self.cfg = cfg
        self._log = logger
        _set_env(cfg)
        if ps2rl_path:
            p = str(Path(ps2rl_path).expanduser().resolve())
            if p not in sys.path:
                sys.path.insert(0, p)

        import jax
        import jax.numpy as jnp

        from ps2rl.cil import quadrotor_landing_backup_cbf as lbcbf
        from ps2rl.evaluation import landing_filter_reach as reach
        from ps2rl.evaluation.landing_phase2_run import parse_field_overrides

        if cfg.qp_solve_dtype == "float64" and not jax.config.jax_enable_x64:
            raise RuntimeError("qp_solve_dtype=float64 needs JAX_ENABLE_X64=1 before JAX is first imported")
        self._jax, self._jnp, self._lbcbf = jax, jnp, lbcbf

        items = [t for t in cfg.filter.split(",") if t.strip()]
        overrides = parse_field_overrides(items, lbcbf.QuadrotorLandingBCBFConfig, extra={"floor_constraint": bool})
        self.bcbf = lbcbf.landing_bcbf_config_from_checkpoint(cfg.backup, qp_solve_dtype=cfg.qp_solve_dtype, **overrides)
        self.filter_overrides = overrides
        lc = self.bcbf.landing
        self.landing = lc
        self.dt = float(lc.dt)
        self.gravity = float(lc.gravity)
        self.pad = np.array([lc.pad_x, lc.pad_y, lc.pad_z], dtype=np.float64)
        self.action_low = np.asarray(self.bcbf.action_low, dtype=np.float64)
        self.action_high = np.asarray(self.bcbf.action_high, dtype=np.float64)
        self.hover_action = np.array([lc.gravity, 0.0, 0.0, 0.0], dtype=np.float64)
        self.hover_position = self.pad + np.array([0.0, 0.0, lc.z_des])
        tan_t = float(np.tan(np.radians(lc.cone_theta_deg)))
        self.apex_depth = float(lc.cone_r0) / tan_t

        projector = lbcbf.QuadrotorLandingBackupCBFProjector(self.bcbf)
        safe, base = lbcbf.landing_sets(self.bcbf)
        self._rec_fn = lbcbf.make_recoverability_fn(self.bcbf, projector.runtime)
        f32 = jnp.float32
        low, high = jnp.asarray(self.action_low, dtype=f32), jnp.asarray(self.action_high, dtype=f32)

        def finish(x, u_nom):
            u_nom = jnp.clip(u_nom.astype(f32), low, high)
            if cfg.use_filter:
                u, slack, used, info = lbcbf.solve_backup_cbf_qp_single_with_info(x, u_nom, self.bcbf, projector.runtime)
                lam = info.get("safeguard_lambda", jnp.asarray(1.0, dtype=f32))
            else:
                u, slack, used, lam = u_nom, jnp.asarray(0.0, f32), jnp.asarray(True), jnp.asarray(1.0, f32)
            comp = safe.component_values(x)
            return (jnp.clip(u.astype(f32), low, high), u_nom, slack, used, lam, comp[0], x[2] - lc.pad_z, base.contains(x))

        self.reference_states = None
        self.reference_duration = None
        if cfg.nominal == "lqr_target":
            self.target = self._parse_target(cfg.target)
            target = jnp.asarray(self.target, dtype=f32)
            hover = jnp.asarray(self.hover_position, dtype=f32)
            e_max = float(cfg.max_position_error)
            if e_max < 0.0:
                raise ValueError(f"max_position_error must be >= 0 (0 = off), got {e_max}")
            lqr = reach.recentred_lqr(self.bcbf)

            def nominal(x):
                e = target - x[0:3]
                n = jnp.sqrt(jnp.sum(e * e) + 1e-12)
                goal = x[0:3] + e * jnp.minimum(1.0, e_max / n) if e_max > 0.0 else target
                return lqr(x, goal - hover)

            self._fn = jax.jit(lambda x: finish(x, nominal(x)))
            self.start_state = None
        else:
            if not cfg.reference_path:
                raise ValueError("nominal 'track_reference' needs reference_path")
            b = np.load(cfg.reference_path)
            self.reference_states = np.asarray(b["states"], dtype=np.float64)
            self._ref_u = np.concatenate([np.asarray(b["a_cmd"], dtype=np.float64)[:, None], np.asarray(b["omega_cmd"], dtype=np.float64)], axis=1)
            self.reference_duration = (len(self.reference_states) - 1) * self.dt
            self.target = self.reference_states[-1, 0:3].copy()
            track = reach.tracking_lqr(self.bcbf)
            self._fn = jax.jit(lambda x, x_ref, u_ref: finish(x, track(x, x_ref, u_ref)))
            self.start_state = self.reference_states[0].copy()
        self._warm_latency = self.warmup()

    # ------------------------------------------------------------------ helpers
    def _parse_target(self, spec: str) -> np.ndarray:
        s = str(spec).strip().lower()
        if s == "pad_centre":
            rel = np.zeros(3)
        elif s == "apex":
            rel = np.array([0.0, 0.0, -self.apex_depth])
        else:
            try:
                rel = np.array([float(t) for t in s.split(",")], dtype=np.float64)
            except ValueError as e:
                raise ValueError(f"target must be one of {TARGETS} or 'x,y,z', got '{spec}'") from e
            if rel.shape != (3,):
                raise ValueError(f"target 'x,y,z' needs three numbers, got '{spec}'")
        return self.pad + rel

    def reference_at(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        """Reference state and feedforward action at time t (held at the last sample afterwards)."""
        idx = min(max(float(t) / self.dt, 0.0), float(len(self.reference_states) - 1))
        i0 = int(np.floor(idx))
        i1 = min(i0 + 1, len(self.reference_states) - 1)
        w = idx - i0
        x_ref = (1.0 - w) * self.reference_states[i0] + w * self.reference_states[i1]
        x_ref[6:10] /= max(float(np.linalg.norm(x_ref[6:10])), 1e-9)
        return x_ref, (1.0 - w) * self._ref_u[i0] + w * self._ref_u[i1]

    def nominal_goal(self, t: float) -> np.ndarray:
        """Where the nominal controller is heading at time t (for logging)."""
        if self.reference_states is None:
            return self.target.copy()
        return self.reference_at(t)[0][0:3]

    # ------------------------------------------------------------------ online calls
    def act(self, x: np.ndarray, t: float = 0.0) -> ControlOutput:
        jnp = self._jnp
        started = time.perf_counter()
        xj = jnp.asarray(np.asarray(x, dtype=np.float32))
        if self.reference_states is None:
            out = self._fn(xj)
        else:
            x_ref, u_ref = self.reference_at(t)
            out = self._fn(xj, jnp.asarray(x_ref, dtype=jnp.float32), jnp.asarray(u_ref, dtype=jnp.float32))
        u, u_nom, slack, used, lam, h_cone, height, in_b = (np.asarray(v) for v in out)
        return ControlOutput(
            u=u.astype(np.float64), u_nominal=u_nom.astype(np.float64), slack=float(slack), qp_used=bool(used),
            safeguard_lambda=float(lam), h_cone=float(h_cone), height=float(height), in_base_set=bool(in_b),
            latency_s=time.perf_counter() - started,
        )

    def recoverable(self, x: np.ndarray) -> bool:
        """Is x in C_N: does the backup reach the base set within its horizon without leaving the safe set?"""
        rec, _, _ = self._rec_fn(self._jnp.asarray(np.asarray(x, dtype=np.float32))[None])
        return bool(np.asarray(rec)[0])

    def margins(self, x: np.ndarray) -> tuple[float, float]:
        """(cone barrier, height above the pad) without running the filter."""
        lc = self.landing
        dx, dy = float(x[0]) - lc.pad_x, float(x[1]) - lc.pad_y
        tan_t = float(np.tan(np.radians(lc.cone_theta_deg)))
        zeta = float(x[2]) - lc.pad_z
        return lc.cone_r0 + tan_t * zeta - float(np.sqrt(dx * dx + dy * dy + lc.cone_eps**2)), zeta

    def warmup(self, iterations: int = 8) -> float:
        """Compile before the vehicle is in the air; returns the steady latency in seconds."""
        x0 = self.start_state if self.start_state is not None else np.r_[self.hover_position, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
        self.recoverable(x0)
        lat = [self.act(x0, 0.0).latency_s for _ in range(iterations)]
        steady = float(min(lat[1:]))
        msg = (f"landing runtime ready | backup {Path(self.cfg.backup).name} | nominal {self.cfg.nominal} -> "
               f"{np.round(self.nominal_goal(1e9), 3).tolist()}"
               + (f" (position error capped at {self.cfg.max_position_error:g} m)" if self.cfg.nominal == "lqr_target" and self.cfg.max_position_error > 0 else "")
               + f" | filter {'on' if self.cfg.use_filter else 'OFF'} "
               f"{self.filter_overrides} ({self.cfg.qp_solve_dtype} QP) | action box {self.action_low.tolist()} .. "
               f"{self.action_high.tolist()} | dt {self.dt:g} s | warm latency {steady * 1e3:.1f} ms")
        (self._log.info if self._log is not None else print)(msg)
        if steady > 0.8 * self.dt:
            warn = f"filter latency {steady * 1e3:.1f} ms is close to the {self.dt * 1e3:.0f} ms control period"
            (self._log.warn if self._log is not None else print)(warn)
        return steady


__all__ = ["ControlOutput", "LandingRuntime", "LandingRuntimeConfig", "NOMINALS", "TARGETS"]
