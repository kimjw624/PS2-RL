"""Online controller for the slow forward + loop reference: the PD controller or a trained policy.

One state in, one command out, for the 50 Hz control loop of the flight node. No JAX: the policy's
actor is a small MLP evaluated in numpy, and the observation, the PD controller and the residual
command come from ``ps2rl.envs.slow_track_common`` - the same functions the training environment
was checked against - so a flight sees exactly what the policy saw in training.

    controller "pd"      the environment's base controller (PD on the measured tracking error, the
                         disturbance observer as feed-forward, yaw held at the reference's)
    controller "policy"  the actor of a ``scripts/train_slow_track.py`` run (mean action, no sampling);
                         in a residual run the command is PD + the actor's correction

Disturbance observer, as in training: ``d_hat = lambda (v - xi)``, ``xi`` advanced every control step with
the commanded thrust along the measured body z (``observer="commanded"``, the training form) or with the
thrust measured by the autopilot (``"measured"``). Before the hand-over the flight logic runs it on the
measured thrust (``observe``), so it is warm when the controller starts, as in training.

Frames: the task frame of the reference (origin at the hand-over point, x east = the vehicle's heading,
which the flight logic holds at yaw 0, z up). The flight logic converts the vehicle's local frame to it.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import pickle
import time

import numpy as np

from ps2rl.deploy.landing_runtime import ControlOutput
from ps2rl.envs.slow_track_common import (SlowTrackEnvConfig, base_action_np, command_np, load_reference,
                                          observation_np)

CONTROLLERS = ("pd", "policy")
OBSERVERS = ("commanded", "measured", "off")
_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class SlowTrackRuntimeConfig:
    controller: str = "pd"            # "pd" | "policy"
    run: str = ""                     # policy: run directory of scripts/train_slow_track.py
    weights: str = "best"             # "best" | "final"
    reference_path: str = ""          # pd: reference .npz (default: the slow-loop reference); policy: the run's
    observer: str = "commanded"
    handover_height: float = 2.0      # height of the hand-over point (task-frame origin) above the take-off point [m]
    extra_time: float = 2.0           # the controller keeps tracking the reference's last point this long [s]


def _resolve(path: str, base: Path | None) -> str:
    if not path:
        return ""
    p = Path(path).expanduser()
    if not p.is_absolute() and base is not None and (base / p).exists():
        p = base / p
    elif not p.is_absolute() and (_ROOT / p).exists():
        p = _ROOT / p
    return str(p)


class _NumpyActor:
    """Mean action of the tanh-Gaussian actor (``ps2rl.utils.policy.actor_mean_action``) in numpy."""

    def __init__(self, params: dict, activation: str, low, high) -> None:
        self.layers = [(np.asarray(l["w"], np.float64), np.asarray(l["b"], np.float64)) for l in params["layers"]]
        if activation != "relu":
            raise ValueError(f"only relu actors are supported here, got {activation!r}")
        self.mid = 0.5 * (np.asarray(high) + np.asarray(low))
        self.half = 0.5 * (np.asarray(high) - np.asarray(low))

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        h = np.asarray(obs, np.float64)
        for i, (w, b) in enumerate(self.layers):
            h = h @ w + b
            if i != len(self.layers) - 1:
                h = np.maximum(h, 0.0)
        mean = np.split(h, 2)[0]
        return self.mid + self.half * np.tanh(mean)


class SlowTrackRuntime:
    def __init__(self, cfg: SlowTrackRuntimeConfig, *, ps2rl_path: str | None = None, logger=None) -> None:
        if cfg.controller not in CONTROLLERS:
            raise ValueError(f"controller must be one of {CONTROLLERS}, got {cfg.controller!r}")
        if cfg.observer not in OBSERVERS:
            raise ValueError(f"observer must be one of {OBSERVERS}, got {cfg.observer!r}")
        self.cfg = cfg
        self._log = logger
        base = Path(ps2rl_path).expanduser() if ps2rl_path else None
        self.actor = None
        if cfg.controller == "policy":
            run = Path(_resolve(cfg.run, base))
            if not (run / "configs.json").exists():
                raise FileNotFoundError(f"policy run '{run}' has no configs.json")
            rc = json.loads((run / "configs.json").read_text())
            self.env_cfg = SlowTrackEnvConfig(**rc["env"])
            with open(run / f"{cfg.weights}_weights.pkl", "rb") as f:
                params = pickle.load(f)["actor_params"]
            self.actor = _NumpyActor(params, rc["actor"].get("activation", "relu"), self.env_cfg.policy_low,
                                     self.env_cfg.policy_high)
            ref_path = _resolve(cfg.reference_path, base) or _resolve(self.env_cfg.reference_path, base)
            self.label = f"policy {run.name} ({cfg.weights}, {self.env_cfg.policy_mode})"
        else:
            self.env_cfg = SlowTrackEnvConfig()
            ref_path = _resolve(cfg.reference_path, base) or _resolve(self.env_cfg.reference_path, base)
            self.label = "PD (the training environment's base controller)"
        self.ref = load_reference(ref_path)
        e = self.env_cfg
        self.dt = float(e.dt)
        self.gravity = float(e.gravity)
        self.action_low = np.asarray(e.action_low, np.float64)
        self.action_high = np.asarray(e.action_high, np.float64)
        self.lam = float(e.observer_lambda)
        self.n_ref = len(self.ref["states"])
        self.episode_duration = float(self.ref["t"][-1]) + float(cfg.extra_time)
        # task frame: origin at the hand-over point; "pad" (the flight logic's anchor) is the take-off point below it
        self.pad = np.array([0.0, 0.0, -float(cfg.handover_height)])
        self.start_state = self.ref["states"][0].copy()
        self.hover_position = self.start_state[0:3].copy()
        self.target = self.ref["states"][-1, 0:3].copy()
        self.filter_overrides: dict = {}
        self.reset()
        self._info(f"slow-track runtime: {self.label}; reference {Path(ref_path).name} ({self.ref['t'][-1]:.1f} s), "
                   f"episode {self.episode_duration:.1f} s; command box {self.action_low.tolist()} .. {self.action_high.tolist()}; "
                   f"observer {cfg.observer} (lambda {self.lam:g})")

    def _info(self, msg: str) -> None:
        (self._log.info if self._log is not None else print)(msg)

    # ------------------------------------------------------------------ flight-logic interface
    def reset(self) -> None:
        self.xi = np.zeros(3)
        self.d_hat = np.zeros(3)
        self.u_prev = np.array([self.gravity, 0.0, 0.0, 0.0])   # previous command (observation)
        self._obs_ready = False
        self._a_last = self.gravity      # thrust that acted over the last interval (observer)
        self._zb_last: np.ndarray | None = None

    def margins(self, x: np.ndarray) -> tuple[float, float]:
        """(no obstacle: always 1, height above the take-off point) - the landing logic's start check."""
        return 1.0, float(x[2] - self.pad[2])

    def recoverable(self, x: np.ndarray) -> bool:
        return True

    def nominal_goal(self, t: float) -> np.ndarray:
        return self.ref["states"][self._k(t), 0:3]

    def _k(self, t: float) -> int:
        return int(np.clip(round(float(t) / self.dt), 0, self.n_ref - 1))

    @staticmethod
    def _zb(q: np.ndarray) -> np.ndarray:
        w, x, y, z = np.asarray(q, np.float64) / np.linalg.norm(q)
        return np.array([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)])

    def reset_observer(self, x: np.ndarray) -> None:
        self.xi = np.asarray(x[3:6], np.float64).copy()
        self.d_hat = np.zeros(3)
        self._obs_ready = True

    def observe(self, x: np.ndarray, a_thrust: float, dt: float) -> None:
        """Advance the observer by dt with a thrust acceleration along body z (measured before the hand-over)."""
        if not self._obs_ready:
            self.reset_observer(x)
            return
        if not (np.isfinite(a_thrust) and 0.0 < dt < 0.2):
            return
        self.xi = self.xi + dt * (-self.gravity * np.array([0.0, 0.0, 1.0]) + self._zb(x[6:10]) * a_thrust + self.d_hat)
        self.d_hat = self.lam * (np.asarray(x[3:6]) - self.xi)
        self._a_last, self._zb_last = float(a_thrust), self._zb(x[6:10])

    def act(self, x: np.ndarray, t: float, *, a_meas: float | None = None, omega_meas=None,
            dt_obs: float | None = None) -> ControlOutput:
        """Command for state x (task frame, z up) at time t since the hand-over."""
        t0 = time.perf_counter()
        x = np.asarray(x, np.float64)
        k = self._k(t)
        e = self.env_cfg
        # observer update with what acted over the last interval
        if self.cfg.observer == "off":
            self.d_hat = np.zeros(3)
        elif self._obs_ready and dt_obs is not None and 0.0 < dt_obs < 0.2:
            # "commanded": the command of the last step (training); "measured": the thrust the autopilot reports
            a = self._a_last if (self.cfg.observer == "commanded" or a_meas is None or not np.isfinite(a_meas)) else a_meas
            zb = self._zb_last if self._zb_last is not None else self._zb(x[6:10])
            self.xi = self.xi + dt_obs * (-self.gravity * np.array([0.0, 0.0, 1.0]) + zb * a + self.d_hat)
            self.d_hat = self.lam * (x[3:6] - self.xi)
        elif not self._obs_ready:
            self.reset_observer(x)
        p, v, q = x[0:3], x[3:6], x[6:10]
        u_pd = base_action_np(p, v, q, k, self.ref, self.d_hat, e)
        if self.actor is None:
            u = u_pd
        else:
            obs = observation_np(p, v, q, k, self.ref, self.d_hat, self.u_prev, e)
            u = command_np(self.actor(obs), p, v, q, k, self.ref, self.d_hat, e)
        u = np.clip(u, self.action_low, self.action_high)
        self.u_prev = u.copy()
        self._a_last, self._zb_last = float(u[0]), self._zb(q)
        rs = self.ref["states"][k]
        info = {"d_hat": self.d_hat.copy(), "k": k, "ref": rs[0:3].copy(), "pos_err": float(np.linalg.norm(p - rs[0:3])),
                "u_pd": u_pd}
        return ControlOutput(u=u, u_nominal=u_pd, slack=0.0, qp_used=True, safeguard_lambda=1.0, h_cone=1.0,
                             height=float(p[2] - self.pad[2]), in_base_set=False, latency_s=time.perf_counter() - t0, info=info)


__all__ = ["SlowTrackRuntime", "SlowTrackRuntimeConfig"]
