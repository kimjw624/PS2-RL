"""A stand-in vehicle for rehearsing a landing flight without PX4 or Gazebo.

It is deliberately not the model the policies were trained on. Where training assumes that a
commanded body rate and thrust act at once, this plant has

* first-order lags on the body rates and on the thrust,
* a transport delay between a command and its effect,
* a thrust-map error (scale and offset) and the vehicle's physical thrust range,
* noise and a bias on the state estimate (height in particular),
* a ground it cannot pass through (the impact speed is recorded),

plus a simple position controller, arming, offboard and land logic so that the whole flight
sequence of ``landing_flight.py`` can run against it. Everything is in a z-up local frame with a
z-up body frame; frame conversions belong to the ROS node and are not exercised here.

It answers "does the flight logic work, and how much model error does the filtered landing
tolerate"; it does not replace the simulator.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math

import numpy as np


@dataclass
class PlantConfig:
    dt_sim: float = 0.004               # integration step [s]
    gravity: float = 9.81
    # actuation
    tau_rate_xy: float = 0.05           # roll/pitch rate lag [s]
    tau_rate_z: float = 0.20            # yaw rate lag [s]
    tau_thrust: float = 0.03            # thrust lag [s]
    command_delay: float = 0.02         # command -> effect, including the controller's computation [s]
    thrust_scale: float = 1.0           # delivered thrust acceleration = scale * commanded + bias
    thrust_bias: float = 0.0
    thrust_accel_min: float = 0.0       # physical range of the vehicle [m/s^2]
    thrust_accel_max: float = 40.0
    # state estimate
    pos_noise_std: float = 0.005
    vel_noise_std: float = 0.02
    height_bias: float = 0.0            # estimate - truth [m]
    height_drift: float = 0.0           # growth of that bias [m/s]
    # the autopilot's own position control and mode logic (rough stand-ins)
    pos_kp: float = 1.0
    vel_kp: float = 3.0
    vel_ki: float = 1.5                 # integral on the velocity error, as the autopilot has
    att_kp: float = 6.0
    yaw_kp: float = 1.5
    max_speed_xy: float = 2.0
    max_speed_z: float = 1.0
    max_accel: float = 4.0
    land_speed: float = 0.5
    mode_delay: float = 0.2             # request -> armed / offboard / land [s]
    disarm_after_land: float = 1.0
    seed: int = 0
    # external translational disturbance (an acceleration, z-up local frame) applied while airborne:
    # constant part + a sinusoid along ext_dir with amplitude ext_amp at ext_freq_hz (all off by default)
    ext_accel: tuple = (0.0, 0.0, 0.0)
    ext_amp: float = 0.0
    ext_freq_hz: float = 0.0
    ext_dir: tuple = (1.0, 0.0, 0.0)


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def _rot(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


class FakeVehicle:
    def __init__(self, cfg: PlantConfig, *, position: np.ndarray | None = None) -> None:
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.t = 0.0
        self.p = np.zeros(3) if position is None else np.asarray(position, dtype=np.float64).copy()
        self.ground_z = float(self.p[2])
        self.v = np.zeros(3)
        self.q = np.array([1.0, 0.0, 0.0, 0.0])
        self.omega = np.zeros(3)
        self.a = 0.0                       # delivered thrust acceleration
        self.armed = self.offboard = False
        self.nav = "hold"                  # "hold", "offboard", "land"
        self.on_ground = True
        self._on_ground_since = 0.0
        self._pending: list[tuple[float, str]] = []
        self._cmd_queue: deque = deque()
        self._cmd = None                   # active command (after the delay)
        self._last_cmd_t = -1e9
        self._land_z = 0.0
        self._acc_int = np.zeros(3)        # integrator of the stand-in position controller
        self.impacts: list[tuple[float, float]] = []   # (time, vertical speed) at each ground contact
        self.min_height_airborne = math.inf
        self._took_off = False

    # ------------------------------------------------------------------ interface used by the flight loop
    def estimate(self) -> tuple[np.ndarray, np.ndarray, float]:
        """(state estimate [p, v, q], measured body rates, measured thrust acceleration)."""
        c = self.cfg
        p = self.p + self.rng.normal(0.0, c.pos_noise_std, 3)
        p[2] += c.height_bias + c.height_drift * self.t
        v = self.v + self.rng.normal(0.0, c.vel_noise_std, 3)
        return np.concatenate([p, v, self.q]), self.omega.copy(), float(self.a)

    @property
    def landed(self) -> bool:
        return self.on_ground and (not self.armed or self.t - self._on_ground_since > 0.5)

    def impacts_during(self, t0: float, t1: float | None) -> list[float]:
        """Vertical speeds of the ground contacts between t0 and t1 (None: until now)."""
        return [sp for (t, sp) in self.impacts if t >= t0 and (t1 is None or t <= t1)]

    def apply(self, cmd) -> None:
        """Take a ``landing_flight.Command``: mode requests and the current setpoint."""
        for name in cmd.requests:
            self._pending.append((self.t + self.cfg.mode_delay, name))
        self._cmd_queue.append((self.t + self.cfg.command_delay, cmd))
        self._last_cmd_t = self.t

    def advance(self, duration: float) -> None:
        n = max(1, int(round(duration / self.cfg.dt_sim)))
        for _ in range(n):
            self._substep()

    def external_accel(self) -> np.ndarray:
        """The external disturbance acting now (zero unless ext_* are set)."""
        c = self.cfg
        d = np.asarray(c.ext_accel, dtype=np.float64).copy()
        if c.ext_amp:
            n = np.asarray(c.ext_dir, dtype=np.float64)
            d += c.ext_amp * math.sin(2.0 * math.pi * c.ext_freq_hz * self.t) * n / max(float(np.linalg.norm(n)), 1e-9)
        return d

    # ------------------------------------------------------------------ internals
    def _modes(self) -> None:
        for item in [it for it in self._pending if it[0] <= self.t]:
            self._pending.remove(item)
            name = item[1]
            streaming = self.t - self._last_cmd_t < 0.5
            if name == "offboard" and streaming:
                self.offboard, self.nav = True, "offboard"
            elif name == "arm" and self.offboard:
                self.armed = True
            elif name == "land":
                self.offboard, self.nav = False, "land"
                self._land_z = float(self.p[2])
        if self.offboard and self.t - self._last_cmd_t > 0.5:   # setpoint stream lost
            self.offboard, self.nav = False, "land"
            self._land_z = float(self.p[2])

    def _position_control(self, p_sp: np.ndarray, yaw_sp: float) -> tuple[float, np.ndarray]:
        c = self.cfg
        v_sp = c.pos_kp * (p_sp - self.p)
        n_xy = float(np.linalg.norm(v_sp[0:2]))
        if n_xy > c.max_speed_xy:
            v_sp[0:2] *= c.max_speed_xy / n_xy
        v_sp[2] = float(np.clip(v_sp[2], -c.max_speed_z, c.max_speed_z))
        self._acc_int = np.clip(self._acc_int + c.vel_ki * (v_sp - self.v) * c.dt_sim, -c.max_accel, c.max_accel)
        acc = np.clip(c.vel_kp * (v_sp - self.v) + self._acc_int, -c.max_accel, c.max_accel)
        f = acc + np.array([0.0, 0.0, c.gravity])
        r = _rot(self.q)
        b = r[:, 2]
        a_cmd = float(np.dot(f, b))
        b_des = f / max(float(np.linalg.norm(f)), 1e-9)
        e_body = r.T @ np.cross(b, b_des)
        yaw = math.atan2(2.0 * (self.q[0] * self.q[3] + self.q[1] * self.q[2]), 1.0 - 2.0 * (self.q[2] ** 2 + self.q[3] ** 2))
        dyaw = (yaw_sp - yaw + math.pi) % (2.0 * math.pi) - math.pi
        return a_cmd, np.array([c.att_kp * e_body[0], c.att_kp * e_body[1], c.yaw_kp * dyaw])

    def _substep(self) -> None:
        c = self.cfg
        dt = c.dt_sim
        self._modes()
        while self._cmd_queue and self._cmd_queue[0][0] <= self.t:
            self._cmd = self._cmd_queue.popleft()[1]

        a_cmd, omega_cmd = 0.0, np.zeros(3)
        if self.armed:
            if self.nav == "land":
                self._land_z -= c.land_speed * dt
                a_cmd, omega_cmd = self._position_control(np.array([self.p[0], self.p[1], self._land_z]), 0.0)
                if self.on_ground:
                    a_cmd = min(a_cmd, 0.5 * c.gravity)
                    if self.t - self._on_ground_since > c.disarm_after_land:
                        self.armed = False
            elif self.nav == "offboard" and self._cmd is not None:
                if self._cmd.mode == "rates":
                    a_cmd, omega_cmd = float(self._cmd.a_cmd), np.asarray(self._cmd.rates, dtype=np.float64)
                else:
                    a_cmd, omega_cmd = self._position_control(np.asarray(self._cmd.position, dtype=np.float64), float(self._cmd.yaw))
            a_target = float(np.clip(c.thrust_scale * a_cmd + c.thrust_bias, c.thrust_accel_min, c.thrust_accel_max))
        else:
            a_target = 0.0
        self.a += (a_target - self.a) * min(1.0, dt / max(c.tau_thrust, dt))
        tau = np.array([c.tau_rate_xy, c.tau_rate_xy, c.tau_rate_z])
        self.omega += (omega_cmd - self.omega) * np.minimum(1.0, dt / np.maximum(tau, dt))

        r = _rot(self.q)
        acc = self.a * r[:, 2] - np.array([0.0, 0.0, c.gravity])
        if not self.on_ground:
            acc = acc + self.external_accel()
        if self.on_ground:
            if acc[2] > 0.0 and self.armed:           # lift-off
                self.on_ground = False
                self._took_off = True
            else:
                self.omega[:] = 0.0
                self.v[:] = 0.0
                self.t += dt
                return
        self.q = self.q + 0.5 * dt * _quat_mul(self.q, np.r_[0.0, self.omega])
        self.q /= np.linalg.norm(self.q)
        self.v = self.v + dt * acc
        self.p = self.p + dt * self.v
        self.min_height_airborne = min(self.min_height_airborne, float(self.p[2] - self.ground_z))
        if self.p[2] <= self.ground_z:
            self.impacts.append((self.t, float(-self.v[2])))
            self.p[2] = self.ground_z
            self.v[:] = 0.0
            self.on_ground = True
            self._on_ground_since = self.t
        self.t += dt


__all__ = ["FakeVehicle", "PlantConfig"]
