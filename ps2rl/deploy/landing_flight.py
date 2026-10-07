"""Flight logic of a filtered landing: what to command in each phase, when to hand over, when to stop.

    BOOT -> ARM -> TAKEOFF -> GOTO_START -> POLICY -> LAND -> DONE
                                              \\-> RECOVER -> LAND   (abort)

Up to GOTO_START the autopilot's own position controller flies the vehicle (position setpoints). In
POLICY the landing runtime commands body rates and thrust acceleration every control step. On
touchdown (or when the vehicle has come to rest just above the pad) the autopilot's land mode
takes over and disarms; a descent that reaches its time limit is also handed to the land mode where
it is. An abort (outside the cone or below the pad plane by a margin, tilt, bad state) first climbs
to a hold point above the pad.

This module is plain Python: no ROS, no PX4 types. The caller (the ROS node, or the offline
rehearsal) feeds it the state in a local z-up frame with a z-up body frame, and executes the
returned :class:`Command`. That keeps the logic testable without a simulator.

Frames
------
``local``  the autopilot's local frame, converted to east-north-up by the caller.
``pad``    the Phase-I checkpoint's frame, in which the pad centre is at the checkpoint's
           (pad_x, pad_y, pad_z). The two differ by a translation fixed before arming: the pad
           centre is the point where the vehicle stands on the ground, plus ``pad_offset``.
           Yaw is not re-aligned: the vehicle is flown to yaw 0 (body x along local x).
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, field
from enum import Enum, auto
import json
import math
from pathlib import Path
from typing import Callable

import numpy as np

from ps2rl.deploy.landing_runtime import ControlOutput, LandingRuntime


class Phase(Enum):
    BOOT = auto()
    ARM = auto()
    TAKEOFF = auto()
    GOTO_START = auto()
    POLICY = auto()
    RECOVER = auto()
    LAND = auto()
    DONE = auto()


@dataclass
class FlightConfig:
    # ---- geometry
    # start of the filtered descent, relative to the pad centre [m]; empty: the reference's first
    # state ("track_reference") or the base-set hover point above the pad ("lqr_target")
    start_position: tuple = ()
    # pad centre relative to the point where the vehicle stands before arming [m]; z > 0 puts the
    # filter's floor above the real ground (a margin against height-estimate error)
    pad_offset: tuple = (0.0, 0.0, 0.0)
    # ---- before the hand-over
    boot_time: float = 2.0             # setpoints must stream this long before offboard is requested [s]
    ground_spread_tol: float = 0.10    # standing still: position samples over boot_time stay within this [m]
    max_prearm_tilt_deg: float = 30.0  # refuse to arm a vehicle lying on its side
    takeoff_tol: float = 0.3           # reached the start height [m]
    start_pos_tol: float = 0.15        # hand-over gates: position [m], speed [m/s], tilt [deg]
    start_speed_tol: float = 0.15
    start_tilt_tol_deg: float = 5.0
    settle_time: float = 2.0           # gates must hold this long [s]
    start_timeout: float = 40.0        # give up and land if the hand-over has not happened by then [s]
    # ---- end of the filtered descent
    zeta_touch: float = 0.02           # touchdown: height above the pad <= this [m] ...
    v_touch: float = 0.3               # ... at speed <= this [m/s] ...
    touch_steps: int = 3               # ... for this many control steps in a row
    rest_speed: float = 0.10           # at rest (the nominal controller has converged, wherever that is):
    rest_time: float = 1.5             # speed <= rest_speed for this long [s]
    policy_timeout: float = 15.0       # longest filtered descent; then the autopilot lands from where the vehicle is [s]
    # ---- aborts during the filtered descent
    cone_abort_margin: float = 0.3     # outside the cone by more than this [m]
    floor_abort_depth: float = 0.3     # below the pad plane by more than this [m]
    tilt_abort_deg: float = 75.0
    recover_height: float = 1.5        # hold this height above the pad after an abort [m] ...
    recover_time: float = 4.0          # ... for this long, then land [s]
    # ---- records
    log_dir: str = ""                  # flight.csv and summary.json go here (empty: no files)


@dataclass
class Command:
    """What the caller must send to the autopilot now."""
    mode: str = "position"                         # "position" or "rates"
    position: np.ndarray | None = None             # local frame [m] (mode "position")
    yaw: float = 0.0                               # about z-up, 0 = body x along local x
    rates: np.ndarray | None = None                # body rates, z-up body frame [rad/s] (mode "rates")
    a_cmd: float = 0.0                             # thrust acceleration along body z [m/s^2] (mode "rates")
    requests: list = field(default_factory=list)   # any of "offboard", "arm", "land"


CSV_COLUMNS = (
    ["t_flight", "phase", "t_policy", "px", "py", "pz", "vx", "vy", "vz", "qw", "qx", "qy", "qz",
     "goal_x", "goal_y", "goal_z", "a_cmd", "wx", "wy", "wz", "a_nom", "wx_nom", "wy_nom", "wz_nom",
     "wx_meas", "wy_meas", "wz_meas", "a_meas", "thrust_norm", "saturated", "slack", "qp_used",
     "safeguard_lambda", "h_cone", "height", "in_base_set", "latency_ms"]
)


def tilt_deg(q: np.ndarray) -> float:
    """Angle between body z and world z for a scalar-first quaternion."""
    cz = 1.0 - 2.0 * (q[1] * q[1] + q[2] * q[2]) / max(float(np.dot(q, q)), 1e-12)
    return math.degrees(math.acos(min(1.0, max(-1.0, cz))))


class LandingFlight:
    def __init__(self, runtime: LandingRuntime, cfg: FlightConfig, *, logger=None,
                 thrust_map: Callable[[float], tuple[float, bool]] | None = None) -> None:
        self.rt, self.cfg = runtime, cfg
        self._log = logger
        # a_cmd -> (normalised thrust, saturated); only recorded here, the caller applies it
        self._thrust_map = thrust_map
        self.phase = Phase.BOOT
        self._t_first: float | None = None
        self._t_phase = 0.0
        self._last_request_t = -1e9
        self._ground_samples: list[tuple[float, np.ndarray]] = []
        self.ground: np.ndarray | None = None        # where the vehicle stood, local frame
        self.shift: np.ndarray | None = None         # local position = pad-frame position + shift
        start = np.asarray(cfg.start_position, dtype=np.float64)
        if start.size == 0:
            start = (runtime.start_state[0:3] if runtime.start_state is not None else runtime.hover_position) - runtime.pad
        if start.shape != (3,):
            raise ValueError(f"start_position needs three numbers, got {cfg.start_position}")
        self.start_pad = runtime.pad + start          # start in the pad frame
        if np.asarray(cfg.pad_offset).shape != (3,):
            raise ValueError(f"pad_offset needs three numbers, got {cfg.pad_offset}")
        h, z = runtime.margins(np.r_[self.start_pad, np.zeros(3), 1.0, 0.0, 0.0, 0.0])
        if h <= 0.0 or z <= 0.0:
            raise ValueError(f"start {start.tolist()} is outside the safe set (cone margin {h:.2f} m, height {z:.2f} m)")
        if not runtime.recoverable(np.r_[self.start_pad, np.zeros(3), 1.0, 0.0, 0.0, 0.0]):
            raise ValueError(f"hovering at the start {start.tolist()} is not recoverable by the backup; choose a start further inside")
        self._gate_since: float | None = None
        self._gate_note_t = -1e9
        self._t0_policy = 0.0
        self._touch_count = 0
        self._rest_since: float | None = None
        self._hold_local: np.ndarray | None = None
        self._land_sent_t: float | None = None
        self.armed = self.offboard = False
        self.preflight_ok = True
        self.landed: bool | None = None               # autopilot's land detector, if the caller has it
        self.landing: bool | None = None              # autopilot is in its land mode, if the caller knows
        self.events: list[dict] = []                  # things the caller wants on record (see add_event)
        self.outcome = ""
        self.stats: dict = {}
        self._rows: list[list] = []
        self._rec: dict[str, list] = {k: [] for k in ("h_cone", "height", "slack", "latency", "lam", "used", "sat",
                                                      "rate_err", "a_err", "speed", "radial", "du")}
        self._handover: dict = {}
        self._csv = self._csv_file = None
        if cfg.log_dir:
            Path(cfg.log_dir).mkdir(parents=True, exist_ok=True)
            self._csv_file = open(Path(cfg.log_dir) / "flight.csv", "w", newline="", encoding="utf-8")
            self._csv = csv.writer(self._csv_file)
            self._csv.writerow(CSV_COLUMNS)
        self._info(f"start of the filtered descent: {np.round(start, 3).tolist()} m from the pad centre; "
                   f"pad centre {list(cfg.pad_offset)} m from the ground point")

    # ------------------------------------------------------------------ small helpers
    def _info(self, msg: str) -> None:
        (self._log.info if self._log is not None else print)(msg)

    def _warn(self, msg: str) -> None:
        if self._log is not None:
            self._log.warn(msg)
        else:
            print("WARNING: " + msg)

    def _set_phase(self, phase: Phase, now: float) -> None:
        if phase is not self.phase:
            self._info(f"phase: {self.phase.name} -> {phase.name}")
            self.phase, self._t_phase = phase, now

    def to_pad(self, x_local: np.ndarray) -> np.ndarray:
        x = np.array(x_local, dtype=np.float64, copy=True)
        x[0:3] -= self.shift
        return x

    def to_local(self, p_pad: np.ndarray) -> np.ndarray:
        return np.asarray(p_pad, dtype=np.float64) + self.shift

    def update_status(self, *, armed: bool, offboard: bool, preflight_ok: bool = True, landed: bool | None = None,
                      landing: bool | None = None) -> None:
        self.armed, self.offboard, self.preflight_ok, self.landed = bool(armed), bool(offboard), bool(preflight_ok), landed
        self.landing = landing

    def add_event(self, now: float, text: str) -> None:
        """Record something the caller noticed (an estimator reset, say); it goes into the summary."""
        t = None if self._t_first is None else float(now - self._t_first)
        self.events.append({"t_flight": t, "phase": self.phase.name, "event": text})
        self._warn(f"{text} (phase {self.phase.name})")

    def _request(self, cmd: Command, name: str, now: float, period: float = 1.0) -> None:
        """Requests are repeated at most once per ``period``: the autopilot acts on the first it accepts."""
        if now - self._last_request_t >= period:
            self._last_request_t = now
            cmd.requests.append(name)

    # ------------------------------------------------------------------ one step
    def step(self, now: float, x_local: np.ndarray, *, omega_meas: np.ndarray | None = None,
             a_meas: float | None = None) -> Command:
        """Advance the flight by one call. ``now`` is the vehicle clock [s] (simulation time in SITL)."""
        x_local = np.asarray(x_local, dtype=np.float64)
        if self._t_first is None:
            self._t_first, self._t_phase = now, now
        handler = {
            Phase.BOOT: self._boot, Phase.ARM: self._arm, Phase.TAKEOFF: self._takeoff,
            Phase.GOTO_START: self._goto_start, Phase.RECOVER: self._recover,
            Phase.LAND: self._land, Phase.DONE: self._done,
        }.get(self.phase)
        if self.phase is Phase.POLICY:
            return self._policy(now, x_local, omega_meas, a_meas)
        phase = self.phase  # the row belongs to the phase that produced this command
        cmd = handler(now, x_local)
        if self.shift is not None:
            self._log_row(now, self.to_pad(x_local), None, None, None, None, None, phase=phase)
        return cmd

    # ------------------------------------------------------------------ phases before the hand-over
    def _hold(self, p_local: np.ndarray) -> Command:
        return Command(mode="position", position=np.asarray(p_local, dtype=np.float64), yaw=0.0)

    def _boot(self, now: float, x: np.ndarray) -> Command:
        c = self.cfg
        if self.armed:
            self._ground_samples.clear()
        else:
            self._ground_samples.append((now, x[0:3].copy()))
            self._ground_samples = [s for s in self._ground_samples if now - s[0] <= c.boot_time]
        pts = np.asarray([s[1] for s in self._ground_samples]) if self._ground_samples else np.zeros((0, 3))
        full = len(pts) >= 10 and now - self._ground_samples[0][0] >= 0.9 * c.boot_time
        spread = float(np.max(pts.max(axis=0) - pts.min(axis=0))) if len(pts) else float("inf")
        if now - self._t_phase >= c.boot_time and full and spread <= c.ground_spread_tol:
            self.ground = np.median(pts, axis=0)
            pad_local = self.ground + np.asarray(self.cfg.pad_offset, dtype=np.float64)
            self.shift = pad_local - self.rt.pad
            self._info(f"ground point {np.round(self.ground, 3).tolist()} (local frame, {len(self._ground_samples)} samples); "
                       f"pad centre at {np.round(pad_local, 3).tolist()}")
            self._set_phase(Phase.ARM, now)
        elif now - self._t_phase >= c.boot_time + 5.0 and now - self._gate_note_t > 5.0:
            self._gate_note_t = now
            self._warn("waiting for the vehicle to stand still and disarmed to take the ground point "
                       f"(position spread {spread:.2f} m over {c.boot_time:g} s, armed {self.armed})")
        return self._hold(x[0:3])

    def _takeoff_point(self) -> np.ndarray:
        p = self.ground.copy()
        p[2] = self.to_local(self.start_pad)[2]
        return p

    def _arm(self, now: float, x: np.ndarray) -> Command:
        cmd = self._hold(self._takeoff_point())  # a setpoint must be streaming before the mode switch
        tilt = tilt_deg(x[6:10])
        if tilt > self.cfg.max_prearm_tilt_deg:
            if now - self._gate_note_t > 5.0:
                self._gate_note_t = now
                self._warn(f"not arming: the vehicle is tilted {tilt:.0f} deg (limit {self.cfg.max_prearm_tilt_deg:.0f}); "
                           "restart the simulation to respawn it, then start this controller again")
            return cmd
        if not self.offboard:
            self._request(cmd, "offboard", now)   # before the pre-flight gate: that check is per mode
        elif not self.armed and self.preflight_ok:
            self._request(cmd, "arm", now)
        if self.armed and self.offboard:
            self._set_phase(Phase.TAKEOFF, now)
        elif now - self._t_phase > 10.0 and now - self._gate_note_t > 5.0:
            self._gate_note_t = now
            self._warn(f"still waiting after {now - self._t_phase:.0f} s: offboard {self.offboard}, armed {self.armed}, "
                       f"pre-flight checks {'pass' if self.preflight_ok else 'FAIL'}")
        return cmd

    def _lost_control(self, now: float) -> bool:
        if self.armed and self.offboard:
            return False
        self._finish(now, "aborted: lost arming or offboard mode before the hand-over", Phase.DONE)
        return True

    def _takeoff(self, now: float, x: np.ndarray) -> Command:
        target = self._takeoff_point()
        if not self._lost_control(now) and float(np.linalg.norm(x[0:3] - target)) < self.cfg.takeoff_tol:
            self._set_phase(Phase.GOTO_START, now)
        return self._hold(target)

    def _gates(self, x: np.ndarray) -> tuple[bool, str]:
        c = self.cfg
        err = float(np.linalg.norm(x[0:3] - self.to_local(self.start_pad)))
        speed, tilt = float(np.linalg.norm(x[3:6])), tilt_deg(x[6:10])
        ok = err < c.start_pos_tol and speed < c.start_speed_tol and tilt < c.start_tilt_tol_deg
        return ok, f"position {err:.2f}/{c.start_pos_tol:.2f} m, speed {speed:.2f}/{c.start_speed_tol:.2f} m/s, tilt {tilt:.1f}/{c.start_tilt_tol_deg:.1f} deg"

    def _goto_start(self, now: float, x: np.ndarray) -> Command:
        """Fly to the start, hold the gates for ``settle_time``, check recoverability, hand over."""
        target = self.to_local(self.start_pad)
        cmd = self._hold(target)
        if self._lost_control(now):
            return cmd
        ok, note = self._gates(x)
        waited = now - self._t_phase
        if not ok:
            self._gate_since = None
            if waited > self.cfg.start_timeout:
                self._hold_local = target
                self._finish(now, f"aborted: start not reached ({note})", Phase.LAND)
            return cmd
        if self._gate_since is None:
            self._gate_since = now
            if now - self._gate_note_t > 5.0:
                self._gate_note_t = now
                self._info(f"at the start ({note}); holding for {self.cfg.settle_time:g} s")
        if now - self._gate_since < self.cfg.settle_time:
            return cmd
        x_pad = self.to_pad(x)
        if not self.rt.recoverable(x_pad):
            if now - self._gate_note_t > 2.0:
                self._gate_note_t = now
                self._warn(f"the state at the start is not recoverable by the backup ({note}); waiting")
            if waited > self.cfg.start_timeout:
                self._hold_local = target
                self._finish(now, "aborted: the start state never became recoverable", Phase.LAND)
            return cmd
        h, z = self.rt.margins(x_pad)
        self._handover = {"position_error_m": float(np.linalg.norm(x[0:3] - target)), "speed_m_s": float(np.linalg.norm(x[3:6])),
                          "tilt_deg": tilt_deg(x[6:10]), "h_cone_m": h, "height_m": z}
        self._info(f"hand-over to the filtered controller: {note}; cone margin {h:.2f} m, height {z:.2f} m")
        self._t0_policy = now
        self._set_phase(Phase.POLICY, now)
        return cmd

    # ------------------------------------------------------------------ the filtered descent
    def _policy(self, now: float, x_local: np.ndarray, omega_meas, a_meas) -> Command:
        c = self.cfg
        x = self.to_pad(x_local)
        t = now - self._t0_policy
        if not (self.armed and self.offboard):
            if self.landed and not self.armed and self._rec["height"]:
                # the autopilot's land detector saw the ground and disarmed before the touchdown test did
                self._finish(now, "touchdown (the autopilot detected the landing and disarmed)", Phase.DONE)
            else:
                self._finish(now, "aborted: lost arming or offboard mode", Phase.DONE)
            return self._hold(x_local[0:3])
        if not np.all(np.isfinite(x)):
            return self._abort(now, x_local, "non-finite state")
        out = self.rt.act(x, t)
        if not np.all(np.isfinite(out.u)):
            return self._abort(now, x_local, "non-finite action")
        thrust_norm, saturated = self._thrust_map(float(out.u[0])) if self._thrust_map is not None else (float("nan"), False)
        self._log_row(now, x, t, out, omega_meas, a_meas, (thrust_norm, saturated), phase=Phase.POLICY)
        r = self._rec
        speed = float(np.linalg.norm(x[3:6]))
        r["h_cone"].append(out.h_cone); r["height"].append(out.height); r["slack"].append(out.slack)
        r["latency"].append(out.latency_s); r["lam"].append(out.safeguard_lambda); r["used"].append(out.qp_used)
        r["sat"].append(bool(saturated)); r["speed"].append(speed)
        r["radial"].append(float(np.linalg.norm(x[0:2] - self.rt.pad[0:2]))); r["du"].append(float(np.linalg.norm(out.u - out.u_nominal)))
        if omega_meas is not None and np.all(np.isfinite(omega_meas)):
            r["rate_err"].append(np.asarray(out.u[1:4]) - np.asarray(omega_meas, dtype=np.float64))
        if a_meas is not None and np.isfinite(a_meas):
            r["a_err"].append(float(out.u[0]) - float(a_meas))

        tilt = tilt_deg(x[6:10])
        if out.h_cone < -c.cone_abort_margin:
            return self._abort(now, x_local, f"outside the cone by {-out.h_cone:.2f} m")
        if out.height < -c.floor_abort_depth:
            return self._abort(now, x_local, f"{-out.height:.2f} m below the pad plane")
        if tilt > c.tilt_abort_deg:
            return self._abort(now, x_local, f"tilt {tilt:.0f} deg")

        touching = out.height <= c.zeta_touch and speed <= c.v_touch
        self._touch_count = self._touch_count + 1 if touching else 0
        resting = speed <= c.rest_speed
        self._rest_since = (self._rest_since if self._rest_since is not None else now) if resting else None
        if self._touch_count >= c.touch_steps:
            self._finish(now, "touchdown", Phase.LAND)
        elif self.landed:
            self._finish(now, "touchdown (reported by the autopilot's land detector)", Phase.LAND)
        elif self._rest_since is not None and now - self._rest_since >= c.rest_time:
            self._finish(now, f"at rest {out.height * 100:.1f} cm above the pad plane (no touchdown)", Phase.LAND)
        elif t >= c.policy_timeout:
            # still inside the safe set, just not done: the land mode descends from here
            self._hold_local = x_local[0:3].copy()
            self._finish(now, f"time-out after {t:.1f} s at height {out.height:.2f} m", Phase.LAND)
        return Command(mode="rates", rates=np.asarray(out.u[1:4], dtype=np.float64), a_cmd=float(out.u[0]))

    def abort(self, now: float, x_local: np.ndarray, reason: str) -> Command:
        """For the caller: stop the filtered descent (stale state, an exception) and recover."""
        if self.phase is Phase.POLICY:
            return self._abort(now, np.asarray(x_local, dtype=np.float64), reason)
        return self._hold(np.asarray(x_local, dtype=np.float64)[0:3])

    def end(self, now: float, x_local: np.ndarray | None, outcome: str) -> None:
        """For the caller: the flight is over for a reason outside this logic (the caller's own failure,
        a restarted autopilot). Records the outcome and writes the summary; commands nothing."""
        if self.phase is Phase.DONE:
            return
        if not self.outcome:
            self.outcome = outcome
            self.stats = self._policy_stats(now)
            self._warn(f"flight ended: {outcome}")
        self._set_phase(Phase.DONE, now)
        self._write_summary(now, x_local)

    def _abort(self, now: float, x_local: np.ndarray, reason: str) -> Command:
        self._hold_local = np.r_[x_local[0:2], max(float(x_local[2]), self.to_local(self.rt.pad)[2] + self.cfg.recover_height)]
        self._finish(now, f"aborted: {reason}", Phase.RECOVER)
        return self._hold(self._hold_local)

    # ------------------------------------------------------------------ after the descent
    def _recover(self, now: float, x: np.ndarray) -> Command:
        if now - self._t_phase > self.cfg.recover_time:
            self._set_phase(Phase.LAND, now)
        return self._hold(self._hold_local)

    def _land(self, now: float, x: np.ndarray) -> Command:
        if self._hold_local is None:
            self._hold_local = x[0:3].copy()
        cmd = self._hold(self._hold_local)
        # repeat the request until the autopilot is in its land mode (or, if the caller cannot tell, has left offboard)
        waiting = (not self.landing) if self.landing is not None else self.offboard
        if self._land_sent_t is None or (waiting and self.armed and now - self._land_sent_t > 1.0):
            self._land_sent_t = now
            cmd.requests.append("land")
        if not self.armed:
            self._set_phase(Phase.DONE, now)
            self._write_summary(now, x)
        return cmd

    def _done(self, now: float, x: np.ndarray) -> Command:
        return self._hold(x[0:3])

    # ------------------------------------------------------------------ records
    def _finish(self, now: float, outcome: str, next_phase: Phase) -> None:
        if not self.outcome:
            self.outcome = outcome
            self.stats = self._policy_stats(now)
            (self._warn if outcome.startswith(("aborted", "time-out")) else self._info)(f"filtered descent ended: {outcome}")
            for line in self._stat_lines():
                self._info(line)
        self._set_phase(next_phase, now)
        if next_phase is Phase.DONE:
            self._write_summary(now, None)

    def _policy_stats(self, now: float) -> dict:
        r = self._rec
        n = len(r["height"])
        s: dict = {"outcome": self.outcome, "steps": n, "duration_s": float(now - self._t0_policy) if n else 0.0,
                   "expected_steps": int(round((now - self._t0_policy) / self.rt.dt)) + 1 if n else 0, "handover": self._handover}
        if not n:
            return s
        h, z, lat = np.asarray(r["h_cone"]), np.asarray(r["height"]), np.asarray(r["latency"]) * 1e3
        used, lam = np.asarray(r["used"], dtype=bool), np.asarray(r["lam"])
        first_low = int(np.argmax(z <= self.cfg.zeta_touch)) if np.any(z <= self.cfg.zeta_touch) else -1
        s.update({
            "min_h_cone_m": float(h.min()), "min_height_m": float(z.min()), "final_height_m": float(z[-1]),
            "final_radial_m": float(r["radial"][-1]), "final_speed_m_s": float(r["speed"][-1]),
            "time_to_touch_height_s": float(first_low * self.rt.dt) if first_low >= 0 else None,
            "speed_at_touch_height_m_s": float(r["speed"][first_low]) if first_low >= 0 else None,
            "max_slack": float(np.max(r["slack"])), "qp_fallback_steps": int((~used).sum()),
            "safeguard_steps": int((lam < 1.0).sum()), "filter_active_fraction": float(np.mean(np.asarray(r["du"]) > 1e-3)),
            "thrust_saturated_steps": int(np.sum(r["sat"])),
            "latency_ms": {"median": float(np.median(lat)), "p95": float(np.percentile(lat, 95)), "max": float(lat.max())},
        })
        if r["rate_err"]:
            e = np.asarray(r["rate_err"])
            s["rate_tracking_rms_rad_s"] = np.sqrt(np.mean(e * e, axis=0)).tolist()
        if r["a_err"]:
            e = np.asarray(r["a_err"])
            s["thrust_tracking_rms_m_s2"] = float(np.sqrt(np.mean(e * e)))
            s["thrust_tracking_mean_m_s2"] = float(np.mean(e))
        return s

    def _stat_lines(self) -> list[str]:
        s = self.stats
        if not s.get("steps"):
            return []
        lines = [
            f"  {s['steps']} control steps in {s['duration_s']:.2f} s (expected {s['expected_steps']}); cone margin min {s['min_h_cone_m']:+.3f} m, "
            f"height min {s['min_height_m']:+.3f} m, final {s['final_height_m']:+.3f} m at {s['final_radial_m']:.2f} m from the pad centre",
            f"  filter: active on {100 * s['filter_active_fraction']:.0f} % of steps, largest slack {s['max_slack']:.1e}, "
            f"QP fallbacks {s['qp_fallback_steps']}, safeguard blends {s['safeguard_steps']}, thrust saturated {s['thrust_saturated_steps']} steps; "
            f"latency median {s['latency_ms']['median']:.1f} / p95 {s['latency_ms']['p95']:.1f} / max {s['latency_ms']['max']:.1f} ms",
        ]
        if s.get("time_to_touch_height_s") is not None:
            lines.append(f"  reached {self.cfg.zeta_touch * 100:.0f} cm after {s['time_to_touch_height_s']:.2f} s at {s['speed_at_touch_height_m_s']:.2f} m/s")
        if "rate_tracking_rms_rad_s" in s:
            e = s["rate_tracking_rms_rad_s"]
            lines.append(f"  rate tracking error (commanded - measured, RMS): roll {e[0]:.2f}, pitch {e[1]:.2f}, yaw {e[2]:.2f} rad/s"
                         + (f"; thrust {s['thrust_tracking_rms_m_s2']:.2f} m/s^2 RMS, mean {s['thrust_tracking_mean_m_s2']:+.2f}" if "thrust_tracking_rms_m_s2" in s else ""))
        return lines

    def _log_row(self, now, x, t, out: ControlOutput | None, omega_meas, a_meas, thrust=None, phase: Phase | None = None) -> None:
        if self._csv is None:
            return
        nan3 = [float("nan")] * 3
        row = [f"{now - self._t_first:.4f}", (phase or self.phase).name, "" if t is None else f"{t:.4f}"] + [f"{v:.6f}" for v in x]
        if out is None:
            row += [""] * (len(CSV_COLUMNS) - len(row))
        else:
            om = nan3 if omega_meas is None else list(np.asarray(omega_meas, dtype=np.float64))
            row += [f"{v:.6f}" for v in self.rt.nominal_goal(t)] + [f"{v:.6f}" for v in out.u] + [f"{v:.6f}" for v in out.u_nominal]
            row += [f"{v:.6f}" for v in om] + [f"{float('nan') if a_meas is None else a_meas:.6f}"]
            row += [f"{thrust[0]:.6f}", int(bool(thrust[1])), f"{out.slack:.6e}", int(out.qp_used), f"{out.safeguard_lambda:.3f}",
                    f"{out.h_cone:.6f}", f"{out.height:.6f}", int(out.in_base_set), f"{out.latency_s * 1e3:.3f}"]
        self._csv.writerow(row)

    def _write_summary(self, now: float, x_local: np.ndarray | None) -> None:
        lc = self.rt.landing
        summary = {
            "outcome": self.outcome or "ended before the filtered descent",
            "policy": self.stats,
            "flight_duration_s": float(now - self._t_first) if self._t_first is not None else 0.0,
            "events": self.events,
            "ground_point_local": None if self.ground is None else self.ground.tolist(),
            "pad_centre_local": None if self.shift is None else self.to_local(self.rt.pad).tolist(),
            "final_position_pad_frame": None if (x_local is None or self.shift is None) else self.to_pad(x_local)[0:3].tolist(),
            "start_pad_frame": self.start_pad.tolist(),
            "runtime": {**asdict(self.rt.cfg), "filter_overrides": {k: (list(v) if isinstance(v, tuple) else v) for k, v in self.rt.filter_overrides.items()},
                        "target": self.rt.nominal_goal(1e9).tolist(), "dt": self.rt.dt,
                        "action_low": self.rt.action_low.tolist(), "action_high": self.rt.action_high.tolist()},
            "geometry": {"pad": self.rt.pad.tolist(), "cone_r0": lc.cone_r0, "cone_theta_deg": lc.cone_theta_deg, "cone_eps": lc.cone_eps,
                         "z_des": lc.z_des, "floor_constraint": bool(lc.floor_constraint)},
            "flight_config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(self.cfg).items()},
        }
        self.summary = summary
        if self._csv_file is not None:
            self._csv_file.flush()
        if self.cfg.log_dir:
            (Path(self.cfg.log_dir) / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
            self._info(f"flight complete ({summary['outcome']}); records in {self.cfg.log_dir}")
        else:
            self._info(f"flight complete ({summary['outcome']})")

    def close(self) -> None:
        if self._csv_file is not None:
            self._csv_file.close()
            self._csv_file = None


__all__ = ["CSV_COLUMNS", "Command", "FlightConfig", "LandingFlight", "Phase", "tilt_deg"]
