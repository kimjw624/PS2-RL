"""Flight logic for the slow forward + loop reference: take off, hand over, track, hover, land.

    BOOT -> ARM -> TAKEOFF -> GOTO_START -> POLICY -> RECOVER (hover) -> LAND -> DONE

* Up to GOTO_START PX4's position controller flies: it climbs to ``handover_height`` above the take-off
  point and holds there at yaw 0 (body x east) until the hand-over gates (position, speed, tilt) have
  held for ``settle_time`` - the sequence and gates of the landing flight (``landing_flight``), which this
  class extends.
* POLICY: the controller (PD or the trained policy, ``slow_track_runtime``) commands body rates and
  collective thrust every control step, for the whole reference + ``extra_time``. The task frame is the
  reference's: origin at the hand-over point, x east, z up.
* At the end of the reference the vehicle hovers where it is (PX4 position hold) for ``end_hold_time``,
  then PX4's land mode lands it.
* Abort - immediately back to PX4 position hold where the vehicle is, then land after ``recover_time``:
  tracking error above ``max_track_error``, tilt above ``tilt_abort_deg``, speed above ``max_speed``,
  lower than ``min_height`` above the take-off point, a non-finite state or command, or loss of arming or
  offboard mode. The node also aborts on stale odometry or an exception in the controller.
* The disturbance observer runs on the measured thrust from take-off on (``observer_warmup``), so it is warm
  at the hand-over, as in training.

Plain Python: no ROS, no PX4 types. ``scripts/rehearse_slow_track_flight.py`` flies it against a stand-in vehicle.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path

import numpy as np

try:   # its own module on the landing-runway-strike branch, inside the cone flight module on main
    from ps2rl.deploy.rate_response import fit_rate_response
except ImportError:  # pragma: no cover
    from ps2rl.deploy.cone_obstacle_flight import fit_rate_response
from ps2rl.deploy.landing_flight import Command, FlightConfig, LandingFlight, Phase, tilt_deg
from ps2rl.deploy.slow_track_runtime import SlowTrackRuntime

SLOW_CSV_COLUMNS = (
    ["t_flight", "phase", "t_policy", "px", "py", "pz", "vx", "vy", "vz", "qw", "qx", "qy", "qz",
     "ref_x", "ref_y", "ref_z", "pos_err", "a_cmd", "wx", "wy", "wz", "a_pd", "wx_pd", "wy_pd", "wz_pd",
     "wx_meas", "wy_meas", "wz_meas", "a_meas", "thrust_norm", "saturated", "dhat_x", "dhat_y", "dhat_z", "latency_ms"]
)


@dataclass
class SlowFlightConfig(FlightConfig):
    policy_duration: float = 0.0       # length of the controller phase [s]; 0 = reference + extra_time
    end_hold_time: float = 3.0         # hover where the reference ended, then land [s]
    max_track_error: float = 0.8       # abort: distance to the reference point [m]
    max_speed: float = 1.5             # abort: speed [m/s] (the reference flies at 0.4)
    min_height: float = 0.7            # abort: height above the take-off point [m]
    observer_warmup: bool = True       # run the disturbance observer on the measured thrust before the hand-over
    settle_time: float = 8.0           # hold at the hand-over point this long before handing over [s]: PX4's vertical
                                       # velocity estimate was still off by up to 0.15 m/s ~10 s after take-off
    tilt_abort_deg: float = 35.0       # the reference needs ~1 deg
    recover_time: float = 3.0          # after an abort: hover this long where the vehicle is, then land [s]


class SlowTrackFlight(LandingFlight):
    def __init__(self, runtime: SlowTrackRuntime, cfg: SlowFlightConfig, *, logger=None, thrust_map=None) -> None:
        runtime.reset()
        super().__init__(runtime, replace(cfg, log_dir=""), logger=logger, thrust_map=thrust_map)
        self.cfg = cfg
        self.rt: SlowTrackRuntime = runtime
        self._duration = float(cfg.policy_duration) if cfg.policy_duration > 0 else float(runtime.episode_duration)
        self._hold_time = float(cfg.recover_time)
        self._t_obs: float | None = None
        self._rec.update({"pos_err": [], "dhat": [], "tilt": [], "u": [], "u_pd": [], "omega": [], "t": [], "x": [], "ref": [],
                          "vz": []})
        if cfg.log_dir:
            Path(cfg.log_dir).mkdir(parents=True, exist_ok=True)
            self._csv_file = open(Path(cfg.log_dir) / "flight.csv", "w", newline="", encoding="utf-8")
            self._csv = csv.writer(self._csv_file)
            self._csv.writerow(SLOW_CSV_COLUMNS)
        self._info(f"slow-track flight: {runtime.label}; {self._duration:.1f} s from the hand-over point "
                   f"{runtime.cfg.handover_height:g} m above the take-off point, then hover {cfg.end_hold_time:g} s and land. "
                   f"Aborts: error > {cfg.max_track_error:g} m, tilt > {cfg.tilt_abort_deg:g} deg, speed > {cfg.max_speed:g} m/s, "
                   f"height < {cfg.min_height:g} m")

    # ------------------------------------------------------------------ the observer before the hand-over
    def step(self, now: float, x_local: np.ndarray, *, omega_meas=None, a_meas=None) -> Command:
        if (self.cfg.observer_warmup and self.phase in (Phase.TAKEOFF, Phase.GOTO_START) and self.shift is not None
                and a_meas is not None and np.isfinite(a_meas) and np.all(np.isfinite(x_local))):
            x = self.to_pad(np.asarray(x_local, dtype=np.float64))
            if self._t_obs is None:
                self.rt.reset_observer(x)
            else:
                self.rt.observe(x, float(a_meas), now - self._t_obs)
            self._t_obs = now
        return super().step(now, x_local, omega_meas=omega_meas, a_meas=a_meas)

    def _goto_start(self, now: float, x: np.ndarray) -> Command:
        was = self.phase
        cmd = super()._goto_start(now, x)
        if was is Phase.GOTO_START and self.phase is Phase.POLICY:
            if self._t_obs is None:
                self.rt.reset_observer(self.to_pad(x))
            self._handover["d_hat"] = self.rt.d_hat.tolist()
            self._info(f"disturbance estimate at the hand-over: {np.round(self.rt.d_hat, 3).tolist()} m/s^2")
        return cmd

    # ------------------------------------------------------------------ the controller phase
    def _policy(self, now: float, x_local: np.ndarray, omega_meas, a_meas) -> Command:
        c = self.cfg
        x = self.to_pad(x_local)
        t = now - self._t0_policy
        if not (self.armed and self.offboard):
            self._finish(now, "aborted: lost arming or offboard mode", Phase.DONE)
            return self._hold(x_local[0:3])
        if not np.all(np.isfinite(x)):
            return self._abort(now, x_local, "non-finite state")
        dt_obs = None if self._t_obs is None else now - self._t_obs
        self._t_obs = now
        out = self.rt.act(x, t, a_meas=a_meas, omega_meas=omega_meas, dt_obs=dt_obs)
        if not np.all(np.isfinite(out.u)):
            return self._abort(now, x_local, "non-finite command")
        thrust_norm, saturated = self._thrust_map(float(out.u[0])) if self._thrust_map is not None else (float("nan"), False)
        self._log_row(now, x, t, out, omega_meas, a_meas, (thrust_norm, saturated), phase=Phase.POLICY)
        r = self._rec
        speed = float(np.linalg.norm(x[3:6]))
        tilt = tilt_deg(x[6:10])
        err = float(out.info["pos_err"])
        r["height"].append(out.height); r["latency"].append(out.latency_s); r["sat"].append(bool(saturated))
        r["speed"].append(speed); r["pos_err"].append(err); r["dhat"].append(np.asarray(out.info["d_hat"]))
        r["tilt"].append(tilt); r["u"].append(np.asarray(out.u)); r["u_pd"].append(np.asarray(out.u_nominal))
        r["omega"].append(np.full(3, np.nan) if omega_meas is None else np.asarray(omega_meas, dtype=np.float64))
        r["t"].append(float(t)); r["x"].append(x[0:3].copy()); r["ref"].append(np.asarray(out.info["ref"]))
        r["vz"].append(float(x[5]))
        if omega_meas is not None and np.all(np.isfinite(omega_meas)):
            r["rate_err"].append(np.asarray(out.u[1:4]) - np.asarray(omega_meas, dtype=np.float64))
        if a_meas is not None and np.isfinite(a_meas):
            r["a_err"].append(float(out.u[0]) - float(a_meas))

        if err > c.max_track_error:
            return self._abort(now, x_local, f"tracking error {err:.2f} m")
        if tilt > c.tilt_abort_deg:
            return self._abort(now, x_local, f"tilt {tilt:.0f} deg")
        if speed > c.max_speed:
            return self._abort(now, x_local, f"speed {speed:.2f} m/s")
        if out.height < c.min_height:
            return self._abort(now, x_local, f"height {out.height:.2f} m above the take-off point")
        if t + 0.5 * self.rt.dt >= self._duration:
            self._hold_local = x_local[0:3].copy()
            self._hold_time = float(c.end_hold_time)
            self._finish(now, f"reference complete after {t:.2f} s (error at the end {err:.3f} m)", Phase.RECOVER)
        return Command(mode="rates", rates=np.asarray(out.u[1:4], dtype=np.float64), a_cmd=float(out.u[0]))

    def _abort(self, now: float, x_local: np.ndarray, reason: str) -> Command:
        """Back to PX4 position hold where the vehicle is (not above it), then land."""
        self._hold_local = np.asarray(x_local, dtype=np.float64)[0:3].copy()
        self._hold_time = float(self.cfg.recover_time)
        self._finish(now, f"aborted: {reason}", Phase.RECOVER)
        return self._hold(self._hold_local)

    def _recover(self, now: float, x: np.ndarray) -> Command:
        if now - self._t_phase > self._hold_time:
            self._set_phase(Phase.LAND, now)
        return self._hold(self._hold_local)

    # ------------------------------------------------------------------ records
    def _policy_stats(self, now: float) -> dict:
        r = self._rec
        n = len(r["pos_err"])
        s: dict = {"outcome": self.outcome, "steps": n, "duration_s": float(now - self._t0_policy) if n else 0.0,
                   "expected_steps": int(round((now - self._t0_policy) / self.rt.dt)) + 1 if n else 0, "handover": self._handover}
        if not n:
            return s
        e = np.asarray(r["pos_err"])
        u = np.asarray(r["u"])
        dh = np.asarray(r["dhat"])
        lat = np.asarray(r["latency"]) * 1e3
        du = np.linalg.norm(np.diff(u, axis=0), axis=1) if n > 1 else np.zeros(1)
        s.update({
            "pos_err_mean_m": float(e.mean()), "pos_err_p95_m": float(np.percentile(e, 95)), "pos_err_max_m": float(e.max()),
            "pos_err_after_5s_m": float(e[int(5.0 / self.rt.dt):].mean()) if n > int(5.0 / self.rt.dt) else None,
            "pos_err_final_m": float(e[-1]), "max_tilt_deg": float(np.max(r["tilt"])), "max_speed_m_s": float(np.max(r["speed"])),
            "min_height_m": float(np.min(r["height"])),
            "thrust_cmd_range_m_s2": [float(u[:, 0].min()), float(u[:, 0].max())],
            "max_rate_cmd_rad_s": float(np.abs(u[:, 1:4]).max()), "command_change_mean": float(du.mean()),
            "thrust_saturated_steps": int(np.sum(r["sat"])),
            "d_hat_final": dh[-1].tolist(), "d_hat_max_norm": float(np.linalg.norm(dh, axis=1).max()),
            "latency_ms": {"median": float(np.median(lat)), "p95": float(np.percentile(lat, 95)), "max": float(lat.max())},
        })
        if n >= 100:
            # estimator consistency: PX4's vertical velocity against the change of its own height estimate, per 5 s.
            # They disagreed by up to 0.15 m/s for 20 s after take-off in two SITL flights (2026-11-18): the controller
            # then holds a phantom climb and sags 15-20 cm.
            tt, zz, vv = np.asarray(r["t"]), np.asarray(r["x"])[:, 2], np.asarray(r["vz"])
            worst = 0.0
            for lo in np.arange(0.0, tt[-1], 5.0):
                m = (tt >= lo) & (tt < lo + 5.0)
                if m.sum() > 50:
                    worst = max(worst, abs(float(vv[m].mean()) - float(np.polyfit(tt[m], zz[m], 1)[0])))
            s["vz_inconsistency_max_m_s"] = worst
        if r["rate_err"]:
            ee = np.asarray(r["rate_err"])
            s["rate_tracking_rms_rad_s"] = np.sqrt(np.mean(ee * ee, axis=0)).tolist()
        if r["a_err"]:
            ee = np.asarray(r["a_err"])
            s["thrust_tracking_rms_m_s2"] = float(np.sqrt(np.mean(ee * ee)))
            s["thrust_tracking_mean_m_s2"] = float(np.mean(ee))
        return s

    def _stat_lines(self) -> list[str]:
        s = self.stats
        if not s.get("steps"):
            return []
        lines = [
            f"  {s['steps']} control steps in {s['duration_s']:.2f} s (expected {s['expected_steps']}); position error mean "
            f"{100 * s['pos_err_mean_m']:.1f} cm, p95 {100 * s['pos_err_p95_m']:.1f} cm, max {100 * s['pos_err_max_m']:.1f} cm, "
            f"at the end {100 * s['pos_err_final_m']:.1f} cm",
            f"  commands: thrust {s['thrust_cmd_range_m_s2'][0]:.2f}..{s['thrust_cmd_range_m_s2'][1]:.2f} m/s^2 "
            f"(saturated {s['thrust_saturated_steps']} steps), rates up to {s['max_rate_cmd_rad_s']:.2f} rad/s, mean change per step "
            f"{s['command_change_mean']:.3f}; tilt up to {s['max_tilt_deg']:.1f} deg, speed up to {s['max_speed_m_s']:.2f} m/s; "
            f"disturbance estimate at the end {np.round(s['d_hat_final'], 2).tolist()} m/s^2; latency median "
            f"{s['latency_ms']['median']:.2f} ms",
        ]
        if "vz_inconsistency_max_m_s" in s:
            v = s["vz_inconsistency_max_m_s"]
            lines.append(f"  estimator check: vertical velocity vs height change differ by up to {v:.3f} m/s"
                         + (" - PX4's estimate was inconsistent: expect a height offset of about 1x this in metres"
                            if v > 0.05 else " (consistent)"))
        if "rate_tracking_rms_rad_s" in s:
            e = s["rate_tracking_rms_rad_s"]
            lines.append(f"  rate tracking error (commanded - measured, RMS): roll {e[0]:.2f}, pitch {e[1]:.2f}, yaw {e[2]:.2f} rad/s"
                         + (f"; thrust {s['thrust_tracking_rms_m_s2']:.2f} m/s^2 RMS, mean {s['thrust_tracking_mean_m_s2']:+.2f}"
                            if "thrust_tracking_rms_m_s2" in s else ""))
        return lines

    def _log_row(self, now, x, t, out, omega_meas, a_meas, thrust=None, phase=None) -> None:
        if self._csv is None:
            return
        row = [f"{now - self._t_first:.4f}", (phase or self.phase).name, "" if t is None else f"{t:.4f}"] + [f"{v:.6f}" for v in x]
        if out is None:
            row += [""] * (len(SLOW_CSV_COLUMNS) - len(row) - 4) + [f"{v:.6f}" for v in self.rt.d_hat] + [""]
        else:
            om = [float("nan")] * 3 if omega_meas is None else list(np.asarray(omega_meas, dtype=np.float64))
            row += [f"{v:.6f}" for v in out.info["ref"]] + [f"{out.info['pos_err']:.6f}"]
            row += [f"{v:.6f}" for v in out.u] + [f"{v:.6f}" for v in out.u_nominal]
            row += [f"{v:.6f}" for v in om] + [f"{float('nan') if a_meas is None else a_meas:.6f}"]
            row += [f"{thrust[0]:.6f}", int(bool(thrust[1]))] + [f"{v:.6f}" for v in out.info["d_hat"]]
            row += [f"{out.latency_s * 1e3:.3f}"]
        self._csv.writerow(row)

    def _write_summary(self, now: float, x_local) -> None:
        r, s = self._rec, self.stats
        if s.get("steps") and "rate_response" not in s and len(r["t"]) >= 30:
            try:
                s["rate_response"] = fit_rate_response(np.asarray(r["t"]), np.asarray(r["u"])[:, 1:4], np.asarray(r["omega"]))
                fit = s["rate_response"]
                if fit:
                    self._info(f"roll/pitch rate response: delay {fit['delay_s'] * 1e3:.0f} ms + lag {fit['tau_s'] * 1e3:.0f} ms "
                               f"(fit error {fit['rms_rad_s']:.2f} rad/s RMS); trained for delay 0-40 ms + 20 ms, lag 20-80 ms")
            except Exception as exc:
                self._warn(f"rate-response fit failed: {exc!r}")
        summary = {
            "outcome": self.outcome or "ended before the controller phase",
            "controller": self.stats,
            "flight_duration_s": float(now - self._t_first) if self._t_first is not None else 0.0,
            "events": self.events,
            "ground_point_local": None if self.ground is None else self.ground.tolist(),
            "handover_point_local": None if self.shift is None else self.to_local(np.zeros(3)).tolist(),
            "final_position_task_frame": None if (x_local is None or self.shift is None) else self.to_pad(x_local)[0:3].tolist(),
            "runtime": {**asdict(self.rt.cfg), "label": self.rt.label, "dt": self.rt.dt, "episode_duration_s": self.rt.episode_duration,
                        "action_low": self.rt.action_low.tolist(), "action_high": self.rt.action_high.tolist()},
            "flight_config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(self.cfg).items()},
        }
        self.summary = summary
        if self._csv_file is not None:
            self._csv_file.flush()
        if self.cfg.log_dir:
            (Path(self.cfg.log_dir) / "summary.json").write_text(json.dumps(summary, indent=2, default=float) + "\n", encoding="utf-8")
            self._info(f"flight complete ({summary['outcome']}); records in {self.cfg.log_dir}")
        else:
            self._info(f"flight complete ({summary['outcome']})")


__all__ = ["SLOW_CSV_COLUMNS", "SlowFlightConfig", "SlowTrackFlight"]
