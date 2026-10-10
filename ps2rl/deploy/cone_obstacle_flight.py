"""Flight logic for the outside-cone task: take off, hand over at the reference start, fly one episode.

    BOOT -> ARM -> TAKEOFF -> GOTO_START -> POLICY -> RECOVER (hold) -> LAND -> DONE

The sequence, the hand-over gates, the request handling and the records are those of the landing
flight (``landing_flight.LandingFlight``, subclassed here); what differs:

* frames: the vehicle's ground point is put at the task's anchor (the ground point below the
  reference start, ``runtime.pad``) plus ``pad_offset``; the funnel axis therefore lies
  ``axis - start`` away horizontally (3 m along local x for L3-1-f). The funnel and the target
  cylinder are virtual: nothing in the simulator stands there;
* the disturbance observer is run on the measured thrust from take-off on, so its estimate of the
  model mismatch is warm at the hand-over;
* POLICY lasts one training episode (reference + extra time, ``policy_duration``), then the
  vehicle holds where it is for ``end_hold_time`` and PX4 lands it. Touching the target cylinder is
  recorded, not an end condition (the reference ends inside it);
* aborts: inside the funnel by more than ``funnel_abort_depth`` (filtered flights only - an
  unfiltered comparison flight is expected to cross the virtual funnel), below the floor by more than
  ``floor_abort_depth``, tilt.

Plain Python: no ROS, no PX4 types (the caller converts frames and publishes).
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path

import numpy as np

from ps2rl.deploy.cone_obstacle_runtime import ConeObstacleRuntime
from ps2rl.deploy.landing_flight import CSV_COLUMNS, Command, FlightConfig, LandingFlight, Phase, tilt_deg

CONE_CSV_COLUMNS = CSV_COLUMNS + ["dhat_x", "dhat_y", "dhat_z", "touch"]


def fit_rate_response(cmd: np.ndarray, meas: np.ndarray, dt: float, max_delay_steps: int = 5) -> dict | None:
    """Fit the measured roll/pitch rates to a delayed first-order response of the commanded ones,

        meas[k+1] = meas[k] + a (cmd[k-n] - meas[k]),   a = 1 - exp(-dt / tau),

    for n = 0..max_delay_steps (least squares in a, the n with the smallest residual kept). Returns the
    delay n dt, the time constant tau and their sum (the effective delay, what ``predict_ahead`` should
    roughly be), or None when there are too few steps or the commands hardly vary."""
    cmd = np.asarray(cmd, np.float64)[:, 0:2]
    meas = np.asarray(meas, np.float64)[:, 0:2]
    ok = np.all(np.isfinite(meas), axis=1) & np.all(np.isfinite(cmd), axis=1)
    if ok.sum() < 30 or np.std(cmd[ok]) < 0.05:
        return None
    best = None
    for n in range(max_delay_steps + 1):
        k = np.arange(n, len(cmd) - 1)
        k = k[ok[k] & ok[k + 1] & ok[k - n]]
        if k.size < 20:
            continue
        y = (meas[k + 1] - meas[k]).ravel()
        x = (cmd[k - n] - meas[k]).ravel()
        if float(x @ x) <= 1e-12:
            continue
        a = float(np.clip((x @ y) / (x @ x), 1e-6, 1.0))
        res = float(np.mean((y - a * x) ** 2))
        if best is None or res < best[0]:
            best = (res, n, a)
    if best is None:
        return None
    _, n, a = best
    tau = 0.0 if a >= 1.0 - 1e-9 else float(-dt / np.log(1.0 - a))
    return {"delay_s": n * dt, "tau_s": tau, "effective_delay_s": n * dt + tau, "steps": int(ok.sum())}


@dataclass
class ConeFlightConfig(FlightConfig):
    policy_duration: float = 0.0       # length of the policy phase [s]; 0 = the training episode (reference + extra time)
    end_hold_time: float = 3.0         # hold where the episode ended for this long, then land [s]
    funnel_abort_depth: float = 0.5    # filtered flights: abort when inside the funnel by more than this [m]
    tilt_abort_deg: float = 110.0      # the tracker through the filter reaches ~100 deg in simulation
    observer_warmup: bool = True       # run the disturbance observer on the measured thrust before the hand-over


class ConeObstacleFlight(LandingFlight):
    def __init__(self, runtime: ConeObstacleRuntime, cfg: ConeFlightConfig, *, logger=None, thrust_map=None) -> None:
        runtime.reset()        # no disturbance estimate or command history from an earlier flight
        super().__init__(runtime, replace(cfg, log_dir=""), logger=logger, thrust_map=thrust_map)
        self.cfg = cfg
        self.rt: ConeObstacleRuntime = runtime
        self._duration = float(cfg.policy_duration) if cfg.policy_duration > 0 else float(runtime.episode_duration)
        self._hold_time = float(cfg.recover_time)
        self._t_obs: float | None = None
        self._first_touch: float | None = None
        self._rec.update({"touch": [], "dhat": [], "tilt": [], "u": [], "omega": []})
        if cfg.log_dir:
            Path(cfg.log_dir).mkdir(parents=True, exist_ok=True)
            self._csv_file = open(Path(cfg.log_dir) / "flight.csv", "w", newline="", encoding="utf-8")
            self._csv = csv.writer(self._csv_file)
            self._csv.writerow(CONE_CSV_COLUMNS)
        ax = runtime.axis - runtime.start_state[0:2]
        self._info(f"outside-cone flight: policy {Path(runtime.cfg.run).name}, filter {runtime.cfg.filter}; episode {self._duration:.2f} s; "
                   f"the (virtual) funnel axis is {np.round(ax, 2).tolist()} m from the take-off point horizontally")

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
            if self._t_obs is None:            # no thrust measurement: start the observer cold
                self.rt.reset_observer(self.to_pad(x))
            self._handover["d_hat"] = self.rt.d_hat.tolist()
            self._info(f"disturbance estimate at the hand-over: {np.round(self.rt.d_hat, 3).tolist()} m/s^2")
        return cmd

    # ------------------------------------------------------------------ the policy episode
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
            return self._abort(now, x_local, "non-finite action")
        thrust_norm, saturated = self._thrust_map(float(out.u[0])) if self._thrust_map is not None else (float("nan"), False)
        self._log_row(now, x, t, out, omega_meas, a_meas, (thrust_norm, saturated), phase=Phase.POLICY)
        r = self._rec
        speed = float(np.linalg.norm(x[3:6]))
        tilt = tilt_deg(x[6:10])
        r["h_cone"].append(out.h_cone); r["height"].append(out.height); r["slack"].append(out.slack)
        r["latency"].append(out.latency_s); r["lam"].append(out.safeguard_lambda); r["used"].append(out.qp_used)
        r["sat"].append(bool(saturated)); r["speed"].append(speed)
        r["radial"].append(float(np.linalg.norm(x[0:2] - self.rt.axis))); r["du"].append(float(np.linalg.norm(out.u - out.u_nominal)))
        r["touch"].append(bool(out.info.get("touch", False))); r["dhat"].append(np.asarray(out.info.get("d_hat", np.zeros(3))))
        r["tilt"].append(tilt); r["u"].append(np.asarray(out.u))
        r["omega"].append(np.full(3, np.nan) if omega_meas is None else np.asarray(omega_meas, dtype=np.float64))
        if omega_meas is not None and np.all(np.isfinite(omega_meas)):
            r["rate_err"].append(np.asarray(out.u[1:4]) - np.asarray(omega_meas, dtype=np.float64))
        if a_meas is not None and np.isfinite(a_meas):
            r["a_err"].append(float(out.u[0]) - float(a_meas))
        if self._first_touch is None and r["touch"][-1]:
            self._first_touch = t
            self._info(f"touched the target cylinder at t = {t:.2f} s")

        if self.rt.cfg.filter != "none" and out.h_cone < -c.funnel_abort_depth:
            return self._abort(now, x_local, f"inside the funnel by {-out.h_cone:.2f} m")
        if out.height < -c.floor_abort_depth:
            return self._abort(now, x_local, f"{-out.height:.2f} m below the floor")
        if tilt > c.tilt_abort_deg:
            return self._abort(now, x_local, f"tilt {tilt:.0f} deg")
        if t + 0.5 * self.rt.dt >= self._duration:
            self._hold_local = x_local[0:3].copy()
            self._hold_time = float(c.end_hold_time)
            touched = f"touched the target at {self._first_touch:.2f} s" if self._first_touch is not None else "did not touch the target"
            self._finish(now, f"episode complete after {t:.2f} s ({touched})", Phase.RECOVER)
        return Command(mode="rates", rates=np.asarray(out.u[1:4], dtype=np.float64), a_cmd=float(out.u[0]))

    def _abort(self, now: float, x_local: np.ndarray, reason: str) -> Command:
        self._hold_time = float(self.cfg.recover_time)
        return super()._abort(now, x_local, reason)

    def _recover(self, now: float, x: np.ndarray) -> Command:
        if now - self._t_phase > self._hold_time:
            self._set_phase(Phase.LAND, now)
        return self._hold(self._hold_local)

    # ------------------------------------------------------------------ records
    def _policy_stats(self, now: float) -> dict:
        s = super()._policy_stats(now)
        r = self._rec
        if not r["height"]:
            return s
        u = np.asarray(r["u"])
        dh = np.asarray(r["dhat"])
        end = self.rt.target
        s.pop("time_to_touch_height_s", None)
        s.pop("speed_at_touch_height_m_s", None)
        s.update({
            "touched": self._first_touch is not None, "time_to_touch_s": self._first_touch,
            "min_floor_margin_m": float(np.min(r["height"])), "max_tilt_deg": float(np.max(r["tilt"])),
            "filter_correction_mean": float(np.mean(r["du"])),
            "max_thrust_cmd_m_s2": float(u[:, 0].max()), "max_rate_cmd_rad_s": float(np.abs(u[:, 1:4]).max()),
            "d_hat_final": dh[-1].tolist(), "d_hat_max_norm": float(np.linalg.norm(dh, axis=1).max()),
            "final_distance_to_reference_end_m": None,
        })
        if self._rows_last_x is not None:
            s["final_distance_to_reference_end_m"] = float(np.linalg.norm(self._rows_last_x[0:3] - end))
        fit = fit_rate_response(u[:, 1:4], np.asarray(r["omega"]), self.rt.dt)
        s["rate_response"] = fit
        s["predict_ahead_s"] = float(self.rt.cfg.predict_ahead)
        return s

    def _stat_lines(self) -> list[str]:
        s = self.stats
        if not s.get("steps"):
            return []
        lines = [
            f"  {s['steps']} control steps in {s['duration_s']:.2f} s (expected {s['expected_steps']}); funnel margin min {s['min_h_cone_m']:+.3f} m, "
            f"floor margin min {s['min_floor_margin_m']:+.3f} m; "
            + (f"touched the target at {s['time_to_touch_s']:.2f} s" if s.get("touched") else "did not touch the target"),
            f"  filter: active on {100 * s['filter_active_fraction']:.0f} % of steps, mean correction {s['filter_correction_mean']:.2f}, "
            f"largest slack {s['max_slack']:.1e}, QP fallbacks {s['qp_fallback_steps']}, safeguard blends {s['safeguard_steps']}, "
            f"thrust saturated {s['thrust_saturated_steps']} steps; latency median {s['latency_ms']['median']:.1f} / "
            f"p95 {s['latency_ms']['p95']:.1f} / max {s['latency_ms']['max']:.1f} ms",
            f"  commands: thrust up to {s['max_thrust_cmd_m_s2']:.1f} m/s^2, rates up to {s['max_rate_cmd_rad_s']:.1f} rad/s, tilt up to "
            f"{s['max_tilt_deg']:.0f} deg; disturbance estimate at the end {np.round(s['d_hat_final'], 2).tolist()} m/s^2 "
            f"(largest |d_hat| {s['d_hat_max_norm']:.2f})",
        ]
        fit = s.get("rate_response")
        if fit:
            lines.append(f"  roll/pitch rate response: delay {fit['delay_s'] * 1e3:.0f} ms + first-order lag {fit['tau_s'] * 1e3:.0f} ms "
                         f"= effective delay {fit['effective_delay_s'] * 1e3:.0f} ms (delay compensation in this flight: "
                         f"{s['predict_ahead_s'] * 1e3:.0f} ms; set predict_ahead to about the effective delay)")
        if "rate_tracking_rms_rad_s" in s:
            e = s["rate_tracking_rms_rad_s"]
            lines.append(f"  rate tracking error (commanded - measured, RMS): roll {e[0]:.2f}, pitch {e[1]:.2f}, yaw {e[2]:.2f} rad/s"
                         + (f"; thrust {s['thrust_tracking_rms_m_s2']:.2f} m/s^2 RMS, mean {s['thrust_tracking_mean_m_s2']:+.2f}"
                            if "thrust_tracking_rms_m_s2" in s else ""))
        return lines

    _rows_last_x = None

    def _log_row(self, now, x, t, out, omega_meas, a_meas, thrust=None, phase=None) -> None:
        self._rows_last_x = np.asarray(x, dtype=np.float64)
        if self._csv is None:
            return
        nan3 = [float("nan")] * 3
        row = [f"{now - self._t_first:.4f}", (phase or self.phase).name, "" if t is None else f"{t:.4f}"] + [f"{v:.6f}" for v in x]
        if out is None:
            row += [""] * (len(CONE_CSV_COLUMNS) - len(row) - 4) + [f"{v:.6f}" for v in self.rt.d_hat] + [""]
        else:
            om = nan3 if omega_meas is None else list(np.asarray(omega_meas, dtype=np.float64))
            row += [f"{v:.6f}" for v in self.rt.nominal_goal(t)] + [f"{v:.6f}" for v in out.u] + [f"{v:.6f}" for v in out.u_nominal]
            row += [f"{v:.6f}" for v in om] + [f"{float('nan') if a_meas is None else a_meas:.6f}"]
            row += [f"{thrust[0]:.6f}", int(bool(thrust[1])), f"{out.slack:.6e}", int(out.qp_used), f"{out.safeguard_lambda:.3f}",
                    f"{out.h_cone:.6f}", f"{out.height:.6f}", int(out.in_base_set), f"{out.latency_s * 1e3:.3f}"]
            row += [f"{v:.6f}" for v in out.info.get("d_hat", np.zeros(3))] + [int(bool(out.info.get("touch", False)))]
        self._csv.writerow(row)

    def _write_summary(self, now: float, x_local) -> None:
        sc = self.rt.scenario
        summary = {
            "outcome": self.outcome or "ended before the policy episode",
            "policy": self.stats,
            "flight_duration_s": float(now - self._t_first) if self._t_first is not None else 0.0,
            "events": self.events,
            "ground_point_local": None if self.ground is None else self.ground.tolist(),
            "task_anchor_local": None if self.shift is None else self.to_local(self.rt.pad).tolist(),
            "funnel_axis_local": None if self.shift is None else self.to_local(np.r_[self.rt.axis, sc.tip_z]).tolist(),
            "final_position_task_frame": None if (x_local is None or self.shift is None) else self.to_pad(x_local)[0:3].tolist(),
            "start_task_frame": self.start_pad.tolist(),
            "runtime": {**asdict(self.rt.cfg), **{k: v for k, v in self.rt.filter_overrides.items()}, "dt": self.rt.dt,
                        "action_low": self.rt.action_low.tolist(), "action_high": self.rt.action_high.tolist(),
                        "episode_duration_s": self.rt.episode_duration},
            "geometry": {"tip_z": sc.tip_z, "cone_theta_deg": sc.cone_theta_deg, "cone_eps": sc.cone_eps, "floor_z": sc.floor_z,
                         "axis": self.rt.axis.tolist(), "target_radius": sc.target_radius, "cylinder_top": sc.cylinder_top,
                         "reference_end": self.rt.target.tolist()},
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


__all__ = ["CONE_CSV_COLUMNS", "ConeFlightConfig", "ConeObstacleFlight", "fit_rate_response"]
