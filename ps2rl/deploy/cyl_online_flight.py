"""Flight logic for the obstacle-only demo: take off, hand over, fly past the (virtual) cylinders to the goal, hover, land.

    BOOT -> ARM -> TAKEOFF -> GOTO_START -> POLICY -> RECOVER (hover) -> LAND -> DONE
                                              \\-> RECOVER (hover where it is) -> LAND   (abort)

* Up to GOTO_START PX4's position controller flies: it climbs to ``flight_height`` above the take-off point and holds
  there at yaw 0 (body x east) until the hand-over gates (position, speed, tilt) have held for ``settle_time`` and the
  filter has a certified plan from the state (``recoverable``) - the sequence and gates of the landing flight
  (``landing_flight``), which this class extends.
* POLICY: the nominal through the online filter (``cyl_online_runtime``) commands body rates and collective thrust every
  control step. The task frame: origin at the take-off point, x east, z up; the course runs along +x.
* End: once the vehicle is inside the goal circle (``goal_radius``, as in the simulations) and slower than
  ``arrive_speed``, PX4 position hold takes over where it is for ``end_hold_time``, then PX4's land mode lands it. At
  ``policy_timeout`` the same happens wherever the vehicle is.
* Abort - immediately back to PX4 position hold where the vehicle is, then land after ``recover_time``: contact with
  a cylinder (clearance < 0: the drone's disc touches a virtual cylinder), tilt above ``tilt_abort_deg``, speed above
  ``max_speed``, lower than ``min_height`` above the take-off point, a non-finite state or command, loss of arming or
  offboard mode. The node also aborts on stale odometry or an exception in the controller.
* The disturbance observer runs on the measured thrust from take-off on (``observer_warmup``), so it is warm at the
  hand-over.

Plain Python: no ROS, no PX4 types. ``scripts/rehearse_cyl_online_flight.py`` flies it against a stand-in vehicle.
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
from ps2rl.deploy.cyl_online_runtime import CylOnlineRuntime
from ps2rl.deploy.landing_flight import Command, FlightConfig, LandingFlight, Phase, tilt_deg
from ps2rl.online.filter import MODES

CYL_CSV_COLUMNS = (
    ["t_flight", "phase", "t_policy", "px", "py", "pz", "vx", "vy", "vz", "qw", "qx", "qy", "qz",
     "goal_x", "goal_y", "a_cmd", "wx", "wy", "wz", "a_nom", "wx_nom", "wy_nom", "wz_nom",
     "wx_meas", "wy_meas", "wz_meas", "a_meas", "thrust_norm", "saturated", "lam", "mode", "n_cert", "rollout_steps",
     "clearance", "known", "in_base", "dhat_x", "dhat_y", "dhat_z", "decide_ms", "latency_ms"]
)


@dataclass
class CylFlightConfig(FlightConfig):
    settle_time: float = 8.0           # hold at the hand-over point this long before handing over [s]: PX4's vertical
                                       # velocity estimate was still off by up to 0.15 m/s ~10 s after take-off
    policy_timeout: float = 30.0       # longest controller phase [s] (the simulations: 4.7 s O3, 6.3 s S1 with the planner;
                                       # 10-14 s with the straight line)
    arrive_speed: float = 0.25         # end: inside the goal circle and slower than this [m/s]
    end_hold_time: float = 3.0         # hover where the controller phase ended, then land [s]
    max_speed: float = 2.0             # abort: speed [m/s] (cruise 1 m/s)
    min_height: float = 0.3            # abort: height above the take-off point [m] (the filter's floor is 0.5 m)
    tilt_abort_deg: float = 35.0       # the nominal tilts up to 20 deg
    contact_abort: bool = True         # abort on contact with a (virtual) cylinder
    observer_warmup: bool = True       # run the disturbance observer on the measured thrust before the hand-over
    recover_time: float = 3.0          # after an abort: hover this long where the vehicle is, then land [s]


class CylOnlineFlight(LandingFlight):
    def __init__(self, runtime: CylOnlineRuntime, cfg: CylFlightConfig, *, logger=None, thrust_map=None) -> None:
        runtime.reset()
        super().__init__(runtime, replace(cfg, log_dir=""), logger=logger, thrust_map=thrust_map)
        self.cfg = cfg
        self.rt: CylOnlineRuntime = runtime
        self._hold_time = float(cfg.recover_time)
        self._t_obs: float | None = None
        self._rec.update({k: [] for k in ("t", "x", "u", "u_nom", "omega", "tilt", "clear", "mode", "steps", "decide",
                                          "dhat", "known", "inb", "intervene", "compute")})
        if cfg.log_dir:
            Path(cfg.log_dir).mkdir(parents=True, exist_ok=True)
            self._csv_file = open(Path(cfg.log_dir) / "flight.csv", "w", newline="", encoding="utf-8")
            self._csv = csv.writer(self._csv_file)
            self._csv.writerow(CYL_CSV_COLUMNS)
        self._info(f"cyl-online flight: {runtime.label}")
        self._info(f"hand-over {runtime.cfg.flight_height:g} m above the take-off point, goal "
                   f"{np.round(runtime.goal[0:2], 2).tolist()} m (x east, y north); end inside {runtime.cfg.goal_radius:g} m of "
                   f"the goal below {cfg.arrive_speed:g} m/s, or after {cfg.policy_timeout:g} s; then hover "
                   f"{cfg.end_hold_time:g} s and land. Aborts: contact, tilt > {cfg.tilt_abort_deg:g} deg, speed > "
                   f"{cfg.max_speed:g} m/s, height < {cfg.min_height:g} m")

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
            self._handover["position_task_frame"] = self.to_pad(x)[0:3].tolist()
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
        out = self.rt.act(x, t, a_meas=a_meas, omega_meas=omega_meas)
        if not np.all(np.isfinite(out.u)):
            return self._abort(now, x_local, "non-finite command")
        thrust_norm, saturated = self._thrust_map(float(out.u[0])) if self._thrust_map is not None else (float("nan"), False)
        self._log_row(now, x, t, out, omega_meas, a_meas, (thrust_norm, saturated), phase=Phase.POLICY)
        r, info = self._rec, out.info
        speed = float(np.linalg.norm(x[3:6]))
        tilt = tilt_deg(x[6:10])
        r["height"].append(out.height); r["latency"].append(out.latency_s); r["sat"].append(bool(saturated))
        r["speed"].append(speed); r["lam"].append(out.safeguard_lambda); r["tilt"].append(tilt)
        r["t"].append(float(t)); r["x"].append(x[0:3].copy()); r["u"].append(np.asarray(out.u)); r["u_nom"].append(np.asarray(out.u_nominal))
        r["omega"].append(np.full(3, np.nan) if omega_meas is None else np.asarray(omega_meas, dtype=np.float64))
        r["clear"].append(float(info["clearance"])); r["mode"].append(info["mode"]); r["steps"].append(int(info["steps"]))
        r["decide"].append(float(info["decide_ms"])); r["dhat"].append(np.asarray(info["d_hat"])); r["known"].append(int(info["known"]))
        r["compute"].append(float(info.get("compute_ms", 1e3 * out.latency_s)))
        r["inb"].append(bool(out.in_base_set)); r["intervene"].append(bool(info["intervene"]))
        if omega_meas is not None and np.all(np.isfinite(omega_meas)):
            r["rate_err"].append(np.asarray(out.u[1:4]) - np.asarray(omega_meas, dtype=np.float64))
        if a_meas is not None and np.isfinite(a_meas):
            r["a_err"].append(float(out.u[0]) - float(a_meas))

        if c.contact_abort and info["clearance"] < 0.0:
            return self._abort(now, x_local, f"contact with a cylinder (clearance {info['clearance']:+.2f} m)")
        if tilt > c.tilt_abort_deg:
            return self._abort(now, x_local, f"tilt {tilt:.0f} deg")
        if speed > c.max_speed:
            return self._abort(now, x_local, f"speed {speed:.2f} m/s")
        if out.height < c.min_height:
            return self._abort(now, x_local, f"height {out.height:.2f} m above the take-off point")
        if info["dist_goal"] < self.rt.cfg.goal_radius and speed < c.arrive_speed:
            self._hold_local = x_local[0:3].copy()
            self._hold_time = float(c.end_hold_time)
            self._finish(now, f"goal reached: inside {self.rt.cfg.goal_radius:g} m after {self.rt.goal_entered_t:.2f} s, "
                              f"below {c.arrive_speed:g} m/s after {t:.2f} s", Phase.RECOVER)
        elif t + 0.5 * self.rt.dt >= c.policy_timeout:
            self._hold_local = x_local[0:3].copy()
            self._hold_time = float(c.end_hold_time)
            self._finish(now, f"time-out after {t:.1f} s, {info['dist_goal']:.2f} m from the goal", Phase.RECOVER)
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
        n = len(r["t"])
        s: dict = {"outcome": self.outcome, "steps": n, "duration_s": float(now - self._t0_policy) if n else 0.0,
                   "expected_steps": int(round((now - self._t0_policy) / self.rt.dt)) + 1 if n else 0, "handover": self._handover}
        if not n:
            return s
        u, lam = np.asarray(r["u"]), np.asarray(r["lam"])
        lat, dec = np.asarray(r["latency"]) * 1e3, np.asarray(r["decide"])
        steps = np.asarray(r["steps"])
        modes = np.asarray(r["mode"])
        dh = np.asarray(r["dhat"])
        du = np.linalg.norm(np.diff(u, axis=0), axis=1) if n > 1 else np.zeros(1)
        P = np.asarray(r["x"])
        s.update({
            "goal_entered_s": self.rt.goal_entered_t, "min_clearance_m": float(np.min(r["clear"])),
            "y_range_m": [float(P[:, 1].min()), float(P[:, 1].max())], "x_max_m": float(P[:, 0].max()),
            "lam_mean": float(lam.mean()), "intervene_fraction": float(np.mean(r["intervene"])),
            "lam_switches": int(np.sum(np.abs(np.diff(lam)) > 0.25)) if n > 1 else 0,
            "modes": {k: float(np.mean(modes == k)) for k in MODES},
            "in_base_fraction": float(np.mean(r["inb"])), "known_final": int(r["known"][-1]),
            "rollout_steps": {"mean": float(steps.mean()), "p99": float(np.percentile(steps, 99)), "max": int(steps.max())},
            "decide_ms": {"median": float(np.median(dec)), "p95": float(np.percentile(dec, 95)), "max": float(dec.max())},
            "latency_ms": {"median": float(np.median(lat)), "p95": float(np.percentile(lat, 95)), "max": float(lat.max())},
            "compute_ms": {"median": float(np.median(r["compute"])), "p95": float(np.percentile(r["compute"], 95)),
                           "max": float(np.max(r["compute"]))},
            "control_period_ms": 1e3 * float(self.rt.dt),
            "max_tilt_deg": float(np.max(r["tilt"])), "max_speed_m_s": float(np.max(r["speed"])),
            "min_height_m": float(np.min(r["height"])),
            "thrust_cmd_range_m_s2": [float(u[:, 0].min()), float(u[:, 0].max())],
            "max_rate_cmd_rad_s": float(np.abs(u[:, 1:4]).max()), "command_change_mean": float(du.mean()),
            "thrust_saturated_steps": int(np.sum(r["sat"])),
            "d_hat_final": dh[-1].tolist(), "d_hat_max_norm": float(np.linalg.norm(dh, axis=1).max()),
        })
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
        ge = s["goal_entered_s"]
        md = s["modes"]
        lines = [
            f"  {s['steps']} control steps in {s['duration_s']:.2f} s (expected {s['expected_steps']}); goal circle "
            + (f"entered after {ge:.2f} s" if ge is not None else "not reached")
            + f"; min clearance {s['min_clearance_m']:+.2f} m; y {s['y_range_m'][0]:+.2f}..{s['y_range_m'][1]:+.2f} m",
            f"  filter: lam mean {s['lam_mean']:.2f}, intervened on {100 * s['intervene_fraction']:.0f} % of steps "
            f"({s['lam_switches']} switches); modes fresh {100 * md['fresh']:.0f} %, committed {100 * md['committed']:.0f} %, "
            f"held {100 * md['held']:.0f} %, none {100 * md['none']:.0f} %; rollout steps mean {s['rollout_steps']['mean']:.0f}, "
            f"max {s['rollout_steps']['max']}",
            f"  compute: decision median {s['decide_ms']['median']:.2f} / p95 {s['decide_ms']['p95']:.2f} / max "
            f"{s['decide_ms']['max']:.2f} ms; whole step median {s['latency_ms']['median']:.2f} / p95 {s['latency_ms']['p95']:.2f} / "
            f"max {s['latency_ms']['max']:.2f} ms against a {1e3 * self.rt.dt:.0f} ms control period ({self.rt.backend})",
            f"  commands: thrust {s['thrust_cmd_range_m_s2'][0]:.2f}..{s['thrust_cmd_range_m_s2'][1]:.2f} m/s^2 (saturated "
            f"{s['thrust_saturated_steps']} steps), rates up to {s['max_rate_cmd_rad_s']:.2f} rad/s; tilt up to "
            f"{s['max_tilt_deg']:.1f} deg, speed up to {s['max_speed_m_s']:.2f} m/s, height min {s['min_height_m']:.2f} m; "
            f"disturbance estimate at the end {np.round(s['d_hat_final'], 2).tolist()} m/s^2",
        ]
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
            row += [""] * (len(CYL_CSV_COLUMNS) - len(row) - 5) + [f"{v:.6f}" for v in self.rt.d_hat] + ["", ""]
        else:
            i = out.info
            om = [float("nan")] * 3 if omega_meas is None else list(np.asarray(omega_meas, dtype=np.float64))
            row += [f"{self.rt.goal[0]:.4f}", f"{self.rt.goal[1]:.4f}"]
            row += [f"{v:.6f}" for v in out.u] + [f"{v:.6f}" for v in out.u_nominal]
            row += [f"{v:.6f}" for v in om] + [f"{float('nan') if a_meas is None else a_meas:.6f}"]
            row += [f"{thrust[0]:.6f}", int(bool(thrust[1])), f"{out.safeguard_lambda:.3f}", i["mode"], i["n_cert"], i["steps"],
                    f"{i['clearance']:.6f}", i["known"], int(out.in_base_set)]
            row += [f"{v:.6f}" for v in i["d_hat"]] + [f"{i['decide_ms']:.3f}", f"{out.latency_s * 1e3:.3f}"]
        self._csv.writerow(row)

    def close(self) -> None:
        super().close()
        try:
            self.rt.close()
        except Exception:
            pass

    def _write_summary(self, now: float, x_local) -> None:
        r, s = self._rec, self.stats
        if s.get("steps") and "rate_response" not in s and len(r["t"]) >= 30:
            try:
                s["rate_response"] = fit_rate_response(np.asarray(r["t"]), np.asarray(r["u"])[:, 1:4], np.asarray(r["omega"]))
                fit = s["rate_response"]
                if fit:
                    self._info(f"roll/pitch rate response: delay {fit['delay_s'] * 1e3:.0f} ms + lag {fit['tau_s'] * 1e3:.0f} ms "
                               f"(fit error {fit['rms_rad_s']:.2f} rad/s RMS); the simulations drew delay 0-40 ms, lag 30-80 ms")
            except Exception as exc:
                self._warn(f"rate-response fit failed: {exc!r}")
        rt = self.rt
        dsc = rt.describe()
        summary = {
            "outcome": self.outcome or "ended before the controller phase",
            "controller": self.stats,
            "flight_duration_s": float(now - self._t_first) if self._t_first is not None else 0.0,
            "events": self.events,
            "course": {"label": dsc["course_label"], "cylinders_xyr": dsc["cylinders_xyr"], "goal": dsc["goal"],
                       "goal_radius": rt.cfg.goal_radius, "r_drone": dsc["r_drone"], "rho_xy": dsc["rho_xy"],
                       "z_floor": dsc["z_floor"]},
            "ground_point_local": None if self.ground is None else self.ground.tolist(),
            "handover_point_local": None if self.shift is None else self.to_local(self.start_pad).tolist(),
            "final_position_task_frame": None if (x_local is None or self.shift is None) else self.to_pad(x_local)[0:3].tolist(),
            "runtime": {**asdict(rt.cfg), "label": dsc["label"], "backend": dsc["backend"], "dt": dsc["dt"], "hold": dsc["hold"],
                        "filter": dsc["filter"], "nominal_config": dsc["nominal_config"],
                        "action_low": np.asarray(rt.action_low).tolist(), "action_high": np.asarray(rt.action_high).tolist(),
                        "backup_meta": dsc["backup_meta"]},
            "flight_config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(self.cfg).items()},
        }
        self.summary = summary
        if self._csv_file is not None:
            self._csv_file.flush()
        if self.cfg.log_dir:
            (Path(self.cfg.log_dir) / "summary.json").write_text(json.dumps(summary, indent=2, default=float) + "\n",
                                                                  encoding="utf-8")
            self._info(f"flight complete ({summary['outcome']}); records in {self.cfg.log_dir}")
        else:
            self._info(f"flight complete ({summary['outcome']})")


__all__ = ["CYL_CSV_COLUMNS", "CylFlightConfig", "CylOnlineFlight", "Phase"]
