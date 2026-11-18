"""Flight logic of a landing flown by a trained Phase-II policy through the landing filter.

    BOOT -> ARM -> TAKEOFF -> GOTO_START -> POLICY -> LAND -> DONE
                                              \\-> RECOVER -> LAND   (abort)

The sequence, the hand-over gates, the touchdown test, the aborts and the records are those of
the filtered landing (``landing_flight.LandingFlight``, subclassed here). What differs:

* the runtime is ``landing_policy_runtime.LandingPolicyRuntime`` (policy + filter + disturbance
  observer + optional delay compensation); it gets the measured thrust and body rates;
* the observer runs on the measured thrust from take-off on, so its estimate is warm at the
  hand-over;
* the start is the policy reference's first position (it is reached at rest; the reference itself
  starts moving, which the policy was trained to absorb within its start perturbation);
* the records add d_hat, and the summary a fit of the vehicle's rate response (delay + lag).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from ps2rl.deploy.landing_flight import CSV_COLUMNS, FlightConfig, LandingFlight, Phase
from ps2rl.deploy.landing_policy_runtime import LandingPolicyRuntime
from ps2rl.deploy.landing_runtime import ControlOutput
from ps2rl.deploy.rate_response import fit_rate_response

POLICY_CSV_COLUMNS = CSV_COLUMNS + ["dhat_x", "dhat_y", "dhat_z"]


@dataclass
class LandingPolicyFlightConfig(FlightConfig):
    policy_timeout: float = 6.0        # the training episode is 2.5 s; then PX4's land mode descends from where it is [s]
    observer_warmup: bool = True       # run the disturbance observer on the measured thrust before the hand-over


class LandingPolicyFlight(LandingFlight):
    def __init__(self, runtime: LandingPolicyRuntime, cfg: LandingPolicyFlightConfig, *, logger=None, thrust_map=None) -> None:
        runtime.reset()
        super().__init__(runtime, replace(cfg, log_dir=""), logger=logger, thrust_map=thrust_map)
        self.cfg = cfg
        self.rt: LandingPolicyRuntime = runtime
        self._t_obs: float | None = None
        self._rec.update({"dhat": [], "u": [], "omega": [], "t": []})
        if cfg.log_dir:
            Path(cfg.log_dir).mkdir(parents=True, exist_ok=True)
            self._csv_file = open(Path(cfg.log_dir) / "flight.csv", "w", newline="", encoding="utf-8")
            self._csv = csv.writer(self._csv_file)
            self._csv.writerow(POLICY_CSV_COLUMNS)
        self._info(f"landing with a trained policy: {Path(runtime.cfg.run).name} through the filter of "
                   f"{Path(runtime.cfg.backup).name}; reference {Path(runtime.reference_path).name}, episode "
                   f"{runtime.episode_duration:.2f} s, policy phase at most {cfg.policy_timeout:g} s")

    # ------------------------------------------------------------------ observer before the hand-over
    def step(self, now: float, x_local: np.ndarray, *, omega_meas=None, a_meas=None):
        if (self.cfg.observer_warmup and self.phase in (Phase.TAKEOFF, Phase.GOTO_START) and self.shift is not None
                and a_meas is not None and np.isfinite(a_meas) and np.all(np.isfinite(x_local))):
            x = self.to_pad(np.asarray(x_local, dtype=np.float64))
            if self._t_obs is None:
                self.rt.reset_observer(x)
            else:
                self.rt.observe(x, float(a_meas), now - self._t_obs)
            self._t_obs = now
        return super().step(now, x_local, omega_meas=omega_meas, a_meas=a_meas)

    def _goto_start(self, now: float, x: np.ndarray):
        was = self.phase
        cmd = super()._goto_start(now, x)
        if was is Phase.GOTO_START and self.phase is Phase.POLICY:
            if self._t_obs is None:
                self.rt.reset_observer(self.to_pad(x))
            self._handover["d_hat"] = self.rt.d_hat.tolist()
            self._info(f"disturbance estimate at the hand-over: {np.round(self.rt.d_hat, 3).tolist()} m/s^2")
        return cmd

    # ------------------------------------------------------------------ the policy step
    def _act(self, now: float, x: np.ndarray, t: float, omega_meas, a_meas) -> ControlOutput:
        dt_obs = None if self._t_obs is None else now - self._t_obs
        self._t_obs = now
        out = self.rt.act(x, t, a_meas=a_meas, omega_meas=omega_meas, dt_obs=dt_obs)
        r = self._rec
        r["dhat"].append(np.asarray(out.info.get("d_hat", np.zeros(3))))
        r["u"].append(np.asarray(out.u))
        r["omega"].append(np.full(3, np.nan) if omega_meas is None else np.asarray(omega_meas, dtype=np.float64))
        r["t"].append(float(t))
        return out

    # ------------------------------------------------------------------ records
    def _log_row(self, now, x, t, out, omega_meas, a_meas, thrust=None, phase=None) -> None:
        if self._csv is None:
            return
        nan3 = [float("nan")] * 3
        row = [f"{now - self._t_first:.4f}", (phase or self.phase).name, "" if t is None else f"{t:.4f}"] + [f"{v:.6f}" for v in x]
        if out is None:
            row += [""] * (len(CSV_COLUMNS) - len(row)) + [f"{v:.6f}" for v in self.rt.d_hat]
        else:
            om = nan3 if omega_meas is None else list(np.asarray(omega_meas, dtype=np.float64))
            row += [f"{v:.6f}" for v in self.rt.nominal_goal(t)] + [f"{v:.6f}" for v in out.u] + [f"{v:.6f}" for v in out.u_nominal]
            row += [f"{v:.6f}" for v in om] + [f"{float('nan') if a_meas is None else a_meas:.6f}"]
            row += [f"{thrust[0]:.6f}", int(bool(thrust[1])), f"{out.slack:.6e}", int(out.qp_used), f"{out.safeguard_lambda:.3f}",
                    f"{out.h_cone:.6f}", f"{out.height:.6f}", int(out.in_base_set), f"{out.latency_s * 1e3:.3f}"]
            row += [f"{v:.6f}" for v in out.info.get("d_hat", np.zeros(3))]
        self._csv.writerow(row)

    def _policy_stats(self, now: float) -> dict:
        s = super()._policy_stats(now)
        r = self._rec
        if r["dhat"]:
            dh = np.asarray(r["dhat"])
            u = np.asarray(r["u"])
            s.update({"d_hat_final": dh[-1].tolist(), "d_hat_max_norm": float(np.linalg.norm(dh, axis=1).max()),
                      "d_hat_median_norm": float(np.median(np.linalg.norm(dh, axis=1))),
                      "max_thrust_cmd_m_s2": float(u[:, 0].max()), "max_rate_cmd_rad_s": float(np.abs(u[:, 1:4]).max()),
                      "predict_ahead_s": float(self.rt.cfg.predict_ahead)})
        return s

    def _stat_lines(self) -> list[str]:
        lines = super()._stat_lines()
        s = self.stats
        if s.get("d_hat_final") is not None:
            lines.append(f"  commands: thrust up to {s['max_thrust_cmd_m_s2']:.1f} m/s^2, rates up to {s['max_rate_cmd_rad_s']:.1f} rad/s; "
                         f"disturbance estimate median |d_hat| {s['d_hat_median_norm']:.2f}, largest {s['d_hat_max_norm']:.2f} m/s^2")
        return lines

    def _write_summary(self, now: float, x_local) -> None:
        r, s = self._rec, self.stats
        if s.get("steps") and "rate_response" not in s and len(r["t"]) >= 30:
            try:
                fit = fit_rate_response(np.asarray(r["t"]), np.asarray(r["u"])[:, 1:4], np.asarray(r["omega"]))
                s["rate_response"] = fit
                if fit:
                    self._info(f"roll/pitch rate response: delay {fit['delay_s'] * 1e3:.0f} ms + first-order lag {fit['tau_s'] * 1e3:.0f} ms "
                               f"= effective delay {fit['effective_delay_s'] * 1e3:.0f} ms (fit error {fit['rms_rad_s']:.2f} rad/s RMS); "
                               f"delay compensation in this flight {s.get('predict_ahead_s', 0.0) * 1e3:.0f} ms - set predict_ahead to "
                               f"about the effective delay")
            except Exception as exc:   # a diagnostic; never let it break the records
                self._warn(f"rate-response fit failed: {exc!r}")
        super()._write_summary(now, x_local)
        if self.cfg.log_dir and getattr(self, "summary", None) is not None:
            self.summary["policy_run"] = {"run": self.rt.cfg.run, "weights": self.rt.cfg.weights, "observer": self.rt.cfg.observer,
                                          "observer_lambda": self.rt.observer_lambda, "predict_ahead": self.rt.cfg.predict_ahead,
                                          "reference": self.rt.reference_path, "episode_duration_s": self.rt.episode_duration}
            import json
            (Path(self.cfg.log_dir) / "summary.json").write_text(json.dumps(self.summary, indent=2, default=float) + "\n", encoding="utf-8")


__all__ = ["LandingPolicyFlight", "LandingPolicyFlightConfig", "POLICY_CSV_COLUMNS"]
