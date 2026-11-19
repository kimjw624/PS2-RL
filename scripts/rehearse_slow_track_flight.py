#!/usr/bin/env python
"""Rehearse a slow-track flight offline: the flight logic and runtime the ROS node runs, against the
stand-in vehicle of ``ps2rl/deploy/landing_plant_sim.py`` (rate / thrust lag, delay, thrust-map error,
estimate noise, an optional push, PX4-like position control and modes). No PX4, Gazebo or ROS needed.

    python scripts/rehearse_slow_track_flight.py --controller pd --out outputs/slow_track_rehearsal/pd
    python scripts/rehearse_slow_track_flight.py --controller pd --sweep
    python scripts/rehearse_slow_track_flight.py --controller policy --run outputs/slow_track/slow_loop_s0 --sweep

``--plant KEY=VAL`` / ``--flight KEY=VAL`` set fields of ``PlantConfig`` / ``SlowFlightConfig``.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

from ps2rl.deploy.landing_plant_sim import FakeVehicle, PlantConfig  # noqa: E402
from ps2rl.deploy.slow_track_flight import Phase, SlowFlightConfig, SlowTrackFlight  # noqa: E402
from ps2rl.deploy.slow_track_runtime import SlowTrackRuntime, SlowTrackRuntimeConfig  # noqa: E402
from rehearse_landing_flight import _typed  # noqa: E402

SWEEP = [
    ("ideal plant", {"tau_rate_xy": 0.0, "tau_rate_z": 0.0, "tau_thrust": 0.0, "command_delay": 0.0,
                     "pos_noise_std": 0.0, "vel_noise_std": 0.0}),
    ("default (lags, 20 ms delay, noise)", {}),
    ("slow rates (0.10 s, yaw 0.40 s)", {"tau_rate_xy": 0.10, "tau_rate_z": 0.40}),
    ("delay 0.06 s", {"command_delay": 0.06}),
    ("thrust 10 % high", {"thrust_scale": 1.10}),
    ("thrust 10 % low", {"thrust_scale": 0.90}),
    ("push 0.5 m/s^2 sideways", {"ext_accel": (0.0, 0.5, 0.0)}),
    ("sinusoid 0.5 m/s^2, 0.1 Hz", {"ext_amp": 0.5, "ext_freq_hz": 0.1, "ext_dir": (1.0, 0.0, 0.0)}),
    ("noisy estimate (1.5 cm, 0.05 m/s)", {"pos_noise_std": 0.015, "vel_noise_std": 0.05}),
    ("all: slow, 60 ms, 10 % low, push", {"tau_rate_xy": 0.10, "tau_rate_z": 0.40, "tau_thrust": 0.08, "command_delay": 0.06,
                                          "thrust_scale": 0.90, "ext_accel": (0.0, 0.5, 0.0)}),
]


def fly(rt: SlowTrackRuntime, fcfg: SlowFlightConfig, pcfg: PlantConfig, *, max_time: float, quiet: bool) -> dict:
    class _Quiet:
        info = staticmethod(lambda m: None)
        warn = staticmethod(lambda m: print("  ! " + m))

    veh = FakeVehicle(pcfg)
    true_ground = veh.p.copy()
    flight = SlowTrackFlight(rt, fcfg, logger=_Quiet() if quiet else None)
    true_err = []
    while flight.phase is not Phase.DONE and veh.t < max_time:
        x, omega, a = veh.estimate()
        flight.update_status(armed=veh.armed, offboard=veh.offboard, landed=veh.landed if flight.phase is Phase.LAND else None,
                             landing=veh.nav == "land")
        was = flight.phase is Phase.POLICY
        t_pol = veh.t - flight._t0_policy
        cmd = flight.step(veh.t, x, omega_meas=omega, a_meas=a)
        if was:   # true tracking error in the task frame
            p_true = veh.p - (true_ground + np.asarray(fcfg.pad_offset)) + rt.pad
            true_err.append(float(np.linalg.norm(p_true - rt.nominal_goal(t_pol))))
        veh.apply(cmd)
        veh.advance(rt.dt)
    flight.close()
    s = flight.stats
    return {"outcome": flight.outcome or f"did not finish (phase {flight.phase.name})", "finished": flight.phase is Phase.DONE,
            "landed": bool(veh.landed), "est_err_mean": s.get("pos_err_mean_m"), "true_err_mean": float(np.mean(true_err)) if true_err else None,
            "true_err_max": float(np.max(true_err)) if true_err else None, "max_tilt": s.get("max_tilt_deg"),
            "thrust_range": s.get("thrust_cmd_range_m_s2"), "max_rate": s.get("max_rate_cmd_rad_s"), "du": s.get("command_change_mean"),
            "d_hat_final": s.get("d_hat_final"), "stats": s}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--controller", default="pd", choices=("pd", "policy"))
    p.add_argument("--run", default="", help="policy: run directory of train_slow_track.py")
    p.add_argument("--weights", default="best", choices=("best", "final"))
    p.add_argument("--reference", default="", help="pd: reference .npz (default: the slow-loop reference)")
    p.add_argument("--observer", default="commanded", choices=("commanded", "measured", "off"))
    p.add_argument("--handover_height", type=float, default=2.0)
    p.add_argument("--thrust_max", type=float, default=16.6, help="physical thrust limit of the stand-in [m/s^2] (stock x500)")
    p.add_argument("--plant", nargs="*", default=[], metavar="KEY=VAL")
    p.add_argument("--flight", nargs="*", default=[], metavar="KEY=VAL")
    p.add_argument("--sweep", action="store_true")
    p.add_argument("--max_time", type=float, default=120.0)
    p.add_argument("--out", default="")
    args = p.parse_args(argv)

    rt = SlowTrackRuntime(SlowTrackRuntimeConfig(controller=args.controller, run=args.run, weights=args.weights,
                                                 reference_path=args.reference, observer=args.observer,
                                                 handover_height=args.handover_height))
    f_over = _typed(SlowFlightConfig, args.flight)
    plant0 = replace(PlantConfig(gravity=rt.gravity, thrust_accel_min=0.0, thrust_accel_max=args.thrust_max),
                     **_typed(PlantConfig, args.plant))
    if not args.sweep:
        res = fly(rt, SlowFlightConfig(**{**f_over, "log_dir": args.out}), plant0, max_time=args.max_time, quiet=False)
        print(f"truth: tracking error mean {res['true_err_mean']:.3f} m, max {res['true_err_max']:.3f} m; landed {res['landed']}")
        return
    rows = []
    print(f"\n{'plant':38s} {'outcome':46s} {'err mean/max [cm]':>18s} {'tilt':>5s} {'thrust':>12s} {'|w|max':>6s} {'du':>6s}  landed")
    for label, over in SWEEP:
        res = fly(rt, SlowFlightConfig(**f_over), replace(plant0, **over), max_time=args.max_time, quiet=True)
        rows.append({"plant": label, **{k: v for k, v in res.items() if k != "stats"}, "stats": res["stats"]})
        f = lambda v, spec: "   -" if v is None else format(v, spec)  # noqa: E731
        tr = res["thrust_range"] or [float("nan")] * 2
        em = None if res["true_err_mean"] is None else 100 * res["true_err_mean"]
        ex = None if res["true_err_max"] is None else 100 * res["true_err_max"]
        print(f"{label:38s} {res['outcome'][:46]:46s} {f(em, '8.1f')}/{f(ex, '7.1f')} {f(res['max_tilt'], '5.1f')} "
              f"{tr[0]:5.2f}-{tr[1]:5.2f} {f(res['max_rate'], '6.2f')} {f(res['du'], '6.3f')}  {res['landed']}")
    if args.out:
        Path(args.out).mkdir(parents=True, exist_ok=True)
        (Path(args.out) / "sweep.json").write_text(json.dumps(rows, indent=1, default=float) + "\n", encoding="utf-8")
        print(f"\nwrote {Path(args.out) / 'sweep.json'}")


if __name__ == "__main__":
    main()
