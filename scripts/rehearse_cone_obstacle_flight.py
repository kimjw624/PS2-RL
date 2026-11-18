#!/usr/bin/env python
"""Rehearse an outside-cone flight offline: the flight logic and the runtime that the ROS node runs,
against a stand-in vehicle with actuator lag, delay, thrust-map error, estimate noise and an optional
external push (``ps2rl/deploy/landing_plant_sim.py``). No PX4, Gazebo or ROS needed.

    # one flight (Phase-II policy through the UE-bCBF), records in a folder
    python scripts/rehearse_cone_obstacle_flight.py --run outputs/cone_obstacle_phase2_ue/cone_ue_ps2_s0 \\
        --backup_from outputs/cone_obstacle_phase1/compare/chosen_backup.json --out outputs/cone_obstacle_rehearsal/one

    # the built-in set of plant variations
    python scripts/rehearse_cone_obstacle_flight.py --run ... --backup_from ... --sweep --out outputs/cone_obstacle_rehearsal/sweep

``--plant KEY=VAL`` and ``--flight KEY=VAL`` set fields of ``PlantConfig`` and ``ConeFlightConfig``.
The table reports what the controller saw (estimates) and what happened (truth, in the task frame):
the smallest true funnel and floor margins during the policy episode and whether the target was touched.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

from ps2rl.deploy.cone_obstacle_flight import ConeFlightConfig, ConeObstacleFlight, Phase  # noqa: E402
from ps2rl.deploy.cone_obstacle_runtime import ConeObstacleRuntime, ConeObstacleRuntimeConfig  # noqa: E402
from ps2rl.deploy.landing_plant_sim import FakeVehicle, PlantConfig  # noqa: E402
from rehearse_landing_flight import _typed  # noqa: E402

SWEEP = [  # (label, plant overrides)
    ("ideal plant (no lag, delay, noise)", {"tau_rate_xy": 0.0, "tau_rate_z": 0.0, "tau_thrust": 0.0, "command_delay": 0.0,
                                            "pos_noise_std": 0.0, "vel_noise_std": 0.0}),
    ("default plant (lags, 20 ms delay, noise)", {}),
    ("slow rates (0.10 s, yaw 0.40 s)", {"tau_rate_xy": 0.10, "tau_rate_z": 0.40}),
    ("delay 0.06 s", {"command_delay": 0.06}),
    ("thrust 10 % high", {"thrust_scale": 1.10}),
    ("thrust 10 % low", {"thrust_scale": 0.90}),
    ("push 0.5 m/s^2 into the wall", {"ext_accel": (0.354, 0.0, 0.354)}),
    ("push 1.0 m/s^2 into the wall", {"ext_accel": (0.707, 0.0, 0.707)}),
    ("sinusoid 0.5 m/s^2, 0.05 Hz, sideways", {"ext_amp": 0.5, "ext_freq_hz": 0.05, "ext_dir": (0.0, 1.0, 0.0)}),
    ("noisy estimate (1.5 cm, 0.05 m/s)", {"pos_noise_std": 0.015, "vel_noise_std": 0.05}),
    ("all of it: slow, delayed, 10 % low, push 0.5", {"tau_rate_xy": 0.10, "tau_rate_z": 0.40, "tau_thrust": 0.08, "command_delay": 0.06,
                                                      "thrust_scale": 0.90, "ext_accel": (0.354, 0.0, 0.354)}),
]


def fly(rt: ConeObstacleRuntime, fcfg: ConeFlightConfig, pcfg: PlantConfig, *, max_time: float, quiet: bool) -> dict:
    class _Quiet:
        info = staticmethod(lambda m: None)
        warn = staticmethod(lambda m: print("  ! " + m))

    veh = FakeVehicle(pcfg)
    true_ground = veh.p.copy()
    flight = ConeObstacleFlight(rt, fcfg, logger=_Quiet() if quiet else None)
    true_h, true_z, touched_true = [], [], False
    while flight.phase is not Phase.DONE and veh.t < max_time:
        x, omega, a = veh.estimate()
        flight.update_status(armed=veh.armed, offboard=veh.offboard, landed=veh.landed if flight.phase is Phase.LAND else None,
                             landing=veh.nav == "land")
        was_policy = flight.phase is Phase.POLICY
        cmd = flight.step(veh.t, x, omega_meas=omega, a_meas=a)
        if was_policy:   # truth: the task frame as it really is (true ground point + offset)
            x_true = np.r_[veh.p - (true_ground + np.asarray(fcfg.pad_offset)) + rt.pad, veh.v, veh.q]
            h, z = rt.margins(x_true)
            true_h.append(h)
            true_z.append(z)
            touched_true = touched_true or rt.touches(x_true)
        veh.apply(cmd)
        veh.advance(rt.dt)
    flight.close()
    s = flight.stats
    return {
        "outcome": flight.outcome or f"did not finish (phase {flight.phase.name})", "finished": flight.phase is Phase.DONE,
        "duration_s": s.get("duration_s"), "touched": s.get("touched"), "touched_true": touched_true, "time_to_touch_s": s.get("time_to_touch_s"),
        "est_min_h_cone": s.get("min_h_cone_m"), "true_min_h_cone": float(min(true_h)) if true_h else None,
        "est_min_floor": s.get("min_floor_margin_m"), "true_min_floor": float(min(true_z)) if true_z else None,
        "max_slack": s.get("max_slack"), "qp_fallbacks": s.get("qp_fallback_steps"), "safeguard_steps": s.get("safeguard_steps"),
        "filter_correction": s.get("filter_correction_mean"), "max_tilt": s.get("max_tilt_deg"), "d_hat_final": s.get("d_hat_final"),
        "latency_ms": s.get("latency_ms"), "stats": s,
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True, help="Phase-II run directory (the policy)")
    p.add_argument("--weights", default="best", choices=("best", "final"))
    p.add_argument("--filter", default="ue", choices=("ue", "nominal", "none"))
    p.add_argument("--backup_from", default="", help="chosen_backup.json (default: the run's own backup)")
    p.add_argument("--qp_solve_dtype", default="float64", choices=("float32", "float64"))
    p.add_argument("--observer", default="measured", choices=("measured", "commanded", "off"))
    p.add_argument("--predict_ahead", type=float, default=0.0, help="delay compensation of the runtime [s] (0 = off)")
    p.add_argument("--thrust_max", type=float, default=60.0, help="physical thrust-acceleration limit of the stand-in [m/s^2]")
    p.add_argument("--plant", nargs="*", default=[], metavar="KEY=VAL")
    p.add_argument("--flight", nargs="*", default=[], metavar="KEY=VAL")
    p.add_argument("--sweep", action="store_true", help="run the built-in plant variations")
    p.add_argument("--max_time", type=float, default=90.0)
    p.add_argument("--out", default="", help="folder for flight.csv and summary.json (one flight), or the sweep table")
    args = p.parse_args(argv)

    rt = ConeObstacleRuntime(ConeObstacleRuntimeConfig(run=args.run, weights=args.weights, filter=args.filter,
                                                       backup_from=args.backup_from, qp_solve_dtype=args.qp_solve_dtype,
                                                       observer=args.observer, predict_ahead=args.predict_ahead))
    f_over = _typed(ConeFlightConfig, args.flight)
    plant0 = replace(PlantConfig(gravity=rt.gravity, thrust_accel_min=0.0, thrust_accel_max=args.thrust_max), **_typed(PlantConfig, args.plant))
    print(f"plant: rate lag {plant0.tau_rate_xy:g}/{plant0.tau_rate_z:g} s, thrust lag {plant0.tau_thrust:g} s, delay {plant0.command_delay:g} s, "
          f"thrust range 0-{plant0.thrust_accel_max:g} m/s^2")

    if not args.sweep:
        res = fly(rt, ConeFlightConfig(**{**f_over, "log_dir": args.out}), plant0, max_time=args.max_time, quiet=False)
        print(f"truth: funnel margin min {res['true_min_h_cone']:+.3f} m, floor margin min {res['true_min_floor']:+.3f} m, "
              f"touched {res['touched_true']}")
        return

    rows = []
    print("columns: funnel / floor = smallest margin during the episode, estimated / true [m]; touch = time of the first touch [s]; "
          "corr = mean filter correction; fb = QP fallbacks; sg = safeguard blends")
    print(f"\n{'plant':46s} {'outcome':40s} {'touch':>6s} {'funnel est/true':>16s} {'floor est/true':>16s} {'corr':>5s} {'tilt':>5s} "
          f"{'slack':>8s} {'fb':>3s} {'sg':>4s} {'lat p95':>8s}")
    for label, over in SWEEP:
        res = fly(rt, ConeFlightConfig(**f_over), replace(plant0, **over), max_time=args.max_time, quiet=True)
        rows.append({"plant": label, "overrides": {k: list(v) if isinstance(v, tuple) else v for k, v in over.items()},
                     **{k: v for k, v in res.items() if k != "stats"}, "stats": res["stats"]})
        f = lambda v, spec: "   -" if v is None else format(v, spec)  # noqa: E731
        lat = res["latency_ms"]["p95"] if res.get("latency_ms") else None
        print(f"{label:46s} {res['outcome'][:40]:40s} {f(res['time_to_touch_s'], '6.2f')} {f(res['est_min_h_cone'], '+7.3f')}/{f(res['true_min_h_cone'], '+7.3f')} "
              f"{f(res['est_min_floor'], '+7.3f')}/{f(res['true_min_floor'], '+7.3f')} {f(res['filter_correction'], '5.2f')} {f(res['max_tilt'], '5.0f')} "
              f"{f(res['max_slack'], '8.1e')} {f(res['qp_fallbacks'], '3d')} {f(res['safeguard_steps'], '4d')} {f(lat, '8.1f')}")
    if args.out:
        Path(args.out).mkdir(parents=True, exist_ok=True)
        (Path(args.out) / "sweep.json").write_text(json.dumps(rows, indent=1, default=float) + "\n", encoding="utf-8")
        print(f"\nwrote {Path(args.out) / 'sweep.json'}")


if __name__ == "__main__":
    main()
