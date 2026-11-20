#!/usr/bin/env python
"""Rehearse an obstacle-only demo flight offline: the flight logic and runtime the ROS node runs (take-off, hold,
hand-over, the filtered controller, hover, land), against the stand-in vehicle of ``ps2rl/deploy/landing_plant_sim.py``
(rate / thrust lag, delay, thrust-map error, estimate noise, an optional push, PX4-like position control and modes).
No PX4, Gazebo or ROS needed; numpy only (numba if installed).

    python scripts/rehearse_cyl_online_flight.py --course S1 --out outputs/cyl_online_rehearsal/S1
    python scripts/rehearse_cyl_online_flight.py --course O3 --nominal straight
    python scripts/rehearse_cyl_online_flight.py --course S1 --sweep              # plant variations
    python scripts/rehearse_cyl_online_flight.py --all                            # O3 / S1 x planner / straight

``--plant KEY=VAL`` / ``--flight KEY=VAL`` / ``--runtime KEY=VAL`` set fields of ``PlantConfig`` / ``CylFlightConfig`` /
``CylOnlineRuntimeConfig``. With ``--out`` a single flight writes flight.csv and summary.json there (the format of the
SITL records: scripts/plot_cyl_online_flight.py draws either).
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

from ps2rl.deploy.cyl_online_flight import CylFlightConfig, CylOnlineFlight, Phase  # noqa: E402
from ps2rl.deploy.cyl_online_runtime import CylOnlineRuntime, CylOnlineRuntimeConfig, make_runtime  # noqa: E402
from ps2rl.deploy.landing_plant_sim import FakeVehicle, PlantConfig  # noqa: E402
from ps2rl.online.model import surface_distance  # noqa: E402
from rehearse_landing_flight import _typed  # noqa: E402

# (label, PlantConfig fields, CylFlightConfig fields). The noisy cases widen the hand-over speed gate (0.15 m/s): a 5 cm/s
# velocity noise per axis keeps resetting it on the stand-in, which PX4's estimate does not in SITL.
SWEEP = [
    ("ideal plant", {"tau_rate_xy": 0.0, "tau_rate_z": 0.0, "tau_thrust": 0.0, "command_delay": 0.0,
                     "pos_noise_std": 0.0, "vel_noise_std": 0.0}, {}),
    ("default (lags, 20 ms delay, noise)", {}, {}),
    ("slow rates (0.08 s, yaw 0.40 s)", {"tau_rate_xy": 0.08, "tau_rate_z": 0.40}, {}),
    ("delay 0.04 s", {"command_delay": 0.04}, {}),
    ("thrust 8 % high", {"thrust_scale": 1.08}, {}),
    ("thrust 8 % low", {"thrust_scale": 0.92}, {}),
    ("noisy estimate (1.5 cm, 0.05 m/s)", {"pos_noise_std": 0.015, "vel_noise_std": 0.05}, {"start_speed_tol": 0.3}),
    ("all: slow, 40 ms, 8 % low, noisy", {"tau_rate_xy": 0.08, "tau_rate_z": 0.40, "tau_thrust": 0.06, "command_delay": 0.04,
                                          "thrust_scale": 0.92, "pos_noise_std": 0.015, "vel_noise_std": 0.05},
     {"start_speed_tol": 0.3}),
]


def fly(rt: CylOnlineRuntime, fcfg: CylFlightConfig, pcfg: PlantConfig, *, max_time: float, quiet: bool) -> dict:
    class _Quiet:
        info = staticmethod(lambda m: None)
        warn = staticmethod(lambda m: print("  ! " + m))

    veh = FakeVehicle(pcfg)
    true_ground = veh.p.copy()
    flight = CylOnlineFlight(rt, fcfg, logger=_Quiet() if quiet else None)
    true_clear = []
    while flight.phase is not Phase.DONE and veh.t < max_time:
        x, omega, a = veh.estimate()
        flight.update_status(armed=veh.armed, offboard=veh.offboard, landed=veh.landed if flight.phase is Phase.LAND else None,
                             landing=veh.nav == "land")
        was = flight.phase is Phase.POLICY
        cmd = flight.step(veh.t, x, omega_meas=omega, a_meas=a)
        if was:   # true clearance in the task frame
            p_true = veh.p - (true_ground + np.asarray(fcfg.pad_offset)) + rt.pad
            true_clear.append(float(surface_distance(p_true[0:2][None], rt.obst[None])[0].min()) - rt.r_drone)
        veh.apply(cmd)
        veh.advance(rt.dt)
    flight.close()
    s = flight.stats
    return {"outcome": flight.outcome or f"did not finish (phase {flight.phase.name})", "finished": flight.phase is Phase.DONE,
            "landed": bool(veh.landed), "goal_entered_s": s.get("goal_entered_s"),
            "true_min_clear": float(np.min(true_clear)) if true_clear else None, "lam_mean": s.get("lam_mean"),
            "intervene": s.get("intervene_fraction"), "switches": s.get("lam_switches"), "y_range": s.get("y_range_m"),
            "max_tilt": s.get("max_tilt_deg"), "decide_ms": (s.get("decide_ms") or {}).get("max"),
            "step_ms": (s.get("latency_ms") or {}).get("p95"),
            "d_hat_final": s.get("d_hat_final"), "stats": s}


def _row(label: str, res: dict) -> str:
    f = lambda v, spec: "   -" if v is None else format(v, spec)  # noqa: E731
    yr = res["y_range"] or [float("nan")] * 2
    return (f"{label:36s} {res['outcome'][:58]:58s} {f(res['goal_entered_s'], '5.1f')} {f(res['true_min_clear'], '+6.2f')} "
            f"{f(res['lam_mean'], '5.2f')} {f(res['intervene'], '5.2f')} {f(res['switches'], '4d')} {yr[1]:5.2f} "
            f"{f(res['max_tilt'], '5.1f')} {f(res['decide_ms'], '6.1f')} {f(res['step_ms'], '6.1f')}  {res['landed']}")


HEADER = (f"{'case':36s} {'outcome':58s} {'goal':>5s} {'clear':>6s} {'lam':>5s} {'intv':>5s} {'sw':>4s} {'ymax':>5s} "
          f"{'tilt':>5s} {'dec ms':>6s} {'stp95':>6s}  landed")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--course", default="S1")
    p.add_argument("--nominal", default="planner", choices=("planner", "straight"))
    p.add_argument("--backup", default="checkpoints/cyl_online/td3_w128_seed0")
    p.add_argument("--backend", default="auto", choices=("auto", "numpy", "numba"))
    p.add_argument("--thrust_max", type=float, default=16.6, help="physical thrust limit of the stand-in [m/s^2] (stock x500)")
    p.add_argument("--plant", nargs="*", default=[], metavar="KEY=VAL")
    p.add_argument("--flight", nargs="*", default=[], metavar="KEY=VAL")
    p.add_argument("--runtime", nargs="*", default=[], metavar="KEY=VAL")
    p.add_argument("--sweep", action="store_true", help="plant variations for this course and nominal")
    p.add_argument("--all", action="store_true", help="O3 and S1, planner and straight line, default plant")
    p.add_argument("--max_time", type=float, default=120.0)
    p.add_argument("--out", default="")
    args = p.parse_args(argv)

    r_over = _typed(CylOnlineRuntimeConfig, args.runtime)
    f_over = _typed(CylFlightConfig, args.flight)

    def runtime(course: str, nominal: str) -> CylOnlineRuntime:
        return make_runtime(CylOnlineRuntimeConfig(**{"backup": args.backup, "course": course, "nominal": nominal,
                                                      "backend": args.backend, **r_over}))

    rt = runtime(args.course, args.nominal)
    plant0 = replace(PlantConfig(gravity=rt.gravity, thrust_accel_min=0.0, thrust_accel_max=args.thrust_max),
                     **_typed(PlantConfig, args.plant))
    if not (args.sweep or args.all):
        res = fly(rt, CylFlightConfig(**{**f_over, "log_dir": args.out}), plant0, max_time=args.max_time, quiet=False)
        print(f"truth: min clearance {res['true_min_clear']:+.3f} m; landed {res['landed']}")
        return
    rows = []
    print("\n" + HEADER)
    cases = ([(f"{c} {n}", c, n, {}, {}) for c in ("O3", "S1") for n in ("planner", "straight")] if args.all
             else [(f"{args.course} {args.nominal}: {label}", args.course, args.nominal, po, fo) for label, po, fo in SWEEP])
    rt.close()
    for label, course, nominal, over, f_case in cases:
        r = runtime(course, nominal)                    # a fresh controller per flight (the flight closes it)
        res = fly(r, CylFlightConfig(**{**f_over, **f_case}), replace(plant0, **over), max_time=args.max_time, quiet=True)
        rows.append({"case": label, **{k: v for k, v in res.items() if k != "stats"}, "stats": res["stats"]})
        print(_row(label, res), flush=True)
    if args.out:
        Path(args.out).mkdir(parents=True, exist_ok=True)
        (Path(args.out) / "rehearsal.json").write_text(json.dumps(rows, indent=1, default=float) + "\n", encoding="utf-8")
        print(f"\nwrote {Path(args.out) / 'rehearsal.json'}")


if __name__ == "__main__":
    main()
