#!/usr/bin/env python
"""Rehearse a filtered landing flight offline: the flight logic and the landing runtime that the
ROS node runs, against a stand-in vehicle with actuator lag, delay, thrust-map error and estimate
noise (``ps2rl/deploy/landing_plant_sim.py``). No PX4, Gazebo or ROS needed.

    # one flight with the default plant, records in a folder
    python scripts/rehearse_landing_flight.py --backup checkpoints/landing_phase1/x500_rec10_td3_seed1 \\
        --start 1.65,0,2 --target 0,0,-0.1 --out outputs/landing_rehearsal/one

    # the built-in set of plant variations (ideal plant, lags, delays, thrust and height errors)
    python scripts/rehearse_landing_flight.py --backup checkpoints/landing_phase1/x500_rec10_td3_seed1 --sweep

``--plant KEY=VAL`` and ``--flight KEY=VAL`` set fields of ``PlantConfig`` and ``FlightConfig``.
The table reports what the controller saw (estimates) and what happened (truth): the lowest true
height before the autopilot's land mode took over and the hardest ground contact in that time.
"""

from __future__ import annotations

import argparse
from dataclasses import fields, replace
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from ps2rl.deploy.landing_flight import FlightConfig, LandingFlight, Phase  # noqa: E402
from ps2rl.deploy.landing_plant_sim import FakeVehicle, PlantConfig  # noqa: E402
from ps2rl.deploy.landing_runtime import LandingRuntime, LandingRuntimeConfig  # noqa: E402

SWEEP = [  # (label, plant overrides)
    ("ideal plant (no lag, delay, noise)", {"tau_rate_xy": 0.0, "tau_rate_z": 0.0, "tau_thrust": 0.0, "command_delay": 0.0,
                                            "pos_noise_std": 0.0, "vel_noise_std": 0.0}),
    ("default plant", {}),
    ("slow rates (0.10 s, yaw 0.40 s)", {"tau_rate_xy": 0.10, "tau_rate_z": 0.40}),
    ("slow thrust (0.08 s)", {"tau_thrust": 0.08}),
    ("delay 0.06 s", {"command_delay": 0.06}),
    ("thrust 5 % high", {"thrust_scale": 1.05}),
    ("thrust 5 % low", {"thrust_scale": 0.95}),
    ("thrust 10 % high", {"thrust_scale": 1.10}),
    ("thrust 10 % low", {"thrust_scale": 0.90}),
    ("height estimate drifts up 2 mm/s", {"height_drift": 0.002}),
    ("height estimate drifts down 2 mm/s", {"height_drift": -0.002}),
    ("noisy estimate (1.5 cm, 0.05 m/s)", {"pos_noise_std": 0.015, "vel_noise_std": 0.05}),
    ("all of it: slow, delayed, 5 % high", {"tau_rate_xy": 0.10, "tau_rate_z": 0.40, "tau_thrust": 0.08, "command_delay": 0.06, "thrust_scale": 1.05}),
    ("all of it: slow, delayed, 5 % low", {"tau_rate_xy": 0.10, "tau_rate_z": 0.40, "tau_thrust": 0.08, "command_delay": 0.06, "thrust_scale": 0.95}),
]


def _typed(cls, items: list[str]) -> dict:
    types = {f.name: f.type for f in fields(cls)}
    defaults = {f.name: f.default for f in fields(cls)}
    out = {}
    for item in items:
        k, v = (t.strip() for t in item.split("=", 1))
        if k not in types:
            raise SystemExit(f"'{k}' is not a field of {cls.__name__}")
        d = defaults[k]
        if isinstance(d, bool):
            out[k] = v.lower() in ("1", "true", "yes", "on")
        elif isinstance(d, tuple):
            out[k] = tuple(float(t) for t in v.split(",") if t.strip())
        elif isinstance(d, (int, float)):
            out[k] = type(d)(v)
        else:
            out[k] = v
    return out


def thrust_range(vehicle_json: str) -> tuple[float, float]:
    """Physical thrust-acceleration range from a vehicle constants file (as design_landing_base_set.py reads it)."""
    v = json.loads(Path(vehicle_json).read_text(encoding="utf-8"))
    n = int(v["rotor_count"])
    mass = float(v["mass_base_kg"]) + n * float(v["rotor_mass_kg"])
    k = float(v["motor_constant_N_per_rad2_s2"])
    return n * k * float(v["rotor_speed_min_rad_s"]) ** 2 / mass, n * k * float(v["rotor_speed_max_rad_s"]) ** 2 / mass


def fly(rt: LandingRuntime, fcfg: FlightConfig, pcfg: PlantConfig, *, max_time: float, quiet: bool) -> dict:
    class _Quiet:
        info = staticmethod(lambda m: None)
        warn = staticmethod(lambda m: print("  ! " + m))

    veh = FakeVehicle(pcfg)
    true_ground = veh.p.copy()
    flight = LandingFlight(rt, fcfg, logger=_Quiet() if quiet else None)
    true_h, true_z, t_policy_end, t_policy_start = [], [], None, None
    while flight.phase is not Phase.DONE and veh.t < max_time:
        x, omega, a = veh.estimate()
        flight.update_status(armed=veh.armed, offboard=veh.offboard, landed=veh.landed if flight.phase is Phase.LAND else None,
                             landing=veh.nav == "land")
        was_policy = flight.phase is Phase.POLICY
        cmd = flight.step(veh.t, x, omega_meas=omega, a_meas=a)
        if flight.phase is Phase.POLICY and t_policy_start is None:
            t_policy_start = veh.t
        if was_policy:  # truth: the pad as it really is (true ground point + offset), not as estimated
            x_true = np.r_[veh.p - (true_ground + np.asarray(fcfg.pad_offset)) + rt.pad, veh.v, veh.q]
            h, _ = rt.margins(x_true)
            true_h.append(h)
            true_z.append(float(veh.p[2] - true_ground[2]))
            if flight.phase is not Phase.POLICY:
                t_policy_end = veh.t
        veh.apply(cmd)
        veh.advance(rt.dt)
    flight.close()
    s = flight.stats
    during = [sp for sp in veh.impacts_during(t_policy_start, t_policy_end)] if t_policy_start is not None else []
    return {
        "outcome": flight.outcome or f"did not finish (phase {flight.phase.name})", "finished": flight.phase is Phase.DONE,
        "duration_s": s.get("duration_s"), "est_min_height": s.get("min_height_m"), "est_min_h_cone": s.get("min_h_cone_m"),
        "true_min_height_above_ground": float(min(true_z)) if true_z else None, "true_min_h_cone": float(min(true_h)) if true_h else None,
        "final_true_height_above_ground": float(true_z[-1]) if true_z else None,
        "hardest_contact_m_s": float(max(during)) if during else 0.0, "contacts": len(during),
        "max_slack": s.get("max_slack"), "qp_fallbacks": s.get("qp_fallback_steps"), "safeguard_steps": s.get("safeguard_steps"),
        "final_radial": s.get("final_radial_m"), "rate_rms": s.get("rate_tracking_rms_rad_s"), "stats": s,
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backup", required=True)
    p.add_argument("--filter", default=LandingRuntimeConfig.filter)
    p.add_argument("--no_filter", action="store_true", help="the nominal controller alone")
    p.add_argument("--nominal", default="lqr_target", choices=("lqr_target", "track_reference"))
    p.add_argument("--target", default=LandingRuntimeConfig.target, help="pad_centre, apex or x,y,z relative to the pad centre")
    p.add_argument("--max_position_error", type=float, default=LandingRuntimeConfig.max_position_error,
                   help="cap on the position error the LQR sees [m] (0 = off)")
    p.add_argument("--reference", default="", help="reference .npz for --nominal track_reference")
    p.add_argument("--qp_solve_dtype", default=LandingRuntimeConfig.qp_solve_dtype, choices=("float32", "float64"))
    p.add_argument("--start", default="", help="x,y,z relative to the pad centre (default: see FlightConfig.start_position)")
    p.add_argument("--pad_offset", default="", help="x,y,z of the pad centre relative to the ground point")
    p.add_argument("--vehicle", default="docker_batch/configs/vehicles/gz_x500.json", help="constants file for the physical thrust range")
    p.add_argument("--plant", nargs="*", default=[], metavar="KEY=VAL")
    p.add_argument("--flight", nargs="*", default=[], metavar="KEY=VAL")
    p.add_argument("--sweep", action="store_true", help="run the built-in plant variations")
    p.add_argument("--max_time", type=float, default=120.0)
    p.add_argument("--out", default="", help="folder for flight.csv and summary.json (one flight), or the sweep table")
    args = p.parse_args(argv)

    rt = LandingRuntime(LandingRuntimeConfig(backup=args.backup, filter=args.filter, use_filter=not args.no_filter, nominal=args.nominal,
                                             target=args.target, max_position_error=args.max_position_error, reference_path=args.reference,
                                             qp_solve_dtype=args.qp_solve_dtype))
    f_over = _typed(FlightConfig, args.flight)
    if args.start:
        f_over["start_position"] = tuple(float(t) for t in args.start.split(","))
    if args.pad_offset:
        f_over["pad_offset"] = tuple(float(t) for t in args.pad_offset.split(","))
    a_lo, a_hi = thrust_range(args.vehicle)
    plant0 = replace(PlantConfig(gravity=rt.gravity, thrust_accel_min=a_lo, thrust_accel_max=a_hi), **_typed(PlantConfig, args.plant))
    print(f"plant: thrust range {a_lo:.2f}-{a_hi:.2f} m/s^2, rate lag {plant0.tau_rate_xy:g}/{plant0.tau_rate_z:g} s, thrust lag {plant0.tau_thrust:g} s, "
          f"delay {plant0.command_delay:g} s")

    if not args.sweep:
        fcfg = FlightConfig(**{**f_over, "log_dir": args.out})
        res = fly(rt, fcfg, plant0, max_time=args.max_time, quiet=False)
        print(f"truth: lowest height above the ground {res['true_min_height_above_ground']:+.3f} m, lowest cone margin {res['true_min_h_cone']:+.3f} m, "
              f"hardest ground contact before the land mode {res['hardest_contact_m_s']:.2f} m/s ({res['contacts']} contacts)")
        return

    rows = []
    print("columns: height = lowest height above the pad plane as estimated / above the real ground in truth [m]; cone = lowest cone margin, "
          "estimated / true [m]; contact = hardest ground contact before the land mode [m/s]; fb = QP fallbacks; sg = safeguard blends")
    print(f"\n{'plant':38s} {'outcome':46s} {'time':>5s} {'height est/true':>20s} {'cone est/true':>15s} {'contact':>8s} {'slack':>8s} {'fb':>3s} {'sg':>4s}")
    for label, over in SWEEP:
        res = fly(rt, FlightConfig(**f_over), replace(plant0, **over), max_time=args.max_time, quiet=True)
        rows.append({"plant": label, "overrides": over, **{k: v for k, v in res.items() if k != "stats"}, "stats": res["stats"]})
        f = lambda v, spec: "   –" if v is None else format(v, spec)  # noqa: E731
        print(f"{label:38s} {res['outcome'][:46]:46s} {f(res['duration_s'], '5.2f')} {f(res['est_min_height'], '+9.3f')} /{f(res['true_min_height_above_ground'], '+8.3f')} "
              f"{f(res['est_min_h_cone'], '+7.3f')}/{f(res['true_min_h_cone'], '+7.3f')} {res['hardest_contact_m_s']:8.2f} {f(res['max_slack'], '8.1e')} "
              f"{f(res['qp_fallbacks'], '3d')} {f(res['safeguard_steps'], '4d')}")
    if args.out:
        Path(args.out).mkdir(parents=True, exist_ok=True)
        (Path(args.out) / "sweep.json").write_text(json.dumps(rows, indent=1) + "\n", encoding="utf-8")
        print(f"\nwrote {Path(args.out) / 'sweep.json'}")


if __name__ == "__main__":
    main()
