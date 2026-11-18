#!/usr/bin/env python
"""Rehearse a landing flown by a trained Phase-II policy through the landing filter, offline: the
flight logic and the runtime the ROS node runs (``ps2rl/deploy/landing_policy_*``), against the
stand-in vehicle of ``ps2rl/deploy/landing_plant_sim.py`` (actuator lag, delay, thrust-map error,
estimate noise, optional external push). No PX4, Gazebo or ROS needed.

    python scripts/rehearse_landing_policy_flight.py --run outputs/landing_phase2_ue/ue_p2_warm_s0 \\
        --backup checkpoints/landing_phase1/floor_rec10_td3_seed0 --predict_ahead 0.04 --out outputs/landing_policy_rehearsal/one
    python scripts/rehearse_landing_policy_flight.py ... --sweep        # the built-in plant variations

``--plant KEY=VAL`` and ``--flight KEY=VAL`` set fields of ``PlantConfig`` and ``LandingPolicyFlightConfig``.
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

from ps2rl.deploy.landing_plant_sim import PlantConfig  # noqa: E402
from ps2rl.deploy.landing_policy_flight import LandingPolicyFlight, LandingPolicyFlightConfig  # noqa: E402
from ps2rl.deploy.landing_policy_runtime import LandingPolicyRuntime, LandingPolicyRuntimeConfig  # noqa: E402
from rehearse_landing_flight import _typed, fly  # noqa: E402

SWEEP = [  # (label, plant overrides)
    ("ideal plant (no lag, delay, noise)", {"tau_rate_xy": 0.0, "tau_rate_z": 0.0, "tau_thrust": 0.0, "command_delay": 0.0,
                                            "pos_noise_std": 0.0, "vel_noise_std": 0.0}),
    ("default plant (lags, 20 ms delay, noise)", {}),
    ("small lags (30 ms, 10 ms delay)", {"tau_rate_xy": 0.03, "tau_rate_z": 0.1, "tau_thrust": 0.02, "command_delay": 0.01}),
    ("slow rates (0.10 s, yaw 0.40 s)", {"tau_rate_xy": 0.10, "tau_rate_z": 0.40}),
    ("thrust 5 % high", {"thrust_scale": 1.05}),
    ("thrust 5 % low", {"thrust_scale": 0.95}),
    ("sinusoid 0.5 m/s^2, 0.05 Hz, sideways (training disturbance)", {"ext_amp": 0.5, "ext_freq_hz": 0.05, "ext_dir": (0.0, 1.0, 0.0)}),
    ("push 0.5 m/s^2 outward (-x)", {"ext_accel": (-0.5, 0.0, 0.0)}),
    ("noisy estimate (1.5 cm, 0.05 m/s)", {"pos_noise_std": 0.015, "vel_noise_std": 0.05}),
]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", default="outputs/landing_phase2_ue/ue_p2_warm_s0")
    p.add_argument("--backup", default="checkpoints/landing_phase1/floor_rec10_td3_seed0")
    p.add_argument("--weights", default="best", choices=("best", "final"))
    p.add_argument("--filter", default=LandingPolicyRuntimeConfig.filter)
    p.add_argument("--no_filter", action="store_true", help="the policy alone")
    p.add_argument("--observer", default="measured", choices=("measured", "commanded", "off"))
    p.add_argument("--predict_ahead", type=float, default=0.0)
    p.add_argument("--thrust_max", type=float, default=60.0, help="physical thrust-acceleration limit of the stand-in [m/s^2]")
    p.add_argument("--plant", nargs="*", default=[], metavar="KEY=VAL")
    p.add_argument("--flight", nargs="*", default=[], metavar="KEY=VAL")
    p.add_argument("--sweep", action="store_true")
    p.add_argument("--max_time", type=float, default=90.0)
    p.add_argument("--out", default="")
    a = p.parse_args(argv)

    rt = LandingPolicyRuntime(LandingPolicyRuntimeConfig(run=a.run, backup=a.backup, weights=a.weights, filter=a.filter,
                                                         use_filter=not a.no_filter, observer=a.observer, predict_ahead=a.predict_ahead))
    f_over = _typed(LandingPolicyFlightConfig, a.flight)
    plant0 = replace(PlantConfig(gravity=rt.gravity, thrust_accel_min=0.0, thrust_accel_max=a.thrust_max), **_typed(PlantConfig, a.plant))
    print(f"plant: rate lag {plant0.tau_rate_xy:g}/{plant0.tau_rate_z:g} s, thrust lag {plant0.tau_thrust:g} s, delay {plant0.command_delay:g} s")
    if not a.sweep:
        res = fly(rt, LandingPolicyFlightConfig(**{**f_over, "log_dir": a.out}), plant0, max_time=a.max_time, quiet=False,
                  flight_cls=LandingPolicyFlight)
        print(f"truth: cone margin min {res['true_min_h_cone']:+.3f} m, lowest {res['true_min_height_above_ground']:+.3f} m above ground, "
              f"hardest contact {res['hardest_contact_m_s']:.2f} m/s, final radial {res['final_radial']}")
        return
    rows = []
    print(f"\n{'plant':58s} {'outcome':42s} {'cone min true':>13s} {'lowest':>7s} {'contact':>7s} {'radial':>6s} {'slack':>8s} {'sg':>3s}")
    for label, over in SWEEP:
        res = fly(rt, LandingPolicyFlightConfig(**f_over), replace(plant0, **over), max_time=a.max_time, quiet=True,
                  flight_cls=LandingPolicyFlight)
        rows.append({"plant": label, **{k: v for k, v in res.items() if k != "stats"}, "stats": res["stats"]})
        f = lambda v, spec: "   -" if v is None else format(v, spec)  # noqa: E731
        print(f"{label:58s} {res['outcome'][:42]:42s} {f(res['true_min_h_cone'], '+13.3f')} {f(res['true_min_height_above_ground'], '+7.3f')} "
              f"{f(res['hardest_contact_m_s'], '7.2f')} {f(res['final_radial'], '6.2f')} {f(res['max_slack'], '8.1e')} {f(res['safeguard_steps'], '3d')}")
    if a.out:
        Path(a.out).mkdir(parents=True, exist_ok=True)
        (Path(a.out) / "sweep.json").write_text(json.dumps(rows, indent=1, default=float) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
