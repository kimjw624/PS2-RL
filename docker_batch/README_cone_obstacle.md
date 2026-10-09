# Cone obstacle (outside of a funnel is safe): CPU study

Findings and recommendation: [`cone_obstacle_study.md`](cone_obstacle_study.md).

A drone flies toward a target cylinder under an apex-down funnel; the outside of the funnel
and the space above the ground are safe. The filter is the same backup CBF as the landing
task (rows for the funnel and the ground along a 2 s backup rollout, base-set rows at its end,
QP plus discrete safeguard), with an LQR base controller as the backup (no Phase I).

| file | what |
|---|---|
| `ps2rl/envs/quadrotor_cone_obstacle_config.py` | geometry (tip height, half-angle, tip rounding), base-set design, LQR weights, target cylinder |
| `ps2rl/sets/quadrotor_cone_obstacle_sets.py` | safe set; base sets `band` (PS2-RL paper), `points` (one or a union of hover ellipsoids), `anywhere` (brake: stop wherever clear) |
| `ps2rl/cil/quadrotor_cone_obstacle_cbf.py` | filter config, runtime, recoverability (C_N) and the QP + safeguard |
| `scripts/cone_obstacle_certify.py` | level bounds (c_U, chart, adversarial Lyapunov) of the 9-D / 7-D / 6-D LQRs, geometric caps, brake-set invariance |
| `scripts/cone_obstacle_coverage.py` | C_N maps (x-z through the axis) per design and geometry |
| `scripts/cone_obstacle_reach.py` | filtered closed loop with simple nominals; touch success for every cylinder radius / top |
| `scripts/cone_obstacle_geometry.py` | does the funnel stand between the start and the cylinder at all? |
| `scripts/cone_obstacle_report.py` | markdown tables and success figures |
| `scripts/test_cone_obstacle_sets.py` | self-checks (B in S, gradients, brake set at rest, filter output) |

Design names (`--designs`): `<kind>[:<point>+<point>]@<level>`, level = a number, `max`
(0.98 x min(certified LQR level, geometric cap)) or `max/<k>`. Named points (relative to
the geometry) are in `scripts/cone_obstacle_common.py` (`approach_low`, `approach_beside`,
`under_tip`, `far_low`).

## Commands (from `PS2-RL/`, CPU)

```bash
python scripts/cone_obstacle_certify.py --tip_z 0.8,1.2,1.6 --out outputs/cone_obstacle/certificate.json   # ~30 s
python scripts/test_cone_obstacle_sets.py                                                                # ~1 min
python scripts/cone_obstacle_coverage.py --tip_z 0.8,1.2,1.6 --res 0.1 --out outputs/cone_obstacle/coverage     # ~1 min
python scripts/cone_obstacle_reach.py --tip_z 0.8,1.2,1.6 --out outputs/cone_obstacle/reach                       # ~15 min
python scripts/cone_obstacle_reach.py --tip_z 1.2 --target_offset_x 1.5 --nominals center,top,rim,horizontal \
    --y0 0 --dz0 0.5,1.0 --v0 0,1.0 --duration 10 --top_offsets 0,0.3 --radii 0.25,0.5 --out outputs/cone_obstacle/reach_far
python scripts/cone_obstacle_geometry.py --out outputs/cone_obstacle/geometry                                    # ~2 min
python scripts/cone_obstacle_report.py --reach outputs/cone_obstacle/reach/reach.json --out outputs/cone_obstacle/report
```

The vehicle limits and LQR weights come from the x500 landing config
(`docker_batch/configs/landing_x500_cone45_r0p3.json`, `--vehicle`).
