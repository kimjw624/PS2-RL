#!/usr/bin/env python
"""Re-derive the landing Phase-I design (action box, hover LQR, base-set level) for another vehicle.

The landing defaults assume the powerloop airframe: thrust up to 4 g and body rates up to
18 rad/s. A vehicle with less authority needs three things changed together, and this script
derives them from the vehicle's constants instead of hand-picked numbers:

1. Action box (``--vehicle`` file: masses, motor constants, rotor speed range, geometry).
     thrust   per-motor thrust lies in [T_idle, T_max]; a fraction ``--thrust_reserve`` of that
              range is kept free at both ends for roll/pitch/yaw torque, so the collective
              thrust the policies may command is [n (T_idle + r), n (T_max - r)] / mass
     rates    the model assumes a commanded body rate is reached within a control step, so the
              limit of each axis is the rate the vehicle reaches in ``--rate_rise_time`` with its
              angular acceleration at hover; roll/pitch are also capped by the autopilot's own
              attitude-loop rate limit if the vehicle file gives one. Yaw gets its own limit
              (``omega_z_max``) because yaw torque is usually far weaker.
   ``--a_cmd_min/--a_cmd_max/--omega_max/--omega_z_max`` override any of these.

2. Hover LQR. The input weights of the reference design are rescaled to the new input ranges,
     r_i <- r_i (m_ref,i / m_i)^2,   m = one-sided input margin around hover,
   which keeps the base set's position/velocity size and makes the controller as gentle as the
   smaller inputs require. State weights are kept.
   Yaw, when it has its own limit: the yaw chain is decoupled (phi_z+ = phi_z - dt omega_z), so
   its two weights are set in closed form such that (i) the base set is ``--base_set_yaw_deg`` wide
   in yaw (default: the yaw range of the reference design's region) and (ii) yaw is exactly as
   input-feasible as the most limiting other input, i.e. neither the binding input nor slower
   than it has to be:
     P = c_B / phi_B^2,  K = omega_z_max sqrt(P / c_U,other),  r = P dt / K - P dt^2,  q = P dt K,
   with phi_B = 2 sin(width / 2). Rescaling the yaw weight like the others would instead shrink
   the base set to a few degrees of yaw. A much wider base set is not available either: the slow
   yaw loop leaves almost no Lyapunov decrease to absorb the yaw/tilt coupling (stock x500: 60 deg
   wide still gives c_B = 15.1; at 80 deg the certified level keeps falling and the script stops).

3. Base-set level. c_B = ``--safety_factor`` x the smallest level bound at z_des (input
   feasibility, quaternion chart, adversarial Lyapunov decrease, cone, ground clearance, and the
   gentle-recovery envelope if the design trains with it).

The reference design is a Phase-I ``configs.json`` (``--reference_design``; geometry, floor and
envelope options and the design region are carried over) or the defaults. With
``--task_references`` the design region's speed/height/tilt/yaw ranges are reset to their defaults
and widened only as far as those references need (plus the start perturbation). With a separate
yaw limit the region's yaw range is also capped at the yaw offset the vehicle can remove within
the backup horizon (omega_z_max T), unless the task needs more: larger offsets cannot be undone
in time, they only cost level and training samples.

Output: a config file for ``train_phase1_landing.py --config_json`` / ``certify_landing_base_set.py
--config_json``, with the derivation recorded under ``provenance``.

    JAX_PLATFORMS=cpu python scripts/design_landing_base_set.py \\
        --vehicle docker_batch/configs/vehicles/gz_x500.json \\
        --reference_design checkpoints/landing_phase1/floor_rec10_td3_seed0/configs.json \\
        --task_references ps2rl/envs/assets/quadrotor_landing_b*_reference.npz \\
        --out docker_batch/configs/landing_x500_cone45_r0p3.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ps2rl.base_controller.quadrotor_landing_dlqr import QuadrotorLandingDLQR
from ps2rl.envs.quadrotor_env import quadrotor_step_euler
from ps2rl.envs.quadrotor_landing_config import QuadrotorLandingConfig
from ps2rl.phase1_sa.landing_design_region import LandingDesignRegionConfig
from ps2rl.sets import landing_certificate as lc
from ps2rl.sets.quadrotor_cone_sets import QuadrotorConeSafeSet

R_KEYS = ("lqr_r_a_cmd", "lqr_r_omega_x", "lqr_r_omega_y", "lqr_r_omega_z")
ENVELOPE_KEYS = ("zeta_max", "v_max", "tilt_max_deg", "yaw_max_deg")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--vehicle", default="", help="vehicle constants (JSON), e.g. docker_batch/configs/vehicles/gz_x500.json")
    p.add_argument("--reference_design", default="",
                   help="Phase-I configs.json (or a config file with landing/region blocks) to carry over; default: the code defaults")
    p.add_argument("--thrust_reserve", type=float, default=0.10,
                   help="fraction of the per-motor thrust range kept free for torque at each end of the thrust box")
    p.add_argument("--rate_rise_time", type=float, default=0.04,
                   help="a body-rate limit is the rate reached in this time at hover [s] (default: two 50 Hz control steps)")
    p.add_argument("--a_cmd_min", type=float, default=None)
    p.add_argument("--a_cmd_max", type=float, default=None)
    p.add_argument("--omega_max", type=float, default=None, help="roll/pitch rate limit [rad/s]")
    p.add_argument("--omega_z_max", type=float, default=None, help="yaw rate limit [rad/s]")
    p.add_argument("--z_des", type=float, default=None, help="hover height of the base set (default: the reference design's)")
    p.add_argument("--base_set_yaw_deg", type=float, default=None,
                   help="yaw width of the base set (default: the yaw range of the reference design's region); "
                        "only with a separate yaw limit")
    p.add_argument("--safety_factor", type=float, default=0.77,
                   help="c_B = safety_factor x the smallest level bound (default: the original design's ratio, 12 / 15.67)")
    p.add_argument("--base_set_c", type=float, default=None, help="set the level instead of deriving it (it is still checked)")
    p.add_argument("--task_references", nargs="*", default=[], help="reference .npz files the design region must cover")
    p.add_argument("--task_start_perturbation", default="pz=0.3,v=0.5,tilt=10,yaw=0",
                   help="Phase-II start perturbation added to the reference envelope (m, m/s, deg, deg)")
    p.add_argument("--lyap_starts", type=int, default=1024,
                   help="starts of the adversarial Lyapunov search (certify_landing_base_set.py repeats it at full effort)")
    p.add_argument("--lyap_iters", type=int, default=150, help="gradient-ascent steps of each search")
    p.add_argument("--out", required=True)
    return p.parse_args(argv)


# --------------------------------------------------------------------------- vehicle
def vehicle_limits(v: dict, *, thrust_reserve: float, rise_time: float, gravity: float) -> dict:
    """Thrust box and body-rate limits from the vehicle's constants (see the module docstring)."""
    n = int(v["rotor_count"])
    mass = float(v["mass_base_kg"]) + n * float(v["rotor_mass_kg"])
    k = float(v["motor_constant_N_per_rad2_s2"])
    t_idle, t_max = k * float(v["rotor_speed_min_rad_s"]) ** 2, k * float(v["rotor_speed_max_rad_s"]) ** 2
    reserve = thrust_reserve * (t_max - t_idle)
    pos = np.asarray(v["rotor_positions_m"], dtype=float)
    inertia = np.asarray(v["inertia_base_kgm2"], dtype=float) + float(v["rotor_mass_kg"]) * np.array([
        np.sum(pos[:, 1] ** 2 + pos[:, 2] ** 2), np.sum(pos[:, 0] ** 2 + pos[:, 2] ** 2), np.sum(pos[:, 0] ** 2 + pos[:, 1] ** 2)])
    t_hover = mass * gravity / n
    d_hover = min(t_max - t_hover, t_hover - t_idle)  # largest thrust change per motor that keeps the total at hover

    def accel(d):
        torque = np.array([d * np.sum(np.abs(pos[:, 1])), d * np.sum(np.abs(pos[:, 0])), d * n * float(v["moment_constant_m"])])
        return torque / inertia

    alpha_hover, alpha_reserve = accel(d_hover), accel(reserve)
    rates = alpha_hover * rise_time
    cap = v.get("autopilot_attitude_rate_limit_deg_s")
    notes = {}
    if cap:
        capped = np.minimum(rates[:2], np.radians(np.asarray(cap[:2], dtype=float)))
        if np.any(capped < rates[:2]):
            notes["omega_max"] = "capped by the autopilot's attitude-loop rate limit"
        rates[:2] = capped
    return {
        "mass_kg": mass, "thrust_per_motor_N": [t_idle, t_max], "thrust_reserve_per_motor_N": reserve,
        "max_thrust_acceleration": n * t_max / mass, "idle_thrust_acceleration": n * t_idle / mass,
        "a_cmd_min": n * (t_idle + reserve) / mass, "a_cmd_max": n * (t_max - reserve) / mass,
        "inertia_kgm2": inertia.tolist(),
        "angular_acceleration_at_hover": alpha_hover.tolist(), "angular_acceleration_at_box_ends": alpha_reserve.tolist(),
        "omega_max": float(min(rates[0], rates[1])), "omega_z_max": float(min(rates[2], rates[0], rates[1])),
        "notes": notes,
    }


# --------------------------------------------------------------------------- task envelope
def references_envelope(paths: list[str], pad_z: float) -> dict:
    env = {"zeta_max": 0.0, "speed_max": 0.0, "tilt_max_deg": 0.0, "yaw_abs_max_deg": 0.0}
    for path in paths:
        s = np.asarray(np.load(path)["states"], dtype=np.float64)
        q = s[:, 6:10] / np.linalg.norm(s[:, 6:10], axis=1, keepdims=True)
        tilt = np.degrees(np.arccos(np.clip(1.0 - 2.0 * (q[:, 1] ** 2 + q[:, 2] ** 2), -1.0, 1.0)))
        yaw = np.degrees(np.arctan2(2.0 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]), 1.0 - 2.0 * (q[:, 2] ** 2 + q[:, 3] ** 2)))
        env["zeta_max"] = max(env["zeta_max"], float(np.max(s[:, 2] - pad_z)))
        env["speed_max"] = max(env["speed_max"], float(np.max(np.linalg.norm(s[:, 3:6], axis=1))))
        env["tilt_max_deg"] = max(env["tilt_max_deg"], float(np.max(tilt)))
        env["yaw_abs_max_deg"] = max(env["yaw_abs_max_deg"], float(np.max(np.abs(yaw))))
    return env


def _ceil_to(x: float, step: float) -> float:
    return math.ceil(x / step - 1e-9) * step


def region_for_task(region: dict, envelope: dict, pert: dict, *, yaw_reach_deg: float | None = None) -> tuple[dict, dict]:
    """Reset the envelope ranges to the class defaults, then widen them to cover the task.

    ``yaw_reach_deg`` (vehicles with a separate yaw limit): the yaw offset the vehicle can remove
    within the backup horizon; the yaw range is capped there unless the task needs more.
    """
    defaults = LandingDesignRegionConfig()
    need = {
        "zeta_max": _ceil_to(envelope["zeta_max"] + pert["pz"], 0.5),
        "v_max": _ceil_to(envelope["speed_max"] + pert["v"], 0.5),
        "tilt_max_deg": _ceil_to(envelope["tilt_max_deg"] + pert["tilt"], 5.0),
        "yaw_max_deg": _ceil_to(envelope["yaw_abs_max_deg"] + pert["yaw"], 5.0),
    }
    out, reasons = dict(region), {}
    for k, v in need.items():
        d = float(getattr(defaults, k))
        out[k] = max(d, float(v))
        why = f"default {d} (task needs {v})" if v <= d + 1e-9 else f"widened from the default {d} to {v} for the task"
        if k == "yaw_max_deg" and yaw_reach_deg is not None:
            cap = _ceil_to(yaw_reach_deg, 5.0)
            if max(cap, float(v)) < out[k]:
                out[k] = max(cap, float(v))
                why = (f"capped at what the vehicle can yaw back within the backup horizon ({yaw_reach_deg:.1f} deg); "
                       f"default {d}, task needs {v}")
        was = f" (the reference design had {region[k]})" if k in region and float(region[k]) != out[k] else ""
        reasons[k] = why + was
    return out, reasons


# --------------------------------------------------------------------------- certificate
def yaw_phi(yaw_deg: float) -> float:
    """Error-quaternion coordinate phi_z of a pure yaw offset."""
    return 2.0 * math.sin(math.radians(yaw_deg) / 2.0)


def yaw_weights(*, c_b: float, width_deg: float, omega_z_max: float, c_u_other: float, dt: float) -> tuple[float, float]:
    """(q_thetaz, r_omega_z) of the decoupled yaw chain: base set ``width_deg`` wide in yaw, yaw as feasible as c_u_other."""
    p = c_b / yaw_phi(width_deg) ** 2
    k = omega_z_max * math.sqrt(p / c_u_other)
    if not 0.0 < k * dt < 1.0:
        raise SystemExit(f"yaw gain {k:g} 1/s is not realisable at dt = {dt:g} s")
    return p * dt * k, p * dt / k - p * dt * dt


def level_bounds(cfg: QuadrotorLandingConfig, *, lyap_starts: int, lyap_iters: int, yaw_pending: bool = False) -> dict:
    """Level bounds at cfg.z_des. ``yaw_pending``: the yaw weights are not set yet, so leave out what
    depends on them (yaw input feasibility and the Lyapunov search); the yaw chain is decoupled in the
    linearisation, so the remaining bounds do not change when the yaw weights do, except c_chart."""
    cone = QuadrotorConeSafeSet.from_config(cfg)
    ctrl = QuadrotorLandingDLQR.from_config(cfg)
    p_inv = np.linalg.inv(ctrl.p_matrix_f64())
    k = ctrl.k_matrix_f64()
    omega_clip = cfg.omega_clip if isinstance(cfg.omega_clip, float) else jnp.asarray(cfg.omega_clip)
    step = lambda x, u: quadrotor_step_euler(x, u, cfg.dt, cfg.gravity, cfg.a_cmd_min, cfg.a_cmd_max, omega_clip)  # noqa: E731
    per_input = [float(m) ** 2 / float(row @ p_inv @ row) for m, row in zip(ctrl._input_margins(), k)]
    b = {"c_U": min(per_input[:3]) if yaw_pending else float(ctrl.max_certified_level)}
    if not yaw_pending:
        b["c_chart"] = lc.c_chart_bound(p_inv)
    c_hi = min(b.values())
    if not yaw_pending:
        b["c_Lyap"] = lc.c_lyap_adversarial(ctrl, step, c_hi=c_hi, n_starts=lyap_starts, ascent_iters=lyap_iters)
    b["c_ground"] = lc.c_ground_bound(p_inv, z_des=cfg.z_des, z_clear=cfg.z_clear)
    b["c_cone"] = lc.c_cone_exact(p_inv, cone, z_des=cfg.z_des, c_hi=10.0 * c_hi)
    if float(cfg.recovery_rate_floor) > 0.0:
        b["c_recovery_floor"] = lc.c_recovery_floor(p_inv, z_des=cfg.z_des, kappa=float(cfg.recovery_rate_floor))
    if float(cfg.recovery_rate_cone) > 0.0:
        b["c_recovery_cone"] = lc.c_recovery_cone(p_inv, cone, z_des=cfg.z_des, kappa=float(cfg.recovery_rate_cone), c_hi=10.0 * c_hi)
    spectral = float(np.abs(np.linalg.eigvals(np.asarray(ctrl.a_d) - np.asarray(ctrl.b_d) @ k)).max())
    return {"bounds": b, "c_bar": float(min(b.values())), "binding": min(b, key=b.get), "c_U_per_input": per_input,
            "spectral_radius": spectral, "p_inv": p_inv, "k_yaw": abs(float(k[3, 8]))}


def extents(p_inv: np.ndarray, c: float) -> dict:
    rho = lc.support_radii(p_inv, c)
    ang = lambda phi: float(np.degrees(2.0 * np.arcsin(min(1.0, 0.5 * phi))))  # noqa: E731
    return {"dx_m": rho["dx"], "dz_m": rho["dz"], "vx_m_s": rho["vx"], "vz_m_s": rho["vz"],
            "tilt_deg": ang(rho["phix"]), "yaw_deg": ang(rho["phiz"])}


def main(argv=None):
    args = parse_args(argv)
    # ---- reference design
    ref_landing, ref_region, ref_name = {}, {}, "code defaults"
    if args.reference_design:
        d = json.loads(Path(args.reference_design).read_text(encoding="utf-8"))
        ref_landing = dict(d.get("landing", {}))
        ref_region = dict(d.get("design_region", d.get("region", {})))
        ref_name = args.reference_design
    ref = QuadrotorLandingConfig().replace(**ref_landing)
    ref_ctrl = QuadrotorLandingDLQR.from_config(ref)
    print(f"[reference design] {ref_name}: thrust [{ref.a_cmd_min:g}, {ref.a_cmd_max:g}] m/s^2, rates "
          f"{', '.join(f'{w:g}' for w in ref.omega_limits)} rad/s, z_des {ref.z_des:g}, c_B {ref.base_set_c:g}")

    # ---- action box
    box = {"a_cmd_min": ref.a_cmd_min, "a_cmd_max": ref.a_cmd_max, "omega_max": ref.omega_max, "omega_z_max": ref.omega_z_max}
    veh = None
    if args.vehicle:
        veh_cfg = json.loads(Path(args.vehicle).read_text(encoding="utf-8"))
        veh = vehicle_limits(veh_cfg, thrust_reserve=args.thrust_reserve, rise_time=args.rate_rise_time, gravity=ref.gravity)
        box.update({k: veh[k] for k in box})
        print(f"[vehicle] {veh_cfg.get('name', args.vehicle)}: mass {veh['mass_kg']:.3f} kg, thrust acceleration "
              f"{veh['idle_thrust_acceleration']:.2f}-{veh['max_thrust_acceleration']:.2f} m/s^2; angular acceleration at hover "
              f"(roll, pitch, yaw) {', '.join(f'{a:.1f}' for a in veh['angular_acceleration_at_hover'])} rad/s^2, "
              f"at the ends of the thrust box {', '.join(f'{a:.1f}' for a in veh['angular_acceleration_at_box_ends'])} rad/s^2")
    for k in box:
        if getattr(args, k) is not None:
            box[k] = float(getattr(args, k))
    box = {k: float(np.round(v, 3)) for k, v in box.items()}
    if box["omega_z_max"] >= box["omega_max"]:
        box["omega_z_max"] = 0.0  # no separate yaw limit needed
    print(f"[action box] thrust [{box['a_cmd_min']:g}, {box['a_cmd_max']:g}] m/s^2, roll/pitch rate {box['omega_max']:g} rad/s, "
          f"yaw rate {box['omega_z_max'] or box['omega_max']:g} rad/s")

    # ---- design region (needed for the yaw rule)
    region, region_reasons, envelope = dict(ref_region), {}, None
    if args.task_references:
        pert = {k: float(v) for k, v in (kv.split("=") for kv in args.task_start_perturbation.split(","))}
        envelope = references_envelope(args.task_references, ref.pad_z)
        yaw_reach = math.degrees(box["omega_z_max"] * float(ref.T)) if box["omega_z_max"] > 0.0 else None
        region, region_reasons = region_for_task(region, envelope, pert, yaw_reach_deg=yaw_reach)
        print(f"[design region] task envelope over {len(args.task_references)} references: height {envelope['zeta_max']:.2f} m, "
              f"speed {envelope['speed_max']:.2f} m/s, tilt {envelope['tilt_max_deg']:.1f} deg, yaw {envelope['yaw_abs_max_deg']:.1f} deg")
        for k in ENVELOPE_KEYS:
            print(f"            {k}: {region[k]:g}  ({region_reasons[k]})")
    yaw_range_deg = float(region.get("yaw_max_deg", LandingDesignRegionConfig().yaw_max_deg))
    base_yaw_deg = float(args.base_set_yaw_deg) if args.base_set_yaw_deg is not None else float(
        ref_region.get("yaw_max_deg", LandingDesignRegionConfig().yaw_max_deg))
    if not 0.0 < base_yaw_deg < 180.0:
        raise SystemExit(f"--base_set_yaw_deg must lie in (0, 180), got {base_yaw_deg}")

    # ---- hover LQR: input weights rescaled to the new input ranges
    new = ref.replace(**box, **({"z_des": args.z_des} if args.z_des is not None else {}))
    m_ref = ref_ctrl._input_margins()
    u_star = np.asarray(ref_ctrl.u_star)
    m_new = np.minimum(u_star - np.asarray(new.action_low), np.asarray(new.action_high) - u_star)
    if np.any(m_new <= 0.0):
        raise SystemExit(f"hover is not strictly inside the action box (margins {m_new.tolist()})")
    scale = (m_ref / m_new) ** 2
    weights = {k: float(getattr(ref, k)) * float(sc) for k, sc in zip(R_KEYS, scale)}
    separate_yaw = float(new.omega_z_max) > 0.0
    if separate_yaw:
        weights["lqr_r_omega_z"] = float(ref.lqr_r_omega_z)  # set by the yaw rule below
    new = new.replace(**weights)
    print("[hover LQR] input margins (thrust, roll, pitch, yaw): reference " + ", ".join(f"{v:g}" for v in m_ref)
          + " -> " + ", ".join(f"{v:g}" for v in m_new))

    # ---- level (and, with a separate yaw limit, the yaw weights, which depend on the level)
    t0 = time.time()
    safety = float(args.safety_factor)

    def pick_level(res):
        return float(args.base_set_c) if args.base_set_c is not None else float(np.floor(10.0 * safety * res["c_bar"]) / 10.0)

    def with_yaw(cfg, c_b, res):
        q, r = yaw_weights(c_b=c_b, width_deg=base_yaw_deg, omega_z_max=float(cfg.omega_z_max),
                           c_u_other=min(res["c_U_per_input"][:3]), dt=float(cfg.dt))
        return cfg.replace(lqr_q_thetaz=q, lqr_r_omega_z=r)

    if separate_yaw:  # first pass: a level from the bounds that do not depend on the yaw weights
        res = level_bounds(new, lyap_starts=args.lyap_starts, lyap_iters=args.lyap_iters, yaw_pending=True)
        new = with_yaw(new, pick_level(res), res)
    for _ in range(8):  # the yaw weights depend on the level and (through the Lyapunov bound) the level on them
        res = level_bounds(new, lyap_starts=args.lyap_starts, lyap_iters=args.lyap_iters)
        c_b = pick_level(res)
        if not separate_yaw:
            break
        prev = (new.lqr_q_thetaz, new.lqr_r_omega_z)
        new = with_yaw(new, c_b, res)
        if np.allclose(prev, (new.lqr_q_thetaz, new.lqr_r_omega_z), rtol=1e-3):
            break
    else:
        raise SystemExit("the yaw weights and the level did not settle; set --base_set_c or a narrower --base_set_yaw_deg")
    new = new.replace(base_set_c=c_b)
    print("            input weights: " + ", ".join(f"{k[6:]} {getattr(ref, k):g} -> {getattr(new, k):.4g}" for k in R_KEYS)
          + (f"; yaw state weight {ref.lqr_q_thetaz:g} -> {new.lqr_q_thetaz:.4g} (yaw rule, base set {base_yaw_deg:g} deg wide)" if separate_yaw else ""))
    ok = c_b <= res["c_bar"] + 1e-9
    print(f"[level bounds at z_des = {new.z_des:g}] " + ", ".join(f"{k} {v:.2f}" for k, v in res["bounds"].items())
          + f" ({time.time() - t0:.0f} s)")
    print("            input feasibility per input (thrust, roll, pitch, yaw): " + ", ".join(f"{v:.1f}" for v in res["c_U_per_input"])
          + f"; closed-loop spectral radius {res['spectral_radius']:.4f}")
    print(f"[level] c_B = {c_b:g} = {c_b / res['c_bar']:.2f} x c_bar ({res['c_bar']:.2f}, binding: {res['binding']})"
          + ("" if args.base_set_c is not None else f"; safety factor {safety:g}")
          + (" -> OK" if ok else " -> NOT CERTIFIED"))
    ext_new = extents(res["p_inv"], c_b)
    ext_ref = extents(np.linalg.inv(ref_ctrl.p_matrix_f64()), float(ref.base_set_c))
    print("[base set extents] " + ", ".join(f"{k} {ext_ref[k]:.2f} -> {ext_new[k]:.2f}" for k in ext_new) + "   (reference -> new)")
    if not ok:
        raise SystemExit("the chosen level is not certified; lower --base_set_c or raise --z_des")
    if ext_new["yaw_deg"] < yaw_range_deg - 0.5:
        print(f"[note] the base set spans {ext_new['yaw_deg']:.0f} deg of yaw, less than the design region's "
              f"{yaw_range_deg:g} deg: starts with a large yaw offset must yaw before they can arrive.")
    if separate_yaw:
        k_yaw = float(res["k_yaw"])
        share = (yaw_phi(yaw_range_deg) / yaw_phi(ext_new["yaw_deg"])) ** 2
        print(f"[yaw] the hover LQR removes a yaw offset with a time constant of {1.0 / k_yaw:.1f} s (limit {new.omega_z_max:g} rad/s); "
              f"a start at the edge of the region's yaw range ({yaw_range_deg:g} deg) uses {share:.2f} of c_B on its own")

    landing_out = {k: v for k, v in new.as_dict().items() if v != getattr(QuadrotorLandingConfig(), k)}
    out = {
        "landing": landing_out,
        "region": {k: v for k, v in region.items() if v != getattr(LandingDesignRegionConfig(), k)},
        "provenance": {
            "designed_by": "scripts/design_landing_base_set.py",
            "vehicle": args.vehicle or None,
            "reference_design": ref_name,
            "thrust_reserve": args.thrust_reserve,
            "rate_rise_time": args.rate_rise_time,
            "vehicle_limits": veh,
            "explicit_overrides": {k: getattr(args, k) for k in ("a_cmd_min", "a_cmd_max", "omega_max", "omega_z_max", "z_des", "base_set_c", "base_set_yaw_deg")
                                   if getattr(args, k) is not None},
            "input_margins_reference": m_ref.tolist(),
            "input_margins": m_new.tolist(),
            "yaw_rule": {"base_set_yaw_deg": base_yaw_deg, "region_yaw_deg": yaw_range_deg, "q_thetaz": new.lqr_q_thetaz, "r_omega_z": new.lqr_r_omega_z} if separate_yaw else None,
            "level_bounds": res["bounds"],
            "c_U_per_input": res["c_U_per_input"],
            "c_bar": res["c_bar"],
            "binding": res["binding"],
            "safety_factor": None if args.base_set_c is not None else safety,
            "closed_loop_spectral_radius": res["spectral_radius"],
            "base_set_extents": ext_new,
            "base_set_extents_reference": ext_ref,
            "task_references": [str(p) for p in args.task_references],
            "task_envelope": envelope,
            "region_reasons": region_reasons,
        },
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
