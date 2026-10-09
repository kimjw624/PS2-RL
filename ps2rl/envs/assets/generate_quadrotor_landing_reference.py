#!/usr/bin/env python
"""Generate a landing reference for the PS2-RL quadrotor environment.

The output npz has the same keys as ``quadrotor_powerloop_reference.npz`` so the
existing loader, env and evaluation code work unchanged:

    t (N,)  states (N,10)=[p, v, q(w,x,y,z)]  a_cmd (N,)  omega_cmd (N,3)
    roll pitch yaw (N,)  quat_norm quat_step_norm (N,)
    max_a_cmd max_abs_omega max_quat_step_norm max_quat_norm_error (1,)

Trajectory
----------
Position is a septic polynomial per axis (pos/vel/acc/jerk pinned at both ends) from an approach state
to a touchdown point. By design the touchdown point is *offset from the pad*, so
the reference leaves the approach cone near the ground and the control-invariant
layer must pull the vehicle onto the pad -- the landing analogue of the powerloop
apex that pierces the ceiling.

Attitude, collective thrust and body rates come from differential flatness
(Mellinger & Kumar 2011) with constant heading, using the env's conventions:

    v_dot = -g e3 + R(q) a e3,     q_dot = 0.5 * Xi(q) @ omega   (body rates)

so ``states``, ``a_cmd`` and ``omega_cmd`` are mutually consistent.

Usage
-----
    python ps2rl/envs/assets/generate_quadrotor_landing_reference.py          # default: misses the pad
    python ps2rl/envs/assets/generate_quadrotor_landing_reference.py --offset 0 \
        --out ps2rl/envs/assets/quadrotor_landing_onpad_reference.npz        # control: lands on the pad
    python ps2rl/envs/assets/generate_quadrotor_landing_reference.py --glide --x0 -1.5 --cross_z 0.8 \
        --out ps2rl/envs/assets/quadrotor_landing_glide_reference.npz        # undershoot glide: rides the cone
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

G = 9.81


# ---------------------------------------------------------------------------
# septic segment: position, velocity, acceleration and jerk fixed at both ends
# ---------------------------------------------------------------------------
# Zero jerk at the endpoints makes the body rates exactly zero there (rates are
# driven by jerk through flatness). That matters at the end: the env clamps the
# reference to its last sample, so a non-zero final rate would keep asking the
# vehicle to rotate after touchdown.
def septic_coeffs(x0, v0, a0, j0, x1, v1, a1, j1, T):
    """Coefficients c0..c7 of x(t)=sum c_k t^k matching pos/vel/acc/jerk at t=0 and t=T."""
    def row(t, d):
        r = np.zeros(8)
        for k in range(d, 8):
            coef = 1.0
            for m in range(d):
                coef *= (k - m)
            r[k] = coef * t ** (k - d)
        return r
    A = np.array([row(0.0, 0), row(0.0, 1), row(0.0, 2), row(0.0, 3),
                  row(T, 0), row(T, 1), row(T, 2), row(T, 3)])
    b = np.array([x0, v0, a0, j0, x1, v1, a1, j1], dtype=np.float64)
    return np.linalg.solve(A, b)


def poly_eval(c, t):
    """Position, velocity, acceleration and jerk of sum c_k t^k at times t."""
    out = []
    for d in range(4):
        val = np.zeros_like(t, dtype=np.float64)
        for k in range(d, len(c)):
            coef = 1.0
            for m in range(d):
                coef *= (k - m)
            val = val + c[k] * coef * t ** (k - d)
        out.append(val)
    return tuple(out)


# ---------------------------------------------------------------------------
# rotations (quaternion order [w, x, y, z], Hamilton, body -> world)
# ---------------------------------------------------------------------------
def rotmat_to_quat(R):
    """Shepperd's method; returns [w, x, y, z] with w >= 0."""
    tr = np.trace(R)
    if tr > 0:
        s = 2.0 * np.sqrt(tr + 1.0)
        q = np.array([0.25 * s, (R[2, 1] - R[1, 2]) / s,
                      (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s])
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        q = np.array([(R[2, 1] - R[1, 2]) / s, 0.25 * s,
                      (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s])
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        q = np.array([(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s,
                      0.25 * s, (R[1, 2] + R[2, 1]) / s])
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        q = np.array([(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s,
                      (R[1, 2] + R[2, 1]) / s, 0.25 * s])
    q = q / np.linalg.norm(q)
    return q if q[0] >= 0 else -q


def quat_to_rotmat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def xi_matrix(q):
    """Quaternion kinematic matrix of the env: q_dot = 0.5 * Xi(q) @ omega_body."""
    w, x, y, z = q
    return np.array([
        [-x, -y, -z],
        [w, -z, y],
        [z, w, -x],
        [-y, x, w],
    ])


def euler_zyx_from_rotmat(R):
    """Roll, pitch, yaw (ZYX) in radians."""
    pitch = -np.arcsin(np.clip(R[2, 0], -1.0, 1.0))
    roll = np.arctan2(R[2, 1], R[2, 2])
    yaw = np.arctan2(R[1, 0], R[0, 0])
    return roll, pitch, yaw


# ---------------------------------------------------------------------------
# differential flatness
# ---------------------------------------------------------------------------
def flat_outputs_to_states(p, v, a, j, yaw):
    """Map (p, v, a, jerk, yaw=const) to quaternion, collective thrust and body rates."""
    n = p.shape[0]
    e3 = np.array([0.0, 0.0, 1.0])
    x_c = np.array([np.cos(yaw), np.sin(yaw), 0.0])

    q_all = np.zeros((n, 4))
    a_cmd = np.zeros(n)
    omega = np.zeros((n, 3))
    for k in range(n):
        f = a[k] + G * e3                       # mass-normalized thrust vector (world)
        c = np.linalg.norm(f)
        z_b = f / c
        y_b = np.cross(z_b, x_c)
        y_b /= np.linalg.norm(y_b)
        x_b = np.cross(y_b, z_b)
        R = np.column_stack([x_b, y_b, z_b])

        # z_b_dot = q x_b - p y_b  and  z_b_dot = (j - (z_b.j) z_b) / c
        h = (j[k] - np.dot(z_b, j[k]) * z_b) / c
        omega[k] = [-np.dot(h, y_b), np.dot(h, x_b), 0.0]   # constant heading -> r = 0

        a_cmd[k] = c
        q_all[k] = rotmat_to_quat(R)

    # keep quaternion sign continuous so linear interpolation in the env is valid
    for k in range(1, n):
        if np.dot(q_all[k], q_all[k - 1]) < 0:
            q_all[k] = -q_all[k]
    return q_all, a_cmd, omega


# ---------------------------------------------------------------------------
# approach cone h(x) = r0 + tan(theta)(z - z_pad) - sqrt(|p_xy - p_pad|^2 + eps^2)
# ---------------------------------------------------------------------------
def h_cone(p, pad_xy, z_pad, r0, theta_deg, eps):
    dp = p[:, :2] - np.asarray(pad_xy)[None, :]
    return r0 + np.tan(np.deg2rad(theta_deg)) * (p[:, 2] - z_pad) - np.sqrt(np.sum(dp**2, axis=1) + eps**2)


# ---------------------------------------------------------------------------
# consistency check: integrate the env dynamics with the reference inputs
# ---------------------------------------------------------------------------
def _model(x, a, w):
    q = x[6:10] / np.linalg.norm(x[6:10])
    R = quat_to_rotmat(q)
    vdot = -G * np.array([0.0, 0.0, 1.0]) + R @ np.array([0.0, 0.0, a])
    qdot = 0.5 * xi_matrix(q) @ w
    return np.concatenate([x[3:6], vdot, qdot])


def rollout(x0, a_cmd, omega, dt, hold=True, substeps=20):
    """RK4 rollout of the env model driven by the reference inputs.

    hold=True  : zero-order hold, as in the env (inputs constant over each step).
    hold=False : inputs linearly interpolated between samples -- this isolates the
                 consistency of states/a_cmd/omega from discretization effects.
    """
    x = x0.copy()
    xs = [x.copy()]
    h = dt / substeps
    for k in range(len(a_cmd) - 1):
        for m in range(substeps):
            def u(frac):
                if hold:
                    return a_cmd[k], omega[k]
                return (a_cmd[k] * (1 - frac) + a_cmd[k + 1] * frac,
                        omega[k] * (1 - frac) + omega[k + 1] * frac)
            s0, s1, s2 = m / substeps, (m + 0.5) / substeps, (m + 1) / substeps
            k1 = _model(x, *u(s0))
            k2 = _model(x + 0.5 * h * k1, *u(s1))
            k3 = _model(x + 0.5 * h * k2, *u(s1))
            k4 = _model(x + h * k3, *u(s2))
            x = x + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
            x[6:10] /= np.linalg.norm(x[6:10])
        xs.append(x.copy())
    return np.array(xs)


# ---------------------------------------------------------------------------
def build(args):
    dt, T = args.dt, args.duration
    n = int(round(T / dt)) + 1
    t = np.linspace(0.0, T, n)

    p0 = np.array([args.x0, args.y0, args.z0])
    pad = np.array([args.pad_x, args.pad_y])

    if args.cross_z is not None:
        # Undershoot: touch down short of the pad on the approach side, on the
        # straight line through the start, such that the path meets the cone wall
        # at altitude cross_z and stays outside it below.
        rel = p0[:2] - pad
        rho0 = float(np.linalg.norm(rel))
        if rho0 < 1e-9:
            raise SystemExit("--cross_z needs a start that is horizontally offset from the pad")
        u_rad = rel / rho0
        zeta0 = args.z0 - args.z_pad
        zc = args.cross_z
        if not (0.0 < zc < zeta0):
            raise SystemExit(f"--cross_z must lie in (0, {zeta0})")
        tan_t = np.tan(np.deg2rad(args.theta_deg))
        rho_b = np.sqrt(max((args.r0 + tan_t * zc) ** 2 - args.eps ** 2, 0.0))
        rho_td = (rho_b - rho0 * zc / zeta0) / (1.0 - zc / zeta0)
        if rho_td <= args.r0:
            raise SystemExit(f"cross_z={zc} gives touchdown inside the pad (rho_td={rho_td:.3f}); raise it")
        if rho0 >= np.sqrt(max((args.r0 + tan_t * zeta0) ** 2 - args.eps ** 2, 0.0)):
            raise SystemExit("start is outside the cone")
        p1 = np.array([*(pad + rho_td * u_rad), args.z_pad + args.z_touch])
    else:
        p1 = np.array([args.pad_x + args.offset * np.cos(np.deg2rad(args.offset_dir_deg)),
                       args.pad_y + args.offset * np.sin(np.deg2rad(args.offset_dir_deg)),
                       args.z_pad + args.z_touch])

    if args.glide:
        # start velocity along the straight line to touchdown -> every axis shares
        # the same normalized profile, so the path is an exact straight segment
        d = p1 - p0
        v0 = args.approach_speed * d / np.linalg.norm(d)
    else:
        v0 = np.array([args.vx0, args.vy0, args.vz0])
    v1 = np.array([0.0, 0.0, -abs(args.vz_touch)])

    P = np.zeros((n, 3)); V = np.zeros((n, 3)); A = np.zeros((n, 3)); J = np.zeros((n, 3))
    for i in range(3):
        c = septic_coeffs(p0[i], v0[i], 0.0, 0.0, p1[i], v1[i], 0.0, 0.0, T)
        P[:, i], V[:, i], A[:, i], J[:, i] = poly_eval(c, t)

    yaw = np.deg2rad(args.yaw_deg)
    Q, a_cmd, omega = flat_outputs_to_states(P, V, A, J, yaw)
    states = np.concatenate([P, V, Q], axis=1)

    rpy = np.array([euler_zyx_from_rotmat(quat_to_rotmat(q)) for q in Q])
    quat_norm = np.linalg.norm(Q, axis=1)
    quat_step = np.r_[0.0, np.linalg.norm(np.diff(Q, axis=0), axis=1)]

    bundle = dict(
        t=t,
        states=states,
        a_cmd=a_cmd,
        omega_cmd=omega,
        roll=rpy[:, 0],
        pitch=rpy[:, 1],
        yaw=rpy[:, 2],
        quat_norm=quat_norm,
        quat_step_norm=quat_step,
        max_a_cmd=np.array([a_cmd.max()]),
        max_abs_omega=np.array([np.abs(omega).max()]),
        max_quat_step_norm=np.array([quat_step.max()]),
        max_quat_norm_error=np.array([np.abs(quat_norm - 1.0).max()]),
    )
    return bundle


def report(bundle, args):
    s = bundle["states"]
    P, V, Q = s[:, 0:3], s[:, 3:6], s[:, 6:10]
    a_cmd, omega, t = bundle["a_cmd"], bundle["omega_cmd"], bundle["t"]
    tilt = np.rad2deg(np.arccos(np.clip(1 - 2 * (Q[:, 1]**2 + Q[:, 2]**2), -1, 1)))

    print(f"points           : {len(t)}  (dt={args.dt}, T={args.duration} s)")
    print(f"start            : p={np.round(P[0], 3)}  v={np.round(V[0], 3)}")
    print(f"touchdown        : p={np.round(P[-1], 3)}  v={np.round(V[-1], 3)}")
    print(f"pad              : ({args.pad_x}, {args.pad_y}, {args.z_pad})  "
          f"touchdown offset from pad = {np.hypot(P[-1,0]-args.pad_x, P[-1,1]-args.pad_y):.3f} m")
    print(f"a_cmd  [m/s^2]   : min {a_cmd.min():.2f}  max {a_cmd.max():.2f}  "
          f"(limits 0..{args.a_max_g * G:.1f})")
    print(f"|omega| [rad/s]  : max per axis {np.round(np.abs(omega).max(0), 3)}  (limit {args.omega_max})")
    print(f"tilt [deg]       : max {tilt.max():.1f}")
    print(f"peak |v| [m/s]   : {np.linalg.norm(V, axis=1).max():.2f}")

    ok = True
    if a_cmd.min() < 0 or a_cmd.max() > args.a_max_g * G:
        print("  !! a_cmd outside the actuator range"); ok = False
    if np.abs(omega).max() > args.omega_max:
        print("  !! body rate exceeds omega_max"); ok = False

    # dynamic consistency: integrate the env model with the reference inputs
    cont = rollout(s[0], a_cmd, omega, args.dt, hold=False)
    zoh = rollout(s[0], a_cmd, omega, args.dt, hold=True)
    e_cont = np.linalg.norm(cont[:, 0:3] - P, axis=1).max()
    e_zoh = np.linalg.norm(zoh[:, 0:3] - P, axis=1).max()
    print(f"consistency      : continuous-input rollout max pos err {e_cont*100:.2f} cm "
          "(should be ~0: states, a_cmd, omega agree)")
    print(f"ZOH drift        : held-input rollout max pos err {e_zoh*100:.1f} cm "
          "(expected open-loop drift from the sample-and-hold; closed-loop tracking removes it)")
    if e_cont > 0.02:
        print("  !! states and inputs are inconsistent"); ok = False

    # approach cone diagnostics
    h = h_cone(P, (args.pad_x, args.pad_y), args.z_pad, args.r0, args.theta_deg, args.eps)
    viol = np.where(h < 0)[0]
    print(f"cone (r0={args.r0}, theta={args.theta_deg} deg, eps={args.eps}):")
    print(f"  h at start {h[0]:.3f}   min h {h.min():.3f}   points outside {len(viol)}/{len(h)}")
    if len(viol):
        k = viol[0]
        print(f"  reference leaves the cone at t={t[k]:.2f} s, z={P[k,2]-args.z_pad:.2f} m above pad")
    else:
        print("  reference never leaves the cone -> the safety layer will rarely act; "
              "increase --offset or narrow --theta_deg if that is not intended")
    if h[0] < 0:
        print("  !! start state is outside the cone (initial state must be safe)"); ok = False

    # worst case over the initial-state randomization box: every sampled start must
    # be inside the cone (PS2-RL needs X0 inside the safe/invariant set)
    r = args.init_p_range
    corners = np.array([[P[0, 0] + sx * r, P[0, 1] + sy * r, P[0, 2] + sz * r]
                        for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    h_init = h_cone(corners, (args.pad_x, args.pad_y), args.z_pad, args.r0, args.theta_deg, args.eps).min()
    print(f"  worst-case h over init box (+-{r} m): {h_init:.3f}")
    if h_init < 0:
        print("  !! some randomized starts lie outside the cone; move the start inward or shrink "
              "--init_px/py/pz_range"); ok = False
    return ok


def plot(bundle, args, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping plot"); return

    s = bundle["states"]; t = bundle["t"]
    P = s[:, 0:3]
    fig, ax = plt.subplots(2, 2, figsize=(11, 7))

    # x-z side view with the cone
    a0 = ax[0, 0]
    z = np.linspace(0, max(P[:, 2].max(), args.z0) + 0.5, 200)
    rad = np.sqrt(np.maximum((args.r0 + np.tan(np.deg2rad(args.theta_deg)) * z)**2 - args.eps**2, 0))
    a0.fill_betweenx(z + args.z_pad, args.pad_x - rad, args.pad_x + rad, color="tab:green", alpha=0.12,
                     label="approach cone")
    a0.plot(P[:, 0], P[:, 2], "k--", lw=2, label="reference")
    a0.plot([args.pad_x - args.r0, args.pad_x + args.r0], [args.z_pad] * 2, color="tab:green", lw=5, label="pad")
    a0.plot(P[0, 0], P[0, 2], "o", color="tab:blue", label="start")
    a0.plot(P[-1, 0], P[-1, 2], "x", color="tab:red", ms=10, mew=2, label="reference touchdown")
    a0.set_xlabel("x [m]"); a0.set_ylabel("z [m]"); a0.set_aspect("equal", adjustable="datalim")
    a0.set_title("Side view: the reference misses the pad"); a0.legend(fontsize=8); a0.grid(alpha=.3)

    ax[0, 1].plot(t, P[:, 2], label="z"); ax[0, 1].plot(t, s[:, 5], label="v_z")
    ax[0, 1].set_title("Altitude and vertical speed"); ax[0, 1].legend(); ax[0, 1].grid(alpha=.3)

    ax[1, 0].plot(t, bundle["a_cmd"]); ax[1, 0].axhline(G, ls=":", c="gray")
    ax[1, 0].set_title("Collective thrust a_cmd [m/s^2]"); ax[1, 0].grid(alpha=.3)

    for i, name in enumerate(["wx", "wy", "wz"]):
        ax[1, 1].plot(t, bundle["omega_cmd"][:, i], label=name)
    ax[1, 1].set_title("Body rates [rad/s]"); ax[1, 1].legend(); ax[1, 1].grid(alpha=.3)
    for a in ax[1]:
        a.set_xlabel("t [s]")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    print(f"saved plot: {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    here = Path(__file__).resolve().parent
    ap.add_argument("--out", type=Path, default=here / "quadrotor_landing_reference.npz",
                    help="output npz; a .png figure and a _config.json are written next to it")
    ap.add_argument("--dt", type=float, default=0.02, help="must match --reference_dt / --env_dt")
    ap.add_argument("--duration", type=float, default=2.0)
    # approach state
    ap.add_argument("--x0", type=float, default=-2.0)
    ap.add_argument("--y0", type=float, default=0.0)
    ap.add_argument("--z0", type=float, default=2.0)
    ap.add_argument("--vx0", type=float, default=1.0)
    ap.add_argument("--vy0", type=float, default=0.0)
    ap.add_argument("--vz0", type=float, default=0.0)
    ap.add_argument("--yaw_deg", type=float, default=0.0)
    # pad and touchdown
    ap.add_argument("--pad_x", type=float, default=0.0)
    ap.add_argument("--pad_y", type=float, default=0.0)
    ap.add_argument("--z_pad", type=float, default=0.0)
    ap.add_argument("--offset", type=float, default=0.8,
                    help="horizontal distance of the reference touchdown from the pad center (0 = on the pad)")
    ap.add_argument("--offset_dir_deg", type=float, default=0.0, help="direction of the miss in the x-y plane")
    ap.add_argument("--z_touch", type=float, default=0.0, help="touchdown height above the pad")
    ap.add_argument("--cross_z", type=float, default=None,
                    help="undershoot mode: touch down short of the pad so the path crosses the cone wall "
                         "at this altitude above the pad (overrides --offset)")
    ap.add_argument("--glide", action="store_true",
                    help="start velocity along the line to touchdown -> straight glide path (overrides --vx0/vy0/vz0)")
    ap.add_argument("--approach_speed", type=float, default=1.0, help="start speed along the glide path [m/s]")
    ap.add_argument("--vz_touch", type=float, default=0.0,
                    help="descent speed at touchdown (0 = soft; >0 makes the reference also violate a descent-rate limit)")
    # cone (for diagnostics and the plot only; the env uses its own settings)
    ap.add_argument("--r0", type=float, default=0.3)
    ap.add_argument("--theta_deg", type=float, default=45.0)
    ap.add_argument("--eps", type=float, default=0.015)
    ap.add_argument("--init_p_range", type=float, default=0.3,
                    help="initial-position randomization used in training (check only)")
    # actuator limits (for checks only)
    ap.add_argument("--a_max_g", type=float, default=4.0)
    ap.add_argument("--omega_max", type=float, default=18.0)
    args = ap.parse_args()

    bundle = build(args)
    ok = report(bundle, args)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **bundle)
    print(f"saved reference: {args.out}")

    import json
    cfg = {
        "landing_config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "landing_trajectory_stats": {
            "num_points": int(len(bundle["t"])),
            "total_duration": float(bundle["t"][-1]),
            "max_a_cmd": float(bundle["max_a_cmd"][0]),
            "max_abs_omega": float(bundle["max_abs_omega"][0]),
            "max_quat_step_norm": float(bundle["max_quat_step_norm"][0]),
            "max_quat_norm_error": float(bundle["max_quat_norm_error"][0]),
        },
    }
    cfg_path = args.out.with_name(args.out.stem + "_config.json")
    cfg_path.write_text(json.dumps(cfg, indent=2))
    print(f"saved config   : {cfg_path}")
    plot(bundle, args, args.out.with_suffix(".png"))
    if not ok:
        raise SystemExit("reference violates a hard check above")


if __name__ == "__main__":
    main()