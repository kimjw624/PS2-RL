#!/usr/bin/env python
"""Slow "forward, then one loop" reference for the tracking policy that is flown in SITL and on hardware.

Same npz keys as the other references (``t, states=[p, v, q(w,x,y,z)], a_cmd, omega_cmd, ...``), so the
repo's loaders and plots work unchanged. Positions are in a *task frame*: the origin is the point where
the policy takes over (the vehicle hovers there, set by PX4 before the hand-over), x is the vehicle's
heading at the hand-over, z is up. The deployment node puts that frame at the hand-over point.

Path (planar, constant height, constant heading = 0)
----------------------------------------------------
    hover  ->  straight L1  ->  curvature ramp  ->  circle of radius R  ->  ramp down  ->  straight L2  ->  hover

* The curvature goes 0 -> 1/R and back with a quintic smootherstep over ``--ramp`` metres (a clothoid-like
  blend), so heading, curvature and its derivative are continuous: the reference's attitude and body rates
  have no jumps. The total turn is ``--turns`` x 360 deg; the circle's arc length is set accordingly.
* Speed along the path: smootherstep from 0 to ``--speed`` in ``--t_acc`` s, cruise, smootherstep back to 0,
  so the reference starts and ends at rest with zero jerk.
* Attitude, collective thrust and body rates come from differential flatness, using the landing generator's
  ``flat_outputs_to_states`` (same conventions as the env).

With the defaults (0.4 m/s, R = 1.5 m) the tilt stays below 1.5 deg and the body rates below 0.02 rad/s.

``--turns 0`` gives a straight line of ``--l1 + --l2`` metres with the same speed profile and hovers (no
curve; ``--radius`` and ``--ramp`` are ignored). The straight-line test reference is

    python ps2rl/envs/assets/generate_quadrotor_slow_loop_reference.py --turns 0 --l1 2.0 --l2 0 --speed 0.3 \
        --out ps2rl/envs/assets/slow_line/quadrotor_slow_line_reference.npz

    python ps2rl/envs/assets/generate_quadrotor_slow_loop_reference.py
    python ps2rl/envs/assets/generate_quadrotor_slow_loop_reference.py --speed 0.3 --radius 2.0 \
        --out ps2rl/envs/assets/slow_loop/quadrotor_slow_loop_v03_r2_reference.npz
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[2]))

from ps2rl.envs.assets.generate_quadrotor_landing_reference import (  # noqa: E402
    G,
    euler_zyx_from_rotmat,
    flat_outputs_to_states,
    quat_to_rotmat,
    rollout,
)

DEFAULT_OUT = _HERE / "slow_loop" / "quadrotor_slow_loop_reference.npz"


# ---------------------------------------------------------------------------
# quintic smootherstep and its integral / derivatives (tau in [0, 1])
# ---------------------------------------------------------------------------
def _ss(tau):
    tau = np.clip(tau, 0.0, 1.0)
    return tau ** 3 * (10.0 - 15.0 * tau + 6.0 * tau ** 2)


def _ss_int(tau):            # integral of _ss from 0 to tau (tau may exceed 1: continues linearly)
    t = np.clip(tau, 0.0, 1.0)
    base = t ** 4 * (2.5 - 3.0 * t + t ** 2)
    return base + np.maximum(np.asarray(tau) - 1.0, 0.0)


def _ss_d1(tau):
    inside = (tau > 0.0) & (tau < 1.0)
    t = np.clip(tau, 0.0, 1.0)
    return np.where(inside, 30.0 * t ** 2 * (1.0 - t) ** 2, 0.0)


def _ss_d2(tau):
    inside = (tau > 0.0) & (tau < 1.0)
    t = np.clip(tau, 0.0, 1.0)
    return np.where(inside, 60.0 * t * (1.0 - t) * (1.0 - 2.0 * t), 0.0)


# ---------------------------------------------------------------------------
# path geometry: curvature profile kappa(s), heading theta(s), position p(s)
# ---------------------------------------------------------------------------
class Path2D:
    def __init__(self, l1, ramp, radius, turns, l2, direction):
        if turns == 0:                               # straight line: no curve at all
            self.k_max, self.ramp = 0.0, 1.0         # (ramp only avoids a division by zero; curvature is 0)
            self.s0 = self.s1 = self.s2 = self.s3 = l1
            self.length = l1 + l2
        else:
            self.k_max = direction / radius
            self.ramp = ramp
            total_turn = 2.0 * np.pi * turns
            arc = total_turn * radius - ramp        # two half-ramps give k_max * ramp of turn together
            if arc < 0.0:
                raise ValueError("ramp too long for this radius / number of turns")
            self.s0 = l1                             # ramp up starts
            self.s1 = self.s0 + ramp                 # full curvature
            self.s2 = self.s1 + arc                  # ramp down starts
            self.s3 = self.s2 + ramp                 # straight again
            self.length = self.s3 + l2
        if self.length <= 0.0:
            raise ValueError("the path has no length")
        # position by numerical integration of the heading on a fine grid
        self._s = np.linspace(0.0, self.length, int(np.ceil(self.length / 1e-4)) + 1)
        th = self.theta(self._s)
        c, s = np.cos(th), np.sin(th)
        ds = np.diff(self._s)
        self._x = np.r_[0.0, np.cumsum(0.5 * (c[1:] + c[:-1]) * ds)]
        self._y = np.r_[0.0, np.cumsum(0.5 * (s[1:] + s[:-1]) * ds)]

    def kappa(self, s):
        up = self.k_max * _ss((s - self.s0) / self.ramp)
        down = self.k_max * _ss((s - self.s2) / self.ramp)
        return up - down

    def dkappa(self, s):
        return self.k_max / self.ramp * (_ss_d1((s - self.s0) / self.ramp) - _ss_d1((s - self.s2) / self.ramp))

    def theta(self, s):
        return self.k_max * self.ramp * (_ss_int((s - self.s0) / self.ramp) - _ss_int((s - self.s2) / self.ramp))

    def position(self, s):
        return np.interp(s, self._s, self._x), np.interp(s, self._s, self._y)


# ---------------------------------------------------------------------------
# speed profile along the path: s(t) and its derivatives
# ---------------------------------------------------------------------------
class SpeedProfile:
    def __init__(self, length, v, t_acc, t_hover0, t_hover1):
        d_acc = 0.5 * v * t_acc                      # distance covered while accelerating (smootherstep mean 1/2)
        if 2.0 * d_acc > length:
            raise ValueError(f"path ({length:.2f} m) shorter than accel + decel distance ({2 * d_acc:.2f} m)")
        self.v, self.ta = v, t_acc
        self.t0 = t_hover0
        self.t1 = self.t0 + t_acc
        self.t2 = self.t1 + (length - 2.0 * d_acc) / v
        self.t3 = self.t2 + t_acc
        self.T = self.t3 + t_hover1
        self.length = length

    def __call__(self, t):
        v, ta = self.v, self.ta
        a = (t - self.t0) / ta                        # acceleration phase coordinate
        d = (t - self.t2) / ta                        # deceleration phase coordinate
        s = v * ta * (_ss_int(a) - _ss_int(d))
        sd = v * (_ss(a) - _ss(d))
        sdd = v / ta * (_ss_d1(a) - _ss_d1(d))
        sddd = v / ta ** 2 * (_ss_d2(a) - _ss_d2(d))
        return np.clip(s, 0.0, self.length), sd, sdd, sddd


def build(args):
    direction = 1.0 if args.direction == "left" else -1.0
    path = Path2D(args.l1, args.ramp, args.radius, args.turns, args.l2, direction)
    prof = SpeedProfile(path.length, args.speed, args.t_acc, args.hover_start, args.hover_end)
    n = int(round(prof.T / args.dt)) + 1
    t = np.arange(n) * args.dt
    s, sd, sdd, sddd = prof(t)

    th, k, dk = path.theta(s), path.kappa(s), path.dkappa(s)
    T = np.stack([np.cos(th), np.sin(th), np.zeros_like(th)], 1)      # unit tangent
    N = np.stack([-np.sin(th), np.cos(th), np.zeros_like(th)], 1)     # unit normal (left)
    px, py = path.position(s)
    P = np.stack([px, py, np.full_like(px, args.z)], 1)
    # p' = T, p'' = k N, p''' = k' N - k^2 T (derivatives w.r.t. arc length)
    V = T * sd[:, None]
    A = (k * sd ** 2)[:, None] * N + sdd[:, None] * T
    J = ((dk * sd ** 3 + 3.0 * k * sd * sdd)[:, None] * N
         + (sddd - k ** 2 * sd ** 3)[:, None] * T)

    Q, a_cmd, omega = flat_outputs_to_states(P, V, A, J, np.deg2rad(args.yaw_deg))
    states = np.concatenate([P, V, Q], axis=1)
    rpy = np.array([euler_zyx_from_rotmat(quat_to_rotmat(q)) for q in Q])
    quat_norm = np.linalg.norm(Q, axis=1)
    quat_step = np.r_[0.0, np.linalg.norm(np.diff(Q, axis=0), axis=1)]
    bundle = dict(t=t, states=states, a_cmd=a_cmd, omega_cmd=omega, roll=rpy[:, 0], pitch=rpy[:, 1], yaw=rpy[:, 2],
                  quat_norm=quat_norm, quat_step_norm=quat_step, max_a_cmd=np.array([a_cmd.max()]),
                  max_abs_omega=np.array([np.abs(omega).max()]), max_quat_step_norm=np.array([quat_step.max()]),
                  max_quat_norm_error=np.array([np.abs(quat_norm - 1.0).max()]))
    phases = {"hover_start_end": prof.t0, "cruise_start": prof.t1, "decel_start": prof.t2, "rest": prof.t3,
              "end": prof.T, "turn_start_s": path.s0, "turn_end_s": path.s3, "path_length": path.length}
    return bundle, phases


def stats(bundle):
    s = bundle["states"]
    tilt = np.degrees(np.arccos(np.clip(1.0 - 2.0 * (s[:, 7] ** 2 + s[:, 8] ** 2), -1.0, 1.0)))
    acc = np.diff(s[:, 3:6], axis=0) / np.diff(bundle["t"])[:, None]
    return {"num_points": int(len(bundle["t"])), "duration": float(bundle["t"][-1]),
            "peak_speed": float(np.linalg.norm(s[:, 3:6], axis=1).max()),
            "peak_horizontal_acc": float(np.linalg.norm(acc[:, :2], axis=1).max()),
            "peak_tilt_deg": float(tilt.max()), "a_cmd_min": float(bundle["a_cmd"].min()),
            "a_cmd_max": float(bundle["a_cmd"].max()), "max_abs_omega": float(np.abs(bundle["omega_cmd"]).max()),
            "x_range": [float(s[:, 0].min()), float(s[:, 0].max())],
            "y_range": [float(s[:, 1].min()), float(s[:, 1].max())],
            "end_point": [float(v) for v in s[-1, 0:3]]}


def check(bundle, dt):
    """Integrate the env model with the reference inputs; the drift shows whether states / inputs agree."""
    s = bundle["states"]
    cont = rollout(s[0], bundle["a_cmd"], bundle["omega_cmd"], dt, hold=False)
    return float(np.linalg.norm(cont[:, 0:3] - s[:, 0:3], axis=1).max())


def plot(bundle, phases, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    s, t = bundle["states"], bundle["t"]
    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    ax[0, 0].plot(s[:, 0], s[:, 1])
    ax[0, 0].plot(s[0, 0], s[0, 1], "go", label="start (hand-over)")
    ax[0, 0].plot(s[-1, 0], s[-1, 1], "rs", label="end")
    ax[0, 0].set_aspect("equal"); ax[0, 0].grid(alpha=.3); ax[0, 0].legend()
    ax[0, 0].set_title("Path, top view (task frame: x = heading at hand-over) [m]")
    ax[0, 1].plot(t, np.linalg.norm(s[:, 3:6], axis=1), label="speed [m/s]")
    for k in ("hover_start_end", "cruise_start", "decel_start", "rest"):
        ax[0, 1].axvline(phases[k], ls=":", c="gray")
    ax[0, 1].legend(); ax[0, 1].grid(alpha=.3); ax[0, 1].set_title("Speed")
    ax[1, 0].plot(t, bundle["a_cmd"]); ax[1, 0].axhline(G, ls=":", c="gray")
    ax[1, 0].set_title("Collective thrust a_cmd [m/s^2]"); ax[1, 0].grid(alpha=.3)
    for i, name in enumerate(("p", "q", "r")):
        ax[1, 1].plot(t, bundle["omega_cmd"][:, i], label=name)
    ax[1, 1].legend(); ax[1, 1].grid(alpha=.3); ax[1, 1].set_title("Body rates [rad/s]")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--speed", type=float, default=0.4, help="cruise speed [m/s]")
    p.add_argument("--t_acc", type=float, default=4.0, help="time to reach / leave the cruise speed [s]")
    p.add_argument("--l1", type=float, default=2.0, help="straight part before the loop [m]")
    p.add_argument("--l2", type=float, default=2.0, help="straight part after the loop [m]")
    p.add_argument("--radius", type=float, default=1.5, help="loop radius [m]")
    p.add_argument("--ramp", type=float, default=1.0, help="curvature blend length at each end of the circle [m]")
    p.add_argument("--turns", type=float, default=1.0, help="number of full turns (0: straight line of l1 + l2)")
    p.add_argument("--direction", choices=("left", "right"), default="left")
    p.add_argument("--hover_start", type=float, default=3.0, help="hover before moving [s]")
    p.add_argument("--hover_end", type=float, default=3.0, help="hover after stopping [s]")
    p.add_argument("--z", type=float, default=0.0, help="height in the task frame (0 = hand-over height) [m]")
    p.add_argument("--yaw_deg", type=float, default=0.0)
    p.add_argument("--dt", type=float, default=0.02)
    p.add_argument("--out", default=str(DEFAULT_OUT))
    args = p.parse_args(argv)

    bundle, phases = build(args)
    st = stats(bundle)
    drift = check(bundle, args.dt)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **bundle)
    cfg = {"slow_loop_config": vars(args) | {"out": str(out)}, "phases": phases, "stats": st,
           "consistency_drift_m": drift}
    out.with_name(out.stem + "_config.json").write_text(json.dumps(cfg, indent=2))
    plot(bundle, phases, out.with_suffix(".png"))
    print(json.dumps({"out": str(out), **st, "consistency_drift_m": drift, "phases": phases}, indent=2))
    if drift > 0.02:
        print("!! states and inputs disagree (drift > 2 cm); check the path parameters")


if __name__ == "__main__":
    main()
