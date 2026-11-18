"""Data stage of the landing Phase-I report: every number and trajectory the report shows.

Called by ``report_landing_phase1.py``; results are cached as a pickle so the figure and
LaTeX stages can be re-run without re-simulating. All evaluation follows the first-hit
semantics of the landing note (Lemma 4): x0 is counted as safely arriving iff the
composed backup pi_b reaches B within N steps before leaving the cone.
"""

from __future__ import annotations

import glob
import json
import os
import pickle
from pathlib import Path
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np

from ps2rl.envs.quadrotor_landing_config import QuadrotorLandingConfig
from ps2rl.phase1_sa.landing_design_region import REGION_NAMES, LandingDesignRegionConfig, heldout_sets, tilt_of
from ps2rl.phase1_sa.quadrotor_landing_sa_env import (
    build_landing_sa_env,
    landing_action_box,
    landing_step_fn,
)
from ps2rl.sets import landing_certificate as lc
from ps2rl.utils.policy import ActorConfig, actor_mean_action

CASE_STEPS = 150  # 3 s: the 2 s backup horizon plus 1 s of base-controller hold


# --------------------------------------------------------------------------- loading
def load_json(p: Path) -> Any:
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def list_runs(root: Path) -> list[str]:
    return sorted(Path(p).parent.name for p in glob.glob(str(root / "*" / "summary.json")))


def load_actor(run_dir: Path) -> tuple[Any, ActorConfig]:
    with open(run_dir / "landing_backup_policy_actor.pkl", "rb") as f:
        d = pickle.load(f)
    ac = dict(d["actor_cfg"])
    ac["hidden_sizes"] = tuple(ac["hidden_sizes"])
    return d["actor_params"], ActorConfig(**ac)


class Ctx:
    """Sets, dynamics and samplers built from the run's saved config."""

    def __init__(self, root: Path, run: str):
        c = load_json(root / run / "configs.json")
        self.cfg = QuadrotorLandingConfig.from_dict(c["landing"])
        self.region_cfg = LandingDesignRegionConfig.from_dict(c["design_region"])
        self.ra_cfg = c["backup_ra"]
        self.env = build_landing_sa_env(self.cfg, self.region_cfg)
        self.cone = self.env.safe_set
        self.B = self.env.base_set
        self.ctrl = self.B.controller
        self.scale, self.low, self.high = landing_action_box(self.cfg)
        self.step = landing_step_fn(self.cfg)
        self.N = int(self.cfg.num_steps)
        self.mid = 0.5 * (self.high + self.low)
        self.half = 0.5 * (self.high - self.low)

    # policies map a single state to a raw action in the box
    def actor_policy(self, params: Any, acfg: ActorConfig) -> Callable:
        def pi(x):
            a = actor_mean_action(params, x, self.scale, acfg, action_low=self.low, action_high=self.high)
            return jnp.clip(a, self.low, self.high)

        return pi

    def lqr_policy(self) -> Callable:
        return self.ctrl.action

    def simulator(self, pi_raw: Callable, steps: int) -> Callable:
        """Batched rollout of the composed backup pi_b (pi_B inside B, pi_raw outside)."""
        ctrl, B, step = self.ctrl, self.B, self.step

        def sim(x0):
            def body(x, _):
                inb = B.contains(x)
                u = jax.lax.select(inb, ctrl.action(x), pi_raw(x))
                return step(x, u), (x, u)

            x_t, (xs, us) = jax.lax.scan(body, x0, None, length=steps)
            return jnp.concatenate([xs, x_t[None]], 0), us

        return jax.jit(jax.vmap(sim))

    def hit_fn(self, pi_raw: Callable) -> Callable:
        """Batched first-hit evaluation without storing trajectories (for dense grids)."""
        ctrl, B, step, cone, N = self.ctrl, self.B, self.step, self.cone, self.N

        def run(x0):
            def body(c, k):
                x, hb, hf, tb = c
                inb = B.contains(x)
                u = jax.lax.select(inb, ctrl.action(x), pi_raw(x))
                xn = step(x, u)
                act = ~(hb | hf)
                nf = act & ~cone.contains(xn)
                nb = act & ~nf & B.contains(xn)
                tb = jnp.where(nb, k + 1, tb)
                return (jnp.where(act, xn, x), hb | nb, hf | nf, tb), None

            (_, hb, hf, tb), _ = jax.lax.scan(body, (x0, False, False, jnp.int32(-1)), jnp.arange(N))
            return hb, hf, tb

        return jax.jit(jax.vmap(run))

    # vectorised set evaluations on numpy arrays of shape (..., 10)
    def h(self, xs: np.ndarray) -> np.ndarray:
        return np.asarray(self.cone.value(jnp.asarray(xs)))

    def V(self, xs: np.ndarray) -> np.ndarray:
        flat = jnp.asarray(xs.reshape(-1, 10))
        return np.asarray(jax.vmap(self.ctrl.quadratic_form)(flat)).reshape(xs.shape[:-1])

    def tilt_deg(self, xs: np.ndarray) -> np.ndarray:
        return np.degrees(np.asarray(tilt_of(jnp.asarray(xs[..., 6:10]))))


# --------------------------------------------------------------- trajectory statistics
def first_hit_stats(ctx: Ctx, xs: np.ndarray, us: np.ndarray) -> dict[str, np.ndarray]:
    """Per-trajectory first-hit outcome and statistics of the pi_SA segment (k < tau_B)."""
    N = ctx.N
    h = ctx.h(xs[:, : N + 1])
    V = ctx.V(xs[:, : N + 1])
    c_b = float(ctx.cfg.base_set_c)
    fail_k = h[:, 1:] < 0.0
    goal_k = (V[:, 1:] <= c_b) & ~fail_k
    big = N + 10
    t_f = np.where(fail_k.any(1), fail_k.argmax(1) + 1, big)
    t_b = np.where(goal_k.any(1), goal_k.argmax(1) + 1, big)
    success = (t_b <= N) & (t_b < t_f)
    crash = (t_f <= N) & (t_f < t_b)
    timeout = ~success & ~crash
    m = xs.shape[0]
    seg_end = np.where(success, t_b, np.minimum(t_f, N))  # steps under pi_SA
    zmin = np.array([xs[i, : seg_end[i] + 1, 2].min() for i in range(m)])
    tilt = ctx.tilt_deg(xs[:, : N + 1])
    tiltmax = np.array([tilt[i, : seg_end[i] + 1].max() for i in range(m)])
    wmax = np.array([np.abs(us[i, : max(seg_end[i], 1), 1:]).max() for i in range(m)])
    amax = np.array([us[i, : max(seg_end[i], 1), 0].max() for i in range(m)])
    hmin = np.array([h[i, 1 : seg_end[i] + 1].min() if seg_end[i] > 0 else h[i, 0] for i in range(m)])
    un = (us[:, :N] - np.asarray(ctx.mid)) / np.asarray(ctx.half)  # normalised in [-1, 1]
    active = np.arange(N)[None, :] < seg_end[:, None]
    thr_hi = (un[..., 0] > 0.98) & active
    thr_lo = (un[..., 0] < -0.98) & active
    rate_sat = (np.abs(un[..., 1:]) > 0.98).any(-1) & active
    du = np.linalg.norm(np.diff(un, axis=1), axis=-1)
    du_active = active[:, 1:] & active[:, :-1]
    n_act = active.sum()
    return {
        "success": success,
        "crash": crash,
        "timeout": timeout,
        "t_b": np.where(success, t_b, -1),
        "t_f": np.where(crash, t_f, -1),
        "zmin": zmin,
        "hmin": hmin,
        "tiltmax": tiltmax,
        "wmax": wmax,
        "amax": amax,
        "counts": {
            "n_act": int(n_act),
            "thr_hi": int(thr_hi.sum()),
            "thr_lo": int(thr_lo.sum()),
            "rate_sat": int(rate_sat.sum()),
            "du_sum": float((du * du_active).sum()),
            "du_n": int(du_active.sum()),
            "m": int(m),
            "thr0_hi": int((un[:, 0, 0] > 0.98).sum()),
            "thr0_lo": int((un[:, 0, 0] < -0.98).sum()),
            "rate0_sat": int((np.abs(un[:, 0, 1:]) > 0.98).any(-1).sum()),
        },
        "thrust_samples": us[:, :N, 0][active],
        "omega_samples": us[:, :N, 1:][active],
    }


def region_table(stats: dict[str, dict], n_steps_dt: float) -> dict[str, dict]:
    out = {}
    for reg, s in stats.items():
        n = len(s["success"])
        tb = s["t_b"][s["success"]]
        zmin_s = s["zmin"][s["success"]]
        out[reg] = {
            "n": n,
            "m": float(s["success"].mean()),
            "crash": float(s["crash"].mean()),
            "timeout": float(s["timeout"].mean()),
            "hw": float(np.sqrt(np.log(2 / 0.05) / (2 * n))),
            "t_mean": float(tb.mean() * n_steps_dt) if tb.size else float("nan"),
            "t_p95": float(np.percentile(tb, 95) * n_steps_dt) if tb.size else float("nan"),
            "below_pad": float((zmin_s < 0.0).mean()) if zmin_s.size else float("nan"),
            "below_pad_5cm": float((zmin_s < -0.05).mean()) if zmin_s.size else float("nan"),
            "zmin_p1": float(np.percentile(zmin_s, 1)) if zmin_s.size else float("nan"),
            "tilt_p50": float(np.median(s["tiltmax"][s["success"]])) if tb.size else float("nan"),
            "tilt_p95": float(np.percentile(s["tiltmax"][s["success"]], 95)) if tb.size else float("nan"),
            "w_p95": float(np.percentile(s["wmax"][s["success"]], 95)) if tb.size else float("nan"),
            "a_p95": float(np.percentile(s["amax"][s["success"]], 95)) if tb.size else float("nan"),
            "hmin_p5": float(np.percentile(s["hmin"][s["success"]], 5)) if tb.size else float("nan"),
        }
    return out


# ---------------------------------------------------------------------- viability
def viability_opt(ctx: Ctx, X0: np.ndarray, *, n_rand: int = 7, iters: int = 600, lr: float = 0.08, seed: int = 0):
    """Open-loop trajectory optimisation: maximise min_k h(x_k) over the 2 s horizon.

    Returns the best achieved min-h per state and the best control/state sequence.
    A negative value after many restarts is numerical evidence (not proof) that the
    state is outside the viability kernel of the cone under the actuator limits.
    """
    mid, half, N, step, cone = ctx.mid, ctx.half, ctx.N, ctx.step, ctx.cone

    def rollout(x0, z):
        u = mid + half * jnp.tanh(z)

        def body(x, uk):
            xn = step(x, uk)
            return xn, (cone.value(xn), xn)

        _, (hs, xs) = jax.lax.scan(body, x0, u)
        return hs, xs, u

    def loss(z, x0, tau):
        hs, _, _ = rollout(x0, z)
        return tau * jax.scipy.special.logsumexp(-hs / tau)

    vg = jax.jit(jax.vmap(jax.value_and_grad(loss), in_axes=(0, 0, None)))
    evalf = jax.jit(jax.vmap(rollout))
    rng = np.random.default_rng(seed)
    m = X0.shape[0]
    hover = np.zeros((m, N, 4))
    hover[..., 0] = np.arctanh((ctx.cfg.gravity - float(mid[0])) / float(half[0]))
    climb = hover.copy()
    climb[..., 0] = np.arctanh(np.clip((30.0 - float(mid[0])) / float(half[0]), -0.99, 0.99))
    inits = [hover, climb] + [hover + 0.8 * rng.standard_normal((m, N, 4)) for _ in range(n_rand)]
    Z = jnp.asarray(np.concatenate(inits, 0))
    Xr = jnp.asarray(np.tile(X0, (len(inits), 1)))
    mom, vel = jnp.zeros_like(Z), jnp.zeros_like(Z)
    for i in range(iters):
        tau = 0.05 if i < iters // 2 else 0.01
        _, g = vg(Z, Xr, tau)
        g = jnp.nan_to_num(g)
        mom = 0.9 * mom + 0.1 * g
        vel = 0.999 * vel + 0.001 * g * g
        Z = Z - lr * (mom / (1 - 0.9 ** (i + 1))) / (jnp.sqrt(vel / (1 - 0.999 ** (i + 1))) + 1e-8)
    hs, xs, us = map(np.asarray, evalf(Xr, Z))
    k = len(inits)
    hmin = hs.min(-1).reshape(k, m)
    best = hmin.argmax(0)
    idx = best * m + np.arange(m)
    return hmin.max(0), xs[idx], us[idx]


# ------------------------------------------------------------------- sensitivity
def sensitivity(ctx: Ctx, pi: Callable, X: np.ndarray) -> dict[str, float]:
    """Actor Jacobian (normalised action per unit state) and peak closed-loop sensitivity.

    Phi_t = d x_t / d x_0 along the composed backup rollout; the CIL constraint rows are
    grad h(x_t) Phi_t, so their size along the horizon is what conditions the QP.
    """
    half, step, B, ctrl, cone, N = ctx.half, ctx.step, ctx.B, ctx.ctrl, ctx.cone, ctx.N
    Xj = jnp.asarray(X)
    J = jax.vmap(jax.jacfwd(lambda x: pi(x) / half))(Xj)
    jn = np.linalg.norm(np.asarray(J).reshape(len(X), -1), axis=1)

    def traj(x0):
        def body(x, _):
            u = jax.lax.select(B.contains(x), ctrl.action(x), pi(x))
            xn = step(x, u)
            return xn, xn

        _, xs = jax.lax.scan(body, x0, None, length=N)
        return xs

    Xsub = Xj[::8]
    Phis = np.asarray(jax.jit(jax.vmap(jax.jacfwd(traj)))(Xsub))
    xs = jax.jit(jax.vmap(traj))(Xsub)
    gh = np.asarray(jax.vmap(jax.vmap(lambda x: cone.values_and_grads(x)[1][0]))(xs))
    peak = np.linalg.norm(Phis, ord=2, axis=(2, 3)).max(1)
    rows = np.linalg.norm(np.einsum("mni,mnij->mnj", gh, Phis), axis=2).max(1)
    return {"jac_med": float(np.median(jn)), "jac_p95": float(np.percentile(jn, 95)),
            "phi_med": float(np.median(peak)), "phi_p95": float(np.percentile(peak, 95)), "phi_max": float(peak.max()),
            "row_p95": float(np.percentile(rows, 95)), "n_jac": int(len(X)), "n_phi": int(Xsub.shape[0])}


# ------------------------------------------------------------------------ slices
def cn_slice(ctx: Ctx, hit: Callable, *, v_out: float, v_z: float, nr: int = 121, nz: int = 91):
    """First-hit outcome on the (r, zeta) half-plane, level attitude, yaw 0, radial velocity v_out."""
    zmax = float(ctx.region_cfg.zeta_max)
    r_max = float(ctx.cone.radius_at(zmax))
    rs = np.linspace(0.0, r_max, nr)
    zs = np.linspace(0.0, zmax, nz)
    rr, zz = np.meshgrid(rs, zs)
    X = np.zeros((rr.size, 10), dtype=np.float32)
    X[:, 0] = rr.ravel() + ctx.cone.pad_x
    X[:, 1] = ctx.cone.pad_y
    X[:, 2] = zz.ravel() + ctx.cone.pad_z
    X[:, 3] = v_out
    X[:, 5] = v_z
    X[:, 6] = 1.0
    inside = ctx.h(X) >= 0.0
    inB = np.asarray(jax.vmap(ctx.B.contains)(jnp.asarray(X)))
    hb, hf, tb = map(np.asarray, hit(jnp.asarray(X)))
    code = np.full(rr.size, np.nan)
    code[inside & ~inB & hb] = 1.0
    code[inside & ~inB & ~hb] = 0.0
    code[inside & inB] = 2.0
    tsec = np.where(hb & inside & ~inB, tb * ctx.cfg.dt, np.nan)
    return {"r": rs, "z": zs, "code": code.reshape(rr.shape), "t": tsec.reshape(rr.shape),
            "frac_success": float(np.mean(hb[inside & ~inB])) if (inside & ~inB).any() else float("nan")}


# ------------------------------------------------------------------- main compute
def compute(root: Path, run: str, compare_run: str, *, log=print) -> dict[str, Any]:
    runs = list_runs(root)
    ctx = Ctx(root, run)
    cfg = ctx.cfg
    data: dict[str, Any] = {"root": str(root), "run": run, "compare_run": compare_run, "runs": runs}
    data["summaries"] = {r: load_json(root / r / "summary.json") for r in runs}
    data["configs"] = {r: load_json(root / r / "configs.json") for r in runs}
    data["histories"] = {r: load_json(root / r / "history.json") for r in runs}
    data["certificate"] = load_json(root / "certificate.json")
    data["cfg"] = cfg.as_dict()
    data["region_cfg"] = ctx.region_cfg.as_dict()

    log("[report] held-out test sets")
    sets = heldout_sets(ctx.region_cfg, ctx.env.sampler, split="test")
    data["test_sets"] = sets

    # ---- LQR design quantities
    P = ctx.ctrl.p_matrix_f64()
    K = ctx.ctrl.k_matrix_f64()
    A_d = np.asarray(ctx.ctrl.a_d)
    B_d = np.asarray(ctx.ctrl.b_d)
    A_c, B_c = ctx.ctrl.linearization_error_coords(cfg.gravity)
    eig_cl = np.linalg.eigvals(A_d - B_d @ K)
    p_inv = np.linalg.inv(P)
    ubar = np.minimum(np.asarray(ctx.ctrl.u_star) - np.asarray(ctx.ctrl.u_low), np.asarray(ctx.ctrl.u_high) - np.asarray(ctx.ctrl.u_star))
    c_u_rows = [float(ubar[i] ** 2 / (K[i] @ p_inv @ K[i])) for i in range(4)]
    data["lqr"] = {
        "A_c": A_c, "B_c": B_c, "A_d": A_d, "B_d": B_d, "P": P, "K": K,
        "eig_cl": eig_cl, "rho_cl": float(np.abs(eig_cl).max()),
        "c_u_rows": c_u_rows, "ubar": ubar,
        "extents": lc.support_radii(p_inv, cfg.base_set_c),
        "ctrb_rank": int(np.linalg.matrix_rank(np.hstack([np.linalg.matrix_power(A_d, i) @ B_d for i in range(9)]))),
    }
    # powerloop-weight comparison (chart/Lyapunov motivation for the yaw re-weighting)
    ctrl_pl = type(ctx.ctrl).from_config(cfg.replace(lqr_q_thetaz=0.16, lqr_r_omega_z=0.004))
    pinv_pl = np.linalg.inv(ctrl_pl.p_matrix_f64())
    data["lqr_powerloop_yaw"] = {"c_u": float(ctrl_pl.max_certified_level), "c_chart": lc.c_chart_bound(pinv_pl)}

    log("[report] Lyapunov decrease curve (adversarial)")
    levels = np.array([2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 13.0, 14.0, 15.0, 15.5, 16.0, 17.0, 18.0])
    worst = [lc.lyapunov_worst_adversarial(ctx.ctrl, ctx.step, level=float(L), n_starts=2048, iters=250)[0] for L in levels]
    worst_pl = [lc.lyapunov_worst_adversarial(ctrl_pl, ctx.step, level=float(L), n_starts=2048, iters=250)[0] for L in levels[:6]]
    data["lyap_curve"] = {"levels": levels, "worst": np.asarray(worst), "levels_pl": levels[:6], "worst_pl": np.asarray(worst_pl)}

    # ---- evaluation on the test sets: all runs + LQR-only baseline
    log("[report] evaluating all runs on the test sets")
    sim_n = {}
    outcomes: dict[str, dict[str, np.ndarray]] = {}
    region_stats: dict[str, dict] = {}
    policies = {"lqr_only": ctx.lqr_policy()}
    for r in runs:
        params, acfg = load_actor(root / r)
        policies[r] = ctx.actor_policy(params, acfg)
    for name, pi in policies.items():
        sim = ctx.simulator(pi, ctx.N)
        per_reg = {}
        for reg in REGION_NAMES:
            xs, us = map(np.asarray, sim(jnp.asarray(sets[reg])))
            per_reg[reg] = first_hit_stats(ctx, xs, us)
        outcomes[name] = {reg: per_reg[reg]["success"] for reg in REGION_NAMES}
        region_stats[name] = per_reg
        sim_n[name] = sim
    data["outcomes"] = outcomes
    data["region_tables"] = {name: region_table(per, cfg.dt) for name, per in region_stats.items()}
    def pooled(name: str) -> dict[str, float]:
        c = {k: sum(region_stats[name][rg]["counts"][k] for rg in REGION_NAMES) for k in region_stats[name]["general"]["counts"]}
        n = max(c["n_act"], 1)
        return {"frac_thr_max": c["thr_hi"] / n, "frac_thr_zero": c["thr_lo"] / n, "frac_rate_sat": c["rate_sat"] / n,
                "mean_du": c["du_sum"] / max(c["du_n"], 1), "n_steps": c["n_act"],
                "frac0_thr_sat": (c["thr0_hi"] + c["thr0_lo"]) / max(c["m"], 1), "frac0_rate_sat": c["rate0_sat"] / max(c["m"], 1)}

    data["actuator"] = {name: pooled(name) for name in (run, compare_run, "lqr_only")}
    data["actuator_samples"] = {
        name: {
            "thrust": np.concatenate([region_stats[name][rg]["thrust_samples"] for rg in REGION_NAMES]),
            "omega": np.concatenate([region_stats[name][rg]["omega_samples"] for rg in REGION_NAMES]),
        }
        for name in (run, compare_run, "lqr_only")
    }
    data["per_state"] = {name: {reg: {k: region_stats[name][reg][k] for k in ("success", "crash", "t_b", "t_f", "zmin", "hmin", "tiltmax", "wmax", "amax")}
                                for reg in REGION_NAMES} for name in (run, compare_run, "lqr_only")}

    # ---- failure overlap and viability
    log("[report] failure overlap and viability of shared failures")
    viab = {}
    for reg in ("general", "edge"):
        F = np.stack([~outcomes[r][reg] for r in runs])
        shared = np.where(F.all(0))[0]
        best, xs_opt, us_opt = viability_opt(ctx, sets[reg][shared])
        viab[reg] = {
            "n": int(F.shape[1]),
            "failed_by_all": int(len(shared)),
            "failed_by_any": int(F.any(0).sum()),
            "shared_idx": shared,
            "best_hmin": best,
            "n_nonviable": int((best < 0).sum()),
            "xs_opt": xs_opt,
            "us_opt": us_opt,
        }
    data["viability"] = viab
    saved_e = np.where(np.stack([outcomes[r]["edge"] for r in runs]).all(0))[0][:60]
    ctrl_best, _, _ = viability_opt(ctx, sets["edge"][saved_e])
    data["viability_control"] = {"n": int(len(saved_e)), "n_viable": int((ctrl_best >= 0).sum())}

    # ---- uncapped z_des sweep of the z-dependent bounds (the certificate caps them at c_rest)
    zg = np.round(np.arange(0.6, 2.5001, 0.02), 4)
    data["sweep"] = [{"z_des": float(z),
                      "c_cone_exact": lc.c_cone_exact(p_inv, ctx.cone, z_des=float(z), c_hi=100.0),
                      "c_cone_note": lc.c_cone_note(p_inv, ctx.cone, z_des=float(z), c_hi=100.0),
                      "c_ground": lc.c_ground_bound(p_inv, z_des=float(z), z_clear=cfg.z_clear)} for z in zg]

    # ---- smoothness and closed-loop sensitivity of every run (what the CIL differentiates)
    log("[report] smoothness and sensitivity")
    Xs = np.concatenate([sets["general"], sets["edge"]])
    data["sensitivity"] = {name: sensitivity(ctx, policies[name], Xs) for name in [*runs, "lqr_only"]}

    # ---- C_N slices (learned vs LQR-only)
    log("[report] C_N slices")
    conds = [("at rest", 0.0, 0.0), ("outward 1 m/s", 1.0, 0.0), ("outward 1 m/s, descending 1 m/s", 1.0, -1.0)]
    data["slices"] = {"conds": conds, "learned": [], "lqr_only": []}
    for name, key in ((run, "learned"), ("lqr_only", "lqr_only")):
        hit = ctx.hit_fn(policies[name])
        for _, vo, vz in conds:
            data["slices"][key].append(cn_slice(ctx, hit, v_out=vo, v_z=vz))

    # ---- overview fan and case studies (3 s rollouts)
    log("[report] case studies")
    sim_long = {name: ctx.simulator(policies[name], CASE_STEPS) for name in (run, compare_run, "lqr_only")}
    ps = data["per_state"][run]

    def traj(name: str, x0: np.ndarray) -> dict[str, np.ndarray]:
        xs, us = map(np.asarray, sim_long[name](jnp.asarray(x0[None])))
        xs, us = xs[0], us[0]
        V = ctx.V(xs)
        return {"x": xs, "u": us, "h": ctx.h(xs), "V": V, "inB": V <= cfg.base_set_c, "tilt": ctx.tilt_deg(xs)}

    rng = np.random.default_rng(3)
    fan = []
    for reg, n in (("general", 60), ("edge", 60)):
        idx = rng.choice(len(sets[reg]), n, replace=False)
        for i in idx:
            fan.append({"region": reg, "x0": sets[reg][i], **traj(run, sets[reg][i]),
                        "success": bool(ps[reg]["success"][i])})
    data["fan"] = fan

    def feats(X):
        dp = X[:, 0:2] - np.array([ctx.cone.pad_x, ctx.cone.pad_y])
        r = np.linalg.norm(dp, axis=1)
        rhat = dp / np.maximum(r, 1e-9)[:, None]
        return {"zeta": X[:, 2] - ctx.cone.pad_z, "r": r, "h": ctx.h(X), "v_out": np.sum(X[:, 3:5] * rhat, 1),
                "vz": X[:, 5], "speed": np.linalg.norm(X[:, 3:6], axis=1), "tilt": ctx.tilt_deg(X)}

    fe = {reg: feats(sets[reg]) for reg in REGION_NAMES}
    data["features"] = fe
    cases = []

    def pick(reg, mask, key, largest=True):
        cand = np.where(mask)[0]
        if cand.size == 0:
            return None
        vals = key[cand]
        return int(cand[np.argmax(vals) if largest else np.argmin(vals)])

    g, e = fe["general"], fe["edge"]
    sg, se = ps["general"]["success"], ps["edge"]["success"]
    i = pick("general", sg & (g["zeta"] > 2.2) & (g["vz"] < -1.0) & (g["h"] < 0.35), ps["general"]["t_b"])
    cases.append(("A", "general", i, "High, off-axis start descending toward the cone wall"))
    i = pick("edge", se & (e["v_out"] > 0.8) & (e["vz"] < 0.0) & (e["zeta"] < 0.6), -e["h"])
    cases.append(("B", "edge", i, "Low-altitude edge state moving outward and down"))
    i = pick("general", sg & (g["tilt"] > 28.0) & (g["zeta"] < 0.8) & (g["vz"] < -0.5), g["speed"])
    cases.append(("C", "general", i, "Large tilt, descending close to the pad"))
    i = pick("shell", ps["shell"]["success"], fe["shell"]["speed"])
    cases.append(("D", "shell", i, "Capture-shell state: short arrival and base-controller hold"))
    shared_e = viab["edge"]["shared_idx"]
    nonv = shared_e[viab["edge"]["best_hmin"] < 0]
    tf_nonv = ps["edge"]["t_f"][nonv]
    data["nonviable_tf"] = tf_nonv * cfg.dt
    j = int(nonv[np.argmax(tf_nonv)])
    cases.append(("E", "edge", j, "Failure: the longest-surviving non-viable edge state"))
    i = pick("general", sg & (ps["general"]["zmin"] < 0.0), -ps["general"]["zmin"])
    cases.append(("F", "general", i, "Successful recovery that dips below the pad plane (no ground in the model)"))
    both = se & data["per_state"][compare_run]["edge"]["success"]
    i = pick("edge", both & (e["v_out"] > 0.5), e["speed"])
    cases.append(("G", "edge", i, "Same edge state under the TD3 and SAC policies"))
    lqr_fail = ~data["per_state"]["lqr_only"]["edge"]["success"]
    cand = np.where(se & lqr_fail & (e["v_out"] > 0.6))[0]
    i = int(cand[np.argmin(np.abs(e["h"][cand] - np.median(e["h"][cand])) + np.abs(e["v_out"][cand] - np.median(e["v_out"][cand])))]) if cand.size else None
    cases.append(("H", "edge", i, "Where the learned policy matters: the base controller alone crashes"))

    case_data = []
    for tag, reg, idx, title in cases:
        if idx is None:
            continue
        x0 = sets[reg][idx]
        entry = {"tag": tag, "region": reg, "idx": idx, "title": title, "x0": x0,
                 "features": {k: float(v[idx]) for k, v in fe[reg].items()},
                 "main": traj(run, x0)}
        if tag == "E":
            k = int(np.where(shared_e == idx)[0][0])
            xs_opt = np.concatenate([x0[None], viab["edge"]["xs_opt"][k]], 0)
            entry["opt"] = {"x": xs_opt, "u": viab["edge"]["us_opt"][k], "h": ctx.h(xs_opt),
                            "best_hmin": float(viab["edge"]["best_hmin"][k])}
        if tag == "G":
            entry["compare"] = traj(compare_run, x0)
        if tag == "H":
            entry["compare"] = traj("lqr_only", x0)
        case_data.append(entry)
    data["cases"] = case_data
    return data


def save(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def load(path: Path) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)
