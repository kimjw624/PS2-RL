"""UE-bCBF control-invariant layer for landing under a bounded disturbance.

Counterpart of ``quadrotor_landing_backup_cbf`` (nominal landing CIL) for the
disturbance-aware Phase-I backup trained by ``scripts/train_phase1_landing_ue.py``.
The geometry, base set, LQR and action box still come from the checkpoint's
``landing_config``; the disturbance bounds and tube settings from its ``ue_config``.

Per solve, given the physical state x, the observer's estimate d_hat and its error bound
e_bar (constant after warm-up):

1. Roll out the composed backup pi_b(., d_hat) under the frozen estimate,
       x_{k+1} = Post(x_k + dt (f(x_k) + g(x_k) pi_b(x_k, d_hat) + E_d d_hat)),
   with exact discrete sensitivities Phi_k = dx_k/dx_0 and Theta_k = dx_k/dd_hat (the
   backup itself depends on d_hat), and the P-metric tube s_k
   (``ps2rl.uncertainty.landing_ue_tube``) along it.
2. Rows (j = cone, floor at every node k; base set at k = N), A u <= b:
       d/dt h_j(phi_k) = grad h_j Phi_k (f0 + E_d d_hat + g0 u) + grad h_j (Phi_k E_d + Theta_k Lambda) e
                         [ - rt_j grad h_j f_cl(phi_k) ]     (relative-time form, as the nominal CIL)
   and |e| <= e_bar gives the UE-bCBF row
       -grad h_j Phi_k g0 u <= alpha_j (h_j(phi_k) - m_j(s_k)) + grad h_j (Phi_k f0_hat - rt_j f_cl)
                               - e_bar || grad h_j (Phi_k E_d + Theta_k Lambda) ||.
   The tube margins m_j(s_k) make the rows hold for the *true* flow (to first order), the
   last term covers the observer error in the derivative (Lambda = observer gain, because
   d_hat_dot = Lambda (d - d_hat)).
3. QP (shared engine's row conditioning, box, slack), then a discrete safeguard on the
   frozen-estimate next state (as in the nominal landing CIL; candidates' tubes reuse the
   current rollout's tube shifted by one step, which is what keeps it cheap).

``rows`` are a function of (x, d_hat, e_bar) only, never of the policy. The Phase-II trainer
caches them at collection time and solves only the 5-variable QP inside the actor loss,
which is exactly the same projection and gradient as rebuilding them.

Experimental in the same sense as ``quadrotor_ue_bcbf_experimental``: the tube is a
first-order centerline tube with an empirical inflation factor and the margins'
dependence on x is not differentiated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import qpax

from ps2rl.backup_policy.backup_policy import BackupPolicy
from ps2rl.backup_policy.quadrotor_learned_backup import load_learned_quadrotor_backup_policy
from ps2rl.cil.backup_cbf import _build_backup_cbf_qp_from_rows
from ps2rl.cil.quadrotor_landing_backup_cbf import (
    CERTIFIED_FIELDS,
    QuadrotorLandingBCBFConfig,
    landing_sets,
    read_checkpoint_metadata,
)
from ps2rl.envs.quadrotor_env import quadrotor_control_affine_terms, quadrotor_dynamics
from ps2rl.envs.quadrotor_landing_config import QuadrotorLandingConfig
from ps2rl.phase1_sa.quadrotor_landing_ue_sa_env import landing_ue_step_fn
from ps2rl.uncertainty.landing_ue_tube import (
    LandingUEConfig,
    q_of_tau,
    tube_constants,
    tube_margins,
)

Array = jax.Array


@dataclass(frozen=True)
class QuadrotorLandingUEBCBFConfig(QuadrotorLandingBCBFConfig):
    """Nominal landing CIL config + UE settings (disturbance bounds come from the checkpoint)."""

    ue: LandingUEConfig = field(default_factory=LandingUEConfig)
    rho_scale: float = 1.0  # multiplies the observer-error term (1 = the UE-bCBF bound)

    @property
    def num_backup_inequalities(self) -> int:
        return self.num_safe_constraints * (self.num_steps + 1) + 1


def ue_bcbf_config_from_checkpoint(path: str | Path, **overrides: Any) -> QuadrotorLandingUEBCBFConfig:
    p = Path(path)
    ckpt = p / "landing_backup_policy_actor.pkl" if p.is_dir() else p
    md = read_checkpoint_metadata(ckpt)
    if "ue_config" not in md:
        raise KeyError(f"{ckpt} has no metadata['ue_config']; train it with scripts/train_phase1_landing_ue.py")
    lc = QuadrotorLandingConfig.from_dict(md["landing_config"])
    ue_ck = LandingUEConfig.from_dict(md["ue_config"])
    ue = overrides.pop("ue", ue_ck)
    # the runtime may assume a *smaller* disturbance than Phase I was trained for, never a larger one
    for name in ("delta_d", "delta_v", "e_bar"):
        if float(getattr(ue, name)) > float(getattr(ue_ck, name)) + 1e-12:
            raise ValueError(f"runtime ue.{name}={getattr(ue, name)} exceeds the Phase-I value {getattr(ue_ck, name)}")
    if float(ue.tube_scale) < float(ue_ck.tube_scale) - 1e-12:
        raise ValueError("runtime tube_scale below the Phase-I value would loosen the tube")
    return QuadrotorLandingUEBCBFConfig(
        backup_policy_mode="learned", learned_backup_policy_path=str(ckpt), landing=lc, ue=ue, **overrides
    )


class UERuntime(NamedTuple):
    pi_b: Callable  # (x, d_hat) -> u
    step: Callable  # (x, d_hat) -> x+ under pi_b and the frozen estimate
    plant: Callable  # (x, u, d_hat) -> x+
    f_cl: Callable  # (x, d_hat) -> x_dot under pi_b (with E_d d_hat)
    safe_vg: Callable  # x -> (h (J,), grad (J, 10))
    base_vg: Callable  # x -> (h (1,), grad (1, 10))
    base_contains: Callable
    safe_contains_tight: Callable  # (x, s) -> bool
    base_contains_tight: Callable  # (x, s) -> bool
    growth_from_f10: Callable  # (x, x_next, F10) -> ||F||_P
    tube: Any
    qp_system: Any


def make_ue_runtime(cfg: QuadrotorLandingUEBCBFConfig) -> UERuntime:
    lc = cfg.landing
    cone, base_set = landing_sets(cfg)
    ctrl = base_set.controller
    learned = load_learned_quadrotor_backup_policy(cfg.learned_backup_policy_path)
    md = learned.metadata
    trained = QuadrotorLandingConfig.from_dict(md["landing_config"]).as_dict()
    runtime = lc.as_dict()
    bad = [k for k in CERTIFIED_FIELDS if not np.isclose(float(trained[k]), float(runtime[k]), rtol=1e-9, atol=1e-12)]
    if bad:
        raise ValueError(f"landing config differs from the checkpoint on {bad}")
    if int(learned.actor_cfg.obs_dim) != 13:
        raise ValueError(f"UE backup must take (x, d_hat) (obs_dim 13), checkpoint has {learned.actor_cfg.obs_dim}")
    low = jnp.asarray(cfg.action_low, dtype=jnp.float32)
    high = jnp.asarray(cfg.action_high, dtype=jnp.float32)
    plant = landing_ue_step_fn(lc)
    tc = tube_constants(lc)
    ue = cfg.ue

    def pi_b(x, d_hat):
        raw = learned.action_single(jnp.concatenate([x, d_hat]).astype(jnp.float32)).astype(x.dtype)
        raw = jnp.clip(jnp.nan_to_num(raw), low, high)
        return jnp.clip(BackupPolicy.select_action(x, raw, base_set), low, high)

    def step(x, d_hat):
        return plant(x, pi_b(x, d_hat), d_hat)

    def f_cl(x, d_hat):
        u = pi_b(x, d_hat)
        xd = quadrotor_dynamics(x, u, lc.gravity, lc.a_cmd_min, lc.a_cmd_max, lc.omega_max)
        return xd.at[3:6].add(d_hat)

    ph = jnp.asarray(tc.p_half, dtype=jnp.float32)
    pmh = jnp.asarray(tc.p_mhalf, dtype=jnp.float32)

    def growth_from_f10(x, x_next, f10):
        xi = ctrl.error_state(x)
        dx_dxi = jax.jacfwd(ctrl.state_from_error)(xi)
        dxi_dx = jax.jacfwd(ctrl.error_state)(x_next)
        m = ph @ (dxi_dx @ f10 @ dx_dxi) @ pmh
        return jnp.sqrt(jnp.maximum(jnp.linalg.eigvalsh(m.T @ m)[-1], 0.0))

    def safe_tight(x, s):
        m_c, m_f, _ = tube_margins(s, ue, tc)
        h = cone.component_values(x)
        ok = h[0] >= m_c
        if cone.floor:
            ok &= h[1] >= m_f
        return ok

    def base_tight(x, s):
        _, _, m_b = tube_margins(s, ue, tc)
        return base_set.margin(x) >= m_b

    qp_system = SimpleNamespace(action_dim=4, action_low=tuple(cfg.action_low), action_high=tuple(cfg.action_high))
    return UERuntime(pi_b=pi_b, step=step, plant=plant, f_cl=f_cl, safe_vg=cone.values_and_grads,
                     base_vg=base_set.values_and_grads, base_contains=base_set.contains,
                     safe_contains_tight=safe_tight, base_contains_tight=base_tight,
                     growth_from_f10=growth_from_f10, tube=tc, qp_system=qp_system)


# ------------------------------------------------------------------------------- rows
def rollout_ue(x0: Array, d_hat: Array, cfg: QuadrotorLandingUEBCBFConfig, rt: UERuntime):
    """Frozen-estimate rollout: xs (N+1, 10), Phi (N+1, 10, 10), Theta (N+1, 10, 3), s (N+1,)."""
    n = int(cfg.num_steps)
    dt = float(cfg.dt)
    tc, ue = rt.tube, cfg.ue
    jx = jax.jacfwd(rt.step, argnums=0)
    jd = jax.jacfwd(rt.step, argnums=1)

    def body(c, k):
        x, phi, theta, s = c
        x_next = rt.step(x, d_hat)
        f10 = jx(x, d_hat)
        g10 = jd(x, d_hat)
        gr = rt.growth_from_f10(x, x_next, f10)
        s_next = gr * s + dt * tc.gamma * q_of_tau(k * dt, ue)
        out = (x_next, f10 @ phi, f10 @ theta + g10, s_next)
        return out, out

    c0 = (x0, jnp.eye(10, dtype=x0.dtype), jnp.zeros((10, 3), x0.dtype), jnp.asarray(0.0, x0.dtype))
    _, (xs, phis, thetas, ss) = jax.lax.scan(body, c0, jnp.arange(n, dtype=x0.dtype))
    cat = lambda a, b: jnp.concatenate([a[None], b], axis=0)
    return cat(c0[0], xs), cat(c0[1], phis), cat(c0[2], thetas), cat(c0[3], ss)


def build_ue_rows(x: Array, d_hat: Array, e_bar: Array, cfg: QuadrotorLandingUEBCBFConfig, rt: UERuntime):
    """A u <= b rows (R = J (N+1) + 1) and diagnostics; depends on (x, d_hat, e_bar) only."""
    xs, phis, thetas, ss = rollout_ue(x, d_hat, cfg, rt)
    f0, g0 = quadrotor_control_affine_terms(x, cfg.gravity)
    ed = jnp.zeros((10, 3), x.dtype).at[3:6].set(jnp.eye(3, dtype=x.dtype))
    f0_hat = f0 + ed @ d_hat
    lam = float(cfg.ue.observer_lambda)
    alpha_vec = cfg.alpha_per_constraint
    rt_vec = cfg.relative_time_per_constraint
    j_safe = cfg.num_safe_constraints
    alphas = jnp.asarray(alpha_vec if alpha_vec is not None else (cfg.alpha,) * j_safe, dtype=x.dtype)
    rts = jnp.asarray(rt_vec if rt_vec is not None else (cfg.include_relative_time_term,) * j_safe, dtype=x.dtype)
    rho_scale = float(cfg.rho_scale)

    def per_node(xi, phi, theta, s):
        h, dh = rt.safe_vg(xi)  # (J,), (J, 10)
        m_c, m_f, _ = tube_margins(s, cfg.ue, rt.tube)
        m = jnp.stack([m_c, m_f])[:j_safe]
        a = -(dh @ (phi @ g0))
        fcl = rt.f_cl(xi, d_hat)
        flow = dh @ (phi @ f0_hat) - rts * (dh @ fcl)
        ue_dir = dh @ (phi @ ed + lam * theta)  # (J, 3)
        rho = e_bar * jnp.linalg.norm(ue_dir, axis=-1)
        b = alphas * (h - m) + flow - rho_scale * rho
        return a, b, h - m, rho

    a_seq, b_seq, h_rob, rho_seq = jax.vmap(per_node)(xs, phis, thetas, ss)
    a_rows = a_seq.reshape((-1, 4))
    b_rows = b_seq.reshape((-1,))

    hb, dhb = rt.base_vg(xs[-1])
    _, _, m_b = tube_margins(ss[-1], cfg.ue, rt.tube)
    a_t = -(dhb @ (phis[-1] @ g0))
    rho_t = e_bar * jnp.linalg.norm(dhb @ (phis[-1] @ ed + lam * thetas[-1]), axis=-1)
    b_t = cfg.base_alpha * (hb - m_b) + dhb @ (phis[-1] @ f0_hat) - rho_scale * rho_t
    a_rows = jnp.concatenate([a_rows, a_t], axis=0)
    b_rows = jnp.concatenate([b_rows, b_t], axis=0)
    finite = jnp.all(jnp.isfinite(a_rows)) & jnp.all(jnp.isfinite(b_rows))
    diag = {
        "tube_T": ss[-1],
        "min_robust_safe_h": jnp.min(h_rob),
        "terminal_robust_h_b": (hb - m_b)[0],
        "max_rho": jnp.maximum(jnp.max(rho_seq), rho_t[0]),
        "rows_finite": finite,
    }
    return a_rows, b_rows, ss, diag


def solve_qp_from_rows(a_rows, b_rows, u_ref, u_backup, cfg: QuadrotorLandingUEBCBFConfig, rt: UERuntime,
                       qp_dtype=None):
    """min |u - u_ref|^2 + w s^2  s.t. rows (with slack), box. Differentiable in u_ref.

    Returns (u, slack, used_solver). Non-finite rows/solution -> u_backup (no gradient).
    """
    dtype = u_ref.dtype
    u_ref = jnp.clip(jnp.nan_to_num(u_ref), jnp.asarray(cfg.action_low, dtype), jnp.asarray(cfg.action_high, dtype))
    q_mat, q_vec, a_eq, b_eq, g, h = _build_backup_cbf_qp_from_rows(a_rows, b_rows, u_ref, cfg, rt.qp_system,
                                                                    dtype=dtype)
    ok_in = jnp.all(jnp.isfinite(g)) & jnp.all(jnp.isfinite(h))
    g = jnp.where(ok_in, g, 0.0)
    h = jnp.where(ok_in, h, 1.0)
    if qp_dtype is None:
        z = qpax.solve_qp_primal(q_mat, q_vec, a_eq, b_eq, g, h, solver_tol=cfg.solver_tol,
                                 target_kappa=cfg.target_kappa)
    else:
        z = qpax.solve_qp_primal(q_mat.astype(qp_dtype), q_vec.astype(qp_dtype), a_eq.astype(qp_dtype),
                                 b_eq.astype(qp_dtype), g.astype(qp_dtype), h.astype(qp_dtype),
                                 solver_tol=cfg.solver_tol, target_kappa=cfg.target_kappa).astype(dtype)
    used = ok_in & jnp.all(jnp.isfinite(z))
    fb = jnp.concatenate([u_backup.astype(dtype), jnp.zeros((1,), dtype)])
    z = jnp.where(used, jnp.nan_to_num(z), jax.lax.stop_gradient(fb))
    u = jnp.clip(z[:4], jnp.asarray(cfg.action_low, dtype), jnp.asarray(cfg.action_high, dtype))
    return u, jnp.maximum(z[4], 0.0), used


def safeguard(x, d_hat, u, ss, cfg: QuadrotorLandingUEBCBFConfig, rt: UERuntime):
    """Largest lambda with plant(x, lambda u + (1 - lambda) pi_b) in the tightened C_N (tube = ss shifted)."""
    n = int(cfg.num_steps)
    lams = jnp.asarray(cfg.safeguard_lambdas, dtype=u.dtype)
    ub = rt.pi_b(x, d_hat)
    lo, hi = jnp.asarray(cfg.action_low, u.dtype), jnp.asarray(cfg.action_high, u.dtype)
    cand = jnp.clip(lams[:, None] * u[None] + (1.0 - lams)[:, None] * ub[None], lo, hi)
    s_shift = jnp.concatenate([ss[1:], ss[-1:]])  # node k of the candidate's rollout ~ node k+1 of x's

    def recoverable(x0):
        def body(c, k):
            z, hb, hf = c
            zn = rt.step(z, d_hat)
            sk = s_shift[jnp.minimum(k + 1, n)]
            act = ~(hb | hf)
            nf = act & ~rt.safe_contains_tight(zn, sk)
            nb = act & ~nf & rt.base_contains_tight(zn, sk)
            return (jnp.where(act, zn, z), hb | nb, hf | nf), None

        hb0 = rt.base_contains_tight(x0, s_shift[0])
        hf0 = ~rt.safe_contains_tight(x0, s_shift[0])
        (_, hb, hf), _ = jax.lax.scan(body, (x0, hb0, hf0), jnp.arange(n))
        return hb & ~hf

    xn = jax.vmap(lambda uu: rt.plant(x, uu, d_hat))(cand)
    ok = jax.vmap(recoverable)(xn)
    found = jnp.any(ok)
    idx = jnp.where(found, jnp.argmax(ok), lams.shape[0] - 1)
    return cand[idx], lams[idx], found


def project_full(x, d_hat, e_bar, u_ref, cfg, rt, qp_dtype=None, use_safeguard=True):
    """rows -> QP -> safeguard for one state. Returns (u_safe, aux dict incl. rows for caching)."""
    a_rows, b_rows, ss, diag = build_ue_rows(x, d_hat, e_bar, cfg, rt)
    ub = rt.pi_b(x, d_hat)
    u_qp, slack, used = solve_qp_from_rows(a_rows, b_rows, u_ref, ub, cfg, rt, qp_dtype)
    if use_safeguard and bool(cfg.discrete_safeguard):
        u, lam, found = safeguard(x, d_hat, u_qp, ss, cfg, rt)
    else:
        u, lam, found = u_qp, jnp.asarray(1.0, u_qp.dtype), jnp.asarray(True)
    aux = {"a_rows": a_rows, "b_rows": b_rows, "u_backup": ub, "u_qp": u_qp, "slack": slack, "used_solver": used,
           "safeguard_lambda": lam, "safeguard_found": found, **diag}
    return u, aux


def make_recoverability_fn_ue(cfg: QuadrotorLandingUEBCBFConfig, rt: UERuntime):
    """(x0, d_hat) batch -> tightened first-hit C_N membership with the exact tube (for initial-state banks)."""
    n = int(cfg.num_steps)
    dt = float(cfg.dt)
    jx = jax.jacfwd(rt.step, argnums=0)

    def one(x0, d_hat):
        def body(c, k):
            z, s, hb, hf = c
            zn = rt.step(z, d_hat)
            gr = rt.growth_from_f10(z, zn, jx(z, d_hat))
            sn = gr * s + dt * rt.tube.gamma * q_of_tau(k * dt, cfg.ue)
            act = ~(hb | hf)
            nf = act & ~rt.safe_contains_tight(zn, sn)
            nb = act & ~nf & rt.base_contains_tight(zn, sn)
            return (jnp.where(act, zn, z), jnp.where(act, sn, s), hb | nb, hf | nf), None

        z0 = jnp.asarray(0.0, x0.dtype)
        (_, _, hb, hf), _ = jax.lax.scan(
            body, (x0, z0, rt.base_contains_tight(x0, z0), ~rt.safe_contains_tight(x0, z0)), jnp.arange(n, dtype=x0.dtype))
        return hb & ~hf

    return jax.jit(jax.vmap(one))


# ------------------------------------------------------------- engine-style facade
# Same call signatures as ``quadrotor_landing_backup_cbf`` (what PS2SystemBinding expects),
# with the "physical state" extended to xd = (x (10), d_hat (3)) = obs[..., :13].
_UE_RUNTIME_CACHE: Dict[tuple, UERuntime] = {}


def get_cached_runtime(cfg: QuadrotorLandingUEBCBFConfig) -> UERuntime:
    key = (cfg.learned_backup_policy_path, cfg.landing, cfg.ue)
    if key not in _UE_RUNTIME_CACHE:
        _UE_RUNTIME_CACHE[key] = make_ue_runtime(cfg)
    return _UE_RUNTIME_CACHE[key]


def _qp_dtype(cfg):
    return jnp.float64 if str(getattr(cfg, "qp_solve_dtype", "float32")) == "float64" else None


def solve_backup_cbf_qp_batch_with_info(xd_batch, u_ref_batch, cfg, runtime=None):
    """Batch UE projection. Returns (u_safe, slack, used_solver, info) like the shared engine."""
    rt = get_cached_runtime(cfg) if runtime is None else runtime
    xd_batch = jnp.asarray(xd_batch)
    x = jnp.nan_to_num(xd_batch[..., :10]).astype(jnp.float32)
    dh = jnp.nan_to_num(xd_batch[..., 10:13]).astype(jnp.float32)
    u_ref = jnp.asarray(u_ref_batch).astype(jnp.float32)
    e_bar = jnp.asarray(float(cfg.ue.e_bar), jnp.float32)
    u, aux = jax.vmap(lambda xx, dd, uu: project_full(xx, dd, e_bar, uu, cfg, rt, _qp_dtype(cfg)))(x, dh, u_ref)
    u = u.astype(u_ref.dtype)
    ones = jnp.ones(u.shape[:1], bool)
    zeros = jnp.zeros(u.shape[:1], u.dtype)
    used = aux["used_solver"]
    delta = u_ref - u
    info = {
        "q_mat_finite": ones, "q_vec_finite": ones, "g_finite": aux["rows_finite"], "h_finite": aux["rows_finite"],
        "inputs_finite": aux["rows_finite"], "z_finite": used, "q_saturated": ~ones, "max_abs_q": zeros,
        "max_abs_b": zeros, "delta_min_u_ref": zeros,
        "u_ref_minus_u_safe_norm": jnp.linalg.norm(delta, axis=-1), "a_ref_minus_a_safe": delta[:, 0],
        "r_ref_minus_r_safe": delta[:, 1],
        "safeguard_lambda": aux["safeguard_lambda"], "tube_T": aux["tube_T"],
    }
    return u, aux["slack"].astype(u.dtype), used, info


def solve_backup_cbf_qp_batch(xd_batch, u_ref_batch, cfg, runtime=None):
    u, slack, _, _ = solve_backup_cbf_qp_batch_with_info(xd_batch, u_ref_batch, cfg, runtime)
    return u, slack


def backup_policy_batch(xd_batch, cfg, runtime=None):
    """pi_b(x, d_hat) for a batch of xd = (x, d_hat)."""
    rt = get_cached_runtime(cfg) if runtime is None else runtime
    xd_batch = jnp.asarray(xd_batch)
    return jax.vmap(rt.pi_b)(xd_batch[..., :10].astype(jnp.float32), xd_batch[..., 10:13].astype(jnp.float32))


class QuadrotorLandingUEBackupCBFProjector:
    """Drop-in for ``QuadrotorLandingBackupCBFProjector`` (what the Phase-II entry/trainer instantiate)."""

    def __init__(self, cfg: QuadrotorLandingUEBCBFConfig, runtime: UERuntime | None = None):
        self.cfg = cfg
        self.runtime = get_cached_runtime(cfg) if runtime is None else runtime
        self._solve_batch_with_info = jax.jit(lambda x, u: solve_backup_cbf_qp_batch_with_info(x, u, self.cfg, self.runtime))

    def solve_batch_with_info(self, xd_batch, u_ref_batch):
        return self._solve_batch_with_info(xd_batch, u_ref_batch)

    def solve_batch(self, xd_batch, u_ref_batch):
        return self._solve_batch_with_info(xd_batch, u_ref_batch)[:2]

    @property
    def num_backup_inequalities(self) -> int:
        return self.cfg.num_backup_inequalities

    @property
    def num_qp_inequalities(self) -> int:
        return self.cfg.num_qp_inequalities


__all__ = [
    "QuadrotorLandingUEBackupCBFProjector",
    "backup_policy_batch",
    "get_cached_runtime",
    "solve_backup_cbf_qp_batch",
    "solve_backup_cbf_qp_batch_with_info",
    "QuadrotorLandingUEBCBFConfig",
    "UERuntime",
    "build_ue_rows",
    "make_recoverability_fn_ue",
    "make_ue_runtime",
    "project_full",
    "rollout_ue",
    "safeguard",
    "solve_qp_from_rows",
    "ue_bcbf_config_from_checkpoint",
]
