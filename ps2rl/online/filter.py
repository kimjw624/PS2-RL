"""Discrete backup safety filter for tall cylinders, sized for a companion computer (numpy, batched over vehicles).

The strike filter (``ps2rl.cil.strike_filter``, static obstacles) with the compute knobs exposed:

  candidates   u(lam, k) = lam u_nom + (1 - lam) pi_k(x),  lam in ``lambdas`` (1 first, 0 last), k in ``backups``
               ("learned": the Phase-I actor, hover LQR inside the base set; "lqr": the re-anchored hover LQR)
  certificate  the candidate is applied for ``period`` control steps (re-evaluated every step from the state: u_nom
               and pi_k are state feedback), every state on the way is safe, and from the state after the window the
               backup reaches the base set within N = horizon / dt steps without leaving the safe set (first hit, the
               Phase-I criterion). With period 1 this is the strike filter's check exactly.
  decision     the largest certified lam (ties: the earlier backup in ``backups``); else the continuation - the held
               hover at the latched anchor, or the committed plan (the backup with the scene view of its certificate,
               frozen) - when it is certified; else no certificate: the continuation is applied anyway (reported).
  hover latch  once a certified plan ends a window in the base set, its anchor is kept and the continuation is the
               hover LQR at that fixed anchor (the controller the base-set certificate covers).

Disturbance: with an estimate d of the acceleration push (``DisturbanceObserver``), the rollouts add d to the model and
the backups and the base set work about the hover equilibrium that cancels it (``ps2rl.online.model``); the nominal is
simulated with d as well. d = 0 is the Phase-I model.

Everything is checked against the known cylinders (``obst``: the ones sensed so far) with ``margin`` added to the drone
radius; new plans need ``fresh_margin`` more (the strike filter's fresh-plan slack): a plan certified with that slack is
still certified as the continuation at the next decision after the estimate noise and the model error of one window,
instead of the certificate flickering off at the boundary. The backups switch to the hover LQR in the stricter base
set (margin + fresh_margin) in every plan, so a continued plan is the plan that was certified. The backup sees the n_slots nearest cylinders within ``obstacle_range`` (the Phase-I region's dist_max), frozen
for the rollout. Between decisions (period > 1) the vehicle applies the stored plan with fresh state feedback
(``command``), which is what the rollout simulated: the nominal is simulated with the goal and the known cylinders of the
decision, so ``command`` must be given the nominal computed with those (``plan_nominal_inputs``).

Compute per decision: (|lambdas| x |backups| + 1) rollouts of up to period + N steps, batched in numpy (the Python
loop runs over steps, not candidates); ``early_exit`` stops once every rollout has reached the base set or failed.

Command hold (``hold``): a control period of hold model steps (hold x dt; the autopilot holds the last rate setpoint
between commands). Every command - the candidate blend, the nominal simulated in the window, each backup command, the
held hover - is computed at a command time and held for hold model steps; the window is period x hold model steps; the
safe set is checked at every model step, the base set (and the latch) only at command times. hold = 1: as before.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import time
from typing import Callable

import numpy as np

from ps2rl.online.model import ABSENT, Actor, CylModel, anchor, combine, held_action, held_level, local_view, lqr_action, \
    margin_parts, margins, rvec, step, surface_distance

MODES = ("fresh", "committed", "held", "none")


@dataclass
class FilterConfig:
    lambdas: tuple = (1.0, 0.5, 0.0)
    backups: tuple = ("learned",)
    horizon: float = 1.0          # backup rollout after the window [s]
    period: int = 1               # control steps (commands) per decision
    hold: int = 1                 # model steps per command (control period = hold x dt)
    obstacle_range: float = 5.0   # the backup's view (Phase I region dist_max) [m]
    margin: float = 0.1           # extra clearance added to the drone radius in every check [m]
    fresh_margin: float = 0.15    # more clearance for new plans (the continuation keeps ``margin`` only) [m]
    hover_latch: bool = True
    early_exit: bool = True
    backend: str = "auto"         # "numpy" | "numba" (compiled rollouts, ps2rl.online.fast) | "auto" (numba if installed)

    def __post_init__(self) -> None:
        lams = sorted({float(v) for v in self.lambdas} | {0.0}, reverse=True)
        self.lambdas = tuple(lams)
        for b in self.backups:
            if b not in ("learned", "lqr"):
                raise ValueError(f"backups must be 'learned' or 'lqr', got {b!r}")
        if len(self.backups) == 0:
            raise ValueError("at least one backup")
        self.backups = tuple(self.backups)
        if int(self.period) < 1:
            raise ValueError("period >= 1")
        if int(self.hold) < 1:
            raise ValueError("hold >= 1")

    @property
    def n_candidates(self) -> int:
        return len(self.lambdas) * len(self.backups)

    def as_dict(self) -> dict:
        d = asdict(self)
        d["lambdas"], d["backups"] = list(self.lambdas), list(self.backups)
        return d


@dataclass
class Plan:
    """What the vehicle applies until the next decision (one entry per vehicle)."""
    lam: np.ndarray        # (E,)
    k: np.ndarray          # (E,) index into backups
    held: np.ndarray       # (E,) bool: hover at the latched anchor
    a0: np.ndarray         # (E, 3)
    view: np.ndarray       # (E, n_slots, 4) the backup's frozen scene
    obst: np.ndarray       # (E, M, 4) the known cylinders of the decision
    goal: np.ndarray       # (E, 3) the nominal's goal of the decision
    d: np.ndarray          # (E, 3) the push of the decision
    mode: np.ndarray       # (E,) index into MODES
    stats: dict = field(default_factory=dict)


class OnlineFilter:
    def __init__(self, m: CylModel, actor: Actor | None, cfg: FilterConfig, nominal: Callable | None = None) -> None:
        if "learned" in cfg.backups and actor is None:
            raise ValueError("backup 'learned' needs an actor (an exported Phase-I run)")
        if cfg.period > 1 and nominal is None:
            raise ValueError("period > 1: the filter simulates the nominal, pass it")
        self.m, self.actor, self.cfg, self.nominal = m, actor, cfg, nominal
        self.N = int(round(cfg.horizon / m.dt))
        lam_c = np.repeat(np.asarray(cfg.lambdas), len(cfg.backups))           # lam-major, backups in order
        k_c = np.tile(np.arange(len(cfg.backups)), len(cfg.lambdas))
        self.cand_lam, self.cand_k = lam_c, k_c
        self.learned_k = np.asarray([b == "learned" for b in cfg.backups])
        self.switch_extra = cfg.margin + cfg.fresh_margin           # where every backup hands over to the hover LQR
        self._force_actor = False                                   # timing only (see _rollouts_numpy)
        self.committed = None
        from ps2rl.online.fast import HAVE_NUMBA, actor_arrays
        if cfg.backend not in ("numpy", "numba", "auto"):
            raise ValueError(f"backend must be numpy, numba or auto, got {cfg.backend!r}")
        if cfg.backend == "numba" and not HAVE_NUMBA:
            raise ValueError("backend 'numba' needs numba (pip install numba)")
        self.use_numba = cfg.backend == "numba" or (cfg.backend == "auto" and HAVE_NUMBA)
        f = lambda a: np.ascontiguousarray(a, dtype=np.float64)
        self._k = (m.dt, m.gravity, f(m.low), f(m.high), m.z_floor, m.r_drone, m.c_B, m.rho_xy, m.z_ext, f(m.S),
                   f(m.DT), f(m.KtT), *actor_arrays(actor, m.obs_dim), m.backup_action == "residual_lqr")

    # ------------------------------------------------------------------ state
    def reset(self, x: np.ndarray, obst: np.ndarray, goal: np.ndarray) -> None:
        E = x.shape[0]
        self.committed = {"view": local_view(self.m, x, obst, self.cfg.obstacle_range), "obst": obst.copy(),
                          "goal": goal.copy(), "k": np.zeros(E, int), "latched": np.zeros(E, bool),
                          "a0": x[:, 0:3].copy()}
        self.plan = None

    # ------------------------------------------------------------------ backups
    def backup_action(self, x: np.ndarray, view: np.ndarray, obst: np.ndarray, learned: np.ndarray, d: np.ndarray,
                      inb: np.ndarray | None = None) -> np.ndarray:
        """Composed backup (as in Phase I): the hover LQR inside the base set, else the backup of the row
        (learned rows: the actor; lqr rows: the hover LQR)."""
        m = self.m
        r = rvec(x, d, m)
        u = lqr_action(m, x, r, d)
        if self.actor is not None and learned.any():
            if inb is None:
                inb = margins(m, x, obst, self.switch_extra, d)[1] >= 0.0
            use = learned if self._force_actor else learned & ~inb
            if use.any():
                idx = np.nonzero(use)[0]
                u[idx] = self.actor(x[idx], view[idx], r[idx], d[idx])
        return u

    # ------------------------------------------------------------------ decision
    def decide(self, x: np.ndarray, obst: np.ndarray, goal: np.ndarray, u_nom: np.ndarray | None = None,
               d: np.ndarray | None = None) -> Plan:
        """x (E, 10) state estimate, obst (E, M, 4) known cylinders, goal (E, 3), u_nom (E, 4) the nominal at x
        (period 1; with period > 1 the filter simulates the nominal itself), d (E, 3) the push estimate (None: 0)."""
        t0 = time.perf_counter()
        m, cfg = self.m, self.cfg
        E, M = x.shape[0], obst.shape[1]
        P, N, H = int(cfg.period), self.N, int(cfg.hold)
        d = np.zeros((E, 3)) if d is None else np.asarray(d, float)
        com = self.committed
        if com["obst"].shape[1] != M:                     # the known set grew: pad the committed copies
            pad = np.zeros((E, M - com["obst"].shape[1], 4))
            com["obst"] = np.concatenate([com["obst"], pad], axis=1)
        fresh_view = local_view(m, x, obst, cfg.obstacle_range)
        C = cfg.n_candidates
        R = C + 1                                          # + the committed continuation (lam 0)
        lam = np.concatenate([self.cand_lam, [0.0]])
        rows_k = np.concatenate([np.broadcast_to(self.cand_k, (E, C)), com["k"][:, None]], axis=1)       # (E, R)
        views = np.concatenate([np.broadcast_to(fresh_view[:, None], (E, C) + fresh_view.shape[1:]),
                                com["view"][:, None]], axis=1).reshape(E * R, m.n_slots, 4)
        obs_r = np.concatenate([np.broadcast_to(obst[:, None], (E, C, M, 4)), com["obst"][:, None]],
                               axis=1).reshape(E * R, M, 4)
        goal_r = np.concatenate([np.broadcast_to(goal[:, None], (E, C, 3)), com["goal"][:, None]], axis=1).reshape(E * R, 3)
        learned = self.learned_k[rows_k].reshape(-1)
        d_r = np.repeat(d, R, axis=0)
        extra = np.tile(np.concatenate([np.full(C, cfg.margin + cfg.fresh_margin), [cfg.margin]]), E)
        lam_r = np.broadcast_to(lam, (E, R)).reshape(-1)
        X = np.repeat(x, R, axis=0)
        fail = np.zeros(E * R, bool)
        steps = 0
        if self.use_numba:
            hit, fail, used, X_P, b_P, steps = self._rollouts_numba(X, P, N, lam_r, u_nom, R, learned, views, obs_r, d_r,
                                                                     extra, goal_r)
        else:
            hit, fail, used, X_P, b_P, steps = self._rollouts_numpy(X, P, N, lam_r, u_nom, R, learned, views, obs_r, d_r,
                                                                     extra, goal_r)
        cert = (hit & ~fail).reshape(E, R)
        ok_f = cert[:, :C].any(axis=1)
        i_f = np.argmax(cert[:, :C], axis=1)
        ok_c = cert[:, C]
        # held hover at the latched anchor, P steps (certified: level <= c_B throughout and the anchor clear)
        latched = com["latched"] & cfg.hover_latch
        ok_h = np.zeros(E, bool)
        Xh_P = x.copy()
        if latched.any():
            a0 = com["a0"]
            Xh = x.copy()
            ok_h = latched.copy()
            for i in range(P * H):
                if i % H == 0:
                    uh = held_action(m, Xh, a0, d)
                Xh = step(m, Xh, uh, d)
                ok_h &= held_level(m, Xh, a0, d) <= m.c_B
            clear = surface_distance(a0[:, 0:2], obst).min(axis=1, initial=ABSENT) - m.rho_xy - m.r_drone - cfg.margin
            ok_h &= (clear >= 0.0) & (a0[:, 2] - m.z_ext - m.z_floor >= 0.0)
            Xh_P = Xh
        return self._choose(x, obst, goal, d, E, R, C, P, lam, rows_k, fresh_view, cert, ok_f, i_f, ok_c, latched, ok_h,
                            Xh_P, X_P, b_P, used, steps, t0)

    def _window_numpy(self, X, P, lam_r, u_nom, R, learned, views, obs_r, d_r, extra, goal_r):
        """The candidates' window: P commands of the blend, each held for hold model steps (rows: the candidates, last per
        vehicle the continuation, lam 0). Returns (X at the window's end, failed, its margin parts, its base margins)."""
        m = self.m
        H = int(self.cfg.hold)
        fail = np.zeros(X.shape[0], bool)
        parts = margin_parts(m, X, obs_r, d_r)
        for i in range(P * H):
            if i % H == 0:
                ub = self.backup_action(X, views, obs_r, learned, d_r, combine(m, parts, self.switch_extra)[1] >= 0.0)
                if i == 0 and u_nom is not None:
                    un = np.repeat(u_nom, R, axis=0)
                elif self.nominal is not None:
                    un = self.nominal(X, goal_r, obs_r, d_r)
                else:
                    raise ValueError("u_nom is needed")
                U = np.clip(lam_r[:, None] * un + (1.0 - lam_r[:, None]) * ub, m.low, m.high)
            X = step(m, X, U, d_r)
            parts = margin_parts(m, X, obs_r, d_r)
            s_m, b_m = combine(m, parts, extra)
            fail |= s_m < 0.0
        return X, fail, parts, b_m

    def _rollouts_numba(self, X, P, N, lam_r, u_nom, R, learned, views, obs_r, d_r, extra, goal_r):
        from ps2rl.online.fast import kernel
        m, k = self.m, self._k
        n = X.shape[0]
        X_P = np.empty((n, 10)); b_P = np.empty(n)
        hit = np.zeros(n, bool); fail = np.zeros(n, bool); used = np.zeros(n, np.int64)
        H = int(self.cfg.hold)
        if P == 1 and u_nom is not None:
            un = np.repeat(u_nom, R, axis=0)
            kernel(np.ascontiguousarray(X), True, not self.cfg.early_exit, np.ascontiguousarray(lam_r, dtype=float), un, learned,
                   np.ascontiguousarray(views), np.ascontiguousarray(obs_r), np.ascontiguousarray(d_r), extra,
                   self.switch_extra, N, fail, hit, *k, X_P, b_P, hit, fail, used, H)
            return hit, fail, used, X_P, b_P, H + int(used.max(initial=0))
        Xw, fw, _, bw = self._window_numpy(X, P, lam_r, u_nom, R, learned, views, obs_r, d_r, extra, goal_r)
        hit_in = (bw >= 0.0) & ~fw
        kernel(np.ascontiguousarray(Xw), False, not self.cfg.early_exit, np.ascontiguousarray(lam_r, dtype=float),
               np.zeros((n, 4)), learned,
               np.ascontiguousarray(views), np.ascontiguousarray(obs_r), np.ascontiguousarray(d_r), extra,
               self.switch_extra, N, fw.copy(), hit_in, *k, X_P, b_P, hit, fail, used, H)
        return hit, fail, used, Xw, bw, P * H + int(used.max(initial=0))

    def _rollouts_numpy(self, X, P, N, lam_r, u_nom, R, learned, views, obs_r, d_r, extra, goal_r):
        """Without early_exit (timing of the worst case) every rollout runs N steps and the actor is evaluated at every
        step (the verdicts are unchanged: inside the base set its action is replaced by the hover LQR's)."""
        m, cfg = self.m, self.cfg
        X, fail, parts, b_m = self._window_numpy(X, P, lam_r, u_nom, R, learned, views, obs_r, d_r, extra, goal_r)
        H = int(cfg.hold)
        steps = P * H
        X_P = X.copy()
        b_P = b_m.copy()
        hit = (b_m >= 0.0) & ~fail
        used = np.zeros(X.shape[0], int)                  # backup steps each rollout needed (onboard cost per vehicle)
        # backup: first hit of the base set within N steps
        for j in range(N):
            active = ~(hit | fail)
            if cfg.early_exit and not active.any():
                break
            if j % H == 0:                                 # a command time: the backup's next command
                inb_sw = combine(m, parts, self.switch_extra)[1] >= 0.0
                self._force_actor = not cfg.early_exit
                ub = self.backup_action(X, views, obs_r, learned, d_r, inb_sw)
                self._force_actor = False
                if not cfg.early_exit:
                    ub = np.where(inb_sw[:, None], lqr_action(m, X, None, d_r), ub)
            X = np.where(active[:, None], step(m, X, ub, d_r), X)
            steps += 1
            used += active
            parts = margin_parts(m, X, obs_r, d_r)
            s_m, b_m = combine(m, parts, extra)
            nf = active & (s_m < 0.0)
            fail |= nf
            if (j + 1) % H == 0:                           # the hand-over to the hover LQR at a command time
                hit |= active & ~nf & (b_m >= 0.0)
        return hit, fail, used, X_P, b_P, steps

    def _choose(self, x, obst, goal, d, E, R, C, P, lam, rows_k, fresh_view, cert, ok_f, i_f, ok_c, latched, ok_h, Xh_P,
                X_P, b_P, used, steps, t0) -> Plan:
        m, cfg, com = self.m, self.cfg, self.committed
        # choose
        mode = np.where(ok_f, 0, np.where(latched, np.where(ok_h, 2, 3), np.where(ok_c, 1, 3)))
        use_held = ~ok_f & latched
        row = np.where(ok_f, i_f, C)                      # fresh candidate or the continuation row
        sel = np.arange(E) * R + row
        lam_out = np.where(ok_f, lam[row], 0.0)
        k_out = rows_k[np.arange(E), row]
        # latch: the plan ends its window in the base set -> hold that anchor (held plans keep theirs)
        xP = np.where(use_held[:, None], Xh_P, X_P[sel])
        inb_P = np.where(use_held, True, b_P[sel] >= 0.0)
        new_a0 = np.where(use_held[:, None], com["a0"], np.where(inb_P[:, None], anchor(m, xP, d=d), com["a0"]))
        new_latched = use_held | inb_P
        view_out = np.where(ok_f[:, None, None], fresh_view, com["view"])
        obst_out = np.where(ok_f[:, None, None], obst, com["obst"])
        goal_out = np.where(ok_f[:, None], goal, com["goal"])
        self.committed = {"view": view_out, "obst": obst_out, "goal": goal_out, "k": k_out,
                          "latched": new_latched & cfg.hover_latch, "a0": new_a0}
        stats = {"time": time.perf_counter() - t0, "steps": steps, "rows": E * R,
                 "steps_vehicle": P * int(cfg.hold) + used.reshape(E, R).max(axis=1), "n_cert": cert[:, :C].sum(axis=1)}
        self.plan = Plan(lam=lam_out, k=k_out, held=use_held, a0=com["a0"].copy(), view=view_out, obst=obst_out,
                         goal=goal_out, d=d.copy(), mode=mode, stats=stats)
        return self.plan

    def plan_nominal_inputs(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(goal, known cylinders, push) the nominal must use until the next decision (what the rollout simulated)."""
        return self.plan.goal, self.plan.obst, self.plan.d

    def command(self, x: np.ndarray, u_nom: np.ndarray) -> np.ndarray:
        """The plan's command at the state x (every control step): lam u_nom + (1 - lam) backup, or the held hover
        (with the decision's push estimate, as simulated)."""
        m, p = self.m, self.plan
        learned = self.learned_k[p.k]
        ub = self.backup_action(x, p.view, p.obst, learned, p.d)
        u = np.clip(p.lam[:, None] * u_nom + (1.0 - p.lam[:, None]) * ub, m.low, m.high)
        if p.held.any():
            u = np.where(p.held[:, None], held_action(m, x, p.a0, p.d), u)
        return u


__all__ = ["FilterConfig", "MODES", "OnlineFilter", "Plan"]
