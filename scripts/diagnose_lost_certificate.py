"""Why the certificate is lost: at the steps without a certificate of a replay, check every backup variant's own plan.

    python scripts/diagnose_lost_certificate.py REPLAY_DIR BACKUP_RUN [--stride 5] [--extra_variants JSON] [--guidance JSON]

REPLAY_DIR: eval_strike_filter.py --save_traj 1 --only_episodes ... output. Approximate: the plans are checked from the
recorded state with a fresh view and no hover latch (the filter's own committed state is not recorded).

For each onset step t (no certificate at t, certificate at t-1) and the step lead before it:
- geometry: h_c, chaser range, obstacle and wall clearance, speed, chasers within 6 m;
- per variant (blend 0, fresh view, no hover latch): path ok with the fresh slack (h_c >= 0.2, obstacles/walls >= 0.15) and
  with zero slack (the safe set), terminal ok, and which margin is the worst along the path (obstacle, wall, h_c);
- the same for extra escape variants (--extra_variants, e.g. [[3.5, 90], [3.5, -90]]) when given.
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run", help="an eval_strike_filter.py --save_traj 1 output folder (strike_filter_traj.npz)")
    p.add_argument("--task", default=str(REPO / "docker_batch/configs/strike_final_ps2.json"),
                   help="task file whose tracker.filter_guidance is the filter under test")
    p.add_argument("backup_run", help="the Phase-I backup run the replay used")
    p.add_argument("--guidance", default="")
    p.add_argument("--extra_variants", default="")
    p.add_argument("--lead", type=int, default=0, help="also check this many steps before the onset")
    p.add_argument("--max_onsets", type=int, default=3, help="per episode")
    p.add_argument("--stride", type=int, default=0, help="> 0: check every stride-th step without a certificate instead of the onsets")
    a = p.parse_args()
    import jax
    import jax.numpy as jnp
    import strike_episode_viz as viz
    from ps2rl.cil.strike_filter import backup_ranges, load_backup, make_filter, model_from_backup
    from ps2rl.cil.strike_guidance import GuidanceConfig, h_chasers, wall_distances
    from ps2rl.cil.strike_filter import full_margins
    task = json.load(open(a.task))
    g0 = json.loads(task["tracker"]["filter_guidance"])
    g0.update(json.loads(a.guidance) if a.guidance else {})
    m = model_from_backup(a.backup_run)
    cfg = m.cfg
    o_rng, b_rng = backup_ranges(a.backup_run)
    bk, prm = load_backup(m, "learned", a.backup_run)
    from ps2rl.envs.strike_env import StrikeEnvConfig
    ec = StrikeEnvConfig.from_dict(task.get("env", {}))
    arena = (0.0, ec.arena_x, ec.arena_y_half, ec.z_max)

    def build(gd):
        gc = GuidanceConfig.from_dict(gd)
        f = make_filter(m, bk, prm, obstacle_range=o_rng, ball_range=b_rng, hover_latch=True, chaser_model="guidance",
                        guidance=gc, arena=arena)
        return f, gc

    f, gc = build(g0)
    filters = [("base", f, len(f.variant_labels))]
    if a.extra_variants:
        ge = dict(g0)
        ge["escape_variants"] = [list(v) for v in g0.get("escape_variants", [])] + json.loads(a.extra_variants)
        fe, _ = build(ge)
        filters.append(("extra", fe, len(fe.variant_labels)))
    no_balls = jnp.zeros((cfg.n_chaser_slots, 4), jnp.float32)

    @jax.jit
    def parts(xs, chs, on, obst):
        so = jax.vmap(lambda x: jnp.minimum(full_margins(m, x, obst, no_balls)[0], x[2] - cfg.z_floor))(xs)
        wall = jax.vmap(lambda x: jnp.min(wall_distances(arena, x[0:2])) - cfg.r_drone)(xs)
        hc = jax.vmap(lambda x, c: jnp.min(h_chasers(m, gc, x[0:3], x[3:6], c, on)))(xs, chs)
        return so, wall, hc

    jits = {}

    def check(fl, nk, x, obst, chp, on):
        if id(fl) not in jits:
            jits[id(fl)] = (jax.jit(fl.candidate, static_argnums=(6,)), jax.jit(fl.fresh_view))
        cand, fv = jits[id(fl)]
        comm = fv(x, obst, chp, on)
        out = []
        for k in range(nk):
            r = {}
            for hm in (gc.fresh_margin, 0.0):
                c = cand(x, jnp.zeros(4), obst, chp, on, comm, k, jnp.float32(0.0), jnp.float32(hm))
                r[hm] = (bool(c["path_ok"]), bool(c["terminal_ok"]))
            so, wall, hc = parts(c["states"], c["chasers"], on, obst)
            worst = {"obstacle": float(jnp.min(so)), "wall": float(jnp.min(wall)), "h_c": float(jnp.min(hc))}
            out.append((r, worst))
        return out

    q = viz.Panel(f"{a.run}:learned:x:x", "#000")
    tally = Counter()
    for e in sorted(q.idx):
        j, k_end = q.ep(e)
        nc = q.nocert[:k_end, j] > 0.5
        onsets = [t for t in range(1, k_end) if nc[t] and not nc[t - 1]][: a.max_onsets]
        if a.stride > 0:
            onsets = [t for t in range(0, k_end) if nc[t]][:: a.stride]
        if not onsets:
            continue
        print(f"== episode {e}: {q.code(e)} at {k_end * q.dt:.1f} s, {int(nc.sum())} steps without certificate, onsets "
              f"{[round(t * q.dt, 2) for t in onsets]} s")
        obst = jnp.asarray(q.obst[j])
        on = jnp.asarray(q.on[j])
        for t0 in onsets:
            for t in ([t0 - a.lead] if a.lead else []) + [t0]:
                x = jnp.asarray(np.concatenate([q.pos[t, j], q.vel[t, j], q.quat[t, j]]), jnp.float32)
                chp = jnp.asarray(q.ch[t, j])
                hc_now = np.asarray(h_chasers(m, gc, x[0:3], x[3:6], chp, on))
                rng = np.linalg.norm(q.ch[t, j] - q.pos[t, j][None, :], axis=1) - (cfg.chaser_radius + cfg.r_drone)
                near = int(np.sum((rng < 6.0) & (q.on[j] > 0.5)))
                so0 = float(full_margins(m, x, obst, no_balls)[0])
                w0 = float(jnp.min(wall_distances(arena, x[0:2])) - cfg.r_drone)
                geo = (float(np.min(rng[q.on[j] > 0.5])), so0, w0, near)
                tally[("geo", "range<1")] += geo[0] < 1.0
                tally[("geo", "obst<1")] += so0 < 1.0
                tally[("geo", "wall<1.5")] += w0 < 1.5
                tally[("geo", "2+ chasers<6m")] += near >= 2
                print(f"  t {t * q.dt:5.2f} s{' (onset)' if t == t0 else ' (before)'}: h_c {np.min(hc_now[q.on[j] > 0.5]):.2f}, "
                      f"range {np.min(rng[q.on[j] > 0.5]):.2f} m, chasers within 6 m {near}, obstacle {so0:.2f} m, wall {w0:.2f} m, "
                      f"z {q.pos[t, j][2]:.2f}, speed {np.linalg.norm(q.vel[t, j]):.2f}")
                for name, fl, nk in filters:
                    res = check(fl, nk, x, obst, chp, on)
                    any_fresh = any(r[gc.fresh_margin][0] and r[gc.fresh_margin][1] for r, _ in res)
                    any_zero = any(r[0.0][0] and r[0.0][1] for r, _ in res)
                    causes = Counter()
                    for r, w in res:
                        if not (r[0.0][0] and r[0.0][1]):
                            causes[("path " + min(w, key=w.get)) if not r[0.0][0] else "terminal"] += 1
                    print(f"    {name}: any variant certified with slack {any_fresh}, with zero slack {any_zero}; "
                          f"zero-slack failures {dict(causes)}")
                    if t == t0:
                        for kk, (r, _) in enumerate(res):
                            if r[gc.fresh_margin][0] and r[gc.fresh_margin][1]:
                                tally[(name, "variant ok", fl.variant_labels[kk])] += 1
                        only = [kk for kk, (r, _) in enumerate(res) if r[gc.fresh_margin][0] and r[gc.fresh_margin][1]]
                        if len(only) == 1:
                            tally[(name, "only variant", fl.variant_labels[only[0]])] += 1
                        tally[(name, "slack")] += any_fresh
                        tally[(name, "zero")] += any_zero
                        tally[(name, "n")] += 1
                        for c, n in causes.items():
                            tally[(name, c)] += n
    print("\nonsets:")
    for kk, vv in sorted(tally.items(), key=lambda t: str(t[0])):
        print("  ", kk, vv)


if __name__ == "__main__":
    main()
