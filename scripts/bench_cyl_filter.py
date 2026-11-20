#!/usr/bin/env python3
"""Time the online filter's decision on this computer (run it on the Raspberry Pi 4B). numpy only.

    python3 scripts/bench_cyl_filter.py                         # default grid, about 3 min on a laptop core
    python3 scripts/bench_cyl_filter.py --quick                 # a smaller grid, under a minute
    python3 scripts/bench_cyl_filter.py --out bench_pi4.json    # also save the table (send it back)

What it measures, for one vehicle (E = 1) and every setting of the grid:
  worst     the decision with every rollout running its full length (period + N steps): the time to budget for
  near      flying at 1 m/s toward a cylinder 1.6 m ahead (the filter at work; rollouts stop when resolved)
  cruise    flying at 1 m/s with two cylinders 6 m ahead (the common case)
  command   one control step between decisions (nominal + the plan's command)
and fits, per backend, worst = a + S (b + c R + e R F) + w (period - 1), S = period + N rollout steps, R rows,
F the actor's kFLOP per evaluation, w the window step (numpy, nominal simulated), so settings outside the grid can be
predicted (scripts/report_cyl_online.py).

The timing does not depend on the weights, so actors of each width get random weights (no trained backup needed); the
model constants (hover LQR, base set) come from ``--model`` (an exported backup directory) or the defaults of
checkpoints/cyl_online/lqr. Single-threaded BLAS (set below before numpy loads), as on the vehicle.
"""

from __future__ import annotations

import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import itertools  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
from pathlib import Path  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from ps2rl.online.filter import FilterConfig, OnlineFilter  # noqa: E402
from ps2rl.online.model import load_backup, random_actor  # noqa: E402
from ps2rl.online.nominal import Nominal, NominalConfig  # noqa: E402

LAMBDA_SETS = {2: (1.0, 0.0), 3: (1.0, 0.5, 0.0), 5: (1.0, 0.75, 0.5, 0.25, 0.0)}


def scenario(kind: str):
    x = np.zeros((1, 10)); x[0, 2] = 1.5; x[0, 3] = 1.0; x[0, 6] = 1.0
    obst = np.zeros((1, 4, 4))
    if kind == "cruise":
        obst[0, 0] = [6.0, 1.0, 0.3, 1.0]; obst[0, 1] = [6.0, -1.5, 0.3, 1.0]
    else:
        obst[0, 0] = [1.75, 0.05, 0.3, 1.0]; obst[0, 1] = [4.0, 2.0, 0.4, 1.0]
    goal = np.array([[14.0, 0.0, 1.5]])
    return x, obst, goal


def time_decide(m, actor, fc: FilterConfig, kind: str, reps: int) -> tuple[np.ndarray, int]:
    """Decision times at E = 1 (``reps`` repeats after a warm-up) and the rollout steps of the last one."""
    nom = Nominal(m, NominalConfig())
    if kind == "worst":
        fc = FilterConfig(**{**fc.as_dict(), "early_exit": False})
    f = OnlineFilter(m, actor, fc, nom)
    x, obst, goal = scenario("cruise" if kind == "worst" else kind)
    d = np.zeros((1, 3))
    f.reset(x, obst, goal)
    u_n = nom(x, goal, obst, d)
    for _ in range(3):                                   # warm-up
        f.decide(x, obst, goal, u_n, d)
    ts, steps = [], 0
    for _ in range(reps):
        t0 = time.perf_counter()
        p = f.decide(x, obst, goal, u_n, d)
        ts.append(time.perf_counter() - t0)
        steps = int(p.stats["steps"])
    return np.asarray(ts), steps


def time_command(m, actor, fc: FilterConfig, reps: int) -> np.ndarray:
    nom = Nominal(m, NominalConfig())
    f = OnlineFilter(m, actor, fc, nom)
    x, obst, goal = scenario("near")
    d = np.zeros((1, 3))
    f.reset(x, obst, goal)
    f.decide(x, obst, goal, nom(x, goal, obst, d), d)
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        g, o, dd = f.plan_nominal_inputs()
        f.command(x, nom(x, g, o, dd))
        ts.append(time.perf_counter() - t0)
    return np.asarray(ts)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=str(ROOT / "checkpoints/cyl_online/lqr"), help="exported backup (model constants)")
    p.add_argument("--widths", default="0,32,64,128,256", help="actor widths (0: hover-LQR backup only)")
    p.add_argument("--lambdas", default="2,3,5", help="number of blends (sets: 2 = 1,0; 3 = 1,.5,0; 5 = 1,.75,.5,.25,0)")
    p.add_argument("--families", default="1,2", help="backup family sizes: 1 = learned (or LQR at width 0), 2 = learned + LQR")
    p.add_argument("--horizons", default="0.5,1.0", help="backup rollout lengths [s]")
    p.add_argument("--periods", default="1,5", help="control steps per decision")
    p.add_argument("--backends", default="numpy,numba", help="numpy and / or numba (skipped if numba is missing)")
    p.add_argument("--reps", type=int, default=20)
    p.add_argument("--quick", action="store_true", help="widths 0,64; lambdas 3; families 1; horizons 1.0; periods 1")
    p.add_argument("--out", default="")
    a = p.parse_args(argv)
    if a.quick:
        a.widths, a.lambdas, a.families, a.horizons, a.periods = "0,64", "3", "1", "1.0", "1"
    from ps2rl.online.fast import HAVE_NUMBA
    backends = [b for b in a.backends.split(",") if b == "numpy" or (b == "numba" and HAVE_NUMBA)]
    if "numba" in a.backends and not HAVE_NUMBA:
        print("[bench] numba is not installed: numpy only (pip install numba for the compiled rollouts)")
    m, _, _ = load_backup(a.model)
    ints = lambda s: [int(v) for v in s.split(",") if v]
    flts = lambda s: [float(v) for v in s.split(",") if v]
    info = {"machine": platform.machine(), "processor": platform.processor(), "python": platform.python_version(),
            "numpy": np.__version__, "node": platform.node()}
    try:
        lines = open("/proc/cpuinfo").read().splitlines()
        for key in ("model name", "hardware", "model"):
            hit = [l.split(":", 1)[1].strip() for l in lines if l.lower().split(":")[0].strip() == key]
            if hit:
                info["cpu_model"] = hit[0]
                break
    except Exception:
        pass
    print("[bench] " + " ".join(f"{k}={v}" for k, v in info.items()), flush=True)
    rows = []
    hdr = (f"{'backend':>7} {'width':>5} {'blends':>6} {'fam':>3} {'rows':>4} {'T[s]':>5} {'P':>2} | {'worst ms':>9} {'steps':>5} "
           f"{'ms/step':>7} | {'near ms':>8} {'steps':>5} | {'cruise ms':>9} {'steps':>5} | {'cmd ms':>6}")
    print(hdr); print("-" * len(hdr), flush=True)
    for be, w, nl, fam, T, P in itertools.product(backends, ints(a.widths), ints(a.lambdas), ints(a.families),
                                                  flts(a.horizons), ints(a.periods)):
        if w == 0 and fam == 2:
            continue
        actor = None if w == 0 else random_actor(m, w, seed=w)
        backups = ("lqr",) if w == 0 else (("learned",) if fam == 1 else ("learned", "lqr"))
        fc = FilterConfig(lambdas=LAMBDA_SETS[nl], backups=backups, horizon=T, period=P, backend=be)
        if be == "numba":
            t0 = time.perf_counter()
            time_decide(m, actor, fc, "near", 1)          # compile (first use; numba caches it on disk)
            info.setdefault("numba_first_call_s", time.perf_counter() - t0)
        res = {"backend": be, "width": w, "blends": nl, "family": len(backups), "rows": fc.n_candidates + 1, "horizon": T, "period": P}
        for kind in ("worst", "near", "cruise"):
            ts, steps = time_decide(m, actor, fc, kind, a.reps)
            res[f"{kind}_ms_median"] = 1e3 * float(np.median(ts))
            res[f"{kind}_ms_p95"] = 1e3 * float(np.percentile(ts, 95))
            res[f"{kind}_steps"] = steps
        tc = time_command(m, actor, fc, max(a.reps, 50))
        res["command_ms_median"] = 1e3 * float(np.median(tc))
        res["worst_ms_per_step"] = res["worst_ms_median"] / max(res["worst_steps"], 1)
        rows.append(res)
        print(f"{be:>7} {w:>5} {nl:>6} {len(backups):>3} {res['rows']:>4} {T:>5.2f} {P:>2} | {res['worst_ms_median']:>9.2f} "
              f"{res['worst_steps']:>5} {res['worst_ms_per_step']:>7.3f} | {res['near_ms_median']:>8.2f} {res['near_steps']:>5} | "
              f"{res['cruise_ms_median']:>9.2f} {res['cruise_steps']:>5} | {res['command_ms_median']:>6.2f}", flush=True)
    # fit worst = a + steps * (b + c * rows * flops_per_row), per backend
    for be in backends:
        sub = [r for r in rows if r["backend"] == be]
        if len(sub) < 4:
            continue
        A, y = [], []
        for r in sub:
            fl = 0.0 if r["width"] == 0 else (m.obs_dim * r["width"] + r["width"] ** 2 + 4 * r["width"]) * 2e-3
            S = r["period"] + round(r["horizon"] / m.dt)
            A.append([1.0, S, S * r["rows"], S * r["rows"] * fl, r["period"] - 1])
            y.append(r["worst_ms_median"])
        coef, *_ = np.linalg.lstsq(np.asarray(A), np.asarray(y), rcond=None)
        info[f"fit_worst_ms_{be}"] = {"a": coef[0], "per_step": coef[1], "per_step_row": coef[2], "per_step_row_kflop": coef[3],
                                      "per_window_step": coef[4]}
        print(f"[fit {be}] worst ms = {coef[0]:.3f} + S x ({coef[1]:.4f} + {coef[2]:.5f} R + {coef[3]:.6f} R x kFLOP) + "
              f"{coef[4]:.3f} (period - 1)")
    if a.out:
        Path(a.out).write_text(json.dumps({"info": info, "rows": rows}, indent=2, default=float))
        print(f"[bench] wrote {a.out}")


if __name__ == "__main__":
    main()
