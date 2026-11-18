#!/usr/bin/env python
"""Pick the best landing tracker among several runs (seeds) and checkpoints by measured tracking error.

Tracker training (SAC without the safety layer) is seed-fragile: some runs collapse mid-training and
some learn slowly. This rolls out every candidate checkpoint (``best_weights.pkl`` and, if present,
``final_weights.pkl`` of every run) alone from the same starts (the reference start + the reset
library) and keeps the one with the smallest mean position error over the reference. The winner is
copied to ``--out`` as a normal run directory (``best_weights.pkl``, configs, reset library), so
the PS2 stage and the evaluator can use it unchanged; ``selection.json`` records every candidate.

    python scripts/select_landing_tracker.py --candidates outputs/landing_phase2/b2_m20_tracker_s0 \\
        outputs/landing_phase2/b2_m20_tracker_s1 --out outputs/landing_phase2/b2_m20_tracker
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--candidates", nargs="+", required=True, help="tracker run directories")
    p.add_argument("--out", required=True, help="directory of the selected tracker")
    p.add_argument("--n_starts", type=int, default=64)
    p.add_argument("--weights", default="best,final", help="checkpoints to consider per run")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    import jax.numpy as jnp

    from ps2rl.cil import quadrotor_landing_backup_cbf as L
    from ps2rl.envs.quadrotor_env import build_quadrotor_env
    from ps2rl.evaluation.landing_phase2_policy_eval import evaluation_starts, load_landing_run, make_mean_policy, rollout_policy

    cands = []
    for d in args.candidates:
        d = Path(d)
        if not (d / "configs.json").exists():
            print(f"[select] skip {d} (no configs.json)")
            continue
        for w in [t.strip() for t in args.weights.split(",") if t.strip()]:
            if (d / f"{w}_weights.pkl").exists():
                cands.append((d, w))
    if not cands:
        raise SystemExit("no candidate checkpoints found")

    first = load_landing_run(cands[0][0], cands[0][1])
    meta = first["meta"]
    filt = {k: (tuple(v) if isinstance(v, list) else v) for k, v in (meta.get("filter") or {}).items()}
    cfg = L.landing_bcbf_config_from_checkpoint(meta["backup"] if Path(meta["backup"]).exists() else
                                                str(REPO / "checkpoints" / "landing_phase1" / Path(meta["backup"]).name), **filt)
    rec_fn = L.make_recoverability_fn(cfg)
    env_cfg = first["env_cfg"]
    env = build_quadrotor_env(env_cfg)
    ref_x = np.asarray(np.load(env_cfg.reference_path)["states"], np.float64)
    lib = np.load(env_cfg.init_state_library)["states"]
    x0, _ = evaluation_starts(ref_x, lib, args.n_starts, cfg, rec_fn)
    steps = int(env_cfg.max_steps)
    k_ref = np.minimum(np.arange(steps + 1), len(ref_x) - 1)

    table = []
    for d, w in cands:
        run = load_landing_run(d, w)
        if Path(run["env_cfg"].reference_path).name != Path(env_cfg.reference_path).name:
            raise SystemExit(f"{d} tracks another reference ({run['env_cfg'].reference_path})")
        res = rollout_policy(make_mean_policy(run, cfg, env.obs_dim), env.observe, cfg, None, x0, steps)
        err = np.linalg.norm(res["traj"][:, :, 0:3] - ref_x[k_ref][None, :, 0:3], axis=-1)[:, : len(ref_x)]
        row = {"run": str(d), "weights": w, "tracking_error_mean": float(err.mean()),
               "tracking_error_p95_over_starts": float(np.percentile(err.mean(1), 95)),
               "tracking_error_max": float(err.max()), "final_error_mean": float(err[:, -1].mean()),
               "leaves_cone_rate": float((np.asarray(L.cone_value(jnp.asarray(res["traj"]), cfg)).min(1) < 0).mean())}
        table.append(row)
        print(f"[select] {d.name:28s} {w:5s}: tracking error mean {row['tracking_error_mean']:.3f} m, worst-start p95 "
              f"{row['tracking_error_p95_over_starts']:.3f} m, max {row['tracking_error_max']:.2f} m, at touchdown "
              f"{row['final_error_mean']:.3f} m, follows the reference out of the cone on {100 * row['leaves_cone_rate']:.0f}% of starts",
              flush=True)

    best = min(range(len(table)), key=lambda i: (table[i]["tracking_error_mean"], table[i]["tracking_error_max"]))
    d, w = cands[best]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name in ("configs.json", "landing_meta.json", "init_states.npz", "summary.json", "history.npz", "metrics.jsonl"):
        if (d / name).exists() and (d / name).resolve() != (out / name).resolve():
            shutil.copy2(d / name, out / name)
    shutil.copy2(d / f"{w}_weights.pkl", out / "best_weights.pkl")
    (out / "selection.json").write_text(json.dumps({"selected": table[best], "candidates": table, "starts": int(len(x0))}, indent=2))
    print(f"[select] -> {out}: {d.name} ({w} weights), tracking error {table[best]['tracking_error_mean']:.3f} m")


if __name__ == "__main__":
    main()
