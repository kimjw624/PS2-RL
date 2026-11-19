#!/usr/bin/env python3
"""Publish a trained slow-track policy into ``checkpoints/slow_track/<name>/`` (tracked by git).

Copies what the deployment runtime (``ps2rl/deploy/slow_track_runtime.py``) and a reader need, and nothing else:

    configs.json        the run's training configuration (the runtime rebuilds the command box, observation
                        scaling, observer and reference path from its "env" block)
    best_weights.pkl    the actor only: {"actor_params": {"layers": [{"w", "b"}, ...]}} (numpy arrays); the
                        critics and the temperature are training state and are left out
    summary.json        training summary (best step and score)
    eval/               the robust evaluation of these weights (eval.json and its figures), if present
    policy_card.json    provenance: source run, sha256 of the weights and of the reference, observation and
                        action layout, evaluation score

After writing, it checks that the exported actor is bit-identical to the run's and that the runtime loads the
exported folder. Run it where the run lives (the container), with system python3 or the training venv:

    python3 scripts/export_slow_track_policy.py --run outputs/slow_track/line_v4_s0
    python3 scripts/export_slow_track_policy.py --run outputs/slow_track/v4_floor_s0
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
from pathlib import Path
import pickle
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _to_numpy(tree):
    if isinstance(tree, dict):
        return {k: _to_numpy(v) for k, v in tree.items()}
    if isinstance(tree, (list, tuple)):
        return [_to_numpy(v) for v in tree]
    return np.asarray(tree)


def _same(a, b) -> bool:
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    return np.array_equal(np.asarray(a), np.asarray(b))


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True, help="run directory of scripts/train_slow_track.py")
    p.add_argument("--weights", default="best", choices=("best", "final"))
    p.add_argument("--name", default="", help="folder name under --dest (default: the run's name)")
    p.add_argument("--dest", default=str(ROOT / "checkpoints" / "slow_track"))
    p.add_argument("--note", default="", help="one line stored in policy_card.json")
    a = p.parse_args(argv)

    run = Path(a.run).expanduser()
    if not run.is_absolute():
        run = (Path.cwd() / run) if (Path.cwd() / run).exists() else ROOT / run
    cfg_path, w_path = run / "configs.json", run / f"{a.weights}_weights.pkl"
    for f in (cfg_path, w_path):
        if not f.exists():
            sys.exit(f"missing {f}")
    out = Path(a.dest).expanduser() / (a.name or run.name)
    out.mkdir(parents=True, exist_ok=True)

    rc = json.loads(cfg_path.read_text())
    with open(w_path, "rb") as f:
        actor = _to_numpy(pickle.load(f)["actor_params"])
    with open(out / "best_weights.pkl", "wb") as f:      # the runtime's default weights name
        pickle.dump({"actor_params": actor}, f, protocol=4)
    shutil.copy2(cfg_path, out / "configs.json")
    if (run / "summary.json").exists():
        shutil.copy2(run / "summary.json", out / "summary.json")
    ev_dir = run / f"eval_{a.weights}"
    ev = None
    if ev_dir.is_dir():
        (out / "eval").mkdir(exist_ok=True)
        for f in ev_dir.iterdir():
            if f.suffix in (".json", ".png"):
                shutil.copy2(f, out / "eval" / f.name)
        if (ev_dir / "eval.json").exists():
            ev = json.loads((ev_dir / "eval.json").read_text())

    env = rc["env"]
    ref = Path(env.get("reference_path", ""))
    ref_abs = ref if ref.is_absolute() else ROOT / ref
    layers = actor["layers"]
    card = {
        "name": out.name,
        "source_run": str(run),
        "source_weights": w_path.name,
        "exported": _dt.datetime.now().isoformat(timespec="seconds"),
        "note": a.note,
        "weights_sha256": _sha256(out / "best_weights.pkl"),
        "reference_path": str(ref),
        "reference_sha256": _sha256(ref_abs) if ref_abs.exists() else None,
        "network": {"activation": rc["actor"].get("activation", "relu"),
                    "layer_shapes": [list(np.shape(l["w"])) for l in layers],
                    "output": "first half of the last layer = mean of the tanh-Gaussian; action = mid + half * tanh(mean)"},
        "observation_dim": int(np.shape(layers[0]["w"])[0]),
        "action": {"order": ["collective thrust / mass [m/s^2]", "p [rad/s]", "q [rad/s]", "r [rad/s]"],
                   "low": [env.get("a_min"), -env.get("omega_xy_max"), -env.get("omega_xy_max"), -env.get("omega_z_max")],
                   "high": [env.get("a_max"), env.get("omega_xy_max"), env.get("omega_xy_max"), env.get("omega_z_max")]},
        "control_period_s": env.get("dt"),
        "observer_lambda": env.get("observer_lambda"),
        "obs_scales": {"position": env.get("obs_pos_scale"), "velocity": env.get("obs_vel_scale")},
        "eval_score": None if ev is None else ev.get("score"),
        "eval_pos_err_mean_cm": None if ev is None else {k: round(100 * v["pos_err_mean"], 2) for k, v in ev["scenarios"].items()},
        "eval_lost_rate": None if ev is None else {k: v["lost_rate"] for k, v in ev["scenarios"].items()},
    }
    (out / "policy_card.json").write_text(json.dumps(card, indent=2))

    # checks: identical actor, and the runtime loads the folder and gives the same action as the run
    with open(out / "best_weights.pkl", "rb") as f:
        again = pickle.load(f)["actor_params"]
    if not _same(actor, again):
        sys.exit("exported actor differs from the run's")
    from ps2rl.deploy.slow_track_runtime import SlowTrackRuntime, SlowTrackRuntimeConfig
    quiet = type("Q", (), {"info": staticmethod(lambda m: None), "warn": staticmethod(lambda m: None)})()
    r_out = SlowTrackRuntime(SlowTrackRuntimeConfig(controller="policy", run=str(out)), ps2rl_path=str(ROOT), logger=quiet)
    r_run = SlowTrackRuntime(SlowTrackRuntimeConfig(controller="policy", run=str(run), weights=a.weights),
                             ps2rl_path=str(ROOT), logger=quiet)
    obs = np.random.default_rng(0).normal(size=(200, card["observation_dim"]))
    diff = max(float(np.abs(r_out.actor(o) - r_run.actor(o)).max()) for o in obs)
    if diff != 0.0:
        sys.exit(f"runtime actions differ by {diff}")
    size_kb = (out / "best_weights.pkl").stat().st_size / 1024
    print(f"exported {run.name} ({a.weights}) -> {out}  [actor {card['network']['layer_shapes']}, {size_kb:.0f} kB, "
          f"eval score {card['eval_score']}]; runtime check passed (200 observations, identical actions)")


if __name__ == "__main__":
    main()
