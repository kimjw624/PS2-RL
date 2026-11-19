#!/usr/bin/env python
"""Evaluate a slow-track run on the robustness scenarios and draw the figures to check before SITL.

    python scripts/eval_slow_track.py --run outputs/slow_track/slow_loop_s0 [--weights best] [--episodes 32]

Writes ``<run>/eval_<weights>/``: ``eval.json`` (per scenario: lost rate, mean / p95 / max position error,
error after the first 5 s, final error, max tilt, command ranges, command smoothness, observer error),
``top_view.png`` (reference and every flown path, one panel per scenario), ``errors.png`` (position error
over time) and ``commands_nominal.png`` (thrust and body-rate commands in the nominal scenario, with the
action box). Every episode starts at the hand-over and flies the whole reference + 2 s.

What to look for before flying: lost rate 0 everywhere; mean error of a few cm in ``nominal`` and
``slow_actuators``; commands far from the box edges (a policy that sits on the box will saturate the real
vehicle too); no high-frequency chatter in the rate commands.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from ps2rl.envs.slow_track_env import SlowTrackEnvConfig, build_slow_track_env  # noqa: E402
from ps2rl.phase2_ps2.slow_track_trainer import (EVAL_SCENARIOS, make_eval_fn, robust_score,  # noqa: E402
                                                 summarize_scenario)
from ps2rl.utils.policy import ActorConfig  # noqa: E402


def load_run(run: Path, weights: str):
    cfg = json.loads((run / "configs.json").read_text())
    env_cfg = SlowTrackEnvConfig(**cfg["env"])
    a = dict(cfg["actor"])
    a["hidden_sizes"] = tuple(a["hidden_sizes"])
    actor_cfg = ActorConfig(**a)
    with open(run / f"{weights}_weights.pkl", "rb") as f:
        params = pickle.load(f)["actor_params"]
    return cfg, env_cfg, actor_cfg, params


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True)
    p.add_argument("--weights", default="best", choices=("best", "final"))
    p.add_argument("--episodes", type=int, default=32)
    p.add_argument("--seed", type=int, default=2024)
    a = p.parse_args(argv)
    run = Path(a.run)
    cfg, env_cfg, actor_cfg, params = load_run(run, a.weights)
    env = build_slow_track_env(env_cfg)
    low = jnp.asarray(env_cfg.policy_low, jnp.float32)
    high = jnp.asarray(env_cfg.policy_high, jnp.float32)
    scale = 0.5 * (high - low)
    fns, n_steps = make_eval_fn(env, env_cfg, actor_cfg, low, high, scale, a.episodes)
    out = run / f"eval_{a.weights}"
    out.mkdir(exist_ok=True)
    half = 0.5 * (np.asarray(env_cfg.action_high) - np.asarray(env_cfg.action_low))
    rest_start = int(round(5.0 / env_cfg.dt))
    trs, ev = {}, {}
    for i, (name, fn) in enumerate(fns.items()):
        tr = jax.device_get(fn(params, jax.random.PRNGKey(a.seed + i)))
        trs[name] = tr
        ev[name] = summarize_scenario(tr, half, rest_start)
        m = ev[name]
        print(f"{name:15s} lost {m['lost_rate']:.2f}  pos err mean {m['pos_err_mean']:.3f} p95 {m['pos_err_p95']:.3f} "
              f"max {m['pos_err_max']:.3f}  after 5 s {m['pos_err_after_5s']:.3f}  tilt max {m['tilt_max_deg']:.1f} deg  "
              f"thrust {m['thrust_min']:.2f}..{m['thrust_max']:.2f}  |rate| xy {m['rate_xy_max']:.2f} z {m['rate_z_max']:.2f}  "
              f"du {m['du_mean']:.3f}", flush=True)
    score = robust_score(ev)
    (out / "eval.json").write_text(json.dumps({"run": str(run), "weights": a.weights, "episodes": a.episodes,
                                               "score": score, "scenarios": ev, "definitions": EVAL_SCENARIOS},
                                              indent=2, default=float))
    print(f"robust score {score:.4f} -> {out / 'eval.json'}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ref = np.asarray(env.ref_states)
    names = list(trs)
    nc = 4
    nr = int(np.ceil(len(names) / nc))
    fig, axs = plt.subplots(nr, nc, figsize=(4.2 * nc, 3.9 * nr), squeeze=False)
    for ax, name in zip(axs.flat, names):
        tr = trs[name]
        ax.plot(ref[:, 0], ref[:, 1], "k--", lw=1.2, label="reference")
        for e in range(tr["x"].shape[0]):
            al = tr["alive"][e].astype(bool)
            c = "tab:red" if tr["lost"][e].any() else "tab:blue"
            ax.plot(tr["x"][e, al, 0], tr["x"][e, al, 1], c=c, lw=0.6, alpha=0.5)
        m = ev[name]
        ax.set_title(f"{name}: {100 * m['pos_err_mean']:.1f} cm mean, lost {m['lost_rate']:.2f}", fontsize=9)
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
    for ax in axs.flat[len(names):]:
        ax.axis("off")
    fig.suptitle(f"{run.name} ({a.weights}): top view, task frame [m]; red = lost")
    fig.tight_layout()
    fig.savefig(out / "top_view.png", dpi=110)
    plt.close(fig)

    t = np.arange(n_steps) * env_cfg.dt
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for name in names:
        tr = trs[name]
        pe = np.where(tr["alive"].astype(bool), tr["pos_err"], np.nan)
        ax.plot(t, np.nanmean(pe, axis=0), lw=1.2, label=name)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("mean position error [m]")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=4)
    fig.tight_layout()
    fig.savefig(out / "errors.png", dpi=110)
    plt.close(fig)

    tr = trs["nominal"]
    fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    lo, hi = np.asarray(env_cfg.action_low), np.asarray(env_cfg.action_high)
    axs[0].plot(t, tr["u"][0, :, 0], lw=0.8)
    axs[0].axhline(lo[0], c="r", ls=":")
    axs[0].axhline(hi[0], c="r", ls=":")
    axs[0].axhline(env_cfg.gravity, c="gray", ls=":")
    axs[0].set_ylabel("thrust cmd [m/s^2]")
    for i, nm in enumerate(("p", "q", "r")):
        axs[1].plot(t, tr["u"][0, :, 1 + i], lw=0.8, label=nm)
    axs[1].axhline(hi[1], c="r", ls=":")
    axs[1].axhline(lo[1], c="r", ls=":")
    axs[1].set_ylabel("rate cmd [rad/s]")
    axs[1].set_xlabel("time [s]")
    axs[1].legend()
    for ax in axs:
        ax.grid(alpha=0.3)
    fig.suptitle("nominal scenario, episode 0: commands (red = action box)")
    fig.tight_layout()
    fig.savefig(out / "commands_nominal.png", dpi=110)
    plt.close(fig)
    print(f"figures in {out}")


if __name__ == "__main__":
    main()
