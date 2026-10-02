"""Shared helpers to load and roll out Phase-II landing policies (runs of ``train_phase2_landing.py``)."""

from __future__ import annotations

import json
from pathlib import Path
import pickle
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from ps2rl.cil import quadrotor_landing_backup_cbf as lbcbf
from ps2rl.envs.quadrotor_env import QuadrotorEnvConfig
from ps2rl.evaluation.landing_phase2_run import resolve_repo_path
from ps2rl.utils.policy import ActorConfig, actor_mean_action

REPO = Path(__file__).resolve().parents[2]


def load_landing_run(run_dir: Path, weights: str = "best") -> dict[str, Any]:
    """Configs, meta and actor parameters of a run (``weights``: "best" or "final")."""
    run_dir = Path(run_dir)
    cfg = json.loads((run_dir / "configs.json").read_text())
    meta = cfg.get("landing_meta") or json.loads((run_dir / "landing_meta.json").read_text())
    env = dict(cfg["env"])
    env["reference_path"] = str(resolve_repo_path(env["reference_path"], repo_root=REPO, run_dir=run_dir))
    lib = env.get("init_state_library", "")
    if lib:
        local = run_dir / "init_states.npz"
        try:
            env["init_state_library"] = str(local if local.exists() else resolve_repo_path(lib, repo_root=REPO, run_dir=run_dir))
        except FileNotFoundError:
            env["init_state_library"] = str(local)
    env_cfg = QuadrotorEnvConfig(**{k: v for k, v in env.items() if k in QuadrotorEnvConfig.__dataclass_fields__})
    with open(run_dir / f"{weights}_weights.pkl", "rb") as fh:
        payload = pickle.load(fh)
    return {"dir": run_dir, "env_cfg": env_cfg, "meta": meta, "hidden": int(cfg["sac"]["hidden_size"]),
            "params": payload["actor_params"], "weights": weights,
            "summary": json.loads((run_dir / "summary.json").read_text()) if (run_dir / "summary.json").exists() else {}}


def make_mean_policy(run: dict[str, Any], cbf_cfg, obs_dim: int):
    """obs (B, obs_dim) -> deterministic (mean) action (B, 4), as the trainer's evaluation uses."""
    actor_cfg = ActorConfig(obs_dim=obs_dim, action_dim=4, hidden_sizes=(run["hidden"], run["hidden"]))
    scale = jnp.asarray(cbf_cfg.action_scale, dtype=jnp.float32)
    lo, hi = jnp.asarray(cbf_cfg.action_low, dtype=jnp.float32), jnp.asarray(cbf_cfg.action_high, dtype=jnp.float32)
    params = jax.tree_util.tree_map(lambda a: jnp.asarray(a, dtype=jnp.float32), run["params"])
    return jax.jit(lambda obs: actor_mean_action(params, obs, scale, actor_cfg, action_low=lo, action_high=hi))


def rollout_policy(policy, observe, cfg, projector, x0: np.ndarray, steps: int) -> dict[str, np.ndarray]:
    """Closed loop on the landing model; ``projector=None`` runs the policy alone."""
    step_b = jax.jit(jax.vmap(lambda x, u: lbcbf.landing_step(x, u, cfg)))
    x = jnp.asarray(x0, dtype=jnp.float32)
    n = x.shape[0]
    xs, us, raws, slacks, used, lams = [x], [], [], [], [], []
    for k in range(steps):
        obs = observe(x, jnp.full((n,), k, dtype=jnp.int32))
        raw = policy(obs)
        if projector is None:
            u, s, ok, lam = raw, jnp.zeros((n,)), jnp.ones((n,), dtype=bool), jnp.ones((n,))
        else:
            u, s, ok, info = projector.solve_batch_with_info(x, raw)
            lam = info.get("safeguard_lambda", jnp.ones((n,)))
        x = step_b(x, jnp.asarray(u, dtype=jnp.float32))
        xs.append(x)
        us.append(u)
        raws.append(raw)
        slacks.append(s)
        used.append(ok)
        lams.append(lam)
    st = lambda a: np.stack([np.asarray(v, dtype=np.float64) for v in a], axis=1)  # noqa: E731
    return {"traj": st(xs), "u": st(us), "u_raw": st(raws), "slack": st(slacks), "qp_used": st(used).astype(bool),
            "safeguard_lambda": st(lams)}


def evaluation_starts(ref_x: np.ndarray, library: np.ndarray, n_starts: int, cfg, rec_fn) -> tuple[np.ndarray, bool]:
    """The reference start (if in the safe set and C_N) followed by library starts."""
    x_ref0 = ref_x[0:1].astype(np.float32)
    ok0 = bool(np.asarray(lbcbf.safe_value(jnp.asarray(x_ref0), cfg))[0] >= 0) and bool(np.asarray(rec_fn(jnp.asarray(x_ref0))[0])[0])
    x0 = np.concatenate([x_ref0, library[: n_starts - 1]] if ok0 else [library[:n_starts]], axis=0).astype(np.float32)
    return x0, ok0


__all__ = ["evaluation_starts", "load_landing_run", "make_mean_policy", "rollout_policy"]
