"""Read a Phase-II landing run and derive what the landing CIL needs from it.

A Phase-II landing run is a regular quadrotor Phase-II run directory (``configs.json``
with ``sac``/``env``/``cbf`` blocks, ``best_weights.pkl``/``final_weights.pkl``) whose
``env.reference_path`` points at a landing reference made by
``ps2rl/envs/assets/generate_quadrotor_landing_reference.py``. That generator writes
``<reference>_config.json`` next to the ``.npz``; its ``landing_config`` block holds the
pad and the approach cone the reference was designed against.

Nothing task-specific is typed in here: the cone and pad come from the reference config,
the dynamics and action box from the run's ``env`` block, the observation layout from
``build_quadrotor_env`` and the actor size from the saved weights.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from ps2rl.envs.quadrotor_env import QuadrotorEnvConfig
from ps2rl.envs.quadrotor_landing_config import QuadrotorLandingConfig
from ps2rl.phase1_sa.landing_design_region import LandingDesignRegionConfig
from ps2rl.utils.paths import load_pickle_payload

# reference-generator key -> QuadrotorLandingConfig field
REFERENCE_GEOMETRY_KEYS = {
    "pad_x": "pad_x",
    "pad_y": "pad_y",
    "z_pad": "pad_z",
    "r0": "cone_r0",
    "theta_deg": "cone_theta_deg",
    "eps": "cone_eps",
}
# Phase-II env field -> QuadrotorLandingConfig field (dynamics and action box)
ENV_DYNAMICS_KEYS = ("dt", "gravity", "a_cmd_min", "a_cmd_max", "omega_max")


def _dataclass_from_dict(cls, payload: dict[str, Any]):
    valid = {f.name for f in fields(cls) if f.init}
    return cls(**{k: v for k, v in payload.items() if k in valid})


def resolve_repo_path(path: str | Path, *, repo_root: Path, run_dir: Path | None = None) -> Path:
    """Resolve a path stored in a run config (relative to the repo root, or to the run)."""
    p = Path(path)
    if p.is_absolute() and p.exists():
        return p
    for base in (repo_root, run_dir, Path.cwd()):
        if base is not None and (base / p).exists():
            return (base / p).resolve()
    if p.is_absolute():
        # Absolute path from another machine (e.g. /root/PS2-RL/...): try its repo-relative tail.
        parts = p.parts
        for i in range(len(parts)):
            cand = repo_root.joinpath(*parts[i:])
            if cand.exists():
                return cand.resolve()
    raise FileNotFoundError(f"cannot resolve '{path}' (tried {repo_root}, {run_dir}, {Path.cwd()})")


@dataclass
class Phase2Run:
    run_dir: Path
    raw: dict[str, Any]
    env_cfg: QuadrotorEnvConfig
    weights_path: Path
    actor_params: Any
    reference_path: Path
    reference_config_path: Path | None
    reference_config: dict[str, Any] | None

    @property
    def sac_dict(self) -> dict[str, Any]:
        return dict(self.raw.get("sac", {}))

    @property
    def actor_obs_dim(self) -> int:
        return int(np.asarray(self.actor_params["layers"][0]["w"]).shape[0])

    @property
    def actor_hidden_sizes(self) -> tuple[int, ...]:
        return tuple(int(np.asarray(l["w"]).shape[1]) for l in self.actor_params["layers"][:-1])

    @property
    def actor_head_dim(self) -> int:
        return int(np.asarray(self.actor_params["layers"][-1]["w"]).shape[1])


def load_phase2_run(
    run_dir: str | Path,
    *,
    repo_root: str | Path,
    checkpoint: str = "best",
    reference_config: str | Path | None = None,
) -> Phase2Run:
    run_dir = Path(run_dir).resolve()
    repo_root = Path(repo_root).resolve()
    raw = json.loads((run_dir / "configs.json").read_text(encoding="utf-8"))
    env_raw = dict(raw.get("env", {}))
    ref_path = resolve_repo_path(env_raw.get("reference_path", ""), repo_root=repo_root, run_dir=run_dir)
    env_raw["reference_path"] = str(ref_path)
    env_cfg = _dataclass_from_dict(QuadrotorEnvConfig, env_raw)

    names = {"best": "best_weights.pkl", "final": "final_weights.pkl"}
    weights_path = run_dir / names.get(checkpoint, checkpoint)
    if not weights_path.exists():
        raise FileNotFoundError(f"{weights_path} not found (checkpoint='{checkpoint}')")
    payload = load_pickle_payload(weights_path)
    actor_params = payload.get("actor_params", payload)

    if reference_config is None:
        cand = ref_path.with_name(ref_path.stem + "_config.json")
        ref_cfg_path = cand if cand.exists() else None
    else:
        ref_cfg_path = Path(reference_config).resolve()
    ref_cfg = json.loads(ref_cfg_path.read_text(encoding="utf-8")) if ref_cfg_path else None
    return Phase2Run(run_dir, raw, env_cfg, weights_path, actor_params, ref_path, ref_cfg_path, ref_cfg)


def reference_geometry(run: Phase2Run) -> dict[str, float] | None:
    """Pad and cone the reference was designed against, as QuadrotorLandingConfig fields."""
    if not run.reference_config:
        return None
    lc = run.reference_config.get("landing_config", run.reference_config)
    missing = [k for k in REFERENCE_GEOMETRY_KEYS if k not in lc]
    if missing:
        raise KeyError(f"{run.reference_config_path} lacks {missing}")
    return {dst: float(lc[src]) for src, dst in REFERENCE_GEOMETRY_KEYS.items()}


def env_dynamics(run: Phase2Run) -> dict[str, float]:
    return {k: float(getattr(run.env_cfg, k)) for k in ENV_DYNAMICS_KEYS}


def reference_envelope(run: Phase2Run, pad_z: float) -> dict[str, float]:
    """Extremes of the reference (height above pad, speed, tilt, yaw)."""
    b = np.load(run.reference_path)
    s = np.asarray(b["states"], dtype=np.float64)
    q = s[:, 6:10] / np.linalg.norm(s[:, 6:10], axis=1, keepdims=True)
    tilt = np.degrees(np.arccos(np.clip(1.0 - 2.0 * (q[:, 1] ** 2 + q[:, 2] ** 2), -1.0, 1.0)))
    yaw = np.degrees(np.arctan2(2.0 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]), 1.0 - 2.0 * (q[:, 2] ** 2 + q[:, 3] ** 2)))
    return {
        "zeta_max": float(np.max(s[:, 2] - pad_z)),
        "speed_max": float(np.max(np.linalg.norm(s[:, 3:6], axis=1))),
        "tilt_max_deg": float(np.max(tilt)),
        "yaw_abs_max_deg": float(np.max(np.abs(yaw))),
        "duration_sec": float(np.asarray(b["t"])[-1]) if "t" in b else float((s.shape[0] - 1) * run.env_cfg.reference_dt),
    }


def _ceil_to(x: float, step: float) -> float:
    return math.ceil(x / step - 1e-9) * step


def phase1_overrides(
    run: Phase2Run,
    *,
    base_landing: QuadrotorLandingConfig | None = None,
    base_region: LandingDesignRegionConfig | None = None,
    geometry: dict[str, float] | None = None,
    v_margin: float | None = None,
) -> dict[str, Any]:
    """Phase-I overrides so the safe-arrival backup matches this Phase-II task.

    * landing geometry: the reference's pad and cone (or ``geometry``);
    * dynamics / action box: the Phase-II env (the backup must drive the same plant);
    * design region: widened (never narrowed) so it covers the reference envelope plus the
      Phase-II initial-state perturbation. The speed margin defaults to ``init_v_range``.
    Base-set level and height are kept; ``train_phase1_landing.py`` re-certifies them.
    """
    base_landing = base_landing or QuadrotorLandingConfig()
    base_region = base_region or LandingDesignRegionConfig()
    geom = geometry if geometry is not None else reference_geometry(run)
    if geom is None:
        raise ValueError(
            f"no reference config next to {run.reference_path}; pass --reference_config or the cone flags explicitly"
        )
    landing = {**geom, **env_dynamics(run)}
    n = int(round(base_landing.T / landing["dt"]))
    if not math.isclose(n * landing["dt"], base_landing.T, rel_tol=1e-9):
        raise ValueError(f"backup horizon T={base_landing.T} is not a multiple of dt={landing['dt']}")
    landing["num_steps"] = n

    env = run.env_cfg
    env_pert = {
        "pz": float(env.init_pz_range),
        "v": float(env.init_v_range),
        "tilt": float(env.init_tilt_deg_range),
        "yaw": float(env.init_yaw_deg_range),
    }
    envl = reference_envelope(run, landing["pad_z"])
    vm = env_pert["v"] if v_margin is None else float(v_margin)
    need = {
        "zeta_max": _ceil_to(envl["zeta_max"] + env_pert["pz"], 0.5),
        "v_max": _ceil_to(envl["speed_max"] + vm, 0.5),
        "tilt_max_deg": _ceil_to(envl["tilt_max_deg"] + env_pert["tilt"], 5.0),
        "yaw_max_deg": _ceil_to(envl["yaw_abs_max_deg"] + env_pert["yaw"], 5.0),
    }
    region = {}
    reasons = {}
    for k, v in need.items():
        cur = float(getattr(base_region, k))
        if v > cur + 1e-9:
            region[k] = float(v)
            reasons[k] = f"widened {cur} -> {v} to cover the reference envelope + initial perturbation"
        else:
            reasons[k] = f"kept {cur} (task needs {v})"
    return {
        "landing": landing,
        "region": region,
        "provenance": {
            "phase2_run": str(run.run_dir),
            "reference_path": str(run.reference_path),
            "reference_config_path": str(run.reference_config_path) if run.reference_config_path else None,
            "reference_envelope": envl,
            "phase2_initial_perturbation": env_pert,
            "speed_margin": vm,
            "region_reasons": reasons,
        },
    }


def parse_field_overrides(items: list[str] | None, cls) -> dict[str, Any]:
    """Parse ``KEY=VALUE`` strings into typed overrides for the dataclass ``cls``.

    Types follow the field defaults (bool accepts 1/0/true/false/yes/no). Unknown keys raise.
    """
    out: dict[str, Any] = {}
    defaults = {f.name: f.default for f in fields(cls)}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"override '{item}' is not KEY=VALUE")
        key, val = (t.strip() for t in item.split("=", 1))
        if key not in defaults:
            raise KeyError(f"'{key}' is not a field of {cls.__name__}")
        d = defaults[key]
        if isinstance(d, bool):
            low = val.lower()
            if low not in ("1", "0", "true", "false", "yes", "no"):
                raise ValueError(f"{key} expects a boolean, got '{val}'")
            out[key] = low in ("1", "true", "yes")
        elif isinstance(d, int) and not isinstance(d, bool):
            out[key] = int(val)
        elif isinstance(d, float) or d is None:
            out[key] = float(val)
        else:
            out[key] = type(d)(val)
    return out


def overrides_tag(overrides: dict[str, Any]) -> str:
    """Short, filesystem-safe tag for a set of overrides (empty if none)."""
    def fmt(v):
        if isinstance(v, bool):
            return "on" if v else "off"
        return f"{v:g}" if isinstance(v, float) else str(v)
    return "_".join(f"{k}-{fmt(v)}" for k, v in sorted(overrides.items()))


def check_geometry_match(run: Phase2Run, landing_cfg: QuadrotorLandingConfig, *, rtol: float = 1e-6) -> list[str]:
    """Mismatches between the backup checkpoint and the Phase-II task (empty = consistent)."""
    bad = []
    geom = reference_geometry(run)
    if geom is not None:
        for k, v in geom.items():
            if not math.isclose(float(getattr(landing_cfg, k)), v, rel_tol=rtol, abs_tol=1e-9):
                bad.append(f"{k}: backup {getattr(landing_cfg, k)} vs reference {v}")
    for k, v in env_dynamics(run).items():
        if not math.isclose(float(getattr(landing_cfg, k)), v, rel_tol=rtol, abs_tol=1e-9):
            bad.append(f"{k}: backup {getattr(landing_cfg, k)} vs Phase-II env {v}")
    return bad


__all__ = [
    "Phase2Run",
    "check_geometry_match",
    "overrides_tag",
    "parse_field_overrides",
    "env_dynamics",
    "load_phase2_run",
    "phase1_overrides",
    "reference_envelope",
    "reference_geometry",
    "resolve_repo_path",
]
