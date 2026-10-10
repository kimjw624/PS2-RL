"""Phase-II system binding for landing under a disturbance (UE-bCBF CIL).

Counterpart of ``landing_ps2_binding`` for ``ps2rl.cil.quadrotor_landing_ue_bcbf``:

    landing (nominal)                         landing UE
    ---------------------------------------   ----------------------------------------------------
    LANDING_BINDING (phys_dim 10 = x)         make_landing_ue_binding(...) (phys_dim 13 = x, d_hat)
    landing_projection_setup(CKPT)            landing_ue_projection_setup(CKPT)

The environment (``quadrotor_landing_ue_env``) puts the observer's estimate right after
the physical state, obs = (x, d_hat, ...), so the trainer's ``obs[..., :phys_dim]`` slice
hands the CIL exactly what it needs.

Residual mode (``residual=True``): the environment appends the nominal tracker's command
u_nom as the last 4 observation entries, the actor outputs a in the residual box, and the
CIL filters u_ref = clip(u_nom + a). This uses the two optional ``PS2SystemBinding`` fields
(``actor_bounds_fn``, ``reference_action_fn``); with both unset the core behaves exactly as before.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

from ps2rl.cil import quadrotor_landing_ue_bcbf as uecbf
from ps2rl.phase2_ps2.ps2_trainer_core import PS2SystemBinding

UE_PHYS_DIM = 13  # x (10) + d_hat (3)


def _action_bounds(_action_scale: Any, cbf_cfg: Any):
    return jnp.asarray(cbf_cfg.action_low, dtype=jnp.float32), jnp.asarray(cbf_cfg.action_high, dtype=jnp.float32)


def _disable_backup_fallback(sac_cfg: Any) -> bool:
    return (not sac_cfg.use_projection) and (not sac_cfg.project_actor_actions)


def make_landing_ue_binding(*, residual: bool = True, res_thrust: float = 9.81, res_rate: float = 8.0) -> PS2SystemBinding:
    extra = {}
    if residual:
        box = jnp.asarray([res_thrust, res_rate, res_rate, res_rate], dtype=jnp.float32)
        extra = {
            "actor_bounds_fn": lambda _scale, _cfg: (-box, box),
            "reference_action_fn": lambda obs, a: obs[..., -4:] + a,  # u_ref = u_nom + u_res
        }
    return PS2SystemBinding(
        phys_dim=UE_PHYS_DIM,
        extended_qp_diagnostics=True,
        project_with_info_fn=uecbf.solve_backup_cbf_qp_batch_with_info,
        project_fn=uecbf.solve_backup_cbf_qp_batch,
        backup_policy_fn=uecbf.backup_policy_batch,
        action_bounds_fn=_action_bounds,
        disable_backup_fallback_fn=_disable_backup_fallback,
        **extra,
    )


LANDING_UE_BINDING = make_landing_ue_binding(residual=False)


def landing_ue_projection_setup(checkpoint: str, *, residual: bool = True, **qp_overrides: Any):
    """(cbf_cfg, projector, projection_ops, binding) for a UE Phase-I landing checkpoint."""
    cbf_cfg = uecbf.ue_bcbf_config_from_checkpoint(checkpoint, **qp_overrides)
    projector = uecbf.QuadrotorLandingUEBackupCBFProjector(cbf_cfg)
    binding = make_landing_ue_binding(residual=residual)
    return cbf_cfg, projector, binding.projection_ops(cbf_cfg, projector.runtime), binding


__all__ = ["LANDING_UE_BINDING", "UE_PHYS_DIM", "landing_ue_projection_setup", "make_landing_ue_binding"]
