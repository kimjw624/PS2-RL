"""Phase-II system binding for the landing task.

The repo's Phase-II trainer specialises the shared PS2 core through a
``PS2SystemBinding`` (see ``quadrotor_ps2_trainer._BINDING``). This module provides
the landing counterpart, so a landing trainer only swaps the binding, the projector
and the config:

    from ps2rl.phase2_ps2.landing_ps2_binding import (
        LANDING_BINDING, landing_projection_setup)

    cbf_cfg, projector, proj_ops = landing_projection_setup(CKPT)
    # instead of:
    #   cbf_cfg = QuadrotorBCBFConfig(...)
    #   projector = QuadrotorBackupCBFProjector(cbf_cfg)
    #   proj_ops = _BINDING.projection_ops(cbf_cfg, projector.runtime)

Everything geometric (cone, pad, base set, LQR, dt, action box) comes from the Phase-I
checkpoint ``CKPT``; see ``ps2rl.cil.quadrotor_landing_backup_cbf``.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

from ps2rl.cil import quadrotor_landing_backup_cbf as lbcbf
from ps2rl.phase2_ps2.ps2_trainer_core import PS2SystemBinding

_PHYS_DIM = 10


def _action_bounds(_action_scale: Any, cbf_cfg: lbcbf.QuadrotorLandingBCBFConfig):
    return jnp.asarray(cbf_cfg.action_low, dtype=jnp.float32), jnp.asarray(cbf_cfg.action_high, dtype=jnp.float32)


def _disable_backup_fallback(sac_cfg: Any) -> bool:
    # Same rule as the powerloop binding: no projection anywhere -> no backup fallback.
    return (not sac_cfg.use_projection) and (not sac_cfg.project_actor_actions)


LANDING_BINDING = PS2SystemBinding(
    phys_dim=_PHYS_DIM,
    extended_qp_diagnostics=True,
    project_with_info_fn=lbcbf.solve_backup_cbf_qp_batch_with_info,
    project_fn=lbcbf.solve_backup_cbf_qp_batch,
    backup_policy_fn=lbcbf.backup_policy_batch,
    action_bounds_fn=_action_bounds,
    disable_backup_fallback_fn=_disable_backup_fallback,
)


def landing_projection_setup(checkpoint: str, **qp_overrides: Any):
    """(cbf_cfg, projector, projection_ops) for a Phase-I landing checkpoint."""
    cbf_cfg = lbcbf.landing_bcbf_config_from_checkpoint(checkpoint, **qp_overrides)
    projector = lbcbf.QuadrotorLandingBackupCBFProjector(cbf_cfg)
    return cbf_cfg, projector, LANDING_BINDING.projection_ops(cbf_cfg, projector.runtime)


__all__ = ["LANDING_BINDING", "landing_projection_setup"]
