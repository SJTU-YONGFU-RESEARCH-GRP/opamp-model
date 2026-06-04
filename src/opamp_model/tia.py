"""Transimpedance amplifier composition (Phase 5+)."""

from __future__ import annotations

from opamp_model.config import OpampConfig, OpampNoiseConfig, TiaConfig


def closed_loop_transimpedance(
    opamp: OpampConfig,
    tia: TiaConfig,
    noise: OpampNoiseConfig | None = None,
) -> object:
    """Return transimpedance vs frequency.

    Raises:
        NotImplementedError: Phase 5.
    """
    _ = (opamp, tia, noise)
    raise NotImplementedError("TIA closed-loop model is implemented in Phase 5.")
