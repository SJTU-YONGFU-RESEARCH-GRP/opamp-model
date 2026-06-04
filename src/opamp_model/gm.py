"""Transconductance (Gm / OTA) model (Phase 5+)."""

from __future__ import annotations

from opamp_model.config import GmConfig, OpampNoiseConfig


def transconductance_transfer(
    gm_cfg: GmConfig,
    frequency_hz: object,
    noise: OpampNoiseConfig | None = None,
) -> object:
    """Return Gm transfer vs frequency.

    Raises:
        NotImplementedError: Phase 5.
    """
    _ = (gm_cfg, frequency_hz, noise)
    raise NotImplementedError("Gm transfer is implemented in Phase 5.")
