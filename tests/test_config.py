"""Tests for configuration dataclasses."""

from __future__ import annotations

from opamp_model.config import GmConfig, OpampConfig, OpampNoiseConfig, TiaConfig


def test_a0_linear_from_db() -> None:
    """a0_linear matches dB conversion."""
    cfg = OpampConfig(a0_db=80.0)
    assert abs(cfg.a0_linear - 10_000.0) < 1.0


def test_noise_disabled_when_ideal_flags_zero() -> None:
    """OpampNoiseConfig.enabled is False for all-zero densities."""
    noise = OpampNoiseConfig(
        en_white_v_per_sqrt_hz=0.0,
        en_flicker_at_1hz_v_per_sqrt_hz=0.0,
    )
    assert noise.enabled is False


def test_tia_and_gm_defaults() -> None:
    """TIA and Gm configs instantiate with finite defaults."""
    assert TiaConfig().rf_ohm > 0.0
    assert GmConfig().gm_s > 0.0
