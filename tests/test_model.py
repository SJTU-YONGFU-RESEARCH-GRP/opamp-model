"""Tests for simulation entry points."""

from __future__ import annotations

import numpy as np

from opamp_model.config import OpampConfig, OpampNoiseConfig
from opamp_model.model import simulate_ac, simulate_noise, simulate_stb


def test_simulate_ac_returns_bode_vectors() -> None:
    """simulate_ac returns aligned frequency, gain, phase, and metrics."""
    cfg = OpampConfig(a0_db=60.0, gbw_hz=1.0e6)
    result = simulate_ac(cfg)
    assert len(result["frequency_hz"]) == len(result["gain_db"]) == len(result["phase_deg"])
    assert result["frequency_hz"][0] > 0.0
    assert np.max(result["gain_db"]) <= cfg.a0_db + 1.0
    assert "gbw_hz" in result["metrics"]


def test_simulate_stb_applies_loop_beta() -> None:
    """STB loop gain differs from open loop when beta != 1."""
    cfg = OpampConfig(a0_db=60.0, gbw_hz=1.0e6, loop_beta=0.25)
    ac = simulate_ac(cfg)
    stb = simulate_stb(cfg)
    assert stb["gain_db"][10] < ac["gain_db"][10]


def test_simulate_noise_spectrum_positive() -> None:
    """Noise simulation returns non-negative density."""
    cfg = OpampConfig(a0_db=40.0)
    noise = OpampNoiseConfig(en_white_v_per_sqrt_hz=1.0e-8)
    result = simulate_noise(cfg, noise)
    assert np.all(result["noise_v_per_sqrt_hz"] >= 0.0)
    assert result["metrics"]["integrated_noise_rms_v"] > 0.0
