"""AC / STB closed-form and metric extraction tests."""

from __future__ import annotations

import math

import numpy as np
import pytest

from opamp_model.ac import extract_gbw_phase_margin
from opamp_model.config import OpampConfig
from opamp_model.core import dominant_pole_rad_s, open_loop_transfer
from opamp_model.io import log_frequency_sweep
from opamp_model.model import simulate_ac, simulate_stb


def test_dominant_pole_matches_gbw_definition() -> None:
    """Unity-gain frequency matches cfg.gbw_hz for a single-pole model."""
    cfg = OpampConfig(a0_db=80.0, gbw_hz=5.0e6)
    wp = dominant_pole_rad_s(cfg)
    w_ug = wp * cfg.a0_linear
    f_ug = w_ug / (2.0 * np.pi)
    assert f_ug == pytest.approx(cfg.gbw_hz, rel=0.02)


def test_open_loop_gain_at_dc() -> None:
    """DC gain matches A0_LINEAR."""
    cfg = OpampConfig(a0_db=60.0, gbw_hz=1.0e6)
    f = np.array([1.0])
    h = open_loop_transfer(cfg, f)
    assert abs(h[0]) == pytest.approx(cfg.a0_linear, rel=1.0e-6)
    assert np.angle(h[0]) == pytest.approx(0.0, abs=0.01)


def test_extract_gbw_near_configured() -> None:
    """0 dB crossover is near the configured GBW."""
    cfg = OpampConfig(a0_db=80.0, gbw_hz=10.0e6)
    result = simulate_ac(cfg)
    metrics = result["metrics"]
    assert metrics["gbw_hz"] == pytest.approx(cfg.gbw_hz, rel=0.15)
    assert metrics["a0_db"] == pytest.approx(cfg.a0_db, rel=0.5)


def test_single_pole_phase_margin() -> None:
    """Single-pole open loop has ~90° phase margin at GBW (phase ≈ -90°)."""
    cfg = OpampConfig(a0_db=80.0, gbw_hz=10.0e6)
    f = log_frequency_sweep(1.0, 100.0e6, 50)
    h = open_loop_transfer(cfg, f)
    gain_db = 20.0 * np.log10(np.abs(h))
    phase_deg = np.angle(h, deg=True)
    metrics = extract_gbw_phase_margin(f, gain_db, phase_deg)
    assert metrics["phase_margin_deg"] == pytest.approx(90.0, abs=10.0)


def test_stb_scales_with_loop_beta() -> None:
    """Loop beta shifts gain but not phase."""
    cfg = OpampConfig(a0_db=60.0, gbw_hz=1.0e6, loop_beta=0.5)
    open_result = simulate_ac(cfg)
    stb_result = simulate_stb(cfg)
    delta_db = 20.0 * math.log10(0.5)
    assert stb_result["gain_db"][0] == pytest.approx(open_result["gain_db"][0] + delta_db, abs=0.1)
    np.testing.assert_allclose(stb_result["phase_deg"], open_result["phase_deg"])
