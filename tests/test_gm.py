"""Tests for Gm / OTA macromodel."""

from __future__ import annotations

import numpy as np
import pytest

from opamp_model.config import GmConfig, OpampNoiseConfig
from opamp_model.gm import (
    extract_gm_metrics,
    simulate_gm_ac,
    transconductance_transfer,
    zout_gm,
)


def test_zout_near_rout_at_low_frequency() -> None:
    """At low frequency, |Zout| approaches Rout."""
    cfg = GmConfig(rout_ohm=1.0e6, cout_f=500.0e-15)
    z = zout_gm(cfg, np.array([1.0]))
    assert abs(z[0]) == pytest.approx(1.0e6, rel=0.01)


def test_transconductance_dc_gain() -> None:
    """Loaded DC gain equals gm * Rout in linear scale."""
    cfg = GmConfig(gm_s=2.0e-3, rout_ohm=50.0e3, cout_f=0.0)
    h = transconductance_transfer(cfg, np.array([10.0]))
    assert abs(h[0]) == pytest.approx(cfg.gm_s * cfg.rout_ohm, rel=1.0e-6)


def test_simulate_gm_ac_returns_bode_and_flat_gm() -> None:
    """simulate_gm_ac returns aligned vectors and constant gm."""
    cfg = GmConfig(gm_s=1.0e-3, rout_ohm=1.0e6, cout_f=500.0e-15)
    result = simulate_gm_ac(cfg)
    n = len(result["frequency_hz"])
    assert n == len(result["gain_db"]) == len(result["phase_deg"]) == len(result["gm_s"])
    assert np.allclose(result["gm_s"], cfg.gm_s)
    assert result["metrics"]["gm_s"] == cfg.gm_s
    assert result["frequency_hz"][0] > 0.0


def test_bandwidth_near_output_rc_pole() -> None:
    """−3 dB bandwidth tracks 1/(2π Rout Cout) within a decade."""
    cfg = GmConfig(gm_s=1.0e-3, rout_ohm=100.0e3, cout_f=1.0e-12)
    result = simulate_gm_ac(cfg)
    f_expected = 1.0 / (2.0 * np.pi * cfg.rout_ohm * cfg.cout_f)
    bw = result["metrics"]["bandwidth_hz"]
    assert bw == pytest.approx(f_expected, rel=0.5)


def test_extract_gm_metrics_from_sweep() -> None:
    """extract_gm_metrics reports configured gm and Rout."""
    cfg = GmConfig(gm_s=3.0e-3, rout_ohm=200.0e3)
    f = np.logspace(0, 6, 50)
    gain_db = np.full(50, 40.0)
    metrics = extract_gm_metrics(cfg, f, gain_db)
    assert metrics["gm_s"] == cfg.gm_s
    assert metrics["rout_ohm"] == cfg.rout_ohm
    assert metrics["gain_db"] == pytest.approx(40.0)


def test_transconductance_accepts_scalar_frequency() -> None:
    """transconductance_transfer accepts a scalar frequency."""
    cfg = GmConfig()
    h = transconductance_transfer(cfg, 1.0e3)
    assert h.shape == (1,)


def test_simulate_gm_ac_with_noise_config() -> None:
    """Noise config is accepted without changing small-signal gm."""
    cfg = GmConfig()
    noise = OpampNoiseConfig(en_white_v_per_sqrt_hz=1.0e-9)
    result = simulate_gm_ac(cfg, noise)
    assert result["gm_s"][0] == cfg.gm_s
