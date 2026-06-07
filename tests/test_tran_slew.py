"""Transient slew-rate simulation and extraction tests."""

from __future__ import annotations

import numpy as np
import pytest

from opamp_model.config import OpampConfig, OpampNoiseConfig
from opamp_model.noise_tran import transient_noise_rms
from opamp_model.tran import (
    extract_slew_rate,
    measure_slew_rates,
    simulate_step_response,
)


def test_step_response_reaches_clipped_final() -> None:
    """Unity-gain follower output settles at the swing-limited target."""
    cfg = OpampConfig(
        gbw_hz=1.0e6,
        slew_pos_vps=5.0e6,
        slew_neg_vps=-5.0e6,
        vcm_v=0.0,
        vswing_high_v=1.0,
        vswing_low_v=-1.0,
    )
    step_v = 1.5
    result = simulate_step_response(
        cfg,
        None,
        step_v=step_v,
        duration_s=5.0e-6,
        dt_s=1.0e-9,
    )
    expected = cfg.vswing_high_v
    assert result["vout_v"][-1] == pytest.approx(expected, abs=1.0e-3)


def test_measured_slew_matches_config() -> None:
    """10–90 % slew rate tracks configured SR+ / SR− within tolerance."""
    slew = 2.0e6
    cfg = OpampConfig(
        gbw_hz=100.0e6,
        slew_pos_vps=slew,
        slew_neg_vps=-slew,
        vcm_v=0.0,
        vswing_high_v=2.0,
        vswing_low_v=-2.0,
    )
    metrics, _, _ = measure_slew_rates(
        cfg,
        None,
        step_v=0.5,
        duration_s=2.0e-6,
        dt_s=5.0e-10,
    )
    assert metrics["slew_pos_vps"] == pytest.approx(slew, rel=0.15)
    assert metrics["slew_neg_vps"] == pytest.approx(-slew, rel=0.15)


def test_step_response_includes_noise_overlay_fields() -> None:
    """Noise-enabled step simulation returns clean and noise traces."""
    cfg = OpampConfig(gbw_hz=1.0e6, slew_pos_vps=5.0e6, slew_neg_vps=-5.0e6)
    noise = OpampNoiseConfig(
        en_white_v_per_sqrt_hz=1.0e-4,
        en_flicker_1hz_v_per_sqrt_hz=0.0,
        noise_seed=1,
    )
    result = simulate_step_response(
        cfg,
        noise,
        step_v=0.5,
        duration_s=1.0e-6,
        dt_s=1.0e-9,
    )
    assert result["vout_clean_v"].shape == result["vout_v"].shape
    assert transient_noise_rms(result["noise_v"]) > 0.0
    assert np.max(np.abs(result["vout_v"] - result["vout_clean_v"])) > 0.0


def test_extract_slew_rate_on_linear_ramp() -> None:
    """10–90 % extraction on an ideal ramp returns the ramp slope."""
    t = np.linspace(0.0, 1.0e-6, 1001)
    slope = 4.0e6
    v = slope * t
    sr_pos, sr_neg = extract_slew_rate(t, v, vfinal_v=float(v[-1]))
    assert sr_pos == pytest.approx(slope, rel=0.02)
    assert not np.isfinite(sr_neg)
