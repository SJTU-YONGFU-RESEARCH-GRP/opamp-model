"""Gain-peaking extraction near GBW."""

from __future__ import annotations

import numpy as np
import pytest

from opamp_model.ac import extract_gain_peaking_near_gbw
from opamp_model.config import OpampConfig
from opamp_model.core import open_loop_transfer
from opamp_model.io import log_frequency_sweep


def test_single_pole_has_near_zero_peaking() -> None:
    """Ideal one-pole response has negligible excess gain near GBW."""
    cfg = OpampConfig(a0_db=80.0, gbw_hz=10.0e6, fp2_hz=0.0)
    f = log_frequency_sweep(1.0, 100.0e6, 50)
    h = open_loop_transfer(cfg, f)
    gain_db = 20.0 * np.log10(np.abs(h))
    phase_deg = np.angle(h, deg=True)
    comp = extract_gain_peaking_near_gbw(f, gain_db, phase_deg)
    assert comp["peak_db"] == pytest.approx(0.0, abs=0.5)


def test_zero_before_second_pole_can_peak() -> None:
    """A mid-band zero can produce positive peaking near GBW."""
    cfg = OpampConfig(
        a0_db=60.0,
        gbw_hz=5.0e6,
        fp2_hz=80.0e6,
        fz_hz=15.0e6,
    )
    f = log_frequency_sweep(1.0e3, 200.0e6, 80)
    h = open_loop_transfer(cfg, f)
    gain_db = 20.0 * np.log10(np.abs(h))
    phase_deg = np.angle(h, deg=True)
    comp = extract_gain_peaking_near_gbw(f, gain_db, phase_deg)
    assert comp["peak_db"] > 0.5
    assert np.isfinite(comp["peak_freq_hz"])
    assert comp["gbw_hz"] == pytest.approx(cfg.gbw_hz, rel=0.2)


def test_cmrr_from_aol_and_acm_matches_macromodel() -> None:
    """CMRR from paired AOL/ACM sweeps matches Python macromodel CMRR."""
    from opamp_model.ac import cmrr_from_aol_and_acm
    from opamp_model.cm_ps import common_mode_transfer, simulate_cmrr

    cfg = OpampConfig(
        a0_db=60.0,
        gbw_hz=5.0e6,
        fp2_hz=80.0e6,
        fz_hz=15.0e6,
        cmrr_db=85.0,
    )
    f = log_frequency_sweep(1.0e3, 200.0e6, 80)
    aol = open_loop_transfer(cfg, f)
    acm = common_mode_transfer(cfg, f)
    aol_db = 20.0 * np.log10(np.abs(aol))
    acm_db = 20.0 * np.log10(np.abs(acm))
    paired = cmrr_from_aol_and_acm(f, aol_db, f, acm_db, source="cmrr_bench")
    reference = simulate_cmrr(cfg)
    assert paired["source"] == "cmrr_bench"
    assert paired["cmrr_dc_db"] == pytest.approx(reference["cmrr_dc_db"], abs=0.05)
    assert paired["cmrr_db"][20] == pytest.approx(reference["cmrr_db"][20], abs=0.05)
