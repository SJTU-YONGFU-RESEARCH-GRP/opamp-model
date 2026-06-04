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
