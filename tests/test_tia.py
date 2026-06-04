"""TIA closed-loop transimpedance model tests."""

from __future__ import annotations

import numpy as np
import pytest

from opamp_model.config import OpampConfig, TiaConfig
from opamp_model.tia import (
    closed_loop_transimpedance,
    extract_tia_metrics,
    feedback_impedance,
    simulate_tia_ac,
)


def test_feedback_impedance_at_dc() -> None:
    """DC feedback impedance equals Rf."""
    tia = TiaConfig(rf_ohm=50.0e3, cf_f=0.0)
    zf = feedback_impedance(tia, np.array([1.0]))
    assert abs(zf[0]) == pytest.approx(tia.rf_ohm, rel=1.0e-6)


def test_high_gain_transimpedance_near_rf() -> None:
    """Large AOL yields |Zt| ≈ Rf at low frequency."""
    opamp = OpampConfig(a0_db=100.0, gbw_hz=10.0e6, rin_ohm=1.0e12)
    tia = TiaConfig(rf_ohm=200.0e3, cf_f=0.0, cs_f=0.0)
    f = np.array([1.0])
    zt = closed_loop_transimpedance(opamp, tia, f)
    assert abs(zt[0]) == pytest.approx(tia.rf_ohm, rel=0.02)


def test_cf_reduces_bandwidth() -> None:
    """Feedback capacitor lowers the −3 dB point vs ideal Rf-only."""
    opamp = OpampConfig(a0_db=80.0, gbw_hz=10.0e6)
    tia_no_c = TiaConfig(rf_ohm=100.0e3, cf_f=0.0)
    tia_with_c = TiaConfig(rf_ohm=100.0e3, cf_f=5.0e-12)
    res_no_c = simulate_tia_ac(opamp, tia_no_c)
    res_c = simulate_tia_ac(opamp, tia_with_c)
    assert res_c["metrics"]["bandwidth_hz"] < res_no_c["metrics"]["bandwidth_hz"]


def test_simulate_tia_ac_bundle() -> None:
    """Simulation returns frequency sweeps and scalar metrics."""
    opamp = OpampConfig(a0_db=60.0, gbw_hz=1.0e6)
    tia = TiaConfig(rf_ohm=10.0e3, cf_f=1.0e-12)
    result = simulate_tia_ac(opamp, tia)
    assert len(result["frequency_hz"]) > 10
    assert len(result["zt_ohm"]) == len(result["frequency_hz"])
    metrics = extract_tia_metrics(result["frequency_hz"], result["zt_db"])
    assert metrics["zt_dc_ohm"] == pytest.approx(result["metrics"]["zt_dc_ohm"])
    assert np.isfinite(result["metrics"]["zt_dc_db"])
