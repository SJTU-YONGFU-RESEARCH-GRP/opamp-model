"""Tests for aggregated metrics reporting."""

from __future__ import annotations

import pytest

from opamp_model.ac import cmrr_from_open_loop
from opamp_model.config import OpampConfig, OpampNoiseConfig
from opamp_model.metrics import build_metrics_report, format_metrics_table
from opamp_model.model import simulate_ac


def test_metrics_report_includes_impedance_and_cmrr() -> None:
    """Report lists impedance and CMRR/PSRR entries."""
    cfg = OpampConfig(a0_db=60.0, cmrr_db=85.0, psrr_db=75.0)
    noise = OpampNoiseConfig()
    ac = simulate_ac(cfg)
    cmrr = cmrr_from_open_loop(
        ac["frequency_hz"],
        ac["gain_db"],
        cmrr_linear=cfg.cmrr_linear,
    )
    report = build_metrics_report(
        cfg,
        noise,
        engine="python",
        ac_result=ac,
        cmrr_result=cmrr,
    )
    assert report["impedance"]["zin_ohm"]["value"] is not None
    assert report["cmrr_psrr"]["cmrr_db"]["value"] == pytest.approx(85.0)
    assert report["cmrr_psrr"]["cmrr_db"]["status"] == "reported"
    assert report["cmrr_psrr"]["acm_db"]["status"] == "reported"
    assert report["noise"]["integrated_noise_rms_v"]["status"] == "planned"
    assert report["ac"]["a0_db"]["source"] == "python_macromodel"
    assert report["cmrr_psrr"]["psrr_db"]["source"] == "OpampConfig"
    text = format_metrics_table(report)
    assert "CMRR / PSRR" in text
    assert "Impedance" in text


def test_metric_sources_per_engine() -> None:
    """Source tags reflect simulation origin per engine."""
    cfg = OpampConfig()
    noise = OpampNoiseConfig()
    ac = simulate_ac(cfg)
    cmrr = cmrr_from_open_loop(
        ac["frequency_hz"],
        ac["gain_db"],
        cmrr_linear=cfg.cmrr_linear,
    )
    python = build_metrics_report(
        cfg,
        noise,
        engine="python",
        ac_result=ac,
        cmrr_result=cmrr,
        slew_pos_measured=5e6,
    )
    ngspice = build_metrics_report(
        cfg,
        noise,
        engine="ngspice",
        ac_result=ac,
        slew_pos_measured=5e6,
    )
    assert python["ac"]["gbw_hz"]["source"] == "python_macromodel"
    assert ngspice["ac"]["gbw_hz"]["source"] == "ngspice_wrdata"
    assert python["large_signal"]["slew_pos_vps"]["source"] == "python_macromodel"
    assert ngspice["large_signal"]["slew_pos_vps"]["source"] == "ngspice_tran_wrdata"
