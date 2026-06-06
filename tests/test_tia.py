"""TIA closed-loop transimpedance model tests."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

from opamp_model.config import OpampConfig, TiaConfig
from opamp_model.io import package_root
from opamp_model.metrics import build_tia_metric_entries
from opamp_model.ngspice_engine import (
    NgspiceNotFoundError,
    render_ngspice_tia_netlist,
    simulate_tia_ac_ngspice,
)
from opamp_model.spectre_engine import render_spectre_tia_netlist
from opamp_model.tia import (
    bode_to_tia_result,
    closed_loop_transimpedance,
    extract_tia_metrics,
    feedback_impedance,
    simulate_tia_ac,
)

_NGSPICE_AVAILABLE = (
    shutil.which("ngspice") is not None or shutil.which("ngspice-shared") is not None
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


def test_bode_to_tia_result_from_ac_columns() -> None:
    """AC gain/phase columns with Iin=1 A map to transimpedance metrics."""
    f = np.array([1.0, 1.0e3])
    gain_db = np.array([100.0, 90.0])
    phase_deg = np.array([-5.0, -45.0])
    result = bode_to_tia_result(f, gain_db, phase_deg)
    assert result["zt_ohm"][0] == pytest.approx(10.0 ** (100.0 / 20.0))
    assert result["metrics"]["zt_dc_db"] == pytest.approx(100.0)


def test_tia_metric_source_tags_engine() -> None:
    """TIA metrics record ngspice/Spectre provenance when engine is set."""
    metrics = extract_tia_metrics(np.array([1.0]), np.array([100.0]))
    py = build_tia_metric_entries(metrics, engine="python")
    ng = build_tia_metric_entries(metrics, engine="ngspice")
    sp = build_tia_metric_entries(metrics, engine="spectre")
    assert py["zt_dc_ohm"]["source"] == "python_macromodel"
    assert ng["zt_dc_ohm"]["source"] == "ngspice_wrdata"
    assert sp["zt_dc_ohm"]["source"] == "spectre_psf"


def test_render_ngspice_tia_netlist_substitutes_params() -> None:
    """Rendered ngspice TIA netlist carries CLI macromodel parameters."""
    repo = package_root()
    template = repo / "testbench" / "ngspice" / "run_tia_ac.cir"
    cfg = OpampConfig(a0_db=70.0, gbw_hz=2.0e6, rin_ohm=1.0e9)
    tia = TiaConfig(rf_ohm=50.0e3, cf_f=200.0e-15, cs_f=1.0e-12)
    text = render_ngspice_tia_netlist(template, cfg, tia)
    assert ".param rf_ohm=50000" in text or ".param rf_ohm=5e+04" in text
    assert ".param a0_db=70" in text
    assert "wrdata tia_zt.raw" in text


def test_render_spectre_tia_netlist_absolutizes_va() -> None:
    """Rendered Spectre TIA netlist uses an absolute configurable_tia.va path."""
    repo = package_root()
    template = repo / "testbench" / "spectre" / "run_tia_ac.scs"
    cfg = OpampConfig()
    tia = TiaConfig()
    text = render_spectre_tia_netlist(template, cfg, tia, repo_root=repo)
    va = (repo / "veriloga/configurable_tia.va").resolve()
    assert f'ahdl_include "{va}"' in text
    assert "configurable_tia" in text


@pytest.mark.skipif(not _NGSPICE_AVAILABLE, reason="ngspice not on PATH")
def test_ngspice_tia_matches_python(tmp_path: Path) -> None:
    """ngspice TIA AC aligns with Python macromodel at DC transimpedance."""
    opamp = OpampConfig(a0_db=80.0, gbw_hz=10.0e6, rout_ohm=1.0e12)
    tia = TiaConfig(rf_ohm=100.0e3, cf_f=0.0)
    py = simulate_tia_ac(opamp, tia)
    try:
        ng = simulate_tia_ac_ngspice(opamp, tia, tmp_path)
    except NgspiceNotFoundError:
        pytest.skip("ngspice not available")
    assert ng["metrics"]["zt_dc_ohm"] == pytest.approx(py["metrics"]["zt_dc_ohm"], rel=0.05)
    assert ng["metrics"]["bandwidth_hz"] == pytest.approx(py["metrics"]["bandwidth_hz"], rel=0.15)
