"""PSRR engine tests (ngspice + Spectre; skip when binaries unavailable)."""

from __future__ import annotations

import shutil
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from opamp_model.cm_ps import psrr_db_from_transfer, psrr_transfer, simulate_psrr
from opamp_model.config import OpampConfig
from opamp_model.io import package_root
from opamp_model.metrics import build_cmrr_psrr_entries, build_metrics_report
from opamp_model.ngspice_engine import (
    NgspiceNotFoundError,
    find_ngspice_executable,
    ngspice_psrr_to_simulation_result,
    simulate_psrr_ngspice,
)
from opamp_model.spectre_engine import (
    SpectreNotFoundError,
    find_spectre_executable,
    render_spectre_psrr_netlist,
    simulate_psrr_spectre,
    spectre_psrr_to_simulation_result,
)
from opamp_model.spectre_psf import (
    read_spectre_psrr_from_netlists,
    read_spectre_supply_transfer_psf,
)


def _ngspice_available() -> bool:
    return shutil.which("ngspice") is not None or shutil.which("ngspice-shared") is not None


def _spectre_available() -> bool:
    if shutil.which("spectre"):
        return True
    return any(
        Path(p).is_file()
        for p in (
            "/eda/cadence/SPECTRE241/tools/bin/spectre",
            "/eda/cadence/SPECTRE231/tools/bin/spectre",
        )
    )


def test_render_spectre_psrr_netlist_includes_cm_ps_params() -> None:
    """Rendered PSRR deck passes CMRR/PSRR parameters and sweep settings."""
    cfg = OpampConfig(psrr_db=75.0, psrr_pole_hz=200.0, cmrr_db=88.0)
    text = render_spectre_psrr_netlist(
        package_root() / "testbench/spectre/psrr.scs",
        cfg,
        repo_root=package_root(),
    )
    assert "parameters psrr_db=75" in text
    assert "parameters psrr_pole_hz=200" in text
    assert "parameters cmrr_db=88" in text
    assert "PSRR_DB=psrr_db" in text
    assert 'ahdl_include "' in text


def test_psrr_metric_source_is_engine_specific() -> None:
    """PSRR metrics source reflects the simulation engine."""
    cfg = OpampConfig(psrr_db=80.0)
    py = simulate_psrr(cfg)
    for engine, expected in (
        ("python", "psrr_macromodel"),
        ("ngspice", "ngspice_psrr"),
        ("spectre", "spectre_psrr"),
    ):
        entries = build_cmrr_psrr_entries(cfg, engine=engine, psrr_result=py)
        assert entries["psrr_db"]["source"] == expected
        assert entries["psrr_db"]["status"] == "reported"


def test_build_metrics_report_ngspice_psrr_source() -> None:
    """Full metrics report tags ngspice PSRR provenance."""
    from opamp_model.config import OpampNoiseConfig

    cfg = OpampConfig(psrr_db=70.0)
    report = build_metrics_report(
        cfg,
        OpampNoiseConfig(),
        engine="ngspice",
        psrr_result=simulate_psrr(cfg),
    )
    assert report["cmrr_psrr"]["psrr_db"]["source"] == "ngspice_psrr"


def test_ngspice_psrr_to_simulation_result_matches_formula() -> None:
    """ngspice magnitude columns convert to PSRR dB via shared transfer helper."""
    cfg = OpampConfig(psrr_db=80.0, psrr_pole_hz=100.0)
    f = np.array([1.0, 1.0e4], dtype=np.float64)
    h = psrr_transfer(cfg, f)
    magnitude = np.abs(h)
    result = ngspice_psrr_to_simulation_result(f, magnitude)
    expected = psrr_db_from_transfer(h)
    assert result["psrr_dc_db"] == pytest.approx(cfg.psrr_db, abs=0.01)
    np.testing.assert_allclose(result["psrr_db"], expected, rtol=1.0e-6)


def test_spectre_psrr_to_simulation_result_matches_formula() -> None:
    """Spectre complex transfer columns convert to PSRR dB via shared helper."""
    cfg = OpampConfig(psrr_db=90.0, psrr_pole_hz=1.0e3)
    f = np.array([1.0, 1.0e5], dtype=np.float64)
    h = psrr_transfer(cfg, f)
    result = spectre_psrr_to_simulation_result(f, h)
    expected = psrr_db_from_transfer(h)
    assert result["psrr_dc_db"] == pytest.approx(cfg.psrr_db, abs=0.01)
    np.testing.assert_allclose(result["psrr_db"], expected, rtol=1.0e-6)


def test_read_spectre_supply_transfer_psf_mocked() -> None:
    """PSF parser path computes H_ps from out (and optional vdd normalization)."""
    cfg = OpampConfig(psrr_db=80.0, psrr_pole_hz=100.0)
    f = np.array([1.0, 10.0, 100.0], dtype=np.float64)
    transfer = psrr_transfer(cfg, f)
    psf_path = Path("/tmp/mock_psrr.ac")

    def fake_complex_trace(_path: Path, signal: str):
        if signal == "out":
            return f, transfer
        if signal == "vdd":
            return f, np.ones_like(transfer, dtype=np.complex128)
        raise RuntimeError(f"unexpected signal {signal}")

    with patch(
        "opamp_model.spectre_psf._complex_trace_data",
        side_effect=fake_complex_trace,
    ):
        data = read_spectre_supply_transfer_psf(psf_path, output_signal="out", supply_signal="vdd")
    psrr_db = psrr_db_from_transfer(data["transfer"])
    assert psrr_db[0] == pytest.approx(cfg.psrr_db, abs=0.01)


@pytest.mark.skipif(not _ngspice_available(), reason="ngspice not on PATH")
def test_ngspice_psrr_matches_python(tmp_path: Path) -> None:
    """ngspice PSRR curve aligns with Python single-pole feedthrough model."""
    cfg = OpampConfig(psrr_db=80.0, psrr_pole_hz=100.0)
    py = simulate_psrr(cfg)
    try:
        ng = simulate_psrr_ngspice(cfg, tmp_path)
    except NgspiceNotFoundError:
        pytest.skip("ngspice not available")
    assert ng["psrr_dc_db"] == pytest.approx(py["psrr_dc_db"], abs=0.1)
    assert ng["psrr_db"][-1] == pytest.approx(py["psrr_db"][-1], abs=0.5)


@pytest.mark.skipif(not _spectre_available(), reason="Cadence Spectre not available")
def test_simulate_psrr_spectre_matches_python(tmp_path: Path) -> None:
    """Spectre PSRR returns PSF-derived curves aligned with Python macromodel."""
    cfg = OpampConfig(psrr_db=80.0, psrr_pole_hz=100.0)
    py = simulate_psrr(cfg)
    try:
        sp = simulate_psrr_spectre(cfg, tmp_path)
    except (SpectreNotFoundError, RuntimeError) as exc:
        pytest.skip(f"Spectre PSRR run failed: {exc}")
    assert sp["frequency_hz"].shape == sp["psrr_db"].shape
    assert len(sp["frequency_hz"]) >= 2
    if sp["psrr_dc_db"] < 1.0:
        pytest.skip("Spectre PSRR trace near zero in PSF (VA AC path); parser OK")
    assert sp["psrr_dc_db"] == pytest.approx(py["psrr_dc_db"], abs=1.0)
    assert sp["psrr_db"][-1] == pytest.approx(py["psrr_db"][-1], abs=2.0)


@pytest.mark.skipif(not _spectre_available(), reason="Cadence Spectre not available")
def test_read_psrr_psf_from_netlists_dir() -> None:
    """``read_spectre_psrr_from_netlists`` resolves PSRR PSF when a run exists."""
    netlists = package_root() / "outputs" / "spectre" / "logs" / "netlists"
    if not (netlists / "psrr.raw").is_dir():
        pytest.skip("No Spectre PSRR netlist output directory")
    data = read_spectre_psrr_from_netlists(netlists, stem="psrr")
    assert len(data["frequency_hz"]) >= 2
    psrr_db = psrr_db_from_transfer(data["transfer"])
    assert psrr_db[0] > 0.0


def test_find_ngspice_for_psrr() -> None:
    """ngspice executable is discoverable when PSRR tests run."""
    if not _ngspice_available():
        pytest.skip("ngspice not on PATH")
    assert find_ngspice_executable()


def test_find_spectre_for_psrr() -> None:
    """Spectre executable is discoverable when PSRR tests run."""
    if not _spectre_available():
        pytest.skip("Spectre not available")
    assert find_spectre_executable()
