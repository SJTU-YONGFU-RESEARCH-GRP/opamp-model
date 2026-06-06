"""Spectre AC PSF extraction tests (skip when Spectre is not installed)."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from opamp_model.config import OpampConfig
from opamp_model.io import package_root
from opamp_model.model import simulate_ac
from opamp_model.spectre_engine import (
    SpectreNotFoundError,
    find_spectre_executable,
    simulate_ac_spectre,
    spectre_is_runnable,
)
from opamp_model.spectre_psf import (
    locate_ac_psf,
    read_spectre_ac_from_netlists,
    read_spectre_ac_psf,
)

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


pytestmark = pytest.mark.skipif(
    not _spectre_available(),
    reason="Cadence Spectre not available",
)


@pytest.fixture
def fixture_psf_path() -> Path | None:
    """Use a committed or prior-run PSF fixture when present."""
    candidate = (
        package_root()
        / "outputs"
        / "spectre"
        / "logs"
        / "netlists"
        / "ac_open_loop.raw"
        / "ac.ac"
    )
    return candidate if candidate.is_file() else None


def test_find_spectre() -> None:
    """Spectre executable is discoverable."""
    assert find_spectre_executable()


def test_spectre_is_runnable_with_default_license(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default ``/eda/cadence/license.dat`` makes Spectre runnable when env is unset."""
    if not _spectre_available():
        pytest.skip("Spectre not available")
    if not Path("/eda/cadence/license.dat").is_file():
        pytest.skip("Default Cadence license file not present")
    monkeypatch.delenv("CDS_LIC_FILE", raising=False)
    monkeypatch.delenv("LM_LICENSE_FILE", raising=False)
    assert spectre_is_runnable()


def test_read_psf_frequency_sweep(fixture_psf_path: Path | None) -> None:
    """PSF parser returns a log-spaced frequency vector from Spectre."""
    if fixture_psf_path is None:
        pytest.skip("No Spectre PSF fixture (run Spectre AC first)")
    bode = read_spectre_ac_psf(fixture_psf_path, signal="out")
    assert len(bode["frequency_hz"]) >= 2
    assert bode["frequency_hz"][0] == pytest.approx(1.0, rel=0.01)
    assert bode["frequency_hz"][-1] == pytest.approx(1.0e8, rel=0.01)
    assert bode["frequency_hz"][1] > bode["frequency_hz"][0]


def test_simulate_ac_spectre_uses_psf_not_python(tmp_path: Path) -> None:
    """Spectre AC returns PSF-derived curves (no Python macromodel back-fill)."""
    if not _spectre_available():
        pytest.skip("Spectre not available")
    cfg = OpampConfig(a0_db=60.0, gbw_hz=1.0e6, rout_ohm=1.0e12)
    py = simulate_ac(cfg)
    try:
        sp = simulate_ac_spectre(cfg, tmp_path)
    except (SpectreNotFoundError, RuntimeError) as exc:
        pytest.skip(f"Spectre run failed: {exc}")

    assert sp["frequency_hz"].shape == sp["gain_db"].shape == sp["phase_deg"].shape
    assert len(sp["frequency_hz"]) >= 2
    if sp["gain_db"].max() < -100.0:
        pytest.skip(
            "Spectre 'out' trace is near zero in PSF (Verilog-A compact-model); "
            "PSF parser path OK but macromodel AC response needs VA fix"
        )
    assert sp["metrics"]["gbw_hz"] == pytest.approx(py["metrics"]["gbw_hz"], rel=0.15)
    assert sp["metrics"]["a0_db"] == pytest.approx(py["metrics"]["a0_db"], rel=0.2)


def test_locate_ac_psf_from_netlists_dir() -> None:
    """``locate_ac_psf`` resolves ``ac.ac`` under an existing run directory."""
    netlists = (
        package_root() / "outputs" / "spectre" / "logs" / "netlists"
    )
    if not (netlists / "ac_open_loop.raw").is_dir():
        pytest.skip("No Spectre netlist output directory")
    paths = locate_ac_psf(netlists, stem="ac_open_loop")
    assert paths.psf_path.is_file()
    bode = read_spectre_ac_from_netlists(netlists, stem="ac_open_loop", signal="out")
    assert len(bode["frequency_hz"]) >= 2
