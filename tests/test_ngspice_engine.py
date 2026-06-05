"""ngspice AC engine tests (skip when ngspice is not installed)."""

from __future__ import annotations

from pathlib import Path

import pytest

from opamp_model.config import OpampConfig
from opamp_model.model import simulate_ac
from opamp_model.ngspice_engine import (
    NgspiceNotFoundError,
    find_ngspice_executable,
    simulate_ac_ngspice,
)

pytestmark = pytest.mark.skipif(
    __import__("shutil").which("ngspice") is None
    and __import__("shutil").which("ngspice-shared") is None,
    reason="ngspice not on PATH",
)


def test_find_ngspice() -> None:
    """ngspice executable is discoverable."""
    assert find_ngspice_executable()


def test_ngspice_ac_matches_python(tmp_path: Path) -> None:
    """ngspice AC Bode aligns with Python one-pole model (PLAN tolerances)."""
    cfg = OpampConfig(
        a0_db=60.0,
        gbw_hz=1.0e6,
        rout_ohm=1.0e12,
        fp2_hz=0.0,
        fz_hz=0.0,
    )
    py = simulate_ac(cfg)
    try:
        ng = simulate_ac_ngspice(cfg, tmp_path)
    except NgspiceNotFoundError:
        pytest.skip("ngspice not available")
    assert ng["metrics"]["gbw_hz"] == pytest.approx(py["metrics"]["gbw_hz"], rel=0.05)
    assert ng["metrics"]["a0_db"] == pytest.approx(py["metrics"]["a0_db"], abs=0.1)
    assert ng["metrics"]["phase_margin_deg"] == pytest.approx(
        py["metrics"]["phase_margin_deg"],
        abs=2.0,
    )

