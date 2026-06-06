"""CMRR and PSRR macromodel tests."""

from __future__ import annotations

import numpy as np
import pytest

from opamp_model.ac import cmrr_from_open_loop, extract_cmrr
from opamp_model.cm_ps import (
    common_mode_transfer,
    psrr_db_from_transfer,
    psrr_transfer,
    simulate_cmrr,
    simulate_psrr,
)
from opamp_model.config import OpampConfig
from opamp_model.core import open_loop_transfer
from opamp_model.model import simulate_ac


def test_common_mode_gain_is_aol_over_cmrr() -> None:
    """ACM magnitude equals |Aol|/CMRR_linear at DC."""
    cfg = OpampConfig(a0_db=70.0, cmrr_db=90.0, gbw_hz=1.0e6)
    f = np.array([1.0])
    acm = common_mode_transfer(cfg, f)
    aol = open_loop_transfer(cfg, f)
    assert abs(acm[0]) == pytest.approx(abs(aol[0]) / cfg.cmrr_linear, rel=1.0e-6)


def test_simulate_cmrr_dc_matches_config() -> None:
    """Simulated CMRR at the lowest frequency matches cmrr_db parameter."""
    cfg = OpampConfig(a0_db=80.0, cmrr_db=85.0, gbw_hz=10.0e6)
    result = simulate_cmrr(cfg)
    assert result["cmrr_dc_db"] == pytest.approx(cfg.cmrr_db, abs=0.01)
    assert result["cmrr_db"] == pytest.approx(cfg.cmrr_db, abs=0.01)


def test_extract_cmrr_from_ac_bode() -> None:
    """CMRR derived from open-loop Bode matches macromodel CMRR."""
    cfg = OpampConfig(a0_db=60.0, cmrr_db=100.0, gbw_hz=5.0e6)
    ac = simulate_ac(cfg)
    acm_db, cmrr_db, cmrr_dc = extract_cmrr(
        ac["frequency_hz"],
        ac["gain_db"],
        cmrr_linear=cfg.cmrr_linear,
    )
    assert cmrr_dc == pytest.approx(cfg.cmrr_db, abs=0.01)
    assert acm_db[0] == pytest.approx(ac["gain_db"][0] - cfg.cmrr_db, abs=0.01)
    assert cmrr_db[10] == pytest.approx(cfg.cmrr_db, abs=0.01)


def test_cmrr_from_open_loop_bundle() -> None:
    """cmrr_from_open_loop returns consistent ACM/CMRR arrays."""
    cfg = OpampConfig(cmrr_db=75.0)
    ac = simulate_ac(cfg)
    bundle = cmrr_from_open_loop(
        ac["frequency_hz"],
        ac["gain_db"],
        cmrr_linear=cfg.cmrr_linear,
    )
    assert bundle["cmrr_dc_db"] == pytest.approx(cfg.cmrr_db, abs=0.01)
    assert bundle["source"] == "hybrid_cmrr"
    assert len(bundle["acm_db"]) == len(ac["frequency_hz"])


def test_psrr_dc_matches_config() -> None:
    """PSRR at DC matches psrr_db for the single-pole feedthrough model."""
    cfg = OpampConfig(psrr_db=80.0, psrr_pole_hz=100.0)
    result = simulate_psrr(cfg)
    assert result["psrr_dc_db"] == pytest.approx(cfg.psrr_db, abs=0.01)


def test_psrr_improves_above_pole() -> None:
    """Feedthrough rolls off above the pole, so reported PSRR rises in dB."""
    cfg = OpampConfig(psrr_db=90.0, psrr_pole_hz=1.0e3)
    f = np.array([10.0, 10.0e3, 10.0e6])
    psrr_db = psrr_db_from_transfer(psrr_transfer(cfg, f))
    assert psrr_db[0] == pytest.approx(cfg.psrr_db, abs=0.01)
    assert psrr_db[-1] > psrr_db[0] + 10.0


def test_psrr_pole_frequency_scaling() -> None:
    """Higher pole frequency keeps feedthrough (and PSRR at DC) valid to higher f."""
    cfg_low = OpampConfig(psrr_db=80.0, psrr_pole_hz=100.0)
    cfg_high = OpampConfig(psrr_db=80.0, psrr_pole_hz=10.0e3)
    f_test = 1.0e3
    h_low = abs(psrr_transfer(cfg_low, np.array([f_test]))[0])
    h_high = abs(psrr_transfer(cfg_high, np.array([f_test]))[0])
    assert h_high > h_low
