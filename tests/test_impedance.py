"""Tests for impedance helpers."""

from __future__ import annotations

import numpy as np
import pytest

from opamp_model.config import OpampConfig
from opamp_model.impedance import zin_diff, zout


def test_zin_near_rin_at_low_frequency() -> None:
    """At low frequency, |Zin| approaches Rin."""
    cfg = OpampConfig(rin_ohm=1.0e6, cin_f=1.0e-12)
    z = zin_diff(cfg, np.array([1.0]))
    assert abs(z[0]) == pytest.approx(1.0e6, rel=0.01)


def test_zout_near_rout_at_low_frequency() -> None:
    """At low frequency, |Zout| approaches Rout."""
    cfg = OpampConfig(rout_ohm=50.0, cout_f=1.0e-12)
    z = zout(cfg, np.array([10.0]))
    assert abs(z[0]) == pytest.approx(50.0, rel=0.01)
