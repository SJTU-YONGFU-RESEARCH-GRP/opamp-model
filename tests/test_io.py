"""Tests for I/O and frequency sweeps."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from opamp_model.io import log_frequency_sweep, read_bode_csv, write_bode_csv


def test_log_frequency_sweep_monotonic() -> None:
    """Log sweep is strictly increasing."""
    f = log_frequency_sweep(1.0, 1.0e6, points_per_decade=10)
    assert f[0] == pytest.approx(1.0)
    assert f[-1] == pytest.approx(1.0e6)
    assert np.all(np.diff(f) > 0.0)


def test_log_frequency_sweep_rejects_invalid_bounds() -> None:
    """Invalid sweep bounds raise ValueError."""
    with pytest.raises(ValueError):
        log_frequency_sweep(0.0, 1.0e6, 10)


def test_bode_csv_roundtrip(tmp_path: Path) -> None:
    """write_bode_csv and read_bode_csv preserve columns."""
    f = np.array([1.0, 10.0, 100.0])
    g = np.array([80.0, 60.0, 40.0])
    p = np.array([0.0, -45.0, -90.0])
    path = tmp_path / "bode.csv"
    write_bode_csv(path, f, g, p)
    loaded = read_bode_csv(path)
    np.testing.assert_allclose(loaded["frequency_hz"], f)
    np.testing.assert_allclose(loaded["gain_db"], g)
    np.testing.assert_allclose(loaded["phase_deg"], p)
