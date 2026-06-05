"""Frequency grids, CSV I/O, and repository path helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

BODE_COLUMNS = ("frequency_hz", "gain_db", "phase_deg")
SLEW_STEP_COLUMNS = ("time_s", "vout_pos_v", "vout_neg_v")
THD_WAVEFORM_COLUMNS = ("time_s", "vout_v")
CMRR_COLUMNS = ("frequency_hz", "acm_db", "cmrr_db")
PSRR_COLUMNS = ("frequency_hz", "psrr_db")
NOISE_COLUMNS = ("frequency_hz", "noise_v_per_sqrt_hz")
NOISE_BREAKDOWN_COLUMNS = (
    "frequency_hz",
    "en_in_white_v_per_sqrt_hz",
    "en_in_flicker_v_per_sqrt_hz",
    "en_in_total_v_per_sqrt_hz",
    "en_out_white_v_per_sqrt_hz",
    "en_out_flicker_v_per_sqrt_hz",
    "en_out_total_v_per_sqrt_hz",
)
TIA_COLUMNS = ("frequency_hz", "zt_ohm", "zt_db", "zt_phase_deg")


def package_root() -> Path:
    """Return the opamp-model repository root (parent of ``src/``)."""
    return Path(__file__).resolve().parents[2]


def log_frequency_sweep(
    f_start_hz: float,
    f_stop_hz: float,
    points_per_decade: int,
) -> NDArray[np.float64]:
    """Return a log-spaced frequency vector (Hz).

    Args:
        f_start_hz: Lowest frequency (Hz); must be positive.
        f_stop_hz: Highest frequency (Hz); must exceed ``f_start_hz``.
        points_per_decade: Samples per decade (>= 1).

    Returns:
        Monotonic frequency samples in Hz.

    Raises:
        ValueError: Invalid sweep bounds or point count.
    """
    if f_start_hz <= 0.0 or f_stop_hz <= f_start_hz:
        msg = f"Invalid sweep: f_start={f_start_hz}, f_stop={f_stop_hz}"
        raise ValueError(msg)
    if points_per_decade < 1:
        raise ValueError("points_per_decade must be >= 1")
    decades = np.log10(f_stop_hz / f_start_hz)
    num_points = max(int(np.ceil(decades * points_per_decade)) + 1, 2)
    return np.logspace(np.log10(f_start_hz), np.log10(f_stop_hz), num_points)


def write_bode_csv(
    path: Path,
    frequency_hz: NDArray[np.float64],
    gain_db: NDArray[np.float64],
    phase_deg: NDArray[np.float64],
) -> None:
    """Write a Bode plot CSV with columns frequency, gain_db, phase_deg."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.column_stack([frequency_hz, gain_db, phase_deg])
    header = ",".join(BODE_COLUMNS)
    np.savetxt(path, data, delimiter=",", header=header, comments="")


def read_ngspice_noise_wrdata(path: Path) -> dict[str, NDArray[np.float64]]:
    """Read ngspice ``wrdata`` from a ``.noise`` analysis.

    With ``wr_singlescale``, leading columns may repeat frequency; density columns
    are taken from the last two columns (``onoise_spectrum``, ``inoise_spectrum``).
    """
    table = np.loadtxt(path)
    if table.ndim == 1:
        table = table.reshape(1, -1)
    if table.shape[1] < 3:
        msg = f"Expected >= 3 columns in noise wrdata, got {table.shape[1]}."
        raise ValueError(msg)
    if table.shape[1] >= 6:
        return {
            "frequency_hz": table[:, 0].astype(np.float64),
            "onoise_v_per_sqrt_hz": table[:, -2].astype(np.float64),
            "inoise_v_per_sqrt_hz": table[:, -1].astype(np.float64),
        }
    if table.shape[1] >= 4:
        return {
            "frequency_hz": table[:, 0].astype(np.float64),
            "onoise_v_per_sqrt_hz": table[:, -1].astype(np.float64),
            "inoise_v_per_sqrt_hz": table[:, -1].astype(np.float64),
        }
    return {
        "frequency_hz": table[:, 0].astype(np.float64),
        "onoise_v_per_sqrt_hz": table[:, 1].astype(np.float64),
        "inoise_v_per_sqrt_hz": table[:, 2].astype(np.float64),
    }


def read_ngspice_ac_wrdata(path: Path) -> dict[str, NDArray[np.float64]]:
    """Read ngspice ``wrdata`` from AC analysis (frequency, vdb, vp columns).

    With ``wr_singlescale``, ngspice may duplicate the frequency column; gain and
    phase are taken from the last two columns.
    """
    table = np.loadtxt(path)
    if table.ndim == 1:
        table = table.reshape(1, -1)
    if table.shape[1] < 3:
        msg = f"Expected >= 3 columns in AC wrdata, got {table.shape[1]}."
        raise ValueError(msg)
    if table.shape[1] >= 5:
        phase = table[:, -1].astype(np.float64)
        if np.max(np.abs(phase)) < 10.0:
            phase = np.degrees(phase)
        return {
            "frequency_hz": table[:, 0].astype(np.float64),
            "gain_db": table[:, -2].astype(np.float64),
            "phase_deg": phase,
        }
    return {
        "frequency_hz": table[:, 0].astype(np.float64),
        "gain_db": table[:, 1].astype(np.float64),
        "phase_deg": table[:, 2].astype(np.float64),
    }


def write_noise_csv(
    path: Path,
    frequency_hz: NDArray[np.float64],
    noise_v_per_sqrt_hz: NDArray[np.float64],
) -> None:
    """Write a noise spectrum CSV with columns frequency_hz, noise_v_per_sqrt_hz."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.column_stack([frequency_hz, noise_v_per_sqrt_hz])
    header = ",".join(NOISE_COLUMNS)
    np.savetxt(path, data, delimiter=",", header=header, comments="")


def write_noise_breakdown_csv(
    path: Path,
    frequency_hz: NDArray[np.float64],
    en_in_white_v_per_sqrt_hz: NDArray[np.float64],
    en_in_flicker_v_per_sqrt_hz: NDArray[np.float64],
    en_in_total_v_per_sqrt_hz: NDArray[np.float64],
    en_out_white_v_per_sqrt_hz: NDArray[np.float64],
    en_out_flicker_v_per_sqrt_hz: NDArray[np.float64],
    en_out_total_v_per_sqrt_hz: NDArray[np.float64],
) -> None:
    """Write white/flicker/total noise breakdown at input and output."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.column_stack(
        [
            frequency_hz,
            en_in_white_v_per_sqrt_hz,
            en_in_flicker_v_per_sqrt_hz,
            en_in_total_v_per_sqrt_hz,
            en_out_white_v_per_sqrt_hz,
            en_out_flicker_v_per_sqrt_hz,
            en_out_total_v_per_sqrt_hz,
        ]
    )
    header = ",".join(NOISE_BREAKDOWN_COLUMNS)
    np.savetxt(path, data, delimiter=",", header=header, comments="")


def write_cmrr_csv(
    path: Path,
    frequency_hz: NDArray[np.float64],
    acm_db: NDArray[np.float64],
    cmrr_db: NDArray[np.float64],
) -> None:
    """Write CMRR bench CSV (frequency, ACM, CMRR in dB)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.column_stack([frequency_hz, acm_db, cmrr_db])
    header = ",".join(CMRR_COLUMNS)
    np.savetxt(path, data, delimiter=",", header=header, comments="")


def write_psrr_csv(
    path: Path,
    frequency_hz: NDArray[np.float64],
    psrr_db: NDArray[np.float64],
) -> None:
    """Write PSRR bench CSV (frequency, PSRR in dB)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.column_stack([frequency_hz, psrr_db])
    header = ",".join(PSRR_COLUMNS)
    np.savetxt(path, data, delimiter=",", header=header, comments="")


def write_slew_step_csv(
    path: Path,
    time_s: NDArray[np.float64],
    vout_pos_v: NDArray[np.float64],
    vout_neg_v: NDArray[np.float64],
) -> None:
    """Write positive/negative step waveforms sharing one time axis."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n = min(len(time_s), len(vout_pos_v), len(vout_neg_v))
    data = np.column_stack([time_s[:n], vout_pos_v[:n], vout_neg_v[:n]])
    header = ",".join(SLEW_STEP_COLUMNS)
    np.savetxt(path, data, delimiter=",", header=header, comments="")


def write_thd_waveform_csv(
    path: Path,
    time_s: NDArray[np.float64],
    vout_v: NDArray[np.float64],
) -> None:
    """Write transient sine waveform CSV (time_s, vout_v)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.column_stack([time_s, vout_v])
    header = ",".join(THD_WAVEFORM_COLUMNS)
    np.savetxt(path, data, delimiter=",", header=header, comments="")


def write_tia_csv(
    path: Path,
    frequency_hz: NDArray[np.float64],
    zt_ohm: NDArray[np.float64],
    zt_db: NDArray[np.float64],
    zt_phase_deg: NDArray[np.float64],
) -> None:
    """Write TIA AC CSV (frequency, |Zt|, Zt dB, phase)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.column_stack([frequency_hz, zt_ohm, zt_db, zt_phase_deg])
    header = ",".join(TIA_COLUMNS)
    np.savetxt(path, data, delimiter=",", header=header, comments="")


def read_bode_csv(path: Path) -> dict[str, NDArray[np.float64]]:
    """Load a Bode CSV written by :func:`write_bode_csv`."""
    table = np.loadtxt(path, delimiter=",", skiprows=1)
    if table.ndim == 1:
        table = table.reshape(1, -1)
    return {
        "frequency_hz": table[:, 0].astype(np.float64),
        "gain_db": table[:, 1].astype(np.float64),
        "phase_deg": table[:, 2].astype(np.float64),
    }
