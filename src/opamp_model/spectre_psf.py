"""Read Cadence Spectre AC results from PSF (binary) under ``*.raw/``."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

_PSF_VERSION_RE = re.compile(r'"PSFversion"\s+"([^"]+)"', re.IGNORECASE)
_ANALYSIS_DATA_RE = re.compile(
    r'"([^"]+)"\s+"analysisInst"\s*\(\s*[^"]*"\s*[^"]*"\s*([^"]+)"',
    re.DOTALL,
)


class SpectrePsfError(RuntimeError):
    """Raised when Spectre PSF artifacts cannot be read."""


@dataclass(frozen=True)
class SpectreAcPsfPaths:
    """Paths to Spectre AC PSF artifacts."""

    raw_dir: Path
    psf_path: Path
    log_file: Path | None


def _load_psf_registry(psf_path: Path):
    try:
        from psf_parser import PsfParser
    except ImportError as exc:
        msg = (
            "psf-parser is required to read Spectre PSF results. "
            "Install with: pip install 'psf-parser>=0.2.1'"
        )
        raise SpectrePsfError(msg) from exc
    return PsfParser(str(psf_path)).parse().registry


def find_spectre_raw_dir(netlists_dir: Path, stem: str) -> Path:
    """Return ``<netlists_dir>/<stem>.raw`` if it exists."""
    raw_dir = netlists_dir / f"{stem}.raw"
    if not raw_dir.is_dir():
        msg = f"Spectre raw directory not found: {raw_dir}"
        raise SpectrePsfError(msg)
    return raw_dir


def _read_logfile_analysis_psf(log_file: Path) -> str | None:
    """Parse ``logFile`` and return the PSF data member for the first analysis."""
    text = log_file.read_text(encoding="utf-8", errors="replace")
    for match in _ANALYSIS_DATA_RE.finditer(text):
        data_member = match.group(2).strip()
        if data_member:
            return data_member
    return None


def resolve_ac_psf_path(raw_dir: Path) -> Path:
    """Resolve the AC PSF member inside a Spectre ``.raw`` directory."""
    log_file = raw_dir / "logFile"
    if log_file.is_file():
        member = _read_logfile_analysis_psf(log_file)
        if member:
            candidate = raw_dir / member
            if candidate.is_file():
                return candidate

    ac_members = sorted(raw_dir.glob("*.ac"))
    if len(ac_members) == 1:
        return ac_members[0]
    if ac_members:
        return ac_members[0]

    psf_members = sorted(raw_dir.glob("*.psf"))
    if len(psf_members) == 1:
        return psf_members[0]

    msg = f"No AC PSF member found under {raw_dir}"
    raise SpectrePsfError(msg)


def locate_ac_psf(netlists_dir: Path, stem: str = "ac_open_loop") -> SpectreAcPsfPaths:
    """Locate PSF files for an AC run under ``output_dir/logs/netlists/``."""
    raw_dir = find_spectre_raw_dir(netlists_dir, stem)
    psf_path = resolve_ac_psf_path(raw_dir)
    log_file = raw_dir / "logFile"
    return SpectreAcPsfPaths(
        raw_dir=raw_dir,
        psf_path=psf_path,
        log_file=log_file if log_file.is_file() else None,
    )


def _complex_trace_data(psf_path: Path, signal: str) -> tuple[NDArray[np.float64], NDArray[np.complex128]]:
    """Return sweep frequency (Hz) and complex samples for ``signal``."""
    registry = _load_psf_registry(psf_path)

    sweep = registry.sweeps[0] if registry.sweeps else None
    if sweep is None or not sweep.data:
        msg = f"No frequency sweep in PSF file {psf_path}"
        raise SpectrePsfError(msg)

    frequency_hz = np.asarray([float(np.real(z)) for z in sweep.data], dtype=np.float64)

    trace = None
    for cand in registry.traces:
        if cand.name == signal:
            trace = cand
            break
    if trace is None:
        names = ", ".join(t.name for t in registry.traces)
        msg = f"Signal '{signal}' not in PSF traces ({names})"
        raise SpectrePsfError(msg)

    values = np.asarray(trace.data, dtype=np.complex128)
    if values.shape[0] != frequency_hz.shape[0]:
        msg = (
            f"PSF sweep length {frequency_hz.shape[0]} != trace '{signal}' "
            f"length {values.shape[0]}"
        )
        raise SpectrePsfError(msg)
    return frequency_hz, values


def complex_to_gain_phase(
    values: NDArray[np.complex128],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Convert complex AC samples to gain (dB) and phase (degrees)."""
    magnitude = np.maximum(np.abs(values), 1.0e-30)
    gain_db = (20.0 * np.log10(magnitude)).astype(np.float64)
    phase_deg = np.degrees(np.angle(values)).astype(np.float64)
    return gain_db, phase_deg


def read_spectre_ac_psf(
    psf_path: Path,
    *,
    signal: str = "out",
) -> dict[str, NDArray[np.float64]]:
    """Read open-loop AC Bode vectors from a Spectre PSF file.

    Returns:
        Dict with ``frequency_hz``, ``gain_db``, and ``phase_deg`` (all from Spectre).
    """
    frequency_hz, values = _complex_trace_data(psf_path, signal)
    gain_db, phase_deg = complex_to_gain_phase(values)
    return {
        "frequency_hz": frequency_hz,
        "gain_db": gain_db,
        "phase_deg": phase_deg,
    }


def read_spectre_ac_from_netlists(
    netlists_dir: Path,
    *,
    stem: str = "ac_open_loop",
    signal: str = "out",
) -> dict[str, NDArray[np.float64]]:
    """Locate and parse AC PSF under a Spectre netlist run directory."""
    paths = locate_ac_psf(netlists_dir, stem=stem)
    return read_spectre_ac_psf(paths.psf_path, signal=signal)
