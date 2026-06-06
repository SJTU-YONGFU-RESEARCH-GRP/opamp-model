"""Read Cadence Spectre AC results from PSF (binary) under ``*.raw/``."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

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


def _resolve_psf_member(raw_dir: Path, *, extensions: tuple[str, ...]) -> Path:
    """Resolve a PSF data member by ``logFile`` hint or unique extension match."""
    log_file = raw_dir / "logFile"
    if log_file.is_file():
        member = _read_logfile_analysis_psf(log_file)
        if member:
            candidate = raw_dir / member
            if candidate.is_file():
                return candidate

    for ext in extensions:
        members = sorted(raw_dir.glob(f"*{ext}"))
        if len(members) == 1:
            return members[0]
        if members:
            return members[0]

    msg = f"No PSF member {extensions} found under {raw_dir}"
    raise SpectrePsfError(msg)


def resolve_ac_psf_path(raw_dir: Path) -> Path:
    """Resolve the AC PSF member inside a Spectre ``.raw`` directory."""
    return _resolve_psf_member(raw_dir, extensions=(".ac", ".psf"))


def resolve_noise_psf_path(raw_dir: Path) -> Path:
    """Resolve the noise-analysis PSF member inside a Spectre ``.raw`` directory."""
    return _resolve_psf_member(raw_dir, extensions=(".noise", ".noi", ".psf"))


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


def read_spectre_supply_transfer_psf(
    psf_path: Path,
    *,
    output_signal: str = "out",
    supply_signal: str = "vdd",
) -> dict[str, NDArray[np.float64]]:
    """Read supply-to-output AC transfer ``H_ps = V(out)/V(vdd)`` from PSF.

    The PSRR bench drives ``vdd`` with unit AC magnitude and keeps differential
    inputs quiet, so ``out`` alone is sufficient when ``supply_signal`` is absent.
    """
    frequency_hz, out_values = _complex_trace_data(psf_path, output_signal)
    try:
        _, vdd_values = _complex_trace_data(psf_path, supply_signal)
        denom = np.maximum(np.abs(vdd_values), 1.0e-30)
        transfer = (out_values / vdd_values).astype(np.complex128)
    except SpectrePsfError:
        transfer = out_values.astype(np.complex128)
    return {
        "frequency_hz": frequency_hz,
        "transfer": transfer,
    }


def read_spectre_psrr_from_netlists(
    netlists_dir: Path,
    *,
    stem: str = "psrr",
    output_signal: str = "out",
    supply_signal: str = "vdd",
) -> dict[str, NDArray[np.float64]]:
    """Locate and parse PSRR supply transfer under a Spectre netlist run directory."""
    paths = locate_ac_psf(netlists_dir, stem=stem)
    return read_spectre_supply_transfer_psf(
        paths.psf_path,
        output_signal=output_signal,
        supply_signal=supply_signal,
    )


def locate_noise_psf(netlists_dir: Path, stem: str = "noise_open_loop") -> SpectreAcPsfPaths:
    """Locate PSF files for a noise run under ``output_dir/logs/netlists/``."""
    raw_dir = find_spectre_raw_dir(netlists_dir, stem)
    psf_path = resolve_noise_psf_path(raw_dir)
    log_file = raw_dir / "logFile"
    return SpectreAcPsfPaths(
        raw_dir=raw_dir,
        psf_path=psf_path,
        log_file=log_file if log_file.is_file() else None,
    )


def _pick_noise_trace(registry, signal: str):
    """Return the PSF trace for output noise spectral density."""
    for cand in registry.traces:
        if cand.name == signal:
            return cand
    for cand in registry.traces:
        name = cand.name.lower()
        if "noise" in name or name in {"out", "outn", "vout"}:
            return cand
    names = ", ".join(t.name for t in registry.traces)
    msg = f"No noise trace matching '{signal}' in PSF ({names})"
    raise SpectrePsfError(msg)


def read_spectre_noise_psf(
    psf_path: Path,
    *,
    signal: str = "out",
) -> dict[str, NDArray[np.float64]]:
    """Read output noise spectral density (V/√Hz) vs frequency from noise PSF."""
    registry = _load_psf_registry(psf_path)
    sweep = registry.sweeps[0] if registry.sweeps else None
    if sweep is None or not sweep.data:
        msg = f"No frequency sweep in PSF file {psf_path}"
        raise SpectrePsfError(msg)
    frequency_hz = np.asarray([float(np.real(z)) for z in sweep.data], dtype=np.float64)
    trace = _pick_noise_trace(registry, signal)
    values = np.asarray(trace.data, dtype=np.complex128)
    if values.shape[0] != frequency_hz.shape[0]:
        msg = (
            f"PSF sweep length {frequency_hz.shape[0]} != trace '{trace.name}' "
            f"length {values.shape[0]}"
        )
        raise SpectrePsfError(msg)
    magnitude = np.abs(values).astype(np.float64)
    # Spectre may store V^2/Hz; heuristically sqrt when magnitudes are squared PSD.
    if np.nanmax(magnitude) > 1.0e-3:
        magnitude = np.sqrt(np.maximum(magnitude, 0.0))
    return {
        "frequency_hz": frequency_hz,
        "noise_v_per_sqrt_hz": magnitude,
    }


def read_spectre_noise_from_netlists(
    netlists_dir: Path,
    *,
    stem: str = "noise_open_loop",
    signal: str = "out",
) -> dict[str, NDArray[np.float64]]:
    """Locate and parse noise PSF under a Spectre netlist run directory."""
    paths = locate_noise_psf(netlists_dir, stem=stem)
    return read_spectre_noise_psf(paths.psf_path, signal=signal)
