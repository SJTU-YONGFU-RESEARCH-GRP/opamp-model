"""Cadence Spectre AC testbench runner."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from opamp_model.ac import extract_gbw_phase_margin
from opamp_model.config import OpampConfig, OpampNoiseConfig
from opamp_model.io import package_root, write_bode_csv
from opamp_model.io import log_frequency_sweep
from opamp_model.model import AcSimulationResult, NoiseSimulationResult
from opamp_model.ngspice_engine import ngspice_output_noise_spectrum
from opamp_model.noise_analysis import extract_noise_metrics
from opamp_model.spectre_psf import (
    SpectrePsfError,
    locate_ac_psf,
    locate_noise_psf,
    read_spectre_ac_from_netlists,
)


class SpectreNotFoundError(RuntimeError):
    """Raised when the ``spectre`` executable is not on PATH."""


class SpectreLicenseError(RuntimeError):
    """Raised when Spectre is installed but the license file/server is unavailable."""


_DEFAULT_CDS_LIC_FILE = Path("/eda/cadence/license.dat")
_LICENSE_FAILURE_MARKERS = (
    "LMC-01902",
    "Failed to initialize license API",
    "Can't find license file",
)


_SCS_INCLUDE = re.compile(
    r'include\s+"\./testbench/spectre/opamp_include\.scs"\s*',
    re.IGNORECASE,
)


def find_spectre_executable() -> str:
    """Return the Spectre binary path or raise ``SpectreNotFoundError``."""
    found = shutil.which("spectre")
    if found:
        return found
    for candidate in (
        "/eda/cadence/SPECTRE241/tools/bin/spectre",
        "/eda/cadence/SPECTRE231/tools/bin/spectre",
    ):
        if Path(candidate).is_file():
            return candidate
    raise SpectreNotFoundError(
        "Cadence Spectre not found on PATH. Install Spectre or use --simulator python."
    )


def spectre_is_runnable() -> bool:
    """Return True when Spectre exists and a license file/env is configured."""
    try:
        find_spectre_executable()
    except SpectreNotFoundError:
        return False
    if os.environ.get("CDS_LIC_FILE") or os.environ.get("LM_LICENSE_FILE"):
        return True
    return _DEFAULT_CDS_LIC_FILE.is_file()


def _spectre_subprocess_env() -> dict[str, str]:
    """Build subprocess env with Spectre bin on PATH and a default license file."""
    env = os.environ.copy()
    executable = find_spectre_executable()
    spectre_bin = str(Path(executable).parent)
    env["PATH"] = f"{spectre_bin}:{env.get('PATH', '')}"
    if not env.get("CDS_LIC_FILE") and _DEFAULT_CDS_LIC_FILE.is_file():
        env["CDS_LIC_FILE"] = str(_DEFAULT_CDS_LIC_FILE)
    return env


def _raise_if_spectre_license_failure(
    completed: subprocess.CompletedProcess[str],
    *,
    log_path: Path,
) -> None:
    """Raise ``SpectreLicenseError`` when Spectre output indicates license failure."""
    output = completed.stdout + completed.stderr
    if any(marker in output for marker in _LICENSE_FAILURE_MARKERS):
        raise SpectreLicenseError(f"Cadence Spectre license unavailable; see {log_path}")


def _raise_if_spectre_failed(
    completed: subprocess.CompletedProcess[str],
    *,
    log_path: Path,
    message: str,
) -> None:
    """Raise on Spectre failure, distinguishing license errors from other faults."""
    if completed.returncode == 0:
        return
    _raise_if_spectre_license_failure(completed, log_path=log_path)
    raise RuntimeError(f"{message} with code {completed.returncode}; see {log_path}")


def _format_param(value: float | int) -> str:
    if isinstance(value, int):
        return str(value)
    return f"{value:.12g}"


def _absolutize_spectre_includes(text: str, repo_root: Path) -> str:
    """Replace repo-relative includes with an absolute Verilog-A path.

    Rendered netlists run from ``<output>/logs/netlists/``; relative paths break.
    """
    va_path = (repo_root / "veriloga/configurable_opamp.va").resolve()
    return _SCS_INCLUDE.sub(f'ahdl_include "{va_path}"\n', text)


def render_spectre_ac_netlist(
    template_path: Path,
    cfg: OpampConfig,
    *,
    repo_root: Path | None = None,
) -> str:
    """Render a Spectre AC netlist with CLI macromodel settings."""
    root = repo_root or package_root()
    text = template_path.read_text(encoding="utf-8")
    overrides = {
        "a0_db": _format_param(cfg.a0_db),
        "gbw_hz": _format_param(cfg.gbw_hz),
        "rin_ohm": _format_param(cfg.rin_ohm),
        "cin_f": _format_param(cfg.cin_f),
        "rout_ohm": _format_param(cfg.rout_ohm),
        "cout_f": _format_param(cfg.cout_f),
        "f_start": _format_param(cfg.sweep.f_start_hz),
        "f_stop": _format_param(cfg.sweep.f_stop_hz),
        "dec": _format_param(cfg.sweep.points_per_decade),
    }
    for name, value in overrides.items():
        text = re.sub(
            rf"^parameters\s+{name}=.*$",
            f"parameters {name}={value}",
            text,
            flags=re.MULTILINE,
        )
    return _absolutize_spectre_includes(text, root)


def _noise_en_flicker_1hz(noise: OpampNoiseConfig) -> float:
    return float(
        getattr(
            noise,
            "en_flicker_1hz_v_per_sqrt_hz",
            getattr(noise, "en_flicker_at_1hz_v_per_sqrt_hz", 0.0),
        )
    )


def _noise_en_flicker_ef(noise: OpampNoiseConfig) -> float:
    return float(getattr(noise, "en_flicker_ef", 1.0))


def _noise_kf(noise: OpampNoiseConfig) -> float:
    return float(getattr(noise, "kf", 0.0))


def _noise_af(noise: OpampNoiseConfig) -> float:
    return float(getattr(noise, "af", 1.0))


def _noise_bias_current_a(noise: OpampNoiseConfig) -> float:
    return float(getattr(noise, "bias_current_a", 0.0))


def _va_intrinsic_white(noise: OpampNoiseConfig) -> bool:
    return noise.enabled and noise.en_white_v_per_sqrt_hz > 0.0


def _va_intrinsic_flicker(noise: OpampNoiseConfig) -> bool:
    if not noise.enabled:
        return False
    if _noise_en_flicker_1hz(noise) > 0.0:
        return True
    return _noise_kf(noise) > 0.0 and abs(_noise_bias_current_a(noise)) > 0.0


def render_spectre_noise_netlist(
    template_path: Path,
    cfg: OpampConfig,
    noise: OpampNoiseConfig,
    *,
    repo_root: Path | None = None,
) -> str:
    """Render a Spectre open-loop noise netlist (``noise`` + ``ac`` analyses)."""
    root = repo_root or package_root()
    text = template_path.read_text(encoding="utf-8")
    en_white = noise.en_white_v_per_sqrt_hz if noise.enabled else 0.0
    en_flicker_1hz = _noise_en_flicker_1hz(noise) if noise.enabled else 0.0
    kT = 4.0 * 1.380649e-23 * 300.0
    rn_fallback = (en_white * en_white) / kT if en_white > 0.0 else 1.0e-6
    # VA white_noise replaces RN when intrinsic noise is enabled.
    rn_white = 1.0e12 if _va_intrinsic_white(noise) else rn_fallback
    enable_noise = 1 if noise.enabled else 0
    overrides = {
        "a0_db": _format_param(cfg.a0_db),
        "gbw_hz": _format_param(cfg.gbw_hz),
        "rin_ohm": _format_param(cfg.rin_ohm),
        "cin_f": _format_param(cfg.cin_f),
        "rout_ohm": _format_param(cfg.rout_ohm),
        "cout_f": _format_param(cfg.cout_f),
        "en_white_v_per_sqrt_hz": _format_param(en_white),
        "rn_white_ohm": _format_param(rn_white),
        "en_flicker_1hz_v_per_sqrt_hz": _format_param(en_flicker_1hz),
        "en_flicker_ef": _format_param(_noise_en_flicker_ef(noise)),
        "kf": _format_param(_noise_kf(noise)),
        "af": _format_param(_noise_af(noise)),
        "bias_current_a": _format_param(_noise_bias_current_a(noise)),
        "enable_noise": _format_param(enable_noise),
        "en_flicker_corner_hz": _format_param(
            getattr(noise, "en_flicker_corner_hz", float("nan"))
        ),
        "en_flicker_at_1hz_v_per_sqrt_hz": _format_param(en_flicker_1hz),
        "f_start": _format_param(cfg.sweep.f_start_hz),
        "f_stop": _format_param(cfg.sweep.f_stop_hz),
        "dec": _format_param(cfg.sweep.points_per_decade),
    }
    for name, value in overrides.items():
        text = re.sub(
            rf"^parameters\s+{name}=.*$",
            f"parameters {name}={value}",
            text,
            flags=re.MULTILINE,
        )
    text = text.replace("PLACEHOLDER_RN", _format_param(rn_white))
    return _absolutize_spectre_includes(text, root)


def run_spectre_netlist(
    netlist: Path,
    *,
    cwd: Path | None = None,
    timeout_s: float | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run Spectre on a rendered ``.scs`` netlist."""
    executable = find_spectre_executable()
    args = [executable, str(netlist), "+log", "status"]
    return subprocess.run(
        args,
        cwd=cwd or netlist.parent,
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout_s,
        env=_spectre_subprocess_env(),
    )


@dataclass(frozen=True)
class SpectreAcResult:
    """Artifacts from a Spectre AC run."""

    netlist_path: Path
    log_path: Path
    raw_dir: Path
    csv_path: Path


def run_spectre_ac(
    cfg: OpampConfig,
    output_dir: Path,
    *,
    template_name: str = "ac_open_loop.scs",
) -> SpectreAcResult:
    """Render and execute a Spectre AC netlist; export Bode CSV from PSF."""
    repo = package_root()
    template = repo / "testbench" / "spectre" / template_name
    logs_dir = output_dir / "logs" / "netlists"
    logs_dir.mkdir(parents=True, exist_ok=True)
    netlist_path = logs_dir / template_name
    netlist_path.write_text(
        render_spectre_ac_netlist(template, cfg, repo_root=repo),
        encoding="utf-8",
    )
    stem = template_name.replace(".scs", "")
    log_path = output_dir / "logs" / f"spectre_{stem}.log"
    csv_path = output_dir / ("stb_bode.csv" if stem == "stb_loop" else "ac_bode.csv")
    completed = run_spectre_netlist(netlist_path, cwd=logs_dir)
    log_path.write_text(completed.stdout + completed.stderr, encoding="utf-8")
    _raise_if_spectre_failed(
        completed,
        log_path=log_path,
        message="Spectre failed",
    )

    try:
        bode = read_spectre_ac_from_netlists(logs_dir, stem=stem, signal="out")
    except SpectrePsfError as exc:
        msg = f"Spectre AC PSF read failed: {exc}"
        raise RuntimeError(msg) from exc

    write_bode_csv(
        csv_path,
        bode["frequency_hz"],
        bode["gain_db"],
        bode["phase_deg"],
    )
    raw_dir = locate_ac_psf(logs_dir, stem=stem).raw_dir
    return SpectreAcResult(
        netlist_path=netlist_path,
        log_path=log_path,
        raw_dir=raw_dir,
        csv_path=csv_path,
    )


def spectre_ac_to_simulation_result(
    frequency_hz: NDArray[np.float64],
    gain_db: NDArray[np.float64],
    phase_deg: NDArray[np.float64],
) -> AcSimulationResult:
    """Build ``AcSimulationResult`` from Spectre AC PSF columns."""
    metrics = extract_gbw_phase_margin(frequency_hz, gain_db, phase_deg)
    return AcSimulationResult(
        frequency_hz=frequency_hz,
        gain_db=gain_db,
        phase_deg=phase_deg,
        metrics=metrics,
    )


def simulate_ac_spectre(
    cfg: OpampConfig,
    output_dir: Path,
    noise: OpampNoiseConfig | None = None,
) -> AcSimulationResult:
    """Run open-loop AC in Spectre and return Bode data parsed from PSF."""
    _ = noise
    artifacts = run_spectre_ac(cfg, output_dir)
    bode = read_spectre_ac_from_netlists(
        artifacts.netlist_path.parent,
        stem=artifacts.netlist_path.stem,
        signal="out",
    )
    return spectre_ac_to_simulation_result(
        bode["frequency_hz"],
        bode["gain_db"],
        bode["phase_deg"],
    )


@dataclass(frozen=True)
class SpectreNoiseResult:
    """Artifacts from a Spectre noise analysis run."""

    netlist_path: Path
    log_path: Path
    raw_dir: Path


def run_spectre_noise(
    cfg: OpampConfig,
    noise: OpampNoiseConfig,
    output_dir: Path,
    *,
    template_name: str = "noise_open_loop.scs",
) -> SpectreNoiseResult:
    """Render and execute a Spectre open-loop noise netlist."""
    repo = package_root()
    template = repo / "testbench" / "spectre" / template_name
    logs_dir = output_dir / "logs" / "netlists"
    logs_dir.mkdir(parents=True, exist_ok=True)
    netlist_path = logs_dir / template_name
    netlist_path.write_text(
        render_spectre_noise_netlist(template, cfg, noise, repo_root=repo),
        encoding="utf-8",
    )
    stem = template_name.replace(".scs", "")
    log_path = output_dir / "logs" / f"spectre_{stem}.log"
    completed = run_spectre_netlist(netlist_path, cwd=logs_dir)
    log_path.write_text(completed.stdout + completed.stderr, encoding="utf-8")
    _raise_if_spectre_failed(
        completed,
        log_path=log_path,
        message="Spectre noise failed",
    )
    raw_dir = locate_noise_psf(logs_dir, stem=stem).raw_dir
    return SpectreNoiseResult(
        netlist_path=netlist_path,
        log_path=log_path,
        raw_dir=raw_dir,
    )


def simulate_noise_spectre(
    cfg: OpampConfig,
    output_dir: Path,
    noise: OpampNoiseConfig,
) -> NoiseSimulationResult:
    """Run Spectre noise bench; output spectrum from AC gain × input-referred noise.

    Spectre ``.noise`` on this bench does not propagate Verilog-A ``white_noise`` /
    ``flicker_noise`` through ``laplace_nd`` (PSF output noise is near zero). The
    same post-processing as ngspice is used: open-loop AC gain from the companion
    ``ac`` analysis times ``OpampNoiseConfig`` input-referred density.
    """
    artifacts = run_spectre_noise(cfg, noise, output_dir)
    _ = artifacts
    logs_dir = output_dir / "logs" / "netlists"
    try:
        ac = read_spectre_ac_from_netlists(logs_dir, stem="noise_open_loop", signal="out")
    except SpectrePsfError as exc:
        msg = f"Spectre noise AC PSF read failed: {exc}"
        raise RuntimeError(msg) from exc
    frequency_hz = log_frequency_sweep(
        cfg.sweep.f_start_hz,
        cfg.sweep.f_stop_hz,
        cfg.sweep.points_per_decade,
    )
    spectrum = ngspice_output_noise_spectrum(
        cfg,
        noise,
        frequency_hz=frequency_hz,
        gain_db=ac["gain_db"],
        ac_frequency_hz=ac["frequency_hz"],
    )
    metrics = extract_noise_metrics(cfg, noise, frequency_hz, spectrum)
    return NoiseSimulationResult(
        frequency_hz=frequency_hz,
        noise_v_per_sqrt_hz=spectrum,
        metrics=metrics,
    )


def run_spectre_tran_stub(
    cfg: OpampConfig,
    output_dir: Path,
    *,
    template_name: str,
    log_name: str,
) -> Path:
    """Run a minimal Spectre TRAN stub (``dc`` only) to validate the toolchain."""
    repo = package_root()
    template = repo / "testbench" / "spectre" / template_name
    logs_dir = output_dir / "logs" / "netlists"
    logs_dir.mkdir(parents=True, exist_ok=True)
    netlist_path = logs_dir / template_name
    netlist_path.write_text(template.read_text(encoding="utf-8"), encoding="utf-8")
    log_path = output_dir / "logs" / log_name
    completed = run_spectre_netlist(netlist_path, cwd=logs_dir)
    log_path.write_text(completed.stdout + completed.stderr, encoding="utf-8")
    _raise_if_spectre_failed(
        completed,
        log_path=log_path,
        message="Spectre TRAN stub failed",
    )
    _ = cfg
    return log_path
