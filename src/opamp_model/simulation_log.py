"""Simulation logging and Verilog-A artifact archival."""

from __future__ import annotations

import shutil
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO

from opamp_model.config import OpampConfig, OpampNoiseConfig
from opamp_model.io import package_root


@dataclass(frozen=True)
class SimulationLogPaths:
    """Paths to simulation logs and archived model files."""

    logs_dir: Path
    veriloga_dir: Path
    ngspice_dir: Path
    spectre_dir: Path
    veriloga_opamp: Path


def prepare_output_dirs(output_dir: Path) -> SimulationLogPaths:
    """Create log and artifact directories under the output folder."""
    logs_dir = output_dir / "logs"
    veriloga_dir = output_dir / "veriloga"
    ngspice_dir = output_dir / "ngspice"
    spectre_dir = output_dir / "spectre"
    for path in (logs_dir, veriloga_dir, ngspice_dir, spectre_dir):
        path.mkdir(parents=True, exist_ok=True)
    return SimulationLogPaths(
        logs_dir=logs_dir,
        veriloga_dir=veriloga_dir,
        ngspice_dir=ngspice_dir,
        spectre_dir=spectre_dir,
        veriloga_opamp=veriloga_dir / "configurable_opamp.va",
    )


def archive_veriloga_artifacts(repo_root: Path, output_dir: Path) -> Path:
    """Copy Verilog-A shells into the output folder for reproducibility."""
    paths = prepare_output_dirs(output_dir)
    sources = [
        repo_root / "veriloga/configurable_opamp.va",
        repo_root / "veriloga/configurable_tia.va",
        repo_root / "veriloga/configurable_gm.va",
    ]
    for src in sources:
        if src.is_file():
            shutil.copy2(src, paths.veriloga_dir / src.name)
    return paths.veriloga_dir


class SimulationLog:
    """Append-only log file for a single bench run."""

    def __init__(self, path: Path) -> None:
        """Open or create a log at ``path``."""
        path.parent.mkdir(parents=True, exist_ok=True)
        self._path = path
        self._file: TextIO = path.open("a", encoding="utf-8")

    def write_header(
        self,
        bench: str,
        engine: str,
        cfg: OpampConfig,
        noise: OpampNoiseConfig,
    ) -> None:
        """Write run metadata and configuration snapshot."""
        stamp = datetime.now(UTC).isoformat()
        self._file.write(f"# bench={bench} engine={engine} utc={stamp}\n")
        self._file.write(f"# a0_db={cfg.a0_db} gbw_hz={cfg.gbw_hz} cmrr_db={cfg.cmrr_db}\n")
        self._file.write(f"# noise_enabled={noise.enabled} seed={noise.noise_seed}\n")
        self._file.flush()

    def write(self, message: str) -> None:
        """Append one line to the log."""
        self._file.write(message.rstrip() + "\n")
        self._file.flush()

    def close(self) -> None:
        """Close the underlying file."""
        self._file.close()


def tee_stdout_to_log(log_path: Path) -> SimulationLog:
    """Return a log handle; callers may redirect stdout when needed."""
    return SimulationLog(log_path)


def log_run_context(
    log: SimulationLog,
    *,
    argv: list[str] | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """Record command line and optional key-value context."""
    log.write(f"argv={' '.join(argv or sys.argv)}")
    if extra:
        for key, value in extra.items():
            log.write(f"{key}={value!r}")


def default_repo_root() -> Path:
    """Return package root for scripts executed from any working directory."""
    return package_root()
