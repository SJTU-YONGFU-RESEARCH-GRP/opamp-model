"""Verilog-A CMRR/PSRR compile and source-structure tests."""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
from pathlib import Path

import pytest

from opamp_model.config import OpampConfig
from opamp_model.io import package_root
from opamp_model.spectre_engine import (
    SpectreNotFoundError,
    _absolutize_spectre_includes,
    find_spectre_executable,
    render_spectre_ac_netlist,
)

_SCS_INCLUDE = re.compile(
    r'include\s+"\./testbench/spectre/opamp_include\.scs"\s*',
    re.IGNORECASE,
)


def _va_path() -> Path:
    return package_root() / "veriloga/configurable_opamp.va"


def _render_spectre_template(template_name: str, cfg: OpampConfig | None = None) -> str:
    root = package_root()
    text = (root / "testbench" / "spectre" / template_name).read_text(encoding="utf-8")
    if cfg is not None and template_name == "ac_open_loop.scs":
        return render_spectre_ac_netlist(
            root / "testbench" / "spectre" / template_name,
            cfg,
            repo_root=root,
        )
    va_path = (root / "veriloga/configurable_opamp.va").resolve()
    return _SCS_INCLUDE.sub(f'ahdl_include "{va_path}"\n', text)


def _spectre_executable() -> str | None:
    try:
        return find_spectre_executable()
    except SpectreNotFoundError:
        pass
    for candidate in (
        "/eda/cadence/SPECTRE241/bin/spectre",
        "/tools/cadence/SPECTRE241/bin/spectre",
    ):
        if Path(candidate).is_file():
            return candidate
    return None


def _run_spectre_compile(netlist_text: str, *, label: str) -> subprocess.CompletedProcess[str]:
    executable = _spectre_executable()
    if executable is None:
        pytest.skip("Cadence Spectre not available")
    with tempfile.TemporaryDirectory(prefix=f"va_cm_ps_{label}_") as tmp:
        work = Path(tmp)
        netlist = work / f"{label}.scs"
        netlist.write_text(netlist_text, encoding="utf-8")
        env = os.environ.copy()
        spectre_bin = str(Path(executable).parent)
        env["PATH"] = f"{spectre_bin}:{env.get('PATH', '')}"
        if not env.get("CDS_LIC_FILE"):
            default_lic = Path("/eda/cadence/license.dat")
            if default_lic.is_file():
                env["CDS_LIC_FILE"] = str(default_lic)
        return subprocess.run(
            [executable, str(netlist), "+log", "status"],
            cwd=work,
            capture_output=True,
            text=True,
            check=False,
            timeout=120.0,
            env=env,
        )


def _assert_spectre_compile_ok(completed: subprocess.CompletedProcess[str]) -> None:
    """Require AHDL compile success; skip when Spectre license is unavailable."""
    output = completed.stdout + completed.stderr
    if "LMC-01902" in output or "Failed to initialize license API" in output:
        pytest.skip("Cadence Spectre license unavailable (LMC-01902)")
    assert completed.returncode == 0, output


def test_veriloga_declares_cm_ps_parameters() -> None:
    """VA module exposes CMRR_DB, PSRR_DB, PSRR_POLE_HZ matching OpampConfig."""
    va = _va_path().read_text(encoding="utf-8")
    assert "parameter real CMRR_DB" in va
    assert "parameter real PSRR_DB" in va
    assert "parameter real PSRR_POLE_HZ" in va
    assert "CMRR_DB" in va
    assert "instance_parameter_list" in va


def test_veriloga_laplace_uses_array_variables_for_cm_ps() -> None:
    """Spectre laplace_nd coeffs for diff, CM, and PS paths use array variables."""
    va = _va_path().read_text(encoding="utf-8")
    assert "real numer[0:1]" in va
    assert "real cm_numer[0:1]" in va
    assert "real ps_numer[0:1]" in va
    assert "laplace_nd(V(inp) - V(inn), numer, denom)" in va
    assert "laplace_nd(0.5 * (V(inp) + V(inn)), cm_numer, cm_denom)" in va
    assert "laplace_nd(V(vdd), ps_numer, ps_denom)" in va
    assert "cm_numer[0] = numer[0] / cmrr_linear" in va
    assert "ps_numer[0] = (1.0 / psrr_linear) * wp_psrr" in va
    assert "{a0_linear * wp" not in va


def test_veriloga_common_mode_voltage_definition() -> None:
    """Common-mode stimulus is half-sum of input pins in the CM laplace path."""
    va = _va_path().read_text(encoding="utf-8")
    assert "0.5 * (V(inp) + V(inn))" in va


def test_psrr_netlist_wires_cm_ps_parameters() -> None:
    """PSRR Spectre deck passes CMRR/PSRR parameters into the op-amp instance."""
    text = _render_spectre_template("psrr.scs")
    assert "CMRR_DB=cmrr_db" in text
    assert "PSRR_DB=psrr_db" in text
    assert "PSRR_POLE_HZ=psrr_pole_hz" in text
    assert "mag=1" in text


def test_spectre_compiles_ac_open_loop_with_cm_ps() -> None:
    """Spectre AHDL compile succeeds for AC netlist with CMRR/PSRR parameters."""
    cfg = OpampConfig(a0_db=70.0, gbw_hz=2.0e6, cmrr_db=85.0, psrr_db=75.0)
    netlist = _render_spectre_template("ac_open_loop.scs", cfg)
    completed = _run_spectre_compile(netlist, label="ac_open_loop")
    _assert_spectre_compile_ok(completed)


def test_spectre_compiles_psrr_netlist() -> None:
    """Spectre AHDL compile succeeds for PSRR supply-AC netlist."""
    netlist = _absolutize_spectre_includes(
        (package_root() / "testbench/spectre/psrr.scs").read_text(encoding="utf-8"),
        package_root(),
    )
    completed = _run_spectre_compile(netlist, label="psrr")
    _assert_spectre_compile_ok(completed)
