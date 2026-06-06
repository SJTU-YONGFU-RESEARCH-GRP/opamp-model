"""ngspice netlist rendering tests (no simulator required)."""

from __future__ import annotations

import math

from opamp_model.config import OpampConfig
from opamp_model.core import dominant_pole_rad_s
from opamp_model.io import package_root
from opamp_model.ngspice_engine import render_ngspice_ac_netlist


def test_render_ac_netlist_one_pole_rc() -> None:
    """Rendered AC netlist places Rlp*Clp = 1/wp and documents laplace_nd equivalence."""
    cfg = OpampConfig(a0_db=70.0, gbw_hz=2.0e6, fp2_hz=0.0)
    template = package_root() / "testbench" / "ngspice" / "ac_open_loop.cir"
    text = render_ngspice_ac_netlist(template, cfg)
    a0 = max(cfg.a0_linear, 1.0)
    wp = dominant_pole_rad_s(cfg)
    assert "laplace_nd" in text
    assert f".param a0_linear={a0}" in text or f".param a0_linear={a0:.12g}" in text
    assert f".param wp_rad_s={wp}" in text or f".param wp_rad_s={wp:.12g}" in text
    assert "Rlp*Clp = 1/wp" in text
    for line in text.splitlines():
        if line.startswith(".param rlp_ohm="):
            rlp = float(line.split("=", 1)[1])
        if line.startswith(".param clp_f="):
            clp = float(line.split("=", 1)[1])
    assert math.isclose(rlp * clp, 1.0 / wp, rel_tol=1.0e-9)
    assert "Efp2byp n3 0 n2 0 1" in text
    assert "Efzbyp n4 0 n3 0 1" in text


def test_render_ac_netlist_fp2_fz_stages() -> None:
    """Rendered AC netlist includes second-pole RC and zero B-source when configured."""
    cfg = OpampConfig(a0_db=70.0, gbw_hz=2.0e6, fp2_hz=50.0e6, fz_hz=10.0e6)
    template = package_root() / "testbench" / "ngspice" / "ac_open_loop.cir"
    text = render_ngspice_ac_netlist(template, cfg)
    wp2 = 2.0 * math.pi * cfg.fp2_hz
    wz = 2.0 * math.pi * cfg.fz_hz
    assert "Rfp2 n2 n3" in text
    assert "Cfp2 n3 0" in text
    assert f"laplace(V(n3), {{{wz}" in text or f"laplace(V(n3), {{{wz:.12g}" in text
    for line in text.splitlines():
        if line.startswith("Rfp2 n2 n3"):
            rfp2 = float(line.split()[-1])
        if line.startswith("Cfp2 n3 0"):
            cfp2 = float(line.split()[-1])
    assert math.isclose(rfp2 * cfp2, 1.0 / wp2, rel_tol=1.0e-9)


def test_render_cm_netlist_acm_gain() -> None:
    """CM netlist scales VCVS gain by A0/CMRR and reuses fp2/fz blocks."""
    from opamp_model.ngspice_engine import render_ngspice_cm_netlist

    cfg = OpampConfig(a0_db=60.0, gbw_hz=1.0e6, cmrr_db=80.0, fp2_hz=0.0)
    template = package_root() / "testbench" / "ngspice" / "ac_cm.cir"
    text = render_ngspice_cm_netlist(template, cfg)
    acm_gain = max(cfg.a0_linear, 1.0) / max(cfg.cmrr_linear, 1.0)
    assert f".param acm_gain={acm_gain}" in text or f".param acm_gain={acm_gain:.12g}" in text
    assert "Eaux n1 0 inp 0 {acm_gain}" in text
    assert "Elink inp inn inn 0 1" in text
