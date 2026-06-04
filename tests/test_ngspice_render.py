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
