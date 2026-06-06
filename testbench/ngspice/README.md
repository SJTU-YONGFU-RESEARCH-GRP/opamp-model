# ngspice testbenches

| Netlist | Script | Description |
| --- | --- | --- |
| `ac_open_loop.cir` | `run_ac.py --ngspice` | One-pole open-loop AC (VCVS + RC ≡ `laplace_nd`) |
| `stb_loop.cir` | `run_stb.py --ngspice` | Same macromodel; ``loop_beta`` applied in Python |
| `psrr.cir` | `run_psrr.py --ngspice` | Supply AC on VDD; single-pole PSRR feedthrough (``vm(out)``) |
| `slew_stub.cir` | `run_slew.py --ngspice` | TRAN toolchain stub (``.op``); step data from Python |
| `thd_stub.cir` | `run_thd.py --ngspice` | TRAN toolchain stub (``.op``); THD data from Python |

The dominant pole follows ``wp = 2π·GBW/A0`` with ``Rlp·Clp = 1/wp`` (see netlist comments).
Optional ``fp2_hz`` / ``fz_hz`` add a second RC pole and B-source zero (see ``render_ngspice_ac_netlist``).
Common-mode ACM is measured via ``ac_cm.cir`` for CMRR extraction in ``run_ac.py``.

AC/STB runs export Bode data with ``wrdata`` (frequency, ``vdb(out)``, ``vp(out)``).
