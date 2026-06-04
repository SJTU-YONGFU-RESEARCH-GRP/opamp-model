# ngspice testbenches

| Netlist | Script | Description |
| --- | --- | --- |
| `ac_open_loop.cir` | `run_ac.py --ngspice` | Open-loop AC via RC-pole + VCVS macromodel |
| `stb_loop.cir` | `run_stb.py --ngspice` | Same as AC; ``loop_beta`` applied in Python |

Phase 1 exports Bode data with ``wrdata`` (frequency, ``vdb(out)``, ``vp(out)``).
