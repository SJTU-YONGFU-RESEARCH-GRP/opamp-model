# Spectre testbenches

Netlists (`.scs`) cover AC, STB, noise, PSRR (supply AC + PSF parse), and stubs for transient (``slew_stub.scs``,
``thd_stub.scs`` use ``dc`` only; waveforms from Python), TIA, and Gm benches.

Requirements:

- Cadence Spectre with Verilog-A (AHDL)
- `spectre` on `PATH`

The Python driver renders netlists under `<output-dir>/logs/netlists/` and rewrites
the template `include` to an absolute `ahdl_include` for `veriloga/configurable_opamp.va`
(so Spectre can run from the output folder).

No external Cadence library cells are required for the behavioral flow.
