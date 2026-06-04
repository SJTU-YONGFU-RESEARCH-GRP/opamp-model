# Spectre testbenches

Netlists (`.scs`) cover AC, STB, noise, PSRR, and stubs for transient, TIA, and Gm benches.

Requirements:

- Cadence Spectre with Verilog-A (AHDL)
- `spectre` on `PATH`

The Python driver renders netlists under `<output-dir>/logs/netlists/` and rewrites
the template `include` to an absolute `ahdl_include` for `veriloga/configurable_opamp.va`
(so Spectre can run from the output folder).

No external Cadence library cells are required for the behavioral flow.
