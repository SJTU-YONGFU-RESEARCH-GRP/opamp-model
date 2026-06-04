# Op-amp behavioral model reference

Deep modeling reference for **opamp-model**.

## Related documentation

- [bench_spec.md](bench_spec.md) — bench definitions
- [metrics_catalog.md](metrics_catalog.md) — metric index
- [golden_metrics.yaml](golden_metrics.yaml) — optional reference targets

## Ideal small-signal VOA

### Open-loop transfer

Unity-gain frequency ``f_ug = gbw_hz`` (Hz). Dominant pole:

\[
\omega_p = \frac{2\pi \cdot \mathrm{GBW}}{A_0}
\qquad
H(s) = \frac{A_0}{1 + s/\omega_p}
\]

Optional second pole ``fp2_hz`` and zero ``fz_hz`` multiply the response in Python
(``src/opamp_model/core.py``). Verilog-A implements the one-pole core via ``laplace_nd``.

### STB / loop gain

\[
L(s) = \beta \cdot H(s)
\]

``loop_beta`` defaults to ``1.0`` (unity feedback). Metrics: GBW (0 dB crossover),
phase margin ``PM = 180° + \angle L(j\omega_{GBW})``.

### Engines (peer implementations)

All engines implement the **same** equations in this document; none is authoritative over the others.

| Engine | Realization | Metrics source (target) |
| --- | --- | --- |
| Python | Closed-form / numerical in ``src/opamp_model/`` | Python arrays |
| ngspice | SPICE netlist macromodel (``testbench/ngspice/``) | Parsed ``wrdata`` / noise output |
| Spectre | Verilog-A ``configurable_opamp.va`` + ``testbench/spectre/`` | PSF / exported simulation data |

**Scaffolding note:** Until PSF and full ngspice parsers land, some benches may still copy Python curves after a stub run. That violates the peer-engine rule and should be removed, not documented as “golden Python.”

## CMRR and PSRR

### Common-mode gain

\[
\mathrm{ACM}(s) = \frac{A_{\mathrm{ol}}(s)}{\mathrm{CMRR}_{\mathrm{linear}}}
\]

``CMRR_linear = 10^{\mathrm{cmrr\_db}/20}``. With constant CMRR, ``\mathrm{CMRR}(f) = \mathrm{cmrr\_db}`` (dB) and ``\mathrm{ACM}(f) = A_{\mathrm{ol}}(f) - \mathrm{cmrr\_db}`` in dB.

### PSRR feedthrough

Supply ripple to output:

\[
H_{\mathrm{ps}}(s) = \frac{1/\mathrm{PSRR}_{\mathrm{linear}}}{1 + s/\omega_{p,\mathrm{psrr}}}
\qquad
\omega_{p,\mathrm{psrr}} = 2\pi \cdot \mathrm{psrr\_pole\_hz}
\]

Reported PSRR in dB: ``\mathrm{PSRR}(f) = -20\log_{10}|H_{\mathrm{ps}}(j\omega)|``.

Implementation: ``src/opamp_model/cm_ps.py``; benches ``run_ac.py`` (CMRR curve) and ``run_psrr.py``.

## Planned sections (future work)

1. Overview — VOA, TIA, Gm
2. Signal chain and processing order
3. Small-signal: \(A(s)\), GBW, phase margin, CMRR, PSRR
4. Impedance: \(Z_{in}(s)\), \(Z_{out}(s)\)
5. Noise: white, flicker, shot; input vs output referred
6. Large-signal: slew, clipping, weak nonlinearity
7. TIA closed-loop: \(Z_t(s)\)
8. Gm: \(I_{out} = g_m V_{diff}\)
9. Testbench catalog — see [bench_spec.md](bench_spec.md)
10. Multi-engine architecture and known discrepancies
11. Python API quick reference

## Authoritative Verilog-A (shells)

| Module | Path | Status |
| --- | --- | --- |
| Voltage op-amp | [../veriloga/configurable_opamp.va](../veriloga/configurable_opamp.va) | Parameter shell |
| TIA | [../veriloga/configurable_tia.va](../veriloga/configurable_tia.va) | Parameter shell |
| Gm / OTA | [../veriloga/configurable_gm.va](../veriloga/configurable_gm.va) | Parameter shell |

## Python package

Implementation lives in `src/opamp_model/`. Configuration dataclasses are defined in [../src/opamp_model/config.py](../src/opamp_model/config.py).

Simulation entry points (`simulate_ac`, `simulate_stb`, …) are in `src/opamp_model/model.py`.
