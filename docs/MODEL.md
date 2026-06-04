# Op-amp behavioral model reference

Deep modeling reference for **opamp-model**.

## Related documentation

- [bench_spec.md](bench_spec.md) — bench definitions and outputs
- [metrics_catalog.md](metrics_catalog.md) — metric index and status
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
(``src/opamp_model/core.py``). Verilog-A implements the one-pole core via ``laplace_nd`` in
[../veriloga/configurable_opamp.va](../veriloga/configurable_opamp.va).

### STB / loop gain

\[
L(s) = \beta \cdot H(s)
\]

``loop_beta`` defaults to ``1.0`` (unity feedback). Metrics: GBW (0 dB crossover),
phase margin ``PM = 180° + \angle L(j\omega_{GBW})``, gain margin at the −180° phase crossing.

### Impedance (passive loading)

Differential input and output impedances are **not** folded into ``H(s)``; they are separate:

\[
Z_{in}(s) = R_{in} \,\|\, \frac{1}{s C_{in}}
\qquad
Z_{out}(s) = R_{out} \,\|\, \frac{1}{s C_{out}}
\]

Implementation: ``src/opamp_model/impedance.py``. ``run_ac.py`` plots ``impedance.svg`` and reports
``|Zin|``, ``|Zout|`` at 1 Hz in ``opamp_metrics.json``.

### CMRR and PSRR

#### Common-mode gain

\[
\mathrm{ACM}(s) = \frac{A_{\mathrm{ol}}(s)}{\mathrm{CMRR}_{\mathrm{linear}}}
\]

``CMRR_linear = 10^{\mathrm{cmrr\_db}/20}``. With constant CMRR, ``\mathrm{CMRR}(f) = \mathrm{cmrr\_db}`` (dB) and ``\mathrm{ACM}(f) = A_{\mathrm{ol}}(f) - \mathrm{cmrr\_db}`` in dB.

#### PSRR feedthrough

Supply ripple to output:

\[
H_{\mathrm{ps}}(s) = \frac{1/\mathrm{PSRR}_{\mathrm{linear}}}{1 + s/\omega_{p,\mathrm{psrr}}}
\qquad
\omega_{p,\mathrm{psrr}} = 2\pi \cdot \mathrm{psrr\_pole\_hz}
\]

Reported PSRR in dB: ``\mathrm{PSRR}(f) = -20\log_{10}|H_{\mathrm{ps}}(j\omega)|``.

Implementation: ``src/opamp_model/cm_ps.py``; benches ``run_ac.py`` (CMRR curve) and ``run_psrr.py``.

### Noise

Input-referred voltage noise density (white + flicker):

\[
e_n(f) = \sqrt{e_{n,\mathrm{white}}^2 + e_{n,\mathrm{flicker@1Hz}}^2 \cdot \frac{f_c}{f}}
\]

Output-referred spectrum scales with open-loop magnitude: ``e_{out}(f) = e_n(f) \cdot |A_{\mathrm{ol}}(j\omega)|``.

Implementation: ``src/opamp_model/noise.py``, ``src/opamp_model/noise_analysis.py``; bench ``run_noise.py``.

### Large-signal (Python TRAN)

| Mechanism | Model | Module |
| --- | --- | --- |
| Slew rate | ``slew_pos_vps`` / ``slew_neg_vps`` limit on unity-gain step | ``src/opamp_model/tran.py`` |
| Weak nonlinearity | ``nl_a2``, ``nl_a3`` on sinusoidal steady state | ``src/opamp_model/tran.py`` |
| Supply / swing | ``vswing_high_v``, ``vswing_low_v``, ``vdd_v``, ``vss_v`` | ``OpampConfig`` |

Benches: ``run_slew.py``, ``run_thd.py`` (Python engine only in batch runs).

## Multi-engine architecture

All engines must implement the **same** equations in this document; none is authoritative over the others.

```text
  docs/MODEL.md (equations)
        │
        ├── Python (src/opamp_model/)
        ├── ngspice (testbench/ngspice/*.cir)
        └── Spectre (veriloga/configurable_opamp.va + testbench/spectre/*.scs)
        │
        ▼
  scripts/run_*.py  →  outputs/<engine>/
```

### Engine parity (current)

| Bench | Python | ngspice | Spectre |
| --- | --- | --- | --- |
| AC open-loop | Macromodel | ``ac_open_loop.cir`` → ``wrdata`` parsed | ``ac_open_loop.scs`` runs; Bode from Python until PSF parser |
| STB loop gain | ``beta * A_open`` | Same AC netlist; ``loop_beta`` applied in ``run_stb.py`` | Same as AC (Python Bode today) |
| Noise | Macromodel spectrum | ``noise_stub.cir`` runs; spectrum from Python | ``noise_stub.scs`` runs; spectrum from Python |
| PSRR | Macromodel | Python only (``psrr.scs`` stub) | Python only (``psrr.scs`` stub) |
| Slew / THD | TRAN macromodel | Not wired | Not wired |

**Scaffolding (temporary):** Spectre AC/STB and ngspice/Spectre noise substitute Python curves after the netlist run. This is not peer-engine behavior and must be replaced by PSF / ngspice noise parsers — do not treat Python as golden for those engines.

``docs/golden_metrics.yaml`` holds optional transistor-level reference targets for future ``compare_engines.py``; it is **not** an engine truth file.

## Verilog-A modules

| Module | Path | Status |
| --- | --- | --- |
| Voltage op-amp | [../veriloga/configurable_opamp.va](../veriloga/configurable_opamp.va) | Dominant-pole ``laplace_nd``, ``Rin‖Cin``, ``Rout‖Cout`` |
| TIA | [../veriloga/configurable_tia.va](../veriloga/configurable_tia.va) | Parameter shell (no closed-loop yet) |
| Gm / OTA | [../veriloga/configurable_gm.va](../veriloga/configurable_gm.va) | Parameter shell |

CMRR, PSRR, noise, slew, and weak nonlinearity are **not** in Verilog-A yet; Python implements them today.

## Python package

| Area | Module |
| --- | --- |
| Configuration | ``src/opamp_model/config.py`` — ``OpampConfig``, ``OpampNoiseConfig``, ``BenchSweepConfig``, ``TiaConfig``, ``GmConfig`` |
| Open-loop / loop | ``src/opamp_model/core.py``, ``src/opamp_model/model.py`` |
| CMRR / PSRR | ``src/opamp_model/cm_ps.py`` |
| Impedance | ``src/opamp_model/impedance.py`` |
| Noise | ``src/opamp_model/noise.py``, ``src/opamp_model/noise_analysis.py`` |
| TRAN | ``src/opamp_model/tran.py`` |
| Metrics / reports | ``src/opamp_model/metrics.py``, ``src/opamp_model/report.py`` |
| ngspice / Spectre | ``src/opamp_model/ngspice_engine.py``, ``src/opamp_model/spectre_engine.py`` |

### API quick reference

```python
from opamp_model import OpampConfig, OpampNoiseConfig, simulate_ac, simulate_stb
from opamp_model import simulate_cmrr, simulate_psrr, simulate_noise

cfg = OpampConfig(a0_db=80.0, gbw_hz=10e6)
noise = OpampNoiseConfig()

ac = simulate_ac(cfg, noise)
stb = simulate_stb(cfg, noise)
cmrr = simulate_cmrr(cfg)
psrr = simulate_psrr(cfg)
n = simulate_noise(cfg, noise)
```

Default frequency sweep: ``BenchSweepConfig`` — 1 Hz–100 MHz, 10 points/decade (see [bench_spec.md](bench_spec.md)).

## Planned (TIA / Gm)

| Model | Target equation | Python status |
| --- | --- | --- |
| TIA closed-loop | ``Z_t(s)`` from ``OpampConfig`` + ``TiaConfig`` | ``NotImplementedError`` in ``tia.py`` |
| Gm | ``I_{out} = g_m V_{diff}`` | ``NotImplementedError`` in ``gm.py`` |

Bench catalog for TIA/Gm: [bench_spec.md](bench_spec.md) (planned scripts).
