# Op-amp behavioral model reference

Deep modeling reference for **opamp-model**.

## Related documentation

- [bench_spec.md](bench_spec.md) — bench definitions and outputs
- [metrics_catalog.md](metrics_catalog.md) — metric index and status
- [golden_metrics.yaml](golden_metrics.yaml) — optional reference targets
- [Flicker_Noise_Analysis_on_Chopper_Amplifier.pdf](Flicker_Noise_Analysis_on_Chopper_Amplifier.pdf) — device flicker noise reference (NEWCAS 2021)

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
e_n(f) = \sqrt{e_{n,\mathrm{white}}^2 + e_{n,\mathrm{flicker}}^2(f)}
\]

Output-referred spectrum scales with open-loop magnitude: ``e_{out}(f) = e_n(f) \cdot |A_{\mathrm{ol}}(j\omega)|``.

Implementation: ``src/opamp_model/noise.py``, ``src/opamp_model/noise_analysis.py``; bench ``run_noise.py``.

Reference: [Flicker_Noise_Analysis_on_Chopper_Amplifier.pdf](Flicker_Noise_Analysis_on_Chopper_Amplifier.pdf) (Zhou et al., NEWCAS 2021).

#### Flicker noise (paper-aligned)

**Phase 1** replaces the legacy scale-factor pair with parameters that match the paper and Spectre ``flicker_noise(pwr, EF, "flicker")``:

\[
e_{n,\mathrm{flicker}}(f) = \frac{e_{n,\mathrm{flicker@1Hz}}}{f^{E_F/2}}
\qquad
e_n(f) = \sqrt{e_{n,\mathrm{white}}^2 + e_{n,\mathrm{flicker}}^2(f)}
\]

| Intended config field | Paper / Spectre | Meaning |
| --- | --- | --- |
| ``en_flicker_1hz_v_per_sqrt_hz`` | ``pwr`` in ``flicker_noise(pwr, EF, …)`` | Input-referred voltage density at **exactly 1 Hz** (V/√Hz) |
| ``en_flicker_ef`` | ``EF`` | Exponent in **PSD** \(S_n(f) \propto 1/f^{E_F}\); default ``1.0`` (classic 1/f) |

Classic MOS flicker PSD (paper Eq. 1):

\[
V_n^2(f) = \frac{K}{C_{ox}\,W\,L\,f}
\qquad\Rightarrow\qquad
e_n(f) = \sqrt{\frac{K}{C_{ox}WL}}\cdot f^{-E_F/2}
\quad (E_F = 1)
\]

**Intended Python API (Phase 1):**

```python
from opamp_model.config import OpampNoiseConfig
from opamp_model.noise import input_referred_en, flicker_corner_hz

noise = OpampNoiseConfig(
    en_white_v_per_sqrt_hz=5.0e-9,
    en_flicker_1hz_v_per_sqrt_hz=50.0e-9,  # true 1 Hz level (paper pwr)
    en_flicker_ef=1.0,
)
f_c = flicker_corner_hz(noise)  # reported metric flicker_corner_hz
```

CLI (intended): ``--en-flicker-1hz-nv-per-sqrt-hz``, ``--en-flicker-ef``.

**Legacy alias (migration):** ``en_flicker_at_1hz_v_per_sqrt_hz`` + ``en_flicker_corner_hz`` implemented the equivalent shape via \(e_{n,\mathrm{flicker}}(f) = e_{\mathrm{coef}}\sqrt{f_c/f}\). When only legacy fields are set, engines derive ``en_flicker_1hz = en_flicker_at_1hz × √en_flicker_corner_hz``.

#### Flicker corner definition

Per the paper (ref. [23] in the PDF), the **flicker corner frequency** \(f_{\mathrm{corner}}\) is the frequency where thermal and flicker noise **equally contribute to the PSD** (not a free scale factor):

\[
e_{n,\mathrm{white}}^2 = \frac{e_{n,\mathrm{flicker@1Hz}}^2}{f_{\mathrm{corner}}^{E_F}}
\qquad\Rightarrow\qquad
f_{\mathrm{corner}} = \left(\frac{e_{n,\mathrm{flicker@1Hz}}}{e_{n,\mathrm{white}}}\right)^{2/E_F}
\]

| Quantity | Role |
| --- | --- |
| ``flicker_corner_hz`` | **Reported** scalar from config (``status: reported``, source ``flicker_corner``) |
| ``en_flicker_corner_hz`` | **Legacy param** — retained as alias; prefer computed ``flicker_corner_hz`` in new reports |

Default example: ``en_white = 5 nV/√Hz``, ``en_flicker_1hz = 50 nV/√Hz``, ``EF = 1`` → ``flicker_corner_hz = 100 Hz``.

#### Coram KF / AF (Phase 2)

The paper’s Verilog-AMS amplifier and switch models use the Coram-corrected compact flicker formulation (ref. [14]):

```verilog
Ir = V(vip, vin) / R;
Pn = KF * pow(abs(Ir), AF);
I(vip, vin) <+ flicker_noise(sign(Ir)*Pn, EF, "flicker");
```

| Intended config field | Paper | Role |
| --- | --- | --- |
| ``kf`` | ``KF`` | Flicker strength constant |
| ``af`` | ``AF`` | Current exponent in ``Pn = KF·|I|^AF`` |
| ``bias_current_a`` | ``Ir`` | Bias current for bias-dependent ``en_flicker_1hz`` |

When ``kf > 0``, ``en_flicker_1hz_v_per_sqrt_hz`` is **derived** from bias rather than taken from the table:

```python
# Intended helper (opamp_model.noise)
def en_flicker_1hz_from_kf(kf: float, af: float, bias_current_a: float) -> float:
    """Map Coram Pn at 1 Hz to input-referred V/√Hz (device-specific calibration)."""
```

Fixed ``en_flicker_1hz`` overrides ``kf``/``af`` when explicitly set (documented precedence).

#### Verilog-A ``flicker_noise`` (Phase 3)

``configurable_opamp.va`` gains native noise sources aligned with Fig. 2 of the paper:

```verilog
I(inp, inn) <+ white_noise(4.0*`P_K*$temperature/RIN_OHM, "thermal");
I(inp, inn) <+ flicker_noise(EN_FLICKER_PWR_1HZ, EN_FLICKER_EF, "flicker");
```

| VA parameter | Maps to |
| --- | --- |
| ``EN_FLICKER_PWR_1HZ`` | ``OpampNoiseConfig.en_flicker_1hz_v_per_sqrt_hz`` |
| ``EN_FLICKER_EF`` | ``OpampNoiseConfig.en_flicker_ef`` |

Spectre ``noise_open_loop.scs`` and ngspice decks pass these at render time. Engine merge rules:

- **spectre:** open-loop AC gain from the companion ``ac`` analysis × ``OpampNoiseConfig`` input-referred density (same as ngspice). VA ``white_noise`` / ``flicker_noise`` do not propagate through ``laplace_nd`` in Spectre ``.noise``.
- **ngspice:** thermal from ``RN``; flicker from VA when the deck includes ``flicker_noise``; else config merge (today).

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
| AC open-loop | Macromodel | ``ac_open_loop.cir``: ideal VCVS + RC with ``Rlp·Clp=1/wp`` (one-pole, same as ``laplace_nd``); ``wrdata`` parsed | ``ac_open_loop.scs`` → PSF ``*.raw/ac.ac`` via ``spectre_psf.py`` |
| STB loop gain | ``beta * A_open`` | ``stb_loop.cir`` (same one-pole macromodel); ``loop_beta`` applied in ``run_stb.py`` | Same AC PSF path; ``loop_beta`` in ``run_stb.py`` |
| Noise | Macromodel spectrum | ``noise_open_loop.cir`` (``.noise`` + AC); VA flicker when enabled | ``noise_open_loop.scs`` (``.noise`` + PSF); VA ``flicker_noise`` when enabled |
| PSRR | Macromodel | Python only (``psrr.scs`` stub) | Python only (``psrr.scs`` stub) |
| Slew / THD | TRAN macromodel | Not wired | Not wired |

**Scaffolding (temporary):** Some benches may still fall back to Python when PSF or ngspice artifacts are missing; see engine code. Spectre AC/STB read Bode from PSF when available (``src/opamp_model/spectre_psf.py``).

### Noise engine limitations

| Engine | Netlist | Spectrum source | Limitation |
| --- | --- | --- | --- |
| **python** | — | ``noise.py`` × ``|A_open(f)|`` | Macromodel for ``--simulator python`` only |
| **ngspice** | ``testbench/ngspice/noise_open_loop.cir`` | ngspice AC gain × ``input_referred_en`` (config) | Ideal VCVS (``E``) does not amplify resistor noise in ``.noise``; thermal ``Rn`` runs; flicker from ``OpampNoiseConfig`` |
| **spectre** | ``testbench/spectre/noise_open_loop.scs`` | Spectre AC gain × ``input_referred_en`` (config) | VA ``white_noise`` / ``flicker_noise`` do not propagate through ``laplace_nd`` in ``.noise``; same post-processing as ngspice |

Spectre and ngspice both derive output-referred noise from the companion AC sweep and ``OpampNoiseConfig`` input-referred density. The Python noise bench remains the reference macromodel for ``--simulator python`` only.

``docs/golden_metrics.yaml`` holds optional transistor-level reference targets for future ``compare_engines.py``; it is **not** an engine truth file.

## Verilog-A modules

| Module | Path | Status |
| --- | --- | --- |
| Voltage op-amp | [../veriloga/configurable_opamp.va](../veriloga/configurable_opamp.va) | Dominant-pole ``laplace_nd``, CMRR common-mode path, PSRR VDD feedthrough, ``Rin‖Cin``, ``Rout‖Cout`` |
| TIA | [../veriloga/configurable_tia.va](../veriloga/configurable_tia.va) | Parameter shell (no closed-loop yet) |
| Gm / OTA | [../veriloga/configurable_gm.va](../veriloga/configurable_gm.va) | Parameter shell |

CMRR and PSRR are in Verilog-A (``CMRR_DB``, ``PSRR_DB``, ``PSRR_POLE_HZ``). ``white_noise`` / ``flicker_noise`` on ``inp, inn`` (paper Fig. 2). Slew and weak nonlinearity remain Python TRAN macromodels.

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
from opamp_model.noise import flicker_corner_frequency

cfg = OpampConfig(a0_db=80.0, gbw_hz=10e6)
noise = OpampNoiseConfig(
    en_flicker_1hz_v_per_sqrt_hz=50.0e-9,
    en_flicker_ef=1.0,
    kf=0.0,  # set kf/af/bias_current_a to derive en_flicker_1hz from bias
)

ac = simulate_ac(cfg, noise)
stb = simulate_stb(cfg, noise)
cmrr = simulate_cmrr(cfg)
psrr = simulate_psrr(cfg)
n = simulate_noise(cfg, noise)
f_c = flicker_corner_frequency(noise)
```

Default frequency sweep: ``BenchSweepConfig`` — 1 Hz–100 MHz, 10 points/decade (see [bench_spec.md](bench_spec.md)).

## Planned (TIA / Gm)

| Model | Target equation | Python status |
| --- | --- | --- |
| TIA closed-loop | ``Z_t(s)`` from ``OpampConfig`` + ``TiaConfig`` | ``NotImplementedError`` in ``tia.py`` |
| Gm | ``I_{out} = g_m V_{diff}`` | ``NotImplementedError`` in ``gm.py`` |

Bench catalog for TIA/Gm: [bench_spec.md](bench_spec.md) (planned scripts).
