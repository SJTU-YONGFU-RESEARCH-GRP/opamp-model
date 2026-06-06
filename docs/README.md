# opamp-model documentation

Index for the **opamp-model** package (voltage op-amp, TIA, and Gm macromodels; multi-engine benches; reporting).

## Documents

| Document | Purpose |
| --- | --- |
| [MODEL.md](MODEL.md) | Macromodel equations, impedance/noise/large-signal, peer-engine rules |
| [bench_spec.md](bench_spec.md) | Bench stimuli, per-engine support, outputs, default sweeps |
| [metrics_catalog.md](metrics_catalog.md) | Metric names, units, implementation status |
| [golden_metrics.yaml](golden_metrics.yaml) | Optional transistor-level reference targets |
| [Flicker_Noise_Analysis_on_Chopper_Amplifier.pdf](Flicker_Noise_Analysis_on_Chopper_Amplifier.pdf) | Flicker noise modeling reference (device-level; NEWCAS 2021) |

## Quick reference

| Engine | AC / STB | Noise | PSRR | Slew / THD |
| --- | --- | --- | --- | --- |
| **python** | Full macromodel | Full macromodel | Full macromodel | Full macromodel |
| **ngspice** | Parsed `wrdata` Bode | Hybrid: SPICE AC × `OpampNoiseConfig` | Python macromodel | Not run in batch |
| **spectre** | PSF Bode (`spectre_psf.py`) | Hybrid: PSF AC × `OpampNoiseConfig` | Python macromodel | Not run in batch |

Run all benches:

```bash
./scripts/run_all_simulations.sh --skip-missing
```

Results land under `outputs/<engine>/` with per-bench `*_REPORT.md`, `opamp_metrics.json`, CSV/SVG sweeps, and a top-level `REPORT.md`.

## Flicker noise (op-amp macromodel)

Open-loop flicker follows [Flicker_Noise_Analysis_on_Chopper_Amplifier.pdf](Flicker_Noise_Analysis_on_Chopper_Amplifier.pdf) device equations (Eq. 1, Coram `KF`/`AF`, Verilog-A `flicker_noise`). Chopper/system-level noise is **out of scope** for this package.

| Layer | Scope |
| --- | --- |
| **Python** | `en_flicker_1hz`, `en_flicker_ef`, `kf`/`af`/`bias_current_a`, computed `flicker_corner_hz` |
| **Verilog-A** | `white_noise` + `flicker_noise` on `configurable_opamp.va` |
| **Benches** | `run_noise.py` (open-loop only) |

See [MODEL.md — Noise](MODEL.md#noise).

### Legacy parameter note

The pre-paper pair `en_flicker_at_1hz_v_per_sqrt_hz` + `en_flicker_corner_hz` (scale-factor form) remains supported as aliases during migration. New benches and `golden_metrics.yaml` prefer **`en_flicker_1hz`** (true density at 1 Hz) and **`en_flicker_ef`** (exponent `EF`).

## Still hybrid / python-only

| Area | Status |
| --- | --- |
| PSRR | Python macromodel on all engines (`python_macromodel`) |
| TIA / Gm AC | Python macromodel on all engines |
| ngspice / Spectre noise | `hybrid_noise_merge` (SPICE AC gain × config `en_in`) |
| Slew / THD | `python_macromodel` in batch; manual ngspice/Spectre TRAN stubs tag `tran_scaffold` |

`compare_engines.py` reports cross-engine spread and marks parity **n/a** when metrics share a non-comparable `source`. Optional `golden_metrics.yaml` reference column only.

## Planned work

- Full ngspice controlled-source noise transfer (remove post-merge flicker where VA supplies it)
- Spectre/ngspice PSRR and full SPICE TRAN netlists
