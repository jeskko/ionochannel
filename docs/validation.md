# Validation

## Against theory

`cargo test --release` ([tests/stats.rs](../tests/stats.rs)) and
`python3 python/tests/test_stats.py` check both implementations against
closed-form results. Tolerances are a few per cent, set by the number of
independent fades in each run.

| check | theory | tolerance |
|---|---|---|
| mean power | 1 | 5 % |
| P(power < x·mean), x = 0.01, 0.1, 1 | 1 − e^(−x) | 0.002, 0.01, 0.02 |
| Doppler spread (2σ), 0.1–55 Hz | from \|R(τ)\| = exp(−2π²σ²τ²) at \|R\| ≈ 0.7 and 0.3 | 6 % |
| Doppler shift | arg R(τ) / 2πτ | 3 % of the spread |
| level-crossing rate at ρ = 0.1, 0.316, 1 | √(4π)·σ·ρ·e^(−ρ²) | 8 % |
| average fade duration | (1 − e^(−ρ²)) / N | 8 % |
| two paths τ apart, correlation at Δf | \|cos(πΔfτ)\| | 0.06 |
| two paths' gains | uncorrelated | < 0.05 |
| fractional delay, tones to 0.375 fs | exact | 2·10⁻³ |
| real audio, +10 Hz shift | tone moves up 10 Hz | 1 % |

## Against skywave

`tools/compare_skywave.py` sends tones through skywave's
`WattersonChannel`, this package's `Channel` and the Rust command line,
all at 12 kS/s real audio. The skywave version was the git head of
2026-11-23 (1c8e99d). Each run is 3000/spread seconds, between 300 and
3000 s. "<−10 dB" is the share of time at least 10 dB under the mean.
"rho" is the correlation of two tones 1/(4τ) apart.

| preset | delay | spread | impl | <−10 dB | spread meas. | rho |
|---|---|---|---|---|---|---|
| good | 0.5 ms | 0.1 Hz | skywave | 9.0 % | 0.10 | 0.69 |
| | | | python | 11.5 % | 0.10 | 0.71 |
| | | | rust | 9.3 % | 0.10 | 0.70 |
| moderate | 1 ms | 0.5 Hz | skywave | 9.6 % | 0.49 | 0.70 |
| | | | python | 9.4 % | 0.50 | 0.71 |
| | | | rust | 9.4 % | 0.50 | 0.71 |
| poor | 2 ms | 1 Hz | skywave | 9.6 % | 1.00 | 0.71 |
| | | | python | 9.4 % | 1.02 | 0.71 |
| | | | rust | 9.6 % | 1.00 | 0.71 |
| flutter | 0.5 ms | 10 Hz | skywave | 9.9 % | 10.06 | 0.71 |
| | | | python | 9.9 % | 9.98 | 0.70 |
| | | | rust | 9.3 % | 10.00 | 0.70 |
| high-moderate | 3 ms | 10 Hz | skywave | 9.6 % | 10.04 | 0.70 |
| | | | python | 9.5 % | 10.14 | 0.71 |
| | | | rust | 9.7 % | 10.01 | 0.71 |
| high-disturbed | 7 ms | 30 Hz | skywave | 9.5 % | 29.58 | 0.71 |
| | | | python | 9.6 % | 29.75 | 0.70 |
| | | | rust | 9.8 % | 29.37 | 0.71 |

Theory: 9.5 %, the spread as given, and rho 0.71. Python's 11.5 % on
"good" is within the scatter expected from about 300 independent fades
(0.1 Hz × 3000 s).
