# References and ideas

A literature survey made on 2026-11-23 to plan the simulator beyond the
basic Watterson model. Each entry says what it would add.

How each entry was checked:
- **[checked]**: read in the source document.
- **[abs]**: only the abstract or a summary was read.
- **[unverified]**: from memory or secondary sources; check before use.

## Standards

- **ITU-R F.1487 (05/2000)**, "Testing of HF modems with bandwidths of up
  to about 12 kHz using ionospheric channel simulators"
  ([pdf](https://www.itu.int/dms_pubrec/itu-r/rec/f/R-REC-F.1487-0-200005-I!!PDF-E.pdf)).
  **[checked]**
  - The model: eq. 1–2, two magneto-ionic components per path, each with
    its own attenuation, shift and spread (2σ).
  - It is valid for 3 kHz channels and may apply up to 12 kHz.
  - Annex 3 has the latitude tables used for the presets.
  - Occurrence (§5): low latitudes exceed 4 ms and 3 Hz about 5 % of the
    time; high latitudes exceed 5 ms and 25 Hz about 5 %.
  - Test length (§6): the longer of 3000/spread and 100/(BER × data
    rate).
- **ITU-R F.520-2 (1992)**, the older recommendation
  ([pdf](https://www.itu.int/dms_pubrec/itu-r/rec/f/R-REC-F.520-2-199203-W!!PDF-E.pdf)).
  **[checked]**
  - Flat fading at 0.2 and 1 Hz.
  - Good 0.5 ms / 0.1 Hz, moderate 1 ms / 0.5 Hz, poor 2 ms / 1 Hz,
    flutter 0.5 ms / 10 Hz.
  - F.1487 says both "low-latitude moderate" (2 ms / 1.5 Hz) and
    "mid-latitude disturbed" (2 ms / 1 Hz) correspond to F.520 "poor".
    The presets follow F.520-2's own figure, 1 Hz.
- **MIL-STD-188-110 Appendix E**: the tap-gain filter used here. A
  Gaussian impulse response exp(−(π d t)²) gives exactly the Gaussian
  power spectrum of 2σ = d, oversampled 32 times over the spread.
  [unverified: taken from skywave's description]
- **ITU-R P.372-18 (09/2026)**, radio noise
  ([pdf](https://www.itu.int/dms_pubrec/itu-r/rec/p/R-REC-P.372-18-202609-I!!PDF-E.pdf)).
  - Atmospheric noise Fa maps by season and time of day, Vd (the
    impulsiveness), and the APD as a function of Vd.
  - Man-made noise Fam = c − d·log10(f/MHz) [unverified]:

    | Environment | c | d |
    |---|---|---|
    | City | 76.8 | 27.7 |
    | Residential | 72.5 | 27.7 |
    | Rural | 67.2 | 27.7 |
    | Quiet rural | 53.6 | 28.6 |
    | Galactic | 52.0 | 23.0 |

  - Its man-made data dates from the 1950s–70s. Later measurements
    (FFI 16-00869, Swedish studies) found lower levels.

## Channel models beyond Watterson

- **Vogler & Hoffmeyer**, wideband HF model. NTIA
  [TR-88-240](https://its.ntia.gov/publications/download/TR-88-240.pdf),
  [TR-90-255](https://its.ntia.gov/publications/download/90-255_ocr.pdf),
  [TR-92-284](https://www.its.ntia.gov/publications/download/92-284.pdf);
  Radio Science 28 (1993).
  - Each mode has a delay-dependent Doppler shift and spread, and a
    skewed delay profile tied to the ionogram trace.
  - It is compared with spread-F on an auroral path and a mid-latitude
    path.
  - **Adds:** spread-F and auroral shapes, and signals wider than
    12 kHz.
- **Mastrangelo et al.**, IEEE Trans. Commun. 45(1) 1997: a real-time
  wideband simulator (V&H + noise + interference, > 1 MS/s); NTIA
  [98-348](https://its.ntia.gov/publications/download/98-348_ocr.pdf).
  [abs] **Adds:** an architecture reference.
- **Hoffmeyer & Nesenbergs**, NTIA
  [TR-87-221](https://its.ntia.gov/publications/download/87-221_ocr.pdf)
  (1987): a review of models and how to validate them.
- **High latitudes (the DAMSON sounder):**
  - Angling et al., Radio Science 33:97 (1998): Doppler spread 2–55 Hz
    and delay spread 1–11 ms on four oblique paths. [abs]
  - Angling's thesis (Leicester): fading correlated between adjacent
    taps and non-Gaussian delay profiles. [abs]
  - Jodalen et al., Radio Science 36 (2001). [abs]
  - Stocker, Warrington & Siddle, Radio Science 48:638 (2013): polar cap
    and trough paths through the solar cycle. [abs]
  - **Adds:** "auroral substorm" and "polar cap" presets, and correlated
    taps.
- **The mid-latitude trough and night-time sporadic E**: Warrington &
  Stocker. [abs] **Adds:** a sporadic E path (short delay, low spread).
- **Equatorial spread-F** (Los Alamos campaigns, OSTI 329722 and
  10170108): Doppler spread over 6 Hz at −10 dB on half of evenings,
  with bottomside spread-F doing the most harm. [abs] **Adds:** a
  "post-sunset equatorial" preset.
- **NVIS and long paths (Vilella, Ads, Hervás et al.)**: Antarctica to
  Spain (Radio Science 2008 and 2012, Remote Sensing 2015), and NVIS
  (Electronics 9:1037, 2020) with ground wave, Es and F2 given
  separately. [abs] **Adds:** NVIS with several paths.
- **Conditions that change over time:**
  - HiCIRF (Adelaide): ray tracing through an ionosphere that changes
    over time, plus scatter.
  - Mikhaleva et al. 2023: travelling ionospheric disturbances (TIDs)
    make Doppler curves sawtooth-shaped and create three-ray regions.
    [abs]
  - The stationary Watterson model holds for minutes only.
  - **Adds:** path parameters that drift, and paths that come and go.
- **Absorption**: NOAA D-RAP, absorption roughly ∝ f^−1.5
  [unverified]. **Adds:** sudden ionospheric disturbance fade-outs and
  polar cap absorption.

## Measured channels and data

- **HamSCI Grape / PSWS** (Collins et al., ESSD 2023;
  [Zenodo 6622112](https://zenodo.org/record/6622112),
  [8122124](https://zenodo.org/records/8122124)): Doppler and amplitude
  of the WWV/CHU carriers, 2019–2022. **Adds:** real traces to drive the
  drifting parameters. Check the licence of each record.
- **Wuhan HF channel modelling**
  ([Zenodo 7012769](https://zenodo.org/record/7012769)): CC BY 4.0,
  MATLAB files whose contents aren't documented.
- **McMurdo–South Pole oblique ionograms**
  ([Zenodo 3478267](https://zenodo.org/record/3478267)).
- **Paderborn HF speech database** (Heitkaemper et al.,
  [arXiv 2106.02472](https://arxiv.org/abs/2106.02472);
  [Zenodo 4247491](https://zenodo.org/records/4247491),
  [code](https://github.com/fgnt/ham_radio)): CC BY 4.0.
  - LSB from Paderborn, received by about 22 KiwiSDRs.
  - The clean LibriSpeech source is time-aligned with the recordings;
    the audio is 8 kHz.
  - The KiwiSDRs' AGC distorts the levels.
  - **Adds:** real channels with ground truth, for checking speech
    enhancement.
- **Machine learning for channel modelling**: no HF-specific generative
  model was found. A WGAN for massive MIMO
  ([arXiv 2403.05321](https://arxiv.org/abs/2403.05321)) is the nearest.
  Open ground.

## Noise and interference

- **Hall (1966)** impulsive noise model
  ([DTIC AD0648650](https://apps.dtic.mil/sti/pdfs/AD0648650.pdf)).
  **Lemmon & Behm**, NTIA
  [TR-91-277](https://its.ntia.gov/publications/download/TR-91-277.pdf):
  wideband HF noise as Gaussian plus Hall impulsive.
- **Middleton Class A and symmetric α-stable noise**
  ([survey](https://www.uni-due.de/imperia/md/content/dc/thoks_survey_aiee_2015.pdf)).
  - Class A parameters: A (impulsive index, about 0.01–1) and Γ
    [unverified].
  - Fit the parameters to the P.372 Vd.
- **Man-made sources**:
  [ITU-R SM.2155](https://www.itu.int/dms_pub/itu-r/opb/rep/R-REP-SM.2155-2009-PDF-E.pdf),
  [SM.2212](https://www.itu.int/dms_pub/itu-r/opb/rep/R-REP-SM.2212-1-2012-PDF-E.pdf)
  (powerline telecom). **Adds:** synthetic signatures:
  - switching-supply combs (20–200 kHz harmonics, drifting, with
    100/120 Hz AM)
  - VDSL with the amateur bands notched
  - powerline OFDM bursts
  - LED buzz.
- **Occupancy**:
  - The Laycock–Gott congestion model (Pantjiaros et al., IEE
    Proc.-Commun. 1998). [abs]
  - **Mendieta-Otero, Pérez-Álvarez, Pérez-Díaz**, "Interference
    simulator for the whole HF band: application to CW-Morse", IEEE TEMC
    56(3) 2014 ([arXiv 2402.04742](https://arxiv.org/abs/2402.04742)).
    **Adds:** a statistical population of interfering stations per band,
    time and place.
  - **Pederick & Cervera**, "Modeling the interference environment in
    the HF band", Radio Science 51:82 (2016): the transmitter database,
    ray tracing and absorption set the interferers' levels.
- **Sferic waveforms**: nothing focused found yet.

## Validation methods

- F.520 and F.1487 test ranges: fade durations 0.05–1.5 s, fading rates
  5–40 a minute, spreads 0.2 and 1 Hz.
- Measure the scattering function from a simulated sounding (PN or
  chirp) and compare with the target: tap delays, per-tap shift, −3 and
  −10 dB widths.
- Watterson, Juroshek & Bensema (1970) checked a Gaussian tap spectrum,
  a Rayleigh envelope and independent taps against measurements
  [unverified]. The tests here check the same, plus the level-crossing
  rate, fade durations and frequency correlation.
- P.372: check the measured APD and Vd against their targets.
- Decoder outcome tables per preset (decode rate against SNR) as
  regression tests.

## Implementations seen

| Project | Language, licence | What it has |
|---|---|---|
| [skywave](https://github.com/openarq-project/skywave) | Python, Apache-2.0 | Watterson on real audio (two equal paths), P.372 noise, QRM, radio-chain effects; used for cross-checking ([validation.md](validation.md)) |
| [ARSFI HFSimulator](https://github.com/ARSFI/HFSimulator) | C, MIT | Watterson simulator (Winlink's, ran on a Teensy) |
| [M0LTE.HfChannel](https://github.com/M0LTE/M0LTE.HfChannel) | .NET | Watterson per MIL-STD-188-110D Appendix E, measured impulse noise |
| codec2 `ch` | C, LGPL-2.1 | MPP/MPG/MPD presets, AWGN |
| GNU Radio `selective_fading_model` | C++, GPL | sum of sinusoids, any delay profile, no HF presets |
| MATLAB `comm.HFChannel` | commercial | F.1487 |
