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
  **[checked]** 2026-11-23.
  - Atmospheric noise (§5.1): Fam maps by season and 4-hour block (Figs
    13a–36a), frequency curves (b), and variability (c).
  - The impulsiveness is the voltage deviation Vd = 20 log10(rms / mean)
    of the envelope, given for 200 Hz. Fig. 37 is the APD for each Vd
    (1.049 = Gaussian, up to 30 dB); Fig. 38 converts the 200 Hz Vd to
    other bandwidths (straight lines in log bandwidth, Vd 2–14 dB, b/200
    from 0.01 to 1000). The text calls it Fig. 39; it is Fig. 38.
  - In this edition the c-figures show Du, Dl and their σ only. They no
    longer give Vdm, although the text still says they do, so Vd has to
    come from elsewhere (older editions or measurements).
  - Figs. 37 and 38 are raster images. Digitised values and how they were
    read are in [tools/check_qrn.py](../tools/check_qrn.py) and
    [qrn.json](../python/ionochannel/qrn.json).
  - Man-made noise (§6.1, eq. 17, Table 1): Fam = c − d·log10(f/MHz),
    0.3–250 MHz:

    | Environment | c | d |
    |---|---|---|
    | City | 76.8 | 27.7 |
    | Residential | 72.5 | 27.7 |
    | Rural | 67.2 | 27.7 |
    | Quiet rural | 53.6 | 28.6 |
    | Galactic | 52.0 | 23.0 |

  - Table 2 gives decile deviations with time (upper 9.2–11.0 dB, lower
    4.6–6.7 dB) and with location (5.8–8.4 dB) for city, residential and
    rural, to be taken as log-normal half distributions.
  - §6 sorts man-made noise into wideband (AWGN-like), impulsive, and
    "single carrier noise" (carriers and humps from nearby sources).
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
  - TR-92-284 (Part III) **[checked]** 2026-11-23: transfer function eq.
    5–8 (delay shape T(τ) = A·e^{α(ln g+1−g)}, slant f_B = fs + b(τ−τc),
    Gaussian ζ = πσf²t with σf = σD(−ln s_v/π)^−½ or Lorentzian), the
    appendix's conditions for α and τℓ, and Table 1 (Fairbanks 88 km
    path, May 1988: σT 22–1600 µs, σc 10.7–300 µs, σD 0.1–9 Hz). The
    threshold s_v isn't given. Used for `vh_mode`. TR-88-240 (Part I)
    fetched; it describes the deterministic delay-frequency model (sech²
    profile), not yet used.
- **Mastrangelo et al.**, IEEE Trans. Commun. 45(1) 1997: a real-time
  wideband simulator (V&H + noise + interference, > 1 MS/s); NTIA
  [98-348](https://its.ntia.gov/publications/download/98-348_ocr.pdf).
  [abs] **Adds:** an architecture reference.
- **Hoffmeyer & Nesenbergs**, NTIA
  [TR-87-221](https://its.ntia.gov/publications/download/87-221_ocr.pdf)
  (1987): a review of models and how to validate them.
- **High latitudes (the DAMSON sounder):**
  - Angling, Cannon, Davies, Willink, Jodalen, Lundborg, Radio Science
    33(1):97–107 (1998), doi:10.1029/97RS02206. **[checked]** 2026-11-23,
    the full paper (a PDF supplied by the user):
    - Paths (Table 1): Svalbard–Tuentangen 2019 km, Svalbard–Kiruna
      1158 km, Harstad–Tuentangen 1019 km, Harstad–Kiruna 194 km;
      3–22 MHz; Doppler resolution 0.62 Hz, range ±40 Hz.
    - §4.2: multipath spread runs from the rising edge of the central
      80 % power region of the first mode to the trailing edge of the
      last; Doppler spread is the central 80 % power width of the
      composite spectrum; "converted to 2σ by multiplying by 0.78".
    - Tables 4 and 5: values exceeded 5 % of the time, per path, for
      0000–2400 and 1900–0100 UT and three bands (2.8–4.7, 6.7–11.2,
      14.4–21.9 MHz), read as the intercepts of the 5 % contour with each
      axis. Table 5 (SNR −5 to 5 dB, no guard, all day): multipath
      0.6–10.7 ms, Doppler spread 1.8–54.6 Hz; Harstad–Kiruna has the
      largest (10.7 ms at 6.7–11.2 MHz; 54.6 Hz at 14.4–21.9 MHz).
    - §4.4.2: Doppler spread generally grows with frequency, except
      Svalbard–Tuentangen (multihop at low frequencies).
    - Used for the presets auroral-7-11mhz and auroral-14-22mhz.
  - Jacobsen, Jodalen, Cannon, Smith, Angling, DAMSON poster (ESA space
    weather workshop 2), read 2026-11-23: an abstract only. "Large
    Doppler spreads (tens of Hz) are observed during disturbed conditions
    (substorms), when the ionospheric reflection point is located within
    the auroral oval"; long delays (several ms) at midday above the
    predicted MUF, possibly ground scatter. No tables.
  - Angling's thesis (Leicester): fading correlated between adjacent
    taps and non-Gaussian delay profiles. [abs]
  - Jodalen et al., Radio Science 36 (2001). [abs]
  - Stocker, Warrington & Siddle, Radio Science 48(5):638–645 (2013),
    doi:10.1002/2013RS005264: polar cap (Qaanaaq–Ny-Ålesund) and trough
    (Uppsala and Nurmijärvi–Bruntingthorpe) paths, 2001 and 2009–2012.
    **[checked]** 2026-11-23 (figshare copy, rds20067.pdf):
    - §2: Doppler spread is "the narrowest spectral width containing 80%
      of the received signal power", corrected for the noise base;
      composite multipath spread runs from the first mode's rising edge
      to the last mode's falling edge, modes more than 12 dB below the
      strongest excluded; the same definitions as Angling et al. 1998.
      Only SNR ≥ 6 dB in 3 kHz; day and night are the 6 h around path
      midpoint noon and midnight.
    - Table 2: 95th percentiles, 14 March 2009 to 1 July 2012, by
      season, day/night and frequency (4.64–18.38 MHz). Trough path NB:
      Doppler spread 0.5–2.0 Hz, multipath 0.1–4.2 ms. Polar cap path
      QN: Doppler spread 1.0–13.0 Hz, multipath 0.1–1.5 ms.
    - §3.1: on QN, Doppler spreads are "typically less than 5 Hz, but at
      times significantly exceed this value rising to upward of 20 Hz";
      swings in Doppler shift with polar patches.
    - Used for the presets polar-cap-14mhz, polar-cap-5mhz and
      trough-5mhz.
  - **Adds:** "auroral substorm" and "polar cap" presets, and correlated
    taps.
- **The mid-latitude trough and night-time sporadic E**: Warrington &
  Stocker. [abs] **Adds:** a sporadic E path (short delay, low spread).
- **Equatorial spread-F** (Los Alamos campaigns, OSTI 329722 and
  10170108): Doppler spread over 6 Hz at −10 dB on half of evenings,
  with bottomside spread-F doing the most harm. [abs] OSTI 10170108
  (Fitzgerald, Argo, Carlos, LA-UR-93-2140, Maloelap–Bikini, 702 km,
  10.2 MHz, August 1990), read 2026-11-23 from a poor OCR scan: "Doppler
  spreads of greater than 20 Hz were not uncommon"; spatial correlation
  down to 50 m and frequency coherence bandwidth to about 1 kHz. No
  definitions or delay spreads readable, so no preset. OSTI 329722 gave
  HTTP 404.
  - Fitzgerald, "Channel probe measurements for the American Sector
    Clutter Experiment, January 1994", LA-UR-94-1811 (OSTI 10156838),
    read 2026-11-23: Piura–Arequipa (Peru) across the magnetic
    equator, 12.87–13.37 MHz CW and an FMCW probe (2.5 ms delay window).
    Qualitative only: Doppler and delay spread grow after spread-F onset;
    about one hour analysed, no statistics.
  - Fitzgerald, Argo & Carlos, "Equatorial spread F effects on an HF
    path: Doppler spread, spatial coherence, and frequency coherence",
    Radio Science 34(1):167–178 (1999), doi:10.1029/1998RS900013.
    **[checked]** 2026-11-23 (a PDF supplied by the user): 700 km
    Maloelap–Bikini, reflection at 10.18°N 168.40°E, CW at 10.224 MHz
    and offsets of 0.5–4 kHz, 13 evenings in August 1990. Abstract and
    Fig. 5: 50 % of evenings Doppler spread > 6 Hz at −10 dB and > 15 Hz
    at −30 dB (relative to peak power), one instance well above 50 Hz;
    75 % of evenings coherence bandwidth < 1.5 kHz, 2 of 12 below
    0.8 kHz; coherence distances defined where coherence falls below
    50 %. The −30 dB pedestal was incoherent. Used for the preset
    equatorial-spread-f.
- **NVIS and long paths (Vilella, Ads, Hervás et al.)**: Antarctica to
  Spain (Radio Science 2008 and 2012, Remote Sensing 2015), and NVIS
  ("Study of NVIS Channel for USN Protocol Definition in Antarctica",
  Electronics 9(6):1037, 2020) with Doppler shift and delay spread of
  ground wave, Es and F2 given separately, near Livingston Island.
  **[checked]** 2026-11-23, the full paper (a PDF supplied by the user;
  the extracted text overlays a review copy, so some lines are
  garbled):
  - 4.3 MHz, 5.7 km (Rockie Glacier) and a second site with ground
    wave (Argentina Cove); E reflects at about 120 km and F2 at about
    310 km (Figure 6).
  - Delay spread is the rms delay spread (eq. 5).
  - The summary table: skywave delay spread at most 2.75 ms, at least
    0.05 ms; Doppler spread at most 0.1 Hz; Doppler shift −17.5 Hz at
    most. With ground wave the delay spread reaches 5.5 ms.
  - Dear et al., "Ionospheric channel impulse response measurement system
    for NVIS propagation mode over Java Island based on low-cost SDR
    platform", Radio Science 59:e2023RS007877 (2024). **[checked]**
    2026-11-23 (a PDF supplied by the user): Pameungpeuk–Bandung, 7 MHz,
    15–21 December 2021, quiet (Kp 0.6–3.6). Fig. 12: rms delay spread
    0–1.3 ms, mean 0.4627 ms; Doppler shift 0.7–1.1 Hz, mean 0.9 Hz. An
    example impulse response has arrivals at 0, 0.213, 0.441, 1.377 and
    1.654 ms in two groups (E/Es and F; foEs about 7.1 MHz). No Doppler
    spread or path powers. Used for the preset nvis-java-7mhz.
  - Not used for a preset. An rms spread of 2.75 ms doesn't fit E and
    F2 at 120 and 310 km. −17.5 Hz looks like an offset between the
    radios. Mode powers aren't given.
  **Adds:** NVIS with several paths, if better numbers turn up.
- **Conditions that change over time:**
  - HiCIRF (Adelaide): ray tracing through an ionosphere that changes
    over time, plus scatter.
  - Mikhaleva et al. 2023: travelling ionospheric disturbances (TIDs)
    make Doppler curves sawtooth-shaped and create three-ray regions.
    [abs]
  - Trop, LaBelle, Erickson, Zhang, McGaw, Kovacs, "Tracking Traveling
    Ionospheric Disturbances through Doppler-shifted AM radio
    transmissions", EGUsphere preprint egusphere-2024-2383 (2024; final
    version Atmos. Meas. Tech. 18:1909, 2025). **[checked]** the preprint,
    2026-11-23:
    - Introduction, citing Hernández-Pajares et al. 2012 and Otsuka et
      al. 2013: MSTIDs have wavelengths of hundreds of km, periods
      15–60 min and phase speeds 250–1000 m/s; LSTIDs have wavelengths
      over 1000 km, periods 30–180 min and phase speeds 400–1000 m/s.
    - Eq. 4: Doppler shift Δf = −2 f v / c, v the layer's vertical
      velocity, so it scales with the carrier frequency. No HF velocity
      or Doppler amplitude is given (its data are 0.8–1.6 MHz).
    - At lower frequencies multiple reflections give "S-shaped features
      that fold back on themselves" (citing Fišer et al. 2017).
    - Occurrence: fewest events in summer; LSTIDs peak at the equinoxes,
      MSTIDs (especially at night) in summer and winter (citing others).
    - Their measured phase velocities: median 347 m/s, 71–1914 m/s.
  - The stationary Watterson model holds for minutes only.
  - Oinats et al., Earth Planets Space 68 (2016), doi:10.1186/
    s40623-016-0390-8: MSTID periods and wavelengths from HF radar
    against solar activity. **[checked]** 2026-11-23 (open-access PDF):
    - From the Hokkaido East (2007–2014) and Ekaterinburg (2013–2014) HF
      radars: the most frequent period is 60–100 min; daytime velocities
      60–100 m/s (Hokkaido E region), 110–180 m/s (Hokkaido F), 30–90 m/s
      (Ekaterinburg F); wavelengths 300–550, 450–800 and 150–400 km.
    - Relative amplitude (RMS slant-range deviation over the mean, taken
      as an estimate of the electron density perturbation): 3–8 % in the
      Hokkaido F region, 4–9 % in the E region; it grows with solar
      activity and with the AE index, while period and wavelength shrink
      with solar activity.
  - **Adds:** path parameters that drift, and paths that come and go.
- **Absorption**: NOAA SWPC "Global D-Region Absorption Prediction
  Documentation" (web page). **[checked]** 2026-11-23:
  - HAF (1 dB, vertical) = 10·log10(flux, W/m²) + 65 MHz at the sub-solar
    point, a fit to the forecasters' table M1 → 15, M5 → 20, X1 → 25,
    X5 → 30 MHz; tapering as cos(χ)^0.75 to the terminator.
  - Polar cap absorption at 30 MHz: A_d = 0.115·√J(E > 5.2 MeV) dB,
    A_n = 0.020·√J(E > 2.2 MeV) dB (Sauer and Wilkinson 2008).
  - A(f) = (f0 / f)^1.5 · A(f0) (Davies 1990; Stonehocker 1970).
  - Used by the absorption helpers.

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
  - Not used. Single impulses with Class A-like statistics make Vd grow
    6–7 dB per decade of bandwidth, against 0.9–3.4 in P.372 Fig. 38.
    The QRN model here uses bursts lasting 0.1–100 ms instead (see the
    README).
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
  - Economou et al., "Models of HF spectral occupancy over a sunspot
    cycle", IEE Proc.-Commun. 152(6):980–988 (2005), doi:10.1049/
    ip-com:20045104: the coefficients behind Mendieta-Otero et al.'s
    levels. No open copy found on 2026-11-23 (IET Digital Library only).
  - **Mendieta-Otero, Pérez-Álvarez, Pérez-Díaz**, "Interference
    simulator for the whole HF band: application to CW-Morse", IEEE TEMC
    56(3) 2014 ([arXiv 2402.04742](https://arxiv.org/abs/2402.04742)).
    **[checked]** 2026-11-23 (author's version):
    - Poisson arrivals and exponential durations (§V.C); in their ARRL
      Field Day 2011 example λ = 6.68 interferers/s over all amateur
      bands (577,181 CW QSOs in 24 h) and mean duration 10 s (Table I).
    - Each QSO sends "PARIS" repeated, dot = duration / 331 (eq. 18),
      WPM = 1.2 / dot (eq. 15), raised-cosine edges of dot / 10 (eq. 19).
    - Levels from the UMIST congestion model (Economou et al., IEE
      Proc.-Commun. 152(6):980, 2005): Q_k = 1/(1 + e^-y_k),
      y_k = α_k + B_k·E (eq. 8–11), so E is logistic with mean −α_k/B_k
      and scale 1/|B_k|; Fig. 3 example B_k = −0.0849, α_k = 10.81. The
      142/139 coefficients for α_k are in Economou et al., not here.
    - Used for `population` (rate left to the user).
  - **Pederick & Cervera**, "Modeling the interference environment in
    the HF band", Radio Science 51:82 (2016): the transmitter database,
    ray tracing and absorption set the interferers' levels. Read
    2026-11-23 (a PDF supplied by the user): it needs transmitter
    databases, ray tracing and a model ionosphere, so it can't be built
    from the paper alone; not used.
- **Nissen & Bello**, "Measured channel parameters for the disturbed
  wide-bandwidth HF channel", MITRE MP 94B0000006 (January 1994; DTIC
  ADA277877). **[checked]** 2026-11-23 (a PDF supplied by the user):
  Søndrestrom–Bedford, 3100 km, 14 March–1 April 1992, DSPN probe at
  1024 kchip/s. Fig. 3: rms (2σ) Doppler spread median 5 Hz, 10th/90th
  percentiles 1.75/13 Hz, largest 17 Hz; Doppler shift magnitude median
  about 0.5 Hz, 90th just below 3 Hz; rms (2σ) delay spread median about
  100 µs, 10th 17 µs, 90th slightly above 400 µs. Delay-axis
  correlations support uncorrelated scattering. Used for the presets
  high-latitude-wb-median and -90.
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
