"""ionochannel: an HF ionospheric channel simulator (Watterson / ITU-R
F.1487) for testing receivers and making training data. numpy only.

    import numpy as np, ionochannel as ic
    ch = ic.Channel(12000, ic.preset("poor"))
    y = ch.apply(x, rng=np.random.default_rng(1))   # x: (..., n) real or complex
    y = ic.awgn(y, snr_db=0, fs=12000, rng=rng, signal_power=ic.power(x))
    y = ic.qrn(y, snr_db=6, vd_db=8, fs=12000, rng=rng, signal_power=ic.power(x))
    y = ic.Receiver.ssb(12000, mistune_hz=rng.uniform(-150, 150, len(x))).apply(y)

The same model as the Rust crate, but whole arrays at a time: fractional
delays in the frequency domain (no latency), each leading index (a batch
row) its own fading realisation. Seeds do not give the same numbers as
the Rust crate; the statistics are the same (both test against theory).
"""
from .channel import Channel, Component, Path, awgn, power, scatter, specular, tap_gain
from .presets import names, preset, presets
from .interference import Arcing, Smps, arcing, arcing_noise, smps, smps_noise
from .noise import qrn, qrn_noise, vd_bandwidth
from .receiver import Agc, Receiver, bandpass_taps, clock_offset
from .recorded import band_density, mix_noise, read_wav, recorded_noise, resample
from .vary import Drift, Ramp, Sine, Vary, tid
from .stations import Am, Cw, Radar, am, am_noise, cw, cw_noise, radar, radar_noise, speech_like

__all__ = ["Channel", "Component", "Path", "awgn", "power", "scatter", "specular", "tap_gain", "names", "preset", "presets",
           "Agc", "Receiver", "bandpass_taps", "clock_offset", "qrn", "qrn_noise", "vd_bandwidth",
           "Arcing", "Smps", "arcing", "arcing_noise", "smps", "smps_noise",
           "Am", "Cw", "Radar", "am", "am_noise", "cw", "cw_noise", "radar", "radar_noise", "speech_like",
           "band_density", "mix_noise", "read_wav", "recorded_noise", "resample",
           "Drift", "Ramp", "Sine", "Vary", "tid"]
__version__ = "0.1.0"
