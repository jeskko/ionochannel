"""The Python channel against theory (as tests/stats.rs for the crate).
Run: python3 python/tests/test_stats.py   (or pytest)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import ionochannel as ic  # noqa: E402

TAU = 2 * np.pi


def gains(fs, paths, secs, seed, rows=1):
    rng = np.random.default_rng(seed)
    x = np.ones((rows, int(fs * secs)), np.complex128)
    return ic.Channel(fs, paths).apply(x, rng)


def acf(g, lag):
    return np.mean(g[..., lag:] * np.conj(g[..., :-lag])) / np.mean(np.abs(g) ** 2)


def test_rayleigh_power_and_depths():
    g = gains(200.0, [ic.scatter(0, 1.0, 1.0)], 2000.0, 1, rows=4)
    p = np.abs(g) ** 2
    m = p.mean()
    assert abs(m - 1) < 0.05, m
    for x, tol in [(0.1, 0.01), (0.01, 0.002), (1.0, 0.02)]:
        f = np.mean(p < x * m)
        assert abs(f - (1 - np.exp(-x))) < tol, (x, f)


def test_doppler_spread_and_shift():
    for spread, shift in [(0.1, 0.0), (0.5, 0.0), (1.0, 3.0), (10.0, -5.0), (30.0, 0.0), (55.0, 12.0)]:
        fs = max(16 * spread + 4 * abs(shift), 20.0)
        secs = min(4000.0 / spread, 4000.0)
        g = gains(fs, [ic.scatter(0, 1.0, spread, shift)], secs, 2)[0]
        sigma = spread / 2
        for target in (0.7, 0.3):
            lag = int(round(np.sqrt(-np.log(target) / 2) / (np.pi * sigma) * fs))
            r = acf(g, lag)
            s = np.sqrt(-np.log(abs(r)) / 2) / (np.pi * lag / fs)
            assert abs(2 * s / spread - 1) < 0.06, (spread, target, 2 * s)
        lag = max(1, int(round(0.05 / (spread + abs(shift)) * fs)))
        f = np.angle(acf(g, lag)) / (TAU * lag / fs)
        assert abs(f - shift) < 0.03 * spread + 0.01, (shift, f)


def test_level_crossings_and_fade_durations():
    spread, fs = 1.0, 100.0
    g = gains(fs, [ic.scatter(0, 1.0, spread)], 5000.0, 3, rows=4)
    rms = np.sqrt(np.mean(np.abs(g) ** 2))
    secs = g.shape[-1] / fs * g.shape[0]
    for rho in (0.1, 0.316, 1.0):
        below = np.abs(g) < rho * rms
        ups = np.sum(below[:, :-1] & ~below[:, 1:])
        n = ups / secs
        nt = np.sqrt(4 * np.pi) * spread / 2 * rho * np.exp(-rho ** 2)
        assert abs(n / nt - 1) < 0.08, (rho, n, nt)
        afd = below.sum() / fs / ups
        at = (1 - np.exp(-rho ** 2)) / nt
        assert abs(afd / at - 1) < 0.08, (rho, afd, at)


def test_two_paths_frequency_correlation():
    fs, tau = 2000.0, 2e-3
    paths = [ic.scatter(0, 0.5, 1.0), ic.scatter(tau, 0.5, 1.0)]
    n = int(fs * 3000)
    t = np.arange(n) / fs
    for df in (62.5, 125.0, 250.0, 500.0):
        x = np.stack([np.exp(1j * TAU * 300 * t), np.exp(1j * TAU * (300 + df) * t)])
        # same realisation for both tones: one row, two runs with the same seed
        h0 = ic.Channel(fs, paths).apply(x[:1], np.random.default_rng(4))[0] * np.exp(-1j * TAU * 300 * t)
        h1 = ic.Channel(fs, paths).apply(x[1:], np.random.default_rng(4))[0] * np.exp(-1j * TAU * (300 + df) * t)
        rho = abs(np.mean(h0 * np.conj(h1))) / np.sqrt(np.mean(abs(h0) ** 2) * np.mean(abs(h1) ** 2))
        assert abs(rho - abs(np.cos(np.pi * df * tau))) < 0.06, (df, rho)


def test_real_shift_moves_audio_up():
    fs, n = 12000.0, 24000
    x = np.cos(TAU * 1000 * np.arange(n) / fs)
    y = ic.Channel(fs, [ic.specular(0, 1.0, 10.0)]).apply(x, np.random.default_rng(7))
    t = np.arange(n) / fs
    amp = lambda f: 2 * abs(np.mean(y[2000:-2000] * np.exp(-1j * TAU * f * t[2000:-2000])))
    assert abs(amp(1010) - 1) < 0.01 and amp(990) < 0.01, (amp(1010), amp(990))


def test_fractional_delay():
    fs, d = 12000.0, 0.37e-3
    n = 12000
    t = np.arange(n) / fs
    for f in (300.0, 1500.0, 4500.0):
        x = np.exp(1j * TAU * f * t)
        y = ic.Channel(fs, [ic.specular(d, 1.0)]).apply(x, np.random.default_rng(6))
        r = (y / np.exp(1j * TAU * f * (t - d)))[1500:-1500]  # the ideal delay rings ~1/k at the edges
        assert np.max(np.abs(r - r.mean())) < 2e-3 and abs(abs(r.mean()) - 1) < 2e-3, f


def test_awgn_snr():
    fs = 12000.0
    rng = np.random.default_rng(8)
    x = np.exp(1j * TAU * 1000 * np.arange(120000) / fs)
    y = ic.awgn(x, 0.0, fs, rng, bw_hz=2500)
    nz = y - x
    p_in_bw = np.mean(np.abs(nz) ** 2) * 2500 / fs
    assert abs(10 * np.log10(1 / p_in_bw)) < 0.1


def test_measured_presets_80_percent_width():
    """The Stocker 2013 and Angling 1998 presets give their 80 %-power width back."""
    for name, w80 in (("polar-cap-14mhz", 13.0), ("polar-cap-5mhz", 3.5), ("trough-5mhz", 1.5),
                      ("auroral-7-11mhz", 30.3), ("auroral-14-22mhz", 54.6)):
        s = ic.preset(name)[1].components[0].spread_hz
        sigma = s / 2
        f = np.linspace(-6 * sigma, 6 * sigma, 200001)
        psd = np.exp(-f ** 2 / (2 * sigma ** 2))
        c = np.cumsum(psd) / psd.sum()
        width = f[np.searchsorted(c, 0.9)] - f[np.searchsorted(c, 0.1)]
        assert abs(width - w80) < 0.01, (name, width)


def test_more_measured_presets():
    """The conversions behind the equatorial, NVIS and MITRE presets."""
    eq = ic.preset("equatorial-spread-f")
    tau, sp = eq[1].delay_s, eq[1].components[0].spread_hz
    assert abs(abs(np.cos(np.pi * 1500 * tau)) - 0.5) < 0.01
    assert abs(sp * np.sqrt(2 * np.log(10)) - 6.0) < 0.02          # -10 dB width of the Gaussian
    nv = ic.preset("nvis-java-7mhz")
    assert abs(nv[1].delay_s / 2 - 0.4627e-3) < 1e-6 and nv[0].components[0].shift_hz == 0.9
    for name, tau, spread in (("high-latitude-wb-median", 0.1e-3, 5.0), ("high-latitude-wb-90", 0.4e-3, 13.0)):
        p = ic.preset(name)
        assert abs(p[1].delay_s - tau) < 1e-12 and p[0].components[0].spread_hz == spread


def test_presets():
    assert len(ic.names()) >= 16
    for k, v in ic.presets().items():
        assert abs(sum(p.power for p in v["paths"]) - 1) < 1e-9, k
        assert v["source"], k


if __name__ == "__main__":
    bad = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("ok  ", name)
            except AssertionError as e:
                bad += 1
                print("FAIL", name, e)
    sys.exit(1 if bad else 0)
