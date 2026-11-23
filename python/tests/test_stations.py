"""Interfering stations against their parameters (as tests/stations.rs).
Run: python3 python/tests/test_stations.py   (or pytest)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import ionochannel as ic  # noqa: E402

TAU = 2 * np.pi
FS = 12000.0


def amp(z, f):
    t = np.arange(z.shape[-1]) / FS
    return np.abs(np.mean(z * np.exp(-1j * TAU * f * t), axis=-1))


def test_morse_timing():
    seq = ic.stations.morse_units("PARIS")
    assert sum(u for _, u in seq) == 50 and sum(u for d, u in seq if d) == 22
    z = ic.cw_noise(2, int(FS * 30), FS, 800.0, np.random.default_rng(1), ic.Cw(wpm=20, text="PARIS"))
    p = np.abs(z) ** 2
    assert abs(p.max() - 1) < 1e-6
    assert abs(p.mean() - 0.44) < 0.01, p.mean()
    # PARIS repeats every 50 units = 3 s at 20 wpm
    e = p[0] - p[0].mean()
    lags = np.arange(int(2.5 * FS), int(3.5 * FS))
    r = [np.dot(e[:-l], e[l:]) for l in lags[::12]]
    assert abs(lags[::12][int(np.argmax(r))] / FS - 3.0) < 0.002
    assert amp(z[0] / np.sqrt(p[0] + 1e-12) * (p[0] > 0.99), 800) > 0.4


def test_cw_rise_and_real():
    z = ic.cw_noise(1, int(FS * 5), FS, 700.0, np.random.default_rng(2), ic.Cw(rise_s=0.005))[0]
    a = np.abs(z)
    up = np.flatnonzero((a[1:] > 0.9) & (a[:-1] <= 0.9))[0]
    start = np.flatnonzero(a[:up] < 0.1)[-1]
    assert abs((up - start) / FS - 0.005 * 0.59) < 0.0002    # 10-90 % of a raised-cosine edge
    x = np.zeros((1, int(FS * 20)), np.float32)
    y = ic.cw(x, -6.0, 700.0, FS, np.random.default_rng(3), ic.Cw(text="PARIS"), signal_power=1.0)
    assert y.dtype == np.float32 and abs(np.mean(y ** 2) / (0.44 * 10 ** -0.6) - 1) < 0.03


def test_am_carrier_and_bandwidth():
    z = ic.am_noise(2, int(FS * 10), FS, 1000.0, np.random.default_rng(4), ic.Am(audio_hz=3000))
    assert np.allclose(amp(z, 1000), 1, atol=0.01)
    side = np.mean(np.abs(z) ** 2) - 1
    assert 0.05 < side < 0.1, side
    Z = np.mean(np.abs(np.fft.fft(z, axis=-1)) ** 2, 0)
    f = np.fft.fftfreq(z.shape[-1], 1 / FS)
    inband = Z[np.abs(f - 1000) <= 3000].sum()
    outband = Z[np.abs(f - 1000) > 3600].sum()
    assert outband / inband < 1e-5, outband / inband
    t = np.arange(int(FS * 2)) / FS
    tone = np.sin(TAU * 400 * t)
    z = ic.am_noise(1, len(t), FS, 0.0, np.random.default_rng(5), ic.Am(depth=1.0), audio=tone)[0]
    c, s = amp(z, 0), amp(z, 400)
    assert abs(s / c - 0.3 * np.sqrt(2) / 2) < 0.01, s / c         # tone scaled to 0.3 rms


def test_am_real_does_not_fold():
    t = np.arange(int(FS * 4)) / FS
    x = np.zeros((1, len(t)))
    y = ic.am(x, 0.0, 500.0, FS, np.random.default_rng(6), ic.Am(), audio=np.sin(TAU * 1500 * t),
              signal_power=1.0)[0]
    k = np.sqrt(2) / 2
    assert abs(amp(y, 500) / k - 1) < 0.01
    assert abs(amp(y, 2000) / k - 0.3 * np.sqrt(2) / 2) < 0.01
    assert amp(y, 1000) / k < 1e-3, amp(y, 1000) / k


def test_radar_sweep():
    spec = ic.Radar(sweep_hz=20e3, rate_hz=50.0)
    z = ic.radar_noise(1, int(FS * 2), FS, 0.0, np.random.default_rng(7), spec)[0]
    a = np.abs(z)
    full = (a[1:] > 0.999) & (a[:-1] > 0.999)
    fi = np.angle(z[1:] * np.conj(z[:-1])) * FS / TAU
    slope = np.diff(fi)[full[1:] & full[:-1]] * FS
    assert abs(np.median(slope) / 1e6 - 1) < 0.01
    frac = np.mean(a > 0.999)
    assert abs(frac - 12000 * 0.9 / 20e3) < 0.02, frac


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
