"""Paths that change over time (as tests/vary.rs).
Run: python3 python/tests/test_vary.py   (or pytest)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import ionochannel as ic  # noqa: E402
from ionochannel.vary import C  # noqa: E402

TAU = 2 * np.pi


def run(fs, paths, vary, x, seed):
    return ic.Channel(fs, paths, vary=vary).apply(x, np.random.default_rng(seed))


def inst_freq(z, fs):
    return np.angle(z[..., 1:] * np.conj(z[..., :-1])) * fs / TAU


def test_doppler_follows_a_sine():
    fs = 1000.0
    y = run(fs, [ic.specular(0, 1.0)], [ic.Vary(shift_hz=(ic.Sine(5.0, 1.0),))], np.ones((1, 4000), complex), 1)[0]
    want = 5 * np.sin(TAU * (np.arange(3999) + 0.5) / fs)
    assert np.max(np.abs(inst_freq(y, fs) - want)) < 0.05


def test_power_ramps():
    fs = 1000.0
    v = ic.Vary(power_db=(ic.Ramp(1.0, 2.0, 0.0, -20.0),))
    a = np.abs(run(fs, [ic.specular(0, 1.0)], [v], np.ones((1, 3000), complex), 2)[0])
    assert abs(a[500] - 1) < 1e-5 and abs(a[2500] - 0.1) < 1e-5 and abs(a[1500] - 10 ** -0.5) < 1e-3


def spread(g, fs, guess):
    sigma = guess / 2
    lag = int(round(np.sqrt(np.log(0.5) / -2) / (np.pi * sigma) * fs))
    r = np.mean(g[..., lag:] * np.conj(g[..., :-lag])) / np.mean(np.abs(g) ** 2)
    return 2 * np.sqrt(-np.log(abs(r)) / 2) / (np.pi * lag / fs)


def test_spread_steps_up_an_octave():
    fs, secs = 100.0, 2000.0
    v = ic.Vary(spread_oct=(ic.Ramp(secs / 2, secs / 2, 0.0, 1.0),))
    y = run(fs, [ic.scatter(0, 1.0, 1.0)], [v] , np.ones((4, int(fs * secs)), complex), 3)
    a, b = y[:, : y.shape[1] // 2], y[:, y.shape[1] // 2:]
    s1, s2 = spread(a, fs, 1.0), spread(b, fs, 2.0)
    assert abs(s1 - 1) < 0.06 and abs(s2 - 2) < 0.12, (s1, s2)
    assert abs(np.mean(np.abs(a) ** 2) - 1) < 0.08 and abs(np.mean(np.abs(b) ** 2) - 1) < 0.08


def test_delay_follows_a_sine_and_matches_fft():
    fs, f = 12000.0, 1000.0
    t = np.arange(int(fs * 4)) / fs
    x = np.exp(1j * TAU * f * t)[None]
    v = ic.Vary(delay_s=(ic.Sine(0.2e-3, 2.0),))
    y = run(fs, [ic.specular(1e-3, 1.0)], [v], x, 4)[0]
    tau = 1e-3 + 0.2e-3 * np.sin(TAU * t / 2)
    got = np.angle(y * np.conj(x[0]))
    d = got - (-TAU * f * tau)
    d = d - d[2000]
    d = (d + np.pi) % TAU - np.pi
    assert np.max(np.abs(d[2000:])) < 0.01, np.max(np.abs(d[2000:]))
    assert np.max(np.abs(np.abs(y[2000:]) - 1)) < 2e-3
    # a delay that "moves" by nothing equals the FFT delay
    z = np.exp(1j * TAU * 1500 * t)[None] * (1 + 0.3 * np.cos(TAU * 3 * t))
    v0 = ic.Vary(delay_s=(ic.Ramp(0, 1, 0.0, 0.0),))
    a = ic.vary.moving_delay(z.astype(np.complex128), np.full(z.shape, 0.37e-3 * fs))
    b = ic.Channel(fs, [ic.specular(0.37e-3, 1.0)]).apply(z, np.random.default_rng(5))
    b = b * np.exp(-1j * np.angle(np.mean(b[0, 3000:-3000] * np.conj(a[0, 3000:-3000]))))
    assert np.max(np.abs(a - b)[0, 3000:-3000]) < 2e-3
    assert run(fs, [ic.specular(0.37e-3, 1.0)], [v0], z, 6).shape == z.shape


def test_drift_rms():
    fs = 200.0
    v = ic.Vary(shift_hz=(ic.Drift(2.0, 1.0),))
    y = run(fs, [ic.specular(0, 1.0)], [v], np.ones((8, int(fs * 300)), complex), 7)
    sd = inst_freq(y, fs).std()
    assert abs(sd / 2 - 1) < 0.1, sd


def test_tid():
    v = ic.tid(10e6, 1000.0, 0.05, 3600.0, 0.3)
    assert abs(v.delay_s[0].amp - 50e3 / C) < 1e-12
    assert abs(v.shift_hz[0].amp - 10e6 * 50e3 * TAU / (C * 3600)) < 1e-9
    val = lambda s, t: s.amp * np.sin(TAU * t / s.period_s + s.phase)  # noqa: E731
    for t in (0.0, 500.0, 1234.0):
        dd = (val(v.delay_s[0], t + 0.01) - val(v.delay_s[0], t - 0.01)) / 0.02
        assert abs(val(v.shift_hz[0], t) + 10e6 * dd) < 1e-6


def test_table_replays_a_record():
    fs = 1000.0
    rec = ic.Table((1.0, 2.0, 3.0), (0.0, 4.0, -2.0))
    y = run(fs, [ic.specular(0, 1.0)], [ic.Vary(shift_hz=(rec,))], np.ones((1, 4000), complex), 8)[0]
    fi = inst_freq(y, fs)
    t = (np.arange(3999) + 0.5) / fs
    assert np.max(np.abs(fi - np.interp(t - 0.5 / fs, rec.t_s, rec.values))) < 0.01
    try:
        ic.Table((1.0, 1.0), (0.0, 1.0))
        raise AssertionError("no error for times that don't increase")
    except ValueError:
        pass


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
