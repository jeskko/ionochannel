"""The Python receiver against its design (as tests/receiver.rs).
Run: python3 python/tests/test_receiver.py   (or pytest)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import ionochannel as ic  # noqa: E402

TAU = 2 * np.pi
FS = 12000.0


def amp(y, f, skip=2000):
    t = np.arange(y.shape[-1]) / FS
    a = np.mean(y[..., skip:-skip] * np.exp(-1j * TAU * f * t[skip:-skip]), axis=-1)
    return np.abs(a) * (1 if np.iscomplexobj(y) else 2)


def db(x):
    return 20 * np.log10(x)


def test_passband_response():
    rx = ic.Receiver(FS, (300.0, 2700.0))
    t = np.arange(24000) / FS
    for f, want in [(350, 0), (1500, 0), (2650, 0), (300, -6.02), (2700, -6.02)]:
        g = db(amp(rx.apply(np.exp(1j * TAU * f * t)), f))
        assert abs(g - want) < (0.02 if want == 0 else 0.1), (f, g)
        g = db(amp(rx.apply(np.cos(TAU * f * t)), f))
        assert abs(g - want) < (0.02 if want == 0 else 0.1), ("real", f, g)
    for f in (250, 2750, 0, -300, -1500, 4000):
        g = db(amp(rx.apply(np.exp(1j * TAU * f * t)), f))
        assert g < -ic.receiver.STOPBAND_DB + 1, (f, g)


def test_no_latency():
    rx = ic.Receiver(FS, (300.0, 2700.0))
    x = np.exp(1j * TAU * 1000 * np.arange(12000) / FS)
    y = rx.apply(x)
    assert np.max(np.abs(y - x)[3000:-3000]) < 2e-3


def test_mistuning_per_row():
    t = np.arange(24000) / FS
    x = np.stack([np.cos(TAU * 1000 * t)] * 2).astype(np.float32)
    y = ic.Receiver(FS, mistune_hz=[150.0, -150.0]).apply(x)
    assert y.dtype == np.float32
    assert abs(amp(y[0], 1150) - 1) < 0.01 and amp(y[0], 1000) < 0.01 and amp(y[0], 850) < 0.01
    assert abs(amp(y[1], 850) - 1) < 0.01 and amp(y[1], 1150) < 0.01


def agc_env(agc, levels):
    x = np.concatenate([np.full(int(s * FS), a) for a, s in levels])
    x = x * np.exp(1j * TAU * 700 * np.arange(len(x)) / FS)
    return np.abs(ic.Receiver(FS, agc=agc).apply(x))


def recovers(env, target, start):
    i0 = int(start * FS)
    return np.argmax(env[i0:] >= target * 10 ** (-1 / 20)) / FS


def test_agc_level_attack_decay_hang():
    a = ic.Agc()
    env = agc_env(a, [(1.0, 1.0), (10.0, 1.0), (1.0, 3.0)])
    near = lambda s, tol: np.all(np.abs(db(env[s] / a.target)) < tol)
    assert near(slice(100, 12000), 0.01)
    up = 12000
    assert env[up + 2] > 2 * a.target
    assert near(slice(up + 72, up + 12000), 1.0) and near(slice(up + 600, up + 12000), 0.01)
    want = a.decay_s * np.log(10 / 10 ** (1 / 20))
    t = recovers(env, a.target, 2.0)
    assert abs(t - want) < 0.003, (t, want)
    env = agc_env(ic.Agc(hang_s=0.2), [(10.0, 1.0), (1.0, 3.0)])
    t = recovers(env, a.target, 1.0)
    assert abs(t - want - 0.2) < 0.003, (t, want + 0.2)
    env = agc_env(a, [(1e-5, 0.5)])
    assert abs(db(env[3000] / 1e-5) - a.max_gain_db) < 0.01


def test_rows_independent_agc():
    t = np.arange(12000) / FS
    x = np.stack([np.exp(1j * TAU * 500 * t), 0.01 * np.exp(1j * TAU * 500 * t)])
    y = np.abs(ic.Receiver(FS, agc=ic.Agc()).apply(x))
    assert np.allclose(y[:, 1000:], 0.25, rtol=1e-3)


def test_clock_offset():
    t = np.arange(int(FS * 10)) / FS
    for ppm in (100.0, -250.0):
        y = ic.clock_offset(np.exp(1j * TAU * 1000 * t)[None], ppm)[0]
        fi = np.angle(y[1:] * np.conj(y[:-1])) * FS / TAU
        assert abs(fi[100:-1000].mean() - 1000 / (1 + ppm * 1e-6)) < 1e-4
        assert np.max(np.abs(np.abs(y[100:-1000]) - 1)) < 2e-4
    y = ic.Receiver(FS, clock_ppm=[50.0, -50.0]).apply(np.cos(TAU * 1000 * t)[None].repeat(2, 0))
    assert y.shape == (2, len(t))


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
