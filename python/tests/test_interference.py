"""Man-made interference against its parameters (as tests/interference.rs).
Run: python3 python/tests/test_interference.py   (or pytest)."""
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


def test_smps_lines_and_ripple():
    spec = ic.Smps(spacing_hz=2500, am_depth=0.0, fm_dev_hz=0.0, drift_hz=0.0)
    z = ic.smps_noise(2, int(FS * 4), FS, 700.0, np.random.default_rng(1), spec)
    lines = ic.interference.smps_lines(FS, 700.0, 2500, False)
    assert list(lines) == [-4300, -1800, 700, 3200, 5700]
    for f in lines:
        assert np.allclose(amp(z, f), 1, atol=1e-6), f
    assert np.all(amp(z, 700 + 1250) < 1e-3)
    # AM sidebands at +-2 mains, m/2 against the carrier
    m = 0.4
    z = ic.smps_noise(1, int(FS * 4), FS, 700.0, np.random.default_rng(2),
                      ic.Smps(spacing_hz=1e6, am_depth=m, fm_dev_hz=0.0, drift_hz=0.0))[0]
    c, s1, s2 = amp(z, 700), amp(z, 800), amp(z, 600)
    assert abs(s1 / c - m / 2) < 1e-3 and abs(s2 / c - m / 2) < 1e-3, (c, s1, s2)
    assert abs(np.mean(np.abs(z) ** 2) - 1) < 1e-3
    # FM: deviation from the instantaneous frequency
    z = ic.smps_noise(1, int(FS * 4), FS, 700.0, np.random.default_rng(3),
                      ic.Smps(spacing_hz=1e6, am_depth=0.0, fm_dev_hz=25.0, drift_hz=0.0))[0]
    fi = np.angle(z[1:] * np.conj(z[:-1])) * FS / TAU
    assert abs((fi.max() - fi.min()) / 2 - 25) < 0.5 and abs(fi.mean() - 700) < 0.5


def test_smps_drift_and_real():
    spec = ic.Smps(spacing_hz=1e6, am_depth=0.0, fm_dev_hz=0.0, drift_hz=30.0, drift_s=1.0)
    z = ic.smps_noise(4, int(1000 * 200), 1000.0, 100.0, np.random.default_rng(4), spec)
    fi = np.angle(z[:, 1:] * np.conj(z[:, :-1])) * 1000.0 / TAU - 100
    assert abs(fi.std() / 30 - 1) < 0.1, fi.std()
    x = np.zeros((1, int(FS * 2)), np.float32)
    y = ic.smps(x, -10.0, 1000.0, FS, np.random.default_rng(5),
                ic.Smps(spacing_hz=1e6, am_depth=0, fm_dev_hz=0, drift_hz=0), signal_power=1.0)
    assert y.dtype == np.float32 and abs(np.mean(y ** 2) - 0.1) < 1e-3


def test_arcing_power_and_mains_lock():
    spec = ic.Arcing(mains_hz=50.0, jitter_s=1e-4, burst_s=1e-4, spread_db=4.0, off_s=0.0)
    z = ic.arcing_noise(4, int(FS * 60), FS, np.random.default_rng(6), spec)
    assert abs(np.mean(np.abs(z) ** 2) - 1) < 0.1, np.mean(np.abs(z) ** 2)
    # fold the power by the half-cycle: it sits within a millisecond of one phase
    half = int(FS / 100)
    p = (np.abs(z[0, : (z.shape[1] // half) * half]) ** 2).reshape(-1, half).sum(0)
    k = np.argmax(p)
    near = sum(p[(k + d) % half] for d in range(-12, 13))
    assert near / p.sum() > 0.9, near / p.sum()


def test_arcing_gate_and_level():
    spec = ic.Arcing(on_s=1.0, off_s=3.0)
    z = ic.arcing_noise(8, int(FS * 60), FS, np.random.default_rng(7), spec)
    half = int(FS / 100)
    e = (np.abs(z[:, : (z.shape[1] // half) * half]) ** 2).reshape(8, -1, half).sum(-1)
    on = np.mean(e > 0.01 * np.median(e[e > np.percentile(e, 90)]))
    assert abs(on - 0.25) < 0.05, on
    x = np.zeros((4, int(FS * 60)), np.float32)
    y = ic.arcing(x, 0.0, FS, np.random.default_rng(8), ic.Arcing(), signal_power=np.ones((4, 1)))
    w = ic.awgn(x, 0.0, FS, np.random.default_rng(8), signal_power=np.ones((4, 1)))
    assert abs(np.mean(y ** 2) / np.mean(w ** 2) - 1) < 0.1


def test_bursts_duty_and_level():
    spec = ic.Bursts(on_s=0.002, off_s=0.02)
    z = ic.bursts_noise(4, int(FS * 60), FS, np.random.default_rng(9), spec)
    on = np.abs(z) > 0
    assert abs(on.mean() - 2 / 22) < 0.01, on.mean()
    assert abs(np.mean(np.abs(z[on]) ** 2) - 1) < 0.02
    # mean length of the on runs (inside the record)
    lens = []
    for r in on:
        e = np.flatnonzero(np.diff(r.astype(int)))
        starts, ends = e[r[e + 1]] + 1, e[~r[e + 1]] + 1
        if len(ends) and len(starts) and ends[0] < starts[0]:
            ends = ends[1:]
        k = min(len(starts), len(ends))
        lens += list(ends[:k] - starts[:k])
    assert abs(np.mean(lens) / FS / 0.002 - 1) < 0.05, np.mean(lens) / FS


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
