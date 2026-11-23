"""QRN against its calibration (as tests/qrn.rs).
Run: python3 python/tests/test_noise.py   (or pytest)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import ionochannel as ic  # noqa: E402
from ionochannel import noise  # noqa: E402


def vd(z, fs, bw):
    """Vd (dB) in an ideal band of bw Hz around 0, per row pooled."""
    n = z.shape[-1]
    Z = np.fft.fft(z, axis=-1)
    Z[..., np.abs(np.fft.fftfreq(n, 1 / fs)) > bw / 2] = 0
    e = np.abs(np.fft.ifft(Z, axis=-1)).ravel()
    return 20 * np.log10(np.sqrt(np.mean(e ** 2)) / np.mean(e))


def test_vd_in_200_hz_and_other_bandwidths():
    fs = 4000.0
    for target in (3.0, 8.0):
        z = ic.qrn_noise(4, int(fs * 150), fs, target, np.random.default_rng(5))
        got = vd(z, fs, 200)
        assert abs(got - target) < 0.5, (target, got)
        wide, want = vd(z, fs, 2000), ic.vd_bandwidth(target, 2000)
        assert abs(wide - want) < 1.0, (target, wide, want)


def test_power_shape_dtype_and_gaussian_limit():
    rng = np.random.default_rng(6)
    z = ic.qrn_noise((2, 3), 8000 * 50, 8000.0, 4.0, rng)
    assert z.shape == (2, 3, 400000) and np.iscomplexobj(z)
    assert abs(np.mean(np.abs(z) ** 2) - 1) < 0.1, np.mean(np.abs(z) ** 2)
    r = ic.qrn_noise(3, 1000, 8000.0, 4.0, rng, complex_=False)
    assert r.shape == (3, 1000) and not np.iscomplexobj(r)
    g = ic.qrn_noise(2, 400000, 4000.0, noise.VD_GAUSS, rng)
    assert abs(vd(g, 4000.0, 200) - noise.VD_GAUSS) < 0.05


def test_qrn_level_matches_awgn():
    fs = 8000.0
    x = np.zeros((4, int(fs * 100)), np.float32)
    p = np.ones((4, 1))
    y = ic.qrn(x, 0.0, 4.0, fs, np.random.default_rng(7), bw_hz=2500, signal_power=p)
    w = ic.awgn(x, 0.0, fs, np.random.default_rng(7), bw_hz=2500, signal_power=p)
    assert y.dtype == np.float32
    assert abs(np.mean(y ** 2) / np.mean(w ** 2) - 1) < 0.1


def test_params_and_conversion():
    r4, g4 = noise.qrn_params(4.0)
    r10, g10 = noise.qrn_params(10.0)
    assert r4 > r10 and g4 > g10
    assert abs(ic.vd_bandwidth(10.0, 200) - 10) < 1e-12
    assert abs(ic.vd_bandwidth(10.0, 200e3) - (10 + 3 * 2.59)) < 1e-9
    try:
        noise.qrn_params(20.0)
        raise AssertionError("no error above the table")
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
