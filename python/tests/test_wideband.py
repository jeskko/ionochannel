"""Vogler & Hoffmeyer layer-modes (as tests/wideband.rs).
Run: python3 python/tests/test_wideband.py   (or pytest)."""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import ionochannel as ic  # noqa: E402


def T(alpha, tell, tc, tau):
    g = (tau - tell) / (tc - tell)
    return math.exp(alpha * (math.log(g) + 1 - g))


def test_delay_shape():
    for tl, tc, tu, sv in ((1e-3, 1.3e-3, 2.6e-3, 0.1), (0.0, 22e-6 * 0.48, 22e-6, 0.01), (2e-3, 2.1e-3, 3.4e-3, 0.3)):
        a, tell = ic.vh_shape(tl, tc, tu, sv)
        assert abs(T(a, tell, tc, tl) - sv) < 1e-9 and abs(T(a, tell, tc, tu) - sv) < 1e-9
        assert abs(T(a, tell, tc, tc) - 1) < 1e-12


def test_mode_slant_width_and_channel():
    p = ic.vh_mode(1e-3, 1600e-6, 300e-6, -0.65, -0.8, 4.0, 0.1)
    assert abs(p[0].components[0].shift_hz + 0.8) < 1e-12
    peak = max(p, key=lambda q: q.power)
    assert abs(peak.delay_s - 1.3e-3) < 1600e-6 / 32
    s = p[0].components[0].spread_hz / 2
    assert abs(math.exp(-16 / (2 * s * s)) - 0.1) < 1e-9
    q = ic.normalize(ic.vh_mode(0.5e-3, 1400e-6, 300e-6, -1.6, -2.0, 9.0, 0.1))
    rng = np.random.default_rng(1)
    x = (rng.standard_normal((6, 120000)) + 1j * rng.standard_normal((6, 120000))) / np.sqrt(2)
    y = ic.Channel(12000.0, q).apply(x, rng)
    assert abs(np.mean(np.abs(y[:, 1000:]) ** 2) - 1) < 0.05


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
