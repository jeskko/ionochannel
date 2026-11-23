"""Recorded noise: level against awgn, band-limited recordings, seamless
loops, WAV reading, resampling (as tests/recorded.rs).
Run: python3 python/tests/test_recorded.py   (or pytest)."""
import os
import struct
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import ionochannel as ic  # noqa: E402

TAU = 2 * np.pi
FS = 12000.0


def test_white_recording_matches_awgn():
    rng = np.random.default_rng(1)
    rec = rng.standard_normal(int(FS * 30))
    x = np.zeros((8, int(FS * 4)))
    y = ic.mix_noise(x, rec, 0.0, FS, rng, signal_power=1.0)
    w = ic.awgn(x, 0.0, FS, rng, signal_power=1.0)
    assert np.allclose(np.mean(y ** 2, -1), np.mean(w ** 2), rtol=0.02), (np.mean(y ** 2, -1), np.mean(w ** 2))
    z = rng.standard_normal(int(FS * 30)) + 1j * rng.standard_normal(int(FS * 30))
    xc = np.zeros((4, int(FS * 4)), complex)
    yc = ic.mix_noise(xc, z, 0.0, FS, rng, signal_power=1.0)
    wc = ic.awgn(xc, 0.0, FS, rng, signal_power=1.0)
    assert np.allclose(np.mean(np.abs(yc) ** 2, -1), np.mean(np.abs(wc) ** 2), rtol=0.03)


def test_band_limited_recording():
    rng = np.random.default_rng(2)
    rec = ic.Receiver(FS, (300.0, 2700.0)).apply(rng.standard_normal(int(FS * 30)) * 7.0)
    x = np.zeros((4, int(FS * 4)))
    y = ic.mix_noise(x, rec, 6.0, FS, rng, signal_power=1.0)
    d = ic.band_density(y, FS, (300.0, 2700.0))
    assert np.allclose(d * 2500, 10 ** -0.6, rtol=1e-6)
    assert np.allclose(np.mean(y ** 2, -1), 10 ** -0.6 * 2400 / 2500, rtol=0.03)


def test_whole_recording_level():
    rng = np.random.default_rng(3)
    rec = rng.standard_normal(int(FS * 60)) * np.repeat([1.0, 3.0], int(FS * 30))
    x = np.zeros((64, int(FS * 1)))
    y = ic.mix_noise(x, rec, 0.0, FS, rng, signal_power=1.0, per_segment=False)
    p = np.mean(y ** 2, -1)
    assert p.max() / p.min() > 5                   # segments keep the recording's swings
    assert abs(np.mean(p) / 2.4 - 1) < 0.1


def test_loop_is_seamless():
    t = np.arange(int(FS * 0.5)) / FS
    rec = np.sin(TAU * 437.3 * t)                   # not a whole number of periods
    y = ic.recorded_noise(rec, 1, int(FS * 4), FS, np.random.default_rng(4))[0]
    assert np.max(np.abs(np.diff(y))) <= np.max(np.abs(np.diff(rec))) * 1.01


def test_read_wav_and_resample():
    rate = 8000
    x = np.sin(TAU * 500 * np.arange(rate) / rate) * 0.5
    with tempfile.TemporaryDirectory() as d:
        for tag, bits, data, ch in [
            (1, 16, (x * 32767).astype("<i2").tobytes(), 1),
            (3, 32, x.astype("<f4").tobytes(), 1),
            (3, 32, np.stack([x, -x], 1).astype("<f4").tobytes(), 2),
        ]:
            path = os.path.join(d, "t.wav")
            blk = ch * bits // 8
            with open(path, "wb") as f:
                f.write(b"RIFF" + struct.pack("<I", 36 + len(data)) + b"WAVEfmt "
                        + struct.pack("<IHHIIHH", 16, tag, ch, rate, rate * blk, blk, bits)
                        + b"data" + struct.pack("<I", len(data)) + data)
            y, fs = ic.read_wav(path)
            assert fs == rate and len(y) == rate
            want = x if ch == 1 else x - 1j * x
            assert np.max(np.abs(y - want)) < 1e-4, (tag, bits, ch)
    y = ic.resample(x, rate, 12000)
    t = np.arange(len(y)) / 12000
    a = 2 * np.abs(np.mean((y * np.exp(-1j * TAU * 500 * t))[1000:-1000]))
    assert len(y) == 12000 and abs(a - 0.5) < 1e-3


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
