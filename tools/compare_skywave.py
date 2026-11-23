"""Cross-check against skywave (github.com/openarq-project/skywave,
Apache-2.0): the same real-audio measurements through skywave's
WattersonChannel, this package and the Rust command line.

  python3 tools/compare_skywave.py SKYWAVE_SRC_DIR [IONOCHANNEL_BIN]

Per preset: share of time >= 10 dB under the mean (Rayleigh 9.5 %), the
measured Doppler spread (2 sigma, from the gain's autocorrelation), and
the correlation of two tones 1/(4 delay) apart (theory cos(pi/4) = 0.71).
"""
import os
import subprocess
import sys
import tempfile
import wave

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
import ionochannel as ic  # noqa: E402

FS = 12000
TAU = 2 * np.pi


def analytic(y):
    Y = np.fft.fft(y)
    f = np.fft.fftfreq(len(y))
    return np.fft.ifft(Y * np.where(f > 0, 2, np.where(f == 0, 1, 0)))


def gain_at(y, f):
    t = np.arange(len(y)) / FS
    g = analytic(y) * np.exp(-1j * TAU * f * t)
    k = 60                                   # 200 Hz rate: well over the spreads used
    g = g[: len(g) // k * k].reshape(-1, k).mean(1)
    return g[50:-50]


def stats(g1, g2, spread):
    p = np.abs(g1) ** 2
    below = np.mean(p < 0.1 * p.mean())
    sigma = spread / 2
    lag = max(1, int(round(np.sqrt(-np.log(0.5) / 2) / (np.pi * sigma) * 200)))
    r = np.mean(g1[lag:] * np.conj(g1[:-lag])) / np.mean(p)
    meas = 2 * np.sqrt(-np.log(abs(r)) / 2) / (np.pi * lag / 200)
    rho = abs(np.mean(g1 * np.conj(g2))) / np.sqrt(np.mean(p) * np.mean(np.abs(g2) ** 2))
    return below, meas, rho


def tones(f1, f2, secs):
    t = np.arange(int(FS * secs)) / FS
    return np.cos(TAU * f1 * t), np.cos(TAU * f2 * t)


def run_skywave(W, x, d, s, seed):
    ch = W.WattersonChannel(FS, d, s, len(x) / FS + 5, seed=seed)
    return np.concatenate([ch.process(b) for b in np.array_split(x, len(x) // 1024)])


def run_rust(binary, x, preset, seed):
    with tempfile.TemporaryDirectory() as tmp:
        a, b = os.path.join(tmp, "in.wav"), os.path.join(tmp, "out.wav")
        with open(a, "wb") as f:
            hdr = b"RIFF" + (36 + 4 * len(x)).to_bytes(4, "little") + b"WAVEfmt " + (16).to_bytes(4, "little")
            hdr += (3).to_bytes(2, "little") + (1).to_bytes(2, "little") + FS.to_bytes(4, "little")
            hdr += (FS * 4).to_bytes(4, "little") + (4).to_bytes(2, "little") + (32).to_bytes(2, "little")
            f.write(hdr + b"data" + (4 * len(x)).to_bytes(4, "little") + x.astype("<f4").tobytes())
        subprocess.run([binary, "--preset", preset, "--seed", str(seed), a, b], check=True, capture_output=True)
        return np.frombuffer(open(b, "rb").read()[44:], "<f4").astype(np.float64)


def main():
    sys.path.insert(0, sys.argv[1])
    from skywave import watterson as W
    binary = sys.argv[2] if len(sys.argv) > 2 else None
    print(f"{'preset':16} {'d ms':>5} {'2s Hz':>6} | {'impl':9} {'<-10dB':>7} {'spread':>7} {'rho':>5}")
    for name in ["good", "moderate", "poor", "flutter", "high-moderate", "high-disturbed"]:
        paths = ic.preset(name)
        d, s = paths[1].delay_s * 1e3, paths[0].components[0].spread_hz
        secs = min(max(3000 / s, 300), 3000)
        f1 = 1000.0
        f2 = f1 + 1 / (4 * d * 1e-3)
        x1, x2 = tones(f1, f2, secs)
        rows = {}
        rows["skywave"] = [gain_at(run_skywave(W, x, d, s, 11), f) for x, f in ((x1, f1), (x2, f2))]
        rows["python"] = [gain_at(ic.Channel(FS, paths).apply(x, np.random.default_rng(11)), f) for x, f in ((x1, f1), (x2, f2))]
        if binary:
            rows["rust"] = [gain_at(run_rust(binary, x, name, 11), f) for x, f in ((x1, f1), (x2, f2))]
        for impl, (g1, g2) in rows.items():
            b, m, r = stats(g1, g2, s)
            print(f"{name:16} {d:5.1f} {s:6.1f} | {impl:9} {100 * b:6.1f}% {m:7.2f} {r:5.2f}")
    print("theory: 9.5 % under -10 dB; spread as given; rho 0.71")


if __name__ == "__main__":
    main()
