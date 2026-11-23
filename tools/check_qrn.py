"""Check the QRN model against ITU-R P.372-18 (a different seed from the
calibration). Run from the repository root:
    nice -n 10 python3 tools/check_qrn.py     (about 2 min on 12 cores)

Targets, digitised from the figures' rasters on 2026-11-23:
- Fig. 37, APD in 200 Hz, dB above the rms envelope, at the percentage of
  time exceeded. The large-P side reads to about 0.3 dB (the Vd = 1.049
  curve is within 0.3 dB of Rayleigh); the small-P side only for Vd 2
  and 4, to about 0.5 dB. The middle, where the curves cross, and the
  small-P side of the higher curves were too crowded to read.
- Fig. 38, Vd in bandwidth b: Vd(200) + slope * log10(b / 200), slopes in
  qrn.json (the lines are straight to 0.05 dB/decade)."""
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
from ionochannel import noise  # noqa: E402

FS, SECS, ROWS = 12000.0, 100, 8
P = [0.98, 0.9, 0.7, 0.5, 0.3, 3e-3, 1.1e-4]
F37 = {2: [-19.8, -12.8, -7.2, -3.9, -0.8, 11.6, 15.9],
       4: [-24.7, -17.4, -11.6, -7.9, -3.9, None, 21.9],
       6: [-29.3, -21.5, -15.4, -11.6, -7.4, None, None],
       8: [-33.5, -25.8, -19.3, -15.2, -10.4, None, None],
       10: [-37.5, -29.8, -23.0, -18.4, None, None, None]}


def measure(v):
    rng = np.random.default_rng(12345)
    n = int(FS * SECS)
    Z = np.fft.fft(noise.qrn_noise(ROWS, n, FS, v, rng), axis=-1)
    f = np.abs(np.fft.fftfreq(n, 1 / FS))
    out = {}
    for b in (20, 200, 2000, 6000):
        Zb = np.where(f <= b / 2, Z, 0)
        e = np.abs(np.fft.ifft(Zb, axis=-1)).ravel()
        out[b] = 20 * np.log10(np.sqrt(np.mean(e ** 2)) / np.mean(e))
        if b == 200:
            s = np.sort(e / np.sqrt(np.mean(e ** 2)))[::-1]
            out["apd"] = [20 * np.log10(s[int(p * len(s))]) for p in P]
    return v, out


if __name__ == "__main__":
    vds = [1.5, 2, 3, 4, 6, 8, 10, 12, 14]
    with Pool(max(1, os.cpu_count() - 4)) as p:
        res = p.map(measure, vds)
    e37, e38 = [], []
    for v, o in res:
        d38 = {b: o[b] - noise.vd_bandwidth(v, b) for b in (20, 2000, 6000)}
        e38 += [abs(x) for x in d38.values()]
        line = f"Vd {v:4}: got {o[200]:5.2f} in 200 Hz | Fig. 38 error " + " ".join(f"{b} Hz {d:+.1f}" for b, d in d38.items())
        if v in F37:
            d37 = [round(float(a - t), 1) for a, t in zip(o["apd"], F37[v]) if t is not None]
            e37 += [abs(x) for x in d37]
            line += f" | Fig. 37 error {d37}"
        print(line)
    print(f"mean |error|: Fig. 37 {np.mean(e37):.2f} dB (max {max(e37):.1f}), Fig. 38 {np.mean(e38):.2f} dB (max {max(e38):.1f})")
