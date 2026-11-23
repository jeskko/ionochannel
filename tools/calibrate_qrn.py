"""Calibrate the QRN burst rate for each Vd (200 Hz) and write the table
into python/ionochannel/qrn.json. Run from the repository root:
    nice -n 10 python3 tools/calibrate_qrn.py   (about 6 min on 12 cores)
Then check it against P.372 with python3 tools/check_qrn.py."""
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
from ionochannel import noise  # noqa: E402

FS, SECS, ROWS = 8000.0, 150, 8
VDS = [1.25, 1.5, 1.75, 2, 2.5, 3, 3.5, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]


def vd200(rate, g, seed):
    rng = np.random.default_rng(seed)
    n = int(FS * SECS)
    env = noise._envelope(ROWS, n, FS, rate, g, rng)
    w = (rng.standard_normal((ROWS, n)) + 1j * rng.standard_normal((ROWS, n))) * np.sqrt(0.5)
    Z = np.fft.fft(np.sqrt(env) * w, axis=-1)
    Z[:, np.abs(np.fft.fftfreq(n, 1 / FS)) > 100] = 0
    e = np.abs(np.fft.ifft(Z, axis=-1)).ravel()
    return 20 * np.log10(np.sqrt(np.mean(e ** 2)) / np.mean(e))


def solve(v):
    g = noise.qrn_floor(v)
    lo, hi = np.log(0.05), np.log(20000)
    for _ in range(14):
        mid = (lo + hi) / 2
        if vd200(np.exp(mid), g, 1) > v:
            lo = mid
        else:
            hi = mid
    return [v, float(np.exp((lo + hi) / 2))]


if __name__ == "__main__":
    with Pool(max(1, os.cpu_count() - 4)) as p:
        tab = p.map(solve, VDS)
    # rates must fall as Vd rises: smooth any noise by a running minimum
    for i in range(1, len(tab)):
        tab[i][1] = min(tab[i][1], tab[i - 1][1])
    path = os.path.join(os.path.dirname(__file__), "..", "python", "ionochannel", "qrn.json")
    q = json.load(open(path))
    q["table"] = [[v, round(r, 4)] for v, r in tab]
    json.dump(q, open(path, "w"), indent=1)
    for v, r in tab:
        print(f"Vd {v:5} dB: {r:9.3f} bursts/s, floor {noise.qrn_floor(v):.4f}")
