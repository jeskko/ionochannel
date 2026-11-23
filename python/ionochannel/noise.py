"""Impulsive atmospheric noise (QRN) set by its voltage deviation Vd, as
ITU-R P.372 describes it: Vd = 20 log10(rms / mean) of the envelope in a
200 Hz bandwidth; Fig. 37 gives the amplitude distribution for each Vd
and Fig. 38 how Vd grows with bandwidth.

The model: Gaussian noise whose power follows a shot-noise envelope. Each
burst (a sferic) arrives at random (Poisson), with energy a^2 from a
Pareto law (P(a^2 > x) = x^-alpha/2, alpha = 4) and an exponential decay
whose amplitude time constant is drawn log-uniformly from 0.1 to 100 ms
(8 bins). A Gaussian floor g carries the rest of the power. Vd sets g by
a fixed law and the burst rate by a table calibrated by simulation (see
qrn.json); the result matches Fig. 37 and Fig. 38 to about 1 dB.
"""
from __future__ import annotations

import json
import math
import os

import numpy as np

_Q = json.load(open(os.path.join(os.path.dirname(__file__), "qrn.json")))
ALPHA = _Q["alpha"]
TAUS = np.geomspace(_Q["tau_s"][0], _Q["tau_s"][1], _Q["bins"])
VD_GAUSS = 20 * math.log10(math.sqrt(4 / math.pi))   # 1.049 dB, Rayleigh envelope
VD_MAX = _Q["table"][-1][0]


def qrn_floor(vd_db: float) -> float:
    """Gaussian share of the power for a 200 Hz Vd."""
    if vd_db >= 2:
        return 0.1 * 10 ** (-0.19 * (vd_db - 2))
    return 10 ** (-(vd_db - VD_GAUSS) / (2 - VD_GAUSS))


def qrn_params(vd_db: float) -> tuple[float, float]:
    """(bursts per second, Gaussian share) for a 200 Hz Vd (dB)."""
    if not (VD_GAUSS <= vd_db <= VD_MAX):
        raise ValueError(f"Vd {vd_db} dB outside {VD_GAUSS:.2f}..{VD_MAX} (200 Hz)")
    t = np.array(_Q["table"], float)
    rate = float(np.exp(np.interp(vd_db, t[:, 0], np.log(t[:, 1]))))
    return rate, qrn_floor(vd_db)


def vd_bandwidth(vd200_db: float, bw_hz: float) -> float:
    """Vd in another bandwidth from the 200 Hz value: P.372 Fig. 38,
    straight lines in log bandwidth (slopes read from the figure)."""
    s = np.interp(vd200_db, *np.array(_Q["fig38_slope"], float).T)
    return max(VD_GAUSS, vd200_db + s * math.log10(bw_hz / 200))


def _envelope(rows: int, n: int, fs: float, rate: float, g: float, rng) -> np.ndarray:
    """Power envelope (rows, n), mean 1 over time and realisations."""
    pre = int(math.ceil(fs * TAUS[-1] / 2 * 25))          # bursts before t=0: e^-25 left
    span = pre + n
    k = rng.poisson(rate * span / fs * rows)
    t = np.sort(rng.integers(0, rows * span, k))
    b = rng.integers(0, len(TAUS), k)
    a2 = (1.0 - rng.random(k)) ** (-2.0 / ALPHA)
    lam = 2.0 / (fs * TAUS)                               # power decay per sample
    m = np.zeros(rows * n)
    grid = (np.arange(rows)[:, None] * span + pre + np.arange(n)).ravel()
    last_burst = np.empty(rows * span, np.int32 if k < 2 ** 31 else np.int64)
    for i, li in enumerate(lam):
        sel = b == i
        ti, pi = t[sel], a2[sel] * (1.0 - math.exp(-li))
        if not len(ti):
            continue
        v = np.empty(len(ti))                             # value just after each burst
        acc, last = 0.0, 0
        for j, (tj, pj) in enumerate(zip(ti.tolist(), pi.tolist())):
            acc = acc * math.exp(-li * (tj - last)) + pj
            last = tj
            v[j] = acc
        # index of the latest burst at or before each sample: a running max
        last_burst.fill(-1)
        end = np.r_[ti[1:] != ti[:-1], True]               # the last of bursts on one sample
        last_burst[ti[end]] = np.flatnonzero(end)
        np.maximum.accumulate(last_burst, out=last_burst)
        idx = last_burst[grid]
        dt = np.where(idx >= 0, (grid - ti[idx]) * li, np.inf)
        live = np.flatnonzero(dt < 40.0)                   # beyond e^-40 it is nothing
        m[live] += v[idx[live]] * np.exp(-dt[live])
    mean = rate / fs * ALPHA / (ALPHA - 2)
    return (g + (1 - g) * m / mean).reshape(rows, n)


def qrn_noise(shape, n: int, fs: float, vd_db: float, rng, complex_: bool = True) -> np.ndarray:
    """QRN of mean power 1, (*shape, n); complex, or real. Each leading
    index is its own realisation. Short clips vary a lot in power: the
    mean is over time and realisations, as for real QRN."""
    shape = tuple(shape) if np.iterable(shape) else (shape,)
    rows = int(np.prod(shape))
    if vd_db <= VD_GAUSS + 1e-9:
        env = np.ones((rows, n))
    else:
        env = _envelope(rows, n, fs, *qrn_params(vd_db), rng)
    if complex_:
        w = (rng.standard_normal((rows, n)) + 1j * rng.standard_normal((rows, n))) * math.sqrt(0.5)
    else:
        w = rng.standard_normal((rows, n))
    return (np.sqrt(env) * w).reshape(*shape, n)


def qrn(x: np.ndarray, snr_db, vd_db: float, fs: float, rng, bw_hz: float = 2500.0,
        signal_power=None) -> np.ndarray:
    """x plus QRN at snr_db in bw_hz (the noise's mean power, as for
    awgn), with Vd vd_db in 200 Hz."""
    x = np.asarray(x)
    p = np.mean(np.abs(x) ** 2, axis=-1, keepdims=True) if signal_power is None else np.asarray(signal_power, float)
    snr = np.asarray(snr_db, float)
    if snr.ndim and snr.shape[-1] != 1:
        snr = snr[..., None]
    complex_ = np.iscomplexobj(x)
    var = p / 10 ** (snr / 10) * (fs if complex_ else fs / 2) / bw_hz
    nz = qrn_noise(x.shape[:-1], x.shape[-1], fs, vd_db, rng, complex_)
    return (x + nz * np.sqrt(var)).astype(x.dtype)
