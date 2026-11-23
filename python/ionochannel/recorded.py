"""Recorded noise mixed in at an SNR, as for awgn(): the noise's level is
its mean spectral density over a measurement band (default 300-2700 Hz
for real audio, the whole band less 5 % at each edge for IQ) times the
reference bandwidth. For white noise this is what awgn() does, so the
two SNRs agree.

Segments start at random points. A recording shorter than the segment is
looped, with a 10 ms equal-power crossfade at the seam, so the seam
doesn't click. read_wav() and resample() are numpy only.
"""
from __future__ import annotations

import math
import struct

import numpy as np



def read_wav(path: str) -> tuple[np.ndarray, float]:
    """(samples, rate): mono as real float, two channels as complex I/Q.
    PCM 16/24/32-bit or float 32/64, scaled to +-1."""
    b = open(path, "rb").read()
    if b[:4] != b"RIFF" or b[8:12] != b"WAVE":
        raise ValueError(f"{path}: not a WAV file")
    i, fmt, data = 12, None, None
    while i + 8 <= len(b):
        cid, size = b[i:i + 4], struct.unpack("<I", b[i + 4:i + 8])[0]
        body = b[i + 8:i + 8 + size]
        if cid == b"fmt ":
            fmt = struct.unpack("<HHIIHH", body[:16])
            if fmt[0] == 0xFFFE and len(body) >= 26:          # WAVE_FORMAT_EXTENSIBLE
                fmt = (struct.unpack("<H", body[24:26])[0],) + fmt[1:]
        elif cid == b"data":
            data = body
        i += 8 + size + (size & 1)
    if fmt is None or data is None:
        raise ValueError(f"{path}: no fmt or data chunk")
    tag, ch, rate, _, _, bits = fmt
    if tag == 3 and bits in (32, 64):
        x = np.frombuffer(data, "<f4" if bits == 32 else "<f8").astype(float)
    elif tag == 1 and bits == 16:
        x = np.frombuffer(data, "<i2") / 32768.0
    elif tag == 1 and bits == 24:
        u = np.frombuffer(data[: len(data) // 3 * 3], np.uint8).reshape(-1, 3).astype(np.int32)
        x = ((u[:, 0] | (u[:, 1] << 8) | (u[:, 2] << 16)) << 8 >> 8) / 8388608.0
    elif tag == 1 and bits == 32:
        x = np.frombuffer(data, "<i4") / 2147483648.0
    else:
        raise ValueError(f"{path}: format {tag}, {bits} bits not supported")
    x = x[: len(x) // ch * ch].reshape(-1, ch)
    if ch == 1:
        return x[:, 0], float(rate)
    if ch == 2:
        return x[:, 0] + 1j * x[:, 1], float(rate)
    raise ValueError(f"{path}: {ch} channels (1 or 2 only)")


def resample(x: np.ndarray, fs_in: float, fs_out: float) -> np.ndarray:
    """Whole-array FFT resampling along the last axis (band-limited to
    the lower Nyquist). Fine for noise; edges ring a little."""
    if fs_in == fs_out:
        return x
    n = x.shape[-1]
    m = int(round(n * fs_out / fs_in))
    X = np.fft.fft(x, axis=-1)
    k = min(n, m)
    Y = np.zeros((*x.shape[:-1], m), complex)
    h = (k + 1) // 2
    Y[..., :h] = X[..., :h]
    Y[..., m - (k - h):] = X[..., n - (k - h):] if k - h else 0
    y = np.fft.ifft(Y, axis=-1) * (m / n)
    return y if np.iscomplexobj(x) else y.real


def default_band(fs: float, complex_: bool) -> tuple[float, float]:
    return (-0.45 * fs, 0.45 * fs) if complex_ else (300.0, min(2700.0, 0.45 * fs))


def band_density(x: np.ndarray, fs: float, band) -> np.ndarray:
    """Mean spectral density (power per Hz; one-sided for real) over band,
    per row (keepdims)."""
    n = x.shape[-1]
    X = np.fft.fft(x, axis=-1)
    f = np.fft.fftfreq(n, 1 / fs)
    lo, hi = band
    sel = (f >= lo) & (f <= hi)
    if not np.iscomplexobj(x):
        sel |= (-f >= lo) & (-f <= hi)
    p = np.sum(np.abs(X[..., sel]) ** 2, axis=-1, keepdims=True) / n ** 2
    return p / (hi - lo)


def _loop(rec: np.ndarray, fs: float) -> np.ndarray:
    """A seamless loop: the last 10 ms crossfaded into the first."""
    m = rec.shape[-1]
    L = min(int(0.01 * fs), m // 4)
    if L < 2:
        return rec
    t = (np.arange(L) + 0.5) / L * (np.pi / 2)
    out = rec[..., : m - L].copy()
    out[..., :L] = rec[..., :L] * np.sin(t) + rec[..., m - L:] * np.cos(t)
    return out


def recorded_noise(rec: np.ndarray, shape, n: int, fs: float, rng) -> np.ndarray:
    """Random segments (*shape, n) of a recording, (m,) or (k, m) for k
    recordings (each row picks one)."""
    shape = tuple(shape) if np.iterable(shape) else (shape,)
    rows = int(np.prod(shape))
    rec = np.atleast_2d(np.asarray(rec))
    k, m = rec.shape
    if m >= n:
        src = rec
        idx = rng.integers(0, m - n + 1, rows)[:, None] + np.arange(n)
    else:
        src = _loop(rec, fs)
        idx = (rng.integers(0, src.shape[-1], rows)[:, None] + np.arange(n)) % src.shape[-1]
    which = rng.integers(0, k, rows)[:, None]
    return src[which, idx].reshape(*shape, n)


def mix_noise(x: np.ndarray, rec: np.ndarray, snr_db, fs: float, rng, bw_hz: float = 2500.0,
              band=None, signal_power=None, per_segment: bool = True) -> np.ndarray:
    """x plus segments of recorded noise at snr_db: the noise's density
    over `band`, times bw_hz, against signal_power (default: x's mean
    power per row). per_segment: level each segment exactly (else one
    level from the whole recording, so segments vary as the recording
    does). The recording must be at fs and real or complex like x."""
    x = np.asarray(x)
    rec = np.asarray(rec)
    if np.iscomplexobj(rec) != np.iscomplexobj(x):
        raise ValueError("recording and signal must both be real or both complex (I/Q)")
    band = default_band(fs, np.iscomplexobj(x)) if band is None else band
    p = np.mean(np.abs(x) ** 2, axis=-1, keepdims=True) if signal_power is None else np.asarray(signal_power, float)
    snr = np.asarray(snr_db, float)
    if snr.ndim and snr.shape[-1] != 1:
        snr = snr[..., None]
    nz = recorded_noise(rec, x.shape[:-1], x.shape[-1], fs, rng)
    d = band_density(nz if per_segment else np.atleast_2d(rec), fs, band)
    if not per_segment:
        d = np.mean(d)
    g = np.sqrt(p / 10 ** (snr / 10) / (np.maximum(d, 1e-300) * bw_hz))
    return (x + nz * g).astype(x.dtype)
