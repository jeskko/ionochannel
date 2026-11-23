"""Man-made interference with a mains signature (ITU-R P.372 §6 calls the
first kind "single carrier noise"; it doesn't give parameters, so the
defaults here are illustrative, not measured):

- smps(): a switching power supply's comb. Lines at offset + k*spacing
  (the supply's harmonics, `spacing` = its switching frequency) that
  wander together (an Ornstein-Uhlenbeck drift, `drift_hz` rms with
  correlation time `drift_s`) and carry AM and FM at twice the mains
  frequency (rectifier ripple). Level: each line's power against the
  signal, in dB.
- arcing(): powerline arcing buzz. Short noise bursts at fixed phases of
  each mains half-cycle (one phase per source), with timing jitter and
  log-normal burst energies, gated on and off over seconds. Level: mean
  power as an SNR in a reference bandwidth, as for awgn().
- bursts(): noise switched on and off at random (exponential on and off
  times), for packet transmitters such as powerline networking seen in a
  narrow band. Level: SNR while on.

VDSL in a 3 kHz window is a steady raised noise floor (use awgn), and
LED drivers are switching supplies (use smps).

Each leading index of x gets its own realisation. The Rust crate's
interference module implements the same models.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .channel import TAU


@dataclass(frozen=True)
class Smps:
    spacing_hz: float = 60e3
    mains_hz: float = 50.0
    am_depth: float = 0.3
    fm_dev_hz: float = 15.0
    drift_hz: float = 20.0
    drift_s: float = 10.0
    line_spread_db: float = 0.0


@dataclass(frozen=True)
class Arcing:
    mains_hz: float = 50.0
    sources: int = 1
    jitter_s: float = 2e-4
    burst_s: float = 1e-4
    spread_db: float = 4.0
    on_s: float = 5.0
    off_s: float = 0.0


def _ou(rows: int, n: int, fs: float, rms: float, tau: float, rng) -> np.ndarray:
    """Ornstein-Uhlenbeck process (rows, n), stationary from the start,
    stepped at up to 200 Hz and linearly interpolated."""
    if rms <= 0:
        return np.zeros((rows, n))
    low = min(fs, 200.0)
    m = int(math.ceil(n / fs * low)) + 2
    rho = math.exp(-1 / (low * tau)) if tau > 0 else 0.0
    w = rng.standard_normal((rows, m)) * rms * math.sqrt(1 - rho * rho)
    y = np.empty((rows, m))
    y[:, 0] = rng.standard_normal(rows) * rms
    for i in range(1, m):
        y[:, i] = y[:, i - 1] * rho + w[:, i]
    pos = np.arange(n) * (low / fs)
    i0 = np.minimum(pos.astype(np.int64), m - 2)
    f = pos - i0
    return y[:, i0] * (1 - f) + y[:, i0 + 1] * f


def smps_lines(fs: float, offset_hz: float, spacing_hz: float, real: bool) -> np.ndarray:
    """Nominal line frequencies inside the band: (-fs/2, fs/2) for complex,
    (0, fs/2) for real."""
    lo, hi = (0.0, fs / 2) if real else (-fs / 2, fs / 2)
    k0 = math.ceil((lo - offset_hz) / spacing_hz)
    k1 = math.floor((hi - offset_hz) / spacing_hz)
    f = offset_hz + spacing_hz * np.arange(k0, k1 + 1)
    return f[(f > lo) & (f < hi)]


def smps_noise(shape, n: int, fs: float, offset_hz: float, rng, spec: Smps = Smps(),
               complex_: bool = True) -> np.ndarray:
    """The comb alone, (*shape, n), each line of mean power 1."""
    shape = tuple(shape) if np.iterable(shape) else (shape,)
    rows = int(np.prod(shape))
    lines = smps_lines(fs, offset_hz, spec.spacing_hz, not complex_)
    t = np.arange(n) / fs
    ripple = TAU * 2 * spec.mains_hz
    th = rng.random((rows, 1)) * TAU
    dev = _ou(rows, n, fs, spec.drift_hz, spec.drift_s, rng)
    # common phase: drift integrated, plus the ripple FM (a sine whose integral is known)
    ph = TAU * np.cumsum(dev, axis=-1) / fs
    if spec.fm_dev_hz and spec.mains_hz:
        ph = ph - TAU * spec.fm_dev_hz / ripple * np.cos(ripple * t + th)
    am = (1 + spec.am_depth * np.sin(ripple * t + th + rng.random((rows, 1)) * TAU)) / math.sqrt(1 + spec.am_depth ** 2 / 2)
    z = np.zeros((rows, n), np.complex128)
    for f in lines:
        a = 10 ** (spec.line_spread_db * rng.standard_normal((rows, 1)) / 20) if spec.line_spread_db else 1.0
        z += a * np.exp(1j * (TAU * f * t + ph + rng.random((rows, 1)) * TAU))
    z *= am
    out = z if complex_ else math.sqrt(2) * z.real
    return out.reshape(*shape, n)


def smps(x: np.ndarray, level_db, offset_hz: float, fs: float, rng, spec: Smps = Smps(),
         signal_power=None) -> np.ndarray:
    """x plus a switching-supply comb; each line level_db against the
    signal's power (default: x's own mean power per row)."""
    x = np.asarray(x)
    p = np.mean(np.abs(x) ** 2, axis=-1, keepdims=True) if signal_power is None else np.asarray(signal_power, float)
    lv = np.asarray(level_db, float)
    if lv.ndim and lv.shape[-1] != 1:
        lv = lv[..., None]
    nz = smps_noise(x.shape[:-1], x.shape[-1], fs, offset_hz, rng, spec, np.iscomplexobj(x))
    return (x + nz * np.sqrt(p * 10 ** (lv / 10))).astype(x.dtype)


def _gate(rows: int, n: int, fs: float, on_s: float, off_s: float, rng) -> np.ndarray:
    """On/off gate (rows, n) bool, exponential on and off times, starting
    in its steady state."""
    if off_s <= 0:
        return np.ones((rows, n), bool)
    g = np.empty((rows, n), bool)
    duty = on_s / (on_s + off_s)
    for r in range(rows):
        state = rng.random() < duty
        i = 0
        while i < n:
            d = int(rng.exponential(on_s if state else off_s) * fs) + 1
            g[r, i:i + d] = state
            i += d
            state = not state
    return g


def arcing_noise(shape, n: int, fs: float, rng, spec: Arcing = Arcing(),
                 complex_: bool = True) -> np.ndarray:
    """Arcing alone, (*shape, n), mean power 1 (over time and realisations)."""
    shape = tuple(shape) if np.iterable(shape) else (shape,)
    rows = int(np.prod(shape))
    half = 1 / (2 * spec.mains_hz)
    lam = 1 / (fs * spec.burst_s / 2)                 # power decay per sample
    d = math.exp(-lam)
    pre = int(math.ceil(25 / lam))
    gate = _gate(rows, n + pre, fs, spec.on_s, spec.off_s, rng)
    env = np.zeros((rows, n + pre))
    k = int(math.ceil((n + pre) / fs / half)) + 1
    for _ in range(spec.sources):
        phase = rng.random((rows, 1)) * half
        tb = phase + np.arange(k) * half + rng.standard_normal((rows, k)) * spec.jitter_s
        ib = np.floor(tb * fs).astype(np.int64)
        e = 10 ** (spec.spread_db * rng.standard_normal((rows, k)) / 10)
        ok = (ib >= 0) & (ib < n + pre)
        r, c = np.nonzero(ok)
        ok_gate = gate[r, ib[r, c]]
        np.add.at(env, (r[ok_gate], ib[r, c][ok_gate]), e[r, c][ok_gate] * (1 - d))
    out = _decay(env, d)[:, pre:]
    duty = 1.0 if spec.off_s <= 0 else spec.on_s / (spec.on_s + spec.off_s)
    mean = spec.sources * 2 * spec.mains_hz / fs * duty * math.exp((spec.spread_db * math.log(10) / 10) ** 2 / 2)
    if complex_:
        w = (rng.standard_normal((rows, n)) + 1j * rng.standard_normal((rows, n))) * math.sqrt(0.5)
    else:
        w = rng.standard_normal((rows, n))
    return (np.sqrt(out / mean) * w).reshape(*shape, n)


def _decay(s: np.ndarray, d: float) -> np.ndarray:
    """y[i] = d*y[i-1] + s[i] along the last axis, in blocks: within a
    block the closed form y = d^i * cumsum(s * d^-i) stays in range."""
    if d <= 0:
        return s.copy()
    blk = max(1, int(30 / -math.log(d)))
    y = np.empty_like(s)
    carry = np.zeros(s.shape[0])
    for a in range(0, s.shape[1], blk):
        seg = s[:, a:a + blk]
        i = np.arange(seg.shape[1])
        p = d ** i
        y[:, a:a + blk] = p * (carry[:, None] * d + np.cumsum(seg / p, axis=-1))
        carry = y[:, a + seg.shape[1] - 1]
    return y


def arcing(x: np.ndarray, snr_db, fs: float, rng, spec: Arcing = Arcing(), bw_hz: float = 2500.0,
           signal_power=None) -> np.ndarray:
    """x plus arcing buzz at snr_db in bw_hz (mean power, as awgn)."""
    x = np.asarray(x)
    p = np.mean(np.abs(x) ** 2, axis=-1, keepdims=True) if signal_power is None else np.asarray(signal_power, float)
    snr = np.asarray(snr_db, float)
    if snr.ndim and snr.shape[-1] != 1:
        snr = snr[..., None]
    complex_ = np.iscomplexobj(x)
    var = p / 10 ** (snr / 10) * (fs if complex_ else fs / 2) / bw_hz
    nz = arcing_noise(x.shape[:-1], x.shape[-1], fs, rng, spec, complex_)
    return (x + nz * np.sqrt(var)).astype(x.dtype)


@dataclass(frozen=True)
class Bursts:
    """Gated noise bursts (powerline networking and other packet
    transmitters seen in a narrow band): exponential on and off times."""
    on_s: float = 0.002
    off_s: float = 0.02


def bursts_noise(shape, n: int, fs: float, rng, spec: Bursts = Bursts(), complex_: bool = True) -> np.ndarray:
    """Noise of power 1 while on, 0 while off, (*shape, n)."""
    shape = tuple(shape) if np.iterable(shape) else (shape,)
    rows = int(np.prod(shape))
    g = _gate(rows, n, fs, spec.on_s, spec.off_s, rng)
    if complex_:
        w = (rng.standard_normal((rows, n)) + 1j * rng.standard_normal((rows, n))) * math.sqrt(0.5)
    else:
        w = rng.standard_normal((rows, n))
    return (w * g).reshape(*shape, n)


def bursts(x: np.ndarray, snr_db, fs: float, rng, spec: Bursts = Bursts(), bw_hz: float = 2500.0,
           signal_power=None) -> np.ndarray:
    """x plus noise bursts at snr_db in bw_hz while on (as awgn)."""
    x = np.asarray(x)
    p = np.mean(np.abs(x) ** 2, axis=-1, keepdims=True) if signal_power is None else np.asarray(signal_power, float)
    snr = np.asarray(snr_db, float)
    if snr.ndim and snr.shape[-1] != 1:
        snr = snr[..., None]
    complex_ = np.iscomplexobj(x)
    var = p / 10 ** (snr / 10) * (fs if complex_ else fs / 2) / bw_hz
    nz = bursts_noise(x.shape[:-1], x.shape[-1], fs, rng, spec, complex_)
    return (x + nz * np.sqrt(var)).astype(x.dtype)
