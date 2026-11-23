"""The channel: tap gains (Gaussian Doppler spectra) and the tapped delay
line, on whole arrays."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

TAU = 2 * np.pi
OVERSAMPLE = 32.0      # low rate of the scatter process over its spread
MIN_LOW_RATE = 0.05


def fast_len(n: int) -> int:
    """The smallest 2^a 3^b 5^c >= n (FFT sizes numpy does quickly)."""
    best = 1 << max(0, (n - 1).bit_length())
    p5 = 1
    while p5 < best:
        p35 = p5
        while p35 < best:
            m = p35
            while m < n:
                m *= 2
            best = min(best, m)
            p35 *= 3
        p5 *= 5
    return best


@dataclass(frozen=True)
class Component:
    """Rayleigh scatter (power, 2-sigma spread, shift) or a specular ray."""
    power: float
    spread_hz: float = 0.0
    shift_hz: float = 0.0
    specular: bool = False


@dataclass(frozen=True)
class Path:
    delay_s: float
    components: tuple[Component, ...]

    @property
    def power(self) -> float:
        return sum(c.power for c in self.components)


def scatter(delay_s: float, power: float, spread_hz: float, shift_hz: float = 0.0) -> Path:
    return Path(delay_s, (Component(power, spread_hz, shift_hz),))


def specular(delay_s: float, power: float, shift_hz: float = 0.0) -> Path:
    return Path(delay_s, (Component(power, 0.0, shift_hz, specular=True),))


def _scatter_gain(shape: tuple[int, ...], n: int, fs: float, spread: float, rng, rate=None) -> np.ndarray:
    """Unit-power complex Gaussian process with a Gaussian spectrum of
    2 sigma = spread: shaped white noise at a low rate (FFT, with a margin
    so the circular wrap is far below the needed correlation), then
    linear interpolation to fs."""
    if spread <= 0:
        raise ValueError("a scatter component needs a spread > 0 (use specular for none)")
    low = max(OVERSAMPLE * spread, MIN_LOW_RATE)
    sigma = spread / 2
    margin = int(np.ceil(4.0 / (np.pi * sigma) * low))   # |R| < 1e-6 beyond
    if rate is not None:
        # time-warped: positions advance by `rate` (<= 1) per sample
        pos = (np.cumsum(rate, axis=-1) - rate[..., :1]) * (low / fs)
        m_low = int(np.ceil(pos.max())) + 2
    else:
        m_low = int(np.ceil(n / fs * low)) + 2
    m = fast_len(m_low + margin)
    w = (rng.standard_normal((*shape, m)) + 1j * rng.standard_normal((*shape, m))) / np.sqrt(2)
    f = np.fft.fftfreq(m, 1 / low)
    h = np.exp(-(f ** 2) / (4 * sigma ** 2))              # |H|^2 = exp(-f^2 / 2 sigma^2)
    h /= np.sqrt(np.mean(h ** 2))
    g = np.fft.ifft(np.fft.fft(w, axis=-1) * h, axis=-1)[..., :m_low]
    g = g.astype(np.complex64)
    if rate is not None:
        i0 = np.minimum(pos.astype(np.int64), m_low - 2)
        frac = (pos - i0).astype(np.float32)
        return np.take_along_axis(g, i0, -1) * (1 - frac) + np.take_along_axis(g, i0 + 1, -1) * frac
    pos = np.arange(n) * (low / fs)
    i0 = np.minimum(pos.astype(np.int64), m_low - 2)
    frac = (pos - i0).astype(np.float32)
    return g[..., i0] * (1 - frac) + g[..., i0 + 1] * frac


def tap_gain(path: Path, n: int, fs: float, rng, shape: tuple[int, ...] = (), rate=None,
             scale: float = 1.0) -> np.ndarray:
    """A path's tap gain over n samples, (*shape, n) complex. `scale`
    multiplies the scatter spreads; `rate` (*shape, n), <= 1, slows the
    scatter down (a spread that changes over time)."""
    t = np.arange(n) / fs
    g = np.zeros((*shape, n), np.complex64)
    for c in path.components:
        rot = np.exp(1j * TAU * c.shift_hz * t).astype(np.complex64)
        if c.specular:
            ph = np.exp(1j * TAU * rng.random((*shape, 1)))
            g += (np.sqrt(c.power) * ph * rot).astype(np.complex64)
        else:
            g += np.float32(np.sqrt(c.power)) * _scatter_gain(shape, n, fs, c.spread_hz * scale, rng, rate) * rot
    return g


def power(x: np.ndarray) -> np.ndarray:
    """Mean power along the last axis (kept for broadcasting)."""
    return np.mean(np.abs(x) ** 2, axis=-1, keepdims=True)


def awgn(x: np.ndarray, snr_db, fs: float, rng, bw_hz: float = 2500.0, signal_power=None) -> np.ndarray:
    """x plus white noise at snr_db (scalar or per row) in bw_hz, against
    signal_power (default: x's own mean power per row; pass the clean
    input's for an SNR before fading)."""
    p = power(x) if signal_power is None else np.asarray(signal_power, float)
    snr = np.asarray(snr_db, float)
    if snr.ndim and snr.shape[-1] != 1:
        snr = snr[..., None]
    complex_ = np.iscomplexobj(x)
    var = p / 10 ** (snr / 10) * (fs if complex_ else fs / 2) / bw_hz
    if complex_:
        nz = (rng.standard_normal(x.shape) + 1j * rng.standard_normal(x.shape)) * np.sqrt(var / 2)
    else:
        nz = rng.standard_normal(x.shape) * np.sqrt(var)
    return (x + nz).astype(x.dtype)


class Channel:
    """Paths at a sample rate; apply() fades whole arrays."""

    def __init__(self, fs: float, paths, vary=None):
        """vary: one Vary per path (or None) for paths that change over
        time; see the vary module."""
        self.fs = float(fs)
        self.paths = tuple(paths)
        if not self.paths:
            raise ValueError("a channel needs at least one path")
        self.vary = tuple(vary) if vary else ()
        if self.vary and len(self.vary) != len(self.paths):
            raise ValueError("vary needs one entry per path")

    def apply(self, x: np.ndarray, rng) -> np.ndarray:
        """Fade x (..., n): complex baseband, or real audio (through its
        analytic signal; a positive shift moves the audio up). Each
        leading index gets its own realisation. No latency: delays are
        applied in the frequency domain over a zero-padded copy."""
        x = np.asarray(x)
        real = not np.iscomplexobj(x)
        n = x.shape[-1]
        shape = x.shape[:-1]
        pad = int(np.ceil(max(p.delay_s for p in self.paths) * self.fs)) + 64
        m = fast_len(n + pad)
        X = np.fft.fft(x, m, axis=-1)
        f = np.fft.fftfreq(m, 1 / self.fs)
        if real:
            X = X * np.where(f > 0, 2.0, np.where(f == 0, 1.0, 0.0))   # analytic signal
            f = np.where(f < 0, f + self.fs, f)                     # no negative half left
        y = np.zeros((*shape, n), np.complex64)
        moving = [v is not None and not v.is_empty() for v in self.vary] or [False] * len(self.paths)
        vrng = np.random.default_rng(rng.integers(2 ** 63)) if any(moving) else None
        rows = int(np.prod(shape)) if shape else 1
        for p, mv, v in zip(self.paths, moving, self.vary or [None] * len(self.paths)):
            if not mv:
                xd = np.fft.ifft(X * np.exp(-1j * TAU * f * p.delay_s), axis=-1)[..., :n].astype(np.complex64)
                y += tap_gain(p, n, self.fs, rng, shape) * xd
                continue
            y += self._moving_path(p, v, X, f, shape, rows, n, rng, vrng)
        return y.real.astype(x.dtype if x.dtype.kind == "f" else np.float64) if real else y.astype(np.result_type(x.dtype, np.complex64))

    def _moving_path(self, p, v, X, f, shape, rows, n, rng, vrng):
        from .vary import bounds, moving_delay, trajectory
        traj = lambda terms: trajectory(terms, rows, n, self.fs, vrng).reshape(*shape, n)  # noqa: E731
        if v.delay_s:
            z = np.fft.ifft(X, axis=-1)[..., :n].astype(np.complex64).reshape(rows, n)
            dmax = max(0.0, (p.delay_s + bounds(v.delay_s)[1]) * self.fs)
            d = np.clip((p.delay_s + traj(v.delay_s).reshape(rows, n)) * self.fs, 0, dmax)
            xd = moving_delay(z, d).reshape(*shape, n)
        else:
            xd = np.fft.ifft(X * np.exp(-1j * TAU * f * p.delay_s), axis=-1)[..., :n].astype(np.complex64)
        if v.spread_oct:
            vmax = bounds(v.spread_oct)[1]
            rate = 2.0 ** (traj(v.spread_oct) - vmax)
            g = tap_gain(p, n, self.fs, rng, shape, rate=rate, scale=2.0 ** vmax)
        else:
            g = tap_gain(p, n, self.fs, rng, shape)
        if v.power_db:
            g = g * (10 ** (traj(v.power_db) / 20)).astype(np.float32)
        if v.shift_hz:
            dop = traj(v.shift_hz)
            cyc = np.concatenate([np.zeros((*shape, 1)), np.cumsum(dop[..., :-1], axis=-1)], axis=-1) / self.fs
            g = g * np.exp(1j * TAU * np.remainder(cyc, 1.0)).astype(np.complex64)
        return g * xd
