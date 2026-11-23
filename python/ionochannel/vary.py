"""Paths that change over time (as the Rust crate's vary module). Each
path's delay (s), Doppler shift (Hz), power (dB) and spread (octaves) can
follow a trajectory, a list of terms added to the path's own value:

    Ramp(t0_s, t1_s, from_, to)     linear, held before and after
    Sine(amp, period_s, phase=0)    amp * sin(2 pi t / period + phase)
    Drift(rms, tau_s)               Ornstein-Uhlenbeck, clamped at +-4 rms
    Table(t_s, values)              a recorded trajectory, linear in between

    vary = [Vary(shift_hz=[Sine(2.0, 600)]), tid(7e6, 800, 0.05, 3600)]
    y = ic.Channel(fs, paths, vary=vary).apply(x, rng)

Spread in octaves multiplies every scatter spread of the path by 2^value
by time-warping its fading process (made for the largest spread reached
and slowed down from it). tid() couples a TID's delay and Doppler.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

C = 299_792_458.0
FD_HALF = 16
PHASES = 512


@dataclass(frozen=True)
class Ramp:
    t0_s: float
    t1_s: float
    from_: float
    to: float


@dataclass(frozen=True)
class Sine:
    amp: float
    period_s: float
    phase: float = 0.0


@dataclass(frozen=True)
class Drift:
    rms: float
    tau_s: float


@dataclass(frozen=True)
class Table:
    """A recorded trajectory: values at times (s, increasing), linear in
    between, held before the first and after the last."""
    t_s: tuple
    values: tuple

    def __post_init__(self):
        t = np.asarray(self.t_s, float)
        if len(t) == 0 or len(t) != len(self.values) or np.any(np.diff(t) <= 0):
            raise ValueError("a table needs increasing times and one value per time")


@dataclass(frozen=True)
class Vary:
    delay_s: tuple = field(default_factory=tuple)
    shift_hz: tuple = field(default_factory=tuple)
    power_db: tuple = field(default_factory=tuple)
    spread_oct: tuple = field(default_factory=tuple)

    def is_empty(self) -> bool:
        return not (self.delay_s or self.shift_hz or self.power_db or self.spread_oct)


def bounds(terms) -> tuple[float, float]:
    lo = hi = 0.0
    for t in terms:
        if isinstance(t, Ramp):
            a, b = min(t.from_, t.to), max(t.from_, t.to)
        elif isinstance(t, Sine):
            a, b = -abs(t.amp), abs(t.amp)
        elif isinstance(t, Table):
            a, b = float(min(t.values)), float(max(t.values))
        else:
            a, b = -4 * abs(t.rms), 4 * abs(t.rms)
        lo, hi = lo + a, hi + b
    return lo, hi


def tid(carrier_hz: float, path_km: float, rel_amp: float, period_s: float, phase: float = 0.0) -> Vary:
    """A travelling ionospheric disturbance: the group path P swings by
    rel_amp * P; delay P / c, Doppler -(f / c) dP/dt a quarter period
    ahead (Trop et al. eq. 4). Oinats et al. 2016: MSTID periods most
    often 60-100 min, relative amplitudes 3-8 %."""
    dp = rel_amp * path_km * 1e3
    return Vary(delay_s=(Sine(dp / C, period_s, phase),),
                shift_hz=(Sine(carrier_hz * dp * 2 * math.pi / (C * period_s), period_s, phase - math.pi / 2),))


def _ou(rows, n, fs, rms, tau, rng):
    """Ornstein-Uhlenbeck at the sample rate, by block-wise closed form."""
    rho = math.exp(-1 / (fs * tau)) if tau > 0 else 0.0
    w = rng.standard_normal((rows, n)) * rms * math.sqrt(1 - rho * rho)
    w[:, 0] = rng.standard_normal(rows) * rms
    if rho == 0:
        return w
    blk = max(1, int(30 / -math.log(rho)))
    y = np.empty_like(w)
    carry = np.zeros(rows)
    for a in range(0, n, blk):
        seg = w[:, a:a + blk]
        p = rho ** np.arange(seg.shape[1])
        y[:, a:a + blk] = p * (carry[:, None] * rho + np.cumsum(seg / p, axis=-1))
        carry = y[:, a + seg.shape[1] - 1]
    return y


def trajectory(terms, rows: int, n: int, fs: float, rng) -> np.ndarray:
    """A trajectory's value (rows, n); each row its own drift."""
    t = np.arange(n) / fs
    v = np.zeros((rows, n))
    for term in terms:
        if isinstance(term, Ramp):
            if term.t1_s > term.t0_s:
                f = np.clip((t - term.t0_s) / (term.t1_s - term.t0_s), 0, 1)
            else:
                f = (t > term.t0_s).astype(float)
            v += term.from_ + (term.to - term.from_) * f
        elif isinstance(term, Sine):
            v += term.amp * np.sin(2 * math.pi * t / term.period_s + term.phase)
        elif isinstance(term, Table):
            v += np.interp(t, term.t_s, term.values)
        else:
            v += np.clip(_ou(rows, n, fs, term.rms, term.tau_s, rng), -4 * term.rms, 4 * term.rms)
    return v


def frac_taps(frac: float) -> np.ndarray:
    """The 33-tap Blackman-windowed sinc fractional delay of the Rust
    crate: taps[k] weights the input k samples back, delay 16 + frac."""
    n = 2 * FD_HALF + 1
    k = np.arange(n)
    x = k - FD_HALF - frac
    w = (k - frac) / (n - 1)
    t = np.sinc(x) * (0.42 - 0.5 * np.cos(2 * np.pi * w) + 0.08 * np.cos(4 * np.pi * w))
    return t / t.sum()


_TABLE = None


def table() -> np.ndarray:
    global _TABLE
    if _TABLE is None:
        _TABLE = np.stack([frac_taps(k / PHASES) for k in range(PHASES + 1)])
    return _TABLE


def moving_delay(x: np.ndarray, d: np.ndarray, chunk: int = 2048) -> np.ndarray:
    """y[n] = x(n - d[n]) for x (rows, n), real or complex, and delays d
    (rows, n) in samples (negative reads ahead), with the fractional-delay
    table; no latency. Outside x is zero."""
    rows, n = x.shape
    dmax = int(math.ceil(max(d.max(), 0.0))) + 1
    ahead = int(math.ceil(max(-d.min(), 0.0))) + 1
    pad = dmax + FD_HALF + 2
    xp = np.concatenate([np.zeros((rows, pad), x.dtype), x, np.zeros((rows, ahead + FD_HALF + 2), x.dtype)], axis=1)
    tab = table().astype(np.float32 if x.dtype in (np.complex64, np.float32) else np.float64)
    y = np.empty_like(x)
    k = np.arange(2 * FD_HALF + 1)
    r = np.arange(rows)[:, None, None]
    for a in range(0, n, chunk):
        dd = d[:, a:a + chunk]
        w = np.floor(dd)
        ph = (dd - w) * PHASES
        k0 = np.minimum(ph.astype(np.int64), PHASES - 1)
        al = (ph - k0)[..., None].astype(tab.dtype)
        taps = tab[k0] * (1 - al) + tab[k0 + 1] * al                     # (rows, c, 33)
        # output a+i is input a+i - w - 16 - frac: taps[k] at (a+i - w + 16 - k)
        idx = (np.arange(a, a + dd.shape[1])[None, :] - w.astype(np.int64))[..., None] + FD_HALF - k + pad
        y[:, a:a + chunk] = np.sum(xp[r, idx] * taps, axis=-1)
    return y
