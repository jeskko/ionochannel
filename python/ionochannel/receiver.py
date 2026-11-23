"""The receiver after the channel and noise: mistuning, passband filter,
AGC, in that order. The same design as the Rust crate's receiver module
(see its docs), on whole arrays with no latency."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .channel import TAU, fast_len

STOPBAND_DB = 60.0
HANG_REARM = 10 ** (-1 / 20)   # a peak within 1 dB of the level re-arms the hang


@dataclass(frozen=True)
class Agc:
    """Peak AGC in 1 ms blocks: attack, hang (after the last peak within
    1 dB of the level), exponential decay; output peak `target`, at most
    `max_gain_db` of gain."""
    attack_s: float = 0.002
    hang_s: float = 0.0
    decay_s: float = 0.5
    target: float = 0.25
    max_gain_db: float = 60.0


def agc_block(fs: float) -> int:
    return max(1, int(np.floor(fs * 1e-3 + 0.5)))


def bandpass_taps(fs: float, lo: float, hi: float, transition_hz: float = 100.0) -> np.ndarray:
    """Complex FIR taps: a Kaiser-windowed lowpass of half width
    (hi - lo) / 2 moved to the band's centre; -6 dB at lo and hi, 60 dB
    stopband, odd length."""
    if hi <= lo:
        raise ValueError("passband: hi must be above lo")
    if transition_hz <= 0:
        raise ValueError("passband: transition must be > 0")
    a = STOPBAND_DB
    beta = 0.1102 * (a - 8.7)
    n = int(np.ceil((a - 7.95) / (14.36 * transition_hz / fs))) + 1
    n |= 1
    half = (n - 1) // 2
    m = np.arange(n) - half
    fc, f0 = (hi - lo) / 2 / fs, (hi + lo) / 2 / fs
    lp = 2 * fc * np.sinc(2 * fc * m) * np.kaiser(n, beta)
    lp /= lp.sum()
    return lp * np.exp(1j * TAU * f0 * m)


def _agc(z: np.ndarray, fs: float, a: Agc) -> np.ndarray:
    """Gain per sample (..., n) for the complex envelope z."""
    b = agc_block(fs)
    n = z.shape[-1]
    nb = -(-n // b)
    env = np.zeros((*z.shape[:-1], nb * b))
    env[..., :n] = np.abs(z)
    p = env.reshape(*z.shape[:-1], nb, b).max(axis=-1)
    g = agc_gains(p, fs, a)
    prev = np.concatenate([g[..., :1], g[..., :-1]], axis=-1)
    ramp = np.arange(1, b + 1) / b
    gain = prev[..., None] + (g - prev)[..., None] * ramp
    return gain.reshape(*z.shape[:-1], nb * b)[..., :n]


def _agc_gains_row(p: np.ndarray, target: float, att: float, dec: float, hb: int, floor: float) -> np.ndarray:
    """agc_gains for one row, on plain floats (numpy per block is slow
    on a single row)."""
    pl = p.tolist()
    level = max(pl[0], floor)
    hang = 0
    g = [target / level]
    for pk in pl[1:]:
        if pk > level:
            level += (pk - level) * att
            hang = hb
        elif hb > 0 and pk >= level * HANG_REARM:
            hang = hb
        elif hang > 0:
            hang -= 1
        else:
            level = max(level * dec, pk, floor)
        g.append(target / max(level, floor))
    return np.array(g, p.dtype)


def agc_gains(p: np.ndarray, fs: float, a: Agc) -> np.ndarray:
    """The AGC's gain at the end of each block (..., nb) from the blocks'
    peak envelopes p (..., nb). Shared by the numpy and torch receivers."""
    b = agc_block(fs)
    dt = b / fs
    att = 1 - np.exp(-dt / a.attack_s) if a.attack_s > 0 else 1.0
    dec = np.exp(-dt / a.decay_s) if a.decay_s > 0 else 0.0
    hb = int(round(a.hang_s / dt))
    floor = a.target / 10 ** (a.max_gain_db / 20)
    if p.ndim == 1:
        return _agc_gains_row(p, a.target, att, dec, hb, floor)
    level = np.maximum(p[..., 0], floor)
    hang = np.zeros(level.shape, np.int64)
    g = np.empty_like(p)
    g[..., 0] = a.target / level
    for k in range(1, p.shape[-1]):
        pk = p[..., k]
        up = pk > level
        rearm = (hb > 0) & ~up & (pk >= level * HANG_REARM)
        held = ~up & ~rearm & (hang > 0)
        fall = ~up & ~rearm & ~held
        level = np.where(up, level + (pk - level) * att,
                         np.where(fall, np.maximum(np.maximum(level * dec, pk), floor), level))
        hang = np.where(up | rearm, hb, np.where(held, hang - 1, hang))
        g[..., k] = a.target / np.maximum(level, floor)
    return g


def clock_offset(x: np.ndarray, ppm) -> np.ndarray:
    """The receiver's sample clock off by ppm (scalar or per row): output
    sample k is the input at k / (1 + ppm 1e-6). A fast clock (ppm > 0)
    stretches the signal: tones come out lower by that factor, a timing
    recovery sees the symbol rate off by it. Same length as x; past the
    end of the input it is zero."""
    x = np.asarray(x)
    n = x.shape[-1]
    shape = x.shape[:-1]
    rows = int(np.prod(shape)) if shape else 1
    e = np.broadcast_to(np.asarray(ppm, float).reshape(-1, 1) if np.ndim(ppm) else np.asarray(ppm, float), (rows, 1)) * 1e-6
    d = np.arange(n)[None, :] * (e / (1 + e))
    from .vary import moving_delay
    return moving_delay(x.reshape(rows, n), d).reshape(x.shape)


class Receiver:
    """Mistuning (Hz; scalar or one per leading index; positive moves the
    audio up), a passband (lo, hi) at -6 dB or None, an Agc or None.
    apply() takes complex baseband or real audio (through its analytic
    signal; the AGC follows its envelope)."""

    def __init__(self, fs: float, passband=None, mistune_hz=0.0, agc: Agc | None = None,
                 transition_hz: float = 100.0, clock_ppm=0.0):
        """clock_ppm: the sample clock's error, applied first (see
        clock_offset)."""
        self.fs = float(fs)
        self.clock_ppm = clock_ppm
        self.passband = passband
        self.mistune_hz = mistune_hz
        self.agc = agc
        self.taps = None if passband is None else bandpass_taps(self.fs, *passband, transition_hz)

    @classmethod
    def ssb(cls, fs: float, mistune_hz=0.0, agc: Agc | None = Agc()) -> "Receiver":
        """300-2700 Hz, default AGC."""
        return cls(fs, (300.0, 2700.0), mistune_hz, agc)

    def apply(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x)
        if np.any(np.asarray(self.clock_ppm) != 0):
            x = clock_offset(x, self.clock_ppm)
        real = not np.iscomplexobj(x)
        n = x.shape[-1]
        ctype = np.complex64 if x.dtype in (np.float32, np.complex64) else np.complex128
        if real:
            m = fast_len(2 * n)
            f = np.fft.fftfreq(m)
            z = np.fft.ifft(np.fft.fft(x, m, axis=-1) * np.where(f > 0, 2.0, np.where(f == 0, 1.0, 0.0)),
                            axis=-1)[..., :n]
        else:
            z = x
        z = z.astype(ctype)
        mis = np.asarray(self.mistune_hz, float)
        if np.any(mis != 0):
            t = np.arange(n) / self.fs
            z = z * np.exp(1j * TAU * mis[..., None] * t).astype(ctype)
        if self.taps is not None:
            k = len(self.taps)
            m = fast_len(n + k - 1)
            h = np.fft.fft(self.taps, m)
            z = np.fft.ifft(np.fft.fft(z, m, axis=-1) * h, axis=-1)[..., (k - 1) // 2:(k - 1) // 2 + n]
            z = z.astype(ctype)
        if self.agc is not None:
            z = z * _agc(z, self.fs, self.agc).astype(z.real.dtype)
        if real:
            return z.real.astype(x.dtype if x.dtype.kind == "f" else np.float64)
        return z
