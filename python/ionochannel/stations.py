"""Interfering stations: CW, AM broadcast and sweeping (FMCW) radar. Each
sits at an offset in the band, at a level in dB against the signal's
power, and can fade on its own (flat Rayleigh fading with a Doppler
spread `fade_hz`). For a plain carrier use smps() with am_depth=0,
fm_dev_hz=0 and a spacing wider than the band.

The defaults are illustrative, not measured. Each leading index is its
own realisation. Real outputs keep only the positive frequencies, so
nothing folds through 0 Hz. The Rust crate's stations module implements
the same models.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .channel import TAU, fast_len, scatter, tap_gain
from .receiver import bandpass_taps

MORSE = {
    "A": ".-", "B": "-...", "C": "-.-.", "D": "-..", "E": ".", "F": "..-.", "G": "--.", "H": "....",
    "I": "..", "J": ".---", "K": "-.-", "L": ".-..", "M": "--", "N": "-.", "O": "---", "P": ".--.",
    "Q": "--.-", "R": ".-.", "S": "...", "T": "-", "U": "..-", "V": "...-", "W": ".--", "X": "-..-",
    "Y": "-.--", "Z": "--..", "0": "-----", "1": ".----", "2": "..---", "3": "...--", "4": "....-",
    "5": ".....", "6": "-....", "7": "--...", "8": "---..", "9": "----.", "/": "-..-.", "?": "..--..",
    "=": "-...-", ".": ".-.-.-", ",": "--..--",
}
GROUP_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"


def morse_units(text: str) -> list[tuple[bool, int]]:
    """(key down, length in dit units) for a text, PARIS timing: dit 1,
    dah 3, gaps 1 inside a character, 3 between characters, 7 between
    words; a word gap after the end."""
    out: list[tuple[bool, int]] = []
    for w, word in enumerate(text.upper().split()):
        if w:
            out.append((False, 7))
        for c, ch in enumerate(word):
            code = MORSE.get(ch)
            if code is None:
                continue
            if c:
                out.append((False, 3))
            for e, sym in enumerate(code):
                if e:
                    out.append((False, 1))
                out.append((True, 1 if sym == "." else 3))
    out.append((False, 7))
    return out


def random_text(rng, words: int = 20) -> str:
    """Random groups of 2-6 characters."""
    lens = rng.integers(2, 7, words)
    chars = np.array(list(GROUP_CHARS))[rng.integers(0, len(GROUP_CHARS), int(lens.sum()))]
    cuts = np.cumsum(lens)[:-1]
    return " ".join("".join(w) for w in np.split(chars, cuts))


def _edges(key: np.ndarray, rise: int) -> np.ndarray:
    """Square keying (rows, n) to raised-cosine edges `rise` samples long."""
    if rise <= 1:
        return key.astype(float)
    k = 0.5 - 0.5 * np.cos(np.pi * (np.arange(rise) + 0.5) / rise)
    k = np.diff(np.r_[0.0, k])                       # derivative of the edge shape
    k /= k.sum()
    m = fast_len(key.shape[-1] + rise)
    y = np.fft.irfft(np.fft.rfft(key.astype(float), m) * np.fft.rfft(k, m), m)[..., :key.shape[-1]]
    return np.clip(y, 0.0, 1.0)


@dataclass(frozen=True)
class Cw:
    wpm: float = 20.0
    rise_s: float = 0.005
    fade_hz: float = 0.0
    text: str | None = None


def _place(z: np.ndarray, fs: float, offset_hz, fade_hz: float, rng, complex_: bool) -> np.ndarray:
    """Complex envelope (rows, n) to the offset, faded, real if asked."""
    rows, n = z.shape
    t = np.arange(n) / fs
    off = np.asarray(offset_hz, float).reshape(-1, 1) if np.ndim(offset_hz) else offset_hz
    z = z * np.exp(1j * (TAU * off * t + rng.random((rows, 1)) * TAU))
    if fade_hz > 0:
        z = z * tap_gain(scatter(0.0, 1.0, fade_hz), n, fs, rng, (rows,))
    if complex_:
        return z
    m = fast_len(n)
    f = np.fft.fftfreq(m)
    z = np.fft.ifft(np.fft.fft(z, m, axis=-1) * (f > 0), axis=-1)[..., :n]
    return math.sqrt(2) * z.real


def cw_noise(shape, n: int, fs: float, offset_hz, rng, spec: Cw = Cw(), complex_: bool = True) -> np.ndarray:
    """A CW station alone, (*shape, n), key-down power 1. Random groups
    of 2-6 characters unless spec.text is given; each row starts at a
    random point of its text."""
    shape = tuple(shape) if np.iterable(shape) else (shape,)
    rows = int(np.prod(shape))
    unit = 1.2 / spec.wpm * fs
    key = np.zeros((rows, n), bool)
    for r in range(rows):
        seq = morse_units(spec.text if spec.text else random_text(rng))
        total = sum(u for _, u in seq)
        start = rng.random() * total * unit
        pos = -start
        while pos < n:
            for down, u in seq:
                a, b = int(round(pos)), int(round(pos + u * unit))
                if down and b > 0 and a < n:
                    key[r, max(a, 0):min(b, n)] = True
                pos += u * unit
                if pos >= n:
                    break
            if not spec.text:
                seq = morse_units(random_text(rng))
    env = _edges(key, int(round(spec.rise_s * fs)))
    return _place(env.astype(complex), fs, np.ravel(offset_hz) if np.ndim(offset_hz) else offset_hz,
                  spec.fade_hz, rng, complex_).reshape(*shape, n)


@dataclass(frozen=True)
class Am:
    depth: float = 1.0          # peak modulation of the normalised audio
    audio_hz: float = 4500.0    # transmitted audio bandwidth (sidebands +-)
    fade_hz: float = 0.0


def speech_like(rows: int, n: int, fs: float, rng) -> np.ndarray:
    """Speech-like noise (rows, n): white noise tilted down above 500 Hz
    and up below 200 Hz (one-pole filters), times a syllabic envelope
    (noise low-passed at 3 Hz, squared). Mean power 1."""
    w = rng.standard_normal((rows, n))
    m = fast_len(n)
    f = np.fft.rfftfreq(m, 1 / fs)
    h = (1j * f / 200) / (1 + 1j * f / 200) / (1 + 1j * f / 500)
    x = np.fft.irfft(np.fft.rfft(w, m) * h, m)[..., :n]
    s = rng.standard_normal((rows, n))
    syl = np.fft.irfft(np.fft.rfft(s, m) / (1 + 1j * f / 3) ** 2, m)[..., :n]
    x = x * syl
    return x / np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True))


def am_noise(shape, n: int, fs: float, offset_hz, rng, spec: Am = Am(), audio=None,
             complex_: bool = True) -> np.ndarray:
    """An AM station alone, (*shape, n), carrier power 1. `audio`
    (n,) or (*shape, n) modulates it, else speech_like(). The audio is
    scaled to 0.3 rms and clipped at +-1 (a broadcast limiter), then
    band-limited to audio_hz."""
    shape = tuple(shape) if np.iterable(shape) else (shape,)
    rows = int(np.prod(shape))
    a = speech_like(rows, n, fs, rng) if audio is None else np.broadcast_to(np.asarray(audio, float), (*shape, n)).reshape(rows, n)
    a = np.clip(0.3 * a / np.sqrt(np.mean(a ** 2, axis=-1, keepdims=True) + 1e-30), -1, 1)
    z = (1 + spec.depth * a).astype(complex)
    if spec.audio_hz < fs / 2:
        h = bandpass_taps(fs, -spec.audio_hz, spec.audio_hz, min(500.0, spec.audio_hz / 4))
        k = len(h)
        m = fast_len(n + k - 1)
        z = np.fft.ifft(np.fft.fft(z, m, axis=-1) * np.fft.fft(h, m), axis=-1)[..., (k - 1) // 2:(k - 1) // 2 + n]
    return _place(z, fs, np.ravel(offset_hz) if np.ndim(offset_hz) else offset_hz,
                  spec.fade_hz, rng, complex_).reshape(*shape, n)


@dataclass(frozen=True)
class Radar:
    sweep_hz: float = 20e3      # swept bandwidth
    rate_hz: float = 50.0       # sweeps per second
    edge: float = 0.05          # taper at the band edges, fraction of the band
    fade_hz: float = 0.0


def radar_noise(shape, n: int, fs: float, offset_hz, rng, spec: Radar = Radar(),
                complex_: bool = True) -> np.ndarray:
    """An FMCW radar alone, (*shape, n): linear up-sweeps of sweep_hz
    centred on offset_hz, rate_hz a second, power 1 while in the band.
    Outside +-fs/2 (complex) or (0, fs/2) (real) it is tapered off, so it
    doesn't alias."""
    shape = tuple(shape) if np.iterable(shape) else (shape,)
    rows = int(np.prod(shape))
    t = np.arange(n) / fs
    period = 1 / spec.rate_hz
    k = spec.sweep_hz * spec.rate_hz                      # Hz per second
    tau = (t + rng.random((rows, 1)) * period) % period   # time into the sweep
    off = np.ravel(offset_hz)[:, None] if np.ndim(offset_hz) else offset_hz
    f = off - spec.sweep_hz / 2 + k * tau
    lo, hi = (0.0, fs / 2) if not complex_ else (-fs / 2, fs / 2)
    e = spec.edge * (hi - lo)
    w = np.clip(np.minimum(f - lo, hi - f) / e, 0, 1)
    w = 0.5 - 0.5 * np.cos(np.pi * w)
    ph = TAU * ((off - spec.sweep_hz / 2) * tau + 0.5 * k * tau ** 2)
    z = w * np.exp(1j * (ph + rng.random((rows, 1)) * TAU))
    if spec.fade_hz > 0:
        z = z * tap_gain(scatter(0.0, 1.0, spec.fade_hz), n, fs, rng, (rows,))
    out = z if complex_ else math.sqrt(2) * z.real
    return out.reshape(*shape, n)


def _add(x, nz_fn, level_db, signal_power):
    x = np.asarray(x)
    p = np.mean(np.abs(x) ** 2, axis=-1, keepdims=True) if signal_power is None else np.asarray(signal_power, float)
    lv = np.asarray(level_db, float)
    if lv.ndim and lv.shape[-1] != 1:
        lv = lv[..., None]
    nz = nz_fn(x.shape[:-1], x.shape[-1], np.iscomplexobj(x))
    return (x + nz * np.sqrt(p * 10 ** (lv / 10))).astype(x.dtype)


def cw(x, level_db, offset_hz, fs, rng, spec: Cw = Cw(), signal_power=None):
    """x plus a CW station, key-down power level_db against the signal."""
    return _add(x, lambda s, n, c: cw_noise(s, n, fs, offset_hz, rng, spec, c), level_db, signal_power)


def am(x, level_db, offset_hz, fs, rng, spec: Am = Am(), audio=None, signal_power=None):
    """x plus an AM station, carrier power level_db against the signal."""
    return _add(x, lambda s, n, c: am_noise(s, n, fs, offset_hz, rng, spec, audio, c), level_db, signal_power)


def radar(x, level_db, offset_hz, fs, rng, spec: Radar = Radar(), signal_power=None):
    """x plus an FMCW radar, in-band power level_db against the signal."""
    return _add(x, lambda s, n, c: radar_noise(s, n, fs, offset_hz, rng, spec, c), level_db, signal_power)


@dataclass(frozen=True)
class Rtty:
    """Amateur RTTY: baud rate, shift (mark the upper tone), stop bits."""
    baud: float = 45.45
    shift_hz: float = 170.0
    stop_bits: float = 1.5
    fade_hz: float = 0.0


@dataclass(frozen=True)
class Psk31:
    """PSK31: BPSK at 31.25 baud, a 0 bit reverses the phase through a
    cosine-shaped zero, a 1 bit holds it. idle: zeros only (two tones at
    +-baud/2)."""
    baud: float = 31.25
    idle: bool = False
    fade_hz: float = 0.0


def _rtty_freqs(n, fs, spec: Rtty, rng) -> np.ndarray:
    """Instantaneous frequency offset (+-shift/2) for one row: characters
    of a start bit, 5 random data bits and stop bits, back to back."""
    unit = fs / spec.baud
    per = (6 + spec.stop_bits) * unit
    chars = int(n / per) + 3
    bits = np.concatenate([np.zeros((chars, 1)), rng.integers(0, 2, (chars, 5)), np.ones((chars, 1))], axis=1)
    dur = np.tile(np.r_[np.ones(6), spec.stop_bits], chars) * unit
    edges = np.round(np.concatenate([[0.0], np.cumsum(dur)]) - rng.random() * per).astype(np.int64)
    f = np.empty(n)
    for v, a, b in zip(bits.ravel(), edges[:-1], edges[1:]):
        lo, hi = max(a, 0), min(b, n)
        if hi > lo:
            f[lo:hi] = (v - 0.5) * spec.shift_hz
    return f


def rtty_noise(shape, n, fs, offset_hz, rng, spec: Rtty = Rtty(), complex_=True):
    """An RTTY station alone, (*shape, n), power 1 (constant envelope)."""
    shape = tuple(shape) if np.iterable(shape) else (shape,)
    rows = int(np.prod(shape))
    f = np.stack([_rtty_freqs(n, fs, spec, rng) for _ in range(rows)])
    z = np.exp(1j * TAU * np.remainder(np.cumsum(f, axis=-1) / fs, 1.0))
    return _place(z, fs, np.ravel(offset_hz) if np.ndim(offset_hz) else offset_hz,
                  spec.fade_hz, rng, complex_).reshape(*shape, n)


def _psk31_bits(count, idle, rng) -> np.ndarray:
    """Random Varicode-like words (start and end with 1, no 00 inside),
    separated by 00; or all zeros when idle."""
    if idle:
        return np.zeros(count, np.int8)
    out = []
    while len(out) < count:
        k = int(rng.integers(1, 11))
        w = [1]
        for _ in range(k - 2):
            w.append(1 if w[-1] == 0 else int(rng.integers(0, 2)))
        if k > 1:
            w.append(1)
        out += w + [0, 0]
    return np.array(out[:count], np.int8)


def _psk31_env(n, fs, spec: Psk31, rng) -> np.ndarray:
    T = fs / spec.baud
    m = int(np.ceil(n / T)) + 2
    bits = _psk31_bits(m, spec.idle, rng)
    sign = np.cumprod(np.where(bits == 0, -1.0, 1.0))           # sign at the end of each symbol
    prev = np.r_[1.0, sign[:-1]]
    t = np.arange(n) + rng.random() * T
    k = (t // T).astype(np.int64)
    u = (t - k * T) / T
    return np.where(bits[k] == 0, prev[k] * np.cos(np.pi * u), sign[k])


def psk31_noise(shape, n, fs, offset_hz, rng, spec: Psk31 = Psk31(), complex_=True):
    """A PSK31 station alone, (*shape, n), peak power 1."""
    shape = tuple(shape) if np.iterable(shape) else (shape,)
    rows = int(np.prod(shape))
    env = np.stack([_psk31_env(n, fs, spec, rng) for _ in range(rows)]).astype(complex)
    return _place(env, fs, np.ravel(offset_hz) if np.ndim(offset_hz) else offset_hz,
                  spec.fade_hz, rng, complex_).reshape(*shape, n)


def rtty(x, level_db, offset_hz, fs, rng, spec: Rtty = Rtty(), signal_power=None):
    """x plus an RTTY station at level_db against the signal."""
    return _add(x, lambda s, n, c: rtty_noise(s, n, fs, offset_hz, rng, spec, c), level_db, signal_power)


def psk31(x, level_db, offset_hz, fs, rng, spec: Psk31 = Psk31(), signal_power=None):
    """x plus a PSK31 station, peak power level_db against the signal."""
    return _add(x, lambda s, n, c: psk31_noise(s, n, fs, offset_hz, rng, spec, c), level_db, signal_power)


@dataclass(frozen=True)
class Population:
    """A population of CW stations coming and going (Mendieta-Otero,
    Pérez-Álvarez & Pérez-Díaz, IEEE TEMC 2014): Poisson arrivals at
    `rate_per_s` (no default: it depends on band, time and place; their
    ARRL Field Day 2011 figure, 6.68/s, is for all amateur bands at once),
    exponential durations (mean 10 s there), frequencies uniform in
    `band_hz`, levels logistic in dB around `level_db` with scale
    `scale_db` (1/|B_k|; their Fig. 3 has B_k = -0.0849, 11.8 dB). Each
    sends "PARIS" with dot = duration / 331 (WPM = 1.2 / dot) and
    raised-cosine edges of dot / 10."""
    rate_per_s: float
    mean_duration_s: float = 10.0
    level_db: float = 0.0
    scale_db: float = 11.8
    band_hz: tuple | None = None


_PARIS = None


def _paris() -> np.ndarray:
    global _PARIS
    if _PARIS is None:
        _PARIS = np.concatenate([np.full(u, d) for d, u in morse_units("PARIS")]).astype(float)
    return _PARIS


def population_plan(n: int, fs: float, rng, spec: Population, complex_: bool = True) -> list:
    """The stations heard in one row of n samples: (start s, duration s,
    frequency Hz, level dB, WPM), including those already on at t = 0."""
    if spec.rate_per_s <= 0:
        return []
    band = spec.band_hz or ((-0.45 * fs, 0.45 * fs) if complex_ else (100.0, 0.45 * fs))
    t0 = -10 * spec.mean_duration_s
    t1 = n / fs
    k = rng.poisson(spec.rate_per_s * (t1 - t0))
    start = rng.uniform(t0, t1, k)
    dur = rng.exponential(spec.mean_duration_s, k)
    keep = start + dur > 0
    start, dur = start[keep], dur[keep]
    f = rng.uniform(band[0], band[1], len(start))
    u = rng.uniform(1e-12, 1 - 1e-12, len(start))
    level = spec.level_db + spec.scale_db * np.log(u / (1 - u))
    wpm = 1.2 * 331 / dur
    return list(zip(start, dur, f, level, wpm))


def population_noise(shape, n, fs, rng, spec: Population, complex_=True):
    """The population alone, (*shape, n): each station at its own level
    (key-down power 10^(level/10))."""
    shape = tuple(shape) if np.iterable(shape) else (shape,)
    rows = int(np.prod(shape))
    out = np.zeros((rows, n), np.complex64)
    pat = _paris()
    for r in range(rows):
        for start, dur, f, level, wpm in population_plan(n, fs, rng, spec, complex_):
            a = max(int(np.floor(start * fs)), 0)
            b = min(int(np.ceil((start + dur) * fs)), n)
            if b <= a:
                continue
            dot = dur / 331
            t = np.arange(a, b) / fs - start
            key = np.where((t >= 0) & (t < dur), pat[(t // dot).astype(np.int64) % len(pat)], 0.0)
            rise = max(1, int(round(dot / 10 * fs)))
            if rise > 1:
                i = np.arange(rise)
                kern = np.diff(np.r_[0.0, 0.5 - 0.5 * np.cos(np.pi * (i + 0.5) / rise)])
                key = np.convolve(key, kern / kern.sum())[: len(key)]
            ph = (TAU * np.remainder(f * np.arange(a, b) / fs + rng.random(), 1.0)).astype(np.float32)
            out[r, a:b] += (10 ** (level / 20) * key).astype(np.float32) * (np.cos(ph) + 1j * np.sin(ph))
    res = out if complex_ else math.sqrt(2) * out.real
    return res.reshape(*shape, n)


def population(x, fs, rng, spec: Population, signal_power=None):
    """x plus a population of CW stations; levels against the signal's power."""
    return _add(x, lambda s, n, c: population_noise(s, n, fs, rng, spec, c), 0.0, signal_power)
