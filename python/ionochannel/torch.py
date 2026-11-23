"""PyTorch backend: the numpy API on tensors, for making training data on
the GPU.

    import torch, ionochannel as ic, ionochannel.torch as ict
    g = torch.Generator("cuda").manual_seed(1)
    x = torch.zeros(128, 48000, device="cuda")              # real audio, or complex I/Q
    y = ict.Channel(12000, ic.preset("poor")).apply(x, g)
    y = ict.awgn(y, snr_db=torch.rand(128, generator=g, device="cuda") * 30 - 10,
                 fs=12000, gen=g, signal_power=ict.power(x))
    y = ict.Receiver.ssb(12000).apply(y)

Same models, same parameters and the same statistics as the numpy
functions; the numbers differ for the same seed. Everything runs on the
input's device in its precision (float32 / complex64 unless you pass
float64). Two parts don't run on the GPU: the AGC's block recursion
(1 ms blocks; the peaks are found on the GPU, the recursion runs on the
CPU in numpy and the gains come back) and building Morse keying, which
are both small. Linear recursions (QRN and arcing bursts, drift) run as
FFT convolutions or block-wise closed forms.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from .channel import MIN_LOW_RATE, OVERSAMPLE, TAU, Component, Path, fast_len, scatter, specular  # noqa: F401
from .interference import Arcing, Smps, smps_lines
from .noise import ALPHA as QRN_ALPHA
from .noise import TAUS as QRN_TAUS
from .noise import VD_GAUSS, qrn_params, vd_bandwidth  # noqa: F401
from .presets import names, preset, presets  # noqa: F401
from .receiver import Agc, agc_block, agc_gains, bandpass_taps
from .recorded import default_band
from .stations import Am, Cw, Radar, morse_units, random_text
from .vary import FD_HALF, PHASES, Drift, Ramp, Sine, Vary, bounds, tid  # noqa: F401
from .vary import table as _vary_table

# ---------------------------------------------------------------- helpers


def _rows(x: torch.Tensor) -> tuple[torch.Tensor, tuple, int]:
    shape = tuple(x.shape[:-1])
    return x.reshape(-1, x.shape[-1]), shape, int(np.prod(shape)) if shape else 1


def _ctype(x: torch.Tensor) -> torch.dtype:
    return torch.complex128 if x.dtype in (torch.float64, torch.complex128) else torch.complex64


def _rtype(c: torch.dtype) -> torch.dtype:
    return torch.float64 if c == torch.complex128 else torch.float32


def _col(v, rows: int, device, dtype=torch.float64) -> torch.Tensor:
    """A scalar or per-row value as a (rows, 1) tensor."""
    t = torch.as_tensor(v, dtype=dtype, device=device)
    if t.numel() == 1:
        return t.reshape(1, 1).expand(rows, 1)
    return t.reshape(rows, 1)


def _randn(shape, gen, device, dtype=torch.float32):
    return torch.randn(shape, generator=gen, device=device, dtype=dtype)


def _rand(shape, gen, device, dtype=torch.float32):
    return torch.rand(shape, generator=gen, device=device, dtype=dtype)


def _crandn(shape, gen, device, ctype=torch.complex64):
    r = _rtype(ctype)
    return torch.complex(_randn(shape, gen, device, r), _randn(shape, gen, device, r)) * math.sqrt(0.5)


def _phasor(cycles: torch.Tensor, ctype) -> torch.Tensor:
    """exp(j 2 pi cycles), with cycles reduced to [0, 1) in float64 first."""
    c = torch.remainder(cycles.to(torch.float64), 1.0).to(_rtype(ctype))
    return torch.polar(torch.ones_like(c), TAU * c).to(ctype)


def _tone(f, n: int, fs: float, rows: int, device, ctype, phase=None) -> torch.Tensor:
    """exp(j 2 pi f t) (rows, n); f scalar or per row."""
    k = torch.arange(n, device=device, dtype=torch.float64)
    cyc = _col(f, rows, device) * k / fs
    if phase is not None:
        cyc = cyc + phase
    return _phasor(cyc, ctype)


def _np_seed(gen) -> np.random.Generator:
    """A numpy generator seeded from the torch one (for the CPU parts)."""
    s = torch.randint(0, 2 ** 62, (1,), generator=gen, device=gen.device).item()
    return np.random.default_rng(int(s))


def _analytic(x: torch.Tensor, m: int) -> torch.Tensor:
    X = torch.fft.fft(x, m, dim=-1)
    f = torch.fft.fftfreq(m, device=x.device)
    w = torch.where(f > 0, 2.0, torch.where(f == 0, 1.0, 0.0)).to(X.dtype)
    return torch.fft.ifft(X * w, dim=-1)


def _one_sided_real(z: torch.Tensor) -> torch.Tensor:
    """sqrt 2 * real part of the positive-frequency part of z."""
    n = z.shape[-1]
    m = fast_len(n)
    Z = torch.fft.fft(z, m, dim=-1)
    f = torch.fft.fftfreq(m, device=z.device)
    return math.sqrt(2) * torch.fft.ifft(Z * (f > 0), dim=-1)[..., :n].real


def _fir(z: torch.Tensor, taps: np.ndarray) -> torch.Tensor:
    """Linear-phase FIR by FFT, its delay removed (no latency)."""
    n, k = z.shape[-1], len(taps)
    m = fast_len(n + k - 1)
    h = torch.fft.fft(torch.as_tensor(taps, dtype=z.dtype, device=z.device), m)
    return torch.fft.ifft(torch.fft.fft(z, m, dim=-1) * h, dim=-1)[..., (k - 1) // 2:(k - 1) // 2 + n]


def _decay_conv(s: torch.Tensor, lam: float, length: int | None = None) -> torch.Tensor:
    """y[i] = sum_j s[i-j] exp(-lam j) along the last axis (real), by FFT,
    the kernel cut where it is below e^-25."""
    n = s.shape[-1]
    L = min(n, int(math.ceil(25 / lam)) + 1) if length is None else length
    m = fast_len(n + L - 1)
    h = torch.exp(-lam * torch.arange(L, device=s.device, dtype=torch.float64)).to(s.dtype)
    y = torch.fft.irfft(torch.fft.rfft(s, m, dim=-1) * torch.fft.rfft(h, m), m, dim=-1)[..., :n]
    return y.clamp_min(0)


def power(x: torch.Tensor) -> torch.Tensor:
    """Mean power along the last axis (kept for broadcasting)."""
    return (x.abs() ** 2).mean(dim=-1, keepdim=True)


def _sig_power(x, signal_power, rows):
    if signal_power is None:
        return power(x.reshape(rows, -1)).to(torch.float64)
    return _col(signal_power, rows, x.device)


def _noise_var(x, snr_db, fs, bw_hz, signal_power, rows):
    p = _sig_power(x, signal_power, rows)
    snr = _col(snr_db, rows, x.device)
    span = fs if x.is_complex() else fs / 2
    return p / 10 ** (snr / 10) * span / bw_hz


# ---------------------------------------------------------------- channel


def _scatter_gain(rows, n, fs, spread, gen, device, ctype, rate=None):
    if spread <= 0:
        raise ValueError("a scatter component needs a spread > 0 (use specular for none)")
    low = max(OVERSAMPLE * spread, MIN_LOW_RATE)
    sigma = spread / 2
    margin = int(math.ceil(4.0 / (math.pi * sigma) * low))
    if rate is not None:
        pos = (torch.cumsum(rate.to(torch.float64), dim=-1) - rate[:, :1]) * (low / fs)
        m_low = int(math.ceil(pos.max().item())) + 2
    else:
        m_low = int(math.ceil(n / fs * low)) + 2
    m = fast_len(m_low + margin)
    w = _crandn((rows, m), gen, device, ctype)
    f = torch.fft.fftfreq(m, 1 / low, device=device, dtype=torch.float64)
    h = torch.exp(-(f ** 2) / (4 * sigma ** 2))
    h = (h / torch.sqrt(torch.mean(h ** 2))).to(_rtype(ctype))
    g = torch.fft.ifft(torch.fft.fft(w, dim=-1) * h, dim=-1)[:, :m_low]
    if rate is not None:
        i0 = torch.clamp(pos.long(), max=m_low - 2)
        frac = (pos - i0).to(_rtype(ctype))
        return torch.gather(g, 1, i0) * (1 - frac) + torch.gather(g, 1, i0 + 1) * frac
    pos = torch.arange(n, device=device, dtype=torch.float64) * (low / fs)
    i0 = torch.clamp(pos.long(), max=m_low - 2)
    frac = (pos - i0).to(_rtype(ctype))
    return g[:, i0] * (1 - frac) + g[:, i0 + 1] * frac


def tap_gain(path: Path, n: int, fs: float, gen, rows: int = 1, device=None,
             ctype=torch.complex64, rate=None, scale: float = 1.0) -> torch.Tensor:
    """A path's tap gain over n samples, (rows, n) complex; `scale` and
    `rate` as in the numpy tap_gain."""
    device = device or gen.device
    g = torch.zeros((rows, n), dtype=ctype, device=device)
    for c in path.components:
        rot = _tone(c.shift_hz, n, fs, 1, device, ctype)
        if c.specular:
            ph = _phasor(_rand((rows, 1), gen, device, torch.float64), ctype)
            g = g + math.sqrt(c.power) * ph * rot
        else:
            g = g + math.sqrt(c.power) * _scatter_gain(rows, n, fs, c.spread_hz * scale, gen, device, ctype,
                                                       rate) * rot
    return g


class Channel:
    """Paths at a sample rate; apply() fades whole tensors."""

    def __init__(self, fs: float, paths, vary=None):
        """vary: one ic.Vary per path (or None), as the numpy Channel."""
        self.fs = float(fs)
        self.paths = tuple(paths)
        if not self.paths:
            raise ValueError("a channel needs at least one path")
        self.vary = tuple(vary) if vary else ()
        if self.vary and len(self.vary) != len(self.paths):
            raise ValueError("vary needs one entry per path")

    def apply(self, x: torch.Tensor, gen) -> torch.Tensor:
        """Fade x (..., n): complex baseband or real audio (through its
        analytic signal). Each leading index its own realisation."""
        xr, shape, rows = _rows(x)
        real = not x.is_complex()
        n = x.shape[-1]
        ct = _ctype(x)
        pad = int(math.ceil(max(p.delay_s for p in self.paths) * self.fs)) + 64
        m = fast_len(n + pad)
        X = torch.fft.fft(xr.to(ct), m, dim=-1)
        f = torch.fft.fftfreq(m, 1 / self.fs, device=x.device, dtype=torch.float64)
        if real:
            X = X * torch.where(f > 0, 2.0, torch.where(f == 0, 1.0, 0.0)).to(X.dtype)
            f = torch.where(f < 0, f + self.fs, f)
        y = torch.zeros((rows, n), dtype=ct, device=x.device)
        vs = self.vary or (None,) * len(self.paths)
        for p, v in zip(self.paths, vs):
            if v is not None and v.delay_s:
                z = torch.fft.ifft(X, dim=-1)[:, :n]
                dmax = max(0.0, (p.delay_s + bounds(v.delay_s)[1]) * self.fs)
                d = torch.clamp((p.delay_s + trajectory(v.delay_s, rows, n, self.fs, gen)) * self.fs, 0, dmax)
                xd = moving_delay(z, d)
            else:
                xd = torch.fft.ifft(X * _phasor(-f * p.delay_s, ct), dim=-1)[:, :n]
            if v is None or v.is_empty():
                y = y + tap_gain(p, n, self.fs, gen, rows, x.device, ct) * xd
                continue
            if v.spread_oct:
                vmax = bounds(v.spread_oct)[1]
                rate = 2.0 ** (trajectory(v.spread_oct, rows, n, self.fs, gen) - vmax)
                g = tap_gain(p, n, self.fs, gen, rows, x.device, ct, rate=rate, scale=2.0 ** vmax)
            else:
                g = tap_gain(p, n, self.fs, gen, rows, x.device, ct)
            if v.power_db:
                g = g * (10 ** (trajectory(v.power_db, rows, n, self.fs, gen) / 20)).to(_rtype(ct))
            if v.shift_hz:
                dop = trajectory(v.shift_hz, rows, n, self.fs, gen)
                cyc = torch.cat([torch.zeros((rows, 1), dtype=torch.float64, device=x.device),
                                 torch.cumsum(dop[:, :-1], dim=-1)], dim=-1) / self.fs
                g = g * _phasor(cyc, ct)
            y = y + g * xd
        out = y.real.to(x.dtype) if real else y
        return out.reshape(*shape, n)


def trajectory(terms, rows: int, n: int, fs: float, gen) -> torch.Tensor:
    """A trajectory's value (rows, n), float64, on gen's device."""
    device = gen.device
    t = torch.arange(n, device=device, dtype=torch.float64) / fs
    v = torch.zeros((rows, n), dtype=torch.float64, device=device)
    for term in terms:
        if isinstance(term, Ramp):
            if term.t1_s > term.t0_s:
                fr = torch.clamp((t - term.t0_s) / (term.t1_s - term.t0_s), 0, 1)
            else:
                fr = (t > term.t0_s).to(torch.float64)
            v = v + term.from_ + (term.to - term.from_) * fr
        elif isinstance(term, Sine):
            v = v + term.amp * torch.sin(TAU * t / term.period_s + term.phase)
        else:
            rho = math.exp(-1 / (fs * term.tau_s)) if term.tau_s > 0 else 0.0
            v = v + torch.clamp(_ar1(rows, n, rho, term.rms, gen, device), -4 * term.rms, 4 * term.rms)
    return v


def _ar1(rows, n, rho, rms, gen, device):
    """Stationary AR(1) of std rms by block-wise closed form."""
    w = _randn((rows, n), gen, device, torch.float64) * rms * math.sqrt(1 - rho * rho)
    w[:, 0] = _randn((rows,), gen, device, torch.float64) * rms
    if rho == 0:
        return w
    blk = max(1, int(30 / -math.log(rho)))
    y = torch.empty_like(w)
    carry = torch.zeros(rows, dtype=torch.float64, device=device)
    for a in range(0, n, blk):
        seg = w[:, a:a + blk]
        p = rho ** torch.arange(seg.shape[1], device=device, dtype=torch.float64)
        y[:, a:a + blk] = p * (carry[:, None] * rho + torch.cumsum(seg / p, dim=-1))
        carry = y[:, a + seg.shape[1] - 1]
    return y


def moving_delay(x: torch.Tensor, d: torch.Tensor, chunk: int = 4096) -> torch.Tensor:
    """y[n] = x(n - d[n]) for complex x (rows, n), delays d (rows, n) in
    samples, with the numpy fractional-delay table; no latency."""
    rows, n = x.shape
    rt = x.dtype if not x.is_complex() else _rtype(x.dtype)
    tab = torch.as_tensor(_vary_table(), dtype=rt, device=x.device)
    pad = int(math.ceil(max(d.max().item(), 0.0))) + 1 + FD_HALF + 2
    ahead = int(math.ceil(max(-d.min().item(), 0.0))) + 1
    xp = torch.cat([torch.zeros((rows, pad), dtype=x.dtype, device=x.device), x,
                    torch.zeros((rows, ahead + FD_HALF + 2), dtype=x.dtype, device=x.device)], dim=1)
    y = torch.empty_like(x)
    k = torch.arange(2 * FD_HALF + 1, device=x.device)
    for a in range(0, n, chunk):
        dd = d[:, a:a + chunk]
        w = torch.floor(dd)
        ph = (dd - w) * PHASES
        k0 = torch.clamp(ph.long(), max=PHASES - 1)
        al = (ph - k0)[..., None].to(rt)
        taps = tab[k0] * (1 - al) + tab[k0 + 1] * al
        i = torch.arange(a, a + dd.shape[1], device=x.device)[None, :] - w.long()
        idx = (i[..., None] + FD_HALF - k + pad).reshape(rows, -1)
        y[:, a:a + chunk] = (torch.gather(xp, 1, idx).reshape(rows, -1, k.numel()) * taps).sum(-1)
    return y


def awgn(x: torch.Tensor, snr_db, fs: float, gen, bw_hz: float = 2500.0, signal_power=None) -> torch.Tensor:
    """x plus white noise at snr_db (scalar or per row) in bw_hz, against
    signal_power (default: x's own mean power per row)."""
    xr, shape, rows = _rows(x)
    var = _noise_var(xr, snr_db, fs, bw_hz, signal_power, rows)
    if x.is_complex():
        nz = _crandn(xr.shape, gen, x.device, x.dtype)
    else:
        nz = _randn(xr.shape, gen, x.device, x.dtype)
    return (xr + nz * torch.sqrt(var).to(_rtype(_ctype(x)))).to(x.dtype).reshape(x.shape)


# ---------------------------------------------------------------- receiver


def clock_offset(x: torch.Tensor, ppm) -> torch.Tensor:
    """As the numpy clock_offset: output k is the input at k / (1 + ppm 1e-6)."""
    xr, shape, rows = _rows(x)
    n = x.shape[-1]
    e = _col(ppm, rows, x.device) * 1e-6
    d = torch.arange(n, device=x.device, dtype=torch.float64)[None, :] * (e / (1 + e))
    return moving_delay(xr, d).reshape(x.shape)


class Receiver:
    """As the numpy Receiver: clock error, mistuning (scalar or per row), a
    passband at -6 dB, an Agc; on complex baseband or real audio."""

    def __init__(self, fs: float, passband=None, mistune_hz=0.0, agc: Agc | None = None,
                 transition_hz: float = 100.0, clock_ppm=0.0):
        self.fs = float(fs)
        self.clock_ppm = clock_ppm
        self.mistune_hz = mistune_hz
        self.agc = agc
        self.taps = None if passband is None else bandpass_taps(self.fs, *passband, transition_hz)

    @classmethod
    def ssb(cls, fs: float, mistune_hz=0.0, agc: Agc | None = Agc()) -> "Receiver":
        return cls(fs, (300.0, 2700.0), mistune_hz, agc)

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        if bool(torch.any(torch.as_tensor(self.clock_ppm) != 0)):
            x = clock_offset(x, self.clock_ppm)
        xr, shape, rows = _rows(x)
        n = x.shape[-1]
        real = not x.is_complex()
        ct = _ctype(x)
        z = _analytic(xr.to(ct), fast_len(2 * n))[:, :n] if real else xr.to(ct)
        mis = torch.as_tensor(self.mistune_hz, dtype=torch.float64, device=x.device)
        if bool(torch.any(mis != 0)):
            z = z * _tone(mis, n, self.fs, rows, x.device, ct)
        if self.taps is not None:
            z = _fir(z, self.taps)
        if self.agc is not None:
            b = agc_block(self.fs)
            nb = -(-n // b)
            env = torch.zeros((rows, nb * b), dtype=_rtype(ct), device=x.device)
            env[:, :n] = z.abs()
            p = env.reshape(rows, nb, b).amax(dim=-1)
            g = torch.as_tensor(agc_gains(p.double().cpu().numpy(), self.fs, self.agc),
                                device=x.device, dtype=_rtype(ct))
            prev = torch.cat([g[:, :1], g[:, :-1]], dim=-1)
            ramp = torch.arange(1, b + 1, device=x.device, dtype=g.dtype) / b
            gain = (prev[..., None] + (g - prev)[..., None] * ramp).reshape(rows, nb * b)[:, :n]
            z = z * gain
        out = z.real.to(x.dtype) if real else z
        return out.reshape(*shape, n)


# ---------------------------------------------------------------- QRN


def _qrn_envelope(rows, n, fs, rate, g, gen, device, rt):
    pre = int(math.ceil(fs * QRN_TAUS[-1] / 2 * 25))
    span = pre + n
    k = int(torch.poisson(torch.tensor([rate * span / fs * rows], dtype=torch.float64, device=device),
                          generator=gen).item())
    pos = torch.randint(0, rows * span, (k,), generator=gen, device=device)
    b = torch.randint(0, len(QRN_TAUS), (k,), generator=gen, device=device)
    a2 = (1.0 - _rand((k,), gen, device, torch.float64)) ** (-2.0 / QRN_ALPHA)
    L = min(span, int(math.ceil(25 * fs * QRN_TAUS[-1] / 2)) + 1)
    m = fast_len(span + L - 1)
    acc = None
    for i, tau in enumerate(QRN_TAUS):
        lam = 2.0 / (fs * tau)
        sel = b == i
        s = torch.zeros(rows * span, dtype=torch.float64, device=device)
        s.index_add_(0, pos[sel], a2[sel] * (1.0 - math.exp(-lam)))
        h = torch.exp(-lam * torch.arange(L, device=device, dtype=torch.float64))
        # float64: bursts reach 1e6 times the mean, float32 round-off would
        # show in the quiet floor between them
        term = torch.fft.rfft(s.reshape(rows, span), m, dim=-1) * torch.fft.rfft(h, m)
        acc = term if acc is None else acc + term
    env = torch.fft.irfft(acc, m, dim=-1)[:, pre:pre + n].clamp_min(0)
    mean = rate / fs * QRN_ALPHA / (QRN_ALPHA - 2)
    return (g + (1 - g) * env / mean).to(rt)


def qrn_noise(shape, n: int, fs: float, vd_db: float, gen, complex_: bool = True,
              dtype=torch.float32) -> torch.Tensor:
    """QRN of mean power 1, (*shape, n), on gen's device."""
    shape = tuple(shape) if np.iterable(shape) else (shape,)
    rows = int(np.prod(shape))
    device = gen.device
    if vd_db <= VD_GAUSS + 1e-9:
        env = torch.ones((rows, n), dtype=dtype, device=device)
    else:
        env = _qrn_envelope(rows, n, fs, *qrn_params(vd_db), gen, device, dtype)
    ct = torch.complex128 if dtype == torch.float64 else torch.complex64
    w = _crandn((rows, n), gen, device, ct) if complex_ else _randn((rows, n), gen, device, dtype)
    return (torch.sqrt(env) * w).reshape(*shape, n)


def qrn(x: torch.Tensor, snr_db, vd_db: float, fs: float, gen, bw_hz: float = 2500.0,
        signal_power=None) -> torch.Tensor:
    """x plus QRN at snr_db in bw_hz (mean power), Vd vd_db in 200 Hz."""
    xr, shape, rows = _rows(x)
    var = _noise_var(xr, snr_db, fs, bw_hz, signal_power, rows)
    rt = _rtype(_ctype(x))
    nz = qrn_noise(rows, x.shape[-1], fs, vd_db, gen, x.is_complex(), rt)
    return (xr + nz * torch.sqrt(var).to(rt)).to(x.dtype).reshape(x.shape)


# ---------------------------------------------------------------- man-made


def _ou(rows, n, fs, rms, tau, gen, device):
    """Ornstein-Uhlenbeck (rows, n) at up to 200 Hz, linearly interpolated;
    the AR(1) recursion by block-wise closed form."""
    if rms <= 0:
        return torch.zeros((rows, n), dtype=torch.float64, device=device)
    low = min(fs, 200.0)
    m = int(math.ceil(n / fs * low)) + 2
    rho = math.exp(-1 / (low * tau)) if tau > 0 else 0.0
    w = _randn((rows, m), gen, device, torch.float64) * rms * math.sqrt(1 - rho * rho)
    w[:, 0] = _randn((rows,), gen, device, torch.float64) * rms
    if rho == 0:
        y = w
    else:
        blk = max(1, int(30 / -math.log(rho)))
        y = torch.empty_like(w)
        carry = torch.zeros(rows, dtype=torch.float64, device=device)
        for a in range(0, m, blk):
            seg = w[:, a:a + blk]
            p = rho ** torch.arange(seg.shape[1], device=device, dtype=torch.float64)
            y[:, a:a + blk] = p * (carry[:, None] * rho + torch.cumsum(seg / p, dim=-1))
            carry = y[:, a + seg.shape[1] - 1]
    pos = torch.arange(n, device=device, dtype=torch.float64) * (low / fs)
    i0 = torch.clamp(pos.long(), max=m - 2)
    f = pos - i0
    return y[:, i0] * (1 - f) + y[:, i0 + 1] * f


def smps_noise(shape, n: int, fs: float, offset_hz: float, gen, spec: Smps = Smps(),
               complex_: bool = True, dtype=torch.float32) -> torch.Tensor:
    """A switching-supply comb, (*shape, n), each line of mean power 1."""
    shape = tuple(shape) if np.iterable(shape) else (shape,)
    rows = int(np.prod(shape))
    device = gen.device
    ct = torch.complex128 if dtype == torch.float64 else torch.complex64
    lines = smps_lines(fs, offset_hz, spec.spacing_hz, not complex_)
    t = torch.arange(n, device=device, dtype=torch.float64) / fs
    ripple = TAU * 2 * spec.mains_hz
    th = _rand((rows, 1), gen, device, torch.float64) * TAU
    cyc = torch.cumsum(_ou(rows, n, fs, spec.drift_hz, spec.drift_s, gen, device), dim=-1) / fs
    if spec.fm_dev_hz and spec.mains_hz:
        cyc = cyc - spec.fm_dev_hz / ripple * torch.cos(ripple * t + th)
    am = (1 + spec.am_depth * torch.sin(ripple * t + th + _rand((rows, 1), gen, device, torch.float64) * TAU)) \
        / math.sqrt(1 + spec.am_depth ** 2 / 2)
    z = torch.zeros((rows, n), dtype=ct, device=device)
    for f in lines:
        a = 10 ** (spec.line_spread_db * _randn((rows, 1), gen, device, torch.float64) / 20) \
            if spec.line_spread_db else torch.ones((rows, 1), dtype=torch.float64, device=device)
        ph = _rand((rows, 1), gen, device, torch.float64)
        z = z + a.to(dtype) * _phasor(f * t + cyc + ph, ct)
    z = z * am.to(dtype)
    out = z if complex_ else math.sqrt(2) * z.real
    return out.reshape(*shape, n)


def smps(x, level_db, offset_hz, fs, gen, spec: Smps = Smps(), signal_power=None):
    """x plus a switching-supply comb, each line level_db against the signal."""
    return _add(x, lambda r, n, c, d: smps_noise(r, n, fs, offset_hz, gen, spec, c, d), level_db, signal_power)


def _gate(rows, n, fs, on_s, off_s, gen, device):
    if off_s <= 0:
        return torch.ones((rows, n), dtype=torch.bool, device=device)
    duty = on_s / (on_s + off_s)
    k = int(math.ceil(2 * n / fs / (on_s + off_s) * 1.5)) + 16
    while True:
        state0 = _rand((rows, 1), gen, device, torch.float64) < duty
        mean = torch.where((torch.arange(k, device=device) % 2 == 0)[None, :] == state0, on_s, off_s)
        d = -torch.log1p(-_rand((rows, k), gen, device, torch.float64)) * mean * fs
        edges = torch.cumsum(torch.floor(d) + 1, dim=-1)
        if bool(torch.all(edges[:, -1] >= n)):
            break
        k *= 2
    t = torch.arange(n, device=device, dtype=torch.float64).expand(rows, n).contiguous()
    i = torch.searchsorted(edges, t, right=True)
    return (i % 2 == 0) == state0


def arcing_noise(shape, n: int, fs: float, gen, spec: Arcing = Arcing(), complex_: bool = True,
                 dtype=torch.float32) -> torch.Tensor:
    """Arcing buzz, (*shape, n), mean power 1."""
    shape = tuple(shape) if np.iterable(shape) else (shape,)
    rows = int(np.prod(shape))
    device = gen.device
    half = 1 / (2 * spec.mains_hz)
    lam = 1 / (fs * spec.burst_s / 2)
    pre = int(math.ceil(25 / lam))
    N = n + pre
    gate = _gate(rows, N, fs, spec.on_s, spec.off_s, gen, device)
    s = torch.zeros((rows, N), dtype=torch.float64, device=device)
    k = int(math.ceil(N / fs / half)) + 1
    for _ in range(spec.sources):
        phase = _rand((rows, 1), gen, device, torch.float64) * half
        tb = phase + torch.arange(k, device=device, dtype=torch.float64) * half \
            + _randn((rows, k), gen, device, torch.float64) * spec.jitter_s
        ib = torch.floor(tb * fs).long()
        e = 10 ** (spec.spread_db * _randn((rows, k), gen, device, torch.float64) / 10)
        ok = (ib >= 0) & (ib < N)
        r = torch.arange(rows, device=device)[:, None].expand(rows, k)
        ib_c = ib.clamp(0, N - 1)
        ok = ok & gate[r, ib_c]
        s.index_put_((r[ok], ib_c[ok]), e[ok] * (1 - math.exp(-lam)), accumulate=True)
    env = _decay_conv(s.to(dtype), lam)[:, pre:]
    duty = 1.0 if spec.off_s <= 0 else spec.on_s / (spec.on_s + spec.off_s)
    mean = spec.sources * 2 * spec.mains_hz / fs * duty * math.exp((spec.spread_db * math.log(10) / 10) ** 2 / 2)
    ct = torch.complex128 if dtype == torch.float64 else torch.complex64
    w = _crandn((rows, n), gen, device, ct) if complex_ else _randn((rows, n), gen, device, dtype)
    return (torch.sqrt(env / mean) * w).reshape(*shape, n)


def arcing(x, snr_db, fs, gen, spec: Arcing = Arcing(), bw_hz: float = 2500.0, signal_power=None):
    """x plus arcing buzz at snr_db in bw_hz (mean power, as awgn)."""
    xr, shape, rows = _rows(x)
    var = _noise_var(xr, snr_db, fs, bw_hz, signal_power, rows)
    rt = _rtype(_ctype(x))
    nz = arcing_noise(rows, x.shape[-1], fs, gen, spec, x.is_complex(), rt)
    return (xr + nz * torch.sqrt(var).to(rt)).to(x.dtype).reshape(x.shape)


# ---------------------------------------------------------------- stations


def _add(x, fn, level_db, signal_power):
    xr, shape, rows = _rows(x)
    p = _sig_power(xr, signal_power, rows)
    lv = _col(level_db, rows, x.device)
    rt = _rtype(_ctype(x))
    nz = fn(rows, x.shape[-1], x.is_complex(), rt)
    return (xr + nz * torch.sqrt(p * 10 ** (lv / 10)).to(rt)).to(x.dtype).reshape(x.shape)


def _place(z, fs, offset_hz, fade_hz, gen, complex_):
    rows, n = z.shape
    ph = _rand((rows, 1), gen, z.device, torch.float64)
    z = z * _tone(offset_hz, n, fs, rows, z.device, z.dtype, ph)
    if fade_hz > 0:
        z = z * tap_gain(scatter(0.0, 1.0, fade_hz), n, fs, gen, rows, z.device, z.dtype)
    return z if complex_ else _one_sided_real(z)


def _edges(key: torch.Tensor, rise: int) -> torch.Tensor:
    if rise <= 1:
        return key
    i = torch.arange(rise, device=key.device, dtype=torch.float64)
    e = 0.5 - 0.5 * torch.cos(math.pi * (i + 0.5) / rise)
    k = torch.diff(torch.cat([torch.zeros(1, device=key.device, dtype=torch.float64), e]))
    k = (k / k.sum()).to(key.dtype)
    n = key.shape[-1]
    m = fast_len(n + rise)
    y = torch.fft.irfft(torch.fft.rfft(key, m, dim=-1) * torch.fft.rfft(k, m), m, dim=-1)[..., :n]
    return y.clamp(0, 1)


def _cw_keys(rows, n, fs, spec: Cw, rng: np.random.Generator) -> np.ndarray:
    """Square keying (rows, n) bool, built on the CPU from edge lists."""
    unit = 1.2 / spec.wpm * fs
    d = np.zeros((rows, n + 1), np.int32)
    for r in range(rows):
        seq = morse_units(spec.text if spec.text else random_text(rng))
        total = sum(u for _, u in seq)
        pos = -rng.random() * total * unit
        while pos < n:
            for down, u in seq:
                a, b = int(round(pos)), int(round(pos + u * unit))
                if down and b > 0 and a < n:
                    d[r, max(a, 0)] += 1
                    d[r, min(b, n)] -= 1
                pos += u * unit
                if pos >= n:
                    break
            if not spec.text:
                seq = morse_units(random_text(rng))
    return np.cumsum(d[:, :n], axis=-1) > 0


def cw_noise(shape, n, fs, offset_hz, gen, spec: Cw = Cw(), complex_=True, dtype=torch.float32):
    """A CW station alone, (*shape, n), key-down power 1."""
    shape = tuple(shape) if np.iterable(shape) else (shape,)
    rows = int(np.prod(shape))
    ct = torch.complex128 if dtype == torch.float64 else torch.complex64
    key = torch.as_tensor(_cw_keys(rows, n, fs, spec, _np_seed(gen)), device=gen.device).to(dtype)
    env = _edges(key, int(round(spec.rise_s * fs))).to(ct)
    return _place(env, fs, offset_hz, spec.fade_hz, gen, complex_).reshape(*shape, n)


def cw(x, level_db, offset_hz, fs, gen, spec: Cw = Cw(), signal_power=None):
    """x plus a CW station, key-down power level_db against the signal."""
    return _add(x, lambda r, n, c, d: cw_noise(r, n, fs, offset_hz, gen, spec, c, d), level_db, signal_power)


def speech_like(rows, n, fs, gen, dtype=torch.float32):
    """As the numpy speech_like(), mean power 1 per row."""
    device = gen.device
    m = fast_len(n)
    f = torch.fft.rfftfreq(m, 1 / fs, device=device, dtype=torch.float64)
    jf = 1j * f
    h = ((jf / 200) / (1 + jf / 200) / (1 + jf / 500)).to(torch.complex64 if dtype == torch.float32 else torch.complex128)
    hs = (1 / (1 + jf / 3) ** 2).to(h.dtype)
    x = torch.fft.irfft(torch.fft.rfft(_randn((rows, n), gen, device, dtype), m, dim=-1) * h, m, dim=-1)[:, :n]
    syl = torch.fft.irfft(torch.fft.rfft(_randn((rows, n), gen, device, dtype), m, dim=-1) * hs, m, dim=-1)[:, :n]
    x = x * syl
    return x / torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True))


def am_noise(shape, n, fs, offset_hz, gen, spec: Am = Am(), audio=None, complex_=True, dtype=torch.float32):
    """An AM station alone, (*shape, n), carrier power 1."""
    shape = tuple(shape) if np.iterable(shape) else (shape,)
    rows = int(np.prod(shape))
    device = gen.device
    ct = torch.complex128 if dtype == torch.float64 else torch.complex64
    if audio is None:
        a = speech_like(rows, n, fs, gen, dtype)
    else:
        a = torch.as_tensor(audio, dtype=dtype, device=device).expand(*shape, n).reshape(rows, n)
    a = torch.clamp(0.3 * a / torch.sqrt(torch.mean(a ** 2, dim=-1, keepdim=True) + 1e-30), -1, 1)
    z = (1 + spec.depth * a).to(ct)
    if spec.audio_hz < fs / 2:
        z = _fir(z, bandpass_taps(fs, -spec.audio_hz, spec.audio_hz, min(500.0, spec.audio_hz / 4)))
    return _place(z, fs, offset_hz, spec.fade_hz, gen, complex_).reshape(*shape, n)


def am(x, level_db, offset_hz, fs, gen, spec: Am = Am(), audio=None, signal_power=None):
    """x plus an AM station, carrier power level_db against the signal."""
    return _add(x, lambda r, n, c, d: am_noise(r, n, fs, offset_hz, gen, spec, audio, c, d), level_db, signal_power)


def radar_noise(shape, n, fs, offset_hz, gen, spec: Radar = Radar(), complex_=True, dtype=torch.float32):
    """An FMCW radar alone, (*shape, n), power 1 while in the band."""
    shape = tuple(shape) if np.iterable(shape) else (shape,)
    rows = int(np.prod(shape))
    device = gen.device
    ct = torch.complex128 if dtype == torch.float64 else torch.complex64
    t = torch.arange(n, device=device, dtype=torch.float64) / fs
    period = 1 / spec.rate_hz
    k = spec.sweep_hz * spec.rate_hz
    tau = torch.remainder(t + _rand((rows, 1), gen, device, torch.float64) * period, period)
    off = _col(offset_hz, rows, device)
    f0 = off - spec.sweep_hz / 2
    f = f0 + k * tau
    lo, hi = (0.0, fs / 2) if not complex_ else (-fs / 2, fs / 2)
    e = spec.edge * (hi - lo)
    w = torch.clamp(torch.minimum(f - lo, hi - f) / e, 0, 1)
    w = 0.5 - 0.5 * torch.cos(math.pi * w)
    z = w.to(dtype) * _phasor(f0 * tau + 0.5 * k * tau ** 2 + _rand((rows, 1), gen, device, torch.float64), ct)
    if spec.fade_hz > 0:
        z = z * tap_gain(scatter(0.0, 1.0, spec.fade_hz), n, fs, gen, rows, device, ct)
    out = z if complex_ else math.sqrt(2) * z.real
    return out.reshape(*shape, n)


def radar(x, level_db, offset_hz, fs, gen, spec: Radar = Radar(), signal_power=None):
    """x plus an FMCW radar, in-band power level_db against the signal."""
    return _add(x, lambda r, n, c, d: radar_noise(r, n, fs, offset_hz, gen, spec, c, d), level_db, signal_power)


# ---------------------------------------------------------------- recorded noise


def band_density(x: torch.Tensor, fs: float, band) -> torch.Tensor:
    """Mean spectral density over band per row (keepdim); one-sided for real."""
    n = x.shape[-1]
    X = torch.fft.fft(x, dim=-1)
    f = torch.fft.fftfreq(n, 1 / fs, device=x.device, dtype=torch.float64)
    lo, hi = band
    sel = (f >= lo) & (f <= hi)
    if not x.is_complex():
        sel = sel | ((-f >= lo) & (-f <= hi))
    p = torch.sum(X[..., sel].abs().to(torch.float64) ** 2, dim=-1, keepdim=True) / n ** 2
    return p / (hi - lo)


def _loop(rec: torch.Tensor, fs: float) -> torch.Tensor:
    m = rec.shape[-1]
    L = min(int(0.01 * fs), m // 4)
    if L < 2:
        return rec
    t = (torch.arange(L, device=rec.device, dtype=torch.float64) + 0.5) / L * (math.pi / 2)
    out = rec[..., : m - L].clone()
    s, c = torch.sin(t).to(rec.real.dtype), torch.cos(t).to(rec.real.dtype)
    out[..., :L] = rec[..., :L] * s + rec[..., m - L:] * c
    return out


def recorded_noise(rec: torch.Tensor, shape, n: int, fs: float, gen) -> torch.Tensor:
    """Random segments (*shape, n) of a recording (m,) or (k, m) on gen's device."""
    shape = tuple(shape) if np.iterable(shape) else (shape,)
    rows = int(np.prod(shape))
    rec = torch.as_tensor(rec, device=gen.device)
    rec = rec.reshape(1, -1) if rec.ndim == 1 else rec
    k, m = rec.shape
    if m >= n:
        src = rec
        start = torch.randint(0, m - n + 1, (rows, 1), generator=gen, device=gen.device)
        idx = start + torch.arange(n, device=gen.device)
    else:
        src = _loop(rec, fs)
        start = torch.randint(0, src.shape[-1], (rows, 1), generator=gen, device=gen.device)
        idx = (start + torch.arange(n, device=gen.device)) % src.shape[-1]
    which = torch.randint(0, k, (rows, 1), generator=gen, device=gen.device)
    return src[which, idx].reshape(*shape, n)


def mix_noise(x: torch.Tensor, rec, snr_db, fs: float, gen, bw_hz: float = 2500.0, band=None,
              signal_power=None, per_segment: bool = True) -> torch.Tensor:
    """x plus segments of recorded noise at snr_db (as the numpy mix_noise)."""
    xr, shape, rows = _rows(x)
    rec = torch.as_tensor(rec, device=x.device)
    if rec.is_complex() != x.is_complex():
        raise ValueError("recording and signal must both be real or both complex (I/Q)")
    rec = rec.to(x.dtype)
    band = default_band(fs, x.is_complex()) if band is None else band
    p = _sig_power(xr, signal_power, rows)
    snr = _col(snr_db, rows, x.device)
    nz = recorded_noise(rec, rows, x.shape[-1], fs, gen)
    d = band_density(nz if per_segment else rec.reshape(-1, rec.shape[-1]), fs, band)
    if not per_segment:
        d = d.mean()
    g = torch.sqrt(p / 10 ** (snr / 10) / (d.clamp_min(1e-300) * bw_hz)).to(_rtype(_ctype(x)))
    return (xr + nz * g).to(x.dtype).reshape(x.shape)
