"""The PyTorch backend: the same statistics as the numpy functions, and
the same output where there is no randomness (the receiver). Runs on
CUDA if there is a GPU, else on the CPU; skipped without torch.
Run: python3 python/tests/test_torch.py   (or pytest)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import ionochannel as ic  # noqa: E402

try:
    import torch

    import ionochannel.torch as ict
except ImportError:          # pragma: no cover
    torch = None

TAU = 2 * np.pi
DEV = os.environ.get("IONOCHANNEL_TORCH_DEVICE") or ("cuda" if torch is not None and torch.cuda.is_available() else "cpu")


def gen(seed):
    return torch.Generator(DEV).manual_seed(seed)


def np_(t):
    return t.detach().cpu().numpy()


def test_rayleigh_and_doppler():
    fs, spread, shift = 200.0, 1.0, 3.0
    x = torch.ones((8, int(fs * 1000)), dtype=torch.complex64, device=DEV)
    g = np_(ict.Channel(fs, [ic.scatter(0, 1.0, spread, shift)]).apply(x, gen(1))).astype(np.complex128)
    p = np.abs(g) ** 2
    assert abs(p.mean() - 1) < 0.05, p.mean()
    for v, tol in [(0.1, 0.01), (1.0, 0.02)]:
        assert abs(np.mean(p < v * p.mean()) - (1 - np.exp(-v))) < tol
    sigma = spread / 2
    lag = int(round(np.sqrt(-np.log(0.5) / 2) / (np.pi * sigma) * fs))
    r = np.mean(g[:, lag:] * np.conj(g[:, :-lag])) / np.mean(p)
    s = np.sqrt(-np.log(abs(r)) / 2) / (np.pi * lag / fs)
    assert abs(2 * s / spread - 1) < 0.06, 2 * s
    lag = 2
    f = np.angle(np.mean(g[:, lag:] * np.conj(g[:, :-lag]))) / (TAU * lag / fs)
    assert abs(f - shift) < 0.05, f


def test_real_shift_and_fractional_delay():
    fs, n = 12000.0, 24000
    t = torch.arange(n, device=DEV, dtype=torch.float64) / fs
    x = torch.cos(TAU * 1000 * t).float()[None]
    y = np_(ict.Channel(fs, [ic.specular(0, 1.0, 10.0)]).apply(x, gen(2)))[0]
    tt = np.arange(n) / fs
    amp = lambda f: 2 * abs(np.mean(y[2000:-2000] * np.exp(-1j * TAU * f * tt[2000:-2000])))
    assert abs(amp(1010) - 1) < 0.01 and amp(990) < 0.01
    d = 0.37e-3
    z = torch.exp(1j * TAU * 1500 * t).to(torch.complex64)[None]
    y = np_(ict.Channel(fs, [ic.specular(d, 1.0)]).apply(z, gen(3)))[0]
    r = (y / np.exp(1j * TAU * 1500 * (tt - d)))[1500:-1500]
    assert np.max(np.abs(r - r.mean())) < 2e-3 and abs(abs(r.mean()) - 1) < 2e-3


def test_awgn_and_per_row_snr():
    fs = 12000.0
    x = torch.zeros((4, 120000), device=DEV)
    snr = torch.tensor([0.0, 10.0, 20.0, -10.0], device=DEV)
    y = np_(ict.awgn(x, snr, fs, gen(4), signal_power=1.0))
    want = 10 ** (-np.array([0, 10, 20, -10]) / 10) * (fs / 2) / 2500
    assert np.allclose(y.var(-1), want, rtol=0.02)


def test_receiver_matches_numpy():
    fs = 12000.0
    rng = np.random.default_rng(5)
    t = np.arange(36000) / fs
    env = np.where((t % 1) < 0.5, 1.0, 0.05)
    x = (env * np.cos(TAU * 900 * t) + 0.02 * rng.standard_normal(len(t)))[None].repeat(2, 0)
    mis = [120.0, -80.0]
    agc = ic.Agc(decay_s=0.3, hang_s=0.1)
    want = ic.Receiver(fs, (300, 2700), np.array(mis), agc).apply(x)
    got = np_(ict.Receiver(fs, (300, 2700), torch.tensor(mis), agc).apply(torch.tensor(x, device=DEV)))
    assert np.max(np.abs(got - want)) < 1e-6 * np.max(np.abs(want)) * 100, np.max(np.abs(got - want))
    z = x + 1j * x[:, ::-1]
    want = ic.Receiver(fs, (-2700, -300), -50.0, agc).apply(z.astype(np.complex64))
    got = np_(ict.Receiver(fs, (-2700, -300), -50.0, agc).apply(torch.tensor(z, dtype=torch.complex64, device=DEV)))
    assert np.max(np.abs(got - want)) < 1e-4 * np.max(np.abs(want)), np.max(np.abs(got - want))


def test_qrn_vd_and_power():
    fs = 4000.0
    for target in (3.0, 8.0):
        z = ict.qrn_noise(8, int(fs * 150), fs, target, gen(6))
        Z = torch.fft.fft(z, dim=-1)
        f = torch.fft.fftfreq(z.shape[-1], 1 / fs, device=DEV)
        e = np_(torch.fft.ifft(Z * (f.abs() <= 100), dim=-1).abs()).ravel().astype(np.float64)
        vd = 20 * np.log10(np.sqrt(np.mean(e ** 2)) / np.mean(e))
        assert abs(vd - target) < 0.5, (target, vd)
    z = np_(ict.qrn_noise(8, 8000 * 50, 8000.0, 4.0, gen(7)))
    assert abs(np.mean(np.abs(z) ** 2) - 1) < 0.1


def test_smps_arcing():
    fs = 12000.0
    z = np_(ict.smps_noise(2, int(fs * 4), fs, 700.0, gen(8),
                           ic.Smps(spacing_hz=2500, am_depth=0.4, fm_dev_hz=0, drift_hz=0)))
    t = np.arange(z.shape[-1]) / fs
    amp = lambda zz, f: np.abs(np.mean(zz * np.exp(-1j * TAU * f * t), -1))
    for f in (-4300, -1800, 3200, 5700):
        assert np.allclose(amp(z, f) / amp(z, 700), 1, atol=1e-3)
    assert np.allclose(amp(z, 800) / amp(z, 700), 0.2, atol=1e-3)
    z = np_(ict.arcing_noise(4, int(fs * 60), fs, gen(9), ic.Arcing(jitter_s=1e-4)))
    assert abs(np.mean(np.abs(z) ** 2) - 1) < 0.1
    half = int(fs / 100)
    p = (np.abs(z[0, : z.shape[1] // half * half]) ** 2).reshape(-1, half).sum(0)
    k = np.argmax(p)
    assert sum(p[(k + d) % half] for d in range(-12, 13)) / p.sum() > 0.9
    z = np_(ict.arcing_noise(8, int(fs * 60), fs, gen(10), ic.Arcing(on_s=1.0, off_s=3.0)))
    e = (np.abs(z[:, : z.shape[1] // half * half]) ** 2).reshape(8, -1, half).sum(-1)
    assert abs(np.mean(e > 1.0) - 0.25) < 0.05


def test_stations():
    fs = 12000.0
    z = np_(ict.cw_noise(2, int(fs * 30), fs, 800.0, gen(11), ic.Cw(text="PARIS")))
    p = np.abs(z) ** 2
    assert abs(p.max() - 1) < 1e-4 and abs(p.mean() - 0.44) < 0.01
    z = np_(ict.am_noise(2, int(fs * 10), fs, 1000.0, gen(12), ic.Am(audio_hz=3000)))
    t = np.arange(z.shape[-1]) / fs
    assert np.allclose(np.abs(np.mean(z * np.exp(-1j * TAU * 1000 * t), -1)), 1, atol=0.01)
    side = np.mean(np.abs(z) ** 2) - 1
    assert 0.05 < side < 0.1
    z = np_(ict.radar_noise(1, int(fs * 2), fs, 0.0, gen(13), ic.Radar()))[0].astype(np.complex128)
    a = np.abs(z)
    full = (a[1:] > 0.999) & (a[:-1] > 0.999)
    fi = np.angle(z[1:] * np.conj(z[:-1])) * fs / TAU
    slope = np.diff(fi)[full[1:] & full[:-1]] * fs
    assert abs(np.median(slope) / 1e6 - 1) < 0.01


def test_mix_noise_matches_awgn():
    fs = 12000.0
    rec = torch.randn(int(fs * 30), device=DEV, generator=gen(14))
    x = torch.zeros((8, int(fs * 4)), device=DEV)
    y = np_(ict.mix_noise(x, rec, 0.0, fs, gen(15), signal_power=1.0))
    assert np.allclose(y.var(-1), 2.4, rtol=0.02)


def test_time_varying_paths():
    fs = 1000.0
    x = torch.ones((2, 4000), dtype=torch.complex64, device=DEV)
    y = np_(ict.Channel(fs, [ic.specular(0, 1.0)], vary=[ic.Vary(shift_hz=(ic.Sine(5.0, 1.0),))]).apply(x, gen(16)))
    fi = np.angle(y[:, 1:] * np.conj(y[:, :-1])) * fs / TAU
    assert np.max(np.abs(fi - 5 * np.sin(TAU * (np.arange(3999) + 0.5) / fs))) < 0.05
    v = ic.Vary(power_db=(ic.Ramp(1.0, 2.0, 0.0, -20.0),))
    a = np.abs(np_(ict.Channel(fs, [ic.specular(0, 1.0)], vary=[v]).apply(x, gen(17))))
    assert np.allclose(a[:, 500], 1, atol=1e-5) and np.allclose(a[:, 2500], 0.1, atol=1e-5)
    fs = 12000.0
    t = np.arange(int(fs * 4)) / fs
    z = torch.tensor(np.exp(1j * TAU * 1000 * t)[None], dtype=torch.complex64, device=DEV)
    v = ic.Vary(delay_s=(ic.Sine(0.2e-3, 2.0),))
    y = np_(ict.Channel(fs, [ic.specular(1e-3, 1.0)], vary=[v]).apply(z, gen(18)))[0]
    tau = 1e-3 + 0.2e-3 * np.sin(TAU * t / 2)
    d = np.angle(y * np.exp(-1j * TAU * 1000 * t)) + TAU * 1000 * tau
    d = (d - d[2000] + np.pi) % TAU - np.pi
    assert np.max(np.abs(d[2000:])) < 0.01
    fs, secs = 100.0, 2000.0
    v = ic.Vary(spread_oct=(ic.Ramp(secs / 2, secs / 2, 0.0, 1.0),))
    g = np_(ict.Channel(fs, [ic.scatter(0, 1.0, 1.0)], vary=[v]).apply(
        torch.ones((4, int(fs * secs)), dtype=torch.complex64, device=DEV), gen(19))).astype(np.complex128)
    def spread(h, guess):
        lag = int(round(np.sqrt(np.log(0.5) / -2) / (np.pi * guess / 2) * fs))
        r = np.mean(h[:, lag:] * np.conj(h[:, :-lag])) / np.mean(np.abs(h) ** 2)
        return 2 * np.sqrt(-np.log(abs(r)) / 2) / (np.pi * lag / fs)
    half = g.shape[1] // 2
    assert abs(spread(g[:, :half], 1.0) - 1) < 0.06 and abs(spread(g[:, half:], 2.0) - 2) < 0.12


def test_clock_offset():
    fs = 12000.0
    t = np.arange(int(fs * 10)) / fs
    x = torch.tensor(np.exp(1j * TAU * 1000 * t)[None], dtype=torch.complex64, device=DEV)
    y = np_(ict.clock_offset(x, 100.0))[0].astype(np.complex128)
    fi = np.angle(y[1:] * np.conj(y[:-1])) * fs / TAU
    assert abs(fi[100:-1000].mean() - 1000 / (1 + 1e-4)) < 1e-3
    r = ict.Receiver(fs, clock_ppm=torch.tensor([50.0, -50.0])).apply(torch.zeros((2, 1000), device=DEV))
    assert r.shape == (2, 1000)


def test_rtty_psk31():
    fs = 12000.0
    z = np_(ict.rtty_noise(2, int(fs * 10), fs, 1000.0, gen(20))).astype(np.complex128)
    fi = np.angle(z[:, 1:] * np.conj(z[:, :-1])) * fs / TAU
    assert np.mean((np.abs(fi - 1085) < 0.5) | (np.abs(fi - 915) < 0.5)) > 0.999
    z = np_(ict.psk31_noise(1, int(fs * 20), fs, 1000.0, gen(21), ic.Psk31(idle=True)))[0]
    t = np.arange(z.shape[-1]) / fs
    a = lambda f: abs(np.mean(z * np.exp(-1j * TAU * f * t)))  # noqa: E731
    assert abs(a(984.375) - 0.5) < 0.01 and abs(a(1015.625) - 0.5) < 0.01 and a(1000) < 0.01


def test_bursts():
    z = np_(ict.bursts_noise(4, int(12000 * 60), 12000.0, gen(22), ic.Bursts()))
    on = np.abs(z) > 0
    assert abs(on.mean() - 2 / 22) < 0.01 and abs(np.mean(np.abs(z[on]) ** 2) - 1) < 0.02


def test_population():
    spec = ic.Population(rate_per_s=0.5, mean_duration_s=20.0, level_db=-6.0, scale_db=0.0, band_hz=(500.0, 2500.0))
    z = np_(ict.population_noise(4, int(12000 * 10), 12000.0, gen(23), spec))
    p = np.abs(z) ** 2
    assert p.max() >= 10 ** -0.6 * 0.99 and np.isfinite(z).all()
    one = ic.Population(rate_per_s=1e-6, mean_duration_s=1e6, level_db=0.0, scale_db=0.0, band_hz=(1000.0, 1000.0001))
    z = np_(ict.population_noise(1, 12000, 12000.0, gen(24), one))
    assert z.shape == (1, 12000)


if __name__ == "__main__":
    if torch is None:
        print("skip: no torch")
        sys.exit(0)
    print("device", DEV)
    bad = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("ok  ", name)
            except AssertionError as e:
                bad += 1
                print("FAIL", name, e)
    sys.exit(1 if bad else 0)
