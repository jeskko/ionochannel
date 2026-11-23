//! The channel against theory: Rayleigh statistics, Doppler spectra, level
//! crossings and fade durations, frequency correlation, delays.

use ionochannel::{Channel, Component, PathSpec, RealChannel, presets};
use num_complex::Complex32;
use std::f64::consts::{PI, TAU};

/// The tap gain over time: a constant input through the channel.
fn gains(fs: f64, paths: &[PathSpec], secs: f64, seed: u64) -> Vec<Complex32> {
    let mut ch = Channel::new(fs, paths, seed);
    let mut x = vec![Complex32::new(1.0, 0.0); (fs * secs) as usize];
    ch.process(&mut x);
    x
}

fn mean_power(g: &[Complex32]) -> f64 {
    g.iter().map(|v| v.norm_sqr() as f64).sum::<f64>() / g.len() as f64
}

/// Normalised autocorrelation at a lag (samples).
fn acf(g: &[Complex32], lag: usize) -> num_complex::Complex64 {
    let n = g.len() - lag;
    let s: num_complex::Complex64 = (0..n)
        .map(|i| {
            let a = g[i + lag];
            let b = g[i].conj();
            num_complex::Complex64::new(a.re as f64, a.im as f64)
                * num_complex::Complex64::new(b.re as f64, b.im as f64)
        })
        .sum();
    s / n as f64 / mean_power(g)
}

#[test]
fn rayleigh_power_and_depths() {
    let g = gains(200.0, &[PathSpec::scatter(0.0, 1.0, 1.0, 0.0)], 8000.0, 1);
    let p: Vec<f64> = g.iter().map(|v| v.norm_sqr() as f64).collect();
    let m = mean_power(&g);
    assert!((m - 1.0).abs() < 0.05, "mean power {m}");
    // Rayleigh: P(power < x * mean) = 1 - exp(-x)
    for (x, tol) in [(0.1, 0.01), (0.01, 0.002), (1.0, 0.02)] {
        let f = p.iter().filter(|&&v| v < x * m).count() as f64 / p.len() as f64;
        let t = 1.0 - (-x).exp();
        assert!((f - t).abs() < tol, "P(<{x}) {f:.4} vs {t:.4}");
    }
}

#[test]
fn doppler_spread_and_shift() {
    for (spread, shift) in [
        (0.1, 0.0),
        (0.5, 0.0),
        (1.0, 3.0),
        (10.0, -5.0),
        (30.0, 0.0),
        (55.0, 12.0),
    ] {
        let fs = (16.0 * spread + 4.0 * f64::abs(shift)).max(20.0);
        let secs = (4000.0 / spread).min(4000.0);
        let g = gains(fs, &[PathSpec::scatter(0.0, 1.0, spread, shift)], secs, 2);
        let sigma = spread / 2.0;
        // |R(tau)| = exp(-2 pi^2 sigma^2 tau^2) for a Gaussian spectrum; arg R = 2 pi shift tau.
        // Two lags (|R| ~ 0.7 and ~ 0.3) check the shape as well as the width.
        for target in [0.7f64, 0.3] {
            let tau = (-target.ln() / 2.0).sqrt() / (PI * sigma);
            let lag = (tau * fs).round() as usize;
            let tau = lag as f64 / fs;
            let r = acf(&g, lag);
            let s = (-(r.norm().ln()) / 2.0).sqrt() / (PI * tau);
            assert!(
                (2.0 * s / spread - 1.0).abs() < 0.06,
                "spread {spread}: measured {:.3} at |R|={target}",
                2.0 * s
            );
        }
        // the shift from the phase at a short lag (no wrap past pi)
        let lag = ((0.05 / (spread + f64::abs(shift))) * fs).round().max(1.0) as usize;
        let f = acf(&g, lag).arg() / (TAU * lag as f64 / fs);
        assert!(
            (f - shift).abs() < 0.03 * spread + 0.01,
            "shift {shift}: measured {f:.3}"
        );
    }
}

#[test]
fn level_crossings_and_fade_durations() {
    // Rice: N = sqrt(4 pi) f_rms rho exp(-rho^2), f_rms = sigma = spread / 2;
    // mean fade duration P(below) / N = (1 - exp(-rho^2)) / N.
    let spread = 1.0;
    let fs = 100.0;
    let g = gains(fs, &[PathSpec::scatter(0.0, 1.0, spread, 0.0)], 20000.0, 3);
    let rms = mean_power(&g).sqrt();
    for rho in [0.1f64, 0.316, 1.0] {
        let lvl = (rho * rms) as f32;
        let below: Vec<bool> = g.iter().map(|v| v.norm() < lvl).collect();
        let ups = below.windows(2).filter(|w| w[0] && !w[1]).count();
        let secs = g.len() as f64 / fs;
        let n = ups as f64 / secs;
        let nt = (4.0 * PI).sqrt() * (spread / 2.0) * rho * (-rho * rho).exp();
        assert!(
            (n / nt - 1.0).abs() < 0.08,
            "rho {rho}: {n:.4}/s vs {nt:.4}/s"
        );
        let afd = below.iter().filter(|&&b| b).count() as f64 / fs / ups as f64;
        let at = (1.0 - (-rho * rho).exp()) / nt;
        assert!(
            (afd / at - 1.0).abs() < 0.08,
            "rho {rho}: fade {afd:.4} s vs {at:.4} s"
        );
    }
}

#[test]
fn two_paths_frequency_correlation() {
    // Two equal paths tau apart: H(f) = g1 + g2 exp(-j 2 pi f tau),
    // corr(H(f1), H(f2)) = |cos(pi df tau)|.
    let fs = 4000.0;
    let tau = 2e-3;
    let paths = [
        PathSpec::scatter(0.0, 0.5, 1.0, 0.0),
        PathSpec::scatter(tau, 0.5, 1.0, 0.0),
    ];
    let secs = 3000.0;
    let n = (fs * secs) as usize;
    let h = |f: f64| -> Vec<num_complex::Complex64> {
        // the same seed gives the same gains whatever the input
        let mut ch = Channel::new(fs, &paths, 4);
        let mut x: Vec<Complex32> = (0..n)
            .map(|i| Complex32::from_polar(1.0, (TAU * f * i as f64 / fs) as f32))
            .collect();
        ch.process(&mut x);
        x.iter()
            .enumerate()
            .map(|(i, y)| {
                num_complex::Complex64::new(y.re as f64, y.im as f64)
                    * num_complex::Complex64::from_polar(1.0, -TAU * f * i as f64 / fs)
            })
            .collect()
    };
    let h0 = h(300.0);
    for df in [125.0, 250.0, 500.0, 62.5] {
        let h1 = h(300.0 + df);
        let c: num_complex::Complex64 = h0.iter().zip(&h1).map(|(a, b)| a * b.conj()).sum();
        let p0: f64 = h0.iter().map(|v| v.norm_sqr()).sum();
        let p1: f64 = h1.iter().map(|v| v.norm_sqr()).sum();
        let rho = c.norm() / (p0 * p1).sqrt();
        let t = (PI * df * tau).cos().abs();
        assert!((rho - t).abs() < 0.06, "df {df}: {rho:.3} vs {t:.3}");
    }
}

#[test]
fn paths_are_independent() {
    let fs = 100.0;
    let a = gains(
        fs,
        &[
            PathSpec::scatter(0.0, 1.0, 1.0, 0.0),
            PathSpec::scatter(0.0, 0.0, 1.0, 0.0),
        ],
        4000.0,
        5,
    );
    let b = gains(
        fs,
        &[
            PathSpec::scatter(0.0, 0.0, 1.0, 0.0),
            PathSpec::scatter(0.0, 1.0, 1.0, 0.0),
        ],
        4000.0,
        5,
    );
    let c: num_complex::Complex64 = a
        .iter()
        .zip(&b)
        .map(|(x, y)| {
            let p = x * y.conj();
            num_complex::Complex64::new(p.re as f64, p.im as f64)
        })
        .sum();
    let r = c.norm() / (mean_power(&a) * mean_power(&b)).sqrt() / a.len() as f64;
    assert!(r < 0.05, "path correlation {r:.3}");
}

#[test]
fn fractional_delay_is_exact_for_tones() {
    let fs = 12000.0;
    let d = 0.37e-3; // 4.44 samples
    for f in [300.0, 1500.0, 4500.0] {
        let mut ch = Channel::new(
            fs,
            &[
                PathSpec::specular(0.0, 0.0, 0.0),
                PathSpec::specular(d, 1.0, 0.0),
            ],
            6,
        );
        let lat = ch.latency() as f64;
        let n = 6000;
        let x: Vec<Complex32> = (0..n)
            .map(|i| Complex32::from_polar(1.0, (TAU * f * i as f64 / fs) as f32))
            .collect();
        let mut y = x.clone();
        ch.process(&mut y);
        // y[i] = e^{j phi0} x(i - lat - d fs): the ratio's phase is constant
        let r: Vec<num_complex::Complex64> = (100..n)
            .map(|i| {
                let ideal = num_complex::Complex64::from_polar(
                    1.0,
                    TAU * f * (i as f64 - lat - d * fs) / fs,
                );
                num_complex::Complex64::new(y[i].re as f64, y[i].im as f64) / ideal
            })
            .collect();
        let m: num_complex::Complex64 = r.iter().sum::<num_complex::Complex64>() / r.len() as f64;
        let dev = r.iter().map(|v| (v - m).norm()).fold(0.0, f64::max);
        assert!(
            (m.norm() - 1.0).abs() < 2e-3 && dev < 2e-3,
            "{f} Hz: gain {:.5}, wobble {dev:.2e}",
            m.norm()
        );
    }
}

#[test]
fn real_channel_shifts_audio_up() {
    let fs = 12000.0;
    let mut ch = RealChannel::new(
        fs,
        &[PathSpec {
            delay_s: 0.0,
            components: vec![Component::Specular {
                power: 1.0,
                shift_hz: 10.0,
            }],
        }],
        7,
    );
    let n = 24000;
    let mut x: Vec<f32> = (0..n)
        .map(|i| (TAU * 1000.0 * i as f64 / fs).cos() as f32)
        .collect();
    ch.process(&mut x);
    let lat = ch.latency();
    let amp = |f: f64| {
        let s: num_complex::Complex64 = x[lat + 2000..]
            .iter()
            .enumerate()
            .map(|(i, v)| num_complex::Complex64::from_polar(*v as f64, -TAU * f * i as f64 / fs))
            .sum();
        2.0 * s.norm() / (x.len() - lat - 2000) as f64
    };
    assert!(
        (amp(1010.0) - 1.0).abs() < 0.01,
        "1010 Hz at {:.4}",
        amp(1010.0)
    );
    assert!(
        amp(990.0) < 0.01 && amp(1000.0) < 0.05,
        "990 {:.4} 1000 {:.4}",
        amp(990.0),
        amp(1000.0)
    );
}

#[test]
fn block_size_and_seed() {
    let p = presets::get("poor").unwrap().paths;
    let x: Vec<Complex32> = (0..20000)
        .map(|i| Complex32::new((i as f32 * 0.01).sin(), (i as f32 * 0.013).cos()))
        .collect();
    let run = |block: usize, seed: u64| {
        let mut ch = Channel::new(12000.0, &p, seed);
        let mut y = x.clone();
        for c in y.chunks_mut(block) {
            ch.process(c);
        }
        y
    };
    let a = run(20000, 1);
    assert_eq!(a, run(1, 1));
    assert_eq!(a, run(333, 1));
    assert_ne!(a, run(20000, 2));
}

#[test]
fn presets_parse_and_sum_to_one() {
    let all = presets::all();
    assert!(all.len() >= 16);
    for p in all {
        let t: f64 = p.paths.iter().map(PathSpec::power).sum();
        assert!((t - 1.0).abs() < 1e-9, "{}: total power {t}", p.name);
        assert!(!p.source.is_empty(), "{}: no source", p.name);
    }
}
