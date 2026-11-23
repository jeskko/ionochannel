//! Vogler & Hoffmeyer layer-modes: the delay shape meets its conditions,
//! the slant and the Doppler width, and a mode runs as a channel.

use ionochannel::wideband::{normalize, vh_mode, vh_shape};
use ionochannel::{Channel, Component, Rng};
use num_complex::Complex32;

fn t(alpha: f64, tell: f64, tc: f64, tau: f64) -> f64 {
    let g = (tau - tell) / (tc - tell);
    (alpha * (g.ln() + 1.0 - g)).exp()
}

#[test]
fn delay_shape_meets_its_conditions() {
    for (tl, tc, tu, sv) in [
        (1e-3, 1.3e-3, 2.6e-3, 0.1),
        (0.0, 22e-6 * 0.48, 22e-6, 0.01),
        (2e-3, 2.1e-3, 3.4e-3, 0.3),
    ] {
        let (a, tell) = vh_shape(tl, tc, tu, sv);
        assert!((t(a, tell, tc, tl) - sv).abs() < 1e-9, "{tl} {tc} {tu}");
        assert!((t(a, tell, tc, tu) - sv).abs() < 1e-9);
        assert!((t(a, tell, tc, tc) - 1.0).abs() < 1e-12);
        assert!(t(a, tell, tc, tc - 1e-6) < 1.0 && t(a, tell, tc, tc + 1e-6) < 1.0);
    }
}

#[test]
fn mode_taps_slant_and_width() {
    let p = vh_mode(1e-3, 1600e-6, 300e-6, -0.65, -0.8, 4.0, 0.1, 1.0);
    let shift = |q: &ionochannel::PathSpec| match q.components[0] {
        Component::Scatter {
            shift_hz,
            spread_hz,
            ..
        } => (shift_hz, spread_hz),
        _ => unreachable!(),
    };
    assert!((shift(&p[0]).0 + 0.8).abs() < 1e-12);
    let peak = p
        .iter()
        .max_by(|a, b| a.power().total_cmp(&b.power()))
        .unwrap();
    assert!((peak.delay_s - 1.3e-3).abs() < 1600e-6 / 32.0);
    // the Gaussian spectrum falls to s_v at +-sigma_D: exp(-f^2 / 2 s^2), s = spread / 2
    let s = shift(&p[0]).1 / 2.0;
    assert!(((-(4.0f64 * 4.0) / (2.0 * s * s)).exp() - 0.1).abs() < 1e-9);
    // powers integrate T: total close to the integral (trapezoid on 33 points)
    let total: f64 = p.iter().map(|q| q.power()).sum();
    let (a, tell) = vh_shape(1e-3, 1.3e-3, 2.6e-3, 0.1);
    let n = 100000;
    let fine: f64 = (0..n)
        .map(|i| t(a, tell, 1.3e-3, 1e-3 + 1.6e-3 * (i as f64 + 0.5) / n as f64))
        .sum::<f64>()
        * 1.6e-3
        / n as f64;
    assert!((total / fine - 1.0).abs() < 0.01, "{total} {fine}");
}

#[test]
fn mode_runs_as_a_channel_with_unit_power() {
    let p = normalize(&vh_mode(0.5e-3, 1400e-6, 300e-6, -1.6, -2.0, 9.0, 0.1, 1.0));
    let mut r = Rng::new(1);
    let fs = 12000.0;
    let mut pw = 0.0;
    for seed in 0..6 {
        let mut x: Vec<Complex32> = (0..(fs * 10.0) as usize)
            .map(|_| r.complex_normal())
            .collect();
        Channel::new(fs, &p, seed).process(&mut x);
        pw += x[1000..].iter().map(|v| v.norm_sqr() as f64).sum::<f64>() / (x.len() - 1000) as f64;
    }
    assert!((pw / 6.0 - 1.0).abs() < 0.05, "{}", pw / 6.0);
}
