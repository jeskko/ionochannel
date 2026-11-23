//! Layer-modes spread over delay, after Vogler & Hoffmeyer (NTIA Report
//! 92-284, 1992, Part III). A mode is given by its delays
//! tau_L < tau_c < tau_U, the Doppler shift `fs` at tau_c and `fsL` at
//! tau_L (the "slant"), the Doppler half-width sigma_D and the threshold
//! s_v at which both spreads are measured (the report gives no value). It
//! becomes taps across [tau_L, tau_U] with power T(tau) = A exp(alpha
//! (ln g + 1 - g)), g = (tau - tau_l)/(tau_c - tau_l) (eq. 8b, alpha and
//! tau_l from the appendix so T(tau_L) = T(tau_U) = s_v A), Doppler shift
//! fs + b (tau - tau_c) (eq. 8c), and a Gaussian Doppler spectrum that
//! falls to s_v at +-sigma_D (eq. 7a): an F.1487 spread of
//! sigma_D (-ln s_v / pi)^-1/2 sqrt(2 / pi). The taps run through the
//! ordinary [`crate::Channel`]. The Python `wideband` module is the same.

use crate::channel::PathSpec;
use std::f64::consts::PI;

fn z_u(zl: f64) -> f64 {
    let c = zl.ln() - zl;
    let (mut lo, mut hi) = (1.0f64, 2.0f64);
    while hi.ln() - hi > c {
        hi *= 2.0;
    }
    for _ in 0..200 {
        let mid = (lo + hi) / 2.0;
        if mid.ln() - mid > c {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    (lo + hi) / 2.0
}

/// (alpha, tau_ell) of the delay shape (the report's appendix).
pub fn vh_shape(tau_l: f64, tau_c: f64, tau_u: f64, threshold: f64) -> (f64, f64) {
    assert!(tau_l < tau_c && tau_c < tau_u, "need tau_L < tau_c < tau_U");
    assert!(
        threshold > 0.0 && threshold < 1.0,
        "threshold must be in (0, 1)"
    );
    let ratio = (tau_c - tau_l) / (tau_u - tau_c);
    assert!(ratio < 1.0, "need tau_c - tau_L < tau_U - tau_c");
    let (mut lo, mut hi) = (1e-12f64, 1.0 - 1e-6);
    for _ in 0..200 {
        let mid = (lo + hi) / 2.0;
        if (1.0 - mid) / (z_u(mid) - 1.0) < ratio {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let zl = (lo + hi) / 2.0;
    let alpha = threshold.ln() / (zl.ln() + 1.0 - zl);
    (alpha, tau_c - (tau_c - tau_l) / (1.0 - zl))
}

/// A layer-mode as taps: tau_L = `delay_s`, tau_U = tau_L + `spread_s`
/// (sigma_T), tau_c = tau_L + `peak_s` (sigma_c); shifts `shift_hz` at
/// tau_c and `shift_low_hz` at tau_L; Doppler half-width
/// `doppler_halfwidth_hz` at `threshold`; peak power density `amplitude`.
/// 33 taps; powers are the integral of T over each step.
#[allow(clippy::too_many_arguments)]
pub fn vh_mode(
    delay_s: f64,
    spread_s: f64,
    peak_s: f64,
    shift_hz: f64,
    shift_low_hz: f64,
    doppler_halfwidth_hz: f64,
    threshold: f64,
    amplitude: f64,
) -> Vec<PathSpec> {
    let (tl, tc, tu) = (delay_s, delay_s + peak_s, delay_s + spread_s);
    let (alpha, tell) = vh_shape(tl, tc, tu, threshold);
    let k = 33;
    let b = (shift_hz - shift_low_hz) / (tc - tl);
    let sigma_f = doppler_halfwidth_hz / (-threshold.ln() / PI).sqrt();
    let spread = sigma_f * (2.0 / PI).sqrt();
    (0..k)
        .map(|i| {
            let t = tl + (tu - tl) * i as f64 / (k - 1) as f64;
            let g = (t - tell) / (tc - tell);
            let p = amplitude * (alpha * (g.ln() + 1.0 - g)).exp();
            let w = spread_s / (k - 1) as f64 * if i == 0 || i == k - 1 { 0.5 } else { 1.0 };
            PathSpec::scatter(t, p * w, spread, shift_hz + b * (t - tc))
        })
        .collect()
}

/// The same paths scaled to a total power of 1.
pub fn normalize(paths: &[PathSpec]) -> Vec<PathSpec> {
    let total: f64 = paths.iter().map(PathSpec::power).sum();
    paths
        .iter()
        .map(|p| {
            let mut q = p.clone();
            for c in &mut q.components {
                match c {
                    crate::Component::Scatter { power, .. }
                    | crate::Component::Specular { power, .. } => *power /= total,
                }
            }
            q
        })
        .collect()
}
