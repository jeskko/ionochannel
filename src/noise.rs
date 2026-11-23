//! Additive white Gaussian noise at a signal-to-noise ratio measured in a
//! reference bandwidth (2500 Hz is usual for HF: WSJT-X, codec2's SNR3k
//! uses 3000).

use crate::rng::Rng;
use num_complex::Complex32;

/// Mean power of a signal.
pub fn power_c(x: &[Complex32]) -> f64 {
    x.iter().map(|v| v.norm_sqr() as f64).sum::<f64>() / x.len().max(1) as f64
}

pub fn power_r(x: &[f32]) -> f64 {
    x.iter().map(|v| (v * v) as f64).sum::<f64>() / x.len().max(1) as f64
}

/// Noise variance per sample giving `snr_db` for a signal of power
/// `signal_power`, the noise counted in `bw_hz`: complex noise spreads
/// over fs, real noise over fs/2.
pub fn noise_power(signal_power: f64, snr_db: f64, bw_hz: f64, fs: f64, complex: bool) -> f64 {
    let in_bw = signal_power / 10f64.powf(snr_db / 10.0);
    let span = if complex { fs } else { fs / 2.0 };
    in_bw * span / bw_hz
}

pub fn add_c(x: &mut [Complex32], variance: f64, rng: &mut Rng) {
    let s = variance.sqrt() as f32;
    for v in x {
        *v += rng.complex_normal() * s;
    }
}

pub fn add_r(x: &mut [f32], variance: f64, rng: &mut Rng) {
    let s = variance.sqrt();
    for v in x {
        *v += (rng.normal() * s) as f32;
    }
}
