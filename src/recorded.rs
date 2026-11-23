//! Recorded noise mixed in at an SNR, as for white noise: the noise's
//! level is its mean spectral density over a measurement band times the
//! reference bandwidth. The density comes from the whole recording,
//! through the passband filter of [`crate::receiver`] (divided by its
//! equivalent noise bandwidth). The recording loops, with a 10 ms
//! equal-power crossfade at the seam, from a random start. The Python
//! `recorded` module does the same on arrays, and can also level each
//! segment on its own.

use crate::receiver::bandpass_taps;
use crate::rng::Rng;
use num_complex::Complex32;
use std::f64::consts::FRAC_PI_2;

/// The default measurement band: 300-2700 Hz for real audio, the whole
/// band less 5 % at each edge for I/Q.
pub fn default_band(fs: f64, real: bool) -> (f64, f64) {
    if real {
        (300.0, 2700.0f64.min(0.45 * fs))
    } else {
        (-0.45 * fs, 0.45 * fs)
    }
}

pub struct Recorded {
    src: Vec<Complex32>,
    pos: usize,
    density: f64,
    real: bool,
}

impl Recorded {
    /// From real audio.
    pub fn new_r(samples: &[f32], fs: f64, band: Option<(f64, f64)>, seed: u64) -> Self {
        let c: Vec<Complex32> = samples.iter().map(|&v| Complex32::new(v, 0.0)).collect();
        Self::build(c, fs, band.unwrap_or(default_band(fs, true)), true, seed)
    }

    /// From I/Q.
    pub fn new_c(samples: &[Complex32], fs: f64, band: Option<(f64, f64)>, seed: u64) -> Self {
        Self::build(
            samples.to_vec(),
            fs,
            band.unwrap_or(default_band(fs, false)),
            false,
            seed,
        )
    }

    fn build(rec: Vec<Complex32>, fs: f64, (lo, hi): (f64, f64), real: bool, seed: u64) -> Self {
        assert!(hi > lo, "band: hi must be above lo");
        let t = bandpass_taps(fs, lo, hi, ((hi - lo) / 4.0).min(100.0));
        assert!(
            rec.len() > 4 * t.len(),
            "recording too short for its band filter"
        );
        let enb = t.iter().map(|v| v.norm_sqr() as f64).sum::<f64>() * fs;
        let mut p = 0.0;
        for i in t.len()..rec.len() {
            let y: Complex32 = t
                .iter()
                .zip(rec[i - t.len()..i].iter().rev())
                .map(|(a, b)| a * b)
                .sum();
            p += y.norm_sqr() as f64;
        }
        p /= (rec.len() - t.len()) as f64;
        let density = if real { 2.0 * p / enb } else { p / enb };
        // seamless loop: the last L samples faded into the first
        let m = rec.len();
        let l = ((0.01 * fs) as usize).min(m / 4);
        let mut src = rec[..m - l].to_vec();
        for i in 0..l {
            let a = (i as f64 + 0.5) / l as f64 * FRAC_PI_2;
            src[i] = rec[i] * a.sin() as f32 + rec[m - l + i] * a.cos() as f32;
        }
        let pos = (Rng::new(seed).next_u64() % src.len() as u64) as usize;
        Recorded {
            src,
            pos,
            density,
            real,
        }
    }

    /// Mean spectral density over the band (power per Hz; one-sided for
    /// real audio).
    pub fn density(&self) -> f64 {
        self.density
    }

    fn gain(&self, in_bw_power: f64, bw_hz: f64) -> f32 {
        (in_bw_power / (self.density.max(1e-300) * bw_hz)).sqrt() as f32
    }

    /// Adds the noise at `in_bw_power` in `bw_hz` (the signal's power over
    /// 10^(SNR/10)).
    pub fn add_c(&mut self, x: &mut [Complex32], in_bw_power: f64, bw_hz: f64) {
        assert!(!self.real, "a real recording on an I/Q signal");
        let g = self.gain(in_bw_power, bw_hz);
        for v in x {
            *v += self.src[self.pos] * g;
            self.pos = (self.pos + 1) % self.src.len();
        }
    }

    pub fn add_r(&mut self, x: &mut [f32], in_bw_power: f64, bw_hz: f64) {
        assert!(self.real, "an I/Q recording on a real signal");
        let g = self.gain(in_bw_power, bw_hz);
        for v in x {
            *v += self.src[self.pos].re * g;
            self.pos = (self.pos + 1) % self.src.len();
        }
    }
}
