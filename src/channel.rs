//! The tapped delay line (ITU-R F.1487 fig. 2): the input delayed per path,
//! multiplied by the path's tap gain, summed. Complex baseband in and out;
//! [`RealChannel`] wraps it for real (audio passband) signals.
//!
//! Delays that are not whole samples go through a windowed-sinc fractional
//! delay (33 taps, Blackman), so every path, and the output, is then late
//! by 16 samples ([`Channel::latency`]); with whole-sample delays only,
//! there is no extra latency.

use crate::fading::{Component, TapGain};
use crate::rng::Rng;
use num_complex::{Complex32, Complex64};
use std::f64::consts::PI;

/// One propagation path: its delay after the first path, its components.
#[derive(Clone, Debug, PartialEq)]
pub struct PathSpec {
    pub delay_s: f64,
    pub components: Vec<Component>,
}

impl PathSpec {
    /// A single Rayleigh-scatter path.
    pub fn scatter(delay_s: f64, power: f64, spread_hz: f64, shift_hz: f64) -> Self {
        PathSpec {
            delay_s,
            components: vec![Component::Scatter {
                power,
                spread_hz,
                shift_hz,
            }],
        }
    }

    /// A steady (unfaded) path.
    pub fn specular(delay_s: f64, power: f64, shift_hz: f64) -> Self {
        PathSpec {
            delay_s,
            components: vec![Component::Specular { power, shift_hz }],
        }
    }

    pub fn power(&self) -> f64 {
        self.components.iter().map(|c| c.power()).sum()
    }
}

/// Half length of the fractional-delay filter.
const FD_HALF: usize = 16;

struct Path {
    /// Whole samples of delay (beyond the filter's own `FD_HALF`, if any).
    whole: usize,
    /// Fractional-delay taps, or empty for a whole-sample delay.
    taps: Vec<f32>,
    gain: TapGain,
}

pub struct Channel {
    paths: Vec<Path>,
    latency: usize,
    /// Input history, newest last; long enough for the longest path.
    hist: Vec<Complex32>,
    keep: usize,
}

fn frac_taps(frac: f64) -> Vec<f32> {
    let n = 2 * FD_HALF + 1;
    let mut t: Vec<f64> = (0..n)
        .map(|k| {
            let x = k as f64 - FD_HALF as f64 - frac;
            let sinc = if x.abs() < 1e-12 {
                1.0
            } else {
                (PI * x).sin() / (PI * x)
            };
            // Blackman window over the filter's span, centred on the delay
            let w = (k as f64 - frac) / (n - 1) as f64;
            let win = 0.42 - 0.5 * (2.0 * PI * w).cos() + 0.08 * (4.0 * PI * w).cos();
            sinc * win
        })
        .collect();
    let s: f64 = t.iter().sum();
    t.iter_mut().for_each(|v| *v /= s);
    t.into_iter().map(|v| v as f32).collect()
}

impl Channel {
    /// A channel at sample rate `fs` (Hz) over the given paths; `seed` fixes
    /// the fading realisation.
    pub fn new(fs: f64, paths: &[PathSpec], seed: u64) -> Self {
        assert!(!paths.is_empty(), "a channel needs at least one path");
        let mut rng = Rng::new(seed);
        let d: Vec<f64> = paths.iter().map(|p| p.delay_s * fs).collect();
        assert!(d.iter().all(|&v| v >= 0.0), "path delays must be >= 0");
        let fractional = d.iter().any(|&v| (v - v.round()).abs() > 1e-6);
        let latency = if fractional { FD_HALF } else { 0 };
        let mut keep = 0;
        let paths: Vec<Path> = paths
            .iter()
            .zip(&d)
            .map(|(p, &dv)| {
                let (whole, taps) = if fractional {
                    let w = dv.floor();
                    (w as usize, frac_taps(dv - w))
                } else {
                    (dv.round() as usize, Vec::new())
                };
                keep = keep.max(whole + taps.len());
                Path {
                    whole,
                    taps,
                    gain: TapGain::new(&p.components, fs, &mut rng.fork()),
                }
            })
            .collect();
        Channel {
            paths,
            latency,
            hist: vec![Complex32::default(); keep],
            keep,
        }
    }

    /// Samples by which the output lags the first path's input.
    pub fn latency(&self) -> usize {
        self.latency
    }

    /// Fades a block in place. Blocks of any size give the same output.
    pub fn process(&mut self, x: &mut [Complex32]) {
        let h = self.keep;
        self.hist.extend_from_slice(x);
        for (i, out) in x.iter_mut().enumerate() {
            // newest input for this output sample sits at hist[h + i]
            let mut acc = Complex64::default();
            for p in &mut self.paths {
                let g = p.gain.sample();
                let at = h + i - p.whole;
                let v = if p.taps.is_empty() {
                    self.hist[at]
                } else {
                    // taps[k] weights the input k samples back
                    p.taps
                        .iter()
                        .enumerate()
                        .map(|(k, t)| self.hist[at - k] * t)
                        .sum()
                };
                acc += g * Complex64::new(v.re as f64, v.im as f64);
            }
            *out = Complex32::new(acc.re as f32, acc.im as f32);
        }
        let n = self.hist.len();
        self.hist.drain(..n - h);
    }
}

/// Half length of the Hilbert transformer for real signals.
const HILBERT_HALF: usize = 127;

/// A channel for real passband signals (audio): the analytic signal
/// (Hilbert FIR, 255 taps, Blackman) through a [`Channel`], the real part
/// out. A positive Doppler shift moves the audio up. Usable from about
/// fs/100 to fs/2 - fs/100 (60-5940 Hz at 12 kS/s).
pub struct RealChannel {
    ch: Channel,
    taps: Vec<f32>,
    hist: Vec<f32>,
    buf: Vec<Complex32>,
}

impl RealChannel {
    pub fn new(fs: f64, paths: &[PathSpec], seed: u64) -> Self {
        let n = 2 * HILBERT_HALF + 1;
        let taps = (0..n)
            .map(|k| {
                let m = k as i64 - HILBERT_HALF as i64;
                if m % 2 == 0 {
                    0.0
                } else {
                    let w = k as f64 / (n - 1) as f64;
                    let win = 0.42 - 0.5 * (2.0 * PI * w).cos() + 0.08 * (4.0 * PI * w).cos();
                    (2.0 / (PI * m as f64) * win) as f32
                }
            })
            .collect();
        RealChannel {
            ch: Channel::new(fs, paths, seed),
            taps,
            hist: vec![0.0; n - 1],
            buf: Vec::new(),
        }
    }

    pub fn latency(&self) -> usize {
        HILBERT_HALF + self.ch.latency()
    }

    pub fn process(&mut self, x: &mut [f32]) {
        let h = self.hist.len();
        self.hist.extend_from_slice(x);
        self.buf.clear();
        for i in 0..x.len() {
            // the analytic sample centred HILBERT_HALF back from the newest
            let w = &self.hist[i..i + h + 1];
            let im: f32 = w.iter().rev().zip(&self.taps).map(|(v, t)| v * t).sum();
            self.buf.push(Complex32::new(w[HILBERT_HALF], im));
        }
        self.ch.process(&mut self.buf);
        for (o, z) in x.iter_mut().zip(&self.buf) {
            *o = z.re;
        }
        let n = self.hist.len();
        self.hist.drain(..n - h);
    }
}
