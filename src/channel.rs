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
use crate::vary::{Run, Vary, bounds};
use num_complex::{Complex32, Complex64};
use std::f64::consts::{PI, TAU};

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
pub(crate) const FD_HALF: usize = 16;

struct Path {
    /// Whole samples of delay (beyond the filter's own `FD_HALF`, if any).
    whole: usize,
    /// Fractional-delay taps, or empty for a whole-sample delay.
    taps: Vec<f32>,
    gain: TapGain,
    moving: Option<Box<Moving>>,
}

/// The parts of a path that change over time.
struct Moving {
    /// Delay run, the path's own delay and the largest delay (samples).
    delay: Option<(Run, f64, f64)>,
    shift: Option<Run>,
    power: Option<Run>,
    /// Spread run (octaves) and its largest value.
    spread: Option<(Run, f64)>,
    /// Integrated extra Doppler, in cycles.
    cycles: f64,
}

/// Phases of the fractional-delay table for moving delays.
pub(crate) const PHASES: usize = 512;

pub struct Channel {
    paths: Vec<Path>,
    latency: usize,
    /// Input history, newest last; long enough for the longest path.
    hist: Vec<Complex32>,
    keep: usize,
    fs: f64,
    /// Fractional-delay taps at PHASES + 1 phases, if any delay moves.
    table: Vec<Vec<f32>>,
}

pub(crate) fn frac_taps(frac: f64) -> Vec<f32> {
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
        Self::with_variation(fs, paths, &[], seed)
    }

    /// A channel whose paths change over time: `vary` has one entry per
    /// path (or is empty). The changes draw on their own random numbers,
    /// so the fading for a seed is the same as from [`Channel::new`]. A
    /// moving delay makes the channel use the fractional-delay filter
    /// (latency 16 samples).
    pub fn with_variation(fs: f64, paths: &[PathSpec], vary: &[Vary], seed: u64) -> Self {
        assert!(!paths.is_empty(), "a channel needs at least one path");
        assert!(
            vary.is_empty() || vary.len() == paths.len(),
            "vary needs one entry per path"
        );
        let none = Vary::default();
        let vary_of = |i: usize| vary.get(i).unwrap_or(&none);
        let mut rng = Rng::new(seed);
        let mut vrng = Rng::new(seed ^ 0x7661_7279_7661_7279);
        let d: Vec<f64> = paths.iter().map(|p| p.delay_s * fs).collect();
        assert!(d.iter().all(|&v| v >= 0.0), "path delays must be >= 0");
        let moving_delay = (0..paths.len()).any(|i| !vary_of(i).delay_s.is_empty());
        let fractional = moving_delay || d.iter().any(|&v| (v - v.round()).abs() > 1e-6);
        let latency = if fractional { FD_HALF } else { 0 };
        let mut keep = 0;
        let paths: Vec<Path> = paths
            .iter()
            .zip(&d)
            .enumerate()
            .map(|(i, (p, &dv))| {
                let v = vary_of(i);
                let (whole, taps) = if fractional {
                    let w = dv.floor();
                    (w as usize, frac_taps(dv - w))
                } else {
                    (dv.round() as usize, Vec::new())
                };
                keep = keep.max(whole + taps.len());
                let gain = if v.spread_oct.is_empty() {
                    TapGain::new(&p.components, fs, &mut rng.fork())
                } else {
                    let vmax = bounds(&v.spread_oct).1;
                    TapGain::scaled(&p.components, fs, &mut rng.fork(), 2f64.powf(vmax))
                };
                let moving = (!v.is_empty()).then(|| {
                    let mut r = vrng.fork();
                    let delay = (!v.delay_s.is_empty()).then(|| {
                        let max = ((p.delay_s + bounds(&v.delay_s).1) * fs).max(0.0);
                        keep = keep.max(max.ceil() as usize + 2 * FD_HALF + 2);
                        (Run::new(&v.delay_s, fs, r.fork()), dv, max)
                    });
                    Box::new(Moving {
                        delay,
                        shift: (!v.shift_hz.is_empty())
                            .then(|| Run::new(&v.shift_hz, fs, r.fork())),
                        power: (!v.power_db.is_empty())
                            .then(|| Run::new(&v.power_db, fs, r.fork())),
                        spread: (!v.spread_oct.is_empty()).then(|| {
                            (
                                Run::new(&v.spread_oct, fs, r.fork()),
                                bounds(&v.spread_oct).1,
                            )
                        }),
                        cycles: 0.0,
                    })
                });
                Path {
                    whole,
                    taps,
                    gain,
                    moving,
                }
            })
            .collect();
        let table = if moving_delay {
            (0..=PHASES)
                .map(|k| frac_taps(k as f64 / PHASES as f64))
                .collect()
        } else {
            Vec::new()
        };
        Channel {
            paths,
            latency,
            hist: vec![Complex32::default(); keep],
            keep,
            fs,
            table,
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
                let Some(m) = p.moving.as_mut() else {
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
                    continue;
                };
                let mut g = match m.spread.as_mut() {
                    Some((run, vmax)) => p.gain.sample_warped(2f64.powf(run.next() - *vmax)),
                    None => p.gain.sample(),
                };
                if let Some(run) = m.power.as_mut() {
                    g *= 10f64.powf(run.next() / 20.0);
                }
                if let Some(run) = m.shift.as_mut() {
                    g *= Complex64::from_polar(1.0, TAU * m.cycles);
                    m.cycles = (m.cycles + run.next() / self.fs).rem_euclid(1.0);
                }
                let v: Complex32 = match m.delay.as_mut() {
                    Some((run, base, max)) => {
                        let dd = (*base + run.next() * self.fs).clamp(0.0, *max);
                        let w = dd.floor();
                        let ph = (dd - w) * PHASES as f64;
                        let k0 = (ph.floor() as usize).min(PHASES - 1);
                        let a = (ph - k0 as f64) as f32;
                        let (t0, t1) = (&self.table[k0], &self.table[k0 + 1]);
                        let at = h + i - w as usize;
                        (0..t0.len())
                            .map(|k| self.hist[at - k] * ((1.0 - a) * t0[k] + a * t1[k]))
                            .sum()
                    }
                    None => {
                        let at = h + i - p.whole;
                        if p.taps.is_empty() {
                            self.hist[at]
                        } else {
                            p.taps
                                .iter()
                                .enumerate()
                                .map(|(k, t)| self.hist[at - k] * t)
                                .sum()
                        }
                    }
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

/// The analytic signal of a real stream: a Hilbert FIR (255 taps,
/// Blackman), the real part delayed to match. Usable from about fs/100 to
/// fs/2 - fs/100.
pub(crate) struct Analytic {
    taps: Vec<f32>,
    hist: Vec<f32>,
}

impl Analytic {
    /// Samples by which the output lags the input.
    pub(crate) const LATENCY: usize = HILBERT_HALF;

    pub(crate) fn new() -> Self {
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
        Analytic {
            taps,
            hist: vec![0.0; n - 1],
        }
    }

    /// Replaces `out` with the analytic signal of `x`.
    pub(crate) fn process(&mut self, x: &[f32], out: &mut Vec<Complex32>) {
        let h = self.hist.len();
        self.hist.extend_from_slice(x);
        out.clear();
        for i in 0..x.len() {
            // the analytic sample centred HILBERT_HALF back from the newest
            let w = &self.hist[i..i + h + 1];
            let im: f32 = w.iter().rev().zip(&self.taps).map(|(v, t)| v * t).sum();
            out.push(Complex32::new(w[HILBERT_HALF], im));
        }
        let n = self.hist.len();
        self.hist.drain(..n - h);
    }
}

/// A channel for real passband signals (audio): the analytic signal
/// (Hilbert FIR, 255 taps, Blackman) through a [`Channel`], the real part
/// out. A positive Doppler shift moves the audio up. Usable from about
/// fs/100 to fs/2 - fs/100 (60-5940 Hz at 12 kS/s).
pub struct RealChannel {
    ch: Channel,
    analytic: Analytic,
    buf: Vec<Complex32>,
}

impl RealChannel {
    pub fn new(fs: f64, paths: &[PathSpec], seed: u64) -> Self {
        Self::with_variation(fs, paths, &[], seed)
    }

    /// As [`Channel::with_variation`], for real audio.
    pub fn with_variation(fs: f64, paths: &[PathSpec], vary: &[Vary], seed: u64) -> Self {
        RealChannel {
            ch: Channel::with_variation(fs, paths, vary, seed),
            analytic: Analytic::new(),
            buf: Vec::new(),
        }
    }

    pub fn latency(&self) -> usize {
        Analytic::LATENCY + self.ch.latency()
    }

    pub fn process(&mut self, x: &mut [f32]) {
        self.analytic.process(x, &mut self.buf);
        self.ch.process(&mut self.buf);
        for (o, z) in x.iter_mut().zip(&self.buf) {
            *o = z.re;
        }
    }
}
