//! What the receiver does to the signal after the channel and the noise:
//! mistuning, the passband filter and AGC, in that order.
//!
//! - **Mistuning** shifts everything by `mistune_hz` (the receiver tuned
//!   that much low: a positive value moves the audio up).
//! - **Passband**: a linear-phase FIR between two -6 dB edges (Kaiser
//!   window, 60 dB stopband, `transition_hz` wide between pass and stop
//!   edges). The taps are complex, so on IQ the passband can be one-sided
//!   (300 to 2700 Hz is USB, -2700 to -300 Hz is LSB).
//! - **AGC**: a peak AGC that works in 1 ms blocks. Each block's peak
//!   envelope updates a level `L`: up towards the peak with time constant
//!   `attack_s`; otherwise held for `hang_s` after the last block whose
//!   peak came within 1 dB of `L`, then down by exp(-t / `decay_s`), never
//!   below the peak or the floor `target / max_gain`. The gain is `target / L`, ramped linearly across
//!   each block. A block's own peak counts towards its gain (1 ms of
//!   look-ahead); a step up in level still overshoots for about the attack
//!   time, as in a receiver. The AGC starts settled on the first block.
//!
//! [`Receiver`] works on complex baseband, [`RealReceiver`] on real audio
//! (through the analytic signal; the AGC follows its envelope). The Python
//! `Receiver` does the same on whole arrays with no latency.

use crate::channel::Analytic;
use num_complex::{Complex32, Complex64};
use std::collections::VecDeque;
use std::f64::consts::{PI, TAU};

/// AGC settings. The defaults: attack 2 ms, no hang, decay 0.5 s, output
/// peak 0.25, at most 60 dB of gain.
#[derive(Clone, Debug, PartialEq)]
pub struct Agc {
    pub attack_s: f64,
    pub hang_s: f64,
    pub decay_s: f64,
    /// Output peak envelope the AGC holds.
    pub target: f64,
    /// Largest gain (dB); below `target` minus this, the gain is fixed.
    pub max_gain_db: f64,
}

impl Default for Agc {
    fn default() -> Self {
        Agc {
            attack_s: 0.002,
            hang_s: 0.0,
            decay_s: 0.5,
            target: 0.25,
            max_gain_db: 60.0,
        }
    }
}

/// What the receiver does; the default does nothing.
#[derive(Clone, Debug, PartialEq)]
pub struct ReceiverSpec {
    pub mistune_hz: f64,
    /// -6 dB edges (Hz), or `None` for no filter.
    pub passband: Option<(f64, f64)>,
    pub transition_hz: f64,
    pub agc: Option<Agc>,
}

impl Default for ReceiverSpec {
    fn default() -> Self {
        ReceiverSpec {
            mistune_hz: 0.0,
            passband: None,
            transition_hz: 100.0,
            agc: None,
        }
    }
}

impl ReceiverSpec {
    /// An SSB receiver: 300-2700 Hz, default AGC.
    pub fn ssb() -> Self {
        ReceiverSpec {
            passband: Some((300.0, 2700.0)),
            agc: Some(Agc::default()),
            ..Default::default()
        }
    }
}

/// Stopband attenuation of the passband filter (dB).
pub const STOPBAND_DB: f64 = 60.0;

/// A peak within this of the level (1 dB) re-arms the hang.
pub const HANG_REARM: f64 = 0.891_250_938_133_745_5;

/// AGC block length: 1 ms, at least one sample.
pub fn agc_block(fs: f64) -> usize {
    ((fs * 1e-3).round() as usize).max(1)
}

fn bessel_i0(x: f64) -> f64 {
    let (mut sum, mut term, mut k) = (1.0, 1.0, 1.0);
    while term > 1e-12 * sum {
        term *= (x / (2.0 * k)).powi(2);
        sum += term;
        k += 1.0;
    }
    sum
}

/// The passband filter's taps: a Kaiser-windowed lowpass of half width
/// (hi - lo) / 2, moved up to the band's centre. Odd length, gain 1 at the
/// centre, -6 dB at `lo` and `hi`.
pub fn bandpass_taps(fs: f64, lo: f64, hi: f64, transition_hz: f64) -> Vec<Complex32> {
    assert!(hi > lo, "passband: hi must be above lo");
    assert!(transition_hz > 0.0, "passband: transition must be > 0");
    // Kaiser's formulas for the length and beta
    let a = STOPBAND_DB;
    let beta = 0.1102 * (a - 8.7);
    let n = ((a - 7.95) / (14.36 * transition_hz / fs)).ceil() as usize + 1;
    let n = n | 1;
    let half = (n - 1) / 2;
    let fc = (hi - lo) / 2.0 / fs;
    let f0 = (hi + lo) / 2.0 / fs;
    let i0b = bessel_i0(beta);
    let lp: Vec<f64> = (0..n)
        .map(|k| {
            let m = k as f64 - half as f64;
            let sinc = if m == 0.0 {
                2.0 * fc
            } else {
                (TAU * fc * m).sin() / (PI * m)
            };
            let r = m / half as f64;
            sinc * bessel_i0(beta * (1.0 - r * r).max(0.0).sqrt()) / i0b
        })
        .collect();
    let s: f64 = lp.iter().sum();
    lp.iter()
        .enumerate()
        .map(|(k, v)| {
            let m = k as f64 - half as f64;
            let z = Complex64::from_polar(v / s, TAU * f0 * m);
            Complex32::new(z.re as f32, z.im as f32)
        })
        .collect()
}

struct Fir {
    taps: Vec<Complex32>,
    hist: Vec<Complex32>,
}

impl Fir {
    fn process(&mut self, x: &mut [Complex32]) {
        let h = self.hist.len();
        self.hist.extend_from_slice(x);
        for (i, out) in x.iter_mut().enumerate() {
            let w = &self.hist[i..i + h + 1];
            *out = w.iter().rev().zip(&self.taps).map(|(v, t)| v * t).sum();
        }
        let n = self.hist.len();
        self.hist.drain(..n - h);
    }
}

struct AgcState {
    block: usize,
    attack: f64,
    decay: f64,
    hang_blocks: usize,
    target: f64,
    floor: f64,
    level: f64,
    hang: usize,
    gain: f64,
    started: bool,
    /// Samples still to pass (as zeros) before the input's first sample
    /// arrives, so blocks line up with the input's start.
    skip: usize,
    pend: Vec<Complex32>,
    out: VecDeque<Complex32>,
}

impl AgcState {
    fn new(a: &Agc, fs: f64, skip: usize) -> Self {
        let block = agc_block(fs);
        let dt = block as f64 / fs;
        let attack = if a.attack_s > 0.0 {
            1.0 - (-dt / a.attack_s).exp()
        } else {
            1.0
        };
        let decay = if a.decay_s > 0.0 {
            (-dt / a.decay_s).exp()
        } else {
            0.0
        };
        AgcState {
            block,
            attack,
            decay,
            hang_blocks: (a.hang_s / dt).round() as usize,
            target: a.target,
            floor: a.target / 10f64.powf(a.max_gain_db / 20.0),
            level: 0.0,
            hang: 0,
            gain: 0.0,
            started: false,
            skip,
            pend: Vec::with_capacity(block),
            out: std::iter::repeat_n(Complex32::default(), block).collect(),
        }
    }

    fn run_block(&mut self) {
        let p = self
            .pend
            .iter()
            .map(|z| z.norm() as f64)
            .fold(0.0, f64::max);
        if !self.started {
            self.level = p.max(self.floor);
            self.gain = self.target / self.level;
            self.started = true;
        } else if p > self.level {
            self.level += (p - self.level) * self.attack;
            self.hang = self.hang_blocks;
        } else if self.hang_blocks > 0 && p >= self.level * HANG_REARM {
            self.hang = self.hang_blocks;
        } else if self.hang > 0 {
            self.hang -= 1;
        } else {
            self.level = (self.level * self.decay).max(p).max(self.floor);
        }
        let g = self.target / self.level.max(self.floor);
        let b = self.block as f64;
        for (i, z) in self.pend.drain(..).enumerate() {
            let gi = self.gain + (g - self.gain) * (i + 1) as f64 / b;
            self.out.push_back(z * gi as f32);
        }
        self.gain = g;
    }

    fn process(&mut self, x: &mut [Complex32]) {
        for v in x {
            if self.skip > 0 {
                self.skip -= 1;
                self.out.push_back(Complex32::default());
                *v = self.out.pop_front().unwrap_or_default();
                continue;
            }
            self.pend.push(*v);
            if self.pend.len() == self.block {
                self.run_block();
            }
            *v = self.out.pop_front().unwrap_or_default();
        }
    }
}

/// A receiver for complex baseband; streams like [`crate::Channel`].
pub struct Receiver {
    step: f64,
    /// Index of the next sample in the input's time (negative while the
    /// samples of a [`RealReceiver`]'s Hilbert filter come out).
    n: i64,
    fir: Option<Fir>,
    agc: Option<AgcState>,
}

impl Receiver {
    pub fn new(fs: f64, spec: &ReceiverSpec) -> Self {
        Self::with_lead(fs, spec, 0)
    }

    /// `lead`: samples the input already lags the stream it came from.
    /// Mistuning phase and AGC blocks count from that stream's first
    /// sample, as the Python version does, so the two agree.
    fn with_lead(fs: f64, spec: &ReceiverSpec, lead: usize) -> Self {
        let fir = spec.passband.map(|(lo, hi)| {
            let taps = bandpass_taps(fs, lo, hi, spec.transition_hz);
            Fir {
                hist: vec![Complex32::default(); taps.len() - 1],
                taps,
            }
        });
        let skip = lead + fir.as_ref().map_or(0, |f| f.hist.len() / 2);
        Receiver {
            step: spec.mistune_hz / fs,
            n: -(lead as i64),
            fir,
            agc: spec.agc.as_ref().map(|a| AgcState::new(a, fs, skip)),
        }
    }

    /// Samples by which the output lags the input: half the filter, plus
    /// one AGC block.
    pub fn latency(&self) -> usize {
        self.fir.as_ref().map_or(0, |f| f.hist.len() / 2) + self.agc.as_ref().map_or(0, |a| a.block)
    }

    pub fn process(&mut self, x: &mut [Complex32]) {
        if self.step != 0.0 {
            for v in x.iter_mut() {
                let ph = TAU * (self.step * self.n as f64).rem_euclid(1.0);
                let (s, c) = ph.sin_cos();
                *v *= Complex32::new(c as f32, s as f32);
                self.n += 1;
            }
        }
        if let Some(f) = self.fir.as_mut() {
            f.process(x);
        }
        if let Some(a) = self.agc.as_mut() {
            a.process(x);
        }
    }
}

/// A receiver for real audio: the analytic signal through a [`Receiver`],
/// the real part out. Usable from about fs/100 to fs/2 - fs/100.
pub struct RealReceiver {
    rx: Receiver,
    analytic: Analytic,
    buf: Vec<Complex32>,
}

impl RealReceiver {
    pub fn new(fs: f64, spec: &ReceiverSpec) -> Self {
        RealReceiver {
            rx: Receiver::with_lead(fs, spec, Analytic::LATENCY),
            analytic: Analytic::new(),
            buf: Vec::new(),
        }
    }

    pub fn latency(&self) -> usize {
        Analytic::LATENCY + self.rx.latency()
    }

    pub fn process(&mut self, x: &mut [f32]) {
        self.analytic.process(x, &mut self.buf);
        self.rx.process(&mut self.buf);
        for (o, z) in x.iter_mut().zip(&self.buf) {
            *o = z.re;
        }
    }
}
