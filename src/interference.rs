//! Man-made interference with a mains signature. ITU-R P.372 §6 names
//! the kinds (its "single carrier noise", impulsive noise) but gives no
//! parameters, so the defaults are illustrative, not measured. The Python
//! `interference` module implements the same models.
//!
//! - [`Smps`]: a switching power supply's comb. Lines at
//!   `offset + k * spacing` that wander together (an Ornstein-Uhlenbeck
//!   drift, `drift_hz` rms, correlation time `drift_s`) and carry AM and
//!   FM at twice the mains frequency (rectifier ripple).
//! - [`Arcing`]: powerline arcing buzz. Short noise bursts at a fixed
//!   phase of each mains half-cycle (one phase per source), with timing
//!   jitter and log-normal burst energies, gated on and off.

use crate::rng::Rng;
use num_complex::Complex32;
use std::f64::consts::{LN_10, TAU};

#[derive(Clone, Debug, PartialEq)]
pub struct SmpsSpec {
    pub spacing_hz: f64,
    pub mains_hz: f64,
    pub am_depth: f64,
    pub fm_dev_hz: f64,
    pub drift_hz: f64,
    pub drift_s: f64,
    /// Spread of the lines' levels (dB, Gaussian).
    pub line_spread_db: f64,
}

impl Default for SmpsSpec {
    fn default() -> Self {
        SmpsSpec {
            spacing_hz: 60e3,
            mains_hz: 50.0,
            am_depth: 0.3,
            fm_dev_hz: 15.0,
            drift_hz: 20.0,
            drift_s: 10.0,
            line_spread_db: 0.0,
        }
    }
}

/// Nominal line frequencies inside the band: (-fs/2, fs/2), or (0, fs/2)
/// for real signals.
pub fn smps_lines(fs: f64, offset_hz: f64, spacing_hz: f64, real: bool) -> Vec<f64> {
    let (lo, hi) = if real {
        (0.0, fs / 2.0)
    } else {
        (-fs / 2.0, fs / 2.0)
    };
    let k0 = ((lo - offset_hz) / spacing_hz).ceil() as i64;
    let k1 = ((hi - offset_hz) / spacing_hz).floor() as i64;
    (k0..=k1)
        .map(|k| offset_hz + spacing_hz * k as f64)
        .filter(|&f| f > lo && f < hi)
        .collect()
}

/// A switching-supply comb; each line of mean power 1.
pub struct Smps {
    fs: f64,
    spec: SmpsSpec,
    rng: Rng,
    /// (frequency, amplitude, phase in cycles)
    lines: Vec<(f64, f64, f64)>,
    n: u64,
    drift: f64,
    rho: f64,
    /// Integrated drift, in cycles.
    cycles: f64,
    th: f64,
    th_am: f64,
    am_norm: f64,
    real: bool,
}

impl Smps {
    pub fn new(fs: f64, offset_hz: f64, spec: &SmpsSpec, real: bool, seed: u64) -> Self {
        let mut rng = Rng::new(seed);
        let lines = smps_lines(fs, offset_hz, spec.spacing_hz, real)
            .into_iter()
            .map(|f| {
                let a = 10f64.powf(spec.line_spread_db * rng.normal() / 20.0);
                (f, a, rng.uniform())
            })
            .collect();
        let rho = if spec.drift_s > 0.0 {
            (-1.0 / (fs * spec.drift_s)).exp()
        } else {
            0.0
        };
        Smps {
            fs,
            lines,
            n: 0,
            drift: rng.normal() * spec.drift_hz,
            rho,
            cycles: 0.0,
            th: rng.uniform() * TAU,
            th_am: rng.uniform() * TAU,
            am_norm: (1.0 + spec.am_depth * spec.am_depth / 2.0).sqrt(),
            real,
            spec: spec.clone(),
            rng,
        }
    }

    fn next(&mut self) -> Complex32 {
        let t = self.n as f64 / self.fs;
        let s = &self.spec;
        let ripple = TAU * 2.0 * s.mains_hz;
        let mut ph = TAU * self.cycles;
        if s.fm_dev_hz != 0.0 && s.mains_hz > 0.0 {
            ph -= TAU * s.fm_dev_hz / ripple * (ripple * t + self.th).cos();
        }
        let am = (1.0 + s.am_depth * (ripple * t + self.th + self.th_am).sin()) / self.am_norm;
        let mut z = num_complex::Complex64::default();
        for &(f, a, p0) in &self.lines {
            let cyc = (f * self.n as f64 / self.fs + p0).fract();
            z += num_complex::Complex64::from_polar(a, TAU * cyc + ph);
        }
        z *= am;
        // drift: integrate this sample's frequency, then step the OU process
        self.cycles = (self.cycles + self.drift / self.fs).fract();
        self.drift = self.drift * self.rho
            + s.drift_hz * (1.0 - self.rho * self.rho).sqrt() * self.rng.normal();
        self.n += 1;
        Complex32::new(z.re as f32, z.im as f32)
    }

    /// Adds the comb, each line at power `power`.
    pub fn add_c(&mut self, x: &mut [Complex32], power: f64) {
        let a = power.sqrt() as f32;
        for v in x {
            *v += self.next() * a;
        }
    }

    /// Adds the comb to a real signal (built with `real = true`).
    pub fn add_r(&mut self, x: &mut [f32], power: f64) {
        assert!(self.real, "Smps for real signals needs real = true");
        let a = (2.0 * power).sqrt() as f32;
        for v in x {
            *v += self.next().re * a;
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ArcingSpec {
    pub mains_hz: f64,
    pub sources: usize,
    pub jitter_s: f64,
    /// Amplitude decay time constant of a burst.
    pub burst_s: f64,
    /// Spread of burst energies (dB, Gaussian).
    pub spread_db: f64,
    pub on_s: f64,
    /// Mean off time; 0 for always on.
    pub off_s: f64,
}

impl Default for ArcingSpec {
    fn default() -> Self {
        ArcingSpec {
            mains_hz: 50.0,
            sources: 1,
            jitter_s: 2e-4,
            burst_s: 1e-4,
            spread_db: 4.0,
            on_s: 5.0,
            off_s: 0.0,
        }
    }
}

/// Powerline arcing buzz; mean power 1.
pub struct Arcing {
    fs: f64,
    rng: Rng,
    spec: ArcingSpec,
    half: f64,
    /// Next half-cycle index for each source, and its phase (samples).
    src: Vec<(u64, f64)>,
    /// Pending bursts (sample index, energy), unordered, few.
    pending: Vec<(i64, f64)>,
    d: f64,
    level: f64,
    mean: f64,
    on: bool,
    left: i64,
    n: i64,
}

impl Arcing {
    pub fn new(fs: f64, spec: &ArcingSpec, seed: u64) -> Self {
        let mut rng = Rng::new(seed);
        let half = fs / (2.0 * spec.mains_hz);
        let src = (0..spec.sources)
            .map(|_| (0, rng.uniform() * half))
            .collect();
        let lam = 2.0 / (fs * spec.burst_s);
        let duty = if spec.off_s > 0.0 {
            spec.on_s / (spec.on_s + spec.off_s)
        } else {
            1.0
        };
        let s = spec.spread_db * LN_10 / 10.0;
        let on = spec.off_s <= 0.0 || rng.uniform() < duty;
        let mut a = Arcing {
            fs,
            half,
            src,
            pending: Vec::new(),
            d: (-lam).exp(),
            level: 0.0,
            mean: spec.sources as f64 / half * duty * (s * s / 2.0).exp(),
            on,
            left: 0,
            n: 0,
            spec: spec.clone(),
            rng,
        };
        a.left = a.hold();
        // start in the steady state: run off the bursts' decay
        for _ in 0..(25.0 / lam).ceil() as usize {
            a.envelope();
        }
        a
    }

    fn hold(&mut self) -> i64 {
        if self.spec.off_s <= 0.0 {
            return i64::MAX;
        }
        let mean = if self.on {
            self.spec.on_s
        } else {
            self.spec.off_s
        };
        (-(1.0 - self.rng.uniform()).ln() * mean * self.fs) as i64 + 1
    }

    fn envelope(&mut self) -> f64 {
        // schedule each source's bursts up to a half-cycle ahead
        let horizon = self.n as f64 + self.half;
        for i in 0..self.src.len() {
            loop {
                let (k, ph) = self.src[i];
                let t = ph + k as f64 * self.half;
                if t > horizon {
                    break;
                }
                let at = (t + self.rng.normal() * self.spec.jitter_s * self.fs).floor() as i64;
                let e = 10f64.powf(self.spec.spread_db * self.rng.normal() / 10.0);
                self.pending.push((at, e));
                self.src[i].0 += 1;
            }
        }
        self.level *= self.d;
        let n = self.n;
        let on = self.on;
        let d = self.d;
        let mut add = 0.0;
        self.pending.retain(|&(at, e)| {
            if at <= n {
                if at == n && on {
                    add += e * (1.0 - d);
                }
                false
            } else {
                true
            }
        });
        self.level += add;
        self.left -= 1;
        if self.left <= 0 {
            self.on = !self.on;
            self.left = self.hold();
        }
        self.n += 1;
        self.level / self.mean
    }

    /// Adds arcing of mean power `variance`.
    pub fn add_c(&mut self, x: &mut [Complex32], variance: f64) {
        for v in x {
            let s = (self.envelope() * variance).sqrt() as f32;
            *v += self.rng.complex_normal() * s;
        }
    }

    pub fn add_r(&mut self, x: &mut [f32], variance: f64) {
        for v in x {
            let s = (self.envelope() * variance).sqrt();
            *v += (self.rng.normal() * s) as f32;
        }
    }
}
