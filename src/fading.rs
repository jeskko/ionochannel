//! Tap-gain processes (ITU-R F.1487 §2, eq. 1-2): each component of a path
//! is either Rayleigh scatter (a complex Gaussian process with a Gaussian
//! Doppler spectrum, 2 sigma = `spread_hz`, shifted by `shift_hz`) or a
//! specular ray (steady amplitude, a random start phase, shifted).
//!
//! Scatter is made at a low rate (32 x the spread, as MIL-STD-188-110
//! Appendix E asks) by filtering white complex noise with a Gaussian FIR
//! h(t) = exp(-(pi d t)^2), whose power response exp(-2 f^2 / d^2) is
//! exactly the Gaussian of 2 sigma = d, then interpolated linearly to the
//! sample rate (32x oversampled: < 0.3 % power lost between points). The
//! shift is applied at the sample rate. Streams forever: no wrap-around.

use crate::rng::Rng;
use num_complex::{Complex32, Complex64};
use std::collections::VecDeque;
use std::f64::consts::{PI, TAU};

/// One component of a path's tap gain.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Component {
    /// Rayleigh scatter: mean power, Doppler spread (2 sigma, Hz), shift (Hz).
    Scatter {
        power: f64,
        spread_hz: f64,
        shift_hz: f64,
    },
    /// A steady ray (Rice when added to scatter): power, shift (Hz).
    Specular { power: f64, shift_hz: f64 },
}

impl Component {
    pub fn power(&self) -> f64 {
        match *self {
            Component::Scatter { power, .. } | Component::Specular { power, .. } => power,
        }
    }
}

/// Oversampling of the low-rate scatter process over its spread.
const OVERSAMPLE: f64 = 32.0;
/// The Gaussian FIR is cut where it has fallen to this fraction of its peak.
const FIR_CUT: f64 = 1e-4;
/// The slowest low rate (Hz), so tiny spreads still interpolate smoothly.
const MIN_LOW_RATE: f64 = 0.05;

struct Scatter {
    taps: Vec<f64>,
    noise: VecDeque<Complex64>,
    /// Low-rate samples either side of the current position.
    a: Complex64,
    b: Complex64,
    /// Position between `a` and `b` (0..1) and its step per output sample.
    pos: f64,
    step: f64,
    rng: Rng,
}

impl Scatter {
    fn new(spread_hz: f64, fs: f64, mut rng: Rng) -> Self {
        assert!(
            spread_hz > 0.0,
            "a scatter component needs a spread > 0 (use specular for none)"
        );
        let low = (OVERSAMPLE * spread_hz).max(MIN_LOW_RATE);
        let half = ((-FIR_CUT.ln()).sqrt() / (PI * spread_hz) * low).ceil() as i64;
        let mut taps: Vec<f64> = (-half..=half)
            .map(|k| {
                let t = k as f64 / low;
                (-(PI * spread_hz * t).powi(2)).exp()
            })
            .collect();
        let e: f64 = taps.iter().map(|v| v * v).sum::<f64>().sqrt();
        taps.iter_mut().for_each(|v| *v /= e);
        let noise = (0..taps.len()).map(|_| c64(rng.complex_normal())).collect();
        let mut s = Scatter {
            taps,
            noise,
            a: Complex64::default(),
            b: Complex64::default(),
            pos: 0.0,
            step: low / fs,
            rng,
        };
        s.a = s.next_low();
        s.b = s.next_low();
        s
    }

    fn next_low(&mut self) -> Complex64 {
        self.noise.pop_front();
        self.noise.push_back(c64(self.rng.complex_normal()));
        self.noise.iter().zip(&self.taps).map(|(n, t)| n * t).sum()
    }

    fn next(&mut self) -> Complex64 {
        let g = self.a + (self.b - self.a) * self.pos;
        self.pos += self.step;
        while self.pos >= 1.0 {
            self.pos -= 1.0;
            self.a = self.b;
            self.b = self.next_low();
        }
        g
    }
}

fn c64(z: Complex32) -> Complex64 {
    Complex64::new(z.re as f64, z.im as f64)
}

/// A running Doppler shift (phasor at the sample rate, renormalised).
struct Shift {
    ph: Complex64,
    rot: Complex64,
    n: u32,
}

impl Shift {
    fn new(shift_hz: f64, fs: f64, phase: f64) -> Self {
        Shift {
            ph: Complex64::from_polar(1.0, phase),
            rot: Complex64::from_polar(1.0, TAU * shift_hz / fs),
            n: 0,
        }
    }

    fn next(&mut self) -> Complex64 {
        let p = self.ph;
        self.ph *= self.rot;
        self.n += 1;
        if self.n == 1024 {
            self.n = 0;
            self.ph /= self.ph.norm();
        }
        p
    }
}

struct Part {
    amp: f64,
    scatter: Option<Scatter>,
    shift: Shift,
}

/// The tap gain of one path: the sum of its components, one sample at a time.
pub struct TapGain {
    parts: Vec<Part>,
}

impl TapGain {
    pub fn new(components: &[Component], fs: f64, rng: &mut Rng) -> Self {
        let parts = components
            .iter()
            .map(|c| match *c {
                Component::Scatter {
                    power,
                    spread_hz,
                    shift_hz,
                } => Part {
                    amp: power.sqrt(),
                    scatter: Some(Scatter::new(spread_hz, fs, rng.fork())),
                    shift: Shift::new(shift_hz, fs, 0.0),
                },
                Component::Specular { power, shift_hz } => Part {
                    amp: power.sqrt(),
                    scatter: None,
                    shift: Shift::new(shift_hz, fs, TAU * rng.uniform()),
                },
            })
            .collect();
        TapGain { parts }
    }

    pub fn sample(&mut self) -> Complex64 {
        self.parts
            .iter_mut()
            .map(|p| {
                let s = p.shift.next() * p.amp;
                match &mut p.scatter {
                    Some(sc) => sc.next() * s,
                    None => s,
                }
            })
            .sum()
    }
}
