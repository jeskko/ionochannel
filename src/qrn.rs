//! Impulsive atmospheric noise (QRN), set by its voltage deviation Vd in a
//! 200 Hz bandwidth as ITU-R P.372 defines it (Vd = 20 log10 of the
//! envelope's rms over its mean; Fig. 37 gives the amplitude distribution
//! for each Vd, Fig. 38 how Vd grows with bandwidth).
//!
//! The model is Gaussian noise whose power follows a shot-noise envelope.
//! Bursts (sferics) arrive at random (Poisson). Each has energy a^2 from a
//! Pareto law (P(a^2 > x) = x^(-alpha/2), alpha = 4) and decays
//! exponentially, with an amplitude time constant drawn log-uniformly from
//! 0.1 to 100 ms (8 bins). A Gaussian floor carries a share of the power
//! that falls with Vd. The burst rate for each Vd comes from a table
//! calibrated by simulation, in `python/ionochannel/qrn.json`, which the
//! Python version reads too. Against P.372-18: Vd in 200 Hz within about
//! 0.3 dB of the target, Fig. 38 within 1 dB from 2 to 6 kHz (2 dB at
//! 20 Hz), Fig. 37 within 2.5 dB where the figure can be read.

use crate::rng::Rng;
use num_complex::Complex32;
use serde_json::Value;
use std::sync::OnceLock;

const JSON: &str = include_str!("../python/ionochannel/qrn.json");

/// Vd of Gaussian noise (Rayleigh envelope): 20 log10(sqrt(4/pi)).
pub const VD_GAUSS: f64 = 1.049_090_7;

struct Model {
    alpha: f64,
    taus: Vec<f64>,
    slope: Vec<(f64, f64)>,
    table: Vec<(f64, f64)>,
}

fn model() -> &'static Model {
    static M: OnceLock<Model> = OnceLock::new();
    M.get_or_init(|| {
        let v: Value = serde_json::from_str(JSON).expect("qrn.json");
        let pairs = |k: &str| -> Vec<(f64, f64)> {
            v[k].as_array()
                .expect("qrn.json table")
                .iter()
                .map(|p| (p[0].as_f64().unwrap(), p[1].as_f64().unwrap()))
                .collect()
        };
        let (t0, t1) = (
            v["tau_s"][0].as_f64().unwrap(),
            v["tau_s"][1].as_f64().unwrap(),
        );
        let bins = v["bins"].as_u64().unwrap() as usize;
        Model {
            alpha: v["alpha"].as_f64().unwrap(),
            taus: (0..bins)
                .map(|i| t0 * (t1 / t0).powf(i as f64 / (bins - 1) as f64))
                .collect(),
            slope: pairs("fig38_slope"),
            table: pairs("table"),
        }
    })
}

fn interp(x: f64, t: &[(f64, f64)]) -> f64 {
    if x <= t[0].0 {
        return t[0].1;
    }
    for w in t.windows(2) {
        if x <= w[1].0 {
            let f = (x - w[0].0) / (w[1].0 - w[0].0);
            return w[0].1 + f * (w[1].1 - w[0].1);
        }
    }
    t[t.len() - 1].1
}

/// Highest Vd (200 Hz, dB) the calibration covers.
pub fn vd_max() -> f64 {
    model().table.last().unwrap().0
}

/// Gaussian share of the power for a 200 Hz Vd.
pub fn floor(vd_db: f64) -> f64 {
    if vd_db >= 2.0 {
        0.1 * 10f64.powf(-0.19 * (vd_db - 2.0))
    } else {
        10f64.powf(-(vd_db - VD_GAUSS) / (2.0 - VD_GAUSS))
    }
}

/// (bursts per second, Gaussian share) for a 200 Hz Vd.
pub fn params(vd_db: f64) -> (f64, f64) {
    assert!(
        (VD_GAUSS..=vd_max()).contains(&vd_db),
        "Vd {vd_db} dB outside {VD_GAUSS:.2}..{} (200 Hz)",
        vd_max()
    );
    let t: Vec<(f64, f64)> = model().table.iter().map(|&(v, r)| (v, r.ln())).collect();
    (interp(vd_db, &t).exp(), floor(vd_db))
}

/// Vd in bandwidth `bw_hz` from the 200 Hz value (P.372 Fig. 38).
pub fn vd_bandwidth(vd200_db: f64, bw_hz: f64) -> f64 {
    let s = interp(vd200_db, &model().slope);
    (vd200_db + s * (bw_hz / 200.0).log10()).max(VD_GAUSS)
}

/// A QRN source, one sample at a time; mean power 1.
pub struct Qrn {
    rng: Rng,
    decay: Vec<f64>,
    state: Vec<f64>,
    /// Bursts per sample.
    rate: f64,
    floor: f64,
    mean: f64,
    /// Arrival time of the next burst, in samples from the current one.
    next: f64,
    gauss: bool,
}

impl Qrn {
    /// QRN at sample rate `fs` with Vd `vd_db` (200 Hz). It starts in
    /// its steady state (bursts from before the start are included).
    pub fn new(fs: f64, vd_db: f64, seed: u64) -> Self {
        let m = model();
        let gauss = vd_db <= VD_GAUSS + 1e-9;
        let (rate, floor) = if gauss { (0.0, 1.0) } else { params(vd_db) };
        let mut rng = Rng::new(seed);
        let rate = rate / fs;
        let next = if gauss {
            f64::INFINITY
        } else {
            -(1.0 - rng.uniform()).ln() / rate
        };
        let mut q = Qrn {
            rng,
            decay: m.taus.iter().map(|t| (-2.0 / (fs * t)).exp()).collect(),
            state: vec![0.0; m.taus.len()],
            rate,
            floor,
            mean: rate * m.alpha / (m.alpha - 2.0),
            next,
            gauss,
        };
        if !gauss {
            let pre = (fs * m.taus[m.taus.len() - 1] / 2.0 * 25.0).ceil() as usize;
            for _ in 0..pre {
                q.envelope();
            }
        }
        q
    }

    /// The power envelope for the next sample (mean 1).
    pub fn envelope(&mut self) -> f64 {
        if self.gauss {
            return 1.0;
        }
        for (s, d) in self.state.iter_mut().zip(&self.decay) {
            *s *= d;
        }
        while self.next < 1.0 {
            let bin = (self.rng.next_u64() % self.state.len() as u64) as usize;
            let a2 = (1.0 - self.rng.uniform()).powf(-2.0 / model().alpha);
            self.state[bin] += a2 * (1.0 - self.decay[bin]);
            self.next += -(1.0 - self.rng.uniform()).ln() / self.rate;
        }
        self.next -= 1.0;
        let sum: f64 = self.state.iter().sum();
        self.floor + (1.0 - self.floor) * sum / self.mean
    }

    /// Adds QRN of mean power `variance` (as [`crate::noise::add_c`]).
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
