//! Paths that change over time. Each path's delay, Doppler shift, power
//! and spread can follow a trajectory: a sum of [`Term`]s added to the
//! path's own value. The stationary Watterson model holds for minutes at
//! most; this is for longer runs and for disturbances.
//!
//! - delay: seconds added to the path's delay (clamped at 0);
//! - shift: Hz added to the whole path (on top of its components' shifts);
//! - power: dB added to the path's power;
//! - spread: octaves: every scatter component's spread is multiplied by
//!   2^value. This time-warps the fading process. The process is made for
//!   the largest spread the trajectory reaches and slowed down from it, so
//!   it stays oversampled at least 32 times.
//!
//! [`tid`] makes the delay and Doppler of a travelling ionospheric
//! disturbance consistent with each other.

use crate::rng::Rng;
use std::f64::consts::{FRAC_PI_2, TAU};

/// Speed of light (m/s).
pub const C: f64 = 299_792_458.0;

/// One term of a trajectory.
#[derive(Clone, Debug, PartialEq)]
pub enum Term {
    /// Linear from `from` at `t0_s` to `to` at `t1_s`, held before and after.
    Ramp {
        t0_s: f64,
        t1_s: f64,
        from: f64,
        to: f64,
    },
    /// `amp * sin(2 pi t / period_s + phase)`.
    Sine { amp: f64, period_s: f64, phase: f64 },
    /// A random drift (Ornstein-Uhlenbeck): `rms`, correlation time
    /// `tau_s`, clamped at +-4 rms. Each realisation draws its own.
    Drift { rms: f64, tau_s: f64 },
    /// A recorded trajectory: values at times (s, increasing), linear in
    /// between, held before the first and after the last. For replaying
    /// measured Doppler or amplitude records.
    Table { t_s: Vec<f64>, values: Vec<f64> },
}

impl Term {
    /// A [`Term::Table`] from (time s, value) pairs, checked.
    pub fn table(points: &[(f64, f64)]) -> Self {
        assert!(!points.is_empty(), "a table needs at least one point");
        assert!(
            points.windows(2).all(|w| w[1].0 > w[0].0),
            "table times must increase"
        );
        Term::Table {
            t_s: points.iter().map(|p| p.0).collect(),
            values: points.iter().map(|p| p.1).collect(),
        }
    }
}

/// What changes on one path; empty trajectories change nothing.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Vary {
    pub delay_s: Vec<Term>,
    pub shift_hz: Vec<Term>,
    pub power_db: Vec<Term>,
    pub spread_oct: Vec<Term>,
}

impl Vary {
    pub fn is_empty(&self) -> bool {
        self.delay_s.is_empty()
            && self.shift_hz.is_empty()
            && self.power_db.is_empty()
            && self.spread_oct.is_empty()
    }
}

/// The range (min, max) a trajectory can take.
pub fn bounds(terms: &[Term]) -> (f64, f64) {
    terms.iter().fold((0.0, 0.0), |(lo, hi), t| {
        let (a, b) = match t {
            Term::Ramp { from, to, .. } => (from.min(*to), from.max(*to)),
            Term::Sine { amp, .. } => (-amp.abs(), amp.abs()),
            Term::Drift { rms, .. } => (-4.0 * rms.abs(), 4.0 * rms.abs()),
            Term::Table { values, .. } => values
                .iter()
                .fold((f64::MAX, f64::MIN), |(l, h), &v| (l.min(v), h.max(v))),
        };
        (lo + a, hi + b)
    })
}

/// A travelling ionospheric disturbance on a path: the group path
/// P = `path_km` swings by `rel_amp` * P with period `period_s`. The delay
/// follows P / c; the Doppler shift is -(f / c) dP/dt (Trop et al.,
/// eq. 4), a quarter period ahead. It treats phase and group path as
/// moving together. Oinats et al. (2016) found MSTIDs most often with
/// periods of 60-100 min and relative amplitudes of 3-8 % (HF radar,
/// F region); medium-scale TIDs run 15-60 min, large-scale 30-180 min.
pub fn tid(carrier_hz: f64, path_km: f64, rel_amp: f64, period_s: f64, phase: f64) -> Vary {
    let dp = rel_amp * path_km * 1e3;
    Vary {
        delay_s: vec![Term::Sine {
            amp: dp / C,
            period_s,
            phase,
        }],
        shift_hz: vec![Term::Sine {
            amp: carrier_hz * dp * TAU / (C * period_s),
            period_s,
            phase: phase - FRAC_PI_2,
        }],
        ..Default::default()
    }
}

/// A trajectory being run, one sample at a time.
pub(crate) struct Run {
    terms: Vec<Term>,
    /// (state, rho, rms) for each drift term, in order.
    ou: Vec<(f64, f64, f64)>,
    rng: Rng,
    fs: f64,
    n: u64,
}

impl Run {
    pub(crate) fn new(terms: &[Term], fs: f64, mut rng: Rng) -> Self {
        let ou = terms
            .iter()
            .filter_map(|t| match *t {
                Term::Drift { rms, tau_s, .. } => {
                    let rho = if tau_s > 0.0 {
                        (-1.0 / (fs * tau_s)).exp()
                    } else {
                        0.0
                    };
                    Some((rng.normal() * rms, rho, rms))
                }
                _ => None,
            })
            .collect();
        Run {
            terms: terms.to_vec(),
            ou,
            rng,
            fs,
            n: 0,
        }
    }

    pub(crate) fn next(&mut self) -> f64 {
        let t = self.n as f64 / self.fs;
        self.n += 1;
        let mut v = 0.0;
        let mut k = 0;
        for term in &self.terms {
            v += match term {
                &Term::Ramp {
                    t0_s,
                    t1_s,
                    from,
                    to,
                } => {
                    if t <= t0_s {
                        from
                    } else if t >= t1_s {
                        to
                    } else {
                        from + (to - from) * (t - t0_s) / (t1_s - t0_s)
                    }
                }
                &Term::Sine {
                    amp,
                    period_s,
                    phase,
                } => amp * (TAU * t / period_s + phase).sin(),
                Term::Table { t_s, values } => {
                    let i = t_s.partition_point(|&x| x <= t);
                    if i == 0 {
                        values[0]
                    } else if i == t_s.len() {
                        values[i - 1]
                    } else {
                        let f = (t - t_s[i - 1]) / (t_s[i] - t_s[i - 1]);
                        values[i - 1] + (values[i] - values[i - 1]) * f
                    }
                }
                Term::Drift { .. } => {
                    let (s, _, rms) = self.ou[k];
                    k += 1;
                    s.clamp(-4.0 * rms, 4.0 * rms)
                }
            };
        }
        for o in &mut self.ou {
            o.0 = o.0 * o.1 + o.2 * (1.0 - o.1 * o.1).sqrt() * self.rng.normal();
        }
        v
    }
}

/// NOAA SWPC D-RAP: the highest frequency (MHz) affected by 1 dB of
/// vertical absorption in a solar X-ray flare of 0.1-0.8 nm flux
/// `flux_wm2`: 10 log10(flux) + 65 MHz at the sub-solar point (exact at
/// M1 -> 15 and X1 -> 25 MHz), times cos(chi)^0.75 away from it.
pub fn flare_haf_mhz(flux_wm2: f64, cos_chi: f64) -> f64 {
    let haf = (10.0 * flux_wm2.max(1e-30).log10() + 65.0).max(0.0);
    haf * cos_chi.clamp(0.0, 1.0).powf(0.75)
}

/// D-RAP frequency scaling: A(f) = (f0 / f)^1.5 A(f0) dB.
pub fn absorption_db(f_mhz: f64, a0_db: f64, f0_mhz: f64) -> f64 {
    (f0_mhz / f_mhz).powf(1.5) * a0_db
}

/// Vertical absorption (dB) at `f_mhz` in a flare (1 dB at the HAF).
pub fn flare_absorption_db(flux_wm2: f64, f_mhz: f64, cos_chi: f64) -> f64 {
    absorption_db(f_mhz, 1.0, flare_haf_mhz(flux_wm2, cos_chi))
}

/// D-RAP polar cap absorption (vertical) at `f_mhz`: 0.115 sqrt(J) dB by
/// day (J: integral proton flux above 5.2 MeV), 0.020 sqrt(J) by night
/// (above 2.2 MeV), at 30 MHz, scaled by (30 / f)^1.5.
pub fn pca_absorption_db(f_mhz: f64, proton_flux: f64, day: bool) -> f64 {
    let a30 = if day { 0.115 } else { 0.020 } * proton_flux.max(0.0).sqrt();
    absorption_db(f_mhz, a30, 30.0)
}
