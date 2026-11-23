//! Paths that change over time: Doppler, power, spread and delay follow
//! their trajectories; the TID helper's delay and Doppler agree.

use ionochannel::vary::{C, tid};
use ionochannel::{Channel, PathSpec, Term, Vary};
use num_complex::{Complex32, Complex64};
use std::f64::consts::{PI, TAU};

fn run(fs: f64, paths: &[PathSpec], vary: &[Vary], x: &mut Vec<Complex32>, seed: u64) -> usize {
    let mut ch = Channel::with_variation(fs, paths, vary, seed);
    let lat = ch.latency();
    x.extend(std::iter::repeat_n(Complex32::default(), lat));
    ch.process(x);
    x.drain(..lat);
    lat
}

fn inst_freq(z: &[Complex32], fs: f64) -> Vec<f64> {
    z.windows(2)
        .map(|w| (w[1] * w[0].conj()).arg() as f64 * fs / TAU)
        .collect()
}

#[test]
fn doppler_follows_a_sine() {
    let fs = 1000.0;
    let v = Vary {
        shift_hz: vec![Term::Sine {
            amp: 5.0,
            period_s: 1.0,
            phase: 0.0,
        }],
        ..Default::default()
    };
    let mut x = vec![Complex32::new(1.0, 0.0); 4000];
    run(fs, &[PathSpec::specular(0.0, 1.0, 0.0)], &[v], &mut x, 1);
    for (i, f) in inst_freq(&x, fs).iter().enumerate() {
        let want = 5.0 * (TAU * (i as f64 + 0.5) / fs).sin();
        assert!((f - want).abs() < 0.05, "{i}: {f} vs {want}");
    }
}

#[test]
fn power_ramps() {
    let fs = 1000.0;
    let v = Vary {
        power_db: vec![Term::Ramp {
            t0_s: 1.0,
            t1_s: 2.0,
            from: 0.0,
            to: -20.0,
        }],
        ..Default::default()
    };
    let mut x = vec![Complex32::new(1.0, 0.0); 3000];
    run(fs, &[PathSpec::specular(0.0, 1.0, 0.0)], &[v], &mut x, 2);
    let a = |t: f64| x[(t * fs) as usize].norm() as f64;
    assert!((a(0.5) - 1.0).abs() < 1e-5 && (a(2.5) - 0.1).abs() < 1e-5);
    assert!((a(1.5) - 10f64.powf(-0.5)).abs() < 1e-3, "{}", a(1.5));
}

/// Spread (2 sigma) from the autocorrelation at the lag where a Gaussian
/// spectrum of `guess` would give 0.5.
fn spread(g: &[Complex32], fs: f64, guess: f64) -> f64 {
    let sigma = guess / 2.0;
    let lag = ((0.5f64.ln() / -2.0).sqrt() / (PI * sigma) * fs).round() as usize;
    let c = |a: Complex32| Complex64::new(a.re as f64, a.im as f64);
    let n = g.len() - lag;
    let r: Complex64 = (0..n)
        .map(|i| c(g[i + lag]) * c(g[i]).conj())
        .sum::<Complex64>()
        / n as f64;
    let p: f64 = g.iter().map(|v| v.norm_sqr() as f64).sum::<f64>() / g.len() as f64;
    2.0 * ((-(r.norm() / p).ln() / 2.0).sqrt() / (PI * lag as f64 / fs))
}

#[test]
fn spread_steps_up_an_octave() {
    let fs = 100.0;
    let secs = 8000.0;
    let v = Vary {
        spread_oct: vec![Term::Ramp {
            t0_s: secs / 2.0,
            t1_s: secs / 2.0,
            from: 0.0,
            to: 1.0,
        }],
        ..Default::default()
    };
    let mut x = vec![Complex32::new(1.0, 0.0); (fs * secs) as usize];
    run(
        fs,
        &[PathSpec::scatter(0.0, 1.0, 1.0, 0.0)],
        &[v],
        &mut x,
        3,
    );
    let (a, b) = x.split_at(x.len() / 2);
    let (s1, s2) = (spread(a, fs, 1.0), spread(b, fs, 2.0));
    assert!(
        (s1 - 1.0).abs() < 0.06 && (s2 - 2.0).abs() < 0.12,
        "{s1} {s2}"
    );
    let pa = a.iter().map(|v| v.norm_sqr() as f64).sum::<f64>() / a.len() as f64;
    let pb = b.iter().map(|v| v.norm_sqr() as f64).sum::<f64>() / b.len() as f64;
    assert!(
        (pa - 1.0).abs() < 0.08 && (pb - 1.0).abs() < 0.08,
        "{pa} {pb}"
    );
}

#[test]
fn delay_follows_a_sine() {
    let fs = 12000.0;
    let f = 1000.0;
    let v = Vary {
        delay_s: vec![Term::Sine {
            amp: 0.2e-3,
            period_s: 2.0,
            phase: 0.0,
        }],
        ..Default::default()
    };
    let n = (fs * 4.0) as usize;
    let mut x: Vec<Complex32> = (0..n)
        .map(|i| {
            let z = Complex64::from_polar(1.0, TAU * f * i as f64 / fs);
            Complex32::new(z.re as f32, z.im as f32)
        })
        .collect();
    let x0 = x.clone();
    run(fs, &[PathSpec::specular(1e-3, 1.0, 0.0)], &[v], &mut x, 4);
    // the specular path has a random phase: take it out at the start
    let ph0 = (x[2000] * x0[2000].conj()).arg() as f64;
    let tau0 = 1e-3 + 0.2e-3 * (TAU * 2000.0 / fs / 2.0).sin();
    for i in (2000..n).step_by(97) {
        let t = i as f64 / fs;
        let tau = 1e-3 + 0.2e-3 * (TAU * t / 2.0).sin();
        let got = (x[i] * x0[i].conj()).arg() as f64 - ph0;
        let want = -TAU * f * (tau - tau0);
        let d = (got - want + PI).rem_euclid(TAU) - PI;
        assert!(d.abs() < 0.01, "t {t}: {got} vs {want}");
        assert!((x[i].norm() - 1.0).abs() < 2e-3);
    }
}

#[test]
fn drift_rms() {
    let fs = 200.0;
    let v = Vary {
        shift_hz: vec![Term::Drift {
            rms: 2.0,
            tau_s: 1.0,
        }],
        ..Default::default()
    };
    let (mut s1, mut s2, mut k) = (0.0, 0.0, 0.0);
    for seed in 0..4 {
        let mut x = vec![Complex32::new(1.0, 0.0); (fs * 300.0) as usize];
        run(
            fs,
            &[PathSpec::specular(0.0, 1.0, 0.0)],
            std::slice::from_ref(&v),
            &mut x,
            seed,
        );
        for f in inst_freq(&x, fs) {
            s1 += f;
            s2 += f * f;
            k += 1.0;
        }
    }
    let sd = (s2 / k - (s1 / k).powi(2)).sqrt();
    assert!((sd / 2.0 - 1.0).abs() < 0.1, "drift rms {sd}");
}

#[test]
fn tid_delay_and_doppler_agree() {
    let (f, km, rel, period) = (10e6, 1000.0, 0.05, 3600.0);
    let v = tid(f, km, rel, period, 0.3);
    let Term::Sine { amp: da, .. } = v.delay_s[0] else {
        panic!()
    };
    let Term::Sine { amp: fa, .. } = v.shift_hz[0] else {
        panic!()
    };
    assert!((da - 50e3 / C).abs() < 1e-12);
    assert!((fa - 10e6 * 50e3 * TAU / (C * 3600.0)).abs() < 1e-9);
    // Doppler = -f d(delay)/dt, numerically
    let val = |terms: &[Term], t: f64| match terms[0] {
        Term::Sine {
            amp,
            period_s,
            phase,
        } => amp * (TAU * t / period_s + phase).sin(),
        _ => unreachable!(),
    };
    for t in [0.0, 500.0, 1234.0, 3000.0] {
        let dt = 0.01;
        let ddelay = (val(&v.delay_s, t + dt) - val(&v.delay_s, t - dt)) / (2.0 * dt);
        assert!((val(&v.shift_hz, t) + f * ddelay).abs() < 1e-6);
    }
}

#[test]
fn streaming_with_variation() {
    let paths = [
        PathSpec::scatter(0.0, 0.5, 2.0, 0.0),
        PathSpec::scatter(2e-3, 0.5, 2.0, 0.0),
    ];
    let v = vec![
        tid(7e6, 800.0, 0.05, 60.0, 0.0),
        Vary {
            power_db: vec![Term::Drift {
                rms: 3.0,
                tau_s: 0.5,
            }],
            spread_oct: vec![Term::Sine {
                amp: 1.0,
                period_s: 3.0,
                phase: 0.0,
            }],
            ..Default::default()
        },
    ];
    let x: Vec<Complex32> = (0..12000)
        .map(|i| Complex32::new((i as f32 * 0.1).sin(), 0.3))
        .collect();
    let mut a = x.clone();
    Channel::with_variation(12000.0, &paths, &v, 9).process(&mut a);
    let mut b = x;
    let mut ch = Channel::with_variation(12000.0, &paths, &v, 9);
    for c in b.chunks_mut(333) {
        ch.process(c);
    }
    assert_eq!(a, b);
}
