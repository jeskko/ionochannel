//! Man-made interference against its parameters: comb lines, ripple AM
//! and FM, drift, arcing power, mains lock and gating, streaming.

use ionochannel::interference::smps_lines;
use ionochannel::{Arcing, ArcingSpec, Bursts, BurstsSpec, Smps, SmpsSpec};
use num_complex::{Complex32, Complex64};
use std::f64::consts::TAU;

const FS: f64 = 12000.0;

fn amp(z: &[Complex32], f: f64, fs: f64) -> f64 {
    let s: Complex64 = z
        .iter()
        .enumerate()
        .map(|(i, v)| {
            Complex64::new(v.re as f64, v.im as f64)
                * Complex64::from_polar(1.0, -TAU * f * i as f64 / fs)
        })
        .sum();
    (s / z.len() as f64).norm()
}

fn smps(fs: f64, secs: f64, offset: f64, spec: SmpsSpec, seed: u64) -> Vec<Complex32> {
    let mut z = vec![Complex32::default(); (fs * secs) as usize];
    Smps::new(fs, offset, &spec, false, seed).add_c(&mut z, 1.0);
    z
}

fn still() -> SmpsSpec {
    SmpsSpec {
        am_depth: 0.0,
        fm_dev_hz: 0.0,
        drift_hz: 0.0,
        ..Default::default()
    }
}

#[test]
fn smps_lines_and_ripple() {
    let spec = SmpsSpec {
        spacing_hz: 2500.0,
        ..still()
    };
    let lines = smps_lines(FS, 700.0, 2500.0, false);
    assert_eq!(lines, vec![-4300.0, -1800.0, 700.0, 3200.0, 5700.0]);
    let z = smps(FS, 4.0, 700.0, spec, 1);
    for f in lines {
        assert!((amp(&z, f, FS) - 1.0).abs() < 1e-5, "{f}");
    }
    assert!(amp(&z, 1950.0, FS) < 1e-3);
    let m = 0.4;
    let z = smps(
        FS,
        4.0,
        700.0,
        SmpsSpec {
            spacing_hz: 1e6,
            am_depth: m,
            ..still()
        },
        2,
    );
    let c = amp(&z, 700.0, FS);
    for f in [600.0, 800.0] {
        assert!((amp(&z, f, FS) / c - m / 2.0).abs() < 1e-3, "AM {f}");
    }
    let z = smps(
        FS,
        4.0,
        700.0,
        SmpsSpec {
            spacing_hz: 1e6,
            fm_dev_hz: 25.0,
            ..still()
        },
        3,
    );
    let fi: Vec<f64> = z
        .windows(2)
        .map(|w| (w[1] * w[0].conj()).arg() as f64 * FS / TAU)
        .collect();
    let (lo, hi) = fi
        .iter()
        .fold((f64::MAX, f64::MIN), |(a, b), &v| (a.min(v), b.max(v)));
    assert!(
        ((hi - lo) / 2.0 - 25.0).abs() < 0.5,
        "FM {}",
        (hi - lo) / 2.0
    );
}

#[test]
fn smps_drift() {
    let spec = SmpsSpec {
        spacing_hz: 1e6,
        drift_hz: 30.0,
        drift_s: 1.0,
        ..still()
    };
    let (mut s1, mut s2, mut n) = (0.0, 0.0, 0.0);
    for seed in 0..4 {
        let z = smps(1000.0, 200.0, 100.0, spec.clone(), seed);
        for w in z.windows(2) {
            let f = (w[1] * w[0].conj()).arg() as f64 * 1000.0 / TAU - 100.0;
            s1 += f;
            s2 += f * f;
            n += 1.0;
        }
    }
    let sd = (s2 / n - (s1 / n).powi(2)).sqrt();
    assert!((sd / 30.0 - 1.0).abs() < 0.1, "drift rms {sd}");
}

fn arcing(secs: f64, spec: &ArcingSpec, seed: u64) -> Vec<Complex32> {
    let mut z = vec![Complex32::default(); (FS * secs) as usize];
    Arcing::new(FS, spec, seed).add_c(&mut z, 1.0);
    z
}

#[test]
fn arcing_power_and_mains_lock() {
    let spec = ArcingSpec {
        jitter_s: 1e-4,
        ..Default::default()
    };
    let mut p = 0.0;
    for seed in 0..4 {
        let z = arcing(60.0, &spec, seed);
        p += z.iter().map(|v| v.norm_sqr() as f64).sum::<f64>() / z.len() as f64 / 4.0;
        if seed == 0 {
            let half = (FS / 100.0) as usize;
            let mut fold = vec![0.0; half];
            for (i, v) in z.iter().enumerate() {
                fold[i % half] += v.norm_sqr() as f64;
            }
            let k = (0..half)
                .max_by(|&a, &b| fold[a].total_cmp(&fold[b]))
                .unwrap();
            let near: f64 = (0..25).map(|d| fold[(k + half + d - 12) % half]).sum();
            let all: f64 = fold.iter().sum();
            assert!(near / all > 0.9, "mains lock {}", near / all);
        }
    }
    assert!((p - 1.0).abs() < 0.1, "power {p}");
}

#[test]
fn arcing_gate() {
    let spec = ArcingSpec {
        on_s: 1.0,
        off_s: 3.0,
        ..Default::default()
    };
    let half = (FS / 100.0) as usize;
    let (mut on, mut all) = (0usize, 0usize);
    for seed in 0..8 {
        let z = arcing(60.0, &spec, seed);
        for c in z.chunks_exact(half) {
            let e: f64 = c.iter().map(|v| v.norm_sqr() as f64).sum();
            on += (e > 1.0) as usize;
            all += 1;
        }
    }
    let f = on as f64 / all as f64;
    assert!((f - 0.25).abs() < 0.05, "duty {f}");
}

#[test]
fn streaming() {
    let spec = SmpsSpec::default();
    let mut a = vec![0.0f32; 7000];
    Smps::new(FS, 1500.0, &spec, true, 4).add_r(&mut a, 1.0);
    let mut b = vec![0.0f32; 7000];
    let mut s = Smps::new(FS, 1500.0, &spec, true, 4);
    for c in b.chunks_mut(97) {
        s.add_r(c, 1.0);
    }
    assert_eq!(a, b);
    let mut a = vec![0.0f32; 7000];
    Arcing::new(FS, &ArcingSpec::default(), 5).add_r(&mut a, 1.0);
    let mut b = vec![0.0f32; 7000];
    let mut s = Arcing::new(FS, &ArcingSpec::default(), 5);
    for c in b.chunks_mut(97) {
        s.add_r(c, 1.0);
    }
    assert_eq!(a, b);
}

#[test]
fn bursts_duty_on_time_and_level() {
    let spec = BurstsSpec::default();
    let mut z = vec![Complex32::default(); (FS * 240.0) as usize];
    Bursts::new(FS, &spec, 9).add_c(&mut z, 1.0);
    let on: Vec<bool> = z.iter().map(|v| v.norm() > 0.0).collect();
    let duty = on.iter().filter(|&&b| b).count() as f64 / on.len() as f64;
    assert!((duty - 2.0 / 22.0).abs() < 0.01, "duty {duty}");
    let p = z.iter().map(|v| v.norm_sqr() as f64).sum::<f64>() / (duty * z.len() as f64);
    assert!((p - 1.0).abs() < 0.02, "power while on {p}");
    let (mut runs, mut len) = (Vec::new(), 0usize);
    for &b in &on[1..] {
        if b {
            len += 1;
        } else if len > 0 {
            runs.push(len);
            len = 0;
        }
    }
    let mean = runs[1..].iter().sum::<usize>() as f64 / (runs.len() - 1) as f64 / FS;
    assert!((mean / 0.002 - 1.0).abs() < 0.05, "mean on {mean}");
}
