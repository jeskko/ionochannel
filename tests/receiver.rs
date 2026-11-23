//! The receiver against its design: filter response, mistuning, latency,
//! AGC attack, hang and decay times, streaming.

use ionochannel::receiver::{STOPBAND_DB, bandpass_taps};
use ionochannel::{Agc, RealReceiver, Receiver, ReceiverSpec};
use num_complex::{Complex32, Complex64};
use std::f64::consts::TAU;

const FS: f64 = 12000.0;

fn tone_c(f: f64, n: usize, amp: f64) -> Vec<Complex32> {
    (0..n)
        .map(|i| {
            let z = Complex64::from_polar(amp, TAU * f * i as f64 / FS);
            Complex32::new(z.re as f32, z.im as f32)
        })
        .collect()
}

/// Complex amplitude of a tone at f in y[from..].
fn amp_c(y: &[Complex32], f: f64, from: usize) -> Complex64 {
    let s: Complex64 = y[from..]
        .iter()
        .enumerate()
        .map(|(i, v)| {
            Complex64::new(v.re as f64, v.im as f64)
                * Complex64::from_polar(1.0, -TAU * f * (i + from) as f64 / FS)
        })
        .sum();
    s / (y.len() - from) as f64
}

fn amp_r(y: &[f32], f: f64, from: usize) -> f64 {
    let c: Vec<Complex32> = y.iter().map(|&v| Complex32::new(v, 0.0)).collect();
    2.0 * amp_c(&c, f, from).norm()
}

fn db(x: f64) -> f64 {
    20.0 * x.log10()
}

#[test]
fn passband_response() {
    let spec = ReceiverSpec {
        passband: Some((300.0, 2700.0)),
        ..Default::default()
    };
    let n = 24000;
    let gain = |f: f64| {
        let mut rx = Receiver::new(FS, &spec);
        let mut x = tone_c(f, n, 1.0);
        rx.process(&mut x);
        db(amp_c(&x, f, 2000).norm())
    };
    for f in [350.0, 1000.0, 1500.0, 2650.0] {
        assert!(gain(f).abs() < 0.02, "pass {f}: {} dB", gain(f));
    }
    for f in [300.0, 2700.0] {
        assert!((gain(f) + 6.02).abs() < 0.1, "edge {f}: {} dB", gain(f));
    }
    for f in [250.0, 2750.0, 0.0, -300.0, -1500.0, 4000.0] {
        assert!(gain(f) < -STOPBAND_DB + 1.0, "stop {f}: {} dB", gain(f));
    }
    // real audio: the same filter on the analytic signal
    let mut rx = RealReceiver::new(FS, &spec);
    for (f, want) in [(1500.0, 0.0), (300.0, -6.02), (2750.0, -100.0)] {
        let mut x: Vec<f32> = (0..n)
            .map(|i| (TAU * f * i as f64 / FS).cos() as f32)
            .collect();
        rx.process(&mut x);
        let g = db(amp_r(&x, f, 2000));
        assert!(
            (g - want).abs() < 0.1 || (want < -60.0 && g < -STOPBAND_DB + 1.0),
            "real {f}: {g} dB"
        );
    }
    // the design: odd length, symmetric about the centre after demodulation
    let t = bandpass_taps(FS, 300.0, 2700.0, 100.0);
    assert!(t.len() % 2 == 1);
}

#[test]
fn latency_is_exact() {
    let spec = ReceiverSpec {
        passband: Some((300.0, 2700.0)),
        ..Default::default()
    };
    let mut rx = Receiver::new(FS, &spec);
    let lat = rx.latency();
    let x = tone_c(1000.0, 12000, 1.0);
    let mut y = x.clone();
    rx.process(&mut y);
    for i in 3000..12000 {
        assert!((y[i] - x[i - lat]).norm() < 2e-3, "sample {i}");
    }
}

#[test]
fn mistuning_moves_audio_up() {
    let spec = ReceiverSpec {
        mistune_hz: 150.0,
        ..Default::default()
    };
    let mut rx = RealReceiver::new(FS, &spec);
    let mut x: Vec<f32> = (0..24000)
        .map(|i| (TAU * 1000.0 * i as f64 / FS).cos() as f32)
        .collect();
    rx.process(&mut x);
    let (up, at, down) = (
        amp_r(&x, 1150.0, 2000),
        amp_r(&x, 1000.0, 2000),
        amp_r(&x, 850.0, 2000),
    );
    assert!(
        (up - 1.0).abs() < 0.01 && at < 0.01 && down < 0.01,
        "{up} {at} {down}"
    );
}

/// A tone whose amplitude steps through `levels`, `secs` each; the output
/// envelope.
fn agc_env(agc: Agc, levels: &[(f64, f64)]) -> Vec<f64> {
    let mut x = Vec::new();
    for &(amp, secs) in levels {
        let n = (secs * FS) as usize;
        let t0 = x.len();
        x.extend((0..n).map(|i| {
            let z = Complex64::from_polar(amp, TAU * 700.0 * (t0 + i) as f64 / FS);
            Complex32::new(z.re as f32, z.im as f32)
        }));
    }
    let mut rx = Receiver::new(
        FS,
        &ReceiverSpec {
            agc: Some(agc),
            ..Default::default()
        },
    );
    let lat = rx.latency();
    x.extend(std::iter::repeat_n(Complex32::default(), lat));
    rx.process(&mut x);
    x[lat..].iter().map(|z| z.norm() as f64).collect()
}

/// First time (s) after `from` that the envelope is within 1 dB below the target.
fn recovers(env: &[f64], target: f64, from: f64) -> f64 {
    let i0 = (from * FS) as usize;
    let k = env[i0..]
        .iter()
        .position(|&e| e >= target / 10f64.powf(1.0 / 20.0))
        .unwrap();
    k as f64 / FS
}

#[test]
fn agc_level_attack_decay_hang() {
    let a = Agc::default();
    let env = agc_env(a.clone(), &[(1.0, 1.0), (10.0, 1.0), (1.0, 3.0)]);
    let near = |i: usize, tol_db: f64| (db(env[i] / a.target)).abs() < tol_db;
    // settled from the start (it starts on the first block's level)
    assert!((100..12000).all(|i| near(i, 0.01)));
    // +20 dB: overshoot, then back within 1 dB in ~2.1 attack times
    let up = 12000;
    assert!(env[up + 2] > 2.0 * a.target);
    assert!((up + 72..up + 12000).all(|i| near(i, 1.0)));
    assert!((up + 600..up + 12000).all(|i| near(i, 0.01)));
    // -20 dB: recovers to -1 dB after decay * ln(10 / 10^(1/20))
    let t = recovers(&env, a.target, 2.0);
    let want = a.decay_s * (10.0f64 / 10f64.powf(1.0 / 20.0)).ln();
    assert!((t - want).abs() < 0.003, "decay {t} vs {want}");
    // hang delays the decay by hang_s
    let h = Agc {
        hang_s: 0.2,
        ..a.clone()
    };
    let env = agc_env(h, &[(10.0, 1.0), (1.0, 3.0)]);
    let t = recovers(&env, a.target, 1.0);
    assert!((t - want - 0.2).abs() < 0.003, "hang {t} vs {}", want + 0.2);
    // below the floor the gain is max_gain
    let env = agc_env(a.clone(), &[(1e-5, 0.5)]);
    let g = env[3000] / 1e-5;
    assert!(
        (db(g) - a.max_gain_db).abs() < 0.01,
        "max gain {} dB",
        db(g)
    );
}

#[test]
fn streaming_block_size_independent() {
    let spec = ReceiverSpec {
        mistune_hz: -120.0,
        ..ReceiverSpec::ssb()
    };
    let x: Vec<f32> = (0..20000)
        .map(|i| {
            let t = i as f64 / FS;
            ((TAU * 900.0 * t).sin() * (1.0 + 5.0 * (TAU * 0.7 * t).sin().max(0.0))) as f32
        })
        .collect();
    let mut a = x.clone();
    RealReceiver::new(FS, &spec).process(&mut a);
    let mut b = x;
    let mut rx = RealReceiver::new(FS, &spec);
    let mut at = 0;
    for (k, len) in [1usize, 7, 13, 500, 3, 4096].iter().cycle().enumerate() {
        if at >= b.len() {
            break;
        }
        let end = (at + len + k % 3).min(b.len());
        rx.process(&mut b[at..end]);
        at = end;
    }
    assert_eq!(a, b);
}
