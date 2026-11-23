//! Recorded noise: level against white noise, band-limited recordings,
//! seamless loops, streaming.

use ionochannel::{RealReceiver, ReceiverSpec, Recorded, Rng, noise};
use num_complex::Complex32;
use std::f64::consts::TAU;

const FS: f64 = 12000.0;

fn white(n: usize, seed: u64, s: f32) -> Vec<f32> {
    let mut r = Rng::new(seed);
    (0..n).map(|_| r.normal() as f32 * s).collect()
}

fn power(x: &[f32]) -> f64 {
    x.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / x.len() as f64
}

#[test]
fn white_recording_matches_white_noise() {
    let rec = white((FS * 30.0) as usize, 1, 3.0);
    let mut x = vec![0.0f32; (FS * 20.0) as usize];
    let target = 1.0; // signal power 1, SNR 0 dB
    Recorded::new_r(&rec, FS, None, 2).add_r(&mut x, target, 2500.0);
    let want = noise::noise_power(1.0, 0.0, 2500.0, FS, false);
    assert!(
        (power(&x) / want - 1.0).abs() < 0.03,
        "{} vs {want}",
        power(&x)
    );
    let mut r = Rng::new(3);
    let rec: Vec<Complex32> = (0..(FS * 30.0) as usize)
        .map(|_| r.complex_normal())
        .collect();
    let mut z = vec![Complex32::default(); (FS * 20.0) as usize];
    Recorded::new_c(&rec, FS, None, 4).add_c(&mut z, target, 2500.0);
    let p = z.iter().map(|v| v.norm_sqr() as f64).sum::<f64>() / z.len() as f64;
    let want = noise::noise_power(1.0, 0.0, 2500.0, FS, true);
    assert!((p / want - 1.0).abs() < 0.03, "{p} vs {want}");
}

#[test]
fn band_limited_recording() {
    let mut rec = white((FS * 30.0) as usize, 5, 7.0);
    let spec = ReceiverSpec {
        passband: Some((300.0, 2700.0)),
        ..Default::default()
    };
    RealReceiver::new(FS, &spec).process(&mut rec);
    let mut x = vec![0.0f32; (FS * 20.0) as usize];
    let target = 10f64.powf(-0.6);
    Recorded::new_r(&rec, FS, None, 6).add_r(&mut x, target, 2500.0);
    // all of its power is in 2400 Hz of density target / 2500
    let want = target * 2400.0 / 2500.0;
    assert!(
        (power(&x) / want - 1.0).abs() < 0.04,
        "{} vs {want}",
        power(&x)
    );
}

#[test]
fn loop_is_seamless_and_streams() {
    let rec: Vec<f32> = (0..(FS * 0.5) as usize)
        .map(|i| (TAU * 437.3 * i as f64 / FS).sin() as f32)
        .collect();
    let mut a = vec![0.0f32; (FS * 4.0) as usize];
    Recorded::new_r(&rec, FS, None, 7).add_r(&mut a, 1.0, 2500.0);
    let step = |v: &[f32]| {
        v.windows(2)
            .map(|w| (w[1] - w[0]).abs())
            .fold(0.0, f32::max)
    };
    let scale = a.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    assert!(step(&a) / scale <= step(&rec) * 1.01, "seam click");
    let mut b = vec![0.0f32; a.len()];
    let mut s = Recorded::new_r(&rec, FS, None, 7);
    for c in b.chunks_mut(777) {
        s.add_r(c, 1.0, 2500.0);
    }
    assert_eq!(a, b);
}
