//! Interfering stations against their parameters: Morse timing and
//! edges, AM carrier and bandwidth, radar sweep rate, streaming.

use ionochannel::stations::morse_units;
use ionochannel::{Am, AmSpec, Cw, CwSpec, Psk31, Psk31Spec, Radar, RadarSpec, Rtty, RttySpec};
use num_complex::{Complex32, Complex64};
use std::f64::consts::TAU;

const FS: f64 = 12000.0;

fn amp(z: &[Complex32], f: f64) -> f64 {
    let s: Complex64 = z
        .iter()
        .enumerate()
        .map(|(i, v)| {
            Complex64::new(v.re as f64, v.im as f64)
                * Complex64::from_polar(1.0, -TAU * f * i as f64 / FS)
        })
        .sum();
    (s / z.len() as f64).norm()
}

#[test]
fn morse_timing() {
    let seq = morse_units("PARIS");
    assert_eq!(seq.iter().map(|s| s.1).sum::<u32>(), 50);
    assert_eq!(seq.iter().filter(|s| s.0).map(|s| s.1).sum::<u32>(), 22);
    let spec = CwSpec {
        text: Some("PARIS".into()),
        ..Default::default()
    };
    let mut z = vec![Complex32::default(); (FS * 30.0) as usize];
    Cw::new(FS, 800.0, &spec, false, 1).add_c(&mut z, 1.0);
    let p: Vec<f64> = z.iter().map(|v| v.norm_sqr() as f64).collect();
    let max = p.iter().cloned().fold(0.0, f64::max);
    let mean = p.iter().sum::<f64>() / p.len() as f64;
    assert!((max - 1.0).abs() < 1e-5, "key-down power {max}");
    assert!((mean - 0.44).abs() < 0.01, "duty {mean}");
    // 10-90 % rise of a 5 ms raised-cosine edge: 0.59 * 5 ms
    let a: Vec<f64> = p.iter().map(|v| v.sqrt()).collect();
    let up = (1..a.len())
        .find(|&i| a[i] > 0.9 && a[i - 1] <= 0.9)
        .unwrap();
    let start = (0..up).rev().find(|&i| a[i] < 0.1).unwrap();
    let rise = (up - start) as f64 / FS;
    assert!((rise - 0.005 * 0.59).abs() < 2e-4, "rise {rise}");
}

#[test]
fn am_carrier_and_bandwidth() {
    let spec = AmSpec {
        audio_hz: 3000.0,
        ..Default::default()
    };
    let mut z = vec![Complex32::default(); (FS * 10.0) as usize];
    Am::new(FS, 1000.0, &spec, false, 4).add_c(&mut z, 1.0);
    let z = &z[2000..];
    assert!(
        (amp(z, 1000.0) - 1.0).abs() < 0.01,
        "carrier {}",
        amp(z, 1000.0)
    );
    let side = z.iter().map(|v| v.norm_sqr() as f64).sum::<f64>() / z.len() as f64 - 1.0;
    assert!(side > 0.05 && side < 0.1, "sideband power {side}");
    for f in [-2900.0, 4600.0, 5500.0] {
        assert!(amp(z, f) < 3e-3, "{f}: {}", amp(z, f));
    }
    // a tone modulates at 0.3 rms: sidebands at 0.3 * sqrt 2 / 2 of the carrier
    let tone: Vec<f32> = (0..24000)
        .map(|i| (TAU * 400.0 * i as f64 / FS).sin() as f32)
        .collect();
    let mut z = vec![Complex32::default(); 24000];
    Am::with_audio(FS, 0.0, &AmSpec::default(), tone, false, 5).add_c(&mut z, 1.0);
    let z = &z[2000..];
    let r = amp(z, 400.0) / amp(z, 0.0);
    assert!(
        (r - 0.3 * 2f64.sqrt() / 2.0).abs() < 0.01,
        "tone sideband {r}"
    );
}

#[test]
fn am_real_does_not_fold() {
    // carrier 500 Hz, tone 1500 Hz: the upper sideband at 2000 Hz stays,
    // the lower one at -1000 Hz is dropped, not folded to +1000 Hz
    let tone: Vec<f32> = (0..48000)
        .map(|i| (TAU * 1500.0 * i as f64 / FS).sin() as f32)
        .collect();
    let mut x = vec![0.0f32; 48000];
    Am::with_audio(FS, 500.0, &AmSpec::default(), tone, true, 6).add_r(&mut x, 1.0);
    let c: Vec<Complex32> = x[2000..].iter().map(|&v| Complex32::new(v, 0.0)).collect();
    let k = 2f64.sqrt() / 2.0; // real amplitude a * sqrt 2 shows as a / sqrt 2 per side
    assert!((amp(&c, 500.0) / k - 1.0).abs() < 0.01, "carrier");
    let usb = amp(&c, 2000.0) / k;
    assert!(
        (usb - 0.3 * 2f64.sqrt() / 2.0).abs() < 0.01,
        "upper sideband {usb}"
    );
    assert!(amp(&c, 1000.0) / k < 1e-3, "folded {}", amp(&c, 1000.0) / k);
}

#[test]
fn radar_sweep() {
    let mut z = vec![Complex32::default(); (FS * 2.0) as usize];
    Radar::new(FS, 0.0, &RadarSpec::default(), false, 7).add_c(&mut z, 1.0);
    let a: Vec<f64> = z.iter().map(|v| v.norm() as f64).collect();
    let fi: Vec<f64> = z
        .windows(2)
        .map(|w| (w[1] * w[0].conj()).arg() as f64 * FS / TAU)
        .collect();
    let mut slopes: Vec<f64> = (1..fi.len())
        .filter(|&i| a[i - 1] > 0.999 && a[i] > 0.999 && a[i + 1] > 0.999)
        .map(|i| (fi[i] - fi[i - 1]) * FS)
        .collect();
    slopes.sort_by(f64::total_cmp);
    let med = slopes[slopes.len() / 2];
    assert!((med / 1e6 - 1.0).abs() < 0.01, "sweep rate {med}");
    let frac = a.iter().filter(|&&v| v > 0.999).count() as f64 / a.len() as f64;
    assert!((frac - 12000.0 * 0.9 / 20e3).abs() < 0.02, "in band {frac}");
}

#[test]
fn streaming_and_fading() {
    let spec = CwSpec {
        fade_hz: 1.0,
        ..Default::default()
    };
    let mut a = vec![0.0f32; 9000];
    Cw::new(FS, 700.0, &spec, true, 8).add_r(&mut a, 1.0);
    let mut b = vec![0.0f32; 9000];
    let mut s = Cw::new(FS, 700.0, &spec, true, 8);
    for c in b.chunks_mut(101) {
        s.add_r(c, 1.0);
    }
    assert_eq!(a, b);
    let mut a = vec![0.0f32; 9000];
    Am::new(FS, 700.0, &AmSpec::default(), true, 9).add_r(&mut a, 1.0);
    let mut b = vec![0.0f32; 9000];
    let mut s = Am::new(FS, 700.0, &AmSpec::default(), true, 9);
    for c in b.chunks_mut(101) {
        s.add_r(c, 1.0);
    }
    assert_eq!(a, b);
}

#[test]
fn rtty_tones_and_timing() {
    let mut z = vec![Complex32::default(); (FS * 20.0) as usize];
    Rtty::new(FS, 1000.0, &RttySpec::default(), false, 9).add_c(&mut z, 1.0);
    assert!(z.iter().all(|v| (v.norm() - 1.0).abs() < 1e-5));
    let fi: Vec<f64> = z
        .windows(2)
        .map(|w| (w[1] * w[0].conj()).arg() as f64 * FS / TAU)
        .collect();
    let on = fi
        .iter()
        .filter(|f| (*f - 1085.0).abs() < 0.5 || (*f - 915.0).abs() < 0.5)
        .count();
    assert!(on as f64 / fi.len() as f64 > 0.999);
    let unit = FS / 45.45;
    let ch: Vec<usize> = (1..fi.len())
        .filter(|&i| (fi[i] > 1000.0) != (fi[i - 1] > 1000.0))
        .collect();
    for &c in &ch {
        let off = ((c - ch[0]) as f64 / unit * 2.0).rem_euclid(1.0);
        assert!(off.min(1.0 - off) < 2.0 / unit, "change at {c}");
    }
}

#[test]
fn psk31_idle_tones_and_shape() {
    let spec = Psk31Spec {
        idle: true,
        ..Default::default()
    };
    let mut z = vec![Complex32::default(); (FS * 20.0) as usize];
    Psk31::new(FS, 1000.0, &spec, false, 10).add_c(&mut z, 1.0);
    for (f, want) in [(984.375, 0.5), (1015.625, 0.5), (1000.0, 0.0)] {
        assert!((amp(&z, f) - want).abs() < 0.01, "{f}: {}", amp(&z, f));
    }
    let mut z = vec![Complex32::default(); (FS * 20.0) as usize];
    Psk31::new(FS, 1000.0, &Psk31Spec::default(), false, 11).add_c(&mut z, 1.0);
    let p: Vec<f64> = z.iter().map(|v| v.norm_sqr() as f64).collect();
    let mean = p.iter().sum::<f64>() / p.len() as f64;
    assert!(p.iter().all(|&v| v <= 1.0 + 1e-5) && p.iter().any(|&v| v < 1e-4));
    assert!(mean > 0.6 && mean < 0.95, "{mean}");
}
