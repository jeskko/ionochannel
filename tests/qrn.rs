//! QRN against its calibration: Vd in 200 Hz, Vd in other bandwidths
//! (P.372 Fig. 38), mean power, the Gaussian limit.

use ionochannel::Qrn;
use ionochannel::qrn::{VD_GAUSS, params, vd_bandwidth};
use num_complex::Complex32;

/// In-place radix-2 FFT (inverse when `inv`), for the measurements only.
fn fft(a: &mut [num_complex::Complex64], inv: bool) {
    let n = a.len();
    let mut j = 0;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j |= bit;
        if i < j {
            a.swap(i, j);
        }
    }
    let mut len = 2;
    while len <= n {
        let ang = std::f64::consts::TAU / len as f64 * if inv { 1.0 } else { -1.0 };
        let w = num_complex::Complex64::from_polar(1.0, ang);
        for c in a.chunks_mut(len) {
            let mut wk = num_complex::Complex64::new(1.0, 0.0);
            for k in 0..len / 2 {
                let (u, v) = (c[k], c[k + len / 2] * wk);
                c[k] = u + v;
                c[k + len / 2] = u - v;
                wk *= w;
            }
        }
        len <<= 1;
    }
}

/// Vd (dB) of the envelope of `x` in an ideal (rectangular) band of
/// `bw` Hz around 0, the definition the calibration uses.
fn vd(x: &[Complex32], fs: f64, bw: f64) -> f64 {
    let n = x.len().next_power_of_two();
    let mut a: Vec<num_complex::Complex64> = x
        .iter()
        .map(|v| num_complex::Complex64::new(v.re as f64, v.im as f64))
        .chain(std::iter::repeat(num_complex::Complex64::default()))
        .take(n)
        .collect();
    fft(&mut a, false);
    for (k, v) in a.iter_mut().enumerate() {
        let f = if k <= n / 2 { k } else { n - k } as f64 * fs / n as f64;
        if f > bw / 2.0 {
            *v = Default::default();
        }
    }
    fft(&mut a, true);
    let e: Vec<f64> = a[..x.len()].iter().map(|v| v.norm()).collect();
    let m1 = e.iter().sum::<f64>() / e.len() as f64;
    let m2 = e.iter().map(|v| v * v).sum::<f64>() / e.len() as f64;
    20.0 * (m2.sqrt() / m1).log10()
}

fn qrn(fs: f64, vd_db: f64, secs: f64, seed: u64) -> Vec<Complex32> {
    let mut x = vec![Complex32::default(); (fs * secs) as usize];
    Qrn::new(fs, vd_db, seed).add_c(&mut x, 1.0);
    x
}

#[test]
fn vd_in_200_hz_and_other_bandwidths() {
    let fs = 4000.0;
    for (target, secs) in [(3.0, 300.0), (8.0, 600.0)] {
        let x = qrn(fs, target, secs, 7);
        let got = vd(&x, fs, 200.0);
        assert!((got - target).abs() < 0.5, "Vd {target}: {got}");
        let wide = vd(&x, fs, 2000.0);
        let want = vd_bandwidth(target, 2000.0);
        assert!(
            (wide - want).abs() < 1.0,
            "Vd {target} in 2 kHz: {wide} vs {want}"
        );
    }
}

#[test]
fn mean_power_and_gaussian_limit() {
    let x = qrn(8000.0, 4.0, 400.0, 3);
    let p = x.iter().map(|v| v.norm_sqr() as f64).sum::<f64>() / x.len() as f64;
    assert!((p - 1.0).abs() < 0.1, "power {p}");
    let x = qrn(4000.0, VD_GAUSS, 100.0, 3);
    let got = vd(&x, 4000.0, 200.0);
    assert!((got - VD_GAUSS).abs() < 0.05, "Gaussian Vd {got}");
}

#[test]
fn params_and_conversion() {
    let (r4, g4) = params(4.0);
    let (r10, g10) = params(10.0);
    assert!(r4 > r10 && g4 > g10);
    assert!((vd_bandwidth(10.0, 200.0) - 10.0).abs() < 1e-12);
    assert!((vd_bandwidth(10.0, 200e3) - (10.0 + 3.0 * 2.59)).abs() < 1e-9);
    assert_eq!(vd_bandwidth(2.0, 2.0), VD_GAUSS.max(2.0 - 2.0 * 0.87));
}

#[test]
fn streaming_and_real() {
    let mut a = vec![0.0f32; 5000];
    Qrn::new(12000.0, 6.0, 9).add_r(&mut a, 1.0);
    let mut b = vec![0.0f32; 5000];
    let mut q = Qrn::new(12000.0, 6.0, 9);
    for c in b.chunks_mut(333) {
        q.add_r(c, 1.0);
    }
    assert_eq!(a, b);
}
