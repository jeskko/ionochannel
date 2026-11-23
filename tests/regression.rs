//! Seeds give the same channel in every version: fingerprints of a few
//! outputs, recorded before the channel learned time-varying paths.

use ionochannel::{Channel, PathSpec, RealChannel, presets};
use num_complex::Complex32;

/// Samples at a few fixed indices: enough to catch a changed realisation,
/// compared with a tolerance because libm's sin/exp/ln may differ in the
/// last bits between platforms.
fn fingerprint(v: &[f32]) -> Vec<f32> {
    [0usize, 1, 997, 4096, 12345, 19999]
        .iter()
        .map(|&i| v[i])
        .collect()
}

fn outputs() -> Vec<Vec<f32>> {
    let mut out = Vec::new();
    for (name, seed) in [("poor", 1u64), ("high-disturbed", 7), ("flutter", 3)] {
        let p = presets::get(name).unwrap().paths;
        let mut x: Vec<Complex32> = (0..20000)
            .map(|i| Complex32::new((i as f32 * 0.01).sin(), (i as f32 * 0.013).cos()))
            .collect();
        Channel::new(12000.0, &p, seed).process(&mut x);
        out.push(fingerprint(
            &x.iter().flat_map(|z| [z.re, z.im]).collect::<Vec<_>>(),
        ));
        let mut r: Vec<f32> = (0..20000).map(|i| (i as f32 * 0.37).sin()).collect();
        RealChannel::new(12000.0, &p, seed).process(&mut r);
        out.push(fingerprint(&r));
    }
    let paths = [
        PathSpec::scatter(0.0, 0.6, 2.0, 1.0),
        PathSpec::specular(0.37e-3, 0.4, -2.0),
    ];
    let mut x = vec![Complex32::new(1.0, 0.0); 20000];
    Channel::new(12000.0, &paths, 11).process(&mut x);
    out.push(fingerprint(
        &x.iter().flat_map(|z| [z.re, z.im]).collect::<Vec<_>>(),
    ));
    out
}

#[test]
fn seeds_give_the_same_channel() {
    let want: Vec<Vec<f32>> = vec![
        vec![
            0.105287395,
            -0.34848228,
            0.16150616,
            -0.36172625,
            0.118507996,
            0.031244686,
        ],
        vec![
            0.0,
            2.6536542e-21,
            -0.45191348,
            -0.17724076,
            1.0520627,
            -1.2223967,
        ],
        vec![
            0.06933959, -0.4971064, 0.737725, 1.8048198, -0.7921318, 1.6304324,
        ],
        vec![
            0.0,
            1.6490799e-21,
            -0.3483711,
            0.98557913,
            0.81543773,
            -0.21378134,
        ],
        vec![
            -0.16945706,
            -0.13667005,
            -0.7634333,
            -0.7627727,
            -0.012247169,
            0.25237677,
        ],
        vec![
            0.0,
            -4.3205804e-21,
            -0.19572341,
            0.89382493,
            -0.57720566,
            1.5112538,
        ],
        vec![
            2.7102384e-34,
            -5.099702e-35,
            0.123764545,
            0.8143394,
            1.2231164,
            -0.19219643,
        ],
    ];
    let got = outputs();
    assert_eq!(got.len(), want.len());
    for (g, w) in got.iter().zip(&want) {
        for (a, b) in g.iter().zip(w) {
            assert!((a - b).abs() <= 1e-5 * (1.0 + b.abs()), "{a} vs {b}");
        }
    }
}

#[test]
#[ignore]
fn print_values() {
    println!("{:?}", outputs());
}
