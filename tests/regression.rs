//! Seeds give the same channel in every version: fingerprints of a few
//! outputs, recorded before the channel learned time-varying paths.

use ionochannel::{Channel, PathSpec, RealChannel, presets};
use num_complex::Complex32;

fn fingerprint(v: &[f32]) -> u64 {
    // FNV-1a over the f32 bit patterns
    v.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, x| {
        (h ^ x.to_bits() as u64).wrapping_mul(0x0100_0000_01b3)
    })
}

fn outputs() -> Vec<u64> {
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
    assert_eq!(
        outputs(),
        vec![
            12376255947366817407,
            4949537275524690498,
            2320915634778736406,
            14169633584398290449,
            2878029206329132387,
            1330066206049277723,
            5539685582080831929
        ]
    );
}
