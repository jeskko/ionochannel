//! A small, fixed random generator (xoshiro256++ seeded by SplitMix64) so a
//! seed draws the same numbers on every platform and in every version,
//! whatever happens to other crates' generators. (Outputs built from them
//! can still differ in the last bits between platforms' maths libraries.)

use num_complex::Complex32;

#[derive(Clone, Debug)]
pub struct Rng {
    s: [u64; 4],
    spare: Option<f64>,
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        let mut z = seed;
        let mut next = || {
            z = z.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut x = z;
            x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            x ^ (x >> 31)
        };
        Rng {
            s: [next(), next(), next(), next()],
            spare: None,
        }
    }

    /// An independent generator for a sub-part (a path, a component).
    pub fn fork(&mut self) -> Self {
        Rng::new(self.next_u64())
    }

    pub fn next_u64(&mut self) -> u64 {
        let s = &mut self.s;
        let r = s[0].wrapping_add(s[3]).rotate_left(23).wrapping_add(s[0]);
        let t = s[1] << 17;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = s[3].rotate_left(45);
        r
    }

    /// Uniform in [0, 1).
    pub fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Standard normal (Box-Muller, both values used).
    pub fn normal(&mut self) -> f64 {
        if let Some(v) = self.spare.take() {
            return v;
        }
        let u1 = 1.0 - self.uniform(); // (0, 1]
        let u2 = self.uniform();
        let r = (-2.0 * u1.ln()).sqrt();
        let (s, c) = (std::f64::consts::TAU * u2).sin_cos();
        self.spare = Some(r * s);
        r * c
    }

    /// Circular complex normal with E|z|^2 = 1.
    pub fn complex_normal(&mut self) -> Complex32 {
        let k = std::f64::consts::FRAC_1_SQRT_2;
        Complex32::new((self.normal() * k) as f32, (self.normal() * k) as f32)
    }
}
