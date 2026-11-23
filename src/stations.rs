//! Interfering stations: CW, AM broadcast and sweeping (FMCW) radar. Each
//! sits at an offset in the band and can fade on its own (flat Rayleigh,
//! Doppler spread `fade_hz`). The `add_*` methods take the station's
//! power: key-down power for CW, carrier power for AM, in-band power for
//! the radar. For a plain carrier use [`crate::Smps`] with no ripple and a
//! spacing wider than the band. Defaults are illustrative, not measured.
//! The Python `stations` module implements the same models.

use crate::fading::{Component, TapGain};
use crate::receiver::bandpass_taps;
use crate::rng::Rng;
use num_complex::{Complex32, Complex64};
use std::collections::VecDeque;
use std::f64::consts::{PI, TAU};

const MORSE: [(char, &str); 44] = [
    ('A', ".-"),
    ('B', "-..."),
    ('C', "-.-."),
    ('D', "-.."),
    ('E', "."),
    ('F', "..-."),
    ('G', "--."),
    ('H', "...."),
    ('I', ".."),
    ('J', ".---"),
    ('K', "-.-"),
    ('L', ".-.."),
    ('M', "--"),
    ('N', "-."),
    ('O', "---"),
    ('P', ".--."),
    ('Q', "--.-"),
    ('R', ".-."),
    ('S', "..."),
    ('T', "-"),
    ('U', "..-"),
    ('V', "...-"),
    ('W', ".--"),
    ('X', "-..-"),
    ('Y', "-.--"),
    ('Z', "--.."),
    ('0', "-----"),
    ('1', ".----"),
    ('2', "..---"),
    ('3', "...--"),
    ('4', "....-"),
    ('5', "....."),
    ('6', "-...."),
    ('7', "--..."),
    ('8', "---.."),
    ('9', "----."),
    ('/', "-..-."),
    ('?', "..--.."),
    ('=', "-...-"),
    ('.', ".-.-.-"),
    (',', "--..--"),
    (' ', ""),
    ('\n', ""),
    ('\t', ""),
];

/// (key down, length in dit units) for a text: dit 1, dah 3, gaps 1
/// inside a character, 3 between characters, 7 between words, and a word
/// gap after the end.
pub fn morse_units(text: &str) -> Vec<(bool, u32)> {
    let mut out = Vec::new();
    for (w, word) in text.split_whitespace().enumerate() {
        if w > 0 {
            out.push((false, 7));
        }
        let mut first = true;
        for ch in word.chars().flat_map(char::to_uppercase) {
            let Some(code) = MORSE.iter().find(|(c, _)| *c == ch).map(|(_, m)| *m) else {
                continue;
            };
            if code.is_empty() {
                continue;
            }
            if !first {
                out.push((false, 3));
            }
            first = false;
            for (e, sym) in code.chars().enumerate() {
                if e > 0 {
                    out.push((false, 1));
                }
                out.push((true, if sym == '.' { 1 } else { 3 }));
            }
        }
    }
    out.push((false, 7));
    out
}

const GROUP_CHARS: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";

/// Random groups of 2-6 characters.
pub fn random_text(rng: &mut Rng, words: usize) -> String {
    (0..words)
        .map(|_| {
            let len = 2 + (rng.next_u64() % 5) as usize;
            (0..len)
                .map(|_| GROUP_CHARS[(rng.next_u64() % GROUP_CHARS.len() as u64) as usize] as char)
                .collect::<String>()
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// Moves a complex envelope to its offset, fades it, and makes it real
/// if asked (keeping positive frequencies only when `one_sided`).
struct Place {
    step: f64,
    n: u64,
    phase0: f64,
    fade: Option<TapGain>,
    real: bool,
    one_sided: Option<(Vec<Complex32>, Vec<Complex32>)>,
}

impl Place {
    fn new(
        fs: f64,
        offset_hz: f64,
        fade_hz: f64,
        real: bool,
        one_sided: bool,
        rng: &mut Rng,
    ) -> Self {
        let fade = (fade_hz > 0.0).then(|| {
            TapGain::new(
                &[Component::Scatter {
                    power: 1.0,
                    spread_hz: fade_hz,
                    shift_hz: 0.0,
                }],
                fs,
                &mut rng.fork(),
            )
        });
        let one_sided = (real && one_sided).then(|| {
            let tr = fs / 100.0;
            let t = bandpass_taps(fs, tr / 2.0, fs / 2.0 - tr / 2.0, tr);
            let h = vec![Complex32::default(); t.len() - 1];
            (t, h)
        });
        Place {
            step: offset_hz / fs,
            n: 0,
            phase0: rng.uniform(),
            fade,
            real,
            one_sided,
        }
    }

    fn put(&mut self, e: Complex64) -> Complex32 {
        let cyc = (self.step * self.n as f64 + self.phase0).rem_euclid(1.0);
        self.n += 1;
        let mut z = e * Complex64::from_polar(1.0, TAU * cyc);
        if let Some(g) = self.fade.as_mut() {
            z *= g.sample();
        }
        let z = Complex32::new(z.re as f32, z.im as f32);
        if let Some((t, h)) = self.one_sided.as_mut() {
            h.push(z);
            let y: Complex32 = h.iter().rev().zip(t.iter()).map(|(v, c)| v * c).sum();
            h.remove(0);
            return y;
        }
        z
    }

    fn real(&self) -> bool {
        self.real
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct CwSpec {
    pub wpm: f64,
    pub rise_s: f64,
    pub fade_hz: f64,
    /// Fixed text, repeated; `None` for random groups.
    pub text: Option<String>,
}

impl Default for CwSpec {
    fn default() -> Self {
        CwSpec {
            wpm: 20.0,
            rise_s: 0.005,
            fade_hz: 0.0,
            text: None,
        }
    }
}

/// A CW station; key-down power 1.
pub struct Cw {
    rng: Rng,
    spec: CwSpec,
    unit: f64,
    seq: Vec<(bool, u32)>,
    el: usize,
    /// End of the current element, in samples from the sequence start.
    end: f64,
    pos: f64,
    n: i64,
    kernel: Vec<f64>,
    keys: VecDeque<f64>,
    place: Place,
}

impl Cw {
    pub fn new(fs: f64, offset_hz: f64, spec: &CwSpec, real: bool, seed: u64) -> Self {
        let mut rng = Rng::new(seed);
        let rise = (spec.rise_s * fs).round().max(1.0) as usize;
        let mut kernel: Vec<f64> = (0..rise)
            .map(|i| {
                let e = |x: f64| 0.5 - 0.5 * (PI * x / rise as f64).cos();
                e(i as f64 + 0.5) - if i == 0 { 0.0 } else { e(i as f64 - 0.5) }
            })
            .collect();
        let s: f64 = kernel.iter().sum();
        kernel.iter_mut().for_each(|k| *k /= s);
        let place = Place::new(fs, offset_hz, spec.fade_hz, real, false, &mut rng);
        let seq = morse_units(&match &spec.text {
            Some(t) => t.clone(),
            None => random_text(&mut rng, 20),
        });
        let unit = 1.2 / spec.wpm * fs;
        let total: u32 = seq.iter().map(|&(_, u)| u).sum();
        let start = rng.uniform() * total as f64 * unit;
        let mut c = Cw {
            unit,
            end: seq[0].1 as f64 * unit,
            seq,
            el: 0,
            pos: start,
            n: 0,
            keys: std::iter::repeat_n(0.0, rise).collect(),
            kernel,
            place,
            spec: spec.clone(),
            rng,
        };
        c.n = start.floor() as i64;
        c.pos = 0.0;
        c
    }

    fn key(&mut self) -> f64 {
        // the element covering sample n: boundaries at rounded unit times
        while self.n >= self.end.round() as i64 {
            self.pos = self.end;
            self.el += 1;
            if self.el == self.seq.len() {
                self.el = 0;
                if self.spec.text.is_none() {
                    self.seq = morse_units(&random_text(&mut self.rng, 20));
                }
            }
            self.end = self.pos + self.seq[self.el].1 as f64 * self.unit;
        }
        self.n += 1;
        if self.seq[self.el].0 { 1.0 } else { 0.0 }
    }

    fn next(&mut self) -> Complex32 {
        let k = self.key();
        self.keys.push_front(k);
        self.keys.pop_back();
        let e: f64 = self.keys.iter().zip(&self.kernel).map(|(a, b)| a * b).sum();
        self.place.put(Complex64::new(e.clamp(0.0, 1.0), 0.0))
    }

    pub fn add_c(&mut self, x: &mut [Complex32], power: f64) {
        let a = power.sqrt() as f32;
        for v in x {
            *v += self.next() * a;
        }
    }

    pub fn add_r(&mut self, x: &mut [f32], power: f64) {
        assert!(self.place.real(), "built for complex signals");
        let a = (2.0 * power).sqrt() as f32;
        for v in x {
            *v += self.next().re * a;
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct AmSpec {
    /// Peak modulation of the normalised audio.
    pub depth: f64,
    /// Transmitted audio bandwidth (sidebands +-).
    pub audio_hz: f64,
    pub fade_hz: f64,
}

impl Default for AmSpec {
    fn default() -> Self {
        AmSpec {
            depth: 1.0,
            audio_hz: 4500.0,
            fade_hz: 0.0,
        }
    }
}

/// Speech-like noise: white noise through a one-pole high-pass at 200 Hz
/// and low-pass at 500 Hz, times a syllabic envelope (white noise through
/// two one-pole low-passes at 3 Hz).
struct SpeechLike {
    rng: Rng,
    a200: f64,
    a500: f64,
    a3: f64,
    hp: f64,
    lp: f64,
    s1: f64,
    s2: f64,
}

impl SpeechLike {
    fn new(fs: f64, rng: Rng) -> Self {
        let a = |f: f64| (-TAU * f / fs).exp();
        SpeechLike {
            rng,
            a200: a(200.0),
            a500: a(500.0),
            a3: a(3.0),
            hp: 0.0,
            lp: 0.0,
            s1: 0.0,
            s2: 0.0,
        }
    }

    fn next(&mut self) -> f64 {
        let w = self.rng.normal();
        self.hp = self.a200 * self.hp + (1.0 - self.a200) * w; // low-passed copy
        let h = w - self.hp;
        self.lp = self.a500 * self.lp + (1.0 - self.a500) * h;
        let s = self.rng.normal();
        self.s1 = self.a3 * self.s1 + (1.0 - self.a3) * s;
        self.s2 = self.a3 * self.s2 + (1.0 - self.a3) * self.s1;
        self.lp * self.s2
    }
}

enum Audio {
    Synth(SpeechLike),
    Samples(Vec<f32>, usize),
}

/// An AM station; carrier power 1. The audio is scaled to 0.3 rms (the
/// synthetic audio from a 2 s estimate) and clipped at +-1, then the
/// sidebands are limited to `audio_hz`.
pub struct Am {
    audio: Audio,
    scale: f64,
    depth: f64,
    fir: Option<(Vec<Complex32>, VecDeque<Complex64>)>,
    place: Place,
}

impl Am {
    /// With speech-like audio.
    pub fn new(fs: f64, offset_hz: f64, spec: &AmSpec, real: bool, seed: u64) -> Self {
        let mut rng = Rng::new(seed);
        let mut s = SpeechLike::new(fs, rng.fork());
        let warm = (2.0 * fs) as usize;
        let ms: f64 = (0..warm).map(|_| s.next().powi(2)).sum::<f64>() / warm as f64;
        Self::build(fs, offset_hz, spec, real, rng, Audio::Synth(s), ms.sqrt())
    }

    /// With your own audio (at `fs`), looped.
    pub fn with_audio(
        fs: f64,
        offset_hz: f64,
        spec: &AmSpec,
        audio: Vec<f32>,
        real: bool,
        seed: u64,
    ) -> Self {
        assert!(!audio.is_empty(), "empty audio");
        let rms =
            (audio.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / audio.len() as f64).sqrt();
        Self::build(
            fs,
            offset_hz,
            spec,
            real,
            Rng::new(seed),
            Audio::Samples(audio, 0),
            rms.max(1e-30),
        )
    }

    fn build(
        fs: f64,
        offset_hz: f64,
        spec: &AmSpec,
        real: bool,
        mut rng: Rng,
        audio: Audio,
        rms: f64,
    ) -> Self {
        let fir = (spec.audio_hz < fs / 2.0).then(|| {
            let t = bandpass_taps(
                fs,
                -spec.audio_hz,
                spec.audio_hz,
                (spec.audio_hz / 4.0).min(500.0),
            );
            let h = std::iter::repeat_n(Complex64::default(), t.len()).collect();
            (t, h)
        });
        Am {
            audio,
            scale: 0.3 / rms,
            depth: spec.depth,
            fir,
            place: Place::new(fs, offset_hz, spec.fade_hz, real, true, &mut rng),
        }
    }

    fn next(&mut self) -> Complex32 {
        let a = match &mut self.audio {
            Audio::Synth(s) => s.next(),
            Audio::Samples(v, i) => {
                let a = v[*i] as f64;
                *i = (*i + 1) % v.len();
                a
            }
        };
        let m = (a * self.scale).clamp(-1.0, 1.0);
        let mut e = Complex64::new(1.0 + self.depth * m, 0.0);
        if let Some((t, h)) = self.fir.as_mut() {
            h.push_front(e);
            h.pop_back();
            e = h
                .iter()
                .zip(t.iter())
                .map(|(v, c)| v * Complex64::new(c.re as f64, c.im as f64))
                .sum();
        }
        self.place.put(e)
    }

    pub fn add_c(&mut self, x: &mut [Complex32], power: f64) {
        let a = power.sqrt() as f32;
        for v in x {
            *v += self.next() * a;
        }
    }

    pub fn add_r(&mut self, x: &mut [f32], power: f64) {
        assert!(self.place.real(), "built for complex signals");
        let a = (2.0 * power).sqrt() as f32;
        for v in x {
            *v += self.next().re * a;
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct RadarSpec {
    pub sweep_hz: f64,
    /// Sweeps per second.
    pub rate_hz: f64,
    /// Taper at the band edges, as a fraction of the band.
    pub edge: f64,
    pub fade_hz: f64,
}

impl Default for RadarSpec {
    fn default() -> Self {
        RadarSpec {
            sweep_hz: 20e3,
            rate_hz: 50.0,
            edge: 0.05,
            fade_hz: 0.0,
        }
    }
}

/// An FMCW radar: linear up-sweeps centred on the offset; power 1 while
/// in the band, tapered off at its edges so it doesn't alias.
pub struct Radar {
    fs: f64,
    spec: RadarSpec,
    offset: f64,
    t0: f64,
    ph0: f64,
    n: u64,
    lo: f64,
    hi: f64,
    fade: Option<TapGain>,
    real: bool,
}

impl Radar {
    pub fn new(fs: f64, offset_hz: f64, spec: &RadarSpec, real: bool, seed: u64) -> Self {
        let mut rng = Rng::new(seed);
        let (lo, hi) = if real {
            (0.0, fs / 2.0)
        } else {
            (-fs / 2.0, fs / 2.0)
        };
        Radar {
            fs,
            offset: offset_hz,
            t0: rng.uniform() / spec.rate_hz,
            ph0: rng.uniform(),
            n: 0,
            lo,
            hi,
            fade: (spec.fade_hz > 0.0).then(|| {
                TapGain::new(
                    &[Component::Scatter {
                        power: 1.0,
                        spread_hz: spec.fade_hz,
                        shift_hz: 0.0,
                    }],
                    fs,
                    &mut rng.fork(),
                )
            }),
            real,
            spec: spec.clone(),
        }
    }

    fn next(&mut self) -> Complex32 {
        let s = &self.spec;
        let period = 1.0 / s.rate_hz;
        let tau = (self.n as f64 / self.fs + self.t0).rem_euclid(period);
        self.n += 1;
        let k = s.sweep_hz * s.rate_hz;
        let f0 = self.offset - s.sweep_hz / 2.0;
        let f = f0 + k * tau;
        let e = s.edge * (self.hi - self.lo);
        let w = ((f - self.lo).min(self.hi - f) / e).clamp(0.0, 1.0);
        let w = 0.5 - 0.5 * (PI * w).cos();
        let cyc = (f0 * tau + 0.5 * k * tau * tau + self.ph0).rem_euclid(1.0);
        let mut z = Complex64::from_polar(w, TAU * cyc);
        if let Some(g) = self.fade.as_mut() {
            z *= g.sample();
        }
        Complex32::new(z.re as f32, z.im as f32)
    }

    pub fn add_c(&mut self, x: &mut [Complex32], power: f64) {
        let a = power.sqrt() as f32;
        for v in x {
            *v += self.next() * a;
        }
    }

    pub fn add_r(&mut self, x: &mut [f32], power: f64) {
        assert!(self.real, "built for complex signals");
        let a = (2.0 * power).sqrt() as f32;
        for v in x {
            *v += self.next().re * a;
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct RttySpec {
    pub baud: f64,
    /// Mark is the upper tone, offset + shift/2.
    pub shift_hz: f64,
    pub stop_bits: f64,
    pub fade_hz: f64,
}

impl Default for RttySpec {
    fn default() -> Self {
        RttySpec {
            baud: 45.45,
            shift_hz: 170.0,
            stop_bits: 1.5,
            fade_hz: 0.0,
        }
    }
}

/// Amateur RTTY: continuous-phase FSK; characters of a start bit (space),
/// 5 random data bits and the stop bits (mark), back to back. Power 1.
pub struct Rtty {
    rng: Rng,
    spec: RttySpec,
    unit: f64,
    /// Bits of the current character as (mark, length in units).
    bits: Vec<(bool, f64)>,
    i: usize,
    end: f64,
    n: f64,
    cycles: f64,
    fs: f64,
    place: Place,
}

impl Rtty {
    pub fn new(fs: f64, offset_hz: f64, spec: &RttySpec, real: bool, seed: u64) -> Self {
        let mut rng = Rng::new(seed);
        let place = Place::new(fs, offset_hz, spec.fade_hz, real, false, &mut rng);
        let unit = fs / spec.baud;
        let mut r = Rtty {
            unit,
            bits: Vec::new(),
            i: 0,
            end: 0.0,
            n: 0.0,
            cycles: 0.0,
            fs,
            place,
            spec: spec.clone(),
            rng,
        };
        r.next_char();
        // start part-way into a character
        r.n = (r.rng.uniform() * (6.0 + spec.stop_bits) * unit).floor();
        r
    }

    fn next_char(&mut self) {
        self.bits.clear();
        self.bits.push((false, 1.0));
        for _ in 0..5 {
            self.bits.push((self.rng.next_u64() & 1 == 1, 1.0));
        }
        self.bits.push((true, self.spec.stop_bits));
        self.i = 0;
        self.end += self.bits[0].1 * self.unit;
    }

    fn next(&mut self) -> Complex32 {
        while self.n >= self.end.round() {
            self.i += 1;
            if self.i == self.bits.len() {
                self.next_char();
            } else {
                self.end += self.bits[self.i].1 * self.unit;
            }
        }
        self.n += 1.0;
        let z = Complex64::from_polar(1.0, TAU * self.cycles);
        let f = if self.bits[self.i].0 { 0.5 } else { -0.5 } * self.spec.shift_hz;
        self.cycles = (self.cycles + f / self.fs).rem_euclid(1.0);
        self.place.put(z)
    }

    pub fn add_c(&mut self, x: &mut [Complex32], power: f64) {
        let a = power.sqrt() as f32;
        for v in x {
            *v += self.next() * a;
        }
    }

    pub fn add_r(&mut self, x: &mut [f32], power: f64) {
        assert!(self.place.real(), "built for complex signals");
        let a = (2.0 * power).sqrt() as f32;
        for v in x {
            *v += self.next().re * a;
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Psk31Spec {
    pub baud: f64,
    /// Zeros only: two tones at +-baud/2.
    pub idle: bool,
    pub fade_hz: f64,
}

impl Default for Psk31Spec {
    fn default() -> Self {
        Psk31Spec {
            baud: 31.25,
            idle: false,
            fade_hz: 0.0,
        }
    }
}

/// PSK31: BPSK; a 0 bit reverses the phase through a cosine-shaped zero,
/// a 1 bit holds it. Random Varicode-like words (start and end with 1, no
/// 00 inside) separated by 00. Peak power 1.
pub struct Psk31 {
    rng: Rng,
    idle: bool,
    period: f64,
    t: f64,
    k: i64,
    bit: u8,
    prev: f64,
    sign: f64,
    queue: VecDeque<u8>,
    place: Place,
}

impl Psk31 {
    pub fn new(fs: f64, offset_hz: f64, spec: &Psk31Spec, real: bool, seed: u64) -> Self {
        let mut rng = Rng::new(seed);
        let place = Place::new(fs, offset_hz, spec.fade_hz, real, false, &mut rng);
        let period = fs / spec.baud;
        let t = rng.uniform() * period;
        Psk31 {
            rng,
            idle: spec.idle,
            period,
            t,
            k: -1,
            bit: 1,
            prev: 1.0,
            sign: 1.0,
            queue: VecDeque::new(),
            place,
        }
    }

    fn next_bit(&mut self) -> u8 {
        if self.idle {
            return 0;
        }
        if self.queue.is_empty() {
            let len = 1 + (self.rng.next_u64() % 10) as usize;
            let mut w = vec![1u8];
            for _ in 0..len.saturating_sub(2) {
                let b = if *w.last().unwrap() == 0 {
                    1
                } else {
                    (self.rng.next_u64() & 1) as u8
                };
                w.push(b);
            }
            if len > 1 {
                w.push(1);
            }
            w.extend([0, 0]);
            self.queue.extend(w);
        }
        self.queue.pop_front().unwrap()
    }

    fn next(&mut self) -> Complex32 {
        let k = (self.t / self.period).floor() as i64;
        while self.k < k {
            self.k += 1;
            self.bit = self.next_bit();
            self.prev = self.sign;
            if self.bit == 0 {
                self.sign = -self.sign;
            }
        }
        let u = (self.t - k as f64 * self.period) / self.period;
        self.t += 1.0;
        let e = if self.bit == 0 {
            self.prev * (PI * u).cos()
        } else {
            self.sign
        };
        self.place.put(Complex64::new(e, 0.0))
    }

    pub fn add_c(&mut self, x: &mut [Complex32], power: f64) {
        let a = power.sqrt() as f32;
        for v in x {
            *v += self.next() * a;
        }
    }

    pub fn add_r(&mut self, x: &mut [f32], power: f64) {
        assert!(self.place.real(), "built for complex signals");
        let a = (2.0 * power).sqrt() as f32;
        for v in x {
            *v += self.next().re * a;
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct PopulationSpec {
    /// Arrivals per second in the band. No default: it depends on band,
    /// time and place (Mendieta-Otero et al.'s ARRL Field Day 2011 figure,
    /// 6.68/s, is for all amateur bands at once).
    pub rate_per_s: f64,
    pub mean_duration_s: f64,
    pub level_db: f64,
    pub scale_db: f64,
    /// Band (Hz); `None`: +-0.45 fs for I/Q, 100 Hz to 0.45 fs for audio.
    pub band_hz: Option<(f64, f64)>,
}

impl PopulationSpec {
    /// The paper's other values: mean duration 10 s, scale 11.8 dB.
    pub fn new(rate_per_s: f64) -> Self {
        PopulationSpec {
            rate_per_s,
            mean_duration_s: 10.0,
            level_db: 0.0,
            scale_db: 11.8,
            band_hz: None,
        }
    }
}

struct PopStation {
    start: f64,
    end: f64,
    dot: f64,
    amp: f64,
    step: f64,
    phase0: f64,
    kernel: Vec<f64>,
    keys: VecDeque<f64>,
}

/// CW stations coming and going (Mendieta-Otero, Pérez-Álvarez &
/// Pérez-Díaz, IEEE TEMC 2014): Poisson arrivals, exponential durations,
/// frequencies uniform in the band, levels logistic in dB (scale
/// 1/|B_k|); each sends "PARIS" with dot = duration / 331 and
/// raised-cosine edges of dot / 10. Each station's key-down power is
/// 10^(level/10); `add_*` scale it all by `power`.
pub struct Population {
    fs: f64,
    spec: PopulationSpec,
    band: (f64, f64),
    rng: Rng,
    active: Vec<PopStation>,
    /// Next arrival, in samples.
    next: f64,
    n: f64,
    paris: Vec<bool>,
    real: bool,
}

impl Population {
    pub fn new(fs: f64, spec: &PopulationSpec, real: bool, seed: u64) -> Self {
        let band = spec.band_hz.unwrap_or(if real {
            (100.0, 0.45 * fs)
        } else {
            (-0.45 * fs, 0.45 * fs)
        });
        let paris = morse_units("PARIS")
            .iter()
            .flat_map(|&(d, u)| std::iter::repeat_n(d, u as usize))
            .collect();
        let mut p = Population {
            fs,
            spec: spec.clone(),
            band,
            rng: Rng::new(seed),
            active: Vec::new(),
            next: f64::INFINITY,
            n: 0.0,
            paris,
            real,
        };
        if spec.rate_per_s > 0.0 {
            // stations already on: arrivals over the last 10 mean durations
            let span = 10.0 * spec.mean_duration_s;
            let mean = spec.rate_per_s * span;
            let k = p.poisson(mean);
            for _ in 0..k {
                let start = -p.rng.uniform() * span * fs;
                p.add(start);
            }
            p.next = p.exp_gap();
        }
        p
    }

    fn poisson(&mut self, mean: f64) -> usize {
        // sum of exponential gaps; fine for the means used here
        let (mut k, mut t) = (0, 0.0);
        loop {
            t += -(1.0 - self.rng.uniform()).ln();
            if t > mean {
                return k;
            }
            k += 1;
        }
    }

    fn exp_gap(&mut self) -> f64 {
        -(1.0 - self.rng.uniform()).ln() / self.spec.rate_per_s * self.fs
    }

    fn add(&mut self, start: f64) {
        let dur = -(1.0 - self.rng.uniform()).ln() * self.spec.mean_duration_s * self.fs;
        if start + dur <= 0.0 {
            return;
        }
        let f = self.band.0 + (self.band.1 - self.band.0) * self.rng.uniform();
        let u = self.rng.uniform().clamp(1e-12, 1.0 - 1e-12);
        let level = self.spec.level_db + self.spec.scale_db * (u / (1.0 - u)).ln();
        let dot = dur / 331.0;
        let rise = ((dot / 10.0).round() as usize).max(1);
        let e = |x: f64| 0.5 - 0.5 * (PI * x / rise as f64).cos();
        let mut kernel: Vec<f64> = (0..rise)
            .map(|i| e(i as f64 + 0.5) - if i == 0 { 0.0 } else { e(i as f64 - 0.5) })
            .collect();
        let s: f64 = kernel.iter().sum();
        kernel.iter_mut().for_each(|k| *k /= s);
        self.active.push(PopStation {
            start,
            end: start + dur,
            dot,
            amp: 10f64.powf(level / 20.0),
            step: f / self.fs,
            phase0: self.rng.uniform(),
            keys: std::iter::repeat_n(0.0, rise).collect(),
            kernel,
        });
    }

    fn next(&mut self) -> Complex32 {
        while self.next <= self.n {
            let at = self.next;
            self.add(at);
            self.next += self.exp_gap();
        }
        let n = self.n;
        let mut z = Complex64::default();
        let paris = &self.paris;
        for s in &mut self.active {
            let key = if n >= s.start && n < s.end {
                let u = ((n - s.start) / s.dot).floor() as usize % paris.len();
                if paris[u] { 1.0 } else { 0.0 }
            } else {
                0.0
            };
            s.keys.push_front(key);
            s.keys.pop_back();
            let e: f64 = s.keys.iter().zip(&s.kernel).map(|(a, b)| a * b).sum();
            if e > 0.0 {
                let cyc = (s.step * n + s.phase0).rem_euclid(1.0);
                z += Complex64::from_polar(s.amp * e.min(1.0), TAU * cyc);
            }
        }
        self.active
            .retain(|s| n < s.end + s.kernel.len() as f64 + 1.0);
        self.n += 1.0;
        Complex32::new(z.re as f32, z.im as f32)
    }

    pub fn add_c(&mut self, x: &mut [Complex32], power: f64) {
        let a = power.sqrt() as f32;
        for v in x {
            *v += self.next() * a;
        }
    }

    pub fn add_r(&mut self, x: &mut [f32], power: f64) {
        assert!(self.real, "built for complex signals");
        let a = (2.0 * power).sqrt() as f32;
        for v in x {
            *v += self.next().re * a;
        }
    }

    /// Stations on now (for tests and logs).
    pub fn active(&self) -> usize {
        self.active
            .iter()
            .filter(|s| self.n >= s.start && self.n < s.end)
            .count()
    }
}
