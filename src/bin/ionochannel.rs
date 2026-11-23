//! ionochannel: put a recording through an HF channel.
//!
//!   ionochannel [--preset NAME | --path D,S[,F[,P]]... | --specular D,P[,F]...]
//!               [--snr DB [--bw HZ]] [--seed N] [--rate HZ] IN OUT
//!   ionochannel --list
//!
//! IN/OUT: .wav (mono: real audio; two channels: I/Q) or .cf32 (complex
//! float32, needs --rate). The output lines up with the input (the
//! filters' latency taken out) and keeps its format.

use ionochannel::{Channel, Component, PathSpec, RealChannel, Rng, noise, presets, wav};
use num_complex::Complex32;
use std::process::exit;

const USAGE: &str =
    "usage: ionochannel [options] IN OUT    (IN/OUT: .wav mono or I/Q, or .cf32 with --rate)
  --preset NAME           a named channel (--list shows them)
  --path D,S[,F[,P]]      a Rayleigh path: delay ms, spread Hz (2 sigma), shift Hz, power
                          (repeat for more paths; powers default to equal shares)
  --specular D,P[,F]      a steady path: delay ms, power, shift Hz
  --snr DB                add white noise: SNR of the input's mean power in --bw
  --bw HZ                 SNR reference bandwidth [2500]
  --seed N                fading and noise realisation [1]
  --rate HZ               sample rate of a .cf32 input
  --list                  list the presets";

fn die(m: &str) -> ! {
    eprintln!("ionochannel: {m}");
    exit(2)
}

fn nums(s: &str, min: usize, max: usize, what: &str) -> Vec<f64> {
    let v: Vec<f64> = s
        .split(',')
        .map(|t| {
            t.trim()
                .parse()
                .unwrap_or_else(|_| die(&format!("{what}: bad number '{t}'")))
        })
        .collect();
    if v.len() < min || v.len() > max {
        die(&format!(
            "{what}: expected {min}-{max} comma-separated numbers"
        ));
    }
    v
}

fn main() {
    let mut a = std::env::args().skip(1);
    let (mut preset, mut paths, mut snr, mut bw, mut seed, mut rate, mut files) =
        (None, Vec::new(), None, 2500.0, 1u64, None, Vec::new());
    let mut explicit_power = false;
    while let Some(arg) = a.next() {
        let mut val = |n: &str| {
            a.next()
                .unwrap_or_else(|| die(&format!("{n} needs a value")))
        };
        match arg.as_str() {
            "-h" | "--help" => {
                println!("{USAGE}");
                return;
            }
            "--list" => {
                for p in presets::all() {
                    let d: Vec<String> = p
                        .paths
                        .iter()
                        .map(|q| {
                            let c: Vec<String> = q
                                .components
                                .iter()
                                .map(|c| match c {
                                    Component::Scatter {
                                        power,
                                        spread_hz,
                                        shift_hz,
                                    } => format!(
                                        "{power}x{spread_hz} Hz{}",
                                        if *shift_hz != 0.0 {
                                            format!("@{shift_hz:+}")
                                        } else {
                                            String::new()
                                        }
                                    ),
                                    Component::Specular { power, shift_hz } => format!(
                                        "{power}x steady{}",
                                        if *shift_hz != 0.0 {
                                            format!("@{shift_hz:+}")
                                        } else {
                                            String::new()
                                        }
                                    ),
                                })
                                .collect();
                            format!("{} ms: {}", q.delay_s * 1e3, c.join(" + "))
                        })
                        .collect();
                    println!("{:20} {}\n{:20} ({})", p.name, d.join(" | "), "", p.source);
                }
                return;
            }
            "--preset" => preset = Some(val("--preset")),
            "--path" => {
                let v = nums(&val("--path"), 2, 4, "--path");
                explicit_power |= v.len() == 4;
                paths.push(PathSpec::scatter(
                    v[0] * 1e-3,
                    v.get(3).copied().unwrap_or(f64::NAN),
                    v[1],
                    v.get(2).copied().unwrap_or(0.0),
                ));
            }
            "--specular" => {
                let v = nums(&val("--specular"), 2, 3, "--specular");
                explicit_power = true;
                paths.push(PathSpec::specular(
                    v[0] * 1e-3,
                    v[1],
                    v.get(2).copied().unwrap_or(0.0),
                ));
            }
            "--snr" => snr = Some(nums(&val("--snr"), 1, 1, "--snr")[0]),
            "--bw" => bw = nums(&val("--bw"), 1, 1, "--bw")[0],
            "--seed" => {
                seed = val("--seed")
                    .parse()
                    .unwrap_or_else(|_| die("--seed: not a number"))
            }
            "--rate" => rate = Some(nums(&val("--rate"), 1, 1, "--rate")[0]),
            s if s.starts_with("--") => die(&format!("unknown option {s}\n{USAGE}")),
            _ => files.push(arg),
        }
    }
    if files.len() != 2 {
        die(USAGE);
    }
    let paths = match (preset, paths.is_empty()) {
        (Some(_), false) => die("--preset or --path/--specular, not both"),
        (Some(n), true) => {
            presets::get(&n)
                .unwrap_or_else(|| die(&format!("no preset '{n}' (see --list)")))
                .paths
        }
        (None, true) => Vec::new(),
        (None, false) => {
            // powers: equal shares when none given, else normalised to 1
            let n = paths.len() as f64;
            let mut p = paths;
            if !explicit_power {
                for q in &mut p {
                    if let Component::Scatter { power, .. } = &mut q.components[0] {
                        *power = 1.0 / n;
                    }
                }
            } else if p.iter().any(|q| q.power().is_nan()) {
                die("give every path a power, or none");
            }
            let total: f64 = p.iter().map(PathSpec::power).sum();
            for q in &mut p {
                for c in &mut q.components {
                    match c {
                        Component::Scatter { power, .. } | Component::Specular { power, .. } => {
                            *power /= total
                        }
                    }
                }
            }
            p
        }
    };

    let (inp, out) = (&files[0], &files[1]);
    let is_cf32 = |f: &str| f.ends_with(".cf32");
    // read as real (mono wav) or complex (I/Q wav, cf32)
    let (fs, mut real, mut cplx, wav_fmt) = if is_cf32(inp) {
        let fs = rate.unwrap_or_else(|| die("a .cf32 input needs --rate"));
        let b = std::fs::read(inp).unwrap_or_else(|e| die(&format!("{inp}: {e}")));
        let c: Vec<Complex32> = b
            .as_chunks::<4>()
            .0
            .as_chunks::<2>()
            .0
            .iter()
            .map(|[re, im]| Complex32::new(f32::from_le_bytes(*re), f32::from_le_bytes(*im)))
            .collect();
        (fs, None, Some(c), None)
    } else {
        let f = std::fs::File::open(inp).unwrap_or_else(|e| die(&format!("{inp}: {e}")));
        let w =
            wav::read(std::io::BufReader::new(f)).unwrap_or_else(|e| die(&format!("{inp}: {e}")));
        let fs = w.rate as f64;
        match w.channels {
            1 => (fs, Some(w.samples), None, Some(w.format)),
            2 => (
                fs,
                None,
                Some(
                    w.samples
                        .as_chunks::<2>()
                        .0
                        .iter()
                        .map(|[i, q]| Complex32::new(*i, *q))
                        .collect(),
                ),
                Some(w.format),
            ),
            n => die(&format!("{inp}: {n} channels (1 or 2 only)")),
        }
    };
    let mut rng = Rng::new(seed ^ 0x6e_6f69_7365);
    if let Some(x) = real.as_mut() {
        let p0 = noise::power_r(x);
        if !paths.is_empty() {
            let mut ch = RealChannel::new(fs, &paths, seed);
            let lat = ch.latency();
            x.extend(std::iter::repeat_n(0.0, lat));
            ch.process(x);
            x.drain(..lat);
        }
        if let Some(s) = snr {
            noise::add_r(x, noise::noise_power(p0, s, bw, fs, false), &mut rng);
        }
    }
    if let Some(x) = cplx.as_mut() {
        let p0 = noise::power_c(x);
        if !paths.is_empty() {
            let mut ch = Channel::new(fs, &paths, seed);
            let lat = ch.latency();
            x.extend(std::iter::repeat_n(Complex32::default(), lat));
            ch.process(x);
            x.drain(..lat);
        }
        if let Some(s) = snr {
            noise::add_c(x, noise::noise_power(p0, s, bw, fs, true), &mut rng);
        }
    }
    let n = real
        .as_ref()
        .map(Vec::len)
        .or(cplx.as_ref().map(Vec::len))
        .unwrap_or(0);
    eprintln!(
        "ionochannel: {n} samples at {fs} Hz, {} path(s){}",
        paths.len(),
        snr.map(|s| format!(", SNR {s} dB in {bw} Hz"))
            .unwrap_or_default()
    );
    if is_cf32(out) {
        let c = cplx.unwrap_or_else(|| die("a real (mono) input can't be written as .cf32"));
        let b: Vec<u8> = c
            .iter()
            .flat_map(|z| z.re.to_le_bytes().into_iter().chain(z.im.to_le_bytes()))
            .collect();
        std::fs::write(out, b).unwrap_or_else(|e| die(&format!("{out}: {e}")));
    } else {
        let (channels, samples) = match (real, cplx) {
            (Some(r), _) => (1, r),
            (_, Some(c)) => (2, c.iter().flat_map(|z| [z.re, z.im]).collect()),
            _ => unreachable!(),
        };
        let w = wav::Wav {
            rate: fs as u32,
            channels,
            format: wav_fmt.unwrap_or(wav::Format::Float32),
            samples,
        };
        let f = std::fs::File::create(out).unwrap_or_else(|e| die(&format!("{out}: {e}")));
        let clipped = wav::write(std::io::BufWriter::new(f), &w)
            .unwrap_or_else(|e| die(&format!("{out}: {e}")));
        if clipped > 0 {
            eprintln!(
                "ionochannel: {clipped} samples clipped in 16-bit output (lower the level or write float)"
            );
        }
    }
}
