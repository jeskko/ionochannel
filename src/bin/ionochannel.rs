//! ionochannel: put a recording through an HF channel.
//!
//!   ionochannel [--preset NAME | --path D,S[,F[,P]]... | --specular D,P[,F]...]
//!               [--snr DB] [--qrn DB,VD] [--smps DB,F[,SP]] [--arcing DB]
//!               [--mains HZ] [--bw HZ] [--seed N] [--rate HZ]
//!               [--mistune HZ] [--passband LO,HI [--transition HZ]]
//!               [--agc DECAY_MS[,HANG_MS[,ATTACK_MS]]] [--ssb] IN OUT
//!   ionochannel --list
//!
//! IN/OUT: .wav (mono: real audio; two channels: I/Q) or .cf32 (complex
//! float32, needs --rate). The output lines up with the input (the
//! filters' latency taken out) and keeps its format.

use ionochannel::{
    Agc, Arcing, ArcingSpec, Channel, Component, PathSpec, Qrn, RealChannel, RealReceiver,
    Receiver, ReceiverSpec, Rng, Smps, SmpsSpec, noise, presets, qrn, wav,
};
use num_complex::Complex32;
use std::process::exit;

const USAGE: &str =
    "usage: ionochannel [options] IN OUT    (IN/OUT: .wav mono or I/Q, or .cf32 with --rate)
  --preset NAME           a named channel (--list shows them)
  --path D,S[,F[,P]]      a Rayleigh path: delay ms, spread Hz (2 sigma), shift Hz, power
                          (repeat for more paths; powers default to equal shares)
  --specular D,P[,F]      a steady path: delay ms, power, shift Hz
  --snr DB                add white noise: SNR of the input's mean power in --bw
  --qrn DB,VD             add impulsive atmospheric noise: SNR in --bw, P.372 Vd (dB, 200 Hz;
                          1.05 = Gaussian, up to 14)
  --smps DB,F[,SP]        a switching-supply comb: each line DB against the signal, one line
                          at F Hz, lines SP Hz apart [60000]; drifting, with mains ripple
  --arcing DB             powerline arcing buzz: SNR in --bw
  --mains HZ              mains frequency for --smps and --arcing [50]
  --bw HZ                 SNR reference bandwidth [2500]
  --seed N                fading and noise realisation [1]
  --rate HZ               sample rate of a .cf32 input
 receiver (after the noise):
  --mistune HZ            tune off by HZ (positive moves the audio up)
  --passband LO,HI        filter, -6 dB edges in Hz (IQ: negative for LSB)
  --transition HZ         filter transition width [100]
  --agc D[,H[,A]]         AGC: decay, hang, attack in ms [500,0,2]; output peak 0.25
  --ssb                   --passband 300,2700 --agc 500
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
    let mut rx = ReceiverSpec::default();
    let mut qrn_arg: Option<(f64, f64)> = None;
    let mut smps_arg: Option<(f64, f64, f64)> = None;
    let mut arcing_arg: Option<f64> = None;
    let mut mains = 50.0;
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
            "--qrn" => {
                let v = nums(&val("--qrn"), 2, 2, "--qrn");
                if !(qrn::VD_GAUSS..=qrn::vd_max()).contains(&v[1]) {
                    die(&format!(
                        "--qrn: Vd must be {:.2}..{} dB",
                        qrn::VD_GAUSS,
                        qrn::vd_max()
                    ));
                }
                qrn_arg = Some((v[0], v[1]));
            }
            "--smps" => {
                let v = nums(&val("--smps"), 2, 3, "--smps");
                let sp = v.get(2).copied().unwrap_or(60e3);
                if sp <= 0.0 {
                    die("--smps: spacing must be > 0");
                }
                smps_arg = Some((v[0], v[1], sp));
            }
            "--arcing" => arcing_arg = Some(nums(&val("--arcing"), 1, 1, "--arcing")[0]),
            "--mains" => {
                mains = nums(&val("--mains"), 1, 1, "--mains")[0];
                if mains <= 0.0 {
                    die("--mains must be > 0");
                }
            }
            "--bw" => bw = nums(&val("--bw"), 1, 1, "--bw")[0],
            "--seed" => {
                seed = val("--seed")
                    .parse()
                    .unwrap_or_else(|_| die("--seed: not a number"))
            }
            "--rate" => rate = Some(nums(&val("--rate"), 1, 1, "--rate")[0]),
            "--mistune" => rx.mistune_hz = nums(&val("--mistune"), 1, 1, "--mistune")[0],
            "--passband" => {
                let v = nums(&val("--passband"), 2, 2, "--passband");
                if v[1] <= v[0] {
                    die("--passband: HI must be above LO");
                }
                rx.passband = Some((v[0], v[1]));
            }
            "--transition" => {
                rx.transition_hz = nums(&val("--transition"), 1, 1, "--transition")[0];
                if rx.transition_hz <= 0.0 {
                    die("--transition must be > 0");
                }
            }
            "--agc" => {
                let v = nums(&val("--agc"), 1, 3, "--agc");
                let d = Agc::default();
                rx.agc = Some(Agc {
                    decay_s: v[0] * 1e-3,
                    hang_s: v.get(1).map_or(d.hang_s, |h| h * 1e-3),
                    attack_s: v.get(2).map_or(d.attack_s, |a| a * 1e-3),
                    ..d
                });
            }
            "--ssb" => {
                rx.passband = Some((300.0, 2700.0));
                rx.agc.get_or_insert_with(Agc::default);
            }
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
        if let Some((s, vd)) = qrn_arg {
            Qrn::new(fs, vd, seed ^ 0x0071_726e).add_r(x, noise::noise_power(p0, s, bw, fs, false));
        }
        if let Some((l, f, sp)) = smps_arg {
            let spec = SmpsSpec {
                spacing_hz: sp,
                mains_hz: mains,
                ..Default::default()
            };
            Smps::new(fs, f, &spec, true, seed ^ 0x736d_7073).add_r(x, p0 * 10f64.powf(l / 10.0));
        }
        if let Some(s) = arcing_arg {
            let spec = ArcingSpec {
                mains_hz: mains,
                ..Default::default()
            };
            Arcing::new(fs, &spec, seed ^ 0x6172_6373)
                .add_r(x, noise::noise_power(p0, s, bw, fs, false));
        }
        if rx != ReceiverSpec::default() {
            let mut r = RealReceiver::new(fs, &rx);
            let lat = r.latency();
            x.extend(std::iter::repeat_n(0.0, lat));
            r.process(x);
            x.drain(..lat);
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
        if let Some((s, vd)) = qrn_arg {
            Qrn::new(fs, vd, seed ^ 0x0071_726e).add_c(x, noise::noise_power(p0, s, bw, fs, true));
        }
        if let Some((l, f, sp)) = smps_arg {
            let spec = SmpsSpec {
                spacing_hz: sp,
                mains_hz: mains,
                ..Default::default()
            };
            Smps::new(fs, f, &spec, false, seed ^ 0x736d_7073).add_c(x, p0 * 10f64.powf(l / 10.0));
        }
        if let Some(s) = arcing_arg {
            let spec = ArcingSpec {
                mains_hz: mains,
                ..Default::default()
            };
            Arcing::new(fs, &spec, seed ^ 0x6172_6373)
                .add_c(x, noise::noise_power(p0, s, bw, fs, true));
        }
        if rx != ReceiverSpec::default() {
            let mut r = Receiver::new(fs, &rx);
            let lat = r.latency();
            x.extend(std::iter::repeat_n(Complex32::default(), lat));
            r.process(x);
            x.drain(..lat);
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
    if let Some((s, vd)) = qrn_arg {
        eprintln!("ionochannel: QRN at {s} dB in {bw} Hz, Vd {vd} dB in 200 Hz");
    }
    if let Some((l, f, sp)) = smps_arg {
        eprintln!(
            "ionochannel: SMPS comb, lines {l} dBc at {f} Hz + k * {sp} Hz, mains {mains} Hz"
        );
    }
    if let Some(s) = arcing_arg {
        eprintln!("ionochannel: arcing at {s} dB in {bw} Hz, mains {mains} Hz");
    }
    if rx != ReceiverSpec::default() {
        eprintln!("ionochannel: receiver {rx:?}");
    }
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
