//! ionochannel: put a recording through an HF channel.
//!
//!   ionochannel [--preset NAME | --path D,S[,F[,P]]... | --specular D,P[,F]...]
//!               [--snr DB] [--qrn DB,VD] [--smps DB,F[,SP]] [--arcing DB]
//!               [--mains HZ] [--cw DB,F[,WPM]]... [--am DB,F]... [--carrier DB,F]...
//!               [--radar DB,F[,SWEEP_HZ,RATE_HZ]]... [--noise-file PATH,DB [--noise-band LO,HI]]
//!               [--tid F,KM,REL,MIN[,DEG]] [--doppler-drift RMS,TAU]
//!               [--bw HZ] [--seed N] [--rate HZ]
//!               [--mistune HZ] [--passband LO,HI [--transition HZ]]
//!               [--agc DECAY_MS[,HANG_MS[,ATTACK_MS]]] [--ssb] IN OUT
//!   ionochannel --list
//!
//! IN/OUT: .wav (mono: real audio; two channels: I/Q) or .cf32 (complex
//! float32, needs --rate). The output lines up with the input (the
//! filters' latency taken out) and keeps its format.

use ionochannel::vary::tid;
use ionochannel::{
    Agc, Am, AmSpec, Arcing, ArcingSpec, Bursts, BurstsSpec, Channel, Clock, Component, Cw, CwSpec,
    PathSpec, Population, PopulationSpec, Psk31, Psk31Spec, Qrn, Radar, RadarSpec, RealChannel,
    RealReceiver, Receiver, ReceiverSpec, Recorded, Rng, Rtty, RttySpec, Smps, SmpsSpec, Term,
    Vary, noise, presets, qrn, wav,
};
use num_complex::Complex32;
use std::process::exit;

const USAGE: &str =
    "usage: ionochannel [options] IN OUT    (IN/OUT: .wav mono or I/Q, or .cf32 with --rate)
  --preset NAME           a named channel (--list shows them)
  --path D,S[,F[,P]]      a Rayleigh path: delay ms, spread Hz (2 sigma), shift Hz, power
                          (repeat for more paths; powers default to equal shares)
  --specular D,P[,F]      a steady path: delay ms, power, shift Hz
  --vh TL,ST,SC,FS,FSL,SD,SV[,A]  a layer-mode spread over delay (Vogler & Hoffmeyer 1992):
                          lowest delay, total and peak spreads (ms), Doppler shift at the
                          peak and at the lowest delay, Doppler half-width (Hz), threshold
                          s_v (0-1) at which the spreads are measured, peak amplitude [1]
  --tid F,KM,REL,MIN[,DEG]  a travelling ionospheric disturbance on every path: carrier MHz,
                          path km, relative amplitude of the path (0.03-0.08 typical),
                          period minutes (60-100 typical), phase degrees; moves delay and
                          Doppler together
  --doppler-drift RMS,TAU  each path's Doppler shift drifts at random: rms Hz, correlation s
  --doppler-file PATH     replay a Doppler record on every path: CSV lines 'time_s,hz'
                          (time increasing; '#' lines skipped), linear in between
  --flare-file PATH,F[,K] a solar flare on every path: CSV 'time_s,xray_flux_W_m2'; power drops
                          by the D-RAP vertical absorption at F MHz, times K [1] for the path
  --snr DB                add white noise: SNR of the input's mean power in --bw
  --qrn DB,VD             add impulsive atmospheric noise: SNR in --bw, P.372 Vd (dB, 200 Hz;
                          1.05 = Gaussian, up to 14)
  --smps DB,F[,SP]        a switching-supply comb: each line DB against the signal, one line
                          at F Hz, lines SP Hz apart [60000]; drifting, with mains ripple
  --arcing DB             powerline arcing buzz: SNR in --bw
  --bursts DB[,ON,OFF]    noise bursts (e.g. powerline networking): SNR in --bw while on,
                          mean on and off times in ms [2,20]
  --mains HZ              mains frequency for --smps and --arcing [50]
 interfering stations (repeatable; DB against the signal's power, F Hz in the band):
  --cw DB,F[,WPM]         CW, random 2-6 character groups [20 wpm]; DB is key-down power
  --am DB,F               AM broadcast, speech-like audio to 4.5 kHz; DB is carrier power
  --carrier DB,F          a steady carrier, drifting slowly
  --radar DB,F[,SW,RATE]  FMCW radar: SW Hz sweeps [20000], RATE a second [50]
  --rtty DB,F             RTTY, 45.45 baud, 170 Hz shift, centred on F
  --psk31 DB,F            PSK31, random Varicode-like words; DB is peak power
  --population R[,DB[,S]] CW stations coming and going (Mendieta-Otero et al. 2014): R arrivals
                          a second, 10 s mean, levels logistic around DB [0] with scale S [11.8]
  --noise-file PATH,DB    mix in a noise recording (WAV at the same rate; mono for audio, I/Q
                          for I/Q), looped from a random start, at an SNR in --bw
  --noise-band LO,HI      band its level is measured in [300,2700 audio; +-0.45 fs I/Q]
  --bw HZ                 SNR reference bandwidth [2500]
  --seed N                fading and noise realisation [1]
  --rate HZ               sample rate of a .cf32 input
 receiver (after the noise):
  --mistune HZ            tune off by HZ (positive moves the audio up)
  --passband LO,HI        filter, -6 dB edges in Hz (IQ: negative for LSB)
  --transition HZ         filter transition width [100]
  --agc D[,H[,A]]         AGC: decay, hang, attack in ms [500,0,2]; output peak 0.25
  --ssb                   --passband 300,2700 --agc 500
  --clock PPM             the receiver's sample clock off by PPM (applied first; a fast clock
                          makes the output longer and its tones lower)
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

#[derive(Debug)]
enum Kind {
    Rtty,
    Psk31,
    Cw(f64),
    Am,
    Carrier,
    Radar(f64, f64),
}

struct Station {
    kind: Kind,
    db: f64,
    f: f64,
}

fn carrier_spec() -> SmpsSpec {
    SmpsSpec {
        spacing_hz: 1e12,
        am_depth: 0.0,
        fm_dev_hz: 0.0,
        drift_hz: 2.0,
        ..Default::default()
    }
}

impl Station {
    fn add_r(&self, x: &mut [f32], fs: f64, p0: f64, seed: u64) {
        let p = p0 * 10f64.powf(self.db / 10.0);
        match self.kind {
            Kind::Cw(wpm) => {
                let spec = CwSpec {
                    wpm,
                    ..Default::default()
                };
                Cw::new(fs, self.f, &spec, true, seed).add_r(x, p)
            }
            Kind::Am => Am::new(fs, self.f, &AmSpec::default(), true, seed).add_r(x, p),
            Kind::Rtty => Rtty::new(fs, self.f, &RttySpec::default(), true, seed).add_r(x, p),
            Kind::Psk31 => Psk31::new(fs, self.f, &Psk31Spec::default(), true, seed).add_r(x, p),
            Kind::Carrier => Smps::new(fs, self.f, &carrier_spec(), true, seed).add_r(x, p),
            Kind::Radar(sweep_hz, rate_hz) => {
                let spec = RadarSpec {
                    sweep_hz,
                    rate_hz,
                    ..Default::default()
                };
                Radar::new(fs, self.f, &spec, true, seed).add_r(x, p)
            }
        }
    }

    fn add_c(&self, x: &mut [Complex32], fs: f64, p0: f64, seed: u64) {
        let p = p0 * 10f64.powf(self.db / 10.0);
        match self.kind {
            Kind::Cw(wpm) => {
                let spec = CwSpec {
                    wpm,
                    ..Default::default()
                };
                Cw::new(fs, self.f, &spec, false, seed).add_c(x, p)
            }
            Kind::Am => Am::new(fs, self.f, &AmSpec::default(), false, seed).add_c(x, p),
            Kind::Rtty => Rtty::new(fs, self.f, &RttySpec::default(), false, seed).add_c(x, p),
            Kind::Psk31 => Psk31::new(fs, self.f, &Psk31Spec::default(), false, seed).add_c(x, p),
            Kind::Carrier => Smps::new(fs, self.f, &carrier_spec(), false, seed).add_c(x, p),
            Kind::Radar(sweep_hz, rate_hz) => {
                let spec = RadarSpec {
                    sweep_hz,
                    rate_hz,
                    ..Default::default()
                };
                Radar::new(fs, self.f, &spec, false, seed).add_c(x, p)
            }
        }
    }
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
    let mut bursts_arg: Option<(f64, f64, f64)> = None;
    let mut mains = 50.0;
    let mut stations: Vec<Station> = Vec::new();
    let mut population_arg: Option<PopulationSpec> = None;
    let mut noise_file: Option<(String, f64)> = None;
    let mut tid_arg: Option<Vec<f64>> = None;
    let mut clock_ppm: Option<f64> = None;
    let mut drift_arg: Option<(f64, f64)> = None;
    let mut doppler_file: Option<Term> = None;
    let mut flare_file: Option<Term> = None;
    let mut noise_band: Option<(f64, f64)> = None;
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
            "--vh" => {
                let v = nums(&val("--vh"), 7, 8, "--vh");
                if !(v[1] > 0.0 && v[2] > 0.0 && 2.0 * v[2] < v[1] && v[6] > 0.0 && v[6] < 1.0) {
                    die("--vh: need 0 < SC < ST / 2 and 0 < SV < 1");
                }
                explicit_power = true;
                paths.extend(ionochannel::wideband::vh_mode(
                    v[0] * 1e-3,
                    v[1] * 1e-3,
                    v[2] * 1e-3,
                    v[3],
                    v[4],
                    v[5],
                    v[6],
                    v.get(7).copied().unwrap_or(1.0),
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
            "--bursts" => {
                let v = nums(&val("--bursts"), 1, 3, "--bursts");
                if v.len() == 2 {
                    die("--bursts: give both ON and OFF, or neither");
                }
                let (on, off) = if v.len() == 3 {
                    (v[1], v[2])
                } else {
                    (2.0, 20.0)
                };
                if on <= 0.0 || off <= 0.0 {
                    die("--bursts: ON and OFF must be > 0");
                }
                bursts_arg = Some((v[0], on * 1e-3, off * 1e-3));
            }
            "--arcing" => arcing_arg = Some(nums(&val("--arcing"), 1, 1, "--arcing")[0]),
            "--cw" => {
                let v = nums(&val("--cw"), 2, 3, "--cw");
                let wpm = v.get(2).copied().unwrap_or(20.0);
                if wpm <= 0.0 {
                    die("--cw: WPM must be > 0");
                }
                stations.push(Station {
                    kind: Kind::Cw(wpm),
                    db: v[0],
                    f: v[1],
                });
            }
            "--am" => {
                let v = nums(&val("--am"), 2, 2, "--am");
                stations.push(Station {
                    kind: Kind::Am,
                    db: v[0],
                    f: v[1],
                });
            }
            "--population" => {
                let v = nums(&val("--population"), 1, 3, "--population");
                if v[0] < 0.0 {
                    die("--population: rate must be >= 0");
                }
                population_arg = Some(PopulationSpec {
                    level_db: v.get(1).copied().unwrap_or(0.0),
                    scale_db: v.get(2).copied().unwrap_or(11.8),
                    ..PopulationSpec::new(v[0])
                });
            }
            "--rtty" => {
                let v = nums(&val("--rtty"), 2, 2, "--rtty");
                stations.push(Station {
                    kind: Kind::Rtty,
                    db: v[0],
                    f: v[1],
                });
            }
            "--psk31" => {
                let v = nums(&val("--psk31"), 2, 2, "--psk31");
                stations.push(Station {
                    kind: Kind::Psk31,
                    db: v[0],
                    f: v[1],
                });
            }
            "--carrier" => {
                let v = nums(&val("--carrier"), 2, 2, "--carrier");
                stations.push(Station {
                    kind: Kind::Carrier,
                    db: v[0],
                    f: v[1],
                });
            }
            "--radar" => {
                let v = nums(&val("--radar"), 2, 4, "--radar");
                if v.len() == 3 {
                    die("--radar: give both SWEEP_HZ and RATE_HZ, or neither");
                }
                let (sw, rate) = if v.len() == 4 {
                    (v[2], v[3])
                } else {
                    (20e3, 50.0)
                };
                if sw <= 0.0 || rate <= 0.0 {
                    die("--radar: SWEEP_HZ and RATE_HZ must be > 0");
                }
                stations.push(Station {
                    kind: Kind::Radar(sw, rate),
                    db: v[0],
                    f: v[1],
                });
            }
            "--clock" => clock_ppm = Some(nums(&val("--clock"), 1, 1, "--clock")[0]),
            "--tid" => {
                let v = nums(&val("--tid"), 4, 5, "--tid");
                if v[0] <= 0.0 || v[1] <= 0.0 || v[3] <= 0.0 {
                    die("--tid: carrier, path length and period must be > 0");
                }
                tid_arg = Some(v);
            }
            "--doppler-file" => {
                let path = val("--doppler-file");
                let text =
                    std::fs::read_to_string(&path).unwrap_or_else(|e| die(&format!("{path}: {e}")));
                let pts: Vec<(f64, f64)> = text
                    .lines()
                    .map(str::trim)
                    .filter(|l| !l.is_empty() && !l.starts_with('#'))
                    .map(|l| {
                        let v = nums(l, 2, 2, &path);
                        (v[0], v[1])
                    })
                    .collect();
                if pts.is_empty() || !pts.windows(2).all(|w| w[1].0 > w[0].0) {
                    die(&format!("{path}: need increasing times"));
                }
                doppler_file = Some(Term::table(&pts));
            }
            "--flare-file" => {
                let v = val("--flare-file");
                let mut parts = v.split(',');
                let path = parts.next().unwrap_or_default().to_string();
                let rest: Vec<f64> = parts
                    .map(|t| {
                        t.trim()
                            .parse()
                            .unwrap_or_else(|_| die("--flare-file: PATH,F_MHZ[,FACTOR]"))
                    })
                    .collect();
                if rest.is_empty() || rest.len() > 2 || rest[0] <= 0.0 {
                    die("--flare-file: PATH,F_MHZ[,FACTOR]");
                }
                let k = rest.get(1).copied().unwrap_or(1.0);
                let text =
                    std::fs::read_to_string(&path).unwrap_or_else(|e| die(&format!("{path}: {e}")));
                let pts: Vec<(f64, f64)> = text
                    .lines()
                    .map(str::trim)
                    .filter(|l| !l.is_empty() && !l.starts_with('#'))
                    .map(|l| {
                        let v = nums(l, 2, 2, &path);
                        (
                            v[0],
                            -k * ionochannel::vary::flare_absorption_db(v[1], rest[0], 1.0),
                        )
                    })
                    .collect();
                if pts.is_empty() || !pts.windows(2).all(|w| w[1].0 > w[0].0) {
                    die(&format!("{path}: need increasing times"));
                }
                flare_file = Some(Term::table(&pts));
            }
            "--doppler-drift" => {
                let v = nums(&val("--doppler-drift"), 2, 2, "--doppler-drift");
                if v[0] < 0.0 || v[1] <= 0.0 {
                    die("--doppler-drift: rms >= 0, correlation time > 0");
                }
                drift_arg = Some((v[0], v[1]));
            }
            "--noise-file" => {
                let v = val("--noise-file");
                let (path, db) = v
                    .rsplit_once(',')
                    .unwrap_or_else(|| die("--noise-file: PATH,DB"));
                let db = db
                    .trim()
                    .parse()
                    .unwrap_or_else(|_| die("--noise-file: bad SNR"));
                noise_file = Some((path.to_string(), db));
            }
            "--noise-band" => {
                let v = nums(&val("--noise-band"), 2, 2, "--noise-band");
                if v[1] <= v[0] {
                    die("--noise-band: HI must be above LO");
                }
                noise_band = Some((v[0], v[1]));
            }
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

    let vary: Vec<Vary> = if tid_arg.is_none()
        && drift_arg.is_none()
        && doppler_file.is_none()
        && flare_file.is_none()
    {
        Vec::new()
    } else {
        paths
            .iter()
            .map(|_| {
                let mut v = tid_arg.as_ref().map_or_else(Vary::default, |t| {
                    tid(
                        t[0] * 1e6,
                        t[1],
                        t[2],
                        t[3] * 60.0,
                        t.get(4).copied().unwrap_or(0.0).to_radians(),
                    )
                });
                if let Some((rms, tau_s)) = drift_arg {
                    v.shift_hz.push(Term::Drift { rms, tau_s });
                }
                if let Some(t) = &doppler_file {
                    v.shift_hz.push(t.clone());
                }
                if let Some(t) = &flare_file {
                    v.power_db.push(t.clone());
                }
                v
            })
            .collect()
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
    let recording = noise_file.as_ref().map(|(path, _)| {
        let f = std::fs::File::open(path).unwrap_or_else(|e| die(&format!("{path}: {e}")));
        let w =
            wav::read(std::io::BufReader::new(f)).unwrap_or_else(|e| die(&format!("{path}: {e}")));
        if w.rate as f64 != fs {
            die(&format!(
                "{path}: {} Hz, the input is {fs} Hz (resample it first)",
                w.rate
            ));
        }
        w
    });
    let mut rng = Rng::new(seed ^ 0x6e_6f69_7365);
    if let Some(x) = real.as_mut() {
        let p0 = noise::power_r(x);
        if !paths.is_empty() {
            let mut ch = RealChannel::with_variation(fs, &paths, &vary, seed);
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
        if let Some((s, on_s, off_s)) = bursts_arg {
            Bursts::new(fs, &BurstsSpec { on_s, off_s }, seed ^ 0x6273_7473)
                .add_r(x, noise::noise_power(p0, s, bw, fs, false));
        }
        for (i, st) in stations.iter().enumerate() {
            st.add_r(x, fs, p0, seed ^ (0x7374_0000 + i as u64));
        }
        if let Some(spec) = &population_arg {
            Population::new(fs, spec, true, seed ^ 0x706f_7075).add_r(x, p0);
        }
        if let (Some(w), Some((path, db))) = (&recording, &noise_file) {
            if w.channels != 1 {
                die(&format!("{path}: mono audio input needs a mono recording"));
            }
            Recorded::new_r(&w.samples, fs, noise_band, seed ^ 0x0072_6563).add_r(
                x,
                p0 / 10f64.powf(db / 10.0),
                bw,
            );
        }
        if let Some(ppm) = clock_ppm {
            let total = (x.len() as f64 * (1.0 + ppm * 1e-6)).round() as u64;
            let mut c = Clock::new(ppm);
            let mut y = Vec::with_capacity(total as usize);
            c.process_r(x, &mut y);
            c.flush_r(total, &mut y);
            *x = y;
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
            let mut ch = Channel::with_variation(fs, &paths, &vary, seed);
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
        if let Some((s, on_s, off_s)) = bursts_arg {
            Bursts::new(fs, &BurstsSpec { on_s, off_s }, seed ^ 0x6273_7473)
                .add_c(x, noise::noise_power(p0, s, bw, fs, true));
        }
        for (i, st) in stations.iter().enumerate() {
            st.add_c(x, fs, p0, seed ^ (0x7374_0000 + i as u64));
        }
        if let Some(spec) = &population_arg {
            Population::new(fs, spec, false, seed ^ 0x706f_7075).add_c(x, p0);
        }
        if let (Some(w), Some((path, db))) = (&recording, &noise_file) {
            if w.channels != 2 {
                die(&format!(
                    "{path}: I/Q input needs a two-channel (I/Q) recording"
                ));
            }
            let c: Vec<Complex32> = w
                .samples
                .as_chunks::<2>()
                .0
                .iter()
                .map(|[i, q]| Complex32::new(*i, *q))
                .collect();
            Recorded::new_c(&c, fs, noise_band, seed ^ 0x0072_6563).add_c(
                x,
                p0 / 10f64.powf(db / 10.0),
                bw,
            );
        }
        if let Some(ppm) = clock_ppm {
            let total = (x.len() as f64 * (1.0 + ppm * 1e-6)).round() as u64;
            let mut c = Clock::new(ppm);
            let mut y = Vec::with_capacity(total as usize);
            c.process_c(x, &mut y);
            c.flush_c(total, &mut y);
            *x = y;
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
    if let Some((s, on, off)) = bursts_arg {
        eprintln!(
            "ionochannel: noise bursts at {s} dB in {bw} Hz while on, {} ms on, {} ms off",
            on * 1e3,
            off * 1e3
        );
    }
    if let Some(s) = arcing_arg {
        eprintln!("ionochannel: arcing at {s} dB in {bw} Hz, mains {mains} Hz");
    }
    for st in &stations {
        eprintln!(
            "ionochannel: station {:?} at {} Hz, {} dB",
            st.kind, st.f, st.db
        );
    }
    if let Some(t) = &tid_arg {
        eprintln!(
            "ionochannel: TID {} MHz, {} km, {} relative, {} min",
            t[0], t[1], t[2], t[3]
        );
    }
    if let Some((rms, tau)) = drift_arg {
        eprintln!("ionochannel: Doppler drift {rms} Hz rms, {tau} s");
    }
    if let Some(sp) = &population_arg {
        eprintln!(
            "ionochannel: CW population, {} arrivals/s, levels around {} dB (scale {} dB)",
            sp.rate_per_s, sp.level_db, sp.scale_db
        );
    }
    if let Some((path, db)) = &noise_file {
        eprintln!("ionochannel: recorded noise {path} at {db} dB in {bw} Hz");
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
