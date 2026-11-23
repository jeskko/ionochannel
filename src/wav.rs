//! Minimal WAV reading and writing for the command line: mono, 16-bit PCM
//! or 32-bit float; two channels are read as I/Q.

use std::io::{self, Read, Write};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Format {
    Pcm16,
    Float32,
}

pub struct Wav {
    pub rate: u32,
    pub channels: u16,
    pub format: Format,
    /// Interleaved samples, scaled to +-1.
    pub samples: Vec<f32>,
}

fn bad(m: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, m.to_string())
}

pub fn read(mut r: impl Read) -> io::Result<Wav> {
    let mut b = Vec::new();
    r.read_to_end(&mut b)?;
    if b.len() < 12 || &b[0..4] != b"RIFF" || &b[8..12] != b"WAVE" {
        return Err(bad("not a RIFF/WAVE file"));
    }
    let u16_at = |i: usize| u16::from_le_bytes([b[i], b[i + 1]]);
    let u32_at = |i: usize| u32::from_le_bytes([b[i], b[i + 1], b[i + 2], b[i + 3]]);
    let mut i = 12;
    let mut fmt = None;
    while i + 8 <= b.len() {
        let id = &b[i..i + 4];
        let len = u32_at(i + 4) as usize;
        let body = i + 8;
        let end = (body + len).min(b.len());
        if id == b"fmt " {
            let tag = u16_at(body);
            let channels = u16_at(body + 2);
            let rate = u32_at(body + 4);
            let bits = u16_at(body + 14);
            let tag = if tag == 0xfffe {
                u16_at(body + 24)
            } else {
                tag
            };
            let format = match (tag, bits) {
                (1, 16) => Format::Pcm16,
                (3, 32) => Format::Float32,
                _ => return Err(bad("only 16-bit PCM or 32-bit float WAV")),
            };
            fmt = Some((rate, channels, format));
        } else if id == b"data" {
            let (rate, channels, format) = fmt.ok_or_else(|| bad("data before fmt"))?;
            let d = &b[body..end];
            let samples = match format {
                Format::Pcm16 => d
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|c| i16::from_le_bytes(*c) as f32 / 32768.0)
                    .collect(),
                Format::Float32 => d
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|c| f32::from_le_bytes(*c))
                    .collect(),
            };
            return Ok(Wav {
                rate,
                channels,
                format,
                samples,
            });
        }
        i = body + len + (len & 1);
    }
    Err(bad("no data chunk"))
}

/// Writes; returns how many samples were clipped (16-bit PCM only).
pub fn write(mut w: impl Write, wav: &Wav) -> io::Result<usize> {
    let (tag, bits) = match wav.format {
        Format::Pcm16 => (1u16, 16u16),
        Format::Float32 => (3, 32),
    };
    let bytes = bits as u32 / 8;
    let data_len = wav.samples.len() as u32 * bytes;
    let mut h = Vec::with_capacity(44);
    h.extend_from_slice(b"RIFF");
    h.extend_from_slice(&(36 + data_len).to_le_bytes());
    h.extend_from_slice(b"WAVEfmt ");
    h.extend_from_slice(&16u32.to_le_bytes());
    h.extend_from_slice(&tag.to_le_bytes());
    h.extend_from_slice(&wav.channels.to_le_bytes());
    h.extend_from_slice(&wav.rate.to_le_bytes());
    h.extend_from_slice(&(wav.rate * wav.channels as u32 * bytes).to_le_bytes());
    h.extend_from_slice(&((wav.channels as u32 * bytes) as u16).to_le_bytes());
    h.extend_from_slice(&bits.to_le_bytes());
    h.extend_from_slice(b"data");
    h.extend_from_slice(&data_len.to_le_bytes());
    w.write_all(&h)?;
    let mut clipped = 0;
    let mut out = Vec::with_capacity(data_len as usize);
    for &s in &wav.samples {
        match wav.format {
            Format::Pcm16 => {
                let v = (s * 32768.0).round();
                if !(-32768.0..=32767.0).contains(&v) {
                    clipped += 1;
                }
                out.extend_from_slice(&(v.clamp(-32768.0, 32767.0) as i16).to_le_bytes());
            }
            Format::Float32 => out.extend_from_slice(&s.to_le_bytes()),
        }
    }
    w.write_all(&out)?;
    Ok(clipped)
}
