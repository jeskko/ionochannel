//! ionochannel: an HF ionospheric channel simulator for testing receivers
//! and decoders and for making training data.
//!
//! The channel is the Watterson model of ITU-R F.1487: a tapped delay
//! line, one tap per propagation path, each tap's gain a sum of Rayleigh
//! scatter (Gaussian Doppler spectrum, spread and shift) and specular
//! components; then noise: white ([`noise`]) or impulsive atmospheric
//! noise set by its P.372 voltage deviation ([`Qrn`]); man-made
//! interference ([`Smps`] combs, [`Arcing`] buzz); interfering stations
//! ([`Cw`], [`Am`], [`Radar`]); recorded noise at an SNR ([`Recorded`]). Complex baseband ([`Channel`]) or real audio
//! ([`RealChannel`]); named conditions in [`presets`]. After that, the
//! receiver: mistuning, passband filter and AGC ([`Receiver`],
//! [`RealReceiver`]).
//!
//! ```
//! use ionochannel::{presets, Channel};
//! use num_complex::Complex32;
//! let p = presets::get("poor").unwrap();
//! let mut ch = Channel::new(12000.0, &p.paths, 1);
//! let mut x = vec![Complex32::new(1.0, 0.0); 12000];
//! ch.process(&mut x);
//! ```

pub mod channel;
pub mod fading;
pub mod interference;
pub mod noise;
pub mod presets;
pub mod qrn;
pub mod receiver;
pub mod recorded;
pub mod rng;
pub mod stations;
pub mod wav;

pub use channel::{Channel, PathSpec, RealChannel};
pub use fading::Component;
pub use interference::{Arcing, ArcingSpec, Smps, SmpsSpec};
pub use qrn::Qrn;
pub use receiver::{Agc, RealReceiver, Receiver, ReceiverSpec};
pub use recorded::Recorded;
pub use rng::Rng;
pub use stations::{Am, AmSpec, Cw, CwSpec, Radar, RadarSpec};
