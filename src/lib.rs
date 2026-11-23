//! ionochannel: an HF ionospheric channel simulator for testing receivers
//! and decoders and for making training data.
//!
//! The channel is the Watterson model of ITU-R F.1487: a tapped delay
//! line, one tap per propagation path, each tap's gain a sum of Rayleigh
//! scatter (Gaussian Doppler spectrum, spread and shift) and specular
//! components; then noise: white ([`noise`]) or impulsive atmospheric
//! noise set by its P.372 voltage deviation ([`Qrn`]). Complex baseband ([`Channel`]) or real audio
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
pub mod noise;
pub mod presets;
pub mod qrn;
pub mod receiver;
pub mod rng;
pub mod wav;

pub use channel::{Channel, PathSpec, RealChannel};
pub use fading::Component;
pub use qrn::Qrn;
pub use receiver::{Agc, RealReceiver, Receiver, ReceiverSpec};
pub use rng::Rng;
