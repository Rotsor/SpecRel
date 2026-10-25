//! Session messages and their compact binary encoding: a 1-byte tag followed by little-endian
//! fields. Floats travel as their exact bits, so both replicas see identical inputs.

use std::fmt;

use crate::dynamic::billiards::Ammo;
use crate::dynamic::helm::ControlMode;
use crate::dynamic::{PlayerId, ShipSample};
use crate::minkowski::Vec4;
use crate::renderer::Frontier;

/// Bumped whenever the encoding or the meaning of a message changes.
pub const PROTOCOL_VERSION: u16 = 5;

const TAG_HELLO: u8 = 0;
const TAG_TRACK: u8 = 1;
const TAG_FIRE: u8 = 2;
const TAG_FLASH: u8 = 3;
const TAG_BYE: u8 = 4;
const TAG_PING: u8 = 5;
const TAG_PONG: u8 = 6;
const TAG_CONTROLS: u8 = 7;

/// One message from a peer.
#[derive(Clone, Debug)]
pub enum Message {
    /// First message of a session.
    Hello { version: u16, player: PlayerId },
    /// The sender's pilot reached `sample`; also a heartbeat while `waiting`. Their hull follows
    /// from their pilot's track: each replica computes it.
    Track { sample: ShipSample, waiting: bool },
    /// The sender fired at event `at` with lab velocity `vel`.
    Fire { at: Frontier, vel: [f64; 3], ammo: Ammo },
    /// The sender emitted a light flash at `at`.
    Flash { at: Vec4 },
    /// The sender quits.
    Bye,
    /// A round-trip probe, stamped with the sender's wall clock (seconds). Answered with `Pong`.
    Ping { sent: f64 },
    /// The answer to a `Ping`, echoing its stamp.
    Pong { sent: f64 },
    /// The sender flies with `controls` (sent before their ship first moves).
    Controls { controls: ControlMode },
}

// `Frontier` has no `PartialEq`, so compare its fields here.
impl PartialEq for Message {
    fn eq(&self, other: &Self) -> bool {
        use Message::*;
        match (self, other) {
            (Hello { version: a, player: p }, Hello { version: b, player: q }) => a == b && p == q,
            (Track { sample: a, waiting: w }, Track { sample: b, waiting: v }) => a == b && w == v,
            (Controls { controls: a }, Controls { controls: b }) => a == b,
            (Fire { at: a, vel: v, ammo: m }, Fire { at: b, vel: w, ammo: n }) => {
                a.origin == b.origin && a.proper_time == b.proper_time && v == w && m == n
            }
            (Flash { at: a }, Flash { at: b }) => a == b,
            (Bye, Bye) => true,
            (Ping { sent: a }, Ping { sent: b }) | (Pong { sent: a }, Pong { sent: b }) => a == b,
            _ => false,
        }
    }
}

/// Why a frame could not be decoded.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DecodeError {
    Truncated,
    UnknownTag(u8),
    BadAmmo(u8),
    BadControls(u8),
    TrailingBytes,
}

impl fmt::Display for DecodeError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Truncated => write!(f, "truncated message"),
            Self::UnknownTag(t) => write!(f, "unknown message tag {t}"),
            Self::BadAmmo(a) => write!(f, "unknown ammo {a}"),
            Self::BadControls(c) => write!(f, "unknown controls {c}"),
            Self::TrailingBytes => write!(f, "trailing bytes after message"),
        }
    }
}

impl Message {
    /// Appends the encoding of `self` to `out`.
    pub fn encode(&self, out: &mut Vec<u8>) {
        match self {
            Self::Hello { version, player } => {
                out.push(TAG_HELLO);
                out.extend_from_slice(&version.to_le_bytes());
                out.push(*player);
            }
            Self::Track { sample, waiting } => {
                out.push(TAG_TRACK);
                put_f64s(out, &[sample.t]);
                put_f64s(out, &sample.pos);
                put_f64s(out, &sample.vel);
                put_f64s(out, &[sample.tau]);
                out.push(*waiting as u8);
            }
            Self::Fire { at, vel, ammo } => {
                out.push(TAG_FIRE);
                put_vec4(out, at.origin);
                out.extend_from_slice(&at.proper_time.to_le_bytes());
                put_f64s(out, vel);
                out.push(ammo_code(*ammo));
            }
            Self::Flash { at } => {
                out.push(TAG_FLASH);
                put_vec4(out, *at);
            }
            Self::Bye => out.push(TAG_BYE),
            Self::Ping { sent } => {
                out.push(TAG_PING);
                put_f64s(out, &[*sent]);
            }
            Self::Pong { sent } => {
                out.push(TAG_PONG);
                put_f64s(out, &[*sent]);
            }
            Self::Controls { controls } => {
                out.push(TAG_CONTROLS);
                out.push(match controls {
                    ControlMode::Delayed => 0,
                    ControlMode::Pod => 1,
                });
            }
        }
    }

    /// Decodes exactly one message occupying all of `bytes`.
    pub fn decode(bytes: &[u8]) -> Result<Message, DecodeError> {
        let mut r = Reader(bytes);
        let msg = match r.u8()? {
            TAG_HELLO => Self::Hello { version: r.u16()?, player: r.u8()? },
            TAG_TRACK => {
                let sample = r.ship_sample()?;
                Self::Track { sample, waiting: r.u8()? != 0 }
            }
            TAG_FIRE => {
                let at = Frontier { origin: r.vec4()?, proper_time: r.f32()? };
                let vel = r.v3()?;
                let ammo = match r.u8()? {
                    0 => Ammo::Puck,
                    1 => Ammo::Sticky,
                    2 => Ammo::Spore,
                    a => return Err(DecodeError::BadAmmo(a)),
                };
                Self::Fire { at, vel, ammo }
            }
            TAG_FLASH => Self::Flash { at: r.vec4()? },
            TAG_BYE => Self::Bye,
            TAG_PING => Self::Ping { sent: r.f64()? },
            TAG_PONG => Self::Pong { sent: r.f64()? },
            TAG_CONTROLS => {
                let controls = match r.u8()? {
                    0 => ControlMode::Delayed,
                    1 => ControlMode::Pod,
                    c => return Err(DecodeError::BadControls(c)),
                };
                Self::Controls { controls }
            }
            t => return Err(DecodeError::UnknownTag(t)),
        };
        if !r.0.is_empty() {
            return Err(DecodeError::TrailingBytes);
        }
        Ok(msg)
    }
}

fn ammo_code(ammo: Ammo) -> u8 {
    match ammo {
        Ammo::Puck => 0,
        Ammo::Sticky => 1,
        Ammo::Spore => 2,
    }
}

fn put_f64s(out: &mut Vec<u8>, xs: &[f64]) {
    for x in xs {
        out.extend_from_slice(&x.to_le_bytes());
    }
}

fn put_vec4(out: &mut Vec<u8>, v: Vec4) {
    for x in [v.t, v.x, v.y, v.z] {
        out.extend_from_slice(&x.to_le_bytes());
    }
}

/// Consumes little-endian fields from the front of a slice.
struct Reader<'a>(&'a [u8]);

impl Reader<'_> {
    fn take<const N: usize>(&mut self) -> Result<[u8; N], DecodeError> {
        if self.0.len() < N {
            return Err(DecodeError::Truncated);
        }
        let (head, rest) = self.0.split_at(N);
        self.0 = rest;
        Ok(head.try_into().unwrap())
    }

    fn u8(&mut self) -> Result<u8, DecodeError> {
        Ok(self.take::<1>()?[0])
    }

    fn u16(&mut self) -> Result<u16, DecodeError> {
        Ok(u16::from_le_bytes(self.take()?))
    }

    fn f32(&mut self) -> Result<f32, DecodeError> {
        Ok(f32::from_le_bytes(self.take()?))
    }

    fn f64(&mut self) -> Result<f64, DecodeError> {
        Ok(f64::from_le_bytes(self.take()?))
    }

    fn v3(&mut self) -> Result<[f64; 3], DecodeError> {
        Ok([self.f64()?, self.f64()?, self.f64()?])
    }

    fn vec4(&mut self) -> Result<Vec4, DecodeError> {
        Ok(Vec4::new(self.f32()?, self.f32()?, self.f32()?, self.f32()?))
    }

    fn ship_sample(&mut self) -> Result<ShipSample, DecodeError> {
        Ok(ShipSample { t: self.f64()?, pos: self.v3()?, vel: self.v3()?, tau: self.f64()? })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn samples() -> Vec<Message> {
        let sample = ShipSample {
            t: 1.0e12 + 0.123456789,
            pos: [-0.0, f64::MIN_POSITIVE, -1.0e300],
            vel: [0.999_999_999, -0.5, 0.0],
            tau: 123.456,
        };
        let at = Frontier { origin: Vec4::new(3.0e7, -0.0, 1.5, f32::MAX), proper_time: 77.25 };
        vec![
            Message::Hello { version: PROTOCOL_VERSION, player: 0 },
            Message::Hello { version: u16::MAX, player: 255 },
            Message::Track { sample, waiting: false },
            Message::Track { sample, waiting: true },
            Message::Fire { at, vel: [0.1, -0.2, f64::EPSILON], ammo: Ammo::Puck },
            Message::Fire { at, vel: [0.0; 3], ammo: Ammo::Sticky },
            Message::Fire { at, vel: [-0.0; 3], ammo: Ammo::Spore },
            Message::Flash { at: Vec4::new(1.0e9, -0.0, f32::MIN_POSITIVE, -2.5) },
            Message::Bye,
            Message::Ping { sent: 12.5 },
            Message::Pong { sent: -0.0 },
            Message::Controls { controls: ControlMode::Delayed },
            Message::Controls { controls: ControlMode::Pod },
        ]
    }

    fn encoded(m: &Message) -> Vec<u8> {
        let mut out = Vec::new();
        m.encode(&mut out);
        out
    }

    #[test]
    fn roundtrip_is_exact() {
        for m in samples() {
            let bytes = encoded(&m);
            let back = Message::decode(&bytes).unwrap();
            assert_eq!(back, m);
            // `==` treats -0.0 == 0.0; the re-encoding checks the exact bits.
            assert_eq!(encoded(&back), bytes);
        }
    }

    #[test]
    fn negative_zero_survives() {
        let m = Message::Flash { at: Vec4::new(-0.0, 0.0, 0.0, 0.0) };
        match Message::decode(&encoded(&m)).unwrap() {
            Message::Flash { at } => assert!(at.t.is_sign_negative() && at.x.is_sign_positive()),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn truncated_input_is_rejected() {
        assert_eq!(Message::decode(&[]), Err(DecodeError::Truncated));
        for m in samples() {
            let bytes = encoded(&m);
            for n in 1..bytes.len() {
                assert_eq!(Message::decode(&bytes[..n]), Err(DecodeError::Truncated), "{m:?} {n}");
            }
        }
    }

    #[test]
    fn bad_input_is_rejected() {
        assert_eq!(Message::decode(&[200]), Err(DecodeError::UnknownTag(200)));
        assert_eq!(Message::decode(&[TAG_BYE, 0]), Err(DecodeError::TrailingBytes));
        let mut bytes = encoded(&samples()[4]);
        *bytes.last_mut().unwrap() = 3;
        assert_eq!(Message::decode(&bytes), Err(DecodeError::BadAmmo(3)));
        assert_eq!(Message::decode(&[TAG_CONTROLS, 2]), Err(DecodeError::BadControls(2)));
        assert_eq!(DecodeError::UnknownTag(9).to_string(), "unknown message tag 9");
    }
}
