//! Causal two-player Dynamic Mode: wire protocol, transports, handshake and pacing. See
//! [`docs/MULTIPLAYER.md`](../../docs/MULTIPLAYER.md) for the physics rule and the protocol.

pub mod game;
pub mod link;
pub mod pacing;
pub mod protocol;
pub mod session;
#[cfg(not(target_arch = "wasm32"))]
pub mod tcp;
