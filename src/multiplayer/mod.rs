//! Causal two-player Dynamic Mode: wire protocol, transports, handshake and pacing. See
//! [`docs/MULTIPLAYER.md`](../../docs/MULTIPLAYER.md) for the physics rule and the protocol.

pub mod game;
pub mod link;
#[cfg(not(target_arch = "wasm32"))]
pub mod p2p;
pub mod pacing;
pub mod protocol;
#[cfg(not(target_arch = "wasm32"))]
pub mod rendezvous;
pub mod session;
#[cfg(not(target_arch = "wasm32"))]
pub mod stun;
#[cfg(not(target_arch = "wasm32"))]
pub mod tcp;
