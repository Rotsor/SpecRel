//! One player's side of a two-player match: the session to the other player, what arrives
//! over it applied to this player's replica of the world, and the pacing that keeps this
//! player out of the other's future light cone.

use super::link::Link;
use super::pacing::Pacer;
use super::protocol::Message;
use super::session::{Event, Session};
use crate::dynamic::billiards::{self, Ammo};
use crate::dynamic::collide::V3;
use crate::dynamic::{DynWorld, PlayerId, ShipSample};
use crate::minkowski::Vec4;
use crate::renderer::{FrameDynamics, Frontier};

pub const PLAYERS: usize = 2;

/// The HUD banner while the local player is held back.
pub const WAITING_BANNER: &str = "WAITING FOR THE OTHER PLAYER";

/// Every player's ship at its spawn event, at rest, clock at zero.
pub fn spawns() -> Vec<ShipSample> {
    (0..PLAYERS as PlayerId).map(|p| ShipSample::at_rest(billiards::spawn_event(p))).collect()
}

pub struct Match {
    session: Session,
    pacer: Pacer,
    remote: PlayerId,
    /// The other ship's latest known event: its spawn until it is heard from.
    remote_latest: ShipSample,
    /// The other player said they are waiting for us.
    remote_waiting: bool,
    /// Why the other player could not join, if they couldn't.
    problem: Option<String>,
}

impl Match {
    pub fn new(local: PlayerId, link: Box<dyn Link>) -> Self {
        let remote = 1 - local;
        Self {
            session: Session::new(local, link),
            pacer: Pacer::default(),
            remote,
            remote_latest: spawns()[remote as usize],
            remote_waiting: false,
            problem: None,
        }
    }

    pub fn local(&self) -> PlayerId {
        self.session.local()
    }

    /// Apply everything the other player did since the last call, in order: their ship's
    /// track and shots to `world`, their flashes to `dynamics`. Returns whether the world
    /// changed (it still needs [`DynWorld::extend`]).
    pub fn receive(&mut self, world: &mut DynWorld, dynamics: &mut FrameDynamics) -> bool {
        let mut changed = false;
        for event in self.session.poll() {
            match event {
                Event::Joined(player) if player != self.remote => {
                    self.problem = Some(format!("BOTH PLAYERS CLAIM PLAYER {}", player + 1));
                }
                Event::Joined(_) => {}
                Event::Track { sample, waiting } => {
                    world.record(self.remote, sample);
                    self.remote_latest = sample;
                    self.remote_waiting = waiting;
                    changed = true;
                }
                Event::Fire { at, vel, ammo } => {
                    world.launch(self.remote, at, vel, ammo);
                    changed = true;
                }
                Event::Flash(at) => dynamics.emit_flash(at),
                Event::Left => {
                    world.leave(self.remote);
                    changed = true;
                }
                Event::Incompatible { theirs } => {
                    self.problem = Some(format!("OTHER PLAYER RUNS PROTOCOL V{theirs}"));
                }
                Event::Error(_) => {}
            }
        }
        changed
    }

    /// Whether the local ship may move from `current` to the tentative `next` event: never
    /// into the other player's future cone, as far as we know it (see [`Pacer`]). Once the
    /// other player has left, nothing holds us back.
    pub fn may_advance(&mut self, current: &ShipSample, next: &ShipSample) -> bool {
        if self.session.peer_left() {
            self.pacer = Pacer::default();
            return true;
        }
        self.pacer.may_advance(current, next, &self.remote_latest)
    }

    /// The local player is held back until the other one catches up.
    pub fn is_waiting(&self) -> bool {
        self.pacer.is_waiting()
    }

    /// Send the local ship's latest event. Sent every frame, also while waiting (a heartbeat
    /// that lets a stalled pair resolve as soon as both know where the other is).
    pub fn send_track(&mut self, sample: ShipSample) {
        let waiting = self.is_waiting();
        self.session.send(&Message::Track { sample, waiting });
    }

    pub fn send_fire(&mut self, at: Frontier, vel: V3, ammo: Ammo) {
        self.session.send(&Message::Fire { at, vel, ammo });
    }

    pub fn send_flash(&mut self, at: Vec4) {
        self.session.send(&Message::Flash { at });
    }

    pub fn leave(&mut self) {
        self.session.leave();
    }

    /// The HUD's line about the match.
    pub fn status(&self) -> String {
        let other = if let Some(problem) = &self.problem {
            problem.clone()
        } else if self.session.peer_left() && !self.session.peer_joined() {
            "COULD NOT CONNECT".to_string()
        } else if self.session.peer_left() {
            "LEFT".to_string()
        } else if !self.session.peer_joined() {
            "NOT CONNECTED YET".to_string()
        } else if self.remote_waiting {
            "WAITING FOR YOU".to_string()
        } else {
            format!("LAST HEARD AT TAU {:+.1}S", self.remote_latest.tau)
        };
        format!("PLAYER {} OF {PLAYERS} | OTHER PLAYER: {other}", self.local() + 1)
    }

    /// A message to show across the screen, if the local player cannot play right now.
    pub fn banner(&self) -> Option<&'static str> {
        self.is_waiting().then_some(WAITING_BANNER)
    }
}
