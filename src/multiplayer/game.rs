//! One player's side of a two-player match: the session to the other player, what arrives
//! over it applied to this player's replica of the world, and the pacing that keeps this
//! player out of the other's future light cone.

use super::link::Link;
use super::pacing::{self, Pacer};
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

/// The HUD banner until the other player has joined: the match starts paused.
pub const NOT_JOINED_BANNER: &str = "WAITING FOR A PLAYER TO JOIN";

/// Wall-clock seconds between round-trip probes.
const PING_INTERVAL: f64 = 0.5;
/// How much of each new round-trip sample goes into the smoothed round trip.
const RTT_SMOOTHING: f64 = 0.25;
/// Wall-clock seconds over which the HUD averages the share of time spent waiting.
const WAIT_WINDOW: f64 = 5.0;

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
    /// The local ship's latest event, as last sent.
    local_latest: Option<ShipSample>,
    /// Wall-clock seconds since the match started, as the frontend measures them.
    clock: f64,
    next_ping: f64,
    /// Smoothed wall-clock round trip to the other player and back, frame times included.
    rtt: Option<f64>,
    /// The share of the last [`WAIT_WINDOW`] or so that the local player spent waiting.
    wait_share: f64,
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
            local_latest: None,
            clock: 0.0,
            next_ping: 0.0,
            rtt: None,
            wait_share: 0.0,
        }
    }

    pub fn local(&self) -> PlayerId {
        self.session.local()
    }

    /// Advance the wall clock by `secs` (the real frame time, unclamped), and probe the round
    /// trip every [`PING_INTERVAL`] once the other player has joined.
    pub fn tick(&mut self, secs: f64) {
        self.clock += secs;
        let waiting = if self.is_waiting() { 1.0 } else { 0.0 };
        self.wait_share += (waiting - self.wait_share) * (1.0 - (-secs / WAIT_WINDOW).exp());
        let live = self.session.peer_joined() && !self.session.peer_left();
        if live && self.clock >= self.next_ping {
            self.session.send(&Message::Ping { sent: self.clock });
            self.next_ping = self.clock + PING_INTERVAL;
        }
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
                Event::Ping { sent } => self.session.send(&Message::Pong { sent }),
                Event::Pong { sent } => {
                    let sample = self.clock - sent;
                    self.rtt = Some(self.rtt.map_or(sample, |r| r + (sample - r) * RTT_SMOOTHING));
                }
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

    /// Whether the local ship may move from `current` to the tentative `next` event: not at
    /// all until the other player has joined, then never into their future cone, as far as we
    /// know it (see [`Pacer`]). Once the other player has left, nothing holds us back.
    pub fn may_advance(&mut self, current: &ShipSample, next: &ShipSample) -> bool {
        if self.session.peer_left() {
            self.pacer = Pacer::default();
            return true;
        }
        if self.awaiting_peer() {
            return false;
        }
        self.pacer.may_advance(current, next, &self.remote_latest)
    }

    /// The other player has not joined yet (and has not given up either).
    fn awaiting_peer(&self) -> bool {
        !self.session.peer_joined() && !self.session.peer_left()
    }

    /// The local player is held back: until the other one joins, then until they catch up.
    pub fn is_waiting(&self) -> bool {
        self.awaiting_peer() || self.pacer.is_waiting()
    }

    /// Send the local ship's latest event. Sent every frame, also while waiting (a heartbeat
    /// that lets a stalled pair resolve as soon as both know where the other is).
    pub fn send_track(&mut self, sample: ShipSample) {
        let waiting = self.is_waiting();
        self.session.send(&Message::Track { sample, waiting });
        self.local_latest = Some(sample);
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

    /// The HUD's line of latency figures, while the other player is in the match: the
    /// wall-clock round trip; and, between the two ships' latest known events, the CM time by
    /// which we lead, the invariant interval, and the CM distance at equal CM time.
    pub fn metrics(&self) -> Option<String> {
        if !self.session.peer_joined() || self.session.peer_left() {
            return None;
        }
        let rtt = self.rtt.map_or("--".to_string(), |r| format!("{:.0}MS", r * 1000.0));
        let Some(local) = &self.local_latest else {
            return Some(format!("NET RTT {rtt}"));
        };
        let lead = pacing::lead(local, &self.remote_latest);
        let sep = pacing::separation(local, &self.remote_latest);
        // Timelike only when the ships are within a step of each other (see `Pacer`).
        let interval = if sep.interval2 >= 0.0 {
            format!("{:.2}LS", sep.interval2.sqrt())
        } else {
            format!("TIMELIKE {:.2}S", (-sep.interval2).sqrt())
        };
        Some(format!(
            "NET RTT {rtt} | CM LEAD {:+.3}S | INTERVAL {interval} | CM DIST {:.2}LS | WAITING {:.0}%",
            lead.dt,
            sep.dist,
            self.wait_share * 100.0,
        ))
    }

    /// A message to show across the screen, if the local player cannot play right now.
    pub fn banner(&self) -> Option<&'static str> {
        if self.awaiting_peer() {
            Some(NOT_JOINED_BANNER)
        } else {
            self.is_waiting().then_some(WAITING_BANNER)
        }
    }
}
