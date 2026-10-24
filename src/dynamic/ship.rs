//! Ships: the input-driven bodies. Each player's camera rides the center of one ship, whose
//! worldline is only known up to that player's current event. In a multiplayer match every
//! ship's latest event is the apex of a future light cone the world may not grow into.

use super::billiards::SHIP_RADIUS;
use super::collide::{self, V3};
use crate::mesh4d::{Mesh3D, WorldlineSample};
use crate::minkowski::{Tetrad, Vec3, Vec4};
use crate::renderer::Frontier;

/// Index of a player (and of their ship) in a match. Single-player worlds have only player 0.
pub type PlayerId = u8;

/// One recorded event of a ship's (camera's) worldline.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ShipSample {
    /// Lab time
    pub t: f64,
    pub pos: V3,
    /// Lab 3-velocity (`c = 1`)
    pub vel: V3,
    /// The ship's proper time (its player's clock)
    pub tau: f64,
}

impl ShipSample {
    pub fn new(frontier: Frontier, u: Vec4) -> Self {
        let o = frontier.origin;
        Self {
            t: o.t as f64,
            pos: [o.x as f64, o.y as f64, o.z as f64],
            vel: [(u.x / u.t) as f64, (u.y / u.t) as f64, (u.z / u.t) as f64],
            tau: frontier.proper_time as f64,
        }
    }

    /// A ship at rest at `event`, its clock reading zero.
    pub fn at_rest(event: Vec4) -> Self {
        let frontier = Frontier { origin: event, proper_time: 0.0 };
        Self::new(frontier, Vec4::new(1.0, 0.0, 0.0, 0.0))
    }

    /// The event, in `f32` render coordinates.
    pub fn event(&self) -> Vec4 {
        let [x, y, z] = self.pos;
        Vec4::new(self.t as f32, x as f32, y as f32, z as f32)
    }

    /// The event as a frontier apex, with the ship's clock reading.
    pub fn frontier(&self) -> Frontier {
        Frontier { origin: self.event(), proper_time: self.tau as f32 }
    }

    /// The 4-velocity `gamma (1, v)`, in `f64`.
    pub fn four_velocity(&self) -> [f64; 4] {
        let g = collide::gamma(self.vel);
        [g, g * self.vel[0], g * self.vel[1], g * self.vel[2]]
    }

    /// Render-side worldline sample.
    pub fn sample(&self) -> WorldlineSample {
        let [vx, vy, vz] = self.vel;
        WorldlineSample {
            center: self.event(),
            proper_time: self.tau as f32,
            phi: Vec3::default(),
            proper_accel_local: Vec3::default(),
            tetrad: Tetrad::from_beta(Vec3::new(vx as f32, vy as f32, vz as f32)),
        }
    }
}

/// A player's ship: a heavy sphere of radius [`SHIP_RADIUS`] around their camera, with the
/// worldline recorded so far.
pub struct Ship {
    pub player: PlayerId,
    /// Recorded worldline, oldest first, pruned behind the history window; never empty.
    pub track: Vec<ShipSample>,
    /// The player left the match: the ship ends at its last sample, and no longer holds the
    /// frontier back.
    pub left: bool,
    /// Projectiles fired so far (they seed the projectiles' uids).
    pub shots: u64,
}

impl Ship {
    pub fn new(player: PlayerId, spawn: ShipSample) -> Self {
        Self { player, track: vec![spawn], left: false, shots: 0 }
    }

    pub fn latest(&self) -> &ShipSample {
        self.track.last().expect("ship tracks are never empty")
    }

    /// The ship is still an input source: its future is open.
    pub fn is_live(&self) -> bool {
        !self.left
    }

    /// Record the ship's next event. Samples are only ever appended, so whatever has been read
    /// from the track stays true. A paused or waiting player repeats its event; a repeat is
    /// dropped unless the velocity changed on the spot (a collision kick).
    pub fn record(&mut self, sample: ShipSample) {
        let last = self.latest();
        if sample.t < last.t || *last == sample {
            return;
        }
        self.track.push(sample);
    }

    /// The track is recorded up to lab time `t`.
    pub fn is_known_at(&self, t: f64) -> bool {
        self.latest().t >= t
    }

    /// The ship's state at lab time `t` (within the recorded track). Between two samples the
    /// ship moved in a straight line at the later sample's velocity: the camera applies a
    /// frame's thrust before it moves.
    pub fn at(&self, t: f64) -> ShipSample {
        let i = self.track.partition_point(|s| s.t < t);
        let Some(&b) = self.track.get(i) else {
            return *self.latest();
        };
        if i == 0 || b.t == t {
            return b;
        }
        let a = self.track[i - 1];
        let f = (t - a.t) / (b.t - a.t);
        let pos = [0, 1, 2].map(|c| a.pos[c] + (b.pos[c] - a.pos[c]) * f);
        ShipSample { t, pos, vel: b.vel, tau: a.tau + (b.tau - a.tau) * f }
    }

    /// Drop samples older than `horizon`, keeping at least the latest.
    pub fn prune(&mut self, horizon: f64) {
        let stale = self.track.partition_point(|s| s.t < horizon);
        self.track.drain(..stale.min(self.track.len() - 1));
    }

    /// What other players see: a clock cube (so they can read this player's proper time)
    /// whose corners reach the contact radius.
    pub fn mesh() -> Mesh3D {
        Mesh3D::cube(Vec3::splat(SHIP_RADIUS as f32 / 3f32.sqrt()))
    }

    /// Clock-face material: cyan bezel for player 0, coral for player 1, and so on.
    pub fn material(&self) -> usize {
        1 + self.player as usize % 4
    }

    /// Track samples worth drawing: the ends, the last sample of every straight (coasting)
    /// stretch, and while thrusting at most one sample per [`THRUST_SAMPLE_DT`] (a ship
    /// records one sample per frame), plus one every `max_segment_dt`.
    pub fn render_indices(&self, max_segment_dt: f64) -> Vec<usize> {
        let track = &self.track;
        let mut keep = vec![0];
        for i in 1..track.len() {
            let last = *keep.last().unwrap();
            let turned = track[i].vel != track[i - 1].vel;
            if turned && last != i - 1 && track[i - 1].vel == track[last].vel {
                keep.push(i - 1); // end of a straight stretch
            }
            let dt = track[i].t - track[*keep.last().unwrap()].t;
            let bent = track[i].vel != track[*keep.last().unwrap()].vel;
            if i == track.len() - 1 || (bent && dt >= THRUST_SAMPLE_DT) || dt >= max_segment_dt {
                keep.push(i);
            }
        }
        keep
    }
}

/// Lab-time spacing of a thrusting ship's rendered samples.
const THRUST_SAMPLE_DT: f64 = 0.25;
