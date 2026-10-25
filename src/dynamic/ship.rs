//! Ships: the input-driven bodies. Each player's camera is the ship's pilot, whose worldline is
//! only known up to that player's current event. In a multiplayer match every pilot's latest
//! event is the apex of a future light cone the world may not grow into.
//!
//! A ship has two worldlines (see `docs/SHIP_CONTROLS_CAUSALITY.md`): the pilot's, where the
//! decisions are made, and the hull's, the rigid body that lasers hit. With delayed commands the
//! pilot sits at the hull's center; in a control pod they float inside the hull.

use super::billiards::SHIP_RADIUS;
use super::collide::{self, V3};
use super::helm::{ControlMode, Hull};
use crate::mesh4d::{Mesh3D, WorldlineSample};
use crate::minkowski::{Tetrad, Vec3, Vec4};
use crate::renderer::Frontier;

/// Index of a player (and of their ship) in a match. Single-player worlds have only player 0.
pub type PlayerId = u8;

/// One recorded event of a ship's worldline (its pilot's or its hull's center).
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

    /// The 4-velocity in `f32` render coordinates.
    pub fn u(&self) -> Vec4 {
        let [t, x, y, z] = self.four_velocity().map(|c| c as f32);
        Vec4::new(t, x, y, z)
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

/// A player's ship: a heavy sphere of radius [`SHIP_RADIUS`] (the hull) with its pilot (the
/// player's camera) inside, and both worldlines recorded so far.
///
/// Only the pilot's track is input. The hull's track is a pure function of it, so every
/// replica computes the same hull.
pub struct Ship {
    pub player: PlayerId,
    /// The pilot's recorded worldline, oldest first, pruned behind the history window; never
    /// empty. Shots and flashes leave from here, and its latest event bounds the frontier.
    pub track: Vec<ShipSample>,
    /// The hull center's worldline: `hull[i]` is at `track[i]`'s lab time. Laser pulses hit
    /// the hull.
    pub hull: Vec<ShipSample>,
    /// How the controls reach the hull.
    pub controls: ControlMode,
    /// [`ControlMode::Pod`]: the hull's state at its latest sample, at full (`f32`) precision.
    helm: Hull,
    /// The player left the match: the ship ends at its last sample, and no longer holds the
    /// frontier back.
    pub left: bool,
    /// Laser pulses fired so far (they number the pulses).
    pub shots: u64,
}

impl Ship {
    pub fn new(player: PlayerId, spawn: ShipSample) -> Self {
        Self {
            player,
            track: vec![spawn],
            hull: vec![spawn],
            controls: ControlMode::default(),
            helm: Hull::from_sample(&spawn),
            left: false,
            shots: 0,
        }
    }

    /// Switch the controls. Only while the ship is still at its spawn event: the pilot and the
    /// hull's center must coincide.
    pub fn set_controls(&mut self, controls: ControlMode) {
        debug_assert_eq!(self.track.len(), 1, "controls change only before the ship moves");
        self.controls = controls;
    }

    pub fn latest(&self) -> &ShipSample {
        self.track.last().expect("ship tracks are never empty")
    }

    pub fn latest_hull(&self) -> &ShipSample {
        self.hull.last().expect("ship tracks are never empty")
    }

    /// The ship is still an input source: its future is open.
    pub fn is_live(&self) -> bool {
        !self.left
    }

    /// Record the pilot's next event, and move the hull's center to the same lab time.
    /// Samples are only ever appended, so whatever has been read from the tracks stays true.
    /// A paused or waiting player repeats its event; a repeat is dropped unless the velocity
    /// changed on the spot (a delayed command coming due while paused).
    ///
    /// With delayed controls the hull's center is the pilot. A control pod's hull steers
    /// toward the pod by what it knows ([`Hull::steering`]), in pure `f32` arithmetic with
    /// `sqrt` as the only function, so every replica gets the same bits: its motion depends
    /// on its pilot's track alone.
    pub fn record(&mut self, pilot: ShipSample) {
        let last = self.latest();
        if pilot.t < last.t || *last == pilot {
            return;
        }
        let hull = match self.controls {
            ControlMode::Delayed => pilot,
            ControlMode::Pod => {
                let accel = self.helm.steering(self);
                self.helm.advance_to(pilot.t as f32, accel);
                ShipSample { t: pilot.t, ..self.helm.sample() } // exactly the pilot's lab time
            }
        };
        self.track.push(pilot);
        self.hull.push(hull);
    }

    /// The tracks are recorded up to lab time `t`.
    pub fn is_known_at(&self, t: f64) -> bool {
        self.latest().t >= t
    }

    /// Shift every event by `dt` in lab time.
    pub fn rebase(&mut self, dt: f64) {
        for s in self.track.iter_mut().chain(&mut self.hull) {
            s.t += dt;
        }
        self.helm.event.t += dt as f32;
    }

    /// The hull's center at lab time `t`, as well as the pilot can know it now: its recorded
    /// state, or, past the recorded track, its latest state coasted on. Only for the pilot's
    /// own decisions (the HUD); the world uses [`Ship::hull_at`].
    pub fn hull_estimate(&self, t: f64) -> Hull {
        let at = if self.is_known_at(t) { self.hull_at(t) } else { *self.latest_hull() };
        let mut hull = Hull::from_sample(&at);
        hull.advance_to(t as f32, Vec4::default());
        hull
    }

    /// The pilot's state at lab time `t` (within the recorded track).
    pub fn at(&self, t: f64) -> ShipSample {
        state_at(&self.track, t)
    }

    /// The hull center's state at lab time `t` (within the recorded track). The hull is rigid
    /// and moves causally (it only acts on commands that have crossed all of it), so every
    /// point of it is where this puts it.
    pub fn hull_at(&self, t: f64) -> ShipSample {
        state_at(&self.hull, t)
    }

    /// The hull center's state when its clock read `tau`. Before the recorded track it was at
    /// rest where the track starts (where it spawned, or the best known).
    pub fn hull_at_tau(&self, tau: f64) -> ShipSample {
        let i = self.hull.partition_point(|s| s.tau < tau);
        match (i, self.hull.get(i)) {
            (0, _) => self.hull[0],
            (_, None) => *self.latest_hull(),
            (_, Some(b)) => {
                let a = self.hull[i - 1];
                self.hull_at(a.t + (b.t - a.t) * (tau - a.tau) / (b.tau - a.tau).max(1e-12))
            }
        }
    }

    /// The pilot's event on the hyperplane through `event` orthogonal to the 4-velocity `u`
    /// (simultaneous with `event` for an observer moving with `u`), clamped to the track.
    pub fn pilot_on_slice(&self, event: Vec4, u: Vec4) -> ShipSample {
        // Along a timelike worldline `<P(t) - event, u>` decreases: it is positive before the
        // slice.
        let before = |s: &ShipSample| (s.event() - event).minkowski_dot(u) > 0.0;
        let i = self.track.partition_point(before);
        if i == 0 || i == self.track.len() {
            return self.track[i.min(self.track.len() - 1)];
        }
        let (mut lo, mut hi) = (self.track[i - 1].t, self.track[i].t);
        for _ in 0..40 {
            let mid = 0.5 * (lo + hi);
            if before(&self.at(mid)) {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        self.at(hi)
    }

    /// Drop samples older than `horizon`, keeping at least the latest.
    pub fn prune(&mut self, horizon: f64) {
        let stale = self.track.partition_point(|s| s.t < horizon).min(self.track.len() - 1);
        self.track.drain(..stale);
        self.hull.drain(..stale);
    }

    /// What other players see of the hull: an open cage of bars along the edges of a cube whose
    /// corners reach the contact radius, so the pilot inside stays visible.
    pub fn hull_mesh() -> Mesh3D {
        let h = SHIP_RADIUS as f32 / 3f32.sqrt();
        let b = 0.018;
        let mut cage = Mesh3D::default();
        for axis in 0..3 {
            for (i, j) in [(-1.0, -1.0), (-1.0, 1.0), (1.0, -1.0), (1.0, 1.0)] {
                let mut center = [0.0; 3];
                center[(axis + 1) % 3] = i * h;
                center[(axis + 2) % 3] = j * h;
                let mut he = [b; 3];
                he[axis] = h + b;
                let [cx, cy, cz] = center;
                append(&mut cage, &Mesh3D::cube(Vec3::new(he[0], he[1], he[2])), Vec3::new(cx, cy, cz));
            }
        }
        cage
    }

    /// What other players see of the pilot: a small clock cube (so they can read this player's
    /// proper time), where the decisions are made.
    pub fn pilot_mesh() -> Mesh3D {
        Mesh3D::cube(Vec3::splat(PILOT_HALF_EXTENT))
    }

    /// Clock-face material: cyan bezel for player 0, coral for player 1, and so on.
    pub fn material(&self) -> usize {
        1 + self.player as usize % 4
    }

    /// The hull cage's material, in the player's color.
    pub fn hull_material(&self) -> usize {
        HULL_MATERIALS + self.player as usize % 4
    }
}

/// First of the four hull cage materials in `Scene4D::common_materials`.
pub const HULL_MATERIALS: usize = 10;

/// Half the side of the pilot's clock cube. It fits inside the pod's range of motion.
const PILOT_HALF_EXTENT: f32 = 0.07;

/// Add `part`, shifted by `offset`, to `mesh`.
fn append(mesh: &mut Mesh3D, part: &Mesh3D, offset: Vec3) {
    let base = mesh.vertices.len();
    mesh.vertices.extend(part.vertices.iter().map(|&v| v + offset));
    mesh.triangles.extend(part.triangles.iter().map(|tri| {
        let mut tri = *tri;
        tri.indices = tri.indices.map(|i| i + base);
        tri
    }));
}

/// The state on `track` at lab time `t` (within the recorded track). Between two samples the
/// ship moved in a straight line at the later sample's velocity: a frame's thrust applies
/// before the frame's motion.
fn state_at(track: &[ShipSample], t: f64) -> ShipSample {
    let i = track.partition_point(|s| s.t < t);
    let Some(&b) = track.get(i) else {
        return *track.last().expect("ship tracks are never empty");
    };
    if i == 0 || b.t == t {
        return b;
    }
    let a = track[i - 1];
    let f = (t - a.t) / (b.t - a.t);
    let pos = [0, 1, 2].map(|c| a.pos[c] + (b.pos[c] - a.pos[c]) * f);
    ShipSample { t, pos, vel: b.vel, tau: a.tau + (b.tau - a.tau) * f }
}

/// Indices of `track` samples worth drawing: the ends, the last sample of every straight
/// (coasting) stretch, and while thrusting at most one sample per [`THRUST_SAMPLE_DT`] (a ship
/// records one sample per frame), plus one every `max_segment_dt`.
pub fn render_indices(track: &[ShipSample], max_segment_dt: f64) -> Vec<usize> {
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

/// Lab-time spacing of a thrusting ship's rendered samples.
const THRUST_SAMPLE_DT: f64 = 0.25;

#[cfg(test)]
mod tests {
    use super::*;

    /// A ship at rest at the origin until `t = 10`, then at 0.5c along `+x`. The pilot rides
    /// 0.1 ahead of the hull's center.
    fn kicked_ship() -> Ship {
        let mut ship = Ship::new(0, ShipSample { t: 0.0, pos: [0.0; 3], vel: [0.0; 3], tau: 0.0 });
        for i in 1..=200 {
            let t = i as f64 * 0.1;
            let x = 0.5 * (t - 10.0).max(0.0);
            let vel = if t > 10.0 + 1e-9 { [0.5, 0.0, 0.0] } else { [0.0; 3] };
            let tau = t.min(10.0) + (t - 10.0).max(0.0) * 0.75f64.sqrt();
            let hull = ShipSample { t, pos: [x, 0.0, 0.0], vel, tau };
            ship.track.push(ShipSample { pos: [x + 0.1, 0.0, 0.0], ..hull });
            ship.hull.push(hull);
        }
        ship
    }

    #[test]
    fn hull_and_pilot_tracks_are_separate() {
        let ship = kicked_ship();
        assert_eq!(ship.track.len(), ship.hull.len());
        assert!((ship.hull_at(12.0).pos[0] - 1.0).abs() < 1e-9);
        assert!((ship.at(12.0).pos[0] - 1.1).abs() < 1e-9);
        // By its clock: tau = 10 + 2 sqrt(0.75) at t = 12.
        let s = ship.hull_at_tau(10.0 + 2.0 * 0.75f64.sqrt());
        assert!((s.t - 12.0).abs() < 1e-9, "{s:?}");
        assert_eq!(ship.hull_at_tau(-5.0), ship.hull[0], "before the track: where it started");
    }

    #[test]
    fn the_pilot_is_found_on_a_moving_observers_slice() {
        let ship = kicked_ship();
        // An observer at the origin of lab time 15, at rest: simultaneous is lab time 15.
        let rest = Vec4::new(1.0, 0.0, 0.0, 0.0);
        let s = ship.pilot_on_slice(Vec4::new(15.0, 0.0, 0.0, 0.0), rest);
        assert!((s.t - 15.0).abs() < 1e-6, "{s:?}");
        // Moving with the ship at 0.5c from its hull center at t = 15: the pilot, 0.1 ahead in
        // the lab, is simultaneous at t' = 15 + 0.5 * 0.1 / (1 - 0.25) in the lab.
        let hull = ship.hull_at(15.0);
        let p = ship.pilot_on_slice(hull.event(), hull.u());
        let expected = 15.0 + 0.05 / 0.75;
        assert!((p.t - expected).abs() < 1e-4, "{} vs {expected}", p.t);
    }

    #[test]
    fn the_hull_cage_is_open() {
        let cage = Ship::hull_mesh();
        assert_eq!(cage.triangles.len(), 12 * 12);
        let h = SHIP_RADIUS as f32 / 3f32.sqrt();
        // No vertex near a face center: you can see through every face.
        for v in &cage.vertices {
            let on_axis = [v.x, v.y, v.z].iter().filter(|c| c.abs() < 0.5 * h).count();
            assert!(on_axis <= 1, "{v:?}");
        }
    }
}
