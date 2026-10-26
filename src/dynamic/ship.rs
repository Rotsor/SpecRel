//! Ships: the input-driven bodies. Each player's camera is the ship's pilot, whose worldline is
//! only known up to that player's current event. In a multiplayer match every pilot's latest
//! event is the apex of a future light cone the world may not grow into.
//!
//! A ship has two worldlines (see `docs/SHIP_CONTROLS_CAUSALITY.md`): the pilot's, where the
//! decisions are made, and the hull's, the rigid body that lasers hit. With delayed commands the
//! pilot sits at the hull's center; in a control pod they float inside the hull.

use super::billiards::SHIP_RADIUS;
use super::collide::{self, V3};
use super::helm::{command_delay, Command, ControlMode, Hull};
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
    /// [`ControlMode::Delayed`]: the orders the pilot had issued by its latest event that the
    /// hull had not executed yet, as far as they have been heard of. They fix the hull's next
    /// [`command_delay`] of motion.
    orders: Vec<Command>,
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
            orders: Vec::new(),
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
        // Orders due by the last event were executed before the step to `pilot`.
        let executed = self.latest().tau as f32;
        self.orders.retain(|c| c.due > executed);
        self.track.push(pilot);
        self.hull.push(hull);
    }

    /// Hear of `orders` the pilot issued (in order) by its latest event.
    pub fn hear_orders(&mut self, orders: &[Command]) {
        self.orders.extend_from_slice(orders);
    }

    /// The hull center's motion past its latest sample, as far as it is already settled by
    /// what has been heard of: with delayed controls, the orders on their way across the hull
    /// ([`command_delay`] ahead of the pilot's latest event); in a control pod, as long as the
    /// hull is still steering by pilot events already recorded (up to [`pod_delay`]
    /// ahead). That covers the hull's events outside the future cone of the pilot's latest
    /// event (all but the far corners of a pod flung out by a hard reversal), so drawing it
    /// shows the whole hull to anyone outside that cone; the recorded track alone ends at the
    /// pilot's lab time, short of the hull's near side.
    ///
    /// Only for drawing: the world itself (contacts, laser hits) waits for recorded samples.
    ///
    /// [`pod_delay`]: super::helm::pod_delay
    pub fn hull_ahead(&self) -> Vec<ShipSample> {
        if self.left {
            return Vec::new();
        }
        match self.controls {
            ControlMode::Delayed => self.delayed_ahead(),
            ControlMode::Pod => self.pod_ahead(),
        }
    }

    /// Fly the hull on through the orders still due, as its pilot would: each frame first
    /// executes what has come due, then moves.
    fn delayed_ahead(&self) -> Vec<ShipSample> {
        let last = self.latest_hull();
        let mut tetrad = Tetrad::rest();
        tetrad.boost_to(last.u());
        let (mut event, mut tau) = (last.event(), last.tau as f32);
        let horizon = tau + command_delay();
        let mut orders = self.orders.iter().peekable();
        let mut ahead = Vec::new();
        while tau < horizon {
            while let Some(c) = orders.next_if(|c| c.due <= tau) {
                c.order.execute(&mut tetrad, c.rapidity);
            }
            let dtau = AHEAD_STEP.min(horizon - tau);
            event += tetrad.e[0] * dtau;
            tau += dtau;
            ahead.push(ShipSample::new(Frontier { origin: event, proper_time: tau }, tetrad.e[0]));
        }
        ahead
    }

    /// Steer the hull on while everything it steers by is recorded.
    fn pod_ahead(&self) -> Vec<ShipSample> {
        let mut hull = self.helm;
        let mut ahead = Vec::new();
        while hull.steering_is_known(self) && ahead.len() < MAX_AHEAD_STEPS {
            let accel = hull.steering(self);
            hull.advance_to(hull.event.t + AHEAD_STEP * hull.u.t, accel);
            ahead.push(hull.sample());
        }
        ahead
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
    ///
    /// The bars' end faces are left out: the three bars meeting at a corner overlap in a cube
    /// there, and each bar's end face lies within the side faces of the other two, so the cage
    /// looks the same without them. That saves a third of its triangles, and of the simplices
    /// of its worldtube, the bulk of a scene with another ship in it.
    pub fn hull_mesh() -> Mesh3D {
        let mut cage = Mesh3D::default();
        for (axis, (mut bar, _)) in Self::hull_bars().into_iter().enumerate().map(|(i, b)| (i / 4, b)) {
            bar.triangles.retain(|tri| [tri.normal.x, tri.normal.y, tri.normal.z][axis] == 0.0);
            append(&mut cage, &bar, Vec3::default());
        }
        cage
    }

    /// The cage's bars, in place in the hull, each with its center: four along each axis in
    /// turn (x, y, z). The cage is not convex, so its tube's end is capped bar by bar, each
    /// fanned to its own center (with its end faces: the fan to them fills the corners).
    pub fn hull_bars() -> Vec<(Mesh3D, Vec3)> {
        let h = SHIP_RADIUS as f32 / 3f32.sqrt();
        let b = 0.018;
        let mut bars = Vec::new();
        for axis in 0..3 {
            for (i, j) in [(-1.0, -1.0), (-1.0, 1.0), (1.0, -1.0), (1.0, 1.0)] {
                let mut center = [0.0; 3];
                center[(axis + 1) % 3] = i * h;
                center[(axis + 2) % 3] = j * h;
                let mut he = [b; 3];
                he[axis] = h + b;
                let [cx, cy, cz] = center;
                let (mut bar, center) = (Mesh3D::default(), Vec3::new(cx, cy, cz));
                append(&mut bar, &Mesh3D::cube(Vec3::new(he[0], he[1], he[2])), center);
                bars.push((bar, center));
            }
        }
        bars
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

/// Proper-time step of [`Ship::hull_ahead`]'s integration (about a frame).
const AHEAD_STEP: f32 = 1.0 / 30.0;

/// A bound on [`Ship::hull_ahead`] in a control pod, a little over [`pod_delay`]'s worth of
/// steps.
///
/// [`pod_delay`]: super::helm::pod_delay
const MAX_AHEAD_STEPS: usize = 40;

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
///
/// "Straight" means the drawn segment strays at most [`MAX_STRAY`] from the track (as
/// [`segment_stray`] measures it, bounded by the spread of the velocities along the segment): a
/// control pod's hull settles onto its pilot only asymptotically, so its velocity never
/// repeats exactly, and an exact test would draw it as if it thrusted all along. (Each drawn
/// segment of the hull's cage costs ~300 simplices.)
pub fn render_indices(track: &[ShipSample], max_segment_dt: f64) -> Vec<usize> {
    RenderScan::default().finish(track, max_segment_dt)
}

/// [`render_indices`]' scan of a track, resumable: a track is only ever appended to (and
/// pruned at the front), so the choices made over a prefix of it stand, and the scene cache
/// picks the scan up where it left off instead of redoing it. Picking it up after the track
/// was pruned ([`RenderScan::realign`]) also keeps the choices made before, where a fresh
/// scan from the new first sample would choose afresh all along (and the cache would have to
/// redraw the whole track).
#[derive(Clone, Debug)]
pub struct RenderScan {
    /// The samples kept so far (indices into the track).
    pub keep: Vec<usize>,
    /// The box spanned by the velocities since the last kept sample (its own included; empty
    /// until the scan looks at it), and their largest speed. A segment whose velocities span
    /// `range` strays from the track by at most `|range| dt / 4` (its velocity is within the
    /// box), and its rest frame from the linear blend by at most `|range|`.
    lo: V3,
    hi: V3,
    fastest: f64,
    /// The next sample to look at.
    next: usize,
}

impl Default for RenderScan {
    fn default() -> Self {
        Self { keep: vec![0], lo: [f64::INFINITY; 3], hi: [f64::NEG_INFINITY; 3], fastest: 0.0, next: 1 }
    }
}

impl RenderScan {
    /// Look at sample `i` of `track` (and at the one after it, if any: otherwise `i` is the
    /// track's last).
    fn step(&mut self, track: &[ShipSample], i: usize, max_segment_dt: f64) {
        let speed = |s: &ShipSample| s.vel.iter().map(|v| v * v).sum::<f64>().sqrt();
        let grow = |lo: &mut V3, hi: &mut V3, s: &ShipSample| {
            for c in 0..3 {
                lo[c] = lo[c].min(s.vel[c]);
                hi[c] = hi[c].max(s.vel[c]);
            }
        };
        let last = *self.keep.last().unwrap();
        if self.lo[0] > self.hi[0] {
            (self.lo, self.hi, self.fastest) = (track[last].vel, track[last].vel, speed(&track[last]));
        }
        grow(&mut self.lo, &mut self.hi, &track[i]);
        self.fastest = self.fastest.max(speed(&track[i]));
        let dt = track[i].t - track[last].t;
        let Some(next) = track.get(i + 1) else {
            self.keep.push(i);
            return;
        };
        // Would the segment still be straight if it ran on to the next sample?
        let (mut lo_next, mut hi_next) = (self.lo, self.hi);
        grow(&mut lo_next, &mut hi_next, next);
        let spread = (0..3).map(|c| (hi_next[c] - lo_next[c]).powi(2)).sum::<f64>().sqrt();
        let radius = SHIP_RADIUS * self.fastest.max(speed(next)) / 2.0;
        let strays = spread * ((next.t - track[last].t) / 4.0 + radius) > MAX_STRAY;
        if (strays && dt >= THRUST_SAMPLE_DT) || dt >= max_segment_dt {
            self.keep.push(i);
            (self.lo, self.hi, self.fastest) = (track[i].vel, track[i].vel, speed(&track[i]));
        }
    }

    /// Scan on up to (not including) sample `end`, which must not be the track's last (a
    /// sample's choice looks at the one after it): these choices stand however the track goes
    /// on.
    pub fn feed(&mut self, track: &[ShipSample], end: usize, max_segment_dt: f64) {
        debug_assert!(end < track.len());
        while self.next < end {
            self.step(track, self.next, max_segment_dt);
            self.next += 1;
        }
    }

    /// Scan the rest of `track`, and return the samples to draw.
    pub fn finish(mut self, track: &[ShipSample], max_segment_dt: f64) -> Vec<usize> {
        while self.next < track.len() {
            self.step(track, self.next, max_segment_dt);
            self.next += 1;
        }
        self.keep
    }

    /// The last sample looked at (to [`realign`](RenderScan::realign) by later).
    pub fn last_seen<'a>(&self, track: &'a [ShipSample]) -> &'a ShipSample {
        &track[self.next - 1]
    }

    /// Carry the scan over to `track`, which the scanned track may have become by being pruned
    /// at the front and appended to; `seen` is the scan's [`last_seen`](RenderScan::last_seen)
    /// sample of the scanned track. Choices before the new first sample are dropped, and that
    /// sample is kept. Starts afresh if `track` did not come about that way (a rebase, a new
    /// world).
    pub fn realign(&mut self, seen: &ShipSample, track: &[ShipSample]) {
        let j = track.partition_point(|s| s.t < seen.t);
        let found = track[j..].iter().take_while(|s| s.t == seen.t).position(|s| s == seen).map(|k| j + k);
        let Some(j) = found.filter(|&j| j < self.next) else {
            *self = Self::default();
            return;
        };
        let shift = self.next - 1 - j;
        if shift == 0 {
            return;
        }
        self.next -= shift;
        self.keep.retain(|&k| k >= shift);
        for k in &mut self.keep {
            *k -= shift;
        }
        if self.keep.first() != Some(&0) {
            self.keep.insert(0, 0);
        }
    }
}

/// How far the segment of `track` drawn straight from sample `a` to sample `b` strays from it,
/// at most, at a sample in between (between samples the track is straight): the center's
/// distance from the segment, plus how far the hull's corners stray off its surface as the
/// segment blends its ends' rest frames linearly, `|v| |dv| / 2` per unit of radius for a
/// velocity `dv` off the blend (the rest of `dv` slides them along the hull's worldtube).
pub fn segment_stray(track: &[ShipSample], a: usize, b: usize) -> f64 {
    let (sa, sb) = (&track[a], &track[b]);
    let span = (sb.t - sa.t).max(1e-12);
    let dist = |x: V3, y: V3| (0..3).map(|c| (x[c] - y[c]).powi(2)).sum::<f64>().sqrt();
    track[a + 1..b]
        .iter()
        .map(|s| {
            let f = (s.t - sa.t) / span;
            let lerp = |x: V3, y: V3| [0, 1, 2].map(|c| x[c] + (y[c] - x[c]) * f);
            let speed = dist(s.vel, [0.0; 3]);
            dist(s.pos, lerp(sa.pos, sb.pos)) + SHIP_RADIUS * speed * dist(s.vel, lerp(sa.vel, sb.vel)) / 2.0
        })
        .fold(0.0, f64::max)
}

/// Lab-time spacing of a thrusting ship's rendered samples.
const THRUST_SAMPLE_DT: f64 = 0.25;

/// How far (`c = 1` units) a drawn ship segment may stray from the recorded track (unless it
/// would be shorter than [`THRUST_SAMPLE_DT`]): well within the hull's 0.036-thick bars.
const MAX_STRAY: f64 = 0.005;

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

    /// A hull settling onto a coasting pod: its velocity creeps toward the pod's forever,
    /// never repeating exactly. Once the drift is negligible it is drawn in long straight
    /// segments, not sampled every `THRUST_SAMPLE_DT` across the whole history.
    #[test]
    fn a_settling_hull_is_drawn_straight() {
        let mut t = 0.0;
        let track: Vec<_> = (0..12_000)
            .map(|i| {
                t += 1.0 / 60.0;
                let v = 0.3 + 0.1 * (-(i as f64) / 60.0).exp();
                ShipSample { t, pos: [0.3 * t, 0.0, 0.0], vel: [v, 0.0, 0.0], tau: t }
            })
            .collect();
        let keep = render_indices(&track, 50.0);
        // Sampling the 200 s throughout would keep 800; settling takes the first few seconds.
        assert!(keep.len() < 30, "{} samples drawn", keep.len());
        assert_eq!(*keep.last().unwrap(), track.len() - 1);
        // Never a burst of samples as the drift crosses the tolerance (save the track's end).
        for w in keep[..keep.len() - 1].windows(2) {
            assert!(track[w[1]].t - track[w[0]].t >= THRUST_SAMPLE_DT, "{w:?}");
        }
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

    /// A cap fanned to a pivot fills the mesh only if the pivot is behind every face. That
    /// fails for the cage and its center (the fan would fill the open space between them), and
    /// holds for each bar and its own center.
    #[test]
    fn the_cage_is_capped_bar_by_bar() {
        let behind_all = |mesh: &Mesh3D, pivot: Vec3| {
            mesh.triangles.iter().all(|tri| (pivot - mesh.vertices[tri.indices[0]]).dot(tri.normal) < 0.0)
        };
        assert!(!behind_all(&Ship::hull_mesh(), Vec3::default()));
        let bars = Ship::hull_bars();
        assert_eq!(bars.len(), 12);
        for (bar, center) in &bars {
            assert!(behind_all(bar, *center), "bar at {center:?}");
        }
    }

    #[test]
    fn the_hull_cage_is_open() {
        let cage = Ship::hull_mesh();
        assert_eq!(cage.triangles.len(), 12 * 8, "the bars' sides only");
        let h = SHIP_RADIUS as f32 / 3f32.sqrt();
        // No vertex near a face center: you can see through every face.
        for v in &cage.vertices {
            let on_axis = [v.x, v.y, v.z].iter().filter(|c| c.abs() < 0.5 * h).count();
            assert!(on_axis <= 1, "{v:?}");
        }
    }

    /// The bars' end faces, left out of the cage, are covered by the side faces of the bars
    /// they meet: from outside the corners look the same.
    #[test]
    fn the_cage_needs_no_bar_ends() {
        use crate::mesh4d::Triangle3D;
        let cage = Ship::hull_mesh();
        let tri = |m: &Mesh3D, t: &Triangle3D| t.indices.map(|i| m.vertices[i]);
        // `p` on the triangle `[a, b, c]` with normal `n`, up to rounding.
        let on = |p: Vec3, [a, b, c]: [Vec3; 3], n: Vec3| {
            let edge = |u: Vec3, v: Vec3| (v - u).cross(p - u).dot(n);
            let e = [edge(a, b), edge(b, c), edge(c, a)];
            (p - a).dot(n).abs() < 1e-6 && (e.iter().all(|&e| e > -1e-7) || e.iter().all(|&e| e < 1e-7))
        };
        let mut ends = 0;
        for (axis, (bar, _)) in Ship::hull_bars().iter().enumerate().map(|(i, b)| (i / 4, b)) {
            for end in bar.triangles.iter().filter(|t| [t.normal.x, t.normal.y, t.normal.z][axis] != 0.0) {
                ends += 1;
                let [a, b, c] = tri(bar, end);
                for (i, j) in (0..=8).flat_map(|i| (0..=8 - i).map(move |j| (i, j))) {
                    let (u, v) = (i as f32 / 8.0, j as f32 / 8.0);
                    let p = a + (b - a) * u + (c - a) * v;
                    let covered = cage.triangles.iter().any(|t| t.normal == end.normal && on(p, tri(&cage, t), t.normal));
                    assert!(covered, "{p:?} on the end of a bar along axis {axis}");
                }
            }
        }
        assert_eq!(ends, 12 * 4);
    }
}
