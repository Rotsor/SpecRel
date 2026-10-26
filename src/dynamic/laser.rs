//! Lasers: the ships' weapon. A pulse of light leaves the pilot along the view direction and
//! flies at `c` until it meets a puck, a ship's hull or the table, and goes out there. Hitting
//! another player's hull scores a hit.
//!
//! A pulse is the only thing in the world that does not have to wait for the frontier: its
//! whole path lies on the boundary of the firing event's future light cone, so the input that
//! fired it fixes it at once. Only where it *ends* depends on others, and only on bodies and
//! hulls, which never depend on it (light pushes nothing here). So a pulse is advanced on the
//! shared lab-time grid behind them: across a step once every puck and hull that could be in
//! its way during that step is known. Then every replica of a match finds the same end, however
//! late it hears of the other player.

use super::billiards::{self, SHIP_RADIUS};
use super::body::{Body, STEP_DT};
use super::collide::{dot3, V3};
use super::ship::{PlayerId, Ship, ShipSample};
use crate::mesh4d::{build_worldtube_cap, build_worldtube_simplices_with_decided, Mesh3D, Simplex4D, WorldlineSample};
use crate::minkowski::{Tetrad, Vec3, Vec4};

/// Length of the rod of light a pulse is drawn as, trailing its head, in the shooter's rest
/// frame at firing (in light-seconds, so it lasts that long there as well).
pub const PULSE_LENGTH: f32 = 0.5;
/// Most lab time one piece of a pulse's worldtube spans. The worldtube is straight, so cutting
/// it leaves it the same; but in one piece its bounding box would be a square in `(t, along)`
/// as long as its whole flight (up to [`LASER_RANGE`]) around a thin diagonal band, which
/// rays all over the view pass through and must test (most of the CPU trace time, a minute
/// into a match). Cut, each piece's box hugs its stretch of the band. Shorter pieces cut the
/// trace further but grow the BVH build: 8 traces as fast as no pulses at all, for 36 more
/// simplices per 8 s of flight.
const PULSE_SEGMENT_DT: f64 = 8.0;
/// Half the thickness of the rod.
const PULSE_HALF_WIDTH: f32 = 0.02;
/// A pulse that has met nothing this far (lab light-seconds) from where it was fired is gone.
pub const LASER_RANGE: f64 = 120.0;
/// Material of pulses in `Scene4D::common_materials`.
pub const MATERIAL_PULSE: usize = 9;

/// What stopped a pulse.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Target {
    /// A puck, by uid.
    Body(u64),
    /// A player's hull: a hit for the shooter.
    Ship(PlayerId),
    /// The felt or a rail.
    Table,
    /// Nothing within [`LASER_RANGE`].
    Nothing,
}

/// Where and when a pulse went out.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PulseEnd {
    /// Lab time
    pub t: f64,
    pub pos: V3,
    pub target: Target,
}

impl PulseEnd {
    /// The event, in `f32` render coordinates.
    pub fn event(&self) -> Vec4 {
        let [x, y, z] = self.pos;
        Vec4::new(self.t as f32, x as f32, y as f32, z as f32)
    }
}

/// What [`Pulse::advance`] did.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Progress {
    /// Something that might be in the way is not known yet.
    Waiting,
    /// Crossed one more grid step unhindered.
    Advanced,
    /// Went out (or already had).
    Ended,
}

/// A pulse of laser light: its head is at `pos0 + dir (t - t0)`.
#[derive(Clone, Debug, PartialEq)]
pub struct Pulse {
    pub shooter: PlayerId,
    /// The shooter's shot count when it was fired: `(shooter, seq)` identifies it.
    pub seq: u64,
    /// Lab time of the firing event
    pub t0: f64,
    pub pos0: V3,
    /// Unit lab direction of flight
    pub dir: V3,
    /// The shooter's lab velocity when it was fired: the pulse is shaped in that rest frame.
    pub vel: V3,
    /// The viewer's proper time when this replica heard of it. Its whole path was decided by
    /// then (it lies on the firing event's light cone).
    pub fired_at: f32,
    /// The path is known to be clear up to grid step `cleared` (or `t0`, if later).
    cleared: i64,
    /// Where it goes out if nothing moving gets in its way first: the table, or out of range.
    /// Static, so known at once.
    stop: PulseEnd,
    pub end: Option<PulseEnd>,
    /// Its hit on a ship is on the viewer's scoreboard: news of it has reached the viewer.
    pub scored: bool,
}

impl Pulse {
    /// Shot `seq` of `shooter`, fired from `from` along the lab direction `dir`; `t_origin` is
    /// the lab time of grid step 0, and `fired_at` the viewer's proper time now.
    pub fn new(shooter: PlayerId, seq: u64, from: ShipSample, dir: V3, t_origin: f64, fired_at: f32) -> Self {
        let (t0, pos0) = (from.t, from.pos);
        let l = dot3(dir, dir).sqrt();
        let dir = dir.map(|c| c / l);
        Self {
            shooter,
            seq,
            t0,
            pos0,
            dir,
            vel: from.vel,
            fired_at,
            cleared: ((t0 - t_origin) / STEP_DT).floor() as i64,
            stop: static_stop(t0, pos0, dir),
            end: None,
            scored: false,
        }
    }

    /// Where the head is at lab time `t`.
    pub fn pos_at(&self, t: f64) -> V3 {
        let s = t - self.t0;
        [0, 1, 2].map(|c| self.pos0[c] + self.dir[c] * s)
    }

    /// Lab time up to which the pulse is known: where it went out, or how far it is known to
    /// have flown unhindered.
    pub fn known_to(&self, t_origin: f64) -> f64 {
        match self.end {
            Some(end) => end.t,
            None => (t_origin + self.cleared as f64 * STEP_DT).max(self.t0),
        }
    }

    /// Fly across the next grid step, unless something that might be in the way during it is
    /// not known yet. Every puck and hull except the shooter's own counts; the first one the
    /// head touches stops it.
    pub fn advance(&mut self, bodies: &[Body], ships: &[Ship], t_origin: f64) -> Progress {
        if self.end.is_some() {
            return Progress::Ended;
        }
        let k = self.cleared;
        let ta = self.known_to(t_origin);
        let tb = (t_origin + (k + 1) as f64 * STEP_DT).min(self.stop.t);
        // A center last known at `pos` at lab time `t` that is still unknown at `tb` can't
        // have come within `r` of the head during `[ta, tb]` unless this holds.
        let could_reach = |pos: V3, t: f64, r: f64| {
            super::body::distance(pos, self.pos_at(tb)) < r + (tb - t) + (tb - ta)
        };

        let mut first: Option<(f64, Target)> = None;
        let mut take = |t: Option<f64>, target: Target| {
            if let Some(t) = t {
                if first.is_none_or(|(best, _)| t < best) {
                    first = Some((t, target));
                }
            }
        };
        for body in bodies {
            if body.first_step() > k {
                continue; // not there yet (or pruned long ago)
            }
            match (body.at(k), body.at(k + 1)) {
                (Some(a), Some(b)) => {
                    // Between grid steps a body moves in a straight line (as it is drawn).
                    let f = |t: f64| (t - a.t) / (b.t - a.t);
                    let at = |t: f64| [0, 1, 2].map(|c| a.pos[c] + (b.pos[c] - a.pos[c]) * f(t));
                    take(self.sweep(ta, tb, at(ta), at(tb), body.radius), Target::Body(body.uid));
                }
                _ if body.ended => {} // gone by step k + 1
                _ => {
                    let last = body.latest();
                    if could_reach(last.pos, last.t, body.radius) {
                        return Progress::Waiting;
                    }
                }
            }
        }
        for ship in ships.iter().filter(|s| s.player != self.shooter) {
            // A ship that left ends at its last sample.
            let known = ship.latest().t;
            if known < tb && ship.is_live() {
                let last = ship.latest_hull();
                if could_reach(last.pos, last.t, SHIP_RADIUS) {
                    return Progress::Waiting;
                }
                continue;
            }
            // The hull moves in a straight line between its samples.
            let mut t = ta;
            let ends = ship.hull.iter().map(|s| s.t).filter(|&s| s > ta && s < tb);
            for u in ends.chain([tb]).map(|u| u.min(known)) {
                if u > t {
                    let hit = self.sweep(t, u, ship.hull_at(t).pos, ship.hull_at(u).pos, SHIP_RADIUS);
                    take(hit, Target::Ship(ship.player));
                    t = u;
                }
            }
        }

        self.end = match first {
            Some((t, target)) => Some(PulseEnd { t, pos: self.pos_at(t), target }),
            None if tb >= self.stop.t => Some(self.stop),
            None => {
                self.cleared = k + 1;
                return Progress::Advanced;
            }
        };
        Progress::Ended
    }

    /// The first lab time in `[ta, tb]` at which the head is within `r` of a center moving in
    /// a straight line from `ca` (at `ta`) to `cb` (at `tb`).
    fn sweep(&self, ta: f64, tb: f64, ca: V3, cb: V3, r: f64) -> Option<f64> {
        let head = self.pos_at(ta);
        let d0: V3 = [0, 1, 2].map(|c| head[c] - ca[c]);
        let c = dot3(d0, d0) - r * r;
        if c <= 0.0 {
            return Some(ta); // already inside (fired from within it)
        }
        let len = tb - ta;
        let w: V3 = [0, 1, 2].map(|i| self.dir[i] - (cb[i] - ca[i]) / len);
        let (a, b) = (dot3(w, w), dot3(d0, w));
        if b >= 0.0 || a == 0.0 {
            return None; // not closing in
        }
        let disc = b * b - a * c;
        if disc < 0.0 {
            return None;
        }
        let s = (-b - disc.sqrt()) / a;
        (s <= len).then_some(ta + s)
    }

    /// The pulse as drawn: an orange-red rod trailing its head, from where it was fired to as
    /// far as it is known. Light has no rest frame, but a pulse does: the one its shooter fired
    /// it in. There it is a rod of [`PULSE_LENGTH`] along the firing direction; shaped in any
    /// other frame (the lab's), it would look skewed to everyone else, as if dragged by an
    /// aether. Its worldtube is cut into the pieces of [`Pulse::pieces`].
    pub fn simplices(&self, t_origin: f64) -> Vec<Simplex4D> {
        self.simplices_in_pieces(t_origin, PULSE_SEGMENT_DT)
    }

    /// [`Pulse::simplices`], with its worldtube cut into pieces of at most `piece_dt` lab time.
    fn simplices_in_pieces(&self, t_origin: f64, piece_dt: f64) -> Vec<Simplex4D> {
        let shape = self.shape();
        let pieces = self.pieces_of(t_origin, piece_dt);
        let mut simplices = Vec::new();
        for k in 0..pieces {
            simplices.extend(self.piece_with(&shape, t_origin, piece_dt, k, pieces));
        }
        simplices
    }

    /// How many pieces of at most [`PULSE_SEGMENT_DT`] lab time the pulse is drawn in (at
    /// least one). They are cut on a grid from where it was fired, `[t0 + k dt, t0 + (k + 1) dt]`,
    /// the last one ending where it is known to: so as the pulse flies on, only its last piece
    /// changes, and the scene cache reuses the others.
    pub fn pieces(&self, t_origin: f64) -> usize {
        self.pieces_of(t_origin, PULSE_SEGMENT_DT)
    }

    fn pieces_of(&self, t_origin: f64, piece_dt: f64) -> usize {
        let span = self.known_to(t_origin) - self.t0;
        // A last piece of no length (the pulse known exactly to a cut) is left out, but for
        // the first.
        let full = (span / piece_dt).floor() as usize;
        if full == 0 {
            return 1;
        }
        full + (full as f64 * piece_dt < span) as usize
    }

    /// Piece `k` of the [`Pulse::pieces`]: its stretch of the worldtube, with the cap where the
    /// pulse was fired on the first and the one where it is known to on the last.
    pub fn piece(&self, t_origin: f64, k: usize) -> Vec<Simplex4D> {
        self.piece_with(&self.shape(), t_origin, PULSE_SEGMENT_DT, k, self.pieces(t_origin))
    }

    /// The rod's mesh in the shooter's rest frame, that frame, and the offset from the head to
    /// the rod's center.
    fn shape(&self) -> (Mesh3D, Tetrad, Vec4) {
        let [vx, vy, vz] = self.vel.map(|c| c as f32);
        let tetrad = Tetrad::from_beta(Vec3::new(vx, vy, vz));
        let [x, y, z] = self.dir.map(|c| c as f32);
        let forward = tetrad.world_to_local_spatial(Vec4::new(1.0, x, y, z)).normalized();
        let back = tetrad.local_spatial_to_world(forward * (-0.5 * PULSE_LENGTH));
        (rod_mesh(forward), tetrad, back)
    }

    fn piece_with(
        &self,
        (mesh, tetrad, back): &(Mesh3D, Tetrad, Vec4),
        t_origin: f64,
        piece_dt: f64,
        k: usize,
        pieces: usize,
    ) -> Vec<Simplex4D> {
        let sample = |t: f64| {
            let [x, y, z] = self.pos_at(t);
            WorldlineSample {
                center: Vec4::new(t as f32, x as f32, y as f32, z as f32) + *back,
                proper_time: 0.0,
                phi: Vec3::default(),
                proper_accel_local: Vec3::default(),
                tetrad: *tetrad,
            }
        };
        let (first, last) = (k == 0, k + 1 == pieces);
        let ta = if first { self.t0 } else { self.t0 + k as f64 * piece_dt };
        let tb = if last { self.known_to(t_origin) } else { self.t0 + (k + 1) as f64 * piece_dt };
        let ends = [sample(ta), sample(tb)];
        let mut simplices =
            build_worldtube_simplices_with_decided(mesh, &ends, Some(&[self.fired_at; 2]), MATERIAL_PULSE);
        for (end, capped) in ends.iter().zip([first, last]) {
            if capped {
                simplices.extend(build_worldtube_cap(mesh, end, self.fired_at, MATERIAL_PULSE));
            }
        }
        simplices
    }

    /// The bits of everything [`Pulse::piece`] `k` is made from (the scene cache's key for
    /// it). A piece but the last stays the same as the pulse flies on.
    pub fn piece_key(&self, t_origin: f64, k: usize) -> Vec<u32> {
        let last = k + 1 == self.pieces(t_origin);
        let end = if last { self.known_to(t_origin) } else { f64::NAN };
        let f64s = [self.t0, end].into_iter().chain(self.pos0).chain(self.dir).chain(self.vel);
        f64s.flat_map(|x| {
            let b = x.to_bits();
            [b as u32, (b >> 32) as u32]
        })
        .chain([self.fired_at.to_bits(), k as u32, last as u32])
        .collect()
    }

    /// Shift every event by `dt` in lab time (`t_origin` moves with it).
    pub fn rebase(&mut self, dt: f64) {
        self.t0 += dt;
        self.stop.t += dt;
        if let Some(end) = &mut self.end {
            end.t += dt;
        }
    }
}

/// Where a pulse fired at `(t0, pos0)` along `dir` meets the static table, or runs out of range.
fn static_stop(t0: f64, pos0: V3, dir: V3) -> PulseEnd {
    let mut best = (LASER_RANGE, Target::Nothing);
    let felt = billiards::TABLE_Y as f64;
    if dir[1] != 0.0 {
        let s = (felt - pos0[1]) / dir[1];
        let half = billiards::FELT_HALF_SIZE as f64;
        let on_felt = |s: f64| (pos0[0] + dir[0] * s).abs() <= half && (pos0[2] + dir[2] * s).abs() <= half;
        if s >= 0.0 && s < best.0 && on_felt(s) {
            best = (s, Target::Table);
        }
    }
    for (center, he) in billiards::rail_boxes() {
        let (c, h) = ([center.x, center.y, center.z], [he.x, he.y, he.z]);
        if let Some(s) = ray_box(pos0, dir, c.map(f64::from), h.map(f64::from)) {
            if s < best.0 {
                best = (s, Target::Table);
            }
        }
    }
    let (s, target) = best;
    PulseEnd { t: t0 + s, pos: [0, 1, 2].map(|c| pos0[c] + dir[c] * s), target }
}

/// Distance along the ray from `o` along unit `dir` to the box at `center` with half extents
/// `he` (0 from inside), if it meets it.
fn ray_box(o: V3, dir: V3, center: V3, he: V3) -> Option<f64> {
    let (mut near, mut far) = (0.0f64, f64::INFINITY);
    for c in 0..3 {
        let (lo, hi) = (center[c] - he[c] - o[c], center[c] + he[c] - o[c]);
        if dir[c] == 0.0 {
            if lo > 0.0 || hi < 0.0 {
                return None;
            }
            continue;
        }
        let (a, b) = (lo / dir[c], hi / dir[c]);
        near = near.max(a.min(b));
        far = far.min(a.max(b));
    }
    (near <= far).then_some(near)
}

/// A thin box along the unit `d`, centered on the origin.
fn rod_mesh(d: Vec3) -> Mesh3D {
    let up = if d.y.abs() < 0.9 { Vec3::new(0.0, 1.0, 0.0) } else { Vec3::new(1.0, 0.0, 0.0) };
    let a = up.cross(d).normalized();
    let b = d.cross(a);
    let to_lab = |v: Vec3| a * v.x + b * v.y + d * v.z;
    let mut rod = Mesh3D::cube(Vec3::new(PULSE_HALF_WIDTH, PULSE_HALF_WIDTH, 0.5 * PULSE_LENGTH));
    for v in &mut rod.vertices {
        *v = to_lab(*v);
    }
    for tri in &mut rod.triangles {
        tri.normal = to_lab(tri.normal);
    }
    rod
}

/// Whether news of `event` has reached an observer at `o`: it lies in `o`'s causal past.
pub fn has_reached(event: Vec4, o: Vec4) -> bool {
    let d = o - event;
    d.t >= d.spatial().length()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pulse(pos0: V3, dir: V3) -> Pulse {
        let from = ShipSample { t: 0.0, pos: pos0, vel: [0.0; 3], tau: 0.0 };
        Pulse::new(0, 0, from, dir, 0.0, 0.0)
    }

    #[test]
    fn pulses_stop_at_the_felt_and_rails_or_run_out_of_range() {
        // Straight down from above the table.
        let p = pulse([0.0, 0.0, 5.0], [0.0, -1.0, 0.0]);
        assert_eq!(p.stop.target, Target::Table);
        assert!((p.stop.t - 2.2).abs() < 1e-6, "{:?}", p.stop);
        // Along the table at rail height: the far rail.
        let p = pulse([0.0, -2.0, 5.0], [0.0, 0.0, 1.0]);
        assert_eq!(p.stop.target, Target::Table);
        assert!((p.stop.pos[2] - billiards::Z_MAX).abs() < 1e-6, "{:?}", p.stop);
        // Up and away.
        let p = pulse([0.0, 0.0, 5.0], [0.0, 1.0, 0.0]);
        assert_eq!(p.stop.target, Target::Nothing);
        assert_eq!(p.stop.t, LASER_RANGE);
    }

    #[test]
    fn the_head_meets_a_moving_sphere_where_they_touch() {
        let p = pulse([0.0; 3], [1.0, 0.0, 0.0]);
        // A sphere of radius 0.5 crossing the pulse's path, centered on it at t = 10.
        let t = p.sweep(9.0, 11.0, [10.0, -0.5, 0.0], [10.0, 0.5, 0.0], 0.5).unwrap();
        // |(t, 0) - (10, 0.5 (t - 10))| = 0.5 at t = 10 - 0.5 / sqrt(1.25).
        assert!((t - (10.0 - 0.5 / 1.25f64.sqrt())).abs() < 1e-9, "{t}");
        // Out of reach, or behind: no hit.
        assert_eq!(p.sweep(9.0, 11.0, [10.0, 2.0, 0.0], [10.0, 2.0, 0.0], 0.5), None);
        assert_eq!(p.sweep(9.0, 11.0, [5.0, 0.0, 0.0], [5.0, 0.0, 0.0], 0.5), None);
    }

    #[test]
    fn the_rod_points_along_the_pulse() {
        let rod = rod_mesh(Vec3::new(0.6, 0.0, 0.8));
        let along = |v: &Vec3| v.x * 0.6 + v.z * 0.8;
        let len = rod.vertices.iter().map(along).fold(f32::MIN, f32::max) * 2.0;
        assert!((len - PULSE_LENGTH).abs() < 1e-5);
        for v in &rod.vertices {
            let across = (*v - Vec3::new(0.6, 0.0, 0.8) * along(v)).length();
            assert!(across < 2.0 * PULSE_HALF_WIDTH);
        }
    }

    /// Fired by a moving shooter, the rod is the one the shooter fired: in their rest frame,
    /// every point of it trails the head along the firing direction, by at most
    /// [`PULSE_LENGTH`]. (Shaped in the lab, it would lean against the shooter's motion.)
    #[test]
    fn the_rod_is_shaped_in_the_shooters_frame() {
        let beta = Vec3::new(0.6, 0.0, 0.3);
        let tetrad = Tetrad::from_beta(beta);
        // Fired sideways (+x) in the shooter's frame: forward in the lab, by aberration.
        let k = tetrad.future_light_ray_direction(Vec3::new(1.0, 0.0, 0.0));
        let dir = [k.x / k.t, k.y / k.t, k.z / k.t].map(|c| c as f64);
        let from = ShipSample { t: 0.0, pos: [0.0; 3], vel: [0.6, 0.0, 0.3], tau: 0.0 };
        let mut p = Pulse::new(0, 0, from, dir, 0.0, 0.0);
        p.cleared = 40; // known to lab time 40 STEP_DT
        let simplices = p.simplices(0.0);
        assert!(!simplices.is_empty());
        for s in &simplices {
            // Where the vertex is relative to the head that passed the firing event at the
            // same shooter time.
            let local = tetrad.world_to_local(s.v0);
            let r = Vec3::new(local.x, local.y, local.z) - Vec3::new(local.t, 0.0, 0.0);
            assert!(r.x <= 1e-4 && r.x >= -PULSE_LENGTH - 1e-4, "{r:?}");
            assert!(r.y.abs() <= 2.0 * PULSE_HALF_WIDTH && r.z.abs() <= 2.0 * PULSE_HALF_WIDTH, "{r:?}");
        }
    }

    /// Cut into pieces, a pulse looks the same, only sharper: rays all over a long pulse's path
    /// hit the pieces where they hit it whole, up to its `f32` rounding (simplices 100 long
    /// around a rod 0.04 thick), which the pieces are rid of.
    #[test]
    fn a_pulse_in_pieces_is_the_whole_pulse() {
        use crate::bvh4d::Bvh4D;
        let from = ShipSample { t: 0.0, pos: [0.0, 0.0, 0.0], vel: [0.0, 0.0, 0.0], tau: 0.0 };
        let mut p = Pulse::new(0, 0, from, [0.2, 0.1, 1.0], 0.0, 0.0);
        p.cleared = (100.0 / STEP_DT) as i64;
        let p = p;
        let whole = p.simplices_in_pieces(0.0, f64::INFINITY);
        let pieces = p.simplices(0.0);
        assert_eq!(whole.len(), 60);
        assert!(pieces.len() > 5 * whole.len(), "{}", pieces.len());
        let (whole, pieces) = (Bvh4D::build(whole), Bvh4D::build(pieces));

        // How far an event is from the rod's surface (shooter at rest: shaped in the lab).
        let d = Vec3::new(p.dir[0] as f32, p.dir[1] as f32, p.dir[2] as f32);
        let a = Vec3::new(0.0, 1.0, 0.0).cross(d).normalized();
        let b = d.cross(a);
        let off_surface = |e: Vec4| {
            let h = p.pos_at(e.t as f64).map(|c| c as f32);
            let r = e.spatial() - Vec3::new(h[0], h[1], h[2]) + d * (0.5 * PULSE_LENGTH);
            let w = PULSE_HALF_WIDTH;
            let m = (r.dot(a).abs() / w).max(r.dot(b).abs() / w).max(r.dot(d).abs() / (0.5 * PULSE_LENGTH));
            (m - 1.0).abs() * w
        };

        let (mut hits, mut differ) = (0, 0);
        let (mut off_whole, mut off_pieces, mut d_lambda) = (0.0f32, 0.0f32, 0.0f32);
        // Past light rays from observers beside the path, at many times, fanned across the rod
        // where they see it.
        for i in 0..40 {
            let t = 5.0 + 2.5 * i as f64;
            let eye = {
                let h = p.pos_at(t);
                [t + 3.0, h[0] + 2.0, h[1] + 0.3, h[2] - 1.0]
            };
            // The head's event on the eye's past light cone.
            let mut seen = t;
            for _ in 0..100 {
                let h = p.pos_at(seen);
                seen = eye[0] - (0..3).map(|c| (eye[c + 1] - h[c]).powi(2)).sum::<f64>().sqrt();
            }
            let h = p.pos_at(seen).map(|c| c as f32);
            let aim = Vec3::new(h[0], h[1], h[2]) - d * (0.5 * PULSE_LENGTH);
            let eye = Vec4::new(eye[0] as f32, eye[1] as f32, eye[2] as f32, eye[3] as f32);
            for j in 0..400 {
                let (u, v) = ((j % 20) as f32 / 19.0 - 0.5, (j / 20) as f32 / 19.0 - 0.5);
                let n = (aim + Vec3::new(u, v, 2.0 * u - v) * 0.15 - eye.spatial()).normalized();
                let dir = Vec4::new(-1.0, n.x, n.y, n.z);
                match (whole.trace(eye, dir, 1e-3, 50.0), pieces.trace(eye, dir, 1e-3, 50.0)) {
                    (Some(h0), Some(h1)) => {
                        hits += 1;
                        off_whole = off_whole.max(off_surface(h0.event));
                        off_pieces = off_pieces.max(off_surface(h1.event));
                        let dn = h0.surface_normal_4d - h1.surface_normal_4d;
                        if dn.euclidean_dot(dn) < 1e-6 {
                            d_lambda = d_lambda.max((h0.lambda - h1.lambda).abs());
                        } else {
                            differ += 1; // through an edge: on either face
                        }
                    }
                    (None, None) => {}
                    (h0, h1) => {
                        differ += 1; // grazing an edge
                        for h in h0.iter().chain(&h1) {
                            assert!(off_surface(h.event) < 1e-3);
                        }
                    }
                }
            }
        }
        assert!(hits > 3000, "{hits}");
        assert!(differ * 50 < hits, "{differ} of {hits}");
        assert!(off_pieces < 5e-5, "{off_pieces}");
        assert!(off_whole < 2e-3 && d_lambda < 4e-3, "{off_whole} {d_lambda}");
    }
}
