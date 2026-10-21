//! Causal Dynamic Mode (see `docs/DYNAMIC_MODE_PLAN.md`).
//!
//! The only source of non-determinism is the player, so an event is *decided* once it has left
//! the open future light cone `I+(O)` of the camera event `O`: no future input can reach it.
//! Every body's history is extended exactly that far — up to the causal frontier `dJ+(O)` —
//! and rendered from a fresh snapshot ([`DynWorld::build_scene`]).
//!
//! Bodies share one lab-time grid. A body advances from step `k` once its next state is decided
//! and everything that could touch it at step `k` is known: other bodies' states at `k`, and the
//! ship's recorded worldline. Its next state is then a pure function of finalized data, so the
//! histories do not depend on how the frontier's motion was split into frames.

pub mod billiards;
pub mod body;
pub mod collide;

use crate::bvh4d::Bvh4D;
use crate::mesh4d::{
    build_worldtube_cap, build_worldtube_simplices, build_worldtube_simplices_with_decided,
    sample_inertial_worldline, Mesh3D,
};
use crate::minkowski::{Vec3, Vec4};
use crate::renderer::{analytic_decision_age, Frontier};
use crate::scene::{DemoScene, Scene4D};
use billiards::{Ammo, SHIP_MASS, SHIP_RADIUS};
use body::{distance, is_decided, Body, BodyState, Kind, STEP_DT};
use collide::{P4, V3};

/// How far behind the camera (in lab time) histories are kept: beyond the renderers' maximum
/// ray length, nothing older can be seen.
pub const HISTORY_WINDOW: f64 = 200.0;

/// Render segments are split at least this often (in lab time), so that the per-vertex
/// `decided_at` interpolated along a merged segment stays close to the per-step stamps.
const MAX_SEGMENT_DT: f64 = 50.0;

const MATERIAL_RAIL: usize = 6;

/// One recorded event of the ship (camera) worldline.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ShipSample {
    pub t: f64,
    pub pos: V3,
    pub vel: V3,
}

impl ShipSample {
    fn new(frontier: Frontier, u: Vec4) -> Self {
        let o = frontier.origin;
        Self {
            t: o.t as f64,
            pos: [o.x as f64, o.y as f64, o.z as f64],
            vel: [(u.x / u.t) as f64, (u.y / u.t) as f64, (u.z / u.t) as f64],
        }
    }
}

/// What a body touches at a grid step.
struct Contacts {
    /// Indices of bodies overlapping it, sorted by uid.
    bodies: Vec<usize>,
    /// The ship's state, if overlapping.
    ship: Option<ShipSample>,
    /// The ship is known to be clear of the body (ends a projectile's launch grace).
    ship_clear: bool,
}

pub struct DynWorld {
    pub demo: DemoScene,
    pub bodies: Vec<Body>,
    /// The ship's (camera's) recorded worldline, one sample per [`DynWorld::advance`], pruned
    /// behind the camera. Bodies near the ship wait for it to be recorded.
    pub ship: Vec<ShipSample>,
    /// The frontier the histories were last extended to.
    pub frontier: Frontier,
    /// Collision 4-momentum delivered to the ship and not yet applied to the camera.
    pub ship_kicks: Vec<P4>,
    /// Lab time of grid step 0.
    t_origin: f64,
    fired: u64,
}

impl DynWorld {
    /// The world as it stands when the camera spawns at the demo's initial event, at rest.
    pub fn spawn(demo: DemoScene) -> Self {
        let frontier = Frontier {
            origin: demo.initial_camera_pos(),
            proper_time: 0.0,
        };
        Self::new(demo, frontier, Vec4::new(1.0, 0.0, 0.0, 0.0))
    }

    /// Start the world with the observer at `frontier`, moving with 4-velocity `u`. Everything
    /// outside `I+(O)` is already decided: histories are filled from [`HISTORY_WINDOW`] in the
    /// past up to the frontier, stamped with the analytic decision time as if the observer had
    /// always coasted with `u`. (The scene is laid out so nothing collides in that past.)
    pub fn new(demo: DemoScene, frontier: Frontier, u: Vec4) -> Self {
        let mut bodies = match demo {
            DemoScene::RelativisticBilliards => billiards::pucks(),
            _ => panic!("{demo:?} is not a dynamic scene"),
        };
        let t_origin = 0.0;
        let first_step = ((frontier.origin.t as f64 - HISTORY_WINDOW - t_origin) / STEP_DT).floor() as i64;
        for body in &mut bodies {
            // Bodies are specified at step 0; walk back to the start of the window.
            let mut s = body.history[0];
            while s.step > first_step {
                s = back_step(&s, body.radius, t_origin + (s.step - 1) as f64 * STEP_DT);
            }
            body.history = vec![s];
            body.decided_at = vec![decision_time(&s, frontier, u)];
        }
        let mut world = Self {
            demo,
            bodies,
            ship: vec![ShipSample::new(frontier, u)],
            frontier,
            ship_kicks: Vec::new(),
            t_origin,
            fired: 0,
        };
        world.extend(frontier, |s| decision_time(s, frontier, u));
        world
    }

    fn step_time(&self, step: i64) -> f64 {
        self.t_origin + step as f64 * STEP_DT
    }

    /// Move the frontier to the ship's new event (moving with 4-velocity `u`) and finalize
    /// every step that just became decided. Returns whether anything changed.
    pub fn advance(&mut self, frontier: Frontier, u: Vec4) -> bool {
        let sample = ShipSample::new(frontier, u);
        match self.ship.last_mut() {
            Some(last) if last.t == sample.t => *last = sample, // paused: no new event
            _ => self.ship.push(sample),
        }
        self.extend(frontier, |_| frontier.proper_time)
    }

    /// Fire a body of `ammo` from the ship at `at`, with lab 3-velocity `vel`. It leaves the
    /// ship's center, so its whole worldline lies in the firing event's causal future; its
    /// first grid state is fixed by this input alone. Returns `false` if the table is full.
    pub fn fire(&mut self, at: Frontier, vel: V3, ammo: Ammo) -> bool {
        let live = self.bodies.iter().filter(|b| !b.ended).count();
        if live >= billiards::MAX_BODIES {
            return false;
        }
        let o = at.origin;
        let step = ((o.t as f64 - self.t_origin) / STEP_DT).ceil() as i64;
        let t = self.step_time(step);
        let dt = t - o.t as f64;
        let pos = [o.x as f64 + vel[0] * dt, o.y as f64 + vel[1] * dt, o.z as f64 + vel[2] * dt];
        let state = BodyState { step, t, pos, vel, tau: dt / collide::gamma(vel) };
        let mut b = billiards::projectile(self.fired, ammo, state);
        b.decided_at = vec![at.proper_time];
        self.bodies.push(b);
        self.fired += 1;
        true
    }

    /// Scheduler: keep passing over the bodies, advancing each as far as it can, until a pass
    /// makes no progress. Then prune what has fallen behind the history window.
    fn extend(&mut self, frontier: Frontier, stamp: impl Fn(&BodyState) -> f32) -> bool {
        self.frontier = frontier;
        let mut changed = false;
        loop {
            let mut progress = false;
            let mut i = 0;
            while i < self.bodies.len() {
                if !self.bodies[i].ended && self.try_advance(i, &stamp) {
                    progress = true;
                }
                i += 1;
            }
            changed |= progress;
            if !progress {
                break;
            }
        }
        self.prune();
        changed
    }

    /// Advance body `i` by one grid step (or end it in a decay or merger). Returns `false` if
    /// it has to wait: for the frontier, or for a neighbor or the ship to be known.
    fn try_advance(&mut self, i: usize, stamp: &impl Fn(&BodyState) -> f32) -> bool {
        let s = *self.bodies[i].latest();
        let k = s.step;

        if let Kind::Spore { lifetime, generation } = self.bodies[i].kind {
            if s.tau >= lifetime {
                return self.decay(i, generation, stamp);
            }
        }

        let Some(contacts) = self.contacts(i, k) else {
            return false;
        };
        let sticky = self.bodies[i].is_sticky();
        if contacts.bodies.iter().any(|&j| sticky || self.bodies[j].is_sticky()) {
            return self.merge_cluster(i, k, stamp);
        }

        let (m, r) = (self.bodies[i].mass, self.bodies[i].radius);
        let mut vel = s.vel;
        for &j in &contacts.bodies {
            let o = self.bodies[j].at(k).expect("contacts exist at step k");
            if let Some((v, _)) = collide::elastic(m, vel, self.bodies[j].mass, o.vel, sub3(o.pos, s.pos)) {
                vel = v;
            }
        }
        let mut kick = None;
        if let Some(ship) = contacts.ship {
            if let Some((v, v_ship)) = collide::elastic(m, vel, SHIP_MASS, ship.vel, sub3(ship.pos, s.pos)) {
                vel = v;
                kick = Some(collide::sub(
                    collide::momentum(SHIP_MASS, v_ship),
                    collide::momentum(SHIP_MASS, ship.vel),
                ));
            }
        }

        let mut next = s.drift(vel, self.step_time(k + 1));
        billiards::constrain(&s, &mut next, r);
        if !is_decided(&next, r, self.frontier.origin) {
            return false;
        }
        let stamp = stamp(&next);
        let body = &mut self.bodies[i];
        body.history.push(next);
        body.decided_at.push(stamp);
        body.ended = billiards::out_of_play(&next);
        if contacts.ship_clear {
            body.ignore_ship = false;
        }
        self.ship_kicks.extend(kick);
        true
    }

    /// Everything body `i` touches at step `k`, or `None` if something that might be touching
    /// it is not known yet: a body still behind step `k` that could reach contact range at
    /// light speed, or the ship beyond its recorded track.
    fn contacts(&self, i: usize, k: i64) -> Option<Contacts> {
        let me = &self.bodies[i];
        let s = me.at(k).expect("body has step k");
        let mut touching = Vec::new();
        for (j, other) in self.bodies.iter().enumerate() {
            if j == i || other.first_step() > k {
                continue;
            }
            let reach = me.radius + other.radius;
            match other.at(k) {
                Some(o) => {
                    if distance(s.pos, o.pos) < reach {
                        touching.push(j);
                    }
                }
                None if other.ended => {} // gone before step k
                None => {
                    let last = other.latest();
                    let light = (k - last.step) as f64 * STEP_DT;
                    if distance(s.pos, last.pos) < reach + light {
                        return None;
                    }
                }
            }
        }
        touching.sort_by_key(|&j| self.bodies[j].uid);

        let reach = me.radius + SHIP_RADIUS;
        let last = *self.ship.last().expect("ship track is never empty");
        let (ship, ship_clear) = if last.t >= s.t {
            let at = self.ship_at(s.t);
            let d = distance(s.pos, at.pos);
            if me.ignore_ship {
                (None, d > reach)
            } else {
                ((d < reach).then_some(at), false)
            }
        } else if !me.ignore_ship && distance(s.pos, last.pos) < reach + (s.t - last.t) {
            return None;
        } else {
            (None, false)
        };
        Some(Contacts { bodies: touching, ship, ship_clear })
    }

    /// The ship's state at lab time `t` (within the recorded track), interpolated linearly.
    fn ship_at(&self, t: f64) -> ShipSample {
        let i = self.ship.partition_point(|s| s.t <= t);
        if i == 0 {
            return self.ship[0];
        }
        let a = self.ship[i - 1];
        let Some(&b) = self.ship.get(i) else {
            return a;
        };
        let f = if b.t > a.t { (t - a.t) / (b.t - a.t) } else { 0.0 };
        let lerp = |x: V3, y: V3| [0, 1, 2].map(|c| x[c] + (y[c] - x[c]) * f);
        ShipSample { t, pos: lerp(a.pos, b.pos), vel: lerp(a.vel, b.vel) }
    }

    /// All sticky contacts of body `i` at step `k`, transitively, merge into one body carrying
    /// their total 4-momentum. Waits (returns `false`) until the whole cluster is known and the
    /// merged body's first state is decided.
    fn merge_cluster(&mut self, i: usize, k: i64, stamp: &impl Fn(&BodyState) -> f32) -> bool {
        let mut members = vec![i];
        let mut next = 0;
        while next < members.len() {
            let m = members[next];
            next += 1;
            let Some(c) = self.contacts(m, k) else {
                return false;
            };
            for j in c.bodies {
                let joins = self.bodies[m].is_sticky() || self.bodies[j].is_sticky();
                if joins && !members.contains(&j) {
                    members.push(j);
                }
            }
        }
        members.sort_by_key(|&m| self.bodies[m].uid);

        let parts: Vec<(f64, V3)> =
            members.iter().map(|&m| (self.bodies[m].mass, self.bodies[m].latest().vel)).collect();
        let (mass, vel) = collide::merge(&parts);
        // Center of energy of the parts.
        let mut pos = [0.0; 3];
        let mut energy = 0.0;
        for &m in &members {
            let s = self.bodies[m].latest();
            let e = self.bodies[m].mass * s.gamma();
            energy += e;
            for c in 0..3 {
                pos[c] += e * s.pos[c];
            }
        }
        let pos = pos.map(|p| p / energy);

        // The parts' last state is step `k`; the merged body's first is `k + 1`, so no step
        // ever holds both (nothing can collide with a body and its own parts at once).
        let at_k = BodyState { step: k, t: self.step_time(k), pos, vel, tau: 0.0 };
        let first = at_k.drift(vel, self.step_time(k + 1));
        if !is_decided(&first, billiards::puck_radius(mass), self.frontier.origin) {
            return false;
        }
        let uids: Vec<u64> = members.iter().map(|&m| self.bodies[m].uid).collect();
        let mut merged = billiards::merged(&uids, mass, first);
        merged.decided_at = vec![stamp(&first)];
        for &m in &members {
            debug_assert_eq!(self.bodies[m].latest().step, k);
            self.bodies[m].ended = true;
        }
        self.bodies.push(merged);
        true
    }

    /// A spore whose clock ran out at step `k` splits there; the two fragments fly from step
    /// `k + 1`. Waits (returns `false`) until both fragments' first states are decided.
    fn decay(&mut self, i: usize, generation: u32, stamp: &impl Fn(&BodyState) -> f32) -> bool {
        let parent = &self.bodies[i];
        let s = *parent.latest();
        let dir = billiards::split_direction(s.vel);
        let parts = collide::decay(parent.mass, s.vel, dir, billiards::SPORE_SPLIT_SPEED);
        let t_next = self.step_time(s.step + 1);
        let mut fragments = billiards::fragments(parent.uid, generation, parts, &s);
        for f in &mut fragments {
            let first = f.history[0].drift(f.history[0].vel, t_next);
            if !is_decided(&first, f.radius, self.frontier.origin) {
                return false;
            }
            f.history = vec![first];
            f.decided_at = vec![stamp(&first)];
        }
        self.bodies[i].ended = true;
        self.bodies.extend(fragments);
        true
    }

    fn prune(&mut self) {
        let horizon = self.frontier.origin.t as f64 - HISTORY_WINDOW;
        self.bodies.retain(|b| !(b.ended && b.latest().t < horizon));
        for body in &mut self.bodies {
            let stale = body.history.partition_point(|s| s.t < horizon);
            let stale = stale.min(body.history.len() - 1);
            if stale > 0 {
                body.born = false; // its birth has scrolled out of the window
            }
            body.history.drain(..stale);
            body.decided_at.drain(..stale);
        }
        let stale = self.ship.partition_point(|s| s.t < horizon);
        self.ship.drain(..stale.min(self.ship.len() - 1));
    }

    /// How far (in the camera's proper time, for a camera moving with 4-velocity `u`) the live
    /// bodies' finalized tips trail the frontier: the largest decision age of a tip, plus one
    /// step so the last prism is in view. The Frontier view looks back by this much so that
    /// its future-cone rays meet the bodies where their histories end. Capped at 5 s.
    pub fn frontier_lag(&self, u: Vec4) -> f32 {
        let o = self.frontier.origin;
        self.bodies
            .iter()
            .filter(|b| !b.ended)
            .map(|b| analytic_decision_age(b.latest().event(), o, u))
            .fold(0.0f32, f32::max)
            .min(5.0)
            + STEP_DT as f32
    }

    /// Shift every event by `dt` in lab time (origin rebase, keeping `f32` coordinates precise).
    pub fn rebase(&mut self, dt: f64) {
        self.t_origin += dt;
        let t_origin = self.t_origin;
        for body in &mut self.bodies {
            for s in &mut body.history {
                s.t = t_origin + s.step as f64 * STEP_DT;
            }
        }
        for s in &mut self.ship {
            s.t += dt;
        }
        self.frontier.origin.t += dt as f32;
    }

    /// Snapshot the world as renderable geometry: the static table plus every body's tube,
    /// sealed with caps where it was born and where it currently ends (its frontier tip, or
    /// the merger, decay or exit that ended it).
    pub fn build_scene(&self) -> Scene4D {
        let o = self.frontier.origin;
        let window = HISTORY_WINDOW as f32;
        let backdrop =
            |pos| sample_inertial_worldline(pos, Vec3::default(), o.t - window, o.t + window, 1);

        let ground = Mesh3D::ground_grid(24.0, 24.0, billiards::TABLE_Y, 4, 4);
        let mut simplices = build_worldtube_simplices(&ground, &backdrop(Vec3::default()), 0);
        for (rail, center) in billiards::rails() {
            simplices.extend(build_worldtube_simplices(&rail, &backdrop(center), MATERIAL_RAIL));
        }

        for body in &self.bodies {
            let keep = render_sample_indices(&body.history);
            let samples: Vec<_> = keep.iter().map(|&i| body.history[i].sample()).collect();
            let decided: Vec<f32> = keep.iter().map(|&i| body.decided_at[i]).collect();
            simplices.extend(build_worldtube_simplices_with_decided(
                &body.mesh,
                &samples,
                Some(&decided),
                body.material_id,
            ));
            let mut cap = |i: usize| {
                simplices.extend(build_worldtube_cap(&body.mesh, &samples[i], decided[i], body.material_id));
            };
            cap(samples.len() - 1);
            if body.born && samples.len() > 1 {
                cap(0);
            }
        }

        Scene4D {
            demo: self.demo,
            bvh: Bvh4D::build(simplices),
            materials: Scene4D::common_materials(),
            light_pos: Vec3::new(0.0, 6.5, 4.0),
            t_min: o.t - window,
            t_max: o.t + window,
        }
    }
}

fn sub3(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// The state one grid step before `s` (for back-filling the collision-free past at spawn):
/// run the step forward in reversed time.
fn back_step(s: &BodyState, radius: f64, t_prev: f64) -> BodyState {
    let reversed = BodyState { t: -s.t, tau: -s.tau, vel: s.vel.map(|v| -v), ..*s };
    let mut prev = reversed.drift(reversed.vel, -t_prev);
    billiards::constrain(&reversed, &mut prev, radius);
    BodyState { step: s.step - 1, t: t_prev, tau: -prev.tau, vel: prev.vel.map(|v| -v), ..prev }
}

/// Camera proper time at which `state` left `I+` of an observer that reached `frontier` coasting
/// with 4-velocity `u` (never later than the frontier itself).
fn decision_time(state: &BodyState, frontier: Frontier, u: Vec4) -> f32 {
    let age = analytic_decision_age(state.event(), frontier.origin, u);
    frontier.proper_time - age.max(0.0)
}

/// History steps to keep as worldtube samples: the ends, both sides of every velocity change
/// (so straight stretches merge into single prisms and kinks stay sharp), and at least one
/// sample every [`MAX_SEGMENT_DT`].
fn render_sample_indices(history: &[BodyState]) -> Vec<usize> {
    let mut keep = vec![0];
    for i in 1..history.len() {
        let last = *keep.last().unwrap();
        let kink = history[i].vel != history[i - 1].vel;
        if kink && last != i - 1 {
            keep.push(i - 1);
        }
        let last_t = history[*keep.last().unwrap()].t;
        if kink || i == history.len() - 1 || history[i].t - last_t >= MAX_SEGMENT_DT {
            keep.push(i);
        }
    }
    keep
}

#[cfg(test)]
mod tests {
    use super::billiards::{puck_radius, PUCK_MASS};
    use super::*;
    use crate::minkowski::Tetrad;

    /// A camera worldline that thrusts sideways, then forward, then coasts: `n` frontiers
    /// `dtau` apart from the billiards spawn event, with the camera's 4-velocity at each.
    fn camera_path(n: usize, dtau: f32) -> Vec<(Frontier, Vec4)> {
        let mut tetrad = Tetrad::rest();
        let mut origin = DemoScene::RelativisticBilliards.initial_camera_pos();
        let mut tau = 0.0;
        (0..n)
            .map(|i| {
                let frac = i as f32 / n as f32;
                if frac < 0.3 {
                    tetrad.boost_local(Vec3::new(0.02, 0.0, 0.0));
                } else if frac < 0.6 {
                    tetrad.boost_local(Vec3::new(0.0, 0.0, 0.015));
                }
                origin += tetrad.e[0] * dtau;
                tau += dtau;
                (Frontier { origin, proper_time: tau }, tetrad.e[0])
            })
            .collect()
    }

    /// A slowly drifting camera (rapidity below 0.5 throughout), so lab time stays close to
    /// camera time and nothing of interest is pruned.
    fn gentle_path(n: usize, dtau: f32) -> Vec<(Frontier, Vec4)> {
        let mut tetrad = Tetrad::rest();
        let mut origin = DemoScene::RelativisticBilliards.initial_camera_pos();
        let mut tau = 0.0;
        (0..n)
            .map(|_| {
                tetrad.boost_local(Vec3::new(0.0005, 0.0, 0.0003));
                origin += tetrad.e[0] * dtau;
                tau += dtau;
                (Frontier { origin, proper_time: tau }, tetrad.e[0])
            })
            .collect()
    }

    fn rest() -> Vec4 {
        Vec4::new(1.0, 0.0, 0.0, 0.0)
    }

    /// Every finalized step is decided (except a projectile's launch state, which the input
    /// that fired it already fixed), stamped no later than now, with non-decreasing stamps and
    /// no gaps in the grid.
    fn assert_causal(world: &DynWorld) {
        let o = world.frontier;
        for body in &world.bodies {
            assert_eq!(body.history.len(), body.decided_at.len());
            for (i, (s, &d)) in body.history.iter().zip(&body.decided_at).enumerate() {
                let launch = body.born && i == 0;
                assert!(
                    launch || is_decided(s, body.radius, o.origin),
                    "body {:x} step {} undecided",
                    body.uid,
                    s.step
                );
                assert!(d <= o.proper_time, "body {:x} stamped in the future", body.uid);
            }
            assert!(body.decided_at.windows(2).all(|w| w[0] <= w[1]));
            assert!(body.history.windows(2).all(|w| w[1].step == w[0].step + 1));
        }
    }

    /// Park the camera (at rest) and advance one frame per 0.05 s for `secs` seconds.
    fn run_parked(world: &mut DynWorld, secs: f32) {
        let mut f = world.frontier;
        for _ in 0..(secs / 0.05) as usize {
            f.origin.t += 0.05;
            f.proper_time += 0.05;
            world.advance(f, rest());
        }
    }

    /// Lab velocity at `speed` from the camera toward the rack's front puck (body 0).
    fn aim_at_rack(world: &DynWorld, speed: f64) -> V3 {
        let target = world.bodies[0].latest().pos;
        let o = world.frontier.origin;
        let d = [target[0] - o.x as f64, target[1] - o.y as f64, target[2] - o.z as f64];
        let l = collide::dot3(d, d).sqrt();
        d.map(|c| c / l * speed)
    }

    #[test]
    fn histories_stay_outside_the_future_cone() {
        let mut world = DynWorld::spawn(DemoScene::RelativisticBilliards);
        assert_causal(&world);
        for (i, (f, u)) in camera_path(400, 0.05).into_iter().enumerate() {
            world.advance(f, u);
            if i % 37 == 0 {
                assert_causal(&world);
            }
        }
        assert_causal(&world);
        // The scheduler is not lazy: every body is either up against the frontier, or waiting
        // for a neighbor that lags behind it and could reach contact range at light speed.
        let o = world.frontier.origin;
        for (i, body) in world.bodies.iter().enumerate() {
            let s = body.latest();
            let dist = (s.event().spatial() - o.spatial()).length() as f64;
            let reach = body::event_margin(s, body.radius) + 2.0 * STEP_DT;
            let at_frontier = s.t - (o.t as f64) > dist - reach;
            assert!(
                at_frontier || world.contacts(i, s.step).is_none(),
                "body {:x} lags the frontier without waiting for anything",
                body.uid
            );
        }
    }

    /// Histories depend only on physics and the ship's recorded worldline, not on how the
    /// frontier's motion was chopped into frames.
    #[test]
    fn history_independent_of_frame_chunking() {
        let path = camera_path(300, 0.07);
        let mut fine = DynWorld::spawn(DemoScene::RelativisticBilliards);
        for &(f, u) in &path {
            fine.advance(f, u);
        }
        let mut coarse = DynWorld::spawn(DemoScene::RelativisticBilliards);
        for &(f, u) in path.iter().skip(42).step_by(43).chain(path.last()) {
            coarse.advance(f, u);
        }
        for (a, b) in fine.bodies.iter().zip(&coarse.bodies) {
            assert_eq!(a.history, b.history);
        }
    }

    #[test]
    fn history_is_pruned_behind_the_camera_and_survives_rebase() {
        let mut world = DynWorld::spawn(DemoScene::RelativisticBilliards);
        for (f, u) in camera_path(60, 5.0) {
            world.advance(f, u);
        }
        let horizon = world.frontier.origin.t as f64 - HISTORY_WINDOW;
        for body in &world.bodies {
            assert!(body.history[0].t >= horizon && body.history[0].t < horizon + STEP_DT);
        }
        world.rebase(-1000.0);
        assert_causal(&world);
        let mut moved = world.frontier;
        moved.origin += Vec4::new(1.0, 0.0, 0.0, 0.0);
        moved.proper_time += 1.0;
        assert!(world.advance(moved, rest()));
        assert_causal(&world);
    }

    #[test]
    fn movers_bounce_between_the_rails_and_never_collide_on_their_own() {
        let mut world = DynWorld::spawn(DemoScene::RelativisticBilliards);
        let mut origin = world.frontier.origin;
        let mut flips = 0;
        for i in 1..=200 {
            origin.t += 1.0;
            world.advance(Frontier { origin, proper_time: i as f32 }, rest());
            for body in &world.bodies {
                for w in body.history.windows(2) {
                    assert!(w[1].pos[0].abs() <= billiards::HALF_X - body.radius + 1e-9);
                    flips += (w[0].vel != w[1].vel) as usize;
                }
            }
        }
        assert!(flips > 0, "movers should bounce off the rails");
        assert_eq!(world.bodies.len(), 13);
        assert!(world.bodies[..10].iter().all(|b| b.latest().vel == [0.0; 3]), "rack stays put");
    }

    /// Frontier-view rays aimed at each live puck hit it when they start `frontier_lag` behind
    /// the camera, and pass behind its tip with the bare static-scene offset.
    #[test]
    fn frontier_view_rays_meet_the_finalized_tips() {
        let mut world = DynWorld::spawn(DemoScene::RelativisticBilliards);
        run_parked(&mut world, 3.0);
        let scene = world.build_scene();
        let o = world.frontier.origin;
        let tetrad = Tetrad::rest();
        let hits = |offset: f32| {
            world
                .bodies
                .iter()
                .filter(|b| !b.ended)
                .filter(|b| {
                    // Aim where the view's light cone crosses the body's worldline.
                    let origin = o - tetrad.e[0] * offset;
                    let crossing = b.history.iter().min_by(|x, y| {
                        let off = |s: &&BodyState| {
                            let e = s.event() - origin;
                            (e.t - e.spatial().length()).abs()
                        };
                        off(x).total_cmp(&off(y))
                    });
                    let dir = crossing.unwrap().event().spatial() - origin.spatial();
                    let ray = tetrad.future_light_ray_direction(dir);
                    scene
                        .bvh
                        .trace(origin, ray, 1e-3, 180.0)
                        .is_some_and(|h| h.material_id == b.material_id)
                })
                .count()
        };
        let live = world.bodies.iter().filter(|b| !b.ended).count();
        let lag = world.frontier_lag(tetrad.e[0]);
        assert!(lag > 0.3 && lag <= 5.0 + STEP_DT as f32, "lag {lag}");
        assert_eq!(hits(crate::renderer::FRONTIER_RAY_OFFSET + lag), live);
        assert_eq!(hits(crate::renderer::FRONTIER_RAY_OFFSET), 0, "static offset misses them");
    }

    #[test]
    fn back_step_inverts_drift() {
        let world = DynWorld::spawn(DemoScene::RelativisticBilliards);
        for body in &world.bodies {
            for w in body.history.windows(2).step_by(97) {
                let back = back_step(&w[1], body.radius, w[0].t);
                assert_eq!(back.step, w[0].step);
                assert!((back.tau - w[0].tau).abs() < 1e-9);
                for c in 0..3 {
                    assert!((back.pos[c] - w[0].pos[c]).abs() < 1e-9);
                    assert_eq!(back.vel[c], w[0].vel[c]);
                }
            }
        }
    }

    #[test]
    fn elastic_shot_breaks_the_rack() {
        let mut world = DynWorld::spawn(DemoScene::RelativisticBilliards);
        let vel = aim_at_rack(&world, 0.5);
        assert!(world.fire(world.frontier, vel, Ammo::Puck));
        run_parked(&mut world, 60.0);
        let moving = |b: &Body| b.latest().vel != [0.0; 3];
        let rack_moving = world.bodies[..10].iter().filter(|b| moving(b)).count();
        assert!(rack_moving >= 2, "only {rack_moving} rack pucks were set moving");
        assert_causal(&world);
    }

    #[test]
    fn sticky_shot_merges_into_a_heavier_puck() {
        let mut world = DynWorld::spawn(DemoScene::RelativisticBilliards);
        let vel = aim_at_rack(&world, 0.6);
        assert!(world.fire(world.frontier, vel, Ammo::Sticky));
        run_parked(&mut world, 60.0);

        let m = world.bodies.iter().find(|b| b.material_id == 9).expect("a merger happened");
        let first = m.history[0];
        let parents: Vec<_> = world
            .bodies
            .iter()
            .filter(|b| b.ended && b.uid != m.uid && b.latest().step == first.step - 1)
            .collect();
        assert!(parents.len() >= 2);
        let total = parents.iter().fold([0.0; 4], |acc, b| {
            collide::add(acc, collide::momentum(b.mass, b.latest().vel))
        });
        let merged = collide::momentum(m.mass, first.vel); // coasting from the merge event
        for c in 0..4 {
            assert!((total[c] - merged[c]).abs() < 1e-9, "4-momentum not conserved");
        }
        let rest_sum: f64 = parents.iter().map(|b| b.mass).sum();
        assert!(m.mass > rest_sum, "rest mass {} should exceed {rest_sum}", m.mass);
        assert!((m.radius - puck_radius(m.mass)).abs() < 1e-12);
        assert_causal(&world);
    }

    /// Time dilation: a spore fired at `v` pops after lab time `gamma(v) * lifetime`.
    #[test]
    fn spore_decays_after_dilated_lifetime() {
        let mut world = DynWorld::spawn(DemoScene::RelativisticBilliards);
        let speed = 0.8;
        let fired_at = world.frontier.origin.t as f64;
        // Straight back, away from the table.
        assert!(world.fire(world.frontier, [0.0, 0.0, -speed], Ammo::Spore));
        run_parked(&mut world, 40.0);
        let is_gen = |b: &&Body, g: u32| matches!(b.kind, Kind::Spore { generation, .. } if generation == g);
        let spore = world.bodies.iter().find(|b| is_gen(b, 0)).expect("spore");
        assert!(spore.ended, "spore should have popped");
        let Kind::Spore { lifetime, .. } = spore.kind else { unreachable!() };
        let expected = lifetime / (1.0 - speed * speed).sqrt();
        let popped = spore.latest().t - fired_at;
        assert!(
            (popped - expected).abs() <= STEP_DT + 1e-9,
            "popped after {popped}, expected {expected}"
        );
        assert_eq!(world.bodies.iter().filter(|b| is_gen(b, 1)).count(), 2);
        assert_causal(&world);
    }

    /// Ramming: a puck hitting the ship bounces off it, and the ship receives exactly the
    /// 4-momentum the puck lost.
    #[test]
    fn puck_hitting_the_ship_kicks_it() {
        let mut world = DynWorld::spawn(DemoScene::RelativisticBilliards);
        let o = world.frontier.origin;
        let step = (o.t as f64 / STEP_DT).floor() as i64 - 40;
        let t = step as f64 * STEP_DT;
        // A puck in front of the camera (off the table), heading straight at it at 0.6c.
        let z = o.z as f64 + 0.5 + 0.6 * (o.t as f64 - t);
        let state = BodyState { step, t, pos: [o.x as f64, o.y as f64, z], vel: [0.0, 0.0, -0.6], tau: 0.0 };
        let mut puck = billiards::body(42, Kind::Puck, PUCK_MASS, state, 7);
        puck.decided_at = vec![0.0];
        world.bodies.push(puck);
        // The parked ship's worldline over the puck's approach.
        let parked = ShipSample { t: t - 1.0, pos: [o.x as f64, o.y as f64, o.z as f64], vel: [0.0; 3] };
        world.ship.insert(0, parked);
        run_parked(&mut world, 10.0);

        assert_eq!(world.ship_kicks.len(), 1, "exactly one collision");
        let puck = world.bodies.iter().find(|b| b.uid == 42).unwrap();
        let before = collide::momentum(PUCK_MASS, [0.0, 0.0, -0.6]);
        let after = collide::momentum(PUCK_MASS, puck.latest().vel);
        let kick = world.ship_kicks[0];
        for c in 0..4 {
            assert!((before[c] - after[c] - kick[c]).abs() < 1e-9);
        }
        assert!(puck.latest().vel[2] > 0.0, "the puck bounced back");
        assert!(kick[3] < 0.0, "the ship is pushed away from the puck");
    }

    /// The causality property: two runs with the same input up to the camera event `O` but
    /// different input afterwards agree on every history step outside `I+(O)`.
    ///
    /// Contact acts across the touching bodies' radii within one step, so a difference can leak
    /// that far outside the cone; the comparison allows exactly that slack.
    #[test]
    fn input_after_o_never_changes_anything_outside_its_future_cone() {
        let path = gentle_path(900, 0.05);
        let diverge = 120;
        let mut same = DynWorld::spawn(DemoScene::RelativisticBilliards);
        let mut other = DynWorld::spawn(DemoScene::RelativisticBilliards);
        for w in [&mut same, &mut other] {
            for &(f, u) in &path[..10] {
                w.advance(f, u);
            }
            // A shared shot before O.
            let vel = aim_at_rack(w, 0.5);
            assert!(w.fire(path[10].0, vel, Ammo::Puck));
            for &(f, u) in &path[10..=diverge] {
                w.advance(f, u);
            }
        }
        let o = path[diverge].0.origin;

        // `same` carries on; `other` fires sticky pucks and spores at the rack right after O.
        for &(f, u) in &path[diverge + 1..] {
            same.advance(f, u);
        }
        for (i, &(f, u)) in path[diverge + 1..].iter().enumerate() {
            if i % 20 == 0 && i < 100 {
                let vel = aim_at_rack(&other, 0.55);
                other.fire(f, vel, if i % 40 == 0 { Ammo::Sticky } else { Ammo::Spore });
            }
            other.advance(f, u);
        }

        let slack = 2.0 * puck_radius(4.0 * PUCK_MASS) + STEP_DT;
        let (mut compared, mut differing_inside) = (0, 0);
        for a in &same.bodies {
            let Some(b) = other.bodies.iter().find(|b| b.uid == a.uid) else {
                differing_inside += 1; // e.g. merged away in `other`
                continue;
            };
            for s in &a.history {
                let Some(t) = b.at(s.step) else {
                    differing_inside += (b.ended && s.step > b.latest().step) as usize;
                    continue;
                };
                if is_decided(s, a.radius + slack, o) {
                    assert_eq!(s, t, "body {:x} differs outside I+(O) at step {}", a.uid, s.step);
                    compared += 1;
                } else if s != t {
                    differing_inside += 1;
                }
            }
        }
        println!("compared {compared} states outside I+(O); {differing_inside} differ inside");
        println!("bodies: same {} other {}", same.bodies.len(), other.bodies.len());
        assert!(compared > 1000, "compared only {compared} states");
        assert!(differing_inside > 0, "the later input should have changed something inside I+(O)");
    }
}

#[cfg(test)]
mod bench {
    /// Snapshot cost, for judging the 10 Hz rebuild budget (`cargo test --release -- --ignored`).
    #[test]
    #[ignore]
    fn time_build_scene() {
        let world = super::DynWorld::spawn(crate::scene::DemoScene::RelativisticBilliards);
        let t0 = std::time::Instant::now();
        for _ in 0..20 {
            std::hint::black_box(world.build_scene());
        }
        println!("build_scene: {:?}", t0.elapsed() / 20);
    }
}
