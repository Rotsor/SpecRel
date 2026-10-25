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
pub mod helm;
pub mod ship;

use crate::bvh4d::Bvh4D;
use crate::mesh4d::{
    build_worldtube_cap, build_worldtube_simplices, build_worldtube_simplices_with_decided,
    sample_inertial_worldline, Mesh3D, DECIDED_AT_STATIC,
};
use crate::minkowski::{Vec3, Vec4};
use crate::renderer::{analytic_decision_age, Frontier};
use crate::scene::{DemoScene, Scene4D};
use billiards::{Ammo, SHIP_MASS, SHIP_RADIUS};
use body::{distance, is_decided, Body, BodyState, Impulse, Kind, STEP_DT};
use collide::{P4, V3};
use helm::ControlMode;
pub use ship::{PlayerId, Ship, ShipSample};

/// How far behind the camera (in lab time) histories are kept: beyond the renderers' maximum
/// ray length, nothing older can be seen.
pub const HISTORY_WINDOW: f64 = 200.0;

/// Render segments are split at least this often (in lab time), so that the per-vertex
/// `decided_at` interpolated along a merged segment stays close to the per-step stamps.
const MAX_SEGMENT_DT: f64 = 50.0;

const MATERIAL_RAIL: usize = 6;

/// What a body touches at a grid step.
struct Contacts {
    /// Indices of bodies overlapping it, sorted by uid.
    bodies: Vec<usize>,
    /// The ships overlapping it, and their states.
    ships: Vec<(PlayerId, ShipSample)>,
    /// The ship that launched it is known to be clear of it (ends the launch grace).
    launcher_clear: bool,
}

pub struct DynWorld {
    pub demo: DemoScene,
    pub bodies: Vec<Body>,
    /// Every player's ship, indexed by [`PlayerId`]. Their latest events bound the decided
    /// region: a step is finalized only once it is outside every live ship's future cone, and
    /// bodies near a ship wait for its track.
    pub ships: Vec<Ship>,
    /// Whose camera this replica of the world is rendered for: its ship is not drawn, its
    /// clock stamps `decided_at`, and only its collision kicks are collected.
    pub viewer: PlayerId,
    /// The viewer's frontier, as the histories were last extended.
    pub frontier: Frontier,
    /// Collision impulses delivered to the viewer's ship. The camera is the ship's center, so
    /// `App` applies each once its news has crossed the hull ([`Impulse::has_reached`]).
    pub ship_kicks: Vec<Impulse>,
    /// Lab time of grid step 0.
    t_origin: f64,
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

    /// A single-player world with the observer at `frontier`, moving with 4-velocity `u`.
    pub fn new(demo: DemoScene, frontier: Frontier, u: Vec4) -> Self {
        Self::new_match(demo, &[ShipSample::new(frontier, u)], 0)
    }

    /// Start the world with one ship per player at the given events, rendered for `viewer`.
    /// Everything outside the ships' future cones is already decided: histories are filled
    /// from [`HISTORY_WINDOW`] in the past up to the frontier, stamped with the analytic
    /// decision time as if the viewer had always coasted. (The scene is laid out so nothing
    /// collides in that past.) Every replica of a match builds identical histories.
    pub fn new_match(demo: DemoScene, spawns: &[ShipSample], viewer: PlayerId) -> Self {
        let mut bodies = match demo {
            DemoScene::RelativisticBilliards => billiards::pucks(),
            _ => panic!("{demo:?} is not a dynamic scene"),
        };
        let me = spawns[viewer as usize];
        let [ut, ux, uy, uz] = me.four_velocity().map(|c| c as f32);
        let (frontier, u) = (me.frontier(), Vec4::new(ut, ux, uy, uz));
        let t_origin = 0.0;
        let t_first = spawns.iter().map(|s| s.t).fold(f64::INFINITY, f64::min) - HISTORY_WINDOW;
        let first_step = ((t_first - t_origin) / STEP_DT).floor() as i64;
        for body in &mut bodies {
            // Bodies are specified at step 0; walk back to the start of the window.
            let mut s = body.history[0];
            while s.step > first_step {
                s = back_step(&s, body.radius, t_origin + (s.step - 1) as f64 * STEP_DT);
            }
            body.history = vec![s];
            body.decided_at = vec![decision_time(&s, frontier, u)];
        }
        let ships = spawns.iter().enumerate().map(|(p, &s)| Ship::new(p as PlayerId, s)).collect();
        let mut world = Self {
            demo,
            bodies,
            ships,
            viewer,
            frontier,
            ship_kicks: Vec::new(),
            t_origin,
        };
        world.extend_with(|s| decision_time(s, frontier, u));
        world
    }

    fn step_time(&self, step: i64) -> f64 {
        self.t_origin + step as f64 * STEP_DT
    }

    /// Move the viewer's frontier to its pilot's new event (moving with 4-velocity `u`) and
    /// finalize every step that just became decided. Returns whether anything changed.
    pub fn advance(&mut self, frontier: Frontier, u: Vec4) -> bool {
        self.record(self.viewer, ShipSample::new(frontier, u));
        self.extend()
    }

    /// Record `player`'s pilot's next event (their hull follows it). Nothing is finalized
    /// until [`DynWorld::extend`].
    pub fn record(&mut self, player: PlayerId, pilot: ShipSample) {
        self.ships[player as usize].record(pilot);
    }

    /// `player` flies with `controls` (only before their ship has moved).
    pub fn set_controls(&mut self, player: PlayerId, controls: ControlMode) {
        self.ships[player as usize].set_controls(controls);
    }

    /// `player` left the match: their ship ends where it was last seen.
    pub fn leave(&mut self, player: PlayerId) {
        self.ships[player as usize].left = true;
    }

    /// Finalize every step that the ships' latest events have decided. Returns whether
    /// anything changed.
    pub fn extend(&mut self) -> bool {
        let tau = self.ships[self.viewer as usize].latest().tau as f32;
        self.extend_with(|_| tau)
    }

    /// Whether the table has room for another projectile.
    pub fn can_fire(&self) -> bool {
        self.bodies.iter().filter(|b| !b.ended).count() < billiards::MAX_BODIES
    }

    /// The viewer fires a body of `ammo` at `at`, if the table has room: see
    /// [`DynWorld::launch`]. Returns whether it was fired.
    pub fn fire(&mut self, at: Frontier, vel: V3, ammo: Ammo) -> bool {
        let fire = self.can_fire();
        if fire {
            self.launch(self.viewer, at, vel, ammo);
        }
        fire
    }

    /// `player`'s ship fires a body of `ammo` at `at`, with lab 3-velocity `vel`. It leaves the
    /// ship's center, so its whole worldline lies in the firing event's causal future; its
    /// first grid state is fixed by this input alone. Always launches: whether there was room
    /// is the shooter's call (live-body counts differ between replicas at any given moment).
    pub fn launch(&mut self, player: PlayerId, at: Frontier, vel: V3, ammo: Ammo) {
        let o = at.origin;
        let step = ((o.t as f64 - self.t_origin) / STEP_DT).ceil() as i64;
        let t = self.step_time(step);
        let dt = t - o.t as f64;
        let pos = [o.x as f64 + vel[0] * dt, o.y as f64 + vel[1] * dt, o.z as f64 + vel[2] * dt];
        let state = BodyState { step, t, pos, vel, tau: dt / collide::gamma(vel) };
        let ship = &mut self.ships[player as usize];
        let mut b = billiards::projectile(player, ship.shots, ammo, state);
        ship.shots += 1;
        b.decided_at = vec![self.frontier.proper_time];
        self.bodies.push(b);
    }

    /// `true` if the whole body at `state` is outside every live ship's future cone.
    fn is_decided(&self, state: &BodyState, radius: f64) -> bool {
        self.ships
            .iter()
            .filter(|s| s.is_live())
            .all(|s| is_decided(state, radius, s.latest().event()))
    }

    /// Scheduler: keep passing over the bodies, advancing each as far as it can, until a pass
    /// makes no progress. Then prune what has fallen behind the history window.
    fn extend_with(&mut self, stamp: impl Fn(&BodyState) -> f32) -> bool {
        self.frontier = self.ships[self.viewer as usize].latest().frontier();
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

    /// Advance body `i` by one grid step (or end it in a decay). Returns `false` if it has to
    /// wait: for the frontier, or for a neighbor or the ship to be known.
    ///
    /// Contacts that begin at step `k` compute their impulse from both bodies' states at the
    /// touching point, but the body only feels it once news of the contact has crossed it
    /// ([`Impulse::has_reached`]); so every state depends only on its own causal past.
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
        let me = &self.bodies[i];
        let r = me.radius;

        // New contacts at step k.
        let mut received = Vec::new();
        for &j in &contacts.bodies {
            let other = &self.bodies[j];
            if me.touching.contains(&other.uid) {
                continue;
            }
            let o = other.at(k).expect("contacts exist at step k");
            if let Some(dp) = contact_impulse(me, &s, other, o) {
                let at = touching_point(s.pos, r, o.pos, other.radius);
                received.push(Impulse { t: s.t, pos: at, dp });
            }
        }
        // Ships are far heavier than pucks, and each player's replica applies the kicks to its
        // own ship; here only the viewer's are collected. A control pod's hull isn't kicked at
        // all: its motion is a function of its pilot's track alone.
        let mut kick = None;
        for &(player, ship) in &contacts.ships {
            if me.touching_ships.contains(&player) {
                continue;
            }
            let sep = sub3(ship.pos, s.pos);
            if let Some((v, _)) = collide::elastic(me.mass, s.vel, SHIP_MASS, ship.vel, sep) {
                let at = touching_point(s.pos, r, ship.pos, SHIP_RADIUS);
                let dp = collide::sub(collide::momentum(me.mass, v), collide::momentum(me.mass, s.vel));
                received.push(Impulse { t: s.t, pos: at, dp });
                if player == self.viewer && self.ships[player as usize].controls == ControlMode::Delayed {
                    kick = Some(Impulse { t: s.t, pos: at, dp: dp.map(|c| -c) });
                }
            }
        }

        // Coast to step k + 1, then feel every impulse whose news has crossed the body by then.
        let mut next = s.drift(s.vel, self.step_time(k + 1));
        billiards::constrain(&s, &mut next, r);
        let (pending, arrived): (Vec<Impulse>, Vec<Impulse>) = me
            .pending
            .iter()
            .chain(&received)
            .copied()
            .partition(|imp| !imp.has_reached(&next, r));
        let mut mass = me.mass;
        for imp in &arrived {
            let (m, v) = collide::mass_and_velocity(collide::add(collide::momentum(mass, next.vel), imp.dp));
            mass = m;
            next.vel = v;
        }
        if !self.is_decided(&next, r) {
            return false;
        }

        let stamp = stamp(&next);
        let touching = contacts.bodies.iter().map(|&j| self.bodies[j].uid).collect();
        let body = &mut self.bodies[i];
        if mass > body.mass * (1.0 + 1e-9) && body.material_at(next.step) != billiards::MATERIAL_HEATED {
            body.material_changes.push((next.step, billiards::MATERIAL_HEATED));
        }
        if mass != body.mass {
            body.mass_changes.push((next.step, mass));
        }
        body.mass = mass;
        body.pending = pending;
        body.touching = touching;
        body.touching_ships = contacts.ships.iter().map(|&(p, _)| p).collect();
        body.history.push(next);
        body.decided_at.push(stamp);
        body.ended = billiards::out_of_play(&next);
        if contacts.launcher_clear {
            body.launcher = None;
        }
        self.ship_kicks.extend(kick);
        true
    }

    /// Everything body `i` touches at step `k`, or `None` if something that might be touching
    /// it is not known yet: a body still behind step `k` that could reach contact range at
    /// light speed, or a ship beyond its recorded track.
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
        let (mut ships, mut launcher_clear) = (Vec::new(), false);
        for ship in &self.ships {
            let launcher = me.launcher == Some(ship.player);
            if ship.is_known_at(s.t) {
                let at = ship.hull_at(s.t);
                let d = distance(s.pos, at.pos);
                if launcher {
                    launcher_clear = d > reach;
                } else if d < reach {
                    ships.push((ship.player, at));
                }
            } else if ship.is_live() && !launcher {
                // Unknown yet: wait if it could be within reach by now.
                let last = ship.latest_hull();
                if distance(s.pos, last.pos) < reach + (s.t - last.t) {
                    return None;
                }
            }
        }
        Some(Contacts { bodies: touching, ships, launcher_clear })
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
            if !self.is_decided(&first, f.radius) {
                return false;
            }
            f.history = vec![first];
            f.decided_at = vec![stamp(&first)];
        }
        self.bodies[i].ended = true;
        self.bodies.extend(fragments);
        true
    }

    /// Drop what is older than [`HISTORY_WINDOW`] behind the *earliest* live ship: a ship that
    /// lags behind holds back bodies near it, and they still need their neighbors' states.
    fn prune(&mut self) {
        let earliest = self.ships.iter().filter(|s| s.is_live()).map(|s| s.latest().t);
        let horizon = earliest.fold(self.frontier.origin.t as f64, f64::min) - HISTORY_WINDOW;
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
        for ship in &mut self.ships {
            ship.prune(horizon);
        }
    }

    /// The latest known events of the other live ships: the apexes of the cones that, besides
    /// the viewer's own, keep the world undecided.
    pub fn remote_origins(&self) -> Vec<Vec4> {
        self.ships
            .iter()
            .filter(|s| s.is_live() && s.player != self.viewer)
            .map(|s| s.latest().event())
            .collect()
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
            for imp in &mut body.pending {
                imp.t += dt;
            }
        }
        for ship in &mut self.ships {
            ship.rebase(dt);
        }
        for kick in &mut self.ship_kicks {
            kick.t += dt;
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
            let material: Vec<usize> =
                keep.iter().map(|&i| body.material_at(body.history[i].step)).collect();
            // One tube per run of prisms of equal material (a prism takes its later end's).
            let mut a = 0;
            while a + 1 < samples.len() {
                let m = material[a + 1];
                let mut b = a + 1;
                while b + 1 < samples.len() && material[b + 1] == m {
                    b += 1;
                }
                simplices.extend(build_worldtube_simplices_with_decided(
                    &body.mesh,
                    &samples[a..=b],
                    Some(&decided[a..=b]),
                    m,
                ));
                a = b;
            }
            let mut cap = |i: usize| {
                simplices.extend(build_worldtube_cap(&body.mesh, &samples[i], decided[i], material[i]));
            };
            cap(samples.len() - 1);
            if body.born && samples.len() > 1 {
                cap(0);
            }
        }

        // The other players' ships, up to their last known events: an open hull cage, and the
        // pilot's clock inside it, where the decisions are made.
        let (hull_mesh, pilot_mesh) = (Ship::hull_mesh(), Ship::pilot_mesh());
        for ship in self.ships.iter().filter(|s| s.player != self.viewer) {
            for (track, mesh, m) in [
                (&ship.hull, &hull_mesh, ship.hull_material()),
                (&ship.track, &pilot_mesh, ship.material()),
            ] {
                let samples: Vec<_> =
                    ship::render_indices(track, MAX_SEGMENT_DT).iter().map(|&i| track[i].sample()).collect();
                simplices.extend(build_worldtube_simplices(mesh, &samples, m));
                simplices.extend(build_worldtube_cap(mesh, samples.last().unwrap(), DECIDED_AT_STATIC, m));
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

/// Where two touching spheres meet: on the line between their centers, split by their radii.
fn touching_point(a: V3, r_a: f64, b: V3, r_b: f64) -> V3 {
    let f = r_a / (r_a + r_b);
    [0, 1, 2].map(|c| a[c] + (b[c] - a[c]) * f)
}

/// The 4-momentum body `me` (at state `s`) receives from a contact beginning with `other` (at
/// `o`), or `None` if they are already separating. Sticky contacts are perfectly inelastic,
/// others elastic. Evaluated in a canonical order (lower uid first) so both bodies compute the
/// same collision and their impulses cancel exactly.
fn contact_impulse(me: &Body, s: &BodyState, other: &Body, o: &BodyState) -> Option<P4> {
    let sep = sub3(o.pos, s.pos);
    let closing = collide::dot3(sub3(o.vel, s.vel), sep) < 0.0;
    if !closing {
        return None;
    }
    let me_first = me.uid < other.uid;
    let ((ma, va), (mb, vb), sep) = if me_first {
        ((me.mass, s.vel), (other.mass_at(s.step), o.vel), sep)
    } else {
        ((other.mass_at(s.step), o.vel), (me.mass, s.vel), sep.map(|c| -c))
    };
    let (dpa, dpb) = if me.is_sticky() || other.is_sticky() {
        collide::stick(ma, va, mb, vb)
    } else {
        let (va2, vb2) = collide::elastic(ma, va, mb, vb, sep)?;
        (
            collide::sub(collide::momentum(ma, va2), collide::momentum(ma, va)),
            collide::sub(collide::momentum(mb, vb2), collide::momentum(mb, vb)),
        )
    };
    Some(if me_first { dpa } else { dpb })
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
    use super::billiards::PUCK_MASS;
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
    pub(super) fn assert_causal(world: &DynWorld) {
        let o = world.frontier;
        for body in &world.bodies {
            assert_eq!(body.history.len(), body.decided_at.len());
            for (i, (s, &d)) in body.history.iter().zip(&body.decided_at).enumerate() {
                let launch = body.born && i == 0;
                assert!(
                    launch || world.is_decided(s, body.radius),
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

    /// Newton's cradle: a line of nearly touching pucks struck at one end. The far puck may not
    /// start moving before light from the first contact could have crossed all of it — rigid
    /// contact would hand the push down the line by a diameter per step, far faster than `c`.
    #[test]
    fn cradle_pushes_no_faster_than_light() {
        let mut world = DynWorld::spawn(DemoScene::RelativisticBilliards);
        let step = (world.frontier.origin.t as f64 / STEP_DT).floor() as i64 - 1;
        let t = step as f64 * STEP_DT;
        let r = billiards::PUCK_RADIUS;
        let gap = 0.01;
        let puck = |uid: u64, x: f64, vx: f64| {
            let state = BodyState { step, t, pos: [x, 0.0, -25.0], vel: [vx, 0.0, 0.0], tau: 0.0 };
            let mut b = billiards::body(uid, Kind::Puck, PUCK_MASS, state, 8);
            b.decided_at = vec![0.0];
            b
        };
        // Off the table, far from the camera: decided well into the future.
        world.bodies = (0..6).map(|n| puck(n, n as f64 * (2.0 * r + gap), 0.0)).collect();
        world.bodies.push(puck(99, -1.0, 0.5));
        run_parked(&mut world, 30.0);

        let striker = world.bodies.iter().find(|b| b.uid == 99).unwrap();
        let first = &world.bodies[0];
        let contact = striker
            .history
            .iter()
            .find(|s| first.at(s.step).is_some_and(|f| distance(s.pos, f.pos) < 2.0 * r))
            .expect("striker reaches the cradle");
        let x0 = contact.pos[0] + r; // the touching point
        let last = &world.bodies[5];
        let moved = last.history.iter().find(|s| s.vel[0] != 0.0).expect("push arrives");
        let news_needed = (last.history[0].pos[0] - x0) + r; // reach its far side
        let took = moved.t - contact.t;
        println!("push crossed the cradle in {took:.2} s; light needs {news_needed:.2} s");
        assert!(took >= news_needed - 1e-9, "push took {took:.2} s, light needs {news_needed:.2} s");
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
    fn sticky_shot_sticks_and_turns_kinetic_energy_into_mass() {
        let mut world = DynWorld::spawn(DemoScene::RelativisticBilliards);
        let vel = aim_at_rack(&world, 0.6);
        assert!(world.fire(world.frontier, vel, Ammo::Sticky));
        run_parked(&mut world, 60.0);

        // The shot and the first puck it hit each felt their inelastic share (the pair may go
        // on to stick to more of the rack): both glow from then on, both are heavier, and right
        // after it they move together.
        let heated: Vec<_> = world.bodies.iter().filter(|b| !b.material_changes.is_empty()).collect();
        let shot = heated.iter().find(|b| b.is_sticky()).expect("the shot heated up");
        let target = heated
            .iter()
            .filter(|b| !b.is_sticky())
            .min_by_key(|b| b.material_changes[0].0)
            .expect("the puck it hit heated up");
        let vel_after = |b: &Body| b.at(b.material_changes[0].0).unwrap().vel;
        let (va, vb) = (vel_after(shot), vel_after(target));
        for c in 0..3 {
            assert!((va[c] - vb[c]).abs() < 1e-9, "co-moving after sticking: {va:?} vs {vb:?}");
        }
        assert!(heated.iter().all(|b| b.mass > PUCK_MASS), "every stuck puck gained rest mass");
        assert!(heated.iter().all(|b| b.material_changes[0].1 == billiards::MATERIAL_HEATED));
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
        let parked = ShipSample { t: t - 1.0, pos: [o.x as f64, o.y as f64, o.z as f64], vel: [0.0; 3], tau: 0.0 };
        world.ships[0].track.insert(0, parked);
        world.ships[0].hull.insert(0, parked);
        run_parked(&mut world, 10.0);

        assert_eq!(world.ship_kicks.len(), 1, "exactly one collision");
        let puck = world.bodies.iter().find(|b| b.uid == 42).unwrap();
        let before = collide::momentum(PUCK_MASS, [0.0, 0.0, -0.6]);
        let after = collide::momentum(PUCK_MASS, puck.latest().vel);
        let kick = world.ship_kicks[0].dp;
        for c in 0..4 {
            assert!((before[c] - after[c] - kick[c]).abs() < 1e-9);
        }
        assert!(puck.latest().vel[2] > 0.0, "the puck bounced back");
        assert!(kick[3] < 0.0, "the ship is pushed away from the puck");
    }

    /// The causality property: two runs with the same input up to the camera event `O` but
    /// different input afterwards agree on every history step outside `I+(O)`.
    ///
    /// No slack: a contact only changes a body once its news has crossed the whole body, so
    /// every state depends on its own causal past alone (see `cradle_pushes_no_faster_than_light`).
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
                if is_decided(s, a.radius, o) {
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

/// Two-player matches: each player runs a replica of the world fed with both ships' inputs,
/// the other player's arriving late.
#[cfg(test)]
mod match_tests {
    use super::tests::assert_causal;
    use super::*;

    /// One player's inputs, in the order their frontend produces them.
    #[derive(Clone, Copy)]
    enum Input {
        Sample(ShipSample),
        Fire(Frontier, V3, Ammo),
    }

    fn spawns() -> Vec<ShipSample> {
        (0..2).map(|p| ShipSample::at_rest(billiards::spawn_event(p))).collect()
    }

    /// `n` frames of `dtau` for `player`: drift toward the table at `speed` along z (no
    /// thrusting, so `f32` camera rounding doesn't matter), firing `ammo` at the rack every
    /// `fire_every` frames.
    fn inputs(player: PlayerId, n: usize, dtau: f32, speed: f32, fire_every: usize, ammo: Ammo) -> Vec<Input> {
        let spawn = billiards::spawn_event(player);
        let u = Vec4::four_velocity_from_beta(Vec3::new(0.0, 0.0, speed));
        let mut out = Vec::new();
        for i in 1..=n {
            let tau = i as f32 * dtau;
            let f = Frontier { origin: spawn + u * tau, proper_time: tau };
            let sample = ShipSample::new(f, u);
            out.push(Input::Sample(sample));
            if i % fire_every == 0 {
                let rack = [0.0, billiards::TABLE_Y as f64 + billiards::PUCK_RADIUS, 9.0];
                let d = sub3(rack, sample.pos);
                let l = collide::dot3(d, d).sqrt();
                out.push(Input::Fire(f, d.map(|c| c / l * 0.6), ammo));
            }
        }
        out
    }

    fn apply(world: &mut DynWorld, player: PlayerId, input: Input) {
        match input {
            Input::Sample(s) => world.record(player, s),
            Input::Fire(at, vel, ammo) => world.launch(player, at, vel, ammo),
        }
    }

    /// Run the viewer's replica: per frame, its own next input and the other player's input
    /// from `delay` frames ago, extending after each frame.
    fn replica(viewer: PlayerId, streams: &[Vec<Input>; 2], delay: usize) -> DynWorld {
        let mut world = DynWorld::new_match(DemoScene::RelativisticBilliards, &spawns(), viewer);
        let other = 1 - viewer;
        let (mine, theirs) = (&streams[viewer as usize], &streams[other as usize]);
        for i in 0..mine.len().max(theirs.len()) + delay {
            if let Some(&input) = mine.get(i) {
                apply(&mut world, viewer, input);
            }
            if let Some(&input) = i.checked_sub(delay).and_then(|j| theirs.get(j)) {
                apply(&mut world, other, input);
            }
            world.extend();
            if i % 97 == 0 {
                assert_causal(&world);
            }
        }
        world
    }

    fn streams() -> [Vec<Input>; 2] {
        [
            inputs(0, 700, 0.05, 0.3, 90, Ammo::Puck),
            inputs(1, 650, 0.05, -0.2, 70, Ammo::Sticky),
        ]
    }

    /// Both replicas finalize bit-identical histories, however late each hears of the other.
    /// (This caught contacts reading the other body's *current* rest mass rather than its
    /// mass at the contact step: whether a sticky hit had already landed depended on timing.)
    #[test]
    fn replicas_agree_whatever_the_latency() {
        let streams = streams();
        let a = replica(0, &streams, 3);
        let b = replica(1, &streams, 41);
        let mut compared = 0;
        for x in &a.bodies {
            let y = b.bodies.iter().find(|y| y.uid == x.uid).expect("same bodies in both replicas");
            for s in &x.history {
                if let Some(t) = y.at(s.step) {
                    assert_eq!(s, t, "body {:x} differs at step {}", x.uid, s.step);
                    compared += 1;
                }
            }
        }
        let fired = a.bodies.iter().filter(|b| b.born).count();
        let heated = a.bodies.iter().filter(|b| !b.material_changes.is_empty()).count();
        println!("compared {compared} states; {fired} born bodies, {heated} heated");
        assert!(compared > 5000);
        assert!(fired >= 10 && heated > 0, "the shots should have hit something");
        assert_eq!(a.bodies.len(), b.bodies.len());
    }

    /// A player who never moves holds back everything near them: nothing is finalized inside
    /// their future cone, however far the other player gets.
    #[test]
    fn an_idle_player_holds_back_the_world_near_them() {
        let mut world = DynWorld::new_match(DemoScene::RelativisticBilliards, &spawns(), 0);
        for input in inputs(0, 400, 0.05, 0.0, usize::MAX, Ammo::Puck) {
            apply(&mut world, 0, input);
            world.extend();
        }
        assert_causal(&world);
        let idle = spawns()[1].event();
        let near = world.bodies.iter().filter(|b| distance(b.latest().pos, spawns()[1].pos) < 8.0);
        for body in near {
            assert!(body.latest().t < idle.t as f64 + 8.0, "body {:x} ran ahead", body.uid);
        }
        // Once they leave, the world catches up with the remaining player.
        world.leave(1);
        world.extend();
        let lag = world.frontier_lag(Vec4::new(1.0, 0.0, 0.0, 0.0));
        assert!(lag < 3.0, "lag {lag}");
    }

    /// Each player sees the other's ship as a hull cage with the pilot's clock inside, up to its
    /// last known event, and not their own.
    #[test]
    fn other_ships_are_drawn_up_to_their_last_known_event() {
        let streams = streams();
        let world = replica(0, &streams, 5);
        let scene = world.build_scene();
        let has = |m: usize| scene.bvh.simplices.iter().any(|s| s.material_id == m);
        assert!(has(world.ships[1].material()), "the other pilot is drawn");
        assert!(has(world.ships[1].hull_material()), "and their hull");
        assert!(!has(world.ships[0].material()), "your own ship is not");
        assert!(!has(world.ships[0].hull_material()));
        let tip = world.ships[1].latest().t as f32;
        let latest = scene
            .bvh
            .simplices
            .iter()
            .filter(|s| s.material_id == world.ships[1].material())
            .map(|s| s.v0.t)
            .fold(f32::MIN, f32::max);
        assert!(latest <= tip + 1.0 && latest >= tip - 1.0, "ship drawn to {latest}, known to {tip}");
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
