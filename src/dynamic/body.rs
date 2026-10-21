//! Bodies with a causally finalized history, and the frontier rule that gates their growth.

use super::collide::{self, V3};
use crate::mesh4d::{Mesh3D, WorldlineSample};
use crate::minkowski::{Tetrad, Vec3, Vec4};

/// Lab-time step of the global history grid shared by all bodies.
pub const STEP_DT: f64 = 0.1;

/// Full dynamic state of a point-like (rigid, non-rotating) body at one grid step, in `f64`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BodyState {
    /// Global grid step; `t` is always the world's time of this step.
    pub step: i64,
    /// Lab time
    pub t: f64,
    pub pos: V3,
    /// Lab 3-velocity (`c = 1`)
    pub vel: V3,
    /// Proper time of the body's center since it came into being
    pub tau: f64,
}

impl BodyState {
    pub fn gamma(&self) -> f64 {
        collide::gamma(self.vel)
    }

    pub fn speed(&self) -> f64 {
        collide::dot3(self.vel, self.vel).sqrt()
    }

    /// The center's spacetime event.
    pub fn event(&self) -> Vec4 {
        let [x, y, z] = self.pos;
        Vec4::new(self.t as f32, x as f32, y as f32, z as f32)
    }

    /// Coast inertially at `vel` to the next grid step at time `t_next`.
    pub fn drift(&self, vel: V3, t_next: f64) -> Self {
        let dt = t_next - self.t;
        let [x, y, z] = self.pos;
        Self {
            step: self.step + 1,
            t: t_next,
            pos: [x + vel[0] * dt, y + vel[1] * dt, z + vel[2] * dt],
            vel,
            tau: self.tau + dt / collide::gamma(vel),
        }
    }

    /// Render-side worldline sample (Born-rigid, unaccelerated between steps).
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

pub fn distance(a: V3, b: V3) -> f64 {
    let d = [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    collide::dot3(d, d).sqrt()
}

/// How far (in the lab frame, space and time combined) any vertex of a rigid body of rest
/// radius `radius` can stick out from its center: `gamma * r` in space plus `gamma * beta * r`
/// in time.
pub fn event_margin(state: &BodyState, radius: f64) -> f64 {
    state.gamma() * (1.0 + state.speed()) * radius
}

/// `true` if the whole body at `state` lies outside the open future light cone `I+(origin)`,
/// i.e. no future choice of the observer at `origin` can still influence it. Requiring the
/// center to clear the light cone by [`event_margin`] guarantees that every vertex does. The
/// boundary `dJ+(origin)` counts as decided.
pub fn is_decided(state: &BodyState, radius: f64, origin: Vec4) -> bool {
    let o = [origin.x as f64, origin.y as f64, origin.z as f64];
    state.t - origin.t as f64 + event_margin(state, radius) <= distance(state.pos, o)
}

/// What a body is, as far as its interactions are concerned.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Kind {
    /// Collides elastically (unless the other body is sticky).
    Puck,
    /// Merges with whatever it touches.
    Sticky,
    /// Splits into two fragments once its own clock reads `lifetime`; fragments of generation
    /// `generation > 0` are spores again with a shorter fuse, the last generation plain pucks.
    Spore { lifetime: f64, generation: u32 },
}

/// A body with a history finalized up to the causal frontier.
pub struct Body {
    /// Stable identity, derived from how the body came to be (so two runs of the world agree
    /// on it even when they create different bodies elsewhere).
    pub uid: u64,
    pub kind: Kind,
    /// Rest mass
    pub mass: f64,
    /// Rest-frame surface mesh
    pub mesh: Mesh3D,
    pub material_id: usize,
    /// Rest-frame bounding radius of `mesh`, also its contact radius
    pub radius: f64,
    /// Finalized states, one per grid step, oldest first (pruned behind the camera).
    pub history: Vec<BodyState>,
    /// Camera proper time at which each history step was finalized (parallel to `history`).
    pub decided_at: Vec<f32>,
    /// The body was created mid-scene (fired, merged, decayed), so its tube gets a start cap.
    pub born: bool,
    /// The body ceased to exist after its last history step (merged, decayed or left).
    pub ended: bool,
    /// Pass through the ship until first clear of it (a projectile leaving the launcher).
    pub ignore_ship: bool,
}

impl Body {
    pub fn latest(&self) -> &BodyState {
        self.history.last().expect("bodies always keep at least one state")
    }

    pub fn first_step(&self) -> i64 {
        self.history[0].step
    }

    /// State at global grid `step`, if finalized (and not yet pruned).
    pub fn at(&self, step: i64) -> Option<&BodyState> {
        let i = step - self.first_step();
        (i >= 0).then(|| self.history.get(i as usize)).flatten()
    }

    pub fn is_sticky(&self) -> bool {
        self.kind == Kind::Sticky
    }
}
