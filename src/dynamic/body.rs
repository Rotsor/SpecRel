//! Bodies with a causally finalized history, and the frontier rule that gates their growth.

use crate::mesh4d::{Mesh3D, WorldlineSample};
use crate::minkowski::{Tetrad, Vec3, Vec4};

/// Lab-time step of every body's history.
pub const STEP_DT: f64 = 0.1;

/// Full dynamic state of a point-like (rigid, non-rotating) body at one history step, in `f64`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BodyState {
    /// Lab time
    pub t: f64,
    pub pos: [f64; 3],
    /// Lab 3-velocity (`c = 1`)
    pub vel: [f64; 3],
    /// Proper time of the body's center
    pub tau: f64,
}

impl BodyState {
    pub fn gamma(&self) -> f64 {
        let [vx, vy, vz] = self.vel;
        1.0 / (1.0 - (vx * vx + vy * vy + vz * vz)).sqrt()
    }

    pub fn speed(&self) -> f64 {
        let [vx, vy, vz] = self.vel;
        (vx * vx + vy * vy + vz * vz).sqrt()
    }

    /// The center's spacetime event.
    pub fn event(&self) -> Vec4 {
        let [x, y, z] = self.pos;
        Vec4::new(self.t as f32, x as f32, y as f32, z as f32)
    }

    /// Coast inertially for lab time `dt`.
    pub fn drift(&self, dt: f64) -> Self {
        let [x, y, z] = self.pos;
        let [vx, vy, vz] = self.vel;
        Self {
            t: self.t + dt,
            pos: [x + vx * dt, y + vy * dt, z + vz * dt],
            vel: self.vel,
            tau: self.tau + dt / self.gamma(),
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

/// `true` if the whole body at `state` lies outside the open future light cone `I+(origin)`,
/// i.e. no future choice of the observer at `origin` can still influence it.
///
/// Every vertex `V` of a rigid body of rest radius `r` is `center + p^i e_i` with `|p| <= r`,
/// so in the lab frame it sits within `gamma * r` in space and `gamma * beta * r` in time of
/// the center. Requiring the center to clear the light cone by `gamma * (1 + beta) * r`
/// therefore guarantees that every vertex does. The boundary `dJ+(origin)` counts as decided.
pub fn is_decided(state: &BodyState, radius: f64, origin: Vec4) -> bool {
    let margin = state.gamma() * (1.0 + state.speed()) * radius;
    let dt = state.t - origin.t as f64;
    let [x, y, z] = state.pos;
    let (dx, dy, dz) = (x - origin.x as f64, y - origin.y as f64, z - origin.z as f64);
    dt + margin <= (dx * dx + dy * dy + dz * dz).sqrt()
}

/// How a body evolves from one history step to the next.
#[derive(Clone, Copy, Debug)]
pub enum Motion {
    /// Coasts forever.
    Inertial,
    /// Coasts and reflects elastically off the walls of an axis-aligned box (half-extents along
    /// `x` and `z`, centered on the origin) shrunk by the body's radius.
    Table { half_x: f64, z_min: f64, z_max: f64 },
}

/// A body with a history finalized up to the causal frontier.
pub struct Body {
    /// Rest-frame surface mesh
    pub mesh: Mesh3D,
    pub material_id: usize,
    /// Rest-frame bounding radius of `mesh`
    pub radius: f64,
    pub motion: Motion,
    /// Finalized states, one per [`STEP_DT`], oldest first (pruned behind the camera).
    pub history: Vec<BodyState>,
    /// Camera proper time at which each history step was finalized (parallel to `history`).
    pub decided_at: Vec<f32>,
}

impl Body {
    pub fn latest(&self) -> &BodyState {
        self.history.last().expect("bodies always keep at least one state")
    }

    /// The state one step after `state`. Inert bodies depend on nothing but themselves, so this
    /// never reads the observer or other bodies.
    pub fn step(&self, state: &BodyState) -> BodyState {
        let mut next = state.drift(STEP_DT);
        if let Motion::Table { half_x, z_min, z_max } = self.motion {
            let r = self.radius;
            reflect(&mut next.pos[0], &mut next.vel[0], -half_x + r, half_x - r);
            reflect(&mut next.pos[2], &mut next.vel[2], z_min + r, z_max - r);
        }
        next
    }

    /// The state one step *before* `state`, for back-filling history that predates the scene.
    pub fn step_back(&self, state: &BodyState) -> BodyState {
        let reversed = BodyState {
            vel: state.vel.map(|v| -v),
            ..*state
        };
        let mut prev = self.step(&BodyState { t: -reversed.t, tau: -reversed.tau, ..reversed });
        prev.t = -prev.t;
        prev.tau = -prev.tau;
        prev.vel = prev.vel.map(|v| -v);
        prev
    }
}

/// Fold `x` back into `[lo, hi]` by mirror reflection, flipping `v` on each bounce.
fn reflect(x: &mut f64, v: &mut f64, lo: f64, hi: f64) {
    if *x > hi {
        *x = 2.0 * hi - *x;
        *v = -*v;
    } else if *x < lo {
        *x = 2.0 * lo - *x;
        *v = -*v;
    }
}
