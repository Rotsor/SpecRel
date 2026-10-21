//! Scene 6: a billiard table of pucks, simulated causally. In Phase 3 the pucks are inert: a
//! rack at rest and a few movers bouncing between the rails, none of which the player can touch
//! yet. Their histories still only exist up to the camera's future light cone.

use super::body::{Body, BodyState, Motion};
use crate::mesh4d::Mesh3D;
use crate::minkowski::Vec3;

pub const HALF_X: f64 = 9.0;
pub const Z_MIN: f64 = 0.0;
pub const Z_MAX: f64 = 13.0;
pub const TABLE_Y: f32 = -2.2;
pub const PUCK_RADIUS: f64 = 0.45;

/// `[V]` matches the fastest mover's (initial) velocity.
pub const COMOVING_BETA: Vec3 = Vec3::new(0.8, 0.0, 0.0);

const MATERIAL_MOVING: usize = 7;
const MATERIAL_RESTING: usize = 8;

/// The pucks' states at lab time `t = 0`.
pub fn pucks() -> Vec<Body> {
    let y = TABLE_Y as f64 + PUCK_RADIUS;
    let puck = |x: f64, z: f64, vx: f64, vz: f64| {
        let moving = vx != 0.0 || vz != 0.0;
        Body {
            mesh: Mesh3D::uv_sphere(PUCK_RADIUS as f32, 4, 6),
            material_id: if moving { MATERIAL_MOVING } else { MATERIAL_RESTING },
            radius: PUCK_RADIUS,
            motion: Motion::Table { half_x: HALF_X, z_min: Z_MIN, z_max: Z_MAX },
            history: vec![BodyState { t: 0.0, pos: [x, y, z], vel: [vx, 0.0, vz], tau: 0.0 }],
            decided_at: Vec::new(),
        }
    };

    let mut bodies = Vec::new();
    // A triangular rack of resting pucks between the movers' lanes.
    for row in 0..4 {
        for i in 0..=row {
            let x = (i as f64 - row as f64 / 2.0) * 1.0;
            bodies.push(puck(x, 7.2 + row as f64 * 0.87, 0.0, 0.0));
        }
    }
    // Movers in separate lanes (they cannot collide until Phase 4).
    bodies.push(puck(-6.0, 2.0, 0.6, 0.0));
    bodies.push(puck(4.0, 4.6, -0.4, 0.0));
    bodies.push(puck(0.0, 11.8, 0.8, 0.0));
    bodies.push(puck(-7.5, 3.3, 0.3, 0.0));
    bodies
}

/// Rest-frame meshes and spatial centers of the table's static rails.
pub fn rails() -> Vec<(Mesh3D, Vec3)> {
    let h = 0.3;
    let y = TABLE_Y + h;
    let half_len_z = ((Z_MAX - Z_MIN) / 2.0) as f32;
    let mid_z = ((Z_MAX + Z_MIN) / 2.0) as f32;
    let rail_x = HALF_X as f32 + h;
    let side = Mesh3D::cube(Vec3::new(h, h, half_len_z + 2.0 * h));
    let end = Mesh3D::cube(Vec3::new(rail_x - h, h, h));
    vec![
        (side.clone(), Vec3::new(-rail_x, y, mid_z)),
        (side, Vec3::new(rail_x, y, mid_z)),
        (end, Vec3::new(0.0, y, Z_MAX as f32 + h)),
    ]
}
