//! Scene 6: relativistic billiards. A rack of resting pucks and a few movers bouncing between
//! the rails. The ships don't touch them (that couldn't be made causal): they shoot lasers
//! ([`super::laser`]), which the pucks stop.

use super::body::{distance, Body, BodyState};
use super::ship::PlayerId;
use crate::mesh4d::Mesh3D;
use crate::minkowski::{Vec3, Vec4};

pub const HALF_X: f64 = 9.0;
pub const Z_MIN: f64 = 0.0;
pub const Z_MAX: f64 = 13.0;
pub const TABLE_Y: f32 = -2.2;
pub const PUCK_RADIUS: f64 = 0.35;
pub const PUCK_MASS: f64 = 1.0;
/// The ship's hull is a sphere around its camera (you never see your own: you are inside it).
pub const SHIP_RADIUS: f64 = 0.4;
/// The ship's proper acceleration (at its center, in `c` per second) in a match. A Born-rigid
/// body accelerating at `a` has a Rindler horizon `1 / a` behind its center, and the hull's
/// trailing side must stay in front of it (its own acceleration `a / (1 - a R)` diverges as
/// `a R -> 1`), so `a < 1 / SHIP_RADIUS`. Thrusters and steering run at 80% of that bound.
pub const SHIP_ACCEL: f64 = 0.8 / SHIP_RADIUS;
/// With a control pod (`docs/SHIP_CONTROLS_CAUSALITY.md`, option C) the pilot floats about
/// this far from the hull's center at most under steady thrust. Nothing stops it going further.
pub const POD_RANGE: f64 = SHIP_RADIUS / 2.0;
/// Bodies this far from the table leave play.
pub const DESPAWN_DISTANCE: f64 = 150.0;
/// Half the side of the felt (the ground grid), centered on the lab origin.
pub const FELT_HALF_SIZE: f32 = 24.0;

/// `[V]` matches the fastest mover's (initial) velocity.
pub const COMOVING_BETA: Vec3 = Vec3::new(0.7, 0.0, 0.0);

const MATERIAL_MOVING: usize = 7;
const MATERIAL_RESTING: usize = 8;

/// SplitMix64-style mixing, for stable body identities.
pub fn mix(a: u64, b: u64) -> u64 {
    let mut z = a ^ b.wrapping_add(0x9E37_79B9_7F4A_7C15).wrapping_add(a << 6);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// A puck of rest mass `mass`, starting at `state`.
pub fn body(uid: u64, mass: f64, state: BodyState, material_id: usize) -> Body {
    let radius = PUCK_RADIUS * (mass / PUCK_MASS).cbrt(); // constant density
    Body {
        uid,
        mass,
        mesh: Mesh3D::uv_sphere(radius as f32, 4, 6),
        material_id,
        radius,
        history: vec![state],
        decided_at: Vec::new(),
        ended: false,
        pending: Vec::new(),
        touching: Vec::new(),
        mass_changes: vec![(state.step, mass)],
    }
}

/// The table's pucks at grid step 0 (`t = 0`).
pub fn pucks() -> Vec<Body> {
    let y = TABLE_Y as f64 + PUCK_RADIUS;
    let mut bodies = Vec::new();
    let mut add = |x: f64, z: f64, vx: f64| {
        let state = BodyState { step: 0, t: 0.0, pos: [x, y, z], vel: [vx, 0.0, 0.0], tau: 0.0 };
        let material = if vx != 0.0 { MATERIAL_MOVING } else { MATERIAL_RESTING };
        let uid = mix(0x7AB1E, bodies.len() as u64);
        bodies.push(body(uid, PUCK_MASS, state, material));
    };
    // A spaced triangular rack. The gaps matter: contact acts across a puck's diameter in one
    // step, so a momentum kick hops from puck to puck faster than the pucks move. With gaps of
    // ~1.1 between neighbours the hop stays below c for impact speeds up to ~0.6c.
    for row in 0..4 {
        for i in 0..=row {
            add((i as f64 - row as f64 / 2.0) * 1.8, 7.0 + row as f64 * 1.56, 0.0);
        }
    }
    // Movers in separate lanes, bouncing between the side rails.
    add(-6.0, 1.5, 0.5);
    add(4.0, 3.0, -0.35);
    add(-1.0, 4.5, 0.7);
    bodies
}

/// Rails and felt. A body over the table that crosses a rail between `prev` and `next` is
/// mirrored back with that velocity component reversed; one that comes down onto the felt lands
/// there and keeps sliding (the felt absorbs vertical motion, so shots fired from above end up
/// in the table plane). The near side (toward the camera's spawn point) is open, and bodies
/// arriving from outside pass over the rails.
pub fn constrain(prev: &BodyState, next: &mut BodyState, r: f64) {
    let over_table = |s: &BodyState| s.pos[0].abs() <= HALF_X && (Z_MIN..=Z_MAX).contains(&s.pos[2]);
    if !over_table(prev) {
        return;
    }
    let mut mirror = |axis: usize, wall: f64, inside_below: bool| {
        let (was_in, now_out) = if inside_below {
            (prev.pos[axis] <= wall, next.pos[axis] > wall)
        } else {
            (prev.pos[axis] >= wall, next.pos[axis] < wall)
        };
        if was_in && now_out {
            next.pos[axis] = 2.0 * wall - next.pos[axis];
            next.vel[axis] = -next.vel[axis];
        }
    };
    mirror(0, HALF_X - r, true);
    mirror(0, -HALF_X + r, false);
    mirror(2, Z_MAX - r, true);
    let felt = TABLE_Y as f64 + r;
    if prev.pos[1] >= felt && next.pos[1] < felt {
        next.pos[1] = felt;
        next.vel[1] = 0.0;
    }
}

pub fn out_of_play(state: &BodyState) -> bool {
    distance(state.pos, [0.0, TABLE_Y as f64, (Z_MIN + Z_MAX) / 2.0]) > DESPAWN_DISTANCE
}

/// The table's static rails, as boxes: `(center, half extents)`.
pub fn rail_boxes() -> [(Vec3, Vec3); 3] {
    let h = 0.3;
    let y = TABLE_Y + h;
    let half_len_z = ((Z_MAX - Z_MIN) / 2.0) as f32;
    let mid_z = ((Z_MAX + Z_MIN) / 2.0) as f32;
    let rail_x = HALF_X as f32 + h;
    let side = Vec3::new(h, h, half_len_z + 2.0 * h);
    [
        (Vec3::new(-rail_x, y, mid_z), side),
        (Vec3::new(rail_x, y, mid_z), side),
        (Vec3::new(0.0, y, Z_MAX as f32 + h), Vec3::new(rail_x - h, h, h)),
    ]
}

/// Rest-frame meshes and spatial centers of the table's static rails.
pub fn rails() -> Vec<(Mesh3D, Vec3)> {
    rail_boxes().into_iter().map(|(center, he)| (Mesh3D::cube(he), center)).collect()
}

/// Where `player` spawns, at `t = 0`: player 0 at the open near end of the table, the others
/// mirrored to the far end.
pub fn spawn_event(player: PlayerId) -> Vec4 {
    let near = crate::scene::DemoScene::RelativisticBilliards.initial_camera_pos();
    if player == 0 {
        near
    } else {
        Vec4::new(near.t, near.x, near.y, (Z_MIN + Z_MAX) as f32 - near.z)
    }
}

/// The spawn heading (yaw from `+z`): everyone starts facing the table.
pub fn spawn_yaw(player: PlayerId) -> f32 {
    if player == 0 {
        0.0
    } else {
        std::f32::consts::PI
    }
}
