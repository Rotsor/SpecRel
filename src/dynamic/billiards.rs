//! Scene 6: relativistic billiards. A rack of resting pucks and a few movers bouncing between
//! the rails; the player fires pucks, sticky pucks or spores from the ship (`Enter`, ammo `\`),
//! or rams them with the ship itself.

use super::body::{distance, Body, BodyState, Kind};
use super::collide::V3;
use crate::mesh4d::Mesh3D;
use crate::minkowski::Vec3;

pub const HALF_X: f64 = 9.0;
pub const Z_MIN: f64 = 0.0;
pub const Z_MAX: f64 = 13.0;
pub const TABLE_Y: f32 = -2.2;
pub const PUCK_RADIUS: f64 = 0.35;
pub const PUCK_MASS: f64 = 1.0;
/// The ship is a heavy sphere around the camera (it is not drawn: you are inside it).
pub const SHIP_RADIUS: f64 = 0.8;
pub const SHIP_MASS: f64 = 6.0;
/// Launch speed of fired pucks in the ship's rest frame.
pub const FIRE_SPEED: f64 = 0.6;
/// Bodies this far from the table leave play.
pub const DESPAWN_DISTANCE: f64 = 150.0;
/// Firing is refused beyond this many live bodies.
pub const MAX_BODIES: usize = 64;

const SPORE_HALF_SIZE: f32 = 0.3;
const SPORE_LIFETIMES: [f64; 2] = [2.5, 1.2];
pub const SPORE_SPLIT_SPEED: f64 = 0.5;

/// `[V]` matches the fastest mover's (initial) velocity.
pub const COMOVING_BETA: Vec3 = Vec3::new(0.7, 0.0, 0.0);

const MATERIAL_MOVING: usize = 7;
const MATERIAL_RESTING: usize = 8;
const MATERIAL_MERGED: usize = 9;
const MATERIAL_STICKY: usize = 10;
const MATERIAL_SPORE: usize = 11;

/// What the ship fires.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ammo {
    Puck,
    Sticky,
    Spore,
}

impl Ammo {
    pub fn next(self) -> Self {
        match self {
            Self::Puck => Self::Sticky,
            Self::Sticky => Self::Spore,
            Self::Spore => Self::Puck,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Puck => "ELASTIC PUCK",
            Self::Sticky => "STICKY PUCK",
            Self::Spore => "SPORE",
        }
    }
}

/// SplitMix64-style mixing, for stable body identities.
pub fn mix(a: u64, b: u64) -> u64 {
    let mut z = a ^ b.wrapping_add(0x9E37_79B9_7F4A_7C15).wrapping_add(a << 6);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Contact radius of a puck of rest mass `mass` (constant density).
pub fn puck_radius(mass: f64) -> f64 {
    PUCK_RADIUS * (mass / PUCK_MASS).cbrt()
}

/// A body of the given kind, starting at `state`.
pub fn body(uid: u64, kind: Kind, mass: f64, state: BodyState, material_id: usize) -> Body {
    let (mesh, radius) = match kind {
        Kind::Spore { .. } => (
            Mesh3D::cube(Vec3::splat(SPORE_HALF_SIZE)),
            SPORE_HALF_SIZE as f64 * 3f64.sqrt(),
        ),
        Kind::Puck | Kind::Sticky => {
            let r = puck_radius(mass);
            (Mesh3D::uv_sphere(r as f32, 4, 6), r)
        }
    };
    Body {
        uid,
        kind,
        mass,
        mesh,
        material_id,
        radius,
        history: vec![state],
        decided_at: Vec::new(),
        born: false,
        ended: false,
        ignore_ship: false,
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
        bodies.push(body(uid, Kind::Puck, PUCK_MASS, state, material));
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

/// A body fired from the ship.
pub fn projectile(seq: u64, ammo: Ammo, state: BodyState) -> Body {
    let uid = mix(0xF1BE, seq);
    let mut b = match ammo {
        Ammo::Puck => body(uid, Kind::Puck, PUCK_MASS, state, MATERIAL_MOVING),
        Ammo::Sticky => body(uid, Kind::Sticky, PUCK_MASS, state, MATERIAL_STICKY),
        Ammo::Spore => {
            let kind = Kind::Spore { lifetime: SPORE_LIFETIMES[0], generation: 0 };
            body(uid, kind, PUCK_MASS, state, MATERIAL_SPORE)
        }
    };
    b.born = true;
    b.ignore_ship = true;
    b
}

/// The body formed when the sticky cluster `uids` merges.
pub fn merged(uids: &[u64], mass: f64, state: BodyState) -> Body {
    let uid = uids.iter().fold(0x3E26E, |acc, &u| mix(acc, u));
    let mut b = body(uid, Kind::Sticky, mass, state, MATERIAL_MERGED);
    b.born = true;
    b
}

/// The two fragments of a spore of `generation` that popped at `state`, with the given masses
/// and velocities. The next generation are spores again until the fuses run out.
pub fn fragments(parent: u64, generation: u32, parts: [(f64, V3); 2], state: &BodyState) -> [Body; 2] {
    let next = generation as usize + 1;
    let mut i = 0;
    parts.map(|(mass, vel)| {
        let s = BodyState { vel, tau: 0.0, ..*state };
        let uid = mix(mix(0xDECA1, parent), i);
        i += 1;
        let mut b = match SPORE_LIFETIMES.get(next) {
            Some(&lifetime) => body(uid, Kind::Spore { lifetime, generation: next as u32 }, mass, s, MATERIAL_SPORE),
            None => body(uid, Kind::Puck, mass, s, MATERIAL_MOVING),
        };
        b.born = true;
        b
    })
}

/// Fragments fly apart perpendicular to the parent's motion (in the table plane).
pub fn split_direction(vel: V3) -> V3 {
    let l = (vel[0] * vel[0] + vel[2] * vel[2]).sqrt();
    if l < 1e-9 {
        [1.0, 0.0, 0.0]
    } else {
        [-vel[2] / l, 0.0, vel[0] / l]
    }
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
