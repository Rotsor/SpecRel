use std::ops::{Add, AddAssign, Div, Mul, Neg, Sub, SubAssign};

/// 3D spatial vector (used for rest-frame coordinates, 3-velocities, colors, etc.)
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Vec3 {
    #[inline]
    pub const fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    #[inline]
    pub const fn splat(v: f32) -> Self {
        Self { x: v, y: v, z: v }
    }

    #[inline]
    pub fn dot(self, other: Self) -> f32 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    #[inline]
    pub fn cross(self, other: Self) -> Self {
        Self {
            x: self.y * other.z - self.z * other.y,
            y: self.z * other.x - self.x * other.z,
            z: self.x * other.y - self.y * other.x,
        }
    }

    #[inline]
    pub fn length_sq(self) -> f32 {
        self.dot(self)
    }

    #[inline]
    pub fn length(self) -> f32 {
        self.length_sq().sqrt()
    }

    #[inline]
    pub fn normalized(self) -> Self {
        let len = self.length();
        if len > 1e-12 {
            self * (1.0 / len)
        } else {
            Self::new(0.0, 0.0, 0.0)
        }
    }

    #[inline]
    pub fn clamp(self, min_val: f32, max_val: f32) -> Self {
        Self {
            x: self.x.clamp(min_val, max_val),
            y: self.y.clamp(min_val, max_val),
            z: self.z.clamp(min_val, max_val),
        }
    }
}

impl Add for Vec3 {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self::new(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z)
    }
}

impl AddAssign for Vec3 {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.x += rhs.x;
        self.y += rhs.y;
        self.z += rhs.z;
    }
}

impl Sub for Vec3 {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self::new(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z)
    }
}

impl SubAssign for Vec3 {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        self.x -= rhs.x;
        self.y -= rhs.y;
        self.z -= rhs.z;
    }
}

impl Mul<f32> for Vec3 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: f32) -> Self {
        Self::new(self.x * rhs, self.y * rhs, self.z * rhs)
    }
}

impl Mul<Vec3> for f32 {
    type Output = Vec3;
    #[inline]
    fn mul(self, rhs: Vec3) -> Vec3 {
        rhs * self
    }
}

impl Mul<Vec3> for Vec3 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        Self::new(self.x * rhs.x, self.y * rhs.y, self.z * rhs.z)
    }
}

impl Div<f32> for Vec3 {
    type Output = Self;
    #[inline]
    fn div(self, rhs: f32) -> Self {
        self * (1.0 / rhs)
    }
}

impl Neg for Vec3 {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self::new(-self.x, -self.y, -self.z)
    }
}

/// 4D Minkowski vector / spacetime event (t, x, y, z) with c = 1 and signature (-, +, +, +).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Vec4 {
    pub t: f32,
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Vec4 {
    #[inline]
    pub const fn new(t: f32, x: f32, y: f32, z: f32) -> Self {
        Self { t, x, y, z }
    }

    #[inline]
    pub const fn from_time_space(t: f32, s: Vec3) -> Self {
        Self {
            t,
            x: s.x,
            y: s.y,
            z: s.z,
        }
    }

    #[inline]
    pub const fn spatial(self) -> Vec3 {
        Vec3::new(self.x, self.y, self.z)
    }

    #[inline]
    pub const fn splat(v: f32) -> Self {
        Self {
            t: v,
            x: v,
            y: v,
            z: v,
        }
    }

    /// Minkowski inner product <u, v>_eta = -u.t * v.t + u.x * v.x + u.y * v.y + u.z * v.z
    #[inline]
    pub fn minkowski_dot(self, other: Self) -> f32 {
        -self.t * other.t + self.x * other.x + self.y * other.y + self.z * other.z
    }

    /// Minkowski squared interval <u, u>_eta (< 0 timelike, == 0 lightlike/null, > 0 spacelike)
    #[inline]
    pub fn minkowski_norm_sq(self) -> f32 {
        self.minkowski_dot(self)
    }

    /// Euclidean 4D inner product (used for affine 4D hyperplane equations and dual basis)
    #[inline]
    pub fn euclidean_dot(self, other: Self) -> f32 {
        self.t * other.t + self.x * other.x + self.y * other.y + self.z * other.z
    }

    #[inline]
    pub fn euclidean_len_sq(self) -> f32 {
        self.euclidean_dot(self)
    }

    #[inline]
    pub fn euclidean_len(self) -> f32 {
        self.euclidean_len_sq().sqrt()
    }

    #[inline]
    pub fn min(self, other: Self) -> Self {
        Self {
            t: self.t.min(other.t),
            x: self.x.min(other.x),
            y: self.y.min(other.y),
            z: self.z.min(other.z),
        }
    }

    #[inline]
    pub fn max(self, other: Self) -> Self {
        Self {
            t: self.t.max(other.t),
            x: self.x.max(other.x),
            y: self.y.max(other.y),
            z: self.z.max(other.z),
        }
    }

    #[inline]
    pub fn axis(self, idx: usize) -> f32 {
        match idx {
            0 => self.t,
            1 => self.x,
            2 => self.y,
            _ => self.z,
        }
    }

    /// 4D triple cross product (Hodge dual *(u ^ v ^ w) in R^4).
    /// Returns a 4-vector N such that for any w' in R^4, euclidean_dot(w', N) = det([w', u, v, w]).
    /// In particular, N is strictly orthogonal (in Euclidean 4D dot product) to u, v, and w.
    #[inline]
    pub fn cross4(u: Self, v: Self, w: Self) -> Self {
        // Precompute 2x2 minors of (v, w)
        let m_xy = v.x * w.y - v.y * w.x;
        let m_xz = v.x * w.z - v.z * w.x;
        let m_yz = v.y * w.z - v.z * w.y;
        let m_tx = v.t * w.x - v.x * w.t;
        let m_ty = v.t * w.y - v.y * w.t;
        let m_tz = v.t * w.z - v.z * w.t;

        let nt = u.x * m_yz - u.y * m_xz + u.z * m_xy;
        let nx = -(u.t * m_yz - u.y * m_tz + u.z * m_ty);
        let ny = u.t * m_xz - u.x * m_tz + u.z * m_tx;
        let nz = -(u.t * m_xy - u.x * m_ty + u.y * m_tx);

        Self::new(nt, nx, ny, nz)
    }

    /// Convert a future-directed 3-velocity beta (|beta| < 1) into a 4-velocity u^mu = (gamma, gamma * beta)
    #[inline]
    pub fn four_velocity_from_beta(beta: Vec3) -> Self {
        let b2 = beta.length_sq().min(0.999_999);
        let gamma = 1.0 / (1.0 - b2).sqrt();
        Self::from_time_space(gamma, beta * gamma)
    }

    /// Extract the 3-velocity beta = (x/t, y/t, z/t) from a timelike 4-velocity
    #[inline]
    pub fn beta_from_four_velocity(self) -> Vec3 {
        if self.t.abs() > 1e-12 {
            self.spatial() / self.t
        } else {
            Vec3::default()
        }
    }
}

impl Add for Vec4 {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self::new(
            self.t + rhs.t,
            self.x + rhs.x,
            self.y + rhs.y,
            self.z + rhs.z,
        )
    }
}

impl AddAssign for Vec4 {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.t += rhs.t;
        self.x += rhs.x;
        self.y += rhs.y;
        self.z += rhs.z;
    }
}

impl Sub for Vec4 {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self::new(
            self.t - rhs.t,
            self.x - rhs.x,
            self.y - rhs.y,
            self.z - rhs.z,
        )
    }
}

impl SubAssign for Vec4 {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        self.t -= rhs.t;
        self.x -= rhs.x;
        self.y -= rhs.y;
        self.z -= rhs.z;
    }
}

impl Mul<f32> for Vec4 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: f32) -> Self {
        Self::new(self.t * rhs, self.x * rhs, self.y * rhs, self.z * rhs)
    }
}

impl Mul<Vec4> for f32 {
    type Output = Vec4;
    #[inline]
    fn mul(self, rhs: Vec4) -> Vec4 {
        rhs * self
    }
}

impl Div<f32> for Vec4 {
    type Output = Self;
    #[inline]
    fn div(self, rhs: f32) -> Self {
        self * (1.0 / rhs)
    }
}

impl Neg for Vec4 {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self::new(-self.t, -self.x, -self.y, -self.z)
    }
}

/// Exact planar Lorentz boost mapping unit future-directed timelike 4-velocity `u1` to `u2`,
/// while leaving the Minkowski-orthogonal complement of span(u1, u2) invariant.
#[inline]
pub fn lorentz_boost_u1_to_u2(v: Vec4, u1: Vec4, u2: Vec4) -> Vec4 {
    let u1_dot_u2 = u1.minkowski_dot(u2); // -gamma_12 <= -1
    let denom = 1.0 - u1_dot_u2; // 1 + gamma_12 >= 2
    let sum = u1 + u2;
    let sum_dot_v = sum.minkowski_dot(v);
    let u1_dot_v = u1.minkowski_dot(v);
    v + sum * (sum_dot_v / denom) - u2 * (2.0 * u1_dot_v)
}

/// An orthonormal Lorentz frame (vierbein / tetrad) in 4D Minkowski spacetime:
/// - `e[0]` is the future-directed unit timelike 4-velocity: <e[0], e[0]>_eta = -1
/// - `e[1]` (right), `e[2]` (up), `e[3]` (forward) are unit spacelike axes in the rest frame:
///   <e[a], e[b]>_eta = delta_ab, and <e[0], e[a]>_eta = 0.
#[derive(Clone, Copy, Debug)]
pub struct Tetrad {
    pub e: [Vec4; 4],
}

impl Default for Tetrad {
    fn default() -> Self {
        Self::rest()
    }
}

impl Tetrad {
    /// Standard identity rest-frame tetrad in lab coordinates.
    #[inline]
    pub const fn rest() -> Self {
        Self {
            e: [
                Vec4::new(1.0, 0.0, 0.0, 0.0), // e_(0): time / 4-velocity
                Vec4::new(0.0, 1.0, 0.0, 0.0), // e_(1): right (+X)
                Vec4::new(0.0, 0.0, 1.0, 0.0), // e_(2): up (+Y)
                Vec4::new(0.0, 0.0, 0.0, 1.0), // e_(3): forward (+Z)
            ],
        }
    }

    /// Construct a tetrad boosted from the lab rest frame to 3-velocity `beta` (|beta| < 1).
    pub fn from_beta(beta: Vec3) -> Self {
        let mut tetrad = Self::rest();
        let u = Vec4::four_velocity_from_beta(beta);
        tetrad.fermi_walker_transport(u);
        tetrad
    }

    /// Transport this tetrad without local rotation (Fermi-Walker step) to a new 4-velocity `new_u0`.
    pub fn fermi_walker_transport(&mut self, new_u0: Vec4) {
        // Normalize new_u0 to unit timelike (-1)
        let norm_sq = new_u0.minkowski_norm_sq();
        let u2 = if norm_sq < -1e-12 {
            let inv = 1.0 / (-norm_sq).sqrt();
            let sign = if new_u0.t >= 0.0 { 1.0 } else { -1.0 };
            new_u0 * (sign * inv)
        } else {
            Vec4::new(1.0, 0.0, 0.0, 0.0)
        };

        let u1 = self.e[0];
        self.e[0] = u2;
        for a in 1..=3 {
            self.e[a] = lorentz_boost_u1_to_u2(self.e[a], u1, u2);
        }
        self.reorthonormalize();
    }

    /// Apply a pure Lorentz boost specified as a rapidity vector `d_rapidity` in the observer's
    /// instantaneous local spatial frame (e[1], e[2], e[3]).
    /// Because rapidity adds hyperbolically, this never exceeds the speed of light.
    pub fn boost_local(&mut self, d_rapidity: Vec3) {
        let eta = d_rapidity.length();
        if eta < 1e-8 {
            return;
        }
        let dir = d_rapidity / eta;
        // Unit spacelike vector in 4D along the boost direction
        let s = self.local_spatial_to_world(dir);
        let ch = eta.cosh();
        let sh = eta.sinh();
        let new_u0 = self.e[0] * ch + s * sh;
        self.fermi_walker_transport(new_u0);
    }

    /// Rotate the spatial triad (e[1], e[2], e[3]) inside the observer's instantaneous rest frame.
    /// `yaw` rotates in the (e[3], e[1]) plane; `pitch` in the (e[3], e[2]) plane; `roll` in (e[1], e[2]).
    pub fn rotate_local(&mut self, yaw: f32, pitch: f32, roll: f32) {
        if yaw.abs() > 1e-8 {
            let (s, c) = yaw.sin_cos();
            let e1 = self.e[1];
            let e3 = self.e[3];
            self.e[1] = e1 * c - e3 * s;
            self.e[3] = e1 * s + e3 * c;
        }
        if pitch.abs() > 1e-8 {
            let (s, c) = pitch.sin_cos();
            let e2 = self.e[2];
            let e3 = self.e[3];
            self.e[2] = e2 * c - e3 * s;
            self.e[3] = e2 * s + e3 * c;
        }
        if roll.abs() > 1e-8 {
            let (s, c) = roll.sin_cos();
            let e1 = self.e[1];
            let e2 = self.e[2];
            self.e[1] = e1 * c - e2 * s;
            self.e[2] = e1 * s + e2 * c;
        }
        self.reorthonormalize();
    }

    /// Map a 3D spatial point/vector in the instantaneous rest frame into 4D spacetime displacement
    /// orthogonal to `e[0]`: r^1 e_(1) + r^2 e_(2) + r^3 e_(3).
    #[inline]
    pub fn local_spatial_to_world(&self, local: Vec3) -> Vec4 {
        self.e[1] * local.x + self.e[2] * local.y + self.e[3] * local.z
    }

    /// Map a 4D vector in local tetrad coordinates (v^0, v^1, v^2, v^3) into lab 4D coordinates.
    #[inline]
    pub fn local_to_world(&self, local: Vec4) -> Vec4 {
        self.e[0] * local.t + self.e[1] * local.x + self.e[2] * local.y + self.e[3] * local.z
    }

    /// Construct a past-directed null ray 4-direction in lab coordinates from a unit spatial
    /// viewing direction `dir_local` in the observer's rest frame.
    /// In local coordinates, k^(A) = (-1, dir_local.x, dir_local.y, dir_local.z),
    /// which satisfies <k, k>_eta = -(-1)^2 + |dir_local|^2 = 0.
    #[inline]
    pub fn past_light_ray_direction(&self, dir_local: Vec3) -> Vec4 {
        let n = dir_local.normalized();
        -self.e[0] + self.e[1] * n.x + self.e[2] * n.y + self.e[3] * n.z
    }

    /// Gram-Schmidt re-orthonormalization with respect to the Minkowski metric (-, +, +, +).
    pub fn reorthonormalize(&mut self) {
        // Normalize e[0] with <e[0], e[0]>_eta = -1
        let n0 = (-self.e[0].minkowski_norm_sq()).max(1e-12).sqrt();
        let sign = if self.e[0].t >= 0.0 { 1.0 } else { -1.0 };
        self.e[0] = self.e[0] * (sign / n0);

        for a in 1..=3 {
            let mut v = self.e[a];
            // Subtract timelike projection: since <e[0], e[0]>_eta = -1,
            // v_orth = v + <v, e[0]>_eta * e[0]
            v += self.e[0] * v.minkowski_dot(self.e[0]);
            // Subtract projections onto previous spacelike vectors e[1..a]
            for b in 1..a {
                v -= self.e[b] * v.minkowski_dot(self.e[b]);
            }
            let na = v.minkowski_norm_sq().max(1e-12).sqrt();
            self.e[a] = v * (1.0 / na);
        }
    }

    /// Current Lorentz factor gamma = e[0].t in the lab frame.
    #[inline]
    pub fn gamma(&self) -> f32 {
        self.e[0].t
    }

    /// Current 3-velocity beta in the lab frame.
    #[inline]
    pub fn beta(&self) -> Vec3 {
        self.e[0].beta_from_four_velocity()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cross4_orthogonality() {
        let u = Vec4::new(1.2, -0.4, 0.7, 2.1);
        let v = Vec4::new(-0.3, 1.5, 0.2, -1.1);
        let w = Vec4::new(0.8, 0.1, -1.9, 0.5);
        let n = Vec4::cross4(u, v, w);

        assert!(n.euclidean_len() > 0.1);
        assert!(u.euclidean_dot(n).abs() < 1e-5);
        assert!(v.euclidean_dot(n).abs() < 1e-5);
        assert!(w.euclidean_dot(n).abs() < 1e-5);
    }

    #[test]
    fn test_tetrad_orthonormality_and_null_rays() {
        let mut tetrad = Tetrad::from_beta(Vec3::new(0.6, -0.3, 0.5));
        tetrad.rotate_local(0.4, -0.25, 0.1);
        tetrad.boost_local(Vec3::new(0.2, 0.1, 0.8));

        // Verify <e_A, e_B>_eta = eta_AB
        assert!((tetrad.e[0].minkowski_norm_sq() + 1.0).abs() < 1e-5);
        for a in 1..=3 {
            assert!((tetrad.e[a].minkowski_norm_sq() - 1.0).abs() < 1e-5);
            assert!(tetrad.e[0].minkowski_dot(tetrad.e[a]).abs() < 1e-5);
            for b in (a + 1)..=3 {
                assert!(tetrad.e[a].minkowski_dot(tetrad.e[b]).abs() < 1e-5);
            }
        }

        // Verify any camera ray is strictly null (<k, k>_eta == 0) and past-directed (k.t < 0)
        let k = tetrad.past_light_ray_direction(Vec3::new(0.3, -0.4, 1.0));
        assert!(k.minkowski_norm_sq().abs() < 1e-5);
        assert!(k.t < 0.0);
    }
}
