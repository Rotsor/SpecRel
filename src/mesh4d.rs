use crate::minkowski::{Tetrad, Vec3, Vec4};

/// Procedural surface pattern rendered in the object's intrinsic rest-frame UV and proper time `tau`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SurfacePattern {
    /// Rest-frame checkerboard + 3D mesh wireframe
    Checkerboard,
    /// Analog 12-tick clock face (1 tick = 1s of proper time) + digital proper-time readout
    ClockFace,
    /// Einstein bouncing-photon light clock cavity
    LightClock,
}

/// Material / appearance properties of a 4D object
#[derive(Clone, Copy, Debug)]
pub struct Material {
    pub base_color: Vec3,
    pub checker_color: Vec3,
    pub emission: f32,
    pub grid_scale: f32,
    pub wireframe_width: f32,
    pub pattern: SurfacePattern,
}

impl Material {
    pub const fn new(
        base_color: Vec3,
        checker_color: Vec3,
        emission: f32,
        grid_scale: f32,
        wireframe_width: f32,
    ) -> Self {
        Self {
            base_color,
            checker_color,
            emission,
            grid_scale,
            wireframe_width,
            pattern: SurfacePattern::Checkerboard,
        }
    }

    pub const fn with_pattern(mut self, pattern: SurfacePattern) -> Self {
        self.pattern = pattern;
        self
    }
}

/// A 4D Axis-Aligned Bounding Box in Minkowski coordinates (t, x, y, z).
#[derive(Clone, Copy, Debug)]
pub struct Aabb4D {
    pub min: Vec4,
    pub max: Vec4,
}

impl Aabb4D {
    #[inline]
    pub const fn empty() -> Self {
        Self {
            min: Vec4::splat(f32::INFINITY),
            max: Vec4::splat(f32::NEG_INFINITY),
        }
    }

    #[inline]
    pub fn from_points(pts: &[Vec4]) -> Self {
        let mut b = Self::empty();
        for &p in pts {
            b.grow_point(p);
        }
        b
    }

    #[inline]
    pub fn grow_point(&mut self, p: Vec4) {
        self.min = self.min.min(p);
        self.max = self.max.max(p);
    }

    #[inline]
    pub fn union(self, other: Self) -> Self {
        Self {
            min: self.min.min(other.min),
            max: self.max.max(other.max),
        }
    }

    #[inline]
    pub fn center(self) -> Vec4 {
        (self.min + self.max) * 0.5
    }

    #[inline]
    pub fn extent(self) -> Vec4 {
        self.max - self.min
    }

    /// 4D slab test for a ray R(lambda) = origin + lambda * dir, with lambda in [t_min, t_max].
    /// Returns `Some(t_enter)` if the ray intersects the 4D AABB, or `None` if it misses.
    #[inline]
    pub fn intersect_ray(
        &self,
        origin: Vec4,
        inv_dir: Vec4,
        mut t_min: f32,
        mut t_max: f32,
    ) -> Option<f32> {
        // Axis 0: t
        let t0 = (self.min.t - origin.t) * inv_dir.t;
        let t1 = (self.max.t - origin.t) * inv_dir.t;
        t_min = t_min.max(t0.min(t1));
        t_max = t_max.min(t0.max(t1));
        if t_max < t_min {
            return None;
        }

        // Axis 1: x
        let x0 = (self.min.x - origin.x) * inv_dir.x;
        let x1 = (self.max.x - origin.x) * inv_dir.x;
        t_min = t_min.max(x0.min(x1));
        t_max = t_max.min(x0.max(x1));
        if t_max < t_min {
            return None;
        }

        // Axis 2: y
        let y0 = (self.min.y - origin.y) * inv_dir.y;
        let y1 = (self.max.y - origin.y) * inv_dir.y;
        t_min = t_min.max(y0.min(y1));
        t_max = t_max.min(y0.max(y1));
        if t_max < t_min {
            return None;
        }

        // Axis 3: z
        let z0 = (self.min.z - origin.z) * inv_dir.z;
        let z1 = (self.max.z - origin.z) * inv_dir.z;
        t_min = t_min.max(z0.min(z1));
        t_max = t_max.min(z0.max(z1));

        if t_max >= t_min {
            Some(t_min)
        } else {
            None
        }
    }
}

/// A 3-simplex (tetrahedron) embedded in 4D Minkowski spacetime.
/// Represents a volumetric element of a 3D hypersurface (worldtube of a 2D surface).
#[derive(Clone, Debug)]
pub struct Simplex4D {
    /// Base vertex V0 in 4D spacetime
    pub v0: Vec4,
    /// Precomputed Euclidean 4D hyperplane normal N = cross4(E1, E2, E3)
    pub hyperplane_normal: Vec4,
    /// Precomputed dual basis vectors B1, B2, B3 in R^4 such that:
    /// B_i . E_j = delta_ij (for i,j in 1..=3) and B_i . N = 0.
    pub dual_b1: Vec4,
    pub dual_b2: Vec4,
    pub dual_b3: Vec4,
    /// UV coordinates at the 4 vertices
    pub uv: [(f32, f32); 4],
    /// Barycentric coordinates on the parent 3D surface triangle at each of the 4 vertices
    pub tri_bary: [Vec3; 4],
    /// Proper time tau of the object at each of the 4 vertices
    pub proper_time: [f32; 4],
    /// Outward surface normal in 4D spacetime (spacelike, orthogonal to object's 4-velocity)
    pub surface_normal_4d: Vec4,
    /// Instantaneous 4-velocity of the surface element
    pub four_velocity: Vec4,
    /// Material index in the scene
    pub material_id: usize,
    /// 4D bounding box
    pub aabb: Aabb4D,
}

/// Information returned when a 4D ray hits a `Simplex4D`.
#[derive(Clone, Copy, Debug)]
pub struct HitRecord4D {
    /// Affine ray parameter lambda > 0 (since k^0 = -1 in observer rest frame, lambda equals
    /// lookback distance / time in the observer's rest frame!)
    pub lambda: f32,
    /// Spacetime event X^mu where the ray intersected the worldtube
    pub event: Vec4,
    /// 4D barycentric coordinates (w0, u, v, w) on the tetrahedron
    pub tet_bary: [f32; 4],
    /// Interpolated barycentric coordinates on the 3D surface triangle
    pub tri_bary: Vec3,
    /// Interpolated surface UV
    pub uv: (f32, f32),
    /// Interpolated object proper time tau at emission
    pub proper_time: f32,
    /// Spacelike outward surface normal in 4D
    pub surface_normal_4d: Vec4,
    /// 4-velocity of the hit surface element at emission
    pub four_velocity: Vec4,
    /// Material index
    pub material_id: usize,
}

impl Simplex4D {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        verts: [Vec4; 4],
        uv: [(f32, f32); 4],
        tri_bary: [Vec3; 4],
        proper_time: [f32; 4],
        surface_normal_4d: Vec4,
        four_velocity: Vec4,
        material_id: usize,
    ) -> Option<Self> {
        let v0 = verts[0];
        let e1 = verts[1] - v0;
        let e2 = verts[2] - v0;
        let e3 = verts[3] - v0;

        let n = Vec4::cross4(e1, e2, e3);
        let n_len_sq = n.euclidean_len_sq();
        if n_len_sq < 1e-16 {
            return None; // Degenerate 3-simplex
        }
        let inv_n_len = 1.0 / n_len_sq.sqrt();
        let unit_n = n * inv_n_len;

        // Compute dual basis vectors inside span(e1, e2, e3)
        let raw_b1 = Vec4::cross4(e2, e3, unit_n);
        let d1 = raw_b1.euclidean_dot(e1);
        let raw_b2 = Vec4::cross4(e3, e1, unit_n);
        let d2 = raw_b2.euclidean_dot(e2);
        let raw_b3 = Vec4::cross4(e1, e2, unit_n);
        let d3 = raw_b3.euclidean_dot(e3);

        if d1.abs() < 1e-12 || d2.abs() < 1e-12 || d3.abs() < 1e-12 {
            return None;
        }

        // Pad AABB slightly for numerical robustness on axis-aligned hyperplanes
        let mut aabb = Aabb4D::from_points(&verts);
        let eps = Vec4::splat(1e-4);
        aabb.min -= eps;
        aabb.max += eps;

        Some(Self {
            v0,
            hyperplane_normal: unit_n,
            dual_b1: raw_b1 * (1.0 / d1),
            dual_b2: raw_b2 * (1.0 / d2),
            dual_b3: raw_b3 * (1.0 / d3),
            uv,
            tri_bary,
            proper_time,
            surface_normal_4d,
            four_velocity,
            material_id,
            aabb,
        })
    }

    /// Fast 4D ray-tetrahedron intersection.
    /// Ray is `origin + lambda * dir` for `lambda in (t_min, t_max)`.
    #[inline]
    pub fn intersect(&self, origin: Vec4, dir: Vec4, t_min: f32, t_max: f32) -> Option<HitRecord4D> {
        let denom = self.hyperplane_normal.euclidean_dot(dir);
        if denom.abs() < 1e-9 {
            return None;
        }

        let lambda = self.hyperplane_normal.euclidean_dot(self.v0 - origin) / denom;
        if lambda <= t_min || lambda >= t_max {
            return None;
        }

        let event = origin + dir * lambda;
        let q = event - self.v0;

        let eps = -1e-5;
        let u = self.dual_b1.euclidean_dot(q);
        if u < eps || u > 1.0 - eps {
            return None;
        }
        let v = self.dual_b2.euclidean_dot(q);
        if v < eps || u + v > 1.0 - eps {
            return None;
        }
        let w = self.dual_b3.euclidean_dot(q);
        if w < eps || u + v + w > 1.0 - eps {
            return None;
        }

        let w0 = 1.0 - u - v - w;
        let tet_bary = [w0, u, v, w];

        let tri_bary = self.tri_bary[0] * w0
            + self.tri_bary[1] * u
            + self.tri_bary[2] * v
            + self.tri_bary[3] * w;

        let uv_u = self.uv[0].0 * w0 + self.uv[1].0 * u + self.uv[2].0 * v + self.uv[3].0 * w;
        let uv_v = self.uv[0].1 * w0 + self.uv[1].1 * u + self.uv[2].1 * v + self.uv[3].1 * w;

        let proper_time = self.proper_time[0] * w0
            + self.proper_time[1] * u
            + self.proper_time[2] * v
            + self.proper_time[3] * w;

        Some(HitRecord4D {
            lambda,
            event,
            tet_bary,
            tri_bary,
            uv: (uv_u, uv_v),
            proper_time,
            surface_normal_4d: self.surface_normal_4d,
            four_velocity: self.four_velocity,
            material_id: self.material_id,
        })
    }
}

/// A triangle in a 3D rest-frame surface mesh.
#[derive(Clone, Copy, Debug)]
pub struct Triangle3D {
    pub indices: [usize; 3],
    pub uv: [(f32, f32); 3],
    pub normal: Vec3,
}

/// A 3D surface mesh defined in an object's comoving rest frame.
#[derive(Clone, Debug, Default)]
pub struct Mesh3D {
    pub vertices: Vec<Vec3>,
    pub triangles: Vec<Triangle3D>,
}

impl Mesh3D {
    /// Create a box/cube centered at the origin in its rest frame with half-extents `he`.
    pub fn cube(he: Vec3) -> Self {
        let mut vertices = Vec::with_capacity(8);
        for &z in &[-he.z, he.z] {
            for &y in &[-he.y, he.y] {
                for &x in &[-he.x, he.x] {
                    vertices.push(Vec3::new(x, y, z));
                }
            }
        }
        let ux_y = (he.x / he.y.max(1e-4)).round().max(1.0);
        let uz_y = (he.z / he.y.max(1e-4)).round().max(1.0);
        let ux_z = (he.x / he.z.max(1e-4)).round().max(1.0);

        // Faces oriented so +u is right and +v is up when viewed from outside the box:
        let faces: [([usize; 4], Vec3, f32, f32); 6] = [
            // -Z (viewed from z < 0 looking +Z: right is +X, up is +Y)
            ([0, 1, 3, 2], Vec3::new(0.0, 0.0, -1.0), ux_y, 1.0),
            // +Z (viewed from z > 0 looking -Z: right is -X, up is +Y)
            ([5, 4, 6, 7], Vec3::new(0.0, 0.0, 1.0), ux_y, 1.0),
            // -X (viewed from x < 0 looking +X: right is -Z, up is +Y)
            ([4, 0, 2, 6], Vec3::new(-1.0, 0.0, 0.0), uz_y, 1.0),
            // +X (viewed from x > 0 looking -X: right is +Z, up is +Y)
            ([1, 5, 7, 3], Vec3::new(1.0, 0.0, 0.0), uz_y, 1.0),
            // -Y (bottom)
            ([4, 5, 1, 0], Vec3::new(0.0, -1.0, 0.0), ux_z, 1.0),
            // +Y (top)
            ([2, 3, 7, 6], Vec3::new(0.0, 1.0, 0.0), ux_z, 1.0),
        ];

        let mut triangles = Vec::with_capacity(12);
        for (idx, normal, us, vs) in faces {
            let uv0 = (0.0, 0.0);
            let uv1 = (us, 0.0);
            let uv2 = (us, vs);
            let uv3 = (0.0, vs);
            triangles.push(Triangle3D {
                indices: [idx[0], idx[1], idx[2]],
                uv: [uv0, uv1, uv2],
                normal,
            });
            triangles.push(Triangle3D {
                indices: [idx[0], idx[2], idx[3]],
                uv: [uv0, uv2, uv3],
                normal,
            });
        }

        Self {
            vertices,
            triangles,
        }
    }

    /// Create a UV sphere in its rest frame (useful for testing Terrell-Penrose circularity).
    pub fn uv_sphere(radius: f32, lat_steps: usize, lon_steps: usize) -> Self {
        let mut vertices = Vec::new();
        let mut triangles = Vec::new();

        for i in 0..=lat_steps {
            let v = i as f32 / lat_steps as f32;
            let theta = v * std::f32::consts::PI;
            let (sin_t, cos_t) = theta.sin_cos();

            for j in 0..=lon_steps {
                let u = j as f32 / lon_steps as f32;
                let phi = u * std::f32::consts::TAU;
                let (sin_p, cos_p) = phi.sin_cos();

                let p = Vec3::new(radius * sin_t * cos_p, radius * cos_t, radius * sin_t * sin_p);
                vertices.push(p);
            }
        }

        let stride = lon_steps + 1;
        for i in 0..lat_steps {
            for j in 0..lon_steps {
                let i00 = i * stride + j;
                let i10 = i00 + 1;
                let i01 = (i + 1) * stride + j;
                let i11 = i01 + 1;

                let u0 = j as f32 / lon_steps as f32 * 4.0;
                let u1 = (j + 1) as f32 / lon_steps as f32 * 4.0;
                let v0 = i as f32 / lat_steps as f32 * 2.0;
                let v1 = (i + 1) as f32 / lat_steps as f32 * 2.0;

                if i > 0 {
                    let n = (vertices[i00] + vertices[i10] + vertices[i11]).normalized();
                    triangles.push(Triangle3D {
                        indices: [i00, i11, i10],
                        uv: [(u0, v0), (u1, v1), (u1, v0)],
                        normal: n,
                    });
                }
                if i + 1 < lat_steps {
                    let n = (vertices[i00] + vertices[i01] + vertices[i11]).normalized();
                    triangles.push(Triangle3D {
                        indices: [i00, i01, i11],
                        uv: [(u0, v0), (u0, v1), (u1, v1)],
                        normal: n,
                    });
                }
            }
        }

        Self {
            vertices,
            triangles,
        }
    }

    /// Create a subdivided ground plane in the XZ plane at height `y`
    pub fn ground_grid(half_x: f32, half_z: f32, y: f32, divs_x: usize, divs_z: usize) -> Self {
        let mut vertices = Vec::with_capacity((divs_x + 1) * (divs_z + 1));
        for iz in 0..=divs_z {
            let fz = iz as f32 / divs_z as f32;
            let z = -half_z + fz * (2.0 * half_z);
            for ix in 0..=divs_x {
                let fx = ix as f32 / divs_x as f32;
                let x = -half_x + fx * (2.0 * half_x);
                vertices.push(Vec3::new(x, y, z));
            }
        }

        let mut triangles = Vec::with_capacity(divs_x * divs_z * 2);
        let stride = divs_x + 1;
        let normal = Vec3::new(0.0, 1.0, 0.0);
        for iz in 0..divs_z {
            for ix in 0..divs_x {
                let i00 = iz * stride + ix;
                let i10 = i00 + 1;
                let i01 = (iz + 1) * stride + ix;
                let i11 = i01 + 1;

                let p00 = vertices[i00];
                let p10 = vertices[i10];
                let p01 = vertices[i01];
                let p11 = vertices[i11];

                triangles.push(Triangle3D {
                    indices: [i00, i01, i11],
                    uv: [(p00.x, p00.z), (p01.x, p01.z), (p11.x, p11.z)],
                    normal,
                });
                triangles.push(Triangle3D {
                    indices: [i00, i11, i10],
                    uv: [(p00.x, p00.z), (p11.x, p11.z), (p10.x, p10.z)],
                    normal,
                });
            }
        }

        Self {
            vertices,
            triangles,
        }
    }
}

/// A keyframe along an object's 4D worldline, with its comoving rest-frame tetrad.
#[derive(Clone, Copy, Debug)]
pub struct WorldlineSample {
    /// Center of mass spacetime event X_center^mu
    pub center: Vec4,
    /// Object's proper time tau at this sample
    pub proper_time: f32,
    /// Orthonormal comoving tetrad (e[0] = 4-velocity u^mu, e[1..=3] = rest-frame spatial axes)
    pub tetrad: Tetrad,
}

/// Extrude a 3D rest-frame mesh (`Mesh3D`) along a sequence of `WorldlineSample`s to produce
/// a watertight 4D tetrahedral hypersurface (`Vec<Simplex4D>`).
///
/// Because each vertex `p_rest` is embedded via the sample's Lorentz-boosted tetrad:
///   `X^mu(tau, p_rest) = X_center^mu(tau) + p_rest.x * e_1^mu(tau) + p_rest.y * e_2^mu(tau) + p_rest.z * e_3^mu(tau)`
/// the cross-section in the instantaneous rest frame is always rigid (Born rigidity), while in
/// any other inertial frame it automatically exhibits Lorentz contraction, relativity of simultaneity,
/// and Thomas precession when accelerated!
pub fn build_worldtube_simplices(
    mesh: &Mesh3D,
    samples: &[WorldlineSample],
    material_id: usize,
) -> Vec<Simplex4D> {
    if samples.len() < 2 {
        return Vec::new();
    }

    let n_verts = mesh.vertices.len();
    let n_steps = samples.len() - 1;
    let mut simplices = Vec::with_capacity(n_steps * mesh.triangles.len() * 3);

    // Precompute 4D vertex positions for each sample step
    let mut embedded_verts: Vec<Vec<Vec4>> = Vec::with_capacity(samples.len());
    for sample in samples {
        let mut step_verts = Vec::with_capacity(n_verts);
        for &p_rest in &mesh.vertices {
            let event = sample.center + sample.tetrad.local_spatial_to_world(p_rest);
            step_verts.push(event);
        }
        embedded_verts.push(step_verts);
    }

    let std_bary = [
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
        Vec3::new(0.0, 0.0, 1.0),
    ];

    for step in 0..n_steps {
        let v_curr = &embedded_verts[step];
        let v_next = &embedded_verts[step + 1];
        let s_curr = &samples[step];
        let s_next = &samples[step + 1];

        let tau0 = s_curr.proper_time;
        let tau1 = s_next.proper_time;

        // Average 4-velocity across the step
        let u_avg_raw = s_curr.tetrad.e[0] + s_next.tetrad.e[0];
        let u_norm = (-u_avg_raw.minkowski_norm_sq()).max(1e-12).sqrt();
        let u_avg = u_avg_raw * (1.0 / u_norm);

        for tri in &mesh.triangles {
            // Sort triangle vertices by index so shared prism faces between adjacent triangles
            // always split along the exact same diagonal (Dompierre watertight prism split).
            let mut perm = [0usize, 1, 2];
            perm.sort_by_key(|&k| tri.indices[k]);

            let ia = tri.indices[perm[0]];
            let ib = tri.indices[perm[1]];
            let ic = tri.indices[perm[2]];

            let a0 = v_curr[ia];
            let b0 = v_curr[ib];
            let c0 = v_curr[ic];
            let a1 = v_next[ia];
            let b1 = v_next[ib];
            let c1 = v_next[ic];

            let uva = tri.uv[perm[0]];
            let uvb = tri.uv[perm[1]];
            let uvc = tri.uv[perm[2]];

            let ba = std_bary[perm[0]];
            let bb = std_bary[perm[1]];
            let bc = std_bary[perm[2]];

            // Map the rest-frame outward triangle normal into 4D spacetime using the step's tetrad
            let n4_curr = s_curr.tetrad.local_spatial_to_world(tri.normal);
            let n4_next = s_next.tetrad.local_spatial_to_world(tri.normal);
            let n4_raw = n4_curr + n4_next;
            let n4_len = n4_raw.minkowski_norm_sq().max(1e-12).sqrt();
            let surface_normal_4d = n4_raw * (1.0 / n4_len);

            // Split prism (A0, B0, C0) -> (A1, B1, C1) into 3 tetrahedra:
            // 1. (A0, B0, C0, C1)
            // 2. (A0, B0, B1, C1)
            // 3. (A0, A1, B1, C1)
            let tets = [
                (
                    [a0, b0, c0, c1],
                    [uva, uvb, uvc, uvc],
                    [ba, bb, bc, bc],
                    [tau0, tau0, tau0, tau1],
                ),
                (
                    [a0, b0, b1, c1],
                    [uva, uvb, uvb, uvc],
                    [ba, bb, bb, bc],
                    [tau0, tau0, tau1, tau1],
                ),
                (
                    [a0, a1, b1, c1],
                    [uva, uva, uvb, uvc],
                    [ba, ba, bb, bc],
                    [tau0, tau1, tau1, tau1],
                ),
            ];

            for (verts, uv, tb, taus) in tets {
                if let Some(simplex) = Simplex4D::new(
                    verts,
                    uv,
                    tb,
                    taus,
                    surface_normal_4d,
                    u_avg,
                    material_id,
                ) {
                    simplices.push(simplex);
                }
            }
        }
    }

    simplices
}

/// Generate a worldline for an inertial object moving at constant 3-velocity `beta`
/// passing through spatial point `pos_at_t0` at lab time `t = 0`, spanning lab time `[t_min, t_max]`.
pub fn sample_inertial_worldline(
    pos_at_t0: Vec3,
    beta: Vec3,
    t_min: f32,
    t_max: f32,
    steps: usize,
) -> Vec<WorldlineSample> {
    let tetrad = Tetrad::from_beta(beta);
    let inv_gamma = 1.0 / tetrad.gamma();
    let mut samples = Vec::with_capacity(steps + 1);

    for i in 0..=steps {
        let frac = i as f32 / steps as f32;
        let t = t_min + frac * (t_max - t_min);
        let pos = pos_at_t0 + beta * t;
        let tau = t * inv_gamma;
        samples.push(WorldlineSample {
            center: Vec4::from_time_space(t, pos),
            proper_time: tau,
            tetrad,
        });
    }
    samples
}

/// Generate an accelerating worldline from an arbitrary smooth spatial trajectory `pos_fn(t)`
/// over lab time `[t_min, t_max]`.
/// Integrates proper time `d_tau = dt / gamma(t)` and Fermi-Walker transports the comoving
/// rest frame `Tetrad` along the trajectory so Born rigidity and Thomas precession emerge naturally!
pub fn sample_accelerated_worldline<F>(
    pos_fn: F,
    t_min: f32,
    t_max: f32,
    steps: usize,
) -> Vec<WorldlineSample>
where
    F: Fn(f32) -> Vec3,
{
    let mut samples = Vec::with_capacity(steps + 1);
    let dt_deriv = 1e-3;

    let velocity_at = |t: f32| -> Vec3 {
        let p_minus = pos_fn(t - dt_deriv);
        let p_plus = pos_fn(t + dt_deriv);
        let raw_v = (p_plus - p_minus) / (2.0 * dt_deriv);
        // Clamp strictly below speed of light (c = 1)
        let speed = raw_v.length();
        if speed >= 0.98 {
            raw_v * (0.98 / speed)
        } else {
            raw_v
        }
    };

    let beta0 = velocity_at(t_min);
    let mut tetrad = Tetrad::from_beta(beta0);
    let mut tau = 0.0;
    let mut prev_t = t_min;

    for i in 0..=steps {
        let frac = i as f32 / steps as f32;
        let t = t_min + frac * (t_max - t_min);
        let pos = pos_fn(t);
        let beta = velocity_at(t);
        let u = Vec4::four_velocity_from_beta(beta);

        if i > 0 {
            let dt = t - prev_t;
            let avg_inv_gamma = 0.5 * (1.0 / tetrad.gamma() + 1.0 / u.t);
            tau += dt * avg_inv_gamma;
            tetrad.fermi_walker_transport(u);
        }
        prev_t = t;

        samples.push(WorldlineSample {
            center: Vec4::from_time_space(t, pos),
            proper_time: tau,
            tetrad,
        });
    }

    samples
}

/// Shift all `proper_time` values along a sampled worldline by a constant offset so that
/// `proper_time == 0.0` at lab coordinate time `t == t_sync`.
pub fn sync_proper_time_at(samples: &mut [WorldlineSample], t_sync: f32) {
    if samples.is_empty() {
        return;
    }
    let mut tau_at_sync = samples[0].proper_time;
    if t_sync <= samples[0].center.t {
        tau_at_sync = samples[0].proper_time;
    } else if let Some(last) = samples.last() {
        if t_sync >= last.center.t {
            tau_at_sync = last.proper_time;
        } else {
            for w in samples.windows(2) {
                let s0 = &w[0];
                let s1 = &w[1];
                if t_sync >= s0.center.t && t_sync <= s1.center.t {
                    let dt = (s1.center.t - s0.center.t).max(1e-6);
                    let frac = (t_sync - s0.center.t) / dt;
                    tau_at_sync = s0.proper_time + frac * (s1.proper_time - s0.proper_time);
                    break;
                }
            }
        }
    }
    for s in samples.iter_mut() {
        s.proper_time -= tau_at_sync;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ray_simplex4d_intersection() {
        let v0 = Vec4::new(0.0, 0.0, 0.0, 0.0);
        let v1 = Vec4::new(1.0, 0.0, 0.0, 0.0);
        let v2 = Vec4::new(0.0, 1.0, 0.0, 0.0);
        let v3 = Vec4::new(0.0, 0.0, 1.0, 0.0);

        let simplex = Simplex4D::new(
            [v0, v1, v2, v3],
            [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)],
            [
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
                Vec3::new(0.0, 0.0, 1.0),
            ],
            [0.0, 1.0, 0.0, 0.0],
            Vec4::new(0.0, 0.0, 0.0, 1.0),
            Vec4::new(1.0, 0.0, 0.0, 0.0),
            0,
        )
        .unwrap();

        // Cast a ray along -z from z = 2 through (t=0.2, x=0.2, y=0.2, z=0)
        let origin = Vec4::new(0.2, 0.2, 0.2, 2.0);
        let dir = Vec4::new(0.0, 0.0, 0.0, -1.0);
        let hit = simplex.intersect(origin, dir, 0.0, 10.0).expect("should hit");
        assert!((hit.lambda - 2.0).abs() < 1e-5);
        assert!((hit.tet_bary[1] - 0.2).abs() < 1e-5);
        assert!((hit.tet_bary[2] - 0.2).abs() < 1e-5);
        assert!((hit.tet_bary[3] - 0.2).abs() < 1e-5);
    }

    #[test]
    fn test_emergent_length_contraction_in_worldtube() {
        // A cube of rest-frame half-width 1.0 (total rest length L0 = 2.0 along X)
        // moving along +X at beta = 0.8 (gamma = 1 / 0.6 = 5/3, so contracted length = 2.0 * 0.6 = 1.2).
        let cube = Mesh3D::cube(Vec3::new(1.0, 1.0, 1.0));
        let beta = Vec3::new(0.8, 0.0, 0.0);
        let samples = sample_inertial_worldline(Vec3::new(0.0, 0.0, 0.0), beta, -20.0, 20.0, 4);
        let simplices = build_worldtube_simplices(&cube, &samples, 0);

        // Slice the 4D worldtube at constant lab time t = 0 by casting two spatial rays along X at t = 0
        // from x = -5 (along +X) and from x = +5 (along -X) through (t=0, y=0, z=0):
        let mut min_x = f32::INFINITY;
        let mut max_x = f32::NEG_INFINITY;
        for s in &simplices {
            if let Some(hit) = s.intersect(
                Vec4::new(0.0, -5.0, 0.1, 0.1),
                Vec4::new(0.0, 1.0, 0.0, 0.0),
                0.0,
                100.0,
            ) {
                min_x = min_x.min(hit.event.x);
                max_x = max_x.max(hit.event.x);
            }
        }
        let measured_lab_length = max_x - min_x;
        // Expected Lorentz-contracted length at constant lab time t=0: L = L_0 / gamma = 2.0 * sqrt(1 - 0.64) = 1.2
        assert!((measured_lab_length - 1.2).abs() < 1e-4, "measured_lab_length = {}", measured_lab_length);
    }
}
