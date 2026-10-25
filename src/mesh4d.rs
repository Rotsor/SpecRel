use crate::minkowski::{Tetrad, Vec3, Vec4};

/// `decided_at` sentinel for precomputed static geometry, which was never inside the ship's
/// future light cone (Decision-Age shading then falls back to an analytic decision time).
pub const DECIDED_AT_STATIC: f32 = -1.0e9;

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
    /// Rest-frame mirror separation `L` (in light-seconds) for `SurfacePattern::LightClock`,
    /// so the bouncing photon obeys `T = 2L/c` and moves at exact speed `c = 1` in the rest frame.
    pub cavity_length: f32,
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
            cavity_length: 1.0,
        }
    }

    pub const fn with_pattern(mut self, pattern: SurfacePattern) -> Self {
        self.pattern = pattern;
        self
    }

    pub const fn with_cavity_length(mut self, cavity_length: f32) -> Self {
        self.cavity_length = cavity_length;
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

    /// Branchless 4D slab test for a ray R(lambda) = origin + lambda * dir, with lambda in [t_min, t_max].
    /// Returns `Some(t_enter)` if the ray intersects the 4D AABB, or `None` if it misses.
    #[inline]
    pub fn intersect_ray(
        &self,
        origin: Vec4,
        inv_dir: Vec4,
        t_min: f32,
        t_max: f32,
    ) -> Option<f32> {
        let t0 = (self.min.t - origin.t) * inv_dir.t;
        let t1 = (self.max.t - origin.t) * inv_dir.t;
        let x0 = (self.min.x - origin.x) * inv_dir.x;
        let x1 = (self.max.x - origin.x) * inv_dir.x;
        let y0 = (self.min.y - origin.y) * inv_dir.y;
        let y1 = (self.max.y - origin.y) * inv_dir.y;
        let z0 = (self.min.z - origin.z) * inv_dir.z;
        let z1 = (self.max.z - origin.z) * inv_dir.z;

        let enter = t_min
            .max(t0.min(t1))
            .max(x0.min(x1))
            .max(y0.min(y1))
            .max(z0.min(z1));
        let exit = t_max
            .min(t0.max(t1))
            .min(x0.max(x1))
            .min(y0.max(y1))
            .min(z0.max(z1));

        if exit >= enter {
            Some(enter)
        } else {
            None
        }
    }
}

/// Compact 80-byte hot affine intersection data for a 4D tetrahedron (`Simplex4D`).
/// Ray-tetrahedron intersection is purely affine (metric-independent): `hyperplane_normal` and
/// `dual_b1..3` are linear functionals (covectors) evaluated via `euclidean_dot`.
#[derive(Clone, Copy, Debug)]
pub struct SimplexGeom4D {
    pub v0: Vec4,
    pub hyperplane_normal: Vec4,
    pub dual_b1: Vec4,
    pub dual_b2: Vec4,
    pub dual_b3: Vec4,
}

impl SimplexGeom4D {
    #[inline]
    pub fn from_simplex(s: &Simplex4D) -> Self {
        Self {
            v0: s.v0,
            hyperplane_normal: s.hyperplane_normal,
            dual_b1: s.dual_b1,
            dual_b2: s.dual_b2,
            dual_b3: s.dual_b3,
        }
    }

    /// Fast affine 4D ray-tetrahedron intersection returning `(lambda, event, tet_bary)`.
    #[inline]
    pub fn intersect_fast(
        &self,
        origin: Vec4,
        dir: Vec4,
        t_min: f32,
        t_max: f32,
    ) -> Option<(f32, Vec4, [f32; 4])> {
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
        Some((lambda, event, [w0, u, v, w]))
    }
}

/// A 3-simplex (tetrahedron) embedded in 4D Minkowski spacetime.
/// Represents a volumetric element of a 3D hypersurface (worldtube of a 2D surface).
#[derive(Clone, Debug)]
pub struct Simplex4D {
    /// Base vertex V0 in 4D spacetime
    pub v0: Vec4,
    /// Precomputed 4D hyperplane 1-form N = cross4(E1, E2, E3)
    pub hyperplane_normal: Vec4,
    /// Precomputed dual basis covectors B1, B2, B3 in R^4 such that:
    /// B_i(E_j) = delta_ij (for i,j in 1..=3) and B_i(N) = 0.
    pub dual_b1: Vec4,
    pub dual_b2: Vec4,
    pub dual_b3: Vec4,
    /// UV coordinates at the 4 vertices
    pub uv: [(f32, f32); 4],
    /// Barycentric coordinates on the parent 3D surface triangle at each of the 4 vertices
    pub tri_bary: [Vec3; 4],
    /// Local proper time `tau_local = tau_c + p . Phi(tau_c)` of an idealized uncoupled point clock
    /// fixed at the material point `p` of each of the 4 vertices.
    pub proper_time: [f32; 4],
    /// Rest-frame spatial proper-time gradient `(d_tau / du, d_tau / dv) = (t_u . Phi, t_v . Phi)`
    /// at the 4 vertices, where `t_u = dp / du` and `t_v = dp / dv` are the 3D rest-frame tangent
    /// vectors of the surface triangle.
    ///
    /// ### Physical Justification: Central Proper Time vs. Local Material-Point Proper Time
    /// On a Born-rigid body whose center of mass follows `X_c(tau_c)` with Fermi-Walker transported
    /// spatial triad `e_i(tau_c)`, the hyperplanes `Sigma(tau_c)` of constant `tau_c` are the body's
    /// instantaneous rest-frame surfaces of simultaneity. By the equivalence principle (Rindler metric
    /// `ds^2 = -(1 + a_local . p)^2 d_tau_c^2 + dp^2`), an **uncoupled point oscillator** fixed at rest-frame
    /// offset `p` ticks at rate `d_tau_local = (1 + a_local . p) d_tau_c`, accumulating a position-dependent
    /// proper time:
    ///   `tau_local(p, tau_c) = tau_c + p . Phi(tau_c)`   where `Phi(tau_c) = \int a_local d_tau_c`.
    ///
    /// However, a macroscopic mechanical or electronic clock face is **not** an uncoupled dust of
    /// independent point oscillators at every `(u, v)` pixel (which would tear a fast-moving hand
    /// into disconnected pieces even in the clock's own rest frame!). Instead, a real clock face is
    /// driven by a **single localized oscillator** at the center of that clock tile `p_tile_center`.
    /// At any rest-frame simultaneity slice `Sigma(tau_c)`, that central oscillator has a single,
    /// well-defined proper time—its **central proper time**:
    ///   `tau_tile_center(tau_c) = tau_c + p_tile_center . Phi(tau_c)`
    ///                           `= tau_local(u, v) - Delta_u * (t_u . Phi) - Delta_v * (t_v . Phi)`.
    /// As that central oscillator broadcasts its phase/time signal radially outward across the dial
    /// at speed `c = 1` in the rest frame, a pixel at physical rest-frame distance `r_phys` from the
    /// dial pivot displays the causally retarded phase `tau_tile_center - r_phys / c`.
    pub tau_uv_grad: [(f32, f32); 4],
    /// Physical rest-frame distances `(|dp / du|, |dp / dv|)` in light-seconds per unit `(u, v)`
    /// on this triangle, used to compute the physical radial signal propagation delay `r_phys / c`
    /// from a clock dial's pivot to each pixel in the clock's comoving rest frame.
    pub uv_world_scale: (f32, f32),
    /// Outward surface normal in 4D spacetime (spacelike, orthogonal to object's 4-velocity)
    pub surface_normal_4d: Vec4,
    /// Instantaneous 4-velocity of the surface element
    pub four_velocity: Vec4,
    /// Camera proper time at which each vertex event left the ship's open future light cone
    /// `I+(O)` in Dynamic Mode ([`DECIDED_AT_STATIC`] for precomputed static geometry)
    pub decided_at: [f32; 4],
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
    /// Interpolated local material-point proper time `tau_local = tau_c + p . Phi` at emission
    pub proper_time: f32,
    /// Interpolated rest-frame UV proper-time gradient `(d_tau / du, d_tau / dv) = (t_u . Phi, t_v . Phi)`
    /// at emission, allowing textures to reconstruct the central proper time of a clock tile's
    /// oscillator (`tau_tile_center = tau_local - Delta_u * d_tau/du - Delta_v * d_tau/dv`)
    /// or the proper time of corner reference oscillators.
    pub tau_uv_grad: (f32, f32),
    /// Physical rest-frame lengths `(|dp / du|, |dp / dv|)` in light-seconds per unit UV, used to
    /// model causal signal propagation (`-r_phys / c`) outward from a clock's central oscillator.
    pub uv_world_scale: (f32, f32),
    /// Spacelike outward surface normal in 4D
    pub surface_normal_4d: Vec4,
    /// 4-velocity of the hit surface element at emission
    pub four_velocity: Vec4,
    /// Interpolated camera proper time at which this event became causally decided
    pub decided_at: f32,
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
        tau_uv_grad: [(f32, f32); 4],
        uv_world_scale: (f32, f32),
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

        // Compute dual basis covectors inside span(e1, e2, e3)
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
            tau_uv_grad,
            uv_world_scale,
            surface_normal_4d,
            four_velocity,
            decided_at: [DECIDED_AT_STATIC; 4],
            material_id,
            aabb,
        })
    }

    /// Stamp per-vertex causal decision times (see [`Simplex4D::decided_at`]).
    #[inline]
    pub fn with_decided_at(mut self, decided_at: [f32; 4]) -> Self {
        self.decided_at = decided_at;
        self
    }

    /// Interpolate vertex attributes at a hit location with 4D barycentric coordinates `tet_bary`.
    #[inline]
    pub fn interpolate_hit(&self, lambda: f32, event: Vec4, tet_bary: [f32; 4]) -> HitRecord4D {
        let [w0, u, v, w] = tet_bary;
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

        let grad_u = self.tau_uv_grad[0].0 * w0
            + self.tau_uv_grad[1].0 * u
            + self.tau_uv_grad[2].0 * v
            + self.tau_uv_grad[3].0 * w;
        let grad_v = self.tau_uv_grad[0].1 * w0
            + self.tau_uv_grad[1].1 * u
            + self.tau_uv_grad[2].1 * v
            + self.tau_uv_grad[3].1 * w;

        let decided_at = self.decided_at[0] * w0
            + self.decided_at[1] * u
            + self.decided_at[2] * v
            + self.decided_at[3] * w;

        HitRecord4D {
            lambda,
            event,
            tet_bary,
            tri_bary,
            uv: (uv_u, uv_v),
            proper_time,
            tau_uv_grad: (grad_u, grad_v),
            uv_world_scale: self.uv_world_scale,
            surface_normal_4d: self.surface_normal_4d,
            four_velocity: self.four_velocity,
            decided_at,
            material_id: self.material_id,
        }
    }

    /// Fast 4D ray-tetrahedron intersection.
    /// Ray is `origin + lambda * dir` for `lambda in (t_min, t_max)`.
    #[inline]
    pub fn intersect(&self, origin: Vec4, dir: Vec4, t_min: f32, t_max: f32) -> Option<HitRecord4D> {
        let geom = SimplexGeom4D::from_simplex(self);
        let (lambda, event, tet_bary) = geom.intersect_fast(origin, dir, t_min, t_max)?;
        Some(self.interpolate_hit(lambda, event, tet_bary))
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
    /// Center's proper time tau_c at this sample
    pub proper_time: f32,
    /// Integrated local proper acceleration (rapidity vector) `Phi(tau_c) = \int a_local d_tau_c`
    /// in the comoving Fermi-Walker frame. For Born-rigid motion `X_p = X_c + p^i e_i`,
    /// `dX_p / d_tau_c = (1 + a_local . p) u`, so a clock fixed at rest-frame offset `p`
    /// ticks at `d_tau_p = (1 + a_local . p) d_tau_c` and reads `tau_p = tau_c + p . phi`.
    pub phi: Vec3,
    /// Instantaneous local proper acceleration `a_local` in the comoving frame (used to check
    /// the Rindler horizon condition `1 + a_local . p > 0`).
    pub proper_accel_local: Vec3,
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
/// gravitational/Rindler proper-time gradients (`tau_p = tau_c + p . phi`), and Thomas precession!
pub fn build_worldtube_simplices(
    mesh: &Mesh3D,
    samples: &[WorldlineSample],
    material_id: usize,
) -> Vec<Simplex4D> {
    build_worldtube_simplices_with_decided(mesh, samples, None, material_id)
}

/// [`build_worldtube_simplices`], additionally stamping every vertex of step `k` with the
/// causal decision time `decided_at[k]` (one entry per sample). `None` marks static geometry
/// ([`DECIDED_AT_STATIC`]).
pub fn build_worldtube_simplices_with_decided(
    mesh: &Mesh3D,
    samples: &[WorldlineSample],
    decided_at: Option<&[f32]>,
    material_id: usize,
) -> Vec<Simplex4D> {
    if samples.len() < 2 {
        return Vec::new();
    }
    if let Some(d) = decided_at {
        assert_eq!(d.len(), samples.len(), "one decided_at entry per worldline sample");
    }

    let n_verts = mesh.vertices.len();
    let n_steps = samples.len() - 1;
    let mut simplices = Vec::with_capacity(n_steps * mesh.triangles.len() * 3);

    // Precompute 4D vertex positions and verify the Rindler horizon bound (1 + a_local . p > 0)
    let mut embedded_verts: Vec<Vec<Vec4>> = Vec::with_capacity(samples.len());
    for sample in samples {
        let mut step_verts = Vec::with_capacity(n_verts);
        for &p_rest in &mesh.vertices {
            let rindler_factor = 1.0 + sample.proper_accel_local.dot(p_rest);
            assert!(
                rindler_factor > 0.0,
                "Born-rigid worldtube crosses its Rindler horizon: 1 + a_local . p = {} <= 0 (p = {:?}, a_local = {:?})",
                rindler_factor,
                p_rest,
                sample.proper_accel_local
            );
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
        let (dec0, dec1) = match decided_at {
            Some(d) => (d[step], d[step + 1]),
            None => (DECIDED_AT_STATIC, DECIDED_AT_STATIC),
        };

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

            // Born-rigid local proper time tau_p = tau_c + p_rest . phi at each vertex
            let pa = mesh.vertices[ia];
            let pb = mesh.vertices[ib];
            let pc = mesh.vertices[ic];
            let tau_a0 = s_curr.proper_time + pa.dot(s_curr.phi);
            let tau_b0 = s_curr.proper_time + pb.dot(s_curr.phi);
            let tau_c0 = s_curr.proper_time + pc.dot(s_curr.phi);
            let tau_a1 = s_next.proper_time + pa.dot(s_next.phi);
            let tau_b1 = s_next.proper_time + pb.dot(s_next.phi);
            let tau_c1 = s_next.proper_time + pc.dot(s_next.phi);

            let uva = tri.uv[perm[0]];
            let uvb = tri.uv[perm[1]];
            let uvc = tri.uv[perm[2]];

            // Compute 3D rest-frame tangent vectors (t_u = dp / du, t_v = dp / dv) on this triangle
            // by inverting the 2x2 UV Jacobian:
            //   p(u, v) = p_a + (u - u_a) * t_u + (v - v_a) * t_v
            // Because local material-point proper time on a constant-tau_c slice is
            //   tau_local(u, v, tau_c) = tau_c + p(u, v) . Phi(tau_c),
            // its directional derivatives along (u, v) in the rest frame are simply:
            //   (d_tau / du, d_tau / dv) = (t_u . Phi(tau_c), t_v . Phi(tau_c)).
            let dp1 = pb - pa;
            let dp2 = pc - pa;
            let du1 = uvb.0 - uva.0;
            let dv1 = uvb.1 - uva.1;
            let du2 = uvc.0 - uva.0;
            let dv2 = uvc.1 - uva.1;
            let uv_det = du1 * dv2 - dv1 * du2;
            let (tu, tv) = if uv_det.abs() > 1e-10 {
                let inv_det = 1.0 / uv_det;
                (
                    (dp1 * dv2 - dp2 * dv1) * inv_det,
                    (dp2 * du1 - dp1 * du2) * inv_det,
                )
            } else {
                (Vec3::default(), Vec3::default())
            };
            let uv_world_scale = (tu.length(), tv.length());
            let tau_uv_0 = (tu.dot(s_curr.phi), tv.dot(s_curr.phi));
            let tau_uv_1 = (tu.dot(s_next.phi), tv.dot(s_next.phi));

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
                    [tau_a0, tau_b0, tau_c0, tau_c1],
                    [tau_uv_0, tau_uv_0, tau_uv_0, tau_uv_1],
                    [dec0, dec0, dec0, dec1],
                ),
                (
                    [a0, b0, b1, c1],
                    [uva, uvb, uvb, uvc],
                    [ba, bb, bb, bc],
                    [tau_a0, tau_b0, tau_b1, tau_c1],
                    [tau_uv_0, tau_uv_0, tau_uv_1, tau_uv_1],
                    [dec0, dec0, dec1, dec1],
                ),
                (
                    [a0, a1, b1, c1],
                    [uva, uva, uvb, uvc],
                    [ba, ba, bb, bc],
                    [tau_a0, tau_a1, tau_b1, tau_c1],
                    [tau_uv_0, tau_uv_1, tau_uv_1, tau_uv_1],
                    [dec0, dec1, dec1, dec1],
                ),
            ];

            for (verts, uv, tb, taus, tau_grads, decs) in tets {
                if let Some(simplex) = Simplex4D::new(
                    verts,
                    uv,
                    tb,
                    taus,
                    tau_grads,
                    uv_world_scale,
                    surface_normal_4d,
                    u_avg,
                    material_id,
                ) {
                    simplices.push(simplex.with_decided_at(decs));
                }
            }
        }
    }

    simplices
}

/// Seal a Born-rigid worldtube with the body's rest-frame cross-section at `sample` (e.g. where
/// a Dynamic Mode history currently ends at the causal frontier): one tetrahedron per surface
/// triangle, fanned to the body center. The cap is a spacelike 3-volume, so rays crossing the
/// end of the tube see the body's interior there. The fan only fills the body if every surface
/// point can see the center: for other shapes, cap each convex part with
/// [`build_worldtube_cap_about`].
pub fn build_worldtube_cap(
    mesh: &Mesh3D,
    sample: &WorldlineSample,
    decided_at: f32,
    material_id: usize,
) -> Vec<Simplex4D> {
    build_worldtube_cap_about(mesh, sample, decided_at, material_id, Vec3::default())
}

/// [`build_worldtube_cap`], fanned to `pivot` (in the body's rest frame) instead of the center:
/// a point that sees all of `mesh`, such as the center of a convex part.
pub fn build_worldtube_cap_about(
    mesh: &Mesh3D,
    sample: &WorldlineSample,
    decided_at: f32,
    material_id: usize,
    pivot: Vec3,
) -> Vec<Simplex4D> {
    let to_world = |p: Vec3| sample.center + sample.tetrad.local_spatial_to_world(p);
    let tau_at = |p: Vec3| sample.proper_time + p.dot(sample.phi);
    let tri_bary = [
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
        Vec3::new(0.0, 0.0, 1.0),
        Vec3::new(1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
    ];
    mesh.triangles
        .iter()
        .filter_map(|tri| {
            let [pa, pb, pc] = tri.indices.map(|i| mesh.vertices[i]);
            let [uva, uvb, uvc] = tri.uv;
            let uv_mid = ((uva.0 + uvb.0 + uvc.0) / 3.0, (uva.1 + uvb.1 + uvc.1) / 3.0);
            Simplex4D::new(
                [to_world(pa), to_world(pb), to_world(pc), to_world(pivot)],
                [uva, uvb, uvc, uv_mid],
                tri_bary,
                [tau_at(pa), tau_at(pb), tau_at(pc), tau_at(pivot)],
                [(0.0, 0.0); 4],
                (1.0, 1.0),
                sample.tetrad.local_spatial_to_world(tri.normal),
                sample.tetrad.e[0],
                material_id,
            )
        })
        .map(|s| s.with_decided_at([decided_at; 4]))
        .collect()
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
    let inv_gamma = 1.0 / (tetrad.gamma() as f64);
    let mut samples = Vec::with_capacity(steps + 1);

    for i in 0..=steps {
        let frac = i as f64 / steps as f64;
        let t = (t_min as f64) + frac * ((t_max - t_min) as f64);
        let pos = pos_at_t0 + beta * (t as f32);
        let tau = (t * inv_gamma) as f32;
        samples.push(WorldlineSample {
            center: Vec4::from_time_space(t as f32, pos),
            proper_time: tau,
            phi: Vec3::default(),
            proper_accel_local: Vec3::default(),
            tetrad,
        });
    }
    samples
}

/// Generate an accelerating worldline from a smooth spatial trajectory `pos_fn(t)`
/// over lab time `[t_min, t_max]`.
///
/// Uses `f64` precision with sub-stepped Fermi-Walker transport and Simpson quadrature for:
/// - proper time `d_tau = dt / gamma(t)`,
/// - Fermi-Walker transported comoving `Tetrad` (capturing Thomas precession), and
/// - integrated local proper acceleration `phi = \int a_local d_tau` (capturing Born-rigid
///   front-to-back clock offsets `tau_p = tau_c + p . phi`).
pub fn sample_accelerated_worldline<F>(
    pos_fn: F,
    t_min: f32,
    t_max: f32,
    steps: usize,
) -> Vec<WorldlineSample>
where
    F: Fn(f64) -> [f64; 3],
{
    let mut samples = Vec::with_capacity(steps + 1);
    let dt_deriv = 1e-5_f64;
    const SUBSTEPS: usize = 8;

    let velocity_f64 = |t: f64| -> [f64; 3] {
        let p_minus = pos_fn(t - dt_deriv);
        let p_plus = pos_fn(t + dt_deriv);
        let inv_2h = 0.5 / dt_deriv;
        let vx = (p_plus[0] - p_minus[0]) * inv_2h;
        let vy = (p_plus[1] - p_minus[1]) * inv_2h;
        let vz = (p_plus[2] - p_minus[2]) * inv_2h;
        let speed_sq = vx * vx + vy * vy + vz * vz;
        assert!(
            speed_sq < 1.0,
            "Worldline becomes superluminal at t = {}: |v| = {} >= c",
            t,
            speed_sq.sqrt()
        );
        [vx, vy, vz]
    };

    let four_vel_f64 = |t: f64| -> (f64, Vec4) {
        let [vx, vy, vz] = velocity_f64(t);
        let b2 = (vx * vx + vy * vy + vz * vz).min(0.999_999_999);
        let inv_gamma = (1.0 - b2).sqrt();
        let gamma = 1.0 / inv_gamma;
        let u = Vec4::new(
            gamma as f32,
            (gamma * vx) as f32,
            (gamma * vy) as f32,
            (gamma * vz) as f32,
        );
        (inv_gamma, u)
    };

    let proper_accel_world = |t: f64| -> Vec4 {
        let (inv_gamma, _) = four_vel_f64(t);
        let (_, u_minus) = four_vel_f64(t - dt_deriv);
        let (_, u_plus) = four_vel_f64(t + dt_deriv);
        let du_dt = (u_plus - u_minus) * (0.5 / dt_deriv as f32);
        // a^mu = du^mu / d_tau = gamma * du^mu / dt
        du_dt * (1.0 / inv_gamma as f32)
    };

    let t_min_d = t_min as f64;
    let t_span_d = (t_max - t_min) as f64;

    let [vx0, vy0, vz0] = velocity_f64(t_min_d);
    let mut tetrad = Tetrad::from_beta(Vec3::new(vx0 as f32, vy0 as f32, vz0 as f32));
    let mut tau = 0.0_f64;
    let mut phi = Vec3::default();
    let mut prev_t = t_min_d;

    for i in 0..=steps {
        let frac = i as f64 / steps as f64;
        let t = t_min_d + frac * t_span_d;

        if i > 0 {
            let sub_dt = (t - prev_t) / (SUBSTEPS as f64);
            for s in 0..SUBSTEPS {
                let ta = prev_t + (s as f64) * sub_dt;
                let tb = ta + sub_dt;
                let tm = 0.5 * (ta + tb);

                let (inv_ga, ua) = four_vel_f64(ta);
                let (inv_gm, _) = four_vel_f64(tm);
                let (inv_gb, ub) = four_vel_f64(tb);

                // Simpson's rule for proper time: d_tau = \int dt / gamma(t)
                tau += (sub_dt / 6.0) * (inv_ga + 4.0 * inv_gm + inv_gb);

                // Advance Fermi-Walker frame and accumulate Born-rigid rapidity vector:
                // d_phi = a_local * d_tau = <du, e_i>_eta
                let du = ub - ua;
                let d_phi_before = tetrad.world_to_local_spatial(du);
                tetrad.fermi_walker_transport(ub);
                let d_phi_after = tetrad.world_to_local_spatial(du);
                phi += (d_phi_before + d_phi_after) * 0.5;
            }
        }
        prev_t = t;

        let [px, py, pz] = pos_fn(t);
        let a_world = proper_accel_world(t);
        let a_local = tetrad.world_to_local_spatial(a_world);

        samples.push(WorldlineSample {
            center: Vec4::new(t as f32, px as f32, py as f32, pz as f32),
            proper_time: tau as f32,
            phi,
            proper_accel_local: a_local,
            tetrad,
        });
    }

    samples
}

/// Shift all `proper_time` and `phi` values along a sampled worldline by a constant offset so that
/// `proper_time == 0.0` and `phi == 0` at lab coordinate time `t == t_sync`.
pub fn sync_proper_time_at(samples: &mut [WorldlineSample], t_sync: f32) {
    if samples.is_empty() {
        return;
    }
    let mut tau_at_sync = samples[0].proper_time;
    let mut phi_at_sync = samples[0].phi;
    if t_sync <= samples[0].center.t {
        tau_at_sync = samples[0].proper_time;
        phi_at_sync = samples[0].phi;
    } else if let Some(last) = samples.last() {
        if t_sync >= last.center.t {
            tau_at_sync = last.proper_time;
            phi_at_sync = last.phi;
        } else {
            for w in samples.windows(2) {
                let s0 = &w[0];
                let s1 = &w[1];
                if t_sync >= s0.center.t && t_sync <= s1.center.t {
                    let dt = (s1.center.t - s0.center.t).max(1e-6);
                    let frac = (t_sync - s0.center.t) / dt;
                    tau_at_sync = s0.proper_time + frac * (s1.proper_time - s0.proper_time);
                    phi_at_sync = s0.phi + (s1.phi - s0.phi) * frac;
                    break;
                }
            }
        }
    }
    for s in samples.iter_mut() {
        s.proper_time -= tau_at_sync;
        s.phi -= phi_at_sync;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Each prism tet takes its vertices' decision times from the worldline step they sit on,
    /// static tubes get the sentinel, and hits interpolate barycentrically.
    #[test]
    fn worldtube_decided_at_follows_steps() {
        let mesh = Mesh3D::cube(Vec3::new(0.5, 0.5, 0.5));
        let samples =
            sample_inertial_worldline(Vec3::default(), Vec3::new(0.3, 0.0, 0.0), 0.0, 4.0, 2);
        assert_eq!(samples.len(), 3);

        let static_tube = build_worldtube_simplices(&mesh, &samples, 0);
        assert!(static_tube
            .iter()
            .all(|s| s.decided_at == [DECIDED_AT_STATIC; 4]));

        let decided = [10.0, 20.0, 30.0];
        let tube = build_worldtube_simplices_with_decided(&mesh, &samples, Some(&decided), 0);
        assert_eq!(tube.len(), static_tube.len());
        let mut steps_seen = [false; 2];
        for s in &tube {
            // Every tet spans exactly one step: its decision times are that step's two values.
            let lo = s.decided_at.iter().cloned().fold(f32::INFINITY, f32::min);
            let hi = s.decided_at.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let step = if lo == 10.0 { 0 } else { 1 };
            assert_eq!((lo, hi), (decided[step], decided[step + 1]));
            steps_seen[step] = true;

            let bary = [0.1, 0.2, 0.3, 0.4];
            let hit = s.interpolate_hit(1.0, s.v0, bary);
            let expected: f32 = (0..4).map(|i| s.decided_at[i] * bary[i]).sum();
            assert!((hit.decided_at - expected).abs() < 1e-4);
        }
        assert_eq!(steps_seen, [true, true]);
    }

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
            [(0.0, 0.0); 4],
            (1.0, 1.0),
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

        // Slice the 4D worldtube at constant lab time t = 0 by casting a spatial ray along +X at t = 0:
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
        assert!(
            (measured_lab_length - 1.2).abs() < 1e-4,
            "measured_lab_length = {}",
            measured_lab_length
        );
    }

    #[test]
    fn test_thomas_precession_one_orbit() {
        // An orbiter moving in a circle of radius R in the XZ plane at constant speed beta = 0.80
        // (gamma = 5/3 = 1.66667) completes one full revolution in lab time T = 2*pi*R / beta.
        // Fermi-Walker transport around the circle must precess the comoving spatial axes in the
        // XZ plane by the exact Thomas precession angle:
        //   Delta_theta_Thomas = 2 * pi * (gamma - 1) = 2 * pi * (2/3) = 4*pi/3 rad.
        let beta = 0.80_f64;
        let gamma = 1.0 / (1.0 - beta * beta).sqrt(); // 5/3
        let radius = 4.0_f64;
        let omega = beta / radius;
        let period = std::f64::consts::TAU / omega;

        let samples = sample_accelerated_worldline(
            |t| {
                let (s, c) = (omega * t).sin_cos();
                [radius * c, 0.0, radius * s]
            },
            0.0,
            period as f32,
            180,
        );

        let first = samples.first().unwrap();
        let last = samples.last().unwrap();

        // Both start and end have the exact same 3-velocity (0, 0, +beta).
        // Project the transported spatial axes back to the lab rest frame via a single pure boost
        // to compare their orientation in the XZ plane:
        let mut t0 = first.tetrad;
        t0.boost_to(Vec4::new(1.0, 0.0, 0.0, 0.0));
        let mut t1 = last.tetrad;
        t1.boost_to(Vec4::new(1.0, 0.0, 0.0, 0.0));

        // Angle of e[1] in the XZ plane:
        let angle0 = t0.e[1].z.atan2(t0.e[1].x);
        let angle1 = t1.e[1].z.atan2(t1.e[1].x);
        let measured_rot = (angle0 - angle1).rem_euclid(std::f32::consts::TAU);
        let expected_rot =
            ((gamma - 1.0) * std::f64::consts::TAU).rem_euclid(std::f64::consts::TAU) as f32;

        assert!(
            (measured_rot - expected_rot).abs() < 5e-3,
            "Thomas precession angle after 1 orbit: got {} rad, expected {} rad",
            measured_rot,
            expected_rot
        );
    }

    #[test]
    fn test_born_rigid_front_back_proper_time_offset() {
        // Accelerate a rigid cube of half-length p = 1.0 smoothly from rest (t = 0) up to a constant
        // cruise velocity beta_cruise = 0.80 along +X, then decelerate back to rest.
        // During the constant-velocity cruise phase, a comoving slice (constant center proper time)
        // must show the front face (+p) ahead of the back face (-p) by:
        //   Delta_tau = 2 * p * atanh(beta_cruise)
        // And after decelerating back to rest, Phi returns to 0 so front and back resynchronize!
        let beta_max = 0.80_f64;
        let expected_phi_cruise = beta_max.atanh() as f32;

        // Smooth velocity profile v(t): 0 at t=0, ramps to beta_max at t=10, returns to 0 at t=20:
        // Position x(t) whose derivative is beta_max * sin^2(pi * t / 20):
        // int_0^t sin^2(k u) du = 0.5 * t - sin(2 k t) / (4 k)
        let k = std::f64::consts::PI / 20.0;
        let mut samples = sample_accelerated_worldline(
            |t| {
                let tc = t.clamp(0.0, 20.0);
                let x = beta_max * (0.5 * tc - (2.0 * k * tc).sin() / (4.0 * k));
                [x, 0.0, 0.0]
            },
            0.0,
            20.0,
            100,
        );
        sync_proper_time_at(&mut samples, 0.0);

        // At midpoint t = 10.0 (sample 50), velocity is beta_max = 0.80, so phi.x == atanh(0.80):
        let mid_sample = &samples[50];
        assert!(
            (mid_sample.phi.x - expected_phi_cruise).abs() < 2e-3,
            "cruise phi.x = {}, expected atanh(0.8) = {}",
            mid_sample.phi.x,
            expected_phi_cruise
        );

        // At end t = 20.0 (sample 100), velocity is back to 0, so phi.x returns to 0:
        let end_sample = samples.last().unwrap();
        assert!(
            end_sample.phi.x.abs() < 2e-3,
            "end phi.x = {}, expected 0.0 after returning to rest",
            end_sample.phi.x
        );

        // Build the 4D worldtube around midpoint sample 50 and slice the -Z clock face in the
        // cube's instantaneous comoving rest frame (constant center proper time tau_c = mid_sample.proper_time).
        // Verify that:
        // 1. Local proper time `hit.proper_time` drifts across `u` by `2 * he.x * phi.x`, and
        // 2. Subtracting `(u - 0.5) * tau_uv_grad.0 + (v - 0.5) * tau_uv_grad.1` recovers the exact
        //    same `central_proper_time == mid_sample.proper_time` at every point on the clock face!
        let cube = Mesh3D::cube(Vec3::new(1.0, 1.0, 1.0));
        let simplices = build_worldtube_simplices(&cube, &samples[49..=51], 0);
        for &x_local in &[-0.75_f32, -0.25, 0.25, 0.75] {
            let ray_origin = mid_sample.center
                + mid_sample
                    .tetrad
                    .local_spatial_to_world(Vec3::new(x_local, 0.2, -3.0));
            let ray_dir = mid_sample
                .tetrad
                .local_spatial_to_world(Vec3::new(0.0, 0.0, 1.0));
            let hit = simplices
                .iter()
                .filter_map(|s| s.intersect(ray_origin, ray_dir, 0.0, 2.5))
                .min_by(|a, b| a.lambda.partial_cmp(&b.lambda).unwrap())
                .expect("comoving ray should hit -Z clock face");

            assert!((hit.uv_world_scale.0 - 2.0).abs() < 1e-4);
            assert!((hit.uv_world_scale.1 - 2.0).abs() < 1e-4);
            let central_tau = hit.proper_time
                - (hit.uv.0 - 0.5) * hit.tau_uv_grad.0
                - (hit.uv.1 - 0.5) * hit.tau_uv_grad.1;
            assert!(
                (central_tau - mid_sample.proper_time).abs() < 1e-3,
                "central_tau = {}, expected mid_sample.proper_time = {} at x_local = {}",
                central_tau,
                mid_sample.proper_time,
                x_local
            );
        }
    }
}
