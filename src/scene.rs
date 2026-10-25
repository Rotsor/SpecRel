use crate::bvh4d::Bvh4D;
use crate::mesh4d::{
    build_worldtube_simplices, sample_accelerated_worldline, sample_inertial_worldline,
    sync_proper_time_at, Material, Mesh3D, Simplex4D, SurfacePattern,
};
use crate::minkowski::{Vec3, Vec4};

/// Selectable scenes: five precomputed Special Relativity thought experiments (keys 1-5) and
/// the causal Dynamic Mode scenes (key 6+), whose geometry grows up to the camera's future light cone.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DemoScene {
    TwinParadox,
    BarnAndPole,
    TerrellPenroseStreet,
    EinsteinTrainSync,
    RelativisticCarousel,
    RelativisticBilliards,
}

impl DemoScene {
    pub const ALL: [DemoScene; 6] = [
        DemoScene::TwinParadox,
        DemoScene::BarnAndPole,
        DemoScene::TerrellPenroseStreet,
        DemoScene::EinsteinTrainSync,
        DemoScene::RelativisticCarousel,
        DemoScene::RelativisticBilliards,
    ];

    /// Dynamic Mode scenes are simulated causally (`crate::dynamic`) instead of precomputed.
    pub fn is_dynamic(self) -> bool {
        matches!(self, Self::RelativisticBilliards)
    }

    pub fn next(self) -> Self {
        match self {
            Self::TwinParadox => Self::BarnAndPole,
            Self::BarnAndPole => Self::TerrellPenroseStreet,
            Self::TerrellPenroseStreet => Self::EinsteinTrainSync,
            Self::EinsteinTrainSync => Self::RelativisticCarousel,
            Self::RelativisticCarousel => Self::RelativisticBilliards,
            Self::RelativisticBilliards => Self::TwinParadox,
        }
    }

    pub fn prev(self) -> Self {
        match self {
            Self::TwinParadox => Self::RelativisticBilliards,
            Self::RelativisticBilliards => Self::RelativisticCarousel,
            Self::BarnAndPole => Self::TwinParadox,
            Self::TerrellPenroseStreet => Self::BarnAndPole,
            Self::EinsteinTrainSync => Self::TerrellPenroseStreet,
            Self::RelativisticCarousel => Self::EinsteinTrainSync,
        }
    }

    pub fn title(self) -> &'static str {
        match self {
            Self::TwinParadox => "1: TWIN PARADOX (STAY-AT-HOME VS ACCELERATED TRAVELER)",
            Self::BarnAndPole => "2: BARN & POLE PARADOX (8M POLE IN 5M BARN AT 0.85C)",
            Self::TerrellPenroseStreet => "3: TERRELL-PENROSE ROTATION & PENROSE SPHERE STREET",
            Self::EinsteinTrainSync => "4: EINSTEIN TRAIN & RELATIVITY OF SIMULTANEITY",
            Self::RelativisticCarousel => "5: RELATIVISTIC CAROUSEL & THOMAS PRECESSION",
            Self::RelativisticBilliards => "6: RELATIVISTIC BILLIARDS (CAUSAL DYNAMIC MODE)",
        }
    }

    pub fn description(self) -> &'static str {
        match self {
            Self::TwinParadox => {
                "LEFT=HOME (V=0), RIGHT=TRAVELER (0.94C PEAK). REUNITE AT T=+14S WITH 8S CLOCK GAP!"
            }
            Self::BarnAndPole => {
                "PAUSE [P] NEAR T=0, PRESS [G] FOR GOD'S EYE & [V] TO MATCH POLE VELOCITY!"
            }
            Self::TerrellPenroseStreet => {
                "FLYING CUBES (0.85C) ROTATE TRAILING CLOCK FACES INTO VIEW; SPHERE STAYS CIRCULAR."
            }
            Self::EinsteinTrainSync => {
                "BOTTOM: 7 LAB-SYNCED CLOCKS. TOP: 7 TRAIN-SYNCED CLOCKS (0.80C). TRY [G] + [V]!"
            }
            Self::RelativisticCarousel => {
                "CENTER=MASTER CLOCK (V=0). RING=6 CLOCKS ORBITING AT 0.80C (GAMMA=1.67) + LIGHT CLOCK!"
            }
            Self::RelativisticBilliards => {
                "RAM PUCKS WITH THE SHIP. [G] FRONTIER/PROPHECY, [M] DECISION AGE"
            }
        }
    }

    pub fn initial_camera_pos(self) -> Vec4 {
        match self {
            Self::TwinParadox => Vec4::new(-15.0, 2.2, 0.0, -3.5),
            Self::BarnAndPole => Vec4::new(-4.5, 0.0, 0.1, -3.2),
            Self::TerrellPenroseStreet => Vec4::new(0.0, 0.0, 0.0, -7.5),
            Self::EinsteinTrainSync => Vec4::new(-3.0, 0.0, 0.0, -4.5),
            Self::RelativisticCarousel => Vec4::new(-18.0, 0.0, 0.8, -2.5),
            Self::RelativisticBilliards => Vec4::new(0.0, 0.0, 0.4, -6.0),
        }
    }

    /// Characteristic 3-velocity of the primary moving frame in this demo scene at lab time `t`
    /// (used by the `[V]` key to jump directly into the moving object's comoving rest frame!).
    pub fn comoving_beta(self, t: f32) -> Vec3 {
        match self {
            Self::TwinParadox => {
                let (_, vx) = twin_traveler_x_and_vx(t as f64);
                Vec3::new(vx as f32, 0.0, 0.0)
            }
            Self::BarnAndPole => Vec3::new(0.85, 0.0, 0.0),
            Self::TerrellPenroseStreet => Vec3::new(0.85, 0.0, 0.0),
            Self::EinsteinTrainSync => Vec3::new(0.80, 0.0, 0.0),
            Self::RelativisticCarousel => Vec3::new(0.80, 0.0, 0.0),
            Self::RelativisticBilliards => crate::dynamic::billiards::COMOVING_BETA,
        }
    }
}

/// Shared analytic position `x(t_rel)` and velocity `vx(t_rel)` for the Traveling Twin in Scene 1.
/// - Rests at `x = 1.4` for `t_rel <= -14.0` and `t_rel >= +14.0` (`vx = 0.0`).
/// - Executes a smooth raised-cosine excursion for `t_rel in [-14.0, +14.0]` peaking at `0.9425c`.
#[inline]
pub fn twin_traveler_x_and_vx(t_rel: f64) -> (f64, f64) {
    let x_start = 1.4_f64;
    let t_dep = -14.0_f64;
    let t_ret = 14.0_f64;
    if t_rel <= t_dep || t_rel >= t_ret {
        (x_start, 0.0)
    } else {
        let period = t_ret - t_dep; // 28.0s
        let omega = std::f64::consts::TAU / period;
        let amp = 4.2_f64; // v_max = amp * omega = 0.9425c (gamma_max = 2.99)
        let phase = omega * (t_rel - t_dep);
        let (s, c) = phase.sin_cos();
        (x_start + amp * (1.0 - c), amp * omega * s)
    }
}

/// Invoke `build_copy(t_offset)` across periodic loop offsets `k in k_range` (`t_offset = k * t_loop`).
fn for_periodic_copies<F>(t_loop: f32, k_range: std::ops::RangeInclusive<i32>, mut build_copy: F)
where
    F: FnMut(f32),
{
    for loop_k in k_range {
        build_copy((loop_k as f32) * t_loop);
    }
}

pub struct Scene4D {
    pub demo: DemoScene,
    pub bvh: Bvh4D,
    pub materials: Vec<Material>,
    /// Spatial position of the stationary point light source's worldline (t_L, light_pos)
    pub light_pos: Vec3,
    /// Valid lab-time window [t_min, t_max] of the generated 4D worldtubes
    pub t_min: f32,
    pub t_max: f32,
}

impl Scene4D {
    pub fn build(demo: DemoScene) -> Self {
        let materials = Self::common_materials();
        match demo {
            DemoScene::TwinParadox => Self::build_twin_paradox(demo, materials),
            DemoScene::BarnAndPole => Self::build_barn_and_pole(demo, materials),
            DemoScene::TerrellPenroseStreet => Self::build_terrell_penrose_street(demo, materials),
            DemoScene::EinsteinTrainSync => Self::build_einstein_train(demo, materials),
            DemoScene::RelativisticCarousel => Self::build_carousel(demo, materials),
            DemoScene::RelativisticBilliards => crate::dynamic::DynWorld::spawn(demo).build_scene(),
        }
    }

    pub fn common_materials() -> Vec<Material> {
        vec![
            // 0: Ground grid (slate gray/blue checkerboard)
            Material::new(
                Vec3::new(0.22, 0.25, 0.30),
                Vec3::new(0.13, 0.15, 0.19),
                0.18,
                1.0,
                0.012,
            ),
            // 1: Stationary / Home Clock (cool cyan/blue bezel)
            Material::new(
                Vec3::new(0.20, 0.72, 0.95),
                Vec3::new(0.10, 0.25, 0.45),
                0.28,
                1.0,
                0.025,
            )
            .with_pattern(SurfacePattern::ClockFace),
            // 2: High-Speed / Traveling Clock (warm coral/red bezel)
            Material::new(
                Vec3::new(0.96, 0.32, 0.24),
                Vec3::new(0.45, 0.12, 0.10),
                0.28,
                1.0,
                0.025,
            )
            .with_pattern(SurfacePattern::ClockFace),
            // 3: Gold / Accelerating Clock (amber/gold bezel)
            Material::new(
                Vec3::new(0.96, 0.78, 0.20),
                Vec3::new(0.45, 0.30, 0.08),
                0.28,
                1.0,
                0.025,
            )
            .with_pattern(SurfacePattern::ClockFace),
            // 4: Magenta / Orbiting Clock (violet/magenta bezel)
            Material::new(
                Vec3::new(0.88, 0.32, 0.88),
                Vec3::new(0.32, 0.10, 0.38),
                0.28,
                1.0,
                0.025,
            )
            .with_pattern(SurfacePattern::ClockFace),
            // 5: Einstein Bouncing-Photon Light Clock (emerald/cyan photon cavity, L = 1.0 light-second)
            Material::new(
                Vec3::new(0.20, 0.95, 0.65),
                Vec3::new(0.08, 0.22, 0.18),
                0.32,
                1.0,
                0.025,
            )
            .with_pattern(SurfacePattern::LightClock)
            .with_cavity_length(1.0),
            // 6: Structural / Barn / Sphere Checkerboard (ivory & teal)
            Material::new(
                Vec3::new(0.85, 0.82, 0.72),
                Vec3::new(0.22, 0.52, 0.62),
                0.18,
                2.0,
                0.025,
            ),
            // 7: Moving puck (bright electric cyan & silver checkerboard)
            Material::new(
                Vec3::new(0.25, 0.92, 1.00),
                Vec3::new(0.85, 0.94, 0.98),
                0.28,
                2.0,
                0.030,
            ),
            // 8: Resting puck (warm coral & cream checkerboard)
            Material::new(
                Vec3::new(0.96, 0.38, 0.22),
                Vec3::new(0.95, 0.86, 0.72),
                0.22,
                2.0,
                0.030,
            ),
            // 9: Merged puck: kinetic energy turned into rest mass (incandescent gold/orange glow)
            Material::new(
                Vec3::new(1.00, 0.82, 0.22),
                Vec3::new(0.98, 0.35, 0.10),
                0.55,
                2.0,
                0.035,
            ),
            // 10: Sticky (inelastic) puck (magenta & plum checkerboard)
            Material::new(
                Vec3::new(0.95, 0.25, 0.72),
                Vec3::new(0.42, 0.10, 0.35),
                0.30,
                2.0,
                0.030,
            ),
            // 11: Spore: clock face counting its own proper time down to the decay (lime bezel)
            Material::new(
                Vec3::new(0.55, 1.00, 0.25),
                Vec3::new(0.12, 0.40, 0.10),
                0.30,
                1.0,
                0.025,
            )
            .with_pattern(SurfacePattern::ClockFace),
            // 12-15: Ship hull cages, in the colors of clock bezels 1-4
            Self::hull_material(Vec3::new(0.20, 0.72, 0.95)),
            Self::hull_material(Vec3::new(0.96, 0.32, 0.24)),
            Self::hull_material(Vec3::new(0.96, 0.78, 0.20)),
            Self::hull_material(Vec3::new(0.88, 0.32, 0.88)),
        ]
    }

    /// A ship's hull cage: plain bars in the player's color.
    const fn hull_material(color: Vec3) -> Material {
        Material::new(color, color, 0.35, 1.0, 0.0)
    }

    /// Scene 1: Twin Paradox (Stay-at-Home Twin vs. Accelerated Round-Trip Twin + Light Clocks)
    fn build_twin_paradox(demo: DemoScene, materials: Vec<Material>) -> Self {
        let t_min = -18.0;
        let t_max = 34.0;
        let t_loop = t_max - t_min; // 52.0s loop period (gives >10s after photon arrival to compare clocks!)
        let t_sync_base = -14.0;
        let mut all_simplices: Vec<Simplex4D> = Vec::new();

        // Ground spanning all loop iterations
        let ground = Mesh3D::ground_grid(22.0, 24.0, -2.2, 4, 4);
        let ground_wl =
            sample_inertial_worldline(Vec3::default(), Vec3::default(), -160.0, 120.0, 1);
        all_simplices.extend(build_worldtube_simplices(&ground, &ground_wl, 0));

        let twin_cube = Mesh3D::cube(Vec3::new(1.15, 1.15, 1.15));
        // he.y = 0.625 => total height H = 1.25m; cavity v in [0.10, 0.90] spans L = 0.80 * 1.25 = 1.00 light-second!
        let light_clock_pillar = Mesh3D::cube(Vec3::new(0.55, 0.625, 0.55));
        let marker_cube = Mesh3D::cube(Vec3::new(0.6, 0.6, 0.6));

        // Generate contiguous loop-iteration copies (loop_k in -2..=1) so that when camera time
        // wraps from t_max (+34s) back to t_min (-18s), delayed photons from the end of loop_k = -1
        // continue reaching the camera seamlessly before the new loop's departure begins!
        for_periodic_copies(t_loop, -2..=1, |t_offset| {
            let seg_t0 = t_min + t_offset;
            let seg_t1 = t_max + t_offset;
            let t_sync = t_sync_base + t_offset;

            // 1. Stay-at-Home Twin at x = -1.2, y = -1.05, z = 6.0
            let mut home_wl = sample_inertial_worldline(
                Vec3::new(-1.2, -1.05, 6.0),
                Vec3::default(),
                seg_t0,
                seg_t1,
                2,
            );
            sync_proper_time_at(&mut home_wl, t_sync);
            all_simplices.extend(build_worldtube_simplices(&twin_cube, &home_wl, 1));

            // Stationary Einstein Light-Clock next to Stay-at-Home Twin (resting on ground y = -2.2)
            let mut home_lc_wl = sample_inertial_worldline(
                Vec3::new(-3.2, -1.575, 6.0),
                Vec3::default(),
                seg_t0,
                seg_t1,
                2,
            );
            sync_proper_time_at(&mut home_lc_wl, t_sync);
            all_simplices.extend(build_worldtube_simplices(&light_clock_pillar, &home_lc_wl, 5));

            // 2. Traveling Twin starting at x = +1.4, y = -1.05, z = 6.0
            let t_off_d = t_offset as f64;
            let mut trav_wl = sample_accelerated_worldline(
                |t| {
                    let (x, _) = twin_traveler_x_and_vx(t - t_off_d);
                    [x, -1.05, 6.0]
                },
                seg_t0,
                seg_t1,
                120,
            );
            sync_proper_time_at(&mut trav_wl, t_sync);
            all_simplices.extend(build_worldtube_simplices(&twin_cube, &trav_wl, 2));

            // Comoving Einstein Light-Clock riding on top of the Traveling Twin (top at y = +0.10)
            let mut trav_lc_wl = sample_accelerated_worldline(
                |t| {
                    let (x, _) = twin_traveler_x_and_vx(t - t_off_d);
                    [x, 0.725, 6.0]
                },
                seg_t0,
                seg_t1,
                120,
            );
            sync_proper_time_at(&mut trav_lc_wl, t_sync);
            all_simplices.extend(build_worldtube_simplices(&light_clock_pillar, &trav_lc_wl, 5));

            // Turnaround marker beacon at x = 9.2
            let mut marker_wl = sample_inertial_worldline(
                Vec3::new(9.2, -1.6, 7.8),
                Vec3::default(),
                seg_t0,
                seg_t1,
                2,
            );
            sync_proper_time_at(&mut marker_wl, t_sync);
            all_simplices.extend(build_worldtube_simplices(&marker_cube, &marker_wl, 3));
        });

        Self {
            demo,
            bvh: Bvh4D::build(all_simplices),
            materials,
            light_pos: Vec3::new(2.0, 6.5, 2.0),
            t_min,
            t_max,
        }
    }

    /// Scene 2: Barn and Pole (Ladder) Paradox
    /// A 5.0m Barn at rest vs. an 8.0m Pole moving at beta_x = 0.85c (gamma = 1.90, contracted length = 4.21m).
    fn build_barn_and_pole(demo: DemoScene, materials: Vec<Material>) -> Self {
        let t_min = -12.0;
        let t_max = 12.0;
        let t_loop = t_max - t_min; // 24.0s loop period
        let mut all_simplices: Vec<Simplex4D> = Vec::new();

        // Ground
        let ground = Mesh3D::ground_grid(22.0, 22.0, -2.2, 4, 4);
        let ground_wl =
            sample_inertial_worldline(Vec3::default(), Vec3::default(), -140.0, 140.0, 1);
        all_simplices.extend(build_worldtube_simplices(&ground, &ground_wl, 0));

        // Open Barn spanning x in [-2.5, +2.5] (interior length = 5.0m) around z = 6.2:
        // 1. Left & Right Barn Door Clock Pillars at x = -2.85 and x = +2.85 (so inner gap is 5.0m: [-2.5, +2.5])
        let door_pillar = Mesh3D::cube(Vec3::new(0.35, 1.5, 0.35));
        for &x_door in &[-2.85, 2.85] {
            let wl = sample_inertial_worldline(
                Vec3::new(x_door, -0.7, 5.2),
                Vec3::default(),
                -140.0,
                140.0,
                1,
            );
            all_simplices.extend(build_worldtube_simplices(&door_pillar, &wl, 1));
        }

        // 2. Overhead Barn Clock Header spanning [-3.2, +3.2] with 4 synchronized Barn Clocks
        let roof_beam = Mesh3D::cube(Vec3::new(3.2, 0.8, 0.6));
        let roof_wl = sample_inertial_worldline(
            Vec3::new(0.0, 1.6, 5.8),
            Vec3::default(),
            -140.0,
            140.0,
            1,
        );
        all_simplices.extend(build_worldtube_simplices(&roof_beam, &roof_wl, 1));

        // 3. Rear Barn Wall at z = 7.4 spanning [-2.5, +2.5]
        let back_wall = Mesh3D::cube(Vec3::new(2.5, 1.5, 0.25));
        let wall_wl = sample_inertial_worldline(
            Vec3::new(0.0, -0.7, 7.4),
            Vec3::default(),
            -140.0,
            140.0,
            1,
        );
        all_simplices.extend(build_worldtube_simplices(&back_wall, &wall_wl, 6));

        // 4. The 8.0m Relativistic Pole (he.x = 4.0 => total rest length L0 = 8.0m > 5.0m Barn!).
        // At beta_x = 0.85c (gamma = 1.898), its lab-frame contracted length is 8.0 / 1.898 = 4.21m < 5.0m!
        let pole_mesh = Mesh3D::cube(Vec3::new(4.0, 0.55, 0.45));
        let beta = Vec3::new(0.85, 0.0, 0.0);
        for_periodic_copies(t_loop, -2..=1, |t_offset| {
            let pos_at_t0 = Vec3::new(0.0, -0.85, 6.0) - beta * t_offset;
            let mut pole_wl = sample_inertial_worldline(
                pos_at_t0,
                beta,
                t_offset - 55.0,
                t_offset + 55.0,
                12,
            );
            sync_proper_time_at(&mut pole_wl, t_offset);
            all_simplices.extend(build_worldtube_simplices(&pole_mesh, &pole_wl, 2));
        });

        Self {
            demo,
            bvh: Bvh4D::build(all_simplices),
            materials,
            light_pos: Vec3::new(0.0, 6.0, 1.5),
            t_min,
            t_max,
        }
    }

    /// Scene 3: Terrell-Penrose Rotation & Penrose Sphere Street with Clock Faces
    fn build_terrell_penrose_street(demo: DemoScene, materials: Vec<Material>) -> Self {
        let t_min = -14.0;
        let t_max = 14.0;
        let t_loop = t_max - t_min; // 28.0s loop period
        let mut all_simplices: Vec<Simplex4D> = Vec::new();

        // 1. Stationary ground plane
        let ground_mesh = Mesh3D::ground_grid(20.0, 28.0, -2.2, 4, 6);
        let ground_wl =
            sample_inertial_worldline(Vec3::default(), Vec3::default(), -140.0, 140.0, 1);
        all_simplices.extend(build_worldtube_simplices(&ground_mesh, &ground_wl, 0));

        // 2. Stationary colonnade of reference Clock Cubes & Light Clocks on Left (-X) and Right (+X)
        let pillar_cube = Mesh3D::cube(Vec3::new(0.95, 0.95, 0.95));
        let lc_pillar = Mesh3D::cube(Vec3::new(0.65, 0.625, 0.65));
        for iz in -3..=3 {
            let z = iz as f32 * 5.0;
            for &x in &[-4.2, 4.2] {
                if iz == 0 {
                    let wl = sample_inertial_worldline(
                        Vec3::new(x, -1.575, z),
                        Vec3::default(),
                        -140.0,
                        140.0,
                        1,
                    );
                    all_simplices.extend(build_worldtube_simplices(&lc_pillar, &wl, 5));
                } else {
                    let wl = sample_inertial_worldline(
                        Vec3::new(x, -1.25, z),
                        Vec3::default(),
                        -140.0,
                        140.0,
                        1,
                    );
                    all_simplices.extend(build_worldtube_simplices(&pillar_cube, &wl, 1));
                }
            }
        }

        // 3 & 4. Loop-iteration copies (`loop_k in -2..=1`) of the high-speed inertial Flyer Clock Cube
        // (beta_x = 0.85c) and the Penrose Sphere (beta_z = -0.86c), shifted by `t_offset = k * t_loop`:
        let flyer_cube = Mesh3D::cube(Vec3::new(1.05, 1.05, 1.05));
        let beta_flyer = Vec3::new(0.85, 0.0, 0.0);
        let sphere_mesh = Mesh3D::uv_sphere(1.05, 10, 16);
        let beta_sphere = Vec3::new(0.0, 0.0, -0.86);

        for_periodic_copies(t_loop, -2..=1, |t_offset| {
            let flyer_pos_t0 = Vec3::new(0.0, -0.75, 4.5) - beta_flyer * t_offset;
            let mut flyer_wl = sample_inertial_worldline(
                flyer_pos_t0,
                beta_flyer,
                t_offset - 55.0,
                t_offset + 55.0,
                10,
            );
            sync_proper_time_at(&mut flyer_wl, t_offset);
            all_simplices.extend(build_worldtube_simplices(&flyer_cube, &flyer_wl, 2));

            let sphere_pos_t0 = Vec3::new(-1.8, 0.6, 5.0) - beta_sphere * t_offset;
            let mut sphere_wl = sample_inertial_worldline(
                sphere_pos_t0,
                beta_sphere,
                t_offset - 55.0,
                t_offset + 55.0,
                10,
            );
            sync_proper_time_at(&mut sphere_wl, t_offset);
            all_simplices.extend(build_worldtube_simplices(&sphere_mesh, &sphere_wl, 6));
        });

        // 5. Relativistically accelerating / oscillating Clock Cube along X
        // Period is exactly T_loop = 28.0s (omega_0 = 2*pi/28 = 0.2244 rad/s, v_max = 3.6 * omega_0 = 0.808c)
        let omega_0 = std::f64::consts::TAU / (t_loop as f64);
        let accel_cube = Mesh3D::cube(Vec3::new(0.95, 0.95, 0.95));
        let accel_wl = sample_accelerated_worldline(
            |t| {
                let a = 3.6_f64;
                [a * (omega_0 * t).sin(), 1.1, 8.5]
            },
            -110.0,
            70.0,
            260,
        );
        all_simplices.extend(build_worldtube_simplices(&accel_cube, &accel_wl, 3));

        // 6. Relativistically orbiting Clock Cube in the XZ plane
        // Angular frequency is omega_0 (1 full orbit per loop, r = 3.2m => v = 0.718c, well inside the light cylinder)
        let orbit_cube = Mesh3D::cube(Vec3::new(0.75, 0.75, 0.75));
        let orbit_wl = sample_accelerated_worldline(
            |t| {
                let r = 3.2_f64;
                let (s, c) = (omega_0 * t).sin_cos();
                [1.5 + r * c, 0.5, 11.5 + r * s]
            },
            -110.0,
            70.0,
            260,
        );
        all_simplices.extend(build_worldtube_simplices(&orbit_cube, &orbit_wl, 4));

        Self {
            demo,
            bvh: Bvh4D::build(all_simplices),
            materials,
            light_pos: Vec3::new(0.0, 6.5, 4.0),
            t_min,
            t_max,
        }
    }

    /// Scene 4: Einstein's Train & Platform Synchronization (Relativity of Simultaneity)
    /// Bottom row: 7 stationary Platform Clocks synchronized in the lab frame.
    /// Top row: A 7-car relativistic Train of Clocks cruising at beta_x = 0.80c (gamma = 1.667),
    /// synchronized in the train's own comoving rest frame!
    fn build_einstein_train(demo: DemoScene, materials: Vec<Material>) -> Self {
        let t_min = -14.0;
        let t_max = 14.0;
        let t_loop = t_max - t_min; // 28.0s loop period
        let mut all_simplices: Vec<Simplex4D> = Vec::new();

        // Ground
        let ground = Mesh3D::ground_grid(24.0, 22.0, -2.2, 4, 4);
        let ground_wl =
            sample_inertial_worldline(Vec3::default(), Vec3::default(), -140.0, 140.0, 1);
        all_simplices.extend(build_worldtube_simplices(&ground, &ground_wl, 0));

        // 1. Stationary Platform Row: 7 synchronized Clock Cubes at x = -7.8 .. +7.8
        let clock_cube = Mesh3D::cube(Vec3::new(0.95, 0.95, 0.95));
        for i in -3..=3 {
            let x = i as f32 * 2.6;
            let wl = sample_inertial_worldline(
                Vec3::new(x, -1.15, 6.8),
                Vec3::default(),
                -140.0,
                140.0,
                1,
            );
            all_simplices.extend(build_worldtube_simplices(&clock_cube, &wl, 1));
        }

        // Central Einstein Light-Clock on the platform at x = 0 (he.y = 0.625 => L = 1.0 light-second)
        let lc_box = Mesh3D::cube(Vec3::new(0.55, 0.625, 0.55));
        let lc_wl = sample_inertial_worldline(
            Vec3::new(0.0, -1.575, 5.1),
            Vec3::default(),
            -140.0,
            140.0,
            1,
        );
        all_simplices.extend(build_worldtube_simplices(&lc_box, &lc_wl, 5));

        // 2. Relativistic Train of 7 Synchronized Clock Wagons moving at beta_x = 0.80c (gamma = 5/3 = 1.667)
        let beta = Vec3::new(0.80, 0.0, 0.0);
        let gamma = 1.0f32 / (1.0f32 - beta.x * beta.x).sqrt(); // 5/3 = 1.6667
        for_periodic_copies(t_loop, -2..=1, |t_offset| {
            for i in -3..=3 {
                let x_rest = i as f32 * 2.6;
                // Position at lab time t = t_offset is x_rest / gamma, so at t = 0:
                let pos_at_t0 = Vec3::new(x_rest / gamma, 1.05, 6.8) - beta * t_offset;
                // Lab time at which wagon `i` of train `loop_k` crosses train proper time tau = 0:
                let t_sync_wagon = t_offset + gamma * beta.x * x_rest;

                let mut wl = sample_inertial_worldline(
                    pos_at_t0,
                    beta,
                    t_offset - 60.0,
                    t_offset + 60.0,
                    10,
                );
                sync_proper_time_at(&mut wl, t_sync_wagon);
                all_simplices.extend(build_worldtube_simplices(&clock_cube, &wl, 2));
            }
        });

        Self {
            demo,
            bvh: Bvh4D::build(all_simplices),
            materials,
            light_pos: Vec3::new(0.0, 6.5, 2.0),
            t_min,
            t_max,
        }
    }

    /// Scene 5: Relativistic Carousel (Transverse Time Dilation & Thomas Precession)
    fn build_carousel(demo: DemoScene, materials: Vec<Material>) -> Self {
        let t_min = -24.0;
        let t_max = 24.0;
        let t_loop = t_max - t_min; // 48.0s loop period
        let t_sync = -20.0;
        let mut all_simplices: Vec<Simplex4D> = Vec::new();

        // Ground
        let ground = Mesh3D::ground_grid(22.0, 24.0, -2.2, 4, 4);
        let ground_wl =
            sample_inertial_worldline(Vec3::default(), Vec3::default(), -140.0, 120.0, 1);
        all_simplices.extend(build_worldtube_simplices(&ground, &ground_wl, 0));

        // Central Stationary Master Clock at (0, -0.7, 8.0) + Einstein Light Clock on top
        let master_cube = Mesh3D::cube(Vec3::new(1.05, 1.05, 1.05));
        let mut master_wl = sample_inertial_worldline(
            Vec3::new(0.0, -0.7, 8.0),
            Vec3::default(),
            -110.0,
            70.0,
            2,
        );
        sync_proper_time_at(&mut master_wl, t_sync);
        all_simplices.extend(build_worldtube_simplices(&master_cube, &master_wl, 1));

        let lc_cube = Mesh3D::cube(Vec3::new(0.55, 0.625, 0.55));
        let mut lc_wl = sample_inertial_worldline(
            Vec3::new(0.0, 0.975, 8.0),
            Vec3::default(),
            -110.0,
            70.0,
            2,
        );
        sync_proper_time_at(&mut lc_wl, t_sync);
        all_simplices.extend(build_worldtube_simplices(&lc_cube, &lc_wl, 5));

        // 6 Relativistically Orbiting Clock Cubes around the Master Clock at v = 0.80c (gamma = 1.667).
        // Orbital angular frequency is chosen so 2 full revolutions occur per 48.0s loop (omega = 4*pi / 48.0).
        let orbiter_cube = Mesh3D::cube(Vec3::new(0.62, 0.62, 0.62));
        let omega = 2.0 * std::f64::consts::TAU / (t_loop as f64); // 2 revolutions per loop = 0.2618 rad/s
        let radius = 0.80 / omega; // v = R * omega = 0.80c (R = 3.056m)
        let t_sync_d = t_sync as f64;
        for k in 0..6 {
            let phase0 = (k as f64) * std::f64::consts::TAU / 6.0;
            let mut wl = sample_accelerated_worldline(
                |t| {
                    let angle = omega * (t - t_sync_d) + phase0;
                    let (s, c) = angle.sin_cos();
                    [
                        radius * c,
                        -0.6 + 0.25 * (2.0 * angle).sin(),
                        8.0 + radius * s,
                    ]
                },
                -110.0,
                70.0,
                320,
            );
            sync_proper_time_at(&mut wl, t_sync);
            let mat_id = if k % 2 == 0 { 2 } else { 3 };
            all_simplices.extend(build_worldtube_simplices(&orbiter_cube, &wl, mat_id));
        }

        Self {
            demo,
            bvh: Bvh4D::build(all_simplices),
            materials,
            light_pos: Vec3::new(0.0, 6.8, 3.0),
            t_min,
            t_max,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::minkowski::Tetrad;
    use crate::renderer::{
        light_clock_bounce_fraction, render_frame, Camera4D, FrameDynamics, RenderConfig, RenderMode,
        ShadowMode, SliceMode, PROPHECY_DELTA_DEFAULT,
    };

    #[test]
    fn test_all_five_paradox_scenes_render() {
        let cfg = RenderConfig {
            width: 80,
            height: 50,
            mode: RenderMode::RelativisticShaded,
            shadow_mode: ShadowMode::LorentzInvariant,
            slice_mode: SliceMode::PastLightCone,
            prophecy_delta: PROPHECY_DELTA_DEFAULT,
        };
        let mut fb = vec![0u32; cfg.width * cfg.height];

        for demo in DemoScene::ALL {
            let scene = Scene4D::build(demo);
            assert!(!scene.bvh.simplices.is_empty());
            let cam = Camera4D::new(demo.initial_camera_pos(), 65.0);
            render_frame(&scene, &cam, &cfg, &FrameDynamics::default(), &mut fb);
            assert!(fb.iter().all(|&px| px != 0));
        }
    }

    /// A player flash (`/`) emitted from a camera at rest is seen lighting a shell of radius
    /// `tau / 2` after `tau` seconds (light out and back): after 10 s the twins' clocks ~5 ls
    /// ahead are visibly lit, and after 120 s the shell (R = 60) is beyond all scene geometry.
    #[test]
    fn test_player_flash_visible_after_round_trip() {
        let cfg = RenderConfig {
            width: 160,
            height: 100,
            mode: RenderMode::RelativisticShaded,
            shadow_mode: ShadowMode::LorentzInvariant,
            slice_mode: SliceMode::PastLightCone,
            prophecy_delta: PROPHECY_DELTA_DEFAULT,
        };
        let demo = DemoScene::TwinParadox;
        let scene = Scene4D::build(demo);
        let lit_fraction = |tau: f32| {
            let mut cam = Camera4D::new(demo.initial_camera_pos(), 65.0);
            let mut dynamics = FrameDynamics::default();
            dynamics.emit_flash(cam.position);
            cam.position += cam.tetrad.e[0] * tau;
            let mut plain = vec![0u32; cfg.width * cfg.height];
            let mut lit = vec![0u32; cfg.width * cfg.height];
            render_frame(&scene, &cam, &cfg, &FrameDynamics::default(), &mut plain);
            render_frame(&scene, &cam, &cfg, &dynamics, &mut lit);
            plain.iter().zip(&lit).filter(|(a, b)| a != b).count() as f32 / plain.len() as f32
        };
        let at_10s = lit_fraction(10.0);
        assert!(at_10s > 0.01, "flash lit only {:.2}% of pixels after 10 s", at_10s * 100.0);
        assert_eq!(lit_fraction(120.0), 0.0);
    }

    #[test]
    fn test_twin_paradox_differential_clock_aging() {
        let scene = Scene4D::build(DemoScene::TwinParadox);

        // Probe the Stay-at-Home Twin (at x = -1.2, y = -1.05, z = 6.0) and the Traveling Twin
        // (reunited at x = +1.4, y = -1.05, z = 6.0) at lab time t = +15.0 (1 second after return):
        let ray_dir = Vec4::new(0.0, 0.0, 0.0, 1.0);
        let hit_home = scene
            .bvh
            .trace(Vec4::new(15.0, -1.2, -1.05, 0.0), ray_dir, 0.1, 20.0)
            .expect("should hit Stay-at-Home Twin");
        let hit_traveler = scene
            .bvh
            .trace(Vec4::new(15.0, 1.4, -1.05, 0.0), ray_dir, 0.1, 20.0)
            .expect("should hit Traveling Twin");

        // Both started synchronized at t = -14.0 (tau = 0).
        // At t = +15.0, Home Twin has aged 29.0s, while Traveling Twin has aged ~21.1s!
        assert!(
            (hit_home.proper_time - 29.0).abs() < 0.05,
            "home proper_time = {}",
            hit_home.proper_time
        );
        assert!(
            hit_traveler.proper_time < hit_home.proper_time - 7.0,
            "traveler proper_time ({}) should be >7s younger than home ({})",
            hit_traveler.proper_time,
            hit_home.proper_time
        );
    }

    #[test]
    fn test_barn_and_pole_world_slice_vs_camera_slice() {
        let scene = Scene4D::build(DemoScene::BarnAndPole);

        // 1. WORLD LAB SLICE (dt = 0, Barn rest frame) at t = 0.0:
        // The 8.0m pole is contracted to 4.21m (x in [-2.105, +2.105]), fitting inside the 5.0m barn!
        // Probing rear (x = -1.5) and front (x = +1.5) at t = 0 shows rear clock is ahead by 4.84s:
        let ray_dir_world = Vec4::new(0.0, 0.0, 0.0, 1.0);
        let hit_rear_world = scene
            .bvh
            .trace(Vec4::new(0.0, -1.5, -0.85, 0.0), ray_dir_world, 0.1, 10.0)
            .expect("should hit rear of pole in world slice");
        let hit_front_world = scene
            .bvh
            .trace(Vec4::new(0.0, 1.5, -0.85, 0.0), ray_dir_world, 0.1, 10.0)
            .expect("should hit front of pole in world slice");

        let measured_delta_tau = hit_rear_world.proper_time - hit_front_world.proper_time;
        let expected_delta_tau = (1.0f32 / (1.0f32 - 0.85 * 0.85).sqrt()) * 0.85 * 3.0;
        assert!(
            (measured_delta_tau - expected_delta_tau).abs() < 0.05,
            "world slice delta_tau = {}, expected = {}",
            measured_delta_tau,
            expected_delta_tau
        );

        // 2. CAMERA REST-FRAME SLICE (dt' = 0, Pole comoving frame at beta_x = +0.85c):
        // A camera boosted to beta_x = +0.85c at O = (0, 0, -0.85, 0) casts rays inside its own
        // simultaneity hyperplane `camera.tetrad.local_spatial_to_world(local_dir)`:
        let cam_tetrad = Tetrad::from_beta(Vec3::new(0.85, 0.0, 0.0));
        let cam_origin = Vec4::new(0.0, 0.0, -0.85, 0.0);
        let dir_rear_cam =
            cam_tetrad.local_spatial_to_world(Vec3::new(-3.5, 0.0, 6.0).normalized());
        let dir_front_cam =
            cam_tetrad.local_spatial_to_world(Vec3::new(3.5, 0.0, 6.0).normalized());

        let hit_rear_cam = scene
            .bvh
            .trace(cam_origin, dir_rear_cam, 0.1, 20.0)
            .expect("should hit pole at rest-frame x' = -3.5m in camera slice");
        let hit_front_cam = scene
            .bvh
            .trace(cam_origin, dir_front_cam, 0.1, 20.0)
            .expect("should hit pole at rest-frame x' = +3.5m in camera slice");

        // In the camera's comoving simultaneity slice (dt' = 0), ALL clocks along the pole
        // read the EXACT same proper time (tau = 0.0s)!
        assert!(
            (hit_rear_cam.proper_time - hit_front_cam.proper_time).abs() < 0.02,
            "camera slice rear tau ({}) and front tau ({}) should be identical!",
            hit_rear_cam.proper_time,
            hit_front_cam.proper_time
        );
    }

    #[test]
    fn test_einstein_train_all_wagons_visible_and_synced() {
        let scene = Scene4D::build(DemoScene::EinsteinTrainSync);
        let beta = 0.80f32;
        let gamma = 1.0f32 / (1.0f32 - beta * beta).sqrt(); // 5/3

        // Check across the entire scene time window t in [-14, +14] that all 7 wagons (i in -3..=3)
        // are present (never clipped) and display exact Lorentz-transformed proper time tau = gamma * (t - beta * x).
        for &t in &[-14.0f32, -7.0, 0.0, 7.0, 14.0] {
            for i in -3..=3 {
                let x_rest = i as f32 * 2.6;
                let x_lab = x_rest / gamma + beta * t;
                let hit = scene
                    .bvh
                    .trace(
                        Vec4::new(t, x_lab, 1.05, 0.0),
                        Vec4::new(0.0, 0.0, 0.0, 1.0),
                        0.1,
                        15.0,
                    )
                    .unwrap_or_else(|| panic!("Wagon {} clipped out at t = {}", i, t));

                let expected_tau = gamma * (t - beta * x_lab);
                assert!(
                    (hit.proper_time - expected_tau).abs() < 0.05,
                    "Wagon {} at t={}: proper_time={}, expected={}",
                    i,
                    t,
                    hit.proper_time,
                    expected_tau
                );
            }
        }
    }

    #[test]
    fn test_loop_boundary_past_light_cone_continuity() {
        // Verify that when the camera wraps from t_max to t_min in BarnAndPole,
        // the past light cone at t_min seamlessly hits the previous loop iteration's pole (`loop_k = -1`)
        // at the exact same spatial position and proper time as just before the wrap at t_max!
        let scene = Scene4D::build(DemoScene::BarnAndPole);
        let cam_tetrad = Tetrad::rest();
        // Aim a past-light-cone ray toward where the pole is at t_hit = t_max - dist (near x = +2.0, y = -0.85, z = 6.0):
        let cam_spatial = Vec3::new(0.0, -0.85, -3.2);
        let target_spatial = Vec3::new(2.0, -0.85, 6.0);
        let local_dir = (target_spatial - cam_spatial).normalized();
        let ray_dir = cam_tetrad.past_light_ray_direction(local_dir);

        let hit_at_tmax = scene
            .bvh
            .trace(
                Vec4::from_time_space(scene.t_max, cam_spatial),
                ray_dir,
                0.1,
                30.0,
            )
            .expect("should hit k=0 pole at t_max");
        let hit_at_tmin = scene
            .bvh
            .trace(
                Vec4::from_time_space(scene.t_min, cam_spatial),
                ray_dir,
                0.1,
                30.0,
            )
            .expect("should hit k=-1 pole at t_min after loop wrap!");

        assert!(
            (hit_at_tmax.event.spatial() - hit_at_tmin.event.spatial()).length() < 1e-3,
            "Spatial hit position jumped across loop wrap: {:?} vs {:?}",
            hit_at_tmax.event.spatial(),
            hit_at_tmin.event.spatial()
        );
        assert!(
            (hit_at_tmax.proper_time - hit_at_tmin.proper_time).abs() < 1e-2,
            "Proper time jumped across loop wrap: {} vs {}",
            hit_at_tmax.proper_time,
            hit_at_tmin.proper_time
        );
    }

    #[test]
    fn test_light_clock_subluminal_exact_speed_c() {
        // Verify that the Einstein light clock has rest-frame mirror separation L = 1.0 light-second,
        // rest-frame round-trip period T_0 = 2L/c = 2.0s (photon speed = 1.0c), and lab-frame period
        // gamma * 2L/c when moving transversely at speed beta = 0.80c (gamma = 5/3).
        let scene = Scene4D::build(DemoScene::TwinParadox);
        let lc_mat = &scene.materials[5];
        assert_eq!(lc_mat.pattern, SurfacePattern::LightClock);
        assert!((lc_mat.cavity_length - 1.0).abs() < 1e-6);

        // At tau = 0.0 -> bottom mirror (0.0); tau = L = 1.0 -> top mirror (1.0); tau = 2L = 2.0 -> bottom (0.0):
        assert!((light_clock_bounce_fraction(0.0, lc_mat.cavity_length) - 0.0).abs() < 1e-6);
        assert!((light_clock_bounce_fraction(0.5, lc_mat.cavity_length) - 0.5).abs() < 1e-6);
        assert!((light_clock_bounce_fraction(1.0, lc_mat.cavity_length) - 1.0).abs() < 1e-6);
        assert!((light_clock_bounce_fraction(2.0, lc_mat.cavity_length) - 0.0).abs() < 1e-6);

        // For a light clock moving transversely at beta = 0.80 (gamma = 5/3), lab time elapsed per
        // round-trip bounce is T_lab = gamma * 2L = 10/3 s, during which photon travels
        // y_dist = L = 1.0 transversely and x_dist = beta * (T_lab / 2) = 4/3 longitudinally,
        // giving total lab speed sqrt(1^2 + (4/3)^2) / (5/3) = 1.000c!
        let beta = 0.80_f32;
        let gamma = 1.0 / (1.0 - beta * beta).sqrt();
        let t_lab_half_period = gamma * lc_mat.cavity_length;
        let tau_at_half_period = t_lab_half_period / gamma;
        assert!(
            (light_clock_bounce_fraction(tau_at_half_period, lc_mat.cavity_length) - 1.0).abs()
                < 1e-5
        );
        let lab_path_len = (lc_mat.cavity_length.powi(2) + (beta * t_lab_half_period).powi(2)).sqrt();
        let lab_photon_speed = lab_path_len / t_lab_half_period;
        assert!((lab_photon_speed - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_penrose_sphere_circular_silhouette() {
        // Penrose (1959): A sphere moving at relativistic speed beta casts a strictly circular
        // silhouette on the observer's sky (constant angular radius around its aberrated sky center),
        // even though it is Lorentz-contracted by 1/gamma in 3D coordinate space!
        let radius = 1.2_f32;
        let sphere = Mesh3D::uv_sphere(radius, 24, 36);
        // Moving transversely at beta_x = 0.80c (gamma = 5/3, 3D x-length contracted by 0.60x):
        let beta = Vec3::new(0.80, 0.0, 0.0);
        let pos_t0 = Vec3::new(0.0, 0.0, 8.0);
        let wl = sample_inertial_worldline(pos_t0, beta, -40.0, 40.0, 4);
        let bvh = Bvh4D::build(build_worldtube_simplices(&sphere, &wl, 0));

        let obs = Vec4::new(0.0, 0.0, 0.0, 0.0);
        let tetrad = Tetrad::rest();

        // Find the retarded time t_ret < 0 where the sphere center's worldline X_c(t) = pos_t0 + beta * t
        // intersects the observer's past light cone |X_c(t)| = -t:
        let a = 1.0 - beta.length_sq();
        let b = -2.0 * pos_t0.dot(beta);
        let c = -pos_t0.length_sq();
        let t_ret = (-b - (b * b - 4.0 * a * c).sqrt()) / (2.0 * a);
        let retarded_dir = (pos_t0 + beta * t_ret).normalized();

        let u_perp = Vec3::new(0.0, 1.0, 0.0).cross(retarded_dir).normalized();

        let find_edge = |center_dir: Vec3, radial_dir: Vec3| -> f32 {
            let mut lo = 0.0_f32;
            let mut hi = 0.45_f32;
            for _ in 0..20 {
                let mid = 0.5 * (lo + hi);
                let dir = center_dir * mid.cos() + radial_dir * mid.sin();
                let ray = tetrad.past_light_ray_direction(dir);
                if bvh.trace(obs, ray, 0.1, 30.0).is_some() {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            0.5 * (lo + hi)
        };

        // Locate the true center of the aberrated circle along u_perp (the motion direction)
        let r_plus = find_edge(retarded_dir, u_perp);
        let r_minus = find_edge(retarded_dir, -u_perp);
        let shift = 0.5 * (r_plus - r_minus);
        let circle_center = (retarded_dir * shift.cos() + u_perp * shift.sin()).normalized();
        let u_circle = Vec3::new(0.0, 1.0, 0.0).cross(circle_center).normalized();
        let v_circle = circle_center.cross(u_circle).normalized();

        // Measure the silhouette angular radius across 8 azimuths around circle_center:
        let mut angular_radii = Vec::new();
        for k in 0..8 {
            let phi = (k as f32) * std::f32::consts::TAU / 8.0;
            let radial_dir = u_circle * phi.cos() + v_circle * phi.sin();
            angular_radii.push(find_edge(circle_center, radial_dir));
        }

        let mean_r: f32 = angular_radii.iter().sum::<f32>() / (angular_radii.len() as f32);
        for (k, &r) in angular_radii.iter().enumerate() {
            assert!(
                (r - mean_r).abs() / mean_r < 0.015,
                "Azimuth {} angular radius {} deviates from circular mean {} (radii = {:?})",
                k,
                r,
                mean_r,
                angular_radii
            );
        }
    }
}
