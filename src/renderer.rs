use crate::hud::glyph_5x7;
use crate::mesh4d::{HitRecord4D, Material, SurfacePattern};
use crate::minkowski::{rest_frame_cos, Tetrad, Vec3, Vec4};
use crate::scene::Scene4D;
use rayon::prelude::*;
use std::sync::OnceLock;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RenderMode {
    /// Standard shaded view with rest-frame surface textures (clocks/checkerboard), wireframe, and 4D shadows
    RelativisticShaded,
    /// Past-light-cone lookback time (lambda) & proper-time isochrone contours
    LookbackIsochrones,
    /// Raw 4D Simplex (tetrahedron) barycentric inspector
    Simplex4DInspector,
}

impl RenderMode {
    pub fn next(self) -> Self {
        match self {
            Self::RelativisticShaded => Self::LookbackIsochrones,
            Self::LookbackIsochrones => Self::Simplex4DInspector,
            Self::Simplex4DInspector => Self::RelativisticShaded,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::RelativisticShaded => "4D LIGHT-CONE SHADED",
            Self::LookbackIsochrones => "PAST LIGHT-CONE ISOCHRONES",
            Self::Simplex4DInspector => "4D TETRAHEDRAL MESH",
        }
    }
}

/// Observer (Camera) state in 4D Minkowski spacetime.
#[derive(Clone, Copy, Debug)]
pub struct Camera4D {
    /// Observer's current spacetime event O^mu = (t, x, y, z)
    pub position: Vec4,
    /// Observer's orthonormal comoving frame {e_0 = u^mu, e_1 = right, e_2 = up, e_3 = forward}
    pub tetrad: Tetrad,
    /// Observer's integrated proper time tau
    pub proper_time: f32,
    /// Vertical field of view in radians (in the observer's rest frame)
    pub fov_y_rad: f32,
}

impl Camera4D {
    pub fn new(position: Vec4, fov_deg: f32) -> Self {
        Self {
            position,
            tetrad: Tetrad::rest(),
            proper_time: 0.0,
            fov_y_rad: fov_deg.to_radians(),
        }
    }
}

/// Controls how rays view 4D Minkowski spacetime from the camera event `O^mu`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ViewMode {
    /// Real camera: True relativistic past light-cone (`ds^2 = 0`): `k^mu = -e_0 + n^a e_a`.
    /// Includes both Lorentz geometry and finite speed-of-light delay (Terrell-Penrose rotation & aberration).
    PastLightCone,
    /// God's eye: Instantaneous 3D hyperplane of simultaneity in the CAMERA'S boosted rest frame (`dt' = 0`):
    /// `<X - O, u_cam>_eta = 0`, i.e., `k^mu = n^1 e_1 + n^2 e_2 + n^3 e_3`.
    /// Strips away light-travel delay in the camera's own frame so you see pure Lorentz contraction
    /// and simultaneity as defined by the camera's current velocity!
    /// (Press `X` to brake to `v = 0` if you want the unboosted World Lab slice!)
    CameraSimultaneity,
}

/// Backwards-compatible alias for [`ViewMode`].
pub type SliceMode = ViewMode;

impl ViewMode {
    pub fn next(self) -> Self {
        match self {
            Self::PastLightCone => Self::CameraSimultaneity,
            Self::CameraSimultaneity => Self::PastLightCone,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::PastLightCone => "REAL CAMERA (DS2=0)",
            Self::CameraSimultaneity => "GOD'S EYE (DT'=0)",
        }
    }
}

/// Controls how 4D shadow rays are cast from surface hit events `X_hit^mu` toward the light worldline.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShadowMode {
    /// No shadow rays cast.
    Off,
    /// Physically accurate Lorentz-invariant retarded-time null shadow rays (`ds^2 = 0`):
    /// `t_emit = t_hit - |x_L - x_hit|`. Because the intersection of the past light cone with
    /// 4D worldtubes is a Minkowski geometric invariant, whether a spacetime event `X_hit` is
    /// in shadow is completely independent of the observer's velocity or camera slice mode!
    LorentzInvariant,
    /// Faked instantaneous shadow cast inside the camera's own 3D simultaneity hyperplane (`dt' = 0`):
    /// `<X_emit - X_hit, u_cam>_eta = 0  ==>  t_emit = t_hit + beta_cam . (x_L - x_hit)`.
    /// Projects shadows strictly within the 3D slice currently defined by the camera's frame.
    CameraSlice,
}

impl ShadowMode {
    pub fn next(self) -> Self {
        match self {
            Self::LorentzInvariant => Self::CameraSlice,
            Self::CameraSlice => Self::Off,
            Self::Off => Self::LorentzInvariant,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Off => "OFF",
            Self::LorentzInvariant => "LORENTZ-INVARIANT (NULL)",
            Self::CameraSlice => "CAMERA-SLICE FAKED (DT'=0)",
        }
    }
}

pub struct RenderConfig {
    pub width: usize,
    pub height: usize,
    pub mode: RenderMode,
    pub shadow_mode: ShadowMode,
    pub slice_mode: SliceMode,
}

pub fn render_frame(
    scene: &Scene4D,
    camera: &Camera4D,
    cfg: &RenderConfig,
    framebuffer: &mut [u32],
) {
    let w = cfg.width;
    let h = cfg.height;
    let aspect = w as f32 / h as f32;
    let half_h = (0.5 * camera.fov_y_rad).tan();
    let half_w = aspect * half_h;
    let lut = gamma_lut();

    framebuffer
        .par_chunks_mut(w)
        .enumerate()
        .for_each(|(py, row)| {
            let ndc_y = 1.0 - 2.0 * ((py as f32 + 0.5) / h as f32);
            let sy = ndc_y * half_h;

            for (px, pixel) in row.iter_mut().enumerate() {
                let ndc_x = 2.0 * ((px as f32 + 0.5) / w as f32) - 1.0;
                let sx = ndc_x * half_w;

                let local_dir = Vec3::new(sx, sy, 1.0).normalized();

                let ray_dir = match cfg.slice_mode {
                    SliceMode::PastLightCone => camera.tetrad.past_light_ray_direction(local_dir),
                    SliceMode::CameraSimultaneity => {
                        camera.tetrad.local_spatial_to_world(local_dir)
                    }
                };

                let color = if let Some(hit) =
                    scene.bvh.trace(camera.position, ray_dir, 1e-3, 180.0)
                {
                    shade_hit(scene, camera, cfg, ray_dir, &hit)
                } else {
                    shade_sky(ray_dir)
                };

                *pixel = pack_rgb_with_lut(color, lut);
            }
        });
}

fn shade_hit(
    scene: &Scene4D,
    camera: &Camera4D,
    cfg: &RenderConfig,
    ray_dir: Vec4,
    hit: &HitRecord4D,
) -> Vec3 {
    let mat = &scene.materials[hit.material_id];

    match cfg.mode {
        RenderMode::RelativisticShaded => {
            let albedo = sample_surface_albedo(hit, mat);
            let lighting = compute_illumination(scene, camera, cfg, ray_dir, hit, mat);
            apply_distance_fog(albedo * lighting, ray_dir, hit.lambda)
        }

        RenderMode::LookbackIsochrones => {
            // Observer's own rest-frame lookback time/distance `hit.lambda`
            let phase = (hit.lambda * 0.12).rem_euclid(1.0);
            let base = hsv_to_rgb(phase * 300.0, 0.8, 0.85);

            // Isochrone contour lines every 1.0 light-second of coordinate time t_hit
            let t_contour = (hit.event.t.rem_euclid(1.0) - 0.5).abs();
            let is_t_ring = t_contour > 0.45;

            // Proper-time contour lines every 1.0 second of local proper time tau
            let tau_contour = (hit.proper_time.rem_euclid(1.0) - 0.5).abs();
            let is_tau_ring = tau_contour > 0.46;

            if is_t_ring {
                Vec3::new(1.0, 1.0, 1.0)
            } else if is_tau_ring {
                Vec3::new(0.1, 0.1, 0.1)
            } else {
                base
            }
        }

        RenderMode::Simplex4DInspector => {
            let b = hit.tet_bary;
            let min_b = b[0].min(b[1]).min(b[2]).min(b[3]);
            let edge = if min_b < 0.035 { 0.12 } else { 1.0 };

            let speed_boost = ((hit.four_velocity.t - 1.0) * 0.8).clamp(0.0, 1.0);
            let tint = Vec3::new(
                0.25 + 0.65 * b[1] + 0.3 * speed_boost,
                0.25 + 0.65 * b[2],
                0.35 + 0.55 * b[3] * (1.0 - 0.5 * speed_boost),
            );
            tint * edge
        }
    }
}

/// Evaluate the intrinsic rest-frame procedural texture at `hit.uv` and `hit.proper_time`.
fn sample_surface_albedo(hit: &HitRecord4D, mat: &Material) -> Vec3 {
    let mut albedo = match mat.pattern {
        SurfacePattern::Checkerboard => {
            let gu = (hit.uv.0 * mat.grid_scale).floor() as i32;
            let gv = (hit.uv.1 * mat.grid_scale).floor() as i32;
            let is_even = (gu + gv).rem_euclid(2) == 0;
            let mut col = if is_even {
                mat.base_color
            } else {
                mat.checker_color
            };

            // 3D surface triangle wireframe on checkerboard meshes
            let min_tri_bary = hit.tri_bary.x.min(hit.tri_bary.y).min(hit.tri_bary.z);
            if min_tri_bary < mat.wireframe_width {
                let edge_t = (min_tri_bary / mat.wireframe_width).clamp(0.0, 1.0);
                let wire_col = Vec3::splat(0.07);
                col = wire_col * (1.0 - edge_t) + col * edge_t;
            }
            col
        }
        SurfacePattern::ClockFace => sample_clock_face_texture(hit, mat),
        SurfacePattern::LightClock => sample_light_clock_texture(hit.uv, hit.proper_time, mat),
    };

    // Subtle border frame on clock cubes
    if mat.pattern != SurfacePattern::Checkerboard {
        let u_tile = (hit.uv.0 * mat.grid_scale).rem_euclid(1.0);
        let v_tile = (hit.uv.1 * mat.grid_scale).rem_euclid(1.0);
        let border_dist = u_tile.min(1.0 - u_tile).min(v_tile.min(1.0 - v_tile));
        if border_dist < 0.025 {
            albedo = albedo * 0.35;
        }
    }

    albedo
}

/// Determine the spacetime event on the stationary light source's worldline `X_L(t_L) = (t_L, light_pos)`
/// that illuminates `hit_event` under `shadow_mode`, together with the spatial distance `|x_L - x_hit|`.
#[inline]
fn light_emission_event(
    hit_event: Vec4,
    light_pos: Vec3,
    camera: &Camera4D,
    shadow_mode: ShadowMode,
) -> (Vec4, f32) {
    let spatial_to_light = light_pos - hit_event.spatial();
    let dist_to_light = spatial_to_light.length().max(1e-4);
    let t_emit = match shadow_mode {
        ShadowMode::CameraSlice => {
            // Instantaneous shadow cast inside the camera's own 3D simultaneity slice:
            // <X_emit - X_hit, u_cam>_eta = 0  ==>  t_emit = t_hit + beta_cam . (x_L - x_hit)
            hit_event.t + camera.tetrad.beta().dot(spatial_to_light)
        }
        ShadowMode::LorentzInvariant | ShadowMode::Off => {
            // Physically accurate Lorentz-invariant retarded-time null ray (ds^2 = 0):
            // t_emit = t_hit - |x_L - x_hit|
            hit_event.t - dist_to_light
        }
    };
    (Vec4::from_time_space(t_emit, light_pos), dist_to_light)
}

/// Compute rest-frame Lambertian + view shading and cast 4D shadow rays when facing the light.
fn compute_illumination(
    scene: &Scene4D,
    camera: &Camera4D,
    cfg: &RenderConfig,
    ray_dir: Vec4,
    hit: &HitRecord4D,
    mat: &Material,
) -> f32 {
    let (light_emit_event, dist_to_light) =
        light_emission_event(hit.event, scene.light_pos, camera, cfg.shadow_mode);

    // 4D direction vector from hit event toward the light emission event (parameterized by spatial distance)
    let to_light_4d = (light_emit_event - hit.event) * (1.0 / dist_to_light);

    // Evaluate one-sided cosines in the surface's comoving rest frame via the spatial projector
    // `v_perp = v + <v, u>_eta * u` (works identically for null rays and spacelike camera-slice rays):
    let cos_light_rest =
        rest_frame_cos(to_light_4d, hit.surface_normal_4d, hit.four_velocity).max(0.0);
    // `-ray_dir` points from the hit surface back toward the observer:
    let cos_view_rest =
        (-rest_frame_cos(ray_dir, hit.surface_normal_4d, hit.four_velocity)).max(0.0);

    if mat.pattern == SurfacePattern::Checkerboard {
        // Only cast a shadow ray if the surface is front-facing with respect to the light in its rest frame
        let in_shadow = if cos_light_rest > 0.0 {
            match cfg.shadow_mode {
                ShadowMode::Off => false,
                ShadowMode::LorentzInvariant | ShadowMode::CameraSlice => scene.bvh.is_occluded(
                    hit.event,
                    to_light_4d,
                    2e-3,
                    dist_to_light - 2e-3,
                ),
            }
        } else {
            false
        };
        let shadow_factor = if in_shadow { 0.32 } else { 1.0 };
        let atten = 1.0 / (1.0 + 0.006 * dist_to_light * dist_to_light);
        mat.emission + 0.30 * cos_view_rest + 0.80 * shadow_factor * cos_light_rest * atten
    } else {
        // High-contrast self-illuminated clock / light-clock display with rest-frame 3D shading
        0.72 + 0.38 * cos_view_rest + 0.22 * cos_light_rest
    }
}

#[inline]
fn apply_distance_fog(surface_color: Vec3, ray_dir: Vec4, lambda: f32) -> Vec3 {
    let fog = (-0.008 * lambda).exp();
    let sky_grad = shade_sky_gradient(ray_dir);
    surface_color * fog + sky_grad * (1.0 - fog)
}

/// Reconstruct the **central proper time** `tau_tile_center` of the oscillator at the center
/// `(u_tile = 0.5, v_tile = 0.5)` of the clock tile containing `uv`.
///
/// Since `proper_time` at `(u, v)` is `tau_local(u, v) = tau_tile_center + Delta_u * d_tau/du + Delta_v * d_tau/dv`,
/// subtracting the linear Born-rigid offset recovers `tau_tile_center`, which is strictly uniform
/// across the entire clock tile on any comoving rest-frame simultaneity slice `Sigma(tau_c)`.
#[inline]
pub fn clock_tile_central_proper_time(
    u_tile: f32,
    v_tile: f32,
    grid_scale: f32,
    local_proper_time: f32,
    tau_uv_grad: (f32, f32),
) -> f32 {
    let inv_grid = 1.0 / grid_scale.max(1e-4);
    let delta_u = (u_tile - 0.5) * inv_grid;
    let delta_v = (v_tile - 0.5) * inv_grid;
    local_proper_time - (delta_u * tau_uv_grad.0 + delta_v * tau_uv_grad.1)
}

/// Procedural Analog + Digital Clock Face texture with:
/// 1. **Centrally-driven main dial** at `(0.50, 0.57)`: driven by a single central oscillator at the
///    clock tile center whose phase signal radiates outward from the dial pivot at speed `c = 1` in
///    the rest frame (`tau_main(r_phys) = central_proper_time - r_phys`). Because `r_phys` is purely
///    radial around the pivot, every hand is a single unbroken curve in the rest frame (curving into
///    a smooth causal spiral rather than tearing into pieces when a Born-rigid proper-time gradient
///    `Phi` is present across the face).
/// 2. **Central digital LED readout** (`+24.7S`) at the bottom center of the tile, driven by
///    `central_proper_time` so digits never tear across their height/width.
/// 3. **Four corner mini-clocks** at `(0.135, 0.865)`, `(0.865, 0.865)`, `(0.135, 0.135)`, `(0.865, 0.135)`,
///    each driven by its own local corner oscillator (`tau_corner = central_proper_time + Delta_p_corner . Phi - r_mini_phys`),
///    making any Rindler / Born-rigid proper-time drift across the face directly visible by comparing
///    the four corners!
///
/// Dial calibration:
/// - 10 decimal notches around the dial (`0..9`), so every full rotation of a faster hand advances
///   the next slower hand by **1 exact notch**, and all 3 hands match the digital `+24.7S` readout:
///   * Fast hand (Cyan):   1 full rotation =   1.0s of proper time (1 notch = 0.1s, tenths digit)
///   * Main hand (Yellow): 1 full rotation =  10.0s of proper time (1 notch = 1.0s = 1 fast-hand rev, ones digit)
///   * Slow hand (Red):    1 full rotation = 100.0s of proper time (1 notch = 10.0s = 1 main-hand rev, tens digit)
fn sample_clock_face_texture(hit: &HitRecord4D, mat: &Material) -> Vec3 {
    let grid_scale = mat.grid_scale.max(1e-4);
    let inv_grid = 1.0 / grid_scale;
    let u = (hit.uv.0 * grid_scale).rem_euclid(1.0);
    let v = (hit.uv.1 * grid_scale).rem_euclid(1.0);

    // Central proper time of the oscillator at the center (0.5, 0.5) of this clock tile:
    let central_proper_time =
        clock_tile_central_proper_time(u, v, grid_scale, hit.proper_time, hit.tau_uv_grad);

    // 1. Four corner mini-clocks showing the local proper time at each corner of the tile:
    const CORNER_CENTERS: [(f32, f32); 4] = [
        (0.135, 0.865),
        (0.865, 0.865),
        (0.135, 0.135),
        (0.865, 0.135),
    ];
    for &(mc_x, mc_y) in &CORNER_CENTERS {
        let mdx = u - mc_x;
        let mdy = v - mc_y;
        let mr = (mdx * mdx + mdy * mdy).sqrt();
        if mr <= 0.095 {
            // Proper time of the corner oscillator at (mc_x, mc_y), plus radial signal delay -mr_phys / c:
            let corner_osc_tau = central_proper_time
                + ((mc_x - 0.5) * inv_grid) * hit.tau_uv_grad.0
                + ((mc_y - 0.5) * inv_grid) * hit.tau_uv_grad.1;
            let mdx_phys = (mdx * inv_grid) * hit.uv_world_scale.0;
            let mdy_phys = (mdy * inv_grid) * hit.uv_world_scale.1;
            let mr_phys = (mdx_phys * mdx_phys + mdy_phys * mdy_phys).sqrt();
            return sample_corner_mini_clock(mdx, mdy, mr, corner_osc_tau - mr_phys, mat);
        }
    }

    // 2. Digital proper-time display panel at bottom-center of clock tile: u in [0.25, 0.75], v in [0.035, 0.205]
    if (0.035..=0.205).contains(&v) && (0.25..=0.75).contains(&u) {
        if !(0.050..=0.190).contains(&v) || !(0.265..=0.735).contains(&u) {
            return mat.base_color * 0.9; // Panel bezel
        }
        if (0.06..=0.18).contains(&v)
            && (0.28..=0.72).contains(&u)
            && sample_digital_tau(u, v, central_proper_time)
        {
            return Vec3::new(1.0, 0.96, 0.30); // Bright high-contrast LED readout
        }
        return Vec3::new(0.03, 0.04, 0.07); // Deep black digital panel background
    }

    // 3. Main analog clock dial centered at (0.50, 0.57) with outer bezel radius 0.330
    let cx = 0.50;
    let cy = 0.57;
    let dx = u - cx;
    let dy = v - cy;
    let r = (dx * dx + dy * dy).sqrt();

    if r > 0.330 {
        // Exterior casing corner checkerboard
        let gu = (u * 4.0).floor() as i32;
        let gv = (v * 4.0).floor() as i32;
        return if (gu + gv).rem_euclid(2) == 0 {
            mat.base_color * 0.78
        } else {
            mat.checker_color * 0.78
        };
    }

    if r > 0.292 {
        // Colored metallic bezel ring
        return mat.base_color * 1.15;
    }

    // Physical rest-frame radial distance r_phys (in light-seconds) from the dial pivot (cx, cy).
    // A radio/phase signal emitted by the central oscillator at the pivot propagates radially outward
    // at speed c = 1 in the clock's comoving rest frame, so a pixel at radius r_phys displays the
    // retarded phase `dial_tau = central_proper_time - r_phys`.
    let dx_phys = (dx * inv_grid) * hit.uv_world_scale.0;
    let dy_phys = (dy * inv_grid) * hit.uv_world_scale.1;
    let r_phys = (dx_phys * dx_phys + dy_phys * dy_phys).sqrt();
    let dial_tau = central_proper_time - r_phys;

    // Inside the circular clock face (r <= 0.292):
    // Clockwise angle from top (+Y, notch 0):
    let angle = dx.atan2(dy).rem_euclid(std::f32::consts::TAU);

    // Main hand: 1 full revolution = 10.0s of proper time (1 notch = 1.0s)
    let main_hand_angle =
        ((dial_tau / 10.0) * std::f32::consts::TAU).rem_euclid(std::f32::consts::TAU);

    let mut dial_col = Vec3::new(0.04, 0.06, 0.10);
    if angle <= main_hand_angle && r < 0.236 {
        dial_col += mat.base_color * 0.22;
    }

    // 10 Notches around the rim (each notch = 1 full rotation of the 1.0s fast hand)
    if (0.220..=0.286).contains(&r) {
        let sector = angle / (std::f32::consts::TAU / 10.0);
        let dist_to_tick = (sector - sector.round()).abs();
        let tick_idx = sector.round() as i32 % 10;
        let is_cardinal = tick_idx == 0 || tick_idx == 5;
        let half_w = if is_cardinal { 0.105 } else { 0.055 };
        let inner_r = if is_cardinal { 0.218 } else { 0.242 };
        if dist_to_tick < half_w && r >= inner_r {
            return if tick_idx == 0 {
                Vec3::new(1.0, 0.88, 0.15) // Gold top (0) marker
            } else if is_cardinal {
                Vec3::new(1.0, 1.0, 1.0) // White bottom (5) marker
            } else {
                Vec3::new(0.75, 0.80, 0.88)
            };
        }
    }

    // Fast sub-second hand: 1 full rotation = 1.0s of proper time (advances main hand by 1 notch per turn!)
    let fast_angle = ((dial_tau / 1.0) * std::f32::consts::TAU).rem_euclid(std::f32::consts::TAU);
    if point_on_hand(dx, dy, fast_angle, 0.254, 0.012) {
        return Vec3::new(0.25, 0.98, 1.0);
    }

    // Main second hand: 1 full rotation = 10.0s of proper time (1 notch = 1.0s, advances slow hand by 1 notch per turn!)
    if point_on_hand(dx, dy, main_hand_angle, 0.228, 0.022) {
        return Vec3::new(1.0, 0.95, 0.30);
    }
    if point_on_hand(dx, dy, main_hand_angle, 0.236, 0.030) {
        return Vec3::new(0.02, 0.02, 0.04);
    }

    // Slow hand: 1 full rotation = 100.0s of proper time (1 notch = 10.0s)
    let slow_angle = ((dial_tau / 100.0) * std::f32::consts::TAU).rem_euclid(std::f32::consts::TAU);
    if point_on_hand(dx, dy, slow_angle, 0.162, 0.026) {
        return Vec3::new(1.0, 0.35, 0.25);
    }

    // Center pivot cap
    if r < 0.034 {
        return Vec3::new(1.0, 1.0, 1.0);
    }

    dial_col
}

/// Render one of the 4 corner reference mini-clocks (`mr <= 0.095`) driven by its local corner
/// proper time `mini_tau`. Comparing the 4 corner dials reveals any Born-rigid / Rindler proper-time
/// gradient (`Delta_tau = Delta_p . Phi`) across the clock face while keeping every dial intact.
#[inline]
fn sample_corner_mini_clock(mdx: f32, mdy: f32, mr: f32, mini_tau: f32, mat: &Material) -> Vec3 {
    if mr > 0.080 {
        return mat.base_color * 1.15; // Metallic mini-bezel
    }

    let angle = mdx.atan2(mdy).rem_euclid(std::f32::consts::TAU);
    let fast_angle = ((mini_tau / 1.0) * std::f32::consts::TAU).rem_euclid(std::f32::consts::TAU);
    let main_angle = ((mini_tau / 10.0) * std::f32::consts::TAU).rem_euclid(std::f32::consts::TAU);

    let mut col = Vec3::new(0.04, 0.06, 0.10);
    // Subtle swept sector up to the 1.0s fast hand so sub-second phase offsets across corners jump out
    if angle <= fast_angle && mr < 0.066 {
        col += mat.base_color * 0.25;
    }

    // 10 rim ticks around the mini-dial
    if (0.062..=0.078).contains(&mr) {
        let sector = angle / (std::f32::consts::TAU / 10.0);
        let dist_to_tick = (sector - sector.round()).abs();
        let tick_idx = sector.round() as i32 % 10;
        let is_cardinal = tick_idx == 0 || tick_idx == 5;
        let half_w = if is_cardinal { 0.13 } else { 0.07 };
        if dist_to_tick < half_w {
            return if tick_idx == 0 {
                Vec3::new(1.0, 0.88, 0.15)
            } else {
                Vec3::new(0.80, 0.85, 0.92)
            };
        }
    }

    // Fast hand (1.0s per revolution, cyan)
    if point_on_hand_mini(mdx, mdy, fast_angle, 0.068, 0.0065) {
        return Vec3::new(0.25, 0.98, 1.0);
    }

    // Main second hand (10.0s per revolution, yellow)
    if point_on_hand_mini(mdx, mdy, main_angle, 0.052, 0.0085) {
        return Vec3::new(1.0, 0.95, 0.30);
    }

    // Mini pivot cap
    if mr < 0.012 {
        return Vec3::new(1.0, 1.0, 1.0);
    }

    col
}

#[inline]
fn point_on_hand(dx: f32, dy: f32, angle_cw: f32, length: f32, half_width: f32) -> bool {
    let (hx, hy) = angle_cw.sin_cos();
    let proj_along = dx * hx + dy * hy;
    let proj_perp = (dx * hy - dy * hx).abs();
    proj_along >= -0.025 && proj_along <= length && proj_perp <= half_width
}

#[inline]
fn point_on_hand_mini(dx: f32, dy: f32, angle_cw: f32, length: f32, half_width: f32) -> bool {
    let (hx, hy) = angle_cw.sin_cos();
    let proj_along = dx * hx + dy * hy;
    let proj_perp = (dx * hy - dy * hx).abs();
    proj_along >= -0.008 && proj_along <= length && proj_perp <= half_width
}

/// Sample a 6-character 5x7 digital proper-time readout (`+12.4S`) inside `u in [0.28, 0.72], v in [0.06, 0.18]`.
fn sample_digital_tau(u: f32, v: f32, proper_time: f32) -> bool {
    let rel_u = (u - 0.28) / (0.72 - 0.28); // [0, 1]
    let rel_v = (0.18 - v) / (0.18 - 0.06); // [0, 1] top-to-bottom

    let total_cols = 35usize; // 6 chars * 6 cols - 1
    let col = (rel_u * total_cols as f32).floor() as usize;
    let row = (rel_v * 7.0).floor() as usize;
    if col >= total_cols || row >= 7 {
        return false;
    }

    let char_idx = col / 6;
    let char_col = col % 6;
    if char_idx >= 6 || char_col >= 5 {
        return false;
    }

    let t_tenths = (proper_time * 10.0).round() as i32;
    let sign_ch = if t_tenths < 0 { '-' } else { '+' };
    let abs_t = t_tenths.unsigned_abs();
    let tenths = (abs_t % 10) as u8;
    let ones = ((abs_t / 10) % 10) as u8;
    let tens = ((abs_t / 100) % 10) as u8;

    let ch = match char_idx {
        0 => sign_ch,
        1 => (b'0' + tens) as char,
        2 => (b'0' + ones) as char,
        3 => '.',
        4 => (b'0' + tenths) as char,
        _ => 'S',
    };

    let glyph = glyph_5x7(ch);
    (glyph[row] >> (4 - char_col)) & 1 != 0
}

/// Fractional bounce position in `[0.0, 1.0]` (`0.0` = bottom mirror, `1.0` = top mirror)
/// for an Einstein light clock with rest-frame mirror separation `cavity_length` (`L`).
/// Because the photon moves at `c = 1` in the rest frame, one-way travel takes `L` seconds
/// of proper time and the round-trip period is `T = 2L/c = 2.0 * cavity_length`.
#[inline]
pub fn light_clock_bounce_fraction(proper_time: f32, cavity_length: f32) -> f32 {
    let period = (2.0 * cavity_length).max(1e-4);
    let cycle = (proper_time / period).rem_euclid(1.0);
    if cycle < 0.5 {
        cycle * 2.0
    } else {
        (1.0 - cycle) * 2.0
    }
}

/// Einstein Bouncing-Photon Light Clock texture:
/// A photon wavepacket bounces vertically between bottom mirror (`v = 0.10`) and top mirror (`v = 0.90`)
/// (spanning `0.80` of the box height). With `he.y = 0.625` (box height `H = 1.25m`), the physical
/// rest-frame mirror separation is `L = 0.80 * 1.25 = 1.00` light-second (`mat.cavity_length = 1.0`),
/// so the photon moves at exact speed `1.00c` in its rest frame and bounces once per `1.0s` of `tau`.
fn sample_light_clock_texture(uv: (f32, f32), proper_time: f32, mat: &Material) -> Vec3 {
    let u = (uv.0 * mat.grid_scale).rem_euclid(1.0);
    let v = (uv.1 * mat.grid_scale).rem_euclid(1.0);

    // Side structural frame
    if !(0.12..=0.88).contains(&u) {
        return mat.base_color * 0.65;
    }

    // Top and bottom reflective mirror plates (inner reflecting surfaces at v = 0.10 and v = 0.90)
    if (0.04..=0.10).contains(&v) || (0.90..=0.96).contains(&v) {
        return Vec3::new(0.75, 0.90, 0.98);
    }
    if !(0.10..=0.90).contains(&v) {
        return Vec3::new(0.08, 0.12, 0.16);
    }

    // Photon position in [0.10, 0.90] with period T = 2L/c = 2.0 * mat.cavity_length
    let bounce = light_clock_bounce_fraction(proper_time, mat.cavity_length);
    let photon_v = 0.10 + 0.80 * bounce;

    let dv = (v - photon_v).abs();
    let du = (u - 0.5).abs();
    let dist = (du * du * 0.7 + dv * dv).sqrt();

    let mut col = Vec3::new(0.05, 0.09, 0.13);

    // Calibration ruler ticks along the cavity
    let ruler_tick = ((v - 0.10) / 0.80 * 8.0).rem_euclid(1.0);
    if (u < 0.22 || u > 0.78) && ruler_tick < 0.08 {
        col += Vec3::new(0.20, 0.35, 0.35);
    }

    // Center guide line
    if du < 0.015 {
        col += Vec3::new(0.06, 0.16, 0.16);
    }

    // Glowing horizontal laser wavefront + bright photon core
    if dv < 0.032 {
        let wave_intensity = (1.0 - dv / 0.032).powi(2);
        col += mat.base_color * (0.9 * wave_intensity);
    }
    if dist < 0.11 {
        let core = (1.0 - dist / 0.11).powi(2);
        col += Vec3::new(1.0, 1.0, 0.85) * (1.4 * core);
    }

    col
}

/// Smooth horizon-to-zenith sky gradient (used for distance fog without trig calls).
#[inline]
fn shade_sky_gradient(ray_dir: Vec4) -> Vec3 {
    let s = ray_dir.spatial().normalized();
    let horizon = (s.y * 0.5 + 0.5).clamp(0.0, 1.0);
    let zenith_col = Vec3::new(0.04, 0.06, 0.13);
    let horizon_col = Vec3::new(0.12, 0.16, 0.25);
    horizon_col * (1.0 - horizon) + zenith_col * horizon
}

fn shade_sky(ray_dir: Vec4) -> Vec3 {
    // Normalize spatial part of ray direction in lab frame so the background star/grid
    // inherently reflects relativistic aberration when the camera boosts!
    let s = ray_dir.spatial().normalized();
    let mut sky = shade_sky_gradient(ray_dir);

    // Celestial coordinate grid on the sky sphere (makes relativistic aberration of the sky visible!)
    let theta = s.y.clamp(-1.0, 1.0).acos(); // [0, pi]
    let phi = s.z.atan2(s.x); // [-pi, pi]
    let grid_u = (phi / std::f32::consts::PI * 12.0).rem_euclid(1.0);
    let grid_v = (theta / std::f32::consts::PI * 6.0).rem_euclid(1.0);
    if grid_u < 0.025 || grid_v < 0.025 {
        sky += Vec3::new(0.06, 0.09, 0.14);
    }
    sky
}

fn hsv_to_rgb(h_deg: f32, s: f32, v: f32) -> Vec3 {
    let c = v * s;
    let h = (h_deg / 60.0).rem_euclid(6.0);
    let x = c * (1.0 - (h.rem_euclid(2.0) - 1.0).abs());
    let m = v - c;
    let (r, g, b) = if h < 1.0 {
        (c, x, 0.0)
    } else if h < 2.0 {
        (x, c, 0.0)
    } else if h < 3.0 {
        (0.0, c, x)
    } else if h < 4.0 {
        (0.0, x, c)
    } else if h < 5.0 {
        (x, 0.0, c)
    } else {
        (c, 0.0, x)
    };
    Vec3::new(r + m, g + m, b + m)
}

pub(crate) const GAMMA_LUT_SIZE: usize = 4096;

pub(crate) fn gamma_lut() -> &'static [u8; GAMMA_LUT_SIZE] {
    static LUT: OnceLock<[u8; GAMMA_LUT_SIZE]> = OnceLock::new();
    LUT.get_or_init(|| {
        let mut table = [0u8; GAMMA_LUT_SIZE];
        for (i, slot) in table.iter_mut().enumerate() {
            let x = (i as f32) / ((GAMMA_LUT_SIZE - 1) as f32);
            *slot = (x.powf(1.0 / 2.2) * 255.0 + 0.5).clamp(0.0, 255.0) as u8;
        }
        table
    })
}

#[inline]
fn pack_rgb_with_lut(c: Vec3, lut: &[u8; GAMMA_LUT_SIZE]) -> u32 {
    let mapped = Vec3::new(
        c.x / (1.0 + c.x * 0.35),
        c.y / (1.0 + c.y * 0.35),
        c.z / (1.0 + c.z * 0.35),
    )
    .clamp(0.0, 1.0);
    let scale = (GAMMA_LUT_SIZE - 1) as f32;
    let r = lut[(mapped.x * scale) as usize] as u32;
    let g = lut[(mapped.y * scale) as usize] as u32;
    let b = lut[(mapped.z * scale) as usize] as u32;
    (r << 16) | (g << 8) | b
}

#[inline]
pub fn pack_rgb(c: Vec3) -> u32 {
    pack_rgb_with_lut(c, gamma_lut())
}
