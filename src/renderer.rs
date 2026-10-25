use crate::hud::glyph_5x7;
use crate::mesh4d::{HitRecord4D, Material, SurfacePattern, DECIDED_AT_STATIC};
use crate::minkowski::{rest_frame_cos, Tetrad, Vec3, Vec4};
use crate::scene::Scene4D;
#[cfg(not(target_arch = "wasm32"))]
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
    /// Causal decision age: how much camera proper time has passed since each visible event
    /// left the camera's open future light cone `I+(O)` and could no longer be influenced.
    DecisionAge,
}

impl RenderMode {
    pub fn next(self) -> Self {
        match self {
            Self::RelativisticShaded => Self::LookbackIsochrones,
            Self::LookbackIsochrones => Self::Simplex4DInspector,
            Self::Simplex4DInspector => Self::DecisionAge,
            Self::DecisionAge => Self::RelativisticShaded,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::RelativisticShaded => "4D LIGHT-CONE SHADED",
            Self::LookbackIsochrones => "PAST LIGHT-CONE ISOCHRONES",
            Self::Simplex4DInspector => "4D TETRAHEDRAL MESH",
            Self::DecisionAge => "CAUSAL DECISION AGE",
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
    /// In a match, the parts of the slice inside another player's future cone are not decided
    /// yet and are drawn as undetermined haze (see [`FrameDynamics::remote_origins`]).
    CameraSimultaneity,
    /// Causal frontier: rays run down the *future* null cone `k^mu = +e_0 + n^a e_a` from
    /// `O' = O - FrameDynamics::frontier_view_offset() * e_0`, landing just outside `I+(O)` (and,
    /// in Dynamic Mode, just behind every body's finalized tip): the world as it is
    /// being decided, and where the camera's current choices are about to propagate.
    Frontier,
    /// Prophecy slice: the God's-eye slice pushed into the camera's future by
    /// `RenderConfig::prophecy_delta`. The ball of radius `delta` around the camera on that
    /// slice lies inside `I+(O)` and is drawn as undetermined haze.
    ProphecySlice,
}

/// Backwards-compatible alias for [`ViewMode`].
pub type SliceMode = ViewMode;

impl ViewMode {
    pub fn next(self) -> Self {
        match self {
            Self::PastLightCone => Self::CameraSimultaneity,
            Self::CameraSimultaneity => Self::Frontier,
            Self::Frontier => Self::ProphecySlice,
            Self::ProphecySlice => Self::PastLightCone,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::PastLightCone => "REAL CAMERA (DS2=0)",
            Self::CameraSimultaneity => "GOD'S EYE (DT'=0)",
            Self::Frontier => "FRONTIER (FUTURE CONE)",
            Self::ProphecySlice => "PROPHECY (DT'=+DELTA)",
        }
    }
}

/// Proper-time offset `delta` of the Frontier view's ray origin `O' = O - delta * e_0`. Every
/// future-cone ray from `O'` beyond `lambda > delta / 2` lies outside `I+(O)`, so rays land on
/// geometry that is already decided instead of grazing the frontier.
pub const FRONTIER_RAY_OFFSET: f32 = 0.14;

/// Range and step of the Prophecy slice's future offset (`PageUp` / `PageDown`).
pub const PROPHECY_DELTA_DEFAULT: f32 = 3.5;
pub const PROPHECY_DELTA_MIN: f32 = 0.5;
pub const PROPHECY_DELTA_MAX: f32 = 20.0;
pub const PROPHECY_DELTA_STEP: f32 = 0.5;

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
    /// Future proper-time offset of [`ViewMode::ProphecySlice`]
    pub prophecy_delta: f32,
}

impl RenderConfig {
    /// Move the prophecy slice `steps` increments further into (or back from) the future.
    pub fn step_prophecy_delta(&mut self, steps: f32) {
        self.prophecy_delta = (self.prophecy_delta + steps * PROPHECY_DELTA_STEP)
            .clamp(PROPHECY_DELTA_MIN, PROPHECY_DELTA_MAX);
    }
}

/// Maximum number of light-pulse flashes uploaded to the GPU uniform block per frame.
pub const MAX_FLASHES: usize = 16;

/// A spherical pulse of light emitted at spacetime event `at`. It illuminates any surface event
/// `E` on its future light cone within shell thickness `width`:
/// `|(E.t - at.t) - |E.x - at.x|| < 0.5 * width`.
#[derive(Clone, Copy, Debug)]
pub struct FlashEvent {
    pub at: Vec4,
    pub width: f32,
    pub intensity: f32,
    pub color: Vec3,
}

/// The causal frontier `dJ+(O)` of the player's ship in Dynamic Mode: everything outside the
/// open future light cone of `origin` is decided.
#[derive(Clone, Copy, Debug)]
pub struct Frontier {
    /// The frontier's apex event `O`.
    pub origin: Vec4,
    /// Ship proper time at `O`.
    pub proper_time: f32,
}

/// Per-frame render inputs that change independently of a scene's (static) geometry, so they
/// are passed alongside the camera rather than baked into scene buffers.
#[derive(Clone, Debug, Default)]
pub struct FrameDynamics {
    /// `Some` while a Dynamic Mode scene is running.
    pub frontier: Option<Frontier>,
    /// Active light pulses; only the first [`MAX_FLASHES`] are rendered.
    pub flashes: Vec<FlashEvent>,
    /// How far (in camera proper time) the simulated world's finalized tips trail the frontier.
    /// Bodies are only finalized once *all* of them is decided, so the Frontier view looks back
    /// this much further to see them. Zero for static scenes.
    pub frontier_lag: f32,
    /// In a match: the latest known events of the other live players' ships. Everything in
    /// their open future cones is still undecided too, even outside the camera's own; only
    /// the first [`MAX_REMOTE_FRONTIERS`] are rendered.
    pub remote_origins: Vec<Vec4>,
}

/// Maximum number of other players' frontiers uploaded to the GPU uniform block per frame.
pub const MAX_REMOTE_FRONTIERS: usize = 4;

/// Maximum number of cones a view draws as undetermined haze: the camera's and the others'.
pub const MAX_HAZE_ORIGINS: usize = MAX_REMOTE_FRONTIERS + 1;

/// Flashes are dropped once their shell is this far (light-seconds) past the emission event:
/// beyond the renderers' maximum ray length, nothing they light can be seen any more.
pub const FLASH_MAX_RADIUS: f32 = 200.0;

/// Color of the flash a laser hit lights.
pub const HIT_FLASH_COLOR: Vec3 = Vec3::new(1.0, 0.35, 0.12);

impl FrameDynamics {
    /// Proper-time offset of the Frontier view's ray origin behind the camera event.
    pub fn frontier_view_offset(&self) -> f32 {
        FRONTIER_RAY_OFFSET + self.frontier_lag
    }

    /// The flashes the renderers actually evaluate (both backends cap at [`MAX_FLASHES`]).
    pub fn active_flashes(&self) -> &[FlashEvent] {
        &self.flashes[..self.flashes.len().min(MAX_FLASHES)]
    }

    /// The other players' frontiers the renderers actually evaluate (both backends cap at
    /// [`MAX_REMOTE_FRONTIERS`]).
    pub fn active_remote_origins(&self) -> &[Vec4] {
        &self.remote_origins[..self.remote_origins.len().min(MAX_REMOTE_FRONTIERS)]
    }

    /// The apexes of the cones whose insides a view draws as undetermined haze. The Prophecy
    /// slice lies in the camera's future, so its own cone counts there. The God's-eye slice
    /// lies outside the camera's cone, so only the other players' can reach it.
    pub fn haze_origins(&self, camera: &Camera4D, slice_mode: SliceMode) -> Vec<Vec4> {
        let remote = self.active_remote_origins().iter().copied();
        match slice_mode {
            SliceMode::ProphecySlice => {
                std::iter::once(self.frontier_or_camera(camera).origin).chain(remote).collect()
            }
            SliceMode::CameraSimultaneity => remote.collect(),
            SliceMode::PastLightCone | SliceMode::Frontier => Vec::new(),
        }
    }

    /// The decision frontier the renderers measure against: the Dynamic Mode frontier if one
    /// is running, else the camera event itself (static scenes, inertial-camera assumption).
    pub fn frontier_or_camera(&self, camera: &Camera4D) -> Frontier {
        self.frontier.unwrap_or(Frontier {
            origin: camera.position,
            proper_time: camera.proper_time,
        })
    }

    /// Emit the player's light flash at event `at`, evicting the oldest if all slots are full.
    pub fn emit_flash(&mut self, at: Vec4) {
        self.push_flash(FlashEvent {
            at,
            width: 0.8,
            intensity: 2.5,
            color: Vec3::new(1.0, 0.9, 0.7),
        });
    }

    /// A laser hit on a ship's hull at `at`: a short orange-red flash.
    pub fn emit_hit_flash(&mut self, at: Vec4) {
        self.push_flash(FlashEvent {
            at,
            width: 0.5,
            intensity: 3.0,
            color: HIT_FLASH_COLOR,
        });
    }

    fn push_flash(&mut self, flash: FlashEvent) {
        if self.flashes.len() >= MAX_FLASHES {
            self.flashes.remove(0);
        }
        self.flashes.push(flash);
    }

    /// Drop flashes whose shell has expanded beyond [`FLASH_MAX_RADIUS`] by lab time `now_t`.
    pub fn retire_flashes(&mut self, now_t: f32) {
        self.flashes.retain(|f| now_t - f.at.t < FLASH_MAX_RADIUS);
    }

    /// Shift every flash by `dt` in lab time, e.g. when a periodic static scene wraps its clock,
    /// so the flash stays attached to the same moment of the loop.
    pub fn shift_time(&mut self, dt: f32) {
        for f in &mut self.flashes {
            f.at.t += dt;
        }
    }
}

/// Camera proper time elapsed since `event` left the open future light cone `I+` of an
/// inertial worldline through `origin` with 4-velocity `u`: `T + R`, where `T` is the
/// rest-frame time since the event and `R` its rest-frame distance. On the past light cone this
/// is `2R`: anything you see was settled a full round trip ago. Negative for events still
/// inside `I+(origin)` (not yet decided).
pub fn analytic_decision_age(event: Vec4, origin: Vec4, u: Vec4) -> f32 {
    let d = event - origin;
    let time_ago = d.minkowski_dot(u);
    // `d = -T u + w` with `w` spacelike and orthogonal to `u`, so `T^2 + d.d = |w|^2 = R^2`.
    let dist = (time_ago * time_ago + d.minkowski_norm_sq()).max(0.0).sqrt();
    time_ago + dist
}

pub fn render_frame(
    scene: &Scene4D,
    camera: &Camera4D,
    cfg: &RenderConfig,
    dynamics: &FrameDynamics,
    framebuffer: &mut [u32],
) {
    let w = cfg.width;
    let h = cfg.height;
    let aspect = w as f32 / h as f32;
    let half_h = (0.5 * camera.fov_y_rad).tan();
    let half_w = aspect * half_h;
    let lut = gamma_lut();
    let haze_origins = dynamics.haze_origins(camera, cfg.slice_mode);

    let ray_origin = match cfg.slice_mode {
        SliceMode::PastLightCone | SliceMode::CameraSimultaneity => camera.position,
        SliceMode::Frontier => {
            camera.position - camera.tetrad.e[0] * dynamics.frontier_view_offset()
        }
        SliceMode::ProphecySlice => camera.position + camera.tetrad.e[0] * cfg.prophecy_delta,
    };

    #[cfg(not(target_arch = "wasm32"))]
    let rows = framebuffer.par_chunks_mut(w);
    #[cfg(target_arch = "wasm32")]
    let rows = framebuffer.chunks_mut(w);

    rows
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
                    SliceMode::CameraSimultaneity | SliceMode::ProphecySlice => {
                        camera.tetrad.local_spatial_to_world(local_dir)
                    }
                    SliceMode::Frontier => camera.tetrad.future_light_ray_direction(local_dir),
                };

                let color = if let Some(hit) = scene.bvh.trace(ray_origin, ray_dir, 1e-3, 180.0)
                {
                    let shaded = shade_hit(scene, camera, cfg, dynamics, ray_dir, &hit);
                    if haze_origins.is_empty() {
                        shaded
                    } else {
                        apply_undecided_haze(shaded, hit.event, &haze_origins)
                    }
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
    dynamics: &FrameDynamics,
    ray_dir: Vec4,
    hit: &HitRecord4D,
) -> Vec3 {
    let mat = &scene.materials[hit.material_id];

    match cfg.mode {
        RenderMode::RelativisticShaded => {
            let albedo = sample_surface_albedo(hit, mat);
            let lighting = compute_illumination(scene, camera, cfg, ray_dir, hit, mat);
            let mut lit = albedo * lighting;
            let flashes = dynamics.active_flashes();
            if !flashes.is_empty() {
                let flash_rgb = compute_flash_illumination(scene, cfg, flashes, hit);
                lit += (albedo * 0.65 + Vec3::splat(0.35)) * flash_rgb;
            }
            apply_distance_fog(lit, ray_dir, hit.lambda)
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

        RenderMode::DecisionAge => {
            let frontier = dynamics.frontier_or_camera(camera);
            let age = if hit.decided_at > DECIDED_AT_STATIC * 0.5 {
                frontier.proper_time - hit.decided_at
            } else {
                analytic_decision_age(hit.event, frontier.origin, camera.tetrad.e[0])
            };
            let cos_view =
                (-rest_frame_cos(ray_dir, hit.surface_normal_4d, hit.four_velocity)).max(0.0);
            apply_distance_fog(
                decision_age_color(age) * (0.55 + 0.45 * cos_view),
                ray_dir,
                hit.lambda,
            )
        }
    }
}

/// Undecided events (`age < 0`) are dark slate; decided ones run red -> yellow -> green -> blue
/// over 60 s (a 30 ls round trip on the past cone), glow gold-white for the first 0.45 s after
/// being decided, and get a contour every 2 s.
fn decision_age_color(age: f32) -> Vec3 {
    if age < 0.0 {
        return Vec3::new(0.14, 0.14, 0.18);
    }
    let mut col = hsv_to_rgb((age * 4.0).min(240.0), 0.82, 0.88);
    if age < 0.45 {
        let f = 1.0 - age / 0.45;
        col = col * (1.0 - f * 0.7) + Vec3::new(1.0, 0.96, 0.65) * (f * 0.9);
    }
    if ((age * 0.5).rem_euclid(1.0) - 0.5).abs() > 0.465 {
        col = Vec3::new(0.95, 0.98, 1.0);
    }
    col
}

const HAZE_COLOR: Vec3 = Vec3::new(0.32, 0.18, 0.62);

/// Prophecy and God's-eye overlay: events inside the open future light cone `I+(O)` of any of
/// `origins` (the frontier, and the other players' in a match) are still undetermined and get
/// an animated violet haze; a cyan shell marks where geometry crosses the boundary of their union.
fn apply_undecided_haze(base: Vec3, event: Vec4, origins: &[Vec4]) -> Vec3 {
    const SHELL: f32 = 0.18;
    // > 0 inside some `I+(O)`, 0 on the frontier (sign is frame-independent; magnitude is
    // lab-frame).
    let inside = origins
        .iter()
        .map(|&o| {
            let d = event - o;
            d.t - d.spatial().length()
        })
        .fold(f32::NEG_INFINITY, f32::max);
    if inside <= -SHELL {
        return base;
    }
    let mut col = base;
    if inside.abs() < SHELL {
        let k = 1.0 - inside.abs() / SHELL;
        col += Vec3::new(0.45, 0.92, 1.0) * (0.85 * k * k);
    }
    if inside > 0.0 {
        let ripple = 0.5
            + 0.5 * (event.x * 3.5 + event.z * 3.5 - event.t * 4.0).sin() * (inside * 5.0).cos();
        let haze = HAZE_COLOR * (1.0 - 0.35 * ripple) + Vec3::new(0.18, 0.72, 0.92) * (0.35 * ripple);
        let alpha = 0.55 + 0.25 * (inside * 0.5).min(1.0);
        col = col * (1.0 - alpha) + haze * alpha;
    }
    col
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

/// Retarded illumination from light-pulse flashes: the hit event `E` is lit by flash `F` when
/// `F` lies on `E`'s past light cone within the pulse's shell width. Shadow rays follow the
/// null segment back to `F` (Lorentz-invariant in every shadow mode except `Off`).
fn compute_flash_illumination(
    scene: &Scene4D,
    cfg: &RenderConfig,
    flashes: &[FlashEvent],
    hit: &HitRecord4D,
) -> Vec3 {
    let mut total = Vec3::default();
    for f in flashes {
        let dt = hit.event.t - f.at.t;
        if dt <= 0.0 {
            continue;
        }
        let dist = (f.at.spatial() - hit.event.spatial()).length().max(1e-4);
        let shell_err = (dt - dist).abs();
        let half_w = (0.5 * f.width).max(1e-3);
        if shell_err >= half_w {
            continue;
        }
        let k = 1.0 - shell_err / half_w;
        let pulse = k * k;
        let to_flash_4d = (f.at - hit.event) * (1.0 / dist);
        let cos_f = rest_frame_cos(to_flash_4d, hit.surface_normal_4d, hit.four_velocity).max(0.0);
        let in_shadow = cos_f > 0.0
            && cfg.shadow_mode != ShadowMode::Off
            && scene.bvh.is_occluded(hit.event, to_flash_4d, 2e-3, dist - 2e-3);
        let shadow_factor = if in_shadow { 0.25 } else { 1.0 };
        let atten = 1.0 / (1.0 + 0.015 * dist * dist);
        total += f.color * (f.intensity * pulse * atten * (0.25 + 0.75 * shadow_factor * cos_f));
    }
    total
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn analytic_decision_age_is_time_ago_plus_distance() {
        let origin = Vec4::new(10.0, 1.0, 2.0, 3.0);
        let u = Tetrad::rest().e[0];
        let at = |dt: f32, r: f32| origin + Vec4::new(dt, r, 0.0, 0.0);
        // Past event 3 s ago, 4 ls away: settled 7 s ago.
        assert!((analytic_decision_age(at(-3.0, 4.0), origin, u) - 7.0).abs() < 1e-5);
        // Spacelike future event: decided once the camera came within its past cone.
        assert!((analytic_decision_age(at(1.0, 3.0), origin, u) - 2.0).abs() < 1e-5);
        // On the frontier itself: decided just now.
        assert!(analytic_decision_age(at(2.0, 2.0), origin, u).abs() < 1e-5);
        // Inside the open future cone: not decided yet.
        assert!(analytic_decision_age(at(3.0, 1.0), origin, u) < 0.0);
    }

    /// For a moving camera, anything seen on its past light cone at rest-frame distance `R` was
    /// decided a round trip `2R` of the camera's own proper time ago.
    #[test]
    fn analytic_decision_age_on_past_cone_is_round_trip() {
        let mut tetrad = Tetrad::rest();
        tetrad.boost_local(Vec3::new(0.4, -0.2, 0.7));
        tetrad.rotate_local(0.3, 0.1, -0.2);
        let origin = Vec4::new(5.0, -1.0, 0.5, 2.0);
        for (dir, lambda) in [(Vec3::new(0.0, 0.0, 1.0), 3.0), (Vec3::new(1.0, -0.5, 0.2), 7.5)] {
            let event = origin + tetrad.past_light_ray_direction(dir) * lambda;
            let age = analytic_decision_age(event, origin, tetrad.e[0]);
            assert!((age - 2.0 * lambda).abs() < 1e-3, "age {age} for lambda {lambda}");
        }
    }

    #[test]
    fn flash_slots_evict_retire_and_shift() {
        let mut d = FrameDynamics::default();
        for i in 0..(MAX_FLASHES + 3) {
            d.emit_flash(Vec4::new(i as f32, 0.0, 0.0, 0.0));
        }
        assert_eq!(d.flashes.len(), MAX_FLASHES);
        assert_eq!(d.flashes[0].at.t, 3.0, "oldest flashes are evicted first");

        d.shift_time(-100.0);
        assert_eq!(d.flashes[0].at.t, -97.0);

        // Flashes now sit at t = -97 ..= -82; at now = 110 only those with t > -90 survive.
        d.retire_flashes(110.0);
        assert!(d.flashes.iter().all(|f| 110.0 - f.at.t < FLASH_MAX_RADIUS));
        assert_eq!(d.flashes.len(), 8);
    }

    /// In a match the God's-eye slice is hazed inside the other player's future cone only;
    /// the Prophecy slice inside the camera's too. Single player has no haze on God's eye.
    #[test]
    fn haze_covers_every_undecided_cone() {
        let cam = Camera4D::new(Vec4::new(10.0, 0.0, 0.0, 0.0), 60.0);
        let mut d = FrameDynamics::default();
        assert!(d.haze_origins(&cam, ViewMode::CameraSimultaneity).is_empty());
        assert_eq!(d.haze_origins(&cam, ViewMode::ProphecySlice), vec![cam.position]);

        // The other player was last heard of 4 s ago (lab time), 6 ls away.
        let remote = Vec4::new(6.0, 6.0, 0.0, 0.0);
        d.remote_origins = vec![remote];
        let origins = d.haze_origins(&cam, ViewMode::CameraSimultaneity);
        assert_eq!(origins, vec![remote]);
        assert!(d.haze_origins(&cam, ViewMode::PastLightCone).is_empty());

        let base = Vec3::new(0.5, 0.5, 0.5);
        let at = |x: f32| Vec4::new(10.0, x, 0.0, 0.0);
        // On the slice t = 10, the remote cone covers |x - 6| < 4.
        assert_eq!(apply_undecided_haze(base, at(-1.0), &origins), base);
        let hazed = apply_undecided_haze(base, at(6.0), &origins);
        assert!(hazed.z > hazed.x && hazed.z > hazed.y, "violet inside the cone: {hazed:?}");
        assert_eq!(apply_undecided_haze(base, at(11.0), &origins), base);
    }

    #[test]
    fn prophecy_delta_steps_are_clamped() {
        let mut cfg = RenderConfig {
            width: 1,
            height: 1,
            mode: RenderMode::RelativisticShaded,
            shadow_mode: ShadowMode::Off,
            slice_mode: ViewMode::ProphecySlice,
            prophecy_delta: PROPHECY_DELTA_DEFAULT,
        };
        cfg.step_prophecy_delta(2.0);
        assert_eq!(cfg.prophecy_delta, PROPHECY_DELTA_DEFAULT + 2.0 * PROPHECY_DELTA_STEP);
        cfg.step_prophecy_delta(-1000.0);
        assert_eq!(cfg.prophecy_delta, PROPHECY_DELTA_MIN);
        cfg.step_prophecy_delta(1000.0);
        assert_eq!(cfg.prophecy_delta, PROPHECY_DELTA_MAX);
    }
}
