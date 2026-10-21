//! Frontend-independent application state: scene selection, the observer camera, render
//! settings, per-frame dynamics and time controls. `main.rs` (minifb) and `web.rs` (browser)
//! translate their own key events into [`Action`]s and [`HeldInput`], and keep only what is
//! genuinely theirs (rendering backend, resolution, HUD visibility, presentation).

use crate::minkowski::{Vec3, Vec4};
use crate::renderer::{
    Camera4D, FrameDynamics, RenderConfig, RenderMode, ShadowMode, ViewMode,
    PROPHECY_DELTA_DEFAULT,
};
use crate::scene::Scene4D;

pub const TIME_SCALES: [f32; 10] = [-4.0, -2.0, -1.0, -0.5, -0.25, 0.25, 0.5, 1.0, 2.0, 4.0];
/// Index of `+1.0x` in [`TIME_SCALES`].
const DEFAULT_TIME_SCALE_IDX: usize = 7;
const FOV_Y_DEG: f32 = 65.0;

/// A discrete, edge-triggered user command.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Action {
    SelectScene(usize),
    NextScene,
    PrevScene,
    CycleRenderMode,
    CycleView,
    CycleShadows,
    TogglePause,
    /// Flip between forward and rewind at the same speed.
    ToggleRewind,
    Slower,
    Faster,
    /// Back to the scene's initial event and rest frame.
    Reset,
    /// Boost to the lab rest frame.
    Brake,
    /// Toggle between the lab frame and the scene's moving object's rest frame.
    MatchVelocity,
    EmitFlash,
    ProphecyDeltaUp,
    ProphecyDeltaDown,
}

/// Keys held down during a frame (continuous controls).
#[derive(Clone, Copy, Debug, Default)]
pub struct HeldInput {
    pub yaw_left: bool,
    pub yaw_right: bool,
    pub pitch_up: bool,
    pub pitch_down: bool,
    pub roll_left: bool,
    pub roll_right: bool,
    pub thrust_forward: bool,
    pub thrust_back: bool,
    pub thrust_right: bool,
    pub thrust_left: bool,
    pub thrust_up: bool,
    pub thrust_down: bool,
    pub scrub_back: bool,
    pub scrub_forward: bool,
    pub move_forward: bool,
    pub move_back: bool,
    pub move_right: bool,
    pub move_left: bool,
    pub move_up: bool,
    pub move_down: bool,
}

pub struct App {
    pub scenes: Vec<Scene4D>,
    pub scene_idx: usize,
    pub camera: Camera4D,
    pub cfg: RenderConfig,
    pub dynamics: FrameDynamics,
    pub paused: bool,
    pub time_scale_idx: usize,
}

impl App {
    pub fn new(scenes: Vec<Scene4D>, width: usize, height: usize) -> Self {
        let camera = Camera4D::new(scenes[0].demo.initial_camera_pos(), FOV_Y_DEG);
        Self {
            scenes,
            scene_idx: 0,
            camera,
            cfg: RenderConfig {
                width,
                height,
                mode: RenderMode::RelativisticShaded,
                shadow_mode: ShadowMode::LorentzInvariant,
                slice_mode: ViewMode::PastLightCone,
                prophecy_delta: PROPHECY_DELTA_DEFAULT,
            },
            dynamics: FrameDynamics::default(),
            paused: false,
            time_scale_idx: DEFAULT_TIME_SCALE_IDX,
        }
    }

    pub fn scene(&self) -> &Scene4D {
        &self.scenes[self.scene_idx]
    }

    pub fn time_scale(&self) -> f32 {
        TIME_SCALES[self.time_scale_idx]
    }

    pub fn apply(&mut self, action: Action) {
        let n = self.scenes.len();
        match action {
            Action::SelectScene(idx) => {
                if idx < n {
                    self.enter_scene(idx);
                }
            }
            Action::NextScene => self.enter_scene((self.scene_idx + 1) % n),
            Action::PrevScene => self.enter_scene((self.scene_idx + n - 1) % n),
            Action::CycleRenderMode => self.cfg.mode = self.cfg.mode.next(),
            Action::CycleView => self.cfg.slice_mode = self.cfg.slice_mode.next(),
            Action::CycleShadows => self.cfg.shadow_mode = self.cfg.shadow_mode.next(),
            Action::TogglePause => self.paused = !self.paused,
            Action::ToggleRewind => {
                self.time_scale_idx = TIME_SCALES.len() - 1 - self.time_scale_idx;
                self.paused = false;
            }
            Action::Slower => self.time_scale_idx = self.time_scale_idx.saturating_sub(1),
            Action::Faster => {
                self.time_scale_idx = (self.time_scale_idx + 1).min(TIME_SCALES.len() - 1)
            }
            Action::Reset => {
                self.enter_scene(self.scene_idx);
                self.time_scale_idx = DEFAULT_TIME_SCALE_IDX;
            }
            Action::Brake => self.camera.tetrad.boost_to(Vec4::new(1.0, 0.0, 0.0, 0.0)),
            Action::MatchVelocity => {
                if self.camera.tetrad.beta().length() > 0.1 {
                    self.camera.tetrad.boost_to(Vec4::new(1.0, 0.0, 0.0, 0.0));
                } else {
                    let beta = self.scene().demo.comoving_beta(self.camera.position.t);
                    self.camera.tetrad.boost_to(Vec4::four_velocity_from_beta(beta));
                }
            }
            Action::EmitFlash => self.dynamics.emit_flash(self.camera.position),
            Action::ProphecyDeltaUp => self.cfg.step_prophecy_delta(1.0),
            Action::ProphecyDeltaDown => self.cfg.step_prophecy_delta(-1.0),
        }
    }

    /// Rotate the view in the observer's rest frame (mouse look; keys go through `step`).
    pub fn look(&mut self, yaw: f32, pitch: f32, roll: f32) {
        if yaw != 0.0 || pitch != 0.0 || roll != 0.0 {
            self.camera.tetrad.rotate_local(yaw, pitch, roll);
        }
    }

    /// Advance one frame of `dt_wall` wall-clock seconds.
    pub fn step(&mut self, dt_wall: f32, input: &HeldInput) {
        let axis = |pos: bool, neg: bool| (pos as i32 - neg as i32) as f32;

        // 1. Local camera orientation in observer's rest frame
        let rot_speed = 1.6 * dt_wall;
        self.look(
            rot_speed * axis(input.yaw_right, input.yaw_left),
            rot_speed * axis(input.pitch_up, input.pitch_down),
            rot_speed * axis(input.roll_right, input.roll_left),
        );

        // 2. Proper acceleration thrusters (Lorentz boost in observer's local rest frame)
        let boost_rate = 1.25 * dt_wall; // Rapidity per second
        self.camera.tetrad.boost_local(
            Vec3::new(
                axis(input.thrust_right, input.thrust_left),
                axis(input.thrust_up, input.thrust_down),
                axis(input.thrust_forward, input.thrust_back),
            ) * boost_rate,
        );

        // 3. Direct coordinate-space & time scrubbing. Because scrubbing `t` at fixed
        // `(x, y, z)` is a coordinate translation rather than motion along the observer's
        // timelike worldline, `camera.proper_time` is left unchanged.
        let scrub_speed = 6.0 * dt_wall;
        let scrub_t = axis(input.scrub_forward, input.scrub_back);
        let mut time_changed = scrub_t != 0.0;
        self.camera.position.t += scrub_t * scrub_speed;
        let shift_local = Vec3::new(
            axis(input.move_right, input.move_left),
            axis(input.move_up, input.move_down),
            axis(input.move_forward, input.move_back),
        ) * scrub_speed;
        self.camera.position += self.camera.tetrad.local_spatial_to_world(shift_local);

        // 4. Advance observer along their timelike worldline: dX^mu = u^mu * d_tau
        if !self.paused {
            let d_tau = dt_wall * self.time_scale();
            self.camera.proper_time += d_tau;
            self.camera.position += self.camera.tetrad.e[0] * d_tau;
            time_changed = true;
        }

        // Loop lab time seamlessly within [scene.t_min, scene.t_max]
        let (t_min, t_max) = (self.scene().t_min, self.scene().t_max);
        let t = self.camera.position.t;
        if time_changed && (t > t_max || t < t_min) {
            let wrapped = t_min + (t - t_min).rem_euclid(t_max - t_min);
            // The scene is periodic, so carry flashes along to the same moment of the loop.
            self.dynamics.shift_time(wrapped - t);
            self.camera.position.t = wrapped;
        }
        self.dynamics.retire_flashes(self.camera.position.t);
    }

    fn enter_scene(&mut self, idx: usize) {
        self.scene_idx = idx;
        self.camera = Camera4D::new(self.scene().demo.initial_camera_pos(), FOV_Y_DEG);
        self.paused = false;
        self.dynamics.flashes.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::DemoScene;

    fn twin_app() -> App {
        App::new(vec![Scene4D::build(DemoScene::TwinParadox)], 64, 40)
    }

    #[test]
    fn step_advances_along_worldline_and_wraps_loop_with_flashes() {
        let mut app = twin_app();
        let t0 = app.camera.position.t;
        app.step(0.1, &HeldInput::default());
        assert!((app.camera.proper_time - 0.1).abs() < 1e-6);
        assert!((app.camera.position.t - (t0 + 0.1)).abs() < 1e-5);

        // Run just past the end of the loop with a flash in flight: both wrap together.
        let (t_min, t_max) = (app.scene().t_min, app.scene().t_max);
        app.camera.position.t = t_max - 0.05;
        app.apply(Action::EmitFlash);
        app.step(0.1, &HeldInput::default());
        let t = app.camera.position.t;
        assert!(t >= t_min && t < t_min + 0.1, "camera wrapped to {t}");
        let flash_age = t - app.dynamics.flashes[0].at.t;
        assert!((flash_age - 0.1).abs() < 1e-4, "flash kept its age, got {flash_age}");
    }

    #[test]
    fn controls_follow_the_observer_frame() {
        let mut app = twin_app();
        app.apply(Action::TogglePause);
        let held = HeldInput { thrust_forward: true, thrust_right: true, ..Default::default() };
        app.step(0.1, &held);
        let beta = app.camera.tetrad.beta();
        assert!(beta.z > 0.0 && beta.x > 0.0 && beta.y.abs() < 1e-6);
        assert_eq!(app.camera.proper_time, 0.0, "paused: thrust boosts but time stands still");

        let p0 = app.camera.position;
        app.step(0.1, &HeldInput { scrub_back: true, ..Default::default() });
        assert!((app.camera.position.t - (p0.t - 0.6)).abs() < 1e-5);
        assert_eq!(app.camera.position.spatial(), p0.spatial());
    }

    #[test]
    fn time_and_scene_actions() {
        let mut app = twin_app();
        assert_eq!(app.time_scale(), 1.0);
        app.apply(Action::ToggleRewind);
        assert_eq!(app.time_scale(), -1.0);
        for _ in 0..3 {
            app.apply(Action::Faster); // -0.5, -0.25, +0.25
        }
        assert_eq!(app.time_scale(), 0.25);

        app.apply(Action::EmitFlash);
        app.camera.proper_time = 5.0;
        app.apply(Action::Reset);
        assert_eq!(app.time_scale(), 1.0);
        assert_eq!(app.camera.proper_time, 0.0);
        assert!(app.dynamics.flashes.is_empty());

        app.apply(Action::SelectScene(7)); // out of range: ignored
        assert_eq!(app.scene_idx, 0);
    }
}
