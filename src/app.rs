//! Frontend-independent application state: scene selection, the observer camera, render
//! settings, per-frame dynamics and time controls. `main.rs` (minifb) and `web.rs` (browser)
//! translate their own key events into [`Action`]s and [`HeldInput`], and keep only what is
//! genuinely theirs (rendering backend, resolution, HUD visibility, presentation).

use crate::dynamic::billiards::{Ammo, FIRE_SPEED, SHIP_MASS, SHIP_RADIUS};
use crate::dynamic::body::{BodyState, Impulse};
use crate::dynamic::collide::P4;
use crate::dynamic::DynWorld;
use crate::minkowski::{Vec3, Vec4};
use crate::renderer::{
    Camera4D, FrameDynamics, Frontier, RenderConfig, RenderMode, ShadowMode, ViewMode,
    PROPHECY_DELTA_DEFAULT,
};
use crate::scene::Scene4D;

pub const TIME_SCALES: [f32; 10] = [-4.0, -2.0, -1.0, -0.5, -0.25, 0.25, 0.5, 1.0, 2.0, 4.0];
/// Index of the slowest *forward* speed: Dynamic Mode cannot run time backwards.
const FIRST_FORWARD_TIME_SCALE_IDX: usize = 5;
/// Index of `+1.0x` in [`TIME_SCALES`].
const DEFAULT_TIME_SCALE_IDX: usize = 7;
const FOV_Y_DEG: f32 = 65.0;
/// Wall-clock interval between geometry snapshots of a growing dynamic world (the plan's
/// "10 Hz full rebuild"; a thing at distance `D` was decided ~`2D` ago, so the lag is invisible
/// in the Real Camera and God's Eye views).
const DYNAMIC_REBUILD_INTERVAL: f32 = 0.1;
/// Rebase a dynamic world's lab time once the camera passes this, to keep `f32` render
/// coordinates precise while time runs forward indefinitely.
const DYNAMIC_REBASE_AFTER_T: f32 = 1000.0;

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
    /// Dynamic Mode: launch the current ammo from the ship along the view direction.
    Fire,
    /// Dynamic Mode: cycle what [`Action::Fire`] launches.
    CycleAmmo,
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
    /// The causal simulation behind the current scene, if it is a Dynamic Mode scene.
    pub world: Option<DynWorld>,
    /// What [`Action::Fire`] launches in Dynamic Mode.
    pub ammo: Ammo,
    /// Ship collision impulses whose news has not crossed the hull yet.
    kicks_in_flight: Vec<Impulse>,
    /// Bumped whenever a scene's geometry is replaced, so frontends know to re-upload it.
    pub scene_generation: u64,
    world_grew: bool,
    rebuild_timer: f32,
}

impl App {
    pub fn new(scenes: Vec<Scene4D>, width: usize, height: usize) -> Self {
        let camera = Camera4D::new(scenes[0].demo.initial_camera_pos(), FOV_Y_DEG);
        let mut app = Self {
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
            world: None,
            ammo: Ammo::Puck,
            kicks_in_flight: Vec::new(),
            scene_generation: 0,
            world_grew: false,
            rebuild_timer: 0.0,
        };
        app.enter_scene(0);
        app
    }

    pub fn scene(&self) -> &Scene4D {
        &self.scenes[self.scene_idx]
    }

    /// Dynamic Mode: the camera is the ship, so only timelike motion forward in time is allowed
    /// (no rewinding, coordinate scrubbing or teleporting).
    pub fn is_dynamic(&self) -> bool {
        self.world.is_some()
    }

    fn camera_frontier(&self) -> Frontier {
        Frontier {
            origin: self.camera.position,
            proper_time: self.camera.proper_time,
        }
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
                if !self.is_dynamic() {
                    self.time_scale_idx = TIME_SCALES.len() - 1 - self.time_scale_idx;
                    self.paused = false;
                }
            }
            Action::Slower => {
                let floor = if self.is_dynamic() { FIRST_FORWARD_TIME_SCALE_IDX } else { 0 };
                self.time_scale_idx = self.time_scale_idx.saturating_sub(1).max(floor);
            }
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
            Action::Fire => self.fire(),
            Action::CycleAmmo => self.ammo = self.ammo.next(),
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
        // timelike worldline, `camera.proper_time` is left unchanged. Such acausal jumps would
        // break "camera = ship" in Dynamic Mode, so they are disabled there.
        let mut time_changed = false;
        if !self.is_dynamic() {
            let scrub_speed = 6.0 * dt_wall;
            let scrub_t = axis(input.scrub_forward, input.scrub_back);
            time_changed = scrub_t != 0.0;
            self.camera.position.t += scrub_t * scrub_speed;
            let shift_local = Vec3::new(
                axis(input.move_right, input.move_left),
                axis(input.move_up, input.move_down),
                axis(input.move_forward, input.move_back),
            ) * scrub_speed;
            self.camera.position += self.camera.tetrad.local_spatial_to_world(shift_local);
        }

        // 4. Advance observer along their timelike worldline: dX^mu = u^mu * d_tau
        if !self.paused {
            let d_tau = dt_wall * self.time_scale();
            self.camera.proper_time += d_tau;
            self.camera.position += self.camera.tetrad.e[0] * d_tau;
            time_changed = true;
        }

        if self.is_dynamic() {
            self.step_world(dt_wall);
        } else {
            // Loop lab time seamlessly within [scene.t_min, scene.t_max]
            let (t_min, t_max) = (self.scene().t_min, self.scene().t_max);
            let t = self.camera.position.t;
            if time_changed && (t > t_max || t < t_min) {
                let wrapped = t_min + (t - t_min).rem_euclid(t_max - t_min);
                // The scene is periodic, so carry flashes along to the same moment of the loop.
                self.dynamics.shift_time(wrapped - t);
                self.camera.position.t = wrapped;
            }
        }
        self.dynamics.retire_flashes(self.camera.position.t);
    }

    /// Dynamic Mode: move the causal frontier to the camera and finalize what it uncovered,
    /// snapshotting the geometry at most every [`DYNAMIC_REBUILD_INTERVAL`].
    fn step_world(&mut self, dt_wall: f32) {
        if self.camera.position.t > DYNAMIC_REBASE_AFTER_T {
            let shift = -DYNAMIC_REBASE_AFTER_T;
            self.camera.position.t += shift;
            self.dynamics.shift_time(shift);
            for kick in &mut self.kicks_in_flight {
                kick.t += shift as f64;
            }
            if let Some(world) = self.world.as_mut() {
                world.rebase(shift as f64);
            }
            self.world_grew = true;
        }

        let frontier = self.camera_frontier();
        self.dynamics.frontier = Some(frontier);
        let world = self.world.as_mut().expect("dynamic scene has a world");
        self.world_grew |= world.advance(frontier, self.camera.tetrad.e[0]);
        self.dynamics.frontier_lag = world.frontier_lag(self.camera.tetrad.e[0]);

        // Collisions with the ship push back on it. The camera is the ship's center: it feels a
        // kick once news of the contact has crossed the hull, like any other body.
        self.kicks_in_flight.append(&mut world.ship_kicks);
        let p = self.camera.position;
        let ship = BodyState {
            step: 0,
            t: p.t as f64,
            pos: [p.x as f64, p.y as f64, p.z as f64],
            vel: [0.0; 3],
            tau: 0.0,
        };
        let (arrived, flying): (Vec<_>, Vec<_>) = std::mem::take(&mut self.kicks_in_flight)
            .into_iter()
            .partition(|k| k.has_reached(&ship, SHIP_RADIUS));
        self.kicks_in_flight = flying;
        for kick in arrived {
            self.kick_ship(kick.dp);
        }

        self.rebuild_timer += dt_wall;
        if self.world_grew && self.rebuild_timer >= DYNAMIC_REBUILD_INTERVAL {
            self.rebuild_dynamic_scene();
        }
    }

    /// Launch the current ammo at [`FIRE_SPEED`] along the view direction in the ship's rest
    /// frame: lab 4-velocity `gamma (e_0 + FIRE_SPEED * e_forward)`.
    fn fire(&mut self) {
        let frontier = self.camera_frontier();
        let Some(world) = self.world.as_mut() else {
            return;
        };
        let e = &self.camera.tetrad.e;
        let u = e[0] + e[3] * FIRE_SPEED as f32;
        let vel = [u.x / u.t, u.y / u.t, u.z / u.t].map(|c| c as f64);
        if world.fire(frontier, vel, self.ammo) {
            self.world_grew = true;
        }
    }

    /// Add collision 4-momentum `kick` to the ship (rest mass [`SHIP_MASS`]) and re-point the
    /// camera's 4-velocity along the result.
    fn kick_ship(&mut self, kick: P4) {
        let u = self.camera.tetrad.e[0];
        let p = [
            SHIP_MASS * u.t as f64 + kick[0],
            SHIP_MASS * u.x as f64 + kick[1],
            SHIP_MASS * u.y as f64 + kick[2],
            SHIP_MASS * u.z as f64 + kick[3],
        ];
        let m = (p[0] * p[0] - p[1] * p[1] - p[2] * p[2] - p[3] * p[3]).sqrt();
        let [t, x, y, z] = p.map(|c| (c / m) as f32);
        self.camera.tetrad.boost_to(Vec4::new(t, x, y, z));
    }

    /// The HUD's second line: the scene description, plus the ammo in Dynamic Mode.
    pub fn scene_description(&self) -> String {
        let desc = self.scene().demo.description();
        if self.is_dynamic() {
            format!("[ENTER] FIRE {} | [\\] AMMO | {desc}", self.ammo.name())
        } else {
            desc.to_string()
        }
    }

    fn rebuild_dynamic_scene(&mut self) {
        if let Some(world) = &self.world {
            self.scenes[self.scene_idx] = world.build_scene();
            self.scene_generation += 1;
            self.world_grew = false;
            self.rebuild_timer = 0.0;
        }
    }

    fn enter_scene(&mut self, idx: usize) {
        self.scene_idx = idx;
        let demo = self.scene().demo;
        self.camera = Camera4D::new(demo.initial_camera_pos(), FOV_Y_DEG);
        self.paused = false;
        self.dynamics.flashes.clear();
        self.kicks_in_flight.clear();

        self.world = demo.is_dynamic().then(|| {
            DynWorld::new(demo, self.camera_frontier(), self.camera.tetrad.e[0])
        });
        self.dynamics.frontier = self.world.as_ref().map(|w| w.frontier);
        let u = self.camera.tetrad.e[0];
        self.dynamics.frontier_lag = self.world.as_ref().map_or(0.0, |w| w.frontier_lag(u));
        if self.is_dynamic() {
            self.time_scale_idx = self.time_scale_idx.max(FIRST_FORWARD_TIME_SCALE_IDX);
            self.rebuild_dynamic_scene();
        }
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

    fn billiards_app() -> App {
        let scenes = vec![
            Scene4D::build(DemoScene::TwinParadox),
            Scene4D::build(DemoScene::RelativisticBilliards),
        ];
        let mut app = App::new(scenes, 64, 40);
        app.apply(Action::SelectScene(1));
        app
    }

    #[test]
    fn dynamic_scene_tracks_the_camera_frontier_and_rebuilds() {
        let mut app = billiards_app();
        assert!(app.is_dynamic());
        let entered = app.scene_generation;
        assert!(entered > 0, "entering a dynamic scene snapshots its world");

        app.step(0.05, &HeldInput::default());
        let f = app.dynamics.frontier.expect("frontier follows the camera");
        assert_eq!(f.origin, app.camera.position);
        assert_eq!(app.scene_generation, entered, "no snapshot before the rebuild interval");
        app.step(0.06, &HeldInput::default());
        assert_eq!(app.scene_generation, entered + 1);

        // Back to a static scene: no world, no frontier.
        app.apply(Action::SelectScene(0));
        assert!(!app.is_dynamic() && app.dynamics.frontier.is_none());
    }

    #[test]
    fn dynamic_scene_only_moves_forward_in_time() {
        let mut app = billiards_app();
        app.apply(Action::ToggleRewind);
        for _ in 0..10 {
            app.apply(Action::Slower);
        }
        assert!(app.time_scale() > 0.0);

        let p0 = app.camera.position;
        app.apply(Action::TogglePause);
        let scrub = HeldInput { scrub_back: true, move_forward: true, ..Default::default() };
        app.step(0.1, &scrub);
        assert_eq!(app.camera.position, p0, "scrubbing and teleporting are disabled");
    }

    #[test]
    fn firing_launches_the_selected_ammo_forward() {
        let mut app = billiards_app();
        let before = app.world.as_ref().unwrap().bodies.len();
        app.apply(Action::CycleAmmo);
        assert_eq!(app.ammo, Ammo::Sticky);
        app.apply(Action::Fire);
        let world = app.world.as_ref().unwrap();
        assert_eq!(world.bodies.len(), before + 1);
        let shot = world.bodies.last().unwrap();
        assert!(shot.is_sticky() && shot.born);
        // Straight ahead (+z) at the launch speed, the camera being at rest.
        let v = shot.latest().vel;
        assert!((v[2] - FIRE_SPEED).abs() < 1e-6 && v[0].abs() < 1e-6);
        assert!(app.scene_description().contains("STICKY PUCK"));
    }

    #[test]
    fn ship_kick_changes_the_camera_velocity() {
        let mut app = billiards_app();
        // The momentum a 0.6c puck hands over when it bounces straight back: (0, 0, 0, -1.5).
        app.kick_ship([0.0, 0.0, 0.0, -1.5]);
        let beta = app.camera.tetrad.beta();
        assert!(beta.z < -0.2 && beta.x.abs() < 1e-6, "ship pushed back: {beta:?}");
    }

    #[test]
    fn dynamic_scene_rebases_lab_time() {
        let mut app = billiards_app();
        app.camera.position.t = DYNAMIC_REBASE_AFTER_T - 0.01;
        app.apply(Action::EmitFlash);
        app.step(0.1, &HeldInput::default());
        let t = app.camera.position.t;
        assert!(t < 1.0, "camera rebased to {t}");
        let world = app.world.as_ref().unwrap();
        assert_eq!(world.frontier.origin, app.camera.position);
        assert!((app.dynamics.flashes[0].at.t - (t - 0.1)).abs() < 1e-3);
    }
}
