//! Frontend-independent application state: scene selection, the observer camera, render
//! settings, per-frame dynamics and time controls. `main.rs` (minifb) and `web.rs` (browser)
//! translate their own key events into [`Action`]s and [`HeldInput`], and keep only what is
//! genuinely theirs (rendering backend, resolution, HUD visibility, presentation).

use crate::dynamic::billiards;
use crate::dynamic::helm::{self, CommandQueue, ControlMode, Hull, Order};
use crate::dynamic::{DynWorld, PlayerId, SceneCache, ShipSample};
use crate::hud::{HudInfo, HudSections};
use crate::minkowski::{Vec3, Vec4};
use crate::renderer::{
    Camera4D, FrameDynamics, Frontier, RenderConfig, RenderMode, ShadowMode, ViewMode,
    PROPHECY_DELTA_DEFAULT,
};
use crate::multiplayer::game::{self, Match};
use crate::multiplayer::link::Link;
use crate::perf::{self, PerfStats};
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
/// The longest frame the frontends report: they clamp `dt_wall` to it.
const MAX_DT_WALL: f32 = 0.1;
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
    /// Boost to the lab rest frame (in a match: decelerate toward it while held, see
    /// [`HeldInput::brake`]).
    Brake,
    /// Toggle between the lab frame and the scene's moving object's rest frame (in a match:
    /// accelerate toward it while held, see [`HeldInput::match_velocity`]).
    MatchVelocity,
    EmitFlash,
    ProphecyDeltaUp,
    ProphecyDeltaDown,
    /// Dynamic Mode: fire a laser pulse from the pilot along the view direction (at the
    /// reticle).
    Fire,
    /// Dynamic Mode: switch between delayed controls and a control pod ([`ControlMode`]).
    /// Alone this restarts the table; in a match it only works before the match starts.
    CycleControls,
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
    /// The [`Action::Brake`] key is (still) down.
    pub brake: bool,
    /// The [`Action::MatchVelocity`] key is (still) down.
    pub match_velocity: bool,
    pub scrub_back: bool,
    pub scrub_forward: bool,
    pub move_forward: bool,
    pub move_back: bool,
    pub move_right: bool,
    pub move_left: bool,
    pub move_up: bool,
    pub move_down: bool,
}

/// Dynamic Mode: an [`Action::Brake`] or [`Action::MatchVelocity`] in progress. The ship is a
/// rigid body of radius [`billiards::SHIP_RADIUS`], so it cannot change velocity at once: it boosts toward
/// `target` at its pilot's acceleration ([`ControlMode::pilot_accel`]) for as long as its key
/// stays down.
#[derive(Clone, Copy)]
struct Steer {
    target: Vec4,
    held: fn(&HeldInput) -> bool,
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
    /// Dynamic Mode: how the ship's controls reach its hull.
    pub controls: ControlMode,
    /// [`ControlMode::Delayed`]: commands on their way across the hull.
    commands: CommandQueue,
    steer: Option<Steer>,
    /// Bumped whenever a scene's geometry is replaced, so frontends know to re-upload it.
    pub scene_generation: u64,
    world_grew: bool,
    rebuild_timer: f32,
    /// Dynamic Mode: the geometry of the last snapshot, reused by the next.
    scene_cache: SceneCache,
    /// Wall-clock interval between geometry snapshots of a growing dynamic world;
    /// [`DYNAMIC_REBUILD_INTERVAL`] unless nothing renders (e.g. headless tests).
    pub rebuild_interval: f32,
    /// The two-player match this app plays in, if any (see `docs/MULTIPLAYER.md`). It pins the
    /// app to the Dynamic Mode scene.
    pub multiplayer: Option<Match>,
    /// Frame timings for the HUD; the app records its own phases, the frontend the rest.
    pub perf: PerfStats,
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
            controls: ControlMode::default(),
            commands: CommandQueue::default(),
            steer: None,
            scene_generation: 0,
            world_grew: false,
            rebuild_timer: 0.0,
            scene_cache: SceneCache::default(),
            rebuild_interval: DYNAMIC_REBUILD_INTERVAL,
            multiplayer: None,
            perf: PerfStats::default(),
        };
        app.enter_scene(0);
        app
    }

    /// Play `local`'s side of a two-player match over `link`, in the Dynamic Mode scene (which
    /// `scenes` must include).
    pub fn new_multiplayer(
        scenes: Vec<Scene4D>,
        width: usize,
        height: usize,
        local: PlayerId,
        link: Box<dyn Link>,
    ) -> Self {
        let idx = scenes
            .iter()
            .position(|s| s.demo.is_dynamic())
            .expect("multiplayer needs a Dynamic Mode scene");
        let mut app = Self::new(scenes, width, height);
        app.multiplayer = Some(Match::new(local, link));
        app.enter_scene(idx);
        app
    }

    /// Multiplayer: this player is held back until the other one catches up. Only looking
    /// around (and view settings) work meanwhile.
    pub fn is_waiting(&self) -> bool {
        self.multiplayer.as_ref().is_some_and(Match::is_waiting)
    }

    /// A message to show across the screen, if the player cannot play right now.
    pub fn banner(&self) -> Option<&'static str> {
        self.multiplayer.as_ref().and_then(Match::banner)
    }

    /// Multiplayer: `secs` of real wall-clock time passed (unclamped, unlike `step`'s
    /// `dt_wall`). Times the round trip to the other player.
    pub fn tick_wall_clock(&mut self, secs: f64) {
        if let Some(m) = self.multiplayer.as_mut() {
            m.tick(secs);
        }
    }

    /// Multiplayer: the HUD's line of latency figures (see [`Match::metrics`]).
    pub fn net_metrics(&self) -> Option<String> {
        self.multiplayer.as_ref().and_then(Match::metrics)
    }

    /// Tell the other player we are leaving (call before quitting).
    pub fn leave_match(&mut self) {
        if let Some(m) = self.multiplayer.as_mut() {
            m.leave();
        }
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
        let multiplayer = self.multiplayer.is_some();
        match action {
            // Both players' worlds must start alike, so a match never leaves its scene.
            Action::SelectScene(_) | Action::NextScene | Action::PrevScene | Action::Reset
                if multiplayer => {}
            // A waiting player's ship is frozen (steering waits with it, see `step`).
            Action::Fire if self.is_waiting() => {}
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
            Action::Brake | Action::MatchVelocity => {
                let rest = Vec4::new(1.0, 0.0, 0.0, 0.0);
                let (target, held): (_, fn(&HeldInput) -> bool) = match action {
                    Action::Brake => (rest, |i| i.brake),
                    _ if self.camera.tetrad.beta().length() > 0.1 => (rest, |i| i.match_velocity),
                    _ => {
                        let beta = self.scene().demo.comoving_beta(self.camera.position.t);
                        (Vec4::four_velocity_from_beta(beta), |i| i.match_velocity)
                    }
                };
                if self.is_dynamic() {
                    self.steer = Some(Steer { target, held });
                } else {
                    self.camera.tetrad.boost_to(target);
                }
            }
            Action::EmitFlash => {
                self.dynamics.emit_flash(self.camera.position);
                if let Some(m) = self.multiplayer.as_mut() {
                    m.send_flash(self.camera.position);
                }
            }
            Action::ProphecyDeltaUp => self.cfg.step_prophecy_delta(1.0),
            Action::ProphecyDeltaDown => self.cfg.step_prophecy_delta(-1.0),
            Action::Fire => self.fire(),
            // The pilot and the hull's center only coincide again when nothing has moved yet.
            Action::CycleControls if multiplayer => {
                if self.camera.proper_time == 0.0 {
                    self.controls = self.controls.next();
                    if let (Some(m), Some(world)) = (self.multiplayer.as_mut(), self.world.as_mut()) {
                        world.set_controls(m.local(), self.controls);
                        m.send_controls(self.controls);
                    }
                }
            }
            Action::CycleControls => {
                self.controls = self.controls.next();
                if self.is_dynamic() {
                    self.enter_scene(self.scene_idx);
                }
            }
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
        let start = perf::now_ms();
        let rebuild_ms = self.step_untimed(dt_wall, input);
        self.perf.record("SIMULATION", perf::since(start) - rebuild_ms);
    }

    /// [`step`](App::step); returns the milliseconds it spent rebuilding the scene.
    fn step_untimed(&mut self, dt_wall: f32, input: &HeldInput) -> f32 {
        let axis = |pos: bool, neg: bool| (pos as i32 - neg as i32) as f32;

        // 0. Multiplayer: what the other player did since the last frame.
        if let (Some(m), Some(world)) = (self.multiplayer.as_mut(), self.world.as_mut()) {
            self.world_grew |= m.receive(world, &mut self.dynamics);
        }

        // 1. Local camera orientation in observer's rest frame
        let rot_speed = 1.6 * dt_wall;
        self.look(
            rot_speed * axis(input.yaw_right, input.yaw_left),
            rot_speed * axis(input.pitch_up, input.pitch_down),
            rot_speed * axis(input.roll_right, input.roll_left),
        );
        let looked = self.camera;
        let commands = self.commands.clone();

        // 2. Proper acceleration thrusters (Lorentz boost in observer's local rest frame)
        let d_tau = if self.paused { 0.0 } else { dt_wall * self.time_scale() };
        let thrust = Vec3::new(
            axis(input.thrust_right, input.thrust_left),
            axis(input.thrust_up, input.thrust_down),
            axis(input.thrust_forward, input.thrust_back),
        );
        if self.is_dynamic() {
            // The ship is rigid, and it learns of the controls no faster than light (see
            // `dynamic::helm`). Thrust is per second of the pilot's own clock, below the
            // Rindler bound in any direction. Steering (`X` / `V`) takes over the thrusters
            // while its key is held.
            let rapidity = self.controls.pilot_accel() * d_tau;
            self.steer = self.steer.filter(|s| (s.held)(input));
            let order = match self.steer {
                Some(s) => Some(Order::Steer(s.target)),
                None if thrust != Vec3::default() => {
                    Some(Order::Thrust(self.camera.tetrad.local_spatial_to_world(thrust.normalized())))
                }
                None => None,
            };
            match self.controls {
                ControlMode::Delayed => {
                    if let Some(order) = order.filter(|_| d_tau > 0.0) {
                        self.commands.issue(self.camera.proper_time, order, rapidity);
                    }
                    self.commands.execute_due(self.camera.proper_time, &mut self.camera.tetrad);
                }
                ControlMode::Pod => {
                    if let Some(order) = order {
                        order.execute(&mut self.camera.tetrad, rapidity);
                    }
                }
            }
        } else {
            let boost_rate = 1.25 * dt_wall; // Rapidity per second
            self.camera.tetrad.boost_local(thrust * boost_rate);
        }

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
            self.camera.proper_time += d_tau;
            self.camera.position += self.camera.tetrad.e[0] * d_tau;
            time_changed = true;
        }

        // 5. Multiplayer: never move into the other player's future light cone. If this frame's
        // thrust and motion would, undo them (commands already issued or executed included) and
        // wait for the other player to catch up.
        if let Some(m) = self.multiplayer.as_mut() {
            let sample = |c: &Camera4D| {
                ShipSample::new(Frontier { origin: c.position, proper_time: c.proper_time }, c.tetrad.e[0])
            };
            if !m.may_advance(&sample(&looked), &sample(&self.camera)) {
                self.camera = looked;
                self.commands = commands;
            }
        }

        let mut rebuild_ms = 0.0;
        if self.is_dynamic() {
            rebuild_ms = self.step_world(dt_wall);
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
        rebuild_ms
    }

    /// Dynamic Mode: the hull's center at the camera's lab time, as the pilot knows it (the world
    /// computes the hull's track from the pilot's).
    fn hull(&self) -> Option<Hull> {
        let world = self.world.as_ref()?;
        Some(world.ships[world.viewer as usize].hull_estimate(self.camera.position.t as f64))
    }

    /// Dynamic Mode: move the causal frontier to the camera and finalize what it uncovered,
    /// snapshotting the geometry at most every [`rebuild_interval`](App::rebuild_interval).
    /// Returns the milliseconds spent snapshotting.
    fn step_world(&mut self, dt_wall: f32) -> f32 {
        // A match never rebases: the two replicas would do it at different moments, and their
        // `f64` rounding would part ways (docs/MULTIPLAYER.md §2).
        if self.multiplayer.is_none() && self.camera.position.t > DYNAMIC_REBASE_AFTER_T {
            let shift = -DYNAMIC_REBASE_AFTER_T;
            self.camera.position.t += shift;
            self.dynamics.shift_time(shift);
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
        self.dynamics.remote_origins = world.remote_origins();
        if let Some(m) = self.multiplayer.as_mut() {
            m.send_track(*world.ships[m.local() as usize].latest(), &self.commands);
        }

        // A laser hit on a hull lights it up (each player sees the flash when its light
        // arrives).
        for hit in world.fresh_hits.drain(..) {
            self.dynamics.emit_hit_flash(hit);
        }

        self.rebuild_timer += dt_wall;
        if self.world_grew && self.rebuild_timer >= self.rebuild_interval {
            let ms = self.rebuild_dynamic_scene();
            // The next frame's `dt_wall` includes this rebuild: count the interval from its
            // end, or a rebuild longer than the interval would make every frame rebuild.
            self.rebuild_timer = -(ms / 1e3).min(MAX_DT_WALL);
            ms
        } else {
            0.0
        }
    }

    /// Fire a laser pulse from the pilot along the view direction in the ship's rest frame:
    /// the null direction `e_0 + e_forward`, which is where the reticle (the screen's center)
    /// looks.
    fn fire(&mut self) {
        let e = &self.camera.tetrad.e;
        let from = ShipSample::new(self.camera_frontier(), e[0]);
        let k = e[0] + e[3];
        let dir = [k.x / k.t, k.y / k.t, k.z / k.t].map(|c| c as f64);
        let Some(world) = self.world.as_mut() else {
            return;
        };
        world.fire_laser(world.viewer, from, dir);
        self.world_grew = true;
        if let Some(m) = self.multiplayer.as_mut() {
            m.send_laser(from, dir);
        }
    }

    /// The HUD's second line: the scene description, plus the laser and controls in Dynamic
    /// Mode.
    pub fn scene_description(&self) -> String {
        let desc = self.scene().demo.description();
        let laser = "[ENTER] FIRE LASER";
        if let Some(m) = &self.multiplayer {
            format!("{laser} | {} | {}", self.controls_status(), m.status())
        } else if self.is_dynamic() {
            format!("{laser} | {} | {desc}", self.controls_status())
        } else {
            desc.to_string()
        }
    }

    /// Dynamic Mode: the laser scoreboard for the HUD. In a match, the hits each side has
    /// landed on the other's hull, counted as news of them arrives; alone, the shots fired.
    pub fn scoreboard(&self) -> Option<String> {
        let world = self.world.as_ref()?;
        let me = world.viewer as usize;
        Some(match &self.multiplayer {
            Some(_) => {
                let theirs: u32 = world.scores.iter().enumerate().filter(|&(p, _)| p != me).map(|(_, s)| s).sum();
                format!("LASER HITS  YOU {} : {} THEM", world.scores[me], theirs)
            }
            None => format!("LASER SHOTS {}", world.ships[me].shots),
        })
    }

    /// The controls, for the HUD: the command lag (and whether the engines are spooling up), or
    /// how far the pod is from the hull's center.
    fn controls_status(&self) -> String {
        match self.controls {
            ControlMode::Delayed => {
                let spooling = if self.commands.pending() > 0 { " SPOOLING" } else { "" };
                format!("[;] DELAYED CONTROLS {:.2}S{spooling}", helm::command_delay())
            }
            ControlMode::Pod => {
                let offset = self.hull().map_or(0.0, |h| h.offset(self.camera.position).minkowski_norm_sq().max(0.0).sqrt());
                format!("[;] CONTROL POD, {offset:.2} OFF CENTER")
            }
        }
    }

    /// Snapshot the dynamic world's geometry (and its BVH). Returns the milliseconds it took.
    fn rebuild_dynamic_scene(&mut self) -> f32 {
        let Some(world) = &self.world else {
            return 0.0;
        };
        let start = perf::now_ms();
        world.rebuild_scene(&mut self.scene_cache, &mut self.scenes[self.scene_idx]);
        self.scene_generation += 1;
        self.world_grew = false;
        self.rebuild_timer = 0.0;
        let ms = perf::since(start);
        self.perf.record("SCENE+BVH REBUILD", ms);
        ms
    }

    /// What the HUD shows this frame. `render_ms` is the average time the frontend spends
    /// producing the image (for the performance section's cost per pixel).
    pub fn hud_info<'a>(&'a self, backend: &'a str, sections: HudSections, render_ms: f32) -> HudInfo<'a> {
        HudInfo {
            demo: self.scene().demo,
            desc: self.scene_description(),
            score: self.scoreboard(),
            net: self.net_metrics(),
            camera: &self.camera,
            cfg: &self.cfg,
            paused: self.paused,
            time_scale: self.time_scale(),
            backend,
            sections,
            perf: &self.perf,
            perf_context: if sections.perf {
                perf::render_context(self.scene(), &self.cfg, render_ms)
            } else {
                Vec::new()
            },
        }
    }

    fn enter_scene(&mut self, idx: usize) {
        self.scene_idx = idx;
        let demo = self.scene().demo;
        let player = self.multiplayer.as_ref().map(Match::local);
        self.camera = match player {
            Some(p) => {
                let mut camera = Camera4D::new(billiards::spawn_event(p), FOV_Y_DEG);
                camera.tetrad.rotate_local(billiards::spawn_yaw(p), 0.0, 0.0);
                camera
            }
            None => Camera4D::new(demo.initial_camera_pos(), FOV_Y_DEG),
        };
        self.paused = false;
        self.dynamics.flashes.clear();
        self.steer = None;
        self.commands.clear();

        self.scene_cache = SceneCache::default();
        self.world = demo.is_dynamic().then(|| match player {
            Some(p) => DynWorld::new_match(demo, &game::spawns(), p),
            None => DynWorld::new(demo, self.camera_frontier(), self.camera.tetrad.e[0]),
        });
        if let Some(world) = self.world.as_mut() {
            world.set_controls(world.viewer, self.controls);
        }
        self.dynamics.frontier = self.world.as_ref().map(|w| w.frontier);
        let u = self.camera.tetrad.e[0];
        self.dynamics.frontier_lag = self.world.as_ref().map_or(0.0, |w| w.frontier_lag(u));
        self.dynamics.remote_origins =
            self.world.as_ref().map_or(Vec::new(), DynWorld::remote_origins);
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
    use crate::dynamic::billiards::SHIP_RADIUS;
    use crate::dynamic::laser;

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

    /// The laser leaves the pilot where the reticle (the screen's center) looks: straight
    /// ahead in the ship's rest frame, where the center pixel's past light ray comes from.
    /// (In the lab, aberration bends the two differently.)
    #[test]
    fn the_laser_fires_at_the_reticle() {
        let mut app = billiards_app();
        app.look(0.4, -0.2, 0.1);
        app.camera.tetrad.boost_local(Vec3::new(0.5, 0.0, 0.3)); // moving: aberration
        app.apply(Action::Fire);
        let world = app.world.as_ref().unwrap();
        let pulse = &world.pulses[0];
        assert_eq!(pulse.t0, app.camera.position.t as f64);
        let [x, y, z] = pulse.dir.map(|c| c as f32);
        let k = Vec4::new(1.0, x, y, z);
        let e = &app.camera.tetrad.e;
        let local = [1, 2, 3].map(|i| k.minkowski_dot(e[i]));
        assert!(local[0].abs() < 1e-5 && local[1].abs() < 1e-5 && local[2] > 0.0, "{local:?}");
        let center = app.camera.tetrad.past_light_ray_direction(Vec3::new(0.0, 0.0, 1.0));
        assert!(center.minkowski_dot(e[3]) > 0.0 && center.minkowski_dot(e[1]).abs() < 1e-6);
        assert!(app.scene_description().contains("FIRE LASER"));
        assert_eq!(app.scoreboard().as_deref(), Some("LASER SHOTS 1"));
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

    /// Two players' apps connected in-process: player 0 and player 1. Nothing renders them,
    /// so they skip the geometry snapshots, which would otherwise take nearly all the time.
    fn match_apps() -> (App, App) {
        let (a, b) = crate::multiplayer::link::LocalLink::pair();
        let app = |p, link| {
            let scenes =
                vec![Scene4D::build(DemoScene::TwinParadox), Scene4D::build(DemoScene::RelativisticBilliards)];
            let mut app = App::new_multiplayer(scenes, 64, 40, p, Box::new(link));
            app.rebuild_interval = f32::INFINITY;
            app
        };
        (app(0, a), app(1, b))
    }

    /// `O_b` lies in the open future light cone of `O_a`, beyond `slack`.
    fn in_future_cone(a: Vec4, b: Vec4, slack: f32) -> bool {
        let d = b - a;
        d.t > d.spatial().length() + slack
    }

    /// The twin paradox, played: player 1 flies out at ~0.9c and back while player 0 stays put.
    /// The traveller covers more lab time per second of their own, runs into the stay-at-home
    /// player's future cone and has to wait; neither ever enters the other's cone, and the
    /// traveller comes home younger.
    #[test]
    fn the_travelling_twin_waits_for_the_stay_at_home_twin() {
        let (mut home, mut away) = match_apps();
        assert_eq!(home.scene().demo, DemoScene::RelativisticBilliards);
        let (mut waited, mut waits_now) = (0, false);
        let dt = 0.05;
        for frame in 0..1200 {
            let secs = frame as f32 * dt;
            let thrust = |from: f32, to: f32| secs >= from && secs < to;
            let held = HeldInput {
                thrust_forward: thrust(0.0, 1.2) || thrust(13.6, 14.8),
                thrust_back: thrust(9.2, 11.6),
                ..Default::default()
            };
            home.step(dt, &HeldInput::default());
            away.step(dt, &held);
            let (h, a) = (home.camera.position, away.camera.position);
            // One frame of slack: a ship level with the other in CM time may step once.
            assert!(!in_future_cone(h, a, 0.1), "frame {frame}: traveller inside home's cone");
            assert!(!in_future_cone(a, h, 0.1), "frame {frame}: home inside traveller's cone");
            assert!(!home.is_waiting(), "frame {frame}: the stay-at-home twin never waits");
            waits_now = away.is_waiting();
            waited += waits_now as usize;
            if waits_now {
                assert_eq!(away.banner(), Some(game::WAITING_BANNER));
            }
        }
        println!(
            "traveller waited {waited} frames; tau home {:.2}, away {:.2}",
            home.camera.proper_time, away.camera.proper_time
        );
        assert!(waited > 20, "the traveller should have had to wait");
        assert!(!waits_now, "and resumed once the stay-at-home twin caught up");
        assert!(away.camera.proper_time < home.camera.proper_time - 1.0, "the traveller is younger");
        assert!(away.scene_description().contains("PLAYER 2 OF 2"));
    }

    /// Rapidity of the camera relative to the lab.
    fn rapidity(app: &App) -> f32 {
        app.camera.tetrad.gamma().acosh()
    }

    /// Step both players `frames` times, player 0 holding `held`.
    fn play(p0: &mut App, p1: &mut App, frames: usize, dt: f32, held: &HeldInput) {
        for _ in 0..frames {
            p0.step(dt, held);
            p1.step(dt, &HeldInput::default());
        }
    }

    /// Frames of `dt` for delayed commands to cross the hull, with one to spare.
    fn delay_frames(dt: f32) -> usize {
        (helm::command_delay() / dt).ceil() as usize + 1
    }

    /// The ship is rigid and hears of its controls at light speed. With delayed controls,
    /// thrust takes effect [`helm::command_delay`] later; then thrust in any direction
    /// (diagonals included) stays below the Rindler bound `1 / SHIP_RADIUS` per second of the
    /// ship's clock, and a paused ship cannot change velocity at all.
    #[test]
    fn match_thrust_is_delayed_and_respects_the_rindler_bound() {
        let (mut p0, mut p1) = match_apps();
        assert_eq!(p0.controls, ControlMode::Delayed);
        let all = HeldInput {
            thrust_forward: true,
            thrust_right: true,
            thrust_up: true,
            ..Default::default()
        };
        let (dt, a) = (0.05, billiards::SHIP_ACCEL as f32);
        let mut before = 0.0;
        for frame in 0..40 {
            let tau = p0.camera.proper_time;
            play(&mut p0, &mut p1, 1, dt, &all);
            let r = rapidity(&p0);
            if tau < helm::command_delay() {
                assert_eq!(r, 0.0, "frame {frame}: the command has not crossed the hull yet");
                assert!(p0.scene_description().contains("SPOOLING"));
            } else {
                let accel = (r - before) / dt;
                assert!((accel - a).abs() < 1e-2, "frame {frame}: proper acceleration {accel}");
                assert!(accel * SHIP_RADIUS as f32 <= 0.8 + 1e-3);
            }
            before = r;
        }

        // Paused, the clock stands still: the command due right now still executes, then
        // nothing more is issued or comes due.
        p0.apply(Action::TogglePause);
        play(&mut p0, &mut p1, 1, dt, &all);
        let u = p0.camera.tetrad.e[0];
        play(&mut p0, &mut p1, 3, dt, &all);
        assert_eq!(p0.camera.tetrad.e[0], u, "no proper time, no boost");
    }

    /// `X` brakes gradually while held, and releasing it mid-brake coasts on (once the
    /// commands have crossed the hull).
    #[test]
    fn match_brake_is_gradual_and_stops_on_release() {
        let (mut p0, mut p1) = match_apps();
        let (dt, a) = (0.05, billiards::SHIP_ACCEL as f32);
        let wait = delay_frames(dt);
        let idle = HeldInput::default();
        play(&mut p0, &mut p1, 10, dt, &HeldInput { thrust_forward: true, ..Default::default() });
        play(&mut p0, &mut p1, wait, dt, &idle);
        let cruise = rapidity(&p0);
        assert!((cruise - 10.0 * dt * a).abs() < 1e-3, "{cruise}");

        let brake = HeldInput { brake: true, ..Default::default() };
        p0.apply(Action::Brake);
        play(&mut p0, &mut p1, 2, dt, &brake);
        play(&mut p0, &mut p1, 1, dt, &idle); // released: coast
        play(&mut p0, &mut p1, 1, dt, &brake); // held again without a new press: still coasting
        play(&mut p0, &mut p1, wait, dt, &idle);
        let braked = rapidity(&p0);
        assert!((cruise - braked - 2.0 * dt * a).abs() < 1e-3, "{cruise} -> {braked}");

        p0.apply(Action::Brake);
        play(&mut p0, &mut p1, 20, dt, &brake);
        play(&mut p0, &mut p1, wait, dt, &idle);
        assert_eq!(p0.camera.tetrad.beta(), Vec3::default(), "came to rest in the lab");
    }

    /// With a control pod the pilot accelerates at once; the hull only follows once news of it
    /// has crossed the hull. The other player's replica, told only the controls and the
    /// pilot's track, computes the very same hull and sees the two apart.
    #[test]
    fn a_control_pod_responds_at_once_and_its_hull_follows() {
        let (mut p0, mut p1) = match_apps();
        p0.apply(Action::CycleControls); // before the match starts
        assert_eq!(p0.controls, ControlMode::Pod);
        let dt = 0.05;
        let forward = HeldInput { thrust_forward: true, ..Default::default() };
        play(&mut p0, &mut p1, 2, dt, &forward);
        assert!(p0.camera.proper_time > 0.0);
        assert!(rapidity(&p0) > 0.9 * helm::pod_accel() * dt, "no lag");
        assert_eq!(p0.hull().unwrap().u, Vec4::new(1.0, 0.0, 0.0, 0.0), "the hull hasn't heard yet");

        p0.apply(Action::CycleControls); // too late now
        assert_eq!(p0.controls, ControlMode::Pod);

        play(&mut p0, &mut p1, 40, dt, &forward);
        assert!(p0.hull().unwrap().u.z > 0.1, "the hull follows: {:?}", p0.hull().unwrap().u);
        let offset = p0.hull().unwrap().offset(p0.camera.position).minkowski_norm_sq().sqrt();
        assert!(offset > 0.02 && offset < SHIP_RADIUS as f32, "pod off center by {offset}");
        assert!(p0.scene_description().contains("CONTROL POD"));

        // Player 1 computed player 0's hull from their pilot's track, bit for bit.
        play(&mut p0, &mut p1, 5, dt, &HeldInput::default());
        let (mine, seen) = (&p0.world.as_ref().unwrap().ships[0], &p1.world.as_ref().unwrap().ships[0]);
        assert_eq!(seen.controls, ControlMode::Pod);
        let n = seen.track.len();
        assert!(n > 40, "player 1 heard player 0's track");
        assert_eq!(mine.track[..n], seen.track[..]);
        assert_eq!(mine.hull[..n], seen.hull[..], "both replicas computed the same hull");
        let (pilot, hull) = (seen.latest(), seen.latest_hull());
        assert!((pilot.pos[2] - hull.pos[2]).abs() > 0.02, "pilot and hull apart");
    }

    /// The other player's hull is drawn on past its pilot's latest event as far as its motion
    /// is settled: far enough for the near side of the hull to be in view from anywhere
    /// outside the pilot's future cone, and just as the hull then really flies, whatever the
    /// pilot does next.
    #[test]
    fn the_other_hull_is_drawn_ahead_as_it_will_fly() {
        for controls in [ControlMode::Delayed, ControlMode::Pod] {
            let (mut p0, mut p1) = match_apps();
            if controls == ControlMode::Pod {
                p0.apply(Action::CycleControls);
            }
            let dt = 0.05;
            play(&mut p0, &mut p1, 20, dt, &HeldInput { thrust_forward: true, ..Default::default() });
            play(&mut p0, &mut p1, 10, dt, &HeldInput { thrust_right: true, ..Default::default() });
            let seen = &p1.world.as_ref().unwrap().ships[0];
            let (ahead, latest) = (seen.hull_ahead(), *seen.latest());
            let end = ahead.last().expect("the hull's next moves are settled");
            let pilot = latest.event();
            let offset = (seen.latest_hull().event().spatial() - pilot.spatial()).length() as f64;
            let reach = end.t - latest.t;
            assert!(reach > SHIP_RADIUS + offset, "{controls:?}: drawn only {reach} ahead");

            // Whatever the pilot does next, the hull flies as drawn.
            play(&mut p0, &mut p1, 40, dt, &HeldInput { thrust_back: true, thrust_up: true, ..Default::default() });
            let flown = &p0.world.as_ref().unwrap().ships[0];
            assert!(flown.latest().t > end.t);
            for a in &ahead {
                let b = flown.hull_at(a.t);
                let gap = (a.event().spatial() - b.event().spatial()).length();
                assert!(gap < 0.02, "{controls:?} at t = {}: drawn {:?}, flew {:?}", a.t, a.pos, b.pos);
            }
            assert!(ahead.iter().any(|a| a.vel != latest.vel), "{controls:?}: the hull turns ahead");
        }
    }

    /// Alone, switching the controls restarts the table.
    #[test]
    fn switching_controls_alone_restarts_the_table() {
        let mut app = billiards_app();
        app.step(0.5, &HeldInput::default());
        assert!(app.camera.proper_time > 0.0);
        app.apply(Action::CycleControls);
        assert_eq!(app.controls, ControlMode::Pod);
        assert_eq!(app.camera.proper_time, 0.0);
        app.apply(Action::CycleControls);
        assert_eq!(app.controls, ControlMode::Delayed);
    }

    /// A match starts paused: nothing moves, and the ship can't thrust or fire, until the
    /// other player's hello arrives.
    #[test]
    fn a_match_starts_paused_until_the_other_player_joins() {
        let (a, _b) = crate::multiplayer::link::LocalLink::pair();
        let scenes = vec![Scene4D::build(DemoScene::RelativisticBilliards)];
        let mut alone = App::new_multiplayer(scenes, 64, 40, 0, Box::new(a));
        let start = alone.camera.position;
        let shots = |app: &App| app.world.as_ref().unwrap().ships[0].shots;
        for _ in 0..40 {
            alone.apply(Action::Fire);
            alone.step(0.05, &HeldInput { thrust_forward: true, ..Default::default() });
        }
        assert_eq!(alone.camera.position, start, "time stands still before anyone joins");
        assert_eq!(alone.camera.proper_time, 0.0);
        assert_eq!(shots(&alone), 0, "no firing while alone");
        assert!(alone.is_waiting());
        assert_eq!(alone.banner(), Some(game::NOT_JOINED_BANNER));

        // Once both sides are there, play starts.
        let (mut p0, mut p1) = match_apps();
        for _ in 0..20 {
            p0.step(0.05, &HeldInput::default());
            p1.step(0.05, &HeldInput::default());
        }
        assert!(p0.camera.proper_time > 0.5 && p1.camera.proper_time > 0.5);
        assert_eq!(p0.banner(), None);
    }

    /// The HUD's latency line: the wall-clock round trip, and the two ships' separation.
    #[test]
    fn a_match_reports_its_latency() {
        let (mut p0, mut p1) = match_apps();
        assert_eq!(p0.net_metrics(), None, "nobody has joined yet");
        for _ in 0..2 {
            p0.step(0.05, &HeldInput::default());
            p1.step(0.05, &HeldInput::default());
        }
        assert!(p0.net_metrics().unwrap().starts_with("NET RTT -- | CM LEAD "));
        p0.tick_wall_clock(1.0); // pings
        p1.step(0.05, &HeldInput::default()); // answers
        p0.tick_wall_clock(0.25);
        p0.step(0.05, &HeldInput::default()); // hears the answer
        let line = p0.net_metrics().unwrap();
        assert!(line.starts_with("NET RTT 250MS | CM LEAD "), "{line}");
        assert!(line.contains("| INTERVAL 25.00LS | CM DIST 25.00LS |"), "{line}"); // the spawns
    }

    /// Laser shots, flashes and the worlds they shape are shared: both replicas agree. The
    /// players spawn 25 ls apart facing each other, so shots straight ahead hit 25 s later;
    /// each hit lights a flash, and lands on a player's scoreboard once news of it arrives
    /// there: at once on the hull that was hit, 25 s later back at the shooter.
    #[test]
    fn a_match_shares_lasers_hits_and_flashes() {
        let (mut p0, mut p1) = match_apps();
        p0.apply(Action::SelectScene(0)); // ignored in a match
        assert!(p0.is_dynamic());
        let mut boards = Vec::new();
        for frame in 0..1500 {
            if frame % 100 == 10 && frame < 500 {
                p0.apply(Action::Fire);
                p1.apply(Action::Fire);
            }
            if frame == 30 {
                p1.apply(Action::EmitFlash);
            }
            p0.step(0.05, &HeldInput::default());
            p1.step(0.05, &HeldInput::default());
            if frame % 50 == 0 {
                boards.push(p0.scoreboard().unwrap());
            }
        }
        let flashes = |app: &App, color| app.dynamics.flashes.iter().filter(|f| f.color == color).count();
        let white = p1.dynamics.flashes[0].color;
        assert_eq!(flashes(&p0, white), 1, "player 1's flash reached player 0");
        assert_eq!(flashes(&p0, crate::renderer::HIT_FLASH_COLOR), 10, "every hit lights a flash");
        let (w0, w1) = (p0.world.as_ref().unwrap(), p1.world.as_ref().unwrap());
        let shots = |w: &DynWorld, p: PlayerId| w.ships[p as usize].shots;
        assert!(shots(w0, 0) == 5 && shots(w0, 1) == 5 && shots(w1, 0) == 5 && shots(w1, 1) == 5);
        for p in &w0.pulses {
            let q = w1.pulses.iter().find(|q| (q.shooter, q.seq) == (p.shooter, p.seq)).unwrap();
            assert_eq!(p.end, q.end);
            assert!(matches!(p.end.unwrap().target, laser::Target::Ship(t) if t != p.shooter));
        }
        assert_eq!((&w0.scores[..], &w1.scores[..]), (&[5, 5][..], &[5, 5][..]));
        // Player 0 heard of the hits on their own hull long before those on player 1's.
        let at = |frame: usize| &boards[frame / 50];
        assert_eq!(at(400), "LASER HITS  YOU 0 : 0 THEM");
        assert_eq!(at(950), "LASER HITS  YOU 0 : 5 THEM");
        assert_eq!(at(1450), "LASER HITS  YOU 5 : 5 THEM");
        let mut compared = 0;
        for x in &w0.bodies {
            let y = w1.bodies.iter().find(|y| y.uid == x.uid).expect("same bodies in both worlds");
            for s in &x.history {
                if let Some(t) = y.at(s.step) {
                    assert_eq!(s, t, "body {:x} differs at step {}", x.uid, s.step);
                    compared += 1;
                }
            }
        }
        assert!(compared > 3000, "compared {compared}");
    }
}
