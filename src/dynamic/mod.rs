//! Causal Dynamic Mode (see `docs/DYNAMIC_MODE_PLAN.md`).
//!
//! The only source of non-determinism is the player, so an event is *decided* once it has left
//! the open future light cone `I+(O)` of the camera event `O`: no future input can reach it.
//! Every body's history is extended exactly that far — up to the causal frontier `dJ+(O)` —
//! and rendered from a fresh snapshot ([`DynWorld::build_scene`]).

pub mod billiards;
pub mod body;

use crate::bvh4d::Bvh4D;
use crate::mesh4d::{
    build_worldtube_cap, build_worldtube_simplices, build_worldtube_simplices_with_decided,
    sample_inertial_worldline, Mesh3D,
};
use crate::minkowski::{Vec3, Vec4};
use crate::renderer::{analytic_decision_age, Frontier};
use crate::scene::{DemoScene, Scene4D};
use body::{is_decided, Body, BodyState};

/// How far behind the camera (in lab time) histories are kept: beyond the renderers' maximum
/// ray length, nothing older can be seen.
pub const HISTORY_WINDOW: f64 = 200.0;

/// Render segments are split at least this often (in lab time), so that the per-vertex
/// `decided_at` interpolated along a merged segment stays close to the per-step stamps.
const MAX_SEGMENT_DT: f64 = 50.0;

const MATERIAL_RAIL: usize = 6;

pub struct DynWorld {
    pub demo: DemoScene,
    pub bodies: Vec<Body>,
    /// The ship's (camera's) recorded worldline, one entry per [`DynWorld::advance`]. Nothing
    /// reads it yet; interacting bodies (Phase 4+) will look up retarded ship states here.
    pub ship: Vec<Frontier>,
    /// The frontier the histories were last extended to.
    pub frontier: Frontier,
}

impl DynWorld {
    /// The world as it stands when the camera spawns at the demo's initial event, at rest.
    pub fn spawn(demo: DemoScene) -> Self {
        let frontier = Frontier {
            origin: demo.initial_camera_pos(),
            proper_time: 0.0,
        };
        Self::new(demo, frontier, Vec4::new(1.0, 0.0, 0.0, 0.0))
    }

    /// Start the world with the observer at `frontier`, moving with 4-velocity `u`. Everything
    /// outside `I+(O)` is already decided: histories are filled from [`HISTORY_WINDOW`] in the
    /// past up to the frontier, stamped with the analytic decision time as if the observer had
    /// always coasted with `u`.
    pub fn new(demo: DemoScene, frontier: Frontier, u: Vec4) -> Self {
        let mut bodies = match demo {
            DemoScene::RelativisticBilliards => billiards::pucks(),
            _ => panic!("{demo:?} is not a dynamic scene"),
        };
        let t_start = frontier.origin.t as f64 - HISTORY_WINDOW;
        for body in &mut bodies {
            // Bodies are specified at t = 0; walk back to the start of the window.
            let mut s = body.history[0];
            while s.t > t_start {
                s = body.step_back(&s);
            }
            body.history = vec![s];
            body.decided_at = vec![decision_time(&s, frontier, u)];
        }
        let mut world = Self {
            demo,
            bodies,
            ship: vec![frontier],
            frontier,
        };
        world.extend(frontier, |s| decision_time(s, frontier, u));
        world
    }

    /// Move the frontier to the camera's new event and finalize every step that just became
    /// decided. Returns whether any history grew.
    pub fn advance(&mut self, frontier: Frontier) -> bool {
        self.ship.push(frontier);
        let horizon = frontier.origin.t as f64 - HISTORY_WINDOW;
        let stale = self.ship.partition_point(|f| (f.origin.t as f64) < horizon);
        self.ship.drain(..stale.min(self.ship.len() - 1));
        self.extend(frontier, |_| frontier.proper_time)
    }

    /// Scheduler. Inert bodies depend on nothing but themselves, so a single pass that advances
    /// each body as far as the frontier allows is complete; interacting bodies (Phase 4) will
    /// need repeated passes until none makes progress.
    fn extend(&mut self, frontier: Frontier, stamp: impl Fn(&BodyState) -> f32) -> bool {
        self.frontier = frontier;
        let horizon = frontier.origin.t as f64 - HISTORY_WINDOW;
        let mut grew = false;
        for body in &mut self.bodies {
            loop {
                let next = body.step(body.latest());
                if !is_decided(&next, body.radius, frontier.origin) {
                    break;
                }
                body.decided_at.push(stamp(&next));
                body.history.push(next);
                grew = true;
            }
            let stale = body.history.partition_point(|s| s.t < horizon);
            let stale = stale.min(body.history.len() - 1);
            body.history.drain(..stale);
            body.decided_at.drain(..stale);
        }
        grew
    }

    /// Shift every event by `dt` in lab time (origin rebase, keeping `f32` coordinates precise).
    pub fn rebase(&mut self, dt: f64) {
        for body in &mut self.bodies {
            for s in &mut body.history {
                s.t += dt;
            }
        }
        for f in self.ship.iter_mut().chain(std::iter::once(&mut self.frontier)) {
            f.origin.t += dt as f32;
        }
    }

    /// Snapshot the world as renderable geometry: the static table plus every body's tube up
    /// to its frontier tip, sealed with an end cap.
    pub fn build_scene(&self) -> Scene4D {
        let o = self.frontier.origin;
        let window = HISTORY_WINDOW as f32;
        let backdrop =
            |pos| sample_inertial_worldline(pos, Vec3::default(), o.t - window, o.t + window, 1);

        let ground = Mesh3D::ground_grid(24.0, 24.0, billiards::TABLE_Y, 4, 4);
        let mut simplices = build_worldtube_simplices(&ground, &backdrop(Vec3::default()), 0);
        for (rail, center) in billiards::rails() {
            simplices.extend(build_worldtube_simplices(&rail, &backdrop(center), MATERIAL_RAIL));
        }

        for body in &self.bodies {
            let keep = render_sample_indices(&body.history);
            let samples: Vec<_> = keep.iter().map(|&i| body.history[i].sample()).collect();
            let decided: Vec<f32> = keep.iter().map(|&i| body.decided_at[i]).collect();
            simplices.extend(build_worldtube_simplices_with_decided(
                &body.mesh,
                &samples,
                Some(&decided),
                body.material_id,
            ));
            simplices.extend(build_worldtube_cap(
                &body.mesh,
                samples.last().expect("history is never empty"),
                *decided.last().expect("history is never empty"),
                body.material_id,
            ));
        }

        Scene4D {
            demo: self.demo,
            bvh: Bvh4D::build(simplices),
            materials: Scene4D::common_materials(),
            light_pos: Vec3::new(0.0, 6.5, 4.0),
            t_min: o.t - window,
            t_max: o.t + window,
        }
    }
}

/// Camera proper time at which `state` left `I+` of an observer that reached `frontier` coasting
/// with 4-velocity `u` (never later than the frontier itself).
fn decision_time(state: &BodyState, frontier: Frontier, u: Vec4) -> f32 {
    let age = analytic_decision_age(state.event(), frontier.origin, u);
    frontier.proper_time - age.max(0.0)
}

/// History steps to keep as worldtube samples: the ends, both sides of every velocity change
/// (so straight stretches merge into single prisms and kinks stay sharp), and at least one
/// sample every [`MAX_SEGMENT_DT`].
fn render_sample_indices(history: &[BodyState]) -> Vec<usize> {
    let mut keep = vec![0];
    for i in 1..history.len() {
        let last = *keep.last().unwrap();
        let kink = history[i].vel != history[i - 1].vel;
        if kink && last != i - 1 {
            keep.push(i - 1);
        }
        let last_t = history[*keep.last().unwrap()].t;
        if kink || i == history.len() - 1 || history[i].t - last_t >= MAX_SEGMENT_DT {
            keep.push(i);
        }
    }
    keep
}

#[cfg(test)]
mod tests {
    use super::body::STEP_DT;
    use super::*;
    use crate::minkowski::Tetrad;

    /// A camera worldline that thrusts sideways, then forward, then coasts: `n` frontiers
    /// `dtau` apart, starting at the billiards spawn event.
    fn camera_path(n: usize, dtau: f32) -> Vec<Frontier> {
        let mut tetrad = Tetrad::rest();
        let mut origin = DemoScene::RelativisticBilliards.initial_camera_pos();
        let mut tau = 0.0;
        (0..n)
            .map(|i| {
                let frac = i as f32 / n as f32;
                if frac < 0.3 {
                    tetrad.boost_local(Vec3::new(0.02, 0.0, 0.0));
                } else if frac < 0.6 {
                    tetrad.boost_local(Vec3::new(0.0, 0.0, 0.015));
                }
                origin += tetrad.e[0] * dtau;
                tau += dtau;
                Frontier { origin, proper_time: tau }
            })
            .collect()
    }

    fn assert_causal(world: &DynWorld) {
        let o = world.frontier;
        for (b, body) in world.bodies.iter().enumerate() {
            assert_eq!(body.history.len(), body.decided_at.len());
            for (s, &d) in body.history.iter().zip(&body.decided_at) {
                assert!(is_decided(s, body.radius, o.origin), "body {b} step at t={} undecided", s.t);
                assert!(d <= o.proper_time, "body {b} stamped in the future");
            }
            assert!(
                body.decided_at.windows(2).all(|w| w[0] <= w[1]),
                "body {b}: decided_at must be non-decreasing"
            );
            // The scheduler stops exactly at the frontier: the next step is still undecided.
            assert!(!is_decided(&body.step(body.latest()), body.radius, o.origin));
        }
    }

    #[test]
    fn histories_stop_exactly_at_the_frontier() {
        let mut world = DynWorld::spawn(DemoScene::RelativisticBilliards);
        assert_causal(&world);
        let path = camera_path(400, 0.05);
        for (i, f) in path.iter().enumerate() {
            world.advance(*f);
            if i % 37 == 0 {
                assert_causal(&world);
            }
        }
        assert_causal(&world);
        // Far bodies are known further into the lab future than near ones.
        let o = world.frontier.origin;
        for body in &world.bodies {
            let s = body.latest();
            let dist = (s.event().spatial() - o.spatial()).length() as f64;
            assert!(s.t - (o.t as f64) > dist - 2.0 * PUCK_REACH, "tip lags far behind the frontier");
        }
    }

    const PUCK_REACH: f64 = billiards::PUCK_RADIUS * 3.0 + STEP_DT;

    /// Histories depend only on physics, not on how the frontier's motion was chopped into
    /// frames: advancing in many small or a few large steps to the same event agrees bit for bit.
    #[test]
    fn history_independent_of_frame_chunking() {
        let path = camera_path(300, 0.07);
        let mut fine = DynWorld::spawn(DemoScene::RelativisticBilliards);
        for f in &path {
            fine.advance(*f);
        }
        let mut coarse = DynWorld::spawn(DemoScene::RelativisticBilliards);
        for f in path.iter().skip(42).step_by(43).chain(path.last()) {
            coarse.advance(*f);
        }
        for (a, b) in fine.bodies.iter().zip(&coarse.bodies) {
            assert_eq!(a.history, b.history);
        }
    }

    #[test]
    fn history_is_pruned_behind_the_camera_and_survives_rebase() {
        let mut world = DynWorld::spawn(DemoScene::RelativisticBilliards);
        let path = camera_path(60, 5.0); // 300 s of camera time
        for f in &path {
            world.advance(*f);
        }
        let horizon = world.frontier.origin.t as f64 - HISTORY_WINDOW;
        for body in &world.bodies {
            assert!(body.history[0].t >= horizon && body.history[0].t < horizon + STEP_DT);
        }

        world.rebase(-1000.0);
        assert_causal(&world);
        let mut moved = world.frontier;
        moved.origin += Vec4::new(1.0, 0.0, 0.0, 0.0);
        moved.proper_time += 1.0;
        assert!(world.advance(moved));
        assert_causal(&world);
    }

    #[test]
    fn pucks_stay_on_the_table_and_bounce() {
        let mut world = DynWorld::spawn(DemoScene::RelativisticBilliards);
        let mut origin = world.frontier.origin;
        let mut flips = 0;
        for i in 1..=200 {
            origin.t += 1.0;
            world.advance(Frontier { origin, proper_time: i as f32 });
            for body in &world.bodies {
                let r = body.radius;
                for w in body.history.windows(2) {
                    let [x, _, z] = w[1].pos;
                    assert!(x.abs() <= billiards::HALF_X - r + 1e-9);
                    assert!(z >= billiards::Z_MIN + r - 1e-9 && z <= billiards::Z_MAX - r + 1e-9);
                    flips += (w[0].vel != w[1].vel) as usize;
                }
            }
        }
        assert!(flips > 0, "movers should bounce off the rails");
    }

    #[test]
    fn step_back_inverts_step() {
        let world = DynWorld::spawn(DemoScene::RelativisticBilliards);
        for body in &world.bodies {
            for s in body.history.iter().step_by(97) {
                let back = body.step_back(&body.step(s));
                assert!((back.t - s.t).abs() < 1e-9 && (back.tau - s.tau).abs() < 1e-9);
                for k in 0..3 {
                    assert!((back.pos[k] - s.pos[k]).abs() < 1e-9);
                    assert_eq!(back.vel[k], s.vel[k]);
                }
            }
        }
    }

    /// Render snapshots stay small enough for a 10 Hz rebuild: straight stretches merge into
    /// a few long prisms.
    #[test]
    fn render_snapshot_merges_straight_segments() {
        let world = DynWorld::spawn(DemoScene::RelativisticBilliards);
        let scene = world.build_scene();
        let steps: usize = world.bodies.iter().map(|b| b.history.len()).sum();
        let simplices = scene.bvh.simplices.len();
        println!("{steps} history steps -> {simplices} simplices");
        assert!(simplices < 40_000, "{simplices} simplices");
        let keep = render_sample_indices(&world.bodies[0].history);
        let resting_span = world.bodies[0].latest().t - world.bodies[0].history[0].t;
        assert!(keep.len() <= (resting_span / MAX_SEGMENT_DT) as usize + 2);
    }
}

#[cfg(test)]
mod bench {
    /// Snapshot cost, for judging the 10 Hz rebuild budget (`cargo test --release -- --ignored`).
    #[test]
    #[ignore]
    fn time_build_scene() {
        let world = super::DynWorld::spawn(crate::scene::DemoScene::RelativisticBilliards);
        let t0 = std::time::Instant::now();
        for _ in 0..20 {
            std::hint::black_box(world.build_scene());
        }
        println!("build_scene: {:?}", t0.elapsed() / 20);
    }
}
