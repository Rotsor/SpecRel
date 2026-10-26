//! Benchmarks, as ignored tests (the crate has no library target for `cargo bench`). Run them
//! optimized, one at a time so they don't share the cores:
//!
//! ```text
//! cargo test --release -- --ignored --nocapture --test-threads=1 bench_
//! ```
//!
//! The headline figure is the worst frame: a scene rebuild runs synchronously in the frame that
//! triggers it, so a frame takes up to the slowest (incremental) rebuild plus a render. The
//! timings are noisy on a loaded machine; compare runs on an idle one.

use std::time::Instant;

use crate::dynamic::helm::{self, ControlMode, Order};
use crate::dynamic::ship::{render_indices, segment_stray};
use crate::dynamic::scene_cache::CacheStats;
use crate::dynamic::{self, billiards, laser, DynWorld, PlayerId, SceneCache, ShipSample, MAX_SEGMENT_DT};
use crate::gpu::PackedSceneGpuData;
use crate::minkowski::{Vec3, Vec4};
use crate::renderer::{
    render_frame, Camera4D, FrameDynamics, Frontier, RenderConfig, RenderMode, ShadowMode, ViewMode,
    PROPHECY_DELTA_DEFAULT,
};
use crate::scene::{DemoScene, Scene4D};

/// Frames per second of match time the pilots are simulated at.
const SIM_FPS: f64 = 60.0;
/// The native frontend's frame cap, and the app's `DYNAMIC_REBUILD_INTERVAL`.
const TARGET_FPS: f64 = 60.0;
const REBUILD_INTERVAL: f64 = 0.1;
const BUILD_ITERS: usize = 20;
const RENDER_ITERS: usize = 5;
const RENDER_SIZE: (usize, usize) = (320, 200);
/// How far behind the other ship's spawn the viewer floats, facing it and the table beyond
/// (`c = 1` units).
const VIEW_DISTANCE: f32 = 2.0;

/// A pilot flying by hand: its event, clock and rest frame, as `App` keeps the camera.
struct Pilot {
    camera: Camera4D,
}

impl Pilot {
    fn new(event: Vec4) -> Self {
        Self { camera: Camera4D::new(event, 65.0) }
    }

    /// One frame of `dt` lab time (both pilots stay level in lab time, so neither ever enters
    /// the other's future cone), thrusting along the lab direction `thrust` (if any) at
    /// `accel` per second of the pilot's clock, as the app's control pod does.
    fn fly(&mut self, dt: f32, thrust: Option<Vec3>, accel: f32) -> ShipSample {
        let c = &mut self.camera;
        let d_tau = dt / c.tetrad.e[0].t;
        if let Some(dir) = thrust {
            Order::Thrust(Vec4::new(0.0, dir.x, dir.y, dir.z)).execute(&mut c.tetrad, accel * d_tau);
        }
        let d_tau = dt / c.tetrad.e[0].t;
        c.proper_time += d_tau;
        c.position += c.tetrad.e[0] * d_tau;
        ShipSample::new(Frontier { origin: c.position, proper_time: c.proper_time }, c.tetrad.e[0])
    }
}

/// The pod pilot's maneuvers: legs of [`LEG`] seconds, each a push along one of [`DIRS`] for
/// [`PUSH`] s, a coast, the opposite push (back to rest) half a leg in, and a coast. Six legs bring it
/// back where it started, so it stays within about a light-second of its spawn. The hull
/// keeps chasing the pod, so it is rarely flying straight.
const LEG: f64 = 5.0;
const PUSH: f64 = 0.7;
const DIRS: [[f32; 3]; 6] =
    [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0], [0.0, -1.0, 0.0]];

fn pod_thrust(t: f64) -> Option<Vec3> {
    let leg = (t / LEG).floor() as usize;
    let s = t - leg as f64 * LEG;
    let [x, y, z] = DIRS[leg % DIRS.len()];
    let d = Vec3::new(x, y, z);
    if s < PUSH {
        Some(d)
    } else if (0.5 * LEG..0.5 * LEG + PUSH).contains(&s) {
        Some(d * -1.0)
    } else {
        None
    }
}

/// A two-player match of the billiards scene, rendered for player 1 (the viewer, coasting
/// [`VIEW_DISTANCE`] behind player 0's spawn and facing it), while player 0 flies a control pod
/// through [`pod_thrust`]'s maneuvers. Both shoot a laser every few seconds: the viewer at
/// the pod, the pod down the table.
struct Scenario {
    world: DynWorld,
    pod: Pilot,
    viewer: Pilot,
    frames: u64,
}

const POD: PlayerId = 0;
const VIEWER: PlayerId = 1;

impl Scenario {
    fn new() -> Self {
        let near = billiards::spawn_event(POD);
        let viewer_at = near - Vec4::new(0.0, 0.0, 0.0, VIEW_DISTANCE);
        let spawns = [ShipSample::at_rest(near), ShipSample::at_rest(viewer_at)];
        let mut world = DynWorld::new_match(DemoScene::RelativisticBilliards, &spawns, VIEWER);
        world.set_controls(POD, ControlMode::Pod);
        Self { world, pod: Pilot::new(near), viewer: Pilot::new(viewer_at), frames: 0 }
    }

    fn time(&self) -> f64 {
        self.frames as f64 / SIM_FPS
    }

    /// Play on until `secs` of match time have passed, calling `snapshot` with the world and the match time every
    /// [`REBUILD_INTERVAL`], as the app snapshots it.
    fn play_until_with(&mut self, secs: f64, mut snapshot: impl FnMut(&DynWorld, f64)) {
        let per_snapshot = (REBUILD_INTERVAL * SIM_FPS).round() as u64;
        let dt = (1.0 / SIM_FPS) as f32;
        while self.time() < secs {
            let t = self.time();
            let pod = self.pod.fly(dt, pod_thrust(t), helm::pod_accel());
            self.world.record(POD, pod);
            let me = self.viewer.fly(dt, None, 0.0);
            self.world.advance(me.frontier(), me.u());
            let every = |period: u64, phase: u64| self.frames % (period * SIM_FPS as u64) == phase * SIM_FPS as u64;
            if every(4, 1) {
                let d = (pod.event() - me.event()).spatial().normalized();
                self.world.fire_laser(VIEWER, me, [d.x, d.y, d.z].map(|c| c as f64));
            }
            if every(5, 3) {
                self.world.fire_laser(POD, pod, [0.0, 0.0, 1.0]);
            }
            self.frames += 1;
            if self.frames % per_snapshot == 0 {
                snapshot(&self.world, self.time());
            }
        }
    }

    /// Play on until `secs` of match time have passed.
    fn play_until(&mut self, secs: f64) {
        self.play_until_with(secs, |_, _| {});
    }

    /// The viewer's camera and frame dynamics, as `App` would render them.
    fn view(&self) -> (Camera4D, FrameDynamics) {
        let u = self.viewer.camera.tetrad.e[0];
        let dynamics = FrameDynamics {
            frontier: Some(self.world.frontier),
            frontier_lag: self.world.frontier_lag(u),
            remote_origins: self.world.remote_origins(),
            ..Default::default()
        };
        (self.viewer.camera, dynamics)
    }
}

/// Median, 99th percentile (nearest rank), maximum and mean of `xs`.
fn stats(xs: &[f64]) -> [f64; 4] {
    let mut v = xs.to_vec();
    v.sort_by(f64::total_cmp);
    let p99 = v[((0.99 * v.len() as f64).ceil() as usize).clamp(1, v.len()) - 1];
    [v[v.len() / 2], p99, v[v.len() - 1], v.iter().sum::<f64>() / v.len() as f64]
}

/// How long [`play_frames`] plays from each history length (match time).
const PLAY_SECS: f64 = 3.0;

/// The longest frame the frontends report (`App`'s `MAX_DT_WALL`).
const MAX_DT_WALL: f64 = 0.1;

/// Play the match on in `scenario` for [`PLAY_SECS`] of match time, frame by frame as the app
/// does: each frame simulates its `dt_wall` (the previous frame's length, clamped to
/// [`MAX_DT_WALL`]), rebuilds `scene` incrementally through `cache` once the frames' `dt_wall`
/// adds up to [`REBUILD_INTERVAL`] (the world grows every frame; the interval counts from the
/// end of the last rebuild), and renders it with `render` (which returns its milliseconds).
/// Frames are at least `1 / TARGET_FPS` long. Returns each frame's work in ms (simulation +
/// rebuild + render), and the rebuilds' ms.
fn play_frames(
    scenario: &mut Scenario,
    cache: &mut SceneCache,
    scene: &mut Scene4D,
    mut render: impl FnMut(&Scene4D, &Scenario) -> f64,
) -> (Vec<f64>, Vec<f64>) {
    let (mut frames, mut rebuilds) = (Vec::new(), Vec::new());
    let (mut timer, mut dt) = (0.0, 1.0 / TARGET_FPS);
    let end = scenario.time() + PLAY_SECS;
    while scenario.time() < end {
        let dt_wall = dt.clamp(0.001, MAX_DT_WALL);
        let start = Instant::now();
        scenario.play_until(scenario.time() + dt_wall);
        timer += dt_wall;
        if timer >= REBUILD_INTERVAL {
            let rebuild = Instant::now();
            scenario.world.rebuild_scene(cache, scene);
            let rebuild_ms = ms(rebuild);
            rebuilds.push(rebuild_ms);
            timer = -(rebuild_ms / 1e3).min(MAX_DT_WALL);
        }
        render(scene, scenario);
        let work = ms(start);
        frames.push(work);
        dt = f64::max(1.0 / TARGET_FPS, work / 1e3);
    }
    (frames, rebuilds)
}

fn ms(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1e3
}

/// The `SCENE+BVH REBUILD` phase with the camera close to another player's control pod (users
/// saw it take ~90% of the frame), at several lengths of match history: the incremental
/// rebuild the app makes every [`REBUILD_INTERVAL`] (`DynWorld::rebuild_scene`), one from an
/// empty cache (a rebase, or a world entered late), and for reference `DynWorld::build_scene`
/// broken down into the geometry it generates (per part) and the BVH build; then the CPU trace
/// of one frame from that camera, and a few seconds of frames played as the app plays them.
#[test]
#[ignore = "benchmark: run with --release --ignored"]
fn bench_scene_rebuild_near_a_control_pod() {
    let parts = [
        dynamic::SCENE_TABLE,
        dynamic::SCENE_BODIES,
        dynamic::SCENE_PULSES,
        dynamic::SCENE_HULL_AHEAD,
        dynamic::SCENE_HULL,
        dynamic::SCENE_PILOT,
    ];
    let cfg = RenderConfig {
        width: RENDER_SIZE.0,
        height: RENDER_SIZE.1,
        mode: RenderMode::RelativisticShaded,
        shadow_mode: ShadowMode::LorentzInvariant,
        slice_mode: ViewMode::PastLightCone,
        prophecy_delta: PROPHECY_DELTA_DEFAULT,
    };
    println!(
        "Scene rebuild near a control pod: {BUILD_ITERS} builds, {RENDER_ITERS} CPU renders at {}x{} on {} threads.",
        cfg.width,
        cfg.height,
        rayon::current_num_threads()
    );
    println!("Times in ms: p50 / p99 / max / mean. The rebuild runs synchronously in the frame that triggers it, so");
    println!("the worst frame is about the worst incremental rebuild (all of them since the last history length)");
    println!("plus a render. Then {PLAY_SECS} s of frames played as `main.rs` paces them ({TARGET_FPS} fps cap, a rebuild");
    println!("every {REBUILD_INTERVAL} s of clamped dt_wall), with the CPU render.");
    let mut scenario = Scenario::new();
    // The incremental rebuild, as the app makes it every `REBUILD_INTERVAL` all along.
    let mut cache = SceneCache::default();
    let mut cached = scenario.world.build_scene();
    // The GPU's copy of the scene (as the web build keeps it), patched after every rebuild.
    let mut packed = PackedSceneGpuData::from_scene(&cached);
    let mut compactions = 0;
    for history in [5.0, 20.0, 60.0, 120.0, 215.0] {
        let history = f64::max(history, scenario.time());
        let since = scenario.time();
        let start = Instant::now();
        let mut incremental: Vec<(f64, f64, CacheStats, f64)> = Vec::new();
        let mut incremental_ms = 0.0;
        scenario.play_until_with(history, |world, t| {
            let start = Instant::now();
            world.rebuild_scene(&mut cache, &mut cached);
            let elapsed = ms(start);
            let pack_start = Instant::now();
            packed.update(&cached);
            let pack_ms = ms(pack_start);
            incremental_ms += elapsed + pack_ms;
            compactions += cache.stats.compacted as usize;
            incremental.push((t, elapsed, cache.stats, pack_ms));
        });
        let play_ms = ms(start) - incremental_ms;
        let world = &scenario.world;
        let other = &world.ships[POD as usize];
        let ahead = other.hull_ahead();
        let hull: Vec<_> = other.hull.iter().chain(&ahead).copied().collect();
        let hull_keep = render_indices(&hull, MAX_SEGMENT_DT);
        let hull_stray = hull_keep.windows(2).map(|w| segment_stray(&hull, w[0], w[1])).fold(0.0, f64::max);
        let drawn_pilot = render_indices(&other.track, MAX_SEGMENT_DT).len();

        // From an empty cache: what a rebase costs (every chunk moved in lab time).
        let full_ms: Vec<f64> = (0..BUILD_ITERS)
            .map(|_| {
                let mut scene = Scene4D { bvh: crate::bvh4d::Bvh4D::empty(), ..world.build_scene() };
                let start = Instant::now();
                world.rebuild_scene(&mut SceneCache::default(), &mut scene);
                ms(start)
            })
            .collect();

        let mut part_ms = vec![vec![0.0; BUILD_ITERS]; parts.len()];
        let mut part_simplices = vec![0; parts.len()];
        let (mut gen_ms, mut bvh_ms, mut total_ms) = (Vec::new(), Vec::new(), Vec::new());
        let mut scene = None;
        for iter in 0..BUILD_ITERS {
            let start = Instant::now();
            let (mut lap_start, mut count) = (start, 0);
            part_simplices.fill(0);
            let simplices = world.scene_simplices(&mut |part, n| {
                let i = parts.iter().position(|&p| p == part).expect("a known part");
                part_ms[i][iter] += ms(lap_start);
                part_simplices[i] += n - count;
                (lap_start, count) = (Instant::now(), n);
            });
            gen_ms.push(ms(start));
            let bvh_start = Instant::now();
            scene = Some(world.scene_of(simplices));
            bvh_ms.push(ms(bvh_start));
            total_ms.push(ms(start));
        }
        let scene = scene.unwrap();

        let (camera, dynamics) = scenario.view();
        let mut fb = vec![0u32; cfg.width * cfg.height];
        render_frame(&scene, &camera, &cfg, &dynamics, &mut fb); // warm-up
        let mut render = |scene: &Scene4D| -> Vec<f64> {
            (0..RENDER_ITERS)
                .map(|_| {
                    let start = Instant::now();
                    render_frame(scene, &camera, &cfg, &dynamics, &mut fb);
                    ms(start)
                })
                .collect()
        };
        // The same frame without the pod's hull cage, or without the laser pulses: how much
        // of the trace each costs.
        let mut render_without = |material: usize| {
            let mut simplices = world.scene_simplices(&mut |_, _| {});
            simplices.retain(|s| s.material_id != material);
            render(&world.scene_of(simplices))
        };
        let render_hullless_ms = render_without(other.hull_material());
        let render_pulseless_ms = render_without(laser::MATERIAL_PULSE);
        let render_scratch_ms = render(&scene);
        let render_ms = render(&cached);
        let lit = fb.iter().filter(|&&px| px != 0).count();
        let pack_ms = |scene: &Scene4D| -> Vec<f64> {
            (0..RENDER_ITERS)
                .map(|_| {
                    let start = Instant::now();
                    std::hint::black_box(PackedSceneGpuData::from_scene(scene));
                    ms(start)
                })
                .collect()
        };
        let pack_scratch_ms = pack_ms(&scene);
        let (in_tree, tree_nodes) = cached.bvh.in_tree();
        let live = in_tree.len();
        if history < dynamic::HISTORY_WINDOW {
            // (Once tracks are pruned, the incremental scan keeps its earlier choices.)
            assert_eq!(live, scene.bvh.simplices.len());
        }
        // Every incremental rebuild since the last history length (but the match's first
        // second, which fills the cache).
        let recent: Vec<_> = incremental.iter().filter(|r| r.0 > since.max(1.0)).collect();
        let recent_ms: Vec<f64> = recent.iter().map(|r| r.1).collect();
        let patch_ms: Vec<f64> = recent.iter().map(|r| r.3).collect();
        let mean_of = |f: &dyn Fn(&CacheStats) -> usize| {
            recent.iter().map(|r| f(&r.2)).sum::<usize>() as f64 / recent.len() as f64
        };

        let row = |label: &str, xs: &[f64], extra: String| {
            let [p50, p99, max, mean] = stats(xs);
            println!("  {label:<26} {p50:>8.2} / {p99:>8.2} / {max:>8.2} / {mean:>8.2}  {extra}");
        };
        println!();
        println!(
            "History {history:.0} s ({} frames, simulated in {play_ms:.0} ms): pod track {} samples, hull drawn \
             in {} samples ({} ahead, straying up to {hull_stray:.4}), pilot in {drawn_pilot}; {} bodies, {} pulses",
            scenario.frames,
            other.track.len(),
            hull_keep.len(),
            ahead.len(),
            world.bodies.len(),
            world.pulses.len(),
        );
        row(
            "incremental rebuild",
            &recent_ms,
            format!(
                "{} rebuilds; per rebuild {:.0} chunks ({:.0} simplices) built, {:.0} ({:.0}) reused",
                recent.len(),
                mean_of(&|s| s.chunks_built),
                mean_of(&|s| s.simplices_built),
                mean_of(&|s| s.chunks_reused),
                mean_of(&|s| s.simplices_reused),
            ),
        );
        row(" GPU pack, patched", &patch_ms, String::new());
        row("rebuild from an empty cache", &full_ms, String::new());
        row("render (CPU)", &render_ms, format!("{lit} lit px of {}", fb.len()));
        row(
            " of build_scene's",
            &render_scratch_ms,
            format!(
                "{tree_nodes} nodes in the tree; arena {} nodes, {} simplices; {compactions} compactions so far",
                cached.bvh.nodes().len(),
                cached.bvh.simplices.len()
            ),
        );
        row(" without the hull", &render_hullless_ms, String::new());
        row(" without the pulses", &render_pulseless_ms, String::new());
        row("build_scene total", &total_ms, format!("{} simplices", scene.bvh.simplices.len()));
        row(" simplex generation", &gen_ms, String::new());
        for (i, part) in parts.iter().enumerate() {
            row(&format!("   {part}"), &part_ms[i], format!("{} simplices", part_simplices[i]));
        }
        row(" Bvh4D::build", &bvh_ms, format!("{} nodes", scene.bvh.nodes().len()));
        row(" GPU pack, from scratch", &pack_scratch_ms, String::new());
        let [rebuild_p50, rebuild_p99, rebuild_max, _] = stats(&recent_ms);
        let render_p50 = stats(&render_ms)[0];
        let worst = rebuild_max + render_p50;
        println!("  WORST FRAME (max incremental rebuild + p50 render): {worst:.1} ms; without the render: {rebuild_max:.1} ms");

        // Play on, frame by frame: the geometry grows meanwhile.
        let mut fb = vec![0u32; cfg.width * cfg.height];
        let (frames, rebuilds) = play_frames(&mut scenario, &mut cache, &mut cached, |scene, scenario| {
            let (camera, dynamics) = scenario.view();
            let start = Instant::now();
            render_frame(scene, &camera, &cfg, &dynamics, &mut fb);
            ms(start)
        });
        packed = PackedSceneGpuData::from_scene(&cached);
        println!(
            "  played {PLAY_SECS} s on in {} frames with {} rebuilds (now {:.0} s of history):",
            frames.len(),
            rebuilds.len(),
            scenario.time()
        );
        row("frame (sim+rebuild+render)", &frames, String::new());
        row("rebuild in play", &rebuilds, String::new());
        let [_, frame_p99, frame_max, _] = stats(&frames);
        println!(
            "  SUMMARY {history:.0} s: {} simplices; rebuild p50 {rebuild_p50:.2} p99 {rebuild_p99:.2} max \
             {rebuild_max:.2}; full build p50 {:.1} max {:.1}; render {render_p50:.1}; worst frame {worst:.1}; played \
             max {frame_max:.1} p99 {frame_p99:.1}",
            live,
            stats(&full_ms)[0],
            stats(&full_ms)[2],
        );
    }
}

/// Every bit of `s` that rendering reads.
fn simplex_bits(s: &crate::mesh4d::Simplex4D) -> Vec<u32> {
    let v4 = |v: Vec4| [v.t, v.x, v.y, v.z];
    let mut f: Vec<f32> = Vec::new();
    for v in [s.v0, s.hyperplane_normal, s.dual_b1, s.dual_b2, s.dual_b3, s.surface_normal_4d, s.four_velocity] {
        f.extend(v4(v));
    }
    f.extend(v4(s.aabb.min).into_iter().chain(v4(s.aabb.max)));
    f.extend(s.uv.iter().flat_map(|&(u, v)| [u, v]));
    f.extend(s.tri_bary.iter().flat_map(|b| [b.x, b.y, b.z]));
    f.extend(s.proper_time.iter().chain(&s.decided_at));
    f.extend(s.tau_uv_grad.iter().flat_map(|&(u, v)| [u, v]));
    f.extend([s.uv_world_scale.0, s.uv_world_scale.1]);
    let mut bits: Vec<u32> = f.into_iter().map(f32::to_bits).collect();
    bits.push(s.material_id as u32);
    bits
}

fn sorted_simplices(scene: &Scene4D) -> Vec<Vec<u32>> {
    let mut all: Vec<_> = scene.bvh.in_tree().0.into_iter().map(simplex_bits).collect();
    all.sort();
    all
}

/// The viewer's past light cone rays in many directions: the share of them that see the
/// same thing (the same distance and material, or nothing) in both scenes.
fn same_hits(a: &Scene4D, b: &Scene4D, from: Vec4) -> f64 {
    let n = 4000;
    let golden = std::f32::consts::PI * (3.0 - 5f32.sqrt());
    let same = (0..n)
        .filter(|&i| {
            let z = 1.0 - 2.0 * (i as f32 + 0.5) / n as f32;
            let r = (1.0 - z * z).sqrt();
            let phi = golden * i as f32;
            let dir = Vec4::new(-1.0, r * phi.cos(), r * phi.sin(), z);
            let hit = |s: &Scene4D| s.bvh.trace(from, dir, 1e-3, 180.0).map(|h| (h.lambda, h.material_id));
            hit(a) == hit(b)
        })
        .count();
    same as f64 / n as f64
}

/// The incremental rebuild of a match that has been snapshotted all along makes the same
/// simplices as a rebuild from scratch, and its BVH finds the same hits. Most of it is reused.
#[test]
fn incremental_rebuild_matches_a_rebuild_from_scratch() {
    let mut scenario = Scenario::new();
    let mut cache = SceneCache::default();
    let mut cached = scenario.world.build_scene();
    let mut stats = Vec::new();
    // The GPU's copy, patched with what each rebuild wrote, stays the same as one packed afresh.
    let mut packed = PackedSceneGpuData::from_scene(&cached);
    let mut patched = 0;
    let same_as_packed_afresh = |packed: &PackedSceneGpuData, scene: &Scene4D| {
        let fresh = PackedSceneGpuData::from_scene(scene);
        let bytes = |p: &PackedSceneGpuData| {
            let b = |x: &[u8]| x.to_vec();
            (b(bytemuck::cast_slice(&p.nodes)), b(bytemuck::cast_slice(&p.geoms)), b(bytemuck::cast_slice(&p.attrs)))
        };
        bytes(packed) == bytes(&fresh) && packed.num_nodes == fresh.num_nodes
    };
    scenario.play_until_with(30.0, |world, _| {
        world.rebuild_scene(&mut cache, &mut cached);
        patched += packed.update(&cached) as usize;
        stats.push(cache.stats);
    });
    assert!(patched * 10 > stats.len() * 9, "{patched} of {} patched", stats.len());
    assert!(same_as_packed_afresh(&packed, &cached));
    let scratch = scenario.world.build_scene();
    assert_eq!(sorted_simplices(&cached), sorted_simplices(&scratch));
    let from = scenario.viewer.camera.position;
    assert_eq!(same_hits(&cached, &scratch, from), 1.0);
    let last = stats.last().unwrap();
    assert!(last.simplices_built * 5 < last.simplices_reused, "{last:?}");

    // Shifted in lab time, everything is rebuilt (and the arena compacted), and still the same.
    scenario.world.rebase(-7.25);
    scenario.viewer.camera.position.t -= 7.25;
    for _ in 0..3 {
        scenario.world.rebuild_scene(&mut cache, &mut cached);
        packed.update(&cached);
    }
    assert!(same_as_packed_afresh(&packed, &cached));
    let scratch = scenario.world.build_scene();
    assert_eq!(sorted_simplices(&cached), sorted_simplices(&scratch));
    assert_eq!(same_hits(&cached, &scratch, scenario.viewer.camera.position), 1.0);
}

/// Once the history window is full, the tracks are pruned at the front every step; the
/// incremental rebuild keeps reusing the rest, and shows what a rebuild from scratch shows
/// (which samples the pruned end of a track afresh).
#[test]
fn incremental_rebuild_keeps_up_with_pruning() {
    let mut scenario = Scenario::new();
    let mut cache = SceneCache::default();
    let mut cached = scenario.world.build_scene();
    let mut stats = Vec::new();
    let end = dynamic::HISTORY_WINDOW + 15.0;
    scenario.play_until_with(end, |world, t| {
        // Snapshot every second until near the end, then as often as the app does.
        if t > end - 2.0 || (t * 10.0).round() as u64 % 10 == 0 {
            world.rebuild_scene(&mut cache, &mut cached);
            stats.push(cache.stats);
        }
    });
    let other = &scenario.world.ships[POD as usize];
    assert!(other.track[0].t > 10.0, "the track was pruned");
    let scratch = scenario.world.build_scene();
    let last = stats.last().unwrap();
    assert!(last.simplices_built * 10 < last.simplices_reused, "{last:?}");
    // The arena's free room stays bounded.
    let live = cached.bvh.in_tree().0.len();
    assert!(last.arena_simplices < 2 * live + (1 << 15), "{live} live: {last:?}");
    let same = same_hits(&cached, &scratch, scenario.viewer.camera.position);
    assert!(same > 0.99, "{same}");
}
