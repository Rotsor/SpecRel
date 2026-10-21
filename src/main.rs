pub mod bvh4d;
pub mod gpu;
pub mod hud;
pub mod mesh4d;
pub mod minkowski;
pub mod renderer;
pub mod scene;
#[cfg(target_arch = "wasm32")]
pub mod web;

#[cfg(not(target_arch = "wasm32"))]
use gpu::GpuRenderer;
#[cfg(not(target_arch = "wasm32"))]
use hud::draw_hud;
#[cfg(not(target_arch = "wasm32"))]
use minkowski::{Vec3, Vec4};
#[cfg(not(target_arch = "wasm32"))]
use minifb::{Key, KeyRepeat, Window, WindowOptions};
#[cfg(not(target_arch = "wasm32"))]
use renderer::{
    render_frame, Camera4D, FrameDynamics, RenderConfig, RenderMode, ShadowMode, ViewMode,
};
#[cfg(not(target_arch = "wasm32"))]
use scene::{DemoScene, Scene4D};
#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;

#[cfg(not(target_arch = "wasm32"))]
const WIDTH: usize = 1280;
#[cfg(not(target_arch = "wasm32"))]
const HEIGHT: usize = 800;

#[cfg(not(target_arch = "wasm32"))]
const TIME_SCALES: [f32; 10] = [
    -4.0, -2.0, -1.0, -0.5, -0.25, 0.25, 0.5, 1.0, 2.0, 4.0,
];

/// Render one frame with the GPU backend if it's available and selected, else on the CPU.
#[cfg(not(target_arch = "wasm32"))]
fn render_with_backend(
    gpu: Option<&mut GpuRenderer>,
    use_gpu: bool,
    scenes: &[Scene4D],
    scene_idx: usize,
    camera: &Camera4D,
    cfg: &RenderConfig,
    dynamics: &FrameDynamics,
    framebuffer: &mut [u32],
) {
    match gpu {
        Some(g) if use_gpu => g.render(scene_idx, camera, cfg, dynamics, framebuffer),
        _ => render_frame(&scenes[scene_idx], camera, cfg, dynamics, framebuffer),
    }
}

#[cfg(target_arch = "wasm32")]
fn main() {}

#[cfg(not(target_arch = "wasm32"))]
fn main() {
    let args: Vec<String> = std::env::args().collect();

    println!("Building 5 Special Relativity 4D Spacetime Demo Scenes and 4D BVHs...");
    let build_start = Instant::now();
    let scenes: Vec<Scene4D> = DemoScene::ALL
        .iter()
        .map(|&demo| Scene4D::build(demo))
        .collect();
    let total_simplices: usize = scenes.iter().map(|s| s.bvh.simplices.len()).sum();
    println!(
        "Built {} scenes ({} total 4D tetrahedral simplices) in {:.2?}",
        scenes.len(),
        total_simplices,
        build_start.elapsed()
    );

    // GPU compute backend (wgpu). `--cpu` forces the CPU reference renderer.
    let mut gpu = if args.iter().any(|a| a == "--cpu") {
        println!("GPU backend disabled (--cpu); using CPU renderer");
        None
    } else {
        let t0 = Instant::now();
        match GpuRenderer::new(&scenes) {
            Ok(g) => {
                println!(
                    "GPU backend ready on {} (init + upload {:.2?})",
                    g.adapter_name(),
                    t0.elapsed()
                );
                Some(g)
            }
            Err(e) => {
                println!("GPU backend unavailable ({e}); falling back to CPU renderer");
                None
            }
        }
    };
    let mut use_gpu = gpu.is_some();

    let mut scene_idx: usize = 0; // Starts on Scene 1: Twin Paradox
    let mut camera = Camera4D::new(scenes[scene_idx].demo.initial_camera_pos(), 65.0);
    let mut cfg = RenderConfig {
        width: WIDTH,
        height: HEIGHT,
        mode: RenderMode::RelativisticShaded,
        shadow_mode: ShadowMode::LorentzInvariant,
        slice_mode: ViewMode::PastLightCone,
    };

    let mut framebuffer = vec![0u32; WIDTH * HEIGHT];

    if args.iter().any(|a| a == "--smoke-test") {
        let mut gpu_fb = vec![0u32; WIDTH * HEIGHT];
        for (i, scene) in scenes.iter().enumerate() {
            let cam = Camera4D::new(scene.demo.initial_camera_pos(), 65.0);
            let t0 = Instant::now();
            render_frame(scene, &cam, &cfg, &FrameDynamics::default(), &mut framebuffer);
            let dt = t0.elapsed();
            let non_zero = framebuffer.iter().filter(|&&px| px != 0).count();
            println!(
                "Scene {} [{}] ({} simplices): CPU rendered {}x{} in {:.2?} ({}/{} non-zero px)",
                i + 1,
                scene.demo.title(),
                scene.bvh.simplices.len(),
                WIDTH,
                HEIGHT,
                dt,
                non_zero,
                framebuffer.len()
            );

            if let Some(g) = gpu.as_mut() {
                g.render(i, &cam, &cfg, &FrameDynamics::default(), &mut gpu_fb); // warm-up (pipeline/driver caches)
                const RUNS: u32 = 10;
                let t0 = Instant::now();
                for _ in 0..RUNS {
                    g.render(i, &cam, &cfg, &FrameDynamics::default(), &mut gpu_fb);
                }
                let dt_gpu = t0.elapsed() / RUNS;
                let mut max_diff = 0u32;
                let mut differing = 0usize;
                for (&a, &b) in framebuffer.iter().zip(&gpu_fb) {
                    let d = (0..3)
                        .map(|c| ((a >> (8 * c)) & 0xFF).abs_diff((b >> (8 * c)) & 0xFF))
                        .max()
                        .unwrap();
                    max_diff = max_diff.max(d);
                    if d > 4 {
                        differing += 1;
                    }
                }
                println!(
                    "    GPU rendered in {:.2?} (avg of {} incl. readback, {:.1}x faster); vs CPU: {} px differ by >4 ({:.3}%), max channel diff {}",
                    dt_gpu,
                    RUNS,
                    dt.as_secs_f64() / dt_gpu.as_secs_f64(),
                    differing,
                    100.0 * differing as f64 / gpu_fb.len() as f64,
                    max_diff
                );
            }
        }
        return;
    }

    let mut res_scale: usize = if use_gpu { 2 } else { 1 };
    cfg.width = WIDTH * res_scale;
    cfg.height = HEIGHT * res_scale;
    framebuffer.resize(cfg.width * cfg.height, 0);

    let mut window = Window::new(
        "SpecRel 4D - Special Relativistic Spacetime Raytracer (Keys 1-5: Switch Paradox Scene)",
        WIDTH,
        HEIGHT,
        WindowOptions {
            resize: true,
            scale: minifb::Scale::X1,
            ..WindowOptions::default()
        },
    )
    .expect("Unable to open minifb window");

    window.set_target_fps(60);

    let mut paused_worldline = false;
    // Static scenes have no causal frontier or flashes; Dynamic Mode will drive these per frame.
    let dynamics = FrameDynamics::default();
    let mut time_scale_idx: usize = 7; // Default +1.0x forward
    let mut show_hud = true;
    let mut last_frame = Instant::now();
    let mut fps_smooth = 60.0f32;

    while window.is_open() && !window.is_key_down(Key::Escape) {
        let now = Instant::now();
        let dt_wall = now.duration_since(last_frame).as_secs_f32().clamp(0.001, 0.1);
        last_frame = now;
        fps_smooth = fps_smooth * 0.9 + (1.0 / dt_wall) * 0.1;

        // Scene selection via keys 1..=5 or N (Next) / B (Previous)
        let mut new_scene_idx = None;
        if window.is_key_pressed(Key::Key1, KeyRepeat::No) {
            new_scene_idx = Some(0);
        } else if window.is_key_pressed(Key::Key2, KeyRepeat::No) {
            new_scene_idx = Some(1);
        } else if window.is_key_pressed(Key::Key3, KeyRepeat::No) {
            new_scene_idx = Some(2);
        } else if window.is_key_pressed(Key::Key4, KeyRepeat::No) {
            new_scene_idx = Some(3);
        } else if window.is_key_pressed(Key::Key5, KeyRepeat::No) {
            new_scene_idx = Some(4);
        } else if window.is_key_pressed(Key::N, KeyRepeat::No) {
            new_scene_idx = Some((scene_idx + 1) % scenes.len());
        } else if window.is_key_pressed(Key::B, KeyRepeat::No) {
            new_scene_idx = Some((scene_idx + scenes.len() - 1) % scenes.len());
        }

        if let Some(idx) = new_scene_idx {
            scene_idx = idx;
            camera = Camera4D::new(scenes[scene_idx].demo.initial_camera_pos(), 65.0);
            paused_worldline = false;
        }

        let scene = &scenes[scene_idx];

        // Discrete toggles
        if window.is_key_pressed(Key::M, KeyRepeat::No) {
            cfg.mode = cfg.mode.next();
        }
        if window.is_key_pressed(Key::G, KeyRepeat::No) {
            cfg.slice_mode = cfg.slice_mode.next();
        }
        if window.is_key_pressed(Key::H, KeyRepeat::No) {
            cfg.shadow_mode = cfg.shadow_mode.next();
        }
        if window.is_key_pressed(Key::P, KeyRepeat::No) {
            paused_worldline = !paused_worldline;
        }
        if window.is_key_pressed(Key::T, KeyRepeat::No)
            || window.is_key_pressed(Key::Backspace, KeyRepeat::No)
        {
            // Flip time direction between Forward (+scale) and Rewind (-scale)
            time_scale_idx = TIME_SCALES.len() - 1 - time_scale_idx;
            paused_worldline = false;
        }
        if window.is_key_pressed(Key::Minus, KeyRepeat::No) {
            time_scale_idx = time_scale_idx.saturating_sub(1);
        }
        if window.is_key_pressed(Key::Equal, KeyRepeat::No) {
            time_scale_idx = (time_scale_idx + 1).min(TIME_SCALES.len() - 1);
        }
        if window.is_key_pressed(Key::Tab, KeyRepeat::No) {
            show_hud = !show_hud;
        }
        if window.is_key_pressed(Key::F, KeyRepeat::No) && gpu.is_some() {
            use_gpu = !use_gpu;
            res_scale = if use_gpu { 2 } else { 1 };
            cfg.width = WIDTH * res_scale;
            cfg.height = HEIGHT * res_scale;
            framebuffer.resize(cfg.width * cfg.height, 0);
        }
        if window.is_key_pressed(Key::Z, KeyRepeat::No)
            || window.is_key_pressed(Key::Y, KeyRepeat::No)
        {
            res_scale = if res_scale == 1 { 2 } else { 1 };
            cfg.width = WIDTH * res_scale;
            cfg.height = HEIGHT * res_scale;
            framebuffer.resize(cfg.width * cfg.height, 0);
        }
        if window.is_key_pressed(Key::R, KeyRepeat::No) {
            camera = Camera4D::new(scene.demo.initial_camera_pos(), 65.0);
            time_scale_idx = 7; // +1.0x
            paused_worldline = false;
        }
        if window.is_key_pressed(Key::X, KeyRepeat::No) {
            // Brake to lab rest frame via a single pure planar Lorentz boost
            camera.tetrad.boost_to(Vec4::new(1.0, 0.0, 0.0, 0.0));
        }
        if window.is_key_pressed(Key::V, KeyRepeat::No) {
            // Toggle between Lab Rest Frame and the moving object's Comoving Rest Frame (Pole / Train / Traveler)!
            if camera.tetrad.beta().length() > 0.1 {
                camera.tetrad.boost_to(Vec4::new(1.0, 0.0, 0.0, 0.0));
            } else {
                let target_beta = scene.demo.comoving_beta(camera.position.t);
                let target_u = Vec4::four_velocity_from_beta(target_beta);
                camera.tetrad.boost_to(target_u);
            }
        }

        // 1. Local camera orientation in observer's rest frame
        let rot_speed = 1.6 * dt_wall;
        let mut yaw = 0.0;
        let mut pitch = 0.0;
        let mut roll = 0.0;
        if window.is_key_down(Key::Left) {
            yaw -= rot_speed;
        }
        if window.is_key_down(Key::Right) {
            yaw += rot_speed;
        }
        if window.is_key_down(Key::Up) {
            pitch += rot_speed;
        }
        if window.is_key_down(Key::Down) {
            pitch -= rot_speed;
        }
        if window.is_key_down(Key::Q) {
            roll -= rot_speed;
        }
        if window.is_key_down(Key::E) {
            roll += rot_speed;
        }
        if yaw != 0.0 || pitch != 0.0 || roll != 0.0 {
            camera.tetrad.rotate_local(yaw, pitch, roll);
        }

        // 2. Proper acceleration thrusters (Lorentz boost in observer's local rest frame)
        let boost_rate = 1.25 * dt_wall; // Rapidity per second
        let mut d_rapidity = Vec3::default();
        if window.is_key_down(Key::W) {
            d_rapidity.z += boost_rate;
        }
        if window.is_key_down(Key::S) {
            d_rapidity.z -= boost_rate;
        }
        if window.is_key_down(Key::D) {
            d_rapidity.x += boost_rate;
        }
        if window.is_key_down(Key::A) {
            d_rapidity.x -= boost_rate;
        }
        if window.is_key_down(Key::Space) {
            d_rapidity.y += boost_rate;
        }
        if window.is_key_down(Key::C) || window.is_key_down(Key::LeftShift) {
            d_rapidity.y -= boost_rate;
        }
        camera.tetrad.boost_local(d_rapidity);

        // 3. Direct coordinate-space & time scrubbing (hold `[` to rewind/scrub back, `]` to scrub forward).
        // Because scrubbing `t` at fixed `(x, y, z)` is a coordinate translation rather than motion
        // along the observer's timelike worldline, `camera.proper_time` is left unchanged.
        let scrub_speed = 6.0 * dt_wall;
        let mut time_changed = false;
        if window.is_key_down(Key::LeftBracket) {
            camera.position.t -= scrub_speed;
            time_changed = true;
        }
        if window.is_key_down(Key::RightBracket) {
            camera.position.t += scrub_speed;
            time_changed = true;
        }
        if window.is_key_down(Key::I) {
            camera.position +=
                camera.tetrad.local_spatial_to_world(Vec3::new(0.0, 0.0, scrub_speed));
        }
        if window.is_key_down(Key::K) {
            camera.position +=
                camera.tetrad.local_spatial_to_world(Vec3::new(0.0, 0.0, -scrub_speed));
        }
        if window.is_key_down(Key::L) {
            camera.position +=
                camera.tetrad.local_spatial_to_world(Vec3::new(scrub_speed, 0.0, 0.0));
        }
        if window.is_key_down(Key::J) {
            camera.position +=
                camera.tetrad.local_spatial_to_world(Vec3::new(-scrub_speed, 0.0, 0.0));
        }
        if window.is_key_down(Key::U) {
            camera.position +=
                camera.tetrad.local_spatial_to_world(Vec3::new(0.0, scrub_speed, 0.0));
        }
        if window.is_key_down(Key::O) {
            camera.position +=
                camera.tetrad.local_spatial_to_world(Vec3::new(0.0, -scrub_speed, 0.0));
        }

        // 4. Advance observer along their timelike worldline: dX^mu = u^mu * d_tau
        let time_scale = TIME_SCALES[time_scale_idx];
        if !paused_worldline {
            let d_tau = dt_wall * time_scale;
            camera.proper_time += d_tau;
            camera.position += camera.tetrad.e[0] * d_tau;
            time_changed = true;
        }

        // Loop lab time seamlessly within [scene.t_min, scene.t_max]
        if time_changed {
            let t_loop = scene.t_max - scene.t_min;
            if camera.position.t > scene.t_max || camera.position.t < scene.t_min {
                camera.position.t =
                    scene.t_min + (camera.position.t - scene.t_min).rem_euclid(t_loop);
            }
        }

        render_with_backend(
            gpu.as_mut(),
            use_gpu,
            &scenes,
            scene_idx,
            &camera,
            &cfg,
            &dynamics,
            &mut framebuffer,
        );

        if show_hud {
            let backend = if gpu.is_some() && use_gpu { "GPU" } else { "CPU" };
            draw_hud(
                &mut framebuffer,
                scene.demo,
                &camera,
                &cfg,
                paused_worldline,
                time_scale,
                fps_smooth,
                backend,
            );
        }

        window
            .update_with_buffer(&framebuffer, cfg.width, cfg.height)
            .expect("Failed to update window buffer");
    }
}
