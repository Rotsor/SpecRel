pub mod app;
pub mod bvh4d;
pub mod dynamic;
pub mod gpu;
pub mod hud;
pub mod mesh4d;
pub mod minkowski;
pub mod multiplayer;
pub mod renderer;
pub mod scene;
#[cfg(target_arch = "wasm32")]
pub mod web;

#[cfg(not(target_arch = "wasm32"))]
use app::{Action, App, HeldInput};
#[cfg(not(target_arch = "wasm32"))]
use gpu::GpuRenderer;
#[cfg(not(target_arch = "wasm32"))]
use hud::{draw_banner, draw_hud};
#[cfg(not(target_arch = "wasm32"))]
use minifb::{Key, KeyRepeat, Window, WindowOptions};
#[cfg(not(target_arch = "wasm32"))]
use renderer::{render_frame, Camera4D, FrameDynamics};
#[cfg(not(target_arch = "wasm32"))]
use scene::{DemoScene, Scene4D};
#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;

#[cfg(not(target_arch = "wasm32"))]
const WIDTH: usize = 1280;
#[cfg(not(target_arch = "wasm32"))]
const HEIGHT: usize = 800;

/// Edge-triggered key bindings (`KeyRepeat::No`).
#[cfg(not(target_arch = "wasm32"))]
const KEY_ACTIONS: &[(Key, Action)] = &[
    (Key::Key1, Action::SelectScene(0)),
    (Key::Key2, Action::SelectScene(1)),
    (Key::Key3, Action::SelectScene(2)),
    (Key::Key4, Action::SelectScene(3)),
    (Key::Key5, Action::SelectScene(4)),
    (Key::Key6, Action::SelectScene(5)),
    (Key::N, Action::NextScene),
    (Key::B, Action::PrevScene),
    (Key::M, Action::CycleRenderMode),
    (Key::G, Action::CycleView),
    (Key::H, Action::CycleShadows),
    (Key::P, Action::TogglePause),
    (Key::T, Action::ToggleRewind),
    (Key::Backspace, Action::ToggleRewind),
    (Key::Minus, Action::Slower),
    (Key::Equal, Action::Faster),
    (Key::R, Action::Reset),
    (Key::X, Action::Brake),
    (Key::V, Action::MatchVelocity),
    (Key::Slash, Action::EmitFlash),
    (Key::Enter, Action::Fire),
    (Key::NumPadEnter, Action::Fire),
    (Key::Backslash, Action::CycleAmmo),
];

/// Key bindings that auto-repeat while held (`KeyRepeat::Yes`).
#[cfg(not(target_arch = "wasm32"))]
const REPEATING_KEY_ACTIONS: &[(Key, Action)] = &[
    (Key::PageUp, Action::ProphecyDeltaUp),
    (Key::PageDown, Action::ProphecyDeltaDown),
];

#[cfg(not(target_arch = "wasm32"))]
fn held_input(window: &Window) -> HeldInput {
    let down = |k: Key| window.is_key_down(k);
    HeldInput {
        yaw_left: down(Key::Left),
        yaw_right: down(Key::Right),
        pitch_up: down(Key::Up),
        pitch_down: down(Key::Down),
        roll_left: down(Key::Q),
        roll_right: down(Key::E),
        thrust_forward: down(Key::W),
        thrust_back: down(Key::S),
        thrust_right: down(Key::D),
        thrust_left: down(Key::A),
        thrust_up: down(Key::Space),
        thrust_down: down(Key::C) || down(Key::LeftShift),
        scrub_back: down(Key::LeftBracket),
        scrub_forward: down(Key::RightBracket),
        move_forward: down(Key::I),
        move_back: down(Key::K),
        move_right: down(Key::L),
        move_left: down(Key::J),
        move_up: down(Key::U),
        move_down: down(Key::O),
    }
}

/// `--host [PORT]` (player 1) or `--join HOST:PORT` (player 2): the link to the other player of
/// a two-player match, if one was asked for. Exits on a bad address.
#[cfg(not(target_arch = "wasm32"))]
fn multiplayer_link(args: &[String]) -> Option<(dynamic::PlayerId, Box<dyn multiplayer::link::Link>)> {
    use multiplayer::tcp::TcpLink;
    const DEFAULT_PORT: u16 = 7777;
    let value_after = |flag: &str| {
        let i = args.iter().position(|a| a == flag)?;
        Some(args.get(i + 1).filter(|v| !v.starts_with("--")).cloned())
    };
    let link = if let Some(port) = value_after("--host") {
        let port = port.map_or(Ok(DEFAULT_PORT), |p| p.parse()).unwrap_or_else(|_| {
            eprintln!("--host takes a port number");
            std::process::exit(2)
        });
        println!("Hosting a 2-player match on port {port}; the other player runs: --join <this-host>:{port}");
        (0, TcpLink::host(port))
    } else if let Some(addr) = value_after("--join") {
        let addr = addr.unwrap_or_else(|| format!("127.0.0.1:{DEFAULT_PORT}"));
        println!("Joining the 2-player match at {addr}...");
        (1, TcpLink::join(&addr))
    } else {
        return None;
    };
    match link {
        (player, Ok(link)) => Some((player, Box::new(link))),
        (_, Err(e)) => {
            eprintln!("Could not start the match: {e}");
            std::process::exit(1)
        }
    }
}

#[cfg(target_arch = "wasm32")]
fn main() {}

#[cfg(not(target_arch = "wasm32"))]
fn main() {
    let args: Vec<String> = std::env::args().collect();

    println!("Building {} demo scenes and 4D BVHs...", DemoScene::ALL.len());
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

    let mut app = match multiplayer_link(&args) {
        Some((player, link)) => App::new_multiplayer(scenes, WIDTH, HEIGHT, player, link),
        None => App::new(scenes, WIDTH, HEIGHT),
    };
    let mut framebuffer = vec![0u32; WIDTH * HEIGHT];

    if args.iter().any(|a| a == "--smoke-test") {
        let cfg = &app.cfg;
        let mut gpu_fb = vec![0u32; WIDTH * HEIGHT];
        for (i, scene) in app.scenes.iter().enumerate() {
            let cam = Camera4D::new(scene.demo.initial_camera_pos(), 65.0);
            let t0 = Instant::now();
            render_frame(scene, &cam, cfg, &FrameDynamics::default(), &mut framebuffer);
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
                g.render(i, &cam, cfg, &FrameDynamics::default(), &mut gpu_fb); // warm-up (pipeline/driver caches)
                const RUNS: u32 = 10;
                let t0 = Instant::now();
                for _ in 0..RUNS {
                    g.render(i, &cam, cfg, &FrameDynamics::default(), &mut gpu_fb);
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
    let set_resolution = |app: &mut App, framebuffer: &mut Vec<u32>, res_scale: usize| {
        app.cfg.width = WIDTH * res_scale;
        app.cfg.height = HEIGHT * res_scale;
        framebuffer.resize(app.cfg.width * app.cfg.height, 0);
    };
    set_resolution(&mut app, &mut framebuffer, res_scale);

    let mut window = Window::new(
        "SpecRel 4D - Special Relativistic Spacetime Raytracer (Keys 1-6: Switch Scene)",
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

    let mut show_hud = true;
    let mut uploaded_generation = app.scene_generation;
    let mut last_frame = Instant::now();
    let mut fps_smooth = 60.0f32;

    while window.is_open() && !window.is_key_down(Key::Escape) {
        let now = Instant::now();
        let dt_wall = now.duration_since(last_frame).as_secs_f32().clamp(0.001, 0.1);
        last_frame = now;
        fps_smooth = fps_smooth * 0.9 + (1.0 / dt_wall) * 0.1;

        for &(key, action) in KEY_ACTIONS {
            if window.is_key_pressed(key, KeyRepeat::No) {
                app.apply(action);
            }
        }
        for &(key, action) in REPEATING_KEY_ACTIONS {
            if window.is_key_pressed(key, KeyRepeat::Yes) {
                app.apply(action);
            }
        }

        // Frontend-only toggles: HUD, rendering backend and resolution.
        if window.is_key_pressed(Key::Tab, KeyRepeat::No) {
            show_hud = !show_hud;
        }
        if window.is_key_pressed(Key::F, KeyRepeat::No) && gpu.is_some() {
            use_gpu = !use_gpu;
            res_scale = if use_gpu { 2 } else { 1 };
            set_resolution(&mut app, &mut framebuffer, res_scale);
        }
        if window.is_key_pressed(Key::Z, KeyRepeat::No)
            || window.is_key_pressed(Key::Y, KeyRepeat::No)
        {
            res_scale = if res_scale == 1 { 2 } else { 1 };
            set_resolution(&mut app, &mut framebuffer, res_scale);
        }

        app.step(dt_wall, &held_input(&window));
        if app.scene_generation != uploaded_generation {
            if let Some(g) = gpu.as_mut() {
                g.update_scene(app.scene_idx, app.scene());
            }
            uploaded_generation = app.scene_generation;
        }

        match gpu.as_mut() {
            Some(g) if use_gpu => g.render(
                app.scene_idx,
                &app.camera,
                &app.cfg,
                &app.dynamics,
                &mut framebuffer,
            ),
            _ => render_frame(app.scene(), &app.camera, &app.cfg, &app.dynamics, &mut framebuffer),
        }

        if show_hud {
            let backend = if gpu.is_some() && use_gpu { "GPU" } else { "CPU" };
            draw_hud(
                &mut framebuffer,
                app.scene().demo,
                &app.scene_description(),
                &app.camera,
                &app.cfg,
                app.paused,
                app.time_scale(),
                fps_smooth,
                backend,
            );
        }
        if let Some(banner) = app.banner() {
            draw_banner(&mut framebuffer, app.cfg.width, app.cfg.height, banner);
        }

        window
            .update_with_buffer(&framebuffer, app.cfg.width, app.cfg.height)
            .expect("Failed to update window buffer");
    }
    app.leave_match();
}
