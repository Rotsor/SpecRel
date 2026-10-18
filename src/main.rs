pub mod bvh4d;
pub mod hud;
pub mod mesh4d;
pub mod minkowski;
pub mod renderer;
pub mod scene;

use hud::{draw_text, fill_rect_alpha};
use minkowski::{Vec3, Vec4};
use minifb::{Key, KeyRepeat, Window, WindowOptions};
use renderer::{render_frame, Camera4D, RenderConfig, RenderMode, ShadowMode, SliceMode};
use scene::{DemoScene, Scene4D};
use std::time::Instant;

const WIDTH: usize = 1280;
const HEIGHT: usize = 800;

const TIME_SCALES: [f32; 10] = [
    -4.0, -2.0, -1.0, -0.5, -0.25, 0.25, 0.5, 1.0, 2.0, 4.0,
];

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

    let mut scene_idx: usize = 0; // Starts on Scene 1: Twin Paradox
    let mut camera = Camera4D::new(scenes[scene_idx].demo.initial_camera_pos(), 65.0);
    let mut cfg = RenderConfig {
        width: WIDTH,
        height: HEIGHT,
        mode: RenderMode::RelativisticShaded,
        shadow_mode: ShadowMode::LorentzInvariant,
        slice_mode: SliceMode::PastLightCone,
    };

    let mut framebuffer = vec![0u32; WIDTH * HEIGHT];

    if args.iter().any(|a| a == "--smoke-test") {
        for (i, scene) in scenes.iter().enumerate() {
            let cam = Camera4D::new(scene.demo.initial_camera_pos(), 65.0);
            let t0 = Instant::now();
            render_frame(scene, &cam, &cfg, &mut framebuffer);
            let dt = t0.elapsed();
            let non_zero = framebuffer.iter().filter(|&&px| px != 0).count();
            println!(
                "Scene {} [{}] ({} simplices): rendered {}x{} in {:.2?} ({}/{} non-zero px)",
                i + 1,
                scene.demo.title(),
                scene.bvh.simplices.len(),
                WIDTH,
                HEIGHT,
                dt,
                non_zero,
                framebuffer.len()
            );
        }
        return;
    }

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
        if window.is_key_pressed(Key::R, KeyRepeat::No) {
            camera = Camera4D::new(scene.demo.initial_camera_pos(), 65.0);
            time_scale_idx = 7; // +1.0x
            paused_worldline = false;
        }
        if window.is_key_pressed(Key::X, KeyRepeat::No) {
            // Brake to lab rest frame while preserving local spatial orientation via Fermi-Walker transport
            camera
                .tetrad
                .fermi_walker_transport(Vec4::new(1.0, 0.0, 0.0, 0.0));
        }
        if window.is_key_pressed(Key::V, KeyRepeat::No) {
            // Toggle between Lab Rest Frame and the moving object's Comoving Rest Frame (Pole / Train / Traveler)!
            if camera.tetrad.beta().length() > 0.1 {
                camera
                    .tetrad
                    .fermi_walker_transport(Vec4::new(1.0, 0.0, 0.0, 0.0));
            } else {
                let target_beta = scene.demo.comoving_beta(camera.position.t);
                let target_u = Vec4::four_velocity_from_beta(target_beta);
                camera.tetrad.fermi_walker_transport(target_u);
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

        // 3. Direct coordinate-space & time scrubbing (hold `[` to rewind/scrub back, `]` to scrub forward)
        let scrub_speed = 6.0 * dt_wall;
        let mut time_changed = false;
        if window.is_key_down(Key::LeftBracket) {
            camera.position.t -= scrub_speed;
            camera.proper_time -= scrub_speed / camera.tetrad.gamma();
            time_changed = true;
        }
        if window.is_key_down(Key::RightBracket) {
            camera.position.t += scrub_speed;
            camera.proper_time += scrub_speed / camera.tetrad.gamma();
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

        render_frame(scene, &camera, &cfg, &mut framebuffer);

        if show_hud {
            draw_hud(
                &mut framebuffer,
                scene.demo,
                &camera,
                &cfg,
                paused_worldline,
                time_scale,
                fps_smooth,
            );
        }

        window
            .update_with_buffer(&framebuffer, WIDTH, HEIGHT)
            .expect("Failed to update window buffer");
    }
}

fn draw_hud(
    fb: &mut [u32],
    demo: DemoScene,
    camera: &Camera4D,
    cfg: &RenderConfig,
    paused: bool,
    time_scale: f32,
    fps: f32,
) {
    fill_rect_alpha(fb, WIDTH, HEIGHT, 12, 12, 1140, 200);

    let beta = camera.tetrad.beta();
    let speed = beta.length();
    let gamma = camera.tetrad.gamma();

    let line_scene = format!("SCENE [1-5 / N,B]: {}", demo.title());
    let line_desc = demo.description();

    let time_status = if paused {
        "PAUSED".to_string()
    } else if time_scale < 0.0 {
        format!("{:+5.2}X (REWIND)", time_scale)
    } else {
        format!("{:+5.2}X (FWD)", time_scale)
    };

    let line1 = format!(
        "OBS EVENT (T,X,Y,Z): ({:+6.2}, {:+6.2}, {:+6.2}, {:+6.2})  OBS TAU: {:+6.2}S  TIME: {}",
        camera.position.t,
        camera.position.x,
        camera.position.y,
        camera.position.z,
        camera.proper_time,
        time_status
    );
    let line2 = format!(
        "OBS 3-VEL: ({:+5.3}, {:+5.3}, {:+5.3})  |V|: {:5.3}C  GAMMA: {:6.3}  FPS: {:4.0}",
        beta.x, beta.y, beta.z, speed, gamma, fps
    );
    let line3 = format!(
        "MODE [M]: {} | SLICE [G]: {} | SHADOWS [H]: {}",
        cfg.mode.name(),
        cfg.slice_mode.name(),
        cfg.shadow_mode.name(),
    );
    let line4 =
        "TIME: [P] PAUSE | [T/BKSP] REWIND | [-/=] SPEED | [ / ] SCRUB T | [R] RESET";
    let line5 =
        "THRUST: WASD/SPC/C | MATCH VEL: [V] | BRAKE: [X] | LOOK: ARROWS/QE | MOVE: IJKL/UO";

    draw_text(fb, WIDTH, HEIGHT, 22, 22, &line_scene, 0xFF9944);
    draw_text(fb, WIDTH, HEIGHT, 22, 46, line_desc, 0xFFD980);
    draw_text(fb, WIDTH, HEIGHT, 22, 74, &line1, 0xFFE866);
    draw_text(fb, WIDTH, HEIGHT, 22, 98, &line2, 0x66F2FF);
    draw_text(fb, WIDTH, HEIGHT, 22, 122, &line3, 0xA8FF80);
    draw_text(fb, WIDTH, HEIGHT, 22, 148, line4, 0xDDDDDD);
    draw_text(fb, WIDTH, HEIGHT, 22, 174, line5, 0xAAAAAA);
}
