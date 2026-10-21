//! WebAssembly (`wasm32-unknown-unknown`) bindings for running SpecRel 4D in the browser.
//!
//! Exports a C-ABI interface over WASM linear memory so JavaScript can:
//! 1. Initialize all 5 Special Relativity 4D scenes and their 4D SAH BVHs in WASM.
//! 2. Read zero-copy slices of the packed GPU storage buffers (`GpuNode`, `GpuGeom`, `GpuAttr`,
//!    `GpuMaterial`, `tables`) and the WGSL compute shader (`RENDER_WGSL`) directly from WASM
//!    memory into browser WebGPU (`navigator.gpu`).
//! 3. Step the relativistic observer camera, Lorentz boosts, proper time, and HUD each frame,
//!    supporting both zero-readback WebGPU compute rendering and CPU reference rendering.

use crate::gpu::{build_tables_data, GpuUniforms, PackedSceneGpuData, RENDER_WGSL};
use crate::hud::{draw_hud, draw_hud_overlay, HUD_BASE_WIDTH, HUD_OVERLAY_HEIGHT};
use crate::minkowski::{Vec3, Vec4};
use crate::renderer::{
    render_frame, Camera4D, FrameDynamics, RenderConfig, RenderMode, ShadowMode, ViewMode,
};
use crate::scene::{DemoScene, Scene4D};
use std::cell::UnsafeCell;

const WIDTH: usize = 1280;
const HEIGHT: usize = 800;

const TIME_SCALES: [f32; 10] = [
    -4.0, -2.0, -1.0, -0.5, -0.25, 0.25, 0.5, 1.0, 2.0, 4.0,
];

// Discrete key action codes sent from JS `keydown`
pub const KEY_1: u32 = 1;
pub const KEY_2: u32 = 2;
pub const KEY_3: u32 = 3;
pub const KEY_4: u32 = 4;
pub const KEY_5: u32 = 5;
pub const KEY_N: u32 = 6;
pub const KEY_B: u32 = 7;
pub const KEY_M: u32 = 8;
pub const KEY_G: u32 = 9;
pub const KEY_H: u32 = 10;
pub const KEY_P: u32 = 11;
pub const KEY_T: u32 = 12;
pub const KEY_MINUS: u32 = 13;
pub const KEY_EQUAL: u32 = 14;
pub const KEY_TAB: u32 = 15;
pub const KEY_F: u32 = 16;
pub const KEY_Z: u32 = 17;
pub const KEY_R: u32 = 18;
pub const KEY_X: u32 = 19;
pub const KEY_V: u32 = 20;

// Continuous held-key bitflags (100 + bit index)
pub const HOLD_LEFT: u32 = 100;
pub const HOLD_RIGHT: u32 = 101;
pub const HOLD_UP: u32 = 102;
pub const HOLD_DOWN: u32 = 103;
pub const HOLD_Q: u32 = 104;
pub const HOLD_E: u32 = 105;
pub const HOLD_W: u32 = 106;
pub const HOLD_S: u32 = 107;
pub const HOLD_D: u32 = 108;
pub const HOLD_A: u32 = 109;
pub const HOLD_SPACE: u32 = 110;
pub const HOLD_C: u32 = 111;
pub const HOLD_BRACKET_LEFT: u32 = 112;
pub const HOLD_BRACKET_RIGHT: u32 = 113;
pub const HOLD_I: u32 = 114;
pub const HOLD_K: u32 = 115;
pub const HOLD_L: u32 = 116;
pub const HOLD_J: u32 = 117;
pub const HOLD_U: u32 = 118;
pub const HOLD_O: u32 = 119;

struct WebState {
    scenes: Vec<Scene4D>,
    packed_scenes: Vec<PackedSceneGpuData>,
    tables: Vec<u32>,
    scene_idx: usize,
    camera: Camera4D,
    cfg: RenderConfig,
    /// Per-frame causal frontier and flashes (empty for static scenes).
    dynamics: FrameDynamics,
    paused_worldline: bool,
    time_scale_idx: usize,
    show_hud: bool,
    gpu_available: bool,
    use_gpu: bool,
    res_scale: usize,
    fps_smooth: f32,
    held_keys: u32,
    uniforms: GpuUniforms,
    hud_overlay: Vec<u32>,
    cpu_fb: Vec<u32>,
    cpu_rgba: Vec<u32>,
}

struct GlobalState(UnsafeCell<Option<WebState>>);
// Safety: `wasm32-unknown-unknown` executes on a single browser thread.
unsafe impl Sync for GlobalState {}

static STATE: GlobalState = GlobalState(UnsafeCell::new(None));

#[inline]
fn with_state<R>(f: impl FnOnce(&mut WebState) -> R) -> R {
    let slot = unsafe { &mut *STATE.0.get() };
    let state = slot.as_mut().expect("web_init() must be called first");
    f(state)
}

#[inline]
fn held(mask: u32, code: u32) -> bool {
    (mask & (1u32 << (code - 100))) != 0
}

#[no_mangle]
pub extern "C" fn web_init(gpu_available: u32) {
    let scenes: Vec<Scene4D> = DemoScene::ALL
        .iter()
        .map(|&demo| Scene4D::build(demo))
        .collect();
    let packed_scenes: Vec<PackedSceneGpuData> = scenes
        .iter()
        .map(PackedSceneGpuData::from_scene)
        .collect();
    let tables = build_tables_data();

    let has_gpu = gpu_available != 0;
    let res_scale = 1usize;
    let scene_idx = 0usize;
    let camera = Camera4D::new(scenes[scene_idx].demo.initial_camera_pos(), 65.0);
    let cfg = RenderConfig {
        width: WIDTH * res_scale,
        height: HEIGHT * res_scale,
        mode: RenderMode::RelativisticShaded,
        shadow_mode: ShadowMode::LorentzInvariant,
        slice_mode: ViewMode::PastLightCone,
    };
    let dynamics = FrameDynamics::default();
    let uniforms = GpuUniforms::build(
        &camera,
        &cfg,
        &dynamics,
        packed_scenes[scene_idx].num_nodes,
        packed_scenes[scene_idx].light_pos,
    );

    let state = WebState {
        scenes,
        packed_scenes,
        tables,
        scene_idx,
        camera,
        cfg,
        dynamics,
        paused_worldline: false,
        time_scale_idx: 7, // +1.0x forward
        show_hud: true,
        gpu_available: has_gpu,
        use_gpu: has_gpu,
        res_scale,
        fps_smooth: 60.0,
        held_keys: 0,
        uniforms,
        hud_overlay: vec![0u32; HUD_BASE_WIDTH * HUD_OVERLAY_HEIGHT],
        cpu_fb: vec![0u32; WIDTH * HEIGHT],
        cpu_rgba: vec![0u32; WIDTH * HEIGHT],
    };

    unsafe {
        *STATE.0.get() = Some(state);
    }
}

#[no_mangle]
pub extern "C" fn web_key_event(code: u32, pressed: u32, repeat: u32) {
    with_state(|s| {
        if (100..132).contains(&code) {
            let bit = 1u32 << (code - 100);
            if pressed != 0 {
                s.held_keys |= bit;
            } else {
                s.held_keys &= !bit;
            }
            return;
        }

        if pressed == 0 || repeat != 0 {
            return;
        }

        let mut new_scene_idx = None;
        match code {
            KEY_1 => new_scene_idx = Some(0),
            KEY_2 => new_scene_idx = Some(1),
            KEY_3 => new_scene_idx = Some(2),
            KEY_4 => new_scene_idx = Some(3),
            KEY_5 => new_scene_idx = Some(4),
            KEY_N => new_scene_idx = Some((s.scene_idx + 1) % s.scenes.len()),
            KEY_B => new_scene_idx = Some((s.scene_idx + s.scenes.len() - 1) % s.scenes.len()),
            KEY_M => s.cfg.mode = s.cfg.mode.next(),
            KEY_G => s.cfg.slice_mode = s.cfg.slice_mode.next(),
            KEY_H => s.cfg.shadow_mode = s.cfg.shadow_mode.next(),
            KEY_P => s.paused_worldline = !s.paused_worldline,
            KEY_T => {
                s.time_scale_idx = TIME_SCALES.len() - 1 - s.time_scale_idx;
                s.paused_worldline = false;
            }
            KEY_MINUS => {
                s.time_scale_idx = s.time_scale_idx.saturating_sub(1);
            }
            KEY_EQUAL => {
                s.time_scale_idx = (s.time_scale_idx + 1).min(TIME_SCALES.len() - 1);
            }
            KEY_TAB => s.show_hud = !s.show_hud,
            KEY_F => {
                if s.gpu_available {
                    s.use_gpu = !s.use_gpu;
                    if !s.use_gpu && s.res_scale > 1 {
                        s.res_scale = 1;
                        s.cfg.width = WIDTH;
                        s.cfg.height = HEIGHT;
                    }
                }
            }
            KEY_Z => {
                s.res_scale = if s.res_scale == 1 { 2 } else { 1 };
                s.cfg.width = WIDTH * s.res_scale;
                s.cfg.height = HEIGHT * s.res_scale;
            }
            KEY_R => {
                let pos = s.scenes[s.scene_idx].demo.initial_camera_pos();
                s.camera = Camera4D::new(pos, 65.0);
                s.time_scale_idx = 7;
                s.paused_worldline = false;
            }
            KEY_X => {
                s.camera.tetrad.boost_to(Vec4::new(1.0, 0.0, 0.0, 0.0));
            }
            KEY_V => {
                if s.camera.tetrad.beta().length() > 0.1 {
                    s.camera.tetrad.boost_to(Vec4::new(1.0, 0.0, 0.0, 0.0));
                } else {
                    let target_beta = s.scenes[s.scene_idx]
                        .demo
                        .comoving_beta(s.camera.position.t);
                    let target_u = Vec4::four_velocity_from_beta(target_beta);
                    s.camera.tetrad.boost_to(target_u);
                }
            }
            _ => {}
        }

        if let Some(idx) = new_scene_idx {
            if idx < s.scenes.len() {
                s.scene_idx = idx;
                let pos = s.scenes[s.scene_idx].demo.initial_camera_pos();
                s.camera = Camera4D::new(pos, 65.0);
                s.paused_worldline = false;
            }
        }
    });
}

#[no_mangle]
pub extern "C" fn web_mouse_look(dyaw: f32, dpitch: f32) {
    with_state(|s| {
        if dyaw != 0.0 || dpitch != 0.0 {
            s.camera.tetrad.rotate_local(dyaw, dpitch, 0.0);
        }
    });
}

#[no_mangle]
pub extern "C" fn web_blur() {
    with_state(|s| {
        s.held_keys = 0;
    });
}

#[no_mangle]
pub extern "C" fn web_step(dt_secs: f32) {
    with_state(|s| {
        let dt_wall = dt_secs.clamp(0.001, 0.1);
        s.fps_smooth = s.fps_smooth * 0.9 + (1.0 / dt_wall) * 0.1;

        let mask = s.held_keys;

        // 1. Local camera orientation in observer's rest frame
        let rot_speed = 1.6 * dt_wall;
        let mut yaw = 0.0;
        let mut pitch = 0.0;
        let mut roll = 0.0;
        if held(mask, HOLD_LEFT) {
            yaw -= rot_speed;
        }
        if held(mask, HOLD_RIGHT) {
            yaw += rot_speed;
        }
        if held(mask, HOLD_UP) {
            pitch += rot_speed;
        }
        if held(mask, HOLD_DOWN) {
            pitch -= rot_speed;
        }
        if held(mask, HOLD_Q) {
            roll -= rot_speed;
        }
        if held(mask, HOLD_E) {
            roll += rot_speed;
        }
        if yaw != 0.0 || pitch != 0.0 || roll != 0.0 {
            s.camera.tetrad.rotate_local(yaw, pitch, roll);
        }

        // 2. Proper acceleration thrusters (Lorentz boost in observer's local rest frame)
        let boost_rate = 1.25 * dt_wall;
        let mut d_rapidity = Vec3::default();
        if held(mask, HOLD_W) {
            d_rapidity.z += boost_rate;
        }
        if held(mask, HOLD_S) {
            d_rapidity.z -= boost_rate;
        }
        if held(mask, HOLD_D) {
            d_rapidity.x += boost_rate;
        }
        if held(mask, HOLD_A) {
            d_rapidity.x -= boost_rate;
        }
        if held(mask, HOLD_SPACE) {
            d_rapidity.y += boost_rate;
        }
        if held(mask, HOLD_C) {
            d_rapidity.y -= boost_rate;
        }
        s.camera.tetrad.boost_local(d_rapidity);

        // 3. Direct coordinate-space & time scrubbing
        let scrub_speed = 6.0 * dt_wall;
        let mut time_changed = false;
        if held(mask, HOLD_BRACKET_LEFT) {
            s.camera.position.t -= scrub_speed;
            time_changed = true;
        }
        if held(mask, HOLD_BRACKET_RIGHT) {
            s.camera.position.t += scrub_speed;
            time_changed = true;
        }
        if held(mask, HOLD_I) {
            s.camera.position += s
                .camera
                .tetrad
                .local_spatial_to_world(Vec3::new(0.0, 0.0, scrub_speed));
        }
        if held(mask, HOLD_K) {
            s.camera.position += s
                .camera
                .tetrad
                .local_spatial_to_world(Vec3::new(0.0, 0.0, -scrub_speed));
        }
        if held(mask, HOLD_L) {
            s.camera.position += s
                .camera
                .tetrad
                .local_spatial_to_world(Vec3::new(scrub_speed, 0.0, 0.0));
        }
        if held(mask, HOLD_J) {
            s.camera.position += s
                .camera
                .tetrad
                .local_spatial_to_world(Vec3::new(-scrub_speed, 0.0, 0.0));
        }
        if held(mask, HOLD_U) {
            s.camera.position += s
                .camera
                .tetrad
                .local_spatial_to_world(Vec3::new(0.0, scrub_speed, 0.0));
        }
        if held(mask, HOLD_O) {
            s.camera.position += s
                .camera
                .tetrad
                .local_spatial_to_world(Vec3::new(0.0, -scrub_speed, 0.0));
        }

        // 4. Advance observer along their timelike worldline: dX^mu = u^mu * d_tau
        let time_scale = TIME_SCALES[s.time_scale_idx];
        if !s.paused_worldline {
            let d_tau = dt_wall * time_scale;
            s.camera.proper_time += d_tau;
            s.camera.position += s.camera.tetrad.e[0] * d_tau;
            time_changed = true;
        }

        let scene = &s.scenes[s.scene_idx];
        if time_changed {
            let t_loop = scene.t_max - scene.t_min;
            if s.camera.position.t > scene.t_max || s.camera.position.t < scene.t_min {
                s.camera.position.t =
                    scene.t_min + (s.camera.position.t - scene.t_min).rem_euclid(t_loop);
            }
        }

        let packed = &s.packed_scenes[s.scene_idx];
        s.uniforms = GpuUniforms::build(
            &s.camera,
            &s.cfg,
            &s.dynamics,
            packed.num_nodes,
            packed.light_pos,
        );

        let backend = if s.gpu_available && s.use_gpu {
            "GPU"
        } else {
            "CPU"
        };

        if s.gpu_available && s.use_gpu {
            if s.show_hud {
                draw_hud_overlay(
                    &mut s.hud_overlay,
                    scene.demo,
                    &s.camera,
                    &s.cfg,
                    s.paused_worldline,
                    time_scale,
                    s.fps_smooth,
                    backend,
                );
            }
        } else {
            let total_px = s.cfg.width * s.cfg.height;
            if s.cpu_fb.len() != total_px {
                s.cpu_fb.resize(total_px, 0);
            }
            render_frame(scene, &s.camera, &s.cfg, &s.dynamics, &mut s.cpu_fb);

            if s.gpu_available {
                // WebGPU is presenting the canvas; composite the HUD in the blit shader.
                if s.show_hud {
                    draw_hud_overlay(
                        &mut s.hud_overlay,
                        scene.demo,
                        &s.camera,
                        &s.cfg,
                        s.paused_worldline,
                        time_scale,
                        s.fps_smooth,
                        backend,
                    );
                }
            } else {
                // Canvas2D fallback: bake HUD directly into `cpu_fb` and swizzle to RGBA bytes.
                if s.show_hud {
                    draw_hud(
                        &mut s.cpu_fb,
                        scene.demo,
                        &s.camera,
                        &s.cfg,
                        s.paused_worldline,
                        time_scale,
                        s.fps_smooth,
                        backend,
                    );
                }
                if s.cpu_rgba.len() != total_px {
                    s.cpu_rgba.resize(total_px, 0);
                }
                for (dst, &px) in s.cpu_rgba.iter_mut().zip(s.cpu_fb.iter()) {
                    let r = (px >> 16) & 0xFF;
                    let g = (px >> 8) & 0xFF;
                    let b = px & 0xFF;
                    *dst = 0xFF00_0000 | (b << 16) | (g << 8) | r;
                }
            }
        }
    });
}

// ---------------------------------------------------------------------------------------------
// Zero-copy buffer & state accessors for JS WebGPU / Canvas2D
// ---------------------------------------------------------------------------------------------

#[no_mangle]
pub extern "C" fn web_shader_ptr() -> *const u8 {
    RENDER_WGSL.as_ptr()
}

#[no_mangle]
pub extern "C" fn web_shader_len() -> usize {
    RENDER_WGSL.len()
}

#[no_mangle]
pub extern "C" fn web_tables_ptr() -> *const u8 {
    with_state(|s| bytemuck::cast_slice::<u32, u8>(&s.tables).as_ptr())
}

#[no_mangle]
pub extern "C" fn web_tables_byte_len() -> usize {
    with_state(|s| bytemuck::cast_slice::<u32, u8>(&s.tables).len())
}

#[no_mangle]
pub extern "C" fn web_scene_count() -> usize {
    with_state(|s| s.packed_scenes.len())
}

#[no_mangle]
pub extern "C" fn web_scene_nodes_ptr(scene_idx: usize) -> *const u8 {
    with_state(|s| bytemuck::cast_slice(&s.packed_scenes[scene_idx].nodes).as_ptr())
}

#[no_mangle]
pub extern "C" fn web_scene_nodes_byte_len(scene_idx: usize) -> usize {
    with_state(|s| bytemuck::cast_slice::<_, u8>(&s.packed_scenes[scene_idx].nodes).len())
}

#[no_mangle]
pub extern "C" fn web_scene_geoms_ptr(scene_idx: usize) -> *const u8 {
    with_state(|s| bytemuck::cast_slice(&s.packed_scenes[scene_idx].geoms).as_ptr())
}

#[no_mangle]
pub extern "C" fn web_scene_geoms_byte_len(scene_idx: usize) -> usize {
    with_state(|s| bytemuck::cast_slice::<_, u8>(&s.packed_scenes[scene_idx].geoms).len())
}

#[no_mangle]
pub extern "C" fn web_scene_attrs_ptr(scene_idx: usize) -> *const u8 {
    with_state(|s| bytemuck::cast_slice(&s.packed_scenes[scene_idx].attrs).as_ptr())
}

#[no_mangle]
pub extern "C" fn web_scene_attrs_byte_len(scene_idx: usize) -> usize {
    with_state(|s| bytemuck::cast_slice::<_, u8>(&s.packed_scenes[scene_idx].attrs).len())
}

#[no_mangle]
pub extern "C" fn web_scene_materials_ptr(scene_idx: usize) -> *const u8 {
    with_state(|s| bytemuck::cast_slice(&s.packed_scenes[scene_idx].materials).as_ptr())
}

#[no_mangle]
pub extern "C" fn web_scene_materials_byte_len(scene_idx: usize) -> usize {
    with_state(|s| bytemuck::cast_slice::<_, u8>(&s.packed_scenes[scene_idx].materials).len())
}

#[no_mangle]
pub extern "C" fn web_width() -> usize {
    with_state(|s| s.cfg.width)
}

#[no_mangle]
pub extern "C" fn web_height() -> usize {
    with_state(|s| s.cfg.height)
}

#[no_mangle]
pub extern "C" fn web_res_scale() -> usize {
    with_state(|s| s.res_scale)
}

#[no_mangle]
pub extern "C" fn web_scene_idx() -> usize {
    with_state(|s| s.scene_idx)
}

#[no_mangle]
pub extern "C" fn web_use_gpu() -> u32 {
    with_state(|s| u32::from(s.gpu_available && s.use_gpu))
}

#[no_mangle]
pub extern "C" fn web_show_hud() -> u32 {
    with_state(|s| u32::from(s.show_hud))
}

#[no_mangle]
pub extern "C" fn web_hud_width() -> usize {
    HUD_BASE_WIDTH
}

#[no_mangle]
pub extern "C" fn web_hud_height() -> usize {
    HUD_OVERLAY_HEIGHT
}

#[no_mangle]
pub extern "C" fn web_uniforms_ptr() -> *const u8 {
    with_state(|s| bytemuck::bytes_of(&s.uniforms).as_ptr())
}

#[no_mangle]
pub extern "C" fn web_uniforms_byte_len() -> usize {
    std::mem::size_of::<GpuUniforms>()
}

#[no_mangle]
pub extern "C" fn web_hud_overlay_ptr() -> *const u8 {
    with_state(|s| bytemuck::cast_slice::<u32, u8>(&s.hud_overlay).as_ptr())
}

#[no_mangle]
pub extern "C" fn web_hud_overlay_byte_len() -> usize {
    with_state(|s| bytemuck::cast_slice::<u32, u8>(&s.hud_overlay).len())
}

#[no_mangle]
pub extern "C" fn web_cpu_fb_ptr() -> *const u8 {
    with_state(|s| bytemuck::cast_slice::<u32, u8>(&s.cpu_fb).as_ptr())
}

#[no_mangle]
pub extern "C" fn web_cpu_fb_byte_len() -> usize {
    with_state(|s| bytemuck::cast_slice::<u32, u8>(&s.cpu_fb).len())
}

#[no_mangle]
pub extern "C" fn web_cpu_rgba_ptr() -> *const u8 {
    with_state(|s| bytemuck::cast_slice::<u32, u8>(&s.cpu_rgba).as_ptr())
}

#[no_mangle]
pub extern "C" fn web_cpu_rgba_byte_len() -> usize {
    with_state(|s| bytemuck::cast_slice::<u32, u8>(&s.cpu_rgba).len())
}
