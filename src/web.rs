//! WebAssembly (`wasm32-unknown-unknown`) bindings for running SpecRel 4D in the browser.
//!
//! Exports a C-ABI interface over WASM linear memory so JavaScript can:
//! 1. Initialize the demo scenes and their 4D SAH BVHs in WASM (re-packing Dynamic Mode scenes
//!    as they grow; JS re-uploads them via `web_take_dirty_scene`).
//! 2. Read zero-copy slices of the packed GPU storage buffers (`GpuNode`, `GpuGeom`, `GpuAttr`,
//!    `GpuMaterial`, `tables`) and the WGSL compute shader (`RENDER_WGSL`) directly from WASM
//!    memory into browser WebGPU (`navigator.gpu`).
//! 3. Step the relativistic observer camera, Lorentz boosts, proper time, and HUD each frame,
//!    supporting both zero-readback WebGPU compute rendering and CPU reference rendering.
//! 4. Play a two-player match (`web_init(_, player)`): JS ferries opaque frames between the
//!    match's [`QueueLink`] and a `BroadcastChannel` through the `web_net_*` exports.

use crate::app::{Action, App, HeldInput};
use crate::gpu::{build_tables_data, GpuUniforms, PackedSceneGpuData, RENDER_WGSL};
use crate::dynamic::PlayerId;
use crate::hud::{draw_hud, draw_hud_overlay, HUD_BASE_WIDTH, HUD_OVERLAY_HEIGHT};
use crate::multiplayer::link::{LinkState, QueueLink, QueueLinkHandle};
use crate::renderer::render_frame;
use crate::scene::{DemoScene, Scene4D};
use std::cell::UnsafeCell;

const WIDTH: usize = 1280;
const HEIGHT: usize = 800;

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
pub const KEY_SLASH: u32 = 21;
pub const KEY_PAGE_UP: u32 = 22;
pub const KEY_PAGE_DOWN: u32 = 23;
pub const KEY_6: u32 = 24;
pub const KEY_ENTER: u32 = 25;
pub const KEY_BACKSLASH: u32 = 26;
pub const KEY_SEMICOLON: u32 = 19;

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
/// `X` and `V` are held keys too: in a match the ship brakes / matches velocity while they are
/// down. Pressing one also fires its [`Action`].
pub const HOLD_X: u32 = 120;
pub const HOLD_V: u32 = 121;

/// Map a discrete JS key code to an app action (frontend-only keys are handled separately).
fn key_action(code: u32) -> Option<Action> {
    Some(match code {
        KEY_1..=KEY_5 => Action::SelectScene((code - KEY_1) as usize),
        KEY_6 => Action::SelectScene(5),
        KEY_ENTER => Action::Fire,
        KEY_BACKSLASH => Action::CycleAmmo,
        KEY_SEMICOLON => Action::CycleControls,
        KEY_N => Action::NextScene,
        KEY_B => Action::PrevScene,
        KEY_M => Action::CycleRenderMode,
        KEY_G => Action::CycleView,
        KEY_H => Action::CycleShadows,
        KEY_P => Action::TogglePause,
        KEY_T => Action::ToggleRewind,
        KEY_MINUS => Action::Slower,
        KEY_EQUAL => Action::Faster,
        KEY_R => Action::Reset,
        KEY_SLASH => Action::EmitFlash,
        KEY_PAGE_UP => Action::ProphecyDeltaUp,
        KEY_PAGE_DOWN => Action::ProphecyDeltaDown,
        _ => return None,
    })
}

fn held_input(mask: u32) -> HeldInput {
    let held = |code: u32| (mask & (1u32 << (code - 100))) != 0;
    HeldInput {
        yaw_left: held(HOLD_LEFT),
        yaw_right: held(HOLD_RIGHT),
        pitch_up: held(HOLD_UP),
        pitch_down: held(HOLD_DOWN),
        roll_left: held(HOLD_Q),
        roll_right: held(HOLD_E),
        thrust_forward: held(HOLD_W),
        thrust_back: held(HOLD_S),
        thrust_right: held(HOLD_D),
        thrust_left: held(HOLD_A),
        thrust_up: held(HOLD_SPACE),
        thrust_down: held(HOLD_C),
        brake: held(HOLD_X),
        match_velocity: held(HOLD_V),
        scrub_back: held(HOLD_BRACKET_LEFT),
        scrub_forward: held(HOLD_BRACKET_RIGHT),
        move_forward: held(HOLD_I),
        move_back: held(HOLD_K),
        move_right: held(HOLD_L),
        move_left: held(HOLD_J),
        move_up: held(HOLD_U),
        move_down: held(HOLD_O),
    }
}

struct WebState {
    app: App,
    packed_scenes: Vec<PackedSceneGpuData>,
    tables: Vec<u32>,
    show_hud: bool,
    gpu_available: bool,
    use_gpu: bool,
    res_scale: usize,
    fps_smooth: f32,
    held_keys: u32,
    uniforms: GpuUniforms,
    /// `App::scene_generation` last packed into `packed_scenes`.
    packed_generation: u64,
    /// Scene whose re-packed buffers JS has not re-uploaded yet.
    dirty_scene: Option<usize>,
    hud_overlay: Vec<u32>,
    cpu_fb: Vec<u32>,
    cpu_rgba: Vec<u32>,
    /// The frontend's side of the match link (multiplayer only).
    net: Option<QueueLinkHandle>,
    /// Incoming frame bytes written by JS before `web_net_push_incoming`.
    net_inbox: Vec<u8>,
    /// The last frame popped by `web_net_take_outgoing`, for JS to read.
    net_outgoing: Vec<u8>,
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

/// Build the scenes and the app. `mp_player` is -1 for single player, or 0/1 to play that side
/// of a two-player match whose frames JS moves through the `web_net_*` exports.
#[no_mangle]
pub extern "C" fn web_init(gpu_available: u32, mp_player: i32) {
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
    let (app, net) = match mp_player {
        0 | 1 => {
            let (link, handle) = QueueLink::new();
            let app =
                App::new_multiplayer(scenes, WIDTH, HEIGHT, mp_player as PlayerId, Box::new(link));
            (app, Some(handle))
        }
        _ => (App::new(scenes, WIDTH, HEIGHT), None),
    };
    let packed_generation = app.scene_generation;
    let uniforms = GpuUniforms::build(
        &app.camera,
        &app.cfg,
        &app.dynamics,
        packed_scenes[app.scene_idx].num_nodes,
        packed_scenes[app.scene_idx].light_pos,
    );

    let state = WebState {
        app,
        packed_scenes,
        tables,
        show_hud: true,
        gpu_available: has_gpu,
        use_gpu: has_gpu,
        res_scale: 1,
        fps_smooth: 60.0,
        held_keys: 0,
        uniforms,
        packed_generation,
        dirty_scene: None,
        hud_overlay: vec![0u32; HUD_BASE_WIDTH * HUD_OVERLAY_HEIGHT],
        cpu_fb: vec![0u32; WIDTH * HEIGHT],
        cpu_rgba: vec![0u32; WIDTH * HEIGHT],
        net,
        net_inbox: Vec::new(),
        net_outgoing: Vec::new(),
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
                let action = match code {
                    HOLD_X => Some(Action::Brake),
                    HOLD_V => Some(Action::MatchVelocity),
                    _ => None,
                };
                if let Some(action) = action.filter(|_| repeat == 0) {
                    s.app.apply(action);
                }
            } else {
                s.held_keys &= !bit;
            }
            return;
        }

        // Prophecy-delta stepping auto-repeats while held, like the native build.
        let repeats = code == KEY_PAGE_UP || code == KEY_PAGE_DOWN;
        if pressed == 0 || (repeat != 0 && !repeats) {
            return;
        }

        if let Some(action) = key_action(code) {
            s.app.apply(action);
            return;
        }

        // Frontend-only toggles: HUD, rendering backend and resolution.
        match code {
            KEY_TAB => s.show_hud = !s.show_hud,
            KEY_F => {
                if s.gpu_available {
                    s.use_gpu = !s.use_gpu;
                    if !s.use_gpu && s.res_scale > 1 {
                        s.res_scale = 1;
                        s.app.cfg.width = WIDTH;
                        s.app.cfg.height = HEIGHT;
                    }
                }
            }
            KEY_Z => {
                s.res_scale = if s.res_scale == 1 { 2 } else { 1 };
                s.app.cfg.width = WIDTH * s.res_scale;
                s.app.cfg.height = HEIGHT * s.res_scale;
            }
            _ => {}
        }
    });
}

#[no_mangle]
pub extern "C" fn web_mouse_look(dyaw: f32, dpitch: f32) {
    with_state(|s| s.app.look(dyaw, dpitch, 0.0));
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
        s.app.tick_wall_clock(dt_secs as f64);
        let dt_wall = dt_secs.clamp(0.001, 0.1);
        s.fps_smooth = s.fps_smooth * 0.9 + (1.0 / dt_wall) * 0.1;

        s.app.step(dt_wall, &held_input(s.held_keys));
        if s.app.scene_generation != s.packed_generation {
            s.packed_scenes[s.app.scene_idx] = PackedSceneGpuData::from_scene(s.app.scene());
            s.packed_generation = s.app.scene_generation;
            s.dirty_scene = Some(s.app.scene_idx);
        }

        let app = &s.app;
        let packed = &s.packed_scenes[app.scene_idx];
        s.uniforms = GpuUniforms::build(
            &app.camera,
            &app.cfg,
            &app.dynamics,
            packed.num_nodes,
            packed.light_pos,
        );

        let backend = if s.gpu_available && s.use_gpu {
            "GPU"
        } else {
            "CPU"
        };
        let demo = app.scene().demo;
        let desc = app.scene_description();
        let net = app.net_metrics();

        if s.gpu_available && s.use_gpu {
            if s.show_hud {
                draw_hud_overlay(
                    &mut s.hud_overlay,
                    demo,
                    &desc,
                    net.as_deref(),
                    &app.camera,
                    &app.cfg,
                    app.paused,
                    app.time_scale(),
                    s.fps_smooth,
                    backend,
                );
            }
        } else {
            let total_px = app.cfg.width * app.cfg.height;
            if s.cpu_fb.len() != total_px {
                s.cpu_fb.resize(total_px, 0);
            }
            render_frame(app.scene(), &app.camera, &app.cfg, &app.dynamics, &mut s.cpu_fb);

            if s.gpu_available {
                // WebGPU is presenting the canvas; composite the HUD in the blit shader.
                if s.show_hud {
                    draw_hud_overlay(
                        &mut s.hud_overlay,
                        demo,
                        &desc,
                        net.as_deref(),
                        &app.camera,
                        &app.cfg,
                        app.paused,
                        app.time_scale(),
                        s.fps_smooth,
                        backend,
                    );
                }
            } else {
                // Canvas2D fallback: bake HUD directly into `cpu_fb` and swizzle to RGBA bytes.
                if s.show_hud {
                    draw_hud(
                        &mut s.cpu_fb,
                        demo,
                        &desc,
                        net.as_deref(),
                        &app.camera,
                        &app.cfg,
                        app.paused,
                        app.time_scale(),
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

/// Index of a scene whose packed buffers changed since JS last uploaded them (then cleared), or -1.
#[no_mangle]
pub extern "C" fn web_take_dirty_scene() -> i32 {
    with_state(|s| s.dirty_scene.take().map_or(-1, |i| i as i32))
}

#[no_mangle]
pub extern "C" fn web_width() -> usize {
    with_state(|s| s.app.cfg.width)
}

#[no_mangle]
pub extern "C" fn web_height() -> usize {
    with_state(|s| s.app.cfg.height)
}

#[no_mangle]
pub extern "C" fn web_res_scale() -> usize {
    with_state(|s| s.res_scale)
}

#[no_mangle]
pub extern "C" fn web_scene_idx() -> usize {
    with_state(|s| s.app.scene_idx)
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

// ---------------------------------------------------------------------------------------------
// Multiplayer: frame ferrying, link state and the "waiting" banner
// ---------------------------------------------------------------------------------------------

#[no_mangle]
pub extern "C" fn web_is_multiplayer() -> u32 {
    with_state(|s| u32::from(s.app.multiplayer.is_some()))
}

/// A buffer of `len` bytes for JS to fill with one incoming frame before calling
/// `web_net_push_incoming(len)`. May grow WASM memory: re-create views after calling.
#[no_mangle]
pub extern "C" fn web_net_inbox_ptr(len: usize) -> *mut u8 {
    with_state(|s| {
        s.net_inbox.resize(len, 0);
        s.net_inbox.as_mut_ptr()
    })
}

/// Hand the first `len` bytes of the inbox buffer to the match as one frame from the peer.
#[no_mangle]
pub extern "C" fn web_net_push_incoming(len: usize) {
    with_state(|s| {
        if let Some(net) = &s.net {
            let len = len.min(s.net_inbox.len());
            net.push_incoming(s.net_inbox[..len].to_vec());
        }
    })
}

/// Pop the next frame to send to the peer into the outgoing buffer (read it via
/// `web_net_outgoing_ptr`). Returns its length, or -1 when there is none.
#[no_mangle]
pub extern "C" fn web_net_take_outgoing() -> i32 {
    with_state(|s| match s.net.as_ref().and_then(QueueLinkHandle::take_outgoing) {
        Some(frame) => {
            s.net_outgoing = frame;
            s.net_outgoing.len() as i32
        }
        None => -1,
    })
}

#[no_mangle]
pub extern "C" fn web_net_outgoing_ptr() -> *const u8 {
    with_state(|s| s.net_outgoing.as_ptr())
}

/// Transport state from JS: 0 connecting, 1 open, 2 closed.
#[no_mangle]
pub extern "C" fn web_net_set_state(state: u32) {
    with_state(|s| {
        if let Some(net) = &s.net {
            net.set_state(match state {
                0 => LinkState::Connecting,
                1 => LinkState::Open,
                _ => LinkState::Closed,
            });
        }
    })
}

/// Tell the other player we are leaving (JS then drains and posts the outgoing frames).
#[no_mangle]
pub extern "C" fn web_leave() {
    with_state(|s| s.app.leave_match());
}

/// The banner to show across the screen (UTF-8), if any; `web_banner_len` is 0 when none.
#[no_mangle]
pub extern "C" fn web_banner_ptr() -> *const u8 {
    with_state(|s| s.app.banner().unwrap_or("").as_ptr())
}

#[no_mangle]
pub extern "C" fn web_banner_len() -> usize {
    with_state(|s| s.app.banner().map_or(0, str::len))
}
