use crate::renderer::{Camera4D, RenderConfig, ViewMode};
use crate::scene::DemoScene;

pub const HUD_BASE_WIDTH: usize = 1280;
pub const HUD_OVERLAY_HEIGHT: usize = 216;

/// Minimal 5x7 ASCII bitmap font for on-screen telemetry HUD with custom scale.
pub fn draw_text_scaled(
    fb: &mut [u32],
    fb_w: usize,
    fb_h: usize,
    x: usize,
    y: usize,
    text: &str,
    color: u32,
    scale: usize,
) {
    let mut cx = x;
    for ch in text.chars() {
        if ch == '\n' {
            continue;
        }
        draw_char(fb, fb_w, fb_h, cx, y, ch, color, scale);
        cx += 6 * scale;
    }
}

/// Minimal 5x7 ASCII bitmap font for on-screen telemetry HUD.
pub fn draw_text(
    fb: &mut [u32],
    fb_w: usize,
    fb_h: usize,
    x: usize,
    y: usize,
    text: &str,
    color: u32,
) {
    let scale = if fb_w >= 1000 { 2 } else { 1 };
    draw_text_scaled(fb, fb_w, fb_h, x, y, text, color, scale);
}

pub fn fill_rect_alpha(
    fb: &mut [u32],
    fb_w: usize,
    fb_h: usize,
    x0: usize,
    y0: usize,
    w: usize,
    h: usize,
) {
    let x1 = (x0 + w).min(fb_w);
    let y1 = (y0 + h).min(fb_h);
    for y in y0..y1 {
        let row = &mut fb[y * fb_w..y * fb_w + fb_w];
        for px in &mut row[x0..x1] {
            let r = ((*px >> 16) & 0xFF) / 3;
            let g = ((*px >> 8) & 0xFF) / 3;
            let b = (*px & 0xFF) / 3;
            *px = (r << 16) | (g << 8) | b;
        }
    }
}

/// A large message centered on a `fb_w x fb_h` framebuffer, over a darkened band (e.g.
/// multiplayer's "waiting for the other player"). Drawn whether or not the HUD is shown.
pub fn draw_banner(fb: &mut [u32], fb_w: usize, fb_h: usize, text: &str) {
    let s = (fb_w / HUD_BASE_WIDTH).max(1);
    let scale = 4 * s;
    let (text_w, text_h) = (text.chars().count() * 6 * scale, 7 * scale);
    let (pad_x, pad_y) = (24 * s, 16 * s);
    let (w, h) = ((text_w + 2 * pad_x).min(fb_w), text_h + 2 * pad_y);
    let (x0, y0) = ((fb_w - w) / 2, fb_h.saturating_sub(h) / 2);
    fill_rect_alpha(fb, fb_w, fb_h, x0, y0, w, h);
    let x = (fb_w.saturating_sub(text_w)) / 2;
    draw_text_scaled(fb, fb_w, fb_h, x, y0 + pad_y, text, 0xFFE866, scale);
}

/// Color of the aiming reticle (the laser's orange-red).
pub const RETICLE_COLOR: u32 = 0xFF6A30;

/// Whether pixel `(x, y)` of a `fb_w x fb_h` framebuffer belongs to the aiming reticle at the
/// screen's center: four short arms around an open middle, so the target stays visible. The
/// web blit shader draws the same shape (`web/index.html`).
pub fn is_reticle_pixel(fb_w: usize, fb_h: usize, x: usize, y: usize) -> bool {
    let s = (fb_w / HUD_BASE_WIDTH).max(1) as i64;
    let (gap, len) = (6 * s, 10 * s);
    let dx = (x as i64 - (fb_w / 2) as i64).abs();
    let dy = (y as i64 - (fb_h / 2) as i64).abs();
    let arm = |along: i64, across: i64| across <= s && along >= gap && along < gap + len;
    arm(dx, dy) || arm(dy, dx)
}

/// Draw the aiming reticle (see [`is_reticle_pixel`]) onto a framebuffer. Drawn whether or not
/// the HUD is shown.
pub fn draw_reticle(fb: &mut [u32], fb_w: usize, fb_h: usize) {
    let r = 16 * (fb_w / HUD_BASE_WIDTH).max(1);
    let (cx, cy) = (fb_w / 2, fb_h / 2);
    for y in cy.saturating_sub(r)..(cy + r).min(fb_h) {
        for x in cx.saturating_sub(r)..(cx + r).min(fb_w) {
            if is_reticle_pixel(fb_w, fb_h, x, y) {
                fb[y * fb_w + x] = RETICLE_COLOR;
            }
        }
    }
}

fn fill_rect_solid(
    fb: &mut [u32],
    fb_w: usize,
    fb_h: usize,
    x0: usize,
    y0: usize,
    w: usize,
    h: usize,
    val: u32,
) {
    let x1 = (x0 + w).min(fb_w);
    let y1 = (y0 + h).min(fb_h);
    for y in y0..y1 {
        let row = &mut fb[y * fb_w..y * fb_w + fb_w];
        row[x0..x1].fill(val);
    }
}

/// Draw the telemetry HUD directly onto a rendered `cfg.width x cfg.height` framebuffer.
#[allow(clippy::too_many_arguments)]
pub fn draw_hud(
    fb: &mut [u32],
    demo: DemoScene,
    desc: &str,
    score: Option<&str>,
    net: Option<&str>,
    camera: &Camera4D,
    cfg: &RenderConfig,
    paused: bool,
    time_scale: f32,
    fps: f32,
    backend: &str,
) {
    let s = (cfg.width / HUD_BASE_WIDTH).max(1);
    let box_w = (cfg.width - 24 * s).min(1240 * s);
    fill_rect_alpha(fb, cfg.width, cfg.height, 12 * s, 12 * s, box_w, 200 * s);
    draw_hud_lines(
        fb,
        cfg.width,
        cfg.height,
        s,
        0,
        demo,
        desc,
        score,
        net,
        camera,
        cfg,
        paused,
        time_scale,
        fps,
        backend,
    );
}

/// Draw the telemetry HUD into a compact 1X overlay buffer (`HUD_BASE_WIDTH x HUD_OVERLAY_HEIGHT`)
/// for GPU compositing (`0x00000000` = transparent, `0x80000000` = 1/3 darkened box,
/// `0xFFRRGGBB` = opaque glyph pixel).
#[allow(clippy::too_many_arguments)]
pub fn draw_hud_overlay(
    overlay: &mut [u32],
    demo: DemoScene,
    desc: &str,
    score: Option<&str>,
    net: Option<&str>,
    camera: &Camera4D,
    cfg: &RenderConfig,
    paused: bool,
    time_scale: f32,
    fps: f32,
    backend: &str,
) {
    assert_eq!(overlay.len(), HUD_BASE_WIDTH * HUD_OVERLAY_HEIGHT);
    overlay.fill(0);
    let box_w = (HUD_BASE_WIDTH - 24).min(1240);
    fill_rect_solid(
        overlay,
        HUD_BASE_WIDTH,
        HUD_OVERLAY_HEIGHT,
        12,
        12,
        box_w,
        200,
        0x8000_0000,
    );
    draw_hud_lines(
        overlay,
        HUD_BASE_WIDTH,
        HUD_OVERLAY_HEIGHT,
        1,
        0xFF00_0000,
        demo,
        desc,
        score,
        net,
        camera,
        cfg,
        paused,
        time_scale,
        fps,
        backend,
    );
}

#[allow(clippy::too_many_arguments)]
fn draw_hud_lines(
    fb: &mut [u32],
    fb_w: usize,
    fb_h: usize,
    s: usize,
    alpha_mask: u32,
    demo: DemoScene,
    desc: &str,
    score: Option<&str>,
    net: Option<&str>,
    camera: &Camera4D,
    cfg: &RenderConfig,
    paused: bool,
    time_scale: f32,
    fps: f32,
    backend: &str,
) {
    let beta = camera.tetrad.beta();
    let speed = beta.length();
    let gamma = camera.tetrad.gamma();

    let line_scene = format!("SCENE [1-6 / N,B]: {}", demo.title());

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
        "OBS 3-VEL: ({:+5.3}, {:+5.3}, {:+5.3})  |V|: {:5.3}C  GAMMA: {:6.3}  FPS: {:4.0} [F] {}  RES: {}x{} [Z]",
        beta.x, beta.y, beta.z, speed, gamma, fps, backend, cfg.width, cfg.height
    );
    let view_name = if cfg.slice_mode == ViewMode::ProphecySlice {
        format!("PROPHECY (DT'=+{:.1}S)", cfg.prophecy_delta)
    } else {
        cfg.slice_mode.name().to_string()
    };
    let line3 = format!(
        "MODE [M]: {} | VIEW [G]: {} | SHADOWS [H]: {}",
        cfg.mode.name(),
        view_name,
        cfg.shadow_mode.name(),
    );
    // In a match the latency figures take the place of the time controls, most of which are
    // disabled there anyway.
    let line4 = net.unwrap_or(
        "TIME: [P] PAUSE | [T] REWIND | [-/=] SPEED | [ / ] SCRUB T | [R] RESET | [/] FLASH | [PGUP/DN] DELTA",
    );
    let line5 =
        "THRUST: WASD/SPC/C | MATCH VEL: [V] | BRAKE: [X] | LOOK: ARROWS/QE | MOVE: IJKL/UO";

    let text_scale = 2 * s;
    draw_text_scaled(fb, fb_w, fb_h, 22 * s, 22 * s, &line_scene, alpha_mask | 0xFF9944, text_scale);
    if let Some(score) = score {
        // Right-aligned in the box, in the laser's color.
        let box_right = 12 * s + (fb_w - 24 * s).min(1240 * s);
        let x = box_right.saturating_sub(10 * s + score.chars().count() * 6 * text_scale);
        draw_text_scaled(fb, fb_w, fb_h, x, 22 * s, score, alpha_mask | RETICLE_COLOR, text_scale);
    }
    draw_text_scaled(fb, fb_w, fb_h, 22 * s, 46 * s, desc, alpha_mask | 0xFFD980, text_scale);
    draw_text_scaled(fb, fb_w, fb_h, 22 * s, 74 * s, &line1, alpha_mask | 0xFFE866, text_scale);
    draw_text_scaled(fb, fb_w, fb_h, 22 * s, 98 * s, &line2, alpha_mask | 0x66F2FF, text_scale);
    draw_text_scaled(fb, fb_w, fb_h, 22 * s, 122 * s, &line3, alpha_mask | 0xA8FF80, text_scale);
    draw_text_scaled(fb, fb_w, fb_h, 22 * s, 148 * s, line4, alpha_mask | 0xDDDDDD, text_scale);
    draw_text_scaled(fb, fb_w, fb_h, 22 * s, 174 * s, line5, alpha_mask | 0xAAAAAA, text_scale);
}

#[allow(clippy::too_many_arguments)]
fn draw_char(
    fb: &mut [u32],
    fb_w: usize,
    fb_h: usize,
    x: usize,
    y: usize,
    ch: char,
    color: u32,
    scale: usize,
) {
    let glyph = glyph_5x7(ch.to_ascii_uppercase());
    for (row_idx, &bits) in glyph.iter().enumerate() {
        for col_idx in 0..5 {
            if (bits >> (4 - col_idx)) & 1 != 0 {
                for sy in 0..scale {
                    let py = y + row_idx * scale + sy;
                    if py >= fb_h {
                        continue;
                    }
                    for sx in 0..scale {
                        let px = x + col_idx * scale + sx;
                        if px < fb_w {
                            fb[py * fb_w + px] = color;
                        }
                    }
                }
            }
        }
    }
}

pub fn glyph_5x7(ch: char) -> [u8; 7] {
    match ch {
        'A' => [0b01110, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001],
        'B' => [0b11110, 0b10001, 0b10001, 0b11110, 0b10001, 0b10001, 0b11110],
        'C' => [0b01110, 0b10001, 0b10000, 0b10000, 0b10000, 0b10001, 0b01110],
        'D' => [0b11100, 0b10010, 0b10001, 0b10001, 0b10001, 0b10010, 0b11100],
        'E' => [0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b11111],
        'F' => [0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b10000],
        'G' => [0b01110, 0b10001, 0b10000, 0b10111, 0b10001, 0b10001, 0b01110],
        'H' => [0b10001, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001],
        'I' => [0b01110, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110],
        'J' => [0b00111, 0b00010, 0b00010, 0b00010, 0b10010, 0b10010, 0b01100],
        'K' => [0b10001, 0b10010, 0b10100, 0b11000, 0b10100, 0b10010, 0b10001],
        'L' => [0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b11111],
        'M' => [0b10001, 0b11011, 0b10101, 0b10101, 0b10001, 0b10001, 0b10001],
        'N' => [0b10001, 0b11001, 0b10101, 0b10011, 0b10001, 0b10001, 0b10001],
        'O' => [0b01110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110],
        'P' => [0b11110, 0b10001, 0b10001, 0b11110, 0b10000, 0b10000, 0b10000],
        'Q' => [0b01110, 0b10001, 0b10001, 0b10001, 0b10101, 0b10010, 0b01101],
        'R' => [0b11110, 0b10001, 0b10001, 0b11110, 0b10100, 0b10010, 0b10001],
        'S' => [0b01111, 0b10000, 0b10000, 0b01110, 0b00001, 0b00001, 0b11110],
        'T' => [0b11111, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100],
        'U' => [0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110],
        'V' => [0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01010, 0b00100],
        'W' => [0b10001, 0b10001, 0b10001, 0b10101, 0b10101, 0b11011, 0b10001],
        'X' => [0b10001, 0b10001, 0b01010, 0b00100, 0b01010, 0b10001, 0b10001],
        'Y' => [0b10001, 0b10001, 0b01010, 0b00100, 0b00100, 0b00100, 0b00100],
        'Z' => [0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b10000, 0b11111],
        '0' => [0b01110, 0b10001, 0b10011, 0b10101, 0b11001, 0b10001, 0b01110],
        '1' => [0b00100, 0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110],
        '2' => [0b01110, 0b10001, 0b00001, 0b00010, 0b00100, 0b01000, 0b11111],
        '3' => [0b11110, 0b00001, 0b00001, 0b01110, 0b00001, 0b00001, 0b11110],
        '4' => [0b00010, 0b00110, 0b01010, 0b10010, 0b11111, 0b00010, 0b00010],
        '5' => [0b11111, 0b10000, 0b11110, 0b00001, 0b00001, 0b10001, 0b01110],
        '6' => [0b01110, 0b10000, 0b10000, 0b11110, 0b10001, 0b10001, 0b01110],
        '7' => [0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b01000, 0b01000],
        '8' => [0b01110, 0b10001, 0b10001, 0b01110, 0b10001, 0b10001, 0b01110],
        '9' => [0b01110, 0b10001, 0b10001, 0b01111, 0b00001, 0b00001, 0b01110],
        '.' => [0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b01100, 0b01100],
        ',' => [0b00000, 0b00000, 0b00000, 0b00000, 0b00100, 0b00100, 0b01000],
        ':' => [0b00000, 0b01100, 0b01100, 0b00000, 0b01100, 0b01100, 0b00000],
        ';' => [0b00000, 0b01100, 0b01100, 0b00000, 0b01100, 0b00100, 0b01000],
        '-' => [0b00000, 0b00000, 0b00000, 0b11111, 0b00000, 0b00000, 0b00000],
        '+' => [0b00000, 0b00100, 0b00100, 0b11111, 0b00100, 0b00100, 0b00000],
        '=' => [0b00000, 0b00000, 0b11111, 0b00000, 0b11111, 0b00000, 0b00000],
        '/' => [0b00001, 0b00010, 0b00100, 0b00100, 0b01000, 0b10000, 0b00000],
        '\\' => [0b10000, 0b01000, 0b00100, 0b00100, 0b00010, 0b00001, 0b00000],
        '(' => [0b00010, 0b00100, 0b01000, 0b01000, 0b01000, 0b00100, 0b00010],
        ')' => [0b01000, 0b00100, 0b00010, 0b00010, 0b00010, 0b00100, 0b01000],
        '[' => [0b01110, 0b01000, 0b01000, 0b01000, 0b01000, 0b01000, 0b01110],
        ']' => [0b01110, 0b00010, 0b00010, 0b00010, 0b00010, 0b00010, 0b01110],
        '|' => [0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100],
        '!' => [0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00000, 0b00100],
        '\'' => [0b00100, 0b00100, 0b01000, 0b00000, 0b00000, 0b00000, 0b00000],
        '*' => [0b00000, 0b10101, 0b01110, 0b11111, 0b01110, 0b10101, 0b00000],
        '&' => [0b01100, 0b10010, 0b10100, 0b01000, 0b10101, 0b10010, 0b01101],
        _ => [0; 7],
    }
}
