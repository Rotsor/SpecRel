use crate::renderer::{Camera4D, RenderConfig, ViewMode};
use crate::perf::{self, PerfStats};
use crate::scene::DemoScene;

pub const HUD_BASE_WIDTH: usize = 1280;
/// Rows of the 1X compositing overlay: the whole 1X screen, room for every panel.
pub const HUD_OVERLAY_HEIGHT: usize = 800;

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

/// Which parts of the HUD are shown, each toggled by an F key (the scene title, description
/// and scoreboard always are).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HudSections {
    /// `[F1]` Key help.
    pub keys: bool,
    /// `[F2]` The observer's event, proper time and velocity.
    pub observer: bool,
    /// `[F3]` Render settings, frame rate, backend and resolution.
    pub render: bool,
    /// `[F4]` Multiplayer latency figures.
    pub net: bool,
    /// `[F5]` Where each frame's time goes ([`PerfStats`]).
    pub perf: bool,
}

impl Default for HudSections {
    fn default() -> Self {
        Self { keys: true, observer: true, render: true, net: true, perf: false }
    }
}

impl HudSections {
    /// The sections' names, in F-key order (`[F1]` first).
    pub const NAMES: [&'static str; 5] = ["KEYS", "OBSERVER", "RENDER", "NET", "PERF"];

    fn flags(&mut self) -> [&mut bool; 5] {
        [&mut self.keys, &mut self.observer, &mut self.render, &mut self.net, &mut self.perf]
    }

    /// Flip the section on key `F{n}` (`n` in `1..=5`; others are ignored).
    pub fn toggle(&mut self, n: usize) {
        if let Some(flag) = n.checked_sub(1).and_then(|i| self.flags().into_iter().nth(i)) {
            *flag = !*flag;
        }
    }

    fn is_on(mut self, i: usize) -> bool {
        *self.flags()[i]
    }
}

/// Everything the HUD shows for one frame.
pub struct HudInfo<'a> {
    pub demo: DemoScene,
    pub desc: String,
    pub score: Option<String>,
    pub net: Option<String>,
    pub camera: &'a Camera4D,
    pub cfg: &'a RenderConfig,
    pub paused: bool,
    pub time_scale: f32,
    pub backend: &'a str,
    pub sections: HudSections,
    pub perf: &'a PerfStats,
    /// Extra performance lines: what the rendering cost scales with.
    pub perf_context: Vec<String>,
}

/// Draw the telemetry HUD directly onto a rendered `cfg.width x cfg.height` framebuffer.
pub fn draw_hud(fb: &mut [u32], info: &HudInfo) {
    let (w, h) = (info.cfg.width, info.cfg.height);
    let s = (w / HUD_BASE_WIDTH).max(1);
    draw_hud_panels(&mut Canvas { fb, w, h, s, overlay: false }, info);
}

/// Draw the telemetry HUD into a 1X overlay buffer (`HUD_BASE_WIDTH x HUD_OVERLAY_HEIGHT`) for
/// GPU compositing (`0x00000000` = transparent, `0x80000000` = 1/3 darkened box,
/// `0xFFRRGGBB` = opaque pixel). Returns how many rows from the top it used.
pub fn draw_hud_overlay(overlay: &mut [u32], info: &HudInfo) -> usize {
    assert_eq!(overlay.len(), HUD_BASE_WIDTH * HUD_OVERLAY_HEIGHT);
    overlay.fill(0);
    let mut canvas = Canvas { fb: overlay, w: HUD_BASE_WIDTH, h: HUD_OVERLAY_HEIGHT, s: 1, overlay: true };
    draw_hud_panels(&mut canvas, info).min(HUD_OVERLAY_HEIGHT)
}

/// A framebuffer drawn on in 1X HUD coordinates (scaled up by `s`).
struct Canvas<'a> {
    fb: &'a mut [u32],
    w: usize,
    h: usize,
    s: usize,
    /// Drawing into a compositing overlay (see [`draw_hud_overlay`]) rather than the image.
    overlay: bool,
}

impl Canvas<'_> {
    /// The width available to panels, in 1X units.
    fn panel_width(&self) -> usize {
        (self.w / self.s).saturating_sub(24).min(1240)
    }

    fn darken(&mut self, x: usize, y: usize, w: usize, h: usize) {
        let s = self.s;
        if self.overlay {
            fill_rect_solid(self.fb, self.w, self.h, x * s, y * s, w * s, h * s, 0x8000_0000);
        } else {
            fill_rect_alpha(self.fb, self.w, self.h, x * s, y * s, w * s, h * s);
        }
    }

    fn fill(&mut self, x: usize, y: usize, w: usize, h: usize, color: u32) {
        let (s, mask) = (self.s, self.mask());
        fill_rect_solid(self.fb, self.w, self.h, x * s, y * s, w * s, h * s, mask | color);
    }

    /// Text at double size (12 x 14 units per character).
    fn text(&mut self, x: usize, y: usize, text: &str, color: u32) {
        let (s, mask) = (self.s, self.mask());
        draw_text_scaled(self.fb, self.w, self.h, x * s, y * s, text, mask | color, 2 * s);
    }

    fn mask(&self) -> u32 {
        if self.overlay { 0xFF00_0000 } else { 0 }
    }
}

/// Width in 1X units of `text` drawn by [`Canvas::text`].
fn text_width(text: &str) -> usize {
    text.chars().count() * 12
}

const LINE_H: usize = 24;
const ROW_H: usize = 20;
const PANEL_GAP: usize = 8;

/// The main panel, then (if on) the performance panel under it. Returns the bottom edge in
/// framebuffer pixels.
fn draw_hud_panels(c: &mut Canvas, info: &HudInfo) -> usize {
    let mut bottom = draw_main_panel(c, info);
    if info.sections.perf {
        bottom = draw_perf_panel(c, info, bottom + PANEL_GAP);
    }
    bottom * c.s
}

/// Returns the panel's bottom edge (1X units).
fn draw_main_panel(c: &mut Canvas, info: &HudInfo) -> usize {
    let (camera, cfg, sections) = (info.camera, info.cfg, info.sections);
    let mut lines: Vec<(String, u32)> = Vec::new();

    if sections.observer {
        let time_status = if info.paused {
            "PAUSED".to_string()
        } else if info.time_scale < 0.0 {
            format!("{:+5.2}X (REWIND)", info.time_scale)
        } else {
            format!("{:+5.2}X (FWD)", info.time_scale)
        };
        lines.push((
            format!(
                "OBS EVENT (T,X,Y,Z): ({:+6.2}, {:+6.2}, {:+6.2}, {:+6.2})  OBS TAU: {:+6.2}S  TIME: {}",
                camera.position.t,
                camera.position.x,
                camera.position.y,
                camera.position.z,
                camera.proper_time,
                time_status
            ),
            0xFFE866,
        ));
        let beta = camera.tetrad.beta();
        lines.push((
            format!(
                "OBS 3-VEL: ({:+5.3}, {:+5.3}, {:+5.3})  |V|: {:5.3}C  GAMMA: {:6.3}",
                beta.x,
                beta.y,
                beta.z,
                beta.length(),
                camera.tetrad.gamma()
            ),
            0x66F2FF,
        ));
    }
    if sections.render {
        let view_name = if cfg.slice_mode == ViewMode::ProphecySlice {
            format!("PROPHECY (DT'=+{:.1}S)", cfg.prophecy_delta)
        } else {
            cfg.slice_mode.name().to_string()
        };
        lines.push((
            format!(
                "MODE [M]: {} | VIEW [G]: {} | SHADOWS [H]: {}",
                cfg.mode.name(),
                view_name,
                cfg.shadow_mode.name(),
            ),
            0xA8FF80,
        ));
        lines.push((
            format!(
                "FPS: {:4.0} | BACKEND [F]: {} | RES [Z]: {}x{}",
                info.perf.fps(),
                info.backend,
                cfg.width,
                cfg.height
            ),
            0x66F2FF,
        ));
    }
    if let Some(net) = info.net.as_ref().filter(|_| sections.net) {
        lines.push((net.clone(), 0xDDDDDD));
    }
    if sections.keys {
        // Most time controls are disabled in a match.
        if info.net.is_none() {
            lines.push((
                "TIME: [P] PAUSE | [T] REWIND | [-/=] SPEED | [ / ] SCRUB T | [R] RESET | [/] FLASH | [PGUP/DN] DELTA".into(),
                0xDDDDDD,
            ));
        }
        lines.push((
            "THRUST: WASD/SPC/C | MATCH VEL: [V] | BRAKE: [X] | LOOK: ARROWS/QE | MOVE: IJKL/UO".into(),
            0xAAAAAA,
        ));
    }

    let (x0, y0, box_w) = (12, 12, c.panel_width());
    let toggles_y = y0 + 62 + LINE_H * lines.len();
    let box_h = toggles_y + 14 + 10 - y0;
    c.darken(x0, y0, box_w, box_h);

    c.text(22, 22, &format!("SCENE [1-6 / N,B]: {}", info.demo.title()), 0xFF9944);
    if let Some(score) = &info.score {
        // Right-aligned in the box, in the laser's color.
        let x = (x0 + box_w).saturating_sub(10 + text_width(score));
        c.text(x, 22, score, RETICLE_COLOR);
    }
    c.text(22, 46, &info.desc, 0xFFD980);
    for (i, (line, color)) in lines.iter().enumerate() {
        c.text(22, y0 + 62 + LINE_H * i, line, *color);
    }

    // The section toggles, lit when on.
    let mut x = 22;
    for (i, name) in HudSections::NAMES.iter().enumerate() {
        let item = format!("[F{}] {name}", i + 1);
        let color = if sections.is_on(i) { 0xFFFFFF } else { 0x666666 };
        c.text(x, toggles_y, &item, color);
        x += text_width(&item) + 24;
    }
    y0 + box_h
}

/// Colors of the performance bars, cycled through the phases.
const BAR_COLORS: [u32; 6] = [0x66F2FF, 0xFF9944, 0xA8FF80, 0xFFE866, 0xFF6A9A, 0xB48CFF];

/// The performance panel at `y0`: one row per frame phase with its average and peak
/// milliseconds, and a bar of its share of the frame. Returns the panel's bottom edge.
fn draw_perf_panel(c: &mut Canvas, info: &HudInfo, y0: usize) -> usize {
    let perf = info.perf;
    let frame = perf.frame();
    let rows = perf.rows();
    let n_rows = 2 + rows.len() + info.perf_context.len();
    let (x0, box_w) = (12, c.panel_width());
    let box_h = 10 + ROW_H * n_rows + 4;
    c.darken(x0, y0, box_w, box_h);

    let mut y = y0 + 10;
    c.text(
        22,
        y,
        &format!(
            "PERFORMANCE: {:.2}MS/FRAME ({:.0} FPS), WORST {:.2}MS IN LAST 1S",
            frame.avg_ms,
            perf.fps(),
            frame.peak_ms
        ),
        0xFFE866,
    );
    y += ROW_H;
    c.text(22, y, &format!("{:<20}{:>7}{:>7}{:>8}", "PHASE", "AVG MS", "FRAME", "PEAK"), 0x999999);
    y += ROW_H;

    // Bars: 100% of the average frame spans the rest of the panel.
    let bar_x = 22 + text_width(&format!("{:<20}{:>7}{:>7}{:>8}", "", "", "", "")) + 16;
    let bar_max = (x0 + box_w).saturating_sub(bar_x + 10);
    let frac = |ms: f32| if frame.avg_ms > 0.0 { (ms / frame.avg_ms).clamp(0.0, 1.0) } else { 0.0 };
    // `peak: None` for rows without one.
    let draw_row = |c: &mut Canvas, y: usize, label: &str, avg: f32, peak: Option<f32>, color: u32, sub: bool| {
        let label = if sub { format!("  {label}") } else { label.to_string() };
        let pct = 100.0 * frac(avg);
        let peak_text = peak.map_or(format!("{:>8}", "-"), |p| format!("{p:>8.2}"));
        c.text(22, y, &format!("{label:<20}{avg:>7.2}{pct:>6.0}%{peak_text}"), if sub { 0xAAAAAA } else { 0xDDDDDD });
        let (bar_y, bar_h) = if sub { (y + 4, 6) } else { (y + 1, 12) };
        let len = (frac(avg) * bar_max as f32).round() as usize;
        c.fill(bar_x, bar_y, len, bar_h, color);
        // The peak, as a tick (capped at the end of the bar: a whole average frame).
        if let Some(peak) = peak {
            let tick = (frac(peak) * bar_max as f32).round() as usize;
            c.fill(bar_x + tick.min(bar_max.saturating_sub(2)), y, 2, 14, 0xFFFFFF);
        }
    };

    let mut color_idx = 0;
    let mut color = BAR_COLORS[0];
    for row in rows {
        if row.label == perf::OTHER {
            color = 0x666666;
        } else if !row.sub {
            color = BAR_COLORS[color_idx % BAR_COLORS.len()];
            color_idx += 1;
        }
        draw_row(c, y, row.label, row.avg_ms, row.peak_ms, color, row.sub);
        y += ROW_H;
    }

    for line in &info.perf_context {
        c.text(22, y, line, 0x999999);
        y += ROW_H;
    }
    y0 + box_h
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
        '%' => [0b11001, 0b11010, 0b00010, 0b00100, 0b01000, 0b01011, 0b10011],
        '*' => [0b00000, 0b10101, 0b01110, 0b11111, 0b01110, 0b10101, 0b00000],
        '&' => [0b01100, 0b10010, 0b10100, 0b01000, 0b10101, 0b10010, 0b01101],
        _ => [0; 7],
    }
}

