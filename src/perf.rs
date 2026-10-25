//! Frame timing for the HUD's performance section (`[F5]`): where each frame's wall-clock time
//! goes, phase by phase (simulation, BVH rebuilds, rendering, readback, HUD, presenting).
//!
//! Frontends and [`App`](crate::app::App) bracket each phase with [`now_ms`] and
//! [`PerfStats::record`] it; the frontend closes each frame with [`PerfStats::end_frame`]. Phases
//! a frame skips count as zero, so periodic work (a BVH rebuild every 0.1 s) shows its amortized
//! cost as the average and its stutter as the peak.

use crate::renderer::RenderConfig;
use crate::scene::Scene4D;

/// Weight of the newest frame in the running averages.
const EMA_ALPHA: f32 = 0.1;
/// Peaks are the worst frame over windows of this much wall-clock time.
const PEAK_WINDOW_MS: f32 = 1000.0;

#[cfg(target_arch = "wasm32")]
#[link(wasm_import_module = "env")]
extern "C" {
    /// `performance.now()`, supplied by the page when it instantiates the module.
    fn perf_now() -> f64;
}

/// Milliseconds on a monotonic clock with an arbitrary origin.
#[cfg(not(target_arch = "wasm32"))]
pub fn now_ms() -> f64 {
    use std::sync::OnceLock;
    use std::time::Instant;
    static START: OnceLock<Instant> = OnceLock::new();
    START.get_or_init(Instant::now).elapsed().as_secs_f64() * 1000.0
}

/// Milliseconds on a monotonic clock with an arbitrary origin.
#[cfg(target_arch = "wasm32")]
pub fn now_ms() -> f64 {
    unsafe { perf_now() }
}

/// Milliseconds since `start` (a [`now_ms`] reading).
pub fn since(start: f64) -> f32 {
    (now_ms() - start) as f32
}

/// The pseudo-phase of time no recorded phase accounts for (event loop, waiting for vsync...).
pub const OTHER: &str = "OTHER (IDLE, VSYNC)";

/// One row of the performance breakdown.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PerfRow {
    pub label: &'static str,
    pub avg_ms: f32,
    /// Worst frame in the last full window (`None` for [`OTHER`]).
    pub peak_ms: Option<f32>,
    /// A sub-timing: part of the phase above it, not counted towards the frame.
    pub sub: bool,
}

/// One timed quantity: its running average, and its peak over the last full window.
#[derive(Clone, Debug)]
pub struct Timing {
    pub label: &'static str,
    pub avg_ms: f32,
    pub peak_ms: f32,
    window_peak_ms: f32,
    this_frame_ms: f32,
    /// Sub-timings only: the phase this one is part of (shown under it, not summed).
    within: Option<&'static str>,
}

impl Timing {
    fn new(label: &'static str, within: Option<&'static str>) -> Self {
        Self { label, avg_ms: 0.0, peak_ms: 0.0, window_peak_ms: 0.0, this_frame_ms: 0.0, within }
    }

    fn add(&mut self, ms: f32) {
        self.avg_ms += EMA_ALPHA * (ms - self.avg_ms);
        self.window_peak_ms = self.window_peak_ms.max(ms);
    }

    fn close_window(&mut self) {
        self.peak_ms = self.window_peak_ms;
        self.window_peak_ms = 0.0;
    }

    /// Long gone: nothing recorded for a whole window and the average has decayed away.
    fn is_stale(&self) -> bool {
        self.peak_ms == 0.0 && self.avg_ms < 0.001
    }
}

#[derive(Clone, Debug)]
pub struct PerfStats {
    /// Disjoint parts of the frame, in the order first recorded.
    phases: Vec<Timing>,
    /// Measurements of part of a phase, taken whenever available (e.g. the GPU kernel's own
    /// time, from timestamp queries).
    subs: Vec<Timing>,
    frame: Timing,
    window_ms: f32,
}

impl Default for PerfStats {
    fn default() -> Self {
        Self {
            phases: Vec::new(),
            subs: Vec::new(),
            frame: Timing::new("FRAME", None),
            window_ms: 0.0,
        }
    }
}

impl PerfStats {
    /// `ms` spent in phase `label` this frame (repeated calls add up).
    pub fn record(&mut self, label: &'static str, ms: f32) {
        match self.phases.iter_mut().find(|t| t.label == label) {
            Some(t) => t.this_frame_ms += ms,
            None => {
                let mut t = Timing::new(label, None);
                t.this_frame_ms = ms;
                self.phases.push(t);
            }
        }
    }

    /// A measurement of part of phase `within` (possibly [`OTHER`]), which need not arrive
    /// every frame.
    pub fn sample(&mut self, label: &'static str, within: &'static str, ms: f32) {
        let i = match self.subs.iter().position(|t| t.label == label) {
            Some(i) => i,
            None => {
                self.subs.push(Timing::new(label, Some(within)));
                self.subs.len() - 1
            }
        };
        self.subs[i].add(ms);
    }

    /// Close a frame that took `frame_ms` of wall-clock time: phases it did not record count as
    /// zero.
    pub fn end_frame(&mut self, frame_ms: f32) {
        self.frame.add(frame_ms);
        for t in &mut self.phases {
            let ms = std::mem::take(&mut t.this_frame_ms);
            t.add(ms);
        }
        self.window_ms += frame_ms;
        if self.window_ms >= PEAK_WINDOW_MS {
            self.window_ms = 0.0;
            self.frame.close_window();
            for t in self.phases.iter_mut().chain(&mut self.subs) {
                t.close_window();
            }
            self.phases.retain(|t| !t.is_stale());
            // A sub-timing that stopped arriving (e.g. after switching to the CPU) is gone.
            self.subs.retain(|t| t.peak_ms > 0.0);
        }
    }

    pub fn frame(&self) -> &Timing {
        &self.frame
    }

    /// Frames per second, from the average frame time.
    pub fn fps(&self) -> f32 {
        if self.frame.avg_ms > 0.0 {
            1000.0 / self.frame.avg_ms
        } else {
            0.0
        }
    }

    /// Average milliseconds of phase or sub-timing `label` (0 if never recorded).
    pub fn avg_ms(&self, label: &str) -> f32 {
        let mut all = self.phases.iter().chain(&self.subs);
        all.find(|t| t.label == label).map_or(0.0, |t| t.avg_ms)
    }

    /// The phases in the order first recorded, then [`OTHER`], each followed by its
    /// sub-timings.
    pub fn rows(&self) -> Vec<PerfRow> {
        let mut rows = Vec::new();
        let phases = self.phases.iter().map(|t| (t.label, t.avg_ms, Some(t.peak_ms)));
        for (label, avg_ms, peak_ms) in phases.chain([(OTHER, self.unaccounted_ms(), None)]) {
            rows.push(PerfRow { label, avg_ms, peak_ms, sub: false });
            let subs = self.subs.iter().filter(|s| s.within == Some(label));
            rows.extend(subs.map(|s| PerfRow { label: s.label, avg_ms: s.avg_ms, peak_ms: Some(s.peak_ms), sub: true }));
        }
        rows
    }

    /// The part of the average frame no phase accounts for.
    pub fn unaccounted_ms(&self) -> f32 {
        (self.frame.avg_ms - self.phases.iter().map(|t| t.avg_ms).sum::<f32>()).max(0.0)
    }
}

/// What the frame's rendering cost scales with: pixels, and geometry the rays traverse.
/// `render_ms` is the average time spent producing the image.
pub fn render_context(scene: &Scene4D, cfg: &RenderConfig, render_ms: f32) -> Vec<String> {
    let pixels = cfg.width * cfg.height;
    let ns_per_px = if pixels > 0 { render_ms * 1e6 / pixels as f32 } else { 0.0 };
    vec![
        format!(
            "PIXELS {}X{} = {:.2} MPX | RENDER {:.2}MS = {:.1} NS/PX",
            cfg.width,
            cfg.height,
            pixels as f32 / 1e6,
            render_ms,
            ns_per_px
        ),
        format!(
            "SCENE {} SIMPLICES, {} BVH NODES | SHADOWS {} | VIEW {}",
            scene.bvh.simplices.len(),
            scene.bvh.nodes().len(),
            cfg.shadow_mode.name(),
            cfg.slice_mode.name(),
        ),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn skipped_phases_average_in_as_zero_and_keep_their_peak() {
        let mut perf = PerfStats::default();
        for frame in 0..200 {
            perf.record("SIM", 1.0);
            if frame % 10 == 0 {
                perf.record("REBUILD", 20.0);
            }
            perf.end_frame(10.0);
        }
        assert!((perf.avg_ms("SIM") - 1.0).abs() < 1e-3);
        let rebuild = perf.rows().into_iter().find(|r| r.label == "REBUILD").unwrap();
        assert!(rebuild.avg_ms > 0.5 && rebuild.avg_ms < 5.0, "amortized: {}", rebuild.avg_ms);
        assert_eq!(rebuild.peak_ms, Some(20.0));
        assert!((perf.fps() - 100.0).abs() < 0.1);
        assert!((perf.unaccounted_ms() - (10.0 - 1.0 - rebuild.avg_ms)).abs() < 1e-3);
    }

    #[test]
    fn sub_timings_follow_their_phase_and_stale_phases_drop_out() {
        let mut perf = PerfStats::default();
        perf.record("A", 1.0);
        perf.record("WAIT", 2.0);
        perf.sample("KERNEL", "WAIT", 1.5);
        perf.sample("ASYNC", OTHER, 0.5);
        perf.end_frame(5.0);
        let labels: Vec<_> = perf.rows().iter().map(|r| (r.label, r.sub)).collect();
        assert_eq!(
            labels,
            [("A", false), ("WAIT", false), ("KERNEL", true), (OTHER, false), ("ASYNC", true)]
        );

        // "A" stops being recorded: it decays and is dropped at a window's end.
        for _ in 0..1000 {
            perf.record("WAIT", 2.0);
            perf.end_frame(5.0);
        }
        assert!(perf.rows().iter().all(|r| r.label != "A"));
    }
}
