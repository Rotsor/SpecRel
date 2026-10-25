//! The ship's controls, made causal (see `docs/SHIP_CONTROLS_CAUSALITY.md`). The hull is a
//! rigid body of radius [`SHIP_RADIUS`]: it may only act on a command once news of it has
//! reached every part of it, so that all parts run their Born-rigid share of the command at the
//! same moment of the ship's clock. Two ways to live with that:
//!
//! * [`ControlMode::Delayed`] (option A): the pilot sits at the hull's center and every command
//!   waits [`command_delay`] of ship time before the whole ship executes it.
//! * [`ControlMode::Pod`] (option C): the pilot flies a small control pod inside the hull with
//!   no lag. The hull keeps station on the pod, steering by what it learned [`pod_delay`] ago,
//!   and a pod that outruns it bounces off the inner wall.

use std::collections::VecDeque;

use super::billiards::{POD_RANGE, SHIP_ACCEL, SHIP_MASS};
use super::collide::P4;
use super::ship::{Ship, ShipSample};
use crate::minkowski::{project_to_rest_space, Tetrad, Vec4};
use crate::renderer::Frontier;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ControlMode {
    /// Option A: commands act on the whole ship after [`command_delay`].
    #[default]
    Delayed,
    /// Option C: a lag-free control pod inside a hull that follows it.
    Pod,
}

impl ControlMode {
    pub fn next(self) -> Self {
        match self {
            Self::Delayed => Self::Pod,
            Self::Pod => Self::Delayed,
        }
    }

    /// The pilot's proper acceleration, in `c` per second of their clock.
    pub fn pilot_accel(self) -> f32 {
        match self {
            Self::Delayed => SHIP_ACCEL as f32,
            Self::Pod => pod_accel(),
        }
    }
}

/// Proper time of the hull's center for light to get from `ahead` in front of the center to
/// `behind` behind it, while the hull accelerates forward at `accel` (Rindler coordinates
/// `ds^2 = -(1 + a x)^2 dtau^2 + dx^2`; light crawls toward the horizon behind the ship). This is
/// the slowest any signal crosses that span under an acceleration program bounded by `accel`.
pub fn signal_delay(accel: f64, ahead: f64, behind: f64) -> f64 {
    ((1.0 + accel * ahead) / (1.0 - accel * behind)).ln() / accel
}

// The constants below feed the hull's motion, which every replica computes, so they are
// literals rather than calls to a platform's `ln`; tests check them against the formulas.

/// Option A: how long a command takes to reach the far side of the hull from its center,
/// `signal_delay(SHIP_ACCEL, 0, SHIP_RADIUS)`.
pub fn command_delay() -> f32 {
    0.804_719
}

/// Option C: how long the pod's push takes to cross the hull, from as far as [`POD_RANGE`]
/// off center to the far wall: `signal_delay(SHIP_ACCEL, POD_RANGE, SHIP_RADIUS)`.
pub fn pod_delay() -> f32 {
    0.972_955
}

/// Option C: how hard the pod may push and still (roughly) stay within [`POD_RANGE`] of the
/// hull's center, which reacts [`pod_delay`] late at up to [`SHIP_ACCEL`]: solves
/// `range = a_c D^2 a_h / (2 (a_h - a_c))` for `a_c`.
pub fn pod_accel() -> f32 {
    0.348_844
}

/// One frame's worth of control input.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Order {
    /// Thrust along this lab-frame spatial direction of the issuing rest frame.
    Thrust(Vec4),
    /// Boost toward this 4-velocity.
    Steer(Vec4),
}

impl Order {
    /// Carry out the order on `tetrad` with a boost of up to `rapidity`.
    pub fn execute(self, tetrad: &mut Tetrad, rapidity: f32) {
        match self {
            Self::Steer(target) => {
                tetrad.boost_toward(target, rapidity);
            }
            Self::Thrust(dir) => {
                // The velocity may have changed since the order was issued: thrust along the
                // part of the direction that is spatial now.
                let u = tetrad.e[0];
                let n = project_to_rest_space(dir, u);
                let len = n.minkowski_norm_sq().max(0.0).sqrt();
                if len > 1e-6 {
                    let n = n / len;
                    tetrad.boost_to(u * rapidity.cosh() + n * rapidity.sinh());
                }
            }
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct Command {
    /// Ship proper time at which the whole hull knows of it.
    due: f32,
    rapidity: f32,
    order: Order,
}

/// Option A: commands broadcast from the center at light speed, waiting until they have
/// reached the whole ship.
#[derive(Clone, Debug, Default)]
pub struct CommandQueue {
    queue: VecDeque<Command>,
}

impl CommandQueue {
    /// Issue `order` at ship proper time `now`, for a boost of `rapidity`.
    pub fn issue(&mut self, now: f32, order: Order, rapidity: f32) {
        self.queue.push_back(Command { due: now + command_delay(), rapidity, order });
    }

    /// Execute on the ship's `tetrad` every command that has crossed the hull by `now`.
    pub fn execute_due(&mut self, now: f32, tetrad: &mut Tetrad) {
        while let Some(c) = self.queue.front().filter(|c| c.due <= now) {
            c.order.execute(tetrad, c.rapidity);
            self.queue.pop_front();
        }
    }

    /// Commands issued but not executed yet: the engines are spooling up.
    pub fn pending(&self) -> usize {
        self.queue.len()
    }

    pub fn clear(&mut self) {
        self.queue.clear();
    }
}

/// Station-keeping gains: critically damped at 2 rad/s (per second of ship time).
const KEEP_P: f32 = 4.0;
const KEEP_D: f32 = 4.0;

/// How much of the pod's speed toward the wall survives a bounce.
const WALL_RESTITUTION: f32 = 0.0;

/// Option C: the hull's center, moving by its own causal rules.
#[derive(Clone, Copy, Debug)]
pub struct Hull {
    pub event: Vec4,
    /// 4-velocity
    pub u: Vec4,
    /// The hull's own clock
    pub tau: f32,
}

impl Hull {
    pub fn sample(&self) -> ShipSample {
        ShipSample::new(Frontier { origin: self.event, proper_time: self.tau }, self.u)
    }

    pub fn from_sample(s: &ShipSample) -> Self {
        Self { event: s.event(), u: s.u(), tau: s.tau as f32 }
    }

    /// The proper acceleration (a spacelike 4-vector orthogonal to `u`, at most
    /// [`SHIP_ACCEL`]) with which the hull steers toward its pilot, judging from `ship`'s
    /// tracks only what has crossed the whole hull by now: see [`Hull::predict_pilot`].
    pub fn steering(&self, ship: &Ship) -> Vec4 {
        let pilot = self.predict_pilot(ship);
        let r = self.offset(pilot.event);
        let w = project_to_rest_space(pilot.u, self.u) / -pilot.u.minkowski_dot(self.u);
        let a = project_to_rest_space(r * KEEP_P + w * KEEP_D, self.u);
        let len = a.minkowski_norm_sq().max(0.0).sqrt();
        let max = SHIP_ACCEL as f32;
        if len > max {
            a * (max / len)
        } else {
            a
        }
    }

    /// Where the pilot is now (at the hull's lab time), as far as the hull can know: as it last
    /// heard of them, [`pod_delay`] ago on its own simultaneity slice, flown on as they were
    /// then flying (keeping up their acceleration), and bouncing off the inner wall wherever
    /// the hull's own track put it. The hull's past is its own business, so this is exact
    /// unless the pilot changed their thrust since; without the wall, the hull would steer
    /// by bounces it hears of too late, and the two would rattle apart.
    fn predict_pilot(&self, ship: &Ship) -> Hull {
        let then = ship.hull_at_tau((self.tau - pod_delay()) as f64);
        let seen = ship.pilot_on_slice(then.event(), then.u());
        // The pilot's 4-acceleration then, from the track (capped: wall bounces are kicks).
        let earlier = ship.at(seen.t - ACCEL_WINDOW);
        let dtau = (seen.tau - earlier.tau) as f32;
        let mut accel = if dtau > 1e-4 { (seen.u() - earlier.u()) / dtau } else { Vec4::default() };
        let len = accel.minkowski_norm_sq().max(0.0).sqrt();
        let cap = 1.5 * pod_accel();
        if len > cap {
            accel = accel * (cap / len);
        }

        let mut pilot = Self::from_sample(&seen);
        let fly = |pilot: &mut Hull, t: f32, hull: &Hull| {
            pilot.advance_to(t, project_to_rest_space(accel, pilot.u));
            if let Some(u) = hull.bounced(pilot.event, pilot.u) {
                pilot.u = u;
            }
        };
        let from = ship.hull.partition_point(|s| s.t <= seen.t);
        for s in &ship.hull[from..] {
            fly(&mut pilot, s.t as f32, &Self::from_sample(s));
        }
        fly(&mut pilot, self.event.t, self);
        pilot
    }

    /// Fly on to lab time `t` under proper acceleration `accel`.
    pub fn advance_to(&mut self, t: f32, accel: Vec4) {
        let dt = t - self.event.t;
        if dt <= 0.0 {
            return;
        }
        let dtau = dt / self.u.t;
        let len = accel.minkowski_norm_sq().max(0.0).sqrt();
        // First order in `dtau`, and `sqrt` only: replicas must agree on every bit.
        let u = if len > 1e-6 {
            normalized(self.u + accel * dtau)
        } else {
            self.u
        };
        let v = normalized(self.u + u);
        self.event += v * (dt / v.t);
        self.tau += dt / v.t;
        self.u = u;
    }

    /// The pilot at `pilot` (at the hull's lab time) relative to the hull's center, in the
    /// hull's rest frame.
    pub fn offset(&self, pilot: Vec4) -> Vec4 {
        project_to_rest_space(pilot - self.event, self.u)
    }

    /// The pilot (at event `pilot`, frame `tetrad`) bounces off the hull's inner wall, at
    /// [`POD_RANGE`] from the center, if they are moving out through it. The hull is far
    /// heavier and doesn't notice. Returns whether they bounced.
    pub fn bounce(&self, pilot: Vec4, tetrad: &mut Tetrad) -> bool {
        let bounced = self.bounced(pilot, tetrad.e[0]);
        if let Some(u) = bounced {
            tetrad.boost_to(u);
        }
        bounced.is_some()
    }

    /// The 4-velocity of a pilot at `pilot` moving with `up` after bouncing off the inner wall,
    /// if they hit it.
    fn bounced(&self, pilot: Vec4, up: Vec4) -> Option<Vec4> {
        let r = self.offset(pilot);
        let dist = r.minkowski_norm_sq().max(0.0).sqrt();
        if dist <= POD_RANGE as f32 {
            return None;
        }
        let n = r / dist;
        let w = project_to_rest_space(up, self.u) / -up.minkowski_dot(self.u);
        let out = w.minkowski_dot(n);
        if out <= 0.0 {
            return None;
        }
        let w = w - n * ((1.0 + WALL_RESTITUTION) * out);
        let gamma = 1.0 / (1.0 - w.minkowski_norm_sq()).max(1e-6).sqrt();
        Some((self.u + w) * gamma)
    }
}

/// Lab time over which the hull estimates its pilot's acceleration.
const ACCEL_WINDOW: f64 = 0.1;

/// Scale a timelike vector to unit norm.
fn normalized(v: Vec4) -> Vec4 {
    v / (-v.minkowski_norm_sq()).sqrt()
}

/// The 4-velocity of a ship (rest mass [`SHIP_MASS`]) moving with `u` once it receives
/// collision 4-momentum `kick`.
pub fn kicked(u: Vec4, kick: P4) -> Vec4 {
    let p = [
        SHIP_MASS * u.t as f64 + kick[0],
        SHIP_MASS * u.x as f64 + kick[1],
        SHIP_MASS * u.y as f64 + kick[2],
        SHIP_MASS * u.z as f64 + kick[3],
    ];
    let m = (p[0] * p[0] - p[1] * p[1] - p[2] * p[2] - p[3] * p[3]).sqrt();
    let [t, x, y, z] = p.map(|c| (c / m) as f32);
    Vec4::new(t, x, y, z)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dynamic::billiards::SHIP_RADIUS;
    use crate::minkowski::Vec3;

    /// In the momentarily comoving frame of a hull accelerating at `a` along `+x`, a part at
    /// rest offset `x0` from the center is at `((1/a + x0) sinh(a tau), (1/a + x0) cosh(a tau) -
    /// 1/a)` when the center's clock reads `tau`.
    fn rindler(a: f64, x0: f64, tau: f64) -> (f64, f64) {
        let r = 1.0 / a + x0;
        (r * (a * tau).sinh(), r * (a * tau).cosh() - 1.0 / a)
    }

    /// Light leaving `ahead` of the center at `tau = 0` reaches the part `behind` it exactly
    /// when the center's clock reads `signal_delay`.
    #[test]
    fn signal_delay_is_the_light_crossing_time_of_an_accelerating_hull() {
        let a = SHIP_ACCEL;
        for (ahead, behind) in [(0.0, SHIP_RADIUS), (POD_RANGE, SHIP_RADIUS), (0.1, 0.05)] {
            let d = signal_delay(a, ahead, behind);
            // The light ray x = ahead - t meets the rear's worldline.
            let gap = |tau: f64| {
                let (t, x) = rindler(a, -behind, tau);
                x - (ahead - t)
            };
            assert!(gap(d).abs() < 1e-9, "{ahead} -> {behind}: {}", gap(d));
            assert!(gap(0.99 * d) < 0.0, "not yet just before");
        }
        let close = |a: f32, b: f64| (a as f64 - b).abs() < 1e-6;
        assert!(close(command_delay(), signal_delay(a, 0.0, SHIP_RADIUS)));
        let d = signal_delay(a, POD_RANGE, SHIP_RADIUS);
        assert!(close(pod_delay(), d));
        assert!(close(pod_accel(), 2.0 * POD_RANGE * a / (d * d * a + 2.0 * POD_RANGE)));
        assert!(signal_delay(1e-9, 0.0, 0.3) > 0.3 - 1e-6, "light speed with no acceleration");
    }

    #[test]
    fn commands_wait_until_they_have_crossed_the_hull() {
        let mut q = CommandQueue::default();
        let mut tetrad = Tetrad::rest();
        q.issue(0.0, Order::Thrust(Vec4::new(0.0, 0.0, 0.0, 1.0)), 0.1);
        q.issue(0.5, Order::Steer(Vec4::new(1.0, 0.0, 0.0, 0.0)), 0.1);
        q.execute_due(command_delay() - 0.01, &mut tetrad);
        assert_eq!(tetrad.beta(), Vec3::default());
        assert_eq!(q.pending(), 2);
        q.execute_due(command_delay(), &mut tetrad);
        assert!((tetrad.e[0].z.asinh() - 0.1).abs() < 1e-5, "{:?}", tetrad.e[0]);
        q.execute_due(command_delay() + 0.5, &mut tetrad);
        assert!(tetrad.beta().length() < 1e-5, "steered back to rest");
        assert_eq!(q.pending(), 0);
    }

    /// One frame of option C, in the order `App::step` runs it.
    struct Pod {
        ship: Ship,
        pilot: Tetrad,
        at: Frontier,
    }

    impl Pod {
        fn new() -> Self {
            let at = Frontier { origin: Vec4::new(0.0, 0.0, 0.0, 0.0), proper_time: 0.0 };
            let mut ship = Ship::new(0, ShipSample::new(at, Vec4::new(1.0, 0.0, 0.0, 0.0)));
            ship.set_controls(ControlMode::Pod);
            Pod { ship, pilot: Tetrad::rest(), at }
        }

        /// The hull now, as the pilot knows it.
        fn hull(&self) -> Hull {
            self.ship.hull_estimate(self.at.origin.t as f64)
        }

        fn step(&mut self, dt: f32, thrust: Option<Vec3>) -> bool {
            let bounced = self.hull().bounce(self.at.origin, &mut self.pilot);
            if let Some(dir) = thrust {
                self.pilot.boost_local(dir * (pod_accel() * dt));
            }
            self.at.origin += self.pilot.e[0] * dt;
            self.at.proper_time += dt;
            self.ship.record(ShipSample::new(self.at, self.pilot.e[0]));
            bounced
        }

        fn offset(&self) -> f32 {
            self.hull().offset(self.at.origin).minkowski_norm_sq().sqrt()
        }
    }

    /// The hull doesn't move until news of the pod's thrust has had time to cross it, then
    /// catches up and keeps the pod near its center; the pod accelerates at once.
    #[test]
    fn the_hull_keeps_station_on_the_pod() {
        let mut pod = Pod::new();
        let dt = 1.0 / 60.0;
        let forward = Vec3::new(0.0, 0.0, 1.0);
        let (mut max_offset, mut bounces) = (0.0f32, 0);
        for frame in 0..600 {
            let thrust = (frame < 240).then_some(forward);
            bounces += pod.step(dt, thrust) as usize;
            max_offset = max_offset.max(pod.offset());
            if pod.at.proper_time < pod_delay() - 2.0 * dt {
                assert_eq!(pod.hull().u, Vec4::new(1.0, 0.0, 0.0, 0.0), "frame {frame}: too early");
            }
        }
        assert!(pod.pilot.beta().z > 0.6, "the pod reached speed: {:?}", pod.pilot.beta());
        println!("max offset {max_offset}, bounces {bounces}, final {}", pod.offset());
        assert!(max_offset < SHIP_RADIUS as f32, "the pod stayed inside the hull");
        assert!(pod.offset() < 0.02, "and ended up back at the center");
        let rel = project_to_rest_space(pod.pilot.e[0], pod.hull().u).minkowski_norm_sq().sqrt();
        assert!(rel < 0.01, "at rest relative to the hull: {rel}");
    }

    /// Bursts of thrust, reversals and coasting, at coarse and fine frame rates: the hull never
    /// lets the pod far out of the wall's range and settles around it once it coasts.
    #[test]
    fn station_keeping_settles_after_any_burst() {
        let x = Vec3::new(1.0, 0.0, 0.0);
        let programs: [&[(f32, Option<Vec3>)]; 4] = [
            &[(1.2, Some(x)), (8.0, None)],
            &[(0.4, Some(x)), (8.0, None)],
            &[(2.0, Some(x)), (2.0, Some(-x)), (8.0, None)],
            &[(0.6, Some(x)), (0.6, Some(-x)), (0.6, Some(x)), (0.6, Some(Vec3::new(0.0, 0.0, 1.0))), (8.0, None)],
        ];
        let mut ok = true;
        for dt in [0.05, 1.0 / 60.0] {
            for (n, program) in programs.iter().enumerate() {
                let mut pod = Pod::new();
                let (mut max_offset, mut bounces) = (0.0f32, 0);
                for &(secs, thrust) in program.iter() {
                    for _ in 0..(secs / dt).round() as usize {
                        bounces += pod.step(dt, thrust) as usize;
                        max_offset = max_offset.max(pod.offset());
                    }
                }
                let rel = project_to_rest_space(pod.pilot.e[0], pod.hull().u).minkowski_norm_sq().sqrt();
                println!("dt {dt:.3} program {n}: max offset {max_offset:.3}, {bounces} bounces, final {:.4} at {rel:.4}", pod.offset());
                ok &= max_offset < 0.25;
                ok &= pod.offset() < 0.02 && rel < 0.01;
            }
        }
        assert!(ok, "the hull lost its pod");
    }

    #[test]
    fn a_pod_outrunning_the_hull_bounces_off_the_wall() {
        let mut pod = Pod::new();
        pod.pilot.boost_to(Vec4::four_velocity_from_beta(Vec3::new(0.3, 0.0, 0.0)));
        let mut bounced = false;
        for _ in 0..120 {
            bounced |= pod.step(1.0 / 60.0, None);
            assert!(pod.offset() < POD_RANGE as f32 + 0.01);
        }
        assert!(bounced);
    }
}
