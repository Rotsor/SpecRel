//! Keeping the local player out of the other player's future light cone (docs/MULTIPLAYER.md
//! §1). All comparisons happen in the centre-of-momentum frame of the two ships, so the rule
//! does not favour either player's frame.

use crate::dynamic::ShipSample;

/// How close (in CM-frame light-seconds) the local ship may get to the remote's future cone.
pub const CONE_MARGIN: f64 = 1.5;

/// The separation `O_local - O_remote` in the two ships' centre-of-momentum frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Lead {
    /// CM time by which the local ship is ahead.
    pub dt: f64,
    /// CM distance between the two events.
    pub dist: f64,
}

pub fn lead(local: &ShipSample, remote: &ShipSample) -> Lead {
    let (a, b) = (local.four_velocity(), remote.four_velocity());
    let s = [a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]];
    let norm = (s[0] * s[0] - s[1] * s[1] - s[2] * s[2] - s[3] * s[3]).sqrt();
    let u = s.map(|c| c / norm);
    let d = [
        local.t - remote.t,
        local.pos[0] - remote.pos[0],
        local.pos[1] - remote.pos[1],
        local.pos[2] - remote.pos[2],
    ];
    let dt = u[0] * d[0] - u[1] * d[1] - u[2] * d[2] - u[3] * d[3];
    let interval = -d[0] * d[0] + d[1] * d[1] + d[2] * d[2] + d[3] * d[3];
    Lead { dt, dist: (dt * dt + interval).max(0.0).sqrt() }
}

/// Whether `local` is ahead in CM time and within [`CONE_MARGIN`] of the remote's future cone.
/// True for every event inside `I⁺(remote)`.
pub fn must_wait(local: &ShipSample, remote: &ShipSample) -> bool {
    let l = lead(local, remote);
    l.dt > 0.0 && l.dt > l.dist - CONE_MARGIN
}

/// Whether the remote is level with or ahead of `local` in CM time.
pub fn caught_up(local: &ShipSample, remote: &ShipSample) -> bool {
    lead(local, remote).dt <= 0.0
}

/// The stop/resume state machine for the local ship.
#[derive(Clone, Copy, Debug, Default)]
pub struct Pacer {
    waiting: bool,
}

impl Pacer {
    pub fn is_waiting(&self) -> bool {
        self.waiting
    }

    /// Whether the local ship may move from `current` to the tentative `next` event, given the
    /// latest known `remote` event.
    ///
    /// Checking `next` (not `current`) means the local event never enters `I⁺(remote)` from
    /// ahead, since [`must_wait`] holds everywhere inside that cone. Once stopped, the ship
    /// resumes only when the remote has caught up in CM time (hysteresis).
    ///
    /// A ship that is not ahead (`caught_up`) always advances. Since `lead` is antisymmetric,
    /// with exact information at least one of the two ships can always move. The price: two
    /// ships within one step of each other in CM time *and* space take turns stepping, so the
    /// trailing one may poke one step into the other's cone until that one catches up.
    pub fn may_advance(&mut self, current: &ShipSample, next: &ShipSample, remote: &ShipSample) -> bool {
        if caught_up(current, remote) {
            self.waiting = false;
            return true;
        }
        if self.waiting {
            return false;
        }
        if must_wait(next, remote) {
            self.waiting = true;
            return false;
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn at(t: f64, pos: [f64; 3], vel: [f64; 3]) -> ShipSample {
        ShipSample { t, pos, vel, tau: 0.0 }
    }

    fn rest(t: f64, x: f64) -> ShipSample {
        at(t, [x, 0.0, 0.0], [0.0; 3])
    }

    /// Lorentz boost with velocity `beta` along x.
    fn boost(s: &ShipSample, beta: f64) -> ShipSample {
        let g = 1.0 / (1.0 - beta * beta).sqrt();
        let u = s.four_velocity();
        let (ut, ux) = (g * (u[0] - beta * u[1]), g * (u[1] - beta * u[0]));
        ShipSample {
            t: g * (s.t - beta * s.pos[0]),
            pos: [g * (s.pos[0] - beta * s.t), s.pos[1], s.pos[2]],
            vel: [ux / ut, u[2] / ut, u[3] / ut],
            tau: s.tau,
        }
    }

    /// Strictly inside the open future light cone of `of`.
    fn in_future_cone(e: &ShipSample, of: &ShipSample) -> bool {
        let d = [e.t - of.t, e.pos[0] - of.pos[0], e.pos[1] - of.pos[1], e.pos[2] - of.pos[2]];
        d[0] > 0.0 && d[0] * d[0] > d[1] * d[1] + d[2] * d[2] + d[3] * d[3]
    }

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9 * (1.0 + a.abs().max(b.abs()))
    }

    #[test]
    fn lead_is_lorentz_invariant() {
        let a = at(3.0, [1.0, -2.0, 0.5], [0.3, 0.1, -0.2]);
        let b = at(1.0, [-4.0, 0.0, 2.0], [-0.6, 0.0, 0.4]);
        let l = lead(&a, &b);
        for beta in [-0.9, -0.3, 0.5, 0.95] {
            let m = lead(&boost(&a, beta), &boost(&b, beta));
            assert!(close(l.dt, m.dt) && close(l.dist, m.dist), "{l:?} vs {m:?}");
        }
    }

    #[test]
    fn lead_is_antisymmetric() {
        let a = at(3.0, [1.0, -2.0, 0.5], [0.3, 0.1, -0.2]);
        let b = at(1.0, [-4.0, 0.0, 2.0], [-0.6, 0.0, 0.4]);
        let (ab, ba) = (lead(&a, &b), lead(&b, &a));
        assert_eq!(ab.dt, -ba.dt);
        assert!(close(ab.dist, ba.dist));
    }

    #[test]
    fn margin_in_the_rest_frame() {
        let (a, b) = (rest(0.0, 0.0), rest(0.0, 10.0));
        assert!(!must_wait(&a, &b) && !must_wait(&b, &a));
        let edge = 10.0 - CONE_MARGIN;
        assert!(!must_wait(&rest(edge - 1e-6, 0.0), &b));
        assert!(must_wait(&rest(edge + 1e-6, 0.0), &b));
        assert!(caught_up(&a, &b) && !caught_up(&rest(1e-6, 0.0), &b));
    }

    #[test]
    fn inside_the_remote_cone_must_wait() {
        let remotes = [rest(0.0, 0.0), at(5.0, [1.0, 2.0, 3.0], [0.9, 0.0, -0.3])];
        let vels = [[0.0; 3], [0.99, 0.0, 0.0], [-0.5, 0.5, 0.5]];
        for r in &remotes {
            for v in vels {
                for (dt, dx) in [(1e-3, 0.0), (1.0, 0.99), (100.0, -99.9), (10.0, 0.0)] {
                    let e = at(r.t + dt, [r.pos[0] + dx, r.pos[1], r.pos[2]], v);
                    assert!(in_future_cone(&e, r));
                    assert!(must_wait(&e, r), "{e:?} vs {r:?}");
                }
            }
        }
    }

    #[test]
    fn pacer_waits_until_caught_up() {
        let remote = |t| rest(t, 10.0);
        let mut p = Pacer::default();
        let (cur, next) = (rest(8.4, 0.0), rest(8.6, 0.0));
        assert!(!p.may_advance(&cur, &next, &remote(0.0)));
        assert!(p.is_waiting());
        // Remote advances, local is still ahead (and even outside the margin): keep waiting.
        assert!(!p.may_advance(&cur, &next, &remote(5.0)));
        assert!(!p.may_advance(&cur, &next, &remote(8.3)));
        assert!(p.is_waiting());
        // Caught up: resume.
        assert!(p.may_advance(&cur, &next, &remote(8.4)));
        assert!(!p.is_waiting());
    }

    /// A ship that changes velocity at fixed proper times.
    struct Flight {
        s: ShipSample,
        pacer: Pacer,
        legs: Vec<(f64, f64)>, // (from proper time, x velocity)
        waited: usize,
    }

    impl Flight {
        fn new(x: f64, legs: Vec<(f64, f64)>) -> Self {
            let mut s = rest(0.0, x);
            s.vel[0] = legs[0].1;
            Self { s, pacer: Pacer::default(), legs, waited: 0 }
        }

        fn vel_at(&self, tau: f64) -> f64 {
            self.legs.iter().rev().find(|l| tau >= l.0).unwrap().1
        }

        fn next(&self, dtau: f64) -> ShipSample {
            let s = &self.s;
            let g = s.four_velocity()[0];
            let mut n = *s;
            n.t += g * dtau;
            n.pos[0] += g * s.vel[0] * dtau;
            n.tau += dtau;
            n.vel[0] = self.vel_at(n.tau + 1e-9);
            n
        }
    }

    fn step(a: &mut Flight, b: &mut Flight, dtau: f64) -> (bool, bool) {
        let (an, bn) = (a.next(dtau), b.next(dtau));
        let go_a = a.pacer.may_advance(&a.s, &an, &b.s);
        let go_b = b.pacer.may_advance(&b.s, &bn, &a.s);
        if go_a {
            a.s = an;
        } else {
            a.waited += 1;
        }
        if go_b {
            b.s = bn;
        } else {
            b.waited += 1;
        }
        (go_a, go_b)
    }

    #[test]
    fn twin_paradox_traveller_waits() {
        // Home sits at x = 0. The traveller starts at x = 5, flies out at 0.8c for 10 s of
        // proper time (to x = 18.3), back for 6.25 s (to x = 10), and parks.
        let mut home = Flight::new(0.0, vec![(0.0, 0.0)]);
        let mut trav = Flight::new(5.0, vec![(0.0, 0.8), (10.0, -0.8), (16.25, 0.0)]);
        for _ in 0..2000 {
            let (go_h, go_t) = step(&mut home, &mut trav, 0.05);
            assert!(go_h || go_t, "deadlock");
            assert!(!in_future_cone(&home.s, &trav.s) && !in_future_cone(&trav.s, &home.s));
        }
        assert_eq!(home.waited, 0);
        assert!(trav.waited > 100, "{}", trav.waited);
        assert!(close(trav.s.pos[0], 10.0) && trav.s.vel[0] == 0.0, "trip not finished");
        // Back at rest, the traveller has aged less.
        assert!(home.s.tau - trav.s.tau > 10.0, "{} {}", home.s.tau, trav.s.tau);
    }

    #[test]
    fn colocated_ships_never_deadlock() {
        let mut a = Flight::new(0.0, vec![(0.0, 0.0)]);
        let mut b = Flight::new(0.0, vec![(0.0, 0.3)]);
        for _ in 0..500 {
            let (go_a, go_b) = step(&mut a, &mut b, 0.05);
            assert!(go_a || go_b, "deadlock");
        }
        assert!(a.s.tau > 10.0 && b.s.tau > 10.0);
    }
}
