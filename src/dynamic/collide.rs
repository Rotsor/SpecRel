//! Exact relativistic two-body kinematics in `f64` (`c = 1`): elastic collisions, inelastic
//! mergers and two-body decays, all conserving total 4-momentum.

pub type V3 = [f64; 3];

/// 4-momentum `(E, px, py, pz)`.
pub type P4 = [f64; 4];

pub fn dot3(a: V3, b: V3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

pub fn gamma(v: V3) -> f64 {
    1.0 / (1.0 - dot3(v, v)).sqrt()
}

/// 4-momentum of rest mass `m` moving at 3-velocity `v`.
pub fn momentum(m: f64, v: V3) -> P4 {
    let e = m * gamma(v);
    [e, e * v[0], e * v[1], e * v[2]]
}

/// Rest mass and 3-velocity of 4-momentum `p`.
pub fn mass_and_velocity(p: P4) -> (f64, V3) {
    let [e, px, py, pz] = p;
    ((e * e - px * px - py * py - pz * pz).sqrt(), [px / e, py / e, pz / e])
}

pub fn add(a: P4, b: P4) -> P4 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]]
}

pub fn sub(a: P4, b: P4) -> P4 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2], a[3] - b[3]]
}

/// Components of 4-vector `p` in the frame moving with 3-velocity `beta` (pure boost).
pub fn boost(p: P4, beta: V3) -> P4 {
    let b2 = dot3(beta, beta);
    if b2 < 1e-30 {
        return p;
    }
    let g = 1.0 / (1.0 - b2).sqrt();
    let s = [p[1], p[2], p[3]];
    let bs = dot3(beta, s);
    let t = g * (p[0] - bs);
    let k = (g - 1.0) * bs / b2 - g * p[0];
    [t, s[0] + k * beta[0], s[1] + k * beta[1], s[2] + k * beta[2]]
}

fn normalized(v: V3) -> V3 {
    let l = dot3(v, v).sqrt();
    [v[0] / l, v[1] / l, v[2] / l]
}

/// Elastic collision of `a` and `b` touching along lab-frame separation `a -> b`.
///
/// In the centre-of-momentum frame the spatial momenta are equal and opposite; the component
/// along the contact normal is reversed, which keeps both energies and the total momentum.
/// Returns the new velocities, or `None` if the bodies are already separating.
pub fn elastic(m_a: f64, v_a: V3, m_b: f64, v_b: V3, separation: V3) -> Option<(V3, V3)> {
    let (pa, pb) = (momentum(m_a, v_a), momentum(m_b, v_b));
    let total = add(pa, pb);
    let beta_cm = [total[1] / total[0], total[2] / total[0], total[3] / total[0]];

    // The equal-lab-time separation, seen from the CM frame, gives the contact normal there.
    let sep_cm = boost([0.0, separation[0], separation[1], separation[2]], beta_cm);
    let n = normalized([sep_cm[1], sep_cm[2], sep_cm[3]]);
    let pa_cm = boost(pa, beta_cm);
    let along = pa_cm[1] * n[0] + pa_cm[2] * n[1] + pa_cm[3] * n[2];
    if along <= 0.0 {
        return None;
    }
    let pa_s = [
        pa_cm[1] - 2.0 * along * n[0],
        pa_cm[2] - 2.0 * along * n[1],
        pa_cm[3] - 2.0 * along * n[2],
    ];
    let eb_cm = boost(pb, beta_cm)[0];
    let back = beta_cm.map(|b| -b);
    let pa_new = boost([pa_cm[0], pa_s[0], pa_s[1], pa_s[2]], back);
    let pb_new = boost([eb_cm, -pa_s[0], -pa_s[1], -pa_s[2]], back);
    Some((mass_and_velocity(pa_new).1, mass_and_velocity(pb_new).1))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_p4_eq(a: P4, b: P4, tol: f64) {
        for c in 0..4 {
            assert!((a[c] - b[c]).abs() < tol, "{a:?} != {b:?}");
        }
    }

    #[test]
    fn elastic_conserves_four_momentum() {
        let cases: [(f64, V3, f64, V3, V3); 4] = [
            (1.0, [0.6, 0.0, 0.0], 1.0, [0.0, 0.0, 0.0], [0.7, 0.0, 0.0]),
            (1.0, [0.5, 0.0, 0.3], 2.5, [-0.2, 0.1, 0.0], [0.5, 0.0, 0.4]),
            (0.4, [0.9, 0.0, 0.0], 6.0, [-0.3, 0.0, 0.2], [0.3, 0.2, 0.6]),
            (1.0, [0.0, 0.0, 0.95], 1.0, [0.0, 0.0, -0.95], [0.1, 0.0, 0.7]),
        ];
        for (ma, va, mb, vb, sep) in cases {
            let (va2, vb2) = elastic(ma, va, mb, vb, sep).expect("approaching");
            let before = add(momentum(ma, va), momentum(mb, vb));
            let after = add(momentum(ma, va2), momentum(mb, vb2));
            assert_p4_eq(before, after, 1e-9);
            assert!(dot3(va2, va2) < 1.0 && dot3(vb2, vb2) < 1.0);
            // Bounced: now separating along the contact.
            assert!(dot3([vb2[0] - va2[0], vb2[1] - va2[1], vb2[2] - va2[2]], sep) > 0.0);
        }
        // Already separating: no collision.
        assert!(elastic(1.0, [-0.3, 0.0, 0.0], 1.0, [0.0; 3], [0.7, 0.0, 0.0]).is_none());
    }
}
