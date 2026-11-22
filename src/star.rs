//! Light-emitting points ("stars"): size-less isotropic emitters on piecewise-inertial
//! worldlines.
//!
//! A point has no area, so a ray almost never hits one. Instead every ray carries a
//! **footprint** ([`RayFootprint`]): the width `W(lambda) = width + spread * lambda` of the
//! beam of directions its pixel stands for (a ray cone). A ray segment "sees" a star when the
//! star's event on the segment's light cone passes within a few `W` of the segment, and it
//! picks up the star's light weighted by a normalized Gaussian kernel of the star's transverse
//! offset `d` from the ray:
//!
//! ```text
//!   L_pixel = I_obs * exp(-d^2 / (2 s^2)) / (2 pi s^2),      s = KERNEL_SIGMA * W(lambda)
//! ```
//!
//! This is the radiance of a point source of intensity `I_obs` (power per steradian) smeared
//! over the beam's cross-section `~W^2`. Because neighbouring pixels' beams tile the plane at
//! the star, the kernels of all pixels sum to one: the star's flux `I_obs / lambda^2` lands in
//! the image exactly once wherever it falls between pixel centres, and it moves smoothly at
//! sub-pixel speeds. The image of a star is then a ~1 px point, and the screen-space bloom
//! (`crate::bloom`) gives it its apparent size, growing with brightness as a real star's does.
//!
//! The test is per ray **segment**, against the flat geometry between the segment's ends, like
//! any other ray-primitive intersection. Nothing projects the stars onto the screen directly,
//! so a future mirror or lens only has to update the footprint where it bends the ray (a
//! curved mirror of radius `R` adds `-+2 W / R` to `spread`; see Amanatides' ray cones), and
//! the star seen in it comes out right, dimmed or brightened by the beam's spreading.
//!
//! A star's worldline is a chain of straight segments ([`StarSegment`]); an inertial star is
//! one unbounded segment, a ship's pilot a polyline through its recorded events that ends at
//! the last one known. Along a timelike worldline, whether an event lies before a ray's cone
//! (past cone, future cone or simultaneity slice) changes exactly once, so a ray finds the one
//! segment its cone crosses by binary search, and a worldline that ends before the cone (or
//! starts after it) is simply not seen.
//!
//! Relativity: on the past light cone the star's light is Doppler shifted by
//! `D = (k . u_cam) / (k . u_star)`; the observed spectrum of a blackbody of temperature `T` is
//! a blackbody of temperature `D T` ([`blackbody_rgb`]), so approaching stars turn blue and
//! bright (`~D^4` bolometric) and receding ones red and dim. Aberration needs nothing extra:
//! the star simply sits where its light-cone event is.

use crate::minkowski::{Vec3, Vec4};

/// Width of the star kernel in units of the pixel's beam width (`sigma = 0.6 px`): wide enough
/// that the kernels of a square grid of pixels sum to one within ~0.3% (so a moving star
/// doesn't twinkle), narrow enough for a sharp point.
pub const KERNEL_SIGMA: f32 = 0.6;
/// The kernel is cut off at this many `sigma` (it keeps 99.97% of the energy there).
pub const KERNEL_CUTOFF_SIGMAS: f32 = 4.0;
/// White point of [`blackbody_rgb`]: a blackbody at this temperature is neutral white.
pub const WHITE_POINT_K: f32 = 6504.0;

/// Bound of an unbounded segment's proper-time range (kept finite for the shader).
pub const UNBOUNDED: f32 = 1e30;

/// A straight piece of a star's worldline: `X(s) = event + s * four_velocity` for
/// `s_min <= s <= s_max` (proper time from `event`).
#[derive(Clone, Copy, Debug)]
pub struct StarSegment {
    pub event: Vec4,
    /// Unit, future-pointing.
    pub four_velocity: Vec4,
    pub s_min: f32,
    pub s_max: f32,
}

/// A size-less light source. Its segments are in time order, each starting where the last
/// ended.
#[derive(Clone, Debug)]
pub struct PointStar {
    pub segments: Vec<StarSegment>,
    /// Rest-frame blackbody temperature in kelvin; it sets the star's color.
    pub temperature: f32,
    /// Rest-frame intensity (power per steradian, isotropic): at rest, at distance `r`, the
    /// star delivers flux `intensity / r^2`, and its color has this luminance.
    pub intensity: f32,
}

impl PointStar {
    /// A star at rest at spatial position `pos`.
    pub fn at_rest(pos: Vec3, temperature: f32, intensity: f32) -> Self {
        Self::moving(pos, Vec3::default(), temperature, intensity)
    }

    /// A star passing through `pos` at `t = 0` with 3-velocity `beta`, forever.
    pub fn moving(pos: Vec3, beta: Vec3, temperature: f32, intensity: f32) -> Self {
        let segment = StarSegment {
            event: Vec4::from_time_space(0.0, pos),
            four_velocity: Vec4::four_velocity_from_beta(beta),
            s_min: -UNBOUNDED,
            s_max: UNBOUNDED,
        };
        Self { segments: vec![segment], temperature, intensity }
    }

    /// A star on the polyline through `events` (in time order, timelike-separated), from the
    /// first to the last: it doesn't exist before or after them.
    pub fn along(events: &[Vec4], temperature: f32, intensity: f32) -> Self {
        let segments = events
            .windows(2)
            .filter_map(|w| {
                let d = w[1] - w[0];
                let s = (-d.minkowski_norm_sq()).max(0.0).sqrt();
                (s > 1e-6 && d.t > 0.0).then(|| StarSegment {
                    event: w[0],
                    four_velocity: d * (1.0 / s),
                    s_min: 0.0,
                    s_max: s,
                })
            })
            .collect();
        Self { segments, temperature, intensity }
    }

    /// Scale from `blackbody_rgb(D * temperature)` to observed radiance-weighted color:
    /// `intensity / luminance(blackbody_rgb(temperature))`, so at `D = 1` the color has
    /// luminance `intensity`.
    pub fn color_norm(&self) -> f32 {
        let lum = luminance(blackbody_rgb(self.temperature));
        if lum > 0.0 {
            self.intensity / lum
        } else {
            0.0
        }
    }
}

/// The beam a ray stands for: its width at `lambda` is `width + spread * lambda`, in the units
/// of the frame the ray's direction is normalized in (the camera's rest frame for camera rays).
#[derive(Clone, Copy, Debug)]
pub struct RayFootprint {
    pub width: f32,
    pub spread: f32,
}

impl RayFootprint {
    /// The footprint of a camera pixel: a cone from the camera's eye whose spread is the
    /// pixel's angular pitch, `sqrt(solid angle)`. `pixel_pitch` is the pixel's size on the
    /// image plane at unit distance, `(sx, sy)` the pixel's point on that plane.
    pub fn camera_pixel(pixel_pitch: f32, sx: f32, sy: f32) -> Self {
        // Solid angle of a pixel at angle theta off-axis: pitch^2 cos^3(theta).
        let inv_len = 1.0 / (sx * sx + sy * sy + 1.0).sqrt();
        Self {
            width: 0.0,
            spread: pixel_pitch * inv_len * inv_len.sqrt(),
        }
    }

    #[inline]
    pub fn width_at(&self, lambda: f32) -> f32 {
        self.width + self.spread * lambda
    }
}

/// Luminance of a linear Rec. 709 color.
#[inline]
pub fn luminance(c: Vec3) -> f32 {
    0.2126 * c.x + 0.7152 * c.y + 0.0722 * c.z
}

/// Planck's law at wavelength `lambda_um` (micrometres), up to a constant factor.
#[inline]
fn planck(lambda_um: f32, temperature: f32) -> f32 {
    // c2 = h c / k_B in um K. Clamped so the shader never takes exp of +inf.
    let x = (14388.0 / (lambda_um * temperature.max(1.0))).min(80.0);
    1.0 / (lambda_um.powi(5) * (x.exp() - 1.0))
}

/// Color of a blackbody at `temperature`: Planck's law sampled at the red, green and blue
/// primaries' wavelengths, each relative to a [`WHITE_POINT_K`] blackbody. It scales as the
/// spectrum does (roughly `T` in the Rayleigh-Jeans tail, exponentially in the Wien tail).
pub fn blackbody_rgb(temperature: f32) -> Vec3 {
    const R: f32 = 0.610;
    const G: f32 = 0.550;
    const B: f32 = 0.465;
    Vec3::new(
        planck(R, temperature) / planck(R, WHITE_POINT_K),
        planck(G, temperature) / planck(G, WHITE_POINT_K),
        planck(B, temperature) / planck(B, WHITE_POINT_K),
    )
}

/// Where along `seg`'s line it crosses the cone of rays `k = cone * e0 + n` (`|n| = 1` in the
/// frame `e0`) from `origin`: `cone = -1` is the past light cone, `+1` the future one, `0` the
/// simultaneity hyperplane. Returns the proper time `s` from `seg.event` there.
pub fn segment_cone_crossing(seg: &StarSegment, origin: Vec4, e0: Vec4, cone: f32) -> f32 {
    let d0 = seg.event - origin;
    let u = seg.four_velocity;
    if cone == 0.0 {
        // Frame time -<D, e0> = 0.
        let tau0 = -d0.minkowski_dot(e0);
        let gamma = -u.minkowski_dot(e0);
        return -tau0 / gamma;
    }
    // <D0 + s u, D0 + s u> = 0  <=>  s^2 - 2 b s - a = 0; the earlier root is on the past cone,
    // the later on the future one. Each in the form that doesn't cancel.
    let a = d0.minkowski_norm_sq();
    let b = d0.minkowski_dot(u);
    let sq = (b * b + a).max(0.0).sqrt();
    if cone < 0.0 {
        if b > 0.0 {
            -a / (b + sq)
        } else {
            b - sq
        }
    } else if b < 0.0 {
        -a / (b - sq)
    } else {
        b + sq
    }
}

/// How far `event` is past the cone from `origin`: `tau - cone * rho` in the frame `e0`
/// (frame time `tau`, distance `rho`). It increases along any timelike worldline.
#[inline]
pub fn past_the_cone(event: Vec4, origin: Vec4, e0: Vec4, cone: f32) -> f32 {
    let d = event - origin;
    let tau = -d.minkowski_dot(e0);
    if cone == 0.0 {
        return tau;
    }
    let rho = (d.minkowski_norm_sq() + tau * tau).max(0.0).sqrt();
    tau - cone * rho
}

/// The star's displacement from `origin` where its worldline crosses the cone, and its
/// four-velocity there, if it exists there: the last segment that starts before the cone (or
/// the first) holds the crossing.
pub fn worldline_cone_event(star: &PointStar, origin: Vec4, e0: Vec4, cone: f32) -> Option<(Vec4, Vec4)> {
    let segs = &star.segments;
    let before = segs.partition_point(|seg| past_the_cone(seg.event, origin, e0, cone) <= 0.0);
    let seg = segs.get(before.max(1) - 1)?;
    let s = segment_cone_crossing(seg, origin, e0, cone);
    let u = seg.four_velocity;
    (s >= seg.s_min && s <= seg.s_max).then(|| (seg.event - origin + u * s, u))
}

/// The light of `stars` a ray segment picks up: the ray leaves `origin` along
/// `dir = cone * e0 + n` and is blocked beyond `lambda_max`. Shared by both renderers' logic
/// (mirrored in `render.wgsl`'s `star_light`).
pub fn star_light(
    stars: &[PointStar],
    origin: Vec4,
    dir: Vec4,
    e0: Vec4,
    cone: f32,
    footprint: RayFootprint,
    lambda_max: f32,
) -> Vec3 {
    let n = dir - e0 * cone;
    let mut total = Vec3::default();
    for star in stars {
        let Some((d, u)) = worldline_cone_event(star, origin, e0, cone) else {
            continue;
        };
        let tau = -d.minkowski_dot(e0);
        let r = d + e0 * (-tau); // d = tau e0 + r
        let along = r.minkowski_dot(n);
        if along <= 0.0 || along >= lambda_max {
            continue;
        }
        let perp = r - n * along;
        let d2 = perp.minkowski_norm_sq().max(0.0);
        let sigma = KERNEL_SIGMA * footprint.width_at(along);
        let cut = KERNEL_CUTOFF_SIGMAS * sigma;
        if d2 >= cut * cut {
            continue;
        }
        let inv_2s2 = 1.0 / (2.0 * sigma * sigma);
        let kernel = (-d2 * inv_2s2).exp() * inv_2s2 * std::f32::consts::FRAC_1_PI;
        // Doppler factor of the light the camera receives (past cone only): photon energy
        // seen by e0 over that seen by the star, `<D, e0> / <D, u>` for D on the past cone.
        let doppler = if cone < 0.0 {
            let lambda = (r.minkowski_norm_sq()).max(0.0).sqrt();
            let du = d.minkowski_dot(u);
            if du > 0.0 {
                lambda / du
            } else {
                1.0
            }
        } else {
            1.0
        };
        total += blackbody_rgb(star.temperature * doppler) * (star.color_norm() * kernel);
    }
    total
}

/// A deterministic backdrop of stars around `center`, between `r_min` and `r_max` away:
/// mostly at rest, a few moving fast so their Doppler colors show.
pub fn starfield(center: Vec3, count: usize, r_min: f32, r_max: f32, seed: u32) -> Vec<PointStar> {
    let mut state = seed.wrapping_mul(747_796_405).wrapping_add(2_891_336_453);
    let mut next = move || {
        // PCG-style hash to [0, 1).
        state = state.wrapping_mul(747_796_405).wrapping_add(2_891_336_453);
        let word = ((state >> ((state >> 28) + 4)) ^ state).wrapping_mul(277_803_737);
        ((word >> 22) ^ word) as f32 / u32::MAX as f32
    };
    (0..count)
        .map(|i| {
            // Uniform on the sphere, biased upward so few stars hide under the floor.
            let z = next() * 1.7 - 0.7;
            let phi = next() * std::f32::consts::TAU;
            let rho = (1.0 - z * z).max(0.0).sqrt();
            let dir = Vec3::new(rho * phi.cos(), z, rho * phi.sin());
            let r = r_min + (r_max - r_min) * next();
            // Temperatures from red dwarfs to blue giants; brightness spread over ~2 decades.
            let temperature = 2800.0 + 14000.0 * next() * next();
            let intensity = r * r * 2e-4 * (10.0f32).powf(2.0 * next());
            let beta = if i % 7 == 3 {
                let a = next() * std::f32::consts::TAU;
                Vec3::new(a.cos(), 0.2 * (next() - 0.5), a.sin()) * 0.6
            } else {
                Vec3::default()
            };
            PointStar::moving(center + dir * r, beta, temperature, intensity)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::minkowski::Tetrad;

    fn rest_star(pos: Vec3) -> PointStar {
        PointStar::at_rest(pos, WHITE_POINT_K, 1.0)
    }

    #[test]
    fn white_point_is_white_and_doppler_shifts_color() {
        let w = blackbody_rgb(WHITE_POINT_K);
        assert!((w.x - 1.0).abs() < 1e-5 && (w.y - 1.0).abs() < 1e-5 && (w.z - 1.0).abs() < 1e-5);
        let red = blackbody_rgb(3000.0);
        let blue = blackbody_rgb(15000.0);
        assert!(red.x > red.z && blue.z > blue.x);
    }

    #[test]
    fn past_cone_event_is_null_and_in_the_past() {
        let star = PointStar::moving(Vec3::new(3.0, 1.0, -4.0), Vec3::new(0.5, 0.0, 0.3), 5000.0, 1.0);
        let origin = Vec4::new(7.0, 0.5, 0.0, 0.0);
        let e0 = Vec4::new(1.0, 0.0, 0.0, 0.0);
        let d = worldline_cone_event(&star, origin, e0, -1.0).unwrap().0;
        assert!(d.minkowski_norm_sq().abs() < 1e-3);
        assert!(d.t < 0.0);
        let f = worldline_cone_event(&star, origin, e0, 1.0).unwrap().0;
        assert!(f.minkowski_norm_sq().abs() < 1e-3 && f.t > 0.0);
        let s = worldline_cone_event(&star, origin, e0, 0.0).unwrap().0;
        assert!(s.t.abs() < 1e-4);
    }

    /// Summed over a grid of pixels, a star's radiance times each pixel's solid angle is its
    /// flux `I / r^2`, wherever it falls between pixel centres.
    #[test]
    fn star_flux_is_conserved_across_subpixel_positions() {
        let pitch = 0.01;
        let e0 = Vec4::new(1.0, 0.0, 0.0, 0.0);
        let tetrad = Tetrad::rest();
        let r = 20.0;
        for &(ox, oy) in &[(0.0, 0.0), (0.3, 0.1), (0.5, 0.5), (0.77, 0.21)] {
            let pos = Vec3::new(ox * pitch * r, oy * pitch * r, r);
            let stars = [rest_star(pos)];
            let mut flux = 0.0;
            for j in -6..=6 {
                for i in -6..=6 {
                    let (sx, sy) = (i as f32 * pitch, j as f32 * pitch);
                    let dir = tetrad.past_light_ray_direction(Vec3::new(sx, sy, 1.0));
                    let fp = RayFootprint::camera_pixel(pitch, sx, sy);
                    let l = star_light(&stars, Vec4::default(), dir, e0, -1.0, fp, 1e30);
                    flux += luminance(l) * fp.spread * fp.spread;
                }
            }
            let expected = 1.0 / (r * r);
            assert!((flux / expected - 1.0).abs() < 0.01, "offset ({ox},{oy}): {flux} vs {expected}");
        }
    }

    /// A polyline star is seen on whichever segment the cone crosses, and not at all once the
    /// cone passes its last event (nothing is known of it there yet).
    #[test]
    fn polyline_star_is_found_by_segment_and_ends_at_its_last_event() {
        // At rest at x = 5 until t = 2, then at 0.5c along +y until t = 6.
        let events = [
            Vec4::new(-30.0, 5.0, 0.0, 0.0),
            Vec4::new(2.0, 5.0, 0.0, 0.0),
            Vec4::new(6.0, 5.0, 2.0, 0.0),
        ];
        let star = PointStar::along(&events, 5000.0, 1.0);
        let e0 = Vec4::new(1.0, 0.0, 0.0, 0.0);
        let at = |t: f32| worldline_cone_event(&star, Vec4::new(t, 0.0, 0.0, 0.0), e0, -1.0);
        // Seen at rest (light left at t = 3 - 5 = -2).
        let (d, u) = at(3.0).unwrap();
        assert!((d.x - 5.0).abs() < 1e-4 && d.y.abs() < 1e-4 && (u.t - 1.0).abs() < 1e-5);
        // Seen moving: light from t in (2, 6).
        let (d, u) = at(9.0).unwrap();
        assert!(d.y > 0.0 && d.y < 2.0 && u.y > 0.0 && d.minkowski_norm_sq().abs() < 1e-3);
        // The past cone of t = 20 crosses the worldline after t = 6: unknown, unseen.
        assert!(at(20.0).is_none());
        // Before the first event: unseen too.
        assert!(at(-40.0).is_none());
    }

    #[test]
    fn star_behind_an_occluder_is_hidden() {
        let e0 = Vec4::new(1.0, 0.0, 0.0, 0.0);
        let stars = [rest_star(Vec3::new(0.0, 0.0, 10.0))];
        let dir = Vec4::new(-1.0, 0.0, 0.0, 1.0);
        let fp = RayFootprint { width: 0.0, spread: 0.01 };
        assert!(luminance(star_light(&stars, Vec4::default(), dir, e0, -1.0, fp, 1e30)) > 0.0);
        assert_eq!(luminance(star_light(&stars, Vec4::default(), dir, e0, -1.0, fp, 9.0)), 0.0);
    }

    #[test]
    fn approaching_star_is_blueshifted_and_brighter() {
        let e0 = Vec4::new(1.0, 0.0, 0.0, 0.0);
        let fp = RayFootprint { width: 0.0, spread: 0.01 };
        let dir = Vec4::new(-1.0, 0.0, 0.0, 1.0);
        let at_rest = [PointStar::at_rest(Vec3::new(0.0, 0.0, 10.0), 5000.0, 1.0)];
        // Moving straight at the camera at 0.5c, passing z = 10 at t = 0: the camera sees it
        // where it was at t = -20, twice as far, with Doppler factor D = sqrt(3).
        let approaching =
            [PointStar::moving(Vec3::new(0.0, 0.0, 10.0), Vec3::new(0.0, 0.0, -0.5), 5000.0, 1.0)];
        let a = star_light(&at_rest, Vec4::default(), dir, e0, -1.0, fp, 1e30);
        let b = star_light(&approaching, Vec4::default(), dir, e0, -1.0, fp, 1e30);
        assert!(b.z / b.x > a.z / a.x, "bluer");
        // Four times less flux from twice as far, but ~9.5x more green light from D T = 8660 K.
        assert!(luminance(b) > luminance(a));
    }
}
