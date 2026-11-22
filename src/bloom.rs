//! Screen-space bloom: the camera's point-spread function, applied to the HDR image.
//!
//! A real lens and eye scatter a small fraction of the light that reaches each pixel into a
//! wide, faint halo. Applied to a star's ~1 px image, that halo is what gives it a visible
//! size, and the size grows with brightness: a dim star's halo stays below what the display
//! shows, a bright one's spreads over many pixels. The filter is linear and conserves energy:
//!
//! ```text
//!   out = (1 - BLOOM_STRENGTH) * hdr + BLOOM_STRENGTH * mean_k(blur_k(hdr))
//! ```
//!
//! with `blur_k` a blur of radius `~2^k` px for `k = 1..=BLOOM_LEVELS`, an equal share of the
//! scattered light per octave of radius (a `~1/r^2` halo, like a real glare PSF).
//!
//! It is computed on a pyramid of half-resolution levels: each level is the one above
//! downsampled with a `[1 3 3 1] / 8` filter per axis; then, from the coarsest level up, each
//! level adds the bilinear 2x upsample of the one below. The GPU kernel (`render.wgsl`'s
//! `bloom_*` entry points) runs the same steps on the same layout, one dispatch per level.

use crate::minkowski::Vec3;

/// Number of pyramid levels below full resolution.
pub const BLOOM_LEVELS: usize = 6;
/// Fraction of each pixel's light scattered into the halo.
pub const BLOOM_STRENGTH: f32 = 0.04;

/// Size of pyramid level `k` (level 0 is the full image).
pub fn level_size(width: usize, height: usize, k: usize) -> (usize, usize) {
    let (mut w, mut h) = (width, height);
    for _ in 0..k {
        w = w.div_ceil(2);
        h = h.div_ceil(2);
    }
    (w, h)
}

/// Offset of level `k` in the packed pyramid, and the pyramid's total size at `k = BLOOM_LEVELS + 1`.
pub fn level_offset(width: usize, height: usize, k: usize) -> usize {
    (0..k).map(|i| {
        let (w, h) = level_size(width, height, i);
        w * h
    })
    .sum()
}

/// Total number of pixels in the packed pyramid (all levels).
pub fn pyramid_len(width: usize, height: usize) -> usize {
    level_offset(width, height, BLOOM_LEVELS + 1)
}

const DOWN_W: [f32; 4] = [0.125, 0.375, 0.375, 0.125];

/// Downsample level `k - 1` into level `k`.
fn downsample(pyr: &mut [Vec3], width: usize, height: usize, k: usize) {
    let (sw, sh) = level_size(width, height, k - 1);
    let (dw, dh) = level_size(width, height, k);
    let (src_off, dst_off) = (level_offset(width, height, k - 1), level_offset(width, height, k));
    let (src, dst) = pyr.split_at_mut(dst_off);
    let src = &src[src_off..];
    for y in 0..dh {
        for x in 0..dw {
            let mut acc = Vec3::default();
            for (j, wy) in DOWN_W.iter().enumerate() {
                let sy = (2 * y as i64 - 1 + j as i64).clamp(0, sh as i64 - 1) as usize;
                let mut row = Vec3::default();
                for (i, wx) in DOWN_W.iter().enumerate() {
                    let sx = (2 * x as i64 - 1 + i as i64).clamp(0, sw as i64 - 1) as usize;
                    row += src[sy * sw + sx] * *wx;
                }
                acc += row * *wy;
            }
            dst[y * dw + x] = acc;
        }
    }
}

/// Bilinear 2x upsample of level `k + 1` (of size `cw x ch`, in `coarse`) at pixel `(x, y)` of
/// level `k`: taps at `m` and `m -+ 1` with weights `3/4` and `1/4`.
#[inline]
pub fn upsample_at(coarse: &[Vec3], cw: usize, ch: usize, x: usize, y: usize) -> Vec3 {
    let taps = |i: usize, n: usize| -> (usize, usize) {
        let m = i / 2;
        let other = if i % 2 == 0 { m as i64 - 1 } else { m as i64 + 1 };
        (m, other.clamp(0, n as i64 - 1) as usize)
    };
    let (x0, x1) = taps(x, cw);
    let (y0, y1) = taps(y, ch);
    let row = |yy: usize| coarse[yy * cw + x0] * 0.75 + coarse[yy * cw + x1] * 0.25;
    row(y0) * 0.75 + row(y1) * 0.25
}

/// Apply the bloom to `hdr` (a `width x height` image) in place.
pub fn apply(hdr: &mut [Vec3], width: usize, height: usize) {
    if width == 0 || height == 0 {
        return;
    }
    let mut pyr = vec![Vec3::default(); pyramid_len(width, height)];
    pyr[..width * height].copy_from_slice(hdr);
    for k in 1..=BLOOM_LEVELS {
        downsample(&mut pyr, width, height, k);
    }
    // U_k = D_k + up(U_{k+1}), coarsest first; in place, as level k only reads level k + 1.
    for k in (1..BLOOM_LEVELS).rev() {
        let (w, h) = level_size(width, height, k);
        let (cw, ch) = level_size(width, height, k + 1);
        let off = level_offset(width, height, k);
        let (fine, coarse) = pyr.split_at_mut(off + w * h);
        let coarse = &coarse[..cw * ch];
        let fine = &mut fine[off..];
        for y in 0..h {
            for x in 0..w {
                fine[y * w + x] += upsample_at(coarse, cw, ch, x, y);
            }
        }
    }
    let (w1, h1) = level_size(width, height, 1);
    let lvl1 = &pyr[level_offset(width, height, 1)..][..w1 * h1];
    let halo = BLOOM_STRENGTH / BLOOM_LEVELS as f32;
    for y in 0..height {
        for x in 0..width {
            let px = &mut hdr[y * width + x];
            *px = *px * (1.0 - BLOOM_STRENGTH) + upsample_at(lvl1, w1, h1, x, y) * halo;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bloom_conserves_energy_and_spreads_a_point() {
        let (w, h) = (97, 64);
        let mut img = vec![Vec3::default(); w * h];
        img[30 * w + 40] = Vec3::splat(1000.0);
        apply(&mut img, w, h);
        let total: f32 = img.iter().map(|c| c.x).sum();
        assert!((total / 1000.0 - 1.0).abs() < 0.01, "total {total}");
        assert!(img[30 * w + 40].x > 900.0);
        // The halo reaches tens of pixels away, fading with distance.
        let (near, far) = (img[30 * w + 50].x, img[30 * w + 70].x);
        assert!(near > far && far > 1e-4, "{near} {far}");
    }

    #[test]
    fn flat_image_is_unchanged() {
        let (w, h) = (33, 20);
        let mut img = vec![Vec3::splat(0.5); w * h];
        apply(&mut img, w, h);
        assert!(img.iter().all(|c| (c.x - 0.5).abs() < 1e-5));
    }
}
