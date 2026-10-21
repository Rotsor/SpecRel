// GPU compute port of the CPU reference renderer (`renderer.rs` + `bvh4d.rs`).
//
// One invocation per pixel: build the camera ray, traverse the 4D BVH with an explicit stack,
// interpolate the hit's cold attributes and shade. Every constant and branch mirrors the Rust
// code so the CPU path can serve as a reference renderer for image-diff tests.
//
// Convention: every `vec4<f32>` holding a 4-vector is laid out as (t, x, y, z), i.e. `.x` is
// time and `.yzw` is the spatial part. Keep this in sync with `gpu.rs`.

const PI: f32 = 3.14159265358979;
const TAU: f32 = 6.28318530717959;
const STACK_SIZE: u32 = 64u;
const GAMMA_LUT_SIZE: u32 = 4096u;

// Enum encodings (see `gpu.rs`).
const MODE_SHADED: u32 = 0u;
const MODE_ISOCHRONES: u32 = 1u;
const MODE_DECISION_AGE: u32 = 3u;
const SHADOW_OFF: u32 = 0u;
const SHADOW_CAMERA_SLICE: u32 = 2u;
const SLICE_PAST_LIGHT_CONE: u32 = 0u;
const SLICE_FRONTIER: u32 = 2u;
const SLICE_PROPHECY: u32 = 3u;
// Mirrors `renderer::FRONTIER_RAY_OFFSET` and `mesh4d::DECIDED_AT_STATIC`.
const FRONTIER_RAY_OFFSET: f32 = 0.14;
const DECIDED_AT_STATIC: f32 = -1.0e9;
const PATTERN_CHECKERBOARD: u32 = 0u;
const PATTERN_CLOCK_FACE: u32 = 1u;

// Glyph indices inside the glyph table.
const GLYPH_PLUS: u32 = 10u;
const GLYPH_MINUS: u32 = 11u;
const GLYPH_DOT: u32 = 12u;
const GLYPH_S: u32 = 13u;

struct Uniforms {
    cam_pos: vec4<f32>,
    e0: vec4<f32>,
    e1: vec4<f32>,
    e2: vec4<f32>,
    e3: vec4<f32>,
    light_pos: vec4<f32>, // (unused, x, y, z)
    half_wh: vec2<f32>,
    size: vec2<u32>,
    mode: u32,
    shadow_mode: u32,
    slice_mode: u32,
    num_nodes: u32,
    // Per-frame dynamics (`renderer::FrameDynamics`). `frontier_origin`/`frontier_tau` fall back
    // to the camera event when no Dynamic Mode frontier is active.
    frontier_origin: vec4<f32>,
    frontier_active: u32,
    num_flashes: u32,
    frontier_tau: f32,
    prophecy_delta: f32,
    flash_at: array<vec4<f32>, 16>,         // emission event (t, x, y, z)
    flash_col_width: array<vec4<f32>, 16>,  // (color * intensity, shell width)
};

struct Node {
    bmin: vec4<f32>,
    bmax: vec4<f32>,
    right_or_first: u32,
    count: u32,
    _pad0: u32,
    _pad1: u32,
};

// 80-byte hot record (`SimplexGeom4D`).
struct Geom {
    v0: vec4<f32>,
    n: vec4<f32>,
    b1: vec4<f32>,
    b2: vec4<f32>,
    b3: vec4<f32>,
};

// Cold per-simplex attributes. Per-vertex quantities are stored "transposed" (one vec4 holds a
// component for all 4 vertices) so interpolation is a single `dot` with the barycentrics.
struct Attr {
    uv_u: vec4<f32>,
    uv_v: vec4<f32>,
    tb_x: vec4<f32>,
    tb_y: vec4<f32>,
    tb_z: vec4<f32>,
    tau: vec4<f32>,
    grad_u: vec4<f32>,
    grad_v: vec4<f32>,
    normal: vec4<f32>,
    four_vel: vec4<f32>,
    misc: vec4<f32>, // (uv_world_scale.0, uv_world_scale.1, bitcast(material_id), 0)
    decided_at: vec4<f32>,
};

struct Material {
    base: vec4<f32>,    // (rgb, emission)
    checker: vec4<f32>, // (rgb, grid_scale)
    params: vec4<f32>,  // (wireframe_width, bitcast(pattern), cavity_length, 0)
};

@group(0) @binding(0) var<storage, read> nodes: array<Node>;
@group(0) @binding(1) var<storage, read> geoms: array<Geom>;
@group(0) @binding(2) var<storage, read> attrs: array<Attr>;
@group(0) @binding(3) var<storage, read> materials: array<Material>;

@group(1) @binding(0) var<uniform> u: Uniforms;
// [0, 4096): gamma LUT (u8 values), then 14 glyphs x 7 rows of 5x7 font bits.
@group(1) @binding(1) var<storage, read> tables: array<u32>;
@group(1) @binding(2) var<storage, read_write> out_pixels: array<u32>;

// ---------------------------------------------------------------------------------------------
// Scalar helpers matching Rust semantics
// ---------------------------------------------------------------------------------------------

// Rust `f32::rem_euclid`.
fn rem_e(x: f32, m: f32) -> f32 {
    let r = x % m;
    return select(r, r + abs(m), r < 0.0);
}

// Rust `f32::round` (half away from zero; WGSL `round` is half-to-even).
fn round_away(x: f32) -> f32 {
    return sign(x) * floor(abs(x) + 0.5);
}

// Rust `Vec3::normalized`.
fn normalized3(v: vec3<f32>) -> vec3<f32> {
    let len = length(v);
    if len > 1e-12 {
        return v * (1.0 / len);
    }
    return vec3<f32>(0.0);
}

fn mdot(a: vec4<f32>, b: vec4<f32>) -> f32 {
    return -a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

fn rest_frame_cos(v: vec4<f32>, n: vec4<f32>, uu: vec4<f32>) -> f32 {
    let vp = v + uu * mdot(v, uu);
    let len_sq = mdot(vp, vp);
    if len_sq > 1e-12 {
        return clamp(mdot(vp, n) / sqrt(len_sq), -1.0, 1.0);
    }
    return 0.0;
}

fn camera_beta() -> vec3<f32> {
    if abs(u.e0.x) > 1e-12 {
        return u.e0.yzw / u.e0.x;
    }
    return vec3<f32>(0.0);
}

fn safe_inv(v: f32) -> f32 {
    if abs(v) > 1e-12 {
        return 1.0 / v;
    }
    return select(-1e12, 1e12, v >= 0.0);
}

// ---------------------------------------------------------------------------------------------
// 4D BVH traversal (port of `Bvh4D::trace` / `Bvh4D::is_occluded`)
// ---------------------------------------------------------------------------------------------

// Slab test; returns the entry parameter, or -1.0 on a miss (`t_min` is always > 0).
fn aabb_enter(ni: u32, o: vec4<f32>, inv: vec4<f32>, t_min: f32, t_max: f32) -> f32 {
    let node = nodes[ni];
    let t0 = (node.bmin - o) * inv;
    let t1 = (node.bmax - o) * inv;
    let lo = min(t0, t1);
    let hi = max(t0, t1);
    let enter = max(max(max(t_min, lo.x), lo.y), max(lo.z, lo.w));
    let exit = min(min(min(t_max, hi.x), hi.y), min(hi.z, hi.w));
    return select(-1.0, enter, exit >= enter);
}

struct GeomHit {
    ok: bool,
    lambda: f32,
    event: vec4<f32>,
    bary: vec4<f32>, // (w0, u, v, w)
};

fn intersect_geom(gi: u32, o: vec4<f32>, d: vec4<f32>, t_min: f32, t_max: f32) -> GeomHit {
    var r: GeomHit;
    r.ok = false;
    let g = geoms[gi];
    let denom = dot(g.n, d);
    if abs(denom) < 1e-9 {
        return r;
    }
    let lambda = dot(g.n, g.v0 - o) / denom;
    if lambda <= t_min || lambda >= t_max {
        return r;
    }
    let event = o + d * lambda;
    let q = event - g.v0;
    let eps = -1e-5;
    let bu = dot(g.b1, q);
    if bu < eps || bu > 1.0 - eps {
        return r;
    }
    let bv = dot(g.b2, q);
    if bv < eps || bu + bv > 1.0 - eps {
        return r;
    }
    let bw = dot(g.b3, q);
    if bw < eps || bu + bv + bw > 1.0 - eps {
        return r;
    }
    r.ok = true;
    r.lambda = lambda;
    r.event = event;
    r.bary = vec4<f32>(1.0 - bu - bv - bw, bu, bv, bw);
    return r;
}

struct TraceHit {
    found: bool,
    idx: u32,
    lambda: f32,
    event: vec4<f32>,
    bary: vec4<f32>,
};

fn trace(o: vec4<f32>, d: vec4<f32>, t_min: f32, t_max_in: f32) -> TraceHit {
    var best: TraceHit;
    best.found = false;
    if u.num_nodes == 0u {
        return best;
    }
    let inv = vec4<f32>(safe_inv(d.x), safe_inv(d.y), safe_inv(d.z), safe_inv(d.w));
    var t_max = t_max_in;
    let root_enter = aabb_enter(0u, o, inv, t_min, t_max);
    if root_enter < 0.0 {
        return best;
    }

    var st_node: array<u32, STACK_SIZE>;
    var st_t: array<f32, STACK_SIZE>;
    st_node[0] = 0u;
    st_t[0] = root_enter;
    var sp = 1u;

    while sp > 0u {
        sp -= 1u;
        let ni = st_node[sp];
        if st_t[sp] >= t_max {
            continue;
        }
        let node = nodes[ni];
        if node.count > 0u {
            let first = node.right_or_first;
            let end = first + node.count;
            for (var k = first; k < end; k++) {
                let h = intersect_geom(k, o, d, t_min, t_max);
                if h.ok {
                    t_max = h.lambda;
                    best.found = true;
                    best.idx = k;
                    best.lambda = h.lambda;
                    best.event = h.event;
                    best.bary = h.bary;
                }
            }
        } else {
            let li = ni + 1u;
            let ri = node.right_or_first;
            let tl = aabb_enter(li, o, inv, t_min, t_max);
            let tr = aabb_enter(ri, o, inv, t_min, t_max);
            let hit_l = tl >= 0.0;
            let hit_r = tr >= 0.0;
            if hit_l && hit_r {
                if sp + 2u <= STACK_SIZE {
                    // Push the farther child first so the nearer one is popped next.
                    if tl <= tr {
                        st_node[sp] = ri;
                        st_t[sp] = tr;
                        st_node[sp + 1u] = li;
                        st_t[sp + 1u] = tl;
                    } else {
                        st_node[sp] = li;
                        st_t[sp] = tl;
                        st_node[sp + 1u] = ri;
                        st_t[sp + 1u] = tr;
                    }
                    sp += 2u;
                }
            } else if hit_l {
                if sp < STACK_SIZE {
                    st_node[sp] = li;
                    st_t[sp] = tl;
                    sp += 1u;
                }
            } else if hit_r {
                if sp < STACK_SIZE {
                    st_node[sp] = ri;
                    st_t[sp] = tr;
                    sp += 1u;
                }
            }
        }
    }
    return best;
}

fn is_occluded(o: vec4<f32>, d: vec4<f32>, t_min: f32, t_max: f32) -> bool {
    if u.num_nodes == 0u {
        return false;
    }
    let inv = vec4<f32>(safe_inv(d.x), safe_inv(d.y), safe_inv(d.z), safe_inv(d.w));
    if aabb_enter(0u, o, inv, t_min, t_max) < 0.0 {
        return false;
    }

    var stack: array<u32, STACK_SIZE>;
    stack[0] = 0u;
    var sp = 1u;

    while sp > 0u {
        sp -= 1u;
        let ni = stack[sp];
        let node = nodes[ni];
        if node.count > 0u {
            let first = node.right_or_first;
            let end = first + node.count;
            for (var k = first; k < end; k++) {
                if intersect_geom(k, o, d, t_min, t_max).ok {
                    return true;
                }
            }
        } else {
            let li = ni + 1u;
            let ri = node.right_or_first;
            let hit_l = aabb_enter(li, o, inv, t_min, t_max) >= 0.0;
            let hit_r = aabb_enter(ri, o, inv, t_min, t_max) >= 0.0;
            if hit_r && sp < STACK_SIZE {
                stack[sp] = ri;
                sp += 1u;
            }
            if hit_l && sp < STACK_SIZE {
                stack[sp] = li;
                sp += 1u;
            }
        }
    }
    return false;
}

// ---------------------------------------------------------------------------------------------
// Hit record (port of `Simplex4D::interpolate_hit`)
// ---------------------------------------------------------------------------------------------

struct HitRecord {
    lambda: f32,
    event: vec4<f32>,
    tet_bary: vec4<f32>,
    tri_bary: vec3<f32>,
    uv: vec2<f32>,
    proper_time: f32,
    tau_uv_grad: vec2<f32>,
    uv_world_scale: vec2<f32>,
    normal: vec4<f32>,
    four_vel: vec4<f32>,
    decided_at: f32,
    material_id: u32,
};

fn make_hit_record(t: TraceHit) -> HitRecord {
    let a = attrs[t.idx];
    let b = t.bary;
    var h: HitRecord;
    h.lambda = t.lambda;
    h.event = t.event;
    h.tet_bary = b;
    h.tri_bary = vec3<f32>(dot(a.tb_x, b), dot(a.tb_y, b), dot(a.tb_z, b));
    h.uv = vec2<f32>(dot(a.uv_u, b), dot(a.uv_v, b));
    h.proper_time = dot(a.tau, b);
    h.tau_uv_grad = vec2<f32>(dot(a.grad_u, b), dot(a.grad_v, b));
    h.uv_world_scale = a.misc.xy;
    h.normal = a.normal;
    h.four_vel = a.four_vel;
    h.decided_at = dot(a.decided_at, b);
    h.material_id = bitcast<u32>(a.misc.z);
    return h;
}

// ---------------------------------------------------------------------------------------------
// Sky
// ---------------------------------------------------------------------------------------------

fn shade_sky_gradient(ray_dir: vec4<f32>) -> vec3<f32> {
    let s = normalized3(ray_dir.yzw);
    let horizon = clamp(s.y * 0.5 + 0.5, 0.0, 1.0);
    let zenith_col = vec3<f32>(0.04, 0.06, 0.13);
    let horizon_col = vec3<f32>(0.12, 0.16, 0.25);
    return horizon_col * (1.0 - horizon) + zenith_col * horizon;
}

fn shade_sky(ray_dir: vec4<f32>) -> vec3<f32> {
    let s = normalized3(ray_dir.yzw);
    var sky = shade_sky_gradient(ray_dir);
    let theta = acos(clamp(s.y, -1.0, 1.0));
    let phi = atan2(s.z, s.x);
    let grid_u = rem_e(phi / PI * 12.0, 1.0);
    let grid_v = rem_e(theta / PI * 6.0, 1.0);
    if grid_u < 0.025 || grid_v < 0.025 {
        sky += vec3<f32>(0.06, 0.09, 0.14);
    }
    return sky;
}

fn hsv_to_rgb(h_deg: f32, s: f32, v: f32) -> vec3<f32> {
    let c = v * s;
    let h = rem_e(h_deg / 60.0, 6.0);
    let x = c * (1.0 - abs(rem_e(h, 2.0) - 1.0));
    let m = v - c;
    var rgb: vec3<f32>;
    if h < 1.0 {
        rgb = vec3<f32>(c, x, 0.0);
    } else if h < 2.0 {
        rgb = vec3<f32>(x, c, 0.0);
    } else if h < 3.0 {
        rgb = vec3<f32>(0.0, c, x);
    } else if h < 4.0 {
        rgb = vec3<f32>(0.0, x, c);
    } else if h < 5.0 {
        rgb = vec3<f32>(x, 0.0, c);
    } else {
        rgb = vec3<f32>(c, 0.0, x);
    }
    return rgb + vec3<f32>(m);
}

// ---------------------------------------------------------------------------------------------
// Procedural rest-frame textures
// ---------------------------------------------------------------------------------------------

fn mat_pattern(mat: Material) -> u32 {
    return bitcast<u32>(mat.params.y);
}

fn clock_tile_central_proper_time(u_tile: f32, v_tile: f32, grid_scale: f32, local_tau: f32, grad: vec2<f32>) -> f32 {
    let inv_grid = 1.0 / max(grid_scale, 1e-4);
    let delta_u = (u_tile - 0.5) * inv_grid;
    let delta_v = (v_tile - 0.5) * inv_grid;
    return local_tau - (delta_u * grad.x + delta_v * grad.y);
}

fn point_on_hand(dx: f32, dy: f32, angle_cw: f32, len: f32, half_width: f32, back: f32) -> bool {
    let hx = sin(angle_cw);
    let hy = cos(angle_cw);
    let proj_along = dx * hx + dy * hy;
    let proj_perp = abs(dx * hy - dy * hx);
    return proj_along >= -back && proj_along <= len && proj_perp <= half_width;
}

fn glyph_row(glyph: u32, row: u32) -> u32 {
    return tables[GAMMA_LUT_SIZE + glyph * 7u + row];
}

fn sample_digital_tau(uu: f32, vv: f32, proper_time: f32) -> bool {
    let rel_u = (uu - 0.28) / (0.72 - 0.28);
    let rel_v = (0.18 - vv) / (0.18 - 0.06);

    let total_cols = 35;
    let col = i32(floor(rel_u * f32(total_cols)));
    let row = i32(floor(rel_v * 7.0));
    if col < 0 || row < 0 || col >= total_cols || row >= 7 {
        return false;
    }
    let char_idx = col / 6;
    let char_col = col % 6;
    if char_idx >= 6 || char_col >= 5 {
        return false;
    }

    let t_tenths = i32(round_away(proper_time * 10.0));
    let abs_t = u32(abs(t_tenths));
    let tenths = abs_t % 10u;
    let ones = (abs_t / 10u) % 10u;
    let tens = (abs_t / 100u) % 10u;

    var glyph: u32;
    switch char_idx {
        case 0: {
            glyph = select(GLYPH_PLUS, GLYPH_MINUS, t_tenths < 0);
        }
        case 1: {
            glyph = tens;
        }
        case 2: {
            glyph = ones;
        }
        case 3: {
            glyph = GLYPH_DOT;
        }
        case 4: {
            glyph = tenths;
        }
        default: {
            glyph = GLYPH_S;
        }
    }
    return ((glyph_row(glyph, u32(row)) >> u32(4 - char_col)) & 1u) != 0u;
}

fn sample_corner_mini_clock(mdx: f32, mdy: f32, mr: f32, mini_tau: f32, mat: Material) -> vec3<f32> {
    if mr > 0.080 {
        return mat.base.rgb * 1.15;
    }
    let angle = rem_e(atan2(mdx, mdy), TAU);
    let fast_angle = rem_e((mini_tau / 1.0) * TAU, TAU);
    let main_angle = rem_e((mini_tau / 10.0) * TAU, TAU);

    var col = vec3<f32>(0.04, 0.06, 0.10);
    if angle <= fast_angle && mr < 0.066 {
        col += mat.base.rgb * 0.25;
    }

    if mr >= 0.062 && mr <= 0.078 {
        let sector = angle / (TAU / 10.0);
        let dist_to_tick = abs(sector - round_away(sector));
        let tick_idx = i32(round_away(sector)) % 10;
        let is_cardinal = tick_idx == 0 || tick_idx == 5;
        let half_w = select(0.07, 0.13, is_cardinal);
        if dist_to_tick < half_w {
            if tick_idx == 0 {
                return vec3<f32>(1.0, 0.88, 0.15);
            }
            return vec3<f32>(0.80, 0.85, 0.92);
        }
    }

    if point_on_hand(mdx, mdy, fast_angle, 0.068, 0.0065, 0.008) {
        return vec3<f32>(0.25, 0.98, 1.0);
    }
    if point_on_hand(mdx, mdy, main_angle, 0.052, 0.0085, 0.008) {
        return vec3<f32>(1.0, 0.95, 0.30);
    }
    if mr < 0.012 {
        return vec3<f32>(1.0, 1.0, 1.0);
    }
    return col;
}

fn sample_clock_face_texture(hit: HitRecord, mat: Material) -> vec3<f32> {
    let grid_scale = max(mat.checker.w, 1e-4);
    let inv_grid = 1.0 / grid_scale;
    let uu = rem_e(hit.uv.x * grid_scale, 1.0);
    let vv = rem_e(hit.uv.y * grid_scale, 1.0);

    let central_tau = clock_tile_central_proper_time(uu, vv, grid_scale, hit.proper_time, hit.tau_uv_grad);

    // 1. Four corner mini-clocks
    var corners = array<vec2<f32>, 4>(
        vec2<f32>(0.135, 0.865),
        vec2<f32>(0.865, 0.865),
        vec2<f32>(0.135, 0.135),
        vec2<f32>(0.865, 0.135),
    );
    for (var i = 0; i < 4; i++) {
        let mc = corners[i];
        let mdx = uu - mc.x;
        let mdy = vv - mc.y;
        let mr = sqrt(mdx * mdx + mdy * mdy);
        if mr <= 0.095 {
            let corner_osc_tau = central_tau
                + ((mc.x - 0.5) * inv_grid) * hit.tau_uv_grad.x
                + ((mc.y - 0.5) * inv_grid) * hit.tau_uv_grad.y;
            let mdx_phys = (mdx * inv_grid) * hit.uv_world_scale.x;
            let mdy_phys = (mdy * inv_grid) * hit.uv_world_scale.y;
            let mr_phys = sqrt(mdx_phys * mdx_phys + mdy_phys * mdy_phys);
            return sample_corner_mini_clock(mdx, mdy, mr, corner_osc_tau - mr_phys, mat);
        }
    }

    // 2. Digital proper-time panel
    if vv >= 0.035 && vv <= 0.205 && uu >= 0.25 && uu <= 0.75 {
        if !(vv >= 0.050 && vv <= 0.190) || !(uu >= 0.265 && uu <= 0.735) {
            return mat.base.rgb * 0.9;
        }
        if vv >= 0.06 && vv <= 0.18 && uu >= 0.28 && uu <= 0.72 && sample_digital_tau(uu, vv, central_tau) {
            return vec3<f32>(1.0, 0.96, 0.30);
        }
        return vec3<f32>(0.03, 0.04, 0.07);
    }

    // 3. Main analog dial
    let cx = 0.50;
    let cy = 0.57;
    let dx = uu - cx;
    let dy = vv - cy;
    let r = sqrt(dx * dx + dy * dy);

    if r > 0.330 {
        let gu = i32(floor(uu * 4.0));
        let gv = i32(floor(vv * 4.0));
        if ((gu + gv) & 1) == 0 {
            return mat.base.rgb * 0.78;
        }
        return mat.checker.rgb * 0.78;
    }
    if r > 0.292 {
        return mat.base.rgb * 1.15;
    }

    let dx_phys = (dx * inv_grid) * hit.uv_world_scale.x;
    let dy_phys = (dy * inv_grid) * hit.uv_world_scale.y;
    let r_phys = sqrt(dx_phys * dx_phys + dy_phys * dy_phys);
    let dial_tau = central_tau - r_phys;

    let angle = rem_e(atan2(dx, dy), TAU);
    let main_hand_angle = rem_e((dial_tau / 10.0) * TAU, TAU);

    var dial_col = vec3<f32>(0.04, 0.06, 0.10);
    if angle <= main_hand_angle && r < 0.236 {
        dial_col += mat.base.rgb * 0.22;
    }

    if r >= 0.220 && r <= 0.286 {
        let sector = angle / (TAU / 10.0);
        let dist_to_tick = abs(sector - round_away(sector));
        let tick_idx = i32(round_away(sector)) % 10;
        let is_cardinal = tick_idx == 0 || tick_idx == 5;
        let half_w = select(0.055, 0.105, is_cardinal);
        let inner_r = select(0.242, 0.218, is_cardinal);
        if dist_to_tick < half_w && r >= inner_r {
            if tick_idx == 0 {
                return vec3<f32>(1.0, 0.88, 0.15);
            } else if is_cardinal {
                return vec3<f32>(1.0, 1.0, 1.0);
            }
            return vec3<f32>(0.75, 0.80, 0.88);
        }
    }

    let fast_angle = rem_e((dial_tau / 1.0) * TAU, TAU);
    if point_on_hand(dx, dy, fast_angle, 0.254, 0.012, 0.025) {
        return vec3<f32>(0.25, 0.98, 1.0);
    }
    if point_on_hand(dx, dy, main_hand_angle, 0.228, 0.022, 0.025) {
        return vec3<f32>(1.0, 0.95, 0.30);
    }
    if point_on_hand(dx, dy, main_hand_angle, 0.236, 0.030, 0.025) {
        return vec3<f32>(0.02, 0.02, 0.04);
    }
    let slow_angle = rem_e((dial_tau / 100.0) * TAU, TAU);
    if point_on_hand(dx, dy, slow_angle, 0.162, 0.026, 0.025) {
        return vec3<f32>(1.0, 0.35, 0.25);
    }
    if r < 0.034 {
        return vec3<f32>(1.0, 1.0, 1.0);
    }
    return dial_col;
}

fn light_clock_bounce_fraction(proper_time: f32, cavity_length: f32) -> f32 {
    let period = max(2.0 * cavity_length, 1e-4);
    let cycle = rem_e(proper_time / period, 1.0);
    if cycle < 0.5 {
        return cycle * 2.0;
    }
    return (1.0 - cycle) * 2.0;
}

fn sample_light_clock_texture(uv: vec2<f32>, proper_time: f32, mat: Material) -> vec3<f32> {
    let grid = mat.checker.w;
    let uu = rem_e(uv.x * grid, 1.0);
    let vv = rem_e(uv.y * grid, 1.0);

    if !(uu >= 0.12 && uu <= 0.88) {
        return mat.base.rgb * 0.65;
    }
    if (vv >= 0.04 && vv <= 0.10) || (vv >= 0.90 && vv <= 0.96) {
        return vec3<f32>(0.75, 0.90, 0.98);
    }
    if !(vv >= 0.10 && vv <= 0.90) {
        return vec3<f32>(0.08, 0.12, 0.16);
    }

    let bounce = light_clock_bounce_fraction(proper_time, mat.params.z);
    let photon_v = 0.10 + 0.80 * bounce;
    let dv = abs(vv - photon_v);
    let du = abs(uu - 0.5);
    let dist = sqrt(du * du * 0.7 + dv * dv);

    var col = vec3<f32>(0.05, 0.09, 0.13);
    let ruler_tick = rem_e((vv - 0.10) / 0.80 * 8.0, 1.0);
    if (uu < 0.22 || uu > 0.78) && ruler_tick < 0.08 {
        col += vec3<f32>(0.20, 0.35, 0.35);
    }
    if du < 0.015 {
        col += vec3<f32>(0.06, 0.16, 0.16);
    }
    if dv < 0.032 {
        let k = 1.0 - dv / 0.032;
        col += mat.base.rgb * (0.9 * k * k);
    }
    if dist < 0.11 {
        let k = 1.0 - dist / 0.11;
        col += vec3<f32>(1.0, 1.0, 0.85) * (1.4 * k * k);
    }
    return col;
}

fn sample_surface_albedo(hit: HitRecord, mat: Material) -> vec3<f32> {
    let pattern = mat_pattern(mat);
    let grid = mat.checker.w;
    var albedo: vec3<f32>;
    if pattern == PATTERN_CHECKERBOARD {
        let gu = i32(floor(hit.uv.x * grid));
        let gv = i32(floor(hit.uv.y * grid));
        var col = select(mat.checker.rgb, mat.base.rgb, ((gu + gv) & 1) == 0);
        let ww = mat.params.x;
        let min_tri = min(min(hit.tri_bary.x, hit.tri_bary.y), hit.tri_bary.z);
        if min_tri < ww {
            let edge_t = clamp(min_tri / ww, 0.0, 1.0);
            col = vec3<f32>(0.07) * (1.0 - edge_t) + col * edge_t;
        }
        albedo = col;
    } else if pattern == PATTERN_CLOCK_FACE {
        albedo = sample_clock_face_texture(hit, mat);
    } else {
        albedo = sample_light_clock_texture(hit.uv, hit.proper_time, mat);
    }

    if pattern != PATTERN_CHECKERBOARD {
        let u_tile = rem_e(hit.uv.x * grid, 1.0);
        let v_tile = rem_e(hit.uv.y * grid, 1.0);
        let border = min(min(u_tile, 1.0 - u_tile), min(v_tile, 1.0 - v_tile));
        if border < 0.025 {
            albedo = albedo * 0.35;
        }
    }
    return albedo;
}

// ---------------------------------------------------------------------------------------------
// Lighting
// ---------------------------------------------------------------------------------------------

fn compute_illumination(ray_dir: vec4<f32>, hit: HitRecord, mat: Material) -> f32 {
    let light = u.light_pos.yzw;
    let spatial_to_light = light - hit.event.yzw;
    let dist_to_light = max(length(spatial_to_light), 1e-4);
    var t_emit: f32;
    if u.shadow_mode == SHADOW_CAMERA_SLICE {
        t_emit = hit.event.x + dot(camera_beta(), spatial_to_light);
    } else {
        t_emit = hit.event.x - dist_to_light;
    }
    let emit_event = vec4<f32>(t_emit, light);
    let to_light_4d = (emit_event - hit.event) * (1.0 / dist_to_light);

    let cos_light_rest = max(rest_frame_cos(to_light_4d, hit.normal, hit.four_vel), 0.0);
    let cos_view_rest = max(-rest_frame_cos(ray_dir, hit.normal, hit.four_vel), 0.0);

    if mat_pattern(mat) == PATTERN_CHECKERBOARD {
        var in_shadow = false;
        if cos_light_rest > 0.0 && u.shadow_mode != SHADOW_OFF {
            in_shadow = is_occluded(hit.event, to_light_4d, 2e-3, dist_to_light - 2e-3);
        }
        let shadow_factor = select(1.0, 0.32, in_shadow);
        let atten = 1.0 / (1.0 + 0.006 * dist_to_light * dist_to_light);
        return mat.base.w + 0.30 * cos_view_rest + 0.80 * shadow_factor * cos_light_rest * atten;
    }
    return 0.72 + 0.38 * cos_view_rest + 0.22 * cos_light_rest;
}

// Mirrors `renderer::compute_flash_illumination`.
fn compute_flash_illumination(hit: HitRecord) -> vec3<f32> {
    var total = vec3<f32>(0.0);
    for (var i = 0u; i < u.num_flashes; i++) {
        let f_at = u.flash_at[i];
        let f_cw = u.flash_col_width[i];
        let dt = hit.event.x - f_at.x;
        if dt <= 0.0 {
            continue;
        }
        let dist = max(length(f_at.yzw - hit.event.yzw), 1e-4);
        let shell_err = abs(dt - dist);
        let half_w = max(0.5 * f_cw.w, 1e-3);
        if shell_err >= half_w {
            continue;
        }
        let k = 1.0 - shell_err / half_w;
        let pulse = k * k;
        let to_flash_4d = (f_at - hit.event) * (1.0 / dist);
        let cos_f = max(rest_frame_cos(to_flash_4d, hit.normal, hit.four_vel), 0.0);
        var in_shadow = false;
        if cos_f > 0.0 && u.shadow_mode != SHADOW_OFF {
            in_shadow = is_occluded(hit.event, to_flash_4d, 2e-3, dist - 2e-3);
        }
        let shadow_factor = select(1.0, 0.25, in_shadow);
        let atten = 1.0 / (1.0 + 0.015 * dist * dist);
        total += f_cw.xyz * (pulse * atten * (0.25 + 0.75 * shadow_factor * cos_f));
    }
    return total;
}

const HAZE_COLOR: vec3<f32> = vec3<f32>(0.32, 0.18, 0.62);

// Mirrors `renderer::analytic_decision_age`.
fn analytic_decision_age(event: vec4<f32>, origin: vec4<f32>, uu: vec4<f32>) -> f32 {
    let d = event - origin;
    let time_ago = mdot(d, uu);
    let dist = sqrt(max(time_ago * time_ago + mdot(d, d), 0.0));
    return time_ago + dist;
}

// Mirrors `renderer::decision_age_color`.
fn decision_age_color(age: f32) -> vec3<f32> {
    if age < 0.0 {
        return vec3<f32>(0.14, 0.14, 0.18);
    }
    var col = hsv_to_rgb(min(age * 4.0, 240.0), 0.82, 0.88);
    if age < 0.45 {
        let f = 1.0 - age / 0.45;
        col = col * (1.0 - f * 0.7) + vec3<f32>(1.0, 0.96, 0.65) * (f * 0.9);
    }
    if abs(rem_e(age * 0.5, 1.0) - 0.5) > 0.465 {
        col = vec3<f32>(0.95, 0.98, 1.0);
    }
    return col;
}

// Mirrors `renderer::apply_undecided_haze`.
fn apply_undecided_haze(base: vec3<f32>, event: vec4<f32>) -> vec3<f32> {
    let shell = 0.18;
    let d = event - u.frontier_origin;
    let inside = d.x - length(d.yzw);
    if inside <= -shell {
        return base;
    }
    var col = base;
    if abs(inside) < shell {
        let k = 1.0 - abs(inside) / shell;
        col += vec3<f32>(0.45, 0.92, 1.0) * (0.85 * k * k);
    }
    if inside > 0.0 {
        let ripple = 0.5 + 0.5 * sin(event.y * 3.5 + event.w * 3.5 - event.x * 4.0) * cos(inside * 5.0);
        let haze = HAZE_COLOR * (1.0 - 0.35 * ripple) + vec3<f32>(0.18, 0.72, 0.92) * (0.35 * ripple);
        let alpha = 0.55 + 0.25 * min(inside * 0.5, 1.0);
        col = col * (1.0 - alpha) + haze * alpha;
    }
    return col;
}

fn apply_distance_fog(surface: vec3<f32>, ray_dir: vec4<f32>, lambda: f32) -> vec3<f32> {
    let fog = exp(-0.008 * lambda);
    return surface * fog + shade_sky_gradient(ray_dir) * (1.0 - fog);
}

fn shade_hit(ray_dir: vec4<f32>, hit: HitRecord) -> vec3<f32> {
    let mat = materials[hit.material_id];

    if u.mode == MODE_SHADED {
        let albedo = sample_surface_albedo(hit, mat);
        let lighting = compute_illumination(ray_dir, hit, mat);
        var lit = albedo * lighting;
        if u.num_flashes > 0u {
            lit += (albedo * 0.65 + vec3<f32>(0.35)) * compute_flash_illumination(hit);
        }
        return apply_distance_fog(lit, ray_dir, hit.lambda);
    }

    if u.mode == MODE_ISOCHRONES {
        let phase = rem_e(hit.lambda * 0.12, 1.0);
        let base = hsv_to_rgb(phase * 300.0, 0.8, 0.85);
        let is_t_ring = abs(rem_e(hit.event.x, 1.0) - 0.5) > 0.45;
        let is_tau_ring = abs(rem_e(hit.proper_time, 1.0) - 0.5) > 0.46;
        if is_t_ring {
            return vec3<f32>(1.0);
        } else if is_tau_ring {
            return vec3<f32>(0.1);
        }
        return base;
    }

    if u.mode == MODE_DECISION_AGE {
        var age: f32;
        if hit.decided_at > DECIDED_AT_STATIC * 0.5 {
            age = u.frontier_tau - hit.decided_at;
        } else {
            age = analytic_decision_age(hit.event, u.frontier_origin, u.e0);
        }
        let cos_view = max(-rest_frame_cos(ray_dir, hit.normal, hit.four_vel), 0.0);
        return apply_distance_fog(decision_age_color(age) * (0.55 + 0.45 * cos_view), ray_dir, hit.lambda);
    }

    // Simplex4DInspector
    let b = hit.tet_bary;
    let min_b = min(min(b.x, b.y), min(b.z, b.w));
    let edge = select(1.0, 0.12, min_b < 0.035);
    let speed_boost = clamp((hit.four_vel.x - 1.0) * 0.8, 0.0, 1.0);
    let tint = vec3<f32>(
        0.25 + 0.65 * b.y + 0.3 * speed_boost,
        0.25 + 0.65 * b.z,
        0.35 + 0.55 * b.w * (1.0 - 0.5 * speed_boost),
    );
    return tint * edge;
}

// Port of `pack_rgb_with_lut`, using the exact same LUT uploaded from the CPU.
fn pack_rgb(c: vec3<f32>) -> u32 {
    let mapped = clamp(c / (vec3<f32>(1.0) + c * 0.35), vec3<f32>(0.0), vec3<f32>(1.0));
    let idx = vec3<u32>(mapped * f32(GAMMA_LUT_SIZE - 1u));
    let r = tables[idx.x];
    let g = tables[idx.y];
    let b = tables[idx.z];
    return (r << 16u) | (g << 8u) | b;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= u.size.x || gid.y >= u.size.y {
        return;
    }
    let w = f32(u.size.x);
    let h = f32(u.size.y);
    let ndc_y = 1.0 - 2.0 * ((f32(gid.y) + 0.5) / h);
    let ndc_x = 2.0 * ((f32(gid.x) + 0.5) / w) - 1.0;
    let sx = ndc_x * u.half_wh.x;
    let sy = ndc_y * u.half_wh.y;

    let local_dir = normalized3(vec3<f32>(sx, sy, 1.0));

    var ray_origin = u.cam_pos;
    var ray_dir: vec4<f32>;
    if u.slice_mode == SLICE_PAST_LIGHT_CONE {
        // `Tetrad::past_light_ray_direction` re-normalizes its input.
        let n = normalized3(local_dir);
        ray_dir = -u.e0 + u.e1 * n.x + u.e2 * n.y + u.e3 * n.z;
    } else if u.slice_mode == SLICE_FRONTIER {
        ray_origin = u.cam_pos - u.e0 * FRONTIER_RAY_OFFSET;
        let n = normalized3(local_dir);
        ray_dir = u.e0 + u.e1 * n.x + u.e2 * n.y + u.e3 * n.z;
    } else {
        if u.slice_mode == SLICE_PROPHECY {
            ray_origin = u.cam_pos + u.e0 * u.prophecy_delta;
        }
        ray_dir = u.e1 * local_dir.x + u.e2 * local_dir.y + u.e3 * local_dir.z;
    }

    let t = trace(ray_origin, ray_dir, 1e-3, 180.0);
    var color: vec3<f32>;
    if t.found {
        let hit = make_hit_record(t);
        color = shade_hit(ray_dir, hit);
        if u.slice_mode == SLICE_PROPHECY {
            color = apply_undecided_haze(color, hit.event);
        }
    } else {
        color = shade_sky(ray_dir);
    }
    out_pixels[gid.y * u.size.x + gid.x] = pack_rgb(color);
}
