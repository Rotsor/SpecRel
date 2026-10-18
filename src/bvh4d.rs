use crate::mesh4d::{Aabb4D, HitRecord4D, Simplex4D, SimplexGeom4D};
use crate::minkowski::Vec4;

#[derive(Clone, Copy, Debug)]
pub(crate) struct BvhNode4D {
    pub(crate) aabb: Aabb4D,
    /// For leaf nodes (`count > 0`): start index into `simplices` / `hot_geoms`.
    /// For internal nodes (`count == 0`): index of the right child node (left child is `node_idx + 1`).
    pub(crate) right_or_first: u32,
    pub(crate) count: u32,
}

/// 4D Spacetime Bounding Volume Hierarchy over `Simplex4D` primitives.
pub struct Bvh4D {
    nodes: Vec<BvhNode4D>,
    hot_geoms: Vec<SimplexGeom4D>,
    pub simplices: Vec<Simplex4D>,
}

impl Bvh4D {
    pub fn build(mut simplices: Vec<Simplex4D>) -> Self {
        let n = simplices.len();
        let mut nodes = Vec::with_capacity(n * 2);
        if n > 0 {
            Self::build_recursive(&mut nodes, &mut simplices, 0, n);
        }
        let hot_geoms = simplices.iter().map(SimplexGeom4D::from_simplex).collect();
        Self {
            nodes,
            hot_geoms,
            simplices,
        }
    }

    /// Flat node array (depth-first; left child of internal node `i` is `i + 1`).
    pub(crate) fn nodes(&self) -> &[BvhNode4D] {
        &self.nodes
    }

    /// Compact hot intersection records, indexed identically to `simplices`.
    pub(crate) fn hot_geoms(&self) -> &[SimplexGeom4D] {
        &self.hot_geoms
    }

    fn build_recursive(
        nodes: &mut Vec<BvhNode4D>,
        simplices: &mut [Simplex4D],
        start: usize,
        end: usize,
    ) -> u32 {
        let node_idx = nodes.len() as u32;
        nodes.push(BvhNode4D {
            aabb: Aabb4D::empty(),
            right_or_first: 0,
            count: 0,
        });

        let mut bounds = Aabb4D::empty();
        let mut centroid_bounds = Aabb4D::empty();
        for s in &simplices[start..end] {
            bounds = bounds.union(s.aabb);
            centroid_bounds.grow_point(s.aabb.center());
        }

        let count = end - start;
        if count <= 3 {
            nodes[node_idx as usize] = BvhNode4D {
                aabb: bounds,
                right_or_first: start as u32,
                count: count as u32,
            };
            return node_idx;
        }

        let ext = centroid_bounds.extent();
        let box_measure = |b: Aabb4D| -> f32 {
            let e = b.extent();
            let spatial_sa = e.x * e.y + e.y * e.z + e.z * e.x;
            let temporal_sa = 0.22 * e.t * (e.x + e.y + e.z);
            spatial_sa + temporal_sa
        };

        // 8-bin 4D SAH across all 4 spacetime axes (t, x, y, z)
        const BINS: usize = 8;
        let mut best_axis = 0usize;
        let mut best_split_pos = 0.0f32;
        let mut best_cost = f32::INFINITY;

        for a in 0..4 {
            let axis_ext = ext.axis(a);
            if axis_ext < 1e-4 {
                continue;
            }
            let axis_min = centroid_bounds.min.axis(a);
            let inv_ext = (BINS as f32) / axis_ext;

            let mut bin_bounds = [Aabb4D::empty(); BINS];
            let mut bin_counts = [0usize; BINS];

            for s in &simplices[start..end] {
                let c = s.aabb.center().axis(a);
                let b = (((c - axis_min) * inv_ext) as usize).min(BINS - 1);
                bin_bounds[b] = bin_bounds[b].union(s.aabb);
                bin_counts[b] += 1;
            }

            let mut left_box = [Aabb4D::empty(); BINS - 1];
            let mut left_cnt = [0usize; BINS - 1];
            let mut cur_b = Aabb4D::empty();
            let mut cur_c = 0usize;
            for i in 0..(BINS - 1) {
                cur_b = cur_b.union(bin_bounds[i]);
                cur_c += bin_counts[i];
                left_box[i] = cur_b;
                left_cnt[i] = cur_c;
            }

            let mut right_b = Aabb4D::empty();
            let mut right_c = 0usize;
            for i in (0..(BINS - 1)).rev() {
                right_b = right_b.union(bin_bounds[i + 1]);
                right_c += bin_counts[i + 1];
                if left_cnt[i] > 0 && right_c > 0 {
                    let cost = (left_cnt[i] as f32) * box_measure(left_box[i])
                        + (right_c as f32) * box_measure(right_b);
                    if cost < best_cost {
                        best_cost = cost;
                        best_axis = a;
                        best_split_pos =
                            axis_min + ((i + 1) as f32 / BINS as f32) * axis_ext;
                    }
                }
            }
        }

        if best_cost == f32::INFINITY {
            let mut max_e = -1.0;
            for a in 0..4 {
                if ext.axis(a) > max_e {
                    max_e = ext.axis(a);
                    best_axis = a;
                }
            }
            if max_e < 1e-5 {
                nodes[node_idx as usize] = BvhNode4D {
                    aabb: bounds,
                    right_or_first: start as u32,
                    count: count as u32,
                };
                return node_idx;
            }
            best_split_pos = centroid_bounds.center().axis(best_axis);
        }

        let mut mid = start;
        for i in start..end {
            if simplices[i].aabb.center().axis(best_axis) < best_split_pos {
                simplices.swap(i, mid);
                mid += 1;
            }
        }

        if mid == start || mid == end {
            mid = (start + end) / 2;
            simplices[start..end].select_nth_unstable_by(mid - start, |a, b| {
                let ca = a.aabb.center().axis(best_axis);
                let cb = b.aabb.center().axis(best_axis);
                ca.partial_cmp(&cb).unwrap_or(std::cmp::Ordering::Equal)
            });
        }

        let _left_idx = Self::build_recursive(nodes, simplices, start, mid);
        let right_idx = Self::build_recursive(nodes, simplices, mid, end);

        nodes[node_idx as usize] = BvhNode4D {
            aabb: bounds,
            right_or_first: right_idx,
            count: 0,
        };

        node_idx
    }

    #[inline]
    fn safe_inv_dir(dir: Vec4) -> Vec4 {
        let inv = |v: f32| {
            if v.abs() > 1e-12 {
                1.0 / v
            } else if v >= 0.0 {
                1e12
            } else {
                -1e12
            }
        };
        Vec4::new(inv(dir.t), inv(dir.x), inv(dir.y), inv(dir.z))
    }

    /// Trace a 4D ray `origin + lambda * dir` (`lambda in (t_min, t_max)`) and return the closest hit.
    /// Uses front-to-back child ordering and compact 80B `SimplexGeom4D` hot records.
    pub fn trace(&self, origin: Vec4, dir: Vec4, t_min: f32, mut t_max: f32) -> Option<HitRecord4D> {
        if self.nodes.is_empty() {
            return None;
        }

        let inv_dir = Self::safe_inv_dir(dir);
        let root_enter = self.nodes[0].aabb.intersect_ray(origin, inv_dir, t_min, t_max)?;

        let mut best_hit: Option<(usize, f32, Vec4, [f32; 4])> = None;
        let mut stack = [(0u32, 0.0f32); 64];
        let mut stack_ptr = 1usize;
        stack[0] = (0, root_enter);

        while stack_ptr > 0 {
            stack_ptr -= 1;
            let (node_idx, t_enter) = stack[stack_ptr];
            if t_enter >= t_max {
                continue;
            }

            let node = &self.nodes[node_idx as usize];
            if node.count > 0 {
                let first = node.right_or_first as usize;
                let end = first + node.count as usize;
                for (offset, geom) in self.hot_geoms[first..end].iter().enumerate() {
                    if let Some((lambda, event, tet_bary)) =
                        geom.intersect_fast(origin, dir, t_min, t_max)
                    {
                        t_max = lambda;
                        best_hit = Some((first + offset, lambda, event, tet_bary));
                    }
                }
            } else {
                let left_idx = node_idx + 1;
                let right_idx = node.right_or_first;

                let t_left = self.nodes[left_idx as usize]
                    .aabb
                    .intersect_ray(origin, inv_dir, t_min, t_max);
                let t_right = self.nodes[right_idx as usize]
                    .aabb
                    .intersect_ray(origin, inv_dir, t_min, t_max);

                match (t_left, t_right) {
                    (Some(tl), Some(tr)) => {
                        if stack_ptr + 2 <= stack.len() {
                            if tl <= tr {
                                stack[stack_ptr] = (right_idx, tr);
                                stack[stack_ptr + 1] = (left_idx, tl);
                            } else {
                                stack[stack_ptr] = (left_idx, tl);
                                stack[stack_ptr + 1] = (right_idx, tr);
                            }
                            stack_ptr += 2;
                        }
                    }
                    (Some(tl), None) => {
                        if stack_ptr < stack.len() {
                            stack[stack_ptr] = (left_idx, tl);
                            stack_ptr += 1;
                        }
                    }
                    (None, Some(tr)) => {
                        if stack_ptr < stack.len() {
                            stack[stack_ptr] = (right_idx, tr);
                            stack_ptr += 1;
                        }
                    }
                    (None, None) => {}
                }
            }
        }

        let (idx, lambda, event, tet_bary) = best_hit?;
        Some(self.simplices[idx].interpolate_hit(lambda, event, tet_bary))
    }

    /// Fast shadow / occlusion query along a 4D ray `origin + lambda * dir` for `lambda in (t_min, t_max)`.
    pub fn is_occluded(&self, origin: Vec4, dir: Vec4, t_min: f32, t_max: f32) -> bool {
        if self.nodes.is_empty() {
            return false;
        }

        let inv_dir = Self::safe_inv_dir(dir);
        if self.nodes[0]
            .aabb
            .intersect_ray(origin, inv_dir, t_min, t_max)
            .is_none()
        {
            return false;
        }

        let mut stack = [0u32; 64];
        let mut stack_ptr = 1usize;
        stack[0] = 0;

        while stack_ptr > 0 {
            stack_ptr -= 1;
            let node_idx = stack[stack_ptr] as usize;
            let node = &self.nodes[node_idx];

            if node.count > 0 {
                let first = node.right_or_first as usize;
                let end = first + node.count as usize;
                for geom in &self.hot_geoms[first..end] {
                    if geom.intersect_fast(origin, dir, t_min, t_max).is_some() {
                        return true;
                    }
                }
            } else {
                let left_idx = (node_idx + 1) as u32;
                let right_idx = node.right_or_first;
                let hit_l = self.nodes[left_idx as usize]
                    .aabb
                    .intersect_ray(origin, inv_dir, t_min, t_max)
                    .is_some();
                let hit_r = self.nodes[right_idx as usize]
                    .aabb
                    .intersect_ray(origin, inv_dir, t_min, t_max)
                    .is_some();

                if hit_r && stack_ptr < stack.len() {
                    stack[stack_ptr] = right_idx;
                    stack_ptr += 1;
                }
                if hit_l && stack_ptr < stack.len() {
                    stack[stack_ptr] = left_idx;
                    stack_ptr += 1;
                }
            }
        }

        false
    }
}
