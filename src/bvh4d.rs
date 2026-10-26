use crate::mesh4d::{Aabb4D, HitRecord4D, Simplex4D, SimplexGeom4D};
use crate::minkowski::Vec4;
use std::ops::Range;

/// The SAH's measure of a box: its spatial surface area, plus its extent in time weighted
/// against the spatial ones.
fn box_measure(b: Aabb4D) -> f32 {
    let e = b.extent();
    let spatial_sa = e.x * e.y + e.y * e.z + e.z * e.x;
    let temporal_sa = 0.22 * e.t * (e.x + e.y + e.z);
    spatial_sa + temporal_sa
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct BvhNode4D {
    pub(crate) aabb: Aabb4D,
    /// For leaf nodes (`count > 0`): start index into `simplices` / `hot_geoms`.
    /// For internal nodes (`count == 0`): index of the right child node.
    pub(crate) right_or_first: u32,
    pub(crate) count: u32,
    /// For internal nodes: index of the left child node. [`Bvh4D::build`] puts it right after
    /// its parent (`node_idx + 1`); a tree assembled from cached parts
    /// ([`Bvh4D::write_part`], [`top_tree`]) links them wherever they lie.
    pub(crate) left: u32,
}

impl BvhNode4D {
    /// An internal node that no ray enters (the root of a tree over nothing).
    pub(crate) const NOTHING: Self = Self { aabb: Aabb4D::empty(), right_or_first: 0, count: 0, left: 0 };

    /// The node, its children moved by `nodes` places and its simplices by `prims`.
    fn moved(self, nodes: u32, prims: u32) -> Self {
        if self.count > 0 {
            Self { right_or_first: self.right_or_first.wrapping_add(prims), ..self }
        } else {
            Self {
                left: self.left.wrapping_add(nodes),
                right_or_first: self.right_or_first.wrapping_add(nodes),
                ..self
            }
        }
    }
}

/// Which arrangement of a [`Bvh4D`]'s arrays a copy of them (GPU buffers) was made from, and
/// what has been written over them in place since.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct BvhVersion {
    /// Changes whenever the arrays are laid out afresh: a copy of another layout is stale.
    pub(crate) layout: u64,
    /// Counts the edits ([`Bvh4D::begin_edit`]) made in place since.
    pub(crate) edit: u64,
}

fn new_layout() -> u64 {
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT: AtomicU64 = AtomicU64::new(1);
    NEXT.fetch_add(1, Ordering::Relaxed)
}

/// 4D Spacetime Bounding Volume Hierarchy over `Simplex4D` primitives.
///
/// Besides being built in one go ([`Bvh4D::build`]), it can serve as an arena that a scene
/// cache edits in place: whole subtrees written into free ranges of its arrays
/// ([`Bvh4D::write_part`]) under a small tree over them ([`top_tree`]). The ranges each edit
/// wrote are kept, so that a copy of the arrays (the GPU's) can be patched instead of redone.
pub struct Bvh4D {
    nodes: Vec<BvhNode4D>,
    hot_geoms: Vec<SimplexGeom4D>,
    pub simplices: Vec<Simplex4D>,
    version: BvhVersion,
    /// The node and simplex ranges the latest edit wrote.
    dirty_nodes: Vec<Range<usize>>,
    dirty_prims: Vec<Range<usize>>,
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
            version: BvhVersion { layout: new_layout(), edit: 0 },
            dirty_nodes: Vec::new(),
            dirty_prims: Vec::new(),
        }
    }

    /// A BVH over nothing.
    pub fn empty() -> Self {
        Self::build(Vec::new())
    }

    /// The bounds of everything in it (empty if it holds nothing).
    pub(crate) fn bounds(&self) -> Aabb4D {
        self.nodes.first().map_or(Aabb4D::empty(), |n| n.aabb)
    }

    pub(crate) fn version(&self) -> BvhVersion {
        self.version
    }

    /// The node and simplex ranges the latest edit wrote (all that changed since the version
    /// before it, `edit - 1` of the same layout).
    pub(crate) fn dirty(&self) -> (&[Range<usize>], &[Range<usize>]) {
        (&self.dirty_nodes, &self.dirty_prims)
    }

    /// Empty the arrays for use as an arena: a new layout, holding only the root (node 0), a
    /// tree over nothing.
    pub(crate) fn reset_arena(&mut self) {
        self.nodes.clear();
        self.hot_geoms.clear();
        self.simplices.clear();
        self.nodes.push(BvhNode4D::NOTHING);
        self.version = BvhVersion { layout: new_layout(), edit: 0 };
        self.dirty_nodes.clear();
        self.dirty_prims.clear();
    }

    /// Start the next edit in place.
    pub(crate) fn begin_edit(&mut self) {
        self.version.edit += 1;
        self.dirty_nodes.clear();
        self.dirty_prims.clear();
    }

    /// Write `src`'s nodes `nodes` and simplices `prims` (a whole subtree and the simplices
    /// under it) to nodes from `node_at` and simplices from `prim_at`, moving its links with
    /// it. The ranges written to must be free, or just past the arrays' ends.
    pub(crate) fn write_part(
        &mut self,
        src: &Bvh4D,
        nodes: Range<usize>,
        prims: Range<usize>,
        node_at: usize,
        prim_at: usize,
    ) {
        let dn = (node_at as u32).wrapping_sub(nodes.start as u32);
        let dp = (prim_at as u32).wrapping_sub(prims.start as u32);
        let moved = src.nodes[nodes.clone()].iter().map(|n| n.moved(dn, dp));
        put(&mut self.nodes, node_at, nodes.len(), moved);
        put(&mut self.hot_geoms, prim_at, prims.len(), src.hot_geoms[prims.clone()].iter().copied());
        put(&mut self.simplices, prim_at, prims.len(), src.simplices[prims.clone()].iter().cloned());
        self.dirty_nodes.push(node_at..node_at + nodes.len());
        self.dirty_prims.push(prim_at..prim_at + prims.len());
    }

    /// Write `nodes` (with their links as they are) from node `at`.
    pub(crate) fn write_nodes(&mut self, at: usize, nodes: &[BvhNode4D]) {
        put(&mut self.nodes, at, nodes.len(), nodes.iter().copied());
        self.dirty_nodes.push(at..at + nodes.len());
    }

    pub(crate) fn node(&self, i: usize) -> BvhNode4D {
        self.nodes[i]
    }

    /// The arrays' lengths: nodes and simplices.
    pub(crate) fn lens(&self) -> (usize, usize) {
        (self.nodes.len(), self.simplices.len())
    }

    /// The simplices in the tree (an arena's arrays also hold free room), and the nodes it
    /// has.
    pub fn in_tree(&self) -> (Vec<&Simplex4D>, usize) {
        let (mut prims, mut nodes) = (Vec::new(), 0);
        let mut stack: Vec<usize> = if self.nodes.is_empty() { vec![] } else { vec![0] };
        while let Some(i) = stack.pop() {
            let n = self.nodes[i];
            nodes += 1;
            if n.count > 0 {
                let first = n.right_or_first as usize;
                prims.extend(&self.simplices[first..first + n.count as usize]);
            } else if n.aabb.min.t <= n.aabb.max.t {
                stack.extend([n.left as usize, n.right_or_first as usize]);
            }
        }
        (prims, nodes)
    }

    /// Flat node array, rooted at node 0.
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
        nodes.push(BvhNode4D::NOTHING);

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
                left: 0,
            };
            return node_idx;
        }

        let ext = centroid_bounds.extent();

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
                    left: 0,
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

        let left_idx = Self::build_recursive(nodes, simplices, start, mid);
        let right_idx = Self::build_recursive(nodes, simplices, mid, end);

        nodes[node_idx as usize] = BvhNode4D {
            aabb: bounds,
            right_or_first: right_idx,
            count: 0,
            left: left_idx,
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
                let left_idx = node.left;
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
                let left_idx = node.left;
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

/// Write `n` items from `items` into `v` from index `at` (at most `v.len()`), growing it as
/// needed.
fn put<T>(v: &mut Vec<T>, at: usize, n: usize, items: impl Iterator<Item = T>) {
    debug_assert!(at <= v.len());
    let overwrite = v.len().min(at + n) - at;
    let mut items = items;
    for (slot, item) in v[at..at + overwrite].iter_mut().zip(&mut items) {
        *slot = item;
    }
    v.extend(items);
}

/// A tree over subtrees already in an arena: `items` are their bounds, their roots' node
/// indices and their weights (simplex counts), at least two of them. The tree's nodes are
/// returned in order; the first is the root and goes to node 0, the `k`th to node
/// `base + k - 1`, and they link to each other and to the subtrees accordingly. A full-sweep
/// SAH over the subtrees' centroids (there are a few hundred).
pub(crate) fn top_tree(items: &mut [(Aabb4D, u32, f32)], base: u32) -> Vec<BvhNode4D> {
    debug_assert!(items.len() >= 2);
    let mut out = Vec::with_capacity(items.len());
    top_tree_recursive(items, base, &mut out);
    out
}

fn top_tree_recursive(items: &mut [(Aabb4D, u32, f32)], base: u32, out: &mut Vec<BvhNode4D>) -> u32 {
    if let [(_, root, _)] = *items {
        return root;
    }
    let k = out.len();
    out.push(BvhNode4D::NOTHING);
    let at = if k == 0 { 0 } else { base + k as u32 - 1 };
    let bounds = items.iter().fold(Aabb4D::empty(), |b, item| b.union(item.0));

    let mut best = (f32::INFINITY, 0usize, items.len() / 2);
    let mut right_box = vec![Aabb4D::empty(); items.len()];
    let mut right_w = vec![0.0f32; items.len()];
    let by_axis = |axis: usize| {
        move |a: &(Aabb4D, u32, f32), b: &(Aabb4D, u32, f32)| a.0.center().axis(axis).total_cmp(&b.0.center().axis(axis))
    };
    for axis in 0..4 {
        items.sort_by(by_axis(axis));
        let mut acc = (Aabb4D::empty(), 0.0);
        for j in (1..items.len()).rev() {
            acc = (acc.0.union(items[j].0), acc.1 + items[j].2);
            right_box[j] = acc.0;
            right_w[j] = acc.1;
        }
        let mut left = (Aabb4D::empty(), 0.0);
        for j in 1..items.len() {
            left = (left.0.union(items[j - 1].0), left.1 + items[j - 1].2);
            let cost = left.1 * box_measure(left.0) + right_w[j] * box_measure(right_box[j]);
            if cost < best.0 {
                best = (cost, axis, j);
            }
        }
    }
    let (_, axis, mid) = best;
    items.sort_by(by_axis(axis));
    let (l, r) = items.split_at_mut(mid);
    let left = top_tree_recursive(l, base, out);
    let right = top_tree_recursive(r, base, out);
    out[k] = BvhNode4D { aabb: bounds, right_or_first: right, count: 0, left };
    at
}
