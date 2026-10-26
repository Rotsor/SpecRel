use crate::mesh4d::{Aabb4D, HitRecord4D, Simplex4D, SimplexGeom4D};
use crate::minkowski::Vec4;
use crate::par::{self, join};
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
    /// For leaf nodes (`count > 0`): start index into `hot_geoms` (and the leaf order of
    /// `simplices`, see [`Bvh4D::leaf_simplex`]).
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
    /// Leaf order: `order[i]` is the index in `simplices` of the primitive at leaf slot `i`.
    order: Vec<u32>,
    /// The primitives, in the order they were given (moving ~300 B each into leaf order would
    /// cost more than the rest of the build). An arena keeps them in leaf order.
    pub simplices: Vec<Simplex4D>,
    version: BvhVersion,
    /// The node and simplex ranges the latest edit wrote.
    dirty_nodes: Vec<Range<usize>>,
    dirty_prims: Vec<Range<usize>>,
}

/// A primitive as the builder sees it: its box, the box's center, and its index in the input.
/// The builder reorders these (52 B each) instead of the ~300 B simplices, which stay put.
#[derive(Clone, Copy)]
struct BuildRef {
    aabb: Aabb4D,
    centroid: Vec4,
    index: u32,
}

/// Subtrees with at least this many primitives build their two children in parallel (native
/// only: on wasm everything builds on the calling thread, into the same tree).
const PARALLEL_MIN: usize = 4096;

/// Slices of refs at most this long are folded on one thread.
const FOLD_CHUNK: usize = 16384;

/// `merge` of `f` over chunks of `refs`, in parallel (native only) for large nodes near the
/// root, whose passes would otherwise run alone. `merge` here is exact (box unions, counts),
/// so the result does not depend on the chunking.
fn fold<T: Send>(
    refs: &[BuildRef],
    f: &(impl Fn(&[BuildRef]) -> T + Sync),
    merge: &(impl Fn(T, T) -> T + Sync),
) -> T {
    if refs.len() <= FOLD_CHUNK {
        return f(refs);
    }
    let (l, r) = refs.split_at(refs.len() / 2);
    let (a, b) = join(|| fold(l, f, merge), || fold(r, f, merge));
    merge(a, b)
}

/// What [`Bvh4D::split`] chose for a node.
enum Split {
    Leaf,
    /// The node's first `mid` primitives go left.
    At(usize),
}

impl Bvh4D {
    pub fn build(simplices: Vec<Simplex4D>) -> Self {
        let n = simplices.len();
        let mut refs: Vec<BuildRef> = par::map(&simplices, |s| BuildRef { aabb: s.aabb, centroid: s.aabb.center(), index: 0 });
        for (i, r) in refs.iter_mut().enumerate() {
            r.index = i as u32;
        }
        let nodes = if n > 0 { Self::build_subtree(&mut refs, 0) } else { Vec::new() };
        let order: Vec<u32> = refs.iter().map(|r| r.index).collect();
        let hot_geoms = par::map(&order, |&i| SimplexGeom4D::from_simplex(&simplices[i as usize]));
        Self {
            nodes,
            hot_geoms,
            order,
            simplices,
            version: BvhVersion { layout: new_layout(), edit: 0 },
            dirty_nodes: Vec::new(),
            dirty_prims: Vec::new(),
        }
    }

    /// The primitive at leaf slot `i` (indexed like `hot_geoms`).
    pub(crate) fn leaf_simplex(&self, i: usize) -> &Simplex4D {
        &self.simplices[self.order[i] as usize]
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
        self.order.clear();
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
    /// it. The ranges written to must be free, or just past the arrays' ends. The simplices
    /// are written in leaf order.
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
        put(&mut self.order, prim_at, prims.len(), (prim_at as u32..).take(prims.len()));
        put(&mut self.simplices, prim_at, prims.len(), prims.clone().map(|i| src.leaf_simplex(i).clone()));
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
                prims.extend((first..first + n.count as usize).map(|i| self.leaf_simplex(i)));
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

    /// Compact hot intersection records, in leaf order (see [`Bvh4D::leaf_simplex`]).
    pub(crate) fn hot_geoms(&self) -> &[SimplexGeom4D] {
        &self.hot_geoms
    }

    /// The subtree over `refs` (which start at primitive `first`), as its own node array (root
    /// first, child links relative to it). Large subtrees build their children in parallel and
    /// splice them together; the tree is the same as a serial build's.
    fn build_subtree(refs: &mut [BuildRef], first: usize) -> Vec<BvhNode4D> {
        if refs.len() < PARALLEL_MIN {
            let mut nodes = Vec::with_capacity(refs.len());
            Self::build_recursive(&mut nodes, refs, first);
            return nodes;
        }
        let (bounds, split) = Self::split(refs);
        let mid = match split {
            Split::Leaf => return vec![Self::leaf(bounds, first, refs.len())],
            Split::At(mid) => mid,
        };
        let (left_refs, right_refs) = refs.split_at_mut(mid);
        let (left, right) = join(
            || Self::build_subtree(left_refs, first),
            || Self::build_subtree(right_refs, first + mid),
        );
        let right_idx = 1 + left.len() as u32;
        let mut nodes = Vec::with_capacity(1 + left.len() + right.len());
        nodes.push(BvhNode4D { aabb: bounds, right_or_first: right_idx, count: 0, left: 1 });
        nodes.extend(left.into_iter().map(|n| n.moved(1, 0)));
        nodes.extend(right.into_iter().map(|n| n.moved(right_idx, 0)));
        nodes
    }

    fn leaf(bounds: Aabb4D, first: usize, count: usize) -> BvhNode4D {
        BvhNode4D { aabb: bounds, right_or_first: first as u32, count: count as u32, left: 0 }
    }

    /// Serial build of the subtree over `refs` (starting at primitive `first`), appended to
    /// `nodes` depth-first with absolute child links. Returns its root's index.
    fn build_recursive(nodes: &mut Vec<BvhNode4D>, refs: &mut [BuildRef], first: usize) -> u32 {
        let node_idx = nodes.len() as u32;
        let (bounds, split) = Self::split(refs);
        let mid = match split {
            Split::Leaf => {
                nodes.push(Self::leaf(bounds, first, refs.len()));
                return node_idx;
            }
            Split::At(mid) => mid,
        };
        nodes.push(BvhNode4D { aabb: bounds, right_or_first: 0, count: 0, left: node_idx + 1 });
        let (left, right) = refs.split_at_mut(mid);
        Self::build_recursive(nodes, left, first);
        let right_idx = Self::build_recursive(nodes, right, first + mid);
        nodes[node_idx as usize].right_or_first = right_idx;
        node_idx
    }

    /// The node over `refs`: its bounds, and whether to make it a leaf or split it (reordering
    /// `refs` so the left child's come first), by an 8-bin SAH across all 4 spacetime axes.
    fn split(refs: &mut [BuildRef]) -> (Aabb4D, Split) {
        let (bounds, centroid_bounds) = fold(
            refs,
            &|refs| {
                let (mut bounds, mut centroid_bounds) = (Aabb4D::empty(), Aabb4D::empty());
                for r in refs {
                    bounds = bounds.union(r.aabb);
                    centroid_bounds.grow_point(r.centroid);
                }
                (bounds, centroid_bounds)
            },
            &|(a, ca), (b, cb)| (a.union(b), ca.union(cb)),
        );

        let count = refs.len();
        if count <= 3 {
            return (bounds, Split::Leaf);
        }

        let ext = centroid_bounds.extent();

        const BINS: usize = 8;
        let mut best_axis = 0usize;
        let mut best_split_pos = 0.0f32;
        let mut best_cost = f32::INFINITY;

        // Bin along all four axes in one pass over the refs.
        let live: [bool; 4] = std::array::from_fn(|a| ext.axis(a) >= 1e-4);
        let axis_min: [f32; 4] = std::array::from_fn(|a| centroid_bounds.min.axis(a));
        let inv_ext: [f32; 4] = std::array::from_fn(|a| (BINS as f32) / ext.axis(a));
        let (bin_bounds, bin_counts) = fold(
            refs,
            &|refs| {
                let mut bin_bounds = [[Aabb4D::empty(); BINS]; 4];
                let mut bin_counts = [[0u32; BINS]; 4];
                for r in refs {
                    for a in 0..4 {
                        if live[a] {
                            let c = r.centroid.axis(a);
                            let b = (((c - axis_min[a]) * inv_ext[a]) as usize).min(BINS - 1);
                            bin_bounds[a][b] = bin_bounds[a][b].union(r.aabb);
                            bin_counts[a][b] += 1;
                        }
                    }
                }
                (bin_bounds, bin_counts)
            },
            &|(mut bounds, mut counts), (b, c)| {
                for a in 0..4 {
                    for i in 0..BINS {
                        bounds[a][i] = bounds[a][i].union(b[a][i]);
                        counts[a][i] += c[a][i];
                    }
                }
                (bounds, counts)
            },
        );

        for a in (0..4).filter(|&a| live[a]) {
            let axis_ext = ext.axis(a);
            let (bin_bounds, bin_counts) = (&bin_bounds[a], &bin_counts[a]);
            let mut left_box = [Aabb4D::empty(); BINS - 1];
            let mut left_cnt = [0u32; BINS - 1];
            let mut cur_b = Aabb4D::empty();
            let mut cur_c = 0u32;
            for i in 0..(BINS - 1) {
                cur_b = cur_b.union(bin_bounds[i]);
                cur_c += bin_counts[i];
                left_box[i] = cur_b;
                left_cnt[i] = cur_c;
            }

            let mut right_b = Aabb4D::empty();
            let mut right_c = 0u32;
            for i in (0..(BINS - 1)).rev() {
                right_b = right_b.union(bin_bounds[i + 1]);
                right_c += bin_counts[i + 1];
                if left_cnt[i] > 0 && right_c > 0 {
                    let cost = (left_cnt[i] as f32) * box_measure(left_box[i])
                        + (right_c as f32) * box_measure(right_b);
                    if cost < best_cost {
                        best_cost = cost;
                        best_axis = a;
                        best_split_pos = axis_min[a] + ((i + 1) as f32 / BINS as f32) * axis_ext;
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
                return (bounds, Split::Leaf);
            }
            best_split_pos = centroid_bounds.center().axis(best_axis);
        }

        let mut mid = 0;
        for i in 0..count {
            if refs[i].centroid.axis(best_axis) < best_split_pos {
                refs.swap(i, mid);
                mid += 1;
            }
        }

        if mid == 0 || mid == count {
            mid = count / 2;
            refs.select_nth_unstable_by(mid, |a, b| {
                let ca = a.centroid.axis(best_axis);
                let cb = b.centroid.axis(best_axis);
                ca.partial_cmp(&cb).unwrap_or(std::cmp::Ordering::Equal)
            });
        }
        (bounds, Split::At(mid))
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
        Some(self.leaf_simplex(idx).interpolate_hit(lambda, event, tet_bary))
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
