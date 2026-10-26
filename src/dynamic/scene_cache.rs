//! Incremental scene rebuilds ([`DynWorld::rebuild_scene`]).
//!
//! The world's histories are append-only: between two snapshots only their tips change (the
//! last segment of a track, its cap, a hull's settled motion ahead, pulses in flight). So the
//! snapshot is cut into chunks, a body's or ship's tube per [`CHUNK_T`] of lab time and a
//! laser pulse per piece ([`super::laser::Pulse::pieces`]), each with its own BVH. A chunk is
//! keyed by everything its geometry is made from (its worldtube samples, materials, mesh), so
//! it is rebuilt exactly when its inputs changed, whatever changed them (a new sample, a track
//! pruned at the front, a rebase, a new world). The few simplices that change every time (the
//! table spanning the time window, the caps) are built afresh as one more chunk.
//!
//! The scene's [`Bvh4D`] is the chunks' arena: each chunk's nodes and simplices stay where
//! they were first written, and a snapshot writes only the new chunks (into the room old ones
//! left) and a small tree over all the chunks' roots ([`top_tree`]). It costs the new chunks'
//! BVH builds plus a tree over a few hundred boxes, whatever the length of the history, and
//! the GPU's copy of the arrays is patched with just what was written. Once the arena holds
//! more free room than chunks, the chunks are moved together (a copy, no BVH build).
//!
//! [`DynWorld::rebuild_scene`]: super::DynWorld::rebuild_scene

use std::collections::HashMap;
use std::hash::{DefaultHasher, Hash, Hasher};

use super::ship::{PlayerId, RenderScan, ShipSample};
use super::MAX_SEGMENT_DT;
use std::ops::Range;

use crate::bvh4d::{top_tree, Bvh4D, BvhNode4D};
use crate::mesh4d::{build_worldtube_simplices_with_decided, Aabb4D, Mesh3D, Simplex4D, WorldlineSample};

/// Lab-time span of a tube chunk (by its segments' first samples).
pub(crate) const CHUNK_T: f32 = 1.0;

/// What a chunk is part of.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) enum Part {
    Body(u64),
    Hull(PlayerId),
    Pilot(PlayerId),
    Pulse(PlayerId, u64),
}

struct Chunk {
    /// The bits of everything the chunk's geometry was made from.
    key: Vec<u32>,
    /// Where its nodes (its root first) and simplices are in the arena (empty: nowhere yet).
    nodes: Range<usize>,
    prims: Range<usize>,
    bounds: Aabb4D,
    /// Its BVH, made this snapshot, until it is written to the arena.
    fresh: Option<Bvh4D>,
    /// The snapshot that last used it.
    used: u64,
}

/// What the last snapshot rebuilt and what it reused (for the benchmark and tests).
#[derive(Clone, Copy, Debug, Default)]
pub struct CacheStats {
    pub chunks_built: usize,
    pub chunks_reused: usize,
    pub simplices_built: usize,
    pub simplices_reused: usize,
    /// The arena's simplex slots, in use or free.
    pub arena_simplices: usize,
    /// The chunks were moved together.
    pub compacted: bool,
}

/// Free ranges of an arena's array, first fit.
#[derive(Default)]
struct Space {
    /// Sorted, disjoint and apart (adjacent ones are merged), all before `end`.
    free: Vec<Range<usize>>,
    /// Past the last slot in use.
    end: usize,
}

impl Space {
    fn alloc(&mut self, n: usize) -> usize {
        if let Some(k) = self.free.iter().position(|r| r.len() >= n) {
            let at = self.free[k].start;
            self.free[k].start += n;
            if self.free[k].is_empty() {
                self.free.remove(k);
            }
            return at;
        }
        self.end += n;
        self.end - n
    }

    fn free(&mut self, r: Range<usize>) {
        if r.is_empty() {
            return;
        }
        let k = self.free.partition_point(|f| f.start < r.start);
        self.free.insert(k, r);
        if k + 1 < self.free.len() && self.free[k].end == self.free[k + 1].start {
            self.free[k].end = self.free.remove(k + 1).end;
        }
        if k > 0 && self.free[k - 1].end == self.free[k].start {
            self.free[k - 1].end = self.free.remove(k).end;
        }
        if self.free.last().is_some_and(|f| f.end == self.end) {
            self.end = self.free.pop().unwrap().start;
        }
    }

    fn holes(&self) -> usize {
        self.free.iter().map(Range::len).sum()
    }
}

/// The chunks of the last snapshot (in the arena: the scene's BVH), and where the ships'
/// render scans left off.
#[derive(Default)]
pub struct SceneCache {
    chunks: HashMap<(Part, i64), Chunk>,
    /// This snapshot's chunks, in the order they were made.
    order: Vec<(Part, i64)>,
    /// Per ship track (`true`: the pilot's), the scan of its recorded samples and the last
    /// sample it looked at.
    scans: HashMap<(PlayerId, bool), (RenderScan, ShipSample)>,
    generation: u64,
    key: Vec<u32>,
    /// The layout of the arena the chunks are in.
    layout: u64,
    nodes: Space,
    prims: Space,
    /// Nodes of the last snapshot's tree over the chunks (but its root, node 0), and of its
    /// volatile part (built afresh every time).
    top: Range<usize>,
    volatile: (Range<usize>, Range<usize>),
    pub stats: CacheStats,
}

impl SceneCache {
    /// Start a snapshot into `arena`. Unless it is the one the cache made the last snapshot
    /// into, start afresh.
    pub(crate) fn begin(&mut self, arena: &mut Bvh4D) {
        self.generation += 1;
        self.order.clear();
        self.stats = CacheStats::default();
        if arena.version().layout != self.layout {
            arena.reset_arena();
            self.layout = arena.version().layout;
            self.chunks.clear();
            self.nodes = Space { free: Vec::new(), end: 1 }; // node 0 is the root
            self.prims = Space::default();
            self.top = 0..0;
            self.volatile = (0..0, 0..0);
        }
        arena.begin_edit();
    }

    /// The chunk `id`, made by `make` unless the cached one was made from the same `key`.
    fn chunk(&mut self, id: (Part, i64), key: &[u32], make: impl FnOnce() -> Vec<Simplex4D>) {
        let generation = self.generation;
        let chunk = self.chunks.entry(id).or_insert_with(|| Chunk {
            key: Vec::new(),
            nodes: 0..0,
            prims: 0..0,
            bounds: Aabb4D::empty(),
            fresh: None,
            used: 0,
        });
        if chunk.used != 0 && chunk.key == key {
            self.stats.chunks_reused += 1;
            self.stats.simplices_reused += chunk.prims.len();
        } else {
            let bvh = Bvh4D::build(make());
            self.nodes.free(std::mem::replace(&mut chunk.nodes, 0..0));
            self.prims.free(std::mem::replace(&mut chunk.prims, 0..0));
            chunk.key.clear();
            chunk.key.extend_from_slice(key);
            self.stats.chunks_built += 1;
            self.stats.simplices_built += bvh.simplices.len();
            chunk.fresh = Some(bvh);
        }
        debug_assert_ne!(chunk.used, generation, "chunk {id:?} made twice");
        chunk.used = generation;
        self.order.push(id);
    }

    /// `mesh` swept along `samples` (with per-sample `decided` stamps, if any), as
    /// [`build_worldtube_simplices_with_decided`] would make it, in chunks of [`CHUNK_T`].
    pub(crate) fn tube(
        &mut self,
        part: Part,
        mesh: &Mesh3D,
        samples: &[WorldlineSample],
        decided: Option<&[f32]>,
        material: usize,
    ) {
        if samples.len() < 2 {
            return;
        }
        let fingerprint = mesh_fingerprint(mesh);
        let bucket = |i: usize| (samples[i].center.t / CHUNK_T).floor() as i64;
        let mut first = 0;
        while first < samples.len() - 1 {
            let b = bucket(first);
            let mut end = first + 1; // segments first..end, samples first..=end
            while end < samples.len() - 1 && bucket(end) == b {
                end += 1;
            }
            let (s, d) = (&samples[first..=end], decided.map(|d| &d[first..=end]));
            let mut key = std::mem::take(&mut self.key);
            key.clear();
            key.extend([material as u32, fingerprint as u32, (fingerprint >> 32) as u32, d.is_some() as u32]);
            for sample in s {
                push_sample(&mut key, sample);
            }
            key.extend(d.into_iter().flatten().map(|x| x.to_bits()));
            self.chunk((part, b), &key, || build_worldtube_simplices_with_decided(mesh, s, d, material));
            self.key = key;
            first = end;
        }
    }

    /// Chunk `index` of `part`, keyed by `key` (the bits of what `make` makes it from).
    pub(crate) fn piece(&mut self, part: Part, index: i64, key: &[u32], make: impl FnOnce() -> Vec<Simplex4D>) {
        self.chunk((part, index), key, make);
    }

    /// The samples of `full` to draw ([`super::ship::render_indices`]), where `full` is the
    /// ship's recorded `track` (of player `player`; `pilot`: its pilot's, else its hull's)
    /// followed by anything drawn ahead of it. Resumes the scan of the recorded samples where
    /// the last snapshot left it, so the samples kept stay put as the track grows and is pruned.
    pub(crate) fn render_indices(
        &mut self,
        player: PlayerId,
        pilot: bool,
        track: &[ShipSample],
        full: &[ShipSample],
    ) -> Vec<usize> {
        let mut scan = match self.scans.remove(&(player, pilot)) {
            Some((mut scan, seen)) => {
                scan.realign(&seen, track);
                scan
            }
            None => RenderScan::default(),
        };
        scan.feed(full, track.len().max(1) - 1, MAX_SEGMENT_DT);
        let seen = *scan.last_seen(full);
        self.scans.insert((player, pilot), (scan.clone(), seen));
        scan.finish(full, MAX_SEGMENT_DT)
    }

    /// Write this snapshot's new chunks and `volatile` into `arena`, under a tree over all
    /// its chunks rooted at node 0, and forget the chunks it did not use.
    pub(crate) fn finish(&mut self, volatile: Bvh4D, arena: &mut Bvh4D) {
        let generation = self.generation;
        let (nodes, prims) = (&mut self.nodes, &mut self.prims);
        self.chunks.retain(|_, c| {
            if c.used != generation {
                nodes.free(c.nodes.clone());
                prims.free(c.prims.clone());
            }
            c.used == generation
        });
        self.nodes.free(std::mem::replace(&mut self.top, 0..0));
        let (vn, vp) = std::mem::replace(&mut self.volatile, (0..0, 0..0));
        self.nodes.free(vn);
        self.prims.free(vp);

        let live = |s: &Space| s.end - s.holes();
        if self.prims.holes() > live(&self.prims) / 2 + (1 << 14) || self.nodes.holes() > live(&self.nodes) / 2 + (1 << 14) {
            self.compact(arena);
        }

        let mut place = |nodes: &mut Space, prims: &mut Space, bvh: &Bvh4D| {
            let (n, p) = bvh.lens();
            let (node_at, prim_at) = (nodes.alloc(n), prims.alloc(p));
            arena.write_part(bvh, 0..n, 0..p, node_at, prim_at);
            (node_at..node_at + n, prim_at..prim_at + p)
        };
        let mut items: Vec<(Aabb4D, u32, f32)> = Vec::with_capacity(self.order.len() + 1);
        for id in &self.order {
            let c = self.chunks.get_mut(id).unwrap();
            if let Some(bvh) = c.fresh.take() {
                (c.nodes, c.prims) = place(&mut self.nodes, &mut self.prims, &bvh);
                c.bounds = bvh.bounds();
            }
            if !c.nodes.is_empty() {
                items.push((c.bounds, c.nodes.start as u32, c.prims.len() as f32));
            }
        }
        if volatile.lens().0 > 0 {
            self.volatile = place(&mut self.nodes, &mut self.prims, &volatile);
            items.push((volatile.bounds(), self.volatile.0.start as u32, volatile.simplices.len() as f32));
        }

        match items.len() {
            0 => arena.write_nodes(0, &[BvhNode4D::NOTHING]),
            1 => arena.write_nodes(0, &[arena.node(items[0].1 as usize)]),
            n => {
                let base = self.nodes.alloc(n - 2);
                self.top = base..base + n - 2;
                let tree = top_tree(&mut items, base as u32);
                arena.write_nodes(0, &tree[..1]);
                arena.write_nodes(base, &tree[1..]);
            }
        }
        self.stats.arena_simplices = self.prims.end;
    }

    /// Move the placed chunks together into a fresh layout of `arena`.
    fn compact(&mut self, arena: &mut Bvh4D) {
        let old = std::mem::replace(arena, Bvh4D::empty());
        arena.reset_arena();
        arena.begin_edit();
        self.layout = arena.version().layout;
        self.nodes = Space { free: Vec::new(), end: 1 };
        self.prims = Space::default();
        let mut placed: Vec<&mut Chunk> = self.chunks.values_mut().filter(|c| !c.nodes.is_empty()).collect();
        placed.sort_by_key(|c| c.prims.start);
        for c in placed {
            let (node_at, prim_at) = (self.nodes.alloc(c.nodes.len()), self.prims.alloc(c.prims.len()));
            arena.write_part(&old, c.nodes.clone(), c.prims.clone(), node_at, prim_at);
            c.nodes = node_at..node_at + c.nodes.len();
            c.prims = prim_at..prim_at + c.prims.len();
        }
        self.stats.compacted = true;
    }
}

fn push_sample(key: &mut Vec<u32>, s: &WorldlineSample) {
    let v4 = |v: crate::minkowski::Vec4| [v.t, v.x, v.y, v.z];
    let v3 = |v: crate::minkowski::Vec3| [v.x, v.y, v.z];
    key.extend(v4(s.center).iter().map(|x| x.to_bits()));
    key.push(s.proper_time.to_bits());
    key.extend(v3(s.phi).iter().chain(&v3(s.proper_accel_local)).map(|x| x.to_bits()));
    for e in s.tetrad.e {
        key.extend(v4(e).iter().map(|x| x.to_bits()));
    }
}

/// A hash of everything in `mesh`.
fn mesh_fingerprint(mesh: &Mesh3D) -> u64 {
    let mut h = DefaultHasher::new();
    for v in &mesh.vertices {
        [v.x, v.y, v.z].map(f32::to_bits).hash(&mut h);
    }
    for tri in &mesh.triangles {
        tri.indices.hash(&mut h);
        tri.uv.map(|(u, v)| (u.to_bits(), v.to_bits())).hash(&mut h);
        [tri.normal.x, tri.normal.y, tri.normal.z].map(f32::to_bits).hash(&mut h);
    }
    h.finish()
}
