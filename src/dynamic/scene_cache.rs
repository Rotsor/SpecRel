//! Incremental scene rebuilds ([`DynWorld::rebuild_scene`]).
//!
//! The world's histories are append-only: between two snapshots only their tips change (the
//! last segment of a track, its cap, a hull's settled motion ahead, pulses in flight). So the
//! snapshot is cut into chunks, a body's or ship's tube per [`CHUNK_T`] of lab time and a
//! laser pulse each, which keep their simplices and their own BVH across snapshots. A chunk
//! is keyed by everything its geometry is made from (its worldtube samples, materials, mesh),
//! so it is rebuilt exactly when its inputs changed, whatever changed them (a new sample, a
//! track pruned at the front, a rebase, a new world). The few simplices that change every time
//! (the table spanning the time window, the caps) are built afresh, and
//! [`Bvh4D::assemble`] splices all the chunks' trees under a small tree over their bounds.
//! A snapshot then costs the changed chunks' BVH builds plus a copy of the rest, instead of a
//! BVH build over the whole history.
//!
//! [`DynWorld::rebuild_scene`]: super::DynWorld::rebuild_scene

use std::collections::HashMap;
use std::hash::{DefaultHasher, Hash, Hasher};

use super::ship::{PlayerId, RenderScan, ShipSample};
use super::MAX_SEGMENT_DT;
use crate::bvh4d::Bvh4D;
use crate::mesh4d::{build_worldtube_simplices_with_decided, Mesh3D, Simplex4D, WorldlineSample};

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
    bvh: Bvh4D,
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
}

/// The chunks of the last snapshot, and where the ships' render scans left off.
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
    pub stats: CacheStats,
}

impl SceneCache {
    pub(crate) fn begin(&mut self) {
        self.generation += 1;
        self.order.clear();
        self.stats = CacheStats::default();
    }

    /// The chunk `id`, made by `make` unless the cached one was made from the same `key`.
    fn chunk(&mut self, id: (Part, i64), key: &[u32], make: impl FnOnce() -> Vec<Simplex4D>) {
        let generation = self.generation;
        let chunk = self.chunks.entry(id).or_insert_with(|| Chunk { key: Vec::new(), bvh: Bvh4D::empty(), used: 0 });
        if chunk.used != 0 && chunk.key == key {
            self.stats.chunks_reused += 1;
            self.stats.simplices_reused += chunk.bvh.simplices.len();
        } else {
            chunk.bvh = Bvh4D::build(make());
            chunk.key.clear();
            chunk.key.extend_from_slice(key);
            self.stats.chunks_built += 1;
            if std::env::var("CACHE_DEBUG").is_ok() { eprintln!("built {id:?} {}", chunk.bvh.simplices.len()); }
            self.stats.simplices_built += chunk.bvh.simplices.len();
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

    /// A chunk of its own for `part`, keyed by `key` (the bits of what `make` makes it from).
    pub(crate) fn single(&mut self, part: Part, key: &[u32], make: impl FnOnce() -> Vec<Simplex4D>) {
        self.chunk((part, 0), key, make);
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

    /// Splice this snapshot's chunks and `volatile` into `out`, and forget the chunks it did
    /// not use.
    pub(crate) fn finish(&mut self, volatile: &Bvh4D, out: &mut Bvh4D) {
        let mut parts: Vec<&Bvh4D> = self.order.iter().map(|id| &self.chunks[id].bvh).collect();
        parts.push(volatile);
        let t0 = std::time::Instant::now();
        out.assemble(&parts);
        if std::env::var("CACHE_DEBUG").is_ok() { eprintln!("assemble {:?} {} parts", t0.elapsed(), parts.len()); }
        let generation = self.generation;
        self.chunks.retain(|_, c| c.used == generation);
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
