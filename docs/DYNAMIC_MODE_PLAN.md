# SpecRel 4D — Dynamic ("Causal") Mode: Design Plan

## 0. The idea in one paragraph

Today every scene is a closed, precomputed 4D block: worldtubes are extruded once in
`Scene4D::build`, uploaded once to the GPU, and lab time loops in `[t_min, t_max]`.
In **Dynamic Mode** the only source of non-determinism is the player. Their ship's worldline
up to the current camera event `O` is known; everything after `O` is not. An event `E` is
**decided** iff no future input can reach it, i.e. iff `E ∉ I⁺(O)` (the open future light
cone of `O`). As the camera advances, `I⁺(O(τ))` shrinks monotonically
(`I⁺(O(τ₂)) ⊂ I⁺(O(τ₁))` for `τ₂ > τ₁`), so the decided region only ever grows.
The simulation therefore extends every object's worldtube **exactly up to the camera's
future light cone** — the *causal frontier* `∂J⁺(O)` — and no further. Far-away objects are
known far into the lab-frame future; nearby ones are known only a hair past "now".

Two facts make this pleasant:

* **Everything the existing views show is already decided.** The past light cone of `O`
  (Real Camera) and the spacelike slice through `O` (God's Eye) both lie outside `I⁺(O)`.
  Nothing on screen ever needs to be "guessed".
* **Causal closure is automatic.** If `E` is decided then its whole causal past is decided
  (if `Q ∈ J⁻(E) ∩ I⁺(O)` then `E ∈ I⁺(O)`). So any interaction rule that only reads an
  event's past (contact, retarded fields, received signals) can always be evaluated at the
  frontier. Instantaneous action-at-a-distance cannot — which is exactly the rule SR imposes.

The growth only becomes *visible* through new views (§4), which is where most of the
"wow" lives.

---

## 1. Where the current code stands (what changes)

| Area | Today | Dynamic mode needs |
|---|---|---|
| `Scene4D` | immutable `Bvh4D` built once | append-only geometry + sim state; static scenes become a special case |
| `build_worldtube_simplices` | Born-rigid: vertex = `center + tetrad·p_rest` | also a **deforming** variant where each vertex has its own sampled worldline |
| `Bvh4D` / `gpu.rs` | one BVH per scene, uploaded at start | two-level (chunked) BVH with an append arena on GPU (§3) |
| `main.rs` time | loops `t` in `[t_min, t_max]`; `[ ]`/`IJKL` teleport | no looping; teleports are acausal → disabled or reinterpreted (§6) |
| `V`/`X` velocity jumps | instant boosts | fine — they're impulsive proper acceleration of the ship |
| Lighting | one static point light | + list of **flash events** (light pulses) with retarded illumination |
| Render modes | Shaded / Isochrones / Inspector | + *Decision age*; views + *Frontier* and *Prophecy slice* |

What stays untouched: `minkowski.rs`, `SimplexGeom4D`/`Simplex4D` intersection, all
shading/clock code, the five existing scenes.

---

## 2. Simulation core (`src/dynamic/`)

### 2.1 Types

```rust
pub struct DynWorld {
    pub bodies: Vec<Body>,          // everything with a history
    pub ship: ShipId,               // the body driven by input (camera rides it)
    pub flashes: Vec<FlashEvent>,   // light pulses: event + duration + color
    pub input_log: Vec<InputEvent>, // (ship proper time, action) — for replay/branching
    pub frontier_origin: Vec4,      // current O
}

pub struct Body {
    pub kind: BodyKind,             // Puck, SoftBody, String, Relay, Moth, ...
    pub history: Vec<BodyState>,    // f64, uniformly stepped in this body's own step index
    pub render: RenderRecipe,       // mesh topology + material(s)
    pub decided_at: Vec<f32>,       // camera proper time when each step was finalized
}
```

`BodyState` is the full dynamic state at one step (positions/4-velocities of all vertices or
beads, proper times, Φ, internal state such as "relay lit"). Storing full state per step
makes truncation/branching trivial: restart integration from any kept step.

### 2.2 The frontier rule

```
decided(E, O)  :=  (E.t - O.t) <= |E.x - O.x|      // E ∉ I⁺(O), boundary counts as decided
```

A body may advance from step `k` to `k+1` only if

1. every new vertex/bead event at step `k+1` is `decided` w.r.t. the current `O`, and
2. every body it interacts with *at contact range* can also advance to that lab-time
   (conservative neighbour rule — contacts are local, so this rarely blocks), and
3. any retarded inputs it reads (other bodies' worldlines on its past light cone) are
   already in their histories — guaranteed by causal closure, but the scheduler must
   process in an order that has computed them.

Scheduler: repeat passes over bodies, advancing each as far as allowed, until a pass makes
no progress. Per frame the frontier moves by `~dt_wall · time_scale`, so typically each body
gains 0–1 steps per frame; this is cheap.

Free bonus: a body nobody can ever influence (`kind.is_inert()`) can be precomputed to the
end of time and skip scheduling — the static scenes are exactly this.

### 2.3 Retarded lookups

Many interactions need "where was body B on my past light cone?". Implement once:

```rust
fn retarded_state(hist: &[BodyState], at: Vec4) -> Option<(f64 /*step frac*/, BodyState)>
```

Find the step bracket where `f(k) = (at.t - X_k.t) - |at.x - X_k.x|` changes sign (binary
search; `f` is monotone along a timelike worldline), then solve linearly within the step.
Cache the last bracket per (body, source) pair — it moves forward by ~0–1 steps per call.

### 2.4 Memory/precision housekeeping

* **Pruning:** the renderer traces past rays up to `λ ≤ 180`. Drop history steps (and their
  GPU chunks) whose events are all further than that in the past of `O`'s past cone.
* **No looping:** `t` grows without bound; rebase the lab origin (subtract a constant `t₀`
  from everything) every ~1000 s to keep `f32` render coordinates precise. Physics in `f64`.
* **Determinism:** fixed step sizes, no wall-clock dependence inside the integrator, input
  applied at recorded ship proper times. Required for replay and branching.

---

## 3. Rendering growing geometry

### 3.1 Deforming worldtubes

Generalise the extruder:

```rust
pub fn build_deforming_tube(
    topo: &Mesh3D,                  // triangles + UVs (vertex positions ignored)
    steps: &[DeformStep],           // per step: per-vertex event, 4-velocity, proper time
    material_id: usize,
) -> Vec<Simplex4D>
```

The Dompierre prism split in `build_worldtube_simplices` never actually needs rigidity — it
only needs vertex events per step, so this is mostly a refactor. Differences:
* per-tet `four_velocity` = average of its vertices' 4-velocities;
* `surface_normal_4d` recomputed per step from the deformed triangle, projected orthogonal
  to the local 4-velocity;
* `proper_time` per vertex = that material point's own integrated τ; `tau_uv_grad = 0`.
  (Clock faces painted on jelly will visibly *smear* under strain — a feature.)
* Guard: skip/flag prisms where a vertex segment is non-timelike or the tet degenerates.

The Born-rigid builder stays for rigid-looking small things (pucks, ship) and for static scenes.

**Kinks.** A collision is an impulse — a Born-rigid shape can't take it consistently. Two
honest options, pick per body kind: (a) *soft impulse*: spread it over `δτ` with proper
acceleration `|a| < 1/r` (the existing Rindler-horizon assert is exactly this bound);
(b) *segment break*: end the tube at the contact event and start a new one, with a short
emissive "contact flash" material covering the seam.

### 3.2 Two-level BVH with an append arena

Geometry is **append-only behind the frontier**, which suggests:

* **Chunks (BLAS):** each body emits geometry in chunks of `K` steps (e.g. 16). A chunk is
  *sealed* once all its steps are decided; only each body's *tip chunk* is rebuilt when it
  grows. Sealed chunks are never touched again (until pruning or branching).
* **TLAS:** a small BVH over chunk AABBs, rebuilt every frame (a few thousand boxes → well
  under 1 ms with the existing SAH builder).
* **GPU arena:** preallocate large `nodes / geoms / attrs` storage buffers; write new/updated
  chunks with `queue.write_buffer` at offsets; free by ring allocation (chunks expire roughly
  in time order). The static scene becomes "one sealed chunk".
* **WGSL:** `trace()`/`is_occluded()` become two-level — TLAS leaf → (node offset, prim
  offset) into the arena, push the BLAS root, continue. Keep the CPU reference renderer on
  the same structure so the GPU-vs-CPU image-diff tests still guard it.

**Cheap first step (recommended):** before writing the TLAS, just rebuild the full BVH at
~10 Hz. In both existing views a thing at distance `D` was decided ≈ `2D` ago (round trip),
so a 0.1 s stale BVH only misses geometry closer than ~0.05 units. Only the frontier views
(§4) need fresh tips every frame. This lets the simulation be built and tuned first.

### 3.3 Flash lighting

`FlashEvent { at: Vec4, width: f32, color }`. A hit event `E` is lit by flash `F` iff
`F` is on `E`'s past light cone within the pulse width:
`| (E.t − F.t) − |E.x − F.x| | < width/2`, plus a shadow ray along that null segment (reuse
the Lorentz-invariant shadow path). Upload a small array (≤ 64) in a storage buffer; cull a
flash once its shell has passed beyond the scene radius. Result: you *watch* a wavefront of
light sweep the scene — with the extra delay of it reaching you.

---

## 4. New views and render modes (how you see the growth)

1. **Frontier view (`G` cycles: Real → God's Eye → Frontier).** Cast rays down the *future*
   null cone, `k = +e₀ + n`, from `O' = O − δ·u` (δ ≈ 0.1 s) so rays land just behind the
   frontier on fully computed geometry instead of grazing ragged tips. You see the world
   "as it is being written": objects appear at their most recently decided state, and the
   view shows where your *current* choices are about to propagate.
2. **Prophecy slice.** God's-Eye slice pushed to `dt' = +Δ` (scroll to change Δ). The part of
   the slice inside `I⁺(O)` is a ball of radius `Δ` around you that is genuinely undecided —
   render it as a shimmering "undetermined" haze (analytic: ray distance `< Δ`). Worldtubes
   visibly end at the bubble's surface. Increase Δ and the bubble swallows the scene.
3. **Decision-age render mode (`M`).** Each simplex vertex carries `decided_at` (camera
   proper time when it was finalized). Colour by `τ_now − decided_at`. In the Real Camera
   view this shows the central lesson: anything you see at distance `D` was settled ≈ `2D`
   ago — *you can never influence what you're currently looking at.*
4. **Spacetime-diagram inset (HUD).** A small `(x, t)` plot (software-drawn in `hud.rs`) of
   worldlines near the ship, your past/future cones, and the frontier sweeping forward.
   Invaluable for debugging the scheduler, and great for teaching.

---

## 5. Objects for a world without rigid bodies

Rigid bodies are out because pushing one end would move the other end instantly. Every
object below either is a point (no extent to be rigid), or is made of parts that talk to
each other no faster than `c`. Rough order = implementation order.

### 5.1 Pucks (relativistic billiards)
Small spheres = point particles with a radius. Elastic collisions solved exactly: boost to
the pair's CM frame, reflect momenta about the contact normal, boost back — 4-momentum
conserved. Fire them from the ship (`F`) at a chosen rapidity.
* **Inelastic mode:** pucks stick and the merged puck's **rest mass exceeds the sum** (kinetic
  energy → mass). Render radius ∝ mass^(1/3) and glow ∝ internal energy — `E = mc²` you can
  see.
* **Your ship is a puck too.** Get rammed at 0.5c and your camera tetrad gets kicked.

### 5.2 Spores (proper-time decays)
A puck with an internal clock that **splits into two pucks after proper time T**
(two-body decay, momenta fixed in its rest frame). Fire spores fast and they travel γ times
further before popping — muon time dilation, interactive. Chains of decays make fireworks.
Use the existing clock face on them so the countdown is readable.

### 5.3 Moths (retarded followers)
Agents that steer toward **where they see you**: at each step, look up the ship's retarded
position on the moth's past light cone and apply bounded proper acceleration toward it.
They swarm around where you *were*.
* **Smart moths** extrapolate your retarded velocity linearly — the same trick that makes a
  uniformly moving charge's field point at its *present* position. Fly inertially and they
  intercept you; swerve and they all overshoot together. Very direct demonstration of
  retardation.

### 5.4 Jelly (retarded-spring soft bodies)
A lattice of beads joined by springs where each bead feels its neighbours at their
**retarded** positions, projected into the bead's rest frame. Signals cross the body at the
spring "sound speed" (tunable 0.2c–c, never above). The skin is rendered with
`build_deforming_tube`.
* Ram a jelly cube: a compression wave crosses it; the far face doesn't budge until the
  wave arrives.
* **Tow it** with a tether: if you accelerate faster than the sound speed allows, it
  stretches; past a strain limit, bonds break and **fracture propagates at ≤ sound speed**.
* Born rigidity appears as the *limiting* behaviour — a slowly towed jelly reproduces the
  front/rear clock offsets that the rigid builder hard-codes today.

### 5.5 Light-speed strings ("the stiffest material possible")
A string whose transverse waves travel at exactly `c` in its rest frame. Discretised 1D wave
equation with leapfrog at Courant number 1 is *exact*, so pulses really ride the light
cone. Anchor strings between pillars; pluck by touching them with the ship or a puck. The
kink travels along your future light cone and reflects off the anchors. A "harp" of strings
at different velocities shows how wave speed transforms.

### 5.6 Relays (causal dominoes / light-cone cellular automaton)
Stationary or moving beacons that **switch colour and emit a flash** when the first flash
reaches them. Trigger one with your own flash (`L`): the set of lit relays at any moment
traces a light cone; a lattice of them draws your light cone in space. Variants:
* **Toggle relays** (flip on/off per received pulse) → a Lorentz-covariant cellular automaton
  whose patterns depend only on null-separated events.
* **A moving sheet of relays** next to a stationary one — they light up in different orders
  in different frames, but the causal order never disagrees.

### 5.7 Mirrors and light pulses
Movable mirrors that reflect flash pulses (null worldlines computed exactly). Build an
interactive light clock: nudge one mirror with your ship and watch the period change. Push
a mirror into a light pulse to see Doppler-shifted reflection (colour the reflected flash).
Photons themselves stay invisible in vacuum — you only see what they illuminate — which is
itself worth showing.

### 5.8 Wingmen (Bell's spaceships, interactive)
A second ship that executes **your thrust commands as received by radio** (i.e. delayed by
the light-travel time from you), with a fragile string between you. With the radio delay
the string behaves one way; switch to "pre-programmed identical thrust in the lab frame"
and it snaps — Bell's paradox, played rather than watched.

### 5.9 Charges (stretch goal)
Particles interacting by a bounded, retarded (Liénard–Wiechert-like) force. Two-body
"dances" that don't close, apparent momentum non-conservation (the field carries it).
Integration is a delay ODE — the retarded lookup from §2.3 does all the work.

---

## 6. Time controls in a causal world

* **Pause / speed (`P`, `-`, `=`):** unchanged; the frontier just moves slower.
* **Rewind (`T`):** move the camera back along its *recorded* worldline. Everything there is
  already decided, so nothing needs computing — it's a replay.
* **Branching:** if you give new input while rewound, that's a new timeline from
  `O_branch`: truncate every body's history to steps outside `I⁺(O_branch)`, drop the input
  log after it, invalidate overlapping chunks, and resume. Optionally keep the discarded
  branch as translucent **"ghost timeline"** geometry for a while — you see what *would*
  have happened, fading out.
* **Coordinate scrubbing (`[ ]`, `IJKL/UO`):** these teleport the camera along non-timelike
  paths, which breaks the "camera = ship" model. Proposal: in Dynamic Mode they detach a
  non-interacting **drone camera** from the ship. The ship holds its last input (or is
  paused); the drone can wander, and anything it looks at inside the ship's undecided
  region is drawn as haze, as in the prophecy slice.
* **No time loop.** Scenes run forward indefinitely (`R` resets).

---

## 7. Suggested dynamic scenes

| Key | Scene | Shows |
|---|---|---|
| 6 | **Relativistic billiards** — rack of pucks, ship fires cue pucks; inelastic toggle | 4-momentum conservation, mass from KE, getting rammed |
| 7 | **Jelly yard** — jelly cubes, a long rod, tow tether | sound-speed propagation, no rigid bodies, fracture |
| 8 | **Causal dominoes** — relay lattice + moving relay sheet | light cones drawn in space, causal order invariance |
| 9 | **Moth swarm** — dumb and smart moths | retardation, "field points at present position" |
| 0 | **Light harp** — light-speed strings | waves riding the light cone |

Spores, mirrors and wingmen can be sprinkled into these or get their own scenes later.

---

## 8. Phased implementation

1. **Refactor (no behaviour change).** Introduce a `World` trait with `StaticScene` (today's
   `Scene4D`) and `DynWorld`; split out `build_deforming_tube`; add tube end caps.
   *All existing tests must still pass.*
2. **Sim core + pucks.** `DynWorld`, frontier rule, scheduler, pruning, ship-as-body, input
   log, puck collisions, firing. Render with the 10 Hz full-rebuild shortcut. HUD inset.
3. **Seeing the frontier.** Frontier view, prophecy slice, decision-age mode, flash lighting.
   Then the real two-level BVH + GPU arena (needed now that tips must be fresh every frame).
4. **Moths and spores** (cheap once 2–3 exist).
5. **Soft bodies**: jelly, tethers, fracture, strings.
6. **Relays and mirrors** (flash-driven state machines).
7. **Rewind + branching + ghost timelines; drone camera.**
8. Stretch: wingmen/Bell, charges.

---

## 9. Tests that keep it honest

* **Causality property test (the big one):** run the world twice with identical input up to
  `O`, then different input; assert every history step outside `I⁺(O)` is bit-identical.
* **Determinism:** replaying the input log reproduces histories bit-exactly.
* **Monotone frontier:** no step is ever finalized inside `I⁺(O)`; `decided_at` is
  non-decreasing along each history.
* **Collisions:** total 4-momentum conserved to 1e-9 (f64) for elastic and inelastic.
* **Strings:** pulse arrival times equal `L/c` to step precision.
* **Jelly:** far face starts moving no earlier than `L / c_sound` after contact; never
  before `L/c` regardless of parameters.
* **Spores:** decay lab-time = `γ·T`.
* **Rendering:** existing GPU-vs-CPU diffs extended to a dynamic snapshot; static scenes
  unchanged pixel-for-pixel after the two-level BVH refactor.

---

## 10. Open questions / risks

* **Prism quality under large deformation** — very stretched jelly may produce slivers; may
  need adaptive time steps or a remesh rule on fracture.
* **Frontier jitter** — the frontier moves continuously but bodies step discretely; the
  frontier view needs the `δ` offset (or interpolated tip caps) to avoid flicker.
* **Scheduler starvation** — a chain of contacting bodies near the ship gates each other;
  keep the neighbour rule conservative but local.
* **Retarded springs aren't momentum-conserving instantaneously** (the "springs" carry
  momentum in transit). Physically fine, but bodies can drift; may want light damping.
* **What the drone does to "input".** If the ship coasts while the drone roams, is the
  world fully determined? (Yes, if the ship's future is fixed — which would make the
  frontier jump to infinity. Simplest is to pause ship time while droning.)
* **Units.** README mixes "m" and "light-seconds" (c = 1); worth pinning down before tuning
  sound speeds and accelerations.
