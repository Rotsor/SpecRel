# SpecRel 4D — Dynamic ("Causal") Mode: Design Plan

> **Revision 2.** The first attempt at this plan produced a large, uncommitted, non-compiling
> diff that tried to touch every phase at once. It is preserved on branch
> **`draft/dynamic-mode-wip`** (commit `aca4c2b`) as a parts bin. This revision keeps the
> physics design (§0, §5–§6) and replaces the implementation plan with small phases that each
> compile, pass tests, and keep the browser build working. §8 says, phase by phase, which draft
> code to reuse.

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

## 1. Ground rules (lessons from the draft)

1. **Every phase lands green.** `cargo test --release` passes, `cargo build --release
   --target wasm32-unknown-unknown` passes with no new warnings, and the web page
   (`web/index.html`) still runs. The browser build is production code; Dynamic Mode must
   never break it, and each feature should reach the browser in the same phase as the
   desktop build, not later.
2. **Static scenes stay pixel-identical** unless a phase says otherwise. The existing
   GPU-vs-CPU image diffs are the guard.
3. **No placeholders in user-facing lists.** A scene enters `DemoScene` (title, keys, HUD
   text) only in the phase that makes it work. The draft added five dynamic scenes whose
   descriptions advertised controls that didn't exist.
4. **Per-frame state is not scene state.** The frontier origin, the frontier proper time and
   active flashes change every frame. They go in a `FrameDynamics` value that is passed to
   `render_frame`, `GpuRenderer::render` and `web.rs` alongside the camera, and is written
   into the uniform block. They are **not** fields of `Scene4D`, `PackedSceneGpuData` or
   `SceneBuffers`. (The draft stored them on the scene, which forced a full BVH re-upload just
   to move the frontier, and `web.rs` silently ignored them.)
5. **Keys don't collide.** Every letter key is already bound (see README controls). New
   actions get non-letter keys (§7), and these are wired in **both** `main.rs` and `web.rs`
   (and the on-screen buttons in `index.html`).
6. **The simulation is single-threaded, deterministic `f64` code** with no `rayon`, no
   wall-clock reads and no `std::time`, so it runs unchanged on `wasm32`.

---

## 2. Simulation core (`src/dynamic/`)

### 2.1 Types

```rust
pub struct DynWorld {
    pub bodies: Vec<Body>,          // everything with a history
    pub ship: BodyId,               // the body driven by input (camera rides it)
    pub flashes: Vec<FlashEvent>,   // light pulses: event + width + color
    pub input_log: Vec<InputEvent>, // (ship proper time, action) — for replay/branching
    pub frontier_origin: Vec4,      // current O
}

pub struct Body {
    pub kind: BodyKind,             // Puck, Spore, Relay, Moth, SoftBody, String, ...
    pub history: Vec<BodyState>,    // f64, uniformly stepped in this body's own step index
    pub decided_at: Vec<f32>,       // camera proper time when each step was finalized
    pub render: RenderRecipe,       // mesh topology + material(s)
}
```

`BodyState` is the full dynamic state at one step: positions and 4-velocities of all vertices
or beads, proper times, `Φ`, and internal state such as "relay lit". Storing full state per step
makes truncation and branching trivial: integration restarts from any kept step.

Proposed layout: `src/dynamic/mod.rs` (`DynWorld`, scene construction), `frontier.rs`
(decided rule, scheduler), `body.rs`, `retarded.rs`, `collide.rs`, `tube.rs` (history →
`Simplex4D`s). Bodies get their own files as they land (`puck.rs`, `relay.rs`, …).

### 2.2 The frontier rule

```
decided(E, O)  :=  (E.t - O.t) <= |E.x - O.x|      // E ∉ I⁺(O); the boundary counts as decided
```

A body may advance from step `k` to `k+1` only if all of the following hold:

1. Every new vertex or bead event at step `k+1` is `decided` w.r.t. the current `O`.
2. Every body it interacts with *at contact range* can also advance to that lab time. This is
   a conservative neighbour rule; contacts are local, so it rarely blocks.
3. Any retarded inputs it reads (other bodies' worldlines on its past light cone) are already
   in their histories. Causal closure guarantees this exists, but the scheduler must process
   bodies in an order that has computed them.

Scheduler: repeatedly pass over the bodies, advancing each as far as allowed, until a pass
makes no progress. Per frame the frontier moves by `~dt_wall · time_scale`, so each body
typically gains 0–1 steps per frame, which is cheap.

A body nobody can ever influence (`kind.is_inert()`) can be precomputed to the end of time and
skip scheduling. The static scenes are exactly this case.

### 2.3 Retarded lookups

Many interactions need "where was body B on my past light cone?". Implement it once:

```rust
fn retarded_state(hist: &[BodyState], at: Vec4) -> Option<(f64 /*step frac*/, BodyState)>
```

Find the step bracket where `f(k) = (at.t - X_k.t) - |at.x - X_k.x|` changes sign (binary
search; `f` is monotone along a timelike worldline), then solve linearly within the step.
Cache the last bracket per (body, source) pair, since it moves forward by ~0–1 steps per call.

### 2.4 Memory and precision housekeeping

* **Pruning:** the renderer traces past rays up to `λ ≤ 180`. Drop history steps (and their
  geometry) whose events are all further than that into the past of `O`'s past cone.
* **No looping:** dynamic scenes skip the `[t_min, t_max]` wrap in `main.rs`/`web.rs`, so
  `t` grows without bound. Rebase the lab origin (subtract a constant `t₀` from everything)
  every ~1000 s to keep `f32` render coordinates precise. Physics stays in `f64`.
* **Determinism:** fixed step sizes, no wall-clock dependence inside the integrator, and input
  applied at recorded ship proper times. This is required for replay and branching.

---

## 3. Rendering growing geometry

### 3.1 Tube builders

* **Rigid tubes** use the existing `build_worldtube_simplices`, plus per-step `decided_at`
  stamping and **end caps** (`build_worldtube_cap`) at the frontier tip and at
  decay/merge events. Both exist in the draft.
* **Deforming tubes** (`build_deforming_tube` / `build_deforming_cap`, also in the draft)
  give each vertex its own sampled worldline:
  * per-tet `four_velocity` = the normalised average of its vertices' 4-velocities;
  * `surface_normal_4d` recomputed per step from the deformed triangle and projected
    orthogonal to the local 4-velocity;
  * `proper_time` per vertex = that material point's own integrated τ, with `tau_uv_grad = 0`.
    Clock faces painted on jelly visibly *smear* under strain, which is a feature.
  * The guard must reject vertex segments that are not **timelike**. The draft only checked
    `Δt > 0`.

**Kinks.** A collision is an impulse, and a Born-rigid shape can't take one consistently. Two
honest options, chosen per body kind:

* **(a) Soft impulse:** spread the impulse over `δτ` with proper acceleration `|a| < 1/r`
  (the existing Rindler-horizon assert is exactly this bound).
* **(b) Segment break:** end the tube at the contact event with a cap and start a new one.
  This is the default for pucks.

### 3.2 Getting geometry to the GPU

* **Step 1 (enough for phases 3–6):** rebuild the dynamic scene's whole BVH at ~10 Hz and
  replace its GPU buffers. Natively, this is the draft's `GpuRenderer::update_scene`. On the
  web it needs a new `web_scene_dirty(idx)` → JS re-creates that scene's buffers and bind
  group; today `index.html` builds them once at startup. In both existing views, a thing at
  distance `D` was decided ≈ `2D` ago, so a 0.1 s stale BVH only misses geometry closer than
  ~0.05 units. Measure `Bvh4D::build` cost on wasm (single-threaded) before relying on it.
* **Step 2 (only when measurements demand it, phase 7):** a two-level BVH with an append
  arena:
  * Each body emits geometry in chunks of `K` steps; a chunk is sealed once all its steps are
    decided, and only tip chunks are rebuilt.
  * A small TLAS over the chunk AABBs is rebuilt every frame.
  * The GPU storage buffers are preallocated, and chunks are written with
    `queue.write_buffer` at offsets.
  * `trace()`/`is_occluded()` in WGSL become two-level, and the CPU renderer mirrors this so
    the image diffs still guard it.

### 3.3 Flash lighting

`FlashEvent { at: Vec4, width: f32, intensity: f32, color: Vec3 }`. A hit event `E` is lit by
flash `F` iff `F` is on `E`'s past light cone within the pulse width,
`| (E.t − F.t) − |E.x − F.x| | < width/2`, plus a shadow ray along that null segment (reusing
the Lorentz-invariant shadow path). Up to `MAX_FLASHES = 16` go in the per-frame uniform
block. A flash is culled once its shell has passed beyond the scene radius. The result: you
*watch* a wavefront of light sweep the scene, with the extra delay of it reaching you.

---

## 4. New views and render modes

1. **Frontier view (`G` cycles: Real → God's Eye → Frontier → Prophecy).** Cast rays down the
   *future* null cone, `k = +e₀ + n`, from `O' = O − δ·u` (δ = 0.14 s). Every point on
   `∂J⁺(O')` beyond `λ > δ/2` lies outside `I⁺(O)`, so rays land on fully computed geometry.
   You see the world "as it is being written".
2. **Prophecy slice.** God's-Eye slice pushed to `dt' = +Δ` (PageUp/PageDown change Δ). The
   part of the slice inside `I⁺(O)` is a ball of radius `Δ` that is genuinely undecided,
   rendered as an animated "undetermined" haze. Worldtubes visibly end at the bubble surface.
3. **Decision-age render mode (`M`).** Colour by `τ_now − decided_at`. For static geometry
   (`decided_at = DECIDED_AT_STATIC`) the age is analytic: for an inertial camera it is
   `T + R` for an event `T` in the past at distance `R`, which is `2R` on the past cone. This
   shows the central lesson: *you can never influence what you're currently looking at.*
4. **Spacetime-diagram inset (HUD).** A small `(x, t)` plot of worldlines near the ship, your
   past and future cones, and the frontier sweeping forward. It is drawn in `hud.rs` so it
   appears in both the native framebuffer and the web HUD overlay. It is invaluable for
   debugging the scheduler, and great for teaching.

Views 1–3 are meaningful on **static** scenes too, so they ship before the simulation exists
(phase 2).

---

## 5. Objects for a world without rigid bodies

Rigid bodies are out, because pushing one end would move the other end instantly. Every
object below is either a point (no extent to be rigid) or made of parts that talk to each
other no faster than `c`. They are listed roughly in implementation order.

### 5.1 Pucks (relativistic billiards)
Small spheres = point particles with a radius. Elastic collisions are solved exactly: boost to
the pair's CM frame, reflect momenta about the contact normal, boost back, conserving
4-momentum. The ship fires them at a chosen rapidity.
* **Inelastic mode:** pucks stick, and the merged puck's **rest mass exceeds the sum**
  (kinetic energy → mass). Render radius ∝ mass^(1/3) and glow ∝ internal energy: `E = mc²`
  you can see.
* **Your ship is a puck too.** Get rammed at 0.5c and your camera tetrad gets kicked.

### 5.2 Spores (proper-time decays)
A puck with an internal clock that **splits into two pucks after proper time T** (two-body
decay, momenta fixed in its rest frame). Fire spores fast and they travel γ times further
before popping: muon time dilation, interactive. Chains of decays make fireworks. They carry
the existing clock face so the countdown is readable.

### 5.3 Relays (causal dominoes / light-cone cellular automaton)
Beacons that **switch colour and emit a flash** when the first flash reaches them. Trigger one
with your own flash: the set of lit relays at any moment traces a light cone, so a lattice of
them draws your light cone in space.
* **Toggle relays** (flip on/off per received pulse) form a Lorentz-covariant cellular
  automaton.
* **A moving sheet of relays** next to a stationary one lights up in a different order in
  different frames, but never in a different causal order.

### 5.4 Moths (retarded followers)
Agents that steer toward **where they see you**: each step, look up the ship's retarded
position and apply bounded proper acceleration toward it. They swarm around where you *were*.
* **Smart moths** extrapolate your retarded velocity linearly, the same trick that makes a
  uniformly moving charge's field point at its *present* position. Fly inertially and they
  intercept you; swerve and they all overshoot together.

### 5.5 Jelly (retarded-spring soft bodies)
A lattice of beads joined by springs, where each bead feels its neighbours at their
**retarded** positions, projected into the bead's rest frame. The spring "sound speed" is
tunable from 0.2c to c, never above. The skin uses `build_deforming_tube`.
* **Ram a jelly cube:** a compression wave crosses it, and the far face doesn't budge until
  the wave arrives.
* **Tow one with a tether:** over-accelerate and it stretches; past a strain limit, bonds
  break and **fracture propagates at ≤ sound speed**.
* **Born rigidity is the limiting behaviour:** a slowly towed jelly reproduces the front/rear
  clock offsets that the rigid builder hard-codes today.

### 5.6 Light-speed strings
Transverse waves travel at exactly `c` in the string's rest frame. A discretised 1D wave
equation with leapfrog at Courant number 1 is *exact*, so pulses really ride the light cone.
Anchor strings between pillars and pluck them with the ship or a puck. A "harp" of strings at
different velocities shows how wave speed transforms.

### 5.7 Mirrors and light pulses
Movable mirrors reflect flash pulses, with null worldlines computed exactly. Build an
interactive light clock: nudge one mirror and watch the period change. Push a mirror into a
pulse to see a Doppler-shifted (recoloured) reflection.

### 5.8 Wingmen (Bell's spaceships, interactive)
A second ship executes **your thrust commands as received by radio**, with a fragile string
between you. Switch to "pre-programmed identical lab-frame thrust" and the string snaps:
Bell's paradox, played rather than watched.

### 5.9 Charges (stretch goal)
Particles interacting through a bounded, retarded (Liénard–Wiechert-like) force. Integration
is a delay ODE, and the retarded lookup from §2.3 does all the work.

---

## 6. Time controls in a causal world

* **Pause / speed (`P`, `-`, `=`):** unchanged; the frontier just moves slower.
* **Rewind (`T`):** move the camera back along its *recorded* worldline. Everything there is
  already decided, so it's a replay.
* **Branching:** new input while rewound starts a new timeline from `O_branch`:
  * truncate every body's history to the steps outside `I⁺(O_branch)`;
  * drop the input log after `O_branch`;
  * invalidate the affected geometry, and resume.

  Optionally keep the discarded branch as fading translucent **"ghost timeline"** geometry.
* **Coordinate scrubbing (`[ ]`, `IJKL/UO`):** these teleport along non-timelike paths, which
  breaks "camera = ship". In Dynamic Mode they detach a non-interacting **drone camera**. Ship
  time pauses while droning (otherwise the ship's future would be fixed and the frontier would
  jump to infinity). Anything the drone sees inside the ship's undecided region is drawn as
  haze.
* **No time loop.** Dynamic scenes run forward indefinitely (`R` resets).

---

## 7. Keys and scenes

All 26 letters are taken, so the new actions use non-letter keys. They must be wired in
`main.rs`, `web.rs` and the `index.html` buttons.

| Key | Action | Phase |
|---|---|---|
| `G` | view cycle gains Frontier, Prophecy | 2 |
| `M` | render-mode cycle gains Decision Age | 2 |
| `PageUp` / `PageDown` | prophecy Δ ± | 2 |
| `/` | emit a light flash from the ship | 2 |
| `Enter` | fire puck (dynamic scenes) | 4 |
| `\` | toggle elastic/inelastic, spore/puck ammo | 4 |
| `6`–`0` | dynamic scenes | 3+ (one key per scene as each lands) |

| Key | Scene | Shows | Phase |
|---|---|---|---|
| 6 | **Relativistic billiards**: rack of pucks + spores | 4-momentum conservation, mass from KE, time dilation, getting rammed | 3–4 |
| 7 | **Causal dominoes**: relay lattice + moving relay sheet | light cones drawn in space, causal-order invariance | 5 |
| 8 | **Moth swarm**: dumb and smart moths | retardation, "field points at present position" | 6 |
| 9 | **Jelly yard**: jelly cubes, a long rod, tow tether | sound-speed propagation, no rigid bodies, fracture | 8 |
| 0 | **Light harp**: light-speed strings | waves riding the light cone | 8 |

---

## 8. Phased implementation

Each phase is one PR. "Draft" refers to branch `draft/dynamic-mode-wip` (`aca4c2b`).

### Phase 0 — WASM baseline ✅ (branch `wasm-port`, commit `3d91f62`)
The browser port was sitting uncommitted under the dynamic work. It is now its own commit:
shared GPU packing (`PackedSceneGpuData`, `GpuUniforms::build`, `build_tables_data`), HUD
moved into `hud.rs` with a GPU overlay variant, `src/web.rs`, `web/index.html`, and `wasm32`
gating of native-only dependencies. Built `.wasm` files are gitignored. `main` has been
fast-forwarded to it.

### Phase 1 — Plumbing + flash lighting, no visible change ✅
* `renderer::FrameDynamics { frontier: Option<Frontier { origin, proper_time }>, flashes }`
  (+ `FlashEvent`, `MAX_FLASHES = 16`) is passed per frame to `render_frame`,
  `GpuRenderer::render` and `GpuUniforms::build`. `main.rs` and `web.rs` hold an empty one;
  dynamic scenes will fill it.
* `GpuUniforms` / WGSL `Uniforms` carry `frontier_origin`, `frontier_active`, `frontier_tau`,
  `num_flashes` and 16 flash slots. The frontier fields are not read by the shader yet.
* Per-vertex `decided_at: [f32; 4]` on `Simplex4D`, interpolated into `HitRecord4D`, packed
  into `GpuAttr` / WGSL `Attr` / `HitRecord`; `DECIDED_AT_STATIC`, `with_decided_at`,
  `build_worldtube_simplices_with_decided`.
* **Flash lighting moved here from phase 2** (CPU `compute_flash_illumination` + WGSL
  mirror), because a GPU-vs-CPU render *with* flashes is the only end-to-end check that the
  new uniform layout is right. With no flashes the term is skipped entirely, so static scenes
  render exactly as before.
* **Tests:** `wgsl_struct_layouts_match_rust` (naga parses `render.wgsl` and compares the size
  and member offsets of `Uniforms`, `Attr`, `Node`, `Geom`, `Material` with the Rust
  mirrors); `gpu_matches_cpu_with_flashes` (two flash shells on every static scene, both
  shadow modes, also asserts the flashes visibly light ≥ 1% of pixels);
  `worldtube_decided_at_follows_steps`.

### Phase 2 — New views on static scenes
Frontier view, Prophecy slice (+ Δ keys), Decision-Age mode with the analytic fallback, and a
`/` key that emits a flash from the ship into `FrameDynamics` (lighting already exists from
phase 1). All are useful before any simulation exists.
* **Reuse from the draft:** `minkowski.rs` `future_light_ray_direction`; `renderer.rs`
  `ViewMode::{Frontier, ProphecySlice}`, `RenderMode::DecisionAge`, `FRONTIER_RAY_OFFSET`,
  `shade_decision_age`, `apply_causal_bubble`, `Camera4D.prophecy_delta`; `render.wgsl`
  equivalents (`SLICE_FRONTIER`, `SLICE_PROPHECY`, `MODE_DECISION_AGE`, bubble). The bubble
  and decision age read the frontier from `FrameDynamics` / `u.frontier_*`.
* **Fix:** in Frontier view every hit sits ~δ outside the frontier and gets a faint ring tint;
  the ring should only draw when `δ` is smaller than the ring width, or move the ring test to
  `|signed_inside + δ|`.
* **Tests:** GPU-vs-CPU diffs for each new view/mode on a static scene; analytic
  decision-age = `T + R`.

### Phase 3 — Dynamic world skeleton + scene 6 (inert)
* Add `src/dynamic/` with `DynWorld`, `Body`, the frontier rule and scheduler, ship-as-body,
  and the input log.
* Add the billiards scene with **inert** pucks only (no collisions yet); their tubes grow up
  to the frontier and end in caps.
* Wire dynamic scenes into the time code (no looping, origin rebase), plus the 10 Hz
  rebuild: `GpuRenderer::update_scene` and `web_scene_dirty`.
* **Reuse from the draft:** `build_worldtube_cap`; `GpuRenderer::update_scene` (+
  `scene_layout` field); materials 7–8 from `common_materials`; billiards spawn/velocity from
  `DemoScene::spawn_offset`/`spawn_velocity`.
* **Tests:** no step is ever finalized inside `I⁺(O)`; `decided_at` non-decreasing; replaying
  the input log is bit-identical; a full dynamic frame renders identically on CPU and GPU.

### Phase 4 — Pucks and spores
Collisions (elastic + inelastic), firing (`Enter`), ammo toggle (`\`), spores, and the ship
being rammed.
* **Reuse from the draft:** materials 7–9.
* **Tests:** 4-momentum is conserved to 1e-9 for elastic and inelastic collisions; spore decay
  lab time = `γ·T`; the **causality property test** (same input up to `O`, different after →
  every step outside `I⁺(O)` is bit-identical).

### Phase 5 — Relays (scene 7) and HUD spacetime inset
Flash-driven state machines; flashes emitted by the simulation feed `FrameDynamics`. The HUD
inset is drawn in `hud.rs` for both native and web.
* **Reuse from the draft:** materials 11–12.
* **Tests:** relay `k` lights at its retarded time from the trigger.

### Phase 6 — Moths (scene 8)
`retarded_state` (§2.3), dumb vs smart steering.
* **Reuse from the draft:** materials 13–14.
* **Tests:** a retarded lookup on an inertial worldline matches the closed form.

### Phase 7 — Two-level BVH + GPU arena (only if needed)
Only if phase 3–6 measurements show the 10 Hz full rebuild is too slow on web or causes
visible stale tips in Frontier view. Static scenes must stay pixel-identical.

### Phase 8 — Soft bodies: jelly (scene 9) and strings (scene 0)
* **Reuse from the draft:** `DeformStep`, `build_deforming_tube`, `build_deforming_cap`
  (fix the timelike guard); materials 10, 15.
* **Tests:** the jelly's far face starts moving no earlier than `L / c_sound` after contact,
  and never before `L/c`; string pulse arrival = `L/c` to step precision.

### Phase 9 — Rewind, branching, ghost timelines, drone camera

### Phase 10 — Stretch: mirrors, wingmen/Bell, charges

### Draft code not carried forward
* The `DemoScene` variants, titles and descriptions for all five dynamic scenes, added in
  bulk. Re-add each with its phase, with descriptions that only mention controls that exist.
* `Scene4D.{flashes, frontier_origin, frontier_tau}`, and their copies in the GPU packing.
* `GpuUniforms::build_with_dynamic` as a separate entry point (folded into `build`).

---

## 9. Tests that keep it honest

* **Causality property test (the big one):** run the world twice with identical input up to
  `O`, then different input; assert that every history step outside `I⁺(O)` is
  bit-identical.
* **Determinism:** replaying the input log reproduces histories bit-exactly. This holds on
  wasm too: a native test can't prove it, so add a check that compares a history hash printed
  by the web build against a hash recorded natively.
* **Monotone frontier:** no step is ever finalized inside `I⁺(O)`; `decided_at` is
  non-decreasing along each history.
* **Collisions:** total 4-momentum conserved to 1e-9 (f64), elastic and inelastic.
* **Strings:** pulse arrival times equal `L/c` to step precision.
* **Jelly:** the far face starts moving no earlier than `L / c_sound` after contact, and never
  before `L/c`.
* **Spores:** decay lab time = `γ·T`.
* **Rendering:** GPU-vs-CPU diffs extended to each new view/mode and to a dynamic snapshot;
  static scenes unchanged pixel-for-pixel through every phase.

---

## 10. Open questions / risks

* **Wasm rebuild budget:** `Bvh4D::build` is single-threaded on the web. If a 10 Hz rebuild
  of a busy dynamic scene costs more than a few ms, phase 7 moves earlier.
* **Prism quality under large deformation:** very stretched jelly may produce slivers; it may
  need adaptive time steps or a remesh rule on fracture.
* **Frontier jitter:** the frontier moves continuously but bodies step discretely; the
  Frontier view relies on the `δ` offset (or interpolated tip caps) to avoid flicker.
* **Scheduler starvation:** a chain of contacting bodies near the ship gates each other; keep
  the neighbour rule conservative but local.
* **Retarded springs aren't instantaneously momentum-conserving** (the "springs" carry
  momentum in transit). This is physically fine, but bodies can drift; light damping may help.
* **Units.** The README mixes "m" and "light-seconds" (c = 1); pin this down before tuning
  sound speeds and accelerations.
