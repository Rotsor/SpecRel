# SpecRel 4D — Special-Relativistic Spacetime Raytracer

A real-time 4D Minkowski spacetime raytracer written in Rust that treats **Special Relativity** as pure 4D geometry rather than post-processing tricks.

Instead of representing a scene as 3D objects at a single global time $t$, **SpecRel 4D** represents every object's trajectory as a **4D tetrahedral mesh (worldtube)** in $(t, x, y, z)$ Minkowski spacetime ($\eta = \text{diag}(-1, 1, 1, 1)$, $c = 1$). At each frame, the observer sits at a 4D event $O^\mu$ with an orthonormal comoving Lorentz frame (tetrad) $\{e_{(0)}^\mu = u^\mu, e_{(1)}^\mu, e_{(2)}^\mu, e_{(3)}^\mu\}$ and casts rays down the **past light cone** ($ds^2 = 0$):

$$X^\mu(\lambda) = O^\mu + \lambda k^\mu, \qquad k^\mu = -e_{(0)}^\mu + n^1 e_{(1)}^\mu + n^2 e_{(2)}^\mu + n^3 e_{(3)}^\mu$$

Every relativistic optical and geometric phenomenon—**time dilation**, **Lorentz length contraction**, **relativity of simultaneity**, **Born-rigid proper-time gradients ($\tau_p = \tau_c + \mathbf{p}\cdot\boldsymbol{\Phi}$)**, **Terrell–Penrose apparent rotation**, **Penrose sphere circularity**, **relativistic aberration**, **Thomas–Wigner rotation**, and **retarded-time shadows**—emerges automatically from ray–tetrahedron intersections in 4D spacetime.

---

## Features

* **True 4D Worldtube Geometry (`Simplex4D`)**:
  * 3D triangle meshes are extruded along arbitrary timelike worldlines using **Fermi–Walker transport** (Born-rigid comoving frames) and decomposed into watertight 4D tetrahedra (3-simplices) using Dompierre prism splitting.
  * Includes the **Born-rigid proper-time gradient** $d\tau_p = (1 + \mathbf{a}_{\text{local}}\cdot\mathbf{p})\,d\tau_c \implies \tau_p = \tau_c + \mathbf{p}\cdot\boldsymbol{\Phi}(\tau_c)$ across accelerated bodies (clocks at the front of an accelerating body run faster than clocks at the rear, and resynchronize when the net rapidity change $\boldsymbol{\Phi}$ returns to zero).
* **4D Spacetime Bounding Volume Hierarchy (`Bvh4D`)**:
  * Custom 8-bin 4D Surface Area Heuristic (SAH) BVH over $(t, x, y, z)$ hyperboxes with front-to-back child traversal and compact 80-byte precomputed affine covector intersection records.
* **Animated Proper-Time ($\tau$) Clock & Light-Clock Textures**:
  * **Analog + Digital Clock Faces**: Driven purely by the interpolated local proper time $\tau$ at the 4D hit event. Features 10 decimal rim notches (`0..9`), three geared hands (`1.0s` cyan tenths hand, `10.0s` yellow seconds hand, `100.0s` red tens-of-seconds hand), and a 5x7 LED digital readout (`+24.7S`).
  * **Einstein Bouncing-Photon Light Clocks**: A laser pulse bounces vertically between two mirrors separated by $L = 1.0\text{ light-second}$ in the rest frame with period $T = 2L/c = 2.0\text{s}$ (exact photon speed $1.0c$ in the rest frame, $1.0\text{s}$ per one-way bounce), visually tracing out a sawtooth light-like path when moving.
* **High-Performance GPU Compute Backend (`wgpu` + WGSL)**:
  * Pure compute shader pipeline (`src/shaders/render.wgsl`) executing 4D BVH traversal, affine ray–tetrahedron intersection, and procedural relativistic shading per pixel directly on the GPU.
  * Static scene data (BVH nodes, 80-byte hot records, cold attributes, materials) is uploaded once into storage buffers; only a compact uniform block (camera event, tetrad, slice/shadow modes) updates per frame.
  * Achieves **~1.5–2.0 ms per frame** at 1280x800 resolution (over **500 FPS**, a **17x–39x speedup** over multithreaded CPU rendering).
  * Automatically detects and uses the GPU backend by default, falling back gracefully to the multithreaded CPU reference renderer if no compatible adapter is found.
  * The CPU path remains an exact **reference renderer** with automated bit-level image comparison tests ensuring pixel-accurate cross-backend consistency.
* **Interactive Spacetime View & Shadow Modes**:
  * Toggle between **Real Camera ($ds^2 = 0$)** (what a physical camera sees down the past light-cone) and **God's Eye ($dt' = 0$)** (an instantaneous 3D simultaneity slice across the observer's current Lorentz frame $\langle X - O, u_{\text{cam}} \rangle_\eta = 0$).
  * **Causal views** (groundwork for the planned Dynamic Mode, see [`docs/DYNAMIC_MODE_PLAN.md`](docs/DYNAMIC_MODE_PLAN.md)): **Frontier** traces rays down the camera's *future* light cone, showing the world exactly as it leaves your reach; **Prophecy** pushes the God's-Eye slice $\Delta$ seconds into your future, where the ball of radius $\Delta$ that you can still influence is drawn as violet haze; the **Causal Decision Age** render mode colours every event by how long ago it became immune to anything you do ($T + R$, i.e. $2R$ for everything you see).
  * **Light flashes** (`/`): emit a spherical light pulse from the camera and watch its wavefront sweep the scene, seen with the extra delay of the light coming back to you.
  * Cycle between **Lorentz-Invariant Null Shadows** ($t_{\text{emit}} = t_{\text{hit}} - \|\mathbf{x}_L - \mathbf{x}_{\text{hit}}\|$), **Camera-Slice Faked Shadows** ($\langle X_{\text{emit}} - X_{\text{hit}}, u_{\text{cam}} \rangle_\eta = 0$), and **Shadows Off**.

---

## 5 Built-In Paradox & Thought-Experiment Scenes

Switch scenes at any time with keys **`1`–`5`** or **`N` / `B`**:

1. **`[1]` Twin Paradox (Stay-at-Home vs. Accelerated Traveler)**
   * Two twins start synchronized ($\tau = 0$) at $t = -14\text{s}$. The right twin executes a smooth relativistic round-trip peaking near $0.94c$ ($\gamma \approx 3$) and parks back next to the stay-at-home twin at $t = +14\text{s}$ with an **$\sim 8\text{s}$ proper-time gap** visible on both their analog/digital dials and their comoving Einstein light clocks. During the high-speed cruise, the traveler's front and rear clock faces also display the Born-rigid proper-time offset ($\Delta\tau = 2 p\,\text{atanh}(\beta) \approx 2\text{s}$) before resynchronizing upon return.
2. **`[2]` Barn & Pole (Ladder) Paradox ($8\text{m}$ Pole in $5\text{m}$ Barn at $0.85c$)**
   * An $8\text{m}$ rest-length pole (tiled with 8 synchronized comoving clocks) flies at $0.85c$ ($\gamma \approx 1.90$) through a $5\text{m}$ barn.
   * Pause (`P`) near $t = 0$ and switch to God's Eye (`G`): in the barn's frame (`X`), the pole is contracted to $4.21\text{m}$ and fits completely inside the barn, with its rear end reading $\beta L_0 = +6.8\text{s}$ ahead of its front end ($+4.84\text{s}$ across a $3\text{m}$ lab separation $\gamma \beta \Delta x$). Press `V` to boost into the pole's rest frame: the pole's clocks all synchronize to $+0.0\text{s}$, the pole stretches back to $8.0\text{m}$, and the barn contracts to $2.63\text{m}$!
3. **`[3]` Terrell–Penrose Rotation & Penrose Sphere Street**
   * High-speed clock cubes ($0.85c$) fly down a colonnade of stationary reference clocks alongside a relativistic sphere ($0.86c$), an oscillating clock, and an orbiting clock. Demonstrates how light-travel delay rotates the trailing face of a cube into view while a sphere's apparent silhouette remains strictly circular.
4. **`[4]` Einstein's Train & Platform Synchronization (Relativity of Simultaneity)**
   * Bottom row: 7 stationary platform clocks synchronized in the lab frame.
   * Top row: A 7-wagon train cruising at $0.80c$ ($\gamma = 1.667$), synchronized in the train's comoving rest frame. Press `G` + `V` to jump between the platform's and the train's hyperplanes of simultaneity and watch which row of clocks is synchronized vs. desynchronized!
5. **`[5]` Relativistic Carousel & Transverse Time Dilation**
   * A central stationary master clock surrounded by 6 clocks orbiting at $0.80c$ ($\gamma = 1.667$), illustrating transverse time dilation and Fermi–Walker / Thomas precession ($\Delta\theta_{\text{Thomas}} = 2\pi(\gamma - 1)$ per orbit).

### Causal Dynamic Mode (in progress)

6. **`[6]` Relativistic Billiards** — the first *causally simulated* scene (see [`docs/DYNAMIC_MODE_PLAN.md`](docs/DYNAMIC_MODE_PLAN.md)). Nothing is precomputed: every puck's history is simulated only up to your **future light cone**, the boundary of what you can no longer influence, so far pucks are known far into the lab-frame future and near ones barely past "now". Everything you normally see is already decided; switch to **Frontier** or **Prophecy** (`G`) to watch the world being written, and **Decision Age** (`M`) to see when each event became fixed. The camera is the ship here, so rewinding and coordinate scrubbing are disabled.
   * **Play:** fire (`Enter`) elastic pucks, sticky pucks or spores (cycle with `\`) along your view direction at 0.6c in your rest frame, or fly into the pucks — the ship is a heavy puck too, and getting rammed kicks your camera. Collisions conserve 4-momentum exactly, and they are **causal**: a puck only changes its motion once news of the contact has crossed all of it, so pucks briefly squish into each other and a push can't run down a line of touching pucks faster than light (there are no rigid bodies in SR). Sticky pucks stick and glow, heavier than before — the kinetic energy lost in the collision became **rest mass**; spores carry a clock face and pop into two fragments when their own clock runs out, so fast spores travel γ times further first.

### Two-player causal multiplayer

Two players each fly a ship over the billiards table, and both see the same causally simulated world, each through their own past light cone (see [`docs/MULTIPLAYER.md`](docs/MULTIPLAYER.md)). An event is decided only once it has left **both** players' future light cones. Your past cone can be drawn only while you stay out of the future cone of the last event you know of the other player. Fly away fast and come back, and you have aged less but covered more lab time than a player who stayed put (the twin paradox, played for real). The game then holds you with **WAITING FOR THE OTHER PLAYER** until the other player catches up in your common centre-of-momentum frame. The other ship is drawn as a clock cube, so you can read their proper time. Shots, flashes and collisions are shared, and both players' replicas of the world agree bit for bit.

* **Desktop:** `cargo run --release -- --host [PORT]` (player 1; port 7777 by default) and `cargo run --release -- --join HOST:PORT` (player 2).
* **Desktop, peer to peer through NATs:** both players run `cargo run --release -- --p2p`, then paste each other's code (each player's public address comes from STUN). To skip the pasting, someone with a public IP runs `cargo run --release -- --rendezvous`, and both players run `--p2p ROOM@SERVER` with the same room name.
* **Browser:** open `index.html?mp=0` and `index.html?mp=1` in two tabs of the same browser, or play on two computers peer to peer over WebRTC: one player opens `index.html?p2p=host`, the other opens `index.html?p2p=join`, and they swap the invite and reply codes. To skip the codes, run the lobby server `node server/signal.mjs --static web`: one player presses **Create online lobby** and sends the link it shows, and the match starts (paused until then) when the other player opens it. A TURN relay on the same server covers networks with no direct path. See [`docs/HOSTING.md`](docs/HOSTING.md) to host it on a VPS and [`docs/MULTIPLAYER.md`](docs/MULTIPLAYER.md) for how it works.

---

## Running

Make sure you have [Rust](https://rustup.rs/) installed, then build and run in release mode:

```bash
cargo run --release
```

To force the CPU reference renderer instead of the GPU backend:

```bash
cargo run --release -- --cpu
```

To run the automated smoke test benchmarking both CPU and GPU backends across all 5 scenes:

```bash
cargo run --release -- --smoke-test
```

To run all unit and GPU-vs-CPU image-diff validation tests:

```bash
cargo test --release
```

### Running in a Browser (WASM + WebGPU)

The entire 4D spacetime engine and WGSL compute shader can also be compiled directly to WebAssembly (`wasm32-unknown-unknown`) with **zero external build tools required**:

```bash
rustup target add wasm32-unknown-unknown
cargo build --release --target wasm32-unknown-unknown
cp target/wasm32-unknown-unknown/release/spec-rel.wasm web/spec-rel.wasm
```

Then serve the `web/` directory with any static HTTP server (for example using Node.js or Python) and open `http://localhost:8080` in a WebGPU-capable browser (Chrome, Edge, Firefox, or Safari):

```bash
python -m http.server 8080 --directory web
```

On touch screens, the **Thrust** and **Teleport** button rows under the viewport stand in for the held `W A S D Space C` and `I J K L U O` keys.

### Companion 2D demo: causal billiards

[`billiards2d/index.html`](billiards2d/index.html) is a standalone, dependency-free JavaScript page (no WASM, no build step: just open it) that runs the same causal collision model in 2D, top-down in the centre-of-momentum frame: offset two-ball hits, Newton's cradle, a billiard break, a converging ring and a periodic relativistic gas. It tracks the energy split (rest mass, heat as rest-mass gain, kinetic, impulses in transit) and total 4-momentum live, conserved to rounding error.

---

## Controls

| Key | Action |
| :--- | :--- |
| **`F`** | Toggle between **GPU Compute** (defaults to 2X 2560x1600 on desktop, 1X 1280x800 on web) and **CPU Reference** (1X 1280x800) backend |
| **`Z`** / **`Y`** | Toggle resolution scale: **1X (1280x800)** vs. **2X (2560x1600)** |
| **`1` – `6`** / **`N`**, **`B`** | Switch between the 5 Special Relativity demo scenes and the Dynamic Mode billiards scene |
| **`W` `A` `S` `D`** / **`Space`** / **`C`** | Fire proper-acceleration thrusters (Lorentz boost in local frame) |
| **`V`** | Toggle velocity match with the scene's moving object (Pole / Train / Traveler) |
| **`X`** | Brake immediately to the Lab rest frame ($\boldsymbol{\beta} = 0$) |
| **`Arrow Keys`** / **`Q` `E`** | Rotate local camera look direction (Yaw / Pitch / Roll) |
| **`G`** | Cycle Spacetime View: **Real Camera ($ds^2 = 0$)** $\to$ **God's Eye ($dt' = 0$)** $\to$ **Frontier (future cone)** $\to$ **Prophecy ($dt' = +\Delta$)** |
| **`PageUp`** / **`PageDown`** | Push the Prophecy slice further into / back from the future ($\Delta$ = 0.5 .. 20 s) |
| **`/`** | Emit a spherical light flash from the camera |
| **`Enter`** / **`\`** | Dynamic Mode: fire the current ammo / cycle ammo (elastic puck, sticky puck, spore) |
| **`H`** | Cycle Shadow Mode: **Lorentz-Invariant (Null)** $\to$ **Camera-Slice Faked** $\to$ **Off** |
| **`M`** | Cycle Render Mode: **Shaded** $\to$ **Lookback Isochrones** $\to$ **4D Tetrahedral Mesh** $\to$ **Causal Decision Age** |
| **`P`** | Pause / Resume observer worldline evolution |
| **`T`** / **`Backspace`** | Toggle **Rewind** (reverse time playback direction) |
| **`-`** / **`=`** | Decrease / Increase time playback speed (`-4x` .. `+4x`) |
| **`[`** / **`]`** | Scrub coordinate time $t$ backward / forward while keeping 3D position $(x, y, z)$ fixed |
| **`I` `J` `K` `L`** / **`U` `O`** | Scrub 3D spatial position $(x, y, z)$ while keeping coordinate time $t$ fixed |
| **`R`** | Reset camera to initial scene event and rest frame |
| **`Tab`** | Show / Hide HUD overlay |
| **`Esc`** | Quit |

---

## Architecture & Recommended Reading Order

If you are reading the code to learn how Special Relativity is implemented geometrically, follow this order:

1. [`src/minkowski.rs`](src/minkowski.rs) — 4D Minkowski vector algebra (`Vec4`), metric $\eta_{\mu\nu} = \text{diag}(-1, 1, 1, 1)$, rest-frame spatial projector (`project_to_rest_space`, `rest_frame_cos`), 4D covector cross product (`cross4`), planar Lorentz boosts, and orthonormal `Tetrad` (`world_to_local`, `local_to_world`, `boost_to`, and stepwise `fermi_walker_transport`).
2. [`src/mesh4d.rs`](src/mesh4d.rs) — 3D rest-frame primitives (`Mesh3D`), `f64` worldline integration (`sample_accelerated_worldline`), Born-rigid worldtube extrusion with local proper-time gradients (`build_worldtube_simplices`), and affine ray–tetrahedron intersection (`SimplexGeom4D`, `Simplex4D`).
3. [`src/bvh4d.rs`](src/bvh4d.rs) — 4D Bounding Volume Hierarchy (`Bvh4D`) with 8-bin 4D SAH construction and branchless slab traversal.
4. [`src/renderer.rs`](src/renderer.rs) — Multithreaded (`rayon`) 4D raytracer reference renderer, rest-frame Lambertian shading (`compute_illumination`), procedural proper-time analog/digital clock and $T = 2L/c$ Einstein light-clock shaders, and 4D shadow modes.
5. [`src/gpu.rs`](src/gpu.rs) — `wgpu` GPU compute backend and target-agnostic GPU buffer packing (`PackedSceneGpuData`, `GpuUniforms`), managing static storage buffer uploads, uniform blocks, pipeline dispatch, readback, and CPU-vs-GPU validation tests.
6. [`src/shaders/render.wgsl`](src/shaders/render.wgsl) — High-performance WGSL compute shader executing full 4D BVH traversal, intersection, procedural clock dials, and lighting on the GPU.
7. [`src/scene.rs`](src/scene.rs) — The 5 Special Relativity demo scenes with multi-iteration periodic worldtubes (`for_periodic_copies`) and physics verification tests.
   - [`src/dynamic/`](src/dynamic/mod.rs) — Causal Dynamic Mode: bodies with `f64` histories extended exactly up to the camera's future light cone (`DynWorld`, the frontier rule `is_decided`), and the billiards scene.
   - [`src/multiplayer/`](src/multiplayer/mod.rs) — Two-player matches: wire protocol, links (TCP, UDP peer to peer with NAT hole punching, in-process, browser queue), STUN and rendezvous, handshake/replay (`Session`), CM-frame pacing (`Pacer`) and the App-facing `Match`.
8. [`src/hud.rs`](src/hud.rs) — Software 5x7 bitmap font renderer for HUD, GPU HUD overlay buffer, and in-scene digital clock readouts.
9. [`src/app.rs`](src/app.rs) — Frontend-independent app state shared by desktop and web: scene selection, observer camera, relativistic flight controls (`Action` / `HeldInput`), time controls and per-frame dynamics.
10. [`src/main.rs`](src/main.rs) — Native desktop window loop (`minifb`): key bindings and GPU/CPU backend switching.
11. [`src/web.rs`](src/web.rs) & [`web/index.html`](web/index.html) — Zero-copy WebAssembly (`wasm32-unknown-unknown`) bindings and browser WebGPU + Canvas2D frontend.
