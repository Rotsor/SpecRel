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
* **Interactive Spacetime Slicing & Shadow Modes**:
  * Toggle between **Past Light-Cone ($ds^2 = 0$)** (what a physical camera sees) and **Camera Simultaneity Slice ($dt' = 0$)** (an instantaneous 3D slice across the observer's current Lorentz frame $\langle X - O, u_{\text{cam}} \rangle_\eta = 0$).
  * Cycle between **Lorentz-Invariant Null Shadows** ($t_{\text{emit}} = t_{\text{hit}} - \|\mathbf{x}_L - \mathbf{x}_{\text{hit}}\|$), **Camera-Slice Faked Shadows** ($\langle X_{\text{emit}} - X_{\text{hit}}, u_{\text{cam}} \rangle_\eta = 0$), and **Shadows Off**.

---

## 5 Built-In Paradox & Thought-Experiment Scenes

Switch scenes at any time with keys **`1`–`5`** or **`N` / `B`**:

1. **`[1]` Twin Paradox (Stay-at-Home vs. Accelerated Traveler)**
   * Two twins start synchronized ($\tau = 0$) at $t = -14\text{s}$. The right twin executes a smooth relativistic round-trip peaking near $0.94c$ ($\gamma \approx 3$) and parks back next to the stay-at-home twin at $t = +14\text{s}$ with an **$\sim 8\text{s}$ proper-time gap** visible on both their analog/digital dials and their comoving Einstein light clocks. During the high-speed cruise, the traveler's front and rear clock faces also display the Born-rigid proper-time offset ($\Delta\tau = 2 p\,\text{atanh}(\beta) \approx 2\text{s}$) before resynchronizing upon return.
2. **`[2]` Barn & Pole (Ladder) Paradox ($8\text{m}$ Pole in $5\text{m}$ Barn at $0.85c$)**
   * An $8\text{m}$ rest-length pole (tiled with 8 synchronized comoving clocks) flies at $0.85c$ ($\gamma \approx 1.90$) through a $5\text{m}$ barn.
   * Pause (`P`) near $t = 0$ and switch to Camera Slice (`G`): in the barn's frame (`X`), the pole is contracted to $4.21\text{m}$ and fits completely inside the barn, with its rear end reading $\beta L_0 = +6.8\text{s}$ ahead of its front end ($+4.84\text{s}$ across a $3\text{m}$ lab separation $\gamma \beta \Delta x$). Press `V` to boost into the pole's rest frame: the pole's clocks all synchronize to $+0.0\text{s}$, the pole stretches back to $8.0\text{m}$, and the barn contracts to $2.63\text{m}$!
3. **`[3]` Terrell–Penrose Rotation & Penrose Sphere Street**
   * High-speed clock cubes ($0.85c$) fly down a colonnade of stationary reference clocks alongside a relativistic sphere ($0.86c$), an oscillating clock, and an orbiting clock. Demonstrates how light-travel delay rotates the trailing face of a cube into view while a sphere's apparent silhouette remains strictly circular.
4. **`[4]` Einstein's Train & Platform Synchronization (Relativity of Simultaneity)**
   * Bottom row: 7 stationary platform clocks synchronized in the lab frame.
   * Top row: A 7-wagon train cruising at $0.80c$ ($\gamma = 1.667$), synchronized in the train's comoving rest frame. Press `G` + `V` to jump between the platform's and the train's hyperplanes of simultaneity and watch which row of clocks is synchronized vs. desynchronized!
5. **`[5]` Relativistic Carousel & Transverse Time Dilation**
   * A central stationary master clock surrounded by 6 clocks orbiting at $0.80c$ ($\gamma = 1.667$), illustrating transverse time dilation and Fermi–Walker / Thomas precession ($\Delta\theta_{\text{Thomas}} = 2\pi(\gamma - 1)$ per orbit).

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

---

## Controls

| Key | Action |
| :--- | :--- |
| **`F`** | Toggle between **GPU Compute** and **CPU Reference** rendering backend in real time |
| **`1` – `5`** / **`N`**, **`B`** | Switch between the 5 Special Relativity demo scenes |
| **`W` `A` `S` `D`** / **`Space`** / **`C`** | Fire proper-acceleration thrusters (Lorentz boost in local frame) |
| **`V`** | Toggle velocity match with the scene's moving object (Pole / Train / Traveler) |
| **`X`** | Brake immediately to the Lab rest frame ($\boldsymbol{\beta} = 0$) |
| **`Arrow Keys`** / **`Q` `E`** | Rotate local camera look direction (Yaw / Pitch / Roll) |
| **`G`** | Toggle Spacetime Slice: **Past Light-Cone ($ds^2 = 0$)** vs. **Camera Slice ($dt' = 0$)** |
| **`H`** | Cycle Shadow Mode: **Lorentz-Invariant (Null)** $\to$ **Camera-Slice Faked** $\to$ **Off** |
| **`M`** | Cycle Render Mode: **Shaded** $\to$ **Lookback Isochrones** $\to$ **4D Tetrahedral Mesh** |
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
5. [`src/gpu.rs`](src/gpu.rs) — `wgpu` GPU compute backend managing static storage buffer uploads, uniform blocks, pipeline dispatch, readback, and CPU-vs-GPU validation tests.
6. [`src/shaders/render.wgsl`](src/shaders/render.wgsl) — High-performance WGSL compute shader executing full 4D BVH traversal, intersection, procedural clock dials, and lighting on the GPU.
7. [`src/scene.rs`](src/scene.rs) — The 5 Special Relativity demo scenes with multi-iteration periodic worldtubes (`for_periodic_copies`) and physics verification tests.
8. [`src/hud.rs`](src/hud.rs) — Software 5x7 bitmap font renderer for HUD and in-scene digital clock readouts.
9. [`src/main.rs`](src/main.rs) — Interactive window loop (`minifb`), dynamic GPU/CPU backend switching, relativistic flight controls, and HUD overlay.
