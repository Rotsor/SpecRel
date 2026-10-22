# Causal billiards in the Rust Dynamic Mode: review and recommendations

Status: review of `claude/dynamic-mode` at `ec05a62` (2026-10-22). Nothing here is implemented yet.

## Where the model stands

`src/dynamic` implements the "contact event + delayed impulse" approach: a contact is solved
with exact two-body kinematics in the pair's centre-of-momentum frame (`collide::elastic`,
`collide::stick`), and each body only feels its impulse once light from the contact event has
crossed all of it (`Impulse::has_reached`). Impulses still in flight are the body's internal
"in transit" momentum, so totals are conserved once everything has arrived.

This is about as far as a rigid-until-informed model can go. A truly rigid, instantly
responding, exactly conserving relativistic ball is impossible (Born rigidity /
Herglotz–Noether; spacelike-separated momentum exchange); the remaining step beyond this model
is a deformable relativistic continuum.

## Measurements

A temporary probe (not committed) fired a flat shot from the open near side at the rack's front
puck, with the camera parked in the lab frame, and compared totals after all impulses arrived.

| Rack | Shot speed | Energy error | Rest masses | Notes |
| :--- | :--- | :--- | :--- | :--- |
| Default (gap ≈ 1.1) | 0.3–0.95c | ~1e-16 | unchanged | max overlap 0.9 of the contact distance at 0.95c |
| Near-touching (gap 0.02) | 0.3–0.6c | 3e-4 – 1e-3 | elastic pucks lost up to 19% | |
| Near-touching (gap 0.02) | ≥ 0.8c | — | — | a 4-momentum went spacelike → NaN velocity; world froze at t ≈ 10 |

The freeze: the NaN state can never satisfy `is_decided`, so that body never advances, its
neighbours wait for it in `contacts()`, and the whole world stalls while the camera moves on.

## Recommendations (most important first)

### 1. Port the 2D demo's contact rules (`billiards2d/index.html`)

The Rust `contact_impulse` solves a new collision from the body's *felt* velocity. When a puck
is hit again before it has felt an earlier hit (dense racks, pile-ups), this mixes inconsistent
states: elastic contacts change rest mass and can produce spacelike 4-momenta.

- Solve new contacts with the **effective** 4-momentum: felt plus still-pending impulses.
- Record each impulse's **dependencies** (the contact events it builds on) and feel impulses in
  order, only once every dependency's light front has crossed the body.
- Store exact 4-momenta per body rather than (mass, velocity) pairs re-derived per impulse.
- Allow **re-contact**: Rust acts once per touching episode (`touching.contains`), so pucks
  pushed together again by a third body while still overlapping can pass through each other.
  Re-contact once a pair has no pending impulses and is still closing.
- Add a guard: `debug_assert!` (and a counter in release) whenever `E² ≤ |p|²` would result,
  instead of letting NaN silently stall the scheduler.

### 2. Make the rails causal

`billiards::constrain` mirrors position and reverses velocity instantly, so the far side of a
puck turns around before light from the rail could reach it. Route rail contacts through the
same pending-impulse path, treating the rail as an infinite-mass partner (impulse
`Δp = −2 (p·n) n` in the rail frame, energy unchanged), placed at the touching event.

### 3. Remove lab-frame dependence

- Contact detection uses the lab-time grid and the *uncontracted* rest radius, while pucks are
  rendered Lorentz-contracted. Detect with the contracted shape (or equivalently in the pair's
  CM frame).
- Rust takes the contact at the overlapped grid position; the 2D demo solves for the exact
  first-touch time within the step. Do the same.
- The contact normal comes from the lab-simultaneous separation; use the separation at equal
  CM time.
- Evaluate "news has crossed the body" in the body's rest frame (the current
  `distance + r <= Δt` lab test is conservative but frame-dependent).
- Add a **covariance test**: run the same setup described in a boosted frame and compare final
  invariants (rest masses, pairwise `p_a · p_b`, scattering angles in the CM frame).

### 4. Show the overlap as compression

At 0.95c pucks interpenetrate by up to 90% of the contact distance while they wait for the news.
That is not a bug: it is the cost of "rigid until fully informed", and a real ball would be
compressed by about that much. Options: render it as a deformation that follows the light front
inside the puck, or keep the overlap and label it in the HUD as compression.

### 5. Track angular momentum and the centre of energy

Delayed impulses can violate the boost-charge part of angular-momentum conservation
(`K = tP − E·X_cm`) unless in-transit momentum is given a location (e.g. on its light front).
Untested so far. Add `L`, `K` and centre-of-energy charts alongside the existing energy and
momentum ones in the 2D demo, and assertions in Rust.

## Beyond this model

The remaining artifacts, overlap and the instant velocity switch once the light front has
crossed, are intrinsic to a rigid-until-informed body. Removing them means modelling each puck as
a relativistic elastic continuum (or a particle mesh with retarded, ≤ c interactions): causal and
locally conserving by construction, at the cost of vibrating pucks and slightly inelastic
collisions, which is the physically correct outcome.
