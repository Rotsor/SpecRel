# Ship controls are still faster than light

**Status:** options A and C are implemented (`src/dynamic/helm.rs`); the player switches between
them with `;`. See [What was built](#what-was-built) at the end. Option B is not implemented.

## The problem

The player's input acts at the ship's centre (the camera). The ship is a rigid body of radius
`R = SHIP_RADIUS`, and every part of it follows the centre's track at once:

* `App::step` boosts the camera tetrad in the same frame as the key press.
* `Ship::sample` / `DynWorld::build_scene` draw the ship around that track as a rigid cube.
* `contacts` puts the hull on a sphere of radius `R` around the centre.

So a key pressed at centre event `C` changes the motion of a hull point `R` away *on `C`'s own
simultaneity slice*. That point is spacelike-separated from `C`. The control signal crosses the
ship instantly.

The Rindler cap from the earlier commit (`SHIP_ACCEL = 0.8 / R`) doesn't fix this. It makes the
rigid motion *possible*: the rear of the hull stays in front of the horizon. But a Born-rigid body
can only accelerate if every part runs its share of the acceleration program at the same moment of
the body's own time. With live input, the parts can't know the program yet.

Only the controls have this problem. Collision kicks already follow the causal rule:
`Impulse::has_reached` changes a body's motion only once news of the contact has reached *every*
part of it, and `App::step_world` applies ship kicks the same way.

## Option A (recommended): a command delay, the same rule as kicks

Treat each control input like a kick. Broadcast the command from the centre at light speed, and
have the whole ship execute it rigidly once it has reached every part. Each part then runs its
Born-rigid share (`a / (1 + a·p)`) at the same ship time. The motion stays rigid and causal, and
the existing rigid rendering and contact code become correct as they stand.

The delay has to be the light-travel time from the centre to the farthest hull point, measured in
the ship's proper time while it accelerates. In Rindler coordinates (`ds² = −(1 + a x)² dτ² +
dx²`), with constant full acceleration pointing away from the point (the worst case):

```
rear:  D = ln(1 / (1 − aR)) / a      front: ln(1 + aR) / a      (→ R as aR → 0)
```

The rear is the slow side because it sits closer to the horizon. Here the horizon matters
twice: it limits how hard the ship can push, and it sets how long commands take to reach the
rear. The cost is input lag. The camera is part of the ship, so it waits too:

| R (`SHIP_RADIUS`) | aR | a (c/s) | lag D (s) |
| ---: | ---: | ---: | ---: |
| 0.40 (now) | 0.8 | 2.0 | 0.80 |
| 0.40 | 0.5 | 1.25 | 0.55 |
| 0.25 | 0.5 | 2.0 | 0.35 |
| 0.20 | 0.4 | 2.0 | 0.26 |
| 0.15 | 0.3 | 2.0 | 0.18 |
| 0.10 | 0.2 | 2.0 | 0.11 |

At a fixed acceleration `a`, the lag shrinks with the ship. The 80%-of-bound choice is the worst
option for lag. A 0.15–0.2 ship at `a = 2` gives about 0.2 s of lag, with the same acceleration
as now.

**Implementation sketch (small):**
* In `App::step`, push each frame's command (`d_tau`, thrust direction, or steering target) onto
  a queue stamped with the ship's proper time. Execute the commands whose stamp is at least `D`
  old.
* The executed boost uses the ship's velocity *at execution time*. The velocity is fixed by
  earlier commands, so every part can compute it.
* Keep the queue out of the multiplayer revert: when `may_advance` fails and the camera is put
  back to `looked`, don't consume the queued commands.
* Optional, for the HUD: show the pending thrust, so the lag reads as "engines spooling up"
  rather than as unresponsive controls.
* `D` is a sufficient delay for any acceleration program bounded by `a`, because constant full
  acceleration away from the point delays light the most. A test can check this by sending
  light from the centre along recorded tracks.
* Revert `Ship::seen_from` (commit `4b4a24b`). Once the hull really moves causally, contacts
  should use the actual rigid hull again.

## Option B: a non-rigid ship with zero lag

Each part runs its Born-rigid share of a command once the command reaches it (each part on its own
delay). The camera then responds instantly. But during every change of thrust, the ship deforms
by roughly `a·R²`, and its shape relaxes back afterwards. This needs per-vertex worldlines for
rendering (`build_worldtube_simplices` assumes a rigid extrusion), a hull shape for contacts that
isn't a sphere, and a rule for how the hull relaxes. That isn't a small change.

## Option C: a control pod with no lag inside a delayed hull

The pilot sits in a small control pod that can move inside the hull. The camera is the pod, and
it accelerates at once. The pod pushes off the ship's inner structure, and that push spreads
outward no faster than light, so the push itself is the command signal. The hull is rigid and
fires its engines once the signal has reached all of it. From outside this is Option A: a rigid
hull obeying causal commands. In the cockpit the controls have no lag. The pod is tiny, so the
Rindler limit barely constrains it.

**The hull can't replay the pod's manoeuvres late.** If the hull flew the pod's exact
acceleration history `D` later, its path would be the pod's path shifted by `D` in time in the
original rest frame (Bell's spaceship paradox). Seen from the ship's current frame, that shift
includes a distance of `D·γv` between the hull's centre and the pod. At `D = 0.8`, `v = 0.6c`
that is 0.6, more than the hull radius of 0.4, so the pilot would leave through the front of the
ship. The hull has to keep station on the pod instead: steer toward where the pod now is, judging
from information `D` old, with spare acceleration beyond the pod's own.

**How hard the pod can push.** Suppose the pod accelerates at `a_c` and the hull at up to `a_h`.
The pod drifts furthest from the centre when the hull reacts `D` late and then burns harder to
stop the relative motion. A rough non-relativistic estimate, ignoring the extra overshoot from
the hull's own delay (so leave a margin):

```
max offset ≈ ½ · a_c · D² · a_h / (a_h − a_c)
```

If the pod may use half the hull's radius:

| hull R | hull a_h (fraction of the limit) | delay D | lag-free pod accel a_c |
| ---: | ---: | ---: | ---: |
| 0.4 (now) | 2.0 (0.8) | 0.80 s | ≈ 0.5 c/s |
| 0.1 | 8.0 (0.8) | 0.20 s | ≈ 1.9 c/s |
| 0.1 | 3.0 (0.3) | 0.12 s | ≈ 2.1 c/s |

The allowed `a_c` scales like `1/R`, the same as the rigidity limit. Keeping today's 2 c/s with
no lag needs a ship of about R = 0.1. At that size Option A lags only about 0.11 s. So Option C
saves about 0.1 s of lag, which is a few frames. At today's size it would cost three quarters of
the acceleration.

**Implementation sketch (fairly contained):**
* **Hull track:** a station-keeping controller, capped at `a_h`, that sees the pod's track only
  after delay `D`. The hull's track is a fixed function of the pod's track, so every replica
  computes it locally and the network protocol doesn't change. `D` must cover the distance to the
  far wall plus the pod's offset (`R + s`), using the formula in Option A.
* **Contacts and drawing:** pucks collide with the hull track, and the other player's ship is
  drawn from it, not from the pod's track. This replaces `Ship::seen_from`.
* **The pod hitting the wall:** a pod that outruns the hull hits the inner wall and gets kicked.
  That's a local, causal hull–pod contact, and it plays like hitting your own bulkhead.
* **Shots and flashes** still leave from the pod, which is where the pilot is.
* **Frontier:** the hull's motion at an event depends only on pod events in that event's causal
  past, so the "outside every player's future cone" rule needs no change.

**Verdict:** the most elegant of the three options, and physically sound. But it only feels
clearly better than Option A for small ships, and there Option A's lag nearly disappears anyway,
for much less code.

## What doesn't help

* **Moving the control point.** The centre already minimises the distance to the farthest part.
* **Making the ship smaller without a delay.** This shrinks the violation but doesn't remove it.
  The hull still reacts on a spacelike slice.
* **Predicting the input.** Input is live by definition.

## Smaller, related gaps

* Ships are drawn with `phi = 0` and `proper_accel_local = 0` (`ShipSample::sample`), so the
  other player's clock faces don't show the Born-rigid proper-time gradient while that player
  accelerates. Option A makes it straightforward to fill these in from the executed commands.

## What was built

Both options run in every Dynamic Mode game, single-player included (`ControlMode` in
`src/dynamic/helm.rs`; `;` switches it, restarting the table when alone and only before the start
in a match). `SHIP_RADIUS` and `SHIP_ACCEL` are unchanged, so the numbers are the ones for today's
ship; shrinking the ship is the lever for less lag, as the tables above show.

**Every ship has two tracks** (`Ship::track`, `Ship::hull`). The pilot's track is the camera: it
bounds the frontier, and shots and flashes leave from it. The hull centre's track, at the same lab
times, is what bodies collide with (`Ship::hull_at`). `Ship::seen_from` is gone: the hull moves
causally, so contacts use the rigid hull as it is. Kicks reach the hull once their news has
crossed it, as before.

**Option A (`ControlMode::Delayed`, the default).** The camera is the hull's centre, and the two
tracks coincide. Each frame's thrust or steering becomes an `Order` stamped with the ship's clock
and queued (`CommandQueue`), and executes `command_delay()` = 0.80 s of ship time later. A thrust
order keeps its lab-frame direction and is re-projected onto the rest frame at execution time;
steering boosts toward its target from the velocity then. When the multiplayer pacer holds the
ship back, the frame's queue changes are undone with the rest of the frame. The HUD shows
**SPOOLING** while commands are on their way.

**Option C (`ControlMode::Pod`).** The camera is the pod. It thrusts at once at `pod_accel()`
(0.35 c/s, the formula above with `POD_RANGE` = R/2 and the corrected delay below) and floats
inside the hull. The hull (`Hull`) steers with a critically damped controller (2 rad/s), capped at
`SHIP_ACCEL`, toward its best knowledge of the pod (`Hull::predict_pilot`): the pod's state on
the hull's own simultaneity slice `pod_delay()` ago, flown on with the acceleration it had then,
and bounced off the inner wall wherever the hull's own recorded track put it. The hull's own past
is known to all of it, so this uses nothing that hasn't crossed the hull. The wall replay
matters: without it the hull steered by bounces it heard of a second late, overshot, and the pod
rattled between the walls with growing amplitude. With it, bursts, reversals and coasting all
settle (`station_keeping_settles_after_any_burst`); the pod peaks about 0.2 off centre. A pod that
outruns the hull hits the inner wall at `POD_RANGE` and loses its outward speed relative to the
hull (`WALL_RESTITUTION` = 0). The hull ignores the push: it is much heavier.

Two corrections to the sketch above:

* **The pod's delay.** Light from a pod `s` ahead of the centre to the rear wall takes
  `ln((1 + a s) / (1 − a R)) / a` of the centre's time, not the Option A formula at `R + s`
  (which would put the rear past the horizon here). With `s` = 0.2 that is `pod_delay()` = 0.97 s
  (`signal_delay`, checked against the Rindler worldlines in a test).
* **The protocol changes.** Instead of each replica recomputing the hull from the pod's track (which
  would need bit-identical `f32` transcendental functions on every platform), the owner sends both
  samples in each `Track` message (protocol version 3).

**Seeing the decision centre.** Other players' ships are drawn as an open cage of bars along the
hull track (`Ship::hull_mesh`, materials 12–15 in the player's colour), with the pilot's clock cube
inside it on the pilot's track (`Ship::pilot_mesh`). With delayed controls the clock sits at the
centre; with a pod it visibly drifts toward the wall the pilot is thrusting at, and the cage
catches up about a second later.
