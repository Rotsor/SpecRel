# Ship controls are still faster than light

**Status:** open. This note describes the problem and the options for fixing it. Nothing here is
implemented yet.

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

## What doesn't help

* **Moving the control point.** The centre already minimises the distance to the farthest part.
* **Making the ship smaller without a delay.** This shrinks the violation but doesn't remove it.
  The hull still reacts on a spacelike slice.
* **Predicting the input.** Input is live by definition.

## Smaller, related gaps

* Ships are drawn with `phi = 0` and `proper_accel_local = 0` (`ShipSample::sample`), so the
  other player's clock faces don't show the Born-rigid proper-time gradient while that player
  accelerates. Option A makes it straightforward to fill these in from the executed commands.
