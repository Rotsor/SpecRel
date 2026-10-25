# X stops braking in a match

**Symptom:** in a fresh room `X` brakes. Later in the same match it stops working: `OBS 3-VEL`
stays non-zero and doesn't change while `X` is held.

**Status:** not fixed yet (another change in flight would conflict). This note lists the likely
causes, how to tell them apart, and the fix for each. Cause 0 matches how it was found: both
players in two browser windows side by side, switching between them while trying to brake both. In a match, `X` is no longer an instant
boost. It sets a steering target (`App::steer`), and `App::step` then boosts toward it by at most
`SHIP_ACCEL · d_tau` per frame, while the key is held. Every frame in which the ship takes no
proper-time step, or in which its step is undone, leaves the velocity exactly as it was. Two
states do that.

## 0. X held while switching windows (most likely)

`X` only starts steering on a *fresh* press: `Action::Brake` sets `App::steer`, and the held bit
(`HeldInput::brake`) merely keeps it going. If the key is released, `steer` is dropped, and
holding the key again without a new press does nothing (by design; see "held again without a new
press: still coasting" in `match_brake_is_gradual_and_stops_on_release`).

With two windows, the natural way to brake both is to hold `X` and click from one window to the
other:

1. Window A loses focus: `web_blur` clears every held bit, so A's `steer` is dropped on its next
   frame. A stops braking, and pressing in A again later works.
2. Window B gains focus while `X` is already down. The browser sends B only auto-repeat
   `keydown`s (`e.repeat = true`). `web_key_event` sets the held bit for those but applies the
   `Action` only when `repeat == 0`, so B never gets `Action::Brake` and never starts steering.
   `OBS 3-VEL` stays put for as long as `X` is held.

The old instant brake was edge-triggered too (a `DISCRETE_KEY_MAP` key, repeats ignored), so it
had the same blind spot. But a single press used to finish the job, so it rarely showed.

**How to tell:** nothing in the HUD. Release `X` and press it again in the focused window: it
brakes.

**Fix options:**
* Simplest: drive `X` from the held state alone. Whenever `input.brake` is down, steer toward
  lab rest, whether or not a press was seen. `X` has a fixed target, so it needs no press
  semantics (unlike `V`, which picks its target when pressed). Then drop the "held again without
  a new press" assertion.
* Or in `web_key_event`, treat a repeat `keydown` for a held code whose bit isn't set yet (bits
  are cleared on blur) as a fresh press.

## 1. The player is paused (my change caused it)

`App::step` computes

```rust
let d_tau = if self.paused { 0.0 } else { dt_wall * self.time_scale() };
let max_boost = SHIP_ACCEL as f32 * d_tau;
```

So while paused (`P`, or the `[P] Pause` button right next to `[X] Brake` on the web page),
`max_boost` is 0 and `X`, `V` and the thrusters do nothing. That's deliberate: a boost with no
proper time passing is infinite acceleration. `match_thrust_respects_the_rindler_bound` even
asserts it. But it's a regression in feel. Before, and still in single-player, you could pause
and then brake. A player who pauses to look around and presses `X` sees exactly this symptom.

**How to tell:** the HUD's `TIME:` field reads `PAUSED`, and `OBS TAU` is frozen.

**Fix options:**
* Let a paused ship's controls queue up and apply over the first proper time after unpausing.
* Disable `P` in a match. A paused player holds the other player back anyway, see cause 2.
* At least say why: e.g. `PAUSED: UNPAUSE TO STEER` when a control is used while paused.

## 2. The pacing rule is holding the ship back

When the tentative next event would come within `CONE_MARGIN` of the other player's future cone,
`Match::may_advance` returns `false`, and `App::step` restores the camera to `looked`. That undoes
this frame's thrust and steering as well as the motion. The `Pacer` then stays stopped until the
other player has caught up in the two ships' centre-of-momentum (CM) frame. The steering target
survives, so braking resumes once the ship is released. But nothing changes until then.

This can last indefinitely when the other player isn't advancing:
* **They are paused** (cause 1, on their side).
* **Their tab is in the background.** Browsers throttle or stop `requestAnimationFrame` in hidden
  tabs, so testing both players in two tabs of one browser triggers it.
* **They have a slow time scale** (`-`), which only slows things down.

There's also a way braking can trigger the stop where coasting wouldn't. The CM frame is built
from both ships' 4-velocities. A brake step changes ours, which changes the frame and the lead
`dt`. So the braked tentative event can pass `must_wait` when coasting wouldn't have. The ship
then stops on the frame you press `X`. It's harmless if the other player keeps playing, but
combined with the bullet list above it looks like "X stopped working".

**How to tell:** the banner `WAITING FOR THE OTHER PLAYER`, and `OBS EVENT (T, ...)` and
`OBS TAU` are frozen. Thrust doesn't work either.

**Fix options:** mostly beyond this bug: the banner is already there. Possibly also:
* Name the reason ("other player paused" / "tab inactive"). The `Track { waiting }` heartbeat
  could carry a `paused` flag.
* Evaluate `must_wait` on the coasted event rather than the boosted one. Then a velocity change
  alone can't start a stop. This needs a proof that the cone guarantee still holds, because the
  boosted event is the one actually reached.

## 3. Less likely, web only

* **Keyboard focus in a text field.** (Clicking into a window on the lobby link or the P2P box
  rather than the canvas would do it.) The `keydown` handler ignores keys while a `textarea` or
  `input` has focus (the P2P code box, the lobby name). Then nothing responds, not only `X`.

## Checked and not the cause

A probe test (not committed) ran a two-app match at 60 fps. Each player reached `|v| ≈ 0.94`
with diagonal thrust and while turning, then held `X`. Both players braked to exactly 0 with no
waiting frames, including player 2, whose camera spawns turned around. So
`Tetrad::boost_toward`, the steering bookkeeping and the frontend key wiring work as designed
when the ship is advancing.

## Suggested regression tests

* `X` held with no `Action::Brake` this match (focus arrived mid-hold): assert the ship brakes
  (with the first fix for cause 0).
* Paused, then `X` held: decide the intended behaviour (cause 1) and assert it.
* Player A coasting fast while player B is idle, *not* stepping (models a background tab): assert
  A shows the waiting banner rather than silently ignoring `X`.
