# SpecRel 4D: Causal Multiplayer

Two players fly ships through the same Dynamic Mode world (scene 6, relativistic billiards). Each
player sees only their own past light cone. The world stays causal and is identical for both of
them. This note covers the physics rule, the architecture, and the network protocol.

## 1. The rule

In single-player Dynamic Mode an event is *decided* once it has left the open future light cone
`I⁺(O)` of the camera event `O` (see [`DYNAMIC_MODE_PLAN.md`](DYNAMIC_MODE_PLAN.md)). With
several players, every player is a source of input, so an event is decided only once it has left
**every** player's future cone:

```
decided(E)  :=  E ∉ I⁺(O₁) ∪ I⁺(O₂)
```

Each peer runs its own replica of the world and knows its own event `O_local` exactly. It knows
the other player's event only as of the last message received, `O_remote`, which lies in the past
of the remote's true event. `I⁺(O_remote)` therefore *contains* the remote's true future cone, so
using it is conservative: nothing a peer finalizes can still be changed by the other player.

**Rendering needs the other player to keep up.** To draw my past light cone `J⁻(O_local)`, every
event on it must be decided. If some `E ∈ J⁻(O_local) ∩ I⁺(O_remote)` existed, then
`O_remote ≪ E ≤ O_local`, so `O_local ∈ I⁺(O_remote)`. The converse also holds. So:

> My past cone is fully known **iff** `O_local ∉ I⁺(O_remote)`, that is, unless I have moved into
> the future light cone of the last event I know of the other player.

That is the "outpacing" in this game. It is ordinary relativity, not lag. A traveller who flies
away and comes back has aged less but has covered more coordinate time than a player who stayed
put (the twin paradox). They arrive back in the stay-at-home player's future, and have to wait for
that player.

**Pacing (`multiplayer::pacing`).** Let `U` be the two ships' centre-of-momentum 4-velocity
(`normalize(u₁ + u₂)`; the ships have equal mass). In the CM frame let `Δt` and `Δx` be the time
and distance from `O_remote` to `O_local`.

* **Stop** before a step would satisfy `Δt > 0 ∧ Δt > |Δx| − CONE_MARGIN`. That means I am ahead
  in CM time and within `CONE_MARGIN` (1.5 s) of the remote's future cone. The margin covers the
  pucks near the other ship, which are finalized a little behind it. The check runs on the
  *tentative* next event, so `O_local` never enters `I⁺(O_remote)`.
* **Resume** only once the other player has caught up in the CM frame: `Δt ≤ 0`.
* While waiting, the player can look around and change views. Thrust and firing are frozen, and
  the HUD shows **WAITING FOR THE OTHER PLAYER**.

A player with `Δt ≤ 0` never needs to stop, so at least one player can always move. Both can be
stopped only briefly, when their information about each other is stale; heartbeats resolve that.
Before the other player has connected, they are taken to be at their spawn event, at rest. This
makes the join screen an ordinary wait.

## 2. Determinism

Both replicas must agree bit for bit. Billiards is chaotic, so any drift would eventually show one
player a collision that the other never sees. Each body's next state is already a pure function of
finalized data (single-player Phase 4). On top of that, multiplayer requires the following:

* **Inputs travel exactly.** Ship samples are sent as `f64` bits. Fire events are sent with their
  `f32` firing event and `f64` velocity.
* **Inputs are applied in order.** A shot fired at `O_f` is sent before the track sample that
  moves past `O_f`. The receiver therefore inserts the projectile before its frontier for that
  ship passes `O_f`, just as the firing peer did. Bodies that could touch the projectile wait for
  that ship's track: the existing ship-proximity rule, extended to every ship.
* **Identities are per player.** Projectile uids are seeded with `(player, shot index)`.
* **Firing capacity is checked by the shooter only.** A received shot is always inserted
  (`DynWorld::launch`), because live-body counts can differ between peers at any given moment.
* **No lab-time rebase in multiplayer.** Rebasing at different moments on the two peers would
  change `f64` rounding. Physics is `f64`, so precision is fine for days. Render precision
  degrades slowly after many hours.
* Pruning keeps everything newer than the *earliest* live frontier minus the history window.
* Both peers must run the same build on the same platform (`f64` arithmetic is IEEE, but the
  `cbrt` in `puck_radius` comes from the platform's libm).

## 3. Architecture

```
src/dynamic/ship.rs        PlayerId, ShipSample (+ proper time)
src/dynamic/mod.rs         DynWorld: N ships, union-of-cones frontier, per-ship tracks, kicks,
                           and other ships drawn as clock cubes
src/multiplayer/protocol.rs Message + compact binary encoding
src/multiplayer/pacing.rs   CM-frame lead, stop/resume rule (Pacer)
src/multiplayer/link.rs     Link trait; LocalLink (in-process pair); QueueLink (frontend-driven)
src/multiplayer/tcp.rs      native TcpLink (--host / --join), u32-length-prefixed frames
src/multiplayer/session.rs  Session: Hello handshake, sequencing, log replay, events
src/app.rs                  App::new_multiplayer; applies remote events; paces the local ship
src/main.rs, src/web.rs, web/index.html   transports, CLI/URL, "waiting" banner
```

### Session protocol

Every frame is `seq: u32 LE` followed by one encoded `Message`:

| Message | Payload |
|---|---|
| `Hello` | protocol version, sender's `PlayerId` |
| `Track` | `ShipSample` (t, pos, vel, τ as `f64`) and the sender's waiting flag; sent every frame, including while waiting, as a heartbeat |
| `Fire` | firing event (`Frontier`, `f32`), lab velocity (`f64` ×3), ammo |
| `Flash` | emission event (`f32` ×4) |
| `Bye` | sender quits |

Links are reliable and ordered once open (TCP, BroadcastChannel, WebSocket), but frames sent
before the peer was listening are lost. A session therefore keeps a log of everything it has sent
until the peer's `Hello` arrives. It then replays the whole log, and the receiver drops sequence
numbers it has already seen. When the peer leaves (`Bye`, or the link closes), its ship stops
being an input source: its track ends and its cone is removed from the frontier.

### Transports

* **Native:** `cargo run --release -- --host 7777` and `cargo run --release -- --join 127.0.0.1:7777`.
* **Web:** `index.html?mp=0` and `index.html?mp=1` in two tabs of the same browser, connected
  through a `BroadcastChannel`. JS ferries opaque frames to and from WASM
  (`web_net_*` exports).

The host is player 0 (the usual spawn point). The joiner is player 1, mirrored to the far side of
the table and facing it. In multiplayer, scene switching and reset are disabled.

## 4. Not done yet

* Ship–ship collisions. Ships pass through each other: a collision needs both tracks at the same
  lab time, which forces lockstep at contact range.
* More than two players (the world supports N ships; session and pacing are 2-player).
* Cross-machine web play (needs a WebSocket relay) and native↔web play.
* Coordinated lab-time rebase for very long sessions.
