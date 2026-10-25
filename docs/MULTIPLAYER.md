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

* **Stop** before a step would satisfy `Δt > 0 ∧ Δt > |Δx|`, that is, before the interval to
  `O_remote` would become timelike. The check runs on the *tentative* next event, so `O_local`
  never enters `I⁺(O_remote)`. There is no margin. (An earlier version stopped 1.5 s short of the
  cone, to cover pucks near the other ship that waited for its track before colliding with it.
  Ships no longer touch bodies in a match, see below.)
* **Resume** once the other player has made up half of the CM-time lead `Δt` I had when I
  stopped (`RESUME_FRACTION`); if my next step would still be timelike then, stop again with the
  new, smaller lead. This hysteresis keeps a ship pressed against the other's cone from flickering
  between moving and waiting every frame. (Earlier, the ship waited for `Δt ≤ 0`.)
* Bodies near the other ship are still finalized up to a shot's reach behind its cone (see §2),
  so right at the cone's edge a puck close to it may be drawn a moment late.
* While waiting, the player can look around and change views. Thrust and firing are frozen, and
  the HUD shows **WAITING FOR THE OTHER PLAYER**.

**The ship is rigid.** The camera is the centre of a ship of radius `SHIP_RADIUS` (0.4). A
Born-rigid body whose centre accelerates at `a` has a Rindler horizon `1/a` behind the centre.
The hull's trailing point, at distance `R`, accelerates at `a / (1 − aR)`, and that blows up as
`aR → 1`. So in a match the ship's proper acceleration is capped at `SHIP_ACCEL = 0.8 / R`, which
is 80% of the bound. The cap is measured per second of the ship's own clock (`d_tau`, so a paused
ship cannot boost at all) and in every direction (diagonal thrust is normalized). `X` (brake to
the lab frame) and `V` (match velocity) do not jump: they boost toward the target for as long as
the key is held. Releasing the key mid-brake leaves the ship coasting at whatever velocity it
reached. (These rules hold for single-player Dynamic Mode too.)

**Ships don't touch bodies.** Getting contacts with the other player's hull right, causally and
identically on both replicas, is hard, so ships and pucks pass through each other. The players
shoot lasers instead (`src/dynamic/laser.rs`): a pulse pushes nothing, so nothing waits for it,
and it goes out at the first puck, hull or rail it meets.

**The hull hears the controls at light speed.** A rigid hull may only act on a command once news
of it has reached all of it (`docs/SHIP_CONTROLS_CAUSALITY.md`, `src/dynamic/helm.rs`). The
player picks one of two ways with `;`, before the match starts (a `Controls` message tells the
other replica):

* **Delayed controls** (option A, the default): the camera is the hull's centre, and every command
  executes `command_delay()` (0.80 s) of ship time after it was issued; the HUD says **SPOOLING**
  meanwhile.
* **Control pod** (option C): the camera is a pod that accelerates at once, at `pod_accel()`
  (0.35 c/s), and nothing holds it back; under steady thrust it floats about `POD_RANGE` (0.2) from
  the hull's centre. The hull keeps station on the pod, steering by what it learned of it
  `pod_delay()` (0.97 s) ago, flown on to now, so it never acts on news that hasn't crossed it.

Each ship therefore has two tracks: the pilot's (the camera: frontier, shots, flashes; the only
one sent in `Track` messages) and the hull centre's (what bodies collide with in single player,
and what is drawn). Every replica computes the hull's track from the pilot's as it records it
(`Ship::record`), in plain `f32` arithmetic with `sqrt` as the only function, so both replicas
get the same bits. The other player's ship is drawn as an open cage along the hull track, with
the pilot's clock cube inside it on the pilot's track, so you can see where its decisions are
made.

**The hull is drawn past the pilot's latest event.** Pacing keeps you outside `I⁺(O_remote)` of
the remote *pilot*, but the hull reaches `SHIP_RADIUS` and more from it, and your past cone meets
its near side up to that much later than its centre. A hull track that stopped at the pilot's lab
time would leave the near bars missing whenever you are within that distance of the remote's
cone, which the pacer allows, and the cut would move with every `Track`. But the hull's next
moves are already settled: with delayed controls by the orders on their way across it (each
`Track` carries those issued since the last one), and in a control pod by pilot events it has
already recorded (it steers by news `pod_delay()` old). `Ship::hull_ahead` flies the hull on
through that stretch, about a second of ship time, which covers every hull event outside
`I⁺(O_remote)`. It is for drawing only: contacts and laser hits still wait for recorded samples.
The tube's end is capped bar by bar, since a cap fanned from the hull's centre would fill the
open cage.

**God's eye shows what is still open.** In single player the God's-eye slice (`dt' = 0` in the
camera frame) lies wholly outside the camera's future cone, so all of it is decided. In a match
the slice can reach into `I⁺(O_remote)` when the other player lags behind in the camera's frame.
Those parts are drawn in the same violet haze as the Prophecy slice: the renderers haze every
event inside the future cone of the latest known event of any other live ship
(`FrameDynamics::remote_origins`), plus, on the Prophecy slice, the camera's own.

A player with `Δt ≤ 0` never needs to stop, so at least one player can always move. Both can be
stopped only briefly, when their information about each other is stale; heartbeats resolve that.
**Latency figures.** In a match the HUD's time-controls line shows, once the other player has
joined:

* `NET RTT`: the wall-clock round trip, from a `Ping` (sent every 0.5 s) to its `Pong`, smoothed.
  Both sides' frame times are included, since messages are read once per frame.
* `CM LEAD`: the CM time `Δt` by which my latest event leads the other ship's latest known one
  (what the pacer compares against).
* `INTERVAL`: the invariant interval between those two events, as a distance. The pacer keeps it
  spacelike, except that a ship within one step of the other may poke one step into its cone;
  it then shows `TIMELIKE` and a proper time.
* `CM DIST`: the distance between the two ships in the CM frame at equal CM time (the other ship
  coasted inertially to my event's CM time).
* `WAITING`: the share of the last ~5 s of wall time this player spent waiting.

Until the other player has joined (their `Hello` has arrived), the match is paused: the ship
cannot move or fire, so the world does not advance either, and the HUD shows **WAITING FOR A
PLAYER TO JOIN**. Their ship is then taken to be at its spawn event, at rest, until their first
track arrives. If the connection fails before they join, the local player is released and flies
alone.

## 2. Determinism

Both replicas must agree bit for bit. Billiards is chaotic, so any drift would eventually show one
player a collision that the other never sees. Each body's next state is already a pure function of
finalized data (single-player Phase 4). On top of that, multiplayer requires the following:

* **Inputs travel exactly.** Ship samples are sent as `f64` bits. Laser shots are sent with
  the pilot's `ShipSample` at firing and their `f64` direction.
* **Inputs are applied in order.** A shot fired at `O_f` is sent before the track sample that
  moves past `O_f`. The pulse lies on `O_f`'s light cone, so the receiver, which never passes
  `O_f`'s future cone before hearing of it, inserts it in time. Where a pulse goes out depends
  on the pucks and on the other ship's hull: it waits for that ship's track wherever it might
  meet the hull (`laser::Pulse::advance`), so both replicas find the same end.
* **Identities are per player.** Pulses are identified by `(player, shot index)`.
* **Scores are local news.** Each replica counts a hit once the hit event lies in its own
  player's past light cone, so the two scoreboards agree once the news has reached both.
* **No lab-time rebase in multiplayer.** Rebasing at different moments on the two peers would
  change `f64` rounding. Physics is `f64`, so precision is fine for days. Render precision
  degrades slowly after many hours.
* Pruning keeps everything newer than the *earliest* live frontier minus the history window.
* Both peers must run the same build on the same platform (`f64` arithmetic is IEEE, but the
  `cbrt` in `billiards::body` comes from the platform's libm).

## 3. Architecture

```
src/dynamic/ship.rs        PlayerId, ShipSample (+ proper time); pilot and hull tracks
src/dynamic/helm.rs        causal controls: delayed commands, or a control pod and its hull
src/dynamic/mod.rs         DynWorld: N ships, union-of-cones frontier, per-ship tracks, kicks,
                           and other ships drawn as hull cages around their pilots' clocks
src/multiplayer/protocol.rs Message + compact binary encoding
src/multiplayer/pacing.rs   CM-frame lead, stop/resume rule (Pacer)
src/multiplayer/link.rs     Link trait; LocalLink (in-process pair); QueueLink (frontend-driven)
src/multiplayer/tcp.rs      native TcpLink (--host / --join), u32-length-prefixed frames
src/multiplayer/p2p.rs      native UdpLink (--p2p): NAT hole punching, reliable ordered frames
                            over UDP; P2pEndpoint, peer codes
src/multiplayer/stun.rs     minimal STUN Binding client (public address discovery)
src/multiplayer/rendezvous.rs  UDP rendezvous server (--rendezvous) and its client
src/multiplayer/session.rs  Session: Hello handshake, sequencing, log replay, events
src/app.rs                  App::new_multiplayer; applies remote events; paces the local ship
src/main.rs, src/web.rs, web/index.html   transports, CLI/URL, "waiting" banner
rendezvous/server.mjs       generic HTTP rendezvous server (rooms of write-once slots, long-poll)
```

### Session protocol

Every frame is `seq: u32 LE` followed by one encoded `Message`:

| Message | Payload |
|---|---|
| `Hello` | protocol version, sender's `PlayerId` |
| `Track` | the pilot's `ShipSample` (t, pos, vel, τ as `f64`), the sender's waiting flag, and the delayed-control orders issued since the last `Track` (count `u16`; each: kind, thrust direction or steering target `f32` ×4, rapidity, due τ); sent every frame, including while waiting, as a heartbeat |
| `Controls` | the sender's `ControlMode` (delayed or pod), sent before their ship first moves |
| `Laser` | the pilot's `ShipSample` at firing (its velocity fixes the frame the pulse is shaped in), lab direction (`f64` ×3) |
| `Flash` | emission event (`f32` ×4) |
| `Bye` | sender quits |
| `Ping` / `Pong` | the sender's wall clock (`f64` seconds); `Pong` echoes a `Ping`'s, timing the round trip |

Links are reliable and ordered once open (TCP, reliable UDP, BroadcastChannel, WebRTC), but frames sent
before the peer was listening are lost. A session therefore keeps a log of everything it has sent
until the peer's `Hello` arrives. It then replays the whole log, and the receiver drops sequence
numbers it has already seen. When the peer leaves (`Bye`, or the link closes), its ship stops
being an input source: its track ends and its cone is removed from the frontier.

### Transports

* **Native, TCP:** `cargo run --release -- --host 7777` and `cargo run --release -- --join 127.0.0.1:7777`.
  The host needs a reachable port.
* **Native, peer to peer:** `cargo run --release -- --p2p` on both machines (see below).
* **Web:** `index.html?mp=0` and `index.html?mp=1` in two tabs of the same browser, connected
  through a `BroadcastChannel`; or `index.html?p2p=host` and `index.html?p2p=join` on two
  computers, connected through a WebRTC data channel. JS ferries opaque frames to and from WASM
  (`web_net_*` exports).

The host is player 0 (the usual spawn point). The joiner is player 1, mirrored to the far side of
the table and facing it. In a native peer-to-peer match, the player with the smaller random nonce
is player 0. In multiplayer, scene switching and reset are disabled.

### Peer to peer and NAT punch-through

Most players are behind a NAT that drops unsolicited incoming packets, so neither of them can be
a TCP host. Both peer-to-peer modes therefore use *hole punching*. Each side learns the
addresses it can be reached at. The two sides swap those addresses, then send packets to each
other at the same time. Each outgoing packet opens a mapping in the sender's own NAT, so the
other side's packets are let in as replies.

**Native (`--p2p`, `src/multiplayer/p2p.rs`).** Everything goes through one UDP socket
(`--udp-port`, default any), so the mapping that STUN reports is the one the peer will reach.

1. *Candidates:* the LAN address and loopback, plus the public address from a STUN Binding
   request (`stun.l.google.com:19302`, `stun.cloudflare.com:3478`, or `--stun HOST:PORT`).
2. *Exchange*, in one of two ways:
   * By hand: each player prints a code (`SR4D-…`, base32 of a random `u64` nonce and the
     candidates) and pastes the other player's. While the program waits for the paste, it pings
     STUN every 10 s so the NAT keeps the public mapping.
   * Through a rendezvous server: `--rendezvous [PORT]` (UDP 7778 by default) on any host with a
     public IP. Both players run `--p2p ROOM@HOST[:PORT]`. The server answers each player with
     the other's nonce, the public address the server saw that player's packets come from, and
     that player's LAN candidates. It carries no game traffic.
3. *Punching:* both sides send `Punch` packets to every candidate of the other every 100 ms.
   The first packet from the peer locks the link onto its source address, and the link becomes
   `Open`. The punch gives up after 3 minutes.
4. *Reliable stream:* the frames form the same `u32`-length-prefixed stream as TCP, cut into
   segments of at most 1200 bytes. Each segment is numbered, acknowledged (cumulatively, plus a
   64-segment selective bitmap) and retransmitted after an adaptive timeout (Jacobson/Karn RTT
   estimate, exponential backoff). At most 256 segments are in flight. An idle link sends a
   keepalive every 250 ms, and 10 s of silence closes it. When the link is dropped, it waits up
   to 0.5 s for its last frames (the session's `Bye`) to be acknowledged.

Every packet carries `b"SR4P"`, a kind byte, and the sender's and receiver's nonces. Stray
datagrams, late STUN answers and our own punches bounced back from our own address are therefore
ignored.

**Web (`?p2p=host` / `?p2p=join`).** The browser's ICE agent does the gathering and the punching.
The players carry the SDP offer and answer by hand. The host's page shows an invite code
(`SR4W…`: deflate-raw of the SDP with every ICE candidate already gathered, in base64url). The
joiner pastes it and gets a reply code, and the host pastes that. The data channel is reliable
and ordered, and both sides pre-negotiate it as id 0.

**Web with a rendezvous server (`?p2p=host&room=NAME`).** To skip the pasting, the browsers swap
the offer and answer through `rendezvous/server.mjs`, a generic rendezvous server that knows
nothing about this game (no dependencies, Node 18+, HTTP port 8787 by default). Both players open
the game with the same `room`. The server is found at the page's own origin, or at
`&signal=https://server.example.com`. The game uses the room `/v1/specrel-4d/VERSION/NAME`, where
VERSION is the first 16 hex digits of the SHA-256 of `spec-rel.wasm`: the host puts its offer in
the slot `offer` and long-polls `answer`, the joiner long-polls `offer` and puts `answer`, and
the host then deletes the room. The SDP is the same one the codes carry, with every candidate in
it, so the server sees just these two messages and no game traffic. A player on another build
gets a version mismatch at once (the server pins each room to the version that first used it),
instead of a match whose worlds drift apart. Locally:

```
node rendezvous/server.mjs                  # rendezvous on :8787
python3 -m http.server 8000                 # the repo root, so the page finds target/…/spec-rel.wasm
# player 1: http://localhost:8000/web/?p2p=host&room=myroom&signal=http://localhost:8787
# player 2: http://localhost:8000/web/?p2p=join&room=myroom&signal=http://localhost:8787
```

The **Create online lobby** button does the same with a fresh random room name (`k7m-q2xp`) and
shows the host the joiner's link to share. A lobby stays open while its host keeps polling for an
answer.

The page needs a secure context (`https://` or `localhost`) for WebGPU, so on a public host the
game and the server sit behind a TLS reverse proxy. See [`HOSTING.md`](HOSTING.md).

**Relay.** When ICE finds no direct path (both players behind symmetric NATs, or UDP blocked),
the data channel goes through a TURN relay instead. The page fetches its ICE servers from the
rendezvous server's `/v1/ice-servers`, which, when `TURN_SECRET` and `TURN_URLS` are set, adds a TURN
server with time-limited credentials (the TURN REST API scheme: username `EXPIRY`,
password `base64(HMAC-SHA1(secret, username))`, valid 24 h). coturn checks them with
`use-auth-secret`. The browser still prefers a direct path, so the relay carries only the
matches that need it.

**Limits.** Hole punching needs at least one side's NAT to keep the same public port for
every destination ("endpoint-independent mapping"), which most home routers do. If both players
are behind symmetric NATs (some carrier-grade and corporate NATs), no direct path exists. The
punch then times out (native: **COULD NOT CONNECT**; web: a message in the panel), unless a web
match has a TURN relay (below). Native matches have no relay. Native peer-to-peer is IPv4 only, and
native and web players cannot play each other.

## 4. Not done yet

* Collisions of ships with bodies or with each other in a match. Ships pass through both: a
  collision needs a ship's track at the same lab time as what it hits, which forces lockstep at
  contact range.
* More than two players (the world supports N ships; session and pacing are 2-player).
* A relay for native peer-to-peer players behind symmetric NATs (web matches have one), and
  native↔web play (native WebRTC).
* Coordinated lab-time rebase for very long sessions.
