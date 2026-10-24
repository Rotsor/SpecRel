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
src/multiplayer/p2p.rs      native UdpLink (--p2p): NAT hole punching, reliable ordered frames
                            over UDP; P2pEndpoint, peer codes
src/multiplayer/stun.rs     minimal STUN Binding client (public address discovery)
src/multiplayer/rendezvous.rs  UDP rendezvous server (--rendezvous) and its client
src/multiplayer/session.rs  Session: Hello handshake, sequencing, log replay, events
src/app.rs                  App::new_multiplayer; applies remote events; paces the local ship
src/main.rs, src/web.rs, web/index.html   transports, CLI/URL, "waiting" banner
server/signal.mjs           HTTP signalling server for web matches (rooms, offer/answer long-poll)
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

**Web with a signalling server (`?p2p=host&room=NAME`).** To skip the pasting, run
`node server/signal.mjs` (no dependencies, Node 18+, HTTP port 8787 by default) on any host the
two browsers can reach. Both players open the game with the same `room`, and with
`&signal=https://server.example.com` when the game is not served by that same server. The host
posts its offer to `/rooms/NAME/offer` and long-polls `/rooms/NAME/answer`. The joiner long-polls
the offer and posts its answer. The SDP is the same one the codes carry, with every candidate in
it, so the server sees just these two messages and no game traffic. A room takes one host and one
joiner. It is dropped 30 s after the host fetches the answer, or after 10 minutes. With
`--static web` the server also serves the game itself, so `signal=` is not needed:

```
node server/signal.mjs --port 8787 --static web
# player 1: http://server:8787/?p2p=host&room=myroom
# player 2: http://server:8787/?p2p=join&room=myroom
```

Browsers allow WebRTC only on `https://` pages and `localhost`, so on a public host put the server
behind a TLS reverse proxy (Caddy, nginx and so on).

**Limits.** Hole punching needs at least one side's NAT to keep the same public port for
every destination ("endpoint-independent mapping"), which most home routers do. If both players
are behind symmetric NATs (some carrier-grade and corporate NATs), no direct path exists. The
punch then times out (native: **COULD NOT CONNECT**; web: a message in the panel). That case
would need a TURN-style relay, which is not implemented. Native peer-to-peer is IPv4 only, and
native and web players cannot play each other.

## 4. Not done yet

* Ship–ship collisions. Ships pass through each other: a collision needs both tracks at the same
  lab time, which forces lockstep at contact range.
* More than two players (the world supports N ships; session and pacing are 2-player).
* A relay for players who are both behind symmetric NATs, and native↔web play (native WebRTC).
* Coordinated lab-time rebase for very long sessions.
