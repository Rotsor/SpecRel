# Hosting the rendezvous server (and the SpecRel game)

Online web matches need a public host for two jobs: introducing the two browsers to each other,
and relaying traffic for the pairs that can't connect directly. Neither job has anything to do
with SpecRel, so the host runs a generic **rendezvous server** that any app can use. An app names
itself and its version in every request, and peers only meet when both match. Shipping a new build
of the game, or of any other app, never needs the server to change or restart.

## What runs on the host

| Component | Port | Role |
|---|---|---|
| [`rendezvous/server.mjs`](../rendezvous/server.mjs) (Node, no dependencies) | 8787, localhost only | Runs the rooms where peers swap their connection messages (each pair's WebRTC offer and answer), hands out relay credentials at `/v1/ice-servers` |
| coturn | 3478 (UDP, TCP), 49160–49999 (UDP) | TURN relay for peers with no direct path between them |
| Caddy | 80, 443 (TCP) | HTTPS in front of the server, with an automatic Let's Encrypt certificate; serves the game's files |

Game traffic goes straight between the browsers whenever it can, and the relay carries only the
matches that can't connect directly. A match uses a few kB/s, so a small VPS (1 vCPU, 1 GB RAM)
is plenty.

What the host deliberately does **not** have:

- **No game code in the server.** The server stores opaque messages under names the client picks
  (`/v1/APP/VERSION/ROOM/SLOT`). SpecRel's version is a hash of its `spec-rel.wasm`, so players on
  different builds are told so instead of getting a broken match.
- **No build toolchain or git checkout.** You build the game on your own machine and copy the
  files over. Caddy serves them as they are, so an update restarts nothing.
- **No database.** Rooms live for minutes, in memory. A restart only drops the lobbies that are
  waiting for their second player, and those players just make a new lobby.
- **No service account.** systemd runs the server as a throwaway user (`DynamicUser`), because it
  never writes a file.

The game itself could be hosted anywhere, for example GitHub Pages. Players then add
`&signal=https://your-rendezvous-host` to the URL. Serving it from the same Caddy keeps one origin,
which avoids that parameter.

The steps below assume Ubuntu or Debian, a public IPv4 address and root access. The example config
files are in [`rendezvous/`](../rendezvous).

## 1. Domain name

The server needs a domain name whose A record points at its IP, so that it can have a
certificate. Without one, use `IP-WITH-DASHES.sslip.io`, for example `203-0-113-10.sslip.io` for
`203.0.113.10`: it resolves to that IP, and Let's Encrypt will issue a certificate for it.

## 2. Packages and the server

```bash
apt-get update
apt-get install -y nodejs caddy coturn rsync
mkdir -p /opt/rendezvous /var/www/app
curl -fsSL -o /opt/rendezvous/server.mjs \
  https://raw.githubusercontent.com/Rotsor/SpecRel/claude/dynamic-mode/rendezvous/server.mjs
```

Node 18 or later is needed (`node --version`). You can also `scp` the file from a checkout.

## 3. Relay (coturn)

Generate a shared secret, which coturn and the rendezvous server both use:

```bash
openssl rand -hex 32
```

Copy [`rendezvous/turnserver.conf`](../rendezvous/turnserver.conf) to `/etc/turnserver.conf` and
fill in the public IP, the domain and the secret. Then copy
[`rendezvous/rendezvous.env`](../rendezvous/rendezvous.env) to `/etc/rendezvous.env` with the
same secret and the public IP in `TURN_URLS`, and protect it:

```bash
chmod 600 /etc/rendezvous.env
systemctl enable --now coturn
systemctl restart coturn
```

Leave out this step and the env file if you don't want a relay. `/v1/ice-servers` then lists only
public STUN servers, and players without a direct path get "Could not connect".

## 4. Rendezvous server (systemd)

Copy [`rendezvous/rendezvous.service`](../rendezvous/rendezvous.service) to
`/etc/systemd/system/`, then:

```bash
systemctl daemon-reload
systemctl enable --now rendezvous
```

It restarts if it crashes and starts at boot. Logs: `journalctl -u rendezvous -f`.

## 5. HTTPS (Caddy)

Put [`rendezvous/Caddyfile`](../rendezvous/Caddyfile) in `/etc/caddy/Caddyfile` with your domain.
It sends `/v1/*` to the rendezvous server and serves everything else from `/var/www/app`, with
the headers that make the game's pages cross-origin isolated, which sharpens `performance.now()`
for the frame timings in the HUD (`F5`). Then:

```bash
systemctl restart caddy
```

## 6. Firewall

Open TCP 22, 80 and 443, TCP and UDP 3478, and UDP 49160–49999. With `ufw`:

```bash
ufw allow 22/tcp
ufw allow 80,443/tcp
ufw allow 3478
ufw allow 49160:49999/udp
ufw enable
```

Your hosting provider's firewall, if it has one, needs the same rules.

## 7. Deploy the game

On your own machine, from the SpecRel checkout:

```bash
cargo build --release --target wasm32-unknown-unknown
rsync -v web/index.html web/online.html target/wasm32-unknown-unknown/release/spec-rel.wasm root@SERVER:/var/www/app/
```

rsync replaces each file in one step, so nobody loads half a file. Run the same two commands for
every new build: nothing on the host restarts. Players already in a match keep playing (their
browsers are connected to each other). Players who load the new build can't join lobbies made
with the old one: the page tells them the other player is on a different build.

## 8. Check it

```bash
curl https://rendezvous.example.com/v1/health
curl https://rendezvous.example.com/v1/ice-servers
```

The first answers `{"ok":true,"rooms":0,"bytes":0}`. The second lists the STUN servers, plus your
TURN URLs with a username and credential if the relay is configured. Then open
`https://rendezvous.example.com/`, press **Create online lobby**, and open the link it shows in
another browser or on another computer. The game stays paused with **WAITING FOR A PLAYER TO
JOIN** until the second player arrives.

## Updating the rendezvous server

Only needed when `rendezvous/server.mjs` itself changes, which is independent of game releases:

```bash
curl -fsSL -o /opt/rendezvous/server.mjs \
  https://raw.githubusercontent.com/Rotsor/SpecRel/claude/dynamic-mode/rendezvous/server.mjs
systemctl restart rendezvous
```

Lobbies waiting for a second player are lost. Matches in progress are not affected.

## Running it locally

For development, no proxy or relay is needed, because `localhost` counts as a secure page. From
the repo root:

```bash
node rendezvous/server.mjs
```

```bash
python3 -m http.server 8000
```

Then open `http://localhost:8000/web/?signal=http://localhost:8787`. The page finds the wasm in
`target/` by itself.
