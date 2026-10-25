# Hosting the SpecRel 4D lobby server

The lobby server lets players start a web match by sharing a link. It is three pieces on one
Linux host:

| Piece | Port | Role |
|---|---|---|
| `server/signal.mjs` (Node, no dependencies) | 8787, localhost only | Serves the game (`web/`), runs the lobbies (swaps each pair's WebRTC offer and answer), hands out relay credentials at `/turn` |
| Caddy | 80, 443 (TCP) | HTTPS in front of it, with an automatic Let's Encrypt certificate |
| coturn | 3478 (UDP, TCP), 49160–49999 (UDP) | TURN relay for players with no direct path between them |

Game traffic goes straight between the two browsers whenever it can. The relay carries only
the matches that can't connect directly. A match uses a few kB/s, so a small VPS (1 vCPU,
1 GB RAM) is plenty. The steps below assume Ubuntu or Debian, a public IPv4 address and root
access. The example files are in [`server/deploy/`](../server/deploy).

## 1. Domain name

The page needs HTTPS (WebGPU only works on secure pages), so the server needs a domain name
whose A record points at its IP. Without one, use `IP-WITH-DASHES.sslip.io`, for example
`203-0-113-10.sslip.io` for `203.0.113.10`: it resolves to that IP, and Let's Encrypt will issue
a certificate for it.

## 2. Packages, user and code

```bash
apt-get update
apt-get install -y nodejs git caddy coturn build-essential curl
useradd -r -m -d /opt/specrel -s /bin/bash specrel
sudo -u specrel -H bash -c '
  git clone https://github.com/Rotsor/SpecRel.git ~/SpecRel
  curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --target wasm32-unknown-unknown'
```

Node 18 or later is needed (`node --version`).

## 3. Build the game for the browser

```bash
sudo -u specrel -H bash -c '
  cd ~/SpecRel
  ~/.cargo/bin/cargo build --release --target wasm32-unknown-unknown
  cp target/wasm32-unknown-unknown/release/spec-rel.wasm web/'
```

Players' browsers only replicate the same world if both load the same build, which they do
when they load it from the same server.

## 4. Relay (coturn)

Generate a shared secret, which both coturn and the lobby server will use:

```bash
openssl rand -hex 32
```

Copy [`server/deploy/turnserver.conf`](../server/deploy/turnserver.conf) to
`/etc/turnserver.conf` and fill in the public IP, the domain and the secret. Then copy
[`server/deploy/specrel-signal.env`](../server/deploy/specrel-signal.env) to
`/etc/specrel-signal.env` with the same secret and the public IP in `TURN_URLS`, and protect it:

```bash
chmod 600 /etc/specrel-signal.env
systemctl enable --now coturn
systemctl restart coturn
```

Leave out this step and the env file if you don't want a relay. `/turn` then lists only public
STUN servers, and players without a direct path get "Could not connect".

## 5. Lobby server (systemd)

```bash
cp /opt/specrel/SpecRel/server/deploy/specrel-signal.service /etc/systemd/system/
systemctl daemon-reload
systemctl enable --now specrel-signal
```

It restarts if it crashes and starts at boot. Logs: `journalctl -u specrel-signal -f`.

## 6. HTTPS (Caddy)

Put [`server/deploy/Caddyfile`](../server/deploy/Caddyfile) in `/etc/caddy/Caddyfile` with your
domain, then:

```bash
systemctl restart caddy
```

## 7. Firewall

Open TCP 22, 80 and 443, TCP and UDP 3478, and UDP 49160–49999. With `ufw`:

```bash
ufw allow 22/tcp
ufw allow 80,443/tcp
ufw allow 3478
ufw allow 49160:49999/udp
ufw enable
```

Your hosting provider's firewall, if it has one, needs the same rules.

## 8. Check it

```bash
curl https://game.example.com/health
curl https://game.example.com/turn
```

The first answers `{"ok":true,"rooms":0}`. The second lists the STUN servers, plus your TURN
URLs with a username and credential if the relay is configured. Then open
`https://game.example.com/`, press **Create online lobby**, and open the link it shows in
another browser or on another computer. The game stays paused with **WAITING FOR A PLAYER TO
JOIN** until the second player arrives.

## Updating

```bash
sudo -u specrel -H bash -c '
  cd ~/SpecRel && git pull &&
  ~/.cargo/bin/cargo build --release --target wasm32-unknown-unknown &&
  cp target/wasm32-unknown-unknown/release/spec-rel.wasm web/'
systemctl restart specrel-signal
```

Players already in a match keep playing (their browsers are connected to each other), but
both need to reload before their next match so they run the same build.

## Running it locally

For development, no proxy or relay is needed, because `localhost` counts as a secure page:

```bash
node server/signal.mjs --static web
```

Then open `http://localhost:8787/`.
