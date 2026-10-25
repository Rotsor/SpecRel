//! A tiny UDP rendezvous server for [peer-to-peer](super::p2p) matches, and its client.
//!
//! Both players send `Register { room, LAN candidates }` from their game socket. The server
//! notes the public address each request came from. Once two players are in the same room, it
//! answers each with `Match`: the other player's nonce, public address and LAN candidates. The
//! players then punch through to each other directly; no game traffic goes through the server.
//! Clients repeat `Register` until matched, which also keeps their entry fresh.

use std::collections::HashMap;
use std::io::{self, ErrorKind};
use std::net::{SocketAddr, SocketAddrV4, UdpSocket};
use std::time::{Duration, Instant};

use super::p2p::{get_addrs, packet, parse_packet, put_addrs, Kind, PeerInfo};

pub const DEFAULT_PORT: u16 = 7778;
/// Players who stopped registering are forgotten after this long.
const ENTRY_TTL: Duration = Duration::from_secs(5);
const REGISTER_INTERVAL: Duration = Duration::from_millis(500);

struct Waiting {
    nonce: u64,
    /// The address the server saw, i.e. the player's public address.
    public: SocketAddrV4,
    lan: Vec<SocketAddrV4>,
    seen: Instant,
}

impl Waiting {
    fn candidates(&self) -> Vec<SocketAddrV4> {
        let mut c = vec![self.public];
        c.extend(self.lan.iter().filter(|&&a| a != self.public));
        c
    }
}

pub struct RendezvousServer {
    socket: UdpSocket,
    rooms: HashMap<String, Vec<Waiting>>,
}

impl RendezvousServer {
    pub fn bind(addr: SocketAddr) -> io::Result<RendezvousServer> {
        Ok(RendezvousServer { socket: UdpSocket::bind(addr)?, rooms: HashMap::new() })
    }

    pub fn local_addr(&self) -> io::Result<SocketAddr> {
        self.socket.local_addr()
    }

    /// Serves until `until` (forever if `None`).
    pub fn serve(&mut self, until: Option<Instant>) -> io::Result<()> {
        let mut buf = [0u8; 2048];
        loop {
            let now = Instant::now();
            let wait = match until {
                Some(t) if t <= now => return Ok(()),
                Some(t) => (t - now).min(Duration::from_secs(1)),
                None => Duration::from_secs(1),
            };
            self.socket.set_read_timeout(Some(wait))?;
            match self.socket.recv_from(&mut buf) {
                Ok((n, SocketAddr::V4(from))) => self.handle(&buf[..n], from, Instant::now()),
                Ok(_) => {} // IPv6 senders are not supported
                Err(e) if matches!(
                    e.kind(),
                    ErrorKind::WouldBlock | ErrorKind::TimedOut | ErrorKind::ConnectionReset | ErrorKind::Interrupted
                ) => {}
                Err(e) => return Err(e),
            }
        }
    }

    fn handle(&mut self, p: &[u8], from: SocketAddrV4, now: Instant) {
        let Some((Kind::Register, nonce, _, body)) = parse_packet(p) else { return };
        let Some((&room_len, rest)) = body.split_first() else { return };
        let Some((room, rest)) = rest.split_at_checked(room_len as usize) else { return };
        let Ok(room) = std::str::from_utf8(room) else { return };
        let Some((lan, _)) = get_addrs(rest) else { return };

        for players in self.rooms.values_mut() {
            players.retain(|w| now - w.seen < ENTRY_TTL);
        }
        self.rooms.retain(|_, players| !players.is_empty());
        let players = self.rooms.entry(room.to_string()).or_default();
        players.retain(|w| w.nonce != nonce);
        let me = Waiting { nonce, public: from, lan, seen: now };
        if let Some(other) = players.last() {
            let tell = |to: &Waiting, about: &Waiting| {
                let mut body = Vec::new();
                put_addrs(&mut body, &about.candidates());
                let _ = self.socket.send_to(&packet(Kind::Match, about.nonce, to.nonce, &body), to.public);
            };
            tell(&me, other);
            tell(other, &me);
        }
        players.push(me);
    }
}

/// Registers `me` in `room` at `server` from `socket` until the server names the other player.
pub(crate) fn meet(
    socket: &UdpSocket,
    me: &PeerInfo,
    server: SocketAddrV4,
    room: &str,
    timeout: Duration,
) -> io::Result<PeerInfo> {
    let room_len = u8::try_from(room.len())
        .map_err(|_| io::Error::new(ErrorKind::InvalidInput, "room name longer than 255 bytes"))?;
    let mut body = vec![room_len];
    body.extend_from_slice(room.as_bytes());
    // The server sees our public address itself; loopback is useless to a remote peer but
    // harmless, and lets two players on one machine meet.
    put_addrs(&mut body, &me.candidates);
    let register = packet(Kind::Register, me.nonce, 0, &body);

    let start = Instant::now();
    let deadline = start + timeout;
    let mut buf = [0u8; 2048];
    let mut next_register = start;
    let result = loop {
        let now = Instant::now();
        if now >= deadline {
            break Err(io::Error::new(ErrorKind::TimedOut, "nobody else joined the room"));
        }
        if now >= next_register {
            socket.send_to(&register, server)?;
            next_register = now + REGISTER_INTERVAL;
        }
        let wake = next_register.min(deadline);
        socket.set_read_timeout(Some((wake - now).max(Duration::from_millis(1))))?;
        match socket.recv_from(&mut buf) {
            Ok((n, from)) if from == SocketAddr::V4(server) => {
                if let Some((Kind::Match, peer, to, body)) = parse_packet(&buf[..n]) {
                    if to == me.nonce && peer != me.nonce {
                        if let Some((candidates, _)) = get_addrs(body) {
                            break Ok(PeerInfo { nonce: peer, candidates });
                        }
                    }
                }
            }
            Ok(_) => {}
            Err(e) if matches!(
                e.kind(),
                ErrorKind::WouldBlock | ErrorKind::TimedOut | ErrorKind::ConnectionReset | ErrorKind::Interrupted
            ) => {}
            Err(e) => break Err(e),
        }
    };
    socket.set_read_timeout(None)?;
    result
}

#[cfg(test)]
mod tests {
    use super::super::link::{Link, LinkState};
    use super::super::p2p::P2pEndpoint;
    use super::*;

    #[test]
    fn two_players_meet_and_connect() {
        let mut server = RendezvousServer::bind("127.0.0.1:0".parse().unwrap()).unwrap();
        let SocketAddr::V4(server_addr) = server.local_addr().unwrap() else { panic!() };
        // Not joined: the server outlives the test, which only checks it never failed.
        let serving = std::thread::spawn(move || {
            server.serve(Some(Instant::now() + Duration::from_secs(60))).unwrap();
        });

        let a = P2pEndpoint::bind(0).unwrap();
        let b = P2pEndpoint::bind(0).unwrap();
        // A third player in another room must not be matched with either.
        let c = P2pEndpoint::bind(0).unwrap();
        let (a_nonce, b_nonce) = (a.info().nonce, b.info().nonce);
        let a_meets = std::thread::spawn(move || {
            let peer = a.rendezvous(server_addr, "room-1", Duration::from_secs(5)).unwrap();
            (a, peer)
        });
        assert!(c.rendezvous(server_addr, "room-2", Duration::from_millis(700)).is_err());
        let b_peer = b.rendezvous(server_addr, "room-1", Duration::from_secs(5)).unwrap();
        let (a, a_peer) = a_meets.join().unwrap();
        assert_eq!(a_peer.nonce, b_nonce);
        assert_eq!(b_peer.nonce, a_nonce);
        // The public address the server saw comes first; on loopback it is the socket's own.
        assert_eq!(b_peer.candidates[0].ip(), &std::net::Ipv4Addr::LOCALHOST);
        assert_eq!(b_peer.candidates[0].port(), a.socket().local_addr().unwrap().port());

        let (pa, mut la) = a.connect(a_peer).unwrap();
        let (pb, mut lb) = b.connect(b_peer).unwrap();
        assert_ne!(pa, pb);
        let start = Instant::now();
        let mut got = None;
        while got.is_none() {
            assert!(start.elapsed() < Duration::from_secs(10), "timed out");
            la.send(b"hello");
            got = lb.recv();
            std::thread::sleep(Duration::from_millis(2));
        }
        assert_eq!(got.as_deref(), Some(&b"hello"[..]));
        assert_eq!(la.state(), LinkState::Open);
        for link in [&la, &lb] {
            assert_ne!(link.peer_addr(), Some(SocketAddr::V4(server_addr)), "locked onto the server");
        }
        assert!(!serving.is_finished(), "the server stopped");
    }
}
