//! Peer-to-peer transport over UDP with NAT hole punching.
//!
//! Each player binds one UDP socket and learns the addresses it can be reached at
//! (*candidates*): its LAN address, and its public address as seen by a STUN server
//! ([`P2pEndpoint::discover`]) or by a rendezvous server ([`P2pEndpoint::rendezvous`]). The two
//! players swap candidates, either by pasting each other's [code](P2pEndpoint::code) or through
//! the rendezvous server, and then both send *punch* packets to every candidate of the other.
//! Outgoing packets open a mapping in the sender's NAT, so once both sides have punched, the
//! first packet that gets through locks the link onto that address and it becomes `Open`.
//!
//! On top of the datagrams, [`UdpLink`] carries the same `u32`-length-prefixed frame stream as
//! TCP, cut into numbered segments that are acknowledged (cumulative plus a 64-segment selective
//! bitmap) and retransmitted, so frames arrive whole, reliably and in order.
//!
//! Every packet starts with `b"SR4P"`, a kind byte, the sender's nonce and the receiver's nonce
//! (`u64` LE each). Nonces are random per endpoint and identify the pair, so stray datagrams and
//! our own punches reflected back to us (e.g. to a LAN address that happens to be ours) are
//! ignored.
//!
//! This works through the common "endpoint-independent" NATs. When both players are behind
//! symmetric NATs (each destination gets a new public port) no direct path exists and the punch
//! times out; that needs a relay, which is not implemented.

use std::collections::hash_map::RandomState;
use std::collections::{BTreeMap, VecDeque};
use std::hash::{BuildHasher, Hasher};
use std::io::{self, ErrorKind};
use std::net::{Ipv4Addr, SocketAddr, SocketAddrV4, UdpSocket};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant, SystemTime};

use super::link::{push_frame, split_frames, Link, LinkState};
use crate::dynamic::PlayerId;

const MAGIC: [u8; 4] = *b"SR4P";
const HEADER: usize = 4 + 1 + 8 + 8;
/// Stream bytes per data packet; keeps datagrams under the 1280-byte IPv6 minimum MTU.
const MAX_SEGMENT: usize = 1200;
/// Unacknowledged segments allowed in flight.
const MAX_IN_FLIGHT: usize = 256;
/// Segments accepted ahead of the next expected one.
const RECV_WINDOW: u32 = 1024;
const PUNCH_INTERVAL: Duration = Duration::from_millis(100);
/// Idle links send a punch this often to keep NAT mappings open and show they are alive.
const KEEPALIVE: Duration = Duration::from_millis(250);
/// Give up when no punch got through by then. Generous, because the other player may still be
/// pasting our code.
const PUNCH_TIMEOUT: Duration = Duration::from_secs(180);
/// While waiting for the other player's code, ping STUN this often: NATs commonly forget idle
/// UDP mappings after 30 s or more.
const MAPPING_KEEPALIVE: Duration = Duration::from_secs(10);
/// The peer is gone after this much silence.
const PEER_TIMEOUT: Duration = Duration::from_secs(10);
const INITIAL_RTO: Duration = Duration::from_millis(300);
const MIN_RTO: Duration = Duration::from_millis(100);
const MAX_RTO: Duration = Duration::from_secs(3);
/// How long dropping an open link waits for its last frames to be acknowledged.
const LINGER: Duration = Duration::from_millis(500);
/// Prefix of the text form of a peer code.
const CODE_PREFIX: &str = "SR4D-";
const CODE_VERSION: u8 = 1;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub(crate) enum Kind {
    Punch = 0,
    Data = 1,
    Ack = 2,
    Bye = 3,
    /// Client to rendezvous server: room, candidates.
    Register = 16,
    /// Rendezvous server to client: the peer's candidates; `from` is the peer's nonce.
    Match = 17,
}

impl Kind {
    fn from_u8(b: u8) -> Option<Kind> {
        Some(match b {
            0 => Kind::Punch,
            1 => Kind::Data,
            2 => Kind::Ack,
            3 => Kind::Bye,
            16 => Kind::Register,
            17 => Kind::Match,
            _ => return None,
        })
    }
}

/// A whole packet: header, then `payload`.
pub(crate) fn packet(kind: Kind, from: u64, to: u64, payload: &[u8]) -> Vec<u8> {
    let mut p = Vec::with_capacity(HEADER + payload.len());
    p.extend_from_slice(&MAGIC);
    p.push(kind as u8);
    p.extend_from_slice(&from.to_le_bytes());
    p.extend_from_slice(&to.to_le_bytes());
    p.extend_from_slice(payload);
    p
}

/// `(kind, from, to, payload)` of a packet of ours.
pub(crate) fn parse_packet(p: &[u8]) -> Option<(Kind, u64, u64, &[u8])> {
    if p.len() < HEADER || p[..4] != MAGIC {
        return None;
    }
    let kind = Kind::from_u8(p[4])?;
    let from = u64::from_le_bytes(p[5..13].try_into().unwrap());
    let to = u64::from_le_bytes(p[13..21].try_into().unwrap());
    Some((kind, from, to, &p[HEADER..]))
}

/// Appends a count byte and the addresses (4 address bytes and a `u16` LE port each).
pub(crate) fn put_addrs(out: &mut Vec<u8>, addrs: &[SocketAddrV4]) {
    let addrs = &addrs[..addrs.len().min(u8::MAX as usize)];
    out.push(addrs.len() as u8);
    for a in addrs {
        out.extend_from_slice(&a.ip().octets());
        out.extend_from_slice(&a.port().to_le_bytes());
    }
}

/// Reads what [`put_addrs`] wrote; returns the addresses and the rest of `b`.
pub(crate) fn get_addrs(b: &[u8]) -> Option<(Vec<SocketAddrV4>, &[u8])> {
    let (&n, mut rest) = b.split_first()?;
    let mut addrs = Vec::with_capacity(n as usize);
    for _ in 0..n {
        let (a, r) = rest.split_first_chunk::<6>()?;
        addrs.push(SocketAddrV4::new(
            Ipv4Addr::new(a[0], a[1], a[2], a[3]),
            u16::from_le_bytes([a[4], a[5]]),
        ));
        rest = r;
    }
    Some((addrs, rest))
}

/// Unpredictable bytes from the standard library's randomly keyed hasher, the clock and a
/// counter. Good enough for nonces and STUN transaction ids.
pub(crate) fn random_bytes<const N: usize>() -> [u8; N] {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let mut out = [0u8; N];
    for chunk in out.chunks_mut(8) {
        let mut h = RandomState::new().build_hasher();
        h.write_u64(COUNTER.fetch_add(1, Ordering::Relaxed));
        let now = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap_or_default();
        h.write_u128(now.as_nanos());
        h.write_u32(std::process::id());
        chunk.copy_from_slice(&h.finish().to_le_bytes()[..chunk.len()]);
    }
    out
}

/// What one player tells the other: who they are and where to punch.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PeerInfo {
    pub nonce: u64,
    pub candidates: Vec<SocketAddrV4>,
}

impl PeerInfo {
    /// The text code for pasting to the other player, e.g. `SR4D-AEBAG...`.
    pub fn code(&self) -> String {
        let mut bytes = vec![CODE_VERSION];
        bytes.extend_from_slice(&self.nonce.to_le_bytes());
        put_addrs(&mut bytes, &self.candidates);
        format!("{CODE_PREFIX}{}", base32_encode(&bytes))
    }

    /// Reads a [code](PeerInfo::code), ignoring case, whitespace and the prefix.
    pub fn from_code(code: &str) -> Option<PeerInfo> {
        let text: String = code.chars().filter(|c| !c.is_whitespace()).collect();
        let text = text.to_ascii_uppercase();
        let text = text.strip_prefix(CODE_PREFIX).unwrap_or(&text);
        let bytes = base32_decode(text)?;
        let (&version, rest) = bytes.split_first()?;
        let (nonce, rest) = rest.split_first_chunk::<8>()?;
        let (candidates, rest) = get_addrs(rest)?;
        (version == CODE_VERSION && rest.is_empty())
            .then(|| PeerInfo { nonce: u64::from_le_bytes(*nonce), candidates })
    }
}

const BASE32: &[u8; 32] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZ234567";

fn base32_encode(bytes: &[u8]) -> String {
    let mut out = String::new();
    let (mut acc, mut bits) = (0u32, 0);
    for &b in bytes {
        acc = (acc << 8) | b as u32;
        bits += 8;
        while bits >= 5 {
            bits -= 5;
            out.push(BASE32[(acc >> bits) as usize & 31] as char);
        }
    }
    if bits > 0 {
        out.push(BASE32[(acc << (5 - bits)) as usize & 31] as char);
    }
    out
}

fn base32_decode(text: &str) -> Option<Vec<u8>> {
    let mut out = Vec::new();
    let (mut acc, mut bits) = (0u32, 0);
    for c in text.bytes() {
        let v = BASE32.iter().position(|&a| a == c)? as u32;
        acc = (acc << 5) | v;
        bits += 5;
        if bits >= 8 {
            bits -= 8;
            out.push((acc >> bits) as u8);
        }
    }
    Some(out)
}

/// Our side of a peer-to-peer connection before it is made: the socket every packet goes
/// through (so NAT mappings learned for it stay valid) and our candidates.
pub struct P2pEndpoint {
    socket: UdpSocket,
    info: PeerInfo,
    /// The STUN server that told us our public address.
    stun_server: Option<SocketAddr>,
}

impl P2pEndpoint {
    /// Binds a UDP socket on all interfaces at `port` (0 picks a free one). Its candidates
    /// start as the LAN and loopback addresses.
    pub fn bind(port: u16) -> io::Result<P2pEndpoint> {
        let socket = UdpSocket::bind((Ipv4Addr::UNSPECIFIED, port))?;
        let port = socket.local_addr()?.port();
        let mut candidates = Vec::new();
        if let Some(ip) = lan_ip() {
            candidates.push(SocketAddrV4::new(ip, port));
        }
        candidates.push(SocketAddrV4::new(Ipv4Addr::LOCALHOST, port));
        let nonce = u64::from_le_bytes(random_bytes());
        Ok(P2pEndpoint { socket, info: PeerInfo { nonce, candidates }, stun_server: None })
    }

    /// Who we are and where we can be reached.
    pub fn info(&self) -> &PeerInfo {
        &self.info
    }

    /// The code to give the other player.
    pub fn code(&self) -> String {
        self.info.code()
    }

    pub fn socket(&self) -> &UdpSocket {
        &self.socket
    }

    /// Learns our public address from STUN `servers` and makes it the first candidate.
    pub fn discover(&mut self, servers: &[&str], timeout: Duration) -> io::Result<SocketAddrV4> {
        match super::stun::public_addr(&self.socket, servers, timeout)? {
            (SocketAddr::V4(addr), server) => {
                self.add_candidate(addr);
                self.stun_server = Some(server);
                Ok(addr)
            }
            _ => Err(io::Error::new(ErrorKind::Unsupported, "STUN gave an IPv6 address")),
        }
    }

    /// Runs `f` (e.g. waiting for the other player's code) while pinging the STUN server that
    /// answered [`discover`](P2pEndpoint::discover), so our NAT keeps the public address it
    /// gave us. Any answers are discarded later as stray datagrams.
    pub fn keep_mapping_while<R>(&self, f: impl FnOnce() -> R) -> R {
        let Some((server, socket)) = self.stun_server.zip(self.socket.try_clone().ok()) else {
            return f();
        };
        let (stop, stopped) = std::sync::mpsc::channel::<()>();
        let pinger = std::thread::spawn(move || {
            let request = super::stun::binding_request(&random_bytes());
            while let Err(std::sync::mpsc::RecvTimeoutError::Timeout) = stopped.recv_timeout(MAPPING_KEEPALIVE) {
                let _ = socket.send_to(&request, server);
            }
        });
        let result = f();
        drop(stop);
        let _ = pinger.join();
        result
    }

    fn add_candidate(&mut self, addr: SocketAddrV4) {
        self.info.candidates.retain(|&c| c != addr);
        self.info.candidates.insert(0, addr);
    }

    /// Meets the other player in `room` at a [rendezvous server](super::rendezvous), which
    /// tells each side the other's public address (as it saw it) and LAN candidates. Blocks
    /// until matched or `timeout`.
    pub fn rendezvous(&self, server: SocketAddrV4, room: &str, timeout: Duration) -> io::Result<PeerInfo> {
        super::rendezvous::meet(&self.socket, &self.info, server, room, timeout)
    }

    /// Starts punching through to `peer`. The player with the smaller nonce is player 0.
    pub fn connect(self, peer: PeerInfo) -> io::Result<(PlayerId, UdpLink)> {
        if peer.nonce == self.info.nonce {
            return Err(io::Error::new(ErrorKind::InvalidInput, "that is our own code"));
        }
        let player = if self.info.nonce < peer.nonce { 0 } else { 1 };
        let link = UdpLink::new(self.socket, self.info.nonce, peer)?;
        Ok((player, link))
    }
}

/// The LAN address the OS would use to reach the internet (no packet is sent).
fn lan_ip() -> Option<Ipv4Addr> {
    let probe = UdpSocket::bind((Ipv4Addr::UNSPECIFIED, 0)).ok()?;
    probe.connect((Ipv4Addr::new(192, 0, 2, 1), 9)).ok()?;
    match probe.local_addr().ok()? {
        SocketAddr::V4(a) if !a.ip().is_unspecified() && !a.ip().is_loopback() => Some(*a.ip()),
        _ => None,
    }
}

struct Segment {
    data: Vec<u8>,
    sent: Instant,
    retransmitted: bool,
}

/// A [`Link`] to one peer over UDP. `Connecting` while punching, `Open` once a packet from the
/// peer got through, `Closed` on `Bye`, silence, or a punch that never got through.
pub struct UdpLink {
    socket: UdpSocket,
    local: u64,
    remote: u64,
    /// Where to punch until we hear from the peer.
    candidates: Vec<SocketAddr>,
    /// The address the peer's packets come from.
    peer: Option<SocketAddr>,
    closed: bool,
    started: Instant,
    last_heard: Instant,
    last_sent: Option<Instant>,
    /// Stream bytes not yet cut into segments.
    pending: Vec<u8>,
    next_seq: u32,
    unacked: BTreeMap<u32, Segment>,
    srtt: Option<Duration>,
    rttvar: Duration,
    rto: Duration,
    /// The next segment to append to `input`.
    recv_next: u32,
    /// Segments received ahead of `recv_next`.
    early: BTreeMap<u32, Vec<u8>>,
    input: Vec<u8>,
    frames: VecDeque<Vec<u8>>,
    ack_due: bool,
    /// Outgoing packets to drop: `(rng state, percent)`.
    #[cfg(test)]
    loss: Option<(u64, u64)>,
}

impl UdpLink {
    /// Punches from `socket` (as `local`) to `peer`.
    pub fn new(socket: UdpSocket, local: u64, peer: PeerInfo) -> io::Result<UdpLink> {
        socket.set_nonblocking(true)?;
        let now = Instant::now();
        Ok(UdpLink {
            socket,
            local,
            remote: peer.nonce,
            candidates: peer.candidates.into_iter().map(SocketAddr::V4).collect(),
            peer: None,
            closed: false,
            started: now,
            last_heard: now,
            last_sent: None,
            pending: Vec::new(),
            next_seq: 0,
            unacked: BTreeMap::new(),
            srtt: None,
            rttvar: Duration::ZERO,
            rto: INITIAL_RTO,
            recv_next: 0,
            early: BTreeMap::new(),
            input: Vec::new(),
            frames: VecDeque::new(),
            ack_due: false,
            #[cfg(test)]
            loss: None,
        })
    }

    /// The address the peer was reached at, once `Open`.
    pub fn peer_addr(&self) -> Option<SocketAddr> {
        self.peer
    }

    fn send_packet(&mut self, to: SocketAddr, kind: Kind, payload: &[u8]) {
        self.last_sent = Some(Instant::now());
        #[cfg(test)]
        if let Some((state, percent)) = &mut self.loss {
            *state ^= *state << 13;
            *state ^= *state >> 7;
            *state ^= *state << 17;
            if *state % 100 < *percent {
                return;
            }
        }
        // A full send buffer loses the packet like the network would; retransmission covers it.
        let _ = self.socket.send_to(&packet(kind, self.local, self.remote, payload), to);
    }

    /// Receives, punches or retransmits as due, and sends acknowledgements and keepalives.
    fn poll(&mut self) {
        if self.closed {
            return;
        }
        let now = Instant::now();
        self.receive(now);
        if self.closed {
            return;
        }
        match self.peer {
            None if now - self.started > PUNCH_TIMEOUT => self.close(),
            None => {
                if self.last_sent.is_none_or(|t| now - t >= PUNCH_INTERVAL) {
                    for to in self.candidates.clone() {
                        self.send_packet(to, Kind::Punch, &[]);
                    }
                }
            }
            Some(_) if now - self.last_heard > PEER_TIMEOUT => self.close(),
            Some(peer) => self.flush(now, peer),
        }
    }

    fn receive(&mut self, now: Instant) {
        let mut buf = [0u8; 2048];
        loop {
            match self.socket.recv_from(&mut buf) {
                Ok((n, from)) => self.handle(&buf[..n], from, now),
                // Windows reports an earlier ICMP "port unreachable" (a dead candidate) this way.
                Err(e) if matches!(e.kind(), ErrorKind::ConnectionReset | ErrorKind::ConnectionRefused) => {}
                Err(e) if e.kind() == ErrorKind::Interrupted => {}
                Err(_) => break,
            }
        }
        if !split_frames(&mut self.input, &mut self.frames) {
            self.close();
        }
    }

    fn handle(&mut self, p: &[u8], from: SocketAddr, now: Instant) {
        let Some((kind, sender, receiver, body)) = parse_packet(p) else { return };
        if sender != self.remote || receiver != self.local {
            return;
        }
        // Follow the peer if its NAT moves it to another port.
        self.peer = Some(from);
        self.last_heard = now;
        match kind {
            Kind::Data => {
                let Some((seq, data)) = body.split_first_chunk::<4>() else { return };
                let seq = u32::from_le_bytes(*seq);
                self.ack_due = true;
                let ahead = seq.wrapping_sub(self.recv_next);
                if ahead == 0 {
                    self.input.extend_from_slice(data);
                    self.recv_next = self.recv_next.wrapping_add(1);
                    while let Some(d) = self.early.remove(&self.recv_next) {
                        self.input.extend_from_slice(&d);
                        self.recv_next = self.recv_next.wrapping_add(1);
                    }
                } else if ahead < RECV_WINDOW {
                    self.early.entry(seq).or_insert_with(|| data.to_vec());
                } // else an old duplicate: the acknowledgement tells the sender
            }
            Kind::Ack => {
                let Some((next, rest)) = body.split_first_chunk::<4>() else { return };
                let Some((sack, _)) = rest.split_first_chunk::<8>() else { return };
                let (next, sack) = (u32::from_le_bytes(*next), u64::from_le_bytes(*sack));
                let acked: Vec<u32> = (self.unacked.keys().copied())
                    .filter(|&s| {
                        let d = s.wrapping_sub(next) as i32;
                        d < 0 || (1..=64).contains(&d) && sack >> (d - 1) & 1 != 0
                    })
                    .collect();
                let mut sample = None;
                for s in acked {
                    let seg = self.unacked.remove(&s).unwrap();
                    // Karn: only segments sent once give an unambiguous round trip.
                    if !seg.retransmitted {
                        sample = Some(now - seg.sent);
                    }
                }
                if let Some(rtt) = sample {
                    self.sample_rtt(rtt);
                }
            }
            Kind::Bye => self.close(),
            Kind::Punch | Kind::Register | Kind::Match => {}
        }
    }

    fn sample_rtt(&mut self, rtt: Duration) {
        let srtt = match self.srtt {
            None => {
                self.rttvar = rtt / 2;
                rtt
            }
            Some(s) => {
                self.rttvar = (self.rttvar * 3 + s.abs_diff(rtt)) / 4;
                (s * 7 + rtt) / 8
            }
        };
        self.srtt = Some(srtt);
        self.rto = (srtt + (self.rttvar * 4).max(Duration::from_millis(10))).clamp(MIN_RTO, MAX_RTO);
    }

    fn flush(&mut self, now: Instant, peer: SocketAddr) {
        let due: Vec<u32> = (self.unacked.iter())
            .filter(|(_, seg)| now - seg.sent >= self.rto)
            .map(|(&s, _)| s)
            .collect();
        for &s in &due {
            let seg = self.unacked.get_mut(&s).unwrap();
            seg.sent = now;
            seg.retransmitted = true;
            let payload = data_payload(s, &seg.data);
            self.send_packet(peer, Kind::Data, &payload);
        }
        if !due.is_empty() {
            self.rto = (self.rto * 2).min(MAX_RTO);
        }

        while !self.pending.is_empty() && self.unacked.len() < MAX_IN_FLIGHT {
            let data: Vec<u8> = self.pending.drain(..self.pending.len().min(MAX_SEGMENT)).collect();
            let seq = self.next_seq;
            self.next_seq = seq.wrapping_add(1);
            self.send_packet(peer, Kind::Data, &data_payload(seq, &data));
            self.unacked.insert(seq, Segment { data, sent: now, retransmitted: false });
        }

        if self.ack_due {
            self.ack_due = false;
            let mut sack = 0u64;
            for &s in self.early.keys() {
                let d = s.wrapping_sub(self.recv_next);
                if (1..=64).contains(&d) {
                    sack |= 1 << (d - 1);
                }
            }
            let mut payload = self.recv_next.to_le_bytes().to_vec();
            payload.extend_from_slice(&sack.to_le_bytes());
            self.send_packet(peer, Kind::Ack, &payload);
        }

        if self.last_sent.is_none_or(|t| now - t >= KEEPALIVE) {
            self.send_packet(peer, Kind::Punch, &[]);
        }
    }

    fn close(&mut self) {
        self.closed = true;
        self.pending.clear();
        self.unacked.clear();
    }
}

fn data_payload(seq: u32, data: &[u8]) -> Vec<u8> {
    let mut p = Vec::with_capacity(4 + data.len());
    p.extend_from_slice(&seq.to_le_bytes());
    p.extend_from_slice(data);
    p
}

impl Link for UdpLink {
    fn send(&mut self, frame: &[u8]) {
        self.poll();
        let Some(peer) = self.peer.filter(|_| !self.closed) else { return };
        push_frame(&mut self.pending, frame);
        self.flush(Instant::now(), peer);
    }

    fn recv(&mut self) -> Option<Vec<u8>> {
        if self.frames.is_empty() {
            self.poll();
        }
        self.frames.pop_front()
    }

    fn state(&self) -> LinkState {
        match (self.closed, self.peer) {
            (true, _) => LinkState::Closed,
            (false, Some(_)) => LinkState::Open,
            (false, None) => LinkState::Connecting,
        }
    }
}

impl Drop for UdpLink {
    /// Gives the last frames (usually the session's `Bye`) a moment to be acknowledged, then
    /// says goodbye at the link level too.
    fn drop(&mut self) {
        let Some(peer) = self.peer.filter(|_| !self.closed) else { return };
        let deadline = Instant::now() + LINGER;
        while !(self.pending.is_empty() && self.unacked.is_empty()) && !self.closed && Instant::now() < deadline {
            self.poll();
            std::thread::sleep(Duration::from_millis(2));
        }
        for _ in 0..3 {
            self.send_packet(peer, Kind::Bye, &[]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn code_round_trip() {
        let info = PeerInfo {
            nonce: 0x0123_4567_89ab_cdef,
            candidates: vec!["203.0.113.7:40000".parse().unwrap(), "192.168.1.20:5555".parse().unwrap()],
        };
        let code = info.code();
        assert!(code.starts_with(CODE_PREFIX));
        assert_eq!(PeerInfo::from_code(&code), Some(info.clone()));
        let messy = format!("  {}\n", code[..20].to_lowercase() + " " + &code[20..]);
        assert_eq!(PeerInfo::from_code(&messy), Some(info));
        assert_eq!(PeerInfo::from_code("SR4D-NOT1VALID"), None);
        assert_eq!(PeerInfo::from_code(""), None);
    }

    #[test]
    fn base32_round_trip() {
        for len in 0..20 {
            let bytes: Vec<u8> = (0..len).map(|i| (i * 37 + 11) as u8).collect();
            assert_eq!(base32_decode(&base32_encode(&bytes)).unwrap(), bytes);
        }
        assert_eq!(base32_encode(b"foobar"), "MZXW6YTBOI");
    }

    #[test]
    fn nonces_differ() {
        let a: [u8; 8] = random_bytes();
        let b: [u8; 8] = random_bytes();
        assert_ne!(a, b);
    }

    /// Two endpoints on loopback, each knowing only the other's code.
    fn pair(loss: u64) -> (PlayerId, UdpLink, PlayerId, UdpLink) {
        let a = P2pEndpoint::bind(0).unwrap();
        let b = P2pEndpoint::bind(0).unwrap();
        let mut a_info = PeerInfo::from_code(&a.code()).unwrap();
        // A dead candidate first, like an unreachable public address.
        a_info.candidates.insert(0, "127.0.0.1:9".parse().unwrap());
        let b_info = PeerInfo::from_code(&b.code()).unwrap();
        let (pa, mut la) = a.connect(b_info).unwrap();
        let (pb, mut lb) = b.connect(a_info).unwrap();
        if loss > 0 {
            la.loss = Some((0x9E37_79B9_7F4A_7C15, loss));
            lb.loss = Some((0x2545_F491_4F6C_DD1D, loss));
        }
        (pa, la, pb, lb)
    }

    fn pump(a: &mut UdpLink, b: &mut UdpLink, mut done: impl FnMut(&mut UdpLink, &mut UdpLink) -> bool) {
        let start = Instant::now();
        while !done(a, b) {
            assert!(start.elapsed() < Duration::from_secs(30), "timed out");
            a.poll();
            b.poll();
            std::thread::sleep(Duration::from_millis(1));
        }
    }

    fn frame(i: usize, len: usize) -> Vec<u8> {
        (0..len).map(|j| (i * 31 + j) as u8).collect()
    }

    fn exchange(loss: u64) {
        let (pa, mut a, pb, mut b) = pair(loss);
        assert_eq!(pa + pb, 1, "one player 0 and one player 1");
        assert_eq!(a.state(), LinkState::Connecting);
        a.send(b"dropped while connecting");
        pump(&mut a, &mut b, |a, b| a.state() == LinkState::Open && b.state() == LinkState::Open);

        let big = frame(7, 300_000);
        a.send(&big);
        b.send(&big);
        for i in 0..1000 {
            a.send(&frame(i, i % 50));
            b.send(&frame(i + 1, i % 70));
        }
        let (mut got_a, mut got_b) = (Vec::new(), Vec::new());
        pump(&mut a, &mut b, |a, b| {
            got_a.extend(std::iter::from_fn(|| a.recv()));
            got_b.extend(std::iter::from_fn(|| b.recv()));
            got_a.len() == 1001 && got_b.len() == 1001
        });
        assert_eq!(got_b[0], big);
        assert_eq!(got_a[0], big);
        for i in 0..1000 {
            assert_eq!(got_b[i + 1], frame(i, i % 50));
            assert_eq!(got_a[i + 1], frame(i + 1, i % 70));
        }

        // The last frame survives the drop, then the peer sees the link close. The drop lingers
        // for acknowledgements, so it runs alongside the peer, like another process would.
        a.send(b"bye");
        let dropper = std::thread::spawn(move || drop(a));
        let start = Instant::now();
        let mut last = Vec::new();
        while b.state() != LinkState::Closed {
            assert!(start.elapsed() < Duration::from_secs(15), "never closed");
            last.extend(std::iter::from_fn(|| b.recv()));
            std::thread::sleep(Duration::from_millis(1));
        }
        last.extend(std::iter::from_fn(|| b.recv()));
        assert_eq!(last, vec![b"bye".to_vec()]);
        dropper.join().unwrap();
    }

    #[test]
    fn punch_and_exchange() {
        exchange(0);
    }

    #[test]
    fn exchange_with_packet_loss() {
        exchange(20);
    }

    #[test]
    fn own_code_is_refused() {
        let a = P2pEndpoint::bind(0).unwrap();
        let own = a.info().clone();
        assert!(a.connect(own).is_err());
    }

    /// Our punches to our own address (as a "candidate") must not open the link to ourselves.
    #[test]
    fn ignores_reflected_punches() {
        let a = P2pEndpoint::bind(0).unwrap();
        let mut fake = a.info().clone();
        fake.nonce ^= 1;
        let (_, mut link) = a.connect(fake).unwrap();
        let start = Instant::now();
        while start.elapsed() < Duration::from_millis(300) {
            assert_eq!(link.recv(), None);
            std::thread::sleep(Duration::from_millis(5));
        }
        assert_eq!(link.state(), LinkState::Connecting);
    }
}
