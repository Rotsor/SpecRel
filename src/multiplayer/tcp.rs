//! Native TCP transport: non-blocking sockets carrying `u32` LE length-prefixed frames.

use std::collections::VecDeque;
use std::io::{self, ErrorKind, Read, Write};
use std::net::{TcpListener, TcpStream, ToSocketAddrs};
use std::time::Duration;

use super::link::{Link, LinkState};

const CONNECT_TIMEOUT: Duration = Duration::from_secs(5);
/// Larger frames are treated as a corrupt stream.
const MAX_FRAME: usize = 64 << 20;

/// A [`Link`] over TCP. The host accepts its first peer lazily, while being polled by
/// `send`/`recv`; until then its state is `Connecting`.
pub struct TcpLink {
    listener: Option<TcpListener>,
    stream: Option<TcpStream>,
    /// Encoded bytes not yet accepted by the socket.
    out: Vec<u8>,
    /// Received bytes not yet forming a whole frame.
    input: Vec<u8>,
    frames: VecDeque<Vec<u8>>,
    closed: bool,
}

impl TcpLink {
    /// Listens on all interfaces at `port` (0 picks a free one; see [`TcpLink::local_port`]).
    pub fn host(port: u16) -> io::Result<TcpLink> {
        Self::listen(("0.0.0.0", port))
    }

    /// Listens on `addr` for one peer.
    pub fn listen(addr: impl ToSocketAddrs) -> io::Result<TcpLink> {
        let listener = TcpListener::bind(addr)?;
        listener.set_nonblocking(true)?;
        Ok(Self::new(Some(listener), None))
    }

    /// Connects to a host at `addr` (e.g. `"127.0.0.1:7777"`).
    pub fn join(addr: &str) -> io::Result<TcpLink> {
        let mut last = io::Error::new(ErrorKind::InvalidInput, "no address");
        for a in addr.to_socket_addrs()? {
            match TcpStream::connect_timeout(&a, CONNECT_TIMEOUT) {
                Ok(stream) => {
                    configure(&stream)?;
                    return Ok(Self::new(None, Some(stream)));
                }
                Err(e) => last = e,
            }
        }
        Err(last)
    }

    fn new(listener: Option<TcpListener>, stream: Option<TcpStream>) -> TcpLink {
        TcpLink {
            listener,
            stream,
            out: Vec::new(),
            input: Vec::new(),
            frames: VecDeque::new(),
            closed: false,
        }
    }

    /// The local port of the listener or connection.
    pub fn local_port(&self) -> Option<u16> {
        let addr = match (&self.listener, &self.stream) {
            (Some(l), _) => l.local_addr(),
            (_, Some(s)) => s.local_addr(),
            _ => return None,
        };
        addr.ok().map(|a| a.port())
    }

    /// Accepts a pending peer, flushes queued output and reads available input.
    fn poll(&mut self) {
        self.accept();
        self.flush();
        self.fill();
    }

    fn accept(&mut self) {
        let Some(listener) = &self.listener else { return };
        match listener.accept() {
            Ok((stream, _)) => {
                if configure(&stream).is_ok() {
                    self.stream = Some(stream);
                    self.listener = None;
                } // else keep listening for another attempt
            }
            Err(e) if e.kind() == ErrorKind::WouldBlock => {}
            Err(_) => self.close(),
        }
    }

    fn flush(&mut self) {
        let Some(stream) = &mut self.stream else { return };
        let mut written = 0;
        let mut failed = false;
        while written < self.out.len() {
            match stream.write(&self.out[written..]) {
                Ok(0) => failed = true,
                Ok(n) => {
                    written += n;
                    continue;
                }
                Err(e) if e.kind() == ErrorKind::Interrupted => continue,
                Err(e) if e.kind() == ErrorKind::WouldBlock => {}
                Err(_) => failed = true,
            }
            break;
        }
        self.out.drain(..written);
        if failed {
            self.close();
        }
    }

    fn fill(&mut self) {
        let Some(stream) = &mut self.stream else { return };
        let mut buf = [0u8; 64 * 1024];
        let mut failed = false;
        loop {
            match stream.read(&mut buf) {
                Ok(0) => failed = true,
                Ok(n) => {
                    self.input.extend_from_slice(&buf[..n]);
                    continue;
                }
                Err(e) if e.kind() == ErrorKind::Interrupted => continue,
                Err(e) if e.kind() == ErrorKind::WouldBlock => {}
                Err(_) => failed = true,
            }
            break;
        }
        failed |= !self.split_frames();
        if failed {
            self.close();
        }
    }

    /// Moves whole frames from `input` to `frames`; false on a corrupt length.
    fn split_frames(&mut self) -> bool {
        let mut start = 0;
        while let Some(header) = self.input.get(start..start + 4) {
            let len = u32::from_le_bytes(header.try_into().unwrap()) as usize;
            if len > MAX_FRAME {
                return false;
            }
            let Some(frame) = self.input.get(start + 4..start + 4 + len) else { break };
            self.frames.push_back(frame.to_vec());
            start += 4 + len;
        }
        self.input.drain(..start);
        true
    }

    fn close(&mut self) {
        self.closed = true;
        self.listener = None;
        self.stream = None;
        self.out.clear();
    }
}

fn configure(stream: &TcpStream) -> io::Result<()> {
    stream.set_nonblocking(true)?;
    stream.set_nodelay(true)
}

impl Link for TcpLink {
    fn send(&mut self, frame: &[u8]) {
        self.poll();
        if self.state() != LinkState::Open {
            return;
        }
        let len = u32::try_from(frame.len()).expect("frame too large");
        self.out.extend_from_slice(&len.to_le_bytes());
        self.out.extend_from_slice(frame);
        self.flush();
    }

    fn recv(&mut self) -> Option<Vec<u8>> {
        if self.frames.is_empty() {
            self.poll();
        }
        self.frames.pop_front()
    }

    fn state(&self) -> LinkState {
        match (self.closed, &self.stream) {
            (true, _) => LinkState::Closed,
            (false, Some(_)) => LinkState::Open,
            (false, None) => LinkState::Connecting,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    /// Polls both ends until `done` holds, failing after a few seconds.
    fn pump(a: &mut TcpLink, b: &mut TcpLink, mut done: impl FnMut(&mut TcpLink, &mut TcpLink) -> bool) {
        let start = Instant::now();
        while !done(a, b) {
            assert!(start.elapsed() < Duration::from_secs(10), "timed out");
            a.poll();
            b.poll();
            std::thread::sleep(Duration::from_millis(1));
        }
    }

    fn frame(i: usize, len: usize) -> Vec<u8> {
        (0..len).map(|j| (i * 31 + j) as u8).collect()
    }

    #[test]
    fn loopback_exchange_and_close() {
        let mut host = TcpLink::listen("127.0.0.1:0").unwrap();
        assert_eq!(host.state(), LinkState::Connecting);
        host.send(b"dropped while connecting");
        let port = host.local_port().unwrap();
        let mut guest = TcpLink::join(&format!("127.0.0.1:{port}")).unwrap();
        assert_eq!(guest.state(), LinkState::Open);
        pump(&mut host, &mut guest, |h, _| h.state() == LinkState::Open);

        // A large frame each way, then many small ones.
        let big = frame(7, 1 << 20);
        host.send(&big);
        guest.send(&big);
        for i in 0..1000 {
            host.send(&frame(i, i % 50));
            guest.send(&frame(i + 1, i % 70));
        }
        let (mut got_h, mut got_g) = (Vec::new(), Vec::new());
        pump(&mut host, &mut guest, |h, g| {
            got_h.extend(std::iter::from_fn(|| h.recv()));
            got_g.extend(std::iter::from_fn(|| g.recv()));
            got_h.len() == 1001 && got_g.len() == 1001
        });
        assert_eq!(got_g[0], big);
        assert_eq!(got_h[0], big);
        for i in 0..1000 {
            assert_eq!(got_g[i + 1], frame(i, i % 50));
            assert_eq!(got_h[i + 1], frame(i + 1, i % 70));
        }

        drop(guest);
        let start = Instant::now();
        while host.state() != LinkState::Closed {
            assert!(start.elapsed() < Duration::from_secs(10), "never closed");
            assert_eq!(host.recv(), None);
            std::thread::sleep(Duration::from_millis(1));
        }
    }
}
