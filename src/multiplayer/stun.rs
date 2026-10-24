//! A minimal STUN client (RFC 5389 Binding): asks a public server which address and port our
//! UDP socket appears to come from on the far side of NATs.

use std::io::{self, ErrorKind};
use std::net::{Ipv4Addr, Ipv6Addr, SocketAddr, ToSocketAddrs, UdpSocket};
use std::time::{Duration, Instant};

/// Public STUN servers tried by default.
pub const DEFAULT_SERVERS: &[&str] = &["stun.l.google.com:19302", "stun.cloudflare.com:3478"];

const MAGIC_COOKIE: u32 = 0x2112_A442;
const BINDING_REQUEST: u16 = 0x0001;
const BINDING_SUCCESS: u16 = 0x0101;
const ATTR_MAPPED_ADDRESS: u16 = 0x0001;
const ATTR_XOR_MAPPED_ADDRESS: u16 = 0x0020;
const RETRY_INTERVAL: Duration = Duration::from_millis(400);

/// A Binding request with the given transaction id.
pub fn binding_request(txid: &[u8; 12]) -> [u8; 20] {
    let mut req = [0u8; 20];
    req[0..2].copy_from_slice(&BINDING_REQUEST.to_be_bytes());
    // Message length 0: no attributes.
    req[4..8].copy_from_slice(&MAGIC_COOKIE.to_be_bytes());
    req[8..20].copy_from_slice(txid);
    req
}

/// The mapped address in a Binding success response to `txid`, or `None` if `msg` is anything
/// else.
pub fn parse_binding_response(msg: &[u8], txid: &[u8; 12]) -> Option<SocketAddr> {
    let header = msg.get(..20)?;
    let kind = u16::from_be_bytes([header[0], header[1]]);
    let len = u16::from_be_bytes([header[2], header[3]]) as usize;
    if kind != BINDING_SUCCESS
        || header[4..8] != MAGIC_COOKIE.to_be_bytes()
        || &header[8..20] != txid
    {
        return None;
    }
    let mut attrs = msg.get(20..20 + len)?;
    let mut mapped = None;
    while attrs.len() >= 4 {
        let kind = u16::from_be_bytes([attrs[0], attrs[1]]);
        let len = u16::from_be_bytes([attrs[2], attrs[3]]) as usize;
        let value = attrs.get(4..4 + len)?;
        match kind {
            ATTR_XOR_MAPPED_ADDRESS => return parse_address(value, Some(txid)),
            ATTR_MAPPED_ADDRESS => mapped = parse_address(value, None),
            _ => {}
        }
        // Attributes are padded to a multiple of 4 bytes.
        attrs = attrs.get((4 + len).next_multiple_of(4)..).unwrap_or(&[]);
    }
    mapped
}

/// A (XOR-)MAPPED-ADDRESS value; `xor` holds the transaction id for the XOR form.
fn parse_address(value: &[u8], xor: Option<&[u8; 12]>) -> Option<SocketAddr> {
    let family = *value.get(1)?;
    let mut port = u16::from_be_bytes([*value.get(2)?, *value.get(3)?]);
    let mut key = [0u8; 16];
    key[..4].copy_from_slice(&MAGIC_COOKIE.to_be_bytes());
    if let Some(txid) = xor {
        key[4..].copy_from_slice(txid);
        port ^= (MAGIC_COOKIE >> 16) as u16;
    }
    let unmask = |bytes: &[u8], out: &mut [u8]| {
        for (i, (o, b)) in out.iter_mut().zip(bytes).enumerate() {
            *o = if xor.is_some() { b ^ key[i] } else { *b };
        }
    };
    match family {
        0x01 => {
            let mut ip = [0u8; 4];
            unmask(value.get(4..8)?, &mut ip);
            Some(SocketAddr::new(Ipv4Addr::from(ip).into(), port))
        }
        0x02 => {
            let mut ip = [0u8; 16];
            unmask(value.get(4..20)?, &mut ip);
            Some(SocketAddr::new(Ipv6Addr::from(ip).into(), port))
        }
        _ => None,
    }
}

/// Asks the IPv4 STUN `servers` (`host:port`) for `socket`'s public address, retrying until
/// `timeout`. Returns the address and the server that answered. Datagrams that are not the
/// answer are discarded, so call this before the socket carries anything else. Leaves the
/// socket blocking without a read timeout.
pub fn public_addr(
    socket: &UdpSocket,
    servers: &[&str],
    timeout: Duration,
) -> io::Result<(SocketAddr, SocketAddr)> {
    let targets: Vec<SocketAddr> = servers
        .iter()
        .filter_map(|s| s.to_socket_addrs().ok())
        .flatten()
        .filter(SocketAddr::is_ipv4)
        .collect();
    if targets.is_empty() {
        return Err(io::Error::new(ErrorKind::NotFound, "no STUN server could be resolved"));
    }
    let txid = super::p2p::random_bytes::<12>();
    let request = binding_request(&txid);
    let start = Instant::now();
    let mut buf = [0u8; 1500];
    let result = loop {
        let elapsed = start.elapsed();
        if elapsed >= timeout {
            break Err(io::Error::new(ErrorKind::TimedOut, "no STUN server answered"));
        }
        for t in &targets {
            let _ = socket.send_to(&request, t);
        }
        let round_end = Instant::now() + RETRY_INTERVAL.min(timeout - elapsed);
        let mut found = None;
        while let Some(left) = round_end.checked_duration_since(Instant::now()).filter(|d| !d.is_zero()) {
            socket.set_read_timeout(Some(left))?;
            match socket.recv_from(&mut buf) {
                Ok((n, from)) => {
                    if let Some(addr) = parse_binding_response(&buf[..n], &txid) {
                        found = Some((addr, from));
                        break;
                    }
                }
                Err(e) if matches!(e.kind(), ErrorKind::WouldBlock | ErrorKind::TimedOut) => break,
                // Windows reports an earlier ICMP "port unreachable" this way.
                Err(e) if e.kind() == ErrorKind::ConnectionReset => {}
                Err(e) => return Err(e),
            }
        }
        if let Some(addr) = found {
            break Ok(addr);
        }
    };
    socket.set_read_timeout(None)?;
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    /// RFC 5769 section 2.2: sample IPv4 response (the integrity and fingerprint values are not
    /// checked by this client, so they are zeroed here).
    #[test]
    fn rfc5769_ipv4_response() {
        let txid = [0xb7, 0xe7, 0xa7, 0x01, 0xbc, 0x34, 0xd6, 0x86, 0xfa, 0x87, 0xdf, 0xae];
        let mut msg = vec![0x01, 0x01, 0x00, 0x3c, 0x21, 0x12, 0xa4, 0x42];
        msg.extend_from_slice(&txid);
        msg.extend_from_slice(&[0x80, 0x22, 0x00, 0x0b]);
        msg.extend_from_slice(b"test vector ");
        msg.extend_from_slice(&[0x00, 0x20, 0x00, 0x08, 0x00, 0x01, 0xa1, 0x47, 0xe1, 0x12, 0xa6, 0x43]);
        msg.extend_from_slice(&[0x00, 0x08, 0x00, 0x14]);
        msg.extend_from_slice(&[0; 20]);
        msg.extend_from_slice(&[0x80, 0x28, 0x00, 0x04, 0, 0, 0, 0]);
        assert_eq!(msg.len(), 20 + 0x3c);
        let addr = parse_binding_response(&msg, &txid).unwrap();
        assert_eq!(addr, "192.0.2.1:32853".parse().unwrap());

        let mut other = txid;
        other[0] ^= 1;
        assert_eq!(parse_binding_response(&msg, &other), None);
        assert_eq!(parse_binding_response(&msg[..30], &txid), None);
    }

    /// RFC 5769 section 2.3: sample IPv6 response.
    #[test]
    fn rfc5769_ipv6_response() {
        let txid = [0xb7, 0xe7, 0xa7, 0x01, 0xbc, 0x34, 0xd6, 0x86, 0xfa, 0x87, 0xdf, 0xae];
        let mut msg = vec![0x01, 0x01, 0x00, 0x18, 0x21, 0x12, 0xa4, 0x42];
        msg.extend_from_slice(&txid);
        msg.extend_from_slice(&[0x00, 0x20, 0x00, 0x14, 0x00, 0x02, 0xa1, 0x47]);
        msg.extend_from_slice(&[
            0x01, 0x13, 0xa9, 0xfa, 0xa5, 0xd3, 0xf1, 0x79, 0xbc, 0x25, 0xf4, 0xb5, 0xbe, 0xd2,
            0xb9, 0xd9,
        ]);
        let addr = parse_binding_response(&msg, &txid).unwrap();
        assert_eq!(addr, "[2001:db8:1234:5678:11:2233:4455:6677]:32853".parse().unwrap());
    }

    #[test]
    fn request_layout() {
        let req = binding_request(&[7; 12]);
        assert_eq!(&req[..8], &[0x00, 0x01, 0x00, 0x00, 0x21, 0x12, 0xa4, 0x42]);
        assert_eq!(&req[8..], &[7; 12]);
    }

    /// Needs the internet: `cargo test -- --ignored public_stun`.
    #[test]
    #[ignore]
    fn public_stun_servers_answer() {
        for server in DEFAULT_SERVERS {
            let socket = UdpSocket::bind("0.0.0.0:0").unwrap();
            let (addr, _) = public_addr(&socket, &[server], Duration::from_secs(5)).unwrap();
            println!("{server}: we are {addr}");
            assert!(addr.is_ipv4() && addr.port() != 0);
        }
    }

    /// A fake server on loopback answers with the classic MAPPED-ADDRESS.
    #[test]
    fn asks_a_server() {
        let server = UdpSocket::bind("127.0.0.1:0").unwrap();
        let server_addr = server.local_addr().unwrap().to_string();
        let thread = std::thread::spawn(move || {
            let mut buf = [0u8; 100];
            let (n, from) = server.recv_from(&mut buf).unwrap();
            assert_eq!(n, 20);
            let SocketAddr::V4(from4) = from else { panic!() };
            let mut resp = vec![0x01, 0x01, 0x00, 0x0c];
            resp.extend_from_slice(&buf[4..20]);
            resp.extend_from_slice(&[0x00, 0x01, 0x00, 0x08, 0x00, 0x01]);
            resp.extend_from_slice(&from4.port().to_be_bytes());
            resp.extend_from_slice(&from4.ip().octets());
            server.send_to(&resp, from).unwrap();
        });
        let client = UdpSocket::bind("127.0.0.1:0").unwrap();
        let (addr, from) = public_addr(&client, &[&server_addr], Duration::from_secs(5)).unwrap();
        assert_eq!(addr, client.local_addr().unwrap());
        assert_eq!(from.to_string(), server_addr);
        thread.join().unwrap();
    }
}
