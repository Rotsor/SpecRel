//! A two-player session over a [`Link`]: `Hello` handshake, sequence numbers, replay of frames
//! the peer may have missed, and decoded [`Event`]s.
//!
//! Every frame is `seq: u32 LE` followed by one encoded [`Message`]. Until the peer's `Hello`
//! arrives, the session logs what it sends and replays the whole log when the peer says hello
//! (and when the link opens). The receiver accepts only the next expected `seq`, so replayed
//! duplicates are dropped and frames after a gap wait for the replay.

use super::link::{Link, LinkState};
use super::protocol::{Message, PROTOCOL_VERSION};
use crate::dynamic::helm::ControlMode;
use crate::dynamic::{PlayerId, ShipSample};
use crate::minkowski::Vec4;
use crate::renderer::Frontier;

/// Something the peer did, in the order it was sent.
#[derive(Clone, Debug)]
pub enum Event {
    /// The peer said hello with a compatible protocol.
    Joined(PlayerId),
    Track { sample: ShipSample, waiting: bool },
    /// The peer flies with `controls`.
    Controls(ControlMode),
    Laser { at: Frontier, dir: [f64; 3] },
    Flash(Vec4),
    /// The peer probes the round trip; answer with `Message::Pong { sent }`.
    Ping { sent: f64 },
    /// The peer answered our `Ping` stamped `sent`.
    Pong { sent: f64 },
    /// The peer quit or the link closed. Emitted once; nothing follows it.
    Left,
    /// The peer speaks another protocol version. Followed by `Left`.
    Incompatible { theirs: u16 },
    /// A frame could not be decoded and was skipped.
    Error(String),
}

pub struct Session {
    local: PlayerId,
    link: Box<dyn Link>,
    next_seq: u32,
    /// Frames sent so far, kept until the peer's `Hello` arrives.
    log: Option<Vec<Vec<u8>>>,
    was_open: bool,
    expected: u32,
    peer: Option<PlayerId>,
    joined: bool,
    left: bool,
}

impl Session {
    /// Starts a session and says hello.
    pub fn new(local: PlayerId, link: Box<dyn Link>) -> Self {
        let mut s = Self {
            local,
            link,
            next_seq: 0,
            log: Some(Vec::new()),
            was_open: false,
            expected: 0,
            peer: None,
            joined: false,
            left: false,
        };
        s.send(&Message::Hello { version: PROTOCOL_VERSION, player: local });
        s.was_open = s.link_state() == LinkState::Open;
        s
    }

    pub fn local(&self) -> PlayerId {
        self.local
    }

    /// The peer's player id, once its `Hello` arrived.
    pub fn peer(&self) -> Option<PlayerId> {
        self.peer
    }

    /// The peer said hello with a compatible protocol (and may have left since).
    pub fn peer_joined(&self) -> bool {
        self.joined
    }

    pub fn peer_left(&self) -> bool {
        self.left
    }

    pub fn link_state(&self) -> LinkState {
        self.link.state()
    }

    pub fn send(&mut self, msg: &Message) {
        let mut frame = self.next_seq.to_le_bytes().to_vec();
        msg.encode(&mut frame);
        self.next_seq += 1;
        self.link.send(&frame);
        if let Some(log) = &mut self.log {
            log.push(frame);
        }
    }

    /// Tells the peer we quit.
    pub fn leave(&mut self) {
        self.send(&Message::Bye);
    }

    /// Receives everything available and returns what the peer did.
    pub fn poll(&mut self) -> Vec<Event> {
        let mut events = Vec::new();
        while let Some(frame) = self.link.recv() {
            if !self.left {
                self.receive(&frame, &mut events);
            }
        }
        let state = self.link_state();
        if state == LinkState::Open && !self.was_open {
            self.replay();
        }
        self.was_open = state == LinkState::Open;
        if state == LinkState::Closed {
            self.peer_gone(&mut events);
        }
        events
    }

    fn receive(&mut self, frame: &[u8], events: &mut Vec<Event>) {
        let Some((seq, body)) = frame.split_first_chunk::<4>() else {
            events.push(Event::Error("frame without sequence number".into()));
            return;
        };
        if u32::from_le_bytes(*seq) != self.expected {
            return; // a replayed duplicate, or past a gap that a replay will fill
        }
        self.expected += 1;
        match Message::decode(body) {
            Ok(msg) => self.handle(msg, events),
            Err(e) => events.push(Event::Error(e.to_string())),
        }
    }

    fn handle(&mut self, msg: Message, events: &mut Vec<Event>) {
        match msg {
            Message::Hello { version, player } => {
                if self.peer.is_some() {
                    return;
                }
                self.peer = Some(player);
                if version != PROTOCOL_VERSION {
                    events.push(Event::Incompatible { theirs: version });
                    self.peer_gone(events);
                    return;
                }
                self.replay();
                self.log = None;
                self.joined = true;
                events.push(Event::Joined(player));
            }
            _ if !self.joined => events.push(Event::Error(format!("{msg:?} before Hello"))),
            Message::Track { sample, waiting } => events.push(Event::Track { sample, waiting }),
            Message::Controls { controls } => events.push(Event::Controls(controls)),
            Message::Laser { at, dir } => events.push(Event::Laser { at, dir }),
            Message::Flash { at } => events.push(Event::Flash(at)),
            Message::Ping { sent } => events.push(Event::Ping { sent }),
            Message::Pong { sent } => events.push(Event::Pong { sent }),
            Message::Bye => self.peer_gone(events),
        }
    }

    fn replay(&mut self) {
        for frame in self.log.iter().flatten() {
            self.link.send(frame);
        }
    }

    fn peer_gone(&mut self, events: &mut Vec<Event>) {
        if !self.left {
            self.left = true;
            events.push(Event::Left);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::multiplayer::link::LocalLink;

    fn track(i: u32) -> Message {
        let sample = ShipSample { t: i as f64, pos: [0.0; 3], vel: [0.0; 3], tau: i as f64 };
        Message::Track { sample, waiting: false }
    }

    /// The `t` of every `Track`, `-1` for `Joined`; panics on anything else.
    fn tracks(events: &[Event]) -> Vec<i64> {
        events
            .iter()
            .map(|e| match e {
                Event::Joined(_) => -1,
                Event::Track { sample, .. } => sample.t as i64,
                e => panic!("unexpected {e:?}"),
            })
            .collect()
    }

    fn sessions() -> (Session, Session, LocalLink) {
        let (a, b) = LocalLink::pair();
        let control = a.clone();
        (Session::new(0, Box::new(a)), Session::new(1, Box::new(b)), control)
    }

    #[test]
    fn hello_and_ordered_messages() {
        let (mut a, mut b, _) = sessions();
        a.send(&track(1));
        a.send(&Message::Flash { at: Vec4::new(1.0, 2.0, 3.0, 4.0) });
        a.send(&track(2));
        let ev = b.poll();
        assert!(matches!(ev[0], Event::Joined(0)));
        assert!(matches!(ev[1], Event::Track { .. }));
        assert!(matches!(ev[2], Event::Flash(v) if v == Vec4::new(1.0, 2.0, 3.0, 4.0)));
        assert!(matches!(ev[3], Event::Track { .. }));
        assert_eq!(ev.len(), 4);
        assert_eq!(b.peer(), Some(0));
        assert!(b.peer_joined() && !b.peer_left());
        // A sees B's hello; its replay to B is dropped as duplicates.
        assert!(matches!(a.poll()[..], [Event::Joined(1)]));
        assert!(b.poll().is_empty());
        a.send(&track(3));
        assert_eq!(tracks(&b.poll()), [3]);
    }

    #[test]
    fn late_peer_gets_everything_once() {
        let (la, lb) = LocalLink::pair_connecting();
        let mut a = Session::new(0, Box::new(la));
        a.send(&track(1)); // lost: nobody is there yet
        a.send(&track(2));
        lb.open();
        a.send(&track(3)); // delivered ahead of the frames B missed
        assert!(a.poll().is_empty()); // sees the link open and replays
        a.send(&track(4));
        let mut b = Session::new(1, Box::new(lb));
        assert_eq!(tracks(&b.poll()), [-1, 1, 2, 3, 4]);
        assert!(matches!(a.poll()[..], [Event::Joined(1)])); // replays again
        a.send(&track(5));
        assert_eq!(tracks(&b.poll()), [5]);
        b.send(&track(9));
        assert_eq!(tracks(&a.poll()), [9]);
    }

    #[test]
    fn hellos_lost_on_both_sides_are_replayed() {
        let (la, lb) = LocalLink::pair_connecting();
        let control = la.clone();
        let mut a = Session::new(0, Box::new(la));
        let mut b = Session::new(1, Box::new(lb));
        a.send(&track(1));
        b.send(&track(2));
        assert!(a.poll().is_empty() && b.poll().is_empty());
        control.open();
        assert!(a.poll().is_empty());
        assert_eq!(tracks(&b.poll()), [-1, 1]);
        assert_eq!(tracks(&a.poll()), [-1, 2]);
    }

    #[test]
    fn bye_means_left() {
        let (mut a, mut b, _) = sessions();
        b.poll();
        a.leave();
        a.send(&track(1)); // ignored after Bye
        assert!(matches!(b.poll()[..], [Event::Left]));
        assert!(b.peer_left());
        assert!(b.poll().is_empty());
    }

    #[test]
    fn closed_link_means_left_once() {
        let (mut a, mut b, control) = sessions();
        a.send(&track(1));
        control.close();
        // Frames that arrived before closing are still delivered.
        assert_eq!(tracks(&b.poll()[..2]), [-1, 1]);
        assert!(b.peer_left());
        assert!(b.poll().is_empty());
        // B's hello reached A before the link closed.
        assert!(matches!(a.poll()[..], [Event::Joined(1), Event::Left]));
        assert_eq!(a.link_state(), LinkState::Closed);
    }

    #[test]
    fn version_mismatch_is_incompatible() {
        let (mut la, lb) = LocalLink::pair();
        let mut frame = 0u32.to_le_bytes().to_vec();
        Message::Hello { version: PROTOCOL_VERSION + 1, player: 1 }.encode(&mut frame);
        la.send(&frame);
        let mut b = Session::new(0, Box::new(lb));
        let ev = b.poll();
        assert!(matches!(ev[..], [Event::Incompatible { theirs }, Event::Left] if theirs == PROTOCOL_VERSION + 1));
        assert!(!b.peer_joined() && b.peer_left());
    }

    #[test]
    fn garbage_is_reported_and_skipped() {
        let (mut la, lb) = LocalLink::pair();
        let mut b = Session::new(1, Box::new(lb));
        la.send(&[1, 2]);
        let mut frame = 0u32.to_le_bytes().to_vec();
        Message::Hello { version: PROTOCOL_VERSION, player: 0 }.encode(&mut frame);
        la.send(&frame);
        la.send(&[1, 0, 0, 0, 99]);
        let mut frame = 2u32.to_le_bytes().to_vec();
        track(7).encode(&mut frame);
        la.send(&frame);
        let ev = b.poll();
        assert!(matches!(ev[..], [Event::Error(_), Event::Joined(0), Event::Error(_), Event::Track { .. }]));
    }
}
