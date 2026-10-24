//! Transports carrying opaque frames between the two peers.

use std::cell::RefCell;
use std::collections::VecDeque;
use std::rc::Rc;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LinkState {
    /// Waiting for the peer; frames sent now may be lost.
    Connecting,
    Open,
    /// The peer is gone (or the transport failed). Frames received before closing can still be
    /// read.
    Closed,
}

/// A frame transport to one peer.
///
/// Once `Open`, frames are delivered whole, reliably and in order. Frames sent while not `Open`
/// may be silently dropped: the `Session` keeps a log and replays it on the peer's `Hello`.
pub trait Link {
    fn send(&mut self, frame: &[u8]);
    /// The next received frame, if any. Callers poll this regularly, which also lets the
    /// transport make progress (accepting, flushing).
    fn recv(&mut self) -> Option<Vec<u8>>;
    fn state(&self) -> LinkState;
}

struct LocalShared {
    /// Frames waiting to be received by end 0 and end 1.
    inbox: [VecDeque<Vec<u8>>; 2],
    state: LinkState,
}

/// One end of an in-process link pair, for tests and local play. A clone is the same end, e.g.
/// kept to `open` or `close` the pair after the original was handed to a session.
#[derive(Clone)]
pub struct LocalLink {
    shared: Rc<RefCell<LocalShared>>,
    side: usize,
}

impl LocalLink {
    /// A connected pair.
    pub fn pair() -> (LocalLink, LocalLink) {
        Self::pair_in(LinkState::Open)
    }

    /// A pair whose peer has not arrived yet: frames are dropped until [`LocalLink::open`].
    pub fn pair_connecting() -> (LocalLink, LocalLink) {
        Self::pair_in(LinkState::Connecting)
    }

    fn pair_in(state: LinkState) -> (LocalLink, LocalLink) {
        let shared = Rc::new(RefCell::new(LocalShared { inbox: Default::default(), state }));
        let end = |side| LocalLink { shared: Rc::clone(&shared), side };
        (end(0), end(1))
    }

    /// The peer arrives: both ends become `Open`.
    pub fn open(&self) {
        self.shared.borrow_mut().state = LinkState::Open;
    }

    /// Both ends see `Closed`.
    pub fn close(&self) {
        self.shared.borrow_mut().state = LinkState::Closed;
    }
}

impl Link for LocalLink {
    fn send(&mut self, frame: &[u8]) {
        let mut s = self.shared.borrow_mut();
        if s.state == LinkState::Open {
            s.inbox[1 - self.side].push_back(frame.to_vec());
        }
    }

    fn recv(&mut self) -> Option<Vec<u8>> {
        self.shared.borrow_mut().inbox[self.side].pop_front()
    }

    fn state(&self) -> LinkState {
        self.shared.borrow().state
    }
}

struct QueueShared {
    incoming: VecDeque<Vec<u8>>,
    outgoing: VecDeque<Vec<u8>>,
    state: LinkState,
}

/// A link whose frames are moved by the frontend (e.g. JavaScript ferrying frames between WASM
/// and a `BroadcastChannel`) through its [`QueueLinkHandle`].
pub struct QueueLink(Rc<RefCell<QueueShared>>);

/// The frontend's side of a [`QueueLink`].
#[derive(Clone)]
pub struct QueueLinkHandle(Rc<RefCell<QueueShared>>);

impl QueueLink {
    /// A `Connecting` link and its handle.
    pub fn new() -> (QueueLink, QueueLinkHandle) {
        let shared = Rc::new(RefCell::new(QueueShared {
            incoming: VecDeque::new(),
            outgoing: VecDeque::new(),
            state: LinkState::Connecting,
        }));
        (QueueLink(Rc::clone(&shared)), QueueLinkHandle(shared))
    }
}

impl Link for QueueLink {
    fn send(&mut self, frame: &[u8]) {
        let mut s = self.0.borrow_mut();
        if s.state == LinkState::Open {
            s.outgoing.push_back(frame.to_vec());
        }
    }

    fn recv(&mut self) -> Option<Vec<u8>> {
        self.0.borrow_mut().incoming.pop_front()
    }

    fn state(&self) -> LinkState {
        self.0.borrow().state
    }
}

impl QueueLinkHandle {
    /// Hands a frame received from the peer to the link.
    pub fn push_incoming(&self, frame: Vec<u8>) {
        self.0.borrow_mut().incoming.push_back(frame);
    }

    /// The next frame to deliver to the peer.
    pub fn take_outgoing(&self) -> Option<Vec<u8>> {
        self.0.borrow_mut().outgoing.pop_front()
    }

    pub fn set_state(&self, state: LinkState) {
        self.0.borrow_mut().state = state;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn local_pair_delivers_in_order() {
        let (mut a, mut b) = LocalLink::pair();
        a.send(b"1");
        a.send(b"2");
        b.send(b"x");
        assert_eq!(b.recv().as_deref(), Some(&b"1"[..]));
        assert_eq!(b.recv().as_deref(), Some(&b"2"[..]));
        assert_eq!(b.recv(), None);
        assert_eq!(a.recv().as_deref(), Some(&b"x"[..]));
        a.close();
        assert_eq!(b.state(), LinkState::Closed);
    }

    #[test]
    fn local_connecting_drops_frames() {
        let (mut a, mut b) = LocalLink::pair_connecting();
        a.send(b"lost");
        assert_eq!(b.state(), LinkState::Connecting);
        b.open();
        a.send(b"kept");
        assert_eq!(a.state(), LinkState::Open);
        assert_eq!(b.recv().as_deref(), Some(&b"kept"[..]));
        assert_eq!(b.recv(), None);
    }

    #[test]
    fn queue_link_moves_frames() {
        let (mut link, handle) = QueueLink::new();
        link.send(b"lost");
        assert_eq!(handle.take_outgoing(), None);
        handle.set_state(LinkState::Open);
        link.send(b"out");
        handle.push_incoming(b"in".to_vec());
        assert_eq!(handle.take_outgoing().as_deref(), Some(&b"out"[..]));
        assert_eq!(link.recv().as_deref(), Some(&b"in"[..]));
        assert_eq!(link.state(), LinkState::Open);
    }
}
