#!/usr/bin/env node
// A rendezvous server: lets two (or more) instances of an app find each other and swap the few
// messages they need to connect directly, for example a WebRTC offer and answer. It knows nothing
// about the apps that use it. It also hands out credentials for a TURN relay (such as coturn),
// which tunnels the traffic of peers that have no direct path between them.
//
// A room belongs to one app, named by the client, and holds write-once slots. The first request
// to a room fixes its app version; any request with another version gets 409 version_mismatch,
// so peers on different builds learn it instead of failing later in some app-specific way.
//
//   PUT    /v1/:app/:version/:room/:slot          store the body in the slot (write-once)
//   GET    /v1/:app/:version/:room/:slot?wait=S   fetch it, long-polling up to S seconds (max 30)
//   DELETE /v1/:app/:version/:room                drop the room once the peers are connected
//   GET    /v1/ice-servers                        RTCIceServer list: STUN, plus TURN credentials
//   GET    /v1/health                             liveness check
//
// Names are 1-64 of A-Z a-z 0-9 . _ -. A slot's body is opaque (at most 64 KiB) and comes back
// with the Content-Type it was stored with. A GET answers 200 with the body, or 204 if the slot
// is still empty when the wait ends. Errors are JSON {"error": code}: slot_taken (409, PUT to a
// filled slot), version_mismatch (409, with the room's "version"), full (503, too many rooms or
// bytes held). A room is dropped ROOM_TTL_MS after the last request that touched it.
//
// Two-peer handshake, as the SpecRel game does it: the host PUTs "offer" and long-polls
// "answer"; the joiner long-polls "offer" and PUTs "answer"; the host DELETEs the room.
//
// With TURN_SECRET and TURN_URLS set (e.g. `turn:relay.example.com:3478?transport=udp,...`),
// /v1/ice-servers adds time-limited credentials for a TURN server running with
// `use-auth-secret` and the same `static-auth-secret` (the TURN REST API scheme).
//
// Usage: node server.mjs [--port 8787] [--host 0.0.0.0]
// Zero dependencies; Node 18 or later.

import http from "node:http";
import crypto from "node:crypto";

const args = process.argv.slice(2);
const opt = (name, fallback) => {
  const i = args.indexOf(`--${name}`);
  return i >= 0 && i + 1 < args.length ? args[i + 1] : fallback;
};
const PORT = Number(opt("port", process.env.PORT || 8787));
const HOST = opt("host", "0.0.0.0");

const ROOM_TTL_MS = 10 * 60 * 1000;
const MAX_WAIT_S = 30;
const MAX_BODY = 64 * 1024;
const MAX_SLOTS = 16;
const MAX_ROOMS = 10000;
const MAX_TOTAL_BYTES = 256 * 1024 * 1024;
const NAME_RE = /^[A-Za-z0-9._-]{1,64}$/;

const STUN_URLS = (process.env.STUN_URLS || "stun:stun.l.google.com:19302,stun:stun.cloudflare.com:3478")
  .split(",").map((u) => u.trim()).filter(Boolean);
const TURN_SECRET = process.env.TURN_SECRET || null;
const TURN_URLS = (process.env.TURN_URLS || "").split(",").map((u) => u.trim()).filter(Boolean);
const TURN_TTL_S = 24 * 60 * 60;

// TURN REST API credentials: the username is the expiry time, the password its HMAC.
function iceServers() {
  const servers = STUN_URLS.length ? [{ urls: STUN_URLS }] : [];
  if (TURN_SECRET && TURN_URLS.length) {
    const username = `${Math.floor(Date.now() / 1000) + TURN_TTL_S}`;
    const credential = crypto.createHmac("sha1", TURN_SECRET).update(username).digest("base64");
    servers.push({ urls: TURN_URLS, username, credential });
  }
  return servers;
}

// "app/room" -> { version, slots: Map(slot -> {type, body}), waiters: Map(slot -> Set), expires }
const rooms = new Map();
let totalBytes = 0;

function touchRoom(app, version, name) {
  const key = `${app}/${name}`;
  let room = rooms.get(key);
  if (!room) {
    if (rooms.size >= MAX_ROOMS) return { error: "full" };
    room = { version, slots: new Map(), waiters: new Map(), expires: 0 };
    rooms.set(key, room);
  }
  if (room.version !== version) return { error: "version_mismatch", version: room.version };
  room.expires = Date.now() + ROOM_TTL_MS;
  return { room };
}

function dropRoom(key) {
  const room = rooms.get(key);
  if (!room) return;
  rooms.delete(key);
  for (const { body } of room.slots.values()) totalBytes -= body.length;
  for (const set of room.waiters.values()) for (const w of set) w(null);
}

setInterval(() => {
  const now = Date.now();
  for (const [key, room] of rooms) if (room.expires <= now) dropRoom(key);
}, 5000).unref();

function log(...parts) {
  console.log(new Date().toISOString(), ...parts);
}

function cors(res) {
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Methods", "GET, PUT, DELETE, OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type");
  res.setHeader("Cache-Control", "no-store");
}

function send(res, status, body) {
  if (res.writableEnded) return;
  if (body === undefined) {
    res.writeHead(status);
    res.end();
    return;
  }
  res.writeHead(status, { "Content-Type": "application/json" });
  res.end(JSON.stringify(body));
}

function sendSlot(res, entry) {
  if (res.writableEnded) return;
  if (!entry) return send(res, 204);
  res.writeHead(200, { "Content-Type": entry.type });
  res.end(entry.body);
}

function readBody(req) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    let size = 0;
    req.on("data", (c) => {
      size += c.length;
      if (size > MAX_BODY) return reject(Object.assign(new Error("too_large"), { status: 413 }));
      chunks.push(c);
    });
    req.on("end", () => resolve(Buffer.concat(chunks)));
    req.on("error", reject);
  });
}

async function putSlot(req, res, app, version, name, slot) {
  const body = await readBody(req);
  if (totalBytes + body.length > MAX_TOTAL_BYTES) return send(res, 503, { error: "full" });
  const { room, ...error } = touchRoom(app, version, name);
  if (!room) return send(res, error.error === "full" ? 503 : 409, error);
  if (room.slots.has(slot)) return send(res, 409, { error: "slot_taken" });
  if (room.slots.size >= MAX_SLOTS) return send(res, 409, { error: "too_many_slots" });
  const entry = { type: req.headers["content-type"] || "application/octet-stream", body };
  room.slots.set(slot, entry);
  totalBytes += body.length;
  log(`${app}@${version} ${name}/${slot}: ${body.length} bytes`);
  const waiters = room.waiters.get(slot);
  if (waiters) for (const w of waiters) w(entry);
  room.waiters.delete(slot);
  send(res, 200, { ok: true });
}

function getSlot(res, app, version, name, slot, url) {
  const { room, ...error } = touchRoom(app, version, name);
  if (!room) return send(res, error.error === "full" ? 503 : 409, error);
  if (room.slots.has(slot)) return sendSlot(res, room.slots.get(slot));
  const wait = Math.min(MAX_WAIT_S, Math.max(0, Number(url.searchParams.get("wait")) || 0));
  if (wait === 0) return send(res, 204);
  if (!room.waiters.has(slot)) room.waiters.set(slot, new Set());
  const waiters = room.waiters.get(slot);
  const waiter = (entry) => {
    clearTimeout(timer);
    sendSlot(res, entry);
  };
  const timer = setTimeout(() => {
    waiters.delete(waiter);
    send(res, 204);
  }, wait * 1000);
  waiters.add(waiter);
  res.on("close", () => {
    clearTimeout(timer);
    waiters.delete(waiter);
  });
}

function deleteRoom(res, app, version, name) {
  const room = rooms.get(`${app}/${name}`);
  if (room && room.version !== version) return send(res, 409, { error: "version_mismatch", version: room.version });
  dropRoom(`${app}/${name}`);
  send(res, 200, { ok: true });
}

const server = http.createServer(async (req, res) => {
  cors(res);
  const url = new URL(req.url, "http://localhost");
  try {
    if (req.method === "OPTIONS") return send(res, 204);
    if (url.pathname === "/v1/ice-servers") return send(res, 200, { iceServers: iceServers() });
    if (url.pathname === "/v1/health") return send(res, 200, { ok: true, rooms: rooms.size, bytes: totalBytes });
    let parts;
    try {
      parts = url.pathname.split("/").slice(1).map(decodeURIComponent);
    } catch {
      return send(res, 400, { error: "bad_name" });
    }
    if (parts[0] !== "v1" || (parts.length !== 4 && parts.length !== 5)) return send(res, 404, { error: "not_found" });
    const [, app, version, name, slot] = parts;
    if (!parts.slice(1).every((p) => NAME_RE.test(p))) return send(res, 400, { error: "bad_name" });
    if (slot === undefined) {
      if (req.method !== "DELETE") return send(res, 405, { error: "method_not_allowed" });
      return deleteRoom(res, app, version, name);
    }
    if (req.method === "PUT") return await putSlot(req, res, app, version, name, slot);
    if (req.method === "GET") return getSlot(res, app, version, name, slot, url);
    return send(res, 405, { error: "method_not_allowed" });
  } catch (e) {
    // Close the connection rather than read the rest of a body we refused.
    res.setHeader("Connection", "close");
    send(res, e.status || 500, { error: e.status ? e.message : "internal" });
  }
});

server.listen(PORT, HOST, () => {
  log(`Rendezvous server on http://${HOST}:${PORT}` + (TURN_SECRET && TURN_URLS.length ? `, TURN relay ${TURN_URLS.join(" ")}` : ""));
});
