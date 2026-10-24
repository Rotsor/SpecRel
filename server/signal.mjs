#!/usr/bin/env node
// SpecRel 4D signalling server for web peer-to-peer matches.
//
// Two browsers in the same room swap one WebRTC offer and one answer through this server, then
// talk to each other directly over a data channel. The SDP carries every ICE candidate the
// browser gathered, so no further signalling is needed and no game traffic passes through here.
//
//   POST /rooms/:room/offer          host stores its offer ({type, sdp} JSON)
//   GET  /rooms/:room/offer?wait=S   joiner fetches it, long-polling up to S seconds
//   POST /rooms/:room/answer         joiner stores its answer
//   GET  /rooms/:room/answer?wait=S  host fetches it, long-polling
//   GET  /health                     liveness check
//
// A timed-out long poll answers 204. Posting an offer to a room that already has one, or an
// answer to a room without an offer or with an answer already, answers 409. Rooms are dropped
// ROOM_TTL_MS after creation, or shortly after the host has fetched the answer.
//
// With `--static DIR` the server also serves DIR (e.g. `web/`), so the game and the signalling
// share one origin and the page needs no `signal=` parameter.
//
// Usage: node server/signal.mjs [--port 8787] [--host 0.0.0.0] [--static web]
// Zero dependencies; Node 18 or later.

import http from "node:http";
import fs from "node:fs";
import path from "node:path";

const args = process.argv.slice(2);
const opt = (name, fallback) => {
  const i = args.indexOf(`--${name}`);
  return i >= 0 && i + 1 < args.length ? args[i + 1] : fallback;
};
const PORT = Number(opt("port", process.env.PORT || 8787));
const HOST = opt("host", "0.0.0.0");
const STATIC_DIR = opt("static", null);

const ROOM_TTL_MS = 10 * 60 * 1000;
const ROOM_LINGER_MS = 30 * 1000;
const MAX_WAIT_S = 30;
const MAX_BODY = 64 * 1024;
const MAX_ROOMS = 10000;
const ROOM_RE = /^[A-Za-z0-9_-]{1,64}$/;

// room name -> { offer, answer, waiters: {offer: Set, answer: Set}, expires }
const rooms = new Map();

function getRoom(name, create) {
  let room = rooms.get(name);
  if (!room && create) {
    if (rooms.size >= MAX_ROOMS) return null;
    room = { offer: null, answer: null, waiters: { offer: new Set(), answer: new Set() }, expires: Date.now() + ROOM_TTL_MS };
    rooms.set(name, room);
  }
  return room;
}

function dropRoom(name) {
  const room = rooms.get(name);
  if (!room) return;
  rooms.delete(name);
  for (const kind of ["offer", "answer"]) for (const w of room.waiters[kind]) w(null);
}

setInterval(() => {
  const now = Date.now();
  for (const [name, room] of rooms) if (room.expires <= now) dropRoom(name);
}, 5000).unref();

function log(...parts) {
  console.log(new Date().toISOString(), ...parts);
}

function cors(res) {
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
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

function readBody(req) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    let size = 0;
    req.on("data", (c) => {
      size += c.length;
      if (size > MAX_BODY) {
        reject(Object.assign(new Error("body too large"), { status: 413 }));
        req.destroy();
        return;
      }
      chunks.push(c);
    });
    req.on("end", () => resolve(Buffer.concat(chunks).toString("utf8")));
    req.on("error", reject);
  });
}

function parseDescription(text, expectedType) {
  let desc;
  try {
    desc = JSON.parse(text);
  } catch {
    return null;
  }
  if (!desc || desc.type !== expectedType || typeof desc.sdp !== "string") return null;
  return { type: desc.type, sdp: desc.sdp };
}

async function handleRoom(req, res, name, kind, url) {
  if (req.method === "POST") {
    const desc = parseDescription(await readBody(req), kind);
    if (!desc) return send(res, 400, { error: `expected {"type":"${kind}","sdp":...}` });
    const room = getRoom(name, kind === "offer");
    if (!room) return send(res, kind === "offer" ? 503 : 409, { error: kind === "offer" ? "server full" : "no such room" });
    if (kind === "answer" && !room.offer) return send(res, 409, { error: "room has no offer" });
    if (room[kind]) return send(res, 409, { error: kind === "offer" ? "room is already hosted" : "room already has an answer" });
    room[kind] = desc;
    log(`room ${name}: ${kind} stored`);
    for (const w of room.waiters[kind]) w(desc);
    room.waiters[kind].clear();
    return send(res, 200, { ok: true });
  }

  // GET: return the description now, or long-poll for it.
  const room = getRoom(name, false);
  const finish = (desc) => {
    if (!desc) return send(res, 204);
    send(res, 200, desc);
    // Once the host has the answer the room has done its job.
    if (kind === "answer") {
      const r = rooms.get(name);
      if (r) r.expires = Math.min(r.expires, Date.now() + ROOM_LINGER_MS);
    }
  };
  if (room && room[kind]) return finish(room[kind]);
  const wait = Math.min(MAX_WAIT_S, Math.max(0, Number(url.searchParams.get("wait")) || 0));
  if (wait === 0) return send(res, 204);
  // Waiting for an offer may create the room (the joiner can arrive first); waiting for an
  // answer needs the host's offer to exist.
  const target = getRoom(name, kind === "offer");
  if (!target) return send(res, kind === "offer" ? 503 : 404, { error: kind === "offer" ? "server full" : "no such room" });
  const waiter = (desc) => {
    clearTimeout(timer);
    finish(desc);
  };
  const timer = setTimeout(() => {
    target.waiters[kind].delete(waiter);
    send(res, 204);
  }, wait * 1000);
  target.waiters[kind].add(waiter);
  res.on("close", () => {
    clearTimeout(timer);
    target.waiters[kind].delete(waiter);
  });
}

const MIME = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript",
  ".mjs": "text/javascript",
  ".css": "text/css",
  ".json": "application/json",
  ".wasm": "application/wasm",
  ".png": "image/png",
  ".svg": "image/svg+xml",
  ".ico": "image/x-icon",
};

function serveStatic(req, res, url) {
  if (!STATIC_DIR || (req.method !== "GET" && req.method !== "HEAD")) return send(res, 404, { error: "not found" });
  const root = path.resolve(STATIC_DIR);
  let rel = decodeURIComponent(url.pathname);
  if (rel.endsWith("/")) rel += "index.html";
  const file = path.resolve(root, "." + rel);
  if (file !== root && !file.startsWith(root + path.sep)) return send(res, 403, { error: "forbidden" });
  fs.stat(file, (err, st) => {
    if (err || !st.isFile()) return send(res, 404, { error: "not found" });
    res.writeHead(200, {
      "Content-Type": MIME[path.extname(file).toLowerCase()] || "application/octet-stream",
      "Content-Length": st.size,
    });
    if (req.method === "HEAD") return res.end();
    fs.createReadStream(file).pipe(res);
  });
}

const server = http.createServer(async (req, res) => {
  cors(res);
  const url = new URL(req.url, "http://localhost");
  try {
    if (req.method === "OPTIONS") return send(res, 204);
    if (url.pathname === "/health") return send(res, 200, { ok: true, rooms: rooms.size });
    const m = url.pathname.match(/^\/rooms\/([^/]+)\/(offer|answer)$/);
    if (m) {
      if (req.method !== "GET" && req.method !== "POST") return send(res, 405, { error: "method not allowed" });
      const name = decodeURIComponent(m[1]);
      if (!ROOM_RE.test(name)) return send(res, 400, { error: "room names are 1-64 of A-Z a-z 0-9 _ -" });
      return await handleRoom(req, res, name, m[2], url);
    }
    return serveStatic(req, res, url);
  } catch (e) {
    send(res, e.status || 500, { error: e.message });
  }
});

server.listen(PORT, HOST, () => {
  log(`SpecRel signalling server on http://${HOST}:${PORT}` + (STATIC_DIR ? `, serving ${path.resolve(STATIC_DIR)}` : ""));
});
