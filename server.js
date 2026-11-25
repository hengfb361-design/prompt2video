// Prompt2Video — Render web service.
//
// A Render static site cannot answer /api/*, so this small Node server does the
// same job the Cloudflare Worker does on Sites: it serves public/ and proxies
// the visitor's Colab Gradio app (tunnel link, status, generation SSE, video).
// No dependencies, no database: the link lives in this instance's memory (and a
// temp file), and the Colab app re-publishes it every 5 minutes, which also
// keeps a free Render instance awake.

import fs from "node:fs";
import http from "node:http";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const PUBLIC_DIR = path.join(__dirname, "public");
const PORT = Number(process.env.PORT || 3000);
const ALLOWED = ["gradio.live", "gradio.dev", "trycloudflare.com"];
const UA = "Mozilla/5.0 (compatible; Prompt2Video/1.1)";
const CALL_TIMEOUT = 30_000;
const STORE_FILE = process.env.P2V_STORE || path.join(os.tmpdir(), "p2v-tunnel.json");

// ---------------------------------------------------------------- tunnel store

let link = { base: null, updated_at: null };

function loadLink() {
  try {
    const raw = JSON.parse(fs.readFileSync(STORE_FILE, "utf8"));
    if (raw && typeof raw.base === "string" && raw.base) {
      link = { base: raw.base, updated_at: Number(raw.updated_at) || null };
    }
  } catch (_) {
    /* no stored link yet */
  }
}

function saveLink() {
  try {
    fs.writeFileSync(STORE_FILE, JSON.stringify(link));
  } catch (_) {
    /* a read-only /tmp only costs us persistence across restarts */
  }
}

loadLink();

// --------------------------------------------------------------------- helpers

function sendJson(res, body, status = 200) {
  const payload = JSON.stringify(body);
  res.writeHead(status, {
    "content-type": "application/json; charset=utf-8",
    "content-length": Buffer.byteLength(payload),
    "cache-control": "no-store",
  });
  res.end(payload);
}

function readBody(req, limit = 64 * 1024) {
  return new Promise((resolve) => {
    let size = 0;
    const chunks = [];
    req.on("data", (chunk) => {
      size += chunk.length;
      if (size > limit) {
        req.destroy();
        resolve(null);
        return;
      }
      chunks.push(chunk);
    });
    req.on("end", () => {
      const text = Buffer.concat(chunks).toString("utf8");
      try {
        resolve(JSON.parse(text || "{}"));
      } catch (_) {
        resolve(null);
      }
    });
    req.on("error", () => resolve(null));
  });
}

// Only https URLs on a known tunnel domain are accepted, so the stored link can
// never point at an arbitrary third-party page.
function cleanBase(value) {
  let url;
  try {
    url = new URL(String(value || "").trim());
  } catch (_) {
    return "";
  }
  if (url.protocol !== "https:") return "";
  const host = url.hostname.toLowerCase();
  if (!ALLOWED.some((d) => host === d || host.endsWith("." + d))) return "";
  return (url.origin + url.pathname).replace(/\/+$/, "");
}

function withTimeout(ms) {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), ms);
  return { signal: ctrl.signal, done: () => clearTimeout(timer) };
}

function htmlToText(raw) {
  return String(raw)
    .replace(/<style[\s\S]*?<\/style>/gi, " ")
    .replace(/<script[\s\S]*?<\/script>/gi, " ")
    .replace(/<[^>]+>/g, " ")
    .replace(/\s+/g, " ")
    .trim()
    .slice(0, 300);
}

// ---------------------------------------------------------------- api handlers

async function tunnel(req, res, url) {
  if (req.method === "GET") {
    return sendJson(res, { base: link.base, updated_at: link.updated_at });
  }

  if (req.method === "POST") {
    const body = await readBody(req);
    if (!body) return sendJson(res, { ok: false, error: "invalid JSON" }, 400);
    const base = cleanBase(body.base);
    if (!base) return sendJson(res, { ok: false, error: "invalid gradio url" }, 400);
    link = { base, updated_at: Date.now() };
    saveLink();
    return sendJson(res, { ok: true, base });
  }

  if (req.method === "DELETE") {
    link = { base: null, updated_at: null };
    saveLink();
    return sendJson(res, { ok: true });
  }

  return sendJson(res, { ok: false, error: "method not allowed" }, 405);
}

// /api/status probes the Gradio app server-side, so the browser never talks to
// gradio.live (which blocks cross-origin requests and dies with the runtime).
async function status(res) {
  if (!link.base) {
    return sendJson(res, { base: null, updated_at: null, online: false, reason: "no-link" });
  }
  const t = withTimeout(12_000);
  try {
    const upstream = await fetch(link.base + "/gradio_api/info", {
      headers: { "user-agent": UA, accept: "application/json" },
      signal: t.signal,
      redirect: "follow",
    });
    return sendJson(res, {
      base: link.base,
      updated_at: link.updated_at,
      online: upstream.ok,
      reason: upstream.ok ? "ok" : "http-" + upstream.status,
    });
  } catch (_) {
    return sendJson(res, { base: link.base, updated_at: link.updated_at, online: false, reason: "unreachable" });
  } finally {
    t.done();
  }
}

// /api/generate forwards one generation to the Colab app and streams Gradio's
// SSE reply straight back, so the chat reads it as a same-origin response.
async function generate(req, res) {
  if (req.method !== "POST") return sendJson(res, { error: "method not allowed" }, 405);

  const body = await readBody(req);
  if (!body) return sendJson(res, { error: "bad-request", message: "JSON body required" }, 400);
  const data = body.data;
  if (!Array.isArray(data) || data.length === 0 || data.length > 12) {
    return sendJson(res, { error: "bad-request", message: "data[] required" }, 400);
  }

  if (!link.base) return sendJson(res, { error: "no-link", message: "Colab has not sent a link yet" }, 409);
  const base = link.base;

  let post;
  const t = withTimeout(CALL_TIMEOUT);
  try {
    post = await fetch(base + "/gradio_api/call/generate", {
      method: "POST",
      headers: { "content-type": "application/json", "user-agent": UA },
      body: JSON.stringify({ data }),
      signal: t.signal,
    });
  } catch (err) {
    return sendJson(res, { error: "offline", message: String((err && err.message) || err) }, 502);
  } finally {
    t.done();
  }

  if (!post.ok) {
    let raw = "";
    try {
      raw = await post.text();
    } catch (_) {
      /* ignore */
    }
    // gradio.live answers a dead tunnel with an HTML page, not JSON.
    const detail = htmlToText(raw);
    if (/no interface is running|tunnel .*not found/i.test(detail)) {
      return sendJson(res, { error: "offline", reason: "dead-link", message: detail }, 502);
    }
    return sendJson(res, { error: "upstream", status: post.status, detail }, 502);
  }

  let payload = null;
  try {
    payload = await post.json();
  } catch (_) {
    payload = null;
  }
  if (payload && Array.isArray(payload.data)) return sendJson(res, { ok: true, data: payload.data });

  const eventId = payload && (payload.event_id || (Array.isArray(payload) && payload[0] && payload[0].event_id));
  if (!eventId) return sendJson(res, { error: "upstream", message: "no event_id" }, 502);

  // Generation can take minutes; this GET is intentionally not time-limited.
  let stream;
  try {
    stream = await fetch(base + "/gradio_api/call/generate/" + encodeURIComponent(eventId), {
      headers: { accept: "text/event-stream", "user-agent": UA },
      redirect: "follow",
    });
  } catch (err) {
    return sendJson(res, { error: "offline", message: String((err && err.message) || err) }, 502);
  }
  if (!stream.ok || !stream.body) return sendJson(res, { error: "upstream", status: stream.status }, 502);

  res.writeHead(200, {
    "content-type": "text/event-stream; charset=utf-8",
    "cache-control": "no-store",
    connection: "keep-alive",
    "x-accel-buffering": "no",
  });
  res.flushHeaders?.();
  // Keep Render's proxy from cutting a long generation short.
  res.socket?.setKeepAlive?.(true, 30_000);

  const reader = stream.body.getReader();
  let closed = false;
  req.on("close", () => {
    closed = true;
    reader.cancel().catch(() => {});
  });

  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done || closed) break;
      if (!res.write(Buffer.from(value))) {
        await new Promise((resolve) => res.once("drain", resolve));
      }
    }
  } catch (_) {
    /* client went away or upstream broke mid-stream */
  }
  res.end();
}

// /api/media proxies the produced mp4 (with Range support) from the tunnel, so
// playback works even when the browser cannot reach gradio.live.
async function media(req, res, url) {
  const target = url.searchParams.get("u") || "";
  if (!link.base) return sendJson(res, { error: "no-link" }, 409);

  let full;
  try {
    full = /^https?:\/\//i.test(target) ? new URL(target) : new URL(link.base + (target || ""));
  } catch (_) {
    return sendJson(res, { error: "bad-request" }, 400);
  }
  if (full.hostname.toLowerCase() !== new URL(link.base).hostname.toLowerCase()) {
    return sendJson(res, { error: "forbidden" }, 403);
  }

  const headers = { "user-agent": UA };
  if (req.headers.range) headers.range = req.headers.range;

  let upstream;
  try {
    upstream = await fetch(full.toString(), { headers, redirect: "follow" });
  } catch (err) {
    return sendJson(res, { error: "offline", message: String((err && err.message) || err) }, 502);
  }
  if (!upstream.ok && upstream.status !== 206) {
    return sendJson(res, { error: "upstream", status: upstream.status }, 502);
  }

  const out = { "cache-control": "public, max-age=600" };
  for (const key of ["content-type", "content-length", "accept-ranges", "content-range"]) {
    const value = upstream.headers.get(key);
    if (value) out[key] = value;
  }
  if (!out["content-type"]) out["content-type"] = "video/mp4";

  res.writeHead(upstream.status, out);
  if (req.method === "HEAD" || !upstream.body) return res.end();

  const reader = upstream.body.getReader();
  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      if (!res.write(Buffer.from(value))) await new Promise((resolve) => res.once("drain", resolve));
    }
  } catch (_) {
    /* client aborted */
  }
  res.end();
}

// ------------------------------------------------------------------- static

const TYPES = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".svg": "image/svg+xml",
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".webp": "image/webp",
  ".ico": "image/x-icon",
  ".mp4": "video/mp4",
  // The Colab one-liner downloads these two with wget/urllib.
  ".sh": "text/x-shellscript; charset=utf-8",
  ".py": "text/plain; charset=utf-8",
  ".ipynb": "application/x-ipynb+json",
};

function serveStatic(req, res, url) {
  let rel;
  try {
    rel = decodeURIComponent(url.pathname);
  } catch (_) {
    return sendJson(res, { error: "bad-request" }, 400);
  }
  if (rel.endsWith("/")) rel += "index.html";

  const file = path.join(PUBLIC_DIR, path.normalize(rel).replace(/^(\.\.[/\\])+/, ""));
  if (file !== PUBLIC_DIR && !file.startsWith(PUBLIC_DIR + path.sep)) {
    res.writeHead(403, { "content-type": "text/plain; charset=utf-8" });
    return res.end("Forbidden");
  }

  fs.stat(file, (err, stat) => {
    if (err || !stat.isFile()) {
      res.writeHead(404, { "content-type": "text/plain; charset=utf-8" });
      return res.end("Not found");
    }
    const type = TYPES[path.extname(file).toLowerCase()] || "application/octet-stream";
    res.writeHead(200, {
      "content-type": type,
      "content-length": stat.size,
      "cache-control": path.extname(file) === ".html" ? "no-cache" : "public, max-age=300",
      "x-content-type-options": "nosniff",
    });
    if (req.method === "HEAD") return res.end();
    const stream = fs.createReadStream(file);
    stream.on("error", () => res.destroy());
    stream.pipe(res);
  });
}

// --------------------------------------------------------------------- server

const server = http.createServer((req, res) => {
  const url = new URL(req.url || "/", "http://" + (req.headers.host || "localhost"));
  const route = (handler) => Promise.resolve(handler()).catch((err) => {
    console.error("[p2v] handler failed:", err);
    if (!res.headersSent) sendJson(res, { error: "server-error" }, 500);
    else res.end();
  });

  switch (url.pathname) {
    case "/api/health":
      return route(() => sendJson(res, { ok: true, uptime: Math.round(process.uptime()) }));
    case "/api/tunnel":
      return route(() => tunnel(req, res, url));
    case "/api/status":
      return route(() => status(res));
    case "/api/generate":
      return route(() => generate(req, res));
    case "/api/media":
      return route(() => media(req, res, url));
    default:
      if (req.method !== "GET" && req.method !== "HEAD") {
        return sendJson(res, { error: "method not allowed" }, 405);
      }
      return serveStatic(req, res, url);
  }
});

server.keepAliveTimeout = 120_000;
server.headersTimeout = 125_000;
server.requestTimeout = 0; // generations run for minutes

server.listen(PORT, "0.0.0.0", () => {
  console.log(`[p2v] listening on http://0.0.0.0:${PORT}`);
});
