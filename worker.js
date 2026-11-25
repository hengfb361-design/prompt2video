const ROOM = "main";
const ALLOWED = ["gradio.live", "gradio.dev", "trycloudflare.com"];
const UA = "Mozilla/5.0 (compatible; Prompt2Video/1.1)";
const CALL_TIMEOUT = 30_000;

const json = (body, status = 200) =>
  new Response(JSON.stringify(body), {
    status,
    headers: {
      "content-type": "application/json; charset=utf-8",
      "cache-control": "no-store",
    },
  });

// Only https URLs on a known tunnel domain are accepted, so the stored link
// can never be an arbitrary third-party page.
function cleanBase(value) {
  try {
    const url = new URL(String(value || "").trim());
    if (url.protocol !== "https:") return "";
    const host = url.hostname.toLowerCase();
    if (!ALLOWED.some((d) => host === d || host.endsWith("." + d))) return "";
    return (url.origin + url.pathname).replace(/\/+$/, "");
  } catch (_) {
    return "";
  }
}

async function ensureTable(db) {
  await db
    .prepare(
      "CREATE TABLE IF NOT EXISTS tunnels (room TEXT PRIMARY KEY, base TEXT NOT NULL, updated_at INTEGER NOT NULL)"
    )
    .run();
}

async function getRow(db) {
  if (!db) return null;
  try {
    return await db
      .prepare("SELECT base, updated_at FROM tunnels WHERE room = ?1")
      .bind(ROOM)
      .first();
  } catch (_) {
    return null;
  }
}

function withTimeout(ms) {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), ms);
  return { signal: ctrl.signal, done: () => clearTimeout(timer) };
}

// /api/tunnel lets the Colab app publish its gradio.live link and the site read it.
async function tunnel(request, env) {
  const db = env.DB;
  if (!db) return json({ base: null, error: "storage not configured" }, 503);

  if (request.method === "GET") {
    const row = await getRow(db);
    return json({ base: row ? row.base : null, updated_at: row ? row.updated_at : null });
  }

  if (request.method === "POST") {
    let body = {};
    try {
      body = await request.json();
    } catch (_) {
      /* handled below */
    }
    const base = cleanBase(body && body.base);
    if (!base) return json({ ok: false, error: "invalid gradio url" }, 400);
    await ensureTable(db);
    await db
      .prepare(
        "INSERT INTO tunnels (room, base, updated_at) VALUES (?1, ?2, ?3) " +
          "ON CONFLICT(room) DO UPDATE SET base = ?2, updated_at = ?3"
      )
      .bind(ROOM, base, Date.now())
      .run();
    return json({ ok: true, base });
  }

  if (request.method === "DELETE") {
    try {
      await db.prepare("DELETE FROM tunnels WHERE room = ?1").bind(ROOM).run();
    } catch (_) {
      /* table not created yet — nothing stored */
    }
    return json({ ok: true });
  }

  return json({ ok: false, error: "method not allowed" }, 405);
}

// /api/status checks the tunnel server-side, so the browser never touches
// gradio.live (which blocks cross-origin requests and dies with the runtime).
async function status(env) {
  const row = await getRow(env.DB);
  if (!row || !row.base) {
    return json({ base: null, updated_at: null, online: false, reason: "no-link" });
  }
  const t = withTimeout(12_000);
  try {
    const res = await fetch(row.base + "/gradio_api/info", {
      headers: { "user-agent": UA, accept: "application/json" },
      signal: t.signal,
      redirect: "follow",
    });
    return json({
      base: row.base,
      updated_at: row.updated_at,
      online: res.ok,
      reason: res.ok ? "ok" : "http-" + res.status,
    });
  } catch (err) {
    return json({
      base: row.base,
      updated_at: row.updated_at,
      online: false,
      reason: "unreachable",
    });
  } finally {
    t.done();
  }
}

// /api/generate forwards one generation to the Colab app and streams Gradio's
// SSE reply straight back, so the chat reads it as a same-origin response.
async function generate(request, env) {
  if (request.method !== "POST") return json({ error: "method not allowed" }, 405);

  let body = {};
  try {
    body = await request.json();
  } catch (_) {
    return json({ error: "bad-request", message: "JSON body required" }, 400);
  }
  const data = body && body.data;
  if (!Array.isArray(data) || data.length === 0 || data.length > 12) {
    return json({ error: "bad-request", message: "data[] required" }, 400);
  }

  const row = await getRow(env.DB);
  if (!row || !row.base) return json({ error: "no-link", message: "Colab has not sent a link yet" }, 409);
  const base = row.base;

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
    return json({ error: "offline", message: String((err && err.message) || err) }, 502);
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
    const detail = raw
      .replace(/<style[\s\S]*?<\/style>/gi, " ")
      .replace(/<script[\s\S]*?<\/script>/gi, " ")
      .replace(/<[^>]+>/g, " ")
      .replace(/\s+/g, " ")
      .trim()
      .slice(0, 300);
    if (/no interface is running|tunnel .*not found/i.test(detail)) {
      return json({ error: "offline", reason: "dead-link", message: detail }, 502);
    }
    return json({ error: "upstream", status: post.status, detail }, 502);
  }

  let payload = null;
  try {
    payload = await post.json();
  } catch (_) {
    payload = null;
  }
  if (payload && Array.isArray(payload.data)) return json({ ok: true, data: payload.data });

  const eventId =
    payload && (payload.event_id || (Array.isArray(payload) && payload[0] && payload[0].event_id));
  if (!eventId) return json({ error: "upstream", message: "no event_id" }, 502);

  // Generation can take minutes; this GET is intentionally not time-limited.
  let stream;
  try {
    stream = await fetch(base + "/gradio_api/call/generate/" + encodeURIComponent(eventId), {
      headers: { accept: "text/event-stream", "user-agent": UA },
      redirect: "follow",
    });
  } catch (err) {
    return json({ error: "offline", message: String((err && err.message) || err) }, 502);
  }
  if (!stream.ok || !stream.body) {
    return json({ error: "upstream", status: stream.status }, 502);
  }

  return new Response(stream.body, {
    status: 200,
    headers: {
      "content-type": "text/event-stream; charset=utf-8",
      "cache-control": "no-store",
      "x-accel-buffering": "no",
    },
  });
}

// /api/media proxies the produced mp4 (with Range support) from the tunnel,
// so playback works even when the browser cannot reach gradio.live.
async function media(request, env) {
  const url = new URL(request.url);
  const target = url.searchParams.get("u") || "";
  const row = await getRow(env.DB);
  if (!row || !row.base) return json({ error: "no-link" }, 409);

  let full = null;
  try {
    full = /^https?:\/\//i.test(target) ? new URL(target) : new URL(row.base + (target || ""));
  } catch (_) {
    return json({ error: "bad-request" }, 400);
  }
  const baseUrl = new URL(row.base);
  if (full.hostname.toLowerCase() !== baseUrl.hostname.toLowerCase()) {
    return json({ error: "forbidden" }, 403);
  }

  const headers = { "user-agent": UA };
  const range = request.headers.get("range");
  if (range) headers.range = range;

  let upstream;
  try {
    upstream = await fetch(full.toString(), { headers, redirect: "follow" });
  } catch (err) {
    return json({ error: "offline", message: String((err && err.message) || err) }, 502);
  }
  if (!upstream.ok && upstream.status !== 206) {
    return json({ error: "upstream", status: upstream.status }, 502);
  }

  const out = new Headers();
  for (const key of ["content-type", "content-length", "accept-ranges", "content-range"]) {
    const value = upstream.headers.get(key);
    if (value) out.set(key, value);
  }
  if (!out.has("content-type")) out.set("content-type", "video/mp4");
  out.set("cache-control", "public, max-age=600");
  return new Response(upstream.body, { status: upstream.status, headers: out });
}

export default {
  async fetch(request, env) {
    const url = new URL(request.url);

    if (url.pathname === "/api/tunnel") return tunnel(request, env);
    if (url.pathname === "/api/status") return status(env);
    if (url.pathname === "/api/generate") return generate(request, env);
    if (url.pathname === "/api/media") return media(request, env);

    const response = await env.ASSETS.fetch(request);

    // The Colab one-liner downloads these with wget/urllib; make the types explicit.
    if (url.pathname.endsWith(".sh")) {
      const headers = new Headers(response.headers);
      headers.set("content-type", "text/x-shellscript; charset=utf-8");
      headers.set("cache-control", "public, max-age=300");
      return new Response(response.body, { status: response.status, headers });
    }
    return response;
  },
};
