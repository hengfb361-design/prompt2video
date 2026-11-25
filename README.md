# Prompt2Video

Chat UI + one-line Google Colab launcher for **LTX-Video 2B** text-to-video.
No API key, no GPU server: the model runs on the visitor's own free Colab T4, and the
site talks to it through a small same-origin proxy (`server.js`), so the browser never
has to call `gradio.live` (no CORS, no dead-link HTML in the chat).

## Deploy on Render

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/hengfb361-design/prompt2video)

1. Click the button → sign in to Render → **Deploy Blueprint**.
2. `render.yaml` creates a free **Node web service**, build `npm install --omit=dev`,
   start `node server.js`, health check `/api/health`. No database, no env vars needed.
3. Render prints the URL (`https://prompt2video.onrender.com`). Open it — the page
   writes its own one-liner with that domain.

A static site would not work here: the site needs `/api/tunnel`, `/api/status`,
`/api/generate` and `/api/media`, and static hosting cannot run them.

## The Colab one-liner

Open a new notebook at <https://colab.research.google.com>, set **Runtime → Change
runtime type → T4 GPU**, then run:

```
!wget -qO /content/p2v.sh https://YOUR-SITE/colab/run.sh && BASE=https://YOUR-SITE bash /content/p2v.sh
```

The site prints this line with its own domain filled in. It installs the dependencies,
downloads `p2v_app.py`, and starts Gradio with `share=True`. The app prints
`https://xxxx.gradio.live` **and** posts the link to `/api/tunnel`, so the chat connects
by itself (green dot in the sidebar). It re-publishes every 5 minutes; on a free Render
instance that also keeps the service awake.

## How the request flows

```
browser ──POST /api/generate──▶ server.js ──POST /gradio_api/call/generate──▶ Colab Gradio
        ◀──SSE (event: complete, video path)──┘ ◀──────────────────────────────┘
browser ──GET /api/media?u=...──▶ server.js ──Range GET──▶ Colab Gradio
```

| Route | Behaviour |
|---|---|
| `GET/POST/DELETE /api/tunnel` | Stores/reads/clears the Colab link (memory + temp file) |
| `GET /api/status` | Probes `{link}/gradio_api/info` (12 s timeout) → `online`, `reason` |
| `POST /api/generate` | Forwards `{data:[...]}` and streams Gradio's SSE straight back |
| `GET /api/media?u=` | Proxies the mp4 with Range support; only the tunnel host is allowed |
| `GET /api/health` | Render health check |

## Structure

| Path | What |
|---|---|
| `server.js` | Render web service: static files + the four proxy routes |
| `public/index.html` | Website: copyable one-liner, prompt studio, settings, connection panel |
| `public/colab/run.sh` | Launcher: GPU check → pip install → download app → start |
| `public/colab/p2v_app.py` | Gradio app: `LTXPipeline`, fp16/bf16, VAE tiling, CPU-offload fallback |
| `public/colab/Prompt2Video_Colab.ipynb` | Ready-made notebook with the same one-liner |
| `worker.js` + `deploy.json` | Same API as a Cloudflare Worker (used by the Freebuff Sites deploy) |
| `render.yaml` | Render Blueprint (Node web service) |

## Model settings that fit a T4 (16 GB)

- Resolution `704×480` or `768×512` (multiples of 32)
- Frames `49 / 65 / 81 / 97 / 121` (`8n+1`)
- Steps `30`, guidance `4.0`, seed `-1` = random

No model follows a prompt with 100% accuracy — generate 2–3 seeds and pick the best one.

## Local run

```
node server.js       # http://localhost:3000
```
