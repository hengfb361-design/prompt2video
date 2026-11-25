#!/usr/bin/env python3
"""
Prompt2Video — text-to-video on a free Google Colab T4 GPU.

Model : Lightricks/LTX-Video (2B, diffusers LTXPipeline)
VRAM  : ~10 GB (documented by diffusers), so it fits a 15 GB T4.
Dtype : fp16 on Turing (T4 has no native bf16), bf16 on Ampere or newer.

Works with Gradio 5 and 6 (the launch options are negotiated at runtime).
Run locally the same way:  python p2v_app.py
"""

import gc
import os
import random
import re
import shutil
import threading
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch  # noqa: E402
import gradio as gr  # noqa: E402
from diffusers import LTXPipeline  # noqa: E402

try:  # diffusers >= 0.32
    from diffusers.utils import export_to_video
except Exception:  # pragma: no cover - very old/new diffusers
    export_to_video = None

MODEL_ID = os.environ.get("P2V_MODEL", "Lightricks/LTX-Video")
OUT_DIR = "/content/p2v_out" if os.path.isdir("/content") else "p2v_out"
os.makedirs(OUT_DIR, exist_ok=True)

# resolutions must be multiples of 32, frames must be 8*n + 1
RESOLUTIONS = {
    "16:9 · 704×480 (លឿន / fast)": (704, 480),
    "3:2 · 768×512 (ស្តង់ដារ / standard)": (768, 512),
    "1:1 · 512×512 (square)": (512, 512),
    "9:16 · 480×704 (ទូរស័ព្ទ / vertical)": (480, 704),
    "2:3 · 512×768 (portrait)": (512, 768),
    "16:9 · 960×544 (ធំ / slow)": (960, 544),
}
FRAMES = {
    "49 · ~2s": 49,
    "65 · ~2.7s": 65,
    "81 · ~3.4s": 81,
    "97 · ~4s": 97,
    "121 · ~5s (slow)": 121,
}

NEG_DEFAULT = (
    "worst quality, inconsistent motion, blurry, jittery, distorted, "
    "watermark, text, logo"
)

_pipe = None
_load_lock = threading.Lock()


def parse_resolution(value):
    """Accept "704×480", "16:9 · 704×480 (fast)" or a (w, h) pair from the API."""
    if isinstance(value, (tuple, list)) and len(value) == 2:
        return int(value[0]), int(value[1])
    text = str(value)
    match = re.search(r"(\d{3,4})\s*[x×]\s*(\d{3,4})", text)
    if match:
        return int(match.group(1)), int(match.group(2))
    return 704, 480


def parse_frames(value):
    """Accept "97", "97f · ~4s" or "97 · ~4s"; snap to 8n + 1."""
    numbers = re.findall(r"\d+", str(value))
    frames = int(numbers[0]) if numbers else 97
    frames = max(9, min(257, frames))
    return frames if (frames - 1) % 8 == 0 else ((frames - 1) // 8) * 8 + 1


def device_report() -> str:
    if not torch.cuda.is_available():
        return "CPU only — សូមបើក Runtime → Change runtime type → T4 GPU"
    props = torch.cuda.get_device_properties(0)
    return f"{torch.cuda.get_device_name(0)} · {props.total_memory / 1e9:.1f} GB"


def pick_dtype() -> torch.dtype:
    """bf16 only where the hardware really has it (Ampere+); T4 gets fp16."""
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        try:
            major, _minor = torch.cuda.get_device_capability(0)
            if major >= 8:
                return torch.bfloat16
        except Exception:
            pass
    return torch.float16


def _ram_total_gb() -> float:
    try:
        with open("/proc/meminfo") as fh:
            return int(re.search(r"MemTotal:\s+(\d+)", fh.read()).group(1)) / 1e6
    except Exception:
        return 0.0


def _mem_report(tag: str) -> None:
    """Print RAM/VRAM at every load stage — a stall at 98% is a memory kill."""
    parts = []
    try:
        with open("/proc/meminfo") as fh:
            info = fh.read()
        total = int(re.search(r"MemTotal:\s+(\d+)", info).group(1)) / 1e6
        avail = int(re.search(r"MemAvailable:\s+(\d+)", info).group(1)) / 1e6
        parts.append(f"RAM {total - avail:.1f} used / {total:.1f} GB")
    except Exception:
        pass
    if torch.cuda.is_available():
        try:
            used = torch.cuda.memory_allocated() / 1e9
            total = torch.cuda.get_device_properties(0).total_memory / 1e9
            parts.append(f"VRAM {used:.1f} / {total:.1f} GB")
        except Exception:
            pass
    if parts:
        print(f"[p2v] {tag} · " + " · ".join(parts), flush=True)


def preflight() -> None:
    """Print the numbers that decide whether a T4 can finish the load."""
    try:
        free_disk = shutil.disk_usage(os.path.expanduser("~")).free / 1e9
    except Exception:
        free_disk = -1.0
    print(
        f"[p2v] preflight · GPU: {device_report()} · RAM {_ram_total_gb():.1f} GB · "
        f"disk free {free_disk:.1f} GB (ចាំបាច់ ~15 GB)",
        flush=True,
    )
    if 0 <= free_disk < 15:
        print(
            "[p2v] ⚠️ ឌីសតិចពេកសម្រាប់ទាញទម្ងន់ — Runtime → Disconnect and delete runtime",
            flush=True,
        )


_stop_watch = threading.Event()


def _watchdog(interval: float = 20.0) -> None:
    """A long load is not a freeze: print RAM/VRAM so a stall is visible."""
    ticks = 0
    while not _stop_watch.wait(interval):
        ticks += 1
        _mem_report(f"loading… {int(ticks * interval)}s")


def load_plan() -> dict:
    """Stream the weights straight onto the GPU instead of through Colab's ~12 GB RAM.

    The fp16 pipeline is ~13.5 GB, which is more than the free Colab VM has in
    system memory, so loading it into RAM first gets the process killed at the
    very last shards (the classic "Loading weights: 98%" freeze).
    """
    if not torch.cuda.is_available():
        return {}
    total_gb = int(torch.cuda.get_device_properties(0).total_memory / 1e9)
    gpu_gb = max(6, total_gb - 3)  # leave room for the CUDA context + activations
    return {"device_map": "balanced", "max_memory": {0: f"{gpu_gb}GiB", "cpu": "6GiB"}}


def load_pipe(mode: str = "balanced") -> None:
    """mode='balanced' streams weights onto the GPU, 'offload' keeps them on CPU,
    'full' moves the whole pipeline to the GPU after loading."""
    global _pipe
    dtype = pick_dtype()
    ram = _ram_total_gb()
    print(f"[p2v] loading {MODEL_ID} as {dtype} · mode={mode}", flush=True)
    if ram and ram < 14:
        print(
            f"[p2v] system RAM only {ram:.1f} GB — weights are streamed to the GPU "
            "so they do not pile up in RAM",
            flush=True,
        )
    _mem_report(f"before load ({mode})")

    kwargs = {"dtype": dtype}
    if mode == "balanced":
        kwargs.update(load_plan())

    _stop_watch.clear()
    watcher = threading.Thread(target=_watchdog, daemon=True)
    watcher.start()
    # diffusers renamed torch_dtype -> dtype; support both.
    try:
        try:
            pipe = LTXPipeline.from_pretrained(MODEL_ID, **kwargs)
        except TypeError:
            kwargs.pop("dtype", None)
            kwargs["torch_dtype"] = dtype
            pipe = LTXPipeline.from_pretrained(MODEL_ID, **kwargs)
    finally:
        _stop_watch.set()
    _mem_report("weights loaded")

    placement = getattr(pipe, "hf_device_map", None)
    if mode == "balanced" and placement:
        on_cpu = [k for k, v in placement.items() if str(v) == "cpu"]
        print(f"[p2v] device_map balanced · on CPU: {on_cpu or 'nothing'}", flush=True)

    if mode == "offload":
        print("[p2v] low-VRAM mode: model CPU offload", flush=True)
        pipe.enable_model_cpu_offload()
    elif mode == "full" or not placement:
        try:
            pipe.to("cuda")
        except Exception as exc:
            print(f"[p2v] pipe.to(cuda) failed ({exc}); falling back to CPU offload", flush=True)
            pipe.enable_model_cpu_offload()

    if mode != "balanced":
        # The VAE is small; float32 keeps fp16 colour artefacts away.
        try:
            pipe.vae.to(torch.float32)
        except Exception:
            pass

    try:
        pipe.vae.enable_tiling()
    except Exception:
        pass
    for name in ("enable_attention_slicing", "enable_vae_slicing"):
        try:
            getattr(pipe, name)()
            break
        except Exception:
            continue

    # Decoding on CPU would take minutes per video, so pull the VAE back if needed.
    if torch.cuda.is_available():
        try:
            vae_dev = next(pipe.vae.parameters()).device
        except Exception:
            vae_dev = torch.device("cpu")
        if vae_dev.type == "cpu":
            try:
                pipe.vae.to("cuda")
                print("[p2v] moved VAE to the GPU", flush=True)
            except Exception as exc:
                print(f"[p2v] VAE stays on CPU ({exc}); decoding will be slow", flush=True)

    _pipe = pipe
    _mem_report("model ready")


LOAD_MODES = ("balanced", "offload", "full")


def ensure_pipe() -> LTXPipeline:
    global _pipe
    if _pipe is not None:
        return _pipe
    with _load_lock:
        if _pipe is None:
            last = None
            for mode in LOAD_MODES:
                try:
                    load_pipe(mode=mode)
                    break
                except Exception as exc:  # OOM, RAM kill, network…
                    last = exc
                    print(f"[p2v] load mode '{mode}' failed: {exc}", flush=True)
                    _pipe = None
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            if _pipe is None:
                raise RuntimeError(f"មិនអាច load model បានទេ — {last}")
    return _pipe


def save_video(frames, path: str, fps: int = 24) -> str:
    if export_to_video is not None:
        try:
            export_to_video(frames, path, fps=fps)
            return path
        except Exception as exc:  # pragma: no cover
            print(f"[p2v] export_to_video failed ({exc}); using imageio", flush=True)
    import numpy as np
    import imageio.v2 as imageio

    writer = imageio.get_writer(
        path, fps=fps, codec="libx264", quality=8,
        macro_block_size=1, ffmpeg_params=["-pix_fmt", "yuv420p"],
    )
    try:
        for frame in frames:
            writer.append_data(np.asarray(frame))
    finally:
        writer.close()
    return path


def generate(prompt, negative, resolution, frames, steps, guidance, seed):
    prompt = (prompt or "").strip()
    if not prompt:
        raise gr.Error("សូមសរសេរ prompt សិន (ណែនាំជាភាសាអង់គ្លេស)។")

    width, height = parse_resolution(resolution)
    num_frames = parse_frames(frames)
    seed = int(seed) if seed is not None and int(seed) >= 0 else random.randint(0, 2**31 - 1)

    pipe = ensure_pipe()
    gen = torch.Generator(device="cuda").manual_seed(seed)
    started = time.time()
    _mem_report(f"generate {width}x{height} · {num_frames}f · {int(steps)} steps")

    kwargs = dict(
        prompt=prompt,
        negative_prompt=(negative or "").strip() or None,
        width=width,
        height=height,
        num_frames=num_frames,
        num_inference_steps=int(steps),
        guidance_scale=float(guidance),
        decode_timestep=0.03,
        decode_noise_scale=0.025,
        generator=gen,
    )

    try:
        video = pipe(**kwargs).frames[0]
    except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
        print(f"[p2v] OOM at {width}x{height}/{num_frames}f ({exc}); retrying smaller", flush=True)
        global _pipe
        _pipe = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        try:
            load_pipe(mode="offload")
        except Exception as exc:
            print(f"[p2v] reload after OOM failed: {exc}", flush=True)
        kwargs.update(width=704, height=480, num_frames=min(num_frames, 81))
        width, height = kwargs["width"], kwargs["height"]
        video = _pipe(**kwargs).frames[0]

    path = save_video(video, os.path.join(OUT_DIR, f"p2v_{seed}.mp4"), fps=24)
    elapsed = time.time() - started
    info = (
        f"✅ រួចរាល់ — {width}×{height} · {kwargs['num_frames']} frames · "
        f"{int(steps)} steps · seed {seed} · {elapsed:.1f}s\n"
        f"GPU: {device_report()}"
    )
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return path, info


def boot() -> None:
    """Warm the model in the background so the public link appears immediately."""
    preflight()
    try:
        ensure_pipe()
    except Exception as exc:  # keep the UI up even if the download fails
        print(f"[p2v] preload failed: {exc}", flush=True)


def publish_link(url: str) -> bool:
    """Push the public link to the website so the chat connects by itself."""
    base = (os.environ.get("P2V_BASE") or "").rstrip("/")
    if not base or not url:
        return False
    import json as _json
    import urllib.request

    body = _json.dumps({"base": url}).encode()
    req = urllib.request.Request(
        base + "/api/tunnel",
        data=body,
        method="POST",
        headers={
            "content-type": "application/json",
            "user-agent": "Mozilla/5.0 (compatible; Prompt2Video/1.0)",
        },
    )
    with urllib.request.urlopen(req, timeout=20) as resp:
        print(f"[p2v] link បានផ្ញើទៅវេបសាយ ({resp.status}) · {url}", flush=True)
        return 200 <= resp.status < 300


def _keep_publishing(url: str) -> None:
    """Re-publish every 5 minutes; retry quickly until the first send works."""
    while True:
        ok = False
        try:
            ok = publish_link(url)
        except Exception as exc:
            print(f"[p2v] ផ្ញើ link មិនបាន: {exc}", flush=True)
        time.sleep(300 if ok else 45)


def launch_app(app) -> None:
    """Launch with whichever options this Gradio version accepts."""
    port = int(os.environ.get("PORT", "7860"))
    base = {"server_name": "0.0.0.0", "server_port": port, "share": True}
    try:
        theme = gr.themes.Soft()
    except Exception:
        theme = None

    candidates = []
    if theme is not None:
        candidates.append(dict(base, theme=theme))
    candidates.append(dict(base))

    try:
        app = app.queue()
    except Exception:
        pass

    for kwargs in candidates:
        try:
            result = app.launch(**kwargs)
        except TypeError as exc:
            print(f"[p2v] launch() rejected {sorted(kwargs)}: {exc}", flush=True)
            continue

        share = ""
        if isinstance(result, (tuple, list)) and len(result) >= 3:
            share = str(result[2] or "")
        if share:
            print(f"[p2v] public link: {share}", flush=True)
            threading.Thread(target=_keep_publishing, args=(share,), daemon=True).start()
        else:
            print("[p2v] គ្មាន public link — សូមផ្ទៀងផ្ទាត់ថា gradio បានបើក share", flush=True)
        return
    raise SystemExit(
        "Gradio មិនអាចបើកបានទេ — សូមផ្ញើ error ខាងលើមកឱ្យខ្ញុំ។"
    )


with gr.Blocks(title="Prompt2Video · LTX-Video") as demo:
    gr.Markdown(
        "# 🎬 Prompt2Video — LTX-Video\n"
        "បង្កើតវីដេអូពី prompt · model រត់លើ GPU របស់ Colab នេះ (គ្មាន API key)។\n\n"
        "**Prompt គួរសរសេរជាភាសាអង់គ្លេស** — រចនាសម្ព័ន្ធ: "
        "`subject + action + camera + lighting + style`។"
    )
    with gr.Row():
        with gr.Column(scale=3):
            prompt = gr.Textbox(
                label="Prompt",
                lines=4,
                placeholder="A red vintage car drives slowly along a rainy Bangkok street at night, "
                "neon reflections on wet asphalt, cinematic, shallow depth of field",
            )
            negative = gr.Textbox(label="Negative prompt", value=NEG_DEFAULT, lines=2)
            with gr.Row():
                resolution = gr.Dropdown(
                    choices=list(RESOLUTIONS),
                    value="16:9 · 704×480 (លឿន / fast)",
                    label="Resolution",
                )
                frames = gr.Dropdown(
                    choices=list(FRAMES),
                    value="97 · ~4s",
                    label="រយៈពេល / duration",
                )
            with gr.Row():
                steps = gr.Slider(minimum=15, maximum=50, value=30, step=1, label="Steps")
                guidance = gr.Slider(minimum=1.5, maximum=6.0, value=4.0, step=0.1, label="Guidance")
                seed = gr.Number(value=-1, precision=0, label="Seed (-1 = random)")
            run = gr.Button("✨ បង្កើតវីដេអូ / Generate", variant="primary")
        with gr.Column(scale=2):
            out = gr.Video(label="លទ្ធផល / output")
            info = gr.Textbox(label="ព័ត៌មាន", lines=3, interactive=False)
            gr.Markdown(
                f"**Model:** `{MODEL_ID}`  \n**GPU:** {device_report()}  \n"
                "T4: 704×480 · 97 frames · 30 steps គឺជាចំណុចមានតុល្យភាព។ "
                "បើ OOM សូមបន្ថយ frames ឬ resolution។  \n"
                "ជួរឈរ `RAM ... / VRAM ...` ក្នុង log បង្ហាញចំណុចដែលអង្គចងចាំអស់។"
            )

    run.click(generate, [prompt, negative, resolution, frames, steps, guidance, seed], [out, info])

    gr.Markdown(
        "---\n"
        "**ចំណាំ:** គ្មាន model ណាមួយធ្វើតាម prompt បាន 100% ទេ។ "
        "ដើម្បីបានជិតបំផុត: prompt អង់គ្លេសច្បាស់, ពិពណ៌នាកាមេរ៉ា, បើក steps 35–45, "
        "និងប្រើ seed ដូចគ្នាដើម្បីប្រៀបធៀប prompt ផ្សេងៗ។"
    )

if __name__ == "__main__":
    threading.Thread(target=boot, daemon=True).start()
    launch_app(demo)
