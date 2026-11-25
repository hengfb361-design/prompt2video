#!/usr/bin/env bash
# Prompt2Video — Colab launcher.
# Usage (one line, inside a Google Colab code cell):
#   !wget -qO /content/p2v.sh https://YOUR-SITE/colab/run.sh && BASE=https://YOUR-SITE bash /content/p2v.sh
set -euo pipefail

BASE="${BASE:-${1:-}}"
APP_URL="${BASE%/}/colab/p2v_app.py"
APP="/content/p2v_app.py"

echo "=============================================="
echo "  Prompt2Video · LTX-Video 2B on Colab"
echo "=============================================="

# 1. GPU check — the model needs a CUDA device.
python - <<'PY'
import torch
print(f"torch {torch.__version__} | cuda available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f"GPU: {torch.cuda.get_device_name(0)} | {p.total_memory/1e9:.1f} GB | bf16: {torch.cuda.is_bf16_supported()}")
else:
    print("\n*** គ្មាន GPU! Runtime → Change runtime type → T4 GPU រួច Run ម្តងទៀត ***\n")
PY

# 2. Dependencies (diffusers >= 0.32 has LTXPipeline).
python -m pip install -q --upgrade \
  "diffusers>=0.32.0" transformers accelerate safetensors \
  gradio imageio imageio-ffmpeg sentencepiece "protobuf<7"

# 3. Download the app itself so the code stays in one place.
case "$APP_URL" in
  http://*|https://*) ;;
  *)
    echo "ERROR: BASE URL missing." >&2
    echo "Run it as:  BASE=https://your-site bash /content/p2v.sh" >&2
    exit 1
    ;;
esac

echo "→ downloading $APP_URL"
UA="Mozilla/5.0 (compatible; Prompt2Video/1.0)"
if command -v curl >/dev/null 2>&1; then
  curl -fsSL -A "$UA" "$APP_URL" -o "$APP"
elif command -v wget >/dev/null 2>&1; then
  wget -q --user-agent="$UA" -O "$APP" "$APP_URL"
else
  python - "$APP_URL" "$APP" "$UA" <<'PY'
import sys, urllib.request
url, dst, ua = sys.argv[1], sys.argv[2], sys.argv[3]
req = urllib.request.Request(url, headers={"User-Agent": ua})
with urllib.request.urlopen(req, timeout=60) as r, open(dst, "wb") as f:
    f.write(r.read())
print("saved", dst)
PY
fi

[ -s "$APP" ] || { echo "ERROR: could not download $APP_URL" >&2; exit 1; }
echo "→ app saved to $APP ($(wc -c <"$APP") bytes)"

# 4. Go. The app prints the public https://xxxx.gradio.live link AND posts it to
#    the website (/api/tunnel), so the chat connects without copying anything.
export P2V_BASE="$BASE"
echo "→ starting the app (first run downloads ~10 GB, one time only)…"
echo "→ link api នឹងបង្ហាញនៅលើវេបសាយដោយស្វ័យប្រវត្តិ"
python "$APP"
