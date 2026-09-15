#!/usr/bin/env bash
# Downloads the Phi-3-mini GGUF model expected by app/config.py (LLM_MODEL_PATH)
# into models/, so a fresh clone doesn't require a manual download.
set -euo pipefail

MODEL_REPO="${MODEL_REPO:-bartowski/Phi-3-mini-128k-instruct-GGUF}"
MODEL_FILE="${MODEL_FILE:-Phi-3-mini-128k-instruct-Q4_K_M.gguf}"
DEST="models/phi-3-mini-128k-instruct.Q4_K_M.gguf"

if [ -f "$DEST" ]; then
  echo "$DEST already exists, nothing to do."
  exit 0
fi

mkdir -p models
URL="https://huggingface.co/${MODEL_REPO}/resolve/main/${MODEL_FILE}"
TMP="${DEST}.part"

echo "Downloading ${MODEL_FILE} from ${MODEL_REPO} (~2.4 GB, this can take a few minutes depending on your connection)..."

if ! curl -fL --retry 3 -o "$TMP" "$URL"; then
  rm -f "$TMP"
  echo "" >&2
  echo "Download failed. If '${MODEL_REPO}' or '${MODEL_FILE}' has moved or been renamed on Hugging Face," >&2
  echo "override the source and retry:" >&2
  echo "  MODEL_REPO=<hf-user>/<hf-repo> MODEL_FILE=<file.gguf> bash scripts/download_model.sh" >&2
  exit 1
fi

mv "$TMP" "$DEST"
echo "Model saved to $DEST ($(du -h "$DEST" | cut -f1))."
