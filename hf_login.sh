#!/usr/bin/env bash

# Load Hugging Face token from .env and export HF_TOKEN
# Usage:  . ./hf_login.sh   (note the leading dot)

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$ROOT_DIR/.env"

if [[ ! -f "$ENV_FILE" ]]; then
  echo ".env file not found at $ENV_FILE" >&2
  return 1 2>/dev/null || exit 1
fi

# Extract HF_TOKEN value from .env (simple KEY=VALUE format)
HF_TOKEN_LINE="$(grep -E '^HF_TOKEN=' "$ENV_FILE" || true)"
if [[ -z "$HF_TOKEN_LINE" ]]; then
  echo "HF_TOKEN not found in $ENV_FILE" >&2
  return 1 2>/dev/null || exit 1
fi

HF_TOKEN_VALUE="${HF_TOKEN_LINE#HF_TOKEN=}"
export HF_TOKEN="$HF_TOKEN_VALUE"

echo "HF_TOKEN exported for this shell."

