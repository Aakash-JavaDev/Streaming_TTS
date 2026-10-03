#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is not installed. Install it from https://docs.astral.sh/uv/ first." >&2
  exit 1
fi

uv venv .venv --python 3.12
uv pip install --python .venv/bin/python --torch-backend=cpu -r requirements.txt

echo "Environment ready. Run: scripts/run_local_smoke.sh"
