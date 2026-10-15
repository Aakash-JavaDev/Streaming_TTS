#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
python -m pytest -q tests
python -m speech_adapter.v2 smoke --output-dir runs/v2-smoke --steps 160
