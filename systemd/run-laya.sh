#!/usr/bin/env bash
set -euo pipefail
LAYA_ROOT="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$LAYA_ROOT"
source "$LAYA_ROOT/npu/env.sh"
exec "$LAYA_ROOT/.venv/bin/uvicorn" app:app --host 0.0.0.0 --port 8000 --workers 1 --limit-concurrency 8 --timeout-keep-alive 5
