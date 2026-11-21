#!/usr/bin/env bash
# Venv alternative to Docker. Env: CHECKPOINT, POLICY_TYPE (act), PORT (8000), DEVICE (auto), POLICY_API_TOKEN (optional)
set -euo pipefail
cd "$(dirname "$0")/.."
if [ ! -d server/.venv ]; then
  python3 -m venv server/.venv
  server/.venv/bin/pip install -r server/requirements.txt
fi
: "${PORT:=8000}"; : "${POLICY_TYPE:=act}"; : "${DEVICE:=auto}"
if [ "${1:-}" = "--dummy" ]; then
  exec server/.venv/bin/python server/policy_server.py --dummy --port "$PORT"
fi
: "${CHECKPOINT:?set CHECKPOINT to a HF repo id or local dir}"
exec server/.venv/bin/python server/policy_server.py --checkpoint "$CHECKPOINT" \
  --policy-type "$POLICY_TYPE" --device "$DEVICE" --port "$PORT"
