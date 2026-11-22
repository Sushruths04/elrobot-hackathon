#!/usr/bin/env bash
# Headless screenshot of the normakit web console (sim backend).
#
# Starts `webapp/run.py --backend sim`, waits for HTTP 200 plus a few seconds of
# WebSocket camera/telemetry streaming, then takes one headless-Chrome screenshot.
# Kills only the server it started. Safe to re-run (idempotent). Chrome dbus
# errors in the output are harmless.
#
#   bash scripts/screenshot_webapp.sh
#   PORT=8930 OUT=docs/twin/webapp_console.png BUDGET=20000 bash scripts/screenshot_webapp.sh
set -euo pipefail
cd "$(dirname "$0")/.."

PORT="${PORT:-8920}"
URL="http://127.0.0.1:${PORT}"
OUT="${OUT:-docs/twin/webapp_console.png}"
BUDGET="${BUDGET:-15000}"    # chrome --virtual-time-budget (ms); raise if the shot is a blank first paint
EXTRA_WAIT="${EXTRA_WAIT:-5}" # extra settle seconds after HTTP 200 so a WS camera frame arrives
CHROME="${CHROME:-/usr/bin/google-chrome}"
LOG="run/webapp_screenshot_server.log"

mkdir -p "$(dirname "$OUT")"

.venv/bin/python webapp/run.py --backend sim --port "$PORT" >"$LOG" 2>&1 &
SERVER_PID=$!
cleanup() {
  kill "$SERVER_PID" 2>/dev/null || true
  wait "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT

echo "waiting for $URL ..."
code="000"
for _ in $(seq 1 80); do
  code=$(curl -s -o /dev/null -w '%{http_code}' "$URL" || true)
  [ "$code" = "200" ] && break
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "server exited early; last log lines:" >&2
    tail -20 "$LOG" >&2
    exit 1
  fi
  sleep 0.5
done
[ "$code" = "200" ] || { echo "server never returned HTTP 200 on $URL" >&2; exit 1; }

echo "up; letting the WS camera/telemetry stream for ${EXTRA_WAIT}s ..."
sleep "$EXTRA_WAIT"

# dbus/Floss noise is expected on headless Linux; keep it out of the log.
"$CHROME" --headless --disable-gpu --no-sandbox \
  --window-size=1600,900 \
  --virtual-time-budget="$BUDGET" \
  --screenshot="$OUT" \
  "$URL" 2>&1 | grep -v -i -E 'dbus|floss|bluez' || true

ls -la "$OUT"
echo "wrote $OUT"
