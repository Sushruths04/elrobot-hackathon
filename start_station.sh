#!/bin/bash
# Start NormaCore Station (web UI http://localhost:8889, TCP 8888) and save its log.
# Usage:  ./start_station.sh          (Ctrl+C to stop)
# Logs:   logs/station-<time>.log     (latest also at logs/station-latest.log)

STATION_DIR="${STATION_DIR:-$HOME/Downloads}"   # override: STATION_DIR=/path ./script
HERE="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$HERE/logs"
LOG="$HERE/logs/station-$(date +%Y%m%d-%H%M%S).log"
ln -sf "$LOG" "$HERE/logs/station-latest.log"

if lsof -nP -iTCP:8888 -sTCP:LISTEN >/dev/null 2>&1; then
  echo "Station already running (port 8888 busy). Stop it first."; exit 1
fi

cd "$STATION_DIR" || exit 1
echo "Starting station from $STATION_DIR  ->  web UI: http://localhost:8889"
echo "Log: $LOG"
./station --web --tcp 2>&1 | tee "$LOG"
