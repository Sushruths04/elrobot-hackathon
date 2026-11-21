#!/bin/bash
# Restart NormaCore Station in the background (graceful stop, then start).
# Usage:  ./restart_station.sh          -> restart (or start if not running)
#         ./restart_station.sh stop     -> stop only
# Logs:   logs/station-<time>.log  (latest: logs/station-latest.log)
# Web UI: http://localhost:8889

STATION_DIR="${STATION_DIR:-$HOME/Downloads}"   # override: STATION_DIR=/path ./script
HERE="$(cd "$(dirname "$0")" && pwd)"

pids() { pgrep -f '^\./station( |$)|/station --web' ; }

stop_station() {
  local p; p=$(pids)
  [ -z "$p" ] && { echo "Station not running."; return 0; }
  echo "Stopping station (pid $p) ..."
  kill -INT $p                       # same as Ctrl+C -> flushes recorded data
  for _ in $(seq 1 30); do           # wait up to 15 s
    sleep 0.5
    [ -z "$(pids)" ] && ! lsof -nP -iTCP:8888 -sTCP:LISTEN >/dev/null 2>&1 && { echo "Stopped."; return 0; }
  done
  echo "Did not stop in 15 s -> force kill"; kill -9 $(pids) 2>/dev/null; sleep 1
}

start_station() {
  mkdir -p "$HERE/logs"
  local log="$HERE/logs/station-$(date +%Y%m%d-%H%M%S).log"
  ln -sf "$log" "$HERE/logs/station-latest.log"
  cd "$STATION_DIR" || exit 1
  nohup ./station --web --tcp >"$log" 2>&1 &
  disown
  for _ in $(seq 1 20); do           # wait up to 10 s for TCP port
    sleep 0.5
    lsof -nP -iTCP:8888 -sTCP:LISTEN >/dev/null 2>&1 && {
      echo "Station running (pid $(pids)). Web UI: http://localhost:8889"
      echo "Log: $log   (follow: tail -f $HERE/logs/station-latest.log)"
      return 0; }
  done
  echo "Station did not open port 8888 - see $log"; tail -20 "$log"; exit 1
}

stop_station
[ "$1" = "stop" ] && exit 0
start_station
