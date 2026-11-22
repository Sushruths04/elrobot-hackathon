#!/usr/bin/env bash
# Waits for ARM, moves to the demo start pose in safe stages, checks, then films the replay.
cd "$(dirname "$0")/.."
T='{"1": 2094, "2": 1055, "3": 1554, "4": 3020, "5": 170, "6": 2836, "7": 3021, "8": 2766}'
echo "WAITING for ARM"; until [ "$(curl -s -m3 localhost:8891/state | python3 -c 'import sys,json;print(json.load(sys.stdin)["armed_s"])')" -gt 120 ] 2>/dev/null; do sleep 1; done
echo "ARMED - mirroring off"; curl -s -m10 -X POST -d '{"on":false}' localhost:8891/mirror; echo
r=$(curl -s -m60 -X POST -d "{\"ticks\":$T,\"speed\":300,\"wait\":true,\"timeout\":30}" localhost:8891/move_joints); echo "all joints: $r"
if ! echo "$r" | grep -q '"ok": true'; then
  for stage in "5 6 7 8" "3 4" "1 2"; do
    body=$(python3 -c "import json;t=json.loads('$T');print(json.dumps({'ticks':{k:t[k] for k in '$stage'.split()},'speed':300,'wait':True,'timeout':30}))")
    echo "stage [$stage]: $(curl -s -m60 -X POST -d "$body" localhost:8891/move_joints)"
  done
fi
.venv/bin/python -u scripts/spoon_replay.py --check 2>&1 | grep -E "worst|path tip|checks passed|ABORT" || true
.venv/bin/python -u scripts/spoon_replay.py --check >/dev/null 2>&1 || { echo "ABORT: checks failed, not replaying"; exit 1; }
echo "STARTING FILMED REPLAY"; .venv/bin/python -u scripts/film_task.py spoon_replay > logs/spoon_replay_film.out 2>&1
grep -E "REPLAYING|DONE|ABORT|HIGH|not following" logs/spoon_replay.log; echo "FINISHED"
