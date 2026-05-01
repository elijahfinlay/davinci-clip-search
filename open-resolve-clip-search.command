#!/bin/zsh

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
STATE_DIR="$ROOT_DIR/.resolve-clip-search"
PID_FILE="$STATE_DIR/server.pid"
LOG_FILE="$STATE_DIR/server.log"
APP_URL="http://127.0.0.1:8000"

mkdir -p "$STATE_DIR"

if [[ ! -x "$PYTHON_BIN" ]]; then
  osascript -e 'display alert "Resolve Clip Search" message "Python virtualenv is missing. Rebuild the app environment first."'
  exit 1
fi

# If a tracked server is already alive AND its API responds, reuse it.
# Otherwise fall through to a fresh start so a stale pidfile or a zombie left
# over by a SIGTERM-during-reindex hang doesn't block the relaunch.
if [[ -f "$PID_FILE" ]]; then
  EXISTING_PID="$(cat "$PID_FILE" 2>/dev/null || true)"
  if [[ -n "$EXISTING_PID" ]] && kill -0 "$EXISTING_PID" 2>/dev/null; then
    if curl -fsS --max-time 2 "$APP_URL/api/health" >/dev/null 2>&1; then
      open "$APP_URL"
      exit 0
    fi
  fi
  rm -f "$PID_FILE"
fi

# Port held but no responsive tracked server — likely a wedged zombie. Try to
# free the port (graceful, then forced) before launching.
if lsof -nP -iTCP:8000 -sTCP:LISTEN >/dev/null 2>&1; then
  PORT_PID="$(lsof -tiTCP:8000 -sTCP:LISTEN 2>/dev/null | head -n 1 || true)"
  if [[ -n "$PORT_PID" ]]; then
    kill "$PORT_PID" 2>/dev/null || true
  fi
  for _ in 1 2 3 4 5 6 7 8; do
    if ! lsof -tiTCP:8000 -sTCP:LISTEN >/dev/null 2>&1; then
      break
    fi
    sleep 1
  done
  if lsof -tiTCP:8000 -sTCP:LISTEN >/dev/null 2>&1; then
    PORT_PID="$(lsof -tiTCP:8000 -sTCP:LISTEN 2>/dev/null | head -n 1 || true)"
    if [[ -n "$PORT_PID" ]]; then
      kill -9 "$PORT_PID" 2>/dev/null || true
    fi
    sleep 1
  fi
fi

nohup "$PYTHON_BIN" run.py >>"$LOG_FILE" 2>&1 &
SERVER_PID=$!
echo "$SERVER_PID" >"$PID_FILE"

for _ in {1..40}; do
  if curl -fsS "$APP_URL/api/health" >/dev/null 2>&1; then
    open "$APP_URL"
    exit 0
  fi
  sleep 0.25
done

osascript -e 'display alert "Resolve Clip Search" message "The local server did not become ready. Check .resolve-clip-search/server.log for details."'
exit 1
