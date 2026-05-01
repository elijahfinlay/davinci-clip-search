#!/bin/zsh

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

PID_FILE="$ROOT_DIR/.resolve-clip-search/server.pid"

# Wait up to `timeout` seconds for either the PID to die or port 8000 to free.
# Escalates SIGTERM -> SIGKILL if the process is wedged inside a Resolve API
# call and won't honour the polite signal.
wait_for_shutdown() {
  local pid="$1"
  local timeout=8
  for _ in $(seq 1 "$timeout"); do
    local pid_alive=0
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      pid_alive=1
    fi
    local port_busy=0
    if lsof -tiTCP:8000 -sTCP:LISTEN >/dev/null 2>&1; then
      port_busy=1
    fi
    if [[ "$pid_alive" -eq 0 && "$port_busy" -eq 0 ]]; then
      return 0
    fi
    sleep 1
  done
  return 1
}

PID=""
if [[ -f "$PID_FILE" ]]; then
  PID="$(cat "$PID_FILE" 2>/dev/null || true)"
  if [[ -n "$PID" ]] && kill -0 "$PID" 2>/dev/null; then
    kill "$PID" 2>/dev/null || true
  fi
fi

if ! wait_for_shutdown "$PID"; then
  # Polite signal didn't take — escalate. Captures both the tracked PID and any
  # other process still holding port 8000.
  if [[ -n "$PID" ]] && kill -0 "$PID" 2>/dev/null; then
    kill -9 "$PID" 2>/dev/null || true
  fi
  PORT_PID="$(lsof -tiTCP:8000 -sTCP:LISTEN 2>/dev/null | head -n 1 || true)"
  if [[ -n "$PORT_PID" ]]; then
    kill -9 "$PORT_PID" 2>/dev/null || true
  fi
  wait_for_shutdown "$PID" || true
fi

rm -f "$PID_FILE"
