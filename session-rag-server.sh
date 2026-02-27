#!/bin/bash
# Launcher for the persistent session-rag HTTP server.
# Starts the server if not running, verifies health.
# Safe to call multiple times (idempotent).
#
# Usage: ./session-rag-server.sh [start|stop|status|restart]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVER_DIR="$HOME/.session-rag"
PID_FILE="$SERVER_DIR/server.pid"
WATCHDOG_PID_FILE="$SERVER_DIR/watchdog.pid"
LOG_FILE="$SERVER_DIR/server.log"
PYTHON="$SCRIPT_DIR/venv/bin/python"
PORT="${SESSION_RAG_PORT:-7102}"
HEALTH_URL="http://127.0.0.1:$PORT/health"
MAX_WAIT=30
WATCHDOG_INTERVAL=30
WATCHDOG_MAX_FAILURES=3

mkdir -p "$SERVER_DIR"

is_running() {
    if [ -f "$PID_FILE" ]; then
        local pid
        pid=$(cat "$PID_FILE")
        if kill -0 "$pid" 2>/dev/null; then
            if curl -sf --max-time 2 "$HEALTH_URL" >/dev/null 2>&1; then
                return 0
            fi
            return 1
        else
            rm -f "$PID_FILE"
        fi
    fi
    return 1
}

stop_watchdog() {
    if [ -f "$WATCHDOG_PID_FILE" ]; then
        local wpid
        wpid=$(cat "$WATCHDOG_PID_FILE")
        if kill -0 "$wpid" 2>/dev/null; then
            kill "$wpid" 2>/dev/null || true
        fi
        rm -f "$WATCHDOG_PID_FILE"
    fi
}

start_watchdog() {
    stop_watchdog

    (
        failures=0
        while true; do
            sleep "$WATCHDOG_INTERVAL"

            if curl -sf --max-time 5 "$HEALTH_URL" >/dev/null 2>&1; then
                failures=0
            else
                failures=$((failures + 1))
                echo "[session-rag-watchdog] Health check failed ($failures/$WATCHDOG_MAX_FAILURES)" >> "$LOG_FILE"
            fi

            if [ "$failures" -ge "$WATCHDOG_MAX_FAILURES" ]; then
                echo "[session-rag-watchdog] Server unresponsive — restarting..." >> "$LOG_FILE"

                # Kill old server if still alive
                if [ -f "$PID_FILE" ]; then
                    local old_pid
                    old_pid=$(cat "$PID_FILE")
                    kill "$old_pid" 2>/dev/null || true
                    sleep 2
                    kill -9 "$old_pid" 2>/dev/null || true
                    rm -f "$PID_FILE"
                fi

                # Kill stale port holders
                local stale_pids
                stale_pids=$(lsof -ti :"$PORT" 2>/dev/null || true)
                if [ -n "$stale_pids" ]; then
                    echo "$stale_pids" | xargs kill 2>/dev/null || true
                    sleep 1
                fi

                # Restart the server
                export PYTHONPATH="$SCRIPT_DIR"
                export HF_HUB_OFFLINE=1
                export TRANSFORMERS_OFFLINE=1
                nohup "$PYTHON" -u "$SCRIPT_DIR/http_server.py" >> "$LOG_FILE" 2>&1 &

                # Wait for it to be healthy
                local waited=0
                while [ $waited -lt $MAX_WAIT ]; do
                    sleep 1
                    waited=$((waited + 1))
                    if curl -sf --max-time 2 "$HEALTH_URL" >/dev/null 2>&1; then
                        echo "[session-rag-watchdog] Server restarted successfully (${waited}s)" >> "$LOG_FILE"
                        break
                    fi
                done
                failures=0
            fi
        done
    ) &
    echo $! > "$WATCHDOG_PID_FILE"
    echo "[session-rag] Watchdog started (PID $!)" >&2
}

do_start() {
    if is_running; then
        echo "[session-rag] Already running (PID $(cat "$PID_FILE"))" >&2
        exit 0
    fi

    rm -f "$PID_FILE"

    # Kill any stale process holding our port
    local stale_pids
    stale_pids=$(lsof -ti :"$PORT" 2>/dev/null || true)
    if [ -n "$stale_pids" ]; then
        echo "[session-rag] Killing stale process(es) on port $PORT: $stale_pids" >&2
        echo "$stale_pids" | xargs kill 2>/dev/null || true
        sleep 1
    fi

    echo "[session-rag] Starting HTTP server on port $PORT..." >&2

    export PYTHONPATH="$SCRIPT_DIR"
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
    nohup "$PYTHON" -u "$SCRIPT_DIR/http_server.py" >> "$LOG_FILE" 2>&1 &
    local server_pid=$!

    local waited=0
    while [ $waited -lt $MAX_WAIT ]; do
        sleep 1
        waited=$((waited + 1))

        if ! kill -0 $server_pid 2>/dev/null; then
            echo "[session-rag] Server process died. Check $LOG_FILE" >&2
            exit 1
        fi

        if curl -sf --max-time 2 "$HEALTH_URL" >/dev/null 2>&1; then
            echo "[session-rag] Server ready (PID $server_pid, ${waited}s)" >&2
            start_watchdog
            exit 0
        fi
    done

    echo "[session-rag] Server failed to become healthy after ${MAX_WAIT}s. Check $LOG_FILE" >&2
    exit 1
}

do_stop() {
    stop_watchdog

    if [ -f "$PID_FILE" ]; then
        local pid
        pid=$(cat "$PID_FILE")
        if kill -0 "$pid" 2>/dev/null; then
            echo "[session-rag] Stopping server (PID $pid)..." >&2
            kill "$pid"
            local waited=0
            while [ $waited -lt 10 ]; do
                if ! kill -0 "$pid" 2>/dev/null; then
                    break
                fi
                sleep 1
                waited=$((waited + 1))
            done
            if kill -0 "$pid" 2>/dev/null; then
                kill -9 "$pid" 2>/dev/null || true
            fi
        fi
        rm -f "$PID_FILE"
        echo "[session-rag] Stopped." >&2
    else
        echo "[session-rag] Not running." >&2
    fi
}

do_status() {
    if is_running; then
        echo "[session-rag] Running (PID $(cat "$PID_FILE"))" >&2
    else
        echo "[session-rag] Not running." >&2
        exit 1
    fi
}

case "${1:-start}" in
    start)  do_start ;;
    stop)   do_stop ;;
    status) do_status ;;
    restart)
        do_stop
        do_start
        ;;
    *)
        echo "Usage: $0 {start|stop|status|restart}" >&2
        exit 1
        ;;
esac
