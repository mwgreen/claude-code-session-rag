#!/bin/bash
# Launcher for the persistent session-rag HTTP server.
# Starts the server if not running, verifies health.
# Safe to call concurrently: a start lock guarantees only one server is spawned.
#
# Usage: ./session-rag-server.sh [start|stop|status|restart]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVER_DIR="$HOME/.session-rag"
PID_FILE="$SERVER_DIR/server.pid"
WATCHDOG_PID_FILE="$SERVER_DIR/watchdog.pid"
LOCK_DIR="$SERVER_DIR/start.lock"
LOG_FILE="$SERVER_DIR/server.log"
PYTHON="$SCRIPT_DIR/venv/bin/python"
PORT="${SESSION_RAG_PORT:-7102}"
HEALTH_URL="http://127.0.0.1:$PORT/health"

# The server preloads an embedding model and backfills the index on boot, which
# routinely takes 60-90s. A supervisor deadline shorter than that kills servers
# that are merely still starting, so the grace period must clear it comfortably.
STARTUP_GRACE="${SESSION_RAG_STARTUP_GRACE:-180}"
# How long `start` blocks before handing readiness off to the watchdog. Kept
# short so the SessionStart hook never stalls a Claude session on a cold boot.
START_CONFIRM="${SESSION_RAG_START_CONFIRM:-15}"
LOCK_WAIT=10
WATCHDOG_INTERVAL=30
WATCHDOG_MAX_FAILURES=3
WATCHDOG_MAX_BACKOFF=300

mkdir -p "$SERVER_DIR"

# PID recorded as the start-lock owner, used to detect a lock whose holder died.
# The watchdog subshell must override this: bash keeps $$ pointing at the parent
# shell, which exits early and would make every lock it takes look stale.
LOCK_OWNER_PID=$$

log()  { echo "[session-rag] $*" >&2; }
wlog() { echo "[session-rag-watchdog] $*" >> "$LOG_FILE"; }

# Echo the server PID if the pidfile names a live process; otherwise fail.
server_pid() {
    [ -f "$PID_FILE" ] || return 1
    local pid
    pid=$(cat "$PID_FILE" 2>/dev/null || true)
    [ -n "$pid" ] || return 1
    kill -0 "$pid" 2>/dev/null || return 1
    echo "$pid"
}

is_healthy() { curl -sf --max-time 5 "$HEALTH_URL" >/dev/null 2>&1; }

# "Running" means the process is alive, whether or not it answers /health yet.
# Treating a booting server as stopped is what previously let each new session
# spawn a rival that then fought over milvus.db and port 7102.
is_running() {
    if server_pid >/dev/null; then
        return 0
    fi
    rm -f "$PID_FILE"
    return 1
}

# --- start lock -------------------------------------------------------------
# mkdir is atomic on every filesystem we care about, and unlike flock it exists
# on stock macOS. The owner PID inside lets us reclaim a lock whose holder died.

acquire_start_lock() {
    local waited=0 owner
    while ! mkdir "$LOCK_DIR" 2>/dev/null; do
        owner=$(cat "$LOCK_DIR/pid" 2>/dev/null || true)
        if [ -z "$owner" ] || ! kill -0 "$owner" 2>/dev/null; then
            log "Clearing stale start lock (owner ${owner:-unknown} gone)"
            rm -rf "$LOCK_DIR"
            continue
        fi
        if [ "$waited" -ge "$LOCK_WAIT" ]; then
            return 1
        fi
        sleep 1
        waited=$((waited + 1))
    done
    echo "$LOCK_OWNER_PID" > "$LOCK_DIR/pid"
    return 0
}

release_start_lock() { rm -rf "$LOCK_DIR"; }

# --- server lifecycle -------------------------------------------------------

# Spawn the server and record its PID immediately. http_server.py writes the
# same pidfile itself, but not until it has imported its dependencies; claiming
# it here closes the window where a concurrent caller sees no live PID.
spawn_server() {
    export PYTHONPATH="$SCRIPT_DIR"
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
    nohup "$PYTHON" -u "$SCRIPT_DIR/http_server.py" >> "$LOG_FILE" 2>&1 &
    SPAWNED_PID=$!
    echo "$SPAWNED_PID" > "$PID_FILE"
}

kill_server() {
    local pid waited=0
    pid=$(cat "$PID_FILE" 2>/dev/null || true)
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
        kill "$pid" 2>/dev/null || true
        while [ "$waited" -lt 10 ] && kill -0 "$pid" 2>/dev/null; do
            sleep 1
            waited=$((waited + 1))
        done
        if kill -0 "$pid" 2>/dev/null; then
            kill -9 "$pid" 2>/dev/null || true
        fi
    fi
    rm -f "$PID_FILE"
}

# Only ever reached once we know no server of ours is alive.
clear_stale_port() {
    local stale_pids
    stale_pids=$(lsof -ti :"$PORT" 2>/dev/null || true)
    if [ -n "$stale_pids" ]; then
        log "Killing stale process(es) on port $PORT: $stale_pids"
        echo "$stale_pids" | xargs kill 2>/dev/null || true
        sleep 1
    fi
}

# --- watchdog ---------------------------------------------------------------

stop_watchdog() {
    if [ -f "$WATCHDOG_PID_FILE" ]; then
        local wpid
        wpid=$(cat "$WATCHDOG_PID_FILE" 2>/dev/null || true)
        if [ -n "$wpid" ] && kill -0 "$wpid" 2>/dev/null; then
            kill "$wpid" 2>/dev/null || true
        fi
        rm -f "$WATCHDOG_PID_FILE"
    fi
}

start_watchdog() {
    stop_watchdog

    (
        # A supervisor must not inherit `set -e`: any incidental non-zero status
        # would silently kill the very process meant to keep the server alive.
        set +e
        # Outlive the terminal that spawned us.
        trap '' HUP

        # bash 3.2 has no $BASHPID; a child's $PPID is this subshell's PID.
        my_pid=$(sh -c 'echo $PPID')
        LOCK_OWNER_PID="$my_pid"

        launched_at=$(date +%s)
        failures=0
        restarts=0

        while true; do
            sleep "$WATCHDOG_INTERVAL"

            # Step aside if a newer watchdog has taken over the pidfile.
            current=$(cat "$WATCHDOG_PID_FILE" 2>/dev/null || true)
            if [ "$current" != "$my_pid" ]; then
                wlog "Superseded by watchdog ${current:-none}; exiting (PID $my_pid)"
                exit 0
            fi

            if is_healthy; then
                if [ "$failures" -ne 0 ] || [ "$restarts" -ne 0 ]; then
                    wlog "Health restored"
                fi
                failures=0
                restarts=0
                continue
            fi

            now=$(date +%s)
            if ! server_pid >/dev/null; then
                wlog "Server process is gone — restarting immediately"
                failures="$WATCHDOG_MAX_FAILURES"
            elif [ $((now - launched_at)) -lt "$STARTUP_GRACE" ]; then
                wlog "Server still starting ($((now - launched_at))s of ${STARTUP_GRACE}s grace) — not intervening"
                continue
            else
                failures=$((failures + 1))
                wlog "Health check failed ($failures/$WATCHDOG_MAX_FAILURES)"
            fi

            if [ "$failures" -lt "$WATCHDOG_MAX_FAILURES" ]; then
                continue
            fi

            wlog "Server unresponsive — restarting..."

            if ! acquire_start_lock; then
                wlog "Another start is in progress; deferring"
                failures=0
                launched_at=$(date +%s)
                continue
            fi

            kill_server
            clear_stale_port
            spawn_server
            release_start_lock

            restarts=$((restarts + 1))
            launched_at=$(date +%s)
            failures=0
            wlog "Restarted (PID $SPAWNED_PID, attempt $restarts)"

            # Back off on repeated restarts so a server that cannot boot at all
            # does not become a spawn loop. Capped, never abandoned.
            if [ "$restarts" -gt 1 ]; then
                backoff=$((WATCHDOG_INTERVAL * restarts))
                if [ "$backoff" -gt "$WATCHDOG_MAX_BACKOFF" ]; then
                    backoff="$WATCHDOG_MAX_BACKOFF"
                fi
                wlog "Backing off ${backoff}s after $restarts consecutive restarts"
                sleep "$backoff"
            fi
        done
    # Detach from the caller's descriptors. Inheriting them keeps the pipe of a
    # piped invocation open for the watchdog's whole life, which hangs the
    # SessionStart hook waiting on a `start` that already finished its work.
    ) </dev/null >> "$LOG_FILE" 2>&1 &

    echo $! > "$WATCHDOG_PID_FILE"
    log "Watchdog started (PID $!)"
}

# --- commands ---------------------------------------------------------------

do_start() {
    if is_running; then
        if is_healthy; then
            log "Already running (PID $(cat "$PID_FILE"))"
        else
            log "Already starting (PID $(cat "$PID_FILE")); model preload in progress"
        fi
        exit 0
    fi

    if ! acquire_start_lock; then
        log "Another start is in progress; leaving it to finish"
        exit 0
    fi
    trap release_start_lock EXIT

    # Re-check under the lock: a racing caller may have started it while we waited.
    if is_running; then
        log "Already started by a concurrent caller (PID $(cat "$PID_FILE"))"
        exit 0
    fi

    rm -f "$PID_FILE"
    clear_stale_port

    log "Starting HTTP server on port $PORT..."
    spawn_server

    local waited=0
    while [ "$waited" -lt "$START_CONFIRM" ]; do
        sleep 1
        waited=$((waited + 1))

        if ! kill -0 "$SPAWNED_PID" 2>/dev/null; then
            log "Server process died. Check $LOG_FILE"
            rm -f "$PID_FILE"
            exit 1
        fi

        if is_healthy; then
            log "Server ready (PID $SPAWNED_PID, ${waited}s)"
            start_watchdog
            exit 0
        fi
    done

    # Still booting is the normal case on a cold start; hand off to the watchdog
    # rather than blocking the caller for the full model preload.
    log "Server starting in background (PID $SPAWNED_PID); readiness handed to watchdog"
    start_watchdog
    exit 0
}

do_stop() {
    stop_watchdog

    if is_running; then
        log "Stopping server (PID $(cat "$PID_FILE"))..."
        kill_server
        log "Stopped."
    else
        rm -f "$PID_FILE"
        log "Not running."
    fi
    release_start_lock
}

do_status() {
    if is_running; then
        if is_healthy; then
            log "Running (PID $(cat "$PID_FILE"))"
        else
            log "Starting (PID $(cat "$PID_FILE")) — not answering /health yet"
        fi
    else
        log "Not running."
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
