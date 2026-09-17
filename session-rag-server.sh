#!/bin/bash
# Launcher for the persistent session-rag HTTP server.
#
# Usage: ./session-rag-server.sh [start|stop|status|restart|logs|install-launchd|uninstall-launchd]
#
# start is idempotent and safe to run from several Claude Code sessions at once
# (a lock directory serialises concurrent starts). It also (re)starts a small
# health watchdog that restarts the server if /health stops answering.
#
# If the launchd agent is installed (install-launchd), launchd owns the process:
# start/stop/restart go through launchctl, so nothing else ever spawns a second
# server. stop unloads the agent (so it stays down for maintenance); start loads
# it again.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVER_DIR="$HOME/.session-rag"
PID_FILE="$SERVER_DIR/server.pid"
WATCHDOG_PID_FILE="$SERVER_DIR/watchdog.pid"
LOG_FILE="$SERVER_DIR/server.log"
LOCK_DIR="$SERVER_DIR/start.lock"
PYTHON="$SCRIPT_DIR/venv/bin/python"
PORT="${SESSION_RAG_PORT:-7102}"
HEALTH_URL="http://127.0.0.1:$PORT/health"
MAX_WAIT="${SESSION_RAG_START_TIMEOUT:-60}"
WATCHDOG_INTERVAL=30
WATCHDOG_MAX_FAILURES=3
LOG_MAX_BYTES=$((10 * 1024 * 1024))
LAUNCHD_LABEL="${SESSION_RAG_LAUNCHD_LABEL:-com.mattgreen.session-rag}"
LAUNCHD_PLIST="$HOME/Library/LaunchAgents/$LAUNCHD_LABEL.plist"
LAUNCHD_DOMAIN="gui/$(id -u)"

mkdir -p "$SERVER_DIR"

log() { echo "[session-rag] $*" >&2; }

healthy() { curl -sf --max-time "${1:-2}" "$HEALTH_URL" >/dev/null 2>&1; }

pid_alive() { [ -n "${1:-}" ] && kill -0 "$1" 2>/dev/null; }

port_owner() { lsof -ti :"$PORT" -sTCP:LISTEN 2>/dev/null | head -n 1 || true; }

server_pid() {
    local pid
    pid=$( [ -f "$PID_FILE" ] && cat "$PID_FILE" 2>/dev/null || true )
    if ! pid_alive "$pid"; then
        # PID file missing or stale: adopt whoever is listening, if it is healthy.
        pid=$(port_owner)
        if pid_alive "$pid" && healthy; then
            echo "$pid" > "$PID_FILE"
        else
            pid=""
        fi
    fi
    echo "$pid"
}

is_running() {
    local pid
    pid=$(server_pid)
    if pid_alive "$pid"; then
        healthy && return 0
        return 1
    fi
    rm -f "$PID_FILE"
    return 1
}

# Wait for an already-launched (but not yet healthy) server to come up.
wait_for_pid() {
    local pid=$1 waited=0
    while [ "$waited" -lt "$MAX_WAIT" ] && pid_alive "$pid"; do
        healthy && return 0
        sleep 1; waited=$((waited + 1))
    done
    return 1
}

rotate_log() {
    if [ -f "$LOG_FILE" ]; then
        local size
        size=$(stat -f%z "$LOG_FILE" 2>/dev/null || stat -c%s "$LOG_FILE" 2>/dev/null || echo 0)
        if [ "$size" -gt "$LOG_MAX_BYTES" ]; then
            mv -f "$LOG_FILE" "$LOG_FILE.1"
        fi
    fi
}

# --- start lock (mkdir is atomic) ---
acquire_lock() {
    local waited=0
    while ! mkdir "$LOCK_DIR" 2>/dev/null; do
        local owner
        owner=$(cat "$LOCK_DIR/pid" 2>/dev/null || true)
        if [ -n "$owner" ] && ! pid_alive "$owner"; then
            rm -rf "$LOCK_DIR"      # stale lock from a dead starter
            continue
        fi
        if [ "$waited" -ge "$MAX_WAIT" ]; then
            log "Another start has held the lock for ${MAX_WAIT}s; giving up"
            return 1
        fi
        sleep 1
        waited=$((waited + 1))
    done
    echo $$ > "$LOCK_DIR/pid"
    trap 'rm -rf "$LOCK_DIR"' EXIT
    return 0
}

release_lock() {
    rm -rf "$LOCK_DIR"
    trap - EXIT
}

# --- server process ---
launch_server() {
    rotate_log
    (
        cd "$SCRIPT_DIR"
        export PYTHONPATH="$SCRIPT_DIR"
        export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
        export GRPC_VERBOSITY="${GRPC_VERBOSITY:-NONE}" GLOG_minloglevel="${GLOG_minloglevel:-2}"
        nohup "$PYTHON" -u "$SCRIPT_DIR/http_server.py" >> "$LOG_FILE" 2>&1 &
        echo $! > "$SERVER_DIR/launch.pid"
    )
    cat "$SERVER_DIR/launch.pid"
}

wait_healthy() {
    local pid=$1 waited=0
    while [ "$waited" -lt "$MAX_WAIT" ]; do
        sleep 1
        waited=$((waited + 1))
        if ! pid_alive "$pid"; then
            return 2
        fi
        if healthy; then
            echo "$waited"
            return 0
        fi
    done
    return 1
}

kill_port_holders() {
    local pids
    pids=$(lsof -ti :"$PORT" 2>/dev/null || true)
    if [ -n "$pids" ]; then
        log "Killing stale process(es) on port $PORT: $pids"
        echo "$pids" | xargs kill 2>/dev/null || true
        sleep 1
        pids=$(lsof -ti :"$PORT" 2>/dev/null || true)
        [ -n "$pids" ] && echo "$pids" | xargs kill -9 2>/dev/null || true
    fi
}

# --- launchd ---
launchd_installed() { [ -f "$LAUNCHD_PLIST" ]; }
launchd_loaded() { launchctl print "$LAUNCHD_DOMAIN/$LAUNCHD_LABEL" >/dev/null 2>&1; }

launchd_load() {
    launchd_loaded && return 0
    launchctl bootstrap "$LAUNCHD_DOMAIN" "$LAUNCHD_PLIST" 2>/dev/null || launchctl load -w "$LAUNCHD_PLIST"
}

launchd_unload() {
    launchd_loaded || return 0
    launchctl bootout "$LAUNCHD_DOMAIN/$LAUNCHD_LABEL" 2>/dev/null || launchctl unload -w "$LAUNCHD_PLIST"
}

wait_for_health() {
    local waited=0
    while [ "$waited" -lt "$MAX_WAIT" ]; do
        healthy && { echo "$waited"; return 0; }
        sleep 1; waited=$((waited + 1))
    done
    return 1
}

wait_for_exit() {
    local waited=0
    while [ "$waited" -lt 20 ]; do
        if [ -z "$(port_owner)" ] && ! pgrep -f "milvus_lite/lib/milvus .*milvus.db" >/dev/null 2>&1; then
            return 0
        fi
        sleep 1; waited=$((waited + 1))
    done
    return 1
}

write_plist() {
    mkdir -p "$(dirname "$LAUNCHD_PLIST")"
    cat > "$LAUNCHD_PLIST" << PLISTEOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>$LAUNCHD_LABEL</string>
    <key>ProgramArguments</key>
    <array>
        <string>$PYTHON</string>
        <string>-u</string>
        <string>$SCRIPT_DIR/http_server.py</string>
    </array>
    <key>WorkingDirectory</key>
    <string>$SCRIPT_DIR</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PYTHONPATH</key>
        <string>$SCRIPT_DIR</string>
        <key>HF_HUB_OFFLINE</key>
        <string>1</string>
        <key>TRANSFORMERS_OFFLINE</key>
        <string>1</string>
        <key>GRPC_VERBOSITY</key>
        <string>NONE</string>
        <key>GLOG_minloglevel</key>
        <string>2</string>
        <key>SESSION_RAG_PORT</key>
        <string>$PORT</string>
    </dict>
    <!-- Start at login so the server is warm before the first Claude Code session. -->
    <key>RunAtLoad</key>
    <true/>
    <!-- launchd restarts the server if it exits; the model choice lives in
         ~/.session-rag/config.json so no environment plumbing is needed here. -->
    <key>KeepAlive</key>
    <true/>
    <key>ThrottleInterval</key>
    <integer>30</integer>
    <key>StandardOutPath</key>
    <string>$LOG_FILE</string>
    <key>StandardErrorPath</key>
    <string>$LOG_FILE</string>
</dict>
</plist>
PLISTEOF
}

do_install_launchd() {
    stop_watchdog
    if launchd_loaded; then launchd_unload; wait_for_exit || true; fi
    # Hand over a hook-started server, if any.
    local pid
    pid=$( [ -f "$PID_FILE" ] && cat "$PID_FILE" 2>/dev/null || true )
    pid_alive "$pid" || pid=$(port_owner)
    if pid_alive "$pid"; then kill "$pid" 2>/dev/null || true; wait_for_exit || true; fi
    write_plist
    launchd_load
    log "launchd agent installed: $LAUNCHD_PLIST"
    local waited
    if waited=$(wait_for_health); then log "Server ready under launchd (${waited}s)"; start_watchdog; else
        log "Server not healthy after ${MAX_WAIT}s. Check $LOG_FILE"; exit 1; fi
}

do_uninstall_launchd() {
    stop_watchdog
    launchd_unload
    rm -f "$LAUNCHD_PLIST"
    wait_for_exit || true
    rm -f "$PID_FILE"
    log "launchd agent removed. Use ./session-rag-server.sh start to run it from the hooks again."
}

# --- watchdog ---
watchdog_alive() {
    local wpid
    wpid=$(cat "$WATCHDOG_PID_FILE" 2>/dev/null || true)
    pid_alive "$wpid"
}

stop_watchdog() {
    local wpid
    wpid=$(cat "$WATCHDOG_PID_FILE" 2>/dev/null || true)
    if pid_alive "$wpid"; then
        kill "$wpid" 2>/dev/null || true
    fi
    rm -f "$WATCHDOG_PID_FILE"
}

start_watchdog() {
    watchdog_alive && return 0
    rm -f "$WATCHDOG_PID_FILE"
    nohup bash -c '
        failures=0
        while true; do
            sleep '"$WATCHDOG_INTERVAL"'
            if curl -sf --max-time 5 "'"$HEALTH_URL"'" >/dev/null 2>&1; then
                failures=0
                continue
            fi
            failures=$((failures + 1))
            echo "[session-rag-watchdog] $(date +%FT%T) health check failed ($failures/'"$WATCHDOG_MAX_FAILURES"')" >> "'"$LOG_FILE"'"
            if [ "$failures" -ge '"$WATCHDOG_MAX_FAILURES"' ]; then
                echo "[session-rag-watchdog] $(date +%FT%T) restarting server" >> "'"$LOG_FILE"'"
                "'"$SCRIPT_DIR/session-rag-server.sh"'" restart --no-watchdog >> "'"$LOG_FILE"'" 2>&1 || true
                failures=0
            fi
        done
    ' >/dev/null 2>&1 &
    echo $! > "$WATCHDOG_PID_FILE"
    disown 2>/dev/null || true
    log "Watchdog started (PID $(cat "$WATCHDOG_PID_FILE"))"
}

# --- commands ---
do_start() {
    local with_watchdog=${1:-yes}
    if is_running; then
        log "Already running (PID $(server_pid))$(launchd_loaded && echo ' under launchd')"
        [ "$with_watchdog" = yes ] && start_watchdog
        return 0
    fi
    if launchd_installed; then
        acquire_lock || exit 1
        if is_running; then release_lock; [ "$with_watchdog" = yes ] && start_watchdog; return 0; fi
        if launchd_loaded; then
            log "Starting via launchd ($LAUNCHD_LABEL) ..."
            launchctl kickstart "$LAUNCHD_DOMAIN/$LAUNCHD_LABEL" 2>/dev/null || true
        else
            log "Loading launchd agent ($LAUNCHD_LABEL) ..."
            launchd_load
        fi
        local waited
        if waited=$(wait_for_health); then
            release_lock
            log "Server ready (PID $(server_pid), ${waited}s, launchd)"
            [ "$with_watchdog" = yes ] && start_watchdog
            return 0
        fi
        release_lock
        log "Server not healthy after ${MAX_WAIT}s. Last log lines:"; tail -n 15 "$LOG_FILE" >&2
        exit 1
    fi
    acquire_lock || exit 1
    if is_running; then   # someone else started it while we waited for the lock
        log "Already running (PID $(server_pid))"
        release_lock
        [ "$with_watchdog" = yes ] && start_watchdog
        return 0
    fi

    # A server may be mid-startup (another session, or the watchdog): give it a chance.
    local old
    old=$( [ -f "$PID_FILE" ] && cat "$PID_FILE" 2>/dev/null || true )
    [ -n "$old" ] || old=$(port_owner)
    if pid_alive "$old"; then
        log "Server PID $old is alive but not healthy yet; waiting ..."
        if wait_for_pid "$old"; then
            echo "$old" > "$PID_FILE"
            log "Server ready (PID $old)"
            release_lock
            [ "$with_watchdog" = yes ] && start_watchdog
            return 0
        fi
        log "Server PID $old never became healthy; stopping it"
        kill "$old" 2>/dev/null || true
        sleep 2
        pid_alive "$old" && kill -9 "$old" 2>/dev/null || true
    fi
    rm -f "$PID_FILE"
    kill_port_holders

    log "Starting HTTP server on port $PORT (model: ${SESSION_RAG_MODEL:-embeddinggemma}) ..."
    local pid waited rc=0
    pid=$(launch_server)
    waited=$(wait_healthy "$pid") || rc=$?
    release_lock
    case $rc in
        0) log "Server ready (PID $pid, ${waited}s)"
           [ "$with_watchdog" = yes ] && start_watchdog
           return 0 ;;
        2) if is_running; then
               log "Our launch lost a race but a healthy server is running (PID $(server_pid))"
               [ "$with_watchdog" = yes ] && start_watchdog
               return 0
           fi
           log "Server process died during startup. Last log lines:"; tail -n 15 "$LOG_FILE" >&2; exit 1 ;;
        *) log "Server not healthy after ${MAX_WAIT}s. Check $LOG_FILE"; exit 1 ;;
    esac
}

do_stop() {
    local keep_watchdog=${1:-no}
    [ "$keep_watchdog" = yes ] || stop_watchdog
    if launchd_loaded; then
        log "Unloading launchd agent ($LAUNCHD_LABEL) so the server stays stopped ..."
        launchd_unload
        wait_for_exit || log "Warning: a milvus process is still shutting down"
        rm -f "$PID_FILE"
        log "Stopped."
        return 0
    fi
    local pid
    pid=$( [ -f "$PID_FILE" ] && cat "$PID_FILE" 2>/dev/null || true )
    pid_alive "$pid" || pid=$(port_owner)
    if pid_alive "$pid"; then
        log "Stopping server (PID $pid) ..."
        kill "$pid"
        local waited=0
        while pid_alive "$pid" && [ "$waited" -lt 15 ]; do
            sleep 1; waited=$((waited + 1))
        done
        pid_alive "$pid" && kill -9 "$pid" 2>/dev/null || true
        # milvus-lite runs as a child process holding the database lock; let it exit.
        waited=0
        while pgrep -f "milvus_lite/lib/milvus .*milvus.db" >/dev/null 2>&1 && [ "$waited" -lt 10 ]; do
            sleep 1; waited=$((waited + 1))
        done
        log "Stopped."
    else
        log "Not running."
    fi
    rm -f "$PID_FILE"
}

do_restart() {
    local with_watchdog=${1:-yes}
    if launchd_loaded; then
        acquire_lock || exit 1
        log "Restarting via launchd ($LAUNCHD_LABEL) ..."
        launchctl kickstart -k "$LAUNCHD_DOMAIN/$LAUNCHD_LABEL"
        sleep 1
        local waited
        if waited=$(wait_for_health); then
            release_lock; log "Server ready (PID $(server_pid), ${waited}s, launchd)"
            [ "$with_watchdog" = yes ] && start_watchdog
            return 0
        fi
        release_lock; log "Server not healthy after ${MAX_WAIT}s. Check $LOG_FILE"; exit 1
    fi
    acquire_lock || exit 1
    [ "$with_watchdog" = yes ] && stop_watchdog
    do_stop yes
    release_lock
    do_start "$with_watchdog"
}

do_status() {
    if is_running; then
        log "Running (PID $(server_pid))$(launchd_loaded && echo " under launchd [$LAUNCHD_LABEL]")"
        curl -s --max-time 3 "http://127.0.0.1:$PORT/status" | "$PYTHON" -m json.tool 2>/dev/null || true
        watchdog_alive && log "Watchdog running (PID $(cat "$WATCHDOG_PID_FILE"))" || log "Watchdog not running"
    else
        log "Not running."
        exit 1
    fi
}

cmd="${1:-start}"
shift || true
case "$cmd" in
    start)   do_start yes ;;
    stop)    do_stop ;;
    status)  do_status ;;
    restart)
        if [ "${1:-}" = "--no-watchdog" ]; then do_restart no; else do_restart yes; fi ;;
    logs)    tail -n "${1:-50}" -f "$LOG_FILE" ;;
    install-launchd)   do_install_launchd ;;
    uninstall-launchd) do_uninstall_launchd ;;
    *)       echo "Usage: $0 {start|stop|status|restart|logs [n]|install-launchd|uninstall-launchd}" >&2; exit 1 ;;
esac
