#!/bin/bash
# Launcher and supervisor for the persistent session-rag HTTP server.
#
# Usage: ./session-rag-server.sh [start|stop|status|restart|logs [n]|install-launchd|uninstall-launchd]
#
# Design
#  * `start` is idempotent and safe to call from several Claude Code sessions at
#    once: starts are serialised through an atomic mkdir lock, and a server that
#    is alive but still booting is never treated as dead.
#  * `start` blocks for at most START_CONFIRM seconds. If the server is still
#    loading the model after that, readiness is handed to the watchdog so the
#    SessionStart hook never stalls a Claude session.
#  * The watchdog restarts the server when /health stays down, but leaves a
#    booting server alone for STARTUP_GRACE seconds and backs off on repeated
#    restarts. Only one watchdog runs; a superseded one exits by itself.
#  * With the launchd agent installed (install-launchd), launchd owns the
#    process: start/stop/restart go through launchctl, so the hooks, the
#    watchdog and launchd can never spawn competing servers. `stop` unloads the
#    agent so the server stays down for maintenance; `start` loads it again.

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

# How long `start`/`restart` block waiting for /health before handing off.
START_CONFIRM="${SESSION_RAG_START_CONFIRM:-15}"
# How long the watchdog leaves a live but unhealthy server alone after a launch.
STARTUP_GRACE="${SESSION_RAG_STARTUP_GRACE:-180}"
LOCK_WAIT="${SESSION_RAG_LOCK_WAIT:-30}"
WATCHDOG_INTERVAL=30
WATCHDOG_MAX_FAILURES=3
WATCHDOG_MAX_BACKOFF=300
LOG_MAX_BYTES=$((10 * 1024 * 1024))

LAUNCHD_LABEL="${SESSION_RAG_LAUNCHD_LABEL:-com.mattgreen.session-rag}"
LAUNCHD_PLIST="$HOME/Library/LaunchAgents/$LAUNCHD_LABEL.plist"
LAUNCHD_DOMAIN="gui/$(id -u)"

mkdir -p "$SERVER_DIR"

log()  { echo "[session-rag] $*" >&2; }
wlog() { echo "[session-rag-watchdog] $(date +%FT%T) $*" >> "$LOG_FILE"; }

healthy()   { curl -sf --max-time "${1:-3}" "$HEALTH_URL" >/dev/null 2>&1; }
pid_alive() { [ -n "${1:-}" ] && kill -0 "$1" 2>/dev/null; }
port_owner() { lsof -ti :"$PORT" -sTCP:LISTEN 2>/dev/null | head -n 1 || true; }

# PID of the live server, or empty. Falls back to whatever healthy process owns
# the port (a stale or missing pidfile must not make us spawn a rival).
server_pid() {
    local pid
    pid=$( [ -f "$PID_FILE" ] && cat "$PID_FILE" 2>/dev/null || true )
    if ! pid_alive "$pid"; then
        pid=$(port_owner)
        if pid_alive "$pid" && healthy; then
            echo "$pid" > "$PID_FILE"
        else
            pid=""
        fi
    fi
    echo "$pid"
}

is_alive()   { [ -n "$(server_pid)" ]; }
is_running() { is_alive && healthy; }

rotate_log() {
    if [ -f "$LOG_FILE" ]; then
        local size
        size=$(stat -f%z "$LOG_FILE" 2>/dev/null || stat -c%s "$LOG_FILE" 2>/dev/null || echo 0)
        [ "$size" -gt "$LOG_MAX_BYTES" ] && mv -f "$LOG_FILE" "$LOG_FILE.1"
    fi
    return 0
}

# --- start lock (mkdir is atomic; the owner PID lets us reclaim a dead holder's lock) ---
acquire_lock() {
    local waited=0 owner
    while ! mkdir "$LOCK_DIR" 2>/dev/null; do
        owner=$(cat "$LOCK_DIR/pid" 2>/dev/null || true)
        if [ -z "$owner" ] || ! pid_alive "$owner"; then
            rm -rf "$LOCK_DIR"
            continue
        fi
        if [ "$waited" -ge "$LOCK_WAIT" ]; then
            log "Another start has held the lock for ${LOCK_WAIT}s; leaving it to finish"
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

# --- launchd ---
launchd_installed() { [ -f "$LAUNCHD_PLIST" ]; }
launchd_loaded()    { launchctl print "$LAUNCHD_DOMAIN/$LAUNCHD_LABEL" >/dev/null 2>&1; }

launchd_load() {
    launchd_loaded && return 0
    launchctl bootstrap "$LAUNCHD_DOMAIN" "$LAUNCHD_PLIST" 2>/dev/null || launchctl load -w "$LAUNCHD_PLIST"
}

launchd_unload() {
    launchd_loaded || return 0
    launchctl bootout "$LAUNCHD_DOMAIN/$LAUNCHD_LABEL" 2>/dev/null || launchctl unload -w "$LAUNCHD_PLIST"
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
    <!-- launchd restarts the server if it exits. The model choice lives in
         ~/.session-rag/config.json, so no environment plumbing is needed here. -->
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

# --- server process ---
spawn_server() {
    rotate_log
    (
        cd "$SCRIPT_DIR"
        export PYTHONPATH="$SCRIPT_DIR"
        export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
        export GRPC_VERBOSITY="${GRPC_VERBOSITY:-NONE}" GLOG_minloglevel="${GLOG_minloglevel:-2}"
        nohup "$PYTHON" -u "$SCRIPT_DIR/http_server.py" </dev/null >> "$LOG_FILE" 2>&1 &
        # Claim the pidfile at once so a concurrent caller sees a live PID before
        # the server has even finished importing its dependencies.
        echo $! > "$PID_FILE"
    )
    cat "$PID_FILE"
}

# Block up to START_CONFIRM seconds. 0 = healthy, 1 = still starting, 2 = process gone.
wait_for_server() {
    local pid=$1 waited=0
    while [ "$waited" -lt "$START_CONFIRM" ]; do
        sleep 1
        waited=$((waited + 1))
        if [ -n "$pid" ] && ! pid_alive "$pid"; then
            return 2
        fi
        if healthy; then
            echo "$waited"
            return 0
        fi
    done
    return 1
}

wait_for_exit() {
    local waited=0
    while [ "$waited" -lt 20 ]; do
        if [ -z "$(port_owner)" ] && ! pgrep -f "milvus_lite/lib/milvus .*milvus.db" >/dev/null 2>&1; then
            return 0
        fi
        sleep 1
        waited=$((waited + 1))
    done
    return 1
}

kill_pid() {
    local pid=$1 waited=0
    pid_alive "$pid" || return 0
    kill "$pid" 2>/dev/null || true
    while pid_alive "$pid" && [ "$waited" -lt 15 ]; do
        sleep 1
        waited=$((waited + 1))
    done
    pid_alive "$pid" && kill -9 "$pid" 2>/dev/null || true
    return 0
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
    return 0
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
    pid_alive "$wpid" && kill "$wpid" 2>/dev/null || true
    rm -f "$WATCHDOG_PID_FILE"
}

start_watchdog() {
    watchdog_alive && return 0
    rm -f "$WATCHDOG_PID_FILE"
    # Fully detached: inheriting the hook's pipe would keep `start` "running"
    # in Claude Code's eyes for the watchdog's whole life.
    nohup "$SCRIPT_DIR/session-rag-server.sh" __watchdog </dev/null >> "$LOG_FILE" 2>&1 &
    echo $! > "$WATCHDOG_PID_FILE"
    log "Watchdog started (PID $!)"
}

do_watchdog() {
    set +e                # a supervisor must not die on an incidental non-zero status
    trap '' HUP           # outlive the terminal that spawned us
    local my_pid=$$ launched_at failures=0 restarts=0 current now age
    launched_at=$(date +%s)
    while true; do
        sleep "$WATCHDOG_INTERVAL"
        current=$(cat "$WATCHDOG_PID_FILE" 2>/dev/null || true)
        if [ "$current" != "$my_pid" ]; then
            wlog "superseded by watchdog ${current:-none}; exiting (PID $my_pid)"
            exit 0
        fi
        if healthy 5; then
            if [ "$failures" -ne 0 ] || [ "$restarts" -ne 0 ]; then
                wlog "health restored"
            fi
            failures=0; restarts=0
            continue
        fi
        now=$(date +%s); age=$((now - launched_at))
        if ! is_alive; then
            if launchd_loaded; then
                wlog "server process gone; launchd will relaunch it"
                launched_at=$now
                continue
            fi
            wlog "server process gone; restarting now"
            failures=$WATCHDOG_MAX_FAILURES
        elif [ "$age" -lt "$STARTUP_GRACE" ]; then
            wlog "server alive but not healthy yet (${age}s of ${STARTUP_GRACE}s grace); waiting"
            continue
        else
            failures=$((failures + 1))
            wlog "health check failed ($failures/$WATCHDOG_MAX_FAILURES)"
        fi
        [ "$failures" -lt "$WATCHDOG_MAX_FAILURES" ] && continue

        wlog "server unresponsive; restarting"
        if "$SCRIPT_DIR/session-rag-server.sh" restart --no-watchdog >> "$LOG_FILE" 2>&1; then
            restarts=$((restarts + 1))
            wlog "restart issued (attempt $restarts)"
        else
            wlog "restart command failed; will retry"
            restarts=$((restarts + 1))
        fi
        launched_at=$(date +%s); failures=0
        if [ "$restarts" -gt 1 ]; then
            local backoff=$((WATCHDOG_INTERVAL * restarts))
            [ "$backoff" -gt "$WATCHDOG_MAX_BACKOFF" ] && backoff=$WATCHDOG_MAX_BACKOFF
            wlog "backing off ${backoff}s after $restarts consecutive restarts"
            sleep "$backoff"
        fi
    done
}

# --- commands ---
finish_start() {   # $1 = with_watchdog, $2 = pid (may be empty), $3 = wait_for_server rc, $4 = seconds
    local with_watchdog=$1 pid=$2 rc=$3 waited=${4:-}
    case $rc in
        0) log "Server ready (PID ${pid:-$(server_pid)}, ${waited}s$(launchd_loaded && echo ', launchd'))" ;;
        1) log "Server starting in background (PID ${pid:-$(server_pid)}); readiness handed to the watchdog" ;;
        2) if is_running; then
               log "Our launch lost a race but a healthy server is running (PID $(server_pid))"
           else
               log "Server process died during startup. Last log lines:"; tail -n 15 "$LOG_FILE" >&2
               return 1
           fi ;;
    esac
    [ "$with_watchdog" = yes ] && start_watchdog
    return 0
}

do_start() {
    local with_watchdog=${1:-yes} pid rc=0 waited=""
    if is_running; then
        log "Already running (PID $(server_pid))$(launchd_loaded && echo ' under launchd')"
        [ "$with_watchdog" = yes ] && start_watchdog
        return 0
    fi
    acquire_lock || return 0
    if is_running; then          # started by a concurrent caller while we waited
        release_lock
        log "Already running (PID $(server_pid))"
        [ "$with_watchdog" = yes ] && start_watchdog
        return 0
    fi
    if launchd_installed; then
        if launchd_loaded; then
            log "Starting via launchd ($LAUNCHD_LABEL) ..."
            launchctl kickstart "$LAUNCHD_DOMAIN/$LAUNCHD_LABEL" 2>/dev/null || true
        else
            log "Loading launchd agent ($LAUNCHD_LABEL) ..."
            launchd_load
        fi
        waited=$(wait_for_server "") || rc=$?
        release_lock
        finish_start "$with_watchdog" "" "$rc" "$waited"
        return $?
    fi
    pid=$( [ -f "$PID_FILE" ] && cat "$PID_FILE" 2>/dev/null || true )
    pid_alive "$pid" || pid=$(port_owner)
    if pid_alive "$pid"; then     # alive but not healthy: it is booting, do not spawn a rival
        release_lock
        log "Already starting (PID $pid); model preload in progress"
        [ "$with_watchdog" = yes ] && start_watchdog
        return 0
    fi
    rm -f "$PID_FILE"
    kill_port_holders
    log "Starting HTTP server on port $PORT ..."
    pid=$(spawn_server)
    waited=$(wait_for_server "$pid") || rc=$?
    release_lock
    finish_start "$with_watchdog" "$pid" "$rc" "$waited"
}

do_stop() {
    local keep_watchdog=${1:-no} pid
    [ "$keep_watchdog" = yes ] || stop_watchdog
    if launchd_loaded; then
        log "Unloading launchd agent ($LAUNCHD_LABEL) so the server stays stopped ..."
        launchd_unload
        wait_for_exit || log "Warning: a milvus process is still shutting down"
        rm -f "$PID_FILE"
        log "Stopped."
        return 0
    fi
    pid=$( [ -f "$PID_FILE" ] && cat "$PID_FILE" 2>/dev/null || true )
    pid_alive "$pid" || pid=$(port_owner)
    if pid_alive "$pid"; then
        log "Stopping server (PID $pid) ..."
        kill_pid "$pid"
        wait_for_exit || log "Warning: a milvus process is still shutting down"
        log "Stopped."
    else
        log "Not running."
    fi
    rm -f "$PID_FILE"
}

do_restart() {
    local with_watchdog=${1:-yes} rc=0 waited=""
    if launchd_loaded; then
        acquire_lock || return 0
        log "Restarting via launchd ($LAUNCHD_LABEL) ..."
        launchctl kickstart -k "$LAUNCHD_DOMAIN/$LAUNCHD_LABEL"
        sleep 1
        waited=$(wait_for_server "") || rc=$?
        release_lock
        finish_start "$with_watchdog" "" "$rc" "$waited"
        return $?
    fi
    [ "$with_watchdog" = yes ] && stop_watchdog
    do_stop yes
    do_start "$with_watchdog"
}

do_status() {
    local pid
    pid=$(server_pid)
    if [ -z "$pid" ]; then
        pid=$(port_owner)
        if pid_alive "$pid"; then
            log "Starting (PID $pid): not answering /health yet"
            return 0
        fi
        log "Not running."
        return 1
    fi
    if healthy; then
        log "Running (PID $pid)$(launchd_loaded && echo " under launchd [$LAUNCHD_LABEL]")"
        curl -s --max-time 3 "http://127.0.0.1:$PORT/status" | "$PYTHON" -m json.tool 2>/dev/null || true
    else
        log "Starting (PID $pid): not answering /health yet"
    fi
    if watchdog_alive; then log "Watchdog running (PID $(cat "$WATCHDOG_PID_FILE"))"; else log "Watchdog not running"; fi
    return 0
}

do_install_launchd() {
    stop_watchdog
    if launchd_loaded; then launchd_unload; wait_for_exit || true; fi
    local pid
    pid=$( [ -f "$PID_FILE" ] && cat "$PID_FILE" 2>/dev/null || true )
    pid_alive "$pid" || pid=$(port_owner)
    if pid_alive "$pid"; then kill_pid "$pid"; wait_for_exit || true; fi
    write_plist
    launchd_load
    log "launchd agent installed: $LAUNCHD_PLIST"
    local rc=0 waited=""
    waited=$(wait_for_server "") || rc=$?
    finish_start yes "" "$rc" "$waited"
}

do_uninstall_launchd() {
    stop_watchdog
    launchd_unload
    rm -f "$LAUNCHD_PLIST"
    wait_for_exit || true
    rm -f "$PID_FILE"
    log "launchd agent removed. ./session-rag-server.sh start runs the server from the hooks again."
}

cmd="${1:-start}"
shift || true
case "$cmd" in
    start)      do_start yes ;;
    stop)       do_stop ;;
    status)     do_status ;;
    restart)    if [ "${1:-}" = "--no-watchdog" ]; then do_restart no; else do_restart yes; fi ;;
    logs)       tail -n "${1:-50}" -f "$LOG_FILE" ;;
    install-launchd)   do_install_launchd ;;
    uninstall-launchd) do_uninstall_launchd ;;
    __watchdog) do_watchdog ;;
    *) echo "Usage: $0 {start|stop|status|restart|logs [n]|install-launchd|uninstall-launchd}" >&2; exit 1 ;;
esac
