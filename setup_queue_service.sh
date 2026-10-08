#!/bin/bash
# Installs the job queue as a systemd USER service: it starts at boot (even before you log in), and systemd restarts it if it
# ever dies. The service runs the supervisor, which runs the queue runner, which watches every queued book.
#   ./setup_queue_service.sh            install and start
#   ./setup_queue_service.sh --remove   stop and uninstall
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
UNIT="$HOME/.config/systemd/user/audiobook-queue.service"
if [ "$1" = "--remove" ]; then
    systemctl --user disable --now audiobook-queue.service 2>/dev/null || true
    rm -f "$UNIT"; systemctl --user daemon-reload
    echo "Removed. The app will start the queue itself when it opens."; exit 0
fi
# a supervisor the app started by itself would clash with the service: stop it first (no job is lost; running jobs resume)
if [ -f "$HERE/work/queue/supervisor.pid" ]; then
    old=$(cat "$HERE/work/queue/supervisor.pid")
    if [ -d "/proc/$old" ] && ! systemctl --user is-active --quiet audiobook-queue.service; then
        kill "$old" 2>/dev/null || true
        pkill -f "[a]udiobook_gen.jobqueue run" 2>/dev/null || true
        sleep 2
    fi
fi
mkdir -p "$(dirname "$UNIT")"
cat > "$UNIT" <<EOF
[Unit]
Description=Audiobook generator job queue (runner and watchdog)

[Service]
Type=simple
WorkingDirectory=$HERE
Environment=PYTHONPATH=$HERE PYTHONUNBUFFERED=1
ExecStart=$HERE/.venv/bin/python -m audiobook_gen.jobqueue supervise
Restart=always
RestartSec=10
KillMode=control-group
TimeoutStopSec=60

[Install]
WantedBy=default.target
EOF
systemctl --user daemon-reload
systemctl --user enable --now audiobook-queue.service
if loginctl enable-linger "$USER" 2>/dev/null; then echo "Starts at boot, before you log in."
else echo "Could not enable start-before-login (needs permission): run  sudo loginctl enable-linger $USER  to get that. It still starts when you log in."; fi
systemctl --user --no-pager status audiobook-queue.service | head -6
