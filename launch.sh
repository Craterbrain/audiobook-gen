#!/usr/bin/env bash
# One-click launcher: starts the Audiobook Gen GUI if it isn't running, waits until it answers, opens it in Firefox
# (Waterfox has no audio playback for the preview players). Set LAUNCH_NO_BROWSER=1 to skip opening the browser.
cd "$(dirname "$(readlink -f "$0")")" || exit 1
PORT=7860; URL="http://127.0.0.1:$PORT"
up() { (exec 3<>"/dev/tcp/127.0.0.1/$PORT") 2>/dev/null; }
if ! up; then
    mkdir -p work
    command -v notify-send >/dev/null && notify-send -i "$PWD/assets/audiobook-gen.svg" "Audiobook Gen" "Starting…"
    # shellcheck disable=SC1091
    source .venv/bin/activate
    nohup python -m audiobook_gen.gui >> work/gui.log 2>&1 &
    for _ in $(seq 1 120); do up && break; sleep 1; done
fi
if ! up; then
    command -v notify-send >/dev/null && notify-send -u critical "Audiobook Gen" "The GUI did not start. See work/gui.log"
    exit 1
fi
[ -n "$LAUNCH_NO_BROWSER" ] && exit 0
if command -v firefox >/dev/null; then firefox "$URL" >/dev/null 2>&1 & else xdg-open "$URL" >/dev/null 2>&1 & fi
