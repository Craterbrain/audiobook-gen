#!/usr/bin/env bash
# Stops the Audiobook Gen GUI server (the generation jobs run separately and are not touched).
if pkill -f "[a]udiobook_gen.gui"; then
    command -v notify-send >/dev/null && notify-send "Audiobook Gen" "Stopped"
else
    command -v notify-send >/dev/null && notify-send "Audiobook Gen" "It wasn't running"
fi
