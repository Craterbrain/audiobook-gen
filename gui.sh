#!/usr/bin/env bash
cd "$(dirname "$0")" && source .venv/bin/activate && exec python -m audiobook_gen.gui
