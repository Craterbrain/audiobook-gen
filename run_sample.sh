#!/usr/bin/env bash
# Minimal end-to-end test: John 1-3 (KJV) -> out/john_kjv.m4b
set -euo pipefail
cd "$(dirname "$0")"
source .venv/bin/activate
python -m audiobook_gen extract samples/john_kjv.txt --max-chapters 3
python -m audiobook_gen parse   samples/john_kjv.txt
python -m audiobook_gen lexicon samples/john_kjv.txt --seed-john
python -m audiobook_gen synth   samples/john_kjv.txt
python -m audiobook_gen assemble samples/john_kjv.txt --title "The Gospel According to John" --author "KJV"
ffprobe -v error -show_chapters -show_entries chapter=start_time,end_time:chapter_tags=title -of compact out/john_kjv.m4b
ffprobe -v error -show_entries stream=codec_name,codec_type -of compact out/john_kjv.m4b
