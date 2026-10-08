#!/usr/bin/env bash
# Optional engines, each in its own virtualenv because their libraries pin different transformers versions.
# They reuse the main .venv's PyTorch XPU build (via a .pth file), so it is not downloaded twice.
#   ./setup_engines.sh chatterbox | qwen3 | all
set -euo pipefail
cd "$(dirname "$0")"
[ -d .venv ] || { echo "Run ./setup.sh first."; exit 1; }
MAIN_SITE=$(.venv/bin/python -c "import site;print(site.getsitepackages()[0])")

make_env() {   # $1 = directory name
    [ -d "$1" ] || .venv/bin/python -m venv "$1"
    echo "$MAIN_SITE" > "$("$1/bin/python" -c "import site;print(site.getsitepackages()[0])")/zz_main_venv.pth"
}

qwen3() {
    make_env .venv-tts2
    .venv-tts2/bin/pip install -q qwen-tts
}

chatterbox() {
    make_env .venv-chatterbox
    # --no-deps: its pins would replace the XPU PyTorch with a CUDA build
    .venv-chatterbox/bin/pip install -q --no-deps chatterbox-tts
    .venv-chatterbox/bin/pip install -q "librosa==0.11.0" s3tokenizer "diffusers==0.29.0" resemble-perth \
        "conformer==0.3.2" spacy-pkuseg "pykakasi==2.3.0" pyloudnorm omegaconf
}

case "${1:-all}" in
    qwen3) qwen3 ;;
    chatterbox) chatterbox ;;
    all) chatterbox; qwen3 ;;
    *) echo "usage: $0 chatterbox|qwen3|all"; exit 1 ;;
esac
echo "Engines installed. Models download on first use."
