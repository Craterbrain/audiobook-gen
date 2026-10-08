#!/usr/bin/env bash
# Creates .venv (Python 3.12), installs PyTorch XPU + deps, spaCy model.
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -d .venv ]; then
  if command -v python3.12 >/dev/null; then
    python3.12 -m venv .venv
  else
    # No system 3.12 (e.g. Arch ships 3.13+): bootstrap uv, let it fetch 3.12.
    python3 -m venv .uvboot && .uvboot/bin/pip install -q uv
    .uvboot/bin/uv venv --python 3.12 --seed .venv
    rm -rf .uvboot
  fi
fi
source .venv/bin/activate
pip install -U pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu
pip install -r requirements.txt
pip uninstall -y torchcodec  # PyPI wheel links CUDA libs; F5 engine loads audio via soundfile
python -m spacy download en_core_web_sm
python -c "import torch; print('xpu:', torch.xpu.is_available(), torch.xpu.get_device_name(0) if torch.xpu.is_available() else '-')"
