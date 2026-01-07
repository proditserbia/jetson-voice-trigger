#!/usr/bin/env bash
set -euo pipefail

# Jetson Orin installer:
# - Installs apt dependencies
# - Creates venv in .venv
# - Builds CTranslate2 from source with CUDA (+cuDNN by default)
# - Installs Python deps
# - Runs a small self-check

VENV_DIR="${VENV_DIR:-.venv}"
CT2_DIR="${CT2_DIR:-CTranslate2}"
CT2_REF="${CT2_REF:-}"          # optional tag/branch/commit
WITH_CUDNN="${WITH_CUDNN:-ON}"  # ON/OFF
CLEAN_BUILD="${CLEAN_BUILD:-0}" # 1 to wipe CTranslate2 checkout
LOG_DIR="${LOG_DIR:-logs}"

mkdir -p "$LOG_DIR"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/install_orin_${TS}.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "============================================================"
echo " Jetson Orin Install (CUDA CTranslate2 + faster-whisper)"
echo " Log: $LOG_FILE"
echo "============================================================"

sudo apt update
sudo apt install -y \
  python3-venv python3-pip \
  build-essential cmake git pkg-config \
  libportaudio2 portaudio19-dev \
  libasound2-dev \
  ffmpeg

if [[ ! -d "$VENV_DIR" ]]; then
  python3 -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
python -m pip install -U pip wheel setuptools

python -m pip install -U -r requirements.txt
python -m pip install -U faster-whisper

if [[ "$CLEAN_BUILD" == "1" ]]; then
  rm -rf "$CT2_DIR"
fi

if [[ ! -d "$CT2_DIR/.git" ]]; then
  git clone https://github.com/OpenNMT/CTranslate2.git "$CT2_DIR"
fi

pushd "$CT2_DIR"
git fetch --all --tags
if [[ -n "$CT2_REF" ]]; then
  git checkout "$CT2_REF"
fi
git submodule update --init --recursive

mkdir -p build
pushd build

cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DWITH_CUDA=ON \
  -DWITH_CUDNN="${WITH_CUDNN}"

cmake --build . -j"$(nproc)"
sudo cmake --install .

popd # build
python -m pip install -U .
popd # CT2_DIR

echo ""
echo "=== Self-check ==="
python - <<'PY'
import ctranslate2 as ct2
print("ctranslate2:", ct2.__version__)
print("cuda devices:", ct2.get_cuda_device_count())
from faster_whisper import WhisperModel
print("faster-whisper: import OK")
try:
    WhisperModel("tiny.en", device="cuda", compute_type="float16")
    print("WhisperModel CUDA init: OK")
except Exception as e:
    print("WhisperModel CUDA init: FAILED:", repr(e))
    WhisperModel("tiny.en", device="cpu", compute_type="int8")
    print("WhisperModel CPU init: OK")
PY

echo ""
echo "✅ Done."
echo "Next:"
echo "  source ${VENV_DIR}/bin/activate"
echo "  python self_test.py --model tiny.en"
echo "  python -m voice_trigger.app --asr_device auto --model medium.en --compute float16 --lang en"
