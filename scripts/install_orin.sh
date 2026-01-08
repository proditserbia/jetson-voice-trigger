#!/usr/bin/env bash
set -euo pipefail

# Jetson Orin installer:
# - Installs apt dependencies
# - Creates venv in .venv
# - Builds CTranslate2 from source with CUDA (+cuDNN by default)
# - Avoids Intel MKL defaults on Jetson (ARM) by disabling MKL, using OpenBLAS,
#   and disabling OpenMP runtime selection (stable baseline).
# - Builds & installs CTranslate2 *Python* bindings from source (CUDA-enabled)
# - Installs Python deps
# - Runs a small self-check

VENV_DIR="${VENV_DIR:-.venv}"
CT2_DIR="${CT2_DIR:-CTranslate2}"
CT2_REF="${CT2_REF:-}"           # optional tag/branch/commit
WITH_CUDNN="${WITH_CUDNN:-ON}"   # ON/OFF
CLEAN_BUILD="${CLEAN_BUILD:-0}"  # 1 to wipe build dir (recommended after errors)
CLEAN_CT2="${CLEAN_CT2:-0}"      # 1 to wipe whole CTranslate2 folder (rarely needed)
LOG_DIR="${LOG_DIR:-logs}"

mkdir -p "$LOG_DIR"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/install_orin_${TS}.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "============================================================"
echo " Jetson Orin Install (CUDA CTranslate2 + faster-whisper)"
echo " Log: $LOG_FILE"
echo "============================================================"

echo ""
echo "[1/6] APT dependencies"
sudo apt update
sudo apt install -y \
  python3-venv python3-pip \
  build-essential cmake git pkg-config \
  libportaudio2 portaudio19-dev \
  libasound2-dev \
  ffmpeg \
  libgomp1 \
  libopenblas-dev \
  libcudnn9-dev

echo ""
echo "[2/6] Python venv: ${VENV_DIR}"
if [[ ! -d "$VENV_DIR" ]]; then
  python3 -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
python -m pip install -U pip wheel setuptools

echo ""
echo "[3/6] Python deps"
python -m pip install -U -r requirements.txt
python -m pip install -U faster-whisper

echo ""
echo "[4/6] CTranslate2 (CUDA build)"

if [[ "$CLEAN_CT2" == "1" ]]; then
  echo "CLEAN_CT2=1 -> removing ${CT2_DIR}"
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

if [[ "$CLEAN_BUILD" == "1" ]]; then
  echo "CLEAN_BUILD=1 -> removing build directory"
  rm -rf build
fi

mkdir -p build
pushd build

cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DWITH_CUDA=ON \
  -DWITH_CUDNN="${WITH_CUDNN}" \
  -DWITH_MKL=OFF \
  -DWITH_OPENBLAS=ON \
  -DOPENMP_RUNTIME=NONE

cmake --build . -j"$(nproc)"
sudo cmake --install .

# Make sure /usr/local libs are visible to the loader
sudo ldconfig

popd # build

echo ""
echo "[5/6] Installing CUDA-enabled CTranslate2 Python bindings into venv"

# IMPORTANT:
# If a CPU-only wheel was installed earlier, remove it first.
python -m pip uninstall -y ctranslate2 || true

# Build wheel from source (the repo root is NOT installable; python/ is)
pushd python
python -m pip install -U -r install_requirements.txt

# C++ lib installed via cmake --install typically lands in /usr/local
export CTRANSLATE2_ROOT="${CTRANSLATE2_ROOT:-/usr/local}"

python setup.py bdist_wheel
python -m pip install -U dist/*.whl
popd # python

popd # CT2_DIR

echo ""
echo "[6/6] Self-check"
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
