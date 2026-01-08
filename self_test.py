#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Self-test for Jetson Orin Voice Trigger stack.

Prints:
- OS / JetPack (nv_tegra_release)
- CUDA toolchain
- CTranslate2 import + CUDA device count
- faster-whisper import
- Model init on CUDA with fallback to CPU

Optional:
- Quick transcription for a provided WAV file
"""

from __future__ import annotations

import argparse
import platform
import subprocess
import sys
from pathlib import Path


def sh(cmd: str) -> str:
    try:
        out = subprocess.check_output(["bash", "-lc", cmd], stderr=subprocess.STDOUT, text=True)
        return out.strip()
    except Exception as e:
        return f"[FAILED] {cmd}: {e}"


def read_file(path: str) -> str:
    try:
        return Path(path).read_text(errors="ignore").strip()
    except Exception as e:
        return f"[FAILED] read {path}: {e}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="tiny.en")
    ap.add_argument("--compute", default="float16")
    ap.add_argument("--wav", default=None, help="Optional: WAV path to transcribe")
    args = ap.parse_args()

    print("=== System ===")
    print("Python:", sys.version.replace("\n", " "))
    print("Platform:", platform.platform())
    print("Machine:", platform.machine())
    print("OS release:\n", read_file("/etc/os-release"))
    print("Jetson release:\n", read_file("/etc/nv_tegra_release"))
    print("nvcc --version:\n", sh("nvcc --version || true"))
    print("nvidia-smi:\n", sh("nvidia-smi || true"))
    print()

    print("=== Packages ===")
    try:
        import ctranslate2 as ct2  # type: ignore

        print("ctranslate2:", getattr(ct2, "__version__", "unknown"))
        try:
            print("cuda devices:", ct2.get_cuda_device_count())
        except Exception as e:
            print("cuda devices: FAILED:", repr(e))
    except Exception as e:
        print("ctranslate2 import FAILED:", repr(e))
        return 2

    try:
        from faster_whisper import WhisperModel  # type: ignore

        print("faster-whisper: import OK")
    except Exception as e:
        print("faster-whisper import FAILED:", repr(e))
        return 3

    print()
    print("=== Model init ===")
    cuda_ok = False
    try:
        WhisperModel(args.model, device="cuda", compute_type=args.compute)
        print(f"CUDA init OK (model={args.model}, compute={args.compute})")
        cuda_ok = True
    except Exception as e:
        print("CUDA init FAILED:", repr(e))

    try:
        WhisperModel(args.model, device="cpu", compute_type="float32")
        print(f"CPU init OK (model={args.model}, compute=float32)")
    except Exception as e:
        print("CPU init FAILED:", repr(e))
        return 4

    if args.wav:
        wav = Path(args.wav)
        if not wav.exists():
            print("WAV not found:", wav)
            return 5

        device = "cuda" if cuda_ok else "cpu"
        compute = args.compute if cuda_ok else "float32"
        print()
        print("=== Transcription test ===")
        print(f"Using device={device}, compute={compute}, wav={wav}")
        m = WhisperModel(args.model, device=device, compute_type=compute)
        segments, _ = m.transcribe(str(wav), beam_size=1, best_of=1)
        text = "".join(seg.text for seg in segments).strip()
        print("Text:", text if text else "(empty)")

    print()
    print("✅ Self-test complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
