from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from faster_whisper import WhisperModel


@dataclass(frozen=True)
class ASRConfig:
    model: str = "tiny.en"
    compute: str = "int8"
    lang: Optional[str] = "en"
    cpu_threads: int = 4
    num_workers: int = 1
    asr_device: str = "auto"  # auto|cuda|cpu


def pcm16_to_float32(pcm16: np.ndarray) -> np.ndarray:
    return (pcm16.astype(np.float32) / 32768.0).clip(-1.0, 1.0)


class Transcriber:
    def __init__(self, cfg: ASRConfig):
        self.cfg = cfg
        self.model = self._init_model()

    def _init_model(self) -> WhisperModel:
        if self.cfg.asr_device not in ("auto", "cuda", "cpu"):
            raise ValueError("asr_device must be auto|cuda|cpu")

        if self.cfg.asr_device in ("auto", "cuda"):
            try:
                logging.info("ASR backend: CUDA (attempt)")
                return WhisperModel(
                    self.cfg.model,
                    device="cuda",
                    compute_type=self.cfg.compute,
                    cpu_threads=self.cfg.cpu_threads,
                    num_workers=self.cfg.num_workers,
                )
            except Exception as e:
                if self.cfg.asr_device == "cuda":
                    raise
                logging.warning("CUDA init failed; falling back to CPU: %r", e)

        logging.info("ASR backend: CPU")
        return WhisperModel(
            self.cfg.model,
            device="cpu",
            compute_type=self.cfg.compute,
            cpu_threads=self.cfg.cpu_threads,
            num_workers=self.cfg.num_workers,
        )

    def warmup(self, seconds: float = 1.0) -> None:
        if seconds <= 0:
            return
        logging.info("ASR warmup: %.2fs", seconds)
        n = int(16000 * seconds)
        audio = np.zeros(n, dtype=np.float32)
        try:
            self.model.transcribe(
                audio,
                language=self.cfg.lang,
                vad_filter=False,
                beam_size=1,
                best_of=1,
                condition_on_previous_text=False,
            )
        except Exception as e:
            logging.debug("Warmup failed (non-fatal): %r", e)

    def transcribe_pcm16(self, pcm16: np.ndarray) -> Tuple[str, float]:
        audio = pcm16_to_float32(pcm16)
        t0 = time.time()
        segments, _ = self.model.transcribe(
            audio,
            language=self.cfg.lang,
            vad_filter=False,
            beam_size=1,
            best_of=1,
            condition_on_previous_text=False,
            word_timestamps=False,
        )
        dt = time.time() - t0
        text = "".join(seg.text for seg in segments).strip()
        return text, dt
