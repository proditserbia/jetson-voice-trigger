from __future__ import annotations

import logging
import queue
from dataclasses import dataclass
from typing import Optional

import numpy as np
import sounddevice as sd


@dataclass(frozen=True)
class AudioConfig:
    sample_rate: int = 16000
    channels: int = 1
    dtype: str = "float32"
    latency: str = "low"


class AudioInput:
    def __init__(self, cfg: AudioConfig, frame_samples: int, device: Optional[str] = None):
        self.cfg = cfg
        self.frame_samples = frame_samples
        self.device = device
        self.q: "queue.Queue[bytes]" = queue.Queue()

    def _cb(self, indata, frames, time_info, status) -> None:
        if status:
            logging.debug("Audio status: %s", status)
        pcm16 = (indata[:, 0] * 32768.0).astype(np.int16).tobytes()
        self.q.put(pcm16)

    def stream(self) -> sd.InputStream:
        return sd.InputStream(
            channels=self.cfg.channels,
            samplerate=self.cfg.sample_rate,
            dtype=self.cfg.dtype,
            blocksize=self.frame_samples,
            device=self.device,
            callback=self._cb,
            latency=self.cfg.latency,
        )
