from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Tuple

import webrtcvad


@dataclass(frozen=True)
class VADConfig:
    sample_rate: int = 16000
    frame_ms: int = 20
    aggressiveness: int = 3
    max_segment_sec: float = 2.0
    min_speech_sec: float = 0.25
    speech_pad_ms: int = 120


class SpeechSegmenter:
    def __init__(self, cfg: VADConfig):
        self.cfg = cfg
        self.vad = webrtcvad.Vad(cfg.aggressiveness)

        self.frame_len = int(cfg.sample_rate * (cfg.frame_ms / 1000.0))
        self.frame_bytes = self.frame_len * 2
        self.pad_frames = int(cfg.speech_pad_ms / cfg.frame_ms)

        self.reset()

    def reset(self) -> None:
        self.speech_frames: list[bytes] = []
        self.trailing_non_speech = 0
        self.in_speech = False
        self.start_time: Optional[float] = None

    def process_one_frame(self, frame_bytes: bytes) -> Tuple[Optional[bytes], bool]:
        if len(frame_bytes) != self.frame_bytes:
            raise ValueError(f"Expected {self.frame_bytes} bytes, got {len(frame_bytes)}")

        is_speech = self.vad.is_speech(frame_bytes, self.cfg.sample_rate)

        if is_speech:
            if not self.in_speech:
                self.in_speech = True
                self.start_time = time.time()
            self.speech_frames.append(frame_bytes)
            self.trailing_non_speech = 0
        else:
            if self.in_speech:
                self.trailing_non_speech += 1
                if self.trailing_non_speech <= self.pad_frames:
                    self.speech_frames.append(frame_bytes)
                if self.trailing_non_speech >= self.pad_frames:
                    seg = b"".join(self.speech_frames)
                    self.reset()
                    return seg, False

        if self.in_speech and self.start_time and (time.time() - self.start_time) > self.cfg.max_segment_sec:
            seg = b"".join(self.speech_frames)
            self.reset()
            return seg, False

        return None, self.in_speech

    def min_frames_before_transcribe(self) -> int:
        return int((self.cfg.min_speech_sec * 1000) / self.cfg.frame_ms)
