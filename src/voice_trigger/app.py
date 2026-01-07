from __future__ import annotations

import argparse
import logging
import queue
import signal
import subprocess
import threading
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .audio import AudioConfig, AudioInput
from .asr import ASRConfig, Transcriber
from .config import load_triggers_json
from .matcher import PhraseMatcher
from .udp_io import UDPConfig, listener_thread, send_udp
from .utils import normalize_text, setup_logging
from .vad import SpeechSegmenter, VADConfig


_LISTENING_EVT = threading.Event()
_LISTENING_EVT.set()


def set_listening(flag: bool) -> None:
    if flag:
        _LISTENING_EVT.set()
        logging.info("Listening RESUMED")
    else:
        _LISTENING_EVT.clear()
        logging.info("Listening PAUSED")


def is_listening() -> bool:
    return _LISTENING_EVT.is_set()


def run_command(cmd: str) -> None:
    try:
        subprocess.Popen(cmd, shell=True)
    except Exception as e:
        logging.warning("Command failed: %r", e)


def prefetch_model(model_name: str) -> None:
    logging.info("Prefetching model: %s", model_name)
    try:
        from faster_whisper import WhisperModel  # type: ignore

        _ = WhisperModel(model_name, device="cpu", compute_type="int8")
        logging.info("Prefetch done (model cached).")
    except Exception as e:
        logging.warning("Prefetch failed (non-fatal): %r", e)


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Offline voice trigger (VAD + faster-whisper)")

    ap.add_argument("--device", type=str, default=None, help="sounddevice input device name/index (optional)")

    ap.add_argument("--model", type=str, default="tiny.en", help="faster-whisper model name or local path")
    ap.add_argument("--compute", type=str, default="int8", help="compute type: int8|int8_float16|float16|float32")
    ap.add_argument("--lang", type=str, default="en", help="language hint, or empty for auto")
    ap.add_argument("--asr_device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="ASR device")
    ap.add_argument("--cpu_threads", type=int, default=4, help="CPU threads for ASR backend")
    ap.add_argument("--num_workers", type=int, default=1, help="CTranslate2 workers")

    ap.add_argument("--triggers", type=str, default=None, help="path to triggers JSON (optional)")
    ap.add_argument("--threshold", type=int, default=85, help="fuzzy match threshold 0-100")
    ap.add_argument("--cooldown", type=float, default=4.0, help="cooldown seconds per trigger")
    ap.add_argument("--debug", action="store_true", help="enable debug logs")

    ap.add_argument("--vad_level", type=int, default=3, help="webrtcvad aggressiveness 0..3")
    ap.add_argument("--max_segment", type=float, default=2.0, help="max speech segment length (sec)")
    ap.add_argument("--min_speech", type=float, default=0.25, help="min speech length (sec) before ASR")
    ap.add_argument("--speech_pad_ms", type=int, default=120, help="padding after speech end (ms)")

    ap.add_argument("--udp_in", action="store_true", help="enable UDP listener")
    ap.add_argument("--udp_host", type=str, default="0.0.0.0", help="UDP listen host")
    ap.add_argument("--udp_port", type=int, default=9999, help="UDP listen port")
    ap.add_argument("--udp_token", type=str, default=None, help="UDP token (recommended)")
    ap.add_argument("--udp_out_host", type=str, default=None, help="UDP out host (optional)")
    ap.add_argument("--udp_out_port", type=int, default=9999, help="UDP out port (optional)")
    ap.add_argument("--allow_udp_cmd", action="store_true", help="allow UDP CMD:<shell> (DANGEROUS)")

    ap.add_argument("--prefetch-model", action="store_true", help="download model weights before start")
    ap.add_argument("--warmup-sec", type=float, default=0.0, help="warmup decode seconds after model init")

    args = ap.parse_args(argv)
    setup_logging(args.debug)

    triggers_raw = load_triggers_json(args.triggers)
    triggers = {normalize_text(k): v for k, v in triggers_raw.items()}

    matcher = PhraseMatcher(triggers, threshold=args.threshold, cooldown_sec=args.cooldown)

    if args.prefetch_model:
        prefetch_model(args.model)

    asr_cfg = ASRConfig(
        model=args.model,
        compute=args.compute,
        lang=(args.lang or None),
        cpu_threads=args.cpu_threads,
        num_workers=args.num_workers,
        asr_device=args.asr_device,
    )
    transcriber = Transcriber(asr_cfg)

    if args.warmup_sec and args.warmup_sec > 0:
        transcriber.warmup(args.warmup_sec)

    vad_cfg = VADConfig(
        sample_rate=16000,
        frame_ms=20,
        aggressiveness=args.vad_level,
        max_segment_sec=args.max_segment,
        min_speech_sec=args.min_speech,
        speech_pad_ms=args.speech_pad_ms,
    )
    segmenter = SpeechSegmenter(vad_cfg)

    audio_cfg = AudioConfig(sample_rate=vad_cfg.sample_rate, channels=1, dtype="float32", latency="low")
    audio = AudioInput(audio_cfg, frame_samples=segmenter.frame_len, device=args.device)

    udp_cfg = UDPConfig(
        enable_in=args.udp_in,
        host=args.udp_host,
        port=args.udp_port,
        token=args.udp_token,
        out_host=args.udp_out_host,
        out_port=args.udp_out_port,
        allow_cmd=args.allow_udp_cmd,
    )

    stop_evt = threading.Event()
    segments_q: "queue.Queue[bytes]" = queue.Queue()

    def on_pause():
        set_listening(False)

    def on_resume():
        set_listening(True)

    def on_trigger(phrase: str):
        p = normalize_text(phrase)
        cmd = triggers.get(p)
        if cmd:
            logging.info("UDP trigger: '%s' -> run", p)
            run_command(cmd)
        else:
            logging.debug("UDP trigger not found: '%s'", p)

    def on_cmd(shell: str):
        logging.info("UDP CMD -> %s", shell)
        run_command(shell)

    if udp_cfg.enable_in:
        threading.Thread(
            target=listener_thread,
            args=(stop_evt, udp_cfg, on_trigger, on_cmd, on_pause, on_resume),
            daemon=True,
        ).start()

    signal.signal(signal.SIGINT, lambda *_: stop_evt.set())

    def asr_worker():
        min_frames = segmenter.min_frames_before_transcribe()
        frame_bytes = segmenter.frame_bytes
        while not stop_evt.is_set():
            try:
                seg = segments_q.get(timeout=0.2)
            except queue.Empty:
                continue

            if not is_listening():
                continue

            total_frames = len(seg) // frame_bytes
            if total_frames < min_frames:
                logging.debug("Segment too short, skipping (%s frames)", total_frames)
                continue

            pcm16 = np.frombuffer(seg, dtype=np.int16)
            text, dt = transcriber.transcribe_pcm16(pcm16)
            if not text:
                logging.debug("Empty transcription.")
                continue

            logging.info("ASR: %s (%.2fs)", text, dt)
            result, best_score = matcher.match(text)
            if result:
                logging.info("TRIGGER matched: '%s' (score %s) -> run", result.phrase, result.score)
                run_command(result.command)
                if udp_cfg.out_host:
                    send_udp(f"TRIGGER:{result.phrase}", udp_cfg.out_host, udp_cfg.out_port, udp_cfg.token)
            else:
                logging.debug("No trigger matched (best score: %s)", best_score)

    threading.Thread(target=asr_worker, daemon=True).start()

    logging.info("Listening... Ctrl+C to stop.")
    with audio.stream():
        while not stop_evt.is_set():
            try:
                frame = audio.q.get(timeout=0.2)
            except queue.Empty:
                continue

            if not is_listening():
                continue

            seg, _in_speech = segmenter.process_one_frame(frame)
            if seg is not None:
                segments_q.put(seg)

    logging.info("Bye.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
