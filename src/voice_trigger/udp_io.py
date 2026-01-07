from __future__ import annotations

import logging
import socket
import threading
from dataclasses import dataclass
from typing import Callable, Optional


@dataclass(frozen=True)
class UDPConfig:
    enable_in: bool = False
    host: str = "0.0.0.0"
    port: int = 9999
    token: Optional[str] = None
    out_host: Optional[str] = None
    out_port: int = 9999
    allow_cmd: bool = False  # security: off by default


def send_udp(msg: str, host: str, port: int, token: Optional[str] = None) -> None:
    try:
        if token:
            msg = f"{token}:{msg}"
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.sendto(msg.encode("utf-8", errors="ignore"), (host, port))
        logging.debug("[UDP OUT] %s:%s <- %s", host, port, msg)
    except Exception as e:
        logging.warning("UDP send failed: %r", e)


def listener_thread(
    stop_evt: threading.Event,
    cfg: UDPConfig,
    on_trigger: Callable[[str], None],
    on_cmd: Callable[[str], None],
    on_pause: Callable[[], None],
    on_resume: Callable[[], None],
) -> None:
    if not cfg.enable_in:
        return

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.bind((cfg.host, cfg.port))
    except Exception as e:
        logging.error("UDP bind failed on %s:%s -> %r", cfg.host, cfg.port, e)
        return

    sock.settimeout(0.5)
    logging.info("UDP listener: %s:%s%s", cfg.host, cfg.port, " (token required)" if cfg.token else "")

    try:
        while not stop_evt.is_set():
            try:
                data, addr = sock.recvfrom(65535)
            except socket.timeout:
                continue
            except Exception as e:
                logging.debug("UDP recv error: %r", e)
                continue

            msg = data.decode("utf-8", errors="ignore").strip()
            logging.debug("[UDP IN] %s -> %s", addr, msg)

            if cfg.token:
                if ":" not in msg or not msg.startswith(cfg.token + ":"):
                    logging.debug("UDP token mismatch; ignoring")
                    continue
                msg = msg.split(":", 1)[1]

            up = msg.upper()

            if up.startswith("CTRL:"):
                action = up.split(":", 1)[1].strip()
                if action == "PAUSE":
                    on_pause()
                elif action == "RESUME":
                    on_resume()
                else:
                    logging.debug("Unknown CTRL action: %s", action)
                continue

            if up.startswith("TRIGGER:"):
                phrase = msg.split(":", 1)[1].strip()
                on_trigger(phrase)
                continue

            if up.startswith("CMD:"):
                if not cfg.allow_cmd:
                    logging.debug("UDP CMD ignored (allow_cmd=False)")
                    continue
                shell = msg.split(":", 1)[1].strip()
                if shell:
                    on_cmd(shell)
                continue

            logging.debug("UDP ignored (unknown format): %s", msg)
    finally:
        sock.close()
        logging.debug("UDP listener stopped.")
