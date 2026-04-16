"""
main.py — Entry point for the DJ-R3X controller.

Start order
-----------
  1. Logging configured (console INFO + rotating file DEBUG).
  2. StateMachine constructed — all hardware opened, serial ports probed.
  3. Startup banner printed showing which hardware was detected.
  4. SIGINT / SIGTERM handlers registered → sm.request_shutdown().
  5. light_speed.mp3 + servo animation run concurrently; both complete before continuing.
  6. Roger Control.mp3 plays through speech path with mouth LEDs and servo animation.
  7. sm.start() — loads wake word models, starts background threads.
  8. Startup chime played (blocking) through music output path.
  9. sm.run() — blocks until SHUTDOWN state plays out and OS halts.

Run
---
    python3 main.py
"""

from __future__ import annotations

import logging
import logging.handlers
import os
import signal
import sys
import time
from pathlib import Path

import config
from states.state_machine import StateMachine

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

_LOG_FILE = config.PROJECT_ROOT / "djr3x.log"
_LOG_FORMAT = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s"
_LOG_DATE   = "%Y-%m-%d %H:%M:%S"
_THIRD_PARTY_LOGGERS = (
    "openai",
    "httpx",
    "httpcore",
    "websockets",
)


def _log_level_from_name(name: str) -> int:
    return getattr(logging, name.upper(), logging.WARNING)


def _setup_logging() -> None:
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)   # capture everything; handlers filter

    # Console — INFO and above so the operator can follow what's happening
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_LOG_DATE))
    root.addHandler(console)

    # Rotating file — DEBUG for post-mortem diagnostics (5 × 1 MB)
    try:
        file_handler = logging.handlers.RotatingFileHandler(
            _LOG_FILE,
            maxBytes=1_048_576,
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_LOG_DATE))
        root.addHandler(file_handler)
    except OSError as exc:
        logging.warning("Could not open log file %s: %s — file logging disabled", _LOG_FILE, exc)

    third_party_level = _log_level_from_name(config.THIRD_PARTY_LOG_LEVEL)
    for logger_name in _THIRD_PARTY_LOGGERS:
        logging.getLogger(logger_name).setLevel(third_party_level)


log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Startup banner
# ---------------------------------------------------------------------------

_TICK  = "CONNECTED"
_CROSS = "MISSING   "

def _hw(ok: bool) -> str:
    return _TICK if ok else _CROSS

def _print_banner(status: dict) -> None:
    w = 62
    sep  = "=" * w
    thin = "-" * w

    models = status["wake_models"]
    wake_detail = f"({models}/2 model{'s' if models != 1 else ''})"
    music = status["music_tracks"]
    music_detail = f"({music} track{'s' if music != 1 else ''})" if music else "(no tracks)"

    transcriber_backend = "mlx-whisper (local)" if config.USE_LOCAL_TRANSCRIPTION else "Whisper API"
    llm_backend         = f"Ollama {config.LOCAL_LLM_MODEL} (local)" if config.USE_LOCAL_LLM else f"GPT-4o-mini"

    lines = [
        "",
        sep,
        f"  DJ R-3X Controller",
        f"  {config.PROJECT_ROOT}",
        thin,
        f"  Servos (Maestro)   {_hw(status['servos'])}   {config.MAESTRO_PORT}",
        f"  Chest LEDs (Nano)  {_hw(status['chest_leds'])}   {config.NANO_CHEST_PORT}",
        f"  Head LEDs  (Nano)  {_hw(status['head_leds'])}   {config.NANO_HEAD_PORT}",
        thin,
        f"  Wake word          {_hw(status['wake_word'])}   {wake_detail}",
        f"  Transcription      {_hw(status['transcriber'])}   {transcriber_backend}",
        f"  LLM                {'READY    '}   {llm_backend}",
        f"  TTS                {'READY    '}   ElevenLabs",
        f"  Camera             {_hw(status['camera'])}",
        f"  Face recognition   {_hw(status['face_recognition'])}",
        f"  Music library      {'READY    ' if music else 'EMPTY    '}   {music_detail}",
        thin,
        f"  HEY HEY HEY!  Rex is {'online' if any([status['servos'], status['chest_leds'], status['head_leds']]) else 'online (no hardware)'}.",
        sep,
        "",
    ]
    print("\n".join(lines))


# ---------------------------------------------------------------------------
# Signal handling
# ---------------------------------------------------------------------------

def _register_signals(sm: StateMachine) -> None:
    """Route SIGINT (Ctrl-C) and SIGTERM to a clean state machine shutdown."""
    def _handler(signum: int, _frame) -> None:
        name = signal.Signals(signum).name
        log.info("Signal %s received — requesting shutdown", name)
        sm.request_shutdown()

    signal.signal(signal.SIGINT,  _handler)
    signal.signal(signal.SIGTERM, _handler)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _started_as_service() -> bool:
    """Best-effort check for systemd/service-style launches.

    Interactive terminal runs should skip the startup enumeration delay, while
    service-managed launches keep it because USB/audio devices may still be
    settling right after boot.
    """
    if os.environ.get("INVOCATION_ID") or os.environ.get("JOURNAL_STREAM"):
        return True
    return not sys.stdin.isatty()


def main() -> None:
    _setup_logging()
    log.info("DJ-R3X controller starting up …")

    # 0. Allow USB serial devices (Maestro ACM, Nano USBs) and audio devices
    #    to fully enumerate after boot.  Multiple USB devices connecting
    #    simultaneously on the Pi can take 3-5 s; the retry logic in
    #    ServoController and LEDController will handle stragglers, but starting
    #    with a generous wait reduces the number of retries needed in practice.
    if _started_as_service():
        log.info("Waiting 5 s for USB serial and audio devices to enumerate …")
        time.sleep(5)
    else:
        log.info("Interactive launch detected — skipping startup enumeration delay")

    # 1. Construct state machine (probes serial ports, opens hardware)
    sm = StateMachine()

    # 2. Banner — hardware status known immediately after __init__
    _print_banner(sm.hardware_status())

    # 3. Signal handlers — after sm exists so the closure is valid
    _register_signals(sm)

    # 4. light_speed.mp3 and servo startup animation run concurrently; both
    #    complete before continuing.  Must run before sm.start() so the servo
    #    idle thread is not fighting the animation's arm movements.
    sm.play_startup_animation()

    # 5. Spoken intro through speech path — mouth LEDs and servo speak animation
    #    are active.  Must run before sm.start() so background threads don't
    #    compete for hardware.  Skipped gracefully if file is missing.
    sm.play_startup_intro()

    # 6. Warmup models and start all background threads
    sm.start()

    # 7. Startup chime — after sm.start() so the AudioPlayer OutputStream is
    #    running, before sm.run() so it plays before entering IDLE/wake-word
    sm.play_startup_chime()

    # 8. Run — blocks until SHUTDOWN state halts the OS or an exception escapes
    sm.run()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        # SIGINT delivered before the handler was registered, or during
        # a non-interruptible C extension (rare). Exit quietly.
        sys.exit(0)
    except Exception:
        log.exception("Fatal unhandled exception — Rex has crashed")
        sys.exit(1)
