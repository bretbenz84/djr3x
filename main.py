"""
main.py — Entry point for the DJ-R3X controller.

Start order
-----------
  1. Logging configured (console INFO + rotating file DEBUG).
  2. StateMachine constructed — all hardware opened, serial ports probed.
  3. Startup banner printed showing which hardware was detected.
  4. SIGINT / SIGTERM handlers registered → sm.request_shutdown().
  5. Startup animation played (blocking, ~2.7 s) before idle thread starts.
  6. sm.start() — loads wake word models, starts background threads.
  7. Startup chime played (blocking) through music output path.
  8. sm.run() — blocks until SHUTDOWN state plays out and OS halts.

Run
---
    python3 main.py
"""

from __future__ import annotations

import logging
import logging.handlers
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
        f"  Transcriber        {_hw(status['transcriber'])}   Whisper",
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

def main() -> None:
    _setup_logging()
    log.info("DJ-R3X controller starting up …")

    # 0. Allow USB audio devices to fully enumerate after boot
    log.info("Waiting for audio devices to initialize...")
    time.sleep(3)

    # 1. Construct state machine (probes serial ports, opens hardware)
    sm = StateMachine()

    # 2. Banner — hardware status known immediately after __init__
    _print_banner(sm.hardware_status())

    # 3. Signal handlers — after sm exists so the closure is valid
    _register_signals(sm)

    # 4. Startup animation — must run before sm.start() starts the servo
    #    idle thread (animation moves arm channels; idle thread would fight it)
    sm.play_startup_animation()

    # 5. Warmup models and start all background threads
    sm.start()

    # 6. Startup chime — after sm.start() so the AudioPlayer OutputStream is
    #    running, before sm.run() so it plays before entering IDLE/wake-word
    sm.play_startup_chime()

    # 7. Run — blocks until SHUTDOWN state halts the OS or an exception escapes
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
