"""
config.py — Configuration loader for DJ-R3X.

Loads .env and exposes all project-wide constants. No other module reads
.env directly — import what you need from here.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT    = Path(__file__).parent.resolve()
ASSETS_DIR      = PROJECT_ROOT / "assets"
AUDIO_CACHE_DIR = ASSETS_DIR / "audio"
MODELS_DIR      = ASSETS_DIR / "models"

# Default Vosk model path — override with VOSK_MODEL_PATH in .env if needed.
VOSK_MODEL_PATH_DEFAULT = MODELS_DIR / "vosk-model-small-en-us-0.15"

# ---------------------------------------------------------------------------
# Load .env
# ---------------------------------------------------------------------------

load_dotenv(PROJECT_ROOT / ".env")

def _require(key: str) -> str:
    """Return env var value or raise at startup with a clear message."""
    val = os.getenv(key)
    if not val:
        raise EnvironmentError(f"Required environment variable '{key}' is not set in .env")
    return val

def _optional(key: str, default: str = "") -> str:
    return os.getenv(key, default)

def _optional_int(key: str, default: int | None = None) -> int | None:
    val = os.getenv(key)
    if val is None or val == "":
        return default
    return int(val)

# ---------------------------------------------------------------------------
# API Keys
# ---------------------------------------------------------------------------

OPENAI_API_KEY      = _require("OPENAI_API_KEY")
ELEVENLABS_API_KEY  = _require("ELEVENLABS_API_KEY")
ELEVENLABS_VOICE_ID = _require("ELEVENLABS_VOICE_ID")

# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------

OPENAI_MODEL        = "gpt-4o-mini"
MAX_HISTORY_TURNS   = 6      # number of user/assistant pairs kept in context

# ---------------------------------------------------------------------------
# Serial — Pololu Maestro Mini
# ---------------------------------------------------------------------------

MAESTRO_PORT = _optional("MAESTRO_PORT", "/dev/ttyACM0")
MAESTRO_BAUD = 9600          # Pololu default; must match Maestro USB settings

# ---------------------------------------------------------------------------
# Serial — Arduino Nano LED controller(s)
# ---------------------------------------------------------------------------

NANO_CHEST_PORT = _optional("NANO_CHEST_PORT", "/dev/ttyUSB0")
NANO_HEAD_PORT  = _optional("NANO_HEAD_PORT",  "/dev/ttyUSB1")
LED_NANO_BAUD   = 9600

# ---------------------------------------------------------------------------
# Servo channel assignments (Maestro channel numbers, 0-based)
# ---------------------------------------------------------------------------

SERVO_HEAD_TILT   = 0
SERVO_HEAD_PAN    = 1
SERVO_VISOR       = 2
SERVO_ARM_LEFT    = 3
SERVO_ARM_RIGHT   = 4
SERVO_HAND_LEFT   = 5
SERVO_HAND_RIGHT  = 6

# Maestro position values are in quarter-microseconds.
# Standard servo center = 1500 µs = 6000 qµs; typical range 4000–8000.

# Home/neutral positions for each channel (qµs)
SERVO_HOME = {
    SERVO_HEAD_TILT:  6000,
    SERVO_HEAD_PAN:   6000,
    SERVO_VISOR:      5500,
    SERVO_ARM_LEFT:   6000,
    SERVO_ARM_RIGHT:  6000,
    SERVO_HAND_LEFT:  6000,
    SERVO_HAND_RIGHT: 6000,
}

# Per-channel (min, max) travel limits enforced in software (qµs)
SERVO_LIMITS = {
    SERVO_HEAD_TILT:  (4500, 7500),
    SERVO_HEAD_PAN:   (4000, 8000),
    SERVO_VISOR:      (4000, 7000),
    SERVO_ARM_LEFT:   (4000, 8000),
    SERVO_ARM_RIGHT:  (4000, 8000),
    SERVO_HAND_LEFT:  (4500, 7500),
    SERVO_HAND_RIGHT: (4500, 7500),
}

# Emotion bias: each emotion overrides (min, max) for affected channels.
# Only channels that differ from SERVO_LIMITS need to be listed.
SERVO_EMOTION_LIMITS = {
    "excited": {
        SERVO_HEAD_TILT: (5500, 7500),   # head up
        SERVO_VISOR:     (5500, 7000),   # visor high
    },
    "sad": {
        SERVO_HEAD_TILT: (4500, 5500),   # head down
        SERVO_VISOR:     (4000, 5000),   # visor low
    },
    "neutral": {},  # use default SERVO_LIMITS
}

# Random idle motion timing (seconds)
SERVO_IDLE_MOVE_INTERVAL_MIN = 1.5   # minimum time between random moves
SERVO_IDLE_MOVE_INTERVAL_MAX = 4.0   # maximum time between random moves

# Maestro move speed (0 = unlimited; units = (0.25µs) / (10ms))
SERVO_DEFAULT_SPEED        = 20
SERVO_EXCITED_SPEED        = 40
SERVO_SAD_SPEED            = 8

# ---------------------------------------------------------------------------
# Audio — recording / wake word
# ---------------------------------------------------------------------------

AUDIO_SAMPLE_RATE  = 16000   # Hz — required by both Vosk and OpenWakeWord
AUDIO_CHANNELS     = 1       # mono mic input
AUDIO_CHUNK_SIZE   = 1024    # frames per buffer read
AUDIO_FORMAT       = 8       # pyaudio.paInt16 == 8 (avoids importing pyaudio here)

MIC_DEVICE_INDEX     = _optional_int("MIC_DEVICE_INDEX")     # None → system default
SPEAKER_DEVICE_INDEX = _optional_int("SPEAKER_DEVICE_INDEX") # None → system default

# Silence-gated recording: stop capturing when RMS drops below threshold
# for SILENCE_DURATION consecutive seconds (or MAX_RECORD_SECONDS elapses)
SILENCE_THRESHOLD    = 300    # RMS amplitude (0–32767)
SILENCE_DURATION     = 1.2    # seconds of silence to end capture
MAX_RECORD_SECONDS   = 12.0   # hard cap on a single utterance

# ---------------------------------------------------------------------------
# Audio — mouth LED brightness
# ---------------------------------------------------------------------------

MOUTH_LED_SMOOTHING = 0.25   # exponential smoothing factor (0=frozen, 1=raw)
MOUTH_LED_GAIN      = 3.0    # multiplier applied to normalised RMS before
                             # mapping to 0–255 brightness

# ---------------------------------------------------------------------------
# Speech — wake word (OpenWakeWord)
# ---------------------------------------------------------------------------

# Paths to the two .onnx wake word model files.  Override in .env if needed.
# Put custom-trained models in assets/models/ and point these at them.
WAKE_WORD_MODEL_1 = Path(
    _optional("WAKE_WORD_MODEL_1",
              str(MODELS_DIR / "hey_rex.onnx"))
)
WAKE_WORD_MODEL_2 = Path(
    _optional("WAKE_WORD_MODEL_2",
              str(MODELS_DIR / "hey_r3x.onnx"))
)

WAKE_WORD_THRESHOLD = 0.6    # minimum score (0–1) to count as a detection
WAKE_WORD_COOLDOWN  = 2.0    # seconds to ignore further detections after one fires

# OpenWakeWord requires exactly 1280 samples (80 ms at 16 kHz) per predict() call.
WAKE_WORD_CHUNK_SIZE = 1280

# ---------------------------------------------------------------------------
# Speech — transcription (Vosk)
# ---------------------------------------------------------------------------

VOSK_MODEL_PATH  = Path(_optional("VOSK_MODEL_PATH") or VOSK_MODEL_PATH_DEFAULT)
VOSK_LOG_LEVEL   = -1   # -1 suppresses Vosk's verbose C-level logging

# ---------------------------------------------------------------------------
# Speech — synthesis (ElevenLabs)
# ---------------------------------------------------------------------------

ELEVENLABS_MODEL_ID   = "eleven_turbo_v2"   # lowest-latency streaming model
ELEVENLABS_STABILITY  = 0.45
ELEVENLABS_SIMILARITY = 0.80

# ---------------------------------------------------------------------------
# Command parser
# ---------------------------------------------------------------------------

# Minimum difflib SequenceMatcher ratio (0–1) for a fuzzy phrase match to be
# accepted. 0.72 catches one-word transcription errors ("louder" → "loudr")
# while rejecting accidental matches between short unrelated phrases.
# Lower = more permissive; raise toward 0.85 if you get false positives.
COMMAND_FUZZY_THRESHOLD = 0.72

# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------

ACTIVE_IDLE_TIMEOUT = 20.0   # seconds of silence before returning to IDLE

# ---------------------------------------------------------------------------
# LED serial command strings sent to Arduino Nanos
# Nanos are dumb executors; all logic lives on the Pi.
# ---------------------------------------------------------------------------

LED_CMD_IDLE      = "IDLE\n"      # slow breathing pulse
LED_CMD_ACTIVE    = "ACTIVE\n"    # full-brightness steady
LED_CMD_LISTENING = "LISTEN\n"    # color shift indicating attention
LED_CMD_SPEAKING  = "SPEAK\n"     # driven per-frame by audio level
LED_CMD_OFF       = "OFF\n"

# Parameterised commands — format() before sending
LED_CMD_BRIGHTNESS = "BRIGHT:{}\n"      # 0–255 integer
LED_CMD_EYE_COLOR  = "EYE:{},{},{}\n"   # R,G,B integers 0–255
